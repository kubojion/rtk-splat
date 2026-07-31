import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rtk_splat.frontends.keyframes import KEYFRAME_PRESETS, KeyframeConfig
from rtk_splat.frontends.pair_graph import PairGraphConfig
from rtk_splat.frontends.planning import make_rig_config, plan_frontend
from rtk_splat.core.segment import POSITION_QUALITY_VOCABULARY, SegmentWriter


def _segment(path: Path):
    n = 5
    writer = SegmentWriter(path)
    images = writer.directory("images")
    left_paths, right_paths = [], []
    # Valid minimal PNG payloads.
    import cv2

    for index in range(n):
        image = np.full((12, 16, 3), 40 + index, dtype=np.uint8)
        ok, payload = cv2.imencode(".png", image)
        assert ok
        for side, paths in (("left", left_paths), ("right", right_paths)):
            relative = f"images/{side}_{index:06d}.png"
            (writer.staging_dir / relative).write_bytes(payload.tobytes())
            paths.append(relative)
    timestamps = np.arange(n, dtype=np.int64) * 1_000_000_000 + 10
    viewmats = np.repeat(np.eye(4)[None], n, axis=0)
    viewmats[:, 0, 3] = -np.arange(n) * 0.1
    frames = {
        "frame_id": np.arange(n, dtype=np.int64),
        "timestamp_ns": timestamps,
        "left_image_path": np.asarray(left_paths),
        "right_image_path": np.asarray(right_paths),
        "right_timestamp_ns": timestamps + 100,
        "stereo_sync_residual_ns": np.full(n, 100, dtype=np.int64),
        "initial_viewmat": viewmats,
        "initial_camera_center_m": np.linalg.inv(viewmats)[:, :3, 3],
        "pose_valid": np.ones(n, dtype=bool),
    }
    camera = {
        "model": "PINHOLE",
        "width": 16,
        "height": 12,
        "K": [[20.0, 0, 8.0], [0, 20.0, 6.0], [0, 0, 1]],
        "distortion": [],
    }
    calibration = {
        "contract_version": 2,
        "cameras": {"left": camera, "right": camera},
        "T_right_left": [
            [1, 0, 0, -0.12],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        "transform_conventions": {"T_right_left": "right_from_left"},
    }
    common = {
        "frame_id": np.arange(n, dtype=np.int64),
        "frame_timestamp_ns": timestamps,
        "source_index": np.arange(n, dtype=np.int64),
        "source_timestamp_ns": timestamps,
    }
    gnss = {
        **common,
        "enu_m": np.column_stack([np.arange(n) * 0.1, np.zeros((n, 2))]),
        "covariance_enu_m2": np.repeat(np.eye(3)[None] * 0.0004, n, axis=0),
        "fix_status": np.ones(n, dtype=np.int16),
        "carrier_status": np.full(n, 2, dtype=np.int16),
        "position_valid": np.ones(n, dtype=bool),
        "position_quality": np.full(n, "rtk_fixed", dtype="<U16"),
        "raw_timestamp_ns": timestamps,
    }
    writer.write_frames(frames)
    writer.write_calibration(calibration)
    writer.write_meta(
        {
            "contract_version": 2,
            "n_frames": n,
            "capabilities": {
                "stereo": True,
                "rgbd": False,
                "single_rtk": True,
                "dual_rtk": False,
                "depth_recorded": False,
                "depth_computed": False,
                "imu_present": False,
                "images_raw": False,
                "images_rectified": True,
            },
            "coordinate_frame": {
                "type": "local_enu",
                "world_frame_id": "map",
                "units": "m",
                "origin_wgs84": {
                    "latitude_deg": 52.0,
                    "longitude_deg": 16.0,
                    "ellipsoidal_altitude_m": 100.0,
                    "ellipsoid": "WGS84",
                    "vertical_datum": "WGS84 ellipsoid",
                },
            },
            "timebase": {
                "frame_timestamp_source": "left_camera_header",
                "observation_timestamp_source": "sensor_header",
                "unit": "ns",
                "association_clock_offset_ns": 0,
            },
            "position_observation": {
                "type": "gnss",
                "quantity": "antenna_phase_center",
                "sensor_frame_id": "gnss",
                "coordinates": "ENU_m",
                "covariance_frame": "ENU_m2",
                "validity_field": "position_valid",
                "quality_field": "position_quality",
                "quality_vocabulary": list(POSITION_QUALITY_VOCABULARY),
            },
            "initial_pose": {
                "camera_frame_id": "left_camera",
                "position_quantity": "left_camera_center",
                "source": "synthetic",
                "lever_arm_applied": True,
                "extrinsic_translation_sigma_m": [0.1, 0.1, 0.1],
                "extrinsic_translation_sigma_frame_id": "left_camera",
            },
        }
    )
    writer.write_manifest({"train": [0, 1, 2, 3], "val": [4], "test": []})
    writer.write_observations("gnss", gnss)
    return writer.finalize()


class FrontendPlanningTests(unittest.TestCase):
    def test_all_frame_control_arm_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader = _segment(Path(tmp) / "segment")
            plan = plan_frontend(
                reader.root,
                keyframe_config=KEYFRAME_PRESETS["all"],
                compute_image_quality=False,
            )
            json.dumps(plan.keyframes, allow_nan=False)
        self.assertEqual(plan.keyframes["frame_ids"], [0, 1, 2, 3, 4])
        self.assertEqual(plan.quality["n_keyframes"], 5)
        self.assertEqual(plan.quality["n_registration_pairs"], 0)
        self.assertTrue(plan.quality["solve_frame_graph_connected"])
        self.assertGreaterEqual(plan.quality["solve_frame_graph_edges"], 4)
        self.assertLessEqual(plan.quality["solve_frame_graph_max_degree"], 8)
        self.assertTrue(
            all(
                record["reasons"][0] == "all_frames"
                for record in plan.keyframes["records"]
            )
        )

    def test_rig_uses_full_right_from_left_transform(self):
        angle = np.deg2rad(10)
        calibration = {
            "cameras": {
                side: {
                    "K": [[100, 0, 40], [0, 101, 30], [0, 0, 1]]
                }
                for side in ("left", "right")
            },
            "T_right_left": [
                [np.cos(angle), -np.sin(angle), 0, -0.12],
                [np.sin(angle), np.cos(angle), 0, 0.01],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        }
        rig = make_rig_config(calibration)[0]["cameras"]
        self.assertTrue(rig[0]["ref_sensor"])
        np.testing.assert_allclose(
            rig[1]["cam_from_rig_translation"], [-0.12, 0.01, 0]
        )
        self.assertAlmostEqual(
            np.linalg.norm(rig[1]["cam_from_rig_rotation"]), 1.0
        )

    def test_plan_keeps_all_frames_and_builds_bounded_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader = _segment(Path(tmp) / "segment")
            plan = plan_frontend(
                reader.root,
                keyframe_config=KeyframeConfig(
                    translation_m=0.19,
                    rotation_deg=30,
                    max_elapsed_s=10,
                    revisit_distance_m=0,
                ),
                pair_config=PairGraphConfig(
                    temporal_max_distance_m=1,
                    temporal_max_seconds=10,
                    revisit_distance_m=0,
                    registration_max_distance_m=1,
                    registration_max_seconds=10,
                ),
                compute_image_quality=True,
            )
        self.assertLess(plan.quality["n_keyframes"], 5)
        self.assertEqual(plan.quality["mandatory_stereo_pairs"], 5)
        self.assertTrue(plan.quality["all_frames_retained_for_gs"])
        self.assertTrue(plan.quality["solve_frame_graph_connected"])
        self.assertEqual(plan.keyframes["frame_ids"][0], 0)
        self.assertEqual(plan.keyframes["frame_ids"][-1], 4)
        self.assertEqual(len(plan.pair_graph.all_image_names), 10)


if __name__ == "__main__":
    unittest.main()
