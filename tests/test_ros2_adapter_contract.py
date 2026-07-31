import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from rtk_splat.adapters.ros2_zed_ublox import (
    FrameRecord,
    RtkTrack,
    publish_segment_v2,
)


BASE_NS = 1_700_000_000_123_456_789
CLOCK_OFFSET_NS = 5_000


def _encoded(extension: str, color: tuple[int, int, int]) -> bytes:
    image = np.full((6, 8, 3), color, dtype=np.uint8)
    ok, payload = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"failed to encode {extension}")
    return payload.tobytes()


def _camera_info(*, right: bool) -> dict:
    projection = np.array(
        [[100.0, 0.0, 4.0, -12.0 if right else 0.0],
         [0.0, 101.0, 3.0, 0.0],
         [0.0, 0.0, 1.0, 0.0]]
    )
    return {
        "width": 8,
        "height": 6,
        "k": [[90.0, 0.0, 4.1], [0.0, 91.0, 3.1], [0.0, 0.0, 1.0]],
        "d": [0.1, -0.02, 0.001, 0.002, 0.0],
        "r": np.eye(3).tolist(),
        "p": projection.tolist(),
        "distortion_model": "plumb_bob",
    }


def _frames_and_poses():
    timestamps = BASE_NS + np.arange(3, dtype=np.int64) * 100_000_000
    jpeg = _encoded(".jpg", (20, 80, 140))
    png = _encoded(".png", (30, 90, 150))
    frames = [
        FrameRecord(
            t=int(timestamp) * 1e-9,
            t_right=(int(timestamp) + 2_000) * 1e-9,
            left_jpeg=jpeg,
            right_jpeg=png,
            left_header_ns=int(timestamp),
            right_header_ns=int(timestamp) + 2_000,
        )
        for timestamp in timestamps
    ]
    poses = []
    for index in range(3):
        center = np.array([float(index), 0.25, 1.5])
        viewmat = np.eye(4)
        viewmat[:3, 3] = -center
        poses.append(
            SimpleNamespace(viewmat=viewmat, cam_center=center)
        )
    return timestamps, frames, poses


def _track(*, dual: bool) -> RtkTrack:
    frame_ns = BASE_NS + np.arange(3, dtype=np.int64) * 100_000_000
    query_ns = frame_ns + CLOCK_OFFSET_NS
    fix_ns = np.array(
        [
            query_ns[0] - 1_000_000,
            query_ns[0] + 40_000_000,
            query_ns[1] - 1_000_000,
            query_ns[1] + 40_000_000,
            query_ns[2] - 1_000_000,
        ],
        dtype=np.int64,
    )
    fix_covariance = np.repeat(
        np.array(
            [[[0.0004, 0.00001, 0.0],
              [0.00001, 0.0005, 0.0],
              [0.0, 0.0, 0.0009]]]
        ),
        len(fix_ns),
        axis=0,
    )
    if dual:
        relpos_ns = query_ns - 500_000
        relpos_log = relpos_ns + 222
        baseline = np.tile([1.20, 0.10, -0.02], (3, 1))
        accuracy = np.array([0.01, 0.011, 0.012])
        carrier = np.full(3, 2, dtype=np.int16)
        flags = np.tile(
            [True, True, True, True, False, False, True, False],
            (3, 1),
        )
    else:
        relpos_ns = np.array([-1], dtype=np.int64)
        relpos_log = np.array([-1], dtype=np.int64)
        baseline = np.full((1, 3), np.nan)
        accuracy = np.full(1, np.nan)
        carrier = np.full(1, -1, dtype=np.int16)
        flags = np.zeros((1, 8), dtype=bool)
    track = RtkTrack(
        fix_t=fix_ns.astype(np.float64) * 1e-9,
        fix_lat=np.linspace(52.0, 52.000004, len(fix_ns)),
        fix_lon=np.linspace(21.0, 21.000004, len(fix_ns)),
        fix_alt=np.linspace(100.0, 100.4, len(fix_ns)),
        fix_status=np.full(len(fix_ns), 2, dtype=np.int16),
        fix_cov_max=np.max(
            np.diagonal(fix_covariance, axis1=1, axis2=2), axis=1
        ),
        relpos_t=relpos_ns.astype(np.float64) * 1e-9,
        relpos_yaw=np.zeros(len(relpos_ns)),
        relpos_carr=carrier,
        fix_header_ns=fix_ns,
        fix_log_ns=fix_ns + 111,
        fix_covariance_enu_m2=fix_covariance,
        fix_covariance_type=np.full(len(fix_ns), 3, dtype=np.int16),
        fix_carrier_status=np.full(len(fix_ns), -1, dtype=np.int16),
        relpos_header_ns=relpos_ns,
        relpos_log_ns=relpos_log,
        relpos_ned_m=baseline,
        relpos_acc_heading_rad=accuracy,
        relpos_flags=flags,
    )
    track.enu_xyz = np.column_stack(
        [np.arange(len(fix_ns)) * 0.25, np.zeros(len(fix_ns)),
         np.linspace(0.0, 0.04, len(fix_ns))]
    )
    return track


def _writer_kwargs(*, dual: bool) -> dict:
    return {
        "camera_frame_id": "zed_left_camera_optical_frame",
        "primary_antenna_frame_id": "gnss_primary_phase_center",
        "secondary_antenna_frame_id": (
            "gnss_secondary_phase_center" if dual else None
        ),
        "enu_definition": {
            "origin_lat_deg": 52.0,
            "origin_lon_deg": 21.0,
            "origin_alt_ellipsoidal_m": 100.0,
            "ellipsoid": "WGS84",
            "vertical_datum": "WGS84 ellipsoid (not orthometric)",
            "world_frame_id": "map",
        },
        "T_camera_primary_antenna": [
            [1.0, 0.0, 0.0, 0.20],
            [0.0, 1.0, 0.0, -0.03],
            [0.0, 0.0, 1.0, 0.65],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "extrinsic_provenance": {
            "method": "rough tape measurement",
            "status": "rough prior; calibration sidecar may refine it",
        },
        "extrinsic_translation_sigma_m": [0.05, 0.05, 0.08],
        "clock_offset_ns": CLOCK_OFFSET_NS,
        "association_tolerance_ns": 2_000_000,
        "stereo_tolerance_ns": 10_000,
        "capabilities": {
            "single_rtk": not dual,
            "dual_rtk": dual,
        },
        "splits": {"train": [0, 2], "val": [1], "test": []},
        "provenance": {
            "sequence": "synthetic",
            "adapter_test": True,
        },
    }


def assert_ros2_contract(
    testcase: unittest.TestCase, reader, *, dual: bool
) -> None:
    reader.validate()
    frames = reader.frames
    gnss = reader.observations("gnss")
    assert gnss is not None
    testcase.assertTrue(reader.meta["capabilities"]["stereo"])
    testcase.assertEqual(reader.meta["capabilities"]["dual_rtk"], dual)
    testcase.assertEqual(reader.meta["capabilities"]["single_rtk"], not dual)
    testcase.assertFalse(reader.meta["capabilities"]["rgbd"])
    testcase.assertFalse(reader.meta["capabilities"]["depth_recorded"])
    testcase.assertFalse(reader.meta["capabilities"]["depth_computed"])
    testcase.assertNotIn("depth_path", frames)
    testcase.assertEqual(frames["timestamp_ns"].dtype, np.dtype(np.int64))
    testcase.assertEqual(
        gnss["raw_timestamp_ns"].dtype, np.dtype(np.int64)
    )
    testcase.assertEqual(
        gnss["raw_log_timestamp_ns"].dtype, np.dtype(np.int64)
    )
    testcase.assertEqual(gnss["raw_covariance_enu_m2"].shape, (5, 3, 3))
    np.testing.assert_array_equal(gnss["raw_carrier_status"], [-1] * 5)
    testcase.assertTrue(gnss["position_valid"].all())
    testcase.assertEqual(
        set(gnss["position_quality"].tolist()), {"differential"}
    )
    testcase.assertEqual(
        reader.meta["position_evidence"]["quantity"],
        "primary antenna phase-center position",
    )
    testcase.assertTrue(
        reader.meta["initial_pose_semantics"]["not_direct_gnss"]
    )
    testcase.assertEqual(reader.meta["crs"]["ellipsoid"], "WGS84")
    testcase.assertIn("vertical_datum", reader.meta["crs"])


class Ros2AdapterContractTests(unittest.TestCase):
    def test_dual_rtk_writer_preserves_complete_evidence_and_calibration(self):
        timestamps, frames, poses = _frames_and_poses()
        track = _track(dual=True)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "segment"
            reader = publish_segment_v2(
                destination,
                frames,
                poses,
                track,
                _camera_info(right=False),
                _camera_info(right=True),
                **_writer_kwargs(dual=True),
            )
            assert_ros2_contract(self, reader, dual=True)

            frame_values = reader.frames
            np.testing.assert_array_equal(frame_values["timestamp_ns"], timestamps)
            np.testing.assert_array_equal(
                frame_values["right_timestamp_ns"], timestamps + 2_000
            )
            self.assertTrue(
                all(str(path).endswith(".jpg")
                    for path in frame_values["left_image_path"])
            )
            self.assertTrue(
                all(str(path).endswith(".png")
                    for path in frame_values["right_image_path"])
            )
            self.assertEqual(
                (destination / str(frame_values["left_image_path"][0])).read_bytes(),
                frames[0].left_jpeg,
            )

            calibration = reader.calibration
            np.testing.assert_allclose(
                calibration["cameras"]["left"]["K"],
                [[100.0, 0.0, 4.0],
                 [0.0, 101.0, 3.0],
                 [0.0, 0.0, 1.0]],
            )
            self.assertEqual(
                calibration["cameras"]["left"]["distortion"], []
            )
            self.assertEqual(
                calibration["rectification"]["source_camera_info"]["left"]["d"],
                [0.1, -0.02, 0.001, 0.002, 0.0],
            )
            np.testing.assert_allclose(
                np.asarray(calibration["T_right_left"])[:3, 3],
                [-0.12, 0.0, 0.0],
            )
            self.assertEqual(
                calibration["sensor_frames"]["primary_antenna"],
                "gnss_primary_phase_center",
            )

            gnss = reader.observations("gnss")
            assert gnss is not None
            np.testing.assert_array_equal(gnss["source_index"], [0, 2, 4])
            np.testing.assert_array_equal(
                gnss["source_residual_ns"], [-1_000_000] * 3
            )
            np.testing.assert_array_equal(
                gnss["source_log_timestamp_ns"],
                gnss["source_timestamp_ns"] + 111,
            )
            np.testing.assert_allclose(
                gnss["covariance_enu_m2"],
                track.fix_covariance_enu_m2[[0, 2, 4]],
            )

            heading = reader.observations("heading")
            assert heading is not None
            np.testing.assert_array_equal(heading["source_index"], [0, 1, 2])
            np.testing.assert_allclose(
                heading["raw_baseline_ned_m"],
                np.tile([1.20, 0.10, -0.02], (3, 1)),
            )
            self.assertTrue(heading["raw_valid"].all())
            self.assertEqual(
                reader.meta["heading_evidence"]["quantity"],
                "primary-to-secondary antenna baseline",
            )
            np.testing.assert_allclose(
                frame_values["initial_camera_center_m"][:, 0], [0, 1, 2]
            )

    def test_single_rtk_writer_omits_heading_without_losing_raw_gnss(self):
        _, frames, poses = _frames_and_poses()
        with tempfile.TemporaryDirectory() as temporary:
            reader = publish_segment_v2(
                Path(temporary) / "segment",
                frames,
                poses,
                _track(dual=False),
                _camera_info(right=False),
                _camera_info(right=True),
                **_writer_kwargs(dual=False),
            )
            assert_ros2_contract(self, reader, dual=False)
            self.assertIsNone(
                reader.observations("heading", required=False)
            )
            self.assertIsNone(reader.meta["heading_evidence"])

    def test_association_tolerance_failure_leaves_no_published_or_staged_data(self):
        _, frames, poses = _frames_and_poses()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "segment"
            options = _writer_kwargs(dual=True)
            options["association_tolerance_ns"] = 100
            with self.assertRaisesRegex(ValueError, "association exceeds"):
                publish_segment_v2(
                    destination,
                    frames,
                    poses,
                    _track(dual=True),
                    _camera_info(right=False),
                    _camera_info(right=True),
                    **options,
                )
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".segment.writing-*")), [])

    def test_existing_destination_is_never_modified(self):
        _, frames, poses = _frames_and_poses()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "segment"
            destination.mkdir()
            marker = destination / "keep"
            marker.write_text("unchanged")
            with self.assertRaises(FileExistsError):
                publish_segment_v2(
                    destination,
                    frames,
                    poses,
                    _track(dual=True),
                    _camera_info(right=False),
                    _camera_info(right=True),
                    **_writer_kwargs(dual=True),
                )
            self.assertEqual(marker.read_text(), "unchanged")


if __name__ == "__main__":
    unittest.main()
