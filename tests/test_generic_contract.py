"""Adapter-independent contract-v2 conformance tests.

The two concrete adapters are exercised through synthetic source data, then
checked with the same assertions.  This file deliberately does not import a
COLMAP backend, the workflow CLI, or helpers from another test module.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from rtk_splat.adapters import agrigs
from rtk_splat.adapters.pose_sources import (
    TrajectoryFile,
    _attach_enu,
    make_pose_source,
)
from rtk_splat.adapters.registry import ADAPTER_NAMES, publish_from_config
from rtk_splat.adapters.ros2_zed_ublox import (
    FrameRecord,
    RtkTrack,
    publish_segment_v2,
)
from rtk_splat.workflows.configio import load_config
from rtk_splat.core.segment import CAPABILITIES, SegmentReader


BASE_NS = 1_700_000_000_123_456_789
CLOCK_OFFSET_NS = 5_000


def _assert_integer_vector(
    testcase: unittest.TestCase,
    values: np.ndarray,
    expected_length: int,
    label: str,
) -> None:
    testcase.assertEqual(values.shape, (expected_length,), label)
    testcase.assertTrue(np.issubdtype(values.dtype, np.integer), label)


def _assert_camera_calibration(
    testcase: unittest.TestCase, camera: dict, label: str
) -> None:
    testcase.assertGreater(camera["width"], 0, label)
    testcase.assertGreater(camera["height"], 0, label)
    testcase.assertIsInstance(camera["model"], str, label)
    testcase.assertTrue(camera["model"], label)
    intrinsics = np.asarray(camera["K"], dtype=float)
    testcase.assertEqual(intrinsics.shape, (3, 3), label)
    testcase.assertTrue(np.isfinite(intrinsics).all(), label)
    testcase.assertGreater(intrinsics[0, 0], 0, label)
    testcase.assertGreater(intrinsics[1, 1], 0, label)
    testcase.assertEqual(np.asarray(camera["distortion"]).ndim, 1, label)


def assert_adapter_conformance(
    testcase: unittest.TestCase,
    reader: SegmentReader,
    *,
    expected_frames: int,
    positioning: str,
) -> None:
    """Apply the common contract gates to any adapter-produced segment."""
    testcase.assertIs(reader.validate(), reader)
    frames = reader.frames
    meta = reader.meta
    calibration = reader.calibration
    capabilities = meta["capabilities"]
    gnss = reader.observations("gnss")
    assert gnss is not None

    testcase.assertEqual(meta["contract_version"], 2)
    testcase.assertEqual(meta["n_frames"], expected_frames)
    testcase.assertTrue(set(CAPABILITIES).issubset(capabilities))
    for name in CAPABILITIES:
        testcase.assertIs(type(capabilities[name]), bool, name)
    testcase.assertTrue(capabilities["stereo"] or capabilities["rgbd"])
    testcase.assertTrue(
        capabilities["images_raw"] or capabilities["images_rectified"]
    )

    _assert_integer_vector(
        testcase, frames["frame_id"], expected_frames, "frame_id"
    )
    np.testing.assert_array_equal(
        frames["frame_id"], np.arange(expected_frames, dtype=np.int64)
    )
    testcase.assertEqual(frames["timestamp_ns"].dtype, np.dtype(np.int64))
    testcase.assertTrue(np.all(np.diff(frames["timestamp_ns"]) > 0))
    testcase.assertEqual(frames["left_image_path"].shape, (expected_frames,))
    for relative in frames["left_image_path"]:
        testcase.assertTrue((reader.root / str(relative)).is_file())

    testcase.assertEqual(calibration["contract_version"], 2)
    testcase.assertIn("left", calibration["cameras"])
    _assert_camera_calibration(
        testcase, calibration["cameras"]["left"], "left camera"
    )

    if capabilities["stereo"]:
        testcase.assertIn("right", calibration["cameras"])
        _assert_camera_calibration(
            testcase, calibration["cameras"]["right"], "right camera"
        )
        transform = np.asarray(calibration["T_right_left"], dtype=float)
        testcase.assertEqual(transform.shape, (4, 4))
        testcase.assertTrue(np.isfinite(transform).all())
        testcase.assertGreater(np.linalg.norm(transform[:3, 3]), 0)
        testcase.assertEqual(
            calibration["transform_conventions"]["T_right_left"],
            "right_from_left",
        )
        testcase.assertEqual(
            frames["right_timestamp_ns"].dtype, np.dtype(np.int64)
        )
        testcase.assertTrue(
            np.all(np.diff(frames["right_timestamp_ns"]) > 0)
        )
        np.testing.assert_array_equal(
            frames["stereo_sync_residual_ns"],
            frames["right_timestamp_ns"] - frames["timestamp_ns"],
        )
        for relative in frames["right_image_path"]:
            testcase.assertTrue((reader.root / str(relative)).is_file())

    if capabilities["rgbd"]:
        testcase.assertIn("depth_path", frames)
        testcase.assertEqual(frames["depth_path"].shape, (expected_frames,))
        testcase.assertIn("depth_observation", meta)
        testcase.assertEqual(meta["depth_observation"]["units"], "m")
        for relative in frames["depth_path"]:
            testcase.assertTrue((reader.root / str(relative)).is_file())

    required_gnss = {
        "frame_id",
        "frame_timestamp_ns",
        "source_index",
        "source_timestamp_ns",
        "enu_m",
        "covariance_enu_m2",
        "fix_status",
        "carrier_status",
        "position_valid",
        "position_quality",
    }
    testcase.assertTrue(required_gnss.issubset(gnss))
    np.testing.assert_array_equal(gnss["frame_id"], frames["frame_id"])
    np.testing.assert_array_equal(
        gnss["frame_timestamp_ns"], frames["timestamp_ns"]
    )
    testcase.assertEqual(
        gnss["frame_timestamp_ns"].dtype, np.dtype(np.int64)
    )
    testcase.assertEqual(
        gnss["source_timestamp_ns"].dtype, np.dtype(np.int64)
    )
    _assert_integer_vector(
        testcase, gnss["source_index"], expected_frames, "GNSS source_index"
    )
    testcase.assertEqual(gnss["enu_m"].shape, (expected_frames, 3))
    testcase.assertEqual(
        gnss["covariance_enu_m2"].shape, (expected_frames, 3, 3)
    )
    _assert_integer_vector(
        testcase, gnss["fix_status"], expected_frames, "GNSS fix_status"
    )
    _assert_integer_vector(
        testcase,
        gnss["carrier_status"],
        expected_frames,
        "GNSS carrier_status",
    )
    testcase.assertEqual(gnss["position_valid"].shape, (expected_frames,))
    testcase.assertTrue(
        np.issubdtype(gnss["position_valid"].dtype, np.bool_)
    )
    testcase.assertEqual(gnss["position_quality"].shape, (expected_frames,))
    testcase.assertTrue(
        np.issubdtype(gnss["position_quality"].dtype, np.str_)
    )
    testcase.assertEqual(
        meta["position_observation"]["validity_field"], "position_valid"
    )
    testcase.assertEqual(
        meta["position_observation"]["quality_field"], "position_quality"
    )
    testcase.assertIn("raw_timestamp_ns", gnss)
    testcase.assertEqual(
        gnss["raw_timestamp_ns"].dtype, np.dtype(np.int64)
    )
    testcase.assertTrue(np.all(np.diff(gnss["raw_timestamp_ns"]) > 0))
    np.testing.assert_array_equal(
        gnss["source_timestamp_ns"],
        gnss["raw_timestamp_ns"][gnss["source_index"]],
    )

    testcase.assertIn("coordinate_frame", meta)
    testcase.assertEqual(meta["coordinate_frame"]["type"], "local_enu")
    testcase.assertEqual(meta["coordinate_frame"]["units"], "m")
    testcase.assertIn("origin_wgs84", meta["coordinate_frame"])
    testcase.assertIn("timebase", meta)
    testcase.assertEqual(meta["timebase"]["unit"], "ns")
    testcase.assertIn("position_observation", meta)

    if positioning == "dual_rtk":
        testcase.assertTrue(capabilities["dual_rtk"])
        testcase.assertFalse(capabilities["single_rtk"])
        testcase.assertTrue(np.isfinite(gnss["enu_m"]).all())
        testcase.assertTrue(
            np.isfinite(gnss["covariance_enu_m2"]).all()
        )
        testcase.assertTrue(gnss["position_valid"].all())
        testcase.assertEqual(
            set(gnss["position_quality"].tolist()), {"rtk_fixed"}
        )
        heading = reader.observations("heading")
        assert heading is not None
        required_heading = {
            "frame_id",
            "frame_timestamp_ns",
            "source_index",
            "source_timestamp_ns",
            "baseline_ned_m",
            "acc_heading_rad",
            "carrier_status",
            "flags",
            "valid",
        }
        testcase.assertTrue(required_heading.issubset(heading))
        testcase.assertEqual(
            heading["baseline_ned_m"].shape, (expected_frames, 3)
        )
        testcase.assertEqual(
            heading["flags"].shape, (expected_frames, 8)
        )
        testcase.assertTrue(
            np.isfinite(heading["baseline_ned_m"][heading["valid"]]).all()
        )
        testcase.assertEqual(
            heading["source_timestamp_ns"].dtype, np.dtype(np.int64)
        )
        testcase.assertIn("raw_baseline_ned_m", heading)
        testcase.assertEqual(heading["raw_baseline_ned_m"].shape[1], 3)
        semantics = meta["heading_observation"]
        testcase.assertEqual(semantics["vector"], "primary_to_secondary")
        testcase.assertEqual(
            semantics["components"], ["north", "east", "down"]
        )
        testcase.assertNotEqual(
            semantics["primary_frame_id"], semantics["secondary_frame_id"]
        )
    elif positioning == "oracle":
        testcase.assertFalse(capabilities["single_rtk"])
        testcase.assertFalse(capabilities["dual_rtk"])
        testcase.assertIsNone(
            reader.observations("heading", required=False)
        )
        testcase.assertEqual(
            meta["position_observation"]["type"], "oracle_groundtruth"
        )
        testcase.assertTrue(np.isnan(gnss["covariance_enu_m2"]).all())
        np.testing.assert_array_equal(
            gnss["fix_status"],
            np.full(expected_frames, -1, dtype=gnss["fix_status"].dtype),
        )
        np.testing.assert_array_equal(
            gnss["carrier_status"],
            np.full(
                expected_frames, -1, dtype=gnss["carrier_status"].dtype
            ),
        )
        testcase.assertTrue(gnss["position_valid"].all())
        testcase.assertEqual(
            set(gnss["position_quality"].tolist()), {"oracle"}
        )
        testcase.assertEqual(
            set(gnss["evidence_type"].tolist()),
            {"oracle_groundtruth"},
        )
    else:  # pragma: no cover - protects accidental misuse of the helper
        raise AssertionError(f"unknown positioning mode: {positioning}")


def _encoded_jpeg(color: tuple[int, int, int]) -> bytes:
    image = np.full((6, 8, 3), color, dtype=np.uint8)
    ok, payload = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("failed to encode synthetic JPEG")
    return payload.tobytes()


def _camera_info(*, right: bool) -> dict:
    projection = np.array(
        [
            [100.0, 0.0, 4.0, -12.0 if right else 0.0],
            [0.0, 101.0, 3.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    return {
        "width": 8,
        "height": 6,
        "k": [
            [90.0, 0.0, 4.1],
            [0.0, 91.0, 3.1],
            [0.0, 0.0, 1.0],
        ],
        "d": [0.1, -0.02, 0.001, 0.002, 0.0],
        "r": np.eye(3).tolist(),
        "p": projection.tolist(),
        "distortion_model": "plumb_bob",
    }


def _synthetic_ros2_inputs():
    frame_ns = BASE_NS + np.arange(3, dtype=np.int64) * 100_000_000
    left = _encoded_jpeg((20, 80, 140))
    right = _encoded_jpeg((30, 90, 150))
    frames = [
        FrameRecord(
            t=int(timestamp) * 1.0e-9,
            t_right=(int(timestamp) + 2_000) * 1.0e-9,
            left_jpeg=left,
            right_jpeg=right,
            left_header_ns=int(timestamp),
            right_header_ns=int(timestamp) + 2_000,
        )
        for timestamp in frame_ns
    ]
    poses = []
    for index in range(len(frame_ns)):
        center = np.array([float(index), 0.25, 1.5])
        viewmat = np.eye(4)
        viewmat[:3, 3] = -center
        poses.append(SimpleNamespace(viewmat=viewmat, cam_center=center))

    sensor_ns = frame_ns + CLOCK_OFFSET_NS
    covariance = np.repeat(
        np.array(
            [
                [
                    [0.0004, 0.00001, 0.0],
                    [0.00001, 0.0005, 0.0],
                    [0.0, 0.0, 0.0009],
                ]
            ]
        ),
        len(sensor_ns),
        axis=0,
    )
    flags = np.tile(
        [True, True, True, True, False, False, True, False],
        (len(sensor_ns), 1),
    )
    track = RtkTrack(
        fix_t=sensor_ns.astype(np.float64) * 1.0e-9,
        fix_lat=np.linspace(52.0, 52.000002, len(sensor_ns)),
        fix_lon=np.linspace(21.0, 21.000002, len(sensor_ns)),
        fix_alt=np.linspace(100.0, 100.2, len(sensor_ns)),
        fix_status=np.full(len(sensor_ns), 2, dtype=np.int16),
        fix_cov_max=np.max(
            np.diagonal(covariance, axis1=1, axis2=2), axis=1
        ),
        relpos_t=sensor_ns.astype(np.float64) * 1.0e-9,
        relpos_yaw=np.zeros(len(sensor_ns)),
        relpos_carr=np.full(len(sensor_ns), 2, dtype=np.int16),
        fix_header_ns=sensor_ns,
        fix_log_ns=sensor_ns + 111,
        fix_covariance_enu_m2=covariance,
        fix_covariance_type=np.full(len(sensor_ns), 3, dtype=np.int16),
        fix_carrier_status=np.full(len(sensor_ns), 2, dtype=np.int16),
        relpos_header_ns=sensor_ns,
        relpos_log_ns=sensor_ns + 222,
        relpos_ned_m=np.tile([1.20, 0.10, -0.02], (len(sensor_ns), 1)),
        relpos_acc_heading_rad=np.full(len(sensor_ns), 0.01),
        relpos_flags=flags,
        pvt_header_ns=sensor_ns,
        pvt_log_ns=sensor_ns + 333,
        pvt_carrier_status=np.full(len(sensor_ns), 2, dtype=np.int16),
    )
    track.enu_xyz = np.column_stack(
        (
            np.arange(len(sensor_ns)) * 0.25,
            np.zeros(len(sensor_ns)),
            np.linspace(0.0, 0.02, len(sensor_ns)),
        )
    )
    return frames, poses, track


def _publish_synthetic_ros2(destination: Path) -> SegmentReader:
    frames, poses, track = _synthetic_ros2_inputs()
    return publish_segment_v2(
        destination,
        frames,
        poses,
        track,
        _camera_info(right=False),
        _camera_info(right=True),
        camera_frame_id="zed_left_camera_optical_frame",
        primary_antenna_frame_id="gnss_primary_phase_center",
        secondary_antenna_frame_id="gnss_secondary_phase_center",
        enu_definition={
            "origin_lat_deg": 52.0,
            "origin_lon_deg": 21.0,
            "origin_alt_ellipsoidal_m": 100.0,
            "ellipsoid": "WGS84",
            "vertical_datum": "WGS84 ellipsoid (not orthometric)",
            "world_frame_id": "map",
        },
        T_camera_primary_antenna=[
            [1.0, 0.0, 0.0, 0.20],
            [0.0, 1.0, 0.0, -0.03],
            [0.0, 0.0, 1.0, 0.65],
            [0.0, 0.0, 0.0, 1.0],
        ],
        extrinsic_translation_sigma_m=[0.05, 0.05, 0.08],
        extrinsic_provenance={
            "method": "synthetic rough-prior measurement",
            "status": "test fixture",
        },
        clock_offset_ns=CLOCK_OFFSET_NS,
        association_tolerance_ns=1_000,
        stereo_tolerance_ns=10_000,
        capabilities={"single_rtk": False, "dual_rtk": True},
        splits={"train": [0, 2], "val": [1], "test": []},
        provenance={"adapter_test": True},
    )


_SYNTHETIC_GEODESY = SimpleNamespace(
    ecef2geodetic=lambda x, y, z: (
        0.0,
        0.0,
        float(x) - 6_378_137.0,
    ),
    ecef2enu=lambda x, y, z, lat, lon, alt: (
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
        np.asarray(x, dtype=float) - 6_378_137.0 - float(alt),
    ),
)


def _write_agrigs_split(
    dataset: Path,
    split: str,
    seconds: tuple[int, int],
    east_m: tuple[float, float],
) -> None:
    root = dataset / split
    rgb = root / "zed_multi" / "cam_1" / "rgb"
    depth = root / "zed_multi" / "cam_1" / "depth"
    rgb.mkdir(parents=True)
    depth.mkdir(parents=True)
    rows = [
        "timestamp,tx,ty,tz,qx,qy,qz,qw",
        *[
            f"{second}-000000123,6378137.0,{east},0.0,0,0,0,1"
            for second, east in zip(seconds, east_m)
        ],
    ]
    (root / "groundtruth_cam_1.csv").write_text("\n".join(rows) + "\n")
    for index, second in enumerate(seconds):
        stem = f"{second}-000000123"
        image = np.full(
            (6, 8, 3), (20 + 10 * index, 80, 140), dtype=np.uint8
        )
        depth_mm = np.full((6, 8), 2_000, dtype=np.uint16)
        depth_mm[0, 0] = 0
        if not cv2.imwrite(str(rgb / f"{stem}.jpg"), image):
            raise RuntimeError("failed to write synthetic AgriGS RGB")
        if not cv2.imwrite(str(depth / f"{stem}.png"), depth_mm):
            raise RuntimeError("failed to write synthetic AgriGS depth")


def _publish_synthetic_agrigs(root: Path) -> SegmentReader:
    dataset = root / "agrigs"
    _write_agrigs_split(dataset, "train", (10, 11), (0.0, 0.2))
    _write_agrigs_split(dataset, "val", (20, 21), (0.4, 0.6))
    with patch.object(agrigs, "pymap3d", _SYNTHETIC_GEODESY):
        return agrigs.ingest(
            dataset,
            "cam_1",
            intrinsic=[10.0, 10.0, 4.0, 3.0],
            distortion=[0.0] * 5,
            out_seg=root / "segment",
            min_z=0.5,
            max_z=3.0,
        )


class AdapterConformanceTests(unittest.TestCase):
    def test_ros2_dual_rtk_output_passes_shared_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            reader = _publish_synthetic_ros2(
                Path(temporary) / "segment"
            )
            assert_adapter_conformance(
                self,
                reader,
                expected_frames=3,
                positioning="dual_rtk",
            )
            heading = reader.observations("heading")
            assert heading is not None
            np.testing.assert_allclose(
                heading["baseline_ned_m"],
                np.tile([1.20, 0.10, -0.02], (3, 1)),
            )

    def test_agrigs_output_passes_shared_contract_without_false_rtk_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            reader = _publish_synthetic_agrigs(Path(temporary))
            assert_adapter_conformance(
                self,
                reader,
                expected_frames=4,
                positioning="oracle",
            )
            np.testing.assert_allclose(
                reader.observations("gnss")["enu_m"],
                reader.frames["initial_camera_center_m"],
            )


class AdapterRegistryTests(unittest.TestCase):
    def test_registry_declares_exact_supported_adapters(self):
        self.assertEqual(
            set(ADAPTER_NAMES),
            {
                "agrigs",
                "ros1_citrusfarm",
                "ros1_rosario_v2",
                "ros2_zed_ublox",
            },
        )

    def test_registry_dispatches_ros2_with_selection(self):
        cfg = SimpleNamespace(adapter="ros2_zed_ublox")
        destination = Path("/new/segment")
        selection = {"t0": 1.0, "t1": 2.0}
        expected = object()
        with patch(
            "rtk_splat.adapters.ros2_zed_ublox.ingest_config_v2",
            return_value=expected,
        ) as ingest:
            result = publish_from_config(
                cfg, destination, selection=selection
            )
        self.assertIs(result, expected)
        ingest.assert_called_once_with(
            cfg, destination, window=selection
        )

    def test_registry_dispatches_ros1(self):
        cfg = SimpleNamespace(adapter="ros1_citrusfarm")
        destination = Path("/new/segment")
        expected = object()
        with patch(
            "rtk_splat.adapters.ros1_citrusfarm.ingest_config_v2",
            return_value=expected,
        ) as ingest:
            result = publish_from_config(cfg, destination)
        self.assertIs(result, expected)
        ingest.assert_called_once_with(cfg, destination, window=None)

    def test_registry_dispatches_agrigs_without_ros_selection(self):
        cfg = SimpleNamespace(adapter="agrigs")
        destination = Path("/new/segment")
        expected = object()
        with patch(
            "rtk_splat.adapters.agrigs.ingest_config_v2",
            return_value=expected,
        ) as ingest:
            result = publish_from_config(cfg, destination)
        self.assertIs(result, expected)
        ingest.assert_called_once_with(cfg, destination)

    def test_registry_rejects_unknown_adapter(self):
        with self.assertRaisesRegex(ValueError, "unknown adapter"):
            publish_from_config(
                SimpleNamespace(adapter="not-real"),
                Path("/new/segment"),
            )

    def test_builtin_adapter_rejects_unvalidated_namespaced_options(self):
        with self.assertRaisesRegex(ValueError, "does not declare any"):
            publish_from_config(
                SimpleNamespace(
                    adapter="ros2_zed_ublox",
                    adapter_options=SimpleNamespace(vendor_mode=True),
                ),
                Path("/new/segment"),
            )

    def test_registry_does_not_pass_a_ros_window_to_agrigs(self):
        with self.assertRaisesRegex(ValueError, "does not consume"):
            publish_from_config(
                SimpleNamespace(adapter="agrigs"),
                Path("/new/segment"),
                selection={"t0": 1.0, "t1": 2.0},
            )


class ConfigTests(unittest.TestCase):
    def test_non_ros_config_does_not_require_bags_or_ublox_messages(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.yaml"
            config.write_text(
                f"paths:\n  workdir: {temporary}/work\n"
            )
            loaded = load_config(config)
        self.assertEqual(loaded.paths.bags, [])
        self.assertFalse(hasattr(loaded.paths, "ublox_msgs_dir"))

    def test_workdir_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.yaml"
            config.write_text("paths: {}\n")
            with self.assertRaisesRegex(ValueError, "paths.workdir"):
                load_config(config)

    def test_dual_antenna_source_requires_its_adapter_types(self):
        cfg = SimpleNamespace(
            paths=SimpleNamespace(),
            pose=SimpleNamespace(source="rtk_dual_antenna"),
        )
        with self.assertRaisesRegex(ValueError, "ublox_msgs_dir"):
            make_pose_source(cfg)

    def test_rtk_track_retains_its_enu_crs_definition(self):
        track = RtkTrack(
            fix_t=np.array([0.0, 1.0]),
            fix_lat=np.array([52.0, 52.000001]),
            fix_lon=np.array([21.0, 21.000001]),
            fix_alt=np.array([100.0, 100.0]),
            fix_status=np.array([2, 2]),
            fix_cov_max=np.array([0.0004, 0.0004]),
            relpos_t=np.array([0.0, 1.0]),
            relpos_yaw=np.array([0.0, 0.0]),
            relpos_carr=np.array([2, 2]),
        )
        with patch(
            "rtk_splat.adapters.pose_sources.LocalEnu.to_enu",
            return_value=np.zeros((2, 3)),
        ):
            _attach_enu(track)
        self.assertEqual(track.enu.crs()["type"], "local_ENU")


class TrajectoryFileTests(unittest.TestCase):
    def test_tum_trajectory_has_complete_track_surface(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory.txt"
            path.write_text(
                "0 1 2 3 0 0 0 1\n"
                "1 2 2 3 0 0 0 1\n"
                "2 3 2 3 0 0 0 1\n"
            )
            cfg = SimpleNamespace(
                pose=SimpleNamespace(trajectory_file=str(path))
            )
            source = TrajectoryFile(cfg)
        self.assertEqual(source.track.fix_status.shape, (3,))
        self.assertEqual(source.track.fix_cov_max.shape, (3,))
        self.assertTrue(np.isnan(source.track.fix_cov_max).all())
        pose = source.pose_frames([0.5])[0]
        np.testing.assert_allclose(
            pose.cam_center, [1.5, 2.0, 3.0]
        )

    def test_tum_timestamps_must_increase(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory.txt"
            path.write_text(
                "1 1 2 3 0 0 0 1\n"
                "1 2 2 3 0 0 0 1\n"
            )
            cfg = SimpleNamespace(
                pose=SimpleNamespace(trajectory_file=str(path))
            )
            with self.assertRaisesRegex(
                RuntimeError, "strictly increasing"
            ):
                TrajectoryFile(cfg)


if __name__ == "__main__":
    unittest.main()
