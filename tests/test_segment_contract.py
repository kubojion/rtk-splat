import tempfile
import unittest
from pathlib import Path

import numpy as np

from rtk_splat.core.segment import (
    POSITION_QUALITY_VOCABULARY,
    SegmentContractError,
    SegmentReader,
    SegmentWriter,
    normalize_navsat_position_quality,
)


def _camera() -> dict:
    return {
        "model": "OPENCV",
        "width": 8,
        "height": 6,
        "K": [[10.0, 0.0, 4.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]],
        "distortion": [0.0, 0.0, 0.0, 0.0],
    }


def _arrays(n: int = 3) -> tuple[dict, dict, dict]:
    frame_time = np.arange(n, dtype=np.int64) * 100_000_000 + 1_000
    raw_time = np.arange(2 * n - 1, dtype=np.int64) * 50_000_000 + 900
    source_index = np.arange(n, dtype=np.int64) * 2
    common = {
        "frame_id": np.arange(n, dtype=np.int64),
        "frame_timestamp_ns": frame_time,
        "source_index": source_index,
        "source_timestamp_ns": raw_time[source_index],
    }
    gnss = {
        **{name: value.copy() for name, value in common.items()},
        "enu_m": np.column_stack((np.arange(n), np.zeros(n), np.ones(n))),
        "covariance_enu_m2": np.repeat(np.eye(3)[None] * 0.0004, n, axis=0),
        "fix_status": np.full(n, 2, dtype=np.int16),
        "carrier_status": np.full(n, 2, dtype=np.int16),
        "position_valid": np.ones(n, dtype=bool),
        "position_quality": np.asarray(["rtk_fixed"] * n),
        "raw_timestamp_ns": raw_time,
        "raw_enu_m": np.zeros((len(raw_time), 3)),
        "raw_covariance_enu_m2": np.repeat(
            np.eye(3)[None] * 0.0004, len(raw_time), axis=0
        ),
        "raw_fix_status": np.full(len(raw_time), 2, dtype=np.int16),
        "raw_carrier_status": np.full(len(raw_time), 2, dtype=np.int16),
    }
    heading = {
        **{name: value.copy() for name, value in common.items()},
        "baseline_ned_m": np.tile([0.4, 0.1, -0.02], (n, 1)),
        "acc_heading_rad": np.full(n, 0.01),
        "valid": np.ones(n, dtype=bool),
        "raw_timestamp_ns": raw_time,
        "raw_baseline_ned_m": np.tile([0.4, 0.1, -0.02], (len(raw_time), 1)),
        "raw_acc_heading_rad": np.full(len(raw_time), 0.01),
        "raw_valid": np.ones(len(raw_time), dtype=bool),
    }
    return common, gnss, heading


def _write_valid(
    destination: Path, *, mutate=None, include_heading: bool = True, imu: dict | None = None
) -> SegmentReader:
    n = 3
    common, gnss, heading = _arrays(n)
    writer = SegmentWriter(destination)
    for directory in ("images", "depth"):
        writer.directory(directory)
    left = np.array([f"images/left_{i}.jpg" for i in range(n)])
    right = np.array([f"images/right_{i}.jpg" for i in range(n)])
    depth = np.array([f"depth/{i}.npz" for i in range(n)])
    for relative in np.concatenate((left, right, depth)):
        path = writer.staging_dir / str(relative)
        path.write_bytes(b"x")
    right_time = common["frame_timestamp_ns"] + 2_000
    viewmats = np.repeat(np.eye(4)[None], n, axis=0)
    viewmats[:, 0, 3] = -np.arange(n)
    frames = {
        "frame_id": common["frame_id"],
        "timestamp_ns": common["frame_timestamp_ns"],
        "left_image_path": left,
        "right_image_path": right,
        "right_timestamp_ns": right_time,
        "stereo_sync_residual_ns": right_time - common["frame_timestamp_ns"],
        "depth_path": depth,
        "initial_viewmat": viewmats,
        "initial_camera_center_m": np.column_stack(
            (np.arange(n), np.zeros(n), np.zeros(n))
        ),
        "pose_valid": np.ones(n, dtype=bool),
    }
    calibration = {
        "contract_version": 2,
        "cameras": {"left": _camera(), "right": _camera()},
        "T_right_left": [
            [1.0, 0.0, 0.0, -0.12],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "transform_conventions": {"T_right_left": "right_from_left"},
    }
    meta = {
        "contract_version": 2,
        "n_frames": n,
        "capabilities": {
            "stereo": True,
            "rgbd": True,
            "single_rtk": False,
            "dual_rtk": True,
            "depth_recorded": True,
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
            "sensor_frame_id": "primary_antenna",
            "coordinates": "ENU_m",
            "covariance_frame": "ENU_m2",
            "validity_field": "position_valid",
            "quality_field": "position_quality",
            "quality_vocabulary": list(POSITION_QUALITY_VOCABULARY),
        },
        "heading_observation": {
            "vector": "primary_to_secondary",
            "components": ["north", "east", "down"],
            "primary_frame_id": "primary_antenna",
            "secondary_frame_id": "secondary_antenna",
        },
        "initial_pose": {
            "camera_frame_id": "left_camera",
            "position_quantity": "left_camera_center",
            "source": "synthetic_rtk_prior",
            "lever_arm_applied": True,
            "extrinsic_translation_sigma_m": [0.1, 0.1, 0.1],
            "extrinsic_translation_sigma_frame_id": "left_camera",
        },
        "depth_observation": {
            "format": "npz_depth_valid",
            "units": "m",
            "quantity": "optical_z",
            "aligned_to": "left",
            "invalid_convention": "valid=false and depth=0",
        },
    }
    manifest = {"train": [0, 2], "val": [1], "test": []}
    payload = {
        "frames": frames,
        "calibration": calibration,
        "meta": meta,
        "manifest": manifest,
        "gnss": gnss,
        "heading": heading,
    }
    if mutate:
        mutate(payload)
    if imu is not None:
        payload["imu"] = imu
    writer.write_frames(payload["frames"])
    writer.write_calibration(payload["calibration"])
    writer.write_meta(payload["meta"])
    writer.write_manifest(payload["manifest"])
    writer.write_observations("gnss", payload["gnss"])
    if include_heading:
        writer.write_observations("heading", payload["heading"])
    if imu is not None:
        writer.write_observations("imu", payload["imu"])
    try:
        return writer.finalize()
    except Exception:
        writer.abort()
        raise


class SegmentContractTests(unittest.TestCase):
    def test_portable_navsat_normalization_accepts_status_zero_without_pvt(self):
        valid, quality = normalize_navsat_position_quality(
            np.array([-1, 0, 1, 0, 0], dtype=np.int16),
            np.array([-1, -1, -1, 1, 2], dtype=np.int16),
            np.ones(5, dtype=bool),
        )
        np.testing.assert_array_equal(valid, [False, True, True, True, True])
        np.testing.assert_array_equal(
            quality,
            ["invalid", "standalone", "differential", "rtk_float", "rtk_fixed"],
        )

    def test_roundtrip_preserves_per_frame_and_raw_sensor_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader = _write_valid(Path(tmp) / "segment")
            self.assertIsInstance(reader, SegmentReader)
            reader.validate()
            gnss = reader.observations("gnss")
            heading = reader.observations("heading")
            assert gnss is not None and heading is not None
            self.assertEqual(len(gnss["frame_id"]), 3)
            self.assertEqual(len(gnss["raw_timestamp_ns"]), 5)
            np.testing.assert_array_equal(
                gnss["source_timestamp_ns"],
                gnss["raw_timestamp_ns"][gnss["source_index"]],
            )
            np.testing.assert_allclose(heading["baseline_ned_m"][0], [0.4, 0.1, -0.02])
            self.assertTrue(reader.meta["capabilities"]["stereo"])
            self.assertTrue(reader.meta["capabilities"]["rgbd"])
            self.assertTrue(reader.meta["capabilities"]["depth_recorded"])
            np.testing.assert_array_equal(
                reader.frames["stereo_sync_residual_ns"], np.full(3, 2_000)
            )
            np.testing.assert_allclose(
                reader.frames["initial_camera_center_m"][:, 0], [0, 1, 2]
            )

    def test_existing_destination_is_never_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "segment"
            destination.mkdir()
            marker = destination / "keep"
            marker.write_text("unchanged")
            with self.assertRaises(FileExistsError):
                SegmentWriter(destination)
            self.assertEqual(marker.read_text(), "unchanged")

    def test_non_monotonic_frame_timestamps_are_rejected(self):
        def mutate(payload):
            payload["frames"]["timestamp_ns"][1] = payload["frames"]["timestamp_ns"][0]
            payload["gnss"]["frame_timestamp_ns"] = payload["frames"]["timestamp_ns"]
            payload["heading"]["frame_timestamp_ns"] = payload["frames"]["timestamp_ns"]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SegmentContractError, "strictly increasing"):
                _write_valid(Path(tmp) / "segment", mutate=mutate)
            self.assertFalse((Path(tmp) / "segment").exists())

    def test_escaping_image_path_is_rejected(self):
        def mutate(payload):
            payload["frames"]["left_image_path"][0] = "../x.jpg"

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SegmentContractError, "relative path"):
                _write_valid(Path(tmp) / "segment", mutate=mutate)

    def test_incomplete_stereo_calibration_is_rejected(self):
        def mutate(payload):
            del payload["calibration"]["T_right_left"]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SegmentContractError, "T_right_left"):
                _write_valid(Path(tmp) / "segment", mutate=mutate)

    def test_source_timestamp_must_match_preserved_raw_sample(self):
        def mutate(payload):
            payload["gnss"]["source_timestamp_ns"][1] += 1

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SegmentContractError, "raw source_index"):
                _write_valid(Path(tmp) / "segment", mutate=mutate)

    def test_single_rtk_does_not_require_heading(self):
        def mutate(payload):
            payload["meta"]["capabilities"]["single_rtk"] = True
            payload["meta"]["capabilities"]["dual_rtk"] = False

        with tempfile.TemporaryDirectory() as tmp:
            reader = _write_valid(
                Path(tmp) / "segment", mutate=mutate, include_heading=False
            )
            self.assertIsNone(reader.observations("heading", required=False))

    def test_external_absolute_trajectory_must_not_be_mislabeled_as_rtk(self):
        def mutate(payload):
            payload["meta"]["capabilities"]["single_rtk"] = False
            payload["meta"]["capabilities"]["dual_rtk"] = False
            payload["meta"]["position_observation"].update(
                type="oracle_groundtruth",
                quantity="left_camera_center",
                sensor_frame_id="left_camera",
            )
            payload["meta"]["initial_pose"]["lever_arm_applied"] = False

        with tempfile.TemporaryDirectory() as tmp:
            reader = _write_valid(
                Path(tmp) / "segment", mutate=mutate, include_heading=False
            )
            self.assertFalse(reader.meta["capabilities"]["single_rtk"])
            self.assertFalse(reader.meta["capabilities"]["dual_rtk"])
            self.assertIsNotNone(reader.observations("gnss"))

    def test_optional_imu_is_validated_when_declared(self):
        imu = {
            "timestamp_ns": np.array([800, 900, 1_000, 1_100], dtype=np.int64),
            "accel_mps2": np.zeros((4, 3)),
            "gyro_radps": np.zeros((4, 3)),
            "orientation_xyzw": np.tile([0.0, 0.0, 0.0, 1.0], (4, 1)),
        }

        def mutate(payload):
            payload["meta"]["capabilities"]["imu_present"] = True

        with tempfile.TemporaryDirectory() as tmp:
            reader = _write_valid(
                Path(tmp) / "segment", mutate=mutate, imu=imu
            )
            self.assertEqual(reader.observations("imu")["accel_mps2"].shape, (4, 3))

    def test_per_frame_observation_ids_must_match_frames(self):
        def mutate(payload):
            payload["gnss"]["frame_id"][1] = 99

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SegmentContractError, "does not match"):
                _write_valid(Path(tmp) / "segment", mutate=mutate)

    def test_position_validity_and_quality_must_agree(self):
        def mutate(payload):
            payload["gnss"]["position_valid"][1] = False

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SegmentContractError, "validity disagree"):
                _write_valid(Path(tmp) / "segment", mutate=mutate)

    def test_normalized_position_fields_are_required(self):
        def mutate(payload):
            del payload["gnss"]["position_quality"]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SegmentContractError, "position_quality"):
                _write_valid(Path(tmp) / "segment", mutate=mutate)

    def test_extrinsic_sigma_frame_must_be_the_camera_frame(self):
        def mutate(payload):
            payload["meta"]["initial_pose"][
                "extrinsic_translation_sigma_frame_id"
            ] = "primary_antenna"

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                SegmentContractError, "expressed in the declared camera frame"
            ):
                _write_valid(Path(tmp) / "segment", mutate=mutate)

    def test_invalid_single_rtk_sample_can_retain_unknown_carrier(self):
        def mutate(payload):
            payload["meta"]["capabilities"]["single_rtk"] = True
            payload["meta"]["capabilities"]["dual_rtk"] = False
            payload["gnss"]["position_valid"][1] = False
            payload["gnss"]["position_quality"][1] = "invalid"
            payload["gnss"]["carrier_status"][1] = -1
            payload["gnss"]["enu_m"][1] = np.nan
            payload["gnss"]["covariance_enu_m2"][1] = np.nan

        with tempfile.TemporaryDirectory() as tmp:
            reader = _write_valid(
                Path(tmp) / "segment", mutate=mutate, include_heading=False
            )
            self.assertFalse(reader.observations("gnss")["position_valid"][1])

    def test_stereo_sync_residual_is_exact(self):
        def mutate(payload):
            payload["frames"]["stereo_sync_residual_ns"][1] += 1

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SegmentContractError, "right minus left"):
                _write_valid(Path(tmp) / "segment", mutate=mutate)

    def test_initial_pose_triplet_is_all_or_nothing(self):
        def mutate(payload):
            del payload["frames"]["pose_valid"]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SegmentContractError, "initial poses require"):
                _write_valid(Path(tmp) / "segment", mutate=mutate)


if __name__ == "__main__":
    unittest.main()
