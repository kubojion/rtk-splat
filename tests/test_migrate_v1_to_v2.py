import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rtk_splat.adapters.migrate_v1_to_v2 import MigrationError, migrate


def _make_v1(root: Path, n: int = 3) -> tuple[Path, Path]:
    (root / "images").mkdir(parents=True)
    (root / "depth").mkdir()
    sidecar_dir = root / "pose_artifacts" / "colmap_stereo"
    sidecar_dir.mkdir(parents=True)
    for frame_id in range(n):
        (root / "images" / f"left_{frame_id:06d}.jpg").write_bytes(b"left")
        (root / "images" / f"right_{frame_id:06d}.jpg").write_bytes(b"right")
        (root / "depth" / f"{frame_id:06d}.npz").write_bytes(b"depth")

    centers = np.column_stack((np.arange(n), np.zeros(n), np.zeros(n))).astype(
        np.float32
    )
    viewmats = np.repeat(np.eye(4, dtype=np.float32)[None], n, axis=0)
    viewmats[:, 0, 3] = -centers[:, 0]
    np.save(root / "viewmats.npy", viewmats)
    np.save(root / "cam_centers.npy", centers)
    (root / "segment_meta.json").write_text(
        json.dumps(
            {
                "n_frames": n,
                "world_origin": {
                    "lat0": 52.0,
                    "lon0": 16.0,
                    "alt0": 100.0,
                },
                "pose_source": "rtk_dual_antenna",
            }
        )
    )
    (root / "manifest.json").write_text(
        json.dumps({"train": [0, 2], "val": [1], "test": []})
    )
    camera = {
        "width": 8,
        "height": 6,
        "fx": 100.0,
        "fy": 100.0,
        "cx": 4.0,
        "cy": 3.0,
        "distortion_model": "rational_polynomial",
        "d": [0.0] * 8,
        "r": np.eye(3).tolist(),
        "p": [[100.0, 0.0, 4.0, 0.0], [0.0, 100.0, 3.0, 0.0], [0, 0, 1, 0]],
    }
    right = dict(camera)
    right["p"] = [
        [100.0, 0.0, 4.0, -12.0],
        [0.0, 100.0, 3.0, 0.0],
        [0, 0, 1, 0],
    ]
    (sidecar_dir / "sidecar_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "baseline_m_camera_info": 0.12,
                "left_calibration": camera,
                "right_calibration": right,
                "coordinate_conventions": {"input_images": "rectified OpenCV"},
            }
        )
    )

    frame_ns = np.arange(n, dtype=np.int64) * 100_000_000 + 1_000_000_000
    raw_ns = (
        np.arange(2 * n - 1, dtype=np.int64) * 50_000_000 + 980_000_000
    )
    raw_count = len(raw_ns)
    flags = np.tile(
        np.array([True, True, True, True, False, False, True, False]),
        (raw_count, 1),
    )
    observations = {
        "stereo_frame_id": np.arange(n, dtype=np.int64),
        "stereo_left_header_ns": frame_ns,
        "stereo_left_log_ns": frame_ns + 5_000,
        "stereo_right_header_ns": frame_ns + 2_000,
        "stereo_right_log_ns": frame_ns + 7_000,
        "stereo_delta_ns": np.full(n, 2_000, dtype=np.int64),
        "stereo_left_sha256": np.full(
            n, hashlib.sha256(b"left").hexdigest(), dtype="<U64"
        ),
        "stereo_right_sha256": np.full(
            n, hashlib.sha256(b"right").hexdigest(), dtype="<U64"
        ),
        "fix_header_ns": raw_ns,
        "fix_log_ns": raw_ns + 10_000,
        "fix_geodetic": np.column_stack(
            (
                np.full(raw_count, 52.0),
                np.full(raw_count, 16.0),
                np.full(raw_count, 100.0),
            )
        ),
        "fix_enu_m": np.column_stack(
            (np.arange(raw_count) * 0.1, np.zeros(raw_count), np.zeros(raw_count))
        ),
        "fix_covariance_enu_m2": np.repeat(
            (np.eye(3) * 0.0004)[None], raw_count, axis=0
        ),
        "fix_status": np.ones(raw_count, dtype=np.int16),
        "fix_service": np.ones(raw_count, dtype=np.uint16),
        "fix_frame_id": np.full(raw_count, "antenna", dtype="<U16"),
        "fix_covariance_type": np.full(raw_count, 3, dtype=np.uint8),
        "pvt_header_ns": raw_ns + 1_000_000,
        "pvt_log_ns": raw_ns + 1_010_000,
        "pvt_carrier_solution": np.full(raw_count, 2, dtype=np.uint8),
        "pvt_status_flags": np.tile(
            np.array([True, True, False, True, True, True]), (raw_count, 1)
        ),
        "relpos_header_ns": raw_ns + 2_000_000,
        "relpos_log_ns": raw_ns + 2_010_000,
        "relpos_ned_m": np.tile([1.2, 0.1, -0.02], (raw_count, 1)),
        "relpos_enu_m": np.tile([0.1, 1.2, 0.02], (raw_count, 1)),
        "relpos_accuracy_heading_deg": np.full(raw_count, 0.4),
        "relpos_carrier_solution": np.full(raw_count, 2, dtype=np.uint8),
        "relpos_flags": flags,
        "relpos_itow_ms": np.arange(raw_count, dtype=np.uint32),
        "baseline_good_for_calibration": np.ones(raw_count, dtype=bool),
    }
    observations_path = root.parent / "observations.npz"
    np.savez_compressed(observations_path, **observations)
    config_path = root.parent / "config.yaml"
    config_path.write_text(
        "pose:\n"
        "  cam_forward_m: 3.18\n"
        "  baseline_m: 0.12\n"
        "calibration:\n"
        "  physical_geometry:\n"
        "    camera_frame: left_optical\n"
    )
    return observations_path, config_path


class MigrationTests(unittest.TestCase):
    def test_migration_roundtrip_preserves_evidence_and_links_bulk_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "v1"
            observations, config = _make_v1(source)
            destination = root / "v2"
            reader = migrate(source, destination, observations, config=config)
            reader.validate()

            self.assertTrue((destination / "images").is_symlink())
            self.assertTrue((destination / "depth").is_symlink())
            self.assertEqual(
                (destination / "images").resolve(), (source / "images").resolve()
            )
            np.testing.assert_array_equal(
                reader.frames["stereo_sync_residual_ns"],
                np.full(3, 2_000),
            )
            gnss = reader.observations("gnss")
            heading = reader.observations("heading")
            assert gnss is not None and heading is not None
            self.assertEqual(len(gnss["frame_id"]), 3)
            self.assertEqual(len(gnss["raw_timestamp_ns"]), 5)
            self.assertEqual(len(gnss["pvt_header_ns"]), 5)
            np.testing.assert_array_equal(gnss["carrier_status"], [2, 2, 2])
            self.assertTrue(gnss["position_valid"].all())
            self.assertEqual(
                set(gnss["position_quality"].tolist()), {"rtk_fixed"}
            )
            np.testing.assert_allclose(
                heading["baseline_ned_m"][0], [1.2, 0.1, -0.02]
            )
            np.testing.assert_allclose(
                heading["acc_heading_rad"], np.deg2rad(0.4)
            )
            self.assertTrue(heading["valid"].all())
            self.assertAlmostEqual(
                reader.calibration["T_right_left"][0][3], -0.12
            )
            self.assertEqual(
                reader.calibration["rough_sensor_geometry"]["pose_parameters"][
                    "cam_forward_m"
                ],
                3.18,
            )
            self.assertEqual(
                reader.calibration["transform_conventions"]["T_right_left"],
                "right_from_left",
            )
            self.assertEqual(
                reader.meta["position_observation"]["quantity"],
                "antenna_phase_center",
            )
            self.assertEqual(
                reader.meta["heading_observation"]["vector"],
                "primary_to_secondary",
            )

    def test_existing_destination_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "v1"
            observations, _ = _make_v1(source)
            destination = root / "v2"
            destination.mkdir()
            marker = destination / "keep"
            marker.write_text("unchanged")
            with self.assertRaises(FileExistsError):
                migrate(source, destination, observations)
            self.assertEqual(marker.read_text(), "unchanged")

    def test_association_outside_tolerance_fails_without_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "v1"
            observations, _ = _make_v1(source)
            destination = root / "v2"
            with self.assertRaisesRegex(MigrationError, "exceed"):
                migrate(
                    source,
                    destination,
                    observations,
                    association_tolerance_ns=1,
                )
            self.assertFalse(destination.exists())

    def test_mismatched_stereo_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "v1"
            observations, _ = _make_v1(source)
            with np.load(observations, allow_pickle=False) as archive:
                values = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            values["stereo_right_log_ns"] = values["stereo_right_log_ns"][:-1]
            np.savez_compressed(observations, **values)
            destination = root / "v2"
            with self.assertRaisesRegex(MigrationError, "count disagrees"):
                migrate(source, destination, observations)
            self.assertFalse(destination.exists())

    def test_destination_inside_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "v1"
            observations, _ = _make_v1(source)
            with self.assertRaisesRegex(MigrationError, "inside"):
                migrate(source, source / "v2", observations)

    def test_recorded_image_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "v1"
            observations, _ = _make_v1(source)
            (source / "images" / "left_000001.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(MigrationError, "sha256 mismatch"):
                migrate(source, root / "v2", observations)
            self.assertFalse((root / "v2").exists())

    def test_float_timestamp_evidence_is_rejected_instead_of_cast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "v1"
            observations, _ = _make_v1(source)
            with np.load(observations, allow_pickle=False) as archive:
                values = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            values["stereo_left_header_ns"] = values[
                "stereo_left_header_ns"
            ].astype(np.float64)
            np.savez_compressed(observations, **values)
            with self.assertRaisesRegex(MigrationError, "integer nanoseconds"):
                migrate(source, root / "v2", observations)
            self.assertFalse((root / "v2").exists())

    def test_validation_failure_removes_unpublished_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "v1"
            observations, _ = _make_v1(source)
            (source / "manifest.json").write_text(
                json.dumps({"train": [0], "val": [1], "test": []})
            )
            destination = root / "v2"
            with self.assertRaisesRegex(ValueError, "cover every"):
                migrate(source, destination, observations)
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".v2.writing-*")), [])


if __name__ == "__main__":
    unittest.main()
