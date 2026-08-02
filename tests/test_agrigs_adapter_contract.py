import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from rtk_splat.adapters import agrigs
from rtk_splat.core.segment import SegmentReader


class AgrigsAdapterContractTests(unittest.TestCase):
    _synthetic_geodesy = SimpleNamespace(
        ecef2geodetic=lambda x, y, z: (0.0, 0.0, float(x) - 6_378_137.0),
        ecef2enu=lambda x, y, z, lat, lon, alt: (
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
            np.asarray(x, dtype=float) - 6_378_137.0 - float(alt),
        ),
    )

    @staticmethod
    def _write_split(
        dataset: Path,
        split: str,
        seconds: tuple[int, int],
        ecef_y_m: tuple[float, float],
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
                for second, east in zip(seconds, ecef_y_m)
            ],
        ]
        (root / "groundtruth_cam_1.csv").write_text(
            "\n".join(rows) + "\n"
        )
        for index, second in enumerate(seconds):
            name = f"{second}-000000123"
            image = np.full(
                (6, 8, 3), (20 + 10 * index, 80, 140), dtype=np.uint8
            )
            depth_mm = np.full((6, 8), 2_000, dtype=np.uint16)
            depth_mm[0, 0] = 0
            self_written = cv2.imwrite(str(rgb / f"{name}.jpg"), image)
            depth_written = cv2.imwrite(
                str(depth / f"{name}.png"), depth_mm
            )
            if not self_written or not depth_written:
                raise RuntimeError("failed to create synthetic AgriGS inputs")

    def test_ingest_writes_valid_v2_without_claiming_rtk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "agrigs"
            self._write_split(dataset, "train", (10, 11), (0.0, 0.2))
            self._write_split(dataset, "val", (20, 21), (0.4, 0.6))
            segment = root / "segment"
            cfg = SimpleNamespace(
                agrigs=SimpleNamespace(
                    dataset_dir=str(dataset),
                    camera="cam_1",
                    intrinsic=[10.0, 10.0, 4.0, 3.0],
                    distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                depth=SimpleNamespace(min_z_m=0.5, max_z_m=3.0),
            )

            with patch.object(agrigs, "pymap3d", self._synthetic_geodesy):
                returned = agrigs.ingest_config_v2(cfg, segment)

            self.assertEqual(returned.root, segment)
            reader = SegmentReader(segment).validate()
            frames = reader.frames
            np.testing.assert_array_equal(
                frames["timestamp_ns"],
                np.array(
                    [
                        10_000_000_123,
                        11_000_000_123,
                        20_000_000_123,
                        21_000_000_123,
                    ],
                    dtype=np.int64,
                ),
            )
            self.assertEqual(reader.manifest["train"], [0, 1])
            self.assertEqual(reader.manifest["val"], [2, 3])
            self.assertEqual(reader.manifest["test"], [])
            self.assertFalse(reader.meta["capabilities"]["single_rtk"])
            self.assertFalse(reader.meta["capabilities"]["dual_rtk"])
            self.assertTrue(reader.meta["capabilities"]["rgbd"])
            self.assertTrue(reader.meta["capabilities"]["depth_recorded"])
            self.assertTrue(reader.meta["capabilities"]["images_rectified"])
            self.assertEqual(
                reader.meta["position_evidence"]["type"],
                "oracle_groundtruth",
            )
            self.assertEqual(
                reader.meta["position_evidence"]["position_provenance"],
                "oracle groundtruth",
            )
            configuration = reader.meta["provenance"]["configuration"]
            self.assertEqual(
                configuration["effective_config"]["agrigs"]["camera"],
                "cam_1",
            )
            self.assertEqual(len(configuration["effective_config_sha256"]), 64)

            gnss = reader.observations("gnss")
            assert gnss is not None
            np.testing.assert_allclose(
                gnss["enu_m"], frames["initial_camera_center_m"]
            )
            self.assertTrue(np.isnan(gnss["covariance_enu_m2"]).all())
            np.testing.assert_array_equal(
                gnss["fix_status"], np.full(4, -1, dtype=np.int16)
            )
            np.testing.assert_array_equal(
                gnss["carrier_status"], np.full(4, -1, dtype=np.int16)
            )
            self.assertTrue(gnss["position_valid"].all())
            self.assertEqual(
                set(gnss["position_quality"].tolist()), {"oracle"}
            )
            self.assertEqual(
                set(gnss["position_provenance"].tolist()),
                {"oracle groundtruth"},
            )
            self.assertEqual(
                set(gnss["evidence_type"].tolist()),
                {"oracle_groundtruth"},
            )
            self.assertEqual(len(gnss["raw_timestamp_ns"]), 4)
            np.testing.assert_array_equal(
                gnss["source_timestamp_ns"],
                gnss["raw_timestamp_ns"][gnss["source_index"]],
            )
            np.testing.assert_allclose(gnss["interpolation_alpha"], 0.0)
            self.assertEqual(
                reader.meta["position_observation"]["type"],
                "oracle_groundtruth",
            )
            self.assertEqual(
                reader.meta["depth_observation"]["quantity"], "optical_z"
            )
            self.assertEqual(
                reader.calibration["cameras"]["left"]["model"], "PINHOLE"
            )
            self.assertFalse((segment / "viewmats.npy").exists())
            self.assertFalse((segment / "cam_centers.npy").exists())

            output = root / "verification.png"
            agrigs.verify(segment, output, pair_gap=1)
            self.assertTrue(output.is_file())

    def test_existing_destination_is_not_modified(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "segment"
            destination.mkdir()
            marker = destination / "keep"
            marker.write_text("unchanged")
            with self.assertRaises(FileExistsError):
                agrigs.ingest(
                    Path(temporary) / "unused",
                    "cam_1",
                    intrinsic=[10.0, 10.0, 4.0, 3.0],
                    distortion=[0.0] * 5,
                    out_seg=destination,
                    min_z=0.5,
                    max_z=3.0,
                )
            self.assertEqual(marker.read_text(), "unchanged")

    def test_pose_interpolation_rejects_extrapolation(self):
        timestamps = np.array([1_000, 2_000], dtype=np.int64)
        positions = np.zeros((2, 3), dtype=np.float64)
        rotations = Rotation.identity(2)
        with self.assertRaisesRegex(ValueError, "outside"):
            agrigs._interp_poses(
                timestamps,
                positions,
                rotations,
                np.array([999], dtype=np.int64),
            )


if __name__ == "__main__":
    unittest.main()
