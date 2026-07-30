import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rtk_splat.cli import cmd_depth
from rtk_splat.colmap_stereo import _load_stereo_calibration
from rtk_splat.configio import load_config
from rtk_splat.bagio import RtkTrack
from rtk_splat.pose_sources import (
    TrajectoryFile,
    _attach_enu,
    make_pose_source,
)


class ConfigTests(unittest.TestCase):
    def test_non_ros_config_does_not_require_bags_or_ublox_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text(f"paths:\n  workdir: {tmp}/work\n")
            loaded = load_config(config)
        self.assertEqual(loaded.paths.bags, [])
        self.assertFalse(hasattr(loaded.paths, "ublox_msgs_dir"))

    def test_workdir_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
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


class PortableSegmentTests(unittest.TestCase):
    def test_stereo_calibration_can_come_from_segment_metadata(self):
        left = {"camera": "left"}
        right = {"camera": "right"}
        meta = {"stereo_calibration": {"left": left, "right": right}}
        cfg = SimpleNamespace(paths=SimpleNamespace(bags=[]))
        self.assertEqual(
            _load_stereo_calibration(meta, cfg, typestore=None),
            (left, right),
        )

    def test_unimplemented_depth_backend_fails_before_writing(self):
        cfg = SimpleNamespace(
            depth=SimpleNamespace(backend="igev"),
            paths=SimpleNamespace(workdir=Path("/path/that/must/not/be/read")),
        )
        with self.assertRaisesRegex(ValueError, "not implemented"):
            cmd_depth(cfg)

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
            "rtk_splat.pose_sources.LocalEnu.to_enu",
            return_value=np.zeros((2, 3)),
        ):
            _attach_enu(track)
        self.assertEqual(track.enu.crs()["type"], "local_ENU")


class TrajectoryFileTests(unittest.TestCase):
    def test_tum_trajectory_has_complete_track_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.txt"
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
        np.testing.assert_allclose(pose.cam_center, [1.5, 2.0, 3.0])

    def test_tum_timestamps_must_increase(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.txt"
            path.write_text(
                "1 1 2 3 0 0 0 1\n"
                "1 2 2 3 0 0 0 1\n"
            )
            cfg = SimpleNamespace(
                pose=SimpleNamespace(trajectory_file=str(path))
            )
            with self.assertRaisesRegex(RuntimeError, "strictly increasing"):
                TrajectoryFile(cfg)


if __name__ == "__main__":
    unittest.main()
