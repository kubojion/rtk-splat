import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from test_frontend_planning import _segment
from workflows.depth import derive_sgbm_depth


class DepthWorkflowTests(unittest.TestCase):
    def test_depth_is_a_new_segment_and_source_remains_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _segment(root / "source")
            source_frames_before = (source.root / "frames.npz").read_bytes()
            cfg = SimpleNamespace(
                depth=SimpleNamespace(
                    min_z_m=1.0,
                    max_z_m=10.0,
                    sgbm=SimpleNamespace(),
                )
            )
            depth = np.full((12, 16), 2.0, dtype=np.float32)
            valid = np.ones((12, 16), dtype=bool)
            with (
                mock.patch("workflows.depth.make_sgbm", return_value=object()),
                mock.patch(
                    "workflows.depth.depth_from_pair",
                    return_value=(depth, valid),
                ),
            ):
                derived = derive_sgbm_depth(
                    source.root, root / "derived", cfg
                )

            self.assertTrue(derived.meta["capabilities"]["depth_computed"])
            self.assertFalse(source.meta["capabilities"]["depth_computed"])
            self.assertEqual(
                (source.root / "frames.npz").read_bytes(), source_frames_before
            )
            self.assertTrue(
                (derived.root / str(derived.frames["left_image_path"][0])).is_symlink()
            )
            self.assertEqual(
                derived.meta["depth_observation"]["quantity"], "optical_z"
            )
            with self.assertRaises(FileExistsError):
                derive_sgbm_depth(source.root, derived.root, cfg)


if __name__ == "__main__":
    unittest.main()
