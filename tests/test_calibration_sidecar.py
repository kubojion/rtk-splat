"""Backend-neutral source resolution for the metric-integrity sidecar."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rtk_splat.calibration_sidecar import resolve_raw_visual_source


class RawVisualSourceTests(unittest.TestCase):
    @staticmethod
    def _touch_inputs(root: Path, backend_dir: str) -> None:
        model = root / backend_dir / "models_text" / "0"
        model.mkdir(parents=True)
        (model / "frames.txt").write_text("# frames\n")
        (model / "rigs.txt").write_text("# rigs\n")
        (root / backend_dir / "database.db").write_bytes(b"database")

    def test_resolves_incremental_artifact_relative_to_current_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "pose"
            self._touch_inputs(root, "colmap")
            source = resolve_raw_visual_source(root, {
                "selected_model": "/old/machine/models_text/0",
                "alignment": {
                    "rotation_visual_world_to_enu": np.eye(3).tolist(),
                },
            })
            self.assertEqual(source.backend, "incremental_stereo")
            self.assertEqual(
                source.model, root / "colmap" / "models_text" / "0")
            self.assertEqual(
                source.database, root / "colmap" / "database.db")

    def test_resolves_global_mapper_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "pose"
            self._touch_inputs(root, "global")
            source = resolve_raw_visual_source(root, {
                "model_stats": {
                    "path": "/old/machine/sparse_global/0",
                },
                "fixed_scale_alignment": {
                    "rotation_visual_world_to_enu": np.eye(3).tolist(),
                },
            })
            self.assertEqual(source.backend, "global_mapper")
            self.assertEqual(
                source.model, root / "global" / "models_text" / "0")
            self.assertEqual(
                source.database, root / "global" / "database.db")

    def test_rejects_invalid_alignment_rotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "pose"
            self._touch_inputs(root, "global")
            with self.assertRaisesRegex(ValueError, "valid rotation"):
                resolve_raw_visual_source(root, {
                    "model_stats": {"path": "/old/sparse_global/0"},
                    "fixed_scale_alignment": {
                        "rotation_visual_world_to_enu":
                            np.diag([1.0, 1.0, 2.0]).tolist(),
                    },
                })


if __name__ == "__main__":
    unittest.main()
