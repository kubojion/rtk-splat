import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from diagnostics.compare_poses import (
    compare_pose_artifacts,
    write_report_noreplace,
)


def _artifact(path: Path, offset: float, angle_deg: float) -> Path:
    path.mkdir()
    count = 4
    viewmats = np.repeat(np.eye(4)[None], count, axis=0)
    angle = np.deg2rad(angle_deg)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    viewmats[:, :3, :3] = rotation
    centers = np.column_stack(
        (np.arange(count, dtype=float) + offset, np.zeros((count, 2)))
    )
    viewmats[:, :3, 3] = -np.einsum(
        "nij,nj->ni", viewmats[:, :3, :3], centers
    )
    np.save(path / "viewmats.npy", viewmats)
    np.save(path / "frame_ids.npy", np.arange(count, dtype=np.int64))
    np.save(
        path / "timestamps_ns.npy",
        np.arange(count, dtype=np.int64) + 100,
    )
    (path / "quality.json").write_text(json.dumps({"passed": True}))
    return path


class ComparePosesTests(unittest.TestCase):
    def test_report_is_frame_exact_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = _artifact(root / "reference", 0.0, 0.0)
            candidate = _artifact(root / "candidate", 0.1, 2.0)
            report = compare_pose_artifacts(reference, candidate)
            self.assertEqual(report["n_common_frames"], 4)
            self.assertAlmostEqual(
                report["translation_delta_m"]["median"], 0.1
            )
            self.assertAlmostEqual(
                report["rotation_delta_deg"]["median"], 2.0
            )
            output = write_report_noreplace(root / "report.json", report)
            self.assertEqual(json.loads(output.read_text())["n_common_frames"], 4)
            with self.assertRaises(FileExistsError):
                write_report_noreplace(output, report)

    def test_different_frame_ids_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = _artifact(root / "reference", 0.0, 0.0)
            candidate = _artifact(root / "candidate", 0.0, 0.0)
            np.save(candidate / "frame_ids.npy", np.arange(4) + 1)
            with self.assertRaisesRegex(ValueError, "frame IDs differ"):
                compare_pose_artifacts(reference, candidate)


if __name__ == "__main__":
    unittest.main()
