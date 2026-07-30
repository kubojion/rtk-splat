import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rtk_splat.pose_artifacts import pose_fingerprint
from rtk_splat.verify_golden import verify


class GoldenVerifierTests(unittest.TestCase):
    def test_exact_and_acceptance_checks_are_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment = root / "segment"
            pose_dir = segment / "pose_artifacts" / "poses"
            run_dir = root / "runs" / "run"
            pose_dir.mkdir(parents=True)
            run_dir.mkdir(parents=True)

            meta = segment / "segment_meta.json"
            meta.write_text(json.dumps({"n_frames": 2}))
            viewmats = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
            np.save(pose_dir / "viewmats.npy", viewmats)
            fingerprint = pose_fingerprint(viewmats)
            (pose_dir / "quality.json").write_text(json.dumps({
                "n_registered_left": 2,
                "pose_fingerprint": fingerprint,
                "alignment": {"scale": 1.0},
            }))
            np.savez(
                pose_dir / "init_cloud.npz",
                xyz=np.zeros((1, 3)),
                pose_fingerprint=np.asarray(fingerprint),
            )
            (run_dir / "metrics.json").write_text(json.dumps([{
                "psnr_masked": 25.0,
                "psnr_masked_cc": 26.0,
                "lpips_cc": 0.2,
                "ssim": 0.7,
            }]))
            digest = hashlib.sha256(meta.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "artifact_layout": {
                    "recorded_workdir": str(root),
                    "pose_artifact": "poses",
                    "run_name": "run",
                },
                "compact_files": [{
                    "path": "segment/segment_meta.json",
                    "sha256": digest,
                }],
                "expected": {
                    "segment": {"n_frames": 2},
                    "pose": {"scale_range": [0.98, 1.02]},
                    "acceptance_gates": {
                        "psnr_masked_min": 24.0,
                        "psnr_masked_cc_min": 25.0,
                        "lpips_cc_max": 0.3,
                        "ssim_min": 0.6,
                    },
                },
            }))

            before = sorted(str(path.relative_to(root))
                            for path in root.rglob("*"))
            checks = verify(manifest, root, mode="exact")
            after = sorted(str(path.relative_to(root))
                           for path in root.rglob("*"))
            self.assertTrue(all(passed for passed, _ in checks))
            self.assertEqual(before, after)

    def test_hash_mismatch_fails_exact_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "artifact_layout": {
                    "recorded_workdir": str(root),
                    "pose_artifact": "missing",
                    "run_name": "missing",
                },
                "compact_files": [{
                    "path": "missing.json",
                    "sha256": "0" * 64,
                }],
                "expected": {
                    "segment": {"n_frames": 1},
                    "pose": {"scale_range": [0.98, 1.02]},
                    "acceptance_gates": {},
                },
            }))
            checks = verify(manifest, root, mode="exact")
            self.assertTrue(any(not passed for passed, _ in checks))


if __name__ == "__main__":
    unittest.main()
