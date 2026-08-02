import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import numpy as np

from rtk_splat.core.pose_artifacts import pose_fingerprint
from rtk_splat.core.verify_golden import (
    discover_default_manifest,
    main,
    verify,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class GoldenVerifierTests(unittest.TestCase):
    def test_default_manifest_is_discovered_only_in_source_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src" / "rtk_splat" / "core" / "verify_golden.py"
            source.parent.mkdir(parents=True)
            source.write_text("")
            self.assertIsNone(discover_default_manifest(source))

            (root / "pyproject.toml").write_text("[project]\n")
            manifest = (
                root
                / "docs"
                / "experiments"
                / "golden"
                / "headland_stereo_ba.json"
            )
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}")
            self.assertEqual(discover_default_manifest(source), manifest)

            installed = (
                root
                / ".venv"
                / "lib"
                / "python3.10"
                / "site-packages"
                / "rtk_splat"
                / "core"
                / "verify_golden.py"
            )
            installed.parent.mkdir(parents=True)
            installed.write_text("")
            self.assertIsNone(discover_default_manifest(installed))

    def test_installed_verifier_requests_manifest_without_traceback(self):
        stderr = io.StringIO()
        with mock.patch(
            "rtk_splat.core.verify_golden.discover_default_manifest",
            return_value=None,
        ), redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main([])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--manifest is required outside", stderr.getvalue())

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
                "schema_version": 2,
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
                    "pose": {
                        "pose_fingerprint": fingerprint,
                        "scale_checks": [{
                            "name": "test scale",
                            "quality_path": "alignment.scale",
                            "expected": 1.0,
                            "range": [0.98, 1.02],
                        }],
                    },
                    "metrics": {"psnr_masked": 25.0},
                    "exact_metric_fields": ["psnr_masked"],
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
                "schema_version": 2,
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
                    "pose": {
                        "pose_fingerprint": "0" * 64,
                        "scale_checks": [{
                            "name": "test scale",
                            "quality_path": "alignment.scale",
                            "expected": 1.0,
                            "range": [0.98, 1.02],
                        }],
                    },
                    "metrics": {},
                    "exact_metric_fields": [],
                    "acceptance_gates": {},
                },
            }))
            checks = verify(manifest, root, mode="exact")
            self.assertTrue(any(not passed for passed, _ in checks))

    def test_global_mapper_uses_declared_scale_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment = root / "segment"
            pose_dir = segment / "pose_artifacts" / "global"
            run_dir = root / "runs" / "run"
            pose_dir.mkdir(parents=True)
            run_dir.mkdir(parents=True)

            (segment / "segment_meta.json").write_text(
                json.dumps({"n_frames": 1})
            )
            viewmats = np.eye(4, dtype=np.float32)[None]
            np.save(pose_dir / "viewmats.npy", viewmats)
            fingerprint = pose_fingerprint(viewmats)
            (pose_dir / "quality.json").write_text(json.dumps({
                "n_registered_left": 1,
                "pose_fingerprint": fingerprint,
                "fixed_scale_alignment": {"scale_applied": 1},
                "sim3_diagnostic_only": {"scale": 0.984},
            }))
            np.savez(
                pose_dir / "init_cloud.npz",
                xyz=np.zeros((1, 3)),
                pose_fingerprint=np.asarray(fingerprint),
            )
            metrics = {
                "psnr_masked": 25.0,
                "psnr_masked_cc": 26.0,
                "lpips_cc": 0.2,
                "ssim": 0.7,
            }
            (run_dir / "metrics.json").write_text(json.dumps([metrics]))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": 2,
                "artifact_layout": {
                    "recorded_workdir": str(root),
                    "pose_artifact": "global",
                    "run_name": "run",
                },
                "compact_files": [],
                "expected": {
                    "segment": {"n_frames": 1},
                    "pose": {
                        "pose_fingerprint": fingerprint,
                        "scale_checks": [
                            {
                                "name": "fixed",
                                "quality_path":
                                    "fixed_scale_alignment.scale_applied",
                                "expected": 1,
                                "range": [1, 1],
                            },
                            {
                                "name": "diagnostic",
                                "quality_path":
                                    "sim3_diagnostic_only.scale",
                                "expected": 0.984,
                                "range": [0.98, 1.02],
                            },
                        ],
                    },
                    "metrics": metrics,
                    "exact_metric_fields": list(metrics),
                    "acceptance_gates": {
                        "psnr_masked_min": 24.0,
                        "psnr_masked_cc_min": 25.0,
                        "lpips_cc_max": 0.3,
                        "ssim_min": 0.6,
                    },
                },
            }))

            checks = verify(manifest, root, mode="exact")
            self.assertTrue(all(passed for passed, _ in checks), checks)

    def test_exact_mode_rejects_changed_recorded_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment = root / "segment"
            pose_dir = segment / "pose_artifacts" / "poses"
            run_dir = root / "runs" / "run"
            pose_dir.mkdir(parents=True)
            run_dir.mkdir(parents=True)
            (segment / "segment_meta.json").write_text(
                json.dumps({"n_frames": 1})
            )
            viewmats = np.eye(4, dtype=np.float32)[None]
            np.save(pose_dir / "viewmats.npy", viewmats)
            fingerprint = pose_fingerprint(viewmats)
            (pose_dir / "quality.json").write_text(json.dumps({
                "n_registered_left": 1,
                "pose_fingerprint": fingerprint,
                "alignment": {"scale": 1.0},
            }))
            np.savez(
                pose_dir / "init_cloud.npz",
                xyz=np.zeros((1, 3)),
                pose_fingerprint=np.asarray(fingerprint),
            )
            (run_dir / "metrics.json").write_text(json.dumps([{
                "psnr_masked": 24.9,
                "psnr_masked_cc": 26.0,
                "lpips_cc": 0.2,
                "ssim": 0.7,
            }]))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": 2,
                "artifact_layout": {
                    "recorded_workdir": str(root),
                    "pose_artifact": "poses",
                    "run_name": "run",
                },
                "compact_files": [],
                "expected": {
                    "segment": {"n_frames": 1},
                    "pose": {
                        "pose_fingerprint": fingerprint,
                        "scale_checks": [{
                            "name": "scale",
                            "quality_path": "alignment.scale",
                            "expected": 1.0,
                            "range": [0.98, 1.02],
                        }],
                    },
                    "metrics": {"psnr_masked": 25.0},
                    "exact_metric_fields": ["psnr_masked"],
                    "acceptance_gates": {
                        "psnr_masked_min": 24.0,
                        "psnr_masked_cc_min": 25.0,
                        "lpips_cc_max": 0.3,
                        "ssim_min": 0.6,
                    },
                },
            }))

            checks = verify(manifest, root, mode="exact")
            self.assertTrue(any(
                not passed and "exact metric psnr_masked" in description
                for passed, description in checks
            ))

    def test_committed_manifests_freeze_accepted_headland_results(self):
        frozen = {
            "headland_stereo_ba.json": {
                "pose": "701d917fdef9b5092499debaeedfa9d24852c1bd461b4816cc476af0b37234af",
                "pose_file": "5247eb98b9a07423b86f28ca470c3a82d4baf23b9d3a1c38b7e5c33a2b96b18d",
                "metrics_file": "8c4563f37943a4d09ceb52dba74f6e970f37a1f0959e6cbbfa42d2c6bdc04174",
                "scale_values": [0.9857944885226491],
                "metrics": {
                    "psnr_masked": 24.444888046809606,
                    "psnr_masked_cc": 25.847966818582442,
                    "lpips_cc": 0.31866925883860814,
                    "ssim": 0.5944193653052762,
                },
            },
            "headland_global_mapper.json": {
                "pose": "8129ff17300a84061cb8df4eb9e0a707c4da63f943bff690f4e1a116150843c7",
                "pose_file": "9efbb66d62af4e8c01d94f9d500be781c057318985dd6b99cdb4b9f1d14a70fa",
                "metrics_file": "85697bb1cb4bb307808a3c3d484c18d96563e0d4afd94161e300c86530f30883",
                "scale_values": [1, 0.9846374983745549],
                "metrics": {
                    "psnr_masked": 24.458305427006312,
                    "psnr_masked_cc": 25.84177161398388,
                    "lpips_cc": 0.3196603652267229,
                    "ssim": 0.5904194045634497,
                },
            },
        }
        golden_dir = REPOSITORY_ROOT / "docs" / "experiments" / "golden"
        for filename, expected in frozen.items():
            with self.subTest(filename=filename):
                manifest = json.loads((golden_dir / filename).read_text())
                self.assertEqual(manifest["schema_version"], 2)
                self.assertEqual(
                    manifest["expected"]["segment"]["n_frames"], 1344
                )
                self.assertEqual(
                    manifest["expected"]["pose"]["pose_fingerprint"],
                    expected["pose"],
                )
                self.assertEqual(
                    [
                        check["expected"]
                        for check in manifest["expected"]["pose"]["scale_checks"]
                    ],
                    expected["scale_values"],
                )
                for metric, value in expected["metrics"].items():
                    self.assertEqual(
                        manifest["expected"]["metrics"][metric], value
                    )
                hashes = {
                    item["path"]: item["sha256"]
                    for item in manifest["compact_files"]
                }
                pose_path = (
                    "segment/pose_artifacts/"
                    f"{manifest['artifact_layout']['pose_artifact']}/"
                    "viewmats.npy"
                )
                metrics_path = (
                    f"runs/{manifest['artifact_layout']['run_name']}/"
                    "metrics.json"
                )
                self.assertEqual(hashes[pose_path], expected["pose_file"])
                self.assertEqual(
                    hashes[metrics_path], expected["metrics_file"]
                )


if __name__ == "__main__":
    unittest.main()
