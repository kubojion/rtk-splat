import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
COMMON_RUNNER = REPOSITORY / "scripts/experiments/_ab_common.sh"


class FeatureReportVerifierTests(unittest.TestCase):
    def _verify(self, commands, profile: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = (
                root
                / "arms/gpu/frontend_artifacts"
                / "feature-profile-ab-gpu-frontend"
                / "stage_reports/features.json"
            )
            report.parent.mkdir(parents=True)
            report.write_text(
                json.dumps(
                    {
                        "profile": profile,
                        "n_images": 2688,
                        "n_keypoints": 10,
                        "n_descriptors": 10,
                        "command": commands,
                    }
                ),
                encoding="utf-8",
            )
            environment = dict(
                os.environ,
                RTK_SPLAT_EXPERIMENT_ROOT=str(root),
                RTK_SPLAT_PYTHON=sys.executable,
            )
            return subprocess.run(
                [
                    "bash",
                    "-c",
                    (
                        'AB_KIND="feature-profile-ab"; '
                        'source "$1"; ab_verify_features gpu "$2"'
                    ),
                    "bash",
                    str(COMMON_RUNNER),
                    profile,
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )

    def test_accepts_every_command_in_a_multi_sensor_report(self):
        gpu = [
            "colmap",
            "feature_extractor",
            "--FeatureExtraction.use_gpu",
            "1",
        ]
        completed = self._verify([gpu, gpu], "gpu")
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_rejects_a_mismatched_command_in_any_sensor(self):
        gpu = [
            "colmap",
            "feature_extractor",
            "--FeatureExtraction.use_gpu",
            "1",
        ]
        cpu = [
            "colmap",
            "feature_extractor",
            "--FeatureExtraction.use_gpu",
            "0",
            "--SiftExtraction.estimate_affine_shape",
            "1",
            "--SiftExtraction.domain_size_pooling",
            "1",
        ]
        completed = self._verify([gpu, cpu], "gpu")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("feature command 1", completed.stderr)


if __name__ == "__main__":
    unittest.main()
