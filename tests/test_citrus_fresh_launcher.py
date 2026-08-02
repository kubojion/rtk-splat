import subprocess
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPOSITORY
    / "scripts"
    / "experiments"
    / "citrusfarm_pose_prior_fresh_l2.sh"
)


class CitrusFreshPosePriorLauncherTests(unittest.TestCase):
    def test_shell_syntax_and_help_are_safe(self):
        syntax = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        help_result = subprocess.run(
            ["bash", str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("preflight", help_result.stdout)
        self.assertIn("approximately 4-10 hours", help_result.stdout)
        self.assertNotIn("tmux", help_result.stdout)

    def test_launcher_freezes_inputs_and_has_only_one_workflow_stage(self):
        contents = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("citrusfarm_05_13d_uturn.yaml", contents)
        self.assertIn("citrus-05-13d-543-735-global-bounded-v1", contents)
        self.assertIn("citrus-05-13d-543-735-rtk-l2-v1", contents)
        self.assertIn(
            "citrus-05-13d-543-735-pose-prior-fresh-l2-v1", contents
        )
        self.assertIn(
            "0f48cab4aa671569ef2a4be6dc4ecadd2027b08f8fa9128430b2661aad8a9fd9",
            contents,
        )
        self.assertEqual(contents.count('"${CLI[@]}" backend-refine-rtk'), 1)
        self.assertNotIn('"${CLI[@]}" backend-export', contents)
        self.assertNotIn('"${CLI[@]}" cloud', contents)
        self.assertNotIn('"${CLI[@]}" train', contents)

    def test_fresh_mode_is_fail_closed_and_comparison_is_terminal(self):
        contents = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--prior-position-loss trivial", contents)
        self.assertIn("--initialization-mode fresh", contents)
        self.assertIn('require("--input_path" not in fresh_command', contents)
        self.assertIn("rtk_splat.diagnostics.initialization_ab", contents)
        self.assertIn("--expected-frames 1495", contents)
        self.assertIn("--expected-evaluation-priors 1495", contents)
        self.assertIn("--expected-calibration-priors 897", contents)
        self.assertIn("--expected-holdout-priors 598", contents)
        self.assertIn("MINIMUM_FREE_KIB=$((20 * 1024 * 1024))", contents)
        self.assertIn(
            "MINIMUM_AVAILABLE_MEMORY_KIB=$((16 * 1024 * 1024))", contents
        )
        self.assertIn("flock -n 9", contents)


if __name__ == "__main__":
    unittest.main()
