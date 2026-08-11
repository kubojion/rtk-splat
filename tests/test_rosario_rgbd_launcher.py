import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPOSITORY
    / "scripts/runs/rosario_v2_sequence5_ppk_140_250_rgbd_diagnostic_overnight.sh"
)


class RosarioRgbdLauncherTests(unittest.TestCase):
    def test_shell_syntax_and_plan_are_non_mutating(self):
        syntax = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        plan = subprocess.run(
            ["bash", str(SCRIPT), "plan"], capture_output=True, text=True
        )
        self.assertEqual(plan.returncode, 0, plan.stderr)
        self.assertIn("1,012 genuinely synchronous", plan.stdout)
        self.assertIn("44,300 presentations", plan.stdout)
        self.assertIn("splat.DIAGNOSTIC_ONLY.ply", plan.stdout)
        self.assertIn("No bag replay", plan.stdout)

    def test_run_refuses_before_writing_without_diagnostic_acknowledgement(self):
        with tempfile.TemporaryDirectory() as temporary:
            workdir = Path(temporary) / "must-not-exist"
            result = subprocess.run(
                ["bash", str(SCRIPT), "run", "--workdir", str(workdir)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "requires --acknowledge-diagnostic-render-only", result.stderr
            )
            self.assertFalse(workdir.exists())

    def test_frozen_transfer_and_render_gates_are_explicit(self):
        contents = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--max-depth-sync-residual-ns 100000", contents)
        self.assertIn("--max-pose-interpolation-gap-ns 300000000", contents)
        self.assertIn("--min-depth-rgb-association-fraction 0.995", contents)
        self.assertIn("--min-projected-depth-coverage-fraction 0.30", contents)
        self.assertIn("--min-projected-depth-retained-fraction 0.50", contents)
        self.assertEqual(contents.count("-m rtk_splat.workflows.rgbd_transfer"), 1)
        self.assertEqual(contents.count('execute_logged cloud "${CLI[@]}" cloud'), 1)
        self.assertEqual(contents.count('execute_logged train "${CLI[@]}" train'), 1)
        self.assertIn("--allow-failed-georeferencing-for-render", contents)
        self.assertNotIn("frontend-", contents)
        self.assertNotIn("backend-solve", contents)


if __name__ == "__main__":
    unittest.main()
