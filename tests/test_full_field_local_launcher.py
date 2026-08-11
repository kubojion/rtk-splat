import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/runs/field1_0703_full_local_prepare.sh"


class FullFieldLocalLauncherTests(unittest.TestCase):
    def test_read_only_actions_and_shell_syntax(self):
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            workdir = Path(temporary) / "must-not-exist"
            for action in ("plan", "status"):
                result = subprocess.run(
                    ["bash", str(SCRIPT), action, "--workdir", str(workdir)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(workdir.exists())

    def test_recovery_binds_every_configuration_layer(self):
        contents = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("cfg.runtime_resolution.source_files", contents)
        self.assertIn("find \"$REPO/src/rtk_splat\"", contents)
        self.assertIn("systemd-inhibit", contents)
        self.assertIn("--resume-existing", contents)
        self.assertIn("segment-materialize", contents)
        self.assertIn("orphaned staging from a hard interruption", contents)
        self.assertNotIn("rm -rf", contents)


if __name__ == "__main__":
    unittest.main()
