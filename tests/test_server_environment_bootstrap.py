import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/tools/bootstrap_server_env.sh"


class ServerEnvironmentBootstrapTests(unittest.TestCase):
    def test_shell_syntax_and_plan_are_read_only(self):
        syntax = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "user" / "rtk-splat"
            result = subprocess.run(
                ["bash", str(SCRIPT), "plan", "--root", str(root)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing broad/unexpected server root", result.stderr)
            self.assertFalse(root.exists())

        root = Path("/data/test-user/rtk-splat")
        result = subprocess.run(
            ["bash", str(SCRIPT), "plan", "--root", str(root)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(root), result.stdout)
        self.assertIn("No system package", result.stdout)

    def test_install_contract_is_user_local_and_pinned(self):
        contents = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('[[ "$EUID" -ne 0 ]]', contents)
        self.assertIn("--prefix \"$PY_ENV\"", contents)
        self.assertIn("--prefix \"$COLMAP_ENV\"", contents)
        self.assertIn('export CONDA_PKGS_DIRS="$CONDA_PACKAGES"', contents)
        self.assertIn('export PIP_CACHE_DIR="$PIP_CACHE"', contents)
        self.assertIn('git -C "$REPO" status --porcelain', contents)
        self.assertNotIn("sudo ", contents)
        self.assertNotIn("apt ", contents)
        self.assertNotIn("conda init", contents)
        self.assertNotIn("rm -rf", contents)


if __name__ == "__main__":
    unittest.main()
