import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/runs/field1_0703_full_server.sh"


class FullFieldServerLauncherTests(unittest.TestCase):
    def test_embedded_python_programs_parse(self):
        lines = SCRIPT.read_text(encoding="utf-8").splitlines()
        programs = []
        index = 0
        while index < len(lines):
            if "<<'PY'" not in lines[index]:
                index += 1
                continue
            start = index + 1
            end = start
            while end < len(lines) and lines[end] != "PY":
                end += 1
            self.assertLess(end, len(lines), "unterminated Python heredoc")
            programs.append("\n".join(lines[start:end]) + "\n")
            index = end + 1
        self.assertGreaterEqual(len(programs), 10)
        for number, program in enumerate(programs):
            ast.parse(program, filename=f"{SCRIPT}:heredoc-{number}")

    def test_shell_syntax_and_read_only_actions_need_no_data(self):
        syntax = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        help_result = subprocess.run(
            ["bash", str(SCRIPT), "--help"], capture_output=True, text=True
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("prepare", help_result.stdout)
        self.assertIn("smoke", help_result.stdout)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workdir = root / "must-not-be-created"
            segment = root / "missing-segment"
            for action in ("plan", "status"):
                result = subprocess.run(
                    [
                        "bash",
                        str(SCRIPT),
                        action,
                        "--workdir",
                        str(workdir),
                        "--segment",
                        str(segment),
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(workdir.exists())

            preflight = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "preflight",
                    "--workdir",
                    str(workdir),
                    "--segment",
                    str(segment),
                    "--python",
                    sys.executable,
                    "--colmap",
                    "/bin/true",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(preflight.returncode, 0)
            self.assertIn("portable segment is missing", preflight.stderr)
            self.assertFalse(workdir.exists())

    def test_pipeline_is_arbitrary_count_and_production_only(self):
        contents = SCRIPT.read_text(encoding="utf-8")
        for stage in (
            "frontend-build",
            "frontend-features",
            "frontend-rig",
            "frontend-priors",
            "frontend-match",
            "backend-prepare",
            "backend-solve",
            "backend-register",
            "backend-quality",
            "backend-export",
            "tiles-plan",
            "cloud",
            "train",
            "scene-publish",
        ):
            self.assertIn(stage, contents)
        self.assertIn("mapfile -t ids < <(tile_ids)", contents)
        self.assertIn('for tile in "${ids[@]}"', contents)
        self.assertNotIn("TILES=(", contents)
        self.assertIn("--scene-mode production", contents)
        self.assertIn("absolute_no_monolithic_reference", contents)
        self.assertNotIn("--reference-run", contents)
        self.assertNotIn("--allow-failed-georeferencing-for-render", contents)

    def test_recovery_and_server_identity_are_fail_closed(self):
        contents = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("scripts/tools/server_preflight.py", contents)
        self.assertIn(
            'SEGMENT_DEFAULT="/data/jkobo/datasets/field1_0703_full77/segment"',
            contents,
        )
        self.assertIn("verify_portable_segment", (
            REPOSITORY / "scripts/tools/server_preflight.py"
        ).read_text(encoding="utf-8"))
        self.assertIn("collect_git_state", contents)
        self.assertIn("production server runs require a clean committed Git checkout", contents)
        self.assertIn(
            "server run identity changed (code/config/segment/environment/path)",
            contents,
        )
        self.assertIn("attempt-%04d", contents)
        self.assertIn("MAXIMUM_VISIBLE_GAUSSIANS=12000000", contents)
        self.assertIn("MINIMUM_RAM_GIB=120", contents)
        self.assertIn('--minimum-ram-gib "$MINIMUM_RAM_GIB"', contents)
        self.assertIn('PREFLIGHT_RECORD="$ORCHESTRATION/server_preflight.json"', contents)
        self.assertIn('"server_preflight": {', contents)
        self.assertIn('"nvidia_driver"', (
            REPOSITORY / "scripts/tools/server_preflight.py"
        ).read_text(encoding="utf-8"))
        self.assertIn('"logical_cpu_count"', (
            REPOSITORY / "scripts/tools/server_preflight.py"
        ).read_text(encoding="utf-8"))
        self.assertIn('"cuda_feature_extraction_smoke"', (
            REPOSITORY / "scripts/tools/server_preflight.py"
        ).read_text(encoding="utf-8"))
        self.assertIn("PRESERVED: incomplete", contents)
        self.assertIn("SKIP: $tile already has one verified completed attempt", contents)
        self.assertIn("systemd-inhibit", contents)
        self.assertIn("XDG_RUNTIME_DIR", contents)
        self.assertIn('candidate="$(dirname "$WORKDIR")/run_locks"', contents)
        self.assertIn("chmod 700", contents)
        self.assertIn("flock -n 9", contents)
        self.assertNotIn('LOCK="/tmp/', contents)
        self.assertNotIn("rm -rf", contents)


if __name__ == "__main__":
    unittest.main()
