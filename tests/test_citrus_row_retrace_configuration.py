import subprocess
import tempfile
import unittest
from pathlib import Path

from rtk_splat.workflows.configio import load_config


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "configs/sequences/citrusfarm_05_13d_row_retrace.yaml"
SCRIPT = REPOSITORY / "scripts/runs/citrusfarm_05_13d_row_retrace.sh"
COMMON_SCRIPT = REPOSITORY / "scripts/runs/_citrusfarm_05_13d_common.sh"
PREFIX = "citrus-05-13d-330-410-row-retrace-v1"


class CitrusRowRetraceConfigurationTests(unittest.TestCase):
    def test_layered_config_selects_one_literal_retrace_submap(self):
        cfg = load_config(CONFIG)

        self.assertEqual(cfg.adapter, "ros1_citrusfarm")
        self.assertEqual(cfg.segment.window_s, [330.0, 410.0])
        self.assertEqual(cfg.segment.window_epoch_source, "first_gnss_log")
        self.assertEqual(cfg.segment.frame_spacing_m, "auto")
        self.assertEqual(cfg.pose.source, "gnss_course")
        self.assertEqual(cfg.pose.time_offset_s, 0.072548749)
        self.assertEqual(len(cfg.paths.camera_bags), 27)
        self.assertEqual(len(cfg.paths.gnss_bags), 2)
        self.assertEqual(cfg.frontend.keyframes.preset, "all")
        self.assertEqual(cfg.frontend.name, f"{PREFIX}-frontend-all-gpu")
        self.assertEqual(cfg.mapper.backend, "global")
        self.assertEqual(cfg.mapper.name, f"{PREFIX}-global")
        self.assertEqual(cfg.mapper.pose_artifact_name, f"{PREFIX}-global")
        self.assertEqual(cfg.train.run_name, f"{PREFIX}-gs")
        self.assertEqual(cfg.cloud.max_points, "auto")
        self.assertEqual(cfg.train.iterations, "auto")
        self.assertEqual(cfg.train.max_gaussians, "auto")

    def test_wrappers_share_one_syntax_valid_pipeline_driver(self):
        for script in (SCRIPT, COMMON_SCRIPT):
            syntax = subprocess.run(
                ["bash", "-n", str(script)], capture_output=True, text=True
            )
            self.assertEqual(syntax.returncode, 0, syntax.stderr)

        wrapper = SCRIPT.read_text(encoding="utf-8")
        common = COMMON_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("_citrusfarm_05_13d_common.sh", wrapper)
        self.assertIn('CITRUS_MIN_OUTPUT_FREE_GIB="30"', wrapper)
        self.assertIn('sha256sum "$SCRIPT_PATH" "$COMMON_SCRIPT_PATH"', common)
        self.assertNotIn("12000000", wrapper)

    def test_plan_is_non_mutating_and_describes_measured_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            untouched = Path(temporary) / "must-not-be-created"
            planned = subprocess.run(
                ["bash", str(SCRIPT), "plan", "--workdir", str(untouched)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(planned.returncode, 0, planned.stderr)
            self.assertFalse(untouched.exists())

        self.assertIn("same-corridor row-retrace", planned.stdout)
        self.assertIn("[330, 410] s", planned.stdout)
        self.assertIn("96.7 m driven", planned.stdout)
        self.assertIn("38 m of direct retrace overlap", planned.stdout)
        self.assertIn("roughly 800 stereo pairs", planned.stdout)
        self.assertIn("likely about 35k iterations", planned.stdout)
        self.assertIn("about 821k points", planned.stdout)
        self.assertIn("approximately 2-4 h", planned.stdout)
        self.assertIn("about 2.46M", planned.stdout)
        self.assertIn("Nothing runs from plan", planned.stdout)

    def test_diagnostic_plan_keeps_georeferencing_failure_explicit(self):
        planned = subprocess.run(
            ["bash", str(SCRIPT), "plan", "--render-on-georef-failure"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(planned.returncode, 0, planned.stderr)
        self.assertIn("diagnostic rendering is enabled", planned.stdout)
        self.assertIn("are NOT relaxed", planned.stdout)
        self.assertIn("visualization-only", planned.stdout)
        self.assertIn("cannot support a", planned.stdout)
        self.assertIn("metric georeferencing claim", planned.stdout)
        self.assertIn(
            "/citrusfarm_05_13d_330_410_row_retrace_v1 ", planned.stdout
        )


if __name__ == "__main__":
    unittest.main()
