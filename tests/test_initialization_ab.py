import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from rtk_splat.diagnostics.initialization_ab import (
    compare_pose_prior_initialization_arms,
)
from rtk_splat.diagnostics.refinement_ab import (
    write_report_content_identical,
)


def _quality(
    *,
    mode: str,
    passed: bool,
    reprojection: float,
    rtk_median: float,
):
    checks = {
        "visual_quality": {"passed": True},
        "trajectory_similarity_scale": {"passed": True, "value": 1.001},
        "stereo_baseline_max_change_m": {
            "passed": True,
            "value": 1e-6,
        },
        "calibration_parameter_max_change": {
            "passed": True,
            "value": 0.0,
        },
    }
    if not passed:
        checks["visual_quality"]["passed"] = False
    return {
        "passed": passed,
        "initialization_mode": mode,
        "position_prior_loss": "trivial",
        "registration_fraction": 1.0,
        "checks": checks,
        "refined_model_stats": {
            "registered_images": 12,
            "points": 101,
            "mean_reprojection_error_px": reprojection,
            "mean_track_length": 4.0,
            "mean_observations_per_image": 67.3,
        },
        "rtk_holdout": {
            "refined": {
                "residual_m": {
                    "median": rtk_median,
                    "p95": 0.30,
                    "p95_euclidean_inliers": 0.28,
                },
                "checks": {
                    "holdout_rtk_inlier_fraction": {"value": 0.95}
                },
            }
        },
    }


def _result(workspace: Path, *, mode: str, fresh: bool):
    shared_input = "/sealed/backend/registered_model"
    command = [
        "/bin/true",
        "pose_prior_mapper",
        "--database_path",
        str(workspace / "database.db"),
        "--image_path",
        "/sealed/frontend/images",
    ]
    if not fresh:
        command.extend(["--input_path", shared_input])
    command.extend(
        [
            "--output_path",
            str(workspace / "refined_model.incomplete"),
            "--use_robust_loss_on_prior_position",
            "0",
        ]
    )
    return {
        "workspace": str(workspace),
        "source_backend": "/sealed/backend",
        "frontend_artifact": "/sealed/frontend",
        "image_path": "/sealed/frontend/images",
        "input_model": shared_input,
        "config": {
            "initialization_mode": mode,
            "prior_position_loss": "trivial",
            "random_seed": 7,
            "fresh_min_mean_track_length": 3.0,
        },
        "mapper_evaluation_config": {"max_reprojection_error_px": 1.0},
        "source_evidence": {"quality_marker_sha256": "a" * 64},
        "optimizer_database_committed_sha256": "b" * 64,
        "optimizer_database": {"database_sha256": "c" * 64},
        "counts": {
            "images": 12,
            "frames": 6,
            "evaluation_priors": 6,
            "calibration_priors": 4,
            "holdout_priors": 2,
        },
        "camera_ids": [1, 2],
        "rig_ids": [1],
        "split": {
            "method": "temporal_blocks",
            "calibration_block_ids": [0, 2],
            "holdout_block_ids": [1],
            "prior_split_sha256": "d" * 64,
            "calibration_names_sha256": "e" * 64,
            "holdout_names_sha256": "f" * 64,
            "all_images_sha256": "0" * 64,
        },
        "refinement_plan_sha256": "1" * 64,
        "quality_marker_sha256": "2" * 64,
        "command": command,
        "quality": _quality(
            mode=mode,
            passed=fresh,
            reprojection=0.40 if fresh else 0.45,
            rtk_median=0.20 if fresh else 0.23,
        ),
    }


class InitializationAbTest(unittest.TestCase):
    def setUp(self):
        self.control_path = Path("/tmp/continuation-arm")
        self.candidate_path = Path("/tmp/fresh-arm")
        self.control = _result(
            self.control_path, mode="continuation", fresh=False
        )
        self.candidate = _result(
            self.candidate_path, mode="fresh", fresh=True
        )

    def _compare(self, control=None, candidate=None):
        arms = [control or self.control, candidate or self.candidate]
        with mock.patch(
            "rtk_splat.diagnostics.initialization_ab."
            "audited_rtk_refinement_result",
            side_effect=arms,
        ) as audited:
            report = compare_pose_prior_initialization_arms(
                self.control_path,
                self.candidate_path,
                expected_frames=6,
                expected_evaluation_priors=6,
                expected_calibration_priors=4,
                expected_holdout_priors=2,
            )
        self.assertEqual(
            audited.call_args_list,
            [mock.call(self.control_path), mock.call(self.candidate_path)],
        )
        return report

    def test_proves_exact_initialization_difference_and_reports_deltas(self):
        report = self._compare()
        self.assertTrue(report["integrity"]["passed"])
        self.assertTrue(report["integrity"]["both_quality_audits_sealed"])
        self.assertEqual(
            report["integrity"]["removed_control_command_pair"]["option"],
            "--input_path",
        )
        self.assertEqual(report["control"]["initialization_mode"], "continuation")
        self.assertEqual(report["candidate"]["initialization_mode"], "fresh")
        self.assertAlmostEqual(
            report["diagnostic_quality_deltas"]["candidate_minus_control"]
            ["mean_reprojection_error_px"],
            -0.05,
        )
        self.assertEqual(
            report["status"], "fresh_candidate_passed_all_existing_gates"
        )
        self.assertFalse(
            report["downstream"]
            ["pose_export_cloud_training_or_ply_started_by_this_experiment"]
        )

    def test_rejects_any_sealed_input_difference(self):
        candidate = deepcopy(self.candidate)
        candidate["rig_ids"] = [9]
        with self.assertRaisesRegex(ValueError, "rig_ids"):
            self._compare(candidate=candidate)

    def test_rejects_extra_mapper_command_difference(self):
        candidate = deepcopy(self.candidate)
        option = candidate["command"].index(
            "--use_robust_loss_on_prior_position"
        )
        candidate["command"][option + 1] = "1"
        with self.assertRaisesRegex(ValueError, "outside removal"):
            self._compare(candidate=candidate)

    def test_rejects_input_model_on_fresh_mapper_command(self):
        candidate = deepcopy(self.candidate)
        candidate["command"][6:6] = [
            "--input_path",
            candidate["input_model"],
        ]
        with self.assertRaisesRegex(ValueError, "unexpectedly contains"):
            self._compare(candidate=candidate)

    def test_rejects_non_l2_arm_or_wrong_initialization_role(self):
        candidate = deepcopy(self.candidate)
        candidate["config"]["prior_position_loss"] = "cauchy"
        with self.assertRaisesRegex(ValueError, "quadratic/L2"):
            self._compare(candidate=candidate)

        control = deepcopy(self.control)
        control["config"]["initialization_mode"] = "fresh"
        with self.assertRaisesRegex(ValueError, "continuation control"):
            self._compare(control=control)

    def test_expected_counts_are_fail_closed(self):
        with mock.patch(
            "rtk_splat.diagnostics.initialization_ab."
            "audited_rtk_refinement_result",
            side_effect=[self.control, self.candidate],
        ):
            with self.assertRaisesRegex(ValueError, "unexpected frames"):
                compare_pose_prior_initialization_arms(
                    self.control_path,
                    self.candidate_path,
                    expected_frames=1495,
                )

    def test_report_publish_is_atomic_and_content_identical(self):
        report = self._compare()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "reports" / "initialization_ab.json"
            first = write_report_content_identical(output, report)
            payload = output.read_bytes()
            second = write_report_content_identical(output, report)
            self.assertEqual(first, second)
            self.assertEqual(output.read_bytes(), payload)

            changed = deepcopy(report)
            changed["status"] = "different"
            with self.assertRaises(FileExistsError):
                write_report_content_identical(output, changed)


if __name__ == "__main__":
    unittest.main()
