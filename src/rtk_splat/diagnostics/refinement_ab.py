"""Fail-closed comparison of two completed RTK-refinement arms."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from rtk_splat.backends.rtk_refinement import audited_rtk_refinement_result
from rtk_splat.frontends.artifact import sha256_file


def _require_equal(
    control: dict[str, Any], candidate: dict[str, Any], fields: tuple[str, ...]
) -> None:
    for field in fields:
        if control[field] != candidate[field]:
            raise ValueError(f"RTK loss arms differ outside the loss: {field}")


def _normalized_command(result: dict[str, Any]) -> list[str]:
    workspace = str(result["workspace"])
    prefix = workspace + os.sep
    normalized: list[str] = []
    for item in result["command"]:
        value = str(item)
        if value == workspace:
            normalized.append("{arm_workspace}")
        elif value.startswith(prefix):
            normalized.append("{arm_workspace}" + value[len(workspace) :])
        else:
            normalized.append(value)
    return normalized


def _quality_summary(result: dict[str, Any]) -> dict[str, Any]:
    quality = result["quality"]
    refined = quality["rtk_holdout"]["refined"]
    return {
        "passed_all_existing_gates": bool(quality["passed"]),
        "registration_fraction": quality["registration_fraction"],
        "mean_reprojection_error_px": quality["refined_model_stats"][
            "mean_reprojection_error_px"
        ],
        "mean_track_length": quality["refined_model_stats"][
            "mean_track_length"
        ],
        "heldout_rtk_median_m": refined["residual_m"]["median"],
        "heldout_rtk_support_fraction": refined["checks"][
            "holdout_rtk_inlier_fraction"
        ]["value"],
        "heldout_rtk_inlier_p95_m": refined["residual_m"][
            "p95_euclidean_inliers"
        ],
        "heldout_rtk_full_p95_m": refined["residual_m"]["p95"],
        "trajectory_similarity_scale": quality["checks"][
            "trajectory_similarity_scale"
        ]["value"],
        "stereo_baseline_max_change_m": quality["checks"][
            "stereo_baseline_max_change_m"
        ]["value"],
        "calibration_parameter_max_change": quality["checks"][
            "calibration_parameter_max_change"
        ]["value"],
        "failed_checks": sorted(
            name
            for name, check in quality["checks"].items()
            if not check["passed"]
        ),
    }


def compare_rtk_refinement_arms(
    control_workspace: str | Path,
    candidate_workspace: str | Path,
    *,
    expected_frames: int | None = None,
    expected_evaluation_priors: int | None = None,
    expected_calibration_priors: int | None = None,
    expected_holdout_priors: int | None = None,
) -> dict[str, Any]:
    """Compare Cauchy and quadratic arms after proving input parity."""
    control = audited_rtk_refinement_result(control_workspace)
    candidate = audited_rtk_refinement_result(candidate_workspace)
    _require_equal(
        control,
        candidate,
        (
            "source_backend",
            "frontend_artifact",
            "image_path",
            "input_model",
            "mapper_evaluation_config",
            "source_evidence",
            "optimizer_database_committed_sha256",
            "optimizer_database",
            "counts",
            "camera_ids",
            "rig_ids",
            "split",
        ),
    )
    expected = {
        "frames": expected_frames,
        "evaluation_priors": expected_evaluation_priors,
        "calibration_priors": expected_calibration_priors,
        "holdout_priors": expected_holdout_priors,
    }
    for field, value in expected.items():
        if value is not None and control["counts"][field] != value:
            raise ValueError(
                f"unexpected {field}: {control['counts'][field]} != {value}"
            )

    control_config = dict(control["config"])
    candidate_config = dict(candidate["config"])
    control_loss = control_config.pop("prior_position_loss")
    candidate_loss = candidate_config.pop("prior_position_loss")
    if control_loss != "cauchy" or candidate_loss != "trivial":
        raise ValueError(
            "loss A/B requires cauchy control and trivial candidate"
        )
    if control_config != candidate_config:
        raise ValueError("RTK loss arm configurations differ outside the loss")

    control_command = _normalized_command(control)
    candidate_command = _normalized_command(candidate)
    if len(control_command) != len(candidate_command):
        raise ValueError("RTK loss arm command lengths differ")
    command_differences = [
        {"index": index, "control": left, "candidate": right}
        for index, (left, right) in enumerate(
            zip(control_command, candidate_command, strict=True)
        )
        if left != right
    ]
    robust_option = "--use_robust_loss_on_prior_position"
    robust_index = control_command.index(robust_option) + 1
    if command_differences != [
        {"index": robust_index, "control": "1", "candidate": "0"}
    ]:
        raise ValueError(
            "COLMAP commands do not differ by exactly the robust-loss flag"
        )

    executable = Path(control["command"][0]).expanduser().resolve()
    if executable != Path(candidate["command"][0]).expanduser().resolve():
        raise ValueError("RTK loss arms record different COLMAP executables")
    if not executable.is_file():
        raise ValueError(f"recorded COLMAP executable is unavailable: {executable}")

    control_quality = _quality_summary(control)
    candidate_quality = _quality_summary(candidate)
    improvement = (
        control_quality["heldout_rtk_median_m"]
        - candidate_quality["heldout_rtk_median_m"]
    )
    support_change = (
        candidate_quality["heldout_rtk_support_fraction"]
        - control_quality["heldout_rtk_support_fraction"]
    )
    p95_change = (
        candidate_quality["heldout_rtk_inlier_p95_m"]
        - control_quality["heldout_rtk_inlier_p95_m"]
    )
    diagnostic_checks = {
        "median_improvement_at_least_5mm": {
            "value_m": improvement,
            "minimum_m": 0.005,
            "passed": improvement >= 0.005,
        },
        "support_not_reduced": {
            "value": support_change,
            "minimum": 0.0,
            "passed": support_change >= 0.0,
        },
        "inlier_p95_regression_at_most_5mm": {
            "value_m": p95_change,
            "maximum_m": 0.005,
            "passed": p95_change <= 0.005,
        },
    }
    diagnostic_supported = all(
        check["passed"] for check in diagnostic_checks.values()
    )
    candidate_passed = candidate_quality["passed_all_existing_gates"]
    return {
        "schema_version": 1,
        "experiment": "rtk_position_prior_loss_ab",
        "status": (
            "candidate_passed_all_existing_gates"
            if candidate_passed
            else "completed_candidate_rejected_by_existing_gates"
        ),
        "integrity": {
            "passed": True,
            "only_meaningful_solver_difference": (
                "use_robust_loss_on_prior_position: 1 -> 0"
            ),
            "source_and_factor_holdout_evidence_identical": True,
            "counts": control["counts"],
            "split": control["split"],
            "command_differences": command_differences,
            "colmap_executable": str(executable),
            "colmap_executable_current_sha256": sha256_file(executable),
            "control_runtime_binary_hash_was_sealed": False,
        },
        "control": {
            "workspace": control["workspace"],
            "loss": control_loss,
            "refinement_plan_sha256": control["refinement_plan_sha256"],
            "quality_marker_sha256": control["quality_marker_sha256"],
            "quality": control_quality,
        },
        "candidate": {
            "workspace": candidate["workspace"],
            "loss": candidate_loss,
            "refinement_plan_sha256": candidate["refinement_plan_sha256"],
            "quality_marker_sha256": candidate["quality_marker_sha256"],
            "quality": candidate_quality,
        },
        "diagnostic_hypothesis": {
            "cauchy_suppression_supported": diagnostic_supported,
            "checks": diagnostic_checks,
            "interpretation": (
                "This tests the incremental effect of removing Cauchy "
                "downweighting. Factor holdouts are not an end-to-end "
                "GNSS-independent benchmark."
            ),
        },
        "downstream": {
            "pose_export_or_gs_started_by_this_experiment": False,
            "candidate_eligible_for_separate_manual_export_review": (
                candidate_passed
            ),
        },
    }


def write_report_content_identical(
    destination: str | Path, report: dict[str, Any]
) -> Path:
    """Publish once, or accept an exactly identical resume result."""
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"existing A/B report differs: {path}")
        return path
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare sealed Cauchy and quadratic RTK refinement arms"
    )
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--expected-evaluation-priors", type=int)
    parser.add_argument("--expected-calibration-priors", type=int)
    parser.add_argument("--expected-holdout-priors", type=int)
    args = parser.parse_args(argv)
    report = compare_rtk_refinement_arms(
        args.control,
        args.candidate,
        expected_frames=args.expected_frames,
        expected_evaluation_priors=args.expected_evaluation_priors,
        expected_calibration_priors=args.expected_calibration_priors,
        expected_holdout_priors=args.expected_holdout_priors,
    )
    output = write_report_content_identical(args.output, report)
    control = report["control"]["quality"]
    candidate = report["candidate"]["quality"]
    print(f"A/B report -> {output}")
    print(
        "held-out median: "
        f"A={control['heldout_rtk_median_m']:.6f} m "
        f"B={candidate['heldout_rtk_median_m']:.6f} m"
    )
    print(
        "held-out support: "
        f"A={100 * control['heldout_rtk_support_fraction']:.2f}% "
        f"B={100 * candidate['heldout_rtk_support_fraction']:.2f}%"
    )
    print(f"result: {report['status']}")
    print("No pose, cloud, training run, or PLY was produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
