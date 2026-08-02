"""Fail-closed comparison of continuation and fresh pose-prior mapping."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from rtk_splat.backends.rtk_refinement import audited_rtk_refinement_result
from rtk_splat.diagnostics.refinement_ab import write_report_content_identical
from rtk_splat.frontends.artifact import sha256_file


_PARITY_FIELDS = (
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
)


def _require_equal(
    control: dict[str, Any], candidate: dict[str, Any]
) -> None:
    for field in _PARITY_FIELDS:
        if control[field] != candidate[field]:
            raise ValueError(
                "pose-prior initialization arms differ outside initialization: "
                f"{field}"
            )


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


def _remove_exact_input_model_pair(
    command: list[str], expected_input_model: str
) -> tuple[list[str], dict[str, Any]]:
    option = "--input_path"
    indices = [index for index, value in enumerate(command) if value == option]
    if len(indices) != 1:
        raise ValueError(
            "continuation command must contain exactly one --input_path"
        )
    index = indices[0]
    if index + 1 >= len(command):
        raise ValueError("continuation --input_path has no value")
    value = command[index + 1]
    if Path(value).expanduser().resolve() != Path(
        expected_input_model
    ).expanduser().resolve():
        raise ValueError(
            "continuation command input model does not match sealed plan"
        )
    return (
        command[:index] + command[index + 2 :],
        {"index": index, "option": option, "value": value},
    )


def _quality_summary(result: dict[str, Any]) -> dict[str, Any]:
    quality = result["quality"]
    refined = quality["rtk_holdout"]["refined"]
    stats = quality["refined_model_stats"]
    checks = quality["checks"]
    return {
        "passed_all_existing_gates": bool(quality["passed"]),
        "registration_fraction": quality["registration_fraction"],
        "mean_reprojection_error_px": stats["mean_reprojection_error_px"],
        "mean_track_length": stats["mean_track_length"],
        "mean_observations_per_image": stats[
            "mean_observations_per_image"
        ],
        "registered_images": stats["registered_images"],
        "points": stats["points"],
        "heldout_rtk_median_m": refined["residual_m"]["median"],
        "heldout_rtk_support_fraction": refined["checks"][
            "holdout_rtk_inlier_fraction"
        ]["value"],
        "heldout_rtk_inlier_p95_m": refined["residual_m"][
            "p95_euclidean_inliers"
        ],
        "heldout_rtk_full_p95_m": refined["residual_m"]["p95"],
        "trajectory_similarity_scale": checks[
            "trajectory_similarity_scale"
        ]["value"],
        "stereo_baseline_max_change_m": checks[
            "stereo_baseline_max_change_m"
        ]["value"],
        "calibration_parameter_max_change": checks[
            "calibration_parameter_max_change"
        ]["value"],
        "failed_checks": sorted(
            name for name, check in checks.items() if not check["passed"]
        ),
    }


def _quality_deltas(
    control: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    return {
        "candidate_minus_control": {
            "registration_fraction": (
                candidate["registration_fraction"]
                - control["registration_fraction"]
            ),
            "mean_reprojection_error_px": (
                candidate["mean_reprojection_error_px"]
                - control["mean_reprojection_error_px"]
            ),
            "mean_track_length": (
                candidate["mean_track_length"] - control["mean_track_length"]
            ),
            "mean_observations_per_image": (
                candidate["mean_observations_per_image"]
                - control["mean_observations_per_image"]
            ),
            "registered_images": (
                candidate["registered_images"] - control["registered_images"]
            ),
            "points": candidate["points"] - control["points"],
            "heldout_rtk_median_m": (
                candidate["heldout_rtk_median_m"]
                - control["heldout_rtk_median_m"]
            ),
            "heldout_rtk_support_fraction": (
                candidate["heldout_rtk_support_fraction"]
                - control["heldout_rtk_support_fraction"]
            ),
            "heldout_rtk_inlier_p95_m": (
                candidate["heldout_rtk_inlier_p95_m"]
                - control["heldout_rtk_inlier_p95_m"]
            ),
            "heldout_rtk_full_p95_m": (
                candidate["heldout_rtk_full_p95_m"]
                - control["heldout_rtk_full_p95_m"]
            ),
            "trajectory_similarity_scale_deviation": (
                abs(candidate["trajectory_similarity_scale"] - 1.0)
                - abs(control["trajectory_similarity_scale"] - 1.0)
            ),
            "stereo_baseline_max_change_m": (
                candidate["stereo_baseline_max_change_m"]
                - control["stereo_baseline_max_change_m"]
            ),
            "calibration_parameter_max_change": (
                candidate["calibration_parameter_max_change"]
                - control["calibration_parameter_max_change"]
            ),
        },
        "interpretation": (
            "Negative reprojection and RTK-residual deltas are improvements; "
            "positive registration, track, observation, and support deltas are "
            "improvements. Existing acceptance gates remain authoritative."
        ),
    }


def compare_pose_prior_initialization_arms(
    control_workspace: str | Path,
    candidate_workspace: str | Path,
    *,
    expected_frames: int | None = None,
    expected_evaluation_priors: int | None = None,
    expected_calibration_priors: int | None = None,
    expected_holdout_priors: int | None = None,
) -> dict[str, Any]:
    """Compare sealed continuation/L2 and fresh/L2 mapper results."""
    control = audited_rtk_refinement_result(control_workspace)
    candidate = audited_rtk_refinement_result(candidate_workspace)
    _require_equal(control, candidate)

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
    control_mode = control_config.pop("initialization_mode")
    candidate_mode = candidate_config.pop("initialization_mode")
    if control_mode != "continuation" or candidate_mode != "fresh":
        raise ValueError(
            "initialization A/B requires continuation control and fresh candidate"
        )
    if control_config.get("prior_position_loss") != "trivial" or (
        candidate_config.get("prior_position_loss") != "trivial"
    ):
        raise ValueError("initialization A/B requires quadratic/L2 RTK priors")
    if control_config != candidate_config:
        raise ValueError(
            "pose-prior initialization configurations differ outside the mode"
        )
    for name, result, mode in (
        ("control", control, control_mode),
        ("candidate", candidate, candidate_mode),
    ):
        quality_mode = result["quality"].get(
            "initialization_mode", "continuation"
        )
        if quality_mode != mode:
            raise ValueError(
                f"{name} quality audit records the wrong initialization mode"
            )
        if result["quality"].get("position_prior_loss") != "trivial":
            raise ValueError(f"{name} quality audit does not record L2 priors")

    control_command = _normalized_command(control)
    candidate_command = _normalized_command(candidate)
    if len(control_command) < 2 or control_command[1] != "pose_prior_mapper":
        raise ValueError("control command is not COLMAP pose_prior_mapper")
    if len(candidate_command) < 2 or candidate_command[1] != "pose_prior_mapper":
        raise ValueError("candidate command is not COLMAP pose_prior_mapper")
    if "--input_path" in candidate_command:
        raise ValueError("fresh mapper command unexpectedly contains --input_path")
    stripped_control, removed_pair = _remove_exact_input_model_pair(
        control_command, control["input_model"]
    )
    if stripped_control != candidate_command:
        raise ValueError(
            "mapper commands differ outside removal of the input-model pair"
        )

    executable = Path(control["command"][0]).expanduser().resolve()
    if executable != Path(candidate["command"][0]).expanduser().resolve():
        raise ValueError(
            "pose-prior initialization arms record different COLMAP executables"
        )
    if not executable.is_file():
        raise ValueError(f"recorded COLMAP executable is unavailable: {executable}")

    control_quality = _quality_summary(control)
    candidate_quality = _quality_summary(candidate)
    candidate_passed = candidate_quality["passed_all_existing_gates"]
    return {
        "schema_version": 1,
        "experiment": "pose_prior_mapper_initialization_ab",
        "status": (
            "fresh_candidate_passed_all_existing_gates"
            if candidate_passed
            else "fresh_candidate_rejected_by_existing_gates"
        ),
        "integrity": {
            "passed": True,
            "only_meaningful_solver_difference": (
                "initialization_mode: continuation -> fresh; mapper "
                "--input_path pair removed"
            ),
            "source_frontend_factor_holdout_and_rig_evidence_identical": True,
            "both_quality_audits_sealed": True,
            "counts": control["counts"],
            "split": control["split"],
            "removed_control_command_pair": removed_pair,
            "commands_identical_after_declared_removal": True,
            "colmap_executable": str(executable),
            "colmap_executable_current_sha256": sha256_file(executable),
            "runtime_binary_hash_was_sealed": False,
        },
        "control": {
            "workspace": control["workspace"],
            "initialization_mode": control_mode,
            "loss": control_config["prior_position_loss"],
            "refinement_plan_sha256": control["refinement_plan_sha256"],
            "quality_marker_sha256": control["quality_marker_sha256"],
            "quality": control_quality,
        },
        "candidate": {
            "workspace": candidate["workspace"],
            "initialization_mode": candidate_mode,
            "loss": candidate_config["prior_position_loss"],
            "refinement_plan_sha256": candidate["refinement_plan_sha256"],
            "quality_marker_sha256": candidate["quality_marker_sha256"],
            "quality": candidate_quality,
        },
        "diagnostic_quality_deltas": _quality_deltas(
            control_quality, candidate_quality
        ),
        "downstream": {
            "pose_export_cloud_training_or_ply_started_by_this_experiment": False,
            "candidate_eligible_for_separate_manual_export_review": (
                candidate_passed
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare sealed continuation and fresh pose-prior mapper arms"
        )
    )
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--expected-evaluation-priors", type=int)
    parser.add_argument("--expected-calibration-priors", type=int)
    parser.add_argument("--expected-holdout-priors", type=int)
    args = parser.parse_args(argv)
    report = compare_pose_prior_initialization_arms(
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
    print(f"initialization A/B report -> {output}")
    print(
        "registered images: "
        f"continuation={control['registered_images']} "
        f"fresh={candidate['registered_images']}"
    )
    print(
        "mean reprojection: "
        f"continuation={control['mean_reprojection_error_px']:.6f} px "
        f"fresh={candidate['mean_reprojection_error_px']:.6f} px"
    )
    print(
        "held-out RTK median: "
        f"continuation={control['heldout_rtk_median_m']:.6f} m "
        f"fresh={candidate['heldout_rtk_median_m']:.6f} m"
    )
    print(f"result: {report['status']}")
    print("No pose, cloud, training run, or PLY was produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
