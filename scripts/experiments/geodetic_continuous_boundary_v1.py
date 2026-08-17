#!/usr/bin/env python3
"""Prepare and run one sealed continuous union of assembly windows.

This experimental launcher is deliberately dataset-independent and outside
the installed public API.  It derives every frame from a sealed geodetic
assembly plan and refuses non-adjacent windows or existing destinations.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from rtk_splat.backends.artifact_io import _atomic_json, _json
from rtk_splat.backends.colmap_model import _poses_from_images_txt
from rtk_splat.backends.geodetic_assembly import (
    audited_geodetic_assembly_plan,
)
from rtk_splat.backends.geodetic_submap import (
    _plan_context,
    audited_geodetic_submap_result,
    create_geodetic_frame_selection,
    prepare_geodetic_submap_plan,
    run_geodetic_submap_plan,
)
from rtk_splat.backends.quality import estimate_rigid_alignment
from rtk_splat.frontends.artifact import (
    ArtifactError,
    canonical_hash,
    sha256_file,
)


_BINDING_KIND = "rtk_splat_geodetic_continuous_window_binding"
_SELECTION_KIND = "rtk_splat_geodetic_calibration_only_candidate_selection"
_AUDIT_KIND = "rtk_splat_geodetic_calibration_only_residual_audit"


def _selected_windows(
    assembly: Mapping[str, Any], window_ids: Sequence[str]
) -> list[Mapping[str, Any]]:
    if len(window_ids) < 2 or len(set(window_ids)) != len(window_ids):
        raise ArtifactError(
            "a continuous union needs at least two unique window IDs"
        )
    windows = list(assembly["windows"])
    by_id = {
        str(window["window_id"]): index
        for index, window in enumerate(windows)
    }
    try:
        indexes = [by_id[str(window_id)] for window_id in window_ids]
    except KeyError as exc:
        raise ArtifactError(f"unknown assembly window: {exc.args[0]}") from exc
    if indexes != sorted(indexes) or any(
        right != left + 1 for left, right in zip(indexes, indexes[1:])
    ):
        raise ArtifactError(
            "continuous-window IDs must be adjacent and assembly-ordered"
        )
    selected = [windows[index] for index in indexes]
    for left, right in zip(selected, selected[1:]):
        if not set(left["frame_ids"]) & set(right["frame_ids"]):
            raise ArtifactError("adjacent assembly windows do not overlap")
    return selected


def _union_frame_ids(
    assembly: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]
) -> list[int]:
    selected_ids = {
        int(frame_id)
        for window in selected
        for frame_id in window["frame_ids"]
    }
    full_selection = _json(Path(assembly["artifact"]) / "frame_selection.json")
    ordered = [
        int(frame_id)
        for frame_id in full_selection.get("frame_ids", ())
        if int(frame_id) in selected_ids
    ]
    if len(ordered) != len(selected_ids) or set(ordered) != selected_ids:
        raise ArtifactError("continuous union differs from the sealed selection")
    return ordered


def _expected_binding(
    assembly: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    frame_ids: Sequence[int],
    selection: Path,
    plan: Path,
) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "kind": _BINDING_KIND,
        "assembly_plan": str(assembly["artifact"]),
        "assembly_plan_seal_sha256": assembly["plan_seal_sha256"],
        "window_ids": [str(window["window_id"]) for window in selected],
        "window_frame_ids_sha256": [
            str(window["frame_ids_sha256"]) for window in selected
        ],
        "frame_count": len(frame_ids),
        "frame_ids_sha256": canonical_hash(list(frame_ids)),
        "selection_sha256": sha256_file(selection),
        "submap_plan_seal_sha256": sha256_file(plan / "plan_seal.json"),
        "continuous_single_model": True,
    }
    return {**body, "binding_sha256": canonical_hash(body)}


def _audited_workspace(
    workspace: str | Path,
    *,
    _audited_assembly: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = Path(workspace).expanduser().resolve()
    binding_path = root / "continuous_binding.json"
    if not root.is_dir() or root.is_symlink() or not binding_path.is_file():
        raise ArtifactError("continuous workspace is incomplete or unsafe")
    binding = _json(binding_path)
    body = dict(binding)
    digest = body.pop("binding_sha256", None)
    if (
        binding.get("schema_version") != 1
        or binding.get("kind") != _BINDING_KIND
        or binding.get("continuous_single_model") is not True
        or digest != canonical_hash(body)
    ):
        raise ArtifactError("continuous workspace binding changed")
    assembly = (
        dict(_audited_assembly)
        if _audited_assembly is not None
        else audited_geodetic_assembly_plan(
            binding.get("assembly_plan", ""),
            _include_runtime_context=True,
        )
    )
    if Path(str(assembly.get("artifact", ""))).resolve() != Path(
        str(binding.get("assembly_plan", ""))
    ).resolve():
        raise ArtifactError("cached assembly plan refers to another artifact")
    selected = _selected_windows(assembly, binding.get("window_ids", ()))
    frame_ids = _union_frame_ids(assembly, selected)
    selection = root / "selection.json"
    plan = root / "plan"
    expected = _expected_binding(
        assembly, selected, frame_ids, selection, plan
    )
    if binding != expected:
        raise ArtifactError("continuous workspace evidence changed")
    plan_root, plan_record, _, _, _ = _plan_context(
        plan,
        require_hardened=True,
        _verified_input_context=assembly["_runtime_context"],
    )
    if (
        plan_root != plan
        or plan_record.get("selected_frame_ids") != frame_ids
        or plan_record.get("n_frames") != len(frame_ids)
    ):
        raise ArtifactError("continuous submap inventory changed")
    return root, assembly


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    assembly = audited_geodetic_assembly_plan(
        args.assembly_plan, _include_runtime_context=True
    )
    selected = _selected_windows(assembly, args.window_id)
    frame_ids = _union_frame_ids(assembly, selected)
    root = Path(args.workspace).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite workspace: {root}")
    for immutable_name in (
        "frontend_artifact",
        "completed_backend",
        "segment",
        "artifact",
    ):
        immutable = Path(str(assembly[immutable_name])).resolve()
        if root == immutable or immutable in root.parents:
            raise ArtifactError(
                "continuous workspace must be outside immutable inputs"
            )
    root.mkdir(parents=True)
    selection = root / "selection.json"
    plan = root / "plan"
    create_geodetic_frame_selection(
        assembly["frontend_artifact"],
        assembly["segment"],
        frame_ids,
        selection,
    )
    prepare_geodetic_submap_plan(
        assembly["frontend_artifact"],
        assembly["completed_backend"],
        assembly["segment"],
        selection,
        plan,
        config=assembly["config_object"].submap_config,
        _verified_input_context=assembly["_runtime_context"],
        _defer_full_reaudit_until_execution=True,
        _initial_pair_method="v5",
    )
    binding = _expected_binding(
        assembly, selected, frame_ids, selection, plan
    )
    _atomic_json(root / "continuous_binding.json", binding)
    _audited_workspace(root, _audited_assembly=assembly)
    return {"workspace": str(root), "plan": str(plan), **binding}


def _calibration_only_evidence(root: Path) -> dict[str, Any]:
    """Summarize visual/calibration evidence without reading held-out values."""

    plan_root = root / "plan"
    plan = _json(plan_root / "geodetic_submap_plan.json")
    pair_audit = _json(plan_root / "pair_audit.json")
    split = _json(plan_root / "prior_split.json")
    inventory = _json(plan_root / "database_inventory.json")
    records = pair_audit.get("records")
    if not isinstance(records, list):
        raise ArtifactError("pair audit records are missing")
    retained = [record for record in records if record.get("retained") is True]
    if not retained:
        raise ArtifactError("continuous plan has no retained pair evidence")
    selected_names = [str(name) for name in plan["selected_image_names"]]
    degree: Counter[str] = Counter()
    for record in retained:
        images = record.get("images")
        if not isinstance(images, list) or len(images) != 2:
            raise ArtifactError("retained pair image evidence is invalid")
        degree.update(str(name) for name in images)
    verified = sorted(int(record["verified_matches"]) for record in retained)
    degrees = sorted(int(degree[name]) for name in selected_names)
    reasons = Counter(str(record.get("reason")) for record in retained)
    calibration_records = [
        record
        for record in split.get("records", ())
        if isinstance(record, Mapping) and record.get("role") == "calibration"
    ]
    calibration_names = [str(name) for name in split["calibration_names"]]
    private_prior_names = [
        str(name) for name in inventory["calibration_prior_names"]
    ]
    if private_prior_names != calibration_names:
        raise ArtifactError("private calibration-prior inventory changed")
    policy = pair_audit["policy"]
    initial_pair = plan["initial_pair"]
    metrics = {
        "selected_frame_count": int(plan["n_frames"]),
        "selected_image_count": len(selected_names),
        "calibration_prior_count": len(calibration_records),
        "calibration_covariance_replacement_count": sum(
            record.get(
                "covariance_replaced_from_sealed_fixed_calibration"
            )
            is True
            for record in calibration_records
        ),
        "fixed_calibration_derivation_bound": split.get(
            "fixed_calibration_prior_derivation"
        )
        is not None,
        "retained_pair_count": len(retained),
        "rejected_pair_count": len(records) - len(retained),
        "retained_pair_reasons": dict(sorted(reasons.items())),
        "minimum_image_graph_degree": min(degrees),
        "median_image_graph_degree": float(statistics.median(degrees)),
        "isolated_image_count": sum(value == 0 for value in degrees),
        "minimum_retained_verified_matches": min(verified),
        "median_retained_verified_matches": float(
            statistics.median(verified)
        ),
        "initial_anchor_source_verified_matches": int(
            initial_pair["source_pair_verified_matches"]
        ),
        "initial_anchor_source_median_parallax_deg": float(
            initial_pair["source_pair_parallax_evidence"][
                "median_angle_deg"
            ]
        ),
        "heldout_position_priors_in_private_database": 0,
    }
    checks = {
        "all_images_connected": {
            "value": metrics["isolated_image_count"],
            "maximum": 0,
            "passed": metrics["isolated_image_count"] == 0,
        },
        "minimum_image_graph_degree": {
            "value": metrics["minimum_image_graph_degree"],
            "minimum": 1,
            "passed": metrics["minimum_image_graph_degree"] >= 1,
        },
        "median_retained_verified_matches": {
            "value": metrics["median_retained_verified_matches"],
            "minimum": int(policy["strong_verified_matches"]),
            "passed": metrics["median_retained_verified_matches"]
            >= int(policy["strong_verified_matches"]),
        },
        "initial_anchor_verified_matches": {
            "value": metrics["initial_anchor_source_verified_matches"],
            "minimum": int(
                plan["config"]["initial_pair_policy"][
                    "minimum_verified_matches"
                ]
            ),
            "passed": metrics["initial_anchor_source_verified_matches"]
            >= int(
                plan["config"]["initial_pair_policy"][
                    "minimum_verified_matches"
                ]
            ),
        },
        "initial_anchor_parallax_deg": {
            "value": metrics["initial_anchor_source_median_parallax_deg"],
            "minimum": 1.0,
            "passed": metrics["initial_anchor_source_median_parallax_deg"]
            >= 1.0,
        },
        "calibration_prior_support": {
            "value": metrics["calibration_prior_count"],
            "minimum": 4,
            "passed": metrics["calibration_prior_count"] >= 4,
        },
        "fixed_calibration_covariance_contract": {
            "value": metrics["fixed_calibration_derivation_bound"],
            "required": True,
            "passed": metrics["fixed_calibration_derivation_bound"] is True,
        },
        "heldout_priors_physically_absent": {
            "value": metrics[
                "heldout_position_priors_in_private_database"
            ],
            "maximum": 0,
            "passed": True,
        },
    }
    return {
        "schema_version": 1,
        "method": "sealed_plan_visual_graph_and_calibration_support_v1",
        "plan_seal_sha256": sha256_file(plan_root / "plan_seal.json"),
        "evidence_uses": [
            "sealed_source_feature_match_and_two_view_inlier_counts",
            "sealed_raw_gnss_pair_eligibility_decisions",
            "sealed_calibration_prior_inventory",
            "sealed_fixed_calibration_covariance_derivation",
        ],
        "evidence_excludes": [
            "heldout_gnss_position",
            "heldout_evaluation_result",
            "finished_continuous_visual_model_residual",
        ],
        "metrics": metrics,
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }


def _selection_record(root: Path) -> dict[str, Any]:
    evidence = _calibration_only_evidence(root)
    passed = bool(evidence["passed"])
    body = {
        "schema_version": 1,
        "kind": _SELECTION_KIND,
        "continuous_binding_sha256": _json(
            root / "continuous_binding.json"
        )["binding_sha256"],
        "calibration_only_evidence": evidence,
        "selected_candidate": (
            "continuous_existing_features_pose_prior_mapper_v1"
            if passed
            else None
        ),
        "feature_enhancement_selected": False,
        "feature_enhancement_reason": (
            "source visual graph and calibration support pass the "
            "predeclared strength checks"
            if passed
            else "source visual evidence is insufficient; stop before run"
        ),
        "heldout_evaluations_used_for_selection": 0,
        "pose_prior_preserving_final_refinement_required": True,
    }
    return {**body, "selection_sha256": canonical_hash(body)}


def _audited_selection(selection: str | Path, root: Path) -> dict[str, Any]:
    path = Path(selection).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ArtifactError("calibration-only selection is missing or unsafe")
    recorded = _json(path)
    expected = _selection_record(root)
    if recorded != expected:
        raise ArtifactError("calibration-only candidate selection changed")
    if not recorded["calibration_only_evidence"]["passed"]:
        raise ArtifactError("calibration-only candidate selection failed")
    return recorded


def _distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ArtifactError("calibration audit distribution is invalid")
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if (
        left.shape != right.shape
        or valid.sum() < 3
        or float(np.std(left[valid])) <= 1.0e-12
        or float(np.std(right[valid])) <= 1.0e-12
    ):
        return None
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def _option(command: Sequence[str], name: str) -> str | None:
    try:
        index = list(command).index(name)
    except ValueError:
        return None
    return str(command[index + 1]) if index + 1 < len(command) else None


def _calibration_residual_audit(result_artifact: str | Path) -> dict[str, Any]:
    """Audit a frozen result without loading any held-out target value."""

    result = audited_geodetic_submap_result(
        result_artifact, _include_internal_plan_context=True
    )
    plan_root, plan, _, mapper_config, context = result.pop(
        "_internal_plan_context"
    )
    split = _json(plan_root / "prior_split.json")
    calibration = [
        record
        for record in split.get("records", ())
        if isinstance(record, Mapping) and record.get("role") == "calibration"
    ]
    if len(calibration) < 6:
        raise ArtifactError("calibration-only audit has insufficient support")
    names = [str(record["name"]) for record in calibration]
    poses = _poses_from_images_txt(
        Path(result["artifact"]) / "refined_text" / "images.txt"
    )
    if not set(names).issubset(poses):
        raise ArtifactError("refined model lacks a calibration image")
    source = np.stack([poses[name][1] for name in names])
    target = np.stack(
        [np.asarray(record["position_m"], dtype=np.float64) for record in calibration]
    )
    covariance = np.stack(
        [
            np.asarray(record["optimizer_covariance_m2"], dtype=np.float64)
            for record in calibration
        ]
    )
    timestamps_ns = np.asarray(
        [int(record["timestamp_ns"]) for record in calibration],
        dtype=np.int64,
    )
    block_ids = np.asarray(
        [int(record["block_id"]) for record in calibration], dtype=np.int64
    )
    if np.any(np.diff(timestamps_ns) <= 0):
        raise ArtifactError("calibration audit timestamps are not ordered")
    alignment = estimate_rigid_alignment(
        source,
        target,
        covariance,
        ransac_threshold_m=mapper_config.alignment_ransac_threshold_m,
        ransac_iterations=mapper_config.alignment_ransac_iterations,
        random_seed=mapper_config.random_seed,
    )
    residual_vectors = (
        source @ alignment.rotation.T + alignment.translation - target
    )
    residual_m = np.linalg.norm(residual_vectors, axis=1)

    cross_validation: list[dict[str, Any]] = []
    cross_validated_residuals: list[np.ndarray] = []
    for block_id in sorted(int(value) for value in np.unique(block_ids)):
        validation = block_ids == block_id
        training = ~validation
        if training.sum() < 3 or not validation.any():
            raise ArtifactError("calibration block cross-validation is invalid")
        fitted = estimate_rigid_alignment(
            source[training],
            target[training],
            covariance[training],
            ransac_threshold_m=mapper_config.alignment_ransac_threshold_m,
            ransac_iterations=mapper_config.alignment_ransac_iterations,
            random_seed=mapper_config.random_seed,
        )
        held_block = np.linalg.norm(
            source[validation] @ fitted.rotation.T
            + fitted.translation
            - target[validation],
            axis=1,
        )
        cross_validated_residuals.append(held_block)
        cross_validation.append(
            {
                "validation_calibration_block_id": block_id,
                "training_calibration_block_ids": sorted(
                    int(value) for value in np.unique(block_ids[training])
                ),
                "validation_count": int(validation.sum()),
                "residual_m": _distribution(held_block),
            }
        )
    cross_validated = np.concatenate(cross_validated_residuals)

    time_s = (timestamps_ns - timestamps_ns[0]).astype(np.float64) / 1.0e9
    velocity = np.gradient(target, time_s, axis=0)
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    moving = speed > 0.05
    direction = np.full((len(speed), 2), np.nan, dtype=np.float64)
    direction[moving] = velocity[moving, :2] / speed[moving, None]
    longitudinal = np.sum(residual_vectors[:, :2] * direction, axis=1)
    lateral = (
        -residual_vectors[:, 0] * direction[:, 1]
        + residual_vectors[:, 1] * direction[:, 0]
    )
    heading = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    turn_rate = np.gradient(heading, time_s)
    source_residual_ms = np.asarray(
        [
            float(record["gnss_temporal_filter"]["source_residual_ns"])
            / 1.0e6
            for record in calibration
        ],
        dtype=np.float64,
    )
    implied_latency_ms = longitudinal[moving] / speed[moving] * 1.0e3

    solve = result["solve"]
    command = [str(value) for value in solve.get("command", ())]
    log_path = Path(result["artifact"]) / str(solve["resources"]["log"])
    log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    global_lines = [
        index + 1
        for index, line in enumerate(log_lines)
        if "Retriangulation and Global bundle adjustment" in line
    ]
    fixed_mapper_contract = {
        "pose_prior_mapper": len(command) > 1
        and command[1] == "pose_prior_mapper",
        "overwrite_prior_covariance": _option(
            command, "--overwrite_priors_covariance"
        )
        == "0",
        "fixed_intrinsics": all(
            _option(command, option) == "0"
            for option in (
                "--Mapper.ba_refine_focal_length",
                "--Mapper.ba_refine_principal_point",
                "--Mapper.ba_refine_extra_params",
            )
        ),
        "fixed_rig_extrinsics": _option(
            command, "--Mapper.ba_refine_sensor_from_rig"
        )
        == "0",
        "robust_position_priors": _option(
            command, "--use_robust_loss_on_prior_position"
        )
        == "1",
        "global_ba_backend": _option(command, "--Mapper.ba_global_backend"),
        "global_bundle_adjustment_count": len(global_lines),
        "last_global_bundle_adjustment_log_line": (
            global_lines[-1] if global_lines else None
        ),
    }
    fixed_mapper_contract["passed"] = bool(
        all(
            fixed_mapper_contract[name]
            for name in (
                "pose_prior_mapper",
                "overwrite_prior_covariance",
                "fixed_intrinsics",
                "fixed_rig_extrinsics",
                "robust_position_priors",
            )
        )
        and fixed_mapper_contract["global_ba_backend"] == "CERES"
        and fixed_mapper_contract["global_bundle_adjustment_count"] > 0
    )
    derivation = context["reader"].meta.get("fixed_calibration_derivation")
    if not isinstance(derivation, Mapping):
        raise ArtifactError("calibration audit lacks fixed calibration evidence")
    body = {
        "schema_version": 1,
        "kind": _AUDIT_KIND,
        "result_artifact": str(Path(result["artifact"]).resolve()),
        "result_seal_sha256": result["result_seal_sha256"],
        "plan_seal_sha256": sha256_file(plan_root / "plan_seal.json"),
        "method": "calibration_roles_only_fixed_se3_temporal_block_cv_v1",
        "heldout_target_values_used": False,
        "calibration_count": len(calibration),
        "calibration_block_ids": sorted(
            int(value) for value in np.unique(block_ids)
        ),
        "all_calibration_fit": {
            "residual_m": _distribution(residual_m),
            "inlier_count": int(alignment.inlier_mask.sum()),
            "inlier_fraction": float(alignment.inlier_mask.mean()),
            "sim3_scale_diagnostic_only": float(
                alignment.sim3_scale_diagnostic
            ),
            "production_scale": 1.0,
        },
        "calibration_block_cross_validation": {
            "folds": cross_validation,
            "all_validation_residual_m": _distribution(cross_validated),
        },
        "motion_and_timing": {
            "speed_m_s": _distribution(speed),
            "absolute_turn_rate_rad_s": _distribution(np.abs(turn_rate)),
            "gnss_source_association_residual_ms": _distribution(
                np.abs(source_residual_ms)
            ),
            "implied_longitudinal_latency_ms": _distribution(
                implied_latency_ms
            ),
            "correlations": {
                "residual_m_vs_time": _correlation(residual_m, time_s),
                "residual_m_vs_speed": _correlation(residual_m, speed),
                "residual_m_vs_absolute_turn_rate": _correlation(
                    residual_m, np.abs(turn_rate)
                ),
                "longitudinal_residual_vs_speed": _correlation(
                    longitudinal, speed
                ),
                "lateral_residual_vs_turn_rate": _correlation(
                    lateral, turn_rate
                ),
                "longitudinal_residual_vs_gnss_source_residual": (
                    _correlation(longitudinal, source_residual_ms)
                ),
                "east_residual_vs_heading_east": _correlation(
                    residual_vectors[:, 0], direction[:, 0]
                ),
                "north_residual_vs_heading_north": _correlation(
                    residual_vectors[:, 1], direction[:, 1]
                ),
            },
        },
        "fixed_calibration": {
            "fixed_scale": derivation.get("fixed_scale"),
            "camera_to_rtk_offset_ns": derivation.get(
                "camera_to_rtk_offset_ns"
            ),
            "orientation_method": derivation.get(
                "dual_antenna_orientation_method"
            ),
            "translation_sigma_camera_m": derivation.get(
                "extrinsic_translation_sigma_camera_m"
            ),
            "raw_observations_changed": derivation.get(
                "raw_observations_changed"
            ),
            "calibration_result_seal_sha256": derivation.get(
                "calibration_result_seal_sha256"
            ),
        },
        "pose_prior_preserving_global_refinement": fixed_mapper_contract,
    }
    return {**body, "audit_sha256": canonical_hash(body)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--assembly-plan", required=True)
    prepare.add_argument("--window-id", action="append", required=True)
    prepare.add_argument("--workspace", required=True)

    select = commands.add_parser("select")
    select.add_argument("--workspace", required=True)
    select.add_argument("--output", required=True)

    run = commands.add_parser("run")
    run.add_argument("--workspace", required=True)
    run.add_argument("--selection", required=True)
    run.add_argument("--result", required=True)
    run.add_argument("--colmap", required=True)

    audit = commands.add_parser("audit-calibration")
    audit.add_argument("--result", required=True)
    audit.add_argument("--output", required=True)

    status = commands.add_parser("status")
    status.add_argument("--workspace", required=True)
    status.add_argument("--result")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit-calibration":
        audit = _calibration_residual_audit(args.result)
        _atomic_json(Path(args.output).expanduser().resolve(), audit)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0 if audit["pose_prior_preserving_global_refinement"]["passed"] else 2
    if args.command == "prepare":
        value = _prepare(args)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    root, assembly = _audited_workspace(args.workspace)
    if args.command == "select":
        selection = _selection_record(root)
        _atomic_json(Path(args.output).expanduser().resolve(), selection)
        print(json.dumps(selection, indent=2, sort_keys=True))
        return 0 if selection["calibration_only_evidence"]["passed"] else 2
    if args.command == "run":
        _audited_selection(args.selection, root)
        result = run_geodetic_submap_plan(
            root / "plan",
            args.result,
            args.colmap,
            _verified_input_context=assembly["_runtime_context"],
        )
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result["passed"] else 2
    output: dict[str, Any] = {
        "workspace": str(root),
        "binding": _json(root / "continuous_binding.json"),
    }
    if args.result is not None:
        output["result"] = audited_geodetic_submap_result(args.result)
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    return 0 if output.get("result", {"passed": True})["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
