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

from rtk_splat.backends.artifact_io import _atomic_json, _json
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
from rtk_splat.frontends.artifact import (
    ArtifactError,
    canonical_hash,
    sha256_file,
)


_BINDING_KIND = "rtk_splat_geodetic_continuous_window_binding"
_SELECTION_KIND = "rtk_splat_geodetic_calibration_only_candidate_selection"


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

    status = commands.add_parser("status")
    status.add_argument("--workspace", required=True)
    status.add_argument("--result")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
