#!/usr/bin/env python3
"""Run a generic, resumable diagnostic geodetic assembly sequence.

All dataset locations, reused results, window order, and destinations are
explicit inputs.  This launcher never converts a diagnostic result into a
production result; the reusable backend retains every production-gate failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rtk_splat.backends.geodetic_assembly import (
    GeodeticDiagnosticPolicy,
    audited_geodetic_assembly_plan,
    audited_geodetic_diagnostic_initialization_fallback_window,
    audited_geodetic_diagnostic_low_parallax_fallback_window,
    audited_geodetic_diagnostic_full_pose_artifact,
    audited_geodetic_diagnostic_overlap_report,
    audited_geodetic_diagnostic_result_inventory,
    audited_geodetic_diagnostic_submap_result,
    prepare_geodetic_assembly_window,
    prepare_geodetic_diagnostic_initialization_fallback_window,
    prepare_geodetic_diagnostic_low_parallax_fallback_window,
    publish_geodetic_diagnostic_full_pose_artifact,
    publish_geodetic_diagnostic_overlap_report,
    publish_geodetic_diagnostic_result_inventory,
)
from rtk_splat.backends.geodetic_submap import (
    audited_geodetic_submap_failure,
    run_geodetic_submap_plan,
)


def _binding(value: str) -> tuple[str, Path]:
    window_id, separator, path = value.partition("=")
    if not separator or not window_id or not path:
        raise argparse.ArgumentTypeError("expected WINDOW_ID=RESULT_PATH")
    return window_id, Path(path).expanduser()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--colmap", required=True, type=Path)
    parser.add_argument("--pose-name", required=True)
    parser.add_argument("--reuse", action="append", type=_binding, default=[])
    parser.add_argument(
        "--run-window",
        action="append",
        default=[],
        help="window ID to solve, in launch order; default is every non-reused window",
    )
    parser.add_argument(
        "--median-rtk-warning-m",
        type=float,
        default=GeodeticDiagnosticPolicy().median_rtk_warning_m,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="verify and continue an existing pilot root without overwriting outputs",
    )
    parser.add_argument(
        "--allow-sealed-auto-initialization-fallback",
        action="store_true",
        help=(
            "diagnostic-only: after a sealed v5 initialization-exhaustion "
            "failure, retry with COLMAP auto-initialization over the same "
            "filtered private database"
        ),
    )
    parser.add_argument(
        "--allow-sealed-low-parallax-initialization-fallback",
        action="store_true",
        help=(
            "diagnostic-only: after a sealed v5 initialization-exhaustion "
            "failure, retry its predeclared calibration pair with the sealed "
            "low-parallax initialization angle before COLMAP auto fallback"
        ),
    )
    return parser


def _result_path(root: Path, window_id: str) -> Path:
    return root / "submaps" / window_id / "result"


def _fallback_workspace(root: Path, window_id: str) -> Path:
    return root / "diagnostic-initialization-fallbacks" / window_id


def _low_parallax_fallback_workspace(root: Path, window_id: str) -> Path:
    return root / "diagnostic-low-parallax-fallbacks" / window_id


def _publish_available_overlaps(
    windows: list[dict],
    results: dict[str, Path],
    root: Path,
    overlap_policy,
    diagnostic_policy: GeodeticDiagnosticPolicy,
    verified_input_context,
) -> list[dict]:
    records = []
    for first, second in zip(windows, windows[1:]):
        first_id = str(first["window_id"])
        second_id = str(second["window_id"])
        if first_id not in results or second_id not in results:
            continue
        destination = root / "overlaps" / f"{first_id}--{second_id}"
        if destination.exists():
            report = audited_geodetic_diagnostic_overlap_report(
                destination,
                _verified_input_context=verified_input_context,
            )
        else:
            published = publish_geodetic_diagnostic_overlap_report(
                [results[first_id], results[second_id]],
                destination,
                overlap_policy=overlap_policy,
                diagnostic_policy=diagnostic_policy,
                _verified_input_context=verified_input_context,
            )
            report = json.loads(
                (published / "diagnostic_overlap.json").read_text(
                    encoding="utf-8"
                )
            )
            report["artifact"] = str(published)
        expected_paths = {
            str(results[first_id].resolve()), str(results[second_id].resolve())
        }
        recorded_paths = {
            str(item["result"])
            for item in report["submap_assessments"]
        }
        if recorded_paths != expected_paths:
            raise RuntimeError(
                f"existing overlap {first_id}--{second_id} binds other results"
            )
        records.append(
            {
                "first_window_id": first_id,
                "second_window_id": second_id,
                "artifact": report["artifact"],
                "production_passed": report["production_passed"],
                "structurally_safe_for_diagnostic": report[
                    "structurally_safe_for_diagnostic"
                ],
                "synchronized_rtk_assessments": report[
                    "synchronized_rtk_assessments"
                ],
            }
        )
    return records


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.pilot_root.expanduser().resolve()
    if args.resume:
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError("--resume requires an existing safe pilot root")
    else:
        root.mkdir(parents=True, exist_ok=False)
    assembly = audited_geodetic_assembly_plan(
        args.plan, _include_runtime_context=True
    )
    windows = list(assembly["windows"])
    by_id = {str(window["window_id"]): window for window in windows}
    reuse = dict(args.reuse)
    if len(reuse) != len(args.reuse) or not set(reuse) <= set(by_id):
        raise ValueError("reused window bindings are duplicate or unknown")
    requested = list(args.run_window) or [
        window_id for window_id in by_id if window_id not in reuse
    ]
    if (
        len(requested) != len(set(requested))
        or not set(requested) <= set(by_id)
        or set(requested) & set(reuse)
        or set(requested) | set(reuse) != set(by_id)
    ):
        raise ValueError(
            "reuse and run-window inputs must partition every assembly window exactly"
        )
    policy = GeodeticDiagnosticPolicy(
        median_rtk_warning_m=args.median_rtk_warning_m,
        allow_sealed_auto_initialization_fallback=(
            args.allow_sealed_auto_initialization_fallback
        ),
        allow_sealed_low_parallax_initialization_fallback=(
            args.allow_sealed_low_parallax_initialization_fallback
        ),
    )
    results: dict[str, Path] = {}
    submap_records = []
    for window_id, path in reuse.items():
        result = path.expanduser().resolve()
        assessment = audited_geodetic_diagnostic_submap_result(
            result,
            policy=policy,
            _verified_input_context=assembly["_runtime_context"],
        )
        results[window_id] = result
        submap_records.append(
            {"window_id": window_id, "state": "reused", **assessment}
        )
    overlap_records = _publish_available_overlaps(
        windows,
        results,
        root,
        assembly["config_object"].overlap_policy,
        policy,
        assembly["_runtime_context"],
    )

    for index, window_id in enumerate(requested, start=1):
        workspace = root / "submaps" / window_id
        primary_result_path = _result_path(root, window_id)
        fallback_workspace = _fallback_workspace(root, window_id)
        fallback_result_path = fallback_workspace / "result"
        low_fallback_workspace = _low_parallax_fallback_workspace(
            root, window_id
        )
        low_fallback_result_path = low_fallback_workspace / "result"
        existing_results = [
            path
            for path in (
                primary_result_path,
                low_fallback_result_path,
                fallback_result_path,
            )
            if path.exists()
        ]
        if len(existing_results) > 1:
            raise RuntimeError(
                f"{window_id} has multiple diagnostic initialization results"
            )
        if low_fallback_result_path.exists():
            audited_geodetic_diagnostic_low_parallax_fallback_window(
                args.plan,
                low_fallback_workspace,
                _audited_plan=assembly,
            )
            result_path = low_fallback_result_path
            state = "verified-diagnostic-low-parallax-initialization-fallback"
        elif fallback_result_path.exists():
            audited_geodetic_diagnostic_initialization_fallback_window(
                args.plan,
                fallback_workspace,
                _audited_plan=assembly,
            )
            result_path = fallback_result_path
            state = "verified-diagnostic-initialization-fallback"
        elif primary_result_path.exists():
            result_path = primary_result_path
            state = "verified"
        else:
            primary_failure_path = workspace / "failure"
            if workspace.exists():
                if not primary_failure_path.exists():
                    raise RuntimeError(
                        f"{window_id} has an incomplete primary workspace"
                    )
                prepared = {
                    "plan": str(workspace / "plan"),
                    "result": str(primary_result_path),
                }
            else:
                prepared = prepare_geodetic_assembly_window(
                    args.plan,
                    window_id,
                    workspace,
                    _audited_plan=assembly,
                )
            if primary_failure_path.exists():
                failure = audited_geodetic_submap_failure(
                    primary_failure_path,
                    _verified_input_context=assembly["_runtime_context"],
                )
            else:
                try:
                    run_geodetic_submap_plan(
                        prepared["plan"],
                        prepared["result"],
                        args.colmap,
                        failure_destination=primary_failure_path,
                        _verified_input_context=assembly["_runtime_context"],
                    )
                except BaseException:
                    failure = audited_geodetic_submap_failure(
                        primary_failure_path,
                        _verified_input_context=assembly["_runtime_context"],
                    )
                else:
                    failure = None
            if failure is None:
                result_path = primary_result_path
                state = "completed"
            else:
                if failure["failure_classification"] != (
                    "initialization_exhausted_without_sparse_model"
                ):
                    raise RuntimeError(
                        f"{window_id} primary solve failed: "
                        f"{failure['failure_classification']}"
                    )
                result_path = None
                low_failure = None
                if args.allow_sealed_low_parallax_initialization_fallback:
                    if low_fallback_workspace.exists():
                        low_fallback = (
                            audited_geodetic_diagnostic_low_parallax_fallback_window(
                                args.plan,
                                low_fallback_workspace,
                                _audited_plan=assembly,
                            )
                        )
                    else:
                        low_fallback = (
                            prepare_geodetic_diagnostic_low_parallax_fallback_window(
                                args.plan,
                                window_id,
                                primary_failure_path,
                                low_fallback_workspace,
                                _audited_plan=assembly,
                            )
                        )
                    low_failure_path = low_fallback_workspace / "failure"
                    if low_failure_path.exists():
                        low_failure = audited_geodetic_submap_failure(
                            low_failure_path,
                            _verified_input_context=assembly["_runtime_context"],
                        )
                    else:
                        try:
                            run_geodetic_submap_plan(
                                low_fallback["plan"],
                                low_fallback["result"],
                                args.colmap,
                                failure_destination=low_failure_path,
                                _verified_input_context=assembly[
                                    "_runtime_context"
                                ],
                            )
                        except BaseException:
                            low_failure = audited_geodetic_submap_failure(
                                low_failure_path,
                                _verified_input_context=assembly[
                                    "_runtime_context"
                                ],
                            )
                    if low_failure is None:
                        result_path = low_fallback_result_path
                        state = (
                            "completed-diagnostic-low-parallax-"
                            "initialization-fallback"
                        )
                    elif low_failure["failure_classification"] != (
                        "initialization_exhausted_without_sparse_model"
                    ):
                        raise RuntimeError(
                            f"{window_id} low-parallax fallback failed: "
                            f"{low_failure['failure_classification']}"
                        )

                if result_path is None:
                    if not args.allow_sealed_auto_initialization_fallback:
                        failed = low_failure or failure
                        raise RuntimeError(
                            f"{window_id} initialization fallback exhausted: "
                            f"{failed['failure_classification']}"
                        )
                    if fallback_workspace.exists():
                        fallback = (
                            audited_geodetic_diagnostic_initialization_fallback_window(
                                args.plan,
                                fallback_workspace,
                                _audited_plan=assembly,
                            )
                        )
                    else:
                        fallback = (
                            prepare_geodetic_diagnostic_initialization_fallback_window(
                                args.plan,
                                window_id,
                                primary_failure_path,
                                fallback_workspace,
                                _audited_plan=assembly,
                            )
                        )
                    fallback_failure_path = fallback_workspace / "failure"
                    if fallback_failure_path.exists():
                        fallback_failure = audited_geodetic_submap_failure(
                            fallback_failure_path,
                            _verified_input_context=assembly["_runtime_context"],
                        )
                        raise RuntimeError(
                            f"{window_id} diagnostic fallback failed: "
                            f"{fallback_failure['failure_classification']}"
                        )
                    run_geodetic_submap_plan(
                        fallback["plan"],
                        fallback["result"],
                        args.colmap,
                        failure_destination=fallback_failure_path,
                        _verified_input_context=assembly["_runtime_context"],
                    )
                    result_path = fallback_result_path
                    state = "completed-diagnostic-initialization-fallback"
        assessment = audited_geodetic_diagnostic_submap_result(
            result_path,
            policy=policy,
            _verified_input_context=assembly["_runtime_context"],
        )
        results[window_id] = result_path.resolve()
        submap_records.append(
            {"window_id": window_id, "state": state, **assessment}
        )
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(requested)}",
                    "window_id": window_id,
                    "state": state,
                    "production_passed": assessment["production_passed"],
                    "production_failed_checks": assessment[
                        "production_failed_checks"
                    ],
                    "structurally_safe_for_diagnostic": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        overlap_records = _publish_available_overlaps(
            windows,
            results,
            root,
            assembly["config_object"].overlap_policy,
            policy,
            assembly["_runtime_context"],
        )

    inventory_path = root / "diagnostic-result-inventory"
    if inventory_path.exists():
        inventory = audited_geodetic_diagnostic_result_inventory(
            inventory_path, _verified_assembly=assembly
        )
    else:
        inventory_path = publish_geodetic_diagnostic_result_inventory(
            args.plan,
            results,
            inventory_path,
            policy=policy,
        )
        inventory = json.loads(
            (inventory_path / "diagnostic_result_inventory.json").read_text(
                encoding="utf-8"
            )
        )
        inventory["artifact"] = str(inventory_path)
    pose_path = root / "pose_artifacts" / args.pose_name
    if pose_path.exists():
        pose = audited_geodetic_diagnostic_full_pose_artifact(pose_path)
    else:
        published_pose = publish_geodetic_diagnostic_full_pose_artifact(
            inventory_path, root / "pose_artifacts", args.pose_name
        )
        pose = {
            "artifact": str(published_pose),
            "quality": json.loads(
                (published_pose / "quality.json").read_text(encoding="utf-8")
            ),
        }
    summary = {
        "schema_version": 1,
        "diagnostic_only": True,
        "pilot_root": str(root),
        "submaps": sorted(submap_records, key=lambda item: item["window_id"]),
        "overlaps": overlap_records,
        "inventory": inventory["artifact"],
        "pose": pose["artifact"],
        "pose_quality": {
            "production_passed": pose["quality"]["passed"],
            "production_failed_checks": pose["quality"][
                "production_failed_checks"
            ],
            "diagnostic_structurally_safe": pose["quality"][
                "diagnostic_structurally_safe"
            ],
            "global_holdout_median_rtk_residual_m": pose["quality"][
                "global_holdout_median_rtk_residual_m"
            ],
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
