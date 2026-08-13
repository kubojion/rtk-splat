#!/usr/bin/env python3
"""Experimental orchestration for sealed overlapping geodetic submaps.

The installed package remains dataset-independent.  Every source, selection,
workspace, result, and pose destination is supplied explicitly by the caller.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from typing import Sequence

from rtk_splat.backends.geodetic_assembly import (
    GeodeticAssemblyConfig,
    GeodeticOverlapPolicy,
    audited_geodetic_assembly_plan,
    audited_geodetic_full_pose_artifact,
    audited_geodetic_overlap_report,
    audited_geodetic_submap_pose_export,
    export_geodetic_submap_poses,
    prepare_geodetic_assembly_plan,
    prepare_geodetic_assembly_window,
    publish_geodetic_full_pose_artifact,
    publish_geodetic_overlap_report,
)
from rtk_splat.backends.geodetic_submap import (
    GeodeticSubmapConfig,
    audited_geodetic_submap_result,
    create_geodetic_frame_selection,
    run_geodetic_submap_plan,
)


def _json_print(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str), flush=True)


def _submap_config(args: argparse.Namespace) -> GeodeticSubmapConfig:
    base = GeodeticSubmapConfig()
    refinement = replace(
        base.refinement,
        num_threads=args.num_threads,
        random_seed=args.random_seed,
        minimum_free_space_gb=args.minimum_free_space_gb,
        minimum_runtime_free_space_gb=args.minimum_runtime_free_space_gb,
    )
    return replace(base, refinement=refinement)


def _assembly_config(args: argparse.Namespace) -> GeodeticAssemblyConfig:
    overlap = GeodeticOverlapPolicy(
        minimum_overlap_frames=args.minimum_overlap_frames,
        max_median_center_disagreement_m=(
            args.max_median_center_disagreement_m
        ),
        max_p95_center_disagreement_m=args.max_p95_center_disagreement_m,
        max_center_disagreement_m=args.max_center_disagreement_m,
        max_median_rotation_disagreement_deg=(
            args.max_median_rotation_disagreement_deg
        ),
        max_p95_rotation_disagreement_deg=(
            args.max_p95_rotation_disagreement_deg
        ),
        max_rotation_disagreement_deg=args.max_rotation_disagreement_deg,
    )
    return GeodeticAssemblyConfig(
        submap_frames=args.submap_frames,
        overlap_frames=args.overlap_frames,
        probe_submaps=args.probe_submaps,
        blend_boundary_power=args.blend_boundary_power,
        overlap_policy=overlap,
        submap_config=_submap_config(args),
    )


def _add_config(parser: argparse.ArgumentParser) -> None:
    defaults = GeodeticAssemblyConfig()
    policy = defaults.overlap_policy
    parser.add_argument("--submap-frames", type=int, default=defaults.submap_frames)
    parser.add_argument("--overlap-frames", type=int, default=defaults.overlap_frames)
    parser.add_argument("--probe-submaps", type=int, default=defaults.probe_submaps)
    parser.add_argument(
        "--blend-boundary-power", type=float, default=defaults.blend_boundary_power
    )
    parser.add_argument(
        "--minimum-overlap-frames",
        type=int,
        default=policy.minimum_overlap_frames,
    )
    parser.add_argument(
        "--max-median-center-disagreement-m",
        type=float,
        default=policy.max_median_center_disagreement_m,
    )
    parser.add_argument(
        "--max-p95-center-disagreement-m",
        type=float,
        default=policy.max_p95_center_disagreement_m,
    )
    parser.add_argument(
        "--max-center-disagreement-m",
        type=float,
        default=policy.max_center_disagreement_m,
    )
    parser.add_argument(
        "--max-median-rotation-disagreement-deg",
        type=float,
        default=policy.max_median_rotation_disagreement_deg,
    )
    parser.add_argument(
        "--max-p95-rotation-disagreement-deg",
        type=float,
        default=policy.max_p95_rotation_disagreement_deg,
    )
    parser.add_argument(
        "--max-rotation-disagreement-deg",
        type=float,
        default=policy.max_rotation_disagreement_deg,
    )
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--minimum-free-space-gb", type=float, default=10.0)
    parser.add_argument(
        "--minimum-runtime-free-space-gb", type=float, default=5.0
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    selection = commands.add_parser("seal-full-selection")
    selection.add_argument("--frontend", required=True)
    selection.add_argument("--segment", required=True)
    selection.add_argument("--output", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--frontend", required=True)
    plan.add_argument("--backend", required=True)
    plan.add_argument("--segment", required=True)
    plan.add_argument("--selection", required=True)
    plan.add_argument("--output", required=True)
    _add_config(plan)

    prepare = commands.add_parser("prepare-all")
    prepare.add_argument("--plan", required=True)
    prepare.add_argument("--submaps-root", required=True)
    prepare.add_argument("--window-id", action="append")
    prepare.add_argument("--limit", type=int)

    run = commands.add_parser("run-all")
    run.add_argument("--plan", required=True)
    run.add_argument("--submaps-root", required=True)
    run.add_argument("--colmap", required=True)
    run.add_argument("--window-id", action="append")
    run.add_argument("--limit", type=int)
    run.add_argument("--jobs", type=int, default=1)

    export = commands.add_parser("export-submap")
    export.add_argument("--result", required=True)
    export.add_argument("--output", required=True)

    overlap = commands.add_parser("overlap")
    overlap.add_argument("--result", action="append", required=True)
    overlap.add_argument("--output", required=True)
    _add_config(overlap)

    assemble = commands.add_parser("assemble")
    assemble.add_argument("--plan", required=True)
    assemble.add_argument("--submaps-root", required=True)
    assemble.add_argument("--pose-root", required=True)
    assemble.add_argument("--pose-name", required=True)

    status = commands.add_parser("status")
    status.add_argument("--plan", required=True)
    status.add_argument("--submaps-root", required=True)
    status.add_argument("--verify", action="store_true")
    status.add_argument("--pose")
    status.add_argument("--overlap-report")
    status.add_argument("--submap-export")
    return parser


def _selected_windows(args: argparse.Namespace, plan: dict) -> list[dict]:
    windows = list(plan["windows"])
    if getattr(args, "window_id", None):
        wanted = list(args.window_id)
        by_id = {item["window_id"]: item for item in windows}
        missing = [item for item in wanted if item not in by_id]
        if missing:
            raise ValueError("unknown window IDs: " + ", ".join(missing))
        windows = [by_id[item] for item in wanted]
    limit = getattr(args, "limit", None)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        windows = windows[:limit]
    return windows


def _prepare(args: argparse.Namespace) -> list[dict[str, str]]:
    plan = audited_geodetic_assembly_plan(
        args.plan, _include_runtime_context=True
    )
    selected = _selected_windows(args, plan)
    root = Path(args.submaps_root).expanduser().resolve()
    prepared = []
    for index, window in enumerate(selected, start=1):
        print(
            f"preparing {window['window_id']} ({index}/{len(selected)})",
            flush=True,
        )
        prepared.append(
            prepare_geodetic_assembly_window(
                args.plan,
                window["window_id"],
                root / window["window_id"],
                _audited_plan=plan,
            )
        )
    return prepared


def _run_one(item: dict[str, str], colmap: str) -> dict:
    result_path = Path(item["result"])
    if result_path.exists():
        result = audited_geodetic_submap_result(result_path)
        if not result["passed"]:
            raise RuntimeError(f"existing {item['window_id']} result failed")
        return {"window_id": item["window_id"], "state": "verified", **result}
    result = run_geodetic_submap_plan(
        item["plan"],
        result_path,
        colmap,
        _verified_input_context=item.get("_verified_input_context"),
    )
    if not result["passed"]:
        raise RuntimeError(f"{item['window_id']} failed acceptance gates")
    return {"window_id": item["window_id"], "state": "completed", **result}


def _prepared_for_run(args: argparse.Namespace) -> list[dict[str, str]]:
    plan = audited_geodetic_assembly_plan(
        args.plan, _include_runtime_context=True
    )
    selected = _selected_windows(args, plan)
    root = Path(args.submaps_root).expanduser().resolve()
    prepared: list[dict[str, str]] = []
    for window in selected:
        workspace = root / window["window_id"]
        selection = workspace / "selection.json"
        local_plan = workspace / "plan"
        binding_path = workspace / "window_binding.json"
        if selection.is_file() and local_plan.is_dir() and binding_path.is_file():
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
            if (
                binding.get("assembly_plan_seal_sha256")
                != plan["plan_seal_sha256"]
                or binding.get("window_id") != window["window_id"]
                or binding.get("frame_ids_sha256")
                != window["frame_ids_sha256"]
            ):
                raise RuntimeError(
                    f"prepared binding changed: {window['window_id']}"
                )
            prepared.append(
                {
                    "workspace": str(workspace),
                    "selection": str(selection),
                    "plan": str(local_plan),
                    "result": str(workspace / "result"),
                    "window_id": window["window_id"],
                    "_verified_input_context": plan["_runtime_context"],
                }
            )
        else:
            item = prepare_geodetic_assembly_window(
                args.plan,
                window["window_id"],
                workspace,
                _audited_plan=plan,
            )
            item["_verified_input_context"] = plan["_runtime_context"]
            prepared.append(item)
    return prepared


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "seal-full-selection":
        manifest = json.loads(
            (Path(args.frontend) / "frame_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        frame_ids = [int(item["frame_id"]) for item in manifest["frames"]]
        print(
            create_geodetic_frame_selection(
                args.frontend, args.segment, frame_ids, args.output
            )
        )
        return 0
    if args.command == "plan":
        print(
            prepare_geodetic_assembly_plan(
                args.frontend,
                args.backend,
                args.segment,
                args.selection,
                args.output,
                config=_assembly_config(args),
            )
        )
        return 0
    if args.command == "prepare-all":
        _json_print({"prepared": _prepare(args)})
        return 0
    if args.command == "run-all":
        if args.jobs < 1 or args.jobs > 8:
            raise ValueError("--jobs must be in [1, 8]")
        prepared = _prepared_for_run(args)
        completed = []
        failures = []
        # The existing monitored mapper installs SIGINT/SIGTERM handlers and
        # therefore must own the main thread of its worker.  Spawned processes
        # also isolate each COLMAP process group and avoid forking this
        # NumPy/SQLite-aware parent after its assembly audit.
        with ProcessPoolExecutor(
            max_workers=args.jobs, mp_context=get_context("spawn")
        ) as executor:
            futures = {
                executor.submit(_run_one, item, args.colmap): item
                for item in prepared
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    record = future.result()
                    completed.append(record)
                    print(
                        f"{item['window_id']}: {record['state']} and passed",
                        flush=True,
                    )
                except BaseException as exc:
                    failures.append(
                        {"window_id": item["window_id"], "error": repr(exc)}
                    )
                    print(f"{item['window_id']}: FAILED: {exc}", flush=True)
        _json_print(
            {
                "completed": sorted(
                    completed, key=lambda item: item["window_id"]
                ),
                "failures": sorted(
                    failures, key=lambda item: item["window_id"]
                ),
            }
        )
        return 1 if failures else 0
    if args.command == "export-submap":
        print(export_geodetic_submap_poses(args.result, args.output))
        return 0
    if args.command == "overlap":
        output = publish_geodetic_overlap_report(
            args.result,
            args.output,
            policy=_assembly_config(args).overlap_policy,
        )
        report = json.loads((output / "overlap.json").read_text(encoding="utf-8"))
        report["artifact"] = str(output)
        _json_print(report)
        return 0 if report["passed"] else 2
    if args.command == "assemble":
        output = publish_geodetic_full_pose_artifact(
            args.plan,
            args.submaps_root,
            args.pose_root,
            args.pose_name,
        )
        _json_print(audited_geodetic_full_pose_artifact(output))
        return 0
    plan = audited_geodetic_assembly_plan(args.plan)
    root = Path(args.submaps_root).expanduser().resolve()
    rows = []
    for window in plan["windows"]:
        result_path = root / window["window_id"] / "result"
        state = "pending"
        if result_path.exists():
            state = "published"
            if args.verify:
                result = audited_geodetic_submap_result(result_path)
                state = "passed" if result["passed"] else "failed"
        elif (root / window["window_id"] / "plan").exists():
            state = "prepared"
        rows.append({"window_id": window["window_id"], "state": state})
    report = {
        "plan": plan["artifact"],
        "window_count": len(rows),
        "states": rows,
    }
    if args.pose:
        report["pose"] = audited_geodetic_full_pose_artifact(args.pose)
    if args.overlap_report:
        report["overlap"] = audited_geodetic_overlap_report(
            args.overlap_report
        )
    if args.submap_export:
        report["submap_export"] = audited_geodetic_submap_pose_export(
            args.submap_export
        )
    _json_print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
