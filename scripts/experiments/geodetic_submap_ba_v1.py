#!/usr/bin/env python3
"""Experimental launcher for one sealed geodetic-submap BA pilot.

This is intentionally outside the installed command-line API.  It never
chooses a dataset, frame range, model, or output location on the caller's
behalf.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from rtk_splat.backends.geodetic_pairs import GeodeticPairPolicy
from rtk_splat.backends.geodetic_submap import (
    GeodeticSubmapConfig,
    audited_geodetic_submap_result,
    create_geodetic_frame_selection,
    prepare_geodetic_submap_plan,
    run_geodetic_submap_plan,
)


def _frame_ids(path: str | Path) -> list[int]:
    values: list[int] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(
                f"invalid frame ID on line {line_number}: {text!r}"
            ) from exc
        values.append(value)
    if not values:
        raise ValueError("frame-ID file is empty")
    return values


def _config(args: argparse.Namespace) -> GeodeticSubmapConfig:
    pair = GeodeticPairPolicy(
        nearby_frame_gap=args.nearby_frame_gap,
        nearby_time_s=args.nearby_time_s,
        revisit_distance_m=args.revisit_distance_m,
        revisit_physical_cap_m=args.revisit_physical_cap_m,
        covariance_sigma_multiplier=args.covariance_sigma_multiplier,
        max_endpoint_position_std_m=args.max_endpoint_position_std_m,
        strong_verified_matches=args.strong_verified_matches,
    )
    base = GeodeticSubmapConfig()
    refinement = replace(
        base.refinement,
        num_threads=args.num_threads,
        random_seed=args.random_seed,
        minimum_free_space_gb=args.minimum_free_space_gb,
        minimum_runtime_free_space_gb=args.minimum_runtime_free_space_gb,
    )
    return GeodeticSubmapConfig(
        pair_policy=pair,
        refinement=refinement,
        position_quality_weights=base.position_quality_weights,
    )


def _pilot_frame_ids(args: argparse.Namespace) -> list[int] | None:
    if args.frame_ids is not None:
        if args.frame_start is not None or args.frame_end is not None:
            raise ValueError("use --frame-ids or --frame-start/--frame-end, not both")
        return _frame_ids(args.frame_ids)
    if (args.frame_start is None) != (args.frame_end is None):
        raise ValueError("--frame-start and --frame-end must be supplied together")
    if args.frame_start is None:
        return None
    if args.frame_start > args.frame_end:
        raise ValueError("--frame-start cannot exceed --frame-end")
    return list(range(args.frame_start, args.frame_end + 1))


def _add_plan_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frontend", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--nearby-frame-gap", type=int, default=2)
    parser.add_argument("--nearby-time-s", type=float, default=2.0)
    parser.add_argument("--revisit-distance-m", type=float, default=1.0)
    parser.add_argument("--revisit-physical-cap-m", type=float, default=1.5)
    parser.add_argument("--covariance-sigma-multiplier", type=float, default=3.0)
    parser.add_argument("--max-endpoint-position-std-m", type=float, default=0.5)
    parser.add_argument("--strong-verified-matches", type=int, default=30)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--minimum-free-space-gb", type=float, default=10.0)
    parser.add_argument(
        "--minimum-runtime-free-space-gb", type=float, default=5.0
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal-selection")
    seal.add_argument("--frontend", required=True)
    seal.add_argument("--segment", required=True)
    seal.add_argument("--frame-ids", required=True)
    seal.add_argument("--output", required=True)

    plan = commands.add_parser("plan")
    _add_plan_inputs(plan)

    run = commands.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--result", required=True)
    run.add_argument("--colmap", required=True)

    pilot = commands.add_parser("pilot")
    _add_plan_inputs(pilot)
    pilot.add_argument("--frame-ids")
    pilot.add_argument("--frame-start", type=int)
    pilot.add_argument("--frame-end", type=int)
    pilot.add_argument("--result", required=True)
    pilot.add_argument("--colmap", required=True)

    status = commands.add_parser("status")
    status.add_argument("--result", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "seal-selection":
        output = create_geodetic_frame_selection(
            args.frontend,
            args.segment,
            _frame_ids(args.frame_ids),
            args.output,
        )
        print(output)
        return 0
    if args.command == "plan":
        output = prepare_geodetic_submap_plan(
            args.frontend,
            args.backend,
            args.segment,
            args.selection,
            args.plan,
            config=_config(args),
        )
        print(output)
        return 0
    if args.command == "run":
        result = run_geodetic_submap_plan(
            args.plan, args.result, args.colmap
        )
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result["passed"] else 2
    if args.command == "pilot":
        frame_ids = _pilot_frame_ids(args)
        if frame_ids is not None:
            create_geodetic_frame_selection(
                args.frontend,
                args.segment,
                frame_ids,
                args.selection,
            )
        plan = prepare_geodetic_submap_plan(
            args.frontend,
            args.backend,
            args.segment,
            args.selection,
            args.plan,
            config=_config(args),
        )
        result = run_geodetic_submap_plan(plan, args.result, args.colmap)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result["passed"] else 2
    result = audited_geodetic_submap_result(args.result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
