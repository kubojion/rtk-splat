#!/usr/bin/env python3
"""Experimental sealed raw-bag calibration orchestration.

All dataset locations and the predeclared calibration window are explicit CLI
inputs.  The reusable diagnostics package contains no field-specific path or
frame selection.
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from rtk_splat.diagnostics.raw_calibration import (
    audited_raw_calibration_plan,
    audited_raw_calibration_result,
    audited_calibrated_segment,
    prepare_raw_calibration_plan,
    publish_calibrated_segment,
    run_raw_calibration_plan,
)


def _print(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--frontend", required=True)
    plan.add_argument("--backend", required=True)
    plan.add_argument("--segment", required=True)
    plan.add_argument("--navigation-bag", required=True)
    plan.add_argument("--camera-bag", required=True)
    plan.add_argument("--ublox-msgs-dir", required=True)
    plan.add_argument("--calibration-config", required=True)
    plan.add_argument("--evaluation-roles", required=True)
    plan.add_argument("--window-start-s", required=True, type=float)
    plan.add_argument("--window-stop-s", required=True, type=float)
    plan.add_argument("--sim3-scale-min", type=float, default=0.98)
    plan.add_argument("--sim3-scale-max", type=float, default=1.02)
    plan.add_argument("--output", required=True)

    run = commands.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--result", required=True)

    audit_plan = commands.add_parser("audit-plan")
    audit_plan.add_argument("--plan", required=True)
    audit_plan.add_argument("--skip-runtime-inputs", action="store_true")

    audit_result = commands.add_parser("audit-result")
    audit_result.add_argument("--result", required=True)

    segment = commands.add_parser("publish-segment")
    segment.add_argument("--source-segment", required=True)
    segment.add_argument("--result", required=True)
    segment.add_argument("--output", required=True)

    audit_segment = commands.add_parser("audit-segment")
    audit_segment.add_argument("--segment", required=True)
    audit_segment.add_argument("--source-segment")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        path = prepare_raw_calibration_plan(
            args.frontend,
            args.backend,
            args.segment,
            args.navigation_bag,
            args.camera_bag,
            args.ublox_msgs_dir,
            args.calibration_config,
            args.evaluation_roles,
            args.output,
            window_start_s=args.window_start_s,
            window_stop_s=args.window_stop_s,
            sim3_scale_range=(args.sim3_scale_min, args.sim3_scale_max),
        )
        print(path, flush=True)
        return 0
    if args.command == "run":
        path, result = run_raw_calibration_plan(args.plan, args.result)
        _print(
            {
                "artifact": str(path),
                "passed": True,
                "calibration_accepted": result[
                    "final_retained_prior_safe"
                ]["calibration_accepted"],
            }
        )
        return 0
    if args.command == "audit-plan":
        result = audited_raw_calibration_plan(
            args.plan, verify_runtime_inputs=not args.skip_runtime_inputs
        )
        result.pop("_runtime_context", None)
        _print(result)
        return 0
    if args.command == "audit-result":
        _print(audited_raw_calibration_result(args.result))
        return 0
    if args.command == "publish-segment":
        print(
            publish_calibrated_segment(
                args.source_segment, args.result, args.output
            ),
            flush=True,
        )
        return 0
    if args.command == "audit-segment":
        _print(
            audited_calibrated_segment(
                args.segment, expected_source_segment=args.source_segment
            )
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
