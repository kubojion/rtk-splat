#!/usr/bin/env python3
"""Run remaining members of a sealed TilePlan sequentially and resumably.

The launcher is intentionally outside the public package API.  Dataset paths,
pose names, run names, reused tile runs, and output locations are explicit
arguments.  Each subprocess retains the normal immutable cloud/run contracts;
the launcher only records progress and stops on the first failed command or
invalid completed run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from rtk_splat.backends.pose_evidence import pose_georeferencing_evidence
from rtk_splat.core.pose_artifacts import (
    load_pose_artifact,
    tile_cloud_path,
)
from rtk_splat.core.segment import SegmentReader
from rtk_splat.frontends.artifact import ArtifactError
from rtk_splat.workflows.cloud import verify_tile_cloud
from rtk_splat.workflows.configio import load_config
from rtk_splat.workflows.tile_scene import (
    _completed_tile_run,
    _expected_tile_binding,
    _training_identity,
    _verify_training_source_binding,
)
from rtk_splat.workflows.tile_seam import verify_tile_seam_probe
from rtk_splat.workflows.tiles import load_tile_execution, verify_tile_plan


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _binding(value: str) -> tuple[str, Path]:
    tile_id, separator, path = str(value).partition("=")
    if not separator or not tile_id or not path:
        raise argparse.ArgumentTypeError("expected TILE_ID=RUN_PATH")
    return tile_id, Path(path).expanduser()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--segment", required=True, type=Path)
    parser.add_argument("--expected-frames", required=True, type=int)
    parser.add_argument("--pose-name", required=True)
    parser.add_argument("--pose-artifact-root", required=True, type=Path)
    parser.add_argument("--tile-plan", required=True, type=Path)
    parser.add_argument("--passed-seam-probe", required=True, type=Path)
    parser.add_argument("--campaign-name", required=True)
    parser.add_argument("--run-name-template", required=True)
    parser.add_argument(
        "--train-iters",
        required=True,
        type=int,
        help=(
            "expected authored iteration count; verified but never injected as "
            "a provenance-changing CLI override"
        ),
    )
    parser.add_argument(
        "--reuse-tile-run", action="append", type=_binding, default=[]
    )
    parser.add_argument(
        "--run-tile",
        action="append",
        default=[],
        help="tile to launch in this order; default is every non-reused tile",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-existing-clouds",
        action="store_true",
        help="verify and reuse sealed tile clouds without re-materializing them",
    )
    parser.add_argument(
        "--allow-failed-georeferencing-for-render",
        action="store_true",
    )
    return parser


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_name(template: str, tile_id: str) -> str:
    try:
        name = str(template).format(tile_id=tile_id)
    except (KeyError, ValueError) as exc:
        raise ValueError("invalid run-name template") from exc
    if "{tile_id}" not in template:
        raise ValueError("run-name template must contain {tile_id}")
    if not _SAFE_NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError("run-name template produced an unsafe name")
    return name


def _run_path(workdir: Path, plan_name: str, tile_id: str, run_name: str) -> Path:
    return (
        workdir
        / "tile_runs"
        / plan_name
        / tile_id
        / run_name
    )


def _stage_command(
    args: argparse.Namespace,
    stage: str,
    tile_id: str,
    *,
    run_name: str | None = None,
) -> list[str]:
    if stage not in {"cloud", "train"}:
        raise ValueError(f"unsupported campaign stage: {stage}")
    command = [
        sys.executable,
        "-m",
        "rtk_splat.workflows.cli",
        stage,
        "--config",
        str(args.config),
        "--workdir",
        str(args.workdir),
        "--segment",
        str(args.segment),
        "--expected-frames",
        str(args.expected_frames),
        "--pose-name",
        str(args.pose_name),
        "--pose-artifact-root",
        str(args.pose_artifact_root),
        "--tile-plan",
        str(args.tile_plan),
        "--tile-id",
        tile_id,
    ]
    if args.allow_failed_georeferencing_for_render:
        command.append("--allow-failed-georeferencing-for-render")
    if stage == "train":
        if run_name is None:
            raise ValueError("train stage needs a run name")
        command.extend(["--run-name", run_name])
    return command


def _run_subprocess(command: Sequence[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            list(command),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"command failed with exit {result.returncode}; inspect {log}"
        )


def _verify_authored_train_iterations(cfg: Any, expected: int) -> None:
    authored = getattr(cfg.train, "iterations", None)
    if (
        isinstance(authored, bool)
        or not isinstance(authored, int)
        or authored != expected
    ):
        raise ValueError(
            "expected train iterations disagree with the authored numeric config"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.expected_frames <= 0 or args.train_iters <= 0:
        raise ValueError("frame and iteration counts must be positive")
    if not _SAFE_NAME.fullmatch(args.campaign_name) or args.campaign_name in {
        ".", ".."
    }:
        raise ValueError("campaign name is unsafe")
    args.config = args.config.expanduser().resolve()
    args.workdir = args.workdir.expanduser().resolve()
    args.segment = args.segment.expanduser().resolve()
    args.pose_artifact_root = args.pose_artifact_root.expanduser().resolve()
    args.tile_plan = args.tile_plan.expanduser().resolve()
    seam = verify_tile_seam_probe(args.passed_seam_probe)
    if seam.get("quality_passed") is not True:
        raise ArtifactError("campaign requires a sealed passing seam probe")

    cfg = load_config(args.config)
    _verify_authored_train_iterations(cfg, args.train_iters)
    cfg.paths.workdir = args.workdir
    cfg.paths.segment = args.segment
    cfg.pose.artifact = args.pose_name
    cfg.pose.artifact_root = args.pose_artifact_root
    reader = SegmentReader(args.segment).validate()
    if int(reader.meta["n_frames"]) != args.expected_frames:
        raise ArtifactError("segment frame count disagrees with campaign input")
    pose_root = args.pose_artifact_root / args.pose_name
    plan = verify_tile_plan(
        args.tile_plan,
        segment=reader.root,
        pose_root=pose_root,
        rehash_sources=True,
    )
    inventory = json.loads(
        (args.tile_plan / "source_inventory.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (args.tile_plan / "provenance.json").read_text(encoding="utf-8")
    )
    georeferencing = provenance.get("georeferencing")
    if not isinstance(georeferencing, dict):
        raise ArtifactError("TilePlan has no georeferencing evidence")
    tiles = list(plan["tiles"])
    tile_ids = [str(tile["tile_id"]) for tile in tiles]
    indices = {tile_id: index for index, tile_id in enumerate(tile_ids)}
    reuse = dict(args.reuse_tile_run)
    if len(reuse) != len(args.reuse_tile_run) or not set(reuse) <= set(tile_ids):
        raise ValueError("reused tile bindings are duplicate or unknown")
    requested = list(args.run_tile) or [
        tile_id for tile_id in tile_ids if tile_id not in reuse
    ]
    if (
        len(requested) != len(set(requested))
        or not set(requested) <= set(tile_ids)
        or set(requested) & set(reuse)
        or set(requested) | set(reuse) != set(tile_ids)
    ):
        raise ValueError("reuse and run tiles must partition the TilePlan")
    if set(seam.get("tile_ids", [])) - set(reuse):
        raise ArtifactError("passing seam tiles must be supplied as reused runs")

    campaign = args.workdir / "campaigns" / args.campaign_name
    if args.resume:
        if not campaign.is_dir() or campaign.is_symlink():
            raise RuntimeError("--resume requires an existing safe campaign")
    else:
        campaign.mkdir(parents=True, exist_ok=False)
    progress_path = campaign / "progress.json"
    progress = {
        "schema_version": 1,
        "campaign_name": args.campaign_name,
        "tile_plan_name": plan["name"],
        "tile_ids": tile_ids,
        "reused_tile_ids": list(reuse),
        "requested_tile_ids": requested,
        "expected_train_iterations": args.train_iters,
        "completed": {},
        "state": "running",
    }
    if args.resume:
        try:
            previous = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("campaign progress is missing or invalid") from exc
        for key in (
            "campaign_name", "tile_plan_name", "tile_ids",
            "reused_tile_ids", "requested_tile_ids",
            "expected_train_iterations",
        ):
            if previous.get(key) != progress[key]:
                raise RuntimeError(f"campaign resume identity changed: {key}")
        progress["completed"] = dict(previous.get("completed", {}))

    common_identity = None

    def verify_run(tile_id: str, run: Path) -> dict[str, Any]:
        nonlocal common_identity
        tile = tiles[indices[tile_id]]
        binding = _expected_tile_binding(
            plan, args.tile_plan, inventory, tile, indices[tile_id]
        )
        run_provenance, params, params_hash, artifacts = _completed_tile_run(
            run,
            binding,
            georeferencing,
            allow_nonproduction_georeferencing_for_diagnostic=(
                args.allow_failed_georeferencing_for_render
            ),
        )
        _verify_training_source_binding(run_provenance, inventory)
        identity = _training_identity(run_provenance)
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise ArtifactError("campaign tile training identities differ")
        return {
            "run": str(run.resolve()),
            "params_sha256": params_hash,
            "source_splat_file": artifacts["splat_file"],
            "source_splat_sha256": artifacts["splat_sha256"],
            "best_step": run_provenance["model_selection"]["best_step"],
            "completed_training_steps": run_provenance["model_selection"][
                "completed_training_steps"
            ],
        }

    for tile_id, run in reuse.items():
        progress["completed"][tile_id] = {
            "state": "reused",
            **verify_run(tile_id, run.expanduser().resolve()),
        }
    _atomic_json(progress_path, progress)

    viewmats, _ = load_pose_artifact(
        reader.root,
        cfg,
        allow_failed_georeferencing_for_render=(
            args.allow_failed_georeferencing_for_render
        ),
    )
    pose_evidence = pose_georeferencing_evidence(reader.root, cfg)
    for position, tile_id in enumerate(requested, start=1):
        run_name = _run_name(args.run_name_template, tile_id)
        run = _run_path(args.workdir, plan["name"], tile_id, run_name)
        execution = load_tile_execution(
            args.tile_plan, tile_id, reader, cfg, rehash_sources=True
        )
        cloud = tile_cloud_path(
            reader.root, cfg, execution.binding["name"], tile_id
        )
        progress.update({
            "state": "running",
            "current_tile_id": tile_id,
            "current_position": position,
        })
        _atomic_json(progress_path, progress)
        if run.exists():
            if not args.resume:
                raise FileExistsError(f"refusing existing tile run: {run}")
            progress["completed"][tile_id] = {
                "state": "verified-existing",
                **verify_run(tile_id, run),
            }
            _atomic_json(progress_path, progress)
            continue
        if cloud.exists():
            if not args.resume and not args.reuse_existing_clouds:
                raise FileExistsError(
                    f"existing tile cloud needs --reuse-existing-clouds: {cloud}"
                )
            verify_tile_cloud(
                cloud,
                execution,
                viewmats,
                georeferencing=pose_evidence,
                allow_failed_georeferencing_for_render=(
                    args.allow_failed_georeferencing_for_render
                ),
            )
        else:
            cloud_log = campaign / "logs" / f"{tile_id}.cloud.log"
            _run_subprocess(
                _stage_command(args, "cloud", tile_id), cloud_log
            )
            verify_tile_cloud(
                cloud,
                execution,
                viewmats,
                georeferencing=pose_evidence,
                allow_failed_georeferencing_for_render=(
                    args.allow_failed_georeferencing_for_render
                ),
            )
        train_log = campaign / "logs" / f"{tile_id}.train.log"
        _run_subprocess(
            _stage_command(
                args, "train", tile_id, run_name=run_name
            ),
            train_log,
        )
        progress["completed"][tile_id] = {
            "state": "trained",
            **verify_run(tile_id, run),
        }
        _atomic_json(progress_path, progress)
        print(
            f"completed {tile_id} ({position}/{len(requested)}) -> {run}",
            flush=True,
        )

    if set(progress["completed"]) != set(tile_ids):
        raise ArtifactError("campaign finished without exact TilePlan coverage")
    progress.pop("current_tile_id", None)
    progress.pop("current_position", None)
    progress["state"] = "complete"
    _atomic_json(progress_path, progress)
    print(json.dumps(progress, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
