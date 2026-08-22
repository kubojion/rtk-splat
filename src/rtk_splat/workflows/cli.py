"""Command-line orchestration for immutable contract-v2 artifacts.

The CLI deliberately exposes one command per artifact stage.  It never
recreates a legacy segment in place and never hides a long COLMAP or Gaussian
Splatting run behind an ``all`` command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from rtk_splat.workflows.configio import load_config
from rtk_splat.workflows.config_ledger import (
    authored_config_plain,
    commit_config_ledger,
    prepare_config_ledger,
)
from rtk_splat.core.segment import SegmentReader


def _plain(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {key: _plain(item) for key, item in sorted(vars(value).items())}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _section(cfg, name: str) -> SimpleNamespace:
    value = getattr(cfg, name, None)
    return value if isinstance(value, SimpleNamespace) else SimpleNamespace()


def _reject_deprecated_mapper_section(cfg) -> None:
    if hasattr(cfg, "global_mapper"):
        raise ValueError(
            "deprecated 'global_mapper:' configuration is not supported; "
            "move the current MapperConfig options under canonical 'mapper:' "
            "and prepare a new backend artifact"
        )


def _dataclass_options(section: Any, cls, *, base: dict[str, Any] | None = None):
    values = {} if base is None else dict(base)
    allowed = {item.name for item in fields(cls)}
    if isinstance(section, SimpleNamespace):
        supplied = vars(section)
    elif isinstance(section, dict):
        supplied = section
    else:
        supplied = {}
    unknown = sorted(set(supplied) - allowed)
    if unknown:
        raise ValueError(
            f"unknown {cls.__name__} option(s): {', '.join(unknown)}"
        )
    values.update(supplied)
    if "acceptable_rtk_status" in values and values["acceptable_rtk_status"] is not None:
        values["acceptable_rtk_status"] = tuple(values["acceptable_rtk_status"])
    return cls(**values)


def _segment_path(cfg) -> Path:
    configured = getattr(cfg.paths, "segment", None)
    return (
        Path(configured).expanduser()
        if configured is not None
        else Path(cfg.paths.workdir).expanduser() / "segment"
    )


def _frontend_name(cfg, args) -> str:
    name = args.frontend_name or getattr(_section(cfg, "frontend"), "name", None)
    if name:
        return str(name)
    preset = args.keyframe_preset or getattr(
        _section(_section(cfg, "frontend"), "keyframes"), "preset", "balanced"
    )
    profile = args.feature_profile or getattr(
        _section(_section(cfg, "frontend"), "features"), "profile", "gpu"
    )
    return f"{preset}-{profile}"


def _frontend_path(cfg, args) -> Path:
    return Path(cfg.paths.workdir) / "frontend_artifacts" / _frontend_name(cfg, args)


def _backend_kind(cfg, args) -> str:
    _reject_deprecated_mapper_section(cfg)
    return str(args.backend or getattr(_section(cfg, "mapper"), "backend", "global"))


def _backend_name(cfg, args) -> str:
    _reject_deprecated_mapper_section(cfg)
    return str(
        args.backend_name
        or getattr(_section(cfg, "mapper"), "name", None)
        or f"{_frontend_name(cfg, args)}-{_backend_kind(cfg, args)}"
    )


def _backend_path(cfg, args) -> Path:
    return Path(cfg.paths.workdir) / "backend_artifacts" / _backend_name(cfg, args)


def _refinement_name(cfg, args) -> str:
    configured = getattr(_section(cfg, "rtk_refinement"), "name", None)
    return str(
        args.refinement_name
        or configured
        or f"{_backend_name(cfg, args)}-rtk-refined"
    )


def _refinement_path(cfg, args) -> Path:
    return (
        Path(cfg.paths.workdir)
        / "refinement_artifacts"
        / _refinement_name(cfg, args)
    )


def _pose_name(cfg, args) -> str:
    return str(
        args.pose_name
        or getattr(_section(cfg, "mapper"), "pose_artifact_name", None)
        or _backend_name(cfg, args)
    )


def _colmap(cfg) -> str:
    return str(
        os.environ.get("COLMAP_BIN")
        or getattr(_section(cfg, "colmap"), "executable", "colmap")
    )


def _validate_expected(reader: SegmentReader, expected: int | None) -> None:
    if expected is not None and reader.meta["n_frames"] != expected:
        raise RuntimeError(
            f"frame-count safety gate failed: expected {expected}, "
            f"found {reader.meta['n_frames']}"
        )


def _reader(cfg, args) -> SegmentReader:
    reader = SegmentReader(_segment_path(cfg)).validate()
    _validate_expected(reader, args.expected_frames)
    return reader


def cmd_validate(cfg, args) -> None:
    reader = _reader(cfg, args)
    print(
        json.dumps(
            {
                "segment": str(reader.root.resolve()),
                "contract_version": reader.meta["contract_version"],
                "n_frames": reader.meta["n_frames"],
                "capabilities": reader.meta["capabilities"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def cmd_ingest(cfg, args) -> None:
    from rtk_splat.adapters.registry import publish_from_config

    window = getattr(_section(cfg, "segment"), "window_s", None)
    if window is not None:
        if len(window) != 2 or float(window[1]) <= float(window[0]):
            raise ValueError("segment.window_s must be [start_s, end_s]")
    destination = _segment_path(cfg)
    reader = publish_from_config(cfg, destination)
    _validate_expected(reader, args.expected_frames)
    print(f"published immutable contract-v2 segment -> {reader.root}")


def cmd_depth(cfg, args) -> None:
    from rtk_splat.workflows.depth import derive_sgbm_depth

    source = _reader(cfg, args)
    destination = (
        args.derived_segment.expanduser()
        if args.derived_segment is not None
        else Path(cfg.paths.workdir) / "segments" / f"{source.root.name}-sgbm"
    )
    result = derive_sgbm_depth(source.root, destination, cfg)
    print(f"published derived depth segment -> {result.root}")


def cmd_segment_materialize(cfg, args) -> None:
    from rtk_splat.workflows.segment_transfer import materialize_portable_segment

    source = _reader(cfg, args)
    destination = args.portable_segment.expanduser()
    result = materialize_portable_segment(
        source.root, destination, link_mode=(args.link_mode or "auto")
    )
    print(f"published portable transfer segment -> {result}")


def cmd_segment_transfer_verify(cfg, args) -> None:
    from rtk_splat.workflows.segment_transfer import verify_portable_segment

    source = _reader(cfg, args)
    result = verify_portable_segment(source.root)
    print(json.dumps(result, indent=2, sort_keys=True))


def _frontend_configs(cfg, args):
    from rtk_splat.frontends.keyframes import KEYFRAME_PRESETS, KeyframeConfig
    from rtk_splat.frontends.pair_graph import PairGraphConfig

    frontend = _section(cfg, "frontend")
    keyframes = _section(frontend, "keyframes")
    preset = str(
        args.keyframe_preset or getattr(keyframes, "preset", "balanced")
    )
    if preset not in KEYFRAME_PRESETS:
        raise ValueError(
            f"unknown keyframe preset {preset!r}; choose "
            + ", ".join(sorted(KEYFRAME_PRESETS))
        )
    overrides = {
        key: value for key, value in vars(keyframes).items() if key != "preset"
    }
    keyframe_config = _dataclass_options(
        overrides, KeyframeConfig, base=asdict(KEYFRAME_PRESETS[preset])
    )
    pair_config = _dataclass_options(
        _section(frontend, "pairs"), PairGraphConfig
    )
    return preset, keyframe_config, pair_config


def cmd_frontend_build(cfg, args) -> None:
    from rtk_splat.core.runtime_resolution import configuration_evidence
    from rtk_splat.frontends.artifact import (
        FrontendArtifactBuilder,
        collect_provenance,
    )
    from rtk_splat.frontends.planning import plan_frontend

    reader = _reader(cfg, args)
    preset, keyframe_config, pair_config = _frontend_configs(cfg, args)
    frontend = _section(cfg, "frontend")
    seed = int(getattr(frontend, "seed", 7))
    plan = plan_frontend(
        reader.root,
        keyframe_config=keyframe_config,
        pair_config=pair_config,
        compute_image_quality=not args.skip_image_quality,
        turn_rate_deg_s=float(getattr(frontend, "turn_rate_deg_s", 0.75)),
    )
    resolved = _plain(cfg)
    resolved["experiment_overrides"] = {
        "frontend_name": _frontend_name(cfg, args),
        "keyframe_preset": preset,
        "skip_image_quality": args.skip_image_quality,
    }
    provenance = collect_provenance(
        reader.root,
        resolved_config=resolved,
        configuration=configuration_evidence(cfg),
        colmap=_colmap(cfg),
        seed=seed,
    )
    artifact = FrontendArtifactBuilder(
        reader.root, cfg.paths.workdir, _frontend_name(cfg, args)
    ).build(
        rig_config=plan.rig_config,
        keyframes=plan.keyframes,
        pairs=plan.pairs,
        provenance=provenance,
        quality=plan.quality,
    )
    print(
        f"frontend foundation -> {artifact}\n"
        f"keyframes {plan.quality['n_keyframes']}/{plan.quality['n_frames']}; "
        f"pairs {plan.quality['n_pairs']}; all frames retained"
    )


def cmd_frontend_features(cfg, args) -> None:
    from rtk_splat.frontends.colmap import run_feature_extraction

    features = _section(_section(cfg, "frontend"), "features")
    profile = str(args.feature_profile or getattr(features, "profile", "gpu"))
    report = run_feature_extraction(
        _frontend_path(cfg, args),
        _colmap(cfg),
        profile=profile,
        max_image_size=int(getattr(features, "max_image_size", -1)),
        num_threads=int(getattr(features, "num_threads", 8)),
        max_num_features=int(getattr(features, "max_num_features", 8192)),
        gpu_index=int(getattr(features, "gpu_index", -1)),
        seed=int(getattr(_section(cfg, "frontend"), "seed", 7)),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_frontend_rig(cfg, args) -> None:
    from rtk_splat.frontends.colmap import run_rig_configurator

    report = run_rig_configurator(_frontend_path(cfg, args), _colmap(cfg))
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_frontend_priors(cfg, args) -> None:
    from rtk_splat.frontends.colmap import insert_pose_priors

    priors = _section(_section(cfg, "frontend"), "pose_priors")
    maximum = getattr(priors, "max_covariance_m2", 0.04)
    maximum = None if maximum is None else float(maximum)
    max_age = getattr(priors, "max_source_residual_s", 0.15)
    max_age = None if max_age is None else float(max_age)
    report = insert_pose_priors(
        _frontend_path(cfg, args),
        covariance_floor_m=float(getattr(priors, "covariance_floor_m", 0.03)),
        max_covariance_m2=maximum,
        min_fix_status=int(getattr(priors, "min_fix_status", 0)),
        min_carrier_status=int(getattr(priors, "min_carrier_status", -1)),
        max_source_residual_s=max_age,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_frontend_match(cfg, args) -> None:
    from rtk_splat.frontends.colmap import run_matches_importer

    matching = _section(_section(cfg, "frontend"), "matching")
    report = run_matches_importer(
        _frontend_path(cfg, args),
        _colmap(cfg),
        use_gpu=bool(getattr(matching, "use_gpu", True)),
        gpu_index=int(getattr(matching, "gpu_index", -1)),
        num_threads=int(getattr(matching, "num_threads", 8)),
        seed=int(getattr(_section(cfg, "frontend"), "seed", 7)),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _mapper_config(cfg, args):
    from rtk_splat.backends.mapper import MapperConfig

    _reject_deprecated_mapper_section(cfg)
    mapper_values = dict(vars(_section(cfg, "mapper")))
    mapper_values.pop("name", None)
    mapper_values.pop("pose_artifact_name", None)
    if args.backend is not None:
        mapper_values["backend"] = args.backend
    return _dataclass_options(mapper_values, MapperConfig)


def _rtk_refinement_config(cfg, args):
    from rtk_splat.backends.rtk_refinement import RtkRefinementConfig

    values = dict(vars(_section(cfg, "rtk_refinement")))
    values.pop("name", None)
    if args.initialization_mode is not None:
        values["initialization_mode"] = args.initialization_mode
    if args.prior_position_loss is not None:
        values["prior_position_loss"] = args.prior_position_loss
    return _dataclass_options(values, RtkRefinementConfig)


def cmd_backend_prepare(cfg, args) -> None:
    from rtk_splat.backends.mapper import prepare_mapper_backend

    workspace = prepare_mapper_backend(
        _frontend_path(cfg, args),
        _backend_path(cfg, args),
        config=_mapper_config(cfg, args),
    )
    print(f"isolated verified mapper workspace -> {workspace}")


def cmd_backend_solve(cfg, args) -> None:
    from rtk_splat.backends.mapper import run_mapper_solve

    report = run_mapper_solve(_backend_path(cfg, args), _colmap(cfg))
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_backend_register(cfg, args) -> None:
    from rtk_splat.backends.mapper import run_image_registration

    report = run_image_registration(_backend_path(cfg, args), _colmap(cfg))
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_backend_quality(cfg, args) -> None:
    from rtk_splat.backends.mapper import run_quality_summary

    report = run_quality_summary(_backend_path(cfg, args), _colmap(cfg))
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_backend_refine_rtk(cfg, args) -> None:
    from rtk_splat.backends.rtk_refinement import run_rtk_refinement

    report = run_rtk_refinement(
        _backend_path(cfg, args),
        _refinement_path(cfg, args),
        _colmap(cfg),
        config=_rtk_refinement_config(cfg, args),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_backend_export(cfg, args) -> None:
    from rtk_splat.backends.mapper import export_pose_artifact

    output = export_pose_artifact(
        _backend_path(cfg, args),
        _pose_name(cfg, args),
        output_root=Path(cfg.paths.workdir) / "pose_artifacts",
        refinement_workspace=(
            _refinement_path(cfg, args)
            if args.refinement_name is not None
            else None
        ),
        allow_failed_georeferencing_for_render=(
            bool(getattr(args, "allow_failed_georeferencing_for_render", False))
        ),
    )
    print(f"fixed-scale ENU pose artifact -> {output}")


def cmd_cloud(cfg, args) -> None:
    from rtk_splat.workflows.cloud import construct_initial_cloud
    from rtk_splat.workflows.tiles import load_tile_execution

    reader = _reader(cfg, args)
    if args.pose_name:
        cfg.pose.artifact = args.pose_name
    if getattr(args, "pose_artifact_root", None) is not None:
        cfg.pose.artifact_root = args.pose_artifact_root.expanduser()
    execution = None
    if getattr(args, "tile_plan", None) is not None:
        execution = load_tile_execution(
            args.tile_plan, args.tile_id, reader, cfg
        )
    output, point_count = construct_initial_cloud(
        reader,
        cfg,
        allow_failed_georeferencing_for_render=(
            bool(getattr(args, "allow_failed_georeferencing_for_render", False))
        ),
        tile_execution=execution,
    )
    print(f"initial cloud: {point_count:,} points -> {output}")


def cmd_tiles_plan(cfg, args) -> None:
    from rtk_splat.workflows.tiles import build_tile_plan

    reader = _reader(cfg, args)
    if args.pose_name:
        cfg.pose.artifact = args.pose_name
    if getattr(args, "pose_artifact_root", None) is not None:
        cfg.pose.artifact_root = args.pose_artifact_root.expanduser()
    tiles = _section(cfg, "tiles")
    if getattr(args, "tile_max_tiles", None) is not None:
        tiles.max_tiles = int(args.tile_max_tiles)
    if getattr(args, "tile_min_visibility_fraction", None) is not None:
        tiles.min_visibility_fraction = float(
            args.tile_min_visibility_fraction
        )
    name = str(
        args.tile_plan_name
        or getattr(tiles, "name", None)
        or "visibility-workload-v1"
    )
    output = build_tile_plan(
        reader,
        cfg,
        name=name,
        output_root=cfg.paths.workdir,
        tile_count=args.tile_count,
        allow_failed_georeferencing_for_render=bool(
            getattr(args, "allow_failed_georeferencing_for_render", False)
        ),
    )
    plan = json.loads((output / "tile_plan.json").read_text())
    print(
        json.dumps(
            {
                "tile_plan": str(output),
                "plot": str(output / "plan.svg"),
                **plan["summary"],
                "tiles": [
                    {"tile_id": item["tile_id"], **item["summary"]}
                    for item in plan["tiles"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


def cmd_train(cfg, args) -> None:
    if args.pose_name:
        cfg.pose.artifact = args.pose_name
    if args.run_name:
        cfg.train.run_name = args.run_name
    from rtk_splat.backends.gsplat import train_tile
    from rtk_splat.core.pose_artifacts import cloud_path, tile_cloud_path
    from rtk_splat.workflows.cloud import load_tile_context_masks, verify_tile_cloud
    from rtk_splat.workflows.runtime_config import resolve_training_controls
    from rtk_splat.workflows.tiles import load_tile_execution

    reader = _reader(cfg, args)
    if getattr(args, "pose_artifact_root", None) is not None:
        cfg.pose.artifact_root = args.pose_artifact_root.expanduser()
    execution = None
    initial_cloud = cloud_path(reader.root, cfg)
    if getattr(args, "tile_plan", None) is not None:
        execution = load_tile_execution(
            args.tile_plan, args.tile_id, reader, cfg
        )
        initial_cloud = tile_cloud_path(
            reader.root,
            cfg,
            execution.binding["name"],
            execution.tile_id,
        )
        from rtk_splat.backends.pose_evidence import pose_georeferencing_evidence
        from rtk_splat.core.pose_artifacts import load_pose_artifact
        viewmats, _ = load_pose_artifact(
            reader.root,
            cfg,
            allow_failed_georeferencing_for_render=bool(
                getattr(args, "allow_failed_georeferencing_for_render", False)
            ),
        )
        verify_tile_cloud(
            initial_cloud,
            execution,
            viewmats,
            georeferencing=pose_georeferencing_evidence(reader.root, cfg),
            allow_failed_georeferencing_for_render=bool(
                getattr(args, "allow_failed_georeferencing_for_render", False)
            ),
        )
    resolve_training_controls(
        cfg,
        reader,
        initial_cloud,
        training_frame_ids=(execution.train_ids if execution else None),
    )
    run = (
        Path(cfg.paths.workdir) / "runs" / str(cfg.train.run_name)
        if execution is None
        else Path(cfg.paths.workdir)
        / "tile_runs"
        / str(execution.binding["name"])
        / execution.tile_id
        / str(cfg.train.run_name)
    )
    context_masks = (
        load_tile_context_masks(initial_cloud, execution)
        if execution is not None
        else None
    )
    result = train_tile(
        reader.root,
        run,
        cfg,
        allow_failed_georeferencing_for_render=(
            bool(getattr(args, "allow_failed_georeferencing_for_render", False))
        ),
        initial_cloud=initial_cloud,
        train_frame_ids=(execution.train_ids if execution else None),
        eval_frame_ids=(execution.val_ids if execution else None),
        tile_plan=(execution.binding if execution else None),
        context_masks=context_masks,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _scene_tile_runs(values: list[str] | None) -> dict[str, Path]:
    result = {}
    for value in values or []:
        tile_id, separator, path = str(value).partition("=")
        if not separator or not tile_id or not path:
            raise ValueError("--scene-tile-run must be TILE_ID=/path/to/run")
        if tile_id in result:
            raise ValueError(f"duplicate --scene-tile-run for {tile_id}")
        result[tile_id] = Path(path).expanduser()
    return result


def cmd_scene_publish(cfg, args) -> None:
    if args.pose_name:
        cfg.pose.artifact = args.pose_name
    cfg.pose.artifact_root = args.pose_artifact_root.expanduser()
    common = {
        "segment": _segment_path(cfg),
        "cfg": cfg,
        "tile_plan_root": args.tile_plan,
        "pose_root": (
            args.pose_artifact_root.expanduser() / str(cfg.pose.artifact)
        ),
        "tile_runs": _scene_tile_runs(args.scene_tile_run),
        "output_root": cfg.paths.workdir,
        "scene_name": args.scene_name,
        "opacity_threshold": (
            0.05 if args.scene_opacity_threshold is None
            else args.scene_opacity_threshold
        ),
        "maximum_combined_gaussians": args.max_combined_gaussians,
        "device": ("cuda" if args.scene_device is None else args.scene_device),
    }
    if (args.scene_mode or "controlled-ab") == "production":
        from rtk_splat.workflows.tile_scene import publish_production_tiled_scene

        result = publish_production_tiled_scene(
            **common,
            allow_nonproduction_georeferencing_for_diagnostic=(
                bool(args.diagnostic_scene)
            ),
        )
    else:
        from rtk_splat.workflows.tile_scene import publish_tiled_scene

        result = publish_tiled_scene(
            **common,
            reference_run=args.reference_run,
            reference_hashes={
                "params_sha256": args.reference_params_sha256,
                "metrics_sha256": args.reference_metrics_sha256,
                "provenance_sha256": args.reference_provenance_sha256,
            },
            maximum_psnr_loss_db=(
                0.30 if args.max_psnr_loss_db is None else args.max_psnr_loss_db
            ),
            maximum_ssim_loss=(
                0.015 if args.max_ssim_loss is None else args.max_ssim_loss
            ),
            maximum_lpips_cc_increase=(
                0.030
                if args.max_lpips_cc_increase is None
                else args.max_lpips_cc_increase
            ),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_seam_probe(cfg, args) -> None:
    if args.pose_name:
        cfg.pose.artifact = args.pose_name
    cfg.pose.artifact_root = args.pose_artifact_root.expanduser()
    from rtk_splat.workflows.tile_seam import publish_tile_seam_probe

    result = publish_tile_seam_probe(
        segment=_segment_path(cfg),
        cfg=cfg,
        tile_plan_root=args.tile_plan,
        pose_root=(
            args.pose_artifact_root.expanduser() / str(cfg.pose.artifact)
        ),
        tile_runs=_scene_tile_runs(args.scene_tile_run),
        anchor_tile_id=args.seam_anchor_tile_id,
        output_root=cfg.paths.workdir,
        probe_name=args.scene_name,
        maximum_psnr_loss_db=(
            0.30 if args.max_psnr_loss_db is None else args.max_psnr_loss_db
        ),
        maximum_ssim_loss=(
            0.015 if args.max_ssim_loss is None else args.max_ssim_loss
        ),
        maximum_lpips_cc_increase=(
            0.030
            if args.max_lpips_cc_increase is None
            else args.max_lpips_cc_increase
        ),
        maximum_visible_gaussians=args.max_combined_gaussians,
        device=("cuda" if args.scene_device is None else args.scene_device),
        allow_nonproduction_georeferencing_for_diagnostic=bool(
            args.diagnostic_scene
        ),
        assembly_policy=(
            args.seam_assembly_policy or "hard_half_open_core_v1"
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


COMMANDS = {
    "validate": cmd_validate,
    "ingest": cmd_ingest,
    "depth": cmd_depth,
    "segment-materialize": cmd_segment_materialize,
    "segment-transfer-verify": cmd_segment_transfer_verify,
    "frontend-build": cmd_frontend_build,
    "frontend-features": cmd_frontend_features,
    "frontend-rig": cmd_frontend_rig,
    "frontend-priors": cmd_frontend_priors,
    "frontend-match": cmd_frontend_match,
    "backend-prepare": cmd_backend_prepare,
    "backend-solve": cmd_backend_solve,
    "backend-register": cmd_backend_register,
    "backend-quality": cmd_backend_quality,
    "backend-refine-rtk": cmd_backend_refine_rtk,
    "backend-export": cmd_backend_export,
    "tiles-plan": cmd_tiles_plan,
    "cloud": cmd_cloud,
    "train": cmd_train,
    "seam-probe": cmd_seam_probe,
    "scene-publish": cmd_scene_publish,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtk-splat")
    parser.add_argument("stage", choices=tuple(COMMANDS))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--config-root",
        type=Path,
        help="external directory containing profiles/ and robots/",
    )
    parser.add_argument("--workdir", type=Path, help="new experiment root override")
    parser.add_argument("--segment", type=Path, help="immutable v2 segment override")
    parser.add_argument(
        "--derived-segment",
        type=Path,
        help="new contract-v2 destination for the depth stage",
    )
    parser.add_argument(
        "--portable-segment",
        type=Path,
        help="new self-contained, checksum-sealed segment destination",
    )
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "copy"),
        help=(
            "portable materialization policy; auto hardlinks on one filesystem "
            "and copies across filesystems"
        ),
    )
    parser.add_argument("--frontend-name")
    parser.add_argument(
        "--feature-profile", choices=("gpu", "cpu_reference")
    )
    parser.add_argument(
        "--keyframe-preset", choices=("all", "dense", "balanced", "sparse")
    )
    parser.add_argument("--backend", choices=("global", "incremental"))
    parser.add_argument("--backend-name")
    parser.add_argument("--refinement-name")
    parser.add_argument(
        "--initialization-mode",
        choices=("continuation", "fresh"),
        help=(
            "pose-prior mapper initialization for backend-refine-rtk only; "
            "fresh omits the existing visual model"
        ),
    )
    parser.add_argument(
        "--prior-position-loss",
        choices=("cauchy", "trivial"),
        help="RTK position-prior loss for backend-refine-rtk only",
    )
    parser.add_argument("--pose-name")
    parser.add_argument(
        "--pose-artifact-root",
        type=Path,
        help=(
            "pose-artifact parent override for tiles-plan, cloud, train, "
            "seam-probe, or scene-publish"
        ),
    )
    parser.add_argument("--tile-plan-name")
    parser.add_argument(
        "--tile-plan",
        type=Path,
        help=(
            "sealed TilePlan artifact to consume during cloud/train/seam-probe/"
            "scene-publish"
        ),
    )
    parser.add_argument("--tile-id")
    parser.add_argument(
        "--tile-count",
        type=int,
        help="explicit tile count for a controlled tiles-plan experiment",
    )
    parser.add_argument(
        "--tile-max-tiles",
        type=int,
        help=(
            "explicit audited ceiling for automatic tiles-plan partitioning; "
            "does not change per-tile training capacity"
        ),
    )
    parser.add_argument(
        "--tile-min-visibility-fraction",
        type=float,
        help=(
            "explicit audited minimum fraction of a frame's sampled depth "
            "support required for automatic tile visibility"
        ),
    )
    parser.add_argument("--run-name")
    parser.add_argument(
        "--scene-tile-run",
        action="append",
        help="completed tiled run as TILE_ID=/path; repeat for every tile",
    )
    parser.add_argument("--scene-name")
    parser.add_argument(
        "--seam-anchor-tile-id",
        help=(
            "seam-probe anchor; its neighbor is selected only from sealed "
            "TilePlan core geometry"
        ),
    )
    parser.add_argument(
        "--seam-assembly-policy",
        choices=(
            "hard_half_open_core_v1",
            "normalized_core_distance_feather_v1",
            "depth_projected_context_composite_v1",
        ),
        help=(
            "seam-probe assembly; feather width is derived from the configured "
            "maximum Gaussian scale and is not a tunable CLI value"
        ),
    )
    parser.add_argument(
        "--scene-mode",
        choices=("controlled-ab", "production"),
        help=(
            "controlled-ab requires a sealed monolithic reference; production "
            "publishes absolute whole-scene/seam evidence without one"
        ),
    )
    parser.add_argument(
        "--diagnostic-scene",
        action="store_true",
        help=(
            "with --scene-mode production, explicitly allow a provisional "
            "diagnostic-only scene from non-production georeferencing"
        ),
    )
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--reference-params-sha256")
    parser.add_argument("--reference-metrics-sha256")
    parser.add_argument("--reference-provenance-sha256")
    parser.add_argument("--scene-opacity-threshold", type=float)
    parser.add_argument("--max-psnr-loss-db", type=float)
    parser.add_argument("--max-ssim-loss", type=float)
    parser.add_argument("--max-lpips-cc-increase", type=float)
    parser.add_argument("--max-combined-gaussians", type=int)
    parser.add_argument("--scene-device")
    parser.add_argument("--train-iters", type=int)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument(
        "--allow-failed-georeferencing-for-render",
        action="store_true",
        help=(
            "explicitly consume or publish a diagnostic-render-only pose "
            "whose held-out georeferencing gate failed; valid only for "
            "backend-export, tiles-plan, cloud, and train"
        ),
    )
    parser.add_argument(
        "--skip-image-quality",
        action="store_true",
        help="skip image decoding only for planning smoke tests",
    )
    return parser


def _stage_overrides(args) -> dict[str, Any]:
    """Return only explicit command-line controls, separate from YAML."""
    names = (
        "workdir", "segment", "derived_segment", "portable_segment",
        "link_mode", "frontend_name",
        "feature_profile", "keyframe_preset", "backend", "backend_name",
        "refinement_name", "initialization_mode", "prior_position_loss",
        "pose_name", "run_name", "train_iters", "expected_frames",
        "pose_artifact_root", "tile_plan_name", "tile_count", "tile_max_tiles",
        "tile_min_visibility_fraction",
        "tile_plan", "tile_id",
        "scene_tile_run", "scene_name", "seam_anchor_tile_id",
        "seam_assembly_policy", "scene_mode",
        "reference_run",
        "reference_params_sha256", "reference_metrics_sha256",
        "reference_provenance_sha256", "scene_opacity_threshold",
        "max_psnr_loss_db", "max_ssim_loss", "max_lpips_cc_increase",
        "max_combined_gaussians", "scene_device",
    )
    result = {
        name: _plain(getattr(args, name, None))
        for name in names
        if getattr(args, name, None) is not None
    }
    if args.skip_image_quality:
        result["skip_image_quality"] = True
    if args.diagnostic_scene:
        result["diagnostic_scene"] = True
    if args.allow_failed_georeferencing_for_render:
        result["allow_failed_georeferencing_for_render"] = True
    return result


def _validate_diagnostic_render_names(args, cfg) -> None:
    """Keep visualization-only outputs outside configured production names."""
    if not args.allow_failed_georeferencing_for_render:
        return
    if args.pose_name is None:
        raise ValueError(
            "--allow-failed-georeferencing-for-render requires an explicit "
            "--pose-name"
        )
    if args.stage == "backend-export":
        configured_pose = str(
            getattr(_section(cfg, "mapper"), "pose_artifact_name", None)
            or _backend_name(cfg, args)
        )
    else:
        configured_pose = str(getattr(_section(cfg, "pose"), "artifact", "rtk"))
    if args.pose_name == configured_pose:
        raise ValueError(
            "diagnostic --pose-name must differ from the configured production "
            "pose name"
        )
    if args.stage == "train":
        if args.run_name is None:
            raise ValueError(
                "--allow-failed-georeferencing-for-render requires an explicit "
                "--run-name for train"
            )
        configured_run = str(getattr(_section(cfg, "train"), "run_name", "default"))
        if args.run_name == configured_run:
            raise ValueError(
                "diagnostic --run-name must differ from the configured production "
                "run name"
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.train_iters is not None and args.train_iters <= 0:
        raise ValueError("--train-iters must be positive")
    if args.expected_frames is not None and args.expected_frames <= 0:
        raise ValueError("--expected-frames must be positive")
    if args.tile_count is not None and args.tile_count <= 0:
        raise ValueError("--tile-count must be positive")
    if args.tile_max_tiles is not None and not 1 <= args.tile_max_tiles <= 4096:
        raise ValueError("--tile-max-tiles must be in [1, 4096]")
    if (
        args.tile_min_visibility_fraction is not None
        and not 0 <= args.tile_min_visibility_fraction <= 1
    ):
        raise ValueError("--tile-min-visibility-fraction must be in [0, 1]")
    if args.stage == "segment-materialize" and args.portable_segment is None:
        raise ValueError("segment-materialize requires --portable-segment")
    if args.stage != "segment-materialize" and args.portable_segment is not None:
        raise ValueError(
            "--portable-segment is valid only for segment-materialize"
        )
    if args.stage != "segment-materialize" and args.link_mode is not None:
        raise ValueError("--link-mode is valid only for segment-materialize")
    plan_only = (
        "tile_plan_name",
        "tile_count",
        "tile_max_tiles",
        "tile_min_visibility_fraction",
    )
    if args.stage != "tiles-plan" and any(
        getattr(args, name) is not None for name in plan_only
    ):
        raise ValueError(
            "TilePlan construction overrides are valid only for tiles-plan"
        )
    if (
        args.pose_artifact_root is not None
        and args.stage
        not in {"tiles-plan", "cloud", "train", "seam-probe", "scene-publish"}
    ):
        raise ValueError(
            "--pose-artifact-root is valid only for tiles-plan, cloud, train, "
            "seam-probe, and scene-publish"
        )
    scene_options = (
        "scene_tile_run", "scene_name", "seam_anchor_tile_id",
        "seam_assembly_policy", "scene_mode",
        "reference_run",
        "reference_params_sha256", "reference_metrics_sha256",
        "reference_provenance_sha256", "scene_opacity_threshold",
        "max_psnr_loss_db", "max_ssim_loss", "max_lpips_cc_increase",
        "max_combined_gaussians", "scene_device",
    )
    if args.stage == "scene-publish":
        mode = args.scene_mode or "controlled-ab"
        required = [
            "tile_plan", "pose_artifact_root", "scene_tile_run", "scene_name",
        ]
        if mode == "controlled-ab":
            required.extend((
                "reference_run", "reference_params_sha256",
                "reference_metrics_sha256", "reference_provenance_sha256",
            ))
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise ValueError(
                "scene-publish is missing: "
                + ", ".join("--" + name.replace("_", "-") for name in missing)
            )
        if args.tile_id is not None:
            raise ValueError("scene-publish consumes the whole plan, not --tile-id")
        if args.seam_anchor_tile_id is not None:
            raise ValueError("--seam-anchor-tile-id requires seam-probe")
        if args.seam_assembly_policy is not None:
            raise ValueError("--seam-assembly-policy requires seam-probe")
        if mode == "controlled-ab" and args.diagnostic_scene:
            raise ValueError("--diagnostic-scene requires --scene-mode production")
        if mode == "production":
            relative_options = (
                "reference_run", "reference_params_sha256",
                "reference_metrics_sha256", "reference_provenance_sha256",
                "max_psnr_loss_db", "max_ssim_loss",
                "max_lpips_cc_increase",
            )
            supplied = [
                name for name in relative_options
                if getattr(args, name) is not None
            ]
            if supplied:
                raise ValueError(
                    "production scene publication accepts absolute evidence only; "
                    "remove "
                    + ", ".join("--" + name.replace("_", "-") for name in supplied)
                )
    elif args.stage == "seam-probe":
        required = (
            "tile_plan",
            "pose_artifact_root",
            "scene_tile_run",
            "scene_name",
            "seam_anchor_tile_id",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise ValueError(
                "seam-probe is missing: "
                + ", ".join("--" + name.replace("_", "-") for name in missing)
            )
        if args.tile_id is not None:
            raise ValueError("seam-probe selects a pair, not --tile-id")
        unsupported = (
            "scene_mode",
            "reference_run",
            "reference_params_sha256",
            "reference_metrics_sha256",
            "reference_provenance_sha256",
            "scene_opacity_threshold",
        )
        supplied = [name for name in unsupported if getattr(args, name) is not None]
        if supplied:
            raise ValueError(
                "seam-probe does not accept "
                + ", ".join("--" + name.replace("_", "-") for name in supplied)
            )
    else:
        if args.diagnostic_scene or any(
            getattr(args, name) is not None for name in scene_options
        ):
            raise ValueError("scene publication options require scene-publish")
        if (args.tile_plan is None) != (args.tile_id is None):
            raise ValueError("--tile-plan and --tile-id must be supplied together")
        if args.tile_plan is not None and args.stage not in {"cloud", "train"}:
            raise ValueError(
                "--tile-plan and --tile-id are valid only for cloud and train"
            )
    if (
        args.prior_position_loss is not None
        and args.stage != "backend-refine-rtk"
    ):
        raise ValueError(
            "--prior-position-loss is valid only for backend-refine-rtk"
        )
    if (
        args.initialization_mode is not None
        and args.stage != "backend-refine-rtk"
    ):
        raise ValueError(
            "--initialization-mode is valid only for backend-refine-rtk"
        )
    if (
        args.allow_failed_georeferencing_for_render
        and args.stage not in {
            "backend-export",
            "tiles-plan",
            "cloud",
            "train",
        }
    ):
        raise ValueError(
            "--allow-failed-georeferencing-for-render is valid only for "
            "backend-export, tiles-plan, cloud, and train"
        )
    cfg = load_config(args.config, config_root=args.config_root)
    _validate_diagnostic_render_names(args, cfg)
    authored_config = authored_config_plain(cfg)
    if args.workdir is not None:
        cfg.paths.workdir = args.workdir.expanduser()
    if args.segment is not None:
        cfg.paths.segment = args.segment.expanduser()
    cfg.paths.workdir = Path(cfg.paths.workdir).expanduser()
    prepare_config_ledger(
        cfg,
        stage=args.stage,
        authored_config=authored_config,
        stage_overrides=_stage_overrides(args),
    )
    if args.train_iters is not None:
        from rtk_splat.core.runtime_resolution import set_runtime_cli_override

        set_runtime_cli_override(
            cfg,
            "train_iterations",
            cfg.train.iterations,
            args.train_iters,
            option="--train-iters",
            config_path="train.iterations",
        )
        cfg.train.iterations = args.train_iters
    COMMANDS[args.stage](cfg, args)
    commit_config_ledger(cfg, authored_config=authored_config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
