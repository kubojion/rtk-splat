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

import cv2
import numpy as np

from rtk_splat.configio import load_config
from rtk_splat.segment import SegmentReader


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    return str(args.backend or getattr(_section(cfg, "mapper"), "backend", "global"))


def _backend_name(cfg, args) -> str:
    return str(
        args.backend_name
        or getattr(_section(cfg, "mapper"), "name", None)
        or f"{_frontend_name(cfg, args)}-{_backend_kind(cfg, args)}"
    )


def _backend_path(cfg, args) -> Path:
    return Path(cfg.paths.workdir) / "backend_artifacts" / _backend_name(cfg, args)


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
    from adapters.registry import publish_from_config

    window = getattr(_section(cfg, "segment"), "window_s", None)
    if window is not None:
        if len(window) != 2 or float(window[1]) <= float(window[0]):
            raise ValueError("segment.window_s must be [start_s, end_s]")
    destination = _segment_path(cfg)
    reader = publish_from_config(cfg, destination)
    _validate_expected(reader, args.expected_frames)
    print(f"published immutable contract-v2 segment -> {reader.root}")


def cmd_depth(cfg, args) -> None:
    from workflows.depth import derive_sgbm_depth

    source = _reader(cfg, args)
    destination = (
        args.derived_segment.expanduser()
        if args.derived_segment is not None
        else Path(cfg.paths.workdir) / "segments" / f"{source.root.name}-sgbm"
    )
    result = derive_sgbm_depth(source.root, destination, cfg)
    print(f"published derived depth segment -> {result.root}")


def _frontend_configs(cfg, args):
    from frontends.keyframes import KEYFRAME_PRESETS, KeyframeConfig
    from frontends.pair_graph import PairGraphConfig

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
    from frontends.artifact import (
        FrontendArtifactBuilder,
        collect_provenance,
    )
    from frontends.planning import plan_frontend

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
        colmap=_colmap(cfg),
        seed=seed,
        repo_root=REPO_ROOT,
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
    from frontends.colmap import run_feature_extraction

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
    from frontends.colmap import run_rig_configurator

    report = run_rig_configurator(_frontend_path(cfg, args), _colmap(cfg))
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_frontend_priors(cfg, args) -> None:
    from frontends.colmap import insert_pose_priors

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
    from frontends.colmap import run_matches_importer

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
    from backends.mapper import MapperConfig

    mapper_values = dict(vars(_section(cfg, "mapper")))
    mapper_values.pop("name", None)
    mapper_values.pop("pose_artifact_name", None)
    if args.backend is not None:
        mapper_values["backend"] = args.backend
    return _dataclass_options(mapper_values, MapperConfig)


def cmd_backend_prepare(cfg, args) -> None:
    from backends.mapper import prepare_mapper_backend

    workspace = prepare_mapper_backend(
        _frontend_path(cfg, args),
        _backend_path(cfg, args),
        config=_mapper_config(cfg, args),
    )
    print(f"isolated verified mapper workspace -> {workspace}")


def cmd_backend_solve(cfg, args) -> None:
    from backends.mapper import run_mapper_solve

    report = run_mapper_solve(_backend_path(cfg, args), _colmap(cfg))
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_backend_register(cfg, args) -> None:
    from backends.mapper import run_image_registration

    report = run_image_registration(_backend_path(cfg, args), _colmap(cfg))
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_backend_quality(cfg, args) -> None:
    from backends.mapper import run_quality_summary

    report = run_quality_summary(_backend_path(cfg, args), _colmap(cfg))
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_backend_export(cfg, args) -> None:
    from backends.mapper import export_pose_artifact

    output = export_pose_artifact(
        _backend_path(cfg, args),
        _pose_name(cfg, args),
        output_root=Path(cfg.paths.workdir) / "pose_artifacts",
    )
    print(f"fixed-scale ENU pose artifact -> {output}")


def cmd_cloud(cfg, args) -> None:
    from rtk_splat.cloud import backproject, to_world, voxel_downsample
    from rtk_splat.pose_artifacts import (
        cloud_path,
        load_pose_artifact,
        pose_artifact_name,
        pose_fingerprint,
    )

    reader = _reader(cfg, args)
    if args.pose_name:
        cfg.pose.artifact = args.pose_name
    frames = reader.frames
    viewmats, _ = load_pose_artifact(reader.root, cfg)
    camera = reader.calibration["cameras"]["left"]
    k = np.asarray(camera["K"], dtype=float)
    intrinsics = {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }
    if "depth_path" not in frames or any(not str(path) for path in frames["depth_path"]):
        raise RuntimeError("cloud construction requires depth for every frame")
    points, colours = [], []
    for frame_id in reader.manifest["train"]:
        index = int(frame_id)
        with np.load(reader.root / str(frames["depth_path"][index])) as depth:
            image = cv2.imread(str(reader.root / str(frames["left_image_path"][index])))
            if image is None:
                raise RuntimeError(f"cannot read frame {index}")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            xyz, colour = backproject(
                depth["depth"].astype(np.float32),
                depth["valid"].astype(bool),
                rgb,
                intrinsics,
                int(cfg.cloud.pixel_stride),
            )
        points.append(to_world(xyz, viewmats[index]))
        colours.append(colour)
    xyz = np.concatenate(points).astype(np.float32)
    rgb = np.concatenate(colours)
    xyz, rgb = voxel_downsample(
        xyz, rgb, float(cfg.cloud.voxel_m), int(cfg.cloud.max_points)
    )
    output = cloud_path(reader.root, cfg)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cloud artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                xyz=xyz,
                rgb=rgb,
                pose_fingerprint=np.asarray(pose_fingerprint(viewmats)),
                pose_artifact=np.asarray(pose_artifact_name(cfg)),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"initial cloud: {len(xyz):,} points -> {output}")


def cmd_train(cfg, args) -> None:
    if args.pose_name:
        cfg.pose.artifact = args.pose_name
    if args.train_iters is not None:
        cfg.train.iterations = args.train_iters
    if args.run_name:
        cfg.train.run_name = args.run_name
    from backends.gsplat import train_tile

    reader = _reader(cfg, args)
    run = Path(cfg.paths.workdir) / "runs" / str(cfg.train.run_name)
    result = train_tile(reader.root, run, cfg)
    print(json.dumps(result, indent=2, sort_keys=True))


COMMANDS = {
    "validate": cmd_validate,
    "ingest": cmd_ingest,
    "depth": cmd_depth,
    "frontend-build": cmd_frontend_build,
    "frontend-features": cmd_frontend_features,
    "frontend-rig": cmd_frontend_rig,
    "frontend-priors": cmd_frontend_priors,
    "frontend-match": cmd_frontend_match,
    "backend-prepare": cmd_backend_prepare,
    "backend-solve": cmd_backend_solve,
    "backend-register": cmd_backend_register,
    "backend-quality": cmd_backend_quality,
    "backend-export": cmd_backend_export,
    "cloud": cmd_cloud,
    "train": cmd_train,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtk-splat")
    parser.add_argument("stage", choices=tuple(COMMANDS))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workdir", type=Path, help="new experiment root override")
    parser.add_argument("--segment", type=Path, help="immutable v2 segment override")
    parser.add_argument(
        "--derived-segment",
        type=Path,
        help="new contract-v2 destination for the depth stage",
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
    parser.add_argument("--pose-name")
    parser.add_argument("--run-name")
    parser.add_argument("--train-iters", type=int)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument(
        "--skip-image-quality",
        action="store_true",
        help="skip image decoding only for planning smoke tests",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.train_iters is not None and args.train_iters <= 0:
        raise ValueError("--train-iters must be positive")
    if args.expected_frames is not None and args.expected_frames <= 0:
        raise ValueError("--expected-frames must be positive")
    cfg = load_config(args.config)
    if args.workdir is not None:
        cfg.paths.workdir = args.workdir.expanduser()
    if args.segment is not None:
        cfg.paths.segment = args.segment.expanduser()
    cfg.paths.workdir = Path(cfg.paths.workdir).expanduser()
    COMMANDS[args.stage](cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
