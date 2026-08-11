"""Non-destructively export a completed GS checkpoint as a sealed PLY bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rtk_splat.backends.pose_evidence import (
    require_render_permission,
    splat_output_name,
    verify_training_run_georeferencing,
)
from rtk_splat.core.segment import publish_directory_noreplace


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _export_name(value: str) -> str:
    if not _SAFE_NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"invalid export name: {value!r}")
    return value


def _completed_run_evidence(
    source_run: Path,
    *,
    allow_failed_georeferencing_for_render: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, str]:
    if not source_run.is_dir():
        raise FileNotFoundError(f"completed training run does not exist: {source_run}")
    georeferencing = _json_object(source_run / "georeferencing.json")
    verify_training_run_georeferencing(source_run, georeferencing)
    require_render_permission(
        georeferencing,
        allow_failed_georeferencing_for_render=(
            allow_failed_georeferencing_for_render
        ),
    )
    best = _json_object(source_run / "best_checkpoint.json")
    provenance = _json_object(source_run / "run_provenance.json")
    provenance_selection = provenance.get("model_selection")
    if not isinstance(provenance_selection, dict):
        raise ValueError("training provenance has no completed model-selection record")
    for label, record in (
        ("best_checkpoint.json", best),
        ("run_provenance.json:model_selection", provenance_selection),
    ):
        if (
            record.get("status") != "complete"
            or record.get("parameter_artifact") != "params.pt"
            or record.get("exported_splat_uses_best_model") is not True
            or record.get("optimizer_state_included") is not False
        ):
            raise ValueError(f"{label} does not declare a completed selected model")
    keys = (
        "criterion",
        "direction",
        "tie_policy",
        "best_step",
        "best_metric",
        "completed_training_steps",
        "parameter_artifact",
        "parameter_artifact_sha256",
    )
    if any(best.get(key) != provenance_selection.get(key) for key in keys):
        raise ValueError("checkpoint and training provenance selection records disagree")
    if provenance.get("iterations") != best.get("completed_training_steps"):
        raise ValueError("training completion count disagrees with run provenance")
    expected_hash = best.get("parameter_artifact_sha256")
    if not isinstance(expected_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_hash
    ):
        raise ValueError("completed checkpoint has no valid params.pt hash")
    params_path = source_run / "params.pt"
    if not params_path.is_file() or params_path.stat().st_size <= 0:
        raise FileNotFoundError(f"completed parameters are missing: {params_path}")
    if _sha256_file(params_path) != expected_hash:
        raise ValueError("params.pt changed after the completed checkpoint was sealed")
    return georeferencing, best, provenance, params_path, expected_hash


def _initial_cloud_path(
    provenance: Mapping[str, Any], explicit: str | Path | None
) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    configuration = provenance.get("configuration")
    effective = (
        configuration.get("effective_config")
        if isinstance(configuration, Mapping)
        else None
    )
    paths = effective.get("paths") if isinstance(effective, Mapping) else None
    workdir = paths.get("workdir") if isinstance(paths, Mapping) else None
    pose_name = provenance.get("pose_artifact")
    if not isinstance(workdir, str) or not isinstance(pose_name, str):
        raise ValueError(
            "cannot infer the initial cloud; pass --initial-cloud explicitly"
        )
    return (
        Path(workdir).expanduser().resolve()
        / "cloud_artifacts"
        / pose_name
        / "init_cloud.npz"
    )


def _crop_bounds(
    provenance: Mapping[str, Any], initial_cloud: str | Path | None
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cloud_path = _initial_cloud_path(provenance, initial_cloud)
    expected_hash = provenance.get("initial_cloud_sha256")
    if not cloud_path.is_file() or not isinstance(expected_hash, str):
        raise ValueError(f"sealed initial cloud is unavailable: {cloud_path}")
    cloud_hash = _sha256_file(cloud_path)
    if cloud_hash != expected_hash:
        raise ValueError("initial cloud changed after the training run was sealed")
    try:
        with np.load(cloud_path, allow_pickle=False) as cloud:
            xyz = np.asarray(cloud["xyz"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"invalid initial cloud: {cloud_path}") from exc
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2 or not np.isfinite(xyz).all():
        raise ValueError("initial cloud xyz must be a finite non-empty Nx3 array")
    low = np.percentile(xyz, 1, axis=0) - 1.0
    high = np.percentile(xyz, 99, axis=0) + 1.0
    return low, high, {
        "enabled": True,
        "initial_cloud": str(cloud_path),
        "initial_cloud_sha256": cloud_hash,
        "lower_percentile": 1.0,
        "upper_percentile": 99.0,
        "margin_m": 1.0,
        "lower_bound_m": low.tolist(),
        "upper_bound_m": high.tolist(),
    }


def export_completed_splat(
    source_run: str | Path,
    export_name: str,
    *,
    opacity_threshold: float,
    crop: bool,
    initial_cloud: str | Path | None = None,
    allow_failed_georeferencing_for_render: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Publish a new export bundle without changing completed-run artifacts."""
    source = Path(source_run).expanduser().resolve()
    name = _export_name(export_name)
    threshold = float(opacity_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
        raise ValueError("opacity_threshold must be finite and in [0, 1)")
    if not crop and initial_cloud is not None:
        raise ValueError("--initial-cloud is valid only with --crop")
    georeferencing, best, provenance, params_path, params_hash = (
        _completed_run_evidence(
            source,
            allow_failed_georeferencing_for_render=(
                allow_failed_georeferencing_for_render
            ),
        )
    )
    crop_policy: dict[str, Any]
    bounds = None
    if crop:
        low, high, crop_policy = _crop_bounds(provenance, initial_cloud)
        bounds = (low, high)
    else:
        crop_policy = {"enabled": False}

    exports = source / "exports"
    destination = exports / name
    exports.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to modify existing export: {destination}")
    staging = exports / f".{name}.writing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        from rtk_splat.backends.gsplat import (
            _load_checkpoint_gaussians,
            export_splat_tensors,
        )

        params = _load_checkpoint_gaussians(params_path, device)
        output_name = splat_output_name(georeferencing)
        output_path = staging / output_name
        total, retained = export_splat_tensors(
            params,
            output_path,
            opacity_threshold=threshold,
            crop_bounds=bounds,
        )
        output_hash = _sha256_file(output_path)
        if _sha256_file(params_path) != params_hash:
            raise ValueError("params.pt changed while the export was running")
        splat_evidence = {
            **georeferencing,
            "splat_file": output_name,
            "splat_sha256": output_hash,
        }
        _write_json(staging / "splat.georeferencing.json", splat_evidence)
        export_provenance = {
            "schema_version": 1,
            "artifact_type": "completed_checkpoint_splat_export",
            "source_run": str(source),
            "source_params_file": "params.pt",
            "source_params_sha256": params_hash,
            "source_best_checkpoint_sha256": _sha256_file(
                source / "best_checkpoint.json"
            ),
            "source_run_provenance_sha256": _sha256_file(
                source / "run_provenance.json"
            ),
            "source_georeferencing_sha256": _sha256_file(
                source / "georeferencing.json"
            ),
            "source_selected_model": {
                key: best.get(key)
                for key in (
                    "criterion",
                    "best_step",
                    "best_metric",
                    "completed_training_steps",
                )
            },
            "policy": {
                "opacity": "sigmoid(logit) > threshold",
                "opacity_threshold": threshold,
                "crop": crop_policy,
            },
            "total_gaussians": total,
            "retained_gaussians": retained,
            "retained_fraction": retained / total,
            "splat_file": output_name,
            "splat_sha256": output_hash,
            "device": device,
        }
        _write_json(staging / "export_provenance.json", export_provenance)
        files = {}
        for path in sorted(staging.iterdir()):
            if path.is_file():
                files[path.name] = {
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
        _write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "artifact_type": "completed_checkpoint_splat_export",
                "name": name,
                "files": files,
            },
        )
        publish_directory_noreplace(staging, destination)
        return {
            "export": str(destination),
            "splat": str(destination / output_name),
            "splat_sha256": output_hash,
            "total_gaussians": total,
            "retained_gaussians": retained,
            "retained_fraction": retained / total,
            "opacity_threshold": threshold,
            "crop": crop_policy,
            "source_metrics_unchanged": True,
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rtk-splat-export",
        description="Export a sealed PLY from a completed GS params.pt",
    )
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--export-name", required=True)
    parser.add_argument("--opacity-threshold", required=True, type=float)
    crop = parser.add_mutually_exclusive_group(required=True)
    crop.add_argument("--crop", action="store_true")
    crop.add_argument("--no-crop", dest="crop", action="store_false")
    parser.add_argument("--initial-cloud", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--allow-failed-georeferencing-for-render",
        action="store_true",
        help="preserve and explicitly authorize a diagnostic-only source run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_completed_splat(
        args.source_run,
        args.export_name,
        opacity_threshold=args.opacity_threshold,
        crop=args.crop,
        initial_cloud=args.initial_cloud,
        allow_failed_georeferencing_for_render=(
            args.allow_failed_georeferencing_for_render
        ),
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
