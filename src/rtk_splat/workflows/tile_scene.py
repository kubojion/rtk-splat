"""Publish and evaluate one immutable scene from completed TilePlan runs.

Training tiles overlap so that both models see useful context at a seam.  The
published scene does not overlap ownership: each final Gaussian centre is
assigned by the TilePlan's sealed, half-open core rule.  Quality is measured
after concatenating those core-owned tensors and rendering the original source
validation split once.  Per-tile metrics are deliberately not averaged.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rtk_splat.backends.pose_evidence import canonical_georeferencing_json
from rtk_splat.core.pose_artifacts import load_pose_artifact, pose_fingerprint
from rtk_splat.core.segment import SegmentReader, publish_directory_noreplace
from rtk_splat.frontends.artifact import (
    ArtifactError,
    canonical_hash,
    collect_package_state,
    sha256_file,
)
from rtk_splat.workflows.tiles import owner_tile_indices, verify_tile_plan


SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METRICS = ("psnr_masked", "psnr_masked_cc", "ssim", "lpips_cc")
_PARAMETERS = ("means", "quats", "scales", "opacities", "sh0", "shN")


def _safe_name(value: str) -> str:
    value = str(value)
    if not _SAFE_NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"invalid scene name: {value!r}")
    return value


def _json(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(value, allow_nan=False)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ArtifactError(f"invalid JSON artifact: {path}") from exc
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ArtifactError(f"scene file is missing or unsafe: {path}")
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _manifest(root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[path.relative_to(root).as_posix()] = _record(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "rtk_splat_tiled_scene",
        "files": files,
    }


def verify_tiled_scene(root: str | Path) -> dict[str, Any]:
    """Verify a scene's terminal file seal and redundant status claims."""
    supplied = Path(root).expanduser()
    if supplied.is_symlink():
        raise ArtifactError(f"tiled scene cannot be a symlink: {supplied}")
    root = supplied.resolve()
    if not root.is_dir():
        raise ArtifactError(f"tiled scene does not exist: {root}")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ArtifactError("tiled scene contains a symlink")
    manifest = _json(root / "manifest.json")
    if manifest != _manifest(root):
        raise ArtifactError("tiled-scene terminal seal verification failed")
    scene = _json(root / "scene.json")
    quality = _json(root / "quality.json")
    metrics = _json(root / "metrics.json")
    provenance = _json(root / "provenance.json")
    if any(
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA_VERSION
        for value in (scene, quality, metrics, provenance)
    ):
        raise ArtifactError("tiled-scene schema is invalid")
    scene_name = str(scene.get("name", ""))
    valid_directory = root.name == scene_name or root.name.startswith(
        f".{scene_name}.writing-"
    )
    if (
        scene.get("artifact_type") != "rtk_splat_tiled_scene"
        or not _SAFE_NAME.fullmatch(scene_name)
        or not valid_directory
        or scene.get("quality_passed") is not quality.get("passed")
        or scene.get("metric_georeferencing_claim_eligible")
        is not provenance.get("georeferencing", {}).get(
            "metric_georeferencing_claim_eligible"
        )
        or scene.get("provisional")
        is not (not scene.get("metric_georeferencing_claim_eligible"))
    ):
        raise ArtifactError("tiled-scene status claims disagree")
    output = root / str(scene.get("splat_file", ""))
    if _record(output) != {
        "sha256": scene.get("splat_sha256"),
        "size_bytes": scene.get("splat_size_bytes"),
    }:
        raise ArtifactError("tiled-scene PLY evidence is invalid")
    checks = quality.get("checks")
    if not isinstance(checks, dict) or not all(
        isinstance(value, bool) for value in checks.values()
    ):
        raise ArtifactError("tiled-scene quality checks are invalid")
    if quality.get("passed") is not all(checks.values()):
        raise ArtifactError("tiled-scene quality result disagrees with its checks")
    if metrics.get("evaluation", {}).get("aggregation") != (
        "single_render_of_concatenated_core_owned_gaussians"
    ):
        raise ArtifactError("tiled-scene evaluation aggregation is invalid")
    publication_mode = scene.get("publication_mode", "controlled_ab")
    if publication_mode not in {"controlled_ab", "production"}:
        raise ArtifactError("tiled-scene publication mode is invalid")
    if publication_mode == "production" and any(
        record.get("publication_mode") != "production"
        for record in (quality, metrics, provenance)
    ):
        raise ArtifactError("production tiled-scene mode claims disagree")
    seam = metrics.get("seam")
    evidence = seam.get("evidence") if isinstance(seam, dict) else None
    if not isinstance(evidence, dict) or (
        evidence.get("n_validation_frames", 0) < 2
        or evidence.get("metric_depth_pixels_in_band", 0) < 512
    ):
        raise ArtifactError("tiled-scene seam evaluation is invalid")
    if publication_mode == "controlled_ab":
        if (
            not isinstance(seam.get("reference"), dict)
            or not isinstance(seam.get("candidate"), dict)
            or not isinstance(seam.get("regressions"), dict)
        ):
            raise ArtifactError("tiled-scene seam evaluation is invalid")
        seam_values = [
            *seam["reference"].values(),
            *seam["candidate"].values(),
            *seam["regressions"].values(),
        ]
    else:
        absolute = metrics.get("absolute")
        seam_absolute = seam.get("absolute")
        completeness = metrics.get("completeness")
        forbidden = ("reference", "candidate", "loss_db", "regressions")
        if (
            not isinstance(absolute, dict)
            or not isinstance(seam_absolute, dict)
            or not isinstance(completeness, dict)
            or any(key in metrics for key in forbidden)
            or any(key in seam for key in forbidden)
            or "reference" in provenance
            or metrics.get("evaluation", {}).get("comparison")
            != "absolute_only_no_monolithic_reference"
            or seam.get("comparison") != "absolute_only_no_reference"
        ):
            raise ArtifactError(
                "production scene must contain absolute, non-relative evidence"
            )
        for label, record in (
            ("whole-scene", absolute), ("seam", seam_absolute)
        ):
            if any(
                isinstance(record.get(key), bool)
                or not isinstance(record.get(key), (int, float))
                or not math.isfinite(float(record[key]))
                for key in _METRICS
            ):
                raise ArtifactError(
                    f"production {label} absolute metrics are invalid"
                )
        planned = completeness.get("planned_tile_ids")
        completed = completeness.get("completed_tile_ids")
        validation = completeness.get("source_validation_frame_ids")
        ownership = scene.get("ownership", {})
        tiles = ownership.get("tiles")
        if (
            not isinstance(planned, list)
            or completed != planned
            or completeness.get("n_planned_tiles") != len(planned)
            or completeness.get("n_completed_tiles") != len(planned)
            or not isinstance(validation, list)
            or completeness.get("n_source_validation_frames") != len(validation)
            or completeness.get("n_evaluated_validation_frames") != len(validation)
            or absolute.get("eval_ids") != validation
            or absolute.get("n_eval") != len(validation)
            or not isinstance(tiles, list)
            or [item.get("tile_id") for item in tiles] != planned
            or any(item.get("core_owned_gaussians", 0) <= 0 for item in tiles)
            or completeness.get("all_tiles_contributed_core_gaussians") is not True
            or completeness.get("source_gaussians")
            != ownership.get("source_gaussians")
            or completeness.get("core_owned_gaussians")
            != ownership.get("core_owned_gaussians")
            or completeness.get("outside_or_other_core_gaussians")
            != sum(
                int(item.get("outside_or_other_core_gaussians", -1))
                for item in tiles
            )
        ):
            raise ArtifactError("production tiled-scene completeness is invalid")
        georeferencing = provenance.get("georeferencing", {})
        production_georeferencing = bool(
            georeferencing.get("artifact_class") == "production"
            and georeferencing.get("georeferencing_status") == "PASSED"
            and georeferencing.get("metric_georeferencing_claim_eligible") is True
        )
        diagnostic = provenance.get("diagnostic_nonproduction_georeferencing")
        if production_georeferencing:
            if diagnostic is not False or scene.get("provisional") is not False:
                raise ArtifactError("production georeferencing was demoted")
        elif (
            diagnostic is not True
            or scene.get("provisional") is not True
            or scene.get("metric_georeferencing_claim_eligible") is not False
        ):
            raise ArtifactError(
                "non-production georeferencing lacks an explicit diagnostic seal"
            )
        seam_values = [seam_absolute[key] for key in _METRICS]
    if not seam_values or not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in seam_values
    ):
        raise ArtifactError("tiled-scene seam metrics are non-finite")
    return scene


def _require_hash(value: str, label: str) -> str:
    value = str(value)
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _selected_reference_metrics(
    metrics_file: Path,
    expected_sha256: str,
    validation_ids: Sequence[int],
) -> tuple[dict[str, float], dict[str, Any]]:
    expected = _require_hash(expected_sha256, "reference metrics hash")
    actual = sha256_file(metrics_file)
    if actual != expected:
        raise ArtifactError("reference metrics changed or the expected hash is wrong")
    history = _json(metrics_file)
    if not isinstance(history, list) or not history:
        raise ArtifactError("reference metrics must be a non-empty history list")
    selected = [
        item for item in history
        if isinstance(item, dict) and item.get("selected_for_export") is True
    ]
    if len(selected) > 1:
        raise ArtifactError("reference metrics select more than one exported model")
    record = selected[0] if selected else history[-1]
    if not isinstance(record, dict):
        raise ArtifactError("reference final metric record is invalid")
    per_frame = record.get("per_frame")
    reference_ids = per_frame.get("eval_ids") if isinstance(per_frame, dict) else None
    expected_ids = [int(value) for value in validation_ids]
    if reference_ids != expected_ids or record.get("n_eval") != len(expected_ids):
        raise ArtifactError(
            "reference metrics were not evaluated on the source validation split"
        )
    result = {}
    for key in _METRICS:
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)):
            raise ArtifactError(f"reference metric {key} is missing or non-finite")
        result[key] = float(value)
    return result, {
        "metrics_file": str(metrics_file.resolve()),
        "metrics_sha256": actual,
        "record_selector": (
            "selected_for_export" if selected else "last_history_record"
        ),
        "record_step": record.get("step"),
        "record_model_step": record.get("model_step"),
    }


def _training_identity(provenance: Mapping[str, Any]) -> dict[str, str]:
    """Return trainer identity without output names or source-file locators."""
    implementation = provenance.get("training_implementation_sha256")
    config = provenance.get("effective_training_config")
    if (
        not isinstance(implementation, str)
        or not _SHA256.fullmatch(implementation)
        or not isinstance(config, dict)
        or not isinstance(config.get("train"), dict)
    ):
        raise ArtifactError("training run has no comparable implementation/config")
    comparable = copy.deepcopy(config)
    comparable["train"].pop("run_name", None)
    # Layer locators prove where an authored file was loaded from, but their
    # absolute checkout paths do not change the resolved scientific settings.
    # Keep every content hash, role, authored value, derivation, and chosen
    # value while removing only these non-causal locators.  This lets a sealed
    # historical tile be compared with an identical tile trained from a fresh
    # isolated worktree without weakening any configuration check.
    runtime = comparable.get("runtime_resolution")
    if isinstance(runtime, dict):
        runtime.pop("config_sources", None)
        source_files = runtime.get("source_files")
        if isinstance(source_files, list):
            for record in source_files:
                if isinstance(record, dict):
                    record.pop("path", None)
        origins = runtime.get("origins")
        if isinstance(origins, dict):
            # The selected frame IDs and complete TilePlan binding are checked
            # independently.  A historical tiles-plan stage may leave this
            # planning-only origin in a later training snapshot, while a fresh
            # execution that consumes the same sealed plan does not.
            origins.pop("tile_max_training_frames", None)
            for record in origins.values():
                if isinstance(record, dict):
                    record.pop("source_path", None)
        derivations = runtime.get("derivations")
        if isinstance(derivations, dict):
            derivations.pop("tile_max_training_frames", None)
            for record in derivations.values():
                if not isinstance(record, dict):
                    continue
                origin = record.get("origin")
                if isinstance(origin, dict):
                    origin.pop("source_path", None)
    return {
        "training_implementation_sha256": implementation,
        "comparable_training_config_sha256": canonical_hash(comparable),
    }


def _reference_run_evidence(
    reference_run: Path,
    expected_hashes: Mapping[str, str],
    plan: Mapping[str, Any],
    inventory: Mapping[str, Any],
    validation_ids: Sequence[int],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Verify the frozen historical control without treating it as a new run."""
    reference_run = reference_run.expanduser().resolve()
    required = {
        "params.pt": "params_sha256",
        "metrics.json": "metrics_sha256",
        "run_provenance.json": "provenance_sha256",
    }
    records = {}
    for filename, key in required.items():
        expected = _require_hash(expected_hashes.get(key, ""), key)
        path = reference_run / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise ArtifactError(f"frozen reference changed: {path}")
        records[filename] = _record(path)
    provenance = _json(reference_run / "run_provenance.json")
    pose = inventory["pose"]
    contracts = inventory["segment"]["contract_files"]
    if (
        provenance.get("pose_fingerprint") != pose.get("fingerprint")
        or provenance.get("pose_artifact") != pose.get("name")
        or provenance.get("manifest_sha256")
        != contracts["manifest.json"]["sha256"]
        or provenance.get("frames_sha256")
        != contracts["frames.npz"]["sha256"]
        or provenance.get("calibration_sha256")
        != contracts["calibration.json"]["sha256"]
        or provenance.get("segment_meta_sha256")
        != contracts["segment_meta.json"]["sha256"]
        or plan["source_binding"]["pose_fingerprint"] != pose.get("fingerprint")
    ):
        raise ArtifactError("frozen reference does not share the TilePlan inputs")
    metrics, metric_evidence = _selected_reference_metrics(
        reference_run / "metrics.json",
        expected_hashes["metrics_sha256"],
        validation_ids,
    )
    return metrics, {
        "run": str(reference_run),
        "files": records,
        **metric_evidence,
        "pose_artifact": provenance.get("pose_artifact"),
        "pose_fingerprint": provenance.get("pose_fingerprint"),
        "training_identity": _training_identity(provenance),
    }


def _expected_tile_binding(
    plan: Mapping[str, Any],
    plan_root: Path,
    inventory: Mapping[str, Any],
    tile: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": plan["name"],
        "manifest_sha256": sha256_file(plan_root / "manifest.json"),
        "tile_plan_sha256": sha256_file(plan_root / "tile_plan.json"),
        "source_inventory_sha256": canonical_hash(inventory),
        "pose_fingerprint": plan["source_binding"]["pose_fingerprint"],
        "tile_id": tile["tile_id"],
        "tile_index": index,
        "train_selection_sha256": canonical_hash(tile["frame_ids"]["train"]),
        "val_selection_sha256": canonical_hash(tile["frame_ids"]["val"]),
        "test_selection_sha256": canonical_hash(tile["frame_ids"]["test"]),
        "n_train": len(tile["frame_ids"]["train"]),
        "n_val": len(tile["frame_ids"]["val"]),
        "n_test": len(tile["frame_ids"]["test"]),
        "partition_origin_enu_m": plan["coordinate_frame"][
            "partition_origin_enu_m"
        ],
        "R_enu_from_partition": plan["coordinate_frame"][
            "R_enu_from_partition"
        ],
        "scene_bounds_uv_m": plan["partition"]["scene_bounds_uv_m"],
        "core_bounds_uv_m": tile["core_bounds_uv_m"],
        "context_bounds_uv_m": tile["context_bounds_uv_m"],
        "boundary_rule": plan["partition"]["boundary_rule"],
        "ownership_tolerance_m": plan["partition"]["ownership_tolerance_m"],
    }


def _completed_tile_run(
    run: Path,
    expected_binding: Mapping[str, Any],
    expected_georeferencing: Mapping[str, Any],
    *,
    allow_nonproduction_georeferencing_for_diagnostic: bool = False,
) -> tuple[dict[str, Any], Path, str, dict[str, Any]]:
    from rtk_splat.workflows.export_splat import _completed_run_evidence

    georeferencing, _, provenance, params, params_hash = _completed_run_evidence(
        run,
        allow_failed_georeferencing_for_render=(
            allow_nonproduction_georeferencing_for_diagnostic
        ),
    )
    if canonical_georeferencing_json(georeferencing) != (
        canonical_georeferencing_json(dict(expected_georeferencing))
    ):
        raise ArtifactError(f"tile run changed georeferencing evidence: {run}")
    if provenance.get("tile_plan") != dict(expected_binding):
        raise ArtifactError(f"tile run has the wrong TilePlan binding: {run}")
    cloud_hash = provenance.get("initial_cloud_sha256")
    config_hash = provenance.get("effective_training_config_sha256")
    if not isinstance(cloud_hash, str) or not _SHA256.fullmatch(cloud_hash):
        raise ArtifactError(f"tile run has no sealed initial-cloud hash: {run}")
    if not isinstance(config_hash, str) or not _SHA256.fullmatch(config_hash):
        raise ArtifactError(f"tile run has no sealed training-config hash: {run}")
    splat_evidence = _json(run / "splat.georeferencing.json")
    splat_name = splat_evidence.get("splat_file")
    splat_hash = splat_evidence.get("splat_sha256")
    if (
        not isinstance(splat_name, str)
        or not isinstance(splat_hash, str)
        or not _SHA256.fullmatch(splat_hash)
        or sha256_file(run / splat_name) != splat_hash
    ):
        raise ArtifactError(f"tile run has invalid source PLY evidence: {run}")
    return provenance, params, params_hash, {
        "initial_cloud_sha256": cloud_hash,
        "splat_file": splat_name,
        "splat_sha256": splat_hash,
    }


def _verify_training_source_binding(
    provenance: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> None:
    """Require a tile run to name the exact segment and pose sealed by its plan."""
    try:
        contracts = inventory["segment"]["contract_files"]
        pose = inventory["pose"]
    except (KeyError, TypeError) as exc:
        raise ArtifactError("TilePlan source inventory is incomplete") from exc
    expected = {
        "manifest_sha256": contracts["manifest.json"]["sha256"],
        "frames_sha256": contracts["frames.npz"]["sha256"],
        "calibration_sha256": contracts["calibration.json"]["sha256"],
        "segment_meta_sha256": contracts["segment_meta.json"]["sha256"],
        "pose_artifact": pose["name"],
        "pose_fingerprint": pose["fingerprint"],
    }
    changed = [
        key for key, value in expected.items()
        if provenance.get(key) != value
    ]
    if changed:
        raise ArtifactError(
            "tile training source binding disagrees with the TilePlan: "
            + ", ".join(changed)
        )


def _concatenate_core_params(
    plan: Mapping[str, Any],
    completed: Sequence[tuple[Mapping[str, Any], Path, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load on CPU, apply exact final-centre ownership, and concatenate."""
    import torch
    from rtk_splat.backends.gsplat import _load_checkpoint_gaussians

    pieces: dict[str, list[Any]] = {name: [] for name in _PARAMETERS}
    records = []
    plan_indices = {
        str(tile["tile_id"]): index
        for index, tile in enumerate(plan["tiles"])
    }
    if len(plan_indices) != len(plan["tiles"]):
        raise ArtifactError("TilePlan contains duplicate tile IDs")
    completed_ids = [str(tile["tile_id"]) for tile, _, _ in completed]
    if len(completed_ids) != len(set(completed_ids)):
        raise ArtifactError("completed tile inputs contain duplicate tile IDs")
    for tile, params_path, expected_hash in completed:
        tile_id = str(tile["tile_id"])
        if tile_id not in plan_indices:
            raise ArtifactError(f"completed tile is absent from TilePlan: {tile_id}")
        params = _load_checkpoint_gaussians(params_path, "cpu")
        if sha256_file(params_path) != expected_hash:
            raise ArtifactError(f"tile params changed while loading: {params_path}")
        means = params["means"].detach().cpu().numpy()
        if means.ndim != 2 or means.shape[1] != 3 or not np.isfinite(means).all():
            raise ArtifactError(f"tile means are invalid: {params_path}")
        owners = owner_tile_indices(plan, means)
        keep = torch.from_numpy(owners == plan_indices[tile_id])
        retained = int(keep.sum().item())
        if retained <= 0:
            raise ArtifactError(f"{tile['tile_id']} owns no final Gaussians")
        for name in _PARAMETERS:
            tensor = params[name].detach().cpu()
            if len(tensor) != len(means) or not bool(torch.isfinite(tensor).all()):
                raise ArtifactError(f"invalid {name} tensor in {params_path}")
            pieces[name].append(tensor[keep].contiguous())
        records.append({
            "tile_id": tile["tile_id"],
            "source_params": str(params_path.resolve()),
            "source_params_sha256": expected_hash,
            "source_gaussians": len(means),
            "core_owned_gaussians": retained,
            "outside_or_other_core_gaussians": int(len(means) - retained),
        })
    combined = {name: torch.cat(values, dim=0) for name, values in pieces.items()}
    return combined, records


def _core_feather_raw_weight(
    uv: np.ndarray,
    core: np.ndarray,
    width_m: float,
) -> np.ndarray:
    """Linear exterior-core weight used to form a partition of unity."""
    outside = np.maximum(np.maximum(core[0] - uv, uv - core[1]), 0.0)
    distance = np.linalg.norm(outside, axis=1)
    return np.clip(1.0 - distance / float(width_m), 0.0, 1.0)


def _concatenate_feathered_params(
    plan: Mapping[str, Any],
    completed: Sequence[tuple[Mapping[str, Any], Path, str]],
    *,
    feather_width_m: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Blend overlapping tile support with normalized optical thickness.

    A tile has raw weight one inside its core and linearly decreasing weight
    outside it.  All raw core-distance weights are normalized at each Gaussian
    centre, so the spatial weights form a partition of unity.  Opacity is
    scaled in optical-thickness space, where coincident weighted contributors
    reproduce the transmittance of one contributor instead of double-darkening
    the seam.
    """
    import torch
    from rtk_splat.backends.gsplat import _load_checkpoint_gaussians

    width = float(feather_width_m)
    if not math.isfinite(width) or width <= 0:
        raise ValueError("feather_width_m must be finite and positive")
    plan_tiles = list(plan.get("tiles", []))
    plan_indices = {
        str(tile["tile_id"]): index for index, tile in enumerate(plan_tiles)
    }
    if not plan_tiles or len(plan_indices) != len(plan_tiles):
        raise ArtifactError("TilePlan contains missing or duplicate tile IDs")
    completed_ids = [str(tile["tile_id"]) for tile, _, _ in completed]
    if len(completed_ids) != len(set(completed_ids)):
        raise ArtifactError("completed tile inputs contain duplicate tile IDs")
    origin = np.asarray(
        plan["coordinate_frame"]["partition_origin_enu_m"], dtype=np.float64
    )
    basis = np.asarray(
        plan["coordinate_frame"]["R_enu_from_partition"], dtype=np.float64
    )
    scene = np.asarray(plan["partition"]["scene_bounds_uv_m"], dtype=np.float64)
    tolerance = float(plan["partition"]["ownership_tolerance_m"])
    all_cores = {
        str(tile["tile_id"]): np.asarray(
            tile["core_bounds_uv_m"], dtype=np.float64
        )
        for tile in plan_tiles
    }
    cores = {tile_id: all_cores[tile_id] for tile_id in completed_ids}
    if (
        origin.shape != (3,)
        or basis.shape != (3, 3)
        or scene.shape != (2, 2)
        or not np.isfinite(origin).all()
        or not np.isfinite(basis).all()
        or not np.isfinite(scene).all()
        or np.any(scene[1] <= scene[0])
    ):
        raise ArtifactError("TilePlan feather geometry is invalid")

    pieces: dict[str, list[Any]] = {name: [] for name in _PARAMETERS}
    records = []
    for tile, params_path, expected_hash in completed:
        tile_id = str(tile["tile_id"])
        if tile_id not in plan_indices:
            raise ArtifactError(f"completed tile is absent from TilePlan: {tile_id}")
        params = _load_checkpoint_gaussians(params_path, "cpu")
        if sha256_file(params_path) != expected_hash:
            raise ArtifactError(f"tile params changed while loading: {params_path}")
        means = params["means"].detach().cpu().numpy()
        if means.ndim != 2 or means.shape[1] != 3 or not np.isfinite(means).all():
            raise ArtifactError(f"tile means are invalid: {params_path}")
        uv = ((means.astype(np.float64, copy=False) - origin) @ basis)[:, :2]
        core = cores[tile_id]
        source_raw = _core_feather_raw_weight(uv, core, width)
        inside_scene = np.all(uv >= scene[0] - tolerance, axis=1) & np.all(
            uv <= scene[1] + tolerance, axis=1
        )
        keep_np = (source_raw > 0.0) & inside_scene
        if not np.any(keep_np):
            raise ArtifactError(f"{tile_id} contributes no feathered Gaussians")
        selected_uv = uv[keep_np]
        denominator = np.zeros(len(selected_uv), dtype=np.float64)
        source_expanded = np.stack((core[0] - width, core[1] + width))
        contributing_cores = []
        for other_id, other_core in cores.items():
            other_expanded = np.stack((
                other_core[0] - width,
                other_core[1] + width,
            ))
            if np.any(
                np.minimum(source_expanded[1], other_expanded[1])
                < np.maximum(source_expanded[0], other_expanded[0])
            ):
                continue
            denominator += _core_feather_raw_weight(
                selected_uv, other_core, width
            )
            contributing_cores.append(other_id)
        selected_raw = source_raw[keep_np]
        if (
            np.any(~np.isfinite(denominator))
            or np.any(denominator <= 0)
            or np.any(selected_raw > denominator + 1e-12)
        ):
            raise ArtifactError("TilePlan feather normalization is invalid")
        weights_np = selected_raw / denominator
        if np.any(weights_np <= 0) or np.any(weights_np > 1 + 1e-12):
            raise ArtifactError("normalized feather weights are invalid")
        keep = torch.from_numpy(keep_np)
        weights = torch.from_numpy(weights_np).to(dtype=params["opacities"].dtype)
        for parameter_name in _PARAMETERS:
            tensor = params[parameter_name].detach().cpu()
            if len(tensor) != len(means) or not bool(torch.isfinite(tensor).all()):
                raise ArtifactError(
                    f"invalid {parameter_name} tensor in {params_path}"
                )
            selected = tensor[keep].contiguous()
            if parameter_name == "opacities":
                alpha = torch.sigmoid(selected)
                epsilon = torch.finfo(alpha.dtype).eps
                alpha = alpha.clamp(epsilon, 1.0 - epsilon)
                optical_thickness = -torch.log1p(-alpha)
                weighted_alpha = -torch.expm1(-optical_thickness * weights)
                weighted_alpha = weighted_alpha.clamp(epsilon, 1.0 - epsilon)
                selected = torch.logit(weighted_alpha).contiguous()
            pieces[parameter_name].append(selected)
        strict_core = np.all(uv >= core[0], axis=1) & np.all(
            uv <= core[1], axis=1
        )
        retained = int(keep_np.sum())
        strict_retained = int(np.sum(keep_np & strict_core))
        records.append({
            "tile_id": tile_id,
            "source_params": str(params_path.resolve()),
            "source_params_sha256": expected_hash,
            "source_gaussians": len(means),
            "retained_gaussians": retained,
            "strict_core_gaussians": strict_retained,
            "feather_support_gaussians": retained - strict_retained,
            "discarded_gaussians": int(len(means) - retained),
            "minimum_normalized_weight": float(np.min(weights_np)),
            "maximum_normalized_weight": float(np.max(weights_np)),
            "mean_normalized_weight": float(np.mean(weights_np)),
            "contributing_core_ids": sorted(contributing_cores),
        })
    combined = {name: torch.cat(values, dim=0) for name, values in pieces.items()}
    if not len(combined["means"]):
        raise ArtifactError("feathered assembly is empty")
    return combined, records


def _shared_core_seam_segment(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    tolerance_m: float = 1e-9,
) -> tuple[int, float, float, float]:
    """Return the positive-length core edge shared by two adjacent tiles."""
    a = np.asarray(first["core_bounds_uv_m"], dtype=np.float64)
    b = np.asarray(second["core_bounds_uv_m"], dtype=np.float64)
    if (
        a.shape != (2, 2)
        or b.shape != (2, 2)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or np.any(a[1] <= a[0])
        or np.any(b[1] <= b[0])
    ):
        raise ArtifactError("tile core geometry is invalid")
    segments = []
    for axis in (0, 1):
        other = 1 - axis
        if math.isclose(
            float(a[1, axis]),
            float(b[0, axis]),
            rel_tol=0.0,
            abs_tol=tolerance_m,
        ):
            coordinate = float((a[1, axis] + b[0, axis]) / 2.0)
        elif math.isclose(
            float(b[1, axis]),
            float(a[0, axis]),
            rel_tol=0.0,
            abs_tol=tolerance_m,
        ):
            coordinate = float((b[1, axis] + a[0, axis]) / 2.0)
        else:
            continue
        low = float(max(a[0, other], b[0, other]))
        high = float(min(a[1, other], b[1, other]))
        if high - low > tolerance_m:
            segments.append((axis, coordinate, low, high))
    if len(segments) != 1:
        raise ArtifactError("tiles do not share exactly one positive-length core edge")
    return segments[0]


def select_geometric_tile_neighbor(
    plan: Mapping[str, Any],
    anchor_tile_id: str,
) -> dict[str, Any]:
    """Select a neighbor using sealed core geometry only.

    The longest shared edge wins; a tile-ID tie break makes the choice stable.
    No images, validation metrics, or finished-model evidence are consulted.
    """
    tiles = plan.get("tiles")
    if not isinstance(tiles, list) or len(tiles) < 2:
        raise ArtifactError("TilePlan needs at least two tiles")
    by_id = {str(tile.get("tile_id")): tile for tile in tiles}
    if len(by_id) != len(tiles) or anchor_tile_id not in by_id:
        raise ArtifactError("anchor tile is unknown or TilePlan IDs are invalid")
    candidates = []
    anchor = by_id[anchor_tile_id]
    for tile_id, tile in by_id.items():
        if tile_id == anchor_tile_id:
            continue
        try:
            axis, coordinate, low, high = _shared_core_seam_segment(anchor, tile)
        except ArtifactError:
            continue
        candidates.append({
            "tile_id": tile_id,
            "axis": int(axis),
            "coordinate_m": float(coordinate),
            "span_m": [float(low), float(high)],
            "shared_edge_length_m": float(high - low),
        })
    if not candidates:
        raise ArtifactError(f"tile has no edge-adjacent neighbor: {anchor_tile_id}")
    ranked = sorted(
        candidates,
        key=lambda item: (-item["shared_edge_length_m"], item["tile_id"]),
    )
    return {
        "policy": "maximum_shared_core_edge_then_tile_id_v1",
        "uses_heldout_evidence": False,
        "anchor_tile_id": anchor_tile_id,
        "selected_tile_id": ranked[0]["tile_id"],
        "selected_segment_uv": [
            ranked[0]["axis"],
            ranked[0]["coordinate_m"],
            *ranked[0]["span_m"],
        ],
        "candidates": ranked,
    }


def _internal_seam_segments(plan: Mapping[str, Any]) -> list[tuple[int, float, float, float]]:
    """Return unique internal UV core edges as axis/coordinate/span tuples."""
    scene = np.asarray(plan["partition"]["scene_bounds_uv_m"], dtype=np.float64)
    segments = set()
    for tile in plan["tiles"]:
        core = np.asarray(tile["core_bounds_uv_m"], dtype=np.float64)
        for axis in (0, 1):
            other = 1 - axis
            for side in (0, 1):
                coordinate = float(core[side, axis])
                if math.isclose(
                    coordinate,
                    float(scene[side, axis]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    continue
                segments.add((
                    axis,
                    coordinate,
                    float(core[0, other]),
                    float(core[1, other]),
                ))
    if not segments:
        raise ArtifactError("multi-tile scene has no internal ownership seam")
    return sorted(segments)


def _build_seam_masks(
    reader: SegmentReader,
    cfg: Any,
    plan: Mapping[str, Any],
    validation_ids: Sequence[int],
    *,
    band_m: float = 1.0,
    minimum_pixels_per_frame: int = 256,
    segments: Sequence[Sequence[float]] | None = None,
    allow_failed_georeferencing_for_render: bool = False,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Project held-out metric depth and retain pixels near internal seams."""
    if not math.isfinite(band_m) or band_m <= 0:
        raise ValueError("seam band must be finite and positive")
    viewmats, _ = load_pose_artifact(
        reader.root,
        cfg,
        allow_failed_georeferencing_for_render=bool(
            allow_failed_georeferencing_for_render
        ),
    )
    camera = reader.calibration["cameras"]["left"]
    intr = np.asarray(camera["K"], dtype=np.float64)
    origin = np.asarray(
        plan["coordinate_frame"]["partition_origin_enu_m"], dtype=np.float64
    )
    basis = np.asarray(
        plan["coordinate_frame"]["R_enu_from_partition"], dtype=np.float64
    )
    if segments is None:
        selected_segments = _internal_seam_segments(plan)
        policy = "projected_metric_depth_internal_core_edges_v1"
    else:
        selected_segments = []
        for item in segments:
            if len(item) != 4:
                raise ValueError("seam segment must be axis/coordinate/low/high")
            axis, coordinate, low, high = item
            if (
                isinstance(axis, bool)
                or int(axis) not in (0, 1)
                or float(axis) != int(axis)
                or not all(math.isfinite(float(value)) for value in (coordinate, low, high))
                or float(high) <= float(low)
            ):
                raise ValueError("seam segment geometry is invalid")
            selected_segments.append(
                (int(axis), float(coordinate), float(low), float(high))
            )
        if not selected_segments:
            raise ValueError("at least one seam segment is required")
        policy = "projected_metric_depth_selected_core_edges_v1"
    masks: dict[int, np.ndarray] = {}
    records = []
    for frame_id in validation_ids:
        depth_path = reader.root / str(reader.frames["depth_path"][int(frame_id)])
        with np.load(depth_path, allow_pickle=False) as archive:
            depth = np.asarray(archive["depth"], dtype=np.float64)
            valid = np.asarray(archive["valid"], dtype=bool)
        valid &= np.isfinite(depth) & (depth > 0)
        rows, columns = np.nonzero(valid)
        if not len(rows):
            continue
        z = depth[rows, columns]
        camera_points = np.column_stack((
            (columns - intr[0, 2]) * z / intr[0, 0],
            (rows - intr[1, 2]) * z / intr[1, 1],
            z,
        ))
        c2w = np.linalg.inv(np.asarray(viewmats[int(frame_id)], dtype=np.float64))
        world = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
        uv = ((world - origin) @ basis)[:, :2]
        minimum_sq = np.full(len(uv), np.inf, dtype=np.float64)
        for axis, coordinate, span_low, span_high in selected_segments:
            other = 1 - axis
            across = uv[:, axis] - coordinate
            along = uv[:, other] - np.clip(
                uv[:, other], span_low, span_high
            )
            minimum_sq = np.minimum(minimum_sq, across * across + along * along)
        selected = minimum_sq <= float(band_m) ** 2
        count = int(selected.sum())
        if count < int(minimum_pixels_per_frame):
            continue
        mask = np.zeros(depth.shape, dtype=bool)
        mask[rows[selected], columns[selected]] = True
        masks[int(frame_id)] = mask
        records.append({
            "frame_id": int(frame_id),
            "metric_depth_pixels_in_band": count,
        })
    total = int(sum(item["metric_depth_pixels_in_band"] for item in records))
    if len(records) < 2 or total < 2 * int(minimum_pixels_per_frame):
        raise ArtifactError(
            "held-out source split has insufficient metric-depth support at seams"
        )
    return masks, {
        "policy": policy,
        "band_m": float(band_m),
        "minimum_pixels_per_frame": int(minimum_pixels_per_frame),
        "n_validation_frames": len(records),
        "metric_depth_pixels_in_band": total,
        "internal_segments_uv": [list(item) for item in selected_segments],
        "frames": records,
    }


def _evaluate_combined(
    params: Mapping[str, Any],
    reader: SegmentReader,
    cfg: Any,
    staging: Path,
    validation_ids: Sequence[int],
    *,
    device: str,
    maximum_visible_gaussians: int,
    maximum_render_depth_m: float | None = None,
    frustum_sigma: float = 3.0,
    context_masks: Mapping[int, np.ndarray] | None = None,
    context_mask_dilate_px: int | None = None,
    allow_failed_georeferencing_for_render: bool = False,
) -> dict[str, Any]:
    """Render once per held-out view with a conservative CPU frustum cull.

    The full uniquely-owned scene remains on CPU.  A sphere with radius
    ``frustum_sigma * max(scale_xyz)`` bounds each oriented Gaussian; a model
    is transferred to CUDA only when that sphere intersects the current image
    frustum and bounded evaluation depth.  This keeps evaluation independent
    of tile occurrence counts without requiring the whole field in VRAM.
    """
    import torch
    from rtk_splat.backends.gsplat import evaluate

    viewmats, _ = load_pose_artifact(
        reader.root,
        cfg,
        allow_failed_georeferencing_for_render=bool(
            allow_failed_georeferencing_for_render
        ),
    )
    expected_fingerprint = pose_fingerprint(viewmats)
    camera = reader.calibration["cameras"]["left"]
    intr = np.asarray(camera["K"], dtype=np.float64)
    width, height = int(camera["width"]), int(camera["height"])
    k_mat = torch.tensor(
        [[intr[0, 0], 0, intr[0, 2]],
         [0, intr[1, 1], intr[1, 2]],
         [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    c2ws = torch.linalg.inv(torch.tensor(
        viewmats, dtype=torch.float32, device=device
    ))
    means = params["means"].detach().cpu().numpy().astype(np.float64, copy=False)
    radii = (
        torch.exp(params["scales"].detach().cpu())
        .amax(dim=1)
        .numpy()
        .astype(np.float64, copy=False)
        * float(frustum_sigma)
    )
    if (
        (
            maximum_render_depth_m is not None
            and (
                not math.isfinite(maximum_render_depth_m)
                or maximum_render_depth_m <= 0
            )
        )
        or not math.isfinite(frustum_sigma)
        or frustum_sigma < 3.0
    ):
        raise ValueError("invalid conservative scene-evaluation frustum policy")
    planes = np.asarray([
        [intr[0, 0], 0.0, intr[0, 2]],
        [-intr[0, 0], 0.0, width - intr[0, 2]],
        [0.0, intr[1, 1], intr[1, 2]],
        [0.0, -intr[1, 1], height - intr[1, 2]],
    ], dtype=np.float64)
    plane_norms = np.linalg.norm(planes, axis=1)
    per_view = []
    records = []
    metric_names = (
        "psnr", "psnr_masked", "psnr_near", "psnr_masked_cc",
        "lpips", "lpips_cc", "ssim",
    )
    for frame_id in validation_ids:
        viewmat = np.asarray(viewmats[int(frame_id)], dtype=np.float64)
        camera = means @ viewmat[:3, :3].T + viewmat[:3, 3]
        signed = camera @ planes.T
        visible = (
            (camera[:, 2] + radii > 1e-3)
            & np.all(signed >= -(radii[:, None] * plane_norms), axis=1)
        )
        if maximum_render_depth_m is not None:
            visible &= camera[:, 2] - radii < maximum_render_depth_m
        selected = int(visible.sum())
        if selected <= 0:
            raise ArtifactError(f"no scene Gaussians see validation frame {frame_id}")
        if selected > maximum_visible_gaussians:
            raise ArtifactError(
                f"validation frame {frame_id} needs {selected:,} Gaussians, "
                f"above the GPU evaluation cap {maximum_visible_gaussians:,}"
            )
        keep = torch.from_numpy(visible)
        device_params = {
            name: tensor[keep].to(device) for name, tensor in params.items()
        }
        observation = evaluate(
            device_params,
            c2ws,
            k_mat,
            width,
            height,
            reader.root,
            staging,
            [int(frame_id)],
            cfg,
            device,
            int(frame_id),
            False,
            frames=reader.frames,
            context_masks=context_masks,
            context_mask_dilate_px=context_mask_dilate_px,
        )
        per_view.append(observation)
        records.append({
            "frame_id": int(frame_id),
            "selected_gaussians": selected,
            "selected_fraction": selected / len(means),
            **{
                key: float(observation[key])
                for key in metric_names
            },
        })
        del device_params
    result = {
        key: float(np.mean([float(item[key]) for item in per_view]))
        for key in metric_names
    }
    result["n_eval"] = len(validation_ids)
    for key in _METRICS:
        if not math.isfinite(float(result.get(key, float("nan")))):
            raise ArtifactError(f"combined scene metric {key} is non-finite")
    result["eval_ids"] = [int(value) for value in validation_ids]
    result["pose_fingerprint"] = expected_fingerprint
    result["frustum_culling"] = {
        "policy": "camera_frustum_sphere_bound_v1",
        "gaussian_radius_sigma": float(frustum_sigma),
        "maximum_render_depth_m": (
            None
            if maximum_render_depth_m is None
            else float(maximum_render_depth_m)
        ),
        "maximum_visible_gaussians": int(maximum_visible_gaussians),
        "full_core_owned_gaussians_on_cpu": len(means),
        "minimum_selected_gaussians": min(
            item["selected_gaussians"] for item in records
        ),
        "maximum_selected_gaussians": max(
            item["selected_gaussians"] for item in records
        ),
        "views": records,
    }
    return result


def publish_tiled_scene(
    *,
    segment: str | Path,
    cfg: Any,
    tile_plan_root: str | Path,
    pose_root: str | Path,
    tile_runs: Mapping[str, str | Path],
    reference_run: str | Path,
    reference_hashes: Mapping[str, str],
    output_root: str | Path,
    scene_name: str,
    opacity_threshold: float = 0.05,
    maximum_psnr_loss_db: float = 0.30,
    maximum_ssim_loss: float = 0.015,
    maximum_lpips_cc_increase: float = 0.030,
    maximum_combined_gaussians: int | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate, export, seal, and atomically publish a tiled GS scene."""
    name = _safe_name(scene_name)
    threshold = float(opacity_threshold)
    maximum_loss = float(maximum_psnr_loss_db)
    maximum_ssim_loss = float(maximum_ssim_loss)
    maximum_lpips_cc_increase = float(maximum_lpips_cc_increase)
    if not math.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError("opacity_threshold must be finite and in [0, 1)")
    if not math.isfinite(maximum_loss) or maximum_loss < 0:
        raise ValueError("maximum_psnr_loss_db must be finite and non-negative")
    if not math.isfinite(maximum_ssim_loss) or maximum_ssim_loss < 0:
        raise ValueError("maximum_ssim_loss must be finite and non-negative")
    if (
        not math.isfinite(maximum_lpips_cc_increase)
        or maximum_lpips_cc_increase < 0
    ):
        raise ValueError(
            "maximum_lpips_cc_increase must be finite and non-negative"
        )
    if maximum_combined_gaussians is None:
        configured_capacity = getattr(getattr(cfg, "train", None), "max_gaussians", None)
        if isinstance(configured_capacity, bool) or not isinstance(
            configured_capacity, (int, np.integer)
        ):
            raise ValueError(
                "scene publication needs an explicit maximum_combined_gaussians "
                "when train.max_gaussians is unresolved"
            )
        maximum_combined_gaussians = int(configured_capacity)
    if maximum_combined_gaussians <= 0:
        raise ValueError("maximum_combined_gaussians must be positive")

    destination = Path(output_root).expanduser() / "scene_artifacts" / name
    if destination.exists():
        raise FileExistsError(f"refusing to modify existing scene: {destination}")

    reader = SegmentReader(segment).validate()
    plan_root = Path(tile_plan_root).expanduser().resolve()
    pose_root = Path(pose_root).expanduser().resolve()
    plan = verify_tile_plan(
        plan_root,
        segment=reader.root,
        pose_root=pose_root,
        rehash_sources=True,
    )
    inventory = _json(plan_root / "source_inventory.json")
    plan_provenance = _json(plan_root / "provenance.json")
    georeferencing = plan_provenance.get("georeferencing")
    if not isinstance(georeferencing, dict):
        raise ArtifactError("TilePlan has no georeferencing evidence")
    tile_ids = [str(tile["tile_id"]) for tile in plan["tiles"]]
    if set(tile_runs) != set(tile_ids):
        raise ArtifactError("tile run mapping must name every planned tile exactly once")

    validation_ids = [int(value) for value in reader.manifest["val"]]
    reference_metrics, reference_evidence = _reference_run_evidence(
        Path(reference_run), reference_hashes, plan, inventory, validation_ids
    )
    run_records = []
    completed = []
    for index, tile in enumerate(plan["tiles"]):
        tile_id = tile["tile_id"]
        run = Path(tile_runs[tile_id]).expanduser().resolve()
        expected_binding = _expected_tile_binding(
            plan, plan_root, inventory, tile, index
        )
        provenance, params, params_hash, artifacts = _completed_tile_run(
            run, expected_binding, georeferencing
        )
        training_identity = _training_identity(provenance)
        if training_identity != reference_evidence["training_identity"]:
            raise ArtifactError(
                "tile and monolithic reference do not share training code/settings"
            )
        run_records.append({
            "tile_id": tile_id,
            "run": str(run),
            "run_provenance_sha256": sha256_file(run / "run_provenance.json"),
            "best_checkpoint_sha256": sha256_file(run / "best_checkpoint.json"),
            "params_sha256": params_hash,
            "initial_cloud_sha256": artifacts["initial_cloud_sha256"],
            "source_splat_file": artifacts["splat_file"],
            "source_splat_sha256": artifacts["splat_sha256"],
            "effective_training_config_sha256": provenance.get(
                "effective_training_config_sha256"
            ),
            "training_identity": training_identity,
            "tile_plan_binding": expected_binding,
            "selected_model": provenance["model_selection"],
        })
        completed.append((tile, params, params_hash))

    params, ownership_records = _concatenate_core_params(plan, completed)
    total = int(len(params["means"]))
    viewmats, _ = load_pose_artifact(reader.root, cfg)
    if pose_fingerprint(viewmats) != plan["source_binding"]["pose_fingerprint"]:
        raise ArtifactError("configured evaluation pose disagrees with the TilePlan")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{name}.writing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        (staging / "renders").mkdir()
        candidate = _evaluate_combined(
            params,
            reader,
            cfg,
            staging,
            validation_ids,
            device=device,
            maximum_visible_gaussians=int(maximum_combined_gaussians),
            maximum_render_depth_m=None,
        )
        seam_masks, seam_evidence = _build_seam_masks(
            reader, cfg, plan, validation_ids, band_m=1.0
        )
        seam_ids = sorted(seam_masks)
        seam_candidate_root = staging / "seam_candidate"
        seam_reference_root = staging / "seam_reference"
        (seam_candidate_root / "renders").mkdir(parents=True)
        (seam_reference_root / "renders").mkdir(parents=True)
        candidate_seam = _evaluate_combined(
            params,
            reader,
            cfg,
            seam_candidate_root,
            seam_ids,
            device=device,
            maximum_visible_gaussians=int(maximum_combined_gaussians),
            maximum_render_depth_m=None,
            context_masks=seam_masks,
            context_mask_dilate_px=1,
        )
        from rtk_splat.backends.gsplat import _load_checkpoint_gaussians
        reference_params = _load_checkpoint_gaussians(
            Path(reference_run) / "params.pt", "cpu"
        )
        reference_seam = _evaluate_combined(
            reference_params,
            reader,
            cfg,
            seam_reference_root,
            seam_ids,
            device=device,
            maximum_visible_gaussians=int(maximum_combined_gaussians),
            maximum_render_depth_m=None,
            context_masks=seam_masks,
            context_mask_dilate_px=1,
        )
        if candidate["pose_fingerprint"] != plan["source_binding"][
            "pose_fingerprint"
        ]:
            raise ArtifactError("combined evaluation used a different pose artifact")
        losses = {
            key: float(reference_metrics[key] - candidate[key])
            for key in ("psnr_masked", "psnr_masked_cc")
        }
        regressions = {
            **{f"{key}_loss_db": value for key, value in losses.items()},
            "ssim_loss": float(
                reference_metrics["ssim"] - candidate["ssim"]
            ),
            "lpips_cc_increase": float(
                candidate["lpips_cc"] - reference_metrics["lpips_cc"]
            ),
        }
        seam_regressions = {
            "psnr_masked_loss_db": float(
                reference_seam["psnr_masked"]
                - candidate_seam["psnr_masked"]
            ),
            "psnr_masked_cc_loss_db": float(
                reference_seam["psnr_masked_cc"]
                - candidate_seam["psnr_masked_cc"]
            ),
            "lpips_cc_increase": float(
                candidate_seam["lpips_cc"] - reference_seam["lpips_cc"]
            ),
        }
        checks = {
            "all_tile_runs_complete_and_bound": True,
            "same_training_implementation_as_monolithic_control": all(
                item["training_identity"]["training_implementation_sha256"]
                == reference_evidence["training_identity"][
                    "training_implementation_sha256"
                ]
                for item in run_records
            ),
            "same_scientific_training_config_as_monolithic_control": all(
                item["training_identity"][
                    "comparable_training_config_sha256"
                ]
                == reference_evidence["training_identity"][
                    "comparable_training_config_sha256"
                ]
                for item in run_records
            ),
            "exact_half_open_core_ownership": True,
            "combined_evaluation_uses_source_validation_once": (
                candidate["eval_ids"] == validation_ids
                and candidate.get("n_eval") == len(validation_ids)
            ),
            "masked_psnr_loss_within_gate": (
                losses["psnr_masked"] <= maximum_loss
            ),
            "corrected_masked_psnr_loss_within_gate": (
                losses["psnr_masked_cc"] <= maximum_loss
            ),
            "ssim_loss_within_gate": (
                regressions["ssim_loss"] <= maximum_ssim_loss
            ),
            "lpips_cc_increase_within_gate": (
                regressions["lpips_cc_increase"]
                <= maximum_lpips_cc_increase
            ),
            "seam_metric_depth_evidence_is_sufficient": (
                seam_evidence["n_validation_frames"] >= 2
                and seam_evidence["metric_depth_pixels_in_band"] >= 512
            ),
            "seam_masked_psnr_loss_within_gate": (
                seam_regressions["psnr_masked_loss_db"] <= maximum_loss
            ),
            "seam_corrected_masked_psnr_loss_within_gate": (
                seam_regressions["psnr_masked_cc_loss_db"] <= maximum_loss
            ),
            "seam_lpips_cc_increase_within_gate": (
                seam_regressions["lpips_cc_increase"]
                <= maximum_lpips_cc_increase
            ),
        }
        quality_passed = all(checks.values())
        splat_name = (
            "scene.ply"
            if quality_passed
            and plan.get("metric_georeferencing_claim_eligible") is True
            else (
                "scene.PROVISIONAL.ply"
                if quality_passed
                else "scene.QUALITY_FAILED.ply"
            )
        )
        from rtk_splat.backends.gsplat import export_splat_tensors

        exported_total, retained = export_splat_tensors(
            params,
            staging / splat_name,
            opacity_threshold=threshold,
            crop_bounds=None,
        )
        if exported_total != total:
            raise ArtifactError("Gaussian exporter changed the combined core count")
        splat_record = _record(staging / splat_name)
        metrics_record = {
            "schema_version": SCHEMA_VERSION,
            "evaluation": {
                "aggregation": (
                    "single_render_of_concatenated_core_owned_gaussians"
                ),
                "validation_split": "source_manifest_val",
                "n_unique_validation_frames": len(validation_ids),
                "eval_ids": validation_ids,
            },
            "reference": reference_metrics,
            "candidate": {key: candidate[key] for key in candidate},
            "loss_db": losses,
            "regressions": regressions,
            "seam": {
                "evaluation": (
                    "projected_metric_depth_within_1m_of_internal_core_edges"
                ),
                "evidence": seam_evidence,
                "reference": {
                    key: reference_seam[key] for key in _METRICS
                },
                "candidate": {
                    key: candidate_seam[key] for key in _METRICS
                },
                "regressions": seam_regressions,
            },
            "maximum_psnr_loss_db": maximum_loss,
            "maximum_ssim_loss": maximum_ssim_loss,
            "maximum_lpips_cc_increase": maximum_lpips_cc_increase,
        }
        quality_record = {
            "schema_version": SCHEMA_VERSION,
            "passed": quality_passed,
            "checks": checks,
            "interpretation": (
                "accepted controlled monolithic-versus-tiled rendering A/B"
                if quality_passed
                else "diagnostic result; rendering acceptance gate failed"
            ),
        }
        scene_record = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "rtk_splat_tiled_scene",
            "name": name,
            "quality_passed": quality_passed,
            "provisional": not bool(
                plan.get("metric_georeferencing_claim_eligible")
            ),
            "metric_georeferencing_claim_eligible": bool(
                plan.get("metric_georeferencing_claim_eligible")
            ),
            "coordinate_frame": plan["coordinate_frame"],
            "partition": plan["partition"],
            "tile_plan": {
                "name": plan["name"],
                "manifest_sha256": sha256_file(plan_root / "manifest.json"),
                "tile_plan_sha256": sha256_file(plan_root / "tile_plan.json"),
                "source_inventory_sha256": canonical_hash(inventory),
            },
            "ownership": {
                "rule": plan["partition"]["boundary_rule"],
                "source_gaussians": int(sum(
                    item["source_gaussians"] for item in ownership_records
                )),
                "core_owned_gaussians": total,
                "tiles": ownership_records,
            },
            "opacity_threshold": threshold,
            "retained_gaussians": retained,
            "splat_file": splat_name,
            "splat_sha256": splat_record["sha256"],
            "splat_size_bytes": splat_record["size_bytes"],
        }
        provenance_record = {
            "schema_version": SCHEMA_VERSION,
            "tile_plan": scene_record["tile_plan"],
            "tile_plan_locator": str(plan_root),
            "source_segment": str(reader.root.resolve()),
            "source_contract_files": inventory["segment"]["contract_files"],
            "pose": inventory["pose"],
            "georeferencing": copy.deepcopy(georeferencing),
            "reference": reference_evidence,
            "tile_runs": run_records,
            "package": collect_package_state(),
            "device": str(device),
        }
        _write_json(staging / "metrics.json", metrics_record)
        _write_json(staging / "quality.json", quality_record)
        _write_json(staging / "scene.json", scene_record)
        _write_json(staging / "provenance.json", provenance_record)

        # Close the most important source races before terminal publication.
        if sha256_file(plan_root / "manifest.json") != (
            scene_record["tile_plan"]["manifest_sha256"]
        ):
            raise ArtifactError("TilePlan changed while scene was being published")
        reference_root = Path(reference_run).expanduser().resolve()
        for filename, record in reference_evidence["files"].items():
            if _record(reference_root / filename) != record:
                raise ArtifactError(
                    "reference run changed while scene was being published"
                )
        for item in run_records:
            run = Path(item["run"])
            if (
                sha256_file(run / "params.pt") != item["params_sha256"]
                or sha256_file(run / "run_provenance.json")
                != item["run_provenance_sha256"]
            ):
                raise ArtifactError("tile run changed while scene was being published")
        _write_json(staging / "manifest.json", _manifest(staging))
        verify_tiled_scene(staging)
        publish_directory_noreplace(staging, destination)
        verify_tiled_scene(destination)
        return {
            "scene": str(destination),
            "splat": str(destination / splat_name),
            "quality_passed": quality_passed,
            "provisional": scene_record["provisional"],
            "n_validation_frames": len(validation_ids),
            "core_owned_gaussians": total,
            "retained_gaussians": retained,
            "masked_psnr_loss_db": losses["psnr_masked"],
            "corrected_masked_psnr_loss_db": losses["psnr_masked_cc"],
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def publish_production_tiled_scene(
    *,
    segment: str | Path,
    cfg: Any,
    tile_plan_root: str | Path,
    pose_root: str | Path,
    tile_runs: Mapping[str, str | Path],
    output_root: str | Path,
    scene_name: str,
    opacity_threshold: float = 0.05,
    maximum_combined_gaussians: int | None = None,
    device: str = "cuda",
    allow_nonproduction_georeferencing_for_diagnostic: bool = False,
) -> dict[str, Any]:
    """Publish an absolute-evidence tiled scene without a monolithic control.

    This is the deployment path.  It verifies all planned tile runs against the
    same immutable segment, pose, georeferencing declaration, TilePlan, trainer,
    and scientific configuration.  It reports whole-scene and seam metrics in
    absolute terms only.  Non-production georeferencing is rejected unless the
    caller explicitly requests a diagnostic-only artifact.
    """
    name = _safe_name(scene_name)
    threshold = float(opacity_threshold)
    if not math.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError("opacity_threshold must be finite and in [0, 1)")
    if maximum_combined_gaussians is None:
        configured_capacity = getattr(
            getattr(cfg, "train", None), "max_gaussians", None
        )
        if isinstance(configured_capacity, bool) or not isinstance(
            configured_capacity, (int, np.integer)
        ):
            raise ValueError(
                "scene publication needs an explicit maximum_combined_gaussians "
                "when train.max_gaussians is unresolved"
            )
        maximum_combined_gaussians = int(configured_capacity)
    if maximum_combined_gaussians <= 0:
        raise ValueError("maximum_combined_gaussians must be positive")

    destination = Path(output_root).expanduser() / "scene_artifacts" / name
    if destination.exists():
        raise FileExistsError(f"refusing to modify existing scene: {destination}")

    reader = SegmentReader(segment).validate()
    plan_root = Path(tile_plan_root).expanduser().resolve()
    pose_root = Path(pose_root).expanduser().resolve()
    plan = verify_tile_plan(
        plan_root,
        segment=reader.root,
        pose_root=pose_root,
        rehash_sources=True,
    )
    inventory = _json(plan_root / "source_inventory.json")
    plan_provenance = _json(plan_root / "provenance.json")
    georeferencing = plan_provenance.get("georeferencing")
    if not isinstance(georeferencing, dict):
        raise ArtifactError("TilePlan has no georeferencing evidence")
    production_georeferencing = bool(
        plan.get("metric_georeferencing_claim_eligible") is True
        and georeferencing.get("artifact_class") == "production"
        and georeferencing.get("georeferencing_status") == "PASSED"
        and georeferencing.get("metric_georeferencing_claim_eligible") is True
    )
    diagnostic_nonproduction = not production_georeferencing
    if diagnostic_nonproduction and not bool(
        allow_nonproduction_georeferencing_for_diagnostic
    ):
        raise ArtifactError(
            "production scene publication requires a PASSED production pose; "
            "use the explicit diagnostic-scene option only for a non-claim artifact"
        )

    tiles = plan.get("tiles")
    if not isinstance(tiles, list) or len(tiles) < 2:
        raise ArtifactError("production tiled-scene publication needs at least two tiles")
    tile_ids = [str(tile["tile_id"]) for tile in tiles]
    if set(tile_runs) != set(tile_ids) or len(tile_runs) != len(tile_ids):
        raise ArtifactError("tile run mapping must name every planned tile exactly once")
    validation_ids = [int(value) for value in reader.manifest["val"]]
    if len(validation_ids) < 2 or validation_ids != sorted(set(validation_ids)):
        raise ArtifactError("source validation split is insufficient or non-canonical")

    run_records = []
    completed = []
    common_training_identity: dict[str, str] | None = None
    for index, tile in enumerate(tiles):
        tile_id = str(tile["tile_id"])
        run = Path(tile_runs[tile_id]).expanduser().resolve()
        expected_binding = _expected_tile_binding(
            plan, plan_root, inventory, tile, index
        )
        provenance, params, params_hash, artifacts = _completed_tile_run(
            run,
            expected_binding,
            georeferencing,
            allow_nonproduction_georeferencing_for_diagnostic=(
                diagnostic_nonproduction
                and allow_nonproduction_georeferencing_for_diagnostic
            ),
        )
        _verify_training_source_binding(provenance, inventory)
        training_identity = _training_identity(provenance)
        if common_training_identity is None:
            common_training_identity = training_identity
        elif training_identity != common_training_identity:
            raise ArtifactError(
                "production tiles do not share one trainer implementation/configuration"
            )
        run_records.append({
            "tile_id": tile_id,
            "run": str(run),
            "run_provenance_sha256": sha256_file(run / "run_provenance.json"),
            "best_checkpoint_sha256": sha256_file(run / "best_checkpoint.json"),
            "params_sha256": params_hash,
            "initial_cloud_sha256": artifacts["initial_cloud_sha256"],
            "source_splat_file": artifacts["splat_file"],
            "source_splat_sha256": artifacts["splat_sha256"],
            "effective_training_config_sha256": provenance.get(
                "effective_training_config_sha256"
            ),
            "training_identity": training_identity,
            "tile_plan_binding": expected_binding,
            "selected_model": provenance["model_selection"],
        })
        completed.append((tile, params, params_hash))

    params, ownership_records = _concatenate_core_params(plan, completed)
    total = int(len(params["means"]))
    source_total = int(sum(
        item["source_gaussians"] for item in ownership_records
    ))
    outside_total = int(sum(
        item["outside_or_other_core_gaussians"] for item in ownership_records
    ))
    if (
        total <= 0
        or source_total != total + outside_total
        or [item["tile_id"] for item in ownership_records] != tile_ids
        or any(item["core_owned_gaussians"] <= 0 for item in ownership_records)
    ):
        raise ArtifactError("final Gaussian ownership accounting is incomplete")
    diagnostic_render_permission = bool(
        diagnostic_nonproduction
        and allow_nonproduction_georeferencing_for_diagnostic
    )
    viewmats, _ = load_pose_artifact(
        reader.root,
        cfg,
        allow_failed_georeferencing_for_render=diagnostic_render_permission,
    )
    if pose_fingerprint(viewmats) != plan["source_binding"]["pose_fingerprint"]:
        raise ArtifactError("configured evaluation pose disagrees with the TilePlan")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{name}.writing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        (staging / "renders").mkdir()
        absolute = _evaluate_combined(
            params,
            reader,
            cfg,
            staging,
            validation_ids,
            device=device,
            maximum_visible_gaussians=int(maximum_combined_gaussians),
            maximum_render_depth_m=None,
            allow_failed_georeferencing_for_render=diagnostic_render_permission,
        )
        seam_masks, seam_evidence = _build_seam_masks(
            reader,
            cfg,
            plan,
            validation_ids,
            band_m=1.0,
            allow_failed_georeferencing_for_render=diagnostic_render_permission,
        )
        seam_ids = sorted(seam_masks)
        seam_root = staging / "seam_absolute"
        (seam_root / "renders").mkdir(parents=True)
        seam_absolute = _evaluate_combined(
            params,
            reader,
            cfg,
            seam_root,
            seam_ids,
            device=device,
            maximum_visible_gaussians=int(maximum_combined_gaussians),
            maximum_render_depth_m=None,
            context_masks=seam_masks,
            context_mask_dilate_px=1,
            allow_failed_georeferencing_for_render=diagnostic_render_permission,
        )
        expected_fingerprint = plan["source_binding"]["pose_fingerprint"]
        if (
            absolute["pose_fingerprint"] != expected_fingerprint
            or seam_absolute["pose_fingerprint"] != expected_fingerprint
        ):
            raise ArtifactError("scene evaluation used a different pose artifact")
        absolute_metrics_finite = all(
            math.isfinite(float(absolute[key])) for key in _METRICS
        )
        seam_metrics_finite = all(
            math.isfinite(float(seam_absolute[key])) for key in _METRICS
        )
        checks = {
            "production_georeferencing_or_explicit_diagnostic": (
                production_georeferencing
                or bool(allow_nonproduction_georeferencing_for_diagnostic)
            ),
            "tile_run_set_matches_plan": (
                [item["tile_id"] for item in run_records] == tile_ids
            ),
            "all_tile_runs_complete_and_bound": True,
            "all_tiles_share_training_implementation": all(
                item["training_identity"]["training_implementation_sha256"]
                == common_training_identity["training_implementation_sha256"]
                for item in run_records
            ),
            "all_tiles_share_scientific_training_config": all(
                item["training_identity"]["comparable_training_config_sha256"]
                == common_training_identity["comparable_training_config_sha256"]
                for item in run_records
            ),
            "exact_half_open_core_ownership": True,
            "every_tile_contributes_core_gaussians": all(
                item["core_owned_gaussians"] > 0 for item in ownership_records
            ),
            "ownership_accounting_is_exact": (
                source_total == total + outside_total
            ),
            "combined_evaluation_uses_source_validation_once": (
                absolute["eval_ids"] == validation_ids
                and absolute.get("n_eval") == len(validation_ids)
            ),
            "whole_scene_absolute_metrics_are_finite": absolute_metrics_finite,
            "seam_metric_depth_evidence_is_sufficient": (
                seam_evidence["n_validation_frames"] >= 2
                and seam_evidence["metric_depth_pixels_in_band"] >= 512
            ),
            "seam_absolute_metrics_are_finite": seam_metrics_finite,
        }
        quality_passed = all(checks.values())
        if not quality_passed:
            raise ArtifactError("production scene failed integrity/completeness checks")

        splat_name = (
            "scene.ply"
            if production_georeferencing
            else "scene.DIAGNOSTIC_ONLY.ply"
        )
        from rtk_splat.backends.gsplat import export_splat_tensors

        exported_total, retained = export_splat_tensors(
            params,
            staging / splat_name,
            opacity_threshold=threshold,
            crop_bounds=None,
        )
        if exported_total != total:
            raise ArtifactError("Gaussian exporter changed the combined core count")
        splat_record = _record(staging / splat_name)
        completeness = {
            "planned_tile_ids": tile_ids,
            "completed_tile_ids": [item["tile_id"] for item in run_records],
            "n_planned_tiles": len(tile_ids),
            "n_completed_tiles": len(run_records),
            "source_validation_frame_ids": validation_ids,
            "n_source_validation_frames": len(validation_ids),
            "n_evaluated_validation_frames": int(absolute["n_eval"]),
            "source_gaussians": source_total,
            "core_owned_gaussians": total,
            "outside_or_other_core_gaussians": outside_total,
            "all_tiles_contributed_core_gaussians": True,
        }
        metrics_record = {
            "schema_version": SCHEMA_VERSION,
            "publication_mode": "production",
            "evaluation": {
                "aggregation": (
                    "single_render_of_concatenated_core_owned_gaussians"
                ),
                "comparison": "absolute_only_no_monolithic_reference",
                "validation_split": "source_manifest_val",
                "n_unique_validation_frames": len(validation_ids),
                "eval_ids": validation_ids,
            },
            "absolute": {key: absolute[key] for key in absolute},
            "seam": {
                "evaluation": (
                    "projected_metric_depth_within_1m_of_internal_core_edges"
                ),
                "comparison": "absolute_only_no_reference",
                "evidence": seam_evidence,
                "absolute": {
                    key: seam_absolute[key] for key in seam_absolute
                },
            },
            "completeness": completeness,
        }
        quality_record = {
            "schema_version": SCHEMA_VERSION,
            "publication_mode": "production",
            "passed": quality_passed,
            "checks": checks,
            "interpretation": (
                "accepted production tiled scene with absolute held-out evidence"
                if production_georeferencing
                else "accepted diagnostic-only tiled render; no metric georeferencing claim"
            ),
        }
        scene_record = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "rtk_splat_tiled_scene",
            "publication_mode": "production",
            "name": name,
            "quality_passed": quality_passed,
            "provisional": diagnostic_nonproduction,
            "metric_georeferencing_claim_eligible": production_georeferencing,
            "coordinate_frame": plan["coordinate_frame"],
            "partition": plan["partition"],
            "tile_plan": {
                "name": plan["name"],
                "manifest_sha256": sha256_file(plan_root / "manifest.json"),
                "tile_plan_sha256": sha256_file(plan_root / "tile_plan.json"),
                "source_inventory_sha256": canonical_hash(inventory),
            },
            "ownership": {
                "rule": plan["partition"]["boundary_rule"],
                "source_gaussians": source_total,
                "core_owned_gaussians": total,
                "tiles": ownership_records,
            },
            "opacity_threshold": threshold,
            "retained_gaussians": retained,
            "splat_file": splat_name,
            "splat_sha256": splat_record["sha256"],
            "splat_size_bytes": splat_record["size_bytes"],
        }
        provenance_record = {
            "schema_version": SCHEMA_VERSION,
            "publication_mode": "production",
            "diagnostic_nonproduction_georeferencing": diagnostic_nonproduction,
            "tile_plan": scene_record["tile_plan"],
            "tile_plan_locator": str(plan_root),
            "source_segment": str(reader.root.resolve()),
            "source_contract_files": inventory["segment"]["contract_files"],
            "pose": inventory["pose"],
            "georeferencing": copy.deepcopy(georeferencing),
            "common_training_identity": common_training_identity,
            "tile_runs": run_records,
            "package": collect_package_state(),
            "device": str(device),
        }
        _write_json(staging / "metrics.json", metrics_record)
        _write_json(staging / "quality.json", quality_record)
        _write_json(staging / "scene.json", scene_record)
        _write_json(staging / "provenance.json", provenance_record)

        # Re-hash every sealed source after the expensive render/export, then
        # close races on all completed-run terminal files before publication.
        verify_tile_plan(
            plan_root,
            segment=reader.root,
            pose_root=pose_root,
            rehash_sources=True,
        )
        if sha256_file(plan_root / "manifest.json") != (
            scene_record["tile_plan"]["manifest_sha256"]
        ):
            raise ArtifactError("TilePlan changed while scene was being published")
        for item in run_records:
            run = Path(item["run"])
            if (
                sha256_file(run / "params.pt") != item["params_sha256"]
                or sha256_file(run / "run_provenance.json")
                != item["run_provenance_sha256"]
                or sha256_file(run / "best_checkpoint.json")
                != item["best_checkpoint_sha256"]
                or sha256_file(run / item["source_splat_file"])
                != item["source_splat_sha256"]
            ):
                raise ArtifactError("tile run changed while scene was being published")
        _write_json(staging / "manifest.json", _manifest(staging))
        verify_tiled_scene(staging)
        publish_directory_noreplace(staging, destination)
        verify_tiled_scene(destination)
        return {
            "scene": str(destination),
            "splat": str(destination / splat_name),
            "publication_mode": "production",
            "quality_passed": quality_passed,
            "provisional": diagnostic_nonproduction,
            "metric_georeferencing_claim_eligible": production_georeferencing,
            "n_tiles": len(tile_ids),
            "n_validation_frames": len(validation_ids),
            "n_seam_validation_frames": len(seam_ids),
            "core_owned_gaussians": total,
            "retained_gaussians": retained,
            "absolute_metrics": {
                key: float(absolute[key]) for key in _METRICS
            },
            "absolute_seam_metrics": {
                key: float(seam_absolute[key]) for key in _METRICS
            },
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
