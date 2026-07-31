"""Mapper-neutral COLMAP reconstruction backend.

The backend reads only a sealed frontend artifact.  Each mapper receives a
private, SHA-256-verified database snapshot, solves the selected stereo
keyframes, and then registers every remaining image against that model.
Global mapping is the default; the incremental mapper is an independent
fallback using the same frontend evidence.

Cartesian pose priors are used only for post-solve fixed-scale
georeferencing and held-out evaluation. They do not constrain GlobalMapper.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from frontends.artifact import (
    ArtifactError,
    FRONTEND_SEAL_FILE,
    StageLedger,
    canonical_hash,
    create_database_snapshot,
    sha256_file,
    sqlite_logical_record,
    verify_frontend_seal,
)


MapperKind = Literal["global", "incremental"]
Runner = Callable[..., Any]

_REQUIRED_FRONTEND_FILES = (
    "frame_manifest.json",
    "rig_config.json",
    "keyframes.json",
    "pairs.txt",
    "provenance.json",
    "quality.json",
    FRONTEND_SEAL_FILE,
    "database.db",
)
_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
_STAT_NAMES = {
    "Rigs": "rigs",
    "Cameras": "cameras",
    "Frames": "frames",
    "Registered frames": "registered_frames",
    "Images": "images",
    "Registered images": "registered_images",
    "Points": "points",
    "Observations": "observations",
    "Mean track length": "mean_track_length",
    "Mean observations per image": "mean_observations_per_image",
    "Mean reprojection error": "mean_reprojection_error_px",
}
_STAT_PATTERN = re.compile(
    r"(Rigs|Cameras|Frames|Registered frames|Images|Registered images|"
    r"Points|Observations|Mean track length|Mean observations per image|"
    r"Mean reprojection error):\s*([-+0-9.eE]+)"
)
_INTEGER_STATS = {
    "rigs",
    "cameras",
    "frames",
    "registered_frames",
    "images",
    "registered_images",
    "points",
    "observations",
}


@dataclass(frozen=True)
class MapperConfig:
    """Shared controls for one isolated mapper attempt."""

    backend: MapperKind = "global"
    num_threads: int = 8
    random_seed: int = 7
    min_num_matches: int = 15
    require_all_keyframes: bool = True
    require_all_frames: bool = True
    max_reprojection_error_px: float = 2.0
    min_mean_track_length: float = 2.0
    alignment_ransac_threshold_m: float = 0.15
    alignment_ransac_iterations: int = 512
    alignment_temporal_blocks: int = 5
    max_rtk_median_error_m: float = 0.08
    max_rtk_p95_inlier_error_m: float = 0.15
    min_rtk_inlier_fraction: float = 0.80

    def __post_init__(self) -> None:
        if self.backend not in ("global", "incremental"):
            raise ValueError("backend must be 'global' or 'incremental'")
        for name in (
            "num_threads",
            "min_num_matches",
            "alignment_ransac_iterations",
            "alignment_temporal_blocks",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.alignment_temporal_blocks < 3:
            raise ValueError("alignment_temporal_blocks must be at least 3")
        if isinstance(self.random_seed, bool) or not isinstance(
            self.random_seed, int
        ):
            raise ValueError("random_seed must be an integer")
        for name in (
            "max_reprojection_error_px",
            "min_mean_track_length",
            "alignment_ransac_threshold_m",
            "max_rtk_median_error_m",
            "max_rtk_p95_inlier_error_m",
        ):
            value = float(getattr(self, name))
            if not value > 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.min_rtk_inlier_fraction <= 1:
            raise ValueError("min_rtk_inlier_fraction must be in (0, 1]")


@dataclass(frozen=True)
class AlignmentResult:
    """Robust fixed-scale transform plus a diagnostic-only similarity scale."""

    rotation: np.ndarray
    translation: np.ndarray
    residuals_m: np.ndarray
    inlier_mask: np.ndarray
    thresholds_m: np.ndarray
    sim3_scale_diagnostic: float
    source_rank: int


@dataclass(frozen=True)
class TemporalAlignmentResult:
    """Calibration-only transform with untouched temporal holdout residuals."""

    alignment: AlignmentResult
    temporal_block_ids: np.ndarray
    calibration_block_ids: tuple[int, ...]
    holdout_block_ids: tuple[int, ...]
    calibration_mask: np.ndarray
    holdout_mask: np.ndarray
    residuals_m: np.ndarray
    thresholds_m: np.ndarray
    calibration_inlier_mask: np.ndarray
    holdout_inlier_mask: np.ndarray


def _alignment_inputs(
    source: np.ndarray,
    target: np.ndarray,
    covariance: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (
        source.ndim != 2
        or source.shape[1:] != (3,)
        or target.shape != source.shape
        or len(source) < 3
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("alignment points must have finite matching shape (N, 3)")
    covariance_supplied = covariance is not None
    if covariance is None:
        covariance = np.repeat(np.eye(3)[None], len(source), axis=0)
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (len(source), 3, 3) or not np.isfinite(covariance).all():
        raise ValueError("alignment covariance must have finite shape (N, 3, 3)")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.any(eigenvalues < -1e-10):
        raise ValueError("alignment covariance must be positive semidefinite")
    sigma = (
        np.sqrt(np.maximum(eigenvalues[:, -1], 1e-8))
        if covariance_supplied
        else np.zeros(len(source), dtype=np.float64)
    )
    weights = 1.0 / np.maximum(np.trace(covariance, axis1=1, axis2=2), 1e-8)
    weights /= weights.mean()
    return source, target, covariance, np.column_stack((weights, sigma))


def _weighted_rigid(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if weights.shape != (len(source),) or not np.isfinite(total) or total <= 0:
        raise ValueError("alignment weights are invalid")
    source_mean = np.sum(source * weights[:, None], axis=0) / total
    target_mean = np.sum(target * weights[:, None], axis=0) / total
    source_centered = source - source_mean
    target_centered = target - target_mean
    rank = int(np.linalg.matrix_rank(source_centered * np.sqrt(weights[:, None])))
    if rank < 2:
        raise ArtifactError("camera-centre priors do not observe a 3-D rigid alignment")
    cross = (target_centered * weights[:, None]).T @ source_centered
    u, _, vt = np.linalg.svd(cross)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    translation = target_mean - rotation @ source_mean
    return rotation, translation, rank


def _similarity_scale(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    rotation: np.ndarray,
) -> float:
    total = float(weights.sum())
    source_mean = np.sum(source * weights[:, None], axis=0) / total
    target_mean = np.sum(target * weights[:, None], axis=0) / total
    source_centered = source - source_mean
    target_centered = target - target_mean
    rotated = source_centered @ rotation.T
    numerator = float(np.sum(weights * np.sum(target_centered * rotated, axis=1)))
    denominator = float(np.sum(weights * np.sum(source_centered**2, axis=1)))
    if denominator <= 1e-12:
        raise ArtifactError("similarity scale is unobservable")
    scale = numerator / denominator
    if not np.isfinite(scale) or scale <= 0:
        raise ArtifactError("similarity scale diagnostic is invalid")
    return scale


def estimate_rigid_alignment(
    source_centers_m: np.ndarray,
    target_centers_m: np.ndarray,
    covariance_m2: np.ndarray | None = None,
    *,
    ransac_threshold_m: float = 0.15,
    ransac_iterations: int = 512,
    random_seed: int = 7,
) -> AlignmentResult:
    """Estimate robust target-from-source SE(3), never applying Sim(3) scale."""
    if not np.isfinite(ransac_threshold_m) or ransac_threshold_m <= 0:
        raise ValueError("ransac_threshold_m must be finite and positive")
    if (
        isinstance(ransac_iterations, bool)
        or not isinstance(ransac_iterations, int)
        or ransac_iterations <= 0
    ):
        raise ValueError("ransac_iterations must be a positive integer")
    source, target, _, weight_sigma = _alignment_inputs(
        source_centers_m, target_centers_m, covariance_m2
    )
    weights = weight_sigma[:, 0]
    thresholds = np.maximum(ransac_threshold_m, 3.0 * weight_sigma[:, 1])
    rng = np.random.default_rng(random_seed)
    samples = [np.arange(len(source), dtype=np.int64)]
    samples.extend(
        rng.choice(len(source), size=3, replace=False)
        for _ in range(ransac_iterations)
    )
    best: tuple[tuple[int, float, float], np.ndarray] | None = None
    for sample in samples:
        try:
            rotation, translation, _ = _weighted_rigid(
                source[sample], target[sample], weights[sample]
            )
        except ArtifactError:
            continue
        residuals = np.linalg.norm(
            source @ rotation.T + translation - target, axis=1
        )
        inliers = residuals <= thresholds
        if inliers.sum() < 3:
            continue
        normalized = residuals[inliers] / thresholds[inliers]
        score = (
            int(inliers.sum()),
            -float(np.median(normalized)),
            -float(np.mean(normalized)),
        )
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None:
        raise ArtifactError("no observable rigid alignment hypothesis")
    inliers = best[1]
    for _ in range(3):
        rotation, translation, rank = _weighted_rigid(
            source[inliers], target[inliers], weights[inliers]
        )
        residuals = np.linalg.norm(
            source @ rotation.T + translation - target, axis=1
        )
        updated = residuals <= thresholds
        if updated.sum() < 3 or np.array_equal(updated, inliers):
            break
        inliers = updated
    rotation, translation, rank = _weighted_rigid(
        source[inliers], target[inliers], weights[inliers]
    )
    residuals = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    scale = _similarity_scale(
        source[inliers], target[inliers], weights[inliers], rotation
    )
    return AlignmentResult(
        rotation=rotation,
        translation=translation,
        residuals_m=residuals,
        inlier_mask=inliers,
        thresholds_m=thresholds,
        sim3_scale_diagnostic=scale,
        source_rank=rank,
    )


def estimate_temporal_heldout_alignment(
    source_centers_m: np.ndarray,
    target_centers_m: np.ndarray,
    timestamps_ns: np.ndarray,
    covariance_m2: np.ndarray | None = None,
    *,
    temporal_blocks: int = 5,
    ransac_threshold_m: float = 0.15,
    ransac_iterations: int = 512,
    random_seed: int = 7,
) -> TemporalAlignmentResult:
    """Fit on alternating contiguous time blocks and evaluate the rest.

    Membership depends only on sorted timestamp order, never on pose residuals.
    Odd-numbered blocks are sealed holdouts. Their target positions are not
    passed to the SE(3) or diagnostic Sim(3) estimators.
    """
    if (
        isinstance(temporal_blocks, bool)
        or not isinstance(temporal_blocks, int)
        or temporal_blocks < 3
    ):
        raise ValueError("temporal_blocks must be an integer of at least 3")
    covariance_supplied = covariance_m2 is not None
    source, target, covariance, weight_sigma = _alignment_inputs(
        source_centers_m, target_centers_m, covariance_m2
    )
    timestamps = np.asarray(timestamps_ns)
    if (
        timestamps.shape != (len(source),)
        or timestamps.dtype.kind not in "iuf"
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError(
            "alignment timestamps must be finite, one-dimensional, and "
            "strictly increasing"
        )
    block_count = min(temporal_blocks, len(source))
    block_ids = (
        np.arange(len(source), dtype=np.int64) * block_count // len(source)
    )
    unique_blocks = tuple(int(value) for value in np.unique(block_ids))
    holdout_blocks = tuple(value for value in unique_blocks if value % 2 == 1)
    calibration_blocks = tuple(
        value for value in unique_blocks if value % 2 == 0
    )
    calibration_mask = np.isin(block_ids, calibration_blocks)
    holdout_mask = np.isin(block_ids, holdout_blocks)
    if calibration_mask.sum() < 3 or not holdout_mask.any():
        raise ArtifactError(
            "temporal holdout requires at least three calibration priors and "
            "one untouched holdout prior"
        )

    alignment = estimate_rigid_alignment(
        source[calibration_mask],
        target[calibration_mask],
        covariance[calibration_mask] if covariance_supplied else None,
        ransac_threshold_m=ransac_threshold_m,
        ransac_iterations=ransac_iterations,
        random_seed=random_seed,
    )
    residuals = np.linalg.norm(
        source @ alignment.rotation.T + alignment.translation - target,
        axis=1,
    )
    thresholds = np.maximum(ransac_threshold_m, 3.0 * weight_sigma[:, 1])
    calibration_inliers = np.zeros(len(source), dtype=bool)
    calibration_inliers[calibration_mask] = alignment.inlier_mask
    holdout_inliers = holdout_mask & (residuals <= thresholds)
    return TemporalAlignmentResult(
        alignment=alignment,
        temporal_block_ids=block_ids,
        calibration_block_ids=calibration_blocks,
        holdout_block_ids=holdout_blocks,
        calibration_mask=calibration_mask,
        holdout_mask=holdout_mask,
        residuals_m=residuals,
        thresholds_m=thresholds,
        calibration_inlier_mask=calibration_inliers,
        holdout_inlier_mask=holdout_inliers,
    )


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{path} must contain a JSON object")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            if path.is_file() and path.read_bytes() == payload:
                return
            raise FileExistsError(f"refusing to overwrite {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    _atomic_write(path, payload)


def _frontend_context(
    artifact: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Validate the mapper-facing portion of a frontend without its segment."""
    root = Path(artifact).expanduser().resolve()
    missing = [name for name in _REQUIRED_FRONTEND_FILES if not (root / name).is_file()]
    if not root.is_dir() or missing:
        raise ArtifactError(f"incomplete frontend artifact {root}: {missing}")
    verify_frontend_seal(root)
    images = root / "images"
    if not images.is_dir():
        raise ArtifactError("frontend artifact has no images directory")
    entries = list(images.iterdir())
    if not entries or any(not entry.is_symlink() for entry in entries):
        raise ArtifactError("frontend images must be non-empty and symlink-only")

    manifest = _json(root / "frame_manifest.json")
    rows = manifest.get("frames")
    if not isinstance(rows, list) or not rows:
        raise ArtifactError("frame_manifest.frames must be a non-empty list")
    frame_ids: list[int] = []
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or type(row.get("frame_id")) is not int:
            raise ArtifactError("invalid frontend frame record")
        frame_ids.append(row["frame_id"])
        for field in ("left_image", "right_image"):
            image = row.get(field)
            if not isinstance(image, dict) or not isinstance(image.get("name"), str):
                raise ArtifactError(f"frontend frame lacks {field}")
            name = image["name"]
            if name in names:
                raise ArtifactError(f"duplicate frontend image name: {name}")
            names.add(name)
    if frame_ids != sorted(set(frame_ids)):
        raise ArtifactError("frontend frame IDs must be sorted and unique")
    if names != {entry.name for entry in entries}:
        raise ArtifactError("frontend image directory differs from its manifest")

    keyframes = _json(root / "keyframes.json")
    selected = keyframes.get("frame_ids")
    if (
        not isinstance(selected, list)
        or any(type(value) is not int for value in selected)
        or selected != sorted(set(selected))
        or not selected
        or not set(selected) <= set(frame_ids)
    ):
        raise ArtifactError("invalid keyframes.frame_ids")
    return root, manifest, keyframes


def _immutable_input_roots(frontend: Path) -> tuple[Path, ...]:
    roots = [frontend.resolve()]
    provenance = _json(frontend / "provenance.json")
    contract = provenance.get("contract_inputs")
    if isinstance(contract, dict) and contract.get("segment_root"):
        roots.append(Path(str(contract["segment_root"])).resolve())
    return tuple(dict.fromkeys(roots))


def _image_sets(
    manifest: Mapping[str, Any], keyframes: Mapping[str, Any]
) -> tuple[list[str], list[str], list[str]]:
    selected = set(keyframes["frame_ids"])
    all_names: list[str] = []
    solve_names: list[str] = []
    registration_names: list[str] = []
    for row in manifest["frames"]:
        names = [row["left_image"]["name"], row["right_image"]["name"]]
        all_names.extend(names)
        (solve_names if row["frame_id"] in selected else registration_names).extend(
            names
        )
    return all_names, solve_names, registration_names


def _frontend_hashes(root: Path) -> dict[str, str]:
    return {
        name: sha256_file(root / name)
        for name in _REQUIRED_FRONTEND_FILES
        if name != "database.db"
    }


def _marker_hash(root: Path, name: str) -> str | None:
    path = root / "stages" / f"{name}.json"
    if not path.is_file():
        return None
    marker = _json(path)
    if marker.get("state") != "complete":
        raise ArtifactError(f"frontend stage {name!r} is not complete")
    outputs = marker.get("outputs")
    if not isinstance(outputs, dict):
        raise ArtifactError(f"frontend stage {name!r} has no output evidence")
    for relative, expected in outputs.items():
        output = root / str(relative)
        if not output.is_file() or sha256_file(output) != expected:
            raise ArtifactError(f"frontend stage output changed: {relative}")
    return sha256_file(path)


def _write_lines(path: Path, values: Sequence[str]) -> None:
    if not values or len(values) != len(set(values)):
        raise ArtifactError(f"{path.name} must be non-empty and unique")
    _atomic_write(path, "".join(f"{value}\n" for value in values).encode())


def _verify_snapshot(workspace: Path, plan: Mapping[str, Any]) -> None:
    snapshot = _json(workspace / "database_snapshot.json")
    database = workspace / "database.db"
    try:
        committed = sqlite_logical_record(database)
    except ArtifactError as exc:
        raise ArtifactError("backend database snapshot changed or is invalid") from exc
    if (
        committed["sha256"] != snapshot.get("snapshot_sha256")
        or committed["sha256"] != plan.get("database_sha256")
        or committed["schema_sha256"] != snapshot.get("schema_sha256")
        or committed["integrity_check"] != "ok"
    ):
        raise ArtifactError("backend database snapshot changed")


def _workspace_context(
    workspace: str | Path,
) -> tuple[Path, dict[str, Any], MapperConfig]:
    root = Path(workspace).expanduser().resolve()
    plan = _json(root / "backend_plan.json")
    try:
        config = MapperConfig(**plan["config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError("invalid backend plan configuration") from exc
    frontend = Path(str(plan.get("frontend_artifact", ""))).resolve()
    seal = verify_frontend_seal(frontend)
    current = _frontend_hashes(frontend)
    if current != plan.get("frontend_inputs"):
        raise ArtifactError("sealed frontend inputs changed after backend preparation")
    if (
        sha256_file(frontend / FRONTEND_SEAL_FILE)
        != plan.get("frontend_seal_sha256")
        or seal["database"]["committed_view"]["sha256"]
        != plan.get("frontend_database_committed_sha256")
        or _marker_hash(frontend, "matching")
        != plan.get("matching_stage_sha256")
    ):
        raise ArtifactError("sealed frontend terminal evidence changed")
    _verify_snapshot(root, plan)
    for filename, expected_key in (
        ("solve_images.txt", "solve_image_names"),
        ("registration_images.txt", "registration_image_names"),
    ):
        values = (root / filename).read_text(encoding="utf-8").splitlines()
        if values != plan[expected_key]:
            raise ArtifactError(f"{filename} changed after backend preparation")
    return root, plan, config


def prepare_mapper_backend(
    frontend_artifact: str | Path,
    destination: str | Path,
    *,
    config: MapperConfig = MapperConfig(),
) -> Path:
    """Create or verify one isolated backend workspace.

    Calling this again with exactly the same inputs verifies and returns the
    existing workspace; changed inputs require a new destination.
    """
    frontend, manifest, keyframes = _frontend_context(frontend_artifact)
    workspace = Path(destination).expanduser().resolve()
    for immutable in _immutable_input_roots(frontend):
        if workspace == immutable or immutable in workspace.parents:
            raise ArtifactError(
                f"backend workspace must be outside immutable input {immutable}"
            )
    all_names, solve_names, registration_names = _image_sets(manifest, keyframes)
    frontend_inputs = _frontend_hashes(frontend)
    seal = verify_frontend_seal(frontend)
    matching_stage_sha256 = _marker_hash(frontend, "matching")
    if matching_stage_sha256 is None:
        raise ArtifactError("sealed frontend matching stage is incomplete or absent")
    inputs = {
        "frontend_artifact": str(frontend),
        "frontend_inputs": frontend_inputs,
        "frontend_seal_sha256": sha256_file(frontend / FRONTEND_SEAL_FILE),
        "frontend_database_raw_sha256": sha256_file(frontend / "database.db"),
        "frontend_database_committed_sha256": seal["database"]["committed_view"][
            "sha256"
        ],
        "matching_stage_sha256": matching_stage_sha256,
        "config": asdict(config),
        "all_image_names": all_names,
        "solve_image_names": solve_names,
        "registration_image_names": registration_names,
    }
    if not workspace.exists():
        snapshot = create_database_snapshot(
            frontend / "database.db",
            workspace,
            backend=config.backend,
            expected_committed_sha256=inputs[
                "frontend_database_committed_sha256"
            ],
        )
        # Close the narrow verification/copy race before publishing a plan.
        verify_frontend_seal(frontend)
    else:
        allowed = {
            "database.db",
            "database_snapshot.json",
            "stages",
            "backend_plan.json",
            "solve_images.txt",
            "registration_images.txt",
        }
        unexpected = {path.name for path in workspace.iterdir()} - allowed
        if unexpected and not (workspace / "backend_plan.json").is_file():
            raise ArtifactError(
                "existing backend prepare workspace has unexpected files: "
                + ", ".join(sorted(unexpected))
            )
        snapshot = _json(workspace / "database_snapshot.json")
        snapshot_digest = sha256_file(workspace / "database.db")
        if (
            snapshot.get("backend") != config.backend
            or Path(str(snapshot.get("source_database", ""))).resolve()
            != (frontend / "database.db").resolve()
            or snapshot.get("snapshot_sha256") != snapshot_digest
            or snapshot.get("source_sha256_before")
            != inputs["frontend_database_raw_sha256"]
            or snapshot.get("source_sha256_after")
            != inputs["frontend_database_raw_sha256"]
            or snapshot.get("source_committed_sha256")
            != inputs["frontend_database_committed_sha256"]
        ):
            raise ArtifactError("existing backend database snapshot is incompatible")
    ledger = StageLedger(workspace)
    plan = {
        "schema_version": 1,
        **inputs,
        "database_sha256": snapshot["snapshot_sha256"],
        "image_path": str(frontend / "images"),
        "n_frames": len(manifest["frames"]),
        "n_images": len(all_names),
        "n_solve_images": len(solve_names),
        "n_registration_images": len(registration_names),
        "all_frames_retained": True,
    }
    if (workspace / "backend_plan.json").exists():
        if _json(workspace / "backend_plan.json") != plan:
            raise ArtifactError("existing backend workspace has different inputs")
    status = ledger.begin("prepare", inputs)
    if status == "complete":
        root, _, _ = _workspace_context(workspace)
        return root
    _atomic_json(workspace / "backend_plan.json", plan)
    _write_lines(workspace / "solve_images.txt", solve_names)
    if registration_names:
        _write_lines(workspace / "registration_images.txt", registration_names)
    else:
        _atomic_write(workspace / "registration_images.txt", b"")
    ledger.complete(
        "prepare",
        inputs,
        [
            "backend_plan.json",
            "database_snapshot.json",
            "solve_images.txt",
            "registration_images.txt",
        ],
    )
    root, _, _ = _workspace_context(workspace)
    return root


def build_mapper_command(
    workspace: str | Path,
    executable: str | Path,
) -> tuple[str, ...]:
    root, plan, config = _workspace_context(workspace)
    shared = [
        str(executable),
        "global_mapper" if config.backend == "global" else "mapper",
        "--database_path",
        str(root / "database.db"),
        "--image_path",
        plan["image_path"],
        "--output_path",
        str(root / "solve.incomplete"),
    ]
    if config.backend == "global":
        shared.extend(
            [
                "--GlobalMapper.image_list_path",
                str(root / "solve_images.txt"),
                "--GlobalMapper.num_threads",
                str(config.num_threads),
                "--GlobalMapper.random_seed",
                str(config.random_seed),
                "--GlobalMapper.min_num_matches",
                str(config.min_num_matches),
                "--GlobalMapper.ba_refine_focal_length",
                "0",
                "--GlobalMapper.ba_refine_principal_point",
                "0",
                "--GlobalMapper.ba_refine_extra_params",
                "0",
                "--GlobalMapper.refine_sensor_from_rig",
                "0",
            ]
        )
    else:
        shared.extend(
            [
                "--Mapper.image_list_path",
                str(root / "solve_images.txt"),
                "--Mapper.num_threads",
                str(config.num_threads),
                "--Mapper.random_seed",
                str(config.random_seed),
                "--Mapper.min_num_matches",
                str(config.min_num_matches),
                "--Mapper.multiple_models",
                "0",
                "--Mapper.ba_refine_focal_length",
                "0",
                "--Mapper.ba_refine_principal_point",
                "0",
                "--Mapper.ba_refine_extra_params",
                "0",
                "--Mapper.ba_refine_sensor_from_rig",
                "0",
            ]
        )
    return tuple(shared)


def build_registration_command(
    workspace: str | Path,
    executable: str | Path,
) -> tuple[str, ...]:
    root, plan, config = _workspace_context(workspace)
    selected = _selected_model(root)
    return (
        str(executable),
        "image_registrator",
        "--database_path",
        str(root / "database.db"),
        "--input_path",
        str(selected),
        "--output_path",
        str(root / "registration.incomplete"),
        "--Mapper.min_num_matches",
        str(config.min_num_matches),
    )


def build_model_analyzer_command(
    model: str | Path, executable: str | Path
) -> tuple[str, ...]:
    return (
        str(executable),
        "model_analyzer",
        "--log_target",
        "stdout",
        "--path",
        str(Path(model).resolve()),
    )


def build_model_converter_command(
    workspace: str | Path, executable: str | Path
) -> tuple[str, ...]:
    root, _, _ = _workspace_context(workspace)
    return (
        str(executable),
        "model_converter",
        "--input_path",
        str(root / "registered_model"),
        "--output_path",
        str(root / "registered_text.incomplete"),
        "--output_type",
        "TXT",
    )


def parse_model_analyzer(output: str) -> dict[str, int | float]:
    """Parse COLMAP's stable model-analyzer labels."""
    result: dict[str, int | float] = {}
    for label, raw in _STAT_PATTERN.findall(output):
        name = _STAT_NAMES[label]
        value = float(raw)
        result[name] = int(round(value)) if name in _INTEGER_STATS else value
    required = {
        "registered_images",
        "points",
        "observations",
        "mean_track_length",
        "mean_reprojection_error_px",
    }
    missing = required - set(result)
    if missing:
        raise ArtifactError(
            "model_analyzer output is missing: " + ", ".join(sorted(missing))
        )
    return result


def _execute(command: Sequence[str], runner: Runner, *, capture: bool = False) -> Any:
    kwargs: dict[str, Any] = {"check": True}
    if capture:
        kwargs.update(
            {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
        )
    return runner(list(command), **kwargs)


def _analyze_model(
    model: Path, executable: str | Path, runner: Runner
) -> dict[str, int | float]:
    result = _execute(
        build_model_analyzer_command(model, executable), runner, capture=True
    )
    output = getattr(result, "stdout", None)
    if not isinstance(output, str):
        raise ArtifactError("model_analyzer runner returned no text output")
    return parse_model_analyzer(output)


def _model_candidates(root: Path) -> list[Path]:
    candidates = []
    if all((root / name).is_file() for name in _MODEL_FILES):
        candidates.append(root)
    if root.is_dir():
        candidates.extend(
            child
            for child in sorted(root.iterdir())
            if child.is_dir() and all((child / name).is_file() for name in _MODEL_FILES)
        )
    return candidates


def _tree_manifest(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise ArtifactError(f"model directory does not exist: {root}")
    files = {
        path.relative_to(root).as_posix(): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    if not files:
        raise ArtifactError(f"model directory is empty: {root}")
    return {"schema_version": 1, "files": files}


def _verify_tree(root: Path, manifest: Mapping[str, Any]) -> None:
    if _tree_manifest(root) != manifest:
        raise ArtifactError(f"published model changed: {root}")


def _selected_model(workspace: Path) -> Path:
    record = _json(_report_path(workspace, "solve"))
    relative = record.get("selected_model_relative")
    if not isinstance(relative, str):
        raise ArtifactError("solve report has no selected model")
    candidate = (workspace / "solve_models" / relative).resolve()
    try:
        candidate.relative_to((workspace / "solve_models").resolve())
    except ValueError as exc:
        raise ArtifactError("selected model escapes solve_models") from exc
    if not all((candidate / name).is_file() for name in _MODEL_FILES):
        raise ArtifactError("selected solve model is incomplete")
    return candidate


def _report_path(root: Path, stage: str) -> Path:
    return root / "reports" / f"{stage}.json"


def _complete_stage(
    root: Path,
    ledger: StageLedger,
    stage: str,
    inputs: Mapping[str, Any],
    report: Mapping[str, Any],
    manifest_name: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    report_path = _report_path(root, stage)
    _atomic_json(report_path, report)
    _atomic_json(root / manifest_name, manifest)
    ledger.complete(stage, inputs, [report_path, manifest_name])
    return dict(report)


def run_mapper_solve(
    workspace: str | Path,
    executable: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Run or verify the keyframe-only mapper stage."""
    root, plan, config = _workspace_context(workspace)
    command = build_mapper_command(root, executable)
    inputs = {
        "command": command,
        "prepare_marker_sha256": sha256_file(root / "stages" / "prepare.json"),
        "database_sha256": plan["database_sha256"],
    }
    ledger = StageLedger(root)
    status = ledger.begin("solve", inputs)
    published = root / "solve_models"
    if status == "complete":
        manifest = _json(root / "solve_model_manifest.json")
        _verify_tree(published, manifest)
        _selected_model(root)
        return _json(_report_path(root, "solve"))

    incomplete = root / "solve.incomplete"
    if not published.exists():
        if incomplete.exists():
            shutil.rmtree(incomplete)
        incomplete.mkdir()
        try:
            _execute(command, runner)
            candidates = _model_candidates(incomplete)
            if not candidates:
                raise ArtifactError("mapper produced no valid COLMAP model")
            analyses = [
                {
                    "relative_path": candidate.relative_to(incomplete).as_posix(),
                    "stats": _analyze_model(candidate, executable, runner),
                }
                for candidate in candidates
            ]
            selected = max(
                analyses,
                key=lambda item: (
                    item["stats"]["registered_images"],
                    item["stats"]["observations"],
                    -item["stats"]["mean_reprojection_error_px"],
                ),
            )
            if config.require_all_keyframes and (
                selected["stats"]["registered_images"] != plan["n_solve_images"]
            ):
                raise ArtifactError(
                    "mapper did not register every selected stereo keyframe image"
                )
            os.rename(incomplete, published)
        except Exception:
            shutil.rmtree(incomplete, ignore_errors=True)
            raise
    else:
        candidates = _model_candidates(published)
        analyses = [
            {
                "relative_path": candidate.relative_to(published).as_posix(),
                "stats": _analyze_model(candidate, executable, runner),
            }
            for candidate in candidates
        ]
        if not analyses:
            raise ArtifactError("published solve model is incomplete")
        selected = max(
            analyses,
            key=lambda item: (
                item["stats"]["registered_images"],
                item["stats"]["observations"],
                -item["stats"]["mean_reprojection_error_px"],
            ),
        )
    report = {
        "schema_version": 1,
        "stage": "solve",
        "backend": config.backend,
        "command": list(command),
        "n_expected_solve_images": plan["n_solve_images"],
        "selected_model_relative": selected["relative_path"],
        "selected_stats": selected["stats"],
        "candidates": analyses,
        "all_keyframes_registered": (
            selected["stats"]["registered_images"] == plan["n_solve_images"]
        ),
    }
    return _complete_stage(
        root,
        ledger,
        "solve",
        inputs,
        report,
        "solve_model_manifest.json",
        _tree_manifest(published),
    )


def run_image_registration(
    workspace: str | Path,
    executable: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Register all non-keyframes from the full private frontend database."""
    root, plan, config = _workspace_context(workspace)
    solve_marker = root / "stages" / "solve.json"
    if not solve_marker.is_file():
        raise ArtifactError("run the solve stage before image registration")
    _verify_tree(root / "solve_models", _json(root / "solve_model_manifest.json"))
    command = build_registration_command(root, executable)
    inputs = {
        "command": command,
        "solve_marker_sha256": sha256_file(solve_marker),
        "database_sha256": plan["database_sha256"],
        "registration_image_names_sha256": canonical_hash(
            plan["registration_image_names"]
        ),
    }
    ledger = StageLedger(root)
    status = ledger.begin("register", inputs)
    published = root / "registered_model"
    if status == "complete":
        _verify_tree(published, _json(root / "registered_model_manifest.json"))
        return _json(_report_path(root, "register"))

    incomplete = root / "registration.incomplete"
    if not published.exists():
        if incomplete.exists():
            shutil.rmtree(incomplete)
        incomplete.mkdir()
        try:
            _execute(command, runner)
            if not all((incomplete / name).is_file() for name in _MODEL_FILES):
                raise ArtifactError("image_registrator produced no valid model")
            stats = _analyze_model(incomplete, executable, runner)
            if config.require_all_frames and stats["registered_images"] != plan["n_images"]:
                raise ArtifactError("image_registrator did not register every image")
            os.rename(incomplete, published)
        except Exception:
            shutil.rmtree(incomplete, ignore_errors=True)
            raise
    else:
        stats = _analyze_model(published, executable, runner)
    solve_stats = _json(_report_path(root, "solve"))["selected_stats"]
    added = int(stats["registered_images"]) - int(solve_stats["registered_images"])
    report = {
        "schema_version": 1,
        "stage": "register",
        "command": list(command),
        "n_expected_images": plan["n_images"],
        "n_expected_non_keyframe_images": plan["n_registration_images"],
        "n_newly_registered_images": added,
        "stats": stats,
        "all_frames_registered_by_count": stats["registered_images"] == plan["n_images"],
    }
    return _complete_stage(
        root,
        ledger,
        "register",
        inputs,
        report,
        "registered_model_manifest.json",
        _tree_manifest(published),
    )


def registered_names_from_images_txt(path: str | Path) -> tuple[str, ...]:
    """Read registered image names from a COLMAP text model."""
    names: list[str] = []
    expect_pose = True
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        if expect_pose:
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) < 10:
                raise ArtifactError("malformed COLMAP images.txt pose row")
            names.append(fields[9])
            expect_pose = False
        else:
            # The observation row may legitimately be empty.
            expect_pose = True
    if not expect_pose:
        raise ArtifactError("COLMAP images.txt ends before an observation row")
    if len(names) != len(set(names)):
        raise ArtifactError("COLMAP text model has duplicate image names")
    return tuple(names)


def _quaternion_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    quaternion = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ArtifactError("COLMAP pose contains an invalid quaternion")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _poses_from_images_txt(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    expect_pose = True
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        if not expect_pose:
            expect_pose = True
            continue
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 10:
            raise ArtifactError("malformed COLMAP images.txt pose row")
        try:
            quaternion = [float(value) for value in fields[1:5]]
            translation = np.asarray(
                [float(value) for value in fields[5:8]], dtype=np.float64
            )
        except ValueError as exc:
            raise ArtifactError("COLMAP pose row contains non-numeric values") from exc
        name = fields[9]
        if name in poses:
            raise ArtifactError(f"duplicate COLMAP pose for {name}")
        rotation = _quaternion_rotation(*quaternion)
        viewmat = np.eye(4, dtype=np.float64)
        viewmat[:3, :3] = rotation
        viewmat[:3, 3] = translation
        center = -(rotation.T @ translation)
        poses[name] = (viewmat, center)
        expect_pose = False
    if not expect_pose:
        raise ArtifactError("COLMAP images.txt ends before an observation row")
    return poses


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_pose_artifact(path: Path) -> dict[str, Any]:
    manifest = _json(path / "manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ArtifactError("pose artifact manifest has no file evidence")
    for relative, evidence in files.items():
        candidate = path / relative
        if (
            not isinstance(evidence, dict)
            or not candidate.is_file()
            or sha256_file(candidate) != evidence.get("sha256")
            or candidate.stat().st_size != evidence.get("size_bytes")
        ):
            raise ArtifactError(f"pose artifact file changed: {relative}")
    return manifest


def _cartesian_camera_priors(
    database: Path, allowed_names: set[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    uri = f"file:{database.resolve()}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
        rows = connection.execute(
            """
            SELECT i.name, p.position, p.position_covariance, p.coordinate_system
            FROM pose_priors AS p
            JOIN images AS i ON i.image_id = p.corr_data_id
            WHERE p.corr_sensor_type = 0
            """
        ).fetchall()
    except sqlite3.Error as exc:
        raise ArtifactError(f"cannot read sealed Cartesian pose priors: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, position_blob, covariance_blob, coordinate_system in rows:
        name = str(name)
        if name not in allowed_names:
            continue
        if int(coordinate_system) != 1:
            raise ArtifactError(f"pose prior for {name} is not Cartesian")
        if position_blob is None or covariance_blob is None:
            raise ArtifactError(f"pose prior for {name} lacks covariance or position")
        position = np.frombuffer(position_blob, dtype="<f8").copy()
        covariance = np.frombuffer(covariance_blob, dtype="<f8").copy()
        if position.shape != (3,) or covariance.shape != (9,):
            raise ArtifactError(f"pose prior for {name} has invalid blob dimensions")
        covariance = covariance.reshape(3, 3, order="F")
        covariance = 0.5 * (covariance + covariance.T)
        if (
            not np.isfinite(position).all()
            or not np.isfinite(covariance).all()
            or np.linalg.eigvalsh(covariance).min() < -1e-10
        ):
            raise ArtifactError(f"pose prior for {name} is numerically invalid")
        if name in result:
            raise ArtifactError(f"duplicate Cartesian pose prior for {name}")
        result[name] = (position, covariance)
    if len(result) < 3:
        raise ArtifactError("fewer than three trusted Cartesian camera-centre priors")
    return result


def _apply_world_alignment(
    viewmats: np.ndarray,
    centers: np.ndarray,
    rotation_target_source: np.ndarray,
    translation_target_source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a unit-scale world transform while preserving OpenCV w2c poses."""
    aligned_centers = centers @ rotation_target_source.T + translation_target_source
    aligned = np.repeat(np.eye(4, dtype=np.float64)[None], len(viewmats), axis=0)
    aligned[:, :3, :3] = (
        viewmats[:, :3, :3] @ rotation_target_source.T
    )
    aligned[:, :3, 3] = -np.einsum(
        "nij,nj->ni", aligned[:, :3, :3], aligned_centers
    )
    recovered = -np.einsum(
        "nji,nj->ni", aligned[:, :3, :3], aligned[:, :3, 3]
    )
    if not np.allclose(recovered, aligned_centers, atol=1e-9, rtol=1e-9):
        raise ArtifactError("aligned view matrices and camera centres disagree")
    return aligned, aligned_centers


def export_pose_artifact(
    workspace: str | Path,
    name: str,
    *,
    output_root: str | Path,
) -> Path:
    """Publish one named, provenance-complete left-camera pose artifact."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"invalid pose artifact name: {name!r}")
    root, plan, config = _workspace_context(workspace)
    quality_marker = root / "stages" / "quality.json"
    if not quality_marker.is_file():
        raise ArtifactError("run the exact-registration quality stage before export")
    quality_report = _json(_report_path(root, "quality"))
    if (
        quality_report.get("missing_images")
        or quality_report.get("unexpected_images")
        or quality_report.get("registration_fraction") != 1.0
    ):
        raise ArtifactError("pose export requires exact left+right registration")
    if not quality_report.get("passed"):
        raise ArtifactError("refusing to export a pose artifact that failed quality")
    text_manifest = _json(root / "text_model_manifest.json")
    _verify_tree(root / "registered_text", text_manifest)
    frontend = Path(plan["frontend_artifact"])
    output = Path(output_root).expanduser().resolve()
    destination = output / name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite pose artifact: {destination}")
    for immutable in _immutable_input_roots(frontend):
        if destination == immutable or immutable in destination.parents:
            raise ArtifactError(
                f"pose artifact output must be outside immutable input {immutable}"
            )
    manifest = _json(frontend / "frame_manifest.json")
    rows = manifest["frames"]
    poses = _poses_from_images_txt(root / "registered_text" / "images.txt")
    left_names = [row["left_image"]["name"] for row in rows]
    missing = sorted(set(left_names) - set(poses))
    if missing:
        raise ArtifactError(f"final model lacks left-camera poses: {missing[:8]}")
    frame_ids = np.asarray([row["frame_id"] for row in rows], dtype=np.int64)
    timestamps = np.asarray([row["timestamp_ns"] for row in rows], dtype=np.int64)
    raw_viewmats = np.stack([poses[name][0] for name in left_names])
    raw_centers = np.stack([poses[name][1] for name in left_names])
    priors = _cartesian_camera_priors(root / "database.db", set(left_names))
    prior_names = [name for name in left_names if name in priors]
    source = np.stack([poses[name][1] for name in prior_names])
    target = np.stack([priors[name][0] for name in prior_names])
    covariance = np.stack([priors[name][1] for name in prior_names])
    prior_timestamps = np.asarray(
        [
            rows[index]["timestamp_ns"]
            for index, name in enumerate(left_names)
            if name in priors
        ],
        dtype=np.int64,
    )
    evaluation = estimate_temporal_heldout_alignment(
        source,
        target,
        prior_timestamps,
        covariance,
        temporal_blocks=config.alignment_temporal_blocks,
        ransac_threshold_m=config.alignment_ransac_threshold_m,
        ransac_iterations=config.alignment_ransac_iterations,
        random_seed=config.random_seed,
    )
    alignment = evaluation.alignment
    viewmats, centers = _apply_world_alignment(
        raw_viewmats,
        raw_centers,
        alignment.rotation,
        alignment.translation,
    )
    holdout_residuals = evaluation.residuals_m[evaluation.holdout_mask]
    holdout_inlier_residuals = evaluation.residuals_m[
        evaluation.holdout_inlier_mask
    ]
    if not holdout_inlier_residuals.size:
        raise ArtifactError(
            "fixed-scale ENU alignment has no RTK inlier in held-out blocks"
        )
    residual_median = float(np.median(holdout_residuals))
    residual_p95 = float(np.percentile(holdout_residuals, 95))
    inlier_p95 = float(np.percentile(holdout_inlier_residuals, 95))
    inlier_fraction = float(
        evaluation.holdout_inlier_mask.sum() / evaluation.holdout_mask.sum()
    )
    rtk_checks = {
        "fixed_scale_se3_applied": {
            "value": 1.0,
            "expected": 1.0,
            "passed": True,
        },
        "median_holdout_rtk_residual_m": {
            "value": residual_median,
            "maximum": config.max_rtk_median_error_m,
            "passed": residual_median <= config.max_rtk_median_error_m,
        },
        "p95_holdout_inlier_rtk_residual_m": {
            "value": inlier_p95,
            "maximum": config.max_rtk_p95_inlier_error_m,
            "passed": inlier_p95 <= config.max_rtk_p95_inlier_error_m,
        },
        "holdout_rtk_inlier_fraction": {
            "value": inlier_fraction,
            "minimum": config.min_rtk_inlier_fraction,
            "passed": inlier_fraction >= config.min_rtk_inlier_fraction,
        },
    }
    if not all(check["passed"] for check in rtk_checks.values()):
        raise ArtifactError("fixed-scale ENU alignment failed RTK residual gates")
    if (
        np.any(np.diff(frame_ids) <= 0)
        or np.any(np.diff(timestamps) <= 0)
        or not np.isfinite(viewmats).all()
        or not np.isfinite(centers).all()
    ):
        raise ArtifactError("exported pose arrays failed integrity checks")

    inputs = {
        "name": name,
        "quality_marker_sha256": sha256_file(quality_marker),
        "text_model_manifest_sha256": sha256_file(
            root / "text_model_manifest.json"
        ),
        "frame_manifest_sha256": sha256_file(frontend / "frame_manifest.json"),
        "backend_plan_sha256": sha256_file(root / "backend_plan.json"),
        "pose_prior_assignment_sha256": canonical_hash(
            [
                {
                    "name": name,
                    "position_m": priors[name][0],
                    "covariance_m2": priors[name][1],
                }
                for name in prior_names
            ]
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{name}.writing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        _atomic_save_npy(staging / "viewmats.npy", viewmats)
        _atomic_save_npy(staging / "cam_centers.npy", centers)
        _atomic_save_npy(staging / "frame_ids.npy", frame_ids)
        _atomic_save_npy(staging / "timestamps_ns.npy", timestamps)
        _atomic_save_npy(staging / "left_image_names.npy", np.asarray(left_names))
        export_quality = {
            "schema_version": 1,
            "source_quality_report_sha256": sha256_file(
                _report_path(root, "quality")
            ),
            "n_frames": len(frame_ids),
            "all_left_camera_poses_present": True,
            "all_frontend_images_registered": bool(quality_report["passed"]),
            "registration_fraction": quality_report["registration_fraction"],
            "mean_reprojection_error_px": quality_report["checks"][
                "mean_reprojection_error_px"
            ]["value"],
            "mean_track_length": quality_report["checks"]["mean_track_length"][
                "value"
            ],
            "coordinate_frame": "local_enu_from_cartesian_pose_priors",
            "n_trusted_rtk_priors": len(prior_names),
            "n_calibration_rtk_priors": int(evaluation.calibration_mask.sum()),
            "n_calibration_rtk_inliers": int(
                evaluation.calibration_inlier_mask.sum()
            ),
            "n_holdout_rtk_priors": int(evaluation.holdout_mask.sum()),
            "n_holdout_rtk_inliers": int(
                evaluation.holdout_inlier_mask.sum()
            ),
            "holdout_rtk_residual_m": {
                "median": residual_median,
                "p95": residual_p95,
                "p95_inliers": inlier_p95,
                "maximum_inliers": float(holdout_inlier_residuals.max()),
            },
            "rtk_alignment_checks": rtk_checks,
            "rtk_alignment_passed": all(
                check["passed"] for check in rtk_checks.values()
            ),
            "sim3_scale_diagnostic": alignment.sim3_scale_diagnostic,
            "sim3_scale_applied": False,
        }
        alignment_record = {
            "schema_version": 1,
            "method": (
                "covariance_weighted_ransac_fixed_scale_se3_"
                "with_temporal_holdout"
            ),
            "source_frame": "colmap_sfm_world",
            "target_frame": "local_enu_cartesian_pose_priors",
            "pose_prior_role": (
                "post_solve_georeferencing_and_heldout_evaluation_only"
            ),
            "pose_priors_constrain_mapper": False,
            "production_scale": 1.0,
            "sim3_scale_diagnostic": alignment.sim3_scale_diagnostic,
            "sim3_scale_fit_membership": "calibration_blocks_only",
            "sim3_scale_applied": False,
            "rotation_target_source": alignment.rotation.tolist(),
            "translation_target_source_m": alignment.translation.tolist(),
            "source_rank": alignment.source_rank,
            "prior_image_names": prior_names,
            "prior_timestamps_ns": prior_timestamps.tolist(),
            "temporal_split": {
                "method": "contiguous_rank_blocks_alternating_v1",
                "requested_block_count": config.alignment_temporal_blocks,
                "realized_block_count": int(
                    len(np.unique(evaluation.temporal_block_ids))
                ),
                "block_id_by_prior": evaluation.temporal_block_ids.tolist(),
                "calibration_block_ids": list(
                    evaluation.calibration_block_ids
                ),
                "holdout_block_ids": list(evaluation.holdout_block_ids),
                "calibration_prior_image_names": [
                    prior_names[index]
                    for index in np.flatnonzero(evaluation.calibration_mask)
                ],
                "holdout_prior_image_names": [
                    prior_names[index]
                    for index in np.flatnonzero(evaluation.holdout_mask)
                ],
            },
            "calibration_inlier_mask": (
                evaluation.calibration_inlier_mask.tolist()
            ),
            "holdout_inlier_mask": evaluation.holdout_inlier_mask.tolist(),
            "residuals_m": evaluation.residuals_m.tolist(),
            "ransac_thresholds_m": evaluation.thresholds_m.tolist(),
            "checks": rtk_checks,
        }
        provenance = {
            "schema_version": 1,
            "frontend_artifact": str(frontend),
            "source_backend_workspace": str(root),
            "frontend_inputs": plan["frontend_inputs"],
            "backend": config.backend,
            "backend_config": plan["config"],
            "database_snapshot_sha256": plan["database_sha256"],
            "solve_marker_sha256": sha256_file(root / "stages" / "solve.json"),
            "register_marker_sha256": sha256_file(
                root / "stages" / "register.json"
            ),
            "quality_marker_sha256": sha256_file(quality_marker),
            "backend_plan_sha256": inputs["backend_plan_sha256"],
            "frame_manifest_sha256": inputs["frame_manifest_sha256"],
            "pose_prior_assignment_sha256": inputs[
                "pose_prior_assignment_sha256"
            ],
            "registered_model_manifest_sha256": sha256_file(
                root / "registered_model_manifest.json"
            ),
            "text_model_manifest_sha256": inputs["text_model_manifest_sha256"],
            "pose_convention": {
                "viewmats": "world_to_left_camera",
                "viewmat_camera_axes": "OpenCV_x_right_y_down_z_forward",
                "cam_centers": "left_camera_center_in_local_enu_m",
                "quaternion_input": "COLMAP_QW_QX_QY_QZ",
                "world_alignment": "fixed_scale_SE3_only",
            },
        }
        _atomic_json(staging / "quality.json", export_quality)
        _atomic_json(staging / "alignment.json", alignment_record)
        _atomic_json(staging / "provenance.json", provenance)
        file_evidence = {
            path.name: {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        _atomic_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "name": name,
                "n_frames": len(frame_ids),
                "files": file_evidence,
            },
        )
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _verify_pose_artifact(destination)
    return destination


def quality_summary(
    expected_image_names: Sequence[str],
    registered_image_names: Sequence[str],
    model_stats: Mapping[str, int | float],
    *,
    max_reprojection_error_px: float = 2.0,
    min_mean_track_length: float = 2.0,
) -> dict[str, Any]:
    """Build explicit registration and geometric quality gates."""
    expected = set(expected_image_names)
    registered = set(registered_image_names)
    missing = sorted(expected - registered)
    unexpected = sorted(registered - expected)
    count_consistent = int(model_stats["registered_images"]) == len(registered)
    checks = {
        "all_expected_images_registered": {
            "value": len(missing),
            "maximum": 0,
            "passed": not missing,
        },
        "no_unexpected_images": {
            "value": len(unexpected),
            "maximum": 0,
            "passed": not unexpected,
        },
        "analyzer_count_consistent": {
            "value": int(model_stats["registered_images"]),
            "expected": len(registered),
            "passed": count_consistent,
        },
        "mean_reprojection_error_px": {
            "value": float(model_stats["mean_reprojection_error_px"]),
            "maximum": float(max_reprojection_error_px),
            "passed": float(model_stats["mean_reprojection_error_px"])
            <= max_reprojection_error_px,
        },
        "mean_track_length": {
            "value": float(model_stats["mean_track_length"]),
            "minimum": float(min_mean_track_length),
            "passed": float(model_stats["mean_track_length"])
            >= min_mean_track_length,
        },
    }
    return {
        "schema_version": 1,
        "n_expected_images": len(expected),
        "n_registered_images": len(registered),
        "registration_fraction": len(expected & registered) / len(expected),
        "missing_images": missing,
        "unexpected_images": unexpected,
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }


def run_quality_summary(
    workspace: str | Path,
    executable: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Convert the final model to text and verify exact image registration."""
    root, plan, config = _workspace_context(workspace)
    register_marker = root / "stages" / "register.json"
    if not register_marker.is_file():
        raise ArtifactError("run image registration before quality summary")
    _verify_tree(
        root / "registered_model", _json(root / "registered_model_manifest.json")
    )
    converter = build_model_converter_command(root, executable)
    analyzer = build_model_analyzer_command(root / "registered_model", executable)
    inputs = {
        "converter_command": converter,
        "analyzer_command": analyzer,
        "register_marker_sha256": sha256_file(register_marker),
        "expected_image_names_sha256": canonical_hash(plan["all_image_names"]),
        "max_reprojection_error_px": config.max_reprojection_error_px,
        "min_mean_track_length": config.min_mean_track_length,
    }
    ledger = StageLedger(root)
    status = ledger.begin("quality", inputs)
    if status == "complete":
        _verify_tree(root / "registered_text", _json(root / "text_model_manifest.json"))
        return _json(_report_path(root, "quality"))

    text_model = root / "registered_text"
    incomplete = root / "registered_text.incomplete"
    if not text_model.exists():
        if incomplete.exists():
            shutil.rmtree(incomplete)
        incomplete.mkdir()
        try:
            _execute(converter, runner)
            if not (incomplete / "images.txt").is_file():
                raise ArtifactError("model_converter produced no images.txt")
            os.rename(incomplete, text_model)
        except Exception:
            shutil.rmtree(incomplete, ignore_errors=True)
            raise
    names = registered_names_from_images_txt(text_model / "images.txt")
    stats = _analyze_model(root / "registered_model", executable, runner)
    report = quality_summary(
        plan["all_image_names"],
        names,
        stats,
        max_reprojection_error_px=config.max_reprojection_error_px,
        min_mean_track_length=config.min_mean_track_length,
    )
    report.update(
        {
            "stage": "quality",
            "converter_command": list(converter),
            "analyzer_command": list(analyzer),
        }
    )
    if config.require_all_frames and (
        report["missing_images"] or report["unexpected_images"]
    ):
        raise ArtifactError("final model failed exact all-image registration")
    return _complete_stage(
        root,
        ledger,
        "quality",
        inputs,
        report,
        "text_model_manifest.json",
        _tree_manifest(text_model),
    )
