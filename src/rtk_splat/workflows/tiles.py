"""Immutable, visibility-derived spatial tile planning.

The planner consumes one validated segment and one global pose artifact.  It
does not copy observations or alter poses.  Stereo/RGB-D depth is sampled into
an oriented ENU support graph, recursively partitioned under a training-view
budget, and expanded only for overlapping training context.  Gaussian centres
will later be owned by the disjoint half-open cores recorded here.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from rtk_splat.backends.pose_evidence import (
    pose_georeferencing_evidence,
    require_render_permission,
)
from rtk_splat.core.pose_artifacts import (
    load_pose_artifact,
    pose_artifact_dir,
    pose_artifact_name,
    pose_fingerprint,
)
from rtk_splat.core.runtime_resolution import (
    record_runtime_resolution,
    runtime_control_value,
)
from rtk_splat.core.segment import SegmentReader, publish_directory_noreplace
from rtk_splat.frontends.artifact import (
    ArtifactError,
    canonical_hash,
    collect_package_state,
    sha256_file,
)
from rtk_splat.workflows.runtime_config import _is_auto


SCHEMA_VERSION = 3
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FILES = (
    "tile_plan.json",
    "visibility.npz",
    "source_inventory.json",
    "quality.json",
    "provenance.json",
    "plan.svg",
)
_SPLITS = ("train", "val", "test")
_SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
_OWNERSHIP_TOLERANCE_M = 1e-9


@dataclass(frozen=True)
class TilePlannerPolicy:
    """Small, generic policy surface; resolved values are sealed in the plan."""

    max_training_frames: int
    max_tiles: int = 64
    max_support_samples: int = 2_000_000
    support_cell_m: float = 0.50
    min_visibility_fraction: float = 0.01
    min_visibility_cells: int = 4
    context_halo_m: float | None = None
    minimum_core_train_frames: int = 8
    max_depthless_frame_fraction: float = 0.02
    min_core_train_support_coverage: float = 0.98

    def validate(self) -> "TilePlannerPolicy":
        if self.max_training_frames <= 0:
            raise ValueError("tiles.max_training_frames must be positive")
        if self.max_tiles <= 0 or self.max_tiles > 4096:
            raise ValueError("tiles.max_tiles must be in [1, 4096]")
        if self.max_support_samples < 1000:
            raise ValueError("tiles.max_support_samples must be at least 1000")
        if not math.isfinite(self.support_cell_m) or self.support_cell_m <= 0:
            raise ValueError("tiles.support_cell_m must be finite and positive")
        if not 0 <= self.min_visibility_fraction <= 1:
            raise ValueError("tiles.min_visibility_fraction must be in [0, 1]")
        if self.min_visibility_cells <= 0:
            raise ValueError("tiles.min_visibility_cells must be positive")
        if self.context_halo_m is not None and (
            not math.isfinite(self.context_halo_m) or self.context_halo_m < 0
        ):
            raise ValueError("tiles.context_halo_m must be nonnegative or auto")
        if self.minimum_core_train_frames <= 0:
            raise ValueError("tiles.minimum_core_train_frames must be positive")
        if not 0 <= self.max_depthless_frame_fraction < 1:
            raise ValueError(
                "tiles.max_depthless_frame_fraction must be in [0, 1)"
            )
        if not 0 <= self.min_core_train_support_coverage <= 1:
            raise ValueError(
                "tiles.min_core_train_support_coverage must be in [0, 1]"
            )
        return self


@dataclass(frozen=True)
class TileExecution:
    """Verified, portable execution contract for exactly one sealed tile."""

    root: Path
    tile_index: int
    tile_plan: Mapping[str, Any]
    train_ids: tuple[int, ...]
    val_ids: tuple[int, ...]
    test_ids: tuple[int, ...]

    @property
    def tile_id(self) -> str:
        return str(self.tile_plan["tile_id"])

    @property
    def binding(self) -> dict[str, Any]:
        """Small identity copied into clouds, training runs, and scenes."""
        plan = self.tile_plan["plan"]
        return {
            "schema_version": 1,
            "name": str(plan["name"]),
            "manifest_sha256": sha256_file(self.root / "manifest.json"),
            "tile_plan_sha256": sha256_file(self.root / "tile_plan.json"),
            "source_inventory_sha256": str(
                plan["source_binding"]["inventory_sha256"]
            ),
            "pose_fingerprint": str(
                plan["source_binding"]["pose_fingerprint"]
            ),
            "tile_id": self.tile_id,
            "tile_index": int(self.tile_index),
            "train_selection_sha256": canonical_hash(list(self.train_ids)),
            "val_selection_sha256": canonical_hash(list(self.val_ids)),
            "test_selection_sha256": canonical_hash(list(self.test_ids)),
            "n_train": len(self.train_ids),
            "n_val": len(self.val_ids),
            "n_test": len(self.test_ids),
            "partition_origin_enu_m": list(
                plan["coordinate_frame"]["partition_origin_enu_m"]
            ),
            "R_enu_from_partition": list(
                plan["coordinate_frame"]["R_enu_from_partition"]
            ),
            "scene_bounds_uv_m": list(
                plan["partition"]["scene_bounds_uv_m"]
            ),
            "core_bounds_uv_m": list(self.tile_plan["core_bounds_uv_m"]),
            "context_bounds_uv_m": list(
                self.tile_plan["context_bounds_uv_m"]
            ),
            "boundary_rule": str(plan["partition"]["boundary_rule"]),
            "ownership_tolerance_m": float(
                plan["partition"]["ownership_tolerance_m"]
            ),
        }


def _plain(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {key: _plain(item) for key, item in vars(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(_plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write(path, _json_bytes(value))


def _deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write an NPZ whose bytes do not depend on wall-clock ZIP metadata."""
    with path.open("xb") as raw:
        with zipfile.ZipFile(
            raw, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for name in sorted(arrays):
                value = np.asarray(arrays[name])
                if value.dtype.hasobject:
                    raise ArtifactError(f"visibility.{name} has an object dtype")
                payload = io.BytesIO()
                np.save(payload, value, allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, payload.getvalue())
        raw.flush()
        os.fsync(raw.fileno())


def _file_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _safe_name(value: str) -> str:
    name = str(value)
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError(f"invalid tile-plan artifact name {name!r}")
    return name


def _safe_relative(value: str) -> str:
    path = PurePosixPath(str(value))
    if (
        path.is_absolute()
        or not path.parts
        or "." in path.parts
        or ".." in path.parts
        or "\\" in str(value)
        or path.as_posix() != str(value)
    ):
        raise ArtifactError(f"unsafe source inventory path: {value!r}")
    return str(value)


def _section(cfg: Any) -> Any:
    value = getattr(cfg, "tiles", None)
    if value is None:
        value = SimpleNamespace()
        cfg.tiles = value
    return value


def resolve_tile_training_budget(cfg: Any) -> int:
    """Preserve the profile's intended image presentations per training view."""
    tiles = _section(cfg)
    current = runtime_control_value(
        cfg,
        "tile_max_training_frames",
        getattr(tiles, "max_training_frames", "auto"),
        config_path="tiles.max_training_frames",
    )
    if not _is_auto(current):
        if isinstance(current, bool) or int(current) != current or int(current) <= 0:
            raise ValueError("tiles.max_training_frames must be auto or positive int")
        chosen = int(current)
        record_runtime_resolution(
            cfg,
            "tile_max_training_frames",
            source="override",
            formula="authored or CLI numeric value; no runtime formula applied",
            inputs={},
            policy={},
            chosen=chosen,
        )
    else:
        derivation = getattr(cfg, "derivation", None)
        training = getattr(derivation, "training", None)
        if training is None:
            raise ValueError(
                "automatic tile capacity requires derivation.training policy"
            )
        presentations = float(training.image_presentations_per_view)
        authored_iterations = runtime_control_value(
            cfg,
            "train_iterations",
            getattr(cfg.train, "iterations", "auto"),
            config_path="train.iterations",
        )
        if _is_auto(authored_iterations):
            iteration_budget = int(training.max_iterations)
            iteration_budget_source = "derivation.training.max_iterations"
        else:
            if (
                isinstance(authored_iterations, bool)
                or int(authored_iterations) != authored_iterations
                or int(authored_iterations) <= 0
            ):
                raise ValueError("train.iterations must be auto or positive int")
            iteration_budget = int(authored_iterations)
            iteration_budget_source = "train.iterations"
        cameras = 2 if bool(getattr(cfg.train, "use_right_camera", False)) else 1
        if presentations <= 0 or iteration_budget <= 0:
            raise ValueError("invalid training policy for automatic tile capacity")
        chosen = int(math.floor(iteration_budget / (presentations * cameras)))
        if chosen <= 0:
            raise ValueError("training policy resolves to a zero-frame tile")
        record_runtime_resolution(
            cfg,
            "tile_max_training_frames",
            source="derived",
            formula=(
                "floor(iteration_budget / (image_presentations_per_view * "
                "supervised_cameras_per_frame))"
            ),
            inputs={
                "supervised_cameras_per_frame": cameras,
                "iteration_budget": iteration_budget,
                "iteration_budget_source": iteration_budget_source,
            },
            policy={
                "image_presentations_per_view": presentations,
            },
            chosen=chosen,
        )
    tiles.max_training_frames = chosen
    return chosen


def planner_policy(cfg: Any) -> TilePlannerPolicy:
    section = _section(cfg)
    strategy = str(
        getattr(section, "strategy", "recursive_visibility_workload_v1")
    )
    if strategy != "recursive_visibility_workload_v1":
        raise ValueError(
            "tiles.strategy must be 'recursive_visibility_workload_v1'"
        )
    halo = getattr(section, "context_halo_m", "auto")
    halo_value = None if _is_auto(halo) else float(halo)
    return TilePlannerPolicy(
        max_training_frames=resolve_tile_training_budget(cfg),
        max_tiles=int(getattr(section, "max_tiles", 64)),
        max_support_samples=int(
            getattr(section, "max_support_samples", 2_000_000)
        ),
        support_cell_m=float(getattr(section, "support_cell_m", 0.50)),
        min_visibility_fraction=float(
            getattr(section, "min_visibility_fraction", 0.01)
        ),
        min_visibility_cells=int(
            getattr(section, "min_visibility_cells", 4)
        ),
        context_halo_m=halo_value,
        minimum_core_train_frames=int(
            getattr(section, "minimum_core_train_frames", 8)
        ),
        max_depthless_frame_fraction=float(
            getattr(section, "max_depthless_frame_fraction", 0.02)
        ),
        min_core_train_support_coverage=float(
            getattr(section, "min_core_train_support_coverage", 0.98)
        ),
    ).validate()


def _partition_basis(
    viewmats: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotations_wc = np.swapaxes(viewmats[:, :3, :3], 1, 2)
    directions = rotations_wc[:, :2, 2]
    norms = np.linalg.norm(directions, axis=1)
    usable = norms > 1e-5
    if int(usable.sum()) >= max(2, len(centers) // 4):
        unit = directions[usable] / norms[usable, None]
        moment = unit.T @ unit
        _, vectors = np.linalg.eigh(moment)
        u = vectors[:, -1]
        source = unit
    else:
        delta = np.diff(centers[:, :2], axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        if int((lengths > 1e-6).sum()) < 2:
            raise ArtifactError("trajectory has no observable horizontal axis")
        source = delta[lengths > 1e-6] / lengths[lengths > 1e-6, None]
        _, vectors = np.linalg.eigh(source.T @ source)
        u = vectors[:, -1]
    sign_index = int(np.argmax(np.abs(u)))
    if u[sign_index] < 0:
        u = -u
    v = np.asarray([-u[1], u[0]], dtype=np.float64)
    origin = np.asarray(
        [centers[:, 0].mean(), centers[:, 1].mean(), 0.0], dtype=np.float64
    )
    basis = np.asarray(
        [[u[0], v[0], 0.0], [u[1], v[1], 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return origin, basis, rotations_wc


def _contract_inventory(reader: SegmentReader) -> dict[str, Any]:
    names = [
        "frames.npz",
        "calibration.json",
        "segment_meta.json",
        "manifest.json",
        "observations/gnss.npz",
    ]
    for optional in ("observations/heading.npz", "observations/imu.npz"):
        if (reader.root / optional).is_file():
            names.append(optional)
    return {
        name: _file_record(reader.root / name)
        for name in sorted(names)
    }


def _image_inventory(reader: SegmentReader) -> list[dict[str, Any]]:
    """Hash every image that a downstream tile is allowed to supervise."""
    frames = reader.frames
    sides = [("left", "left_image_path")]
    if reader.meta["capabilities"]["stereo"]:
        sides.append(("right", "right_image_path"))
    records = []
    for index, frame_id in enumerate(frames["frame_id"].astype(int)):
        for side, path_field in sides:
            relative = _safe_relative(str(frames[path_field][index]))
            path = reader.root / relative
            record = _file_record(path)
            supplied = None
            for field in (
                f"{side}_image_sha256",
                f"stereo_{side}_sha256",
                f"{side}_sha256",
            ):
                if field in frames:
                    if frames[field].shape != frames["frame_id"].shape:
                        raise ArtifactError(f"frames.{field} has an invalid shape")
                    supplied = str(frames[field][index]).lower()
                    if not _SHA256.fullmatch(supplied):
                        raise ArtifactError(f"frames.{field}[{index}] is not SHA-256")
                    break
            if supplied is not None and supplied != record["sha256"]:
                raise ArtifactError(
                    f"{side} image content disagrees with its supplied hash: {path}"
                )
            records.append(
                {
                    "frame_id": frame_id,
                    "camera": side,
                    "path": relative,
                    **record,
                }
            )
    return records


def _pose_inventory(reader: SegmentReader, cfg: Any) -> dict[str, Any]:
    name = pose_artifact_name(cfg)
    if name == "rtk":
        return {"kind": "segment_initial_pose", "files": {}}
    root = pose_artifact_dir(reader.root, cfg)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise ArtifactError(f"named pose artifact has no manifest: {root}")
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read pose manifest: {root}") from exc
    files = value.get("files")
    if not isinstance(files, dict):
        raise ArtifactError("pose artifact manifest has no file evidence")
    records = {"manifest.json": _file_record(manifest)}
    for relative in sorted(files):
        candidate = root / relative
        if not candidate.is_file():
            raise ArtifactError(f"pose artifact file is missing: {candidate}")
        current = _file_record(candidate)
        if current != files[relative]:
            raise ArtifactError(f"pose artifact file changed: {candidate}")
        records[relative] = current
    for filename, expected in (
        ("frame_ids.npy", reader.frames["frame_id"]),
        ("timestamps_ns.npy", reader.frames["timestamp_ns"]),
    ):
        path = root / filename
        if path.is_file() and not np.array_equal(
            np.load(path, allow_pickle=False), expected
        ):
            raise ArtifactError(f"pose {filename} does not match source segment")
    return {"kind": "named_pose_artifact", "files": records}


def _depth_payload(path: Path) -> tuple[bytes, dict[str, Any]]:
    payload = path.read_bytes()
    return payload, {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _sample_support(
    reader: SegmentReader,
    viewmats: np.ndarray,
    centers: np.ndarray,
    basis: np.ndarray,
    origin: np.ndarray,
    policy: TilePlannerPolicy,
) -> tuple[
    list[np.ndarray], list[dict[str, Any]], dict[str, Any], np.ndarray
]:
    frames = reader.frames
    if "depth_path" not in frames or any(not str(item) for item in frames["depth_path"]):
        raise ArtifactError(
            "final visibility planning requires metric depth for every frame"
        )
    camera = reader.calibration["cameras"]["left"]
    width, height = int(camera["width"]), int(camera["height"])
    k = np.asarray(camera["K"], dtype=np.float64)
    stride = max(
        1,
        int(
            math.ceil(
                math.sqrt(
                    len(frames["frame_id"]) * width * height
                    / policy.max_support_samples
                )
            )
        ),
    )
    ys = np.arange(stride // 2, height, stride, dtype=np.int64)
    xs = np.arange(stride // 2, width, stride, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    rotations_wc = np.swapaxes(viewmats[:, :3, :3], 1, 2)
    # The derived depth artifact has already applied its configured physical
    # range. This additional finite/positive check protects the planner from a
    # malformed file without reinterpreting the depth policy.
    minimum = 0.01
    support: list[np.ndarray] = []
    depth_records: list[dict[str, Any]] = []
    sampled_depths: list[np.ndarray] = []
    for index, relative_value in enumerate(frames["depth_path"]):
        relative = str(relative_value)
        path = reader.root / relative
        payload, record = _depth_payload(path)
        depth_records.append({"frame_id": index, "path": relative, **record})
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                depth = np.asarray(archive["depth"], dtype=np.float32)
                valid = np.asarray(archive["valid"], dtype=bool)
        except (OSError, KeyError, ValueError) as exc:
            raise ArtifactError(f"invalid depth artifact: {path}") from exc
        if depth.shape != (height, width) or valid.shape != depth.shape:
            raise ArtifactError(f"depth shape disagrees with calibration: {path}")
        sampled = depth[np.ix_(ys, xs)]
        selected = valid[np.ix_(ys, xs)] & np.isfinite(sampled) & (sampled >= minimum)
        z = sampled[selected].astype(np.float64)
        if len(z) == 0:
            support.append(np.empty((0, 2), dtype=np.float64))
            continue
        x = (grid_x[selected] - k[0, 2]) * z / k[0, 0]
        y = (grid_y[selected] - k[1, 2]) * z / k[1, 1]
        camera_xyz = np.column_stack([x, y, z])
        world = camera_xyz @ rotations_wc[index].T + centers[index]
        partition = (world - origin) @ basis
        ij = np.floor(partition[:, :2] / policy.support_cell_m).astype(np.int64)
        ij = np.unique(ij, axis=0)
        support.append((ij.astype(np.float64) + 0.5) * policy.support_cell_m)
        sampled_depths.append(z)
    counts = np.asarray([len(item) for item in support], dtype=np.int64)
    if len(sampled_depths) == 0 or counts.sum() < 100:
        raise ArtifactError("insufficient valid metric depth for tile planning")
    missing = np.flatnonzero(counts == 0)
    missing_fraction = float(len(missing) / len(counts))
    if missing_fraction > policy.max_depthless_frame_fraction:
        raise ArtifactError(
            "tile planner depthless-frame fraction exceeds its configured gate: "
            f"{len(missing)}/{len(counts)} ({missing_fraction:.6f}) > "
            f"{policy.max_depthless_frame_fraction:.6f}"
        )
    depths = np.concatenate(sampled_depths)
    diagnostics = {
        "sampling_record": {
            "depth_sample_stride_px": stride,
            "image_width_px": width,
            "image_height_px": height,
        },
        "n_support_cells": int(counts.sum()),
        "support_cells_per_frame": {
            "minimum": int(counts.min()),
            "median": float(np.median(counts)),
            "maximum": int(counts.max()),
        },
        "depth_support": {
            "n_supported_frames": int(np.sum(counts > 0)),
            "n_depthless_frames": int(len(missing)),
            "depthless_fraction": missing_fraction,
            "maximum_depthless_fraction": policy.max_depthless_frame_fraction,
            "depthless_frame_ids": missing.astype(int).tolist(),
            "fallback": "nearest_supported_temporal_neighbours",
        },
        "sampled_depth_m": {
            "median": float(np.median(depths)),
            "p75": float(np.quantile(depths, 0.75)),
            "maximum": float(depths.max()),
        },
    }
    return support, depth_records, diagnostics, depths.astype(np.float32)


def _split_codes(reader: SegmentReader) -> np.ndarray:
    n = int(reader.meta["n_frames"])
    result = np.full(n, 255, dtype=np.uint8)
    for name in _SPLITS:
        ids = np.asarray(reader.manifest[name], dtype=np.int64)
        result[ids] = _SPLIT_CODE[name]
    if np.any(result == 255):
        raise ArtifactError("segment split does not cover every frame")
    return result


def _visibility_threshold(counts: np.ndarray, policy: TilePlannerPolicy) -> np.ndarray:
    return np.maximum(
        policy.min_visibility_cells,
        np.ceil(counts * policy.min_visibility_fraction).astype(np.int64),
    )


def _in_bounds(points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.all(points >= bounds[0], axis=1) & np.all(points < bounds[1], axis=1)


def _frame_counts(
    support: Sequence[np.ndarray], bounds: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [int(_in_bounds(points, bounds).sum()) for points in support],
        dtype=np.int32,
    )


def _context_bounds(core: np.ndarray, halo: float) -> np.ndarray:
    return np.asarray([core[0] - halo, core[1] + halo], dtype=np.float64)


def _unique_support(
    support: Sequence[np.ndarray],
    bounds: np.ndarray,
    frame_mask: np.ndarray | None = None,
) -> np.ndarray:
    if frame_mask is None:
        frame_mask = np.ones(len(support), dtype=bool)
    chunks = [
        points[_in_bounds(points, bounds)]
        for points, include in zip(support, frame_mask)
        if include and len(points)
    ]
    chunks = [chunk for chunk in chunks if len(chunk)]
    if not chunks:
        return np.empty((0, 2), dtype=np.float64)
    return np.unique(np.concatenate(chunks, axis=0), axis=0)


def _support_coverage(reference: np.ndarray, covered: np.ndarray) -> float:
    if len(reference) == 0:
        return 0.0
    covered_rows = {tuple(row) for row in covered.tolist()}
    return float(
        sum(tuple(row) in covered_rows for row in reference.tolist())
        / len(reference)
    )


_COVERAGE_COMPLETION = {
    "method": "greedy_missing_core_support_v1",
    "split": "train",
    "candidate_scope": "unselected_training_frames_with_core_support",
    "ranking": "maximum_new_core_cells_then_lowest_frame_id",
    "capacity": "never_exceed_max_training_frames",
}


def _complete_core_training_coverage(
    support: Sequence[np.ndarray],
    core: np.ndarray,
    selected: np.ndarray,
    split_codes: np.ndarray,
    *,
    max_training_frames: int,
    minimum_coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Add the fewest greedily useful train views needed for core coverage.

    Visibility remains the primary selection rule.  This deterministic second
    pass admits only an otherwise-unselected training frame that covers core
    support cells still missing from the selected training set.  It never
    changes validation/test membership or exceeds the sealed training-view
    capacity.
    """
    result = np.asarray(selected, dtype=bool).copy()
    codes = np.asarray(split_codes, dtype=np.uint8)
    if result.shape != codes.shape or result.shape != (len(support),):
        raise ValueError("coverage completion inputs have inconsistent shapes")
    added = np.zeros(len(support), dtype=bool)
    reference_array = _unique_support(support, core)
    if len(reference_array) == 0:
        return result, added
    reference = {tuple(row) for row in reference_array.tolist()}
    required = int(
        math.ceil(float(minimum_coverage) * len(reference) - 1e-12)
    )
    train = codes == _SPLIT_CODE["train"]

    def core_cells(frame_id: int) -> set[tuple[float, float]]:
        points = support[frame_id]
        if len(points) == 0:
            return set()
        return {
            tuple(row)
            for row in points[_in_bounds(points, core)].tolist()
        }

    covered: set[tuple[float, float]] = set()
    for frame_id in np.flatnonzero(result & train):
        covered.update(core_cells(int(frame_id)))
    if len(covered) >= required:
        return result, added
    capacity = int(max_training_frames) - int(np.sum(result & train))
    if capacity <= 0:
        return result, added

    import heapq

    candidates: dict[int, set[tuple[float, float]]] = {}
    heap: list[tuple[int, int]] = []
    uncovered = reference - covered
    for frame_id in np.flatnonzero(train & ~result):
        cells = core_cells(int(frame_id))
        gain = len(cells & uncovered)
        if gain:
            frame = int(frame_id)
            candidates[frame] = cells
            heapq.heappush(heap, (-gain, frame))
    while heap and capacity > 0 and len(reference) - len(uncovered) < required:
        negative_gain, frame_id = heapq.heappop(heap)
        cells = candidates[frame_id]
        gain = len(cells & uncovered)
        if gain == 0:
            continue
        if gain != -negative_gain:
            heapq.heappush(heap, (-gain, frame_id))
            continue
        result[frame_id] = True
        added[frame_id] = True
        uncovered.difference_update(cells)
        capacity -= 1
    return result, added


def _selected(
    support: Sequence[np.ndarray], core: np.ndarray, halo: float,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    core_count = _frame_counts(support, core)
    context_count = _frame_counts(support, _context_bounds(core, halo))
    selected = context_count >= thresholds
    supported = np.asarray([len(points) > 0 for points in support], dtype=bool)
    if not supported.all():
        valid_ids = np.flatnonzero(supported)
        for frame_id in np.flatnonzero(~supported):
            insertion = int(np.searchsorted(valid_ids, frame_id))
            neighbours = []
            if insertion > 0:
                neighbours.append(int(valid_ids[insertion - 1]))
            if insertion < len(valid_ids):
                neighbours.append(int(valid_ids[insertion]))
            # The segment contract is chronological. Including a depth-poor
            # frame wherever either bracketing supported observation is visible
            # preserves motion continuity without inventing spatial evidence.
            selected[frame_id] = any(selected[index] for index in neighbours)
    return core_count, context_count, selected


def _candidate_cuts(values: np.ndarray, low: float, high: float) -> list[float]:
    inside = values[(values > low) & (values < high)]
    if len(inside) < 2:
        return []
    quantiles = np.linspace(0.15, 0.85, 15)
    raw = np.quantile(inside, quantiles)
    return sorted(
        set(float(item) for item in raw if low + 1e-6 < item < high - 1e-6)
    )


def _best_split(
    core: np.ndarray,
    support: Sequence[np.ndarray],
    all_points: np.ndarray,
    split_codes: np.ndarray,
    thresholds: np.ndarray,
    halo: float,
    policy: TilePlannerPolicy,
) -> tuple[np.ndarray, np.ndarray] | None:
    best: tuple[tuple[Any, ...], np.ndarray, np.ndarray] | None = None
    for axis in (0, 1):
        for cut in _candidate_cuts(all_points[:, axis], core[0, axis], core[1, axis]):
            left, right = core.copy(), core.copy()
            left[1, axis] = cut
            right[0, axis] = cut
            records = []
            valid = True
            for child in (left, right):
                core_count, _, selected = _selected(
                    support, child, halo, thresholds
                )
                core_train = int(
                    np.sum((core_count >= thresholds) & (split_codes == 0))
                )
                context_train = int(np.sum(selected & (split_codes == 0)))
                if core_train < policy.minimum_core_train_frames:
                    valid = False
                    break
                records.append((context_train, selected))
            if not valid:
                continue
            maximum = max(records[0][0], records[1][0])
            excess = max(0, maximum - policy.max_training_frames)
            imbalance = abs(records[0][0] - records[1][0])
            duplicate = int(np.sum(records[0][1] & records[1][1] & (split_codes == 0)))
            # Capacity and balanced compute come first. Among equally useful
            # cuts, prefer less duplicated context. This avoids choosing a tiny
            # sliver merely to reduce overlap before workload is balanced.
            score = (excess, maximum, imbalance, duplicate, axis, cut)
            if best is None or score < best[0]:
                best = (score, left, right)
    return None if best is None else (best[1], best[2])


def _partition(
    support: Sequence[np.ndarray],
    centers_uv: np.ndarray,
    split_codes: np.ndarray,
    policy: TilePlannerPolicy,
    halo: float,
    tile_count: int | None,
) -> tuple[list[np.ndarray], np.ndarray]:
    all_points = np.concatenate([*support, centers_uv], axis=0)
    margin = max(policy.support_cell_m, 1e-3)
    root = np.asarray(
        [all_points.min(axis=0) - margin, all_points.max(axis=0) + margin],
        dtype=np.float64,
    )
    thresholds = _visibility_threshold(
        np.asarray([len(item) for item in support], dtype=np.int64), policy
    )
    cores = [root]
    requested = None if tile_count is None else int(tile_count)
    if requested is not None and (requested <= 0 or requested > policy.max_tiles):
        raise ValueError(f"--tile-count must be in [1, {policy.max_tiles}]")
    while True:
        counts = [
            int(np.sum(_selected(support, core, halo, thresholds)[2] & (split_codes == 0)))
            for core in cores
        ]
        if requested is not None:
            if len(cores) >= requested:
                break
        elif max(counts) <= policy.max_training_frames:
            break
        if len(cores) >= policy.max_tiles:
            raise ArtifactError(
                "tile planner reached max_tiles before satisfying capacity"
            )
        order = sorted(range(len(cores)), key=lambda index: (-counts[index], index))
        split_index = None
        children = None
        for index in order:
            proposed = _best_split(
                cores[index], support, all_points, split_codes, thresholds,
                halo, policy,
            )
            if proposed is not None:
                split_index, children = index, proposed
                break
        if children is None or split_index is None:
            raise ArtifactError(
                "visibility/workload evidence cannot form another safe tile"
            )
        cores[split_index : split_index + 1] = [children[0], children[1]]
    cores.sort(key=lambda item: (item[0, 1], item[0, 0], item[1, 1], item[1, 0]))
    final_counts = np.asarray(
        [
            int(np.sum(_selected(support, core, halo, thresholds)[2] & (split_codes == 0)))
            for core in cores
        ],
        dtype=np.int64,
    )
    if np.any(final_counts > policy.max_training_frames):
        raise ArtifactError(
            "requested tile count violates max_training_frames: "
            + ", ".join(str(int(item)) for item in final_counts)
        )
    return cores, thresholds


def owner_tile_indices(plan: Mapping[str, Any], points_enu_m: np.ndarray) -> np.ndarray:
    """Return the unique core owner, or -1 outside the sealed scene extent."""
    points = np.asarray(points_enu_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("points_enu_m must have shape (N,2) or (N,3)")
    coordinate = plan["coordinate_frame"]
    origin = np.asarray(coordinate["partition_origin_enu_m"], dtype=np.float64)
    basis = np.asarray(coordinate["R_enu_from_partition"], dtype=np.float64)
    xyz = np.zeros((len(points), 3), dtype=np.float64)
    xyz[:, : points.shape[1]] = points
    uv = ((xyz - origin) @ basis)[:, :2]
    scene = np.asarray(plan["partition"]["scene_bounds_uv_m"], dtype=np.float64)
    owners = np.full(len(points), -1, dtype=np.int64)
    for index, tile in enumerate(plan["tiles"]):
        bounds = np.asarray(tile["core_bounds_uv_m"], dtype=np.float64)
        lower = uv >= bounds[0]
        upper = uv < bounds[1]
        for axis in (0, 1):
            if bounds[0, axis] == scene[0, axis]:
                lower[:, axis] = (
                    uv[:, axis] >= bounds[0, axis] - _OWNERSHIP_TOLERANCE_M
                )
            if bounds[1, axis] == scene[1, axis]:
                upper[:, axis] = (
                    uv[:, axis] <= bounds[1, axis] + _OWNERSHIP_TOLERANCE_M
                )
        inside = np.all(lower & upper, axis=1)
        if np.any(inside & (owners >= 0)):
            raise ArtifactError("tile cores assign a point more than once")
        owners[inside] = index
    return owners


def _svg(
    plan: Mapping[str, Any], centers: np.ndarray, support: Sequence[np.ndarray]
) -> bytes:
    origin = np.asarray(plan["coordinate_frame"]["partition_origin_enu_m"])
    basis = np.asarray(plan["coordinate_frame"]["R_enu_from_partition"])
    uv = ((centers - origin) @ basis)[:, :2]
    scene = np.asarray(plan["partition"]["scene_bounds_uv_m"])
    width, height, pad = 1100.0, 720.0, 45.0
    span = np.maximum(scene[1] - scene[0], 1e-9)
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])
    def xy(values: np.ndarray) -> tuple[float, float]:
        return (
            pad + (float(values[0]) - scene[0, 0]) * scale,
            height - pad - (float(values[1]) - scene[0, 1]) * scale,
        )
    colours = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2")
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="720" viewBox="0 0 1100 720">',
        '<rect width="1100" height="720" fill="white"/>',
        '<text x="45" y="28" font-family="sans-serif" font-size="18">RTK-Splat sealed visibility tile plan</text>',
    ]
    unique_support = np.unique(np.concatenate(support), axis=0)
    support_stride = max(1, int(math.ceil(len(unique_support) / 6000)))
    for point in unique_support[::support_stride]:
        px, py = xy(point)
        lines.append(
            f'<circle cx="{px:.3f}" cy="{py:.3f}" r="0.8" fill="#999" fill-opacity="0.35"/>'
        )
    for index, tile in enumerate(plan["tiles"]):
        bounds = np.asarray(tile["core_bounds_uv_m"])
        context = np.asarray(tile["context_bounds_uv_m"])
        cx0, cy1 = xy(context[0]); cx1, cy0 = xy(context[1])
        x0, y1 = xy(bounds[0]); x1, y0 = xy(bounds[1])
        colour = colours[index % len(colours)]
        lines.append(
            f'<rect x="{cx0:.3f}" y="{cy0:.3f}" width="{cx1-cx0:.3f}" height="{cy1-cy0:.3f}" '
            f'fill="none" stroke="{colour}" stroke-opacity="0.55" stroke-dasharray="6 4"/>'
        )
        lines.append(
            f'<rect x="{x0:.3f}" y="{y0:.3f}" width="{x1-x0:.3f}" height="{y1-y0:.3f}" '
            f'fill="{colour}" fill-opacity="0.12" stroke="{colour}" stroke-width="2"/>'
        )
        lines.append(
            f'<text x="{x0+5:.3f}" y="{y0+18:.3f}" font-family="sans-serif" font-size="14">{tile["tile_id"]}</text>'
        )
    path = " ".join(
        ("M" if index == 0 else "L") + f" {xy(point)[0]:.3f} {xy(point)[1]:.3f}"
        for index, point in enumerate(uv)
    )
    lines.extend([
        f'<path d="{path}" fill="none" stroke="#222" stroke-width="1.2"/>',
        "</svg>",
    ])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _manifest(root: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "rtk_splat_tile_plan",
        "files": {name: _file_record(root / name) for name in _FILES},
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON artifact is not an object: {path}")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"JSON artifact has non-finite/unsafe values: {path}") from exc
    return value


def verify_tile_plan(
    root: str | Path,
    *,
    segment: str | Path | None = None,
    pose_root: str | Path | None = None,
    rehash_sources: bool = True,
) -> dict[str, Any]:
    """Verify the terminal seal, geometry, selections, and optional live sources."""
    supplied_root = Path(root).expanduser()
    if supplied_root.is_symlink():
        raise ArtifactError(f"tile-plan artifact cannot be a symlink: {supplied_root}")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise ArtifactError(f"tile-plan artifact does not exist safely: {root}")
    present = sorted(path.name for path in root.iterdir())
    expected = sorted((*_FILES, "manifest.json"))
    if present != expected or any(path.is_symlink() for path in root.iterdir()):
        raise ArtifactError("tile-plan artifact has missing, extra, or symlink files")
    manifest = _read_json(root / "manifest.json")
    if manifest != _manifest(root):
        raise ArtifactError("tile-plan terminal seal verification failed")
    plan = _read_json(root / "tile_plan.json")
    quality = _read_json(root / "quality.json")
    inventory = _read_json(root / "source_inventory.json")
    provenance = _read_json(root / "provenance.json")
    if (
        plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("artifact_type") != "rtk_splat_tile_plan"
        or quality.get("schema_version") != SCHEMA_VERSION
        or provenance.get("schema_version") != SCHEMA_VERSION
        or not quality.get("passed")
    ):
        raise ArtifactError("tile-plan schema or quality status is invalid")
    selection_record = plan.get("selection")
    coverage_completion = selection_record == {
        "primary": "configured_visibility_threshold_v1",
        "coverage_completion": _COVERAGE_COMPLETION,
    }
    if selection_record is not None and not coverage_completion:
        raise ArtifactError("tile-plan selection policy is unsupported")
    plan_name = str(plan.get("name", ""))
    valid_directory = root.name == plan_name or root.name.startswith(
        f".{plan_name}.writing-"
    )
    if not valid_directory or not _SAFE_NAME.fullmatch(plan_name):
        raise ArtifactError("tile-plan name is invalid or disagrees with its directory")
    inventory_hash = canonical_hash(inventory)
    if (
        plan.get("source_binding", {}).get("inventory_sha256") != inventory_hash
        or provenance.get("source_inventory_sha256") != inventory_hash
        or inventory.get("schema_version") != SCHEMA_VERSION
    ):
        raise ArtifactError("tile-plan source inventory binding is invalid")
    georeferencing = provenance.get("georeferencing", {})
    pose_record = inventory.get("pose", {})
    pose_files = pose_record.get("files", {})
    if (
        not isinstance(pose_files, dict)
        or georeferencing.get("pose_artifact") != pose_record.get("name")
        or plan.get("source_binding", {}).get("pose_fingerprint")
        != pose_record.get("fingerprint")
        or not _SHA256.fullmatch(str(pose_record.get("fingerprint", "")))
    ):
        raise ArtifactError("tile-plan pose identity binding is invalid")
    expected_pose_hashes = {
        "pose_manifest_sha256": (
            pose_files.get("manifest.json", {}).get("sha256")
        ),
        "pose_quality_sha256": (
            pose_files.get("quality.json", {}).get("sha256")
        ),
        "pose_georeferencing_sha256": (
            pose_files.get("georeferencing.json", {}).get("sha256")
        ),
    }
    if any(
        georeferencing.get(field) != expected
        for field, expected in expected_pose_hashes.items()
    ):
        raise ArtifactError("tile-plan georeferencing hashes disagree with its pose")
    has_modern_pose_declaration = bool(
        pose_record.get("kind") == "named_pose_artifact"
        and all(
            isinstance(expected, str) and _SHA256.fullmatch(expected)
            for expected in expected_pose_hashes.values()
        )
    )
    if not has_modern_pose_declaration and (
        georeferencing.get("artifact_class") != "legacy_unassessed"
        or georeferencing.get("georeferencing_status") != "legacy_unassessed"
        or georeferencing.get("metric_georeferencing_claim_eligible") is not False
    ):
        raise ArtifactError("a pose without a modern declaration cannot be promoted")
    claim_eligible = bool(
        has_modern_pose_declaration
        and georeferencing.get("artifact_class") == "production"
        and georeferencing.get("georeferencing_status") == "PASSED"
        and georeferencing.get("metric_georeferencing_claim_eligible") is True
    )
    diagnostic_pose = bool(
        has_modern_pose_declaration
        and georeferencing.get("artifact_class") == "diagnostic_render_only"
        and georeferencing.get("metric_georeferencing_claim_eligible") is False
    )
    expected_tier = (
        "metric_depth_and_production_global_pose"
        if claim_eligible
        else (
            "metric_depth_and_diagnostic_global_pose"
            if diagnostic_pose
            else "metric_depth_and_unassessed_global_pose"
        )
    )
    if (
        plan.get("metric_georeferencing_claim_eligible") is not claim_eligible
        or plan.get("provisional") is not (not claim_eligible)
        or plan.get("diagnostic_render_only", False) is not diagnostic_pose
        or plan.get("evidence_tier") != expected_tier
    ):
        raise ArtifactError("tile-plan pose status was promoted or changed")
    basis = np.asarray(plan["coordinate_frame"]["R_enu_from_partition"], dtype=float)
    if basis.shape != (3, 3) or not np.isfinite(basis).all() or not np.allclose(
        basis.T @ basis, np.eye(3), atol=1e-9
    ) or not np.isclose(np.linalg.det(basis), 1.0, atol=1e-9):
        raise ArtifactError("tile-plan partition basis is invalid")
    scene = np.asarray(plan["partition"]["scene_bounds_uv_m"], dtype=float)
    if plan["partition"].get("boundary_rule") != (
        "lower_closed_upper_open_global_max_closed"
    ) or plan["partition"].get("ownership_tolerance_m") != (
        _OWNERSHIP_TOLERANCE_M
    ):
        raise ArtifactError("tile-plan ownership boundary rule is unsupported")
    tiles = plan.get("tiles")
    if scene.shape != (2, 2) or np.any(scene[1] <= scene[0]) or not isinstance(tiles, list) or not tiles:
        raise ArtifactError("tile-plan scene or tiles are invalid")
    area = 0.0
    bounds_list = []
    identifiers = []
    for tile in tiles:
        identifiers.append(tile.get("tile_id"))
        bounds = np.asarray(tile.get("core_bounds_uv_m"), dtype=float)
        context = np.asarray(tile.get("context_bounds_uv_m"), dtype=float)
        if bounds.shape != (2, 2) or np.any(bounds[1] <= bounds[0]):
            raise ArtifactError("tile core bounds are invalid")
        if np.any(bounds[0] < scene[0] - 1e-9) or np.any(bounds[1] > scene[1] + 1e-9):
            raise ArtifactError("tile core lies outside the scene")
        if context.shape != (2, 2) or np.any(context[0] > bounds[0]) or np.any(context[1] < bounds[1]):
            raise ArtifactError("tile context does not contain its core")
        halo = float(plan["resolved_policy"]["context_halo_m"])
        if not np.allclose(context, _context_bounds(bounds, halo), atol=1e-12):
            raise ArtifactError("tile context disagrees with the sealed halo")
        area += float(np.prod(bounds[1] - bounds[0]))
        bounds_list.append(bounds)
        for split in _SPLITS:
            values = tile["frame_ids"][split]
            if values != sorted(set(values)):
                raise ArtifactError(f"{tile['tile_id']} {split} IDs are not sorted unique")
    if identifiers != [f"tile-{index:04d}" for index in range(len(tiles))]:
        raise ArtifactError("tile identifiers are not canonical")
    for first in range(len(bounds_list)):
        for second in range(first + 1, len(bounds_list)):
            overlap = np.minimum(bounds_list[first][1], bounds_list[second][1]) - np.maximum(bounds_list[first][0], bounds_list[second][0])
            if np.all(overlap > 1e-10):
                raise ArtifactError("tile cores overlap")
    if not np.isclose(area, float(np.prod(scene[1] - scene[0])), rtol=1e-9, atol=1e-8):
        raise ArtifactError("tile cores do not exhaustively cover the scene")
    try:
        with np.load(root / "visibility.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ArtifactError("invalid visibility evidence") from exc
    row_fields = {
        "tile_index",
        "frame_id",
        "split_code",
        "selected",
        "core_cell_count",
        "context_cell_count",
    }
    if coverage_completion:
        row_fields |= {"visibility_selected", "coverage_selected"}
    required = row_fields | {
        "support_frame_ptr",
        "support_uv_m",
        "sampled_depth_m",
    }
    if set(arrays) != required:
        raise ArtifactError("visibility evidence schema is not exact")
    n_rows = len(arrays["tile_index"])
    if any(arrays[name].shape != (n_rows,) for name in row_fields):
        raise ArtifactError("visibility evidence row shapes disagree")
    n_frames = int(plan["summary"]["n_source_frames"])
    if n_rows != n_frames * len(tiles):
        raise ArtifactError("visibility evidence does not cover every tile/frame")
    exact_dtypes = {
        "tile_index": np.dtype("int32"),
        "frame_id": np.dtype("int64"),
        "split_code": np.dtype("uint8"),
        "selected": np.dtype("bool"),
        "core_cell_count": np.dtype("int32"),
        "context_cell_count": np.dtype("int32"),
        "support_frame_ptr": np.dtype("int64"),
        "support_uv_m": np.dtype("float64"),
        "sampled_depth_m": np.dtype("float32"),
    }
    if coverage_completion:
        exact_dtypes.update({
            "visibility_selected": np.dtype("bool"),
            "coverage_selected": np.dtype("bool"),
        })
    if any(arrays[name].dtype != dtype for name, dtype in exact_dtypes.items()):
        raise ArtifactError("visibility evidence dtypes or split codes are invalid")
    if np.any(~np.isin(arrays["split_code"], [0, 1, 2])) or np.any(
        arrays["core_cell_count"] < 0
    ) or np.any(arrays["context_cell_count"] < arrays["core_cell_count"]):
        raise ArtifactError("visibility evidence values are invalid")
    pointers = arrays["support_frame_ptr"]
    support_uv = arrays["support_uv_m"]
    sampled_depth_m = arrays["sampled_depth_m"]
    if (
        pointers.shape != (n_frames + 1,)
        or pointers[0] != 0
        or np.any(np.diff(pointers) < 0)
        or pointers[-1] != len(support_uv)
        or support_uv.ndim != 2
        or support_uv.shape[1] != 2
        or not np.isfinite(support_uv).all()
        or sampled_depth_m.ndim != 1
        or len(sampled_depth_m) < len(support_uv)
        or not np.isfinite(sampled_depth_m).all()
        or np.any(sampled_depth_m <= 0)
    ):
        raise ArtifactError("visibility support geometry is invalid")
    support = [
        support_uv[pointers[index] : pointers[index + 1]]
        for index in range(n_frames)
    ]
    if not np.all(_in_bounds(support_uv, scene)):
        raise ArtifactError("visibility support lies outside the scene")
    policy = TilePlannerPolicy(
        **{
            key: value
            for key, value in plan["resolved_policy"].items()
            if key != "forced_tile_count"
        }
    ).validate()
    thresholds = _visibility_threshold(
        np.asarray([len(points) for points in support], dtype=np.int64), policy
    )
    for index, tile in enumerate(tiles):
        rows = arrays["tile_index"] == index
        if int(rows.sum()) != n_frames or not np.array_equal(
            arrays["frame_id"][rows], np.arange(n_frames, dtype=np.int64)
        ):
            raise ArtifactError("visibility tile rows do not cover canonical frame IDs")
        bounds = np.asarray(tile["core_bounds_uv_m"], dtype=np.float64)
        expected_core, expected_context, expected_visibility = _selected(
            support, bounds, policy.context_halo_m, thresholds
        )
        if coverage_completion:
            expected_selected, expected_coverage = (
                _complete_core_training_coverage(
                    support,
                    bounds,
                    expected_visibility,
                    arrays["split_code"][rows],
                    max_training_frames=policy.max_training_frames,
                    minimum_coverage=(
                        policy.min_core_train_support_coverage
                    ),
                )
            )
        else:
            expected_selected = expected_visibility
            expected_coverage = np.zeros(n_frames, dtype=bool)
        if (
            not np.array_equal(arrays["core_cell_count"][rows], expected_core)
            or not np.array_equal(
                arrays["context_cell_count"][rows], expected_context
            )
            or not np.array_equal(arrays["selected"][rows], expected_selected)
        ):
            raise ArtifactError("visibility counts or selection semantics changed")
        if coverage_completion and (
            not np.array_equal(
                arrays["visibility_selected"][rows], expected_visibility
            )
            or not np.array_equal(
                arrays["coverage_selected"][rows], expected_coverage
            )
            or np.any(
                arrays["coverage_selected"][rows]
                & arrays["visibility_selected"][rows]
            )
        ):
            raise ArtifactError("coverage-completion selection evidence changed")
        for split in _SPLITS:
            code = _SPLIT_CODE[split]
            actual = arrays["frame_id"][rows & (arrays["split_code"] == code) & arrays["selected"]].astype(int).tolist()
            if actual != tile["frame_ids"][split]:
                raise ArtifactError("JSON and NPZ frame selections disagree")
            if len(actual) != int(tile["summary"][f"n_{split}"]):
                raise ArtifactError("tile frame-count summary disagrees")
        unique_core = _unique_support(support, bounds)
        selected_train = expected_selected & (
            arrays["split_code"][rows] == _SPLIT_CODE["train"]
        )
        train_core = _unique_support(support, bounds, selected_train)
        core_visible_train = int(
            np.sum(
                (expected_core >= thresholds)
                & (arrays["split_code"][rows] == _SPLIT_CODE["train"])
            )
        )
        summary_expected = {
            "n_core_visible_frames": int(np.sum(expected_core >= thresholds)),
            "n_core_visible_train_frames": core_visible_train,
            "n_unique_core_support_cells": int(len(unique_core)),
            "n_train_covered_core_support_cells": int(len(train_core)),
            "train_core_support_coverage": _support_coverage(
                unique_core, train_core
            ),
        }
        if coverage_completion:
            summary_expected.update({
                "n_visibility_selected_train": int(np.sum(
                    expected_visibility
                    & (
                        arrays["split_code"][rows]
                        == _SPLIT_CODE["train"]
                    )
                )),
                "n_coverage_selected_train": int(
                    np.sum(expected_coverage)
                ),
            })
        for key, value in summary_expected.items():
            stored = tile["summary"].get(key)
            if isinstance(value, float):
                agrees = isinstance(stored, (float, int)) and math.isclose(
                    float(stored), value, rel_tol=0.0, abs_tol=1e-12
                )
            else:
                agrees = stored == value
            if not agrees:
                raise ArtifactError(f"tile summary {key} disagrees with evidence")
    split_matrix = arrays["split_code"].reshape(len(tiles), n_frames)
    if np.any(split_matrix != split_matrix[0]):
        raise ArtifactError("visibility split labels differ between tiles")
    if int(arrays["selected"].sum()) != plan["summary"]["n_selected_frame_occurrences"]:
        raise ArtifactError("tile-plan selected occurrence summary disagrees")
    if coverage_completion and int(
        arrays["coverage_selected"].sum()
    ) != plan["summary"].get("n_coverage_selected_training_occurrences"):
        raise ArtifactError("tile-plan coverage-completion summary disagrees")
    selected_matrix = arrays["selected"].reshape(len(tiles), n_frames)
    maximum_train = max(len(tile["frame_ids"]["train"]) for tile in tiles)
    minimum_core_train = min(
        int(tile["summary"]["n_core_visible_train_frames"]) for tile in tiles
    )
    checks = {
        "all_frames_selected_at_least_once": bool(
            np.all(np.any(selected_matrix, axis=0))
        ),
        "all_support_inside_scene": bool(np.all(_in_bounds(support_uv, scene))),
        "tile_training_capacity": bool(
            maximum_train <= policy.max_training_frames
        ),
        "every_tile_has_training": bool(
            all(tile["frame_ids"]["train"] for tile in tiles)
        ),
        "minimum_core_training_support": bool(
            minimum_core_train >= policy.minimum_core_train_frames
        ),
        "core_train_support_coverage": bool(
            all(
                float(tile["summary"]["train_core_support_coverage"])
                >= policy.min_core_train_support_coverage
                for tile in tiles
            )
        ),
        "validation_split_preserved": True,
    }
    if (
        quality.get("checks") != checks
        or quality.get("passed") is not all(checks.values())
        or not all(checks.values())
        or quality.get("max_selected_training_frames") != maximum_train
        or quality.get("minimum_core_visible_training_frames")
        != minimum_core_train
    ):
        raise ArtifactError("tile-plan quality claims disagree with sealed evidence")
    if (
        plan["summary"].get("n_tiles") != len(tiles)
        or plan["summary"].get("n_source_frames") != n_frames
        or plan["summary"].get("n_train_source_frames")
        != int(np.sum(split_matrix[0] == _SPLIT_CODE["train"]))
        or plan["summary"].get("n_support_cell_observations")
        != len(support_uv)
        or plan["summary"].get("n_unique_support_cells")
        != len(np.unique(support_uv, axis=0))
    ):
        raise ArtifactError("tile-plan support summary disagrees")
    support_lengths = np.diff(pointers)
    depthless_ids = np.flatnonzero(support_lengths == 0)
    expected_depth_support = {
        "n_supported_frames": int(np.sum(support_lengths > 0)),
        "n_depthless_frames": int(len(depthless_ids)),
        "depthless_fraction": float(len(depthless_ids) / n_frames),
        "maximum_depthless_fraction": policy.max_depthless_frame_fraction,
        "depthless_frame_ids": depthless_ids.astype(int).tolist(),
        "fallback": "nearest_supported_temporal_neighbours",
    }
    expected_cell_stats = {
        "minimum": int(support_lengths.min()),
        "median": float(np.median(support_lengths)),
        "maximum": int(support_lengths.max()),
    }
    depths64 = sampled_depth_m.astype(np.float64)
    expected_depth_stats = {
        "median": float(np.median(depths64)),
        "p75": float(np.quantile(depths64, 0.75)),
        "maximum": float(depths64.max()),
    }
    support_quality = quality.get("support", {})
    if (
        support_quality.get("n_support_cells") != len(support_uv)
        or support_quality.get("support_cells_per_frame") != expected_cell_stats
        or support_quality.get("depth_support") != expected_depth_support
        or support_quality.get("sampled_depth_m") != expected_depth_stats
    ):
        raise ArtifactError("tile-plan support diagnostics disagree with evidence")
    sampling_record = support_quality.get("sampling_record", {})
    expected_stride = int(
        math.ceil(
            math.sqrt(
                n_frames
                * int(sampling_record.get("image_width_px", 0))
                * int(sampling_record.get("image_height_px", 0))
                / policy.max_support_samples
            )
        )
    ) if (
        isinstance(sampling_record, dict)
        and isinstance(sampling_record.get("image_width_px"), int)
        and isinstance(sampling_record.get("image_height_px"), int)
        and sampling_record.get("image_width_px", 0) > 0
        and sampling_record.get("image_height_px", 0) > 0
    ) else -1
    expected_stride = max(1, expected_stride)
    if sampling_record.get("depth_sample_stride_px") != expected_stride:
        raise ArtifactError("tile-plan depth sampling record is inconsistent")
    if segment is not None:
        reader = SegmentReader(segment).validate()
        current = _contract_inventory(reader)
        if current != inventory["segment"]["contract_files"]:
            raise ArtifactError("source segment contract changed")
        if int(reader.meta["n_frames"]) != plan["summary"]["n_source_frames"]:
            raise ArtifactError("source segment frame count changed")
        if rehash_sources:
            if _image_inventory(reader) != inventory["segment"]["image_files"]:
                raise ArtifactError("source image content changed")
            expected_depth = inventory["segment"]["depth_files"]
            current_depth = []
            for item in expected_depth:
                relative = _safe_relative(item["path"])
                path = reader.root / relative
                current_depth.append({"frame_id": item["frame_id"], "path": relative, **_file_record(path)})
            if current_depth != expected_depth:
                raise ArtifactError("source depth content changed")
    if pose_root is not None:
        pose_root = Path(pose_root)
        expected_pose = inventory["pose"]
        files = {"manifest.json": _file_record(pose_root / "manifest.json")}
        manifest_value = _read_json(pose_root / "manifest.json")
        for relative in sorted(manifest_value["files"]):
            safe = _safe_relative(relative)
            candidate = (pose_root / safe).resolve()
            resolved_pose = pose_root.resolve()
            if resolved_pose not in candidate.parents:
                raise ArtifactError("pose manifest path escapes its artifact")
            files[safe] = _file_record(candidate)
        if files != expected_pose["files"]:
            raise ArtifactError("source pose artifact changed")
        try:
            current_viewmats = np.load(pose_root / "viewmats.npy", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ArtifactError("cannot reload source pose geometry") from exc
        if pose_fingerprint(current_viewmats) != expected_pose["fingerprint"]:
            raise ArtifactError("source pose fingerprint changed")
    return plan


def load_tile_execution(
    root: str | Path,
    tile_id: str,
    reader: SegmentReader,
    cfg: Any,
    *,
    rehash_sources: bool = True,
) -> TileExecution:
    """Load one tile only after re-verifying its plan and live inputs.

    Absolute source locations are deliberately absent from the returned
    binding.  A copied plan remains portable, while the live segment and pose
    contents must still match the hashes sealed by the planner.
    """
    reader.validate()
    pose_root = None
    if pose_artifact_name(cfg) != "rtk":
        pose_root = pose_artifact_dir(reader.root, cfg)
    plan = verify_tile_plan(
        root,
        segment=reader.root,
        pose_root=pose_root,
        rehash_sources=rehash_sources,
    )
    matches = [
        (index, tile)
        for index, tile in enumerate(plan["tiles"])
        if tile["tile_id"] == str(tile_id)
    ]
    if len(matches) != 1:
        available = ", ".join(tile["tile_id"] for tile in plan["tiles"])
        raise ArtifactError(
            f"unknown tile ID {tile_id!r}; available: {available}"
        )
    index, tile = matches[0]
    train_ids = tuple(int(value) for value in tile["frame_ids"]["train"])
    val_ids = tuple(int(value) for value in tile["frame_ids"]["val"])
    test_ids = tuple(int(value) for value in tile["frame_ids"]["test"])
    if not train_ids:
        raise ArtifactError(f"{tile_id} has no training frames")
    if not val_ids:
        raise ArtifactError(
            f"{tile_id} has no held-out validation frames; training model "
            "selection would be undefined"
        )
    manifest = reader.manifest
    for split, selected in (
        ("train", train_ids), ("val", val_ids), ("test", test_ids)
    ):
        if not set(selected) <= set(int(value) for value in manifest[split]):
            raise ArtifactError(
                f"{tile_id} {split} selection escapes the source split"
            )
    return TileExecution(
        root=Path(root).expanduser().resolve(),
        tile_index=index,
        tile_plan={**tile, "plan": plan},
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )


def build_tile_plan(
    reader: SegmentReader,
    cfg: Any,
    *,
    name: str,
    output_root: str | Path,
    tile_count: int | None = None,
    allow_failed_georeferencing_for_render: bool = False,
) -> Path:
    """Build and atomically publish one non-overwriting final-mode TilePlan."""
    reader.validate()
    name = _safe_name(name)
    destination = Path(output_root).expanduser() / "tile_plan_artifacts" / name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite tile plan: {destination}")
    policy = planner_policy(cfg)
    contract_before = _contract_inventory(reader)
    pose_before = _pose_inventory(reader, cfg)
    images_before = _image_inventory(reader)
    georeferencing = pose_georeferencing_evidence(reader.root, cfg)
    require_render_permission(
        georeferencing,
        allow_failed_georeferencing_for_render=(
            allow_failed_georeferencing_for_render
        ),
    )
    viewmats, centers = load_pose_artifact(
        reader.root,
        cfg,
        allow_failed_georeferencing_for_render=(
            allow_failed_georeferencing_for_render
        ),
    )
    if (
        _contract_inventory(reader) != contract_before
        or _pose_inventory(reader, cfg) != pose_before
    ):
        raise ArtifactError("source changed while poses were being loaded")
    pose_claim_eligible = bool(
        georeferencing.get("artifact_class") == "production"
        and georeferencing.get("georeferencing_status") == "PASSED"
        and georeferencing.get("metric_georeferencing_claim_eligible") is True
    )
    diagnostic_pose = bool(
        georeferencing.get("artifact_class") == "diagnostic_render_only"
        and georeferencing.get("metric_georeferencing_claim_eligible") is False
    )
    origin, basis, _ = _partition_basis(viewmats, centers)
    support, depth_records, support_quality, sampled_depth_m = _sample_support(
        reader, viewmats, centers, basis, origin, policy
    )
    halo = policy.context_halo_m
    if halo is None:
        halo = float(np.clip(0.5 * support_quality["sampled_depth_m"]["p75"], 2.0, 8.0))
    split_codes = _split_codes(reader)
    centers_partition = ((centers - origin) @ basis)[:, :2]
    cores, thresholds = _partition(
        support, centers_partition, split_codes, policy, halo, tile_count
    )
    frame_ids = reader.frames["frame_id"].astype(np.int64)
    evidence = {name: [] for name in (
        "tile_index", "frame_id", "split_code", "selected",
        "visibility_selected", "coverage_selected",
        "core_cell_count", "context_cell_count",
    )}
    tiles = []
    selected_any = np.zeros(len(frame_ids), dtype=bool)
    for tile_index, core in enumerate(cores):
        core_count, context_count, visibility_selected = _selected(
            support, core, halo, thresholds
        )
        selected, coverage_selected = _complete_core_training_coverage(
            support,
            core,
            visibility_selected,
            split_codes,
            max_training_frames=policy.max_training_frames,
            minimum_coverage=policy.min_core_train_support_coverage,
        )
        selected_any |= selected
        selections = {
            split: frame_ids[selected & (split_codes == code)].astype(int).tolist()
            for split, code in _SPLIT_CODE.items()
        }
        unique_core = _unique_support(support, core)
        selected_train = selected & (split_codes == _SPLIT_CODE["train"])
        train_core = _unique_support(support, core, selected_train)
        train_support_coverage = _support_coverage(unique_core, train_core)
        core_visible_train = int(
            np.sum(
                (core_count >= thresholds)
                & (split_codes == _SPLIT_CODE["train"])
            )
        )
        tiles.append({
            "tile_id": f"tile-{tile_index:04d}",
            "core_bounds_uv_m": core.tolist(),
            "context_bounds_uv_m": _context_bounds(core, halo).tolist(),
            "frame_ids": selections,
            "summary": {
                "n_train": len(selections["train"]),
                "n_val": len(selections["val"]),
                "n_test": len(selections["test"]),
                "n_visibility_selected_train": int(np.sum(
                    visibility_selected
                    & (split_codes == _SPLIT_CODE["train"])
                )),
                "n_coverage_selected_train": int(np.sum(coverage_selected)),
                "n_core_visible_frames": int(np.sum(core_count >= thresholds)),
                "n_core_visible_train_frames": core_visible_train,
                "n_unique_core_support_cells": int(len(unique_core)),
                "n_train_covered_core_support_cells": int(len(train_core)),
                "train_core_support_coverage": train_support_coverage,
            },
        })
        for key, values in (
            ("tile_index", np.full(len(frame_ids), tile_index, dtype=np.int32)),
            ("frame_id", frame_ids),
            ("split_code", split_codes),
            ("selected", selected),
            ("visibility_selected", visibility_selected),
            ("coverage_selected", coverage_selected),
            ("core_cell_count", core_count),
            ("context_cell_count", context_count),
        ):
            evidence[key].append(values)
    evidence_arrays = {
        key: np.concatenate(values) for key, values in evidence.items()
    }
    all_support = np.concatenate(support)
    unique_scene_support = np.unique(all_support, axis=0)
    support_lengths = np.asarray([len(points) for points in support], dtype=np.int64)
    support_frame_ptr = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(support_lengths)]
    )
    evidence_arrays["support_frame_ptr"] = support_frame_ptr
    evidence_arrays["support_uv_m"] = all_support.astype(np.float64, copy=False)
    evidence_arrays["sampled_depth_m"] = sampled_depth_m
    scene = np.asarray([
        np.min([tile["core_bounds_uv_m"][0] for tile in tiles], axis=0),
        np.max([tile["core_bounds_uv_m"][1] for tile in tiles], axis=0),
    ])
    inventory = {
        "schema_version": SCHEMA_VERSION,
        "segment": {
            "contract_files": contract_before,
            "image_files": images_before,
            "depth_files": depth_records,
        },
        "pose": {
            **pose_before,
            "name": pose_artifact_name(cfg),
            "fingerprint": pose_fingerprint(viewmats),
        },
    }
    plan = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "rtk_splat_tile_plan",
        "name": name,
        "evidence_tier": (
            "metric_depth_and_production_global_pose"
            if pose_claim_eligible
            else (
                "metric_depth_and_diagnostic_global_pose"
                if diagnostic_pose
                else "metric_depth_and_unassessed_global_pose"
            )
        ),
        "provisional": not pose_claim_eligible,
        "metric_georeferencing_claim_eligible": pose_claim_eligible,
        **({"diagnostic_render_only": True} if diagnostic_pose else {}),
        "coordinate_frame": {
            "source": reader.meta["coordinate_frame"],
            "partition_origin_enu_m": origin.tolist(),
            "R_enu_from_partition": basis.tolist(),
            "ownership_axes": ["u", "v"],
        },
        "partition": {
            "method": "recursive_visibility_workload_v1",
            "scene_bounds_uv_m": scene.tolist(),
            "boundary_rule": "lower_closed_upper_open_global_max_closed",
            "ownership_tolerance_m": _OWNERSHIP_TOLERANCE_M,
            "z_ownership": "unbounded",
            "context_halo_m": halo,
        },
        "selection": {
            "primary": "configured_visibility_threshold_v1",
            "coverage_completion": _COVERAGE_COMPLETION,
        },
        "source_binding": {
            "inventory_sha256": canonical_hash(inventory),
            "pose_fingerprint": pose_fingerprint(viewmats),
        },
        "resolved_policy": {
            **_plain(policy.__dict__),
            "context_halo_m": halo,
            "forced_tile_count": tile_count,
        },
        "tiles": tiles,
        "summary": {
            "n_tiles": len(tiles),
            "n_source_frames": len(frame_ids),
            "n_train_source_frames": int(np.sum(split_codes == 0)),
            "n_selected_frame_occurrences": int(evidence_arrays["selected"].sum()),
            "n_coverage_selected_training_occurrences": int(
                evidence_arrays["coverage_selected"].sum()
            ),
            "n_support_cell_observations": len(all_support),
            "n_unique_support_cells": len(unique_scene_support),
        },
    }
    max_train = max(tile["summary"]["n_train"] for tile in tiles)
    minimum_core_train = min(
        tile["summary"]["n_core_visible_train_frames"] for tile in tiles
    )
    quality = {
        "schema_version": SCHEMA_VERSION,
        "passed": bool(selected_any.all() and max_train <= policy.max_training_frames),
        "checks": {
            "all_frames_selected_at_least_once": bool(selected_any.all()),
            "all_support_inside_scene": bool(np.all(_in_bounds(all_support, scene))),
            "tile_training_capacity": bool(max_train <= policy.max_training_frames),
            "every_tile_has_training": bool(all(tile["summary"]["n_train"] > 0 for tile in tiles)),
            "minimum_core_training_support": bool(
                minimum_core_train >= policy.minimum_core_train_frames
            ),
            "core_train_support_coverage": bool(all(
                tile["summary"]["train_core_support_coverage"]
                >= policy.min_core_train_support_coverage
                for tile in tiles
            )),
            "validation_split_preserved": True,
        },
        "max_selected_training_frames": max_train,
        "minimum_core_visible_training_frames": minimum_core_train,
        "capacity_scope": (
            "training-view presentation budget; tile cloud size and measured "
            "VRAM remain downstream acceptance gates"
        ),
        "unselected_frame_ids": frame_ids[~selected_any].astype(int).tolist(),
        "support": support_quality,
    }
    if not all(quality["checks"].values()):
        failed = [key for key, passed in quality["checks"].items() if not passed]
        raise ArtifactError("tile-plan quality gates failed: " + ", ".join(failed))
    from rtk_splat.core.runtime_resolution import configuration_evidence
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "segment_locator": str(reader.root.resolve()),
        "pose_locator": (
            "segment:initial_pose"
            if pose_artifact_name(cfg) == "rtk"
            else str(pose_artifact_dir(reader.root, cfg).resolve())
        ),
        "georeferencing": georeferencing,
        **(
            {"allow_failed_georeferencing_for_render": True}
            if allow_failed_georeferencing_for_render
            else {}
        ),
        "configuration": configuration_evidence(cfg),
        "package": collect_package_state(),
        "source_inventory_sha256": canonical_hash(inventory),
    }
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{name}.writing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        _write_json(staging / "tile_plan.json", plan)
        _deterministic_npz(staging / "visibility.npz", evidence_arrays)
        _write_json(staging / "source_inventory.json", inventory)
        _write_json(staging / "quality.json", quality)
        _write_json(staging / "provenance.json", provenance)
        _write(staging / "plan.svg", _svg(plan, centers, support))
        contract_after = _contract_inventory(reader)
        pose_after = _pose_inventory(reader, cfg)
        depth_after = [
            {"frame_id": item["frame_id"], "path": item["path"], **_file_record(reader.root / item["path"])}
            for item in depth_records
        ]
        images_after = _image_inventory(reader)
        if (
            contract_after != contract_before
            or pose_after != pose_before
            or depth_after != depth_records
            or images_after != images_before
        ):
            raise ArtifactError("source changed while tile plan was being built")
        _write_json(staging / "manifest.json", _manifest(staging))
        verify_tile_plan(staging)
        publish_directory_noreplace(staging, destination)
        verify_tile_plan(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
