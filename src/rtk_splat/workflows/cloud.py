"""Initial point-cloud workflow built from depth and a sealed pose artifact."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rtk_splat.core.cloud import backproject, to_world, voxel_downsample
from rtk_splat.backends.pose_evidence import (
    canonical_georeferencing_json,
    pose_georeferencing_evidence,
    require_render_permission,
)
from rtk_splat.core.pose_artifacts import (
    cloud_path,
    load_pose_artifact,
    pose_artifact_name,
    pose_fingerprint,
    tile_cloud_path,
    verify_cloud_matches_poses,
    verify_cloud_tile_binding,
)
from rtk_splat.core.segment import SegmentReader
from rtk_splat.workflows.runtime_config import resolve_cloud_max_points
from rtk_splat.workflows.tiles import TileExecution


@dataclass(frozen=True)
class PackedTileContextMasks:
    """Verified packed masks; unpack only the frame used by this iteration."""

    frame_ids: np.ndarray
    shape: tuple[int, int]
    packed: np.ndarray

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, frame_id: int) -> np.ndarray:
        position = int(np.searchsorted(self.frame_ids, int(frame_id)))
        if position >= len(self.frame_ids) or self.frame_ids[position] != frame_id:
            raise KeyError(f"tile cloud has no context mask for frame {frame_id}")
        count = self.shape[0] * self.shape[1]
        return np.unpackbits(
            self.packed[position], count=count, bitorder="little"
        ).reshape(self.shape).astype(bool, copy=False)


def _canonical_json(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tile_cloud_manifest_path(cloud_file: Path) -> Path:
    return cloud_file.with_name("manifest.json")


def _publish_json_noreplace(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        payload = json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        with temporary.open("xb") as stream:
            stream.write(payload.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tile_cloud_manifest(
    cloud_file: Path,
    execution: TileExecution,
    point_count: int,
    mask_count: int,
    mask_sha256: str,
) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "rtk_splat_tile_cloud",
        "file": cloud_file.name,
        "sha256": _sha256_file(cloud_file),
        "size_bytes": cloud_file.stat().st_size,
        "point_count": int(point_count),
        "mask_count": int(mask_count),
        "tile_context_mask_sha256": str(mask_sha256),
        "tile_plan": execution.binding,
    }


def _mask_sha256(
    frame_ids: np.ndarray, shape: np.ndarray, packed: np.ndarray
) -> str:
    digest = hashlib.sha256()
    for value in (frame_ids.astype("<i8", copy=False),
                  shape.astype("<i8", copy=False), packed):
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _context_mask(
    depth: np.ndarray,
    valid: np.ndarray,
    intrinsics: dict[str, float],
    viewmat: np.ndarray,
    execution: TileExecution,
) -> np.ndarray:
    """Full-resolution pixels whose measured 3-D point is in tile context."""
    valid = np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 0)
    rows, cols = np.nonzero(valid)
    result = np.zeros(valid.shape, dtype=bool)
    if not len(rows):
        return result
    distance = depth[rows, cols].astype(np.float64, copy=False)
    points = np.column_stack((
        (cols - intrinsics["cx"]) / intrinsics["fx"] * distance,
        (rows - intrinsics["cy"]) / intrinsics["fy"] * distance,
        distance,
    ))
    world = to_world(points, viewmat)
    binding = execution.binding
    origin = np.asarray(binding["partition_origin_enu_m"], dtype=np.float64)
    basis = np.asarray(binding["R_enu_from_partition"], dtype=np.float64)
    bounds = np.asarray(binding["context_bounds_uv_m"], dtype=np.float64)
    uv = ((world - origin) @ basis)[:, :2]
    inside = np.all((uv >= bounds[0]) & (uv <= bounds[1]), axis=1)
    result[rows[inside], cols[inside]] = True
    return result


def load_tile_context_masks(
    cloud_file: str | Path,
    execution: TileExecution,
) -> PackedTileContextMasks:
    """Load and verify packed per-frame context masks from a tile cloud."""
    cloud_file = Path(cloud_file)
    verify_cloud_tile_binding(cloud_file, execution.binding)
    try:
        with np.load(cloud_file, allow_pickle=False) as cloud:
            frame_ids = np.asarray(cloud["tile_mask_frame_ids"], dtype=np.int64)
            shape = np.asarray(cloud["tile_context_mask_shape"], dtype=np.int64)
            packed = np.asarray(cloud["tile_context_mask_bits"], dtype=np.uint8)
            stored_hash = str(cloud["tile_context_mask_sha256"].item())
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"invalid tile context masks in {cloud_file}") from exc
    expected_ids = np.asarray(
        sorted(set(execution.train_ids) | set(execution.val_ids)),
        dtype=np.int64,
    )
    if (
        shape.shape != (2,)
        or np.any(shape <= 0)
        or not np.array_equal(frame_ids, expected_ids)
        or packed.shape != (
            len(frame_ids), (int(shape[0]) * int(shape[1]) + 7) // 8
        )
        or _mask_sha256(frame_ids, shape, packed) != stored_hash
    ):
        raise ValueError("tile context mask identity or shape is invalid")
    return PackedTileContextMasks(
        frame_ids=frame_ids,
        shape=(int(shape[0]), int(shape[1])),
        packed=packed,
    )


def verify_tile_cloud(
    cloud_file: str | Path,
    execution: TileExecution,
    viewmats: np.ndarray,
    *,
    georeferencing: dict | None = None,
    allow_failed_georeferencing_for_render: bool = False,
) -> tuple[int, int]:
    """Verify a completed tile cloud strongly enough to skip rebuilding it."""
    cloud_file = Path(cloud_file)
    manifest_path = _tile_cloud_manifest_path(cloud_file)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"tile cloud has no valid terminal manifest: {cloud_file}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != "rtk_splat_tile_cloud"
        or manifest.get("file") != cloud_file.name
        or manifest.get("sha256") != _sha256_file(cloud_file)
        or manifest.get("size_bytes") != cloud_file.stat().st_size
        or manifest.get("tile_plan") != execution.binding
    ):
        raise ValueError("tile cloud terminal manifest verification failed")
    verify_cloud_matches_poses(cloud_file, viewmats, require_fingerprint=True)
    verify_cloud_tile_binding(cloud_file, execution.binding)
    if georeferencing is not None:
        from rtk_splat.backends.pose_evidence import cloud_georeferencing_evidence
        cloud_georeferencing_evidence(
            cloud_file,
            georeferencing,
            allow_failed_georeferencing_for_render=(
                allow_failed_georeferencing_for_render
            ),
        )
    masks = load_tile_context_masks(cloud_file, execution)
    with np.load(cloud_file, allow_pickle=False) as cloud:
        xyz = np.asarray(cloud["xyz"], dtype=np.float64)
        rgb = np.asarray(cloud["rgb"])
        mask_sha256 = str(cloud["tile_context_mask_sha256"].item())
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or not len(xyz)
        or not np.isfinite(xyz).all()
        or rgb.shape != (len(xyz), 3)
        or rgb.dtype != np.uint8
        or manifest.get("point_count") != len(xyz)
        or manifest.get("mask_count") != len(masks)
        or manifest.get("tile_context_mask_sha256") != mask_sha256
    ):
        raise ValueError("tile cloud point arrays are invalid")
    binding = execution.binding
    origin = np.asarray(binding["partition_origin_enu_m"], dtype=np.float64)
    basis = np.asarray(binding["R_enu_from_partition"], dtype=np.float64)
    bounds = np.asarray(binding["context_bounds_uv_m"], dtype=np.float64)
    uv = ((xyz - origin) @ basis)[:, :2]
    if not np.all((uv >= bounds[0] - 1e-6) & (uv <= bounds[1] + 1e-6)):
        raise ValueError("tile cloud contains points outside its context bounds")
    return len(xyz), len(masks)


def construct_initial_cloud(
    reader: SegmentReader,
    cfg,
    *,
    allow_failed_georeferencing_for_render: bool = False,
    tile_execution: TileExecution | None = None,
) -> tuple[Path, int]:
    """Publish a pose-bound cloud without hiding orchestration in the CLI."""
    frames = reader.frames
    georeferencing = pose_georeferencing_evidence(reader.root, cfg)
    require_render_permission(
        georeferencing,
        allow_failed_georeferencing_for_render=(
            allow_failed_georeferencing_for_render
        ),
    )
    resolve_cloud_max_points(cfg)
    viewmats, _ = load_pose_artifact(
        reader.root,
        cfg,
        allow_failed_georeferencing_for_render=(
            allow_failed_georeferencing_for_render
        ),
    )
    camera = reader.calibration["cameras"]["left"]
    k = np.asarray(camera["K"], dtype=float)
    intrinsics = {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }
    if "depth_path" not in frames or any(
        not str(path) for path in frames["depth_path"]
    ):
        raise RuntimeError("cloud construction requires depth for every frame")
    points, colours = [], []
    train_ids = (
        reader.manifest["train"]
        if tile_execution is None
        else list(tile_execution.train_ids)
    )
    mask_ids = (
        []
        if tile_execution is None
        else sorted(set(tile_execution.train_ids) | set(tile_execution.val_ids))
    )
    mask_shape = np.asarray(
        [int(camera["height"]), int(camera["width"])], dtype=np.int64
    )
    mask_bits = None
    if tile_execution is not None:
        mask_bits = np.empty(
            (
                len(mask_ids),
                (int(mask_shape[0]) * int(mask_shape[1]) + 7) // 8,
            ),
            dtype=np.uint8,
        )
    train_set = set(int(value) for value in train_ids)
    for mask_row, frame_id in enumerate(
        mask_ids if tile_execution is not None else train_ids
    ):
        index = int(frame_id)
        with np.load(reader.root / str(frames["depth_path"][index])) as depth:
            depth_values = depth["depth"].astype(np.float32)
            valid = depth["valid"].astype(bool)
            if tile_execution is not None:
                context = _context_mask(
                    depth_values,
                    valid,
                    intrinsics,
                    viewmats[index],
                    tile_execution,
                )
                mask_bits[mask_row] = np.packbits(
                    context.reshape(-1), bitorder="little"
                )
                valid &= context
            if index not in train_set:
                continue
            image = cv2.imread(
                str(reader.root / str(frames["left_image_path"][index]))
            )
            if image is None:
                raise RuntimeError(f"cannot read frame {index}")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            xyz, colour = backproject(
                depth_values,
                valid,
                rgb,
                intrinsics,
                int(cfg.cloud.pixel_stride),
            )
        if len(xyz):
            points.append(to_world(xyz, viewmats[index]))
            colours.append(colour)
    if not points:
        raise RuntimeError("cloud construction retained no points")
    xyz = np.concatenate(points).astype(np.float32)
    rgb = np.concatenate(colours)
    xyz, rgb = voxel_downsample(
        xyz, rgb, float(cfg.cloud.voxel_m), int(cfg.cloud.max_points)
    )
    output = (
        cloud_path(reader.root, cfg)
        if tile_execution is None
        else tile_cloud_path(
            reader.root,
            cfg,
            tile_execution.binding["name"],
            tile_execution.tile_id,
        )
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cloud artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            values = dict(
                xyz=xyz,
                rgb=rgb,
                pose_fingerprint=np.asarray(pose_fingerprint(viewmats)),
                pose_artifact=np.asarray(pose_artifact_name(cfg)),
                georeferencing_json=np.asarray(
                    canonical_georeferencing_json(georeferencing)
                ),
                artifact_class=np.asarray(
                    georeferencing["artifact_class"]
                ),
                georeferencing_status=np.asarray(
                    georeferencing["georeferencing_status"]
                ),
                metric_georeferencing_claim_eligible=np.asarray(
                    georeferencing["metric_georeferencing_claim_eligible"],
                    dtype=np.bool_,
                ),
                pose_quality_sha256=np.asarray(
                    georeferencing["pose_quality_sha256"] or ""
                ),
                pose_manifest_sha256=np.asarray(
                    georeferencing["pose_manifest_sha256"] or ""
                ),
                pose_georeferencing_sha256=np.asarray(
                    georeferencing["pose_georeferencing_sha256"] or ""
                ),
            )
            if tile_execution is not None:
                frame_ids = np.asarray(mask_ids, dtype=np.int64)
                assert mask_bits is not None
                values.update(
                    tile_plan_json=np.asarray(
                        _canonical_json(tile_execution.binding)
                    ),
                    tile_mask_frame_ids=frame_ids,
                    tile_context_mask_shape=mask_shape,
                    tile_context_mask_bits=mask_bits,
                    tile_context_mask_sha256=np.asarray(
                        _mask_sha256(frame_ids, mask_shape, mask_bits)
                    ),
                )
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    if tile_execution is not None:
        with np.load(output, allow_pickle=False) as cloud:
            mask_sha256 = str(cloud["tile_context_mask_sha256"].item())
        _publish_json_noreplace(
            _tile_cloud_manifest_path(output),
            _tile_cloud_manifest(
                output,
                tile_execution,
                len(xyz),
                len(mask_ids),
                mask_sha256,
            ),
        )
        verify_tile_cloud(
            output,
            tile_execution,
            viewmats,
            georeferencing=georeferencing,
            allow_failed_georeferencing_for_render=(
                allow_failed_georeferencing_for_render
            ),
        )
    return output, len(xyz)
