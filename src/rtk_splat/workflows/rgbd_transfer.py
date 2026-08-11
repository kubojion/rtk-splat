"""Derive an RGB-D segment from calibrated RGB and accepted stereo poses.

This workflow deliberately sits after mapping.  It does not estimate poses,
change the mapper, or consume IMU data.  A validated metric optical-z image is
motion-compensated from the source camera into each RGB observation, while the
accepted and RTK-initial poses are interpolated and transformed by one sealed
camera calibration.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from rtk_splat.adapters.rgb_observations import load_rectified_rgb_observations
from rtk_splat.backends.mapper_config import MapperConfig
from rtk_splat.backends.pose_evidence import verify_pose_georeferencing_artifact
from rtk_splat.backends.quality import rtk_residual_quality
from rtk_splat.core.segment import (
    CONTRACT_VERSION,
    POSITION_QUALITY_VOCABULARY,
    SegmentReader,
    SegmentWriter,
    publish_directory_noreplace,
)
from rtk_splat.frontends.artifact import (
    ArtifactError,
    canonical_hash,
    image_inventory,
    sha256_file,
)


__all__ = ["RgbdTransferConfig", "derive_rgbd_segment"]


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_DIAGNOSTIC_STATUS = {
    "artifact_class": "diagnostic_render_only",
    "georeferencing_status": "PASSED",
    "metric_georeferencing_claim_eligible": False,
    "diagnostic_export_requested": True,
    "diagnostic_export_override_used": False,
}


@dataclass(frozen=True)
class RgbdTransferConfig:
    """Explicit gates for one immutable camera-to-camera transfer."""

    max_depth_sync_residual_ns: int
    max_pose_interpolation_gap_ns: int
    min_depth_rgb_association_fraction: float
    min_projected_depth_coverage_fraction: float
    min_projected_depth_retained_fraction: float
    position_sigma_floor_m: float = 0.01
    depth_fill_min_valid_neighbors: int = 3
    depth_fill_max_spread_abs_m: float = 0.05
    depth_fill_max_spread_rel: float = 0.02
    held_out_registration_evidence: str | None = None
    min_held_out_correspondences: int = 20
    max_held_out_median_reprojection_error_px: float = 2.0
    max_held_out_p95_reprojection_error_px: float = 5.0
    clock_observability_probe_ns: int = 10_000_000
    min_clock_observability_px_per_s: float = 0.1
    max_extrinsic_observability_condition_number: float = 1_000_000.0

    def __post_init__(self) -> None:
        for name in (
            "max_depth_sync_residual_ns",
            "max_pose_interpolation_gap_ns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not math.isfinite(self.position_sigma_floor_m) or (
            self.position_sigma_floor_m <= 0
        ):
            raise ValueError("position_sigma_floor_m must be finite and positive")
        for name in (
            "min_depth_rgb_association_fraction",
            "min_projected_depth_coverage_fraction",
            "min_projected_depth_retained_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if (
            isinstance(self.depth_fill_min_valid_neighbors, bool)
            or not isinstance(self.depth_fill_min_valid_neighbors, int)
            or not 1 <= self.depth_fill_min_valid_neighbors <= 8
        ):
            raise ValueError(
                "depth_fill_min_valid_neighbors must be an integer in [1, 8]"
            )
        for name in ("depth_fill_max_spread_abs_m", "depth_fill_max_spread_rel"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.min_held_out_correspondences, bool)
            or not isinstance(self.min_held_out_correspondences, int)
            or self.min_held_out_correspondences < 3
        ):
            raise ValueError("min_held_out_correspondences must be an integer >= 3")
        if (
            isinstance(self.clock_observability_probe_ns, bool)
            or not isinstance(self.clock_observability_probe_ns, int)
            or self.clock_observability_probe_ns <= 0
        ):
            raise ValueError("clock_observability_probe_ns must be a positive integer")
        for name in (
            "max_held_out_median_reprojection_error_px",
            "max_held_out_p95_reprojection_error_px",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            self.max_held_out_p95_reprojection_error_px
            < self.max_held_out_median_reprojection_error_px
        ):
            raise ValueError("held-out p95 gate cannot be tighter than median gate")
        if (
            not math.isfinite(self.min_clock_observability_px_per_s)
            or self.min_clock_observability_px_per_s <= 0
        ):
            raise ValueError(
                "min_clock_observability_px_per_s must be finite and positive"
            )
        if (
            not math.isfinite(
                self.max_extrinsic_observability_condition_number
            )
            or self.max_extrinsic_observability_condition_number <= 1.0
        ):
            raise ValueError(
                "max_extrinsic_observability_condition_number must exceed one"
            )
        if self.held_out_registration_evidence is not None and (
            not isinstance(self.held_out_registration_evidence, str)
            or not self.held_out_registration_evidence.strip()
        ):
            raise ValueError(
                "held_out_registration_evidence must be a non-empty path string"
            )


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON artifact is not an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
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


def _rigid(value: Any, label: str) -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{label} must be numeric") from exc
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5
        )
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-5)
    ):
        raise ArtifactError(f"{label} must be a finite proper rigid 4x4 transform")
    return transform


def _associate_depth_to_rgb(
    depth_ns: np.ndarray, corrected_rgb_ns: np.ndarray, tolerance_ns: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Select at most one nearest RGB for each source depth timestamp.

    Ties choose the earlier RGB. If two depth frames choose the same RGB, the
    smaller absolute residual wins (then the earlier depth index). Losers and
    out-of-gate depth frames remain explicit missing evidence.
    """
    depth = np.asarray(depth_ns, dtype=np.int64)
    rgb = np.asarray(corrected_rgb_ns, dtype=np.int64)
    if (
        depth.ndim != 1
        or rgb.ndim != 1
        or not len(depth)
        or not len(rgb)
        or np.any(np.diff(depth) <= 0)
        or np.any(np.diff(rgb) <= 0)
    ):
        raise ArtifactError("depth and corrected RGB timestamps must increase")
    insertion = np.searchsorted(rgb, depth, side="left")
    before = np.clip(insertion - 1, 0, len(rgb) - 1)
    after = np.clip(insertion, 0, len(rgb) - 1)
    candidate = np.where(
        np.abs(rgb[before] - depth) <= np.abs(rgb[after] - depth), before, after
    ).astype(np.int64)
    rgb_minus_depth = rgb[candidate] - depth
    in_gate = np.abs(rgb_minus_depth) <= tolerance_ns
    selected = np.zeros(len(depth), dtype=bool)
    duplicate_losers: list[int] = []
    for rgb_index in np.unique(candidate[in_gate]):
        contenders = np.flatnonzero(in_gate & (candidate == rgb_index))
        winner = min(
            contenders.tolist(),
            key=lambda index: (abs(int(rgb_minus_depth[index])), index),
        )
        selected[winner] = True
        duplicate_losers.extend(
            int(index) for index in contenders if int(index) != winner
        )
    source_indices = np.flatnonzero(selected).astype(np.int64)
    rgb_indices = candidate[source_indices]
    if len(source_indices) < 3 or np.any(np.diff(rgb_indices) <= 0):
        raise ArtifactError(
            "depth association failed: fewer than three unique in-gate "
            "RGB-D associations"
        )
    missing_gate = np.flatnonzero(~in_gate).astype(np.int64)
    missing_duplicate = np.asarray(sorted(duplicate_losers), dtype=np.int64)
    used_rgb = np.zeros(len(rgb), dtype=bool)
    used_rgb[rgb_indices] = True
    report = {
        "method": "source_depth_to_nearest_corrected_rgb_unique_v1",
        "residual_convention": "corrected_rgb_timestamp_minus_source_depth_timestamp",
        "tolerance_ns": tolerance_ns,
        "n_source_depth": len(depth),
        "n_rgb_observations": len(rgb),
        "n_matched": len(source_indices),
        "n_missing_source_depth": int(len(depth) - len(source_indices)),
        "n_missing_outside_gate": len(missing_gate),
        "n_missing_duplicate_rgb_conflict": len(missing_duplicate),
        "n_unused_rgb": int((~used_rgb).sum()),
        "median_abs_residual_ns": float(
            np.median(np.abs(rgb_minus_depth[source_indices]))
        ),
        "maximum_abs_residual_ns": int(
            np.max(np.abs(rgb_minus_depth[source_indices]))
        ),
        "missing_outside_gate_source_indices": missing_gate.tolist(),
        "missing_duplicate_source_indices": missing_duplicate.tolist(),
        "unused_rgb_indices": np.flatnonzero(~used_rgb).astype(int).tolist(),
    }
    return (
        source_indices,
        rgb_indices,
        rgb_minus_depth[source_indices].astype(np.int64),
        report,
    )


def _source_depth_timeline(source_frames: Mapping[str, np.ndarray]) -> np.ndarray:
    """Require explicit depth acquisition time and its signed sync evidence."""
    required = {"timestamp_ns", "depth_timestamp_ns", "depth_sync_residual_ns"}
    missing = sorted(required - set(source_frames))
    if missing:
        raise ArtifactError(
            "source depth timing evidence is incomplete: " + ", ".join(missing)
        )
    frame_ns = np.asarray(source_frames["timestamp_ns"])
    depth_ns = np.asarray(source_frames["depth_timestamp_ns"])
    residual_ns = np.asarray(source_frames["depth_sync_residual_ns"])
    if (
        depth_ns.dtype != np.dtype(np.int64)
        or residual_ns.dtype != np.dtype(np.int64)
        or depth_ns.shape != frame_ns.shape
        or residual_ns.shape != frame_ns.shape
        or np.any(np.diff(depth_ns) <= 0)
        or not np.array_equal(residual_ns, depth_ns - frame_ns)
    ):
        raise ArtifactError(
            "source depth timestamps must be strictly increasing int64 and "
            "depth_sync_residual_ns must equal depth minus frame timestamp"
        )
    if "depth_log_timestamp_ns" in source_frames:
        log_ns = np.asarray(source_frames["depth_log_timestamp_ns"])
        if log_ns.dtype != np.dtype(np.int64) or log_ns.shape != frame_ns.shape:
            raise ArtifactError("source depth log timestamps are invalid")
    return depth_ns.astype(np.int64, copy=False)


def _projected_depth_quality(
    source_valid_counts: np.ndarray,
    scattered_valid_counts: np.ndarray,
    filled_valid_counts: np.ndarray,
    output_pixels_per_frame: int,
    config: RgbdTransferConfig,
) -> dict[str, Any]:
    """Gate the projected supervision area.

    Coverage is judged on the artifact actually written (after lattice fill);
    retention is judged on the physical scatter alone, so the fill can never
    mask genuine projection loss.
    """
    source_counts = np.asarray(source_valid_counts, dtype=np.int64)
    scattered_counts = np.asarray(scattered_valid_counts, dtype=np.int64)
    filled_counts = np.asarray(filled_valid_counts, dtype=np.int64)
    if (
        source_counts.ndim != 1
        or scattered_counts.shape != source_counts.shape
        or filled_counts.shape != source_counts.shape
        or not len(source_counts)
        or np.any(source_counts < 0)
        or np.any(scattered_counts < 0)
        or np.any(scattered_counts > source_counts)
        or np.any(filled_counts < scattered_counts)
        or np.any(filled_counts > output_pixels_per_frame)
        or output_pixels_per_frame <= 0
    ):
        raise ArtifactError("projected depth count evidence is invalid")
    total_source = int(source_counts.sum())
    total_scattered = int(scattered_counts.sum())
    total_filled = int(filled_counts.sum())
    coverage = total_filled / float(len(source_counts) * output_pixels_per_frame)
    retained = total_scattered / float(total_source) if total_source else 0.0
    per_frame_coverage = filled_counts / float(output_pixels_per_frame)
    per_frame_retained = np.divide(
        scattered_counts,
        source_counts,
        out=np.zeros(len(source_counts), dtype=np.float64),
        where=source_counts > 0,
    )
    passed = (
        coverage >= config.min_projected_depth_coverage_fraction
        and retained >= config.min_projected_depth_retained_fraction
    )
    report = {
        "method": "aggregate_valid_pixel_coverage_and_retention_v2_lattice_fill",
        "n_frames": len(source_counts),
        "output_pixels_per_frame": output_pixels_per_frame,
        "total_source_valid_pixels": total_source,
        "total_scattered_valid_pixels": total_scattered,
        "total_projected_valid_pixels": total_filled,
        "projected_depth_coverage_fraction": coverage,
        "scattered_depth_coverage_fraction": (
            total_scattered / float(len(source_counts) * output_pixels_per_frame)
        ),
        "projected_depth_retained_fraction": retained,
        "minimum_projected_depth_coverage_fraction": (
            config.min_projected_depth_coverage_fraction
        ),
        "minimum_projected_depth_retained_fraction": (
            config.min_projected_depth_retained_fraction
        ),
        "median_frame_coverage_fraction": float(np.median(per_frame_coverage)),
        "minimum_frame_coverage_fraction": float(np.min(per_frame_coverage)),
        "median_frame_retained_fraction": float(np.median(per_frame_retained)),
        "minimum_frame_retained_fraction": float(np.min(per_frame_retained)),
        "n_zero_projected_frames": int(np.count_nonzero(filled_counts == 0)),
        "resampling_fill": {
            "method": "single_pass_3x3_median_fill_v1",
            "min_valid_neighbors": config.depth_fill_min_valid_neighbors,
            "max_spread_abs_m": config.depth_fill_max_spread_abs_m,
            "max_spread_rel": config.depth_fill_max_spread_rel,
            "total_filled_pixels": total_filled - total_scattered,
            "filled_fraction_of_output": (
                (total_filled - total_scattered)
                / float(len(source_counts) * output_pixels_per_frame)
            ),
        },
        "passed": passed,
    }
    if not passed:
        raise ArtifactError(
            "projected depth gate failed: coverage "
            f"{coverage:.6f} (minimum "
            f"{config.min_projected_depth_coverage_fraction:.6f}), retained "
            f"{retained:.6f} (minimum "
            f"{config.min_projected_depth_retained_fraction:.6f})"
        )
    return report


def _interpolate_viewmats(
    timestamps_ns: np.ndarray,
    viewmats: np.ndarray,
    query_ns: np.ndarray,
    *,
    max_gap_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate OpenCV world-to-camera poses without epoch float loss."""
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    query = np.asarray(query_ns, dtype=np.int64)
    poses = np.asarray(viewmats, dtype=np.float64)
    if (
        timestamps.shape != (len(poses),)
        or poses.shape[1:] != (4, 4)
        or np.any(np.diff(timestamps) <= 0)
        or not np.isfinite(poses).all()
    ):
        raise ArtifactError("pose interpolation inputs are invalid")
    if np.any(query < timestamps[0]) or np.any(query > timestamps[-1]):
        raise ArtifactError("RGB pose query lies outside the accepted pose timeline")
    upper = np.searchsorted(timestamps, query, side="left")
    upper = np.clip(upper, 0, len(timestamps) - 1)
    exact = timestamps[upper] == query
    lower = np.where(exact, upper, upper - 1).astype(np.int64)
    gaps = timestamps[upper] - timestamps[lower]
    if np.any(~exact & (gaps > max_gap_ns)):
        raise ArtifactError(
            "RGB pose interpolation crosses a gap larger than "
            f"{max_gap_ns} ns"
        )
    alpha = np.zeros(len(query), dtype=np.float64)
    interpolate = ~exact
    alpha[interpolate] = (
        (query[interpolate] - timestamps[lower[interpolate]])
        / gaps[interpolate].astype(np.float64)
    )
    c2w = np.linalg.inv(poses)
    output_c2w = np.repeat(np.eye(4)[None], len(query), axis=0)
    output_c2w[:, :3, 3] = (
        (1.0 - alpha[:, None]) * c2w[lower, :3, 3]
        + alpha[:, None] * c2w[upper, :3, 3]
    )
    output_c2w[exact, :3, :3] = c2w[upper[exact], :3, :3]
    for index in np.flatnonzero(interpolate):
        rotations = Rotation.from_matrix(
            np.stack((c2w[lower[index], :3, :3], c2w[upper[index], :3, :3]))
        )
        output_c2w[index, :3, :3] = Slerp([0.0, 1.0], rotations)(
            [alpha[index]]
        ).as_matrix()[0]
    return np.linalg.inv(output_c2w), lower, upper, alpha


def _reproject_optical_z(
    depth_m: np.ndarray,
    valid: np.ndarray,
    source_K: np.ndarray,
    output_K: np.ndarray,
    T_output_source: np.ndarray,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject metric optical-z with a nearest-surface z-buffer."""
    depth = np.asarray(depth_m, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    source_k = np.asarray(source_K, dtype=np.float64)
    output_k = np.asarray(output_K, dtype=np.float64)
    transform = _rigid(T_output_source, "depth reprojection transform")
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ArtifactError("source depth and validity mask shapes disagree")
    height, width = output_shape
    if height <= 0 or width <= 0:
        raise ArtifactError("output depth dimensions must be positive")
    mask &= np.isfinite(depth) & (depth > 0)
    y, x = np.nonzero(mask)
    output = np.zeros((height, width), dtype=np.float32)
    if not len(x):
        return output, np.zeros_like(output, dtype=bool)
    z = depth[y, x].astype(np.float64)
    points = np.stack(
        (
            (x - source_k[0, 2]) * z / source_k[0, 0],
            (y - source_k[1, 2]) * z / source_k[1, 1],
            z,
            np.ones_like(z),
        )
    )
    transformed = transform @ points
    target_z = transformed[2]
    visible = np.isfinite(transformed).all(axis=0) & (target_z > 0)
    u = np.rint(
        output_k[0, 0] * transformed[0] / target_z + output_k[0, 2]
    ).astype(np.int64)
    v = np.rint(
        output_k[1, 1] * transformed[1] / target_z + output_k[1, 2]
    ).astype(np.int64)
    visible &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not visible.any():
        return output, np.zeros_like(output, dtype=bool)
    flat = np.full(height * width, np.inf, dtype=np.float64)
    flat_index = v[visible] * width + u[visible]
    np.minimum.at(flat, flat_index, target_z[visible])
    output_valid = np.isfinite(flat).reshape(height, width)
    output[output_valid] = flat.reshape(height, width)[output_valid].astype(np.float32)
    return output, output_valid


def _fill_projection_lattice(
    depth_m: np.ndarray,
    valid: np.ndarray,
    *,
    min_valid_neighbors: int,
    max_spread_abs_m: float,
    max_spread_rel: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fill single-pixel resampling holes left by nearest-pixel scattering.

    Projecting into a camera with a longer focal length places source samples
    more than one target pixel apart, so a fixed lattice of target rows and
    columns never receives a sample even where the surface is continuously
    observed. One 3x3 pass fills an invalid pixel with the median of its valid
    neighbours only when enough neighbours exist and they agree on one surface
    (spread within max(abs, rel * nearest)). Depth discontinuities, occlusion
    boundaries and out-of-frustum regions fail those guards and stay invalid;
    holes wider than one pixel cannot be bridged by a single 3x3 pass.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ArtifactError("depth and validity mask shapes disagree")
    height, width = depth.shape
    neighbors = np.full((8, height, width), np.nan, dtype=np.float32)
    source = np.where(mask, depth, np.nan).astype(np.float32)
    shifts = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    for plane, (dy, dx) in enumerate(shifts):
        # neighbors[plane][y, x] = source[y + dy, x + dx] where that exists
        destination_y = slice(max(-dy, 0), height - max(dy, 0))
        destination_x = slice(max(-dx, 0), width - max(dx, 0))
        source_y = slice(max(dy, 0), height - max(-dy, 0))
        source_x = slice(max(dx, 0), width - max(-dx, 0))
        neighbors[plane][destination_y, destination_x] = source[source_y, source_x]
    known = ~np.isnan(neighbors)
    count = known.sum(axis=0)
    nearest = np.where(known, neighbors, np.inf).min(axis=0)
    farthest = np.where(known, neighbors, -np.inf).max(axis=0)
    agree = (farthest - nearest) <= np.maximum(
        np.float32(max_spread_abs_m), np.float32(max_spread_rel) * nearest
    )
    fillable = (~mask) & (count >= min_valid_neighbors) & agree
    if not fillable.any():
        return depth, mask, 0
    filled_depth = depth.copy()
    filled_depth[fillable] = np.nanmedian(
        neighbors[:, fillable], axis=0
    ).astype(np.float32)
    filled_valid = mask | fillable
    return filled_depth, filled_valid, int(np.count_nonzero(fillable))


def _load_depth(path: Path, expected_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            depth = np.asarray(archive["depth"], dtype=np.float32)
            valid = np.asarray(archive["valid"], dtype=bool)
    except (OSError, ValueError, KeyError) as exc:
        raise ArtifactError(f"cannot read metric depth {path}: {exc}") from exc
    if depth.shape != expected_shape or valid.shape != expected_shape:
        raise ArtifactError(
            f"depth {path} has shape {depth.shape}, expected {expected_shape}"
        )
    invalid = valid & (~np.isfinite(depth) | (depth <= 0))
    if invalid.any():
        raise ArtifactError(f"depth {path} marks invalid optical-z values as valid")
    return depth, valid


def _mapper_config(source_provenance: dict[str, Any]) -> tuple[MapperConfig, list[str]]:
    stored = source_provenance.get("backend_config_effective")
    if not isinstance(stored, dict):
        raise ArtifactError("source pose provenance has no effective mapper gates")
    names = {field.name for field in fields(MapperConfig)}
    selected = {name: value for name, value in stored.items() if name in names}
    missing = sorted(names - set(selected))
    try:
        return MapperConfig(**selected), missing
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"source mapper gate policy is invalid: {exc}") from exc


def _floored_covariance(covariance: np.ndarray, floor_m: float) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    output = np.empty_like(covariance)
    for index, matrix in enumerate(covariance):
        if not np.isfinite(matrix).all():
            raise ArtifactError("target GNSS covariance is not finite")
        eigenvalues, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
        output[index] = (vectors * np.maximum(eigenvalues, floor_m**2)) @ vectors.T
    return output


def _source_hashes(reader: SegmentReader) -> dict[str, str]:
    paths = [
        "frames.npz",
        "calibration.json",
        "segment_meta.json",
        "manifest.json",
        "observations/gnss.npz",
    ]
    if reader.meta["capabilities"]["dual_rtk"]:
        paths.append("observations/heading.npz")
    return {relative: sha256_file(reader.root / relative) for relative in paths}


def _verify_pose_source_binding(
    source: SegmentReader, pose_provenance: Mapping[str, Any]
) -> dict[str, Any]:
    binding = pose_provenance.get("source_segment_binding")
    if not isinstance(binding, dict):
        raise ArtifactError("source pose artifact has no sealed source-segment binding")
    contract = binding.get("contract_inputs")
    if (
        not isinstance(contract, dict)
        or binding.get("contract_inputs_sha256") != canonical_hash(contract)
        or contract.get("segment_root") != str(source.root.resolve())
    ):
        raise ArtifactError("source pose artifact is bound to a different segment")
    acquisition_id = source.meta.get("acquisition_id")
    if (
        not isinstance(acquisition_id, str)
        or binding.get("acquisition_id") != acquisition_id
        or contract.get("acquisition_id") != acquisition_id
    ):
        raise ArtifactError("source pose artifact acquisition binding is invalid")
    files = contract.get("files")
    required = set(_source_hashes(source))
    if not isinstance(files, dict) or not required <= set(files):
        raise ArtifactError("source pose binding lacks segment core-file evidence")
    for relative, record in files.items():
        path = source.root / str(relative)
        if (
            not isinstance(relative, str)
            or not isinstance(record, dict)
            or not path.is_file()
            or record.get("sha256") != sha256_file(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise ArtifactError(
                f"source pose segment binding changed: {relative!r}"
            )
    inventory = image_inventory(source)
    if (
        contract.get("image_inventory_count") != len(inventory)
        or contract.get("image_inventory_sha256") != canonical_hash(inventory)
    ):
        raise ArtifactError("source pose image-inventory binding is invalid")
    return {
        "contract_inputs_sha256": binding["contract_inputs_sha256"],
        "acquisition_id": acquisition_id,
        "image_inventory_count": len(inventory),
        "image_inventory_sha256": contract["image_inventory_sha256"],
    }


def _sealed_registration_files(root: Path) -> tuple[dict[str, Any], str]:
    required = {
        "validation.json",
        "source_points_xyz_m.npy",
        "source_depth_timestamp_ns.npy",
        "rgb_pixels_px.npy",
        "rgb_timestamp_ns.npy",
    }
    manifest_path = root / "manifest.json"
    manifest = _json(manifest_path)
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type")
        != "held_out_rgb_depth_registration_evidence"
        or not isinstance(files, dict)
        or not required <= set(files)
    ):
        raise ArtifactError("held-out registration evidence manifest is incomplete")
    for relative, record in files.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(record, dict)
        ):
            raise ArtifactError("held-out registration manifest path is invalid")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or (
            record.get("sha256") != sha256_file(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise ArtifactError(
                f"held-out registration evidence changed: {relative}"
            )
    return manifest, sha256_file(manifest_path)


def _held_out_registration_quality(
    evidence_path: str | None,
    *,
    source: SegmentReader,
    source_pose_root: Path,
    source_pose_timestamps: np.ndarray,
    source_pose_viewmats: np.ndarray,
    rgb_root: Path,
    rgb_frames: Mapping[str, np.ndarray],
    rgb_to_source_clock_offset_ns: int,
    rgb_calibration: Mapping[str, Any],
    config: RgbdTransferConfig,
) -> dict[str, Any]:
    """Recompute a sealed independent correspondence reprojection check.

    RGB plus depth images alone do not generically prove cross-modal
    registration. Production promotion therefore requires independently
    selected, held-out 3-D/source-camera to 2-D/RGB correspondences.
    """
    if evidence_path is None:
        return {
            "status": "NOT_SUPPLIED",
            "passed": False,
            "production_promotion_allowed": False,
            "reason": (
                "RGB and depth alone do not generically prove cross-modal "
                "pixel registration; sealed held-out correspondences are required"
            ),
        }
    root = Path(evidence_path).expanduser().resolve()
    if not root.is_dir():
        raise ArtifactError(
            f"held-out registration evidence does not exist: {root}"
        )
    _, manifest_sha256 = _sealed_registration_files(root)
    declaration = _json(root / "validation.json")
    expected_bindings = {
        "source_segment_manifest_sha256": sha256_file(
            source.root / "manifest.json"
        ),
        "rgb_observations_manifest_sha256": sha256_file(
            rgb_root / "manifest.json"
        ),
        "rgb_calibration_sha256": sha256_file(rgb_root / "calibration.json"),
        "source_pose_manifest_sha256": sha256_file(
            source_pose_root / "manifest.json"
        ),
        "rgb_to_source_clock_offset_ns": rgb_to_source_clock_offset_ns,
        "source_camera_frame_id": str(
            source.meta["initial_pose"]["camera_frame_id"]
        ),
        "rgb_camera_frame_id": str(rgb_calibration["camera_frame_id"]),
    }
    if (
        declaration.get("schema_version") != 1
        or declaration.get("artifact_type")
        != "held_out_rgb_depth_registration_evidence"
        or declaration.get("validation_split") != "held_out"
        or declaration.get("independent_of_extrinsic_estimation") is not True
        or declaration.get("correspondence_geometry")
        != "source_depth_xyz_at_depth_time_to_rgb_pixels_at_rgb_time"
        or not isinstance(declaration.get("method"), str)
        or not declaration["method"].strip()
        or any(declaration.get(key) != value for key, value in expected_bindings.items())
    ):
        raise ArtifactError(
            "held-out registration declaration is invalid or bound to different inputs"
        )
    try:
        source_points = np.load(
            root / "source_points_xyz_m.npy", allow_pickle=False
        )
        source_depth_ns = np.load(
            root / "source_depth_timestamp_ns.npy", allow_pickle=False
        )
        rgb_pixels = np.load(root / "rgb_pixels_px.npy", allow_pickle=False)
        evidence_rgb_ns = np.load(root / "rgb_timestamp_ns.npy", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"cannot read held-out registration arrays: {exc}") from exc
    source_points = np.asarray(source_points, dtype=np.float64)
    source_depth_ns = np.asarray(source_depth_ns)
    rgb_pixels = np.asarray(rgb_pixels, dtype=np.float64)
    evidence_rgb_ns = np.asarray(evidence_rgb_ns)
    n = len(source_points) if source_points.ndim == 2 else 0
    if (
        source_points.shape != (n, 3)
        or rgb_pixels.shape != (n, 2)
        or source_depth_ns.dtype != np.dtype(np.int64)
        or source_depth_ns.shape != (n,)
        or evidence_rgb_ns.dtype != np.dtype(np.int64)
        or evidence_rgb_ns.shape != (n,)
        or n < config.min_held_out_correspondences
        or not np.isfinite(source_points).all()
        or not np.isfinite(rgb_pixels).all()
    ):
        raise ArtifactError("held-out registration correspondences are invalid")
    source_depth_timestamps = _source_depth_timeline(source.frames)
    held_out_ids = set(source.manifest["val"]) | set(source.manifest["test"])
    depth_to_id = {
        int(timestamp): int(frame_id)
        for timestamp, frame_id in zip(
            source_depth_timestamps, source.frames["frame_id"], strict=True
        )
    }
    if (
        not held_out_ids
        or any(
            depth_to_id.get(int(timestamp)) not in held_out_ids
            for timestamp in source_depth_ns
        )
        or not np.isin(evidence_rgb_ns, rgb_frames["timestamp_ns"]).all()
    ):
        raise ArtifactError(
            "registration correspondences are not bound to held-out source/RGB times"
        )
    unique_depth_ns = np.unique(source_depth_ns)
    unique_rgb_ns = np.unique(evidence_rgb_ns)
    if len(unique_depth_ns) < 2 or len(unique_rgb_ns) < 2:
        raise ArtifactError(
            "held-out registration needs at least two distinct depth/RGB times"
        )
    corrected_rgb_ns = evidence_rgb_ns + np.int64(rgb_to_source_clock_offset_ns)
    if np.any((corrected_rgb_ns - evidence_rgb_ns) != rgb_to_source_clock_offset_ns):
        raise ArtifactError("held-out RGB clock correction overflows int64")
    depth_viewmats, _, _, _ = _interpolate_viewmats(
        source_pose_timestamps,
        source_pose_viewmats,
        source_depth_ns,
        max_gap_ns=config.max_pose_interpolation_gap_ns,
    )
    rgb_time_viewmats, _, _, _ = _interpolate_viewmats(
        source_pose_timestamps,
        source_pose_viewmats,
        corrected_rgb_ns,
        max_gap_ns=config.max_pose_interpolation_gap_ns,
    )
    transform = _rigid(
        rgb_calibration["T_rgb_source_camera"], "T_rgb_source_camera"
    )
    homogeneous = np.column_stack((source_points, np.ones(n, dtype=np.float64)))
    world_points = np.einsum(
        "nij,nj->ni", np.linalg.inv(depth_viewmats), homogeneous
    )
    source_at_rgb_time = np.einsum(
        "nij,nj->ni", rgb_time_viewmats, world_points
    )
    transformed = np.einsum("ij,nj->ni", transform, source_at_rgb_time)[:, :3]
    camera = rgb_calibration["camera"]
    K = np.asarray(camera["K"], dtype=np.float64)
    if np.any(transformed[:, 2] <= 0):
        raise ArtifactError("held-out registration points project behind RGB camera")
    predicted = np.column_stack(
        (
            K[0, 0] * transformed[:, 0] / transformed[:, 2] + K[0, 2],
            K[1, 1] * transformed[:, 1] / transformed[:, 2] + K[1, 2],
        )
    )
    width, height = int(camera["width"]), int(camera["height"])
    if np.any(
        (predicted[:, 0] < 0)
        | (predicted[:, 0] >= width)
        | (predicted[:, 1] < 0)
        | (predicted[:, 1] >= height)
    ):
        raise ArtifactError("held-out registration points project outside RGB image")
    jacobian_rows: list[np.ndarray] = []
    for point in transformed:
        x, y, z = point
        projection_jacobian = np.asarray(
            [
                [K[0, 0] / z, 0.0, -K[0, 0] * x / (z * z)],
                [0.0, K[1, 1] / z, -K[1, 1] * y / (z * z)],
            ],
            dtype=np.float64,
        )
        skew = np.asarray(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
            dtype=np.float64,
        )
        jacobian_rows.append(
            projection_jacobian @ np.column_stack((np.eye(3), -skew))
        )
    extrinsic_jacobian = np.concatenate(jacobian_rows, axis=0)
    column_norms = np.linalg.norm(extrinsic_jacobian, axis=0)
    if np.any(column_norms <= np.finfo(np.float64).eps):
        extrinsic_rank = 0
        extrinsic_condition = None
    else:
        singular_values = np.linalg.svd(
            extrinsic_jacobian / column_norms[None, :], compute_uv=False
        )
        extrinsic_rank = int(
            np.count_nonzero(singular_values > singular_values[0] * 1e-6)
        )
        extrinsic_condition = (
            float(singular_values[0] / singular_values[-1])
            if singular_values[-1] > 0
            else None
        )
    extrinsic_observable = (
        extrinsic_rank == 6
        and extrinsic_condition is not None
        and extrinsic_condition
        <= config.max_extrinsic_observability_condition_number
    )

    probe_ns = config.clock_observability_probe_ns
    can_increase = corrected_rgb_ns <= int(source_pose_timestamps[-1]) - probe_ns
    can_decrease = corrected_rgb_ns >= int(source_pose_timestamps[0]) + probe_ns
    probe_sign = np.where(can_increase, 1, np.where(can_decrease, -1, 0)).astype(
        np.int64
    )
    clock_sensitivity = np.zeros(n, dtype=np.float64)
    if np.all(probe_sign != 0):
        perturbed_rgb_ns = corrected_rgb_ns + probe_sign * np.int64(probe_ns)
        perturbed_viewmats, _, _, _ = _interpolate_viewmats(
            source_pose_timestamps,
            source_pose_viewmats,
            perturbed_rgb_ns,
            max_gap_ns=config.max_pose_interpolation_gap_ns,
        )
        perturbed_source_points = np.einsum(
            "nij,nj->ni", perturbed_viewmats, world_points
        )
        perturbed_rgb_points = np.einsum(
            "ij,nj->ni", transform, perturbed_source_points
        )[:, :3]
        if np.all(perturbed_rgb_points[:, 2] > 0):
            perturbed_pixels = np.column_stack(
                (
                    K[0, 0]
                    * perturbed_rgb_points[:, 0]
                    / perturbed_rgb_points[:, 2]
                    + K[0, 2],
                    K[1, 1]
                    * perturbed_rgb_points[:, 1]
                    / perturbed_rgb_points[:, 2]
                    + K[1, 2],
                )
            )
            clock_sensitivity = (
                np.linalg.norm(perturbed_pixels - predicted, axis=1)
                / (probe_ns * 1e-9)
            )
    median_clock_sensitivity = float(np.median(clock_sensitivity))
    clock_observable = (
        median_clock_sensitivity >= config.min_clock_observability_px_per_s
    )
    errors = np.linalg.norm(predicted - rgb_pixels, axis=1)
    median = float(np.median(errors))
    p95 = float(np.percentile(errors, 95))
    passed = (
        median <= config.max_held_out_median_reprojection_error_px
        and p95 <= config.max_held_out_p95_reprojection_error_px
    )
    production_promotion_allowed = False
    report = {
        "status": "PASSED" if passed else "FAILED",
        "passed": passed,
        "production_promotion_allowed": production_promotion_allowed,
        "method": "sealed_held_out_cross_modal_correspondence_reprojection_v1",
        "evidence_path": str(root),
        "evidence_manifest_sha256": manifest_sha256,
        "declaration": declaration,
        "n_correspondences": n,
        "n_unique_source_depth_timestamps": int(len(unique_depth_ns)),
        "n_unique_rgb_timestamps": int(len(unique_rgb_ns)),
        "source_depth_time_span_ns": int(unique_depth_ns[-1] - unique_depth_ns[0]),
        "rgb_time_span_ns": int(unique_rgb_ns[-1] - unique_rgb_ns[0]),
        "rgb_to_source_clock_offset_ns": rgb_to_source_clock_offset_ns,
        "clock_observability": {
            "probe_ns": probe_ns,
            "median_pixel_sensitivity_px_per_s": median_clock_sensitivity,
            "minimum_pixel_sensitivity_px_per_s": (
                config.min_clock_observability_px_per_s
            ),
            "observable": clock_observable,
        },
        "extrinsic_observability": {
            "normalized_jacobian_rank": extrinsic_rank,
            "normalized_jacobian_condition_number": extrinsic_condition,
            "maximum_condition_number": (
                config.max_extrinsic_observability_condition_number
            ),
            "observable": extrinsic_observable,
        },
        "minimum_correspondences": config.min_held_out_correspondences,
        "median_reprojection_error_px": median,
        "p95_reprojection_error_px": p95,
        "maximum_median_reprojection_error_px": (
            config.max_held_out_median_reprojection_error_px
        ),
        "maximum_p95_reprojection_error_px": (
            config.max_held_out_p95_reprojection_error_px
        ),
    }
    if passed:
        report["reason"] = "joint_extrinsic_clock_observability_not_established"
    if not passed:
        raise ArtifactError(
            "held-out RGB-depth registration gate failed: median "
            f"{median:.3f} px, p95 {p95:.3f} px"
        )
    return report


def _write_pose_artifact(
    staging: Path,
    name: str,
    viewmats: np.ndarray,
    centers: np.ndarray,
    timestamps_ns: np.ndarray,
    image_names: np.ndarray,
    quality: dict[str, Any],
    provenance: dict[str, Any],
    transfer: dict[str, Any],
    transfer_arrays: Mapping[str, np.ndarray],
    status: Mapping[str, Any],
) -> None:
    staging.mkdir()
    frame_ids = np.arange(len(viewmats), dtype=np.int64)
    _atomic_npy(staging / "viewmats.npy", viewmats.astype(np.float64))
    _atomic_npy(staging / "cam_centers.npy", centers.astype(np.float64))
    _atomic_npy(staging / "frame_ids.npy", frame_ids)
    _atomic_npy(staging / "timestamps_ns.npy", timestamps_ns.astype(np.int64))
    _atomic_npy(staging / "left_image_names.npy", image_names.astype(np.str_))
    base = {"schema_version": 1, **status}
    _atomic_json(
        staging / "quality.json",
        {
            **base,
            "n_frames": len(viewmats),
            "rtk_alignment_passed": True,
            "transfer_rtk_quality": quality,
            "pixel_registration_gates_passed": bool(
                transfer.get("held_out_registration_quality", {}).get(
                    "passed", False
                )
            ),
            "production_transfer_validation_passed": bool(
                transfer.get("held_out_registration_quality", {}).get(
                    "production_promotion_allowed", False
                )
            ),
        },
    )
    _atomic_json(
        staging / "alignment.json",
        {
            **base,
            "method": "calibrated_camera_transfer_no_world_refit",
            "production_scale": 1.0,
            "world_alignment_fitted": False,
            "pose_priors_constrain_transfer": False,
        },
    )
    _atomic_json(staging / "provenance.json", {**base, **provenance})
    _atomic_json(
        staging / "georeferencing.json",
        {
            **base,
            "rtk_alignment_passed": True,
            "source_fixed_scale_alignment_inherited": True,
            "source_rtk_consistency_rechecked": True,
            "target_rtk_acquired": False,
            "new_world_alignment_fitted": False,
            "held_out_rgb_depth_registration_passed": bool(
                transfer.get("held_out_registration_quality", {}).get(
                    "passed", False
                )
            ),
            "production_promotion_allowed": False,
            "warning": (
                None
                if status["artifact_class"] == "production"
                else transfer.get("held_out_registration_quality", {}).get(
                    "reason",
                    "held-out RGB-depth registration is not production-eligible",
                )
            ),
        },
    )
    _atomic_json(staging / "transfer_evaluation.json", transfer)
    with (staging / "transfer_evidence.npz").open("xb") as stream:
        np.savez_compressed(stream, **transfer_arrays)
    evidence = {
        path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(staging.iterdir())
        if path.is_file()
    }
    _atomic_json(
        staging / "manifest.json",
        {
            "schema_version": 1,
            "name": name,
            "n_frames": len(viewmats),
            **status,
            "files": evidence,
        },
    )


def derive_rgbd_segment(
    source_segment: str | Path,
    source_pose_artifact: str | Path,
    rgb_observations: str | Path,
    destination_segment: str | Path,
    destination_pose_artifact: str | Path,
    *,
    config: RgbdTransferConfig,
) -> tuple[SegmentReader, Path]:
    """Publish a new RGB-D segment and accepted RGB pose artifact.

    Both inputs remain immutable.  The source pose must already be a modern,
    manifest-sealed production artifact.  Any failed synchronization,
    calibration, depth, or georeferencing gate publishes neither output.
    """
    destination_segment = Path(destination_segment).expanduser().resolve()
    destination_pose = Path(destination_pose_artifact).expanduser().resolve()
    if destination_segment.exists() or destination_pose.exists():
        raise FileExistsError("refusing to modify an existing transfer destination")
    if not _SAFE_NAME.fullmatch(destination_pose.name):
        raise ValueError(f"invalid pose artifact name: {destination_pose.name!r}")
    source = SegmentReader(source_segment).validate()
    capabilities = source.meta["capabilities"]
    depth_semantics = source.meta.get("depth_observation", {})
    if not (
        capabilities["stereo"]
        and capabilities["images_rectified"]
        and (
            capabilities["rgbd"]
            or capabilities["depth_recorded"]
            or capabilities["depth_computed"]
        )
        and not capabilities["imu_present"]
        and depth_semantics.get("format") == "npz_depth_valid"
        and depth_semantics.get("units") == "m"
        and depth_semantics.get("quantity") == "optical_z"
        and depth_semantics.get("aligned_to") == "left"
    ):
        raise ArtifactError(
            "transfer requires IMU-free rectified stereo with validated metric "
            "optical-z depth aligned to the source left camera"
        )
    if not (capabilities["single_rtk"] ^ capabilities["dual_rtk"]):
        raise ArtifactError("transfer requires exactly one declared RTK acquisition mode")
    source_pose = Path(source_pose_artifact).expanduser().resolve()
    source_evidence = verify_pose_georeferencing_artifact(
        source_pose, expected_name=source_pose.name
    )
    if (
        source_evidence["artifact_class"] != "production"
        or source_evidence["georeferencing_status"] != "PASSED"
        or source_evidence["metric_georeferencing_claim_eligible"] is not True
    ):
        raise ArtifactError("source pose artifact is not accepted production georeferencing")
    pose_manifest = _json(source_pose / "manifest.json")
    source_pose_provenance = _json(source_pose / "provenance.json")
    source_pose_binding = _verify_pose_source_binding(
        source, source_pose_provenance
    )
    for name in ("frame_ids.npy", "timestamps_ns.npy", "viewmats.npy", "cam_centers.npy"):
        if name not in pose_manifest.get("files", {}):
            raise ArtifactError(f"source pose manifest does not seal {name}")
    source_frames = source.frames
    source_depth_timestamps = _source_depth_timeline(source_frames)
    source_ids = np.load(source_pose / "frame_ids.npy", allow_pickle=False)
    source_timestamps = np.load(source_pose / "timestamps_ns.npy", allow_pickle=False)
    accepted_viewmats = np.load(source_pose / "viewmats.npy", allow_pickle=False)
    accepted_centers = np.load(source_pose / "cam_centers.npy", allow_pickle=False)
    if (
        not np.array_equal(source_ids, source_frames["frame_id"])
        or not np.array_equal(source_timestamps, source_frames["timestamp_ns"])
        or accepted_viewmats.shape != (len(source_ids), 4, 4)
        or accepted_centers.shape != (len(source_ids), 3)
        or not np.allclose(
            np.linalg.inv(accepted_viewmats)[:, :3, 3], accepted_centers, atol=2e-4
        )
        or not np.asarray(source_frames["pose_valid"], dtype=bool).all()
    ):
        raise ArtifactError("source pose artifact and source segment timeline disagree")
    rgb = load_rectified_rgb_observations(rgb_observations)
    source_acquisition_id = source.meta.get("acquisition_id")
    source_meta_provenance = source.meta.get("provenance")
    rgb_acquisition_id = rgb.meta.get("acquisition_id")
    if (
        not isinstance(source_acquisition_id, str)
        or not source_acquisition_id
        or not isinstance(source_meta_provenance, dict)
        or source_meta_provenance.get("acquisition_id") != source_acquisition_id
        or rgb_acquisition_id != source_acquisition_id
        or rgb.meta.get("provenance", {}).get("acquisition_id")
        != source_acquisition_id
    ):
        raise ArtifactError(
            "RGB observations are not sealed to this source acquisition"
        )
    source_camera_frame_id = str(source.meta["initial_pose"]["camera_frame_id"])
    if rgb.calibration["source_camera_frame_id"] != source_camera_frame_id:
        raise ArtifactError(
            "RGB extrinsic source_camera_frame_id does not match the source "
            "segment left-camera pose frame"
        )
    all_rgb_timestamps = np.asarray(rgb.frames["timestamp_ns"], dtype=np.int64)
    clock_offset = int(rgb.meta["clock"]["rgb_to_source_clock_offset_ns"])
    all_corrected_rgb_timestamps = all_rgb_timestamps + np.int64(clock_offset)
    if np.any((all_corrected_rgb_timestamps - all_rgb_timestamps) != clock_offset):
        raise ArtifactError("RGB clock correction overflows int64")
    depth_indices, rgb_indices, rgb_minus_depth_residual, association = (
        _associate_depth_to_rgb(
            source_depth_timestamps,
            all_corrected_rgb_timestamps,
            config.max_depth_sync_residual_ns,
        )
    )
    association_fraction = len(depth_indices) / float(len(source_depth_timestamps))
    association.update(
        matched_source_depth_fraction=association_fraction,
        minimum_matched_source_depth_fraction=(
            config.min_depth_rgb_association_fraction
        ),
        passed=(association_fraction >= config.min_depth_rgb_association_fraction),
        source_depth_timestamp_field="frames.depth_timestamp_ns",
    )
    if not association["passed"]:
        raise ArtifactError(
            "depth association fraction gate failed: matched "
            f"{association_fraction:.6f}, minimum "
            f"{config.min_depth_rgb_association_fraction:.6f}"
        )
    rgb_timestamps = all_rgb_timestamps[rgb_indices]
    query_source_clock = all_corrected_rgb_timestamps[rgb_indices]
    selected_depth_timestamps = source_depth_timestamps[depth_indices]
    # The segment's generic observation convention is source minus query.
    # The transfer report also retains the opposite, explicitly named residual.
    depth_residual = -rgb_minus_depth_residual
    accepted_interpolated, lower, upper, alpha = _interpolate_viewmats(
        source_timestamps,
        accepted_viewmats,
        query_source_clock,
        max_gap_ns=config.max_pose_interpolation_gap_ns,
    )
    depth_interpolated, depth_lower, depth_upper, depth_alpha = (
        _interpolate_viewmats(
            source_timestamps,
            accepted_viewmats,
            selected_depth_timestamps,
            max_gap_ns=config.max_pose_interpolation_gap_ns,
        )
    )
    initial_interpolated, initial_lower, initial_upper, initial_alpha = (
        _interpolate_viewmats(
            source_timestamps,
            np.asarray(source_frames["initial_viewmat"], dtype=np.float64),
            query_source_clock,
            max_gap_ns=config.max_pose_interpolation_gap_ns,
        )
    )
    if not (
        np.array_equal(lower, initial_lower)
        and np.array_equal(upper, initial_upper)
        and np.allclose(alpha, initial_alpha)
    ):
        raise ArtifactError("accepted and initial pose interpolation brackets disagree")
    T_rgb_source = _rigid(
        rgb.calibration["T_rgb_source_camera"], "T_rgb_source_camera"
    )
    rgb_viewmats = T_rgb_source[None] @ accepted_interpolated
    rgb_initial_viewmats = T_rgb_source[None] @ initial_interpolated
    source_depth_c2w = np.linalg.inv(depth_interpolated)
    rgb_c2w = np.linalg.inv(rgb_viewmats)
    rgb_initial_c2w = np.linalg.inv(rgb_initial_viewmats)
    rgb_centers = rgb_c2w[:, :3, 3]
    rgb_initial_centers = rgb_initial_c2w[:, :3, 3]

    source_camera = source.calibration["cameras"]["left"]
    source_k = np.asarray(source_camera["K"], dtype=np.float64)
    source_shape = (int(source_camera["height"]), int(source_camera["width"]))
    output_camera = rgb.calibration["camera"]
    output_k = np.asarray(output_camera["K"], dtype=np.float64)
    output_shape = (int(output_camera["height"]), int(output_camera["width"]))

    writer = SegmentWriter(destination_segment)
    pose_staging = destination_pose.parent / (
        f".{destination_pose.name}.writing-{uuid.uuid4().hex}"
    )
    segment_published = False
    pose_published = False
    try:
        image_directory = writer.directory("images")
        depth_directory = writer.directory("depth")
        image_paths: list[str] = []
        depth_paths: list[str] = []
        source_valid_counts: list[int] = []
        scattered_valid_counts: list[int] = []
        projected_valid_counts: list[int] = []
        for index, (source_index, rgb_index) in enumerate(
            zip(depth_indices, rgb_indices, strict=True)
        ):
            source_image = rgb.root / str(rgb.frames["image_path"][rgb_index])
            input_image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            if input_image is None or input_image.shape[:2] != output_shape:
                raise ArtifactError(
                    f"RGB observation {int(rgb_index)} does not match calibration"
                )
            image_name = f"rgb_{index:06d}.png"
            shutil.copyfile(source_image, image_directory / image_name)
            image_paths.append(f"images/{image_name}")

            source_depth, source_valid = _load_depth(
                source.root / str(source_frames["depth_path"][source_index]),
                source_shape,
            )
            T_rgb_at_query_from_source_at_depth = (
                rgb_viewmats[index] @ source_depth_c2w[index]
            )
            depth, valid = _reproject_optical_z(
                source_depth,
                source_valid,
                source_k,
                output_k,
                T_rgb_at_query_from_source_at_depth,
                output_shape,
            )
            scattered = int(valid.sum())
            depth, valid, _ = _fill_projection_lattice(
                depth,
                valid,
                min_valid_neighbors=config.depth_fill_min_valid_neighbors,
                max_spread_abs_m=config.depth_fill_max_spread_abs_m,
                max_spread_rel=config.depth_fill_max_spread_rel,
            )
            depth_name = f"{index:06d}.npz"
            np.savez_compressed(
                depth_directory / depth_name,
                depth=depth.astype(np.float32),
                valid=valid.astype(bool),
            )
            depth_paths.append(f"depth/{depth_name}")
            source_valid_counts.append(int(source_valid.sum()))
            scattered_valid_counts.append(scattered)
            projected_valid_counts.append(int(valid.sum()))

        projected_depth_quality = _projected_depth_quality(
            np.asarray(source_valid_counts, dtype=np.int64),
            np.asarray(scattered_valid_counts, dtype=np.int64),
            np.asarray(projected_valid_counts, dtype=np.int64),
            output_shape[0] * output_shape[1],
            config,
        )
        registration_quality = _held_out_registration_quality(
            config.held_out_registration_evidence,
            source=source,
            source_pose_root=source_pose,
            source_pose_timestamps=source_timestamps,
            source_pose_viewmats=accepted_viewmats,
            rgb_root=rgb.root,
            rgb_frames=rgb.frames,
            rgb_to_source_clock_offset_ns=clock_offset,
            rgb_calibration=rgb.calibration,
            config=config,
        )
        artifact_status = _DIAGNOSTIC_STATUS

        source_gnss = source.observations("gnss")
        assert source_gnss is not None
        if not np.asarray(source_gnss["position_valid"], dtype=bool)[
            depth_indices
        ].all():
            raise ArtifactError("every derived RGB frame requires a valid RTK observation")
        gnss_covariance = np.asarray(
            source_gnss["covariance_enu_m2"], dtype=np.float64
        )[depth_indices]
        source_sigma = np.asarray(
            source.meta["initial_pose"]["extrinsic_translation_sigma_m"],
            dtype=np.float64,
        )
        rgb_sigma = np.asarray(
            rgb.calibration["extrinsic_translation_sigma_m"], dtype=np.float64
        )
        sigma_rgb_covariance = (
            T_rgb_source[:3, :3] @ np.diag(source_sigma**2) @ T_rgb_source[:3, :3].T
            + np.diag(rgb_sigma**2)
        )
        world_rgb_rotation = rgb_initial_c2w[:, :3, :3]
        effective_covariance = gnss_covariance + np.einsum(
            "nij,jk,nlk->nil",
            world_rgb_rotation,
            sigma_rgb_covariance,
            world_rgb_rotation,
        )
        effective_covariance = _floored_covariance(
            effective_covariance, config.position_sigma_floor_m
        )
        mapper_config, defaulted_gate_fields = _mapper_config(
            source_pose_provenance
        )
        residual_vectors = rgb_centers - rgb_initial_centers
        sigma_max = np.sqrt(np.linalg.eigvalsh(effective_covariance)[:, -1])
        thresholds = np.maximum(
            mapper_config.alignment_ransac_threshold_m, 3.0 * sigma_max
        )
        inlier_mask = np.linalg.norm(residual_vectors, axis=1) <= thresholds
        transfer_quality = rtk_residual_quality(
            residual_vectors,
            effective_covariance,
            inlier_mask,
            config=mapper_config,
        )
        if not transfer_quality["passed"]:
            raise ArtifactError(
                "RGB camera transfer failed the source artifact's RTK residual gates"
            )

        frames = {
            "frame_id": np.arange(len(rgb_timestamps), dtype=np.int64),
            "timestamp_ns": rgb_timestamps,
            "left_image_path": np.asarray(image_paths, dtype=np.str_),
            "depth_path": np.asarray(depth_paths, dtype=np.str_),
            "initial_viewmat": rgb_initial_viewmats.astype(np.float64),
            "initial_camera_center_m": rgb_initial_centers.astype(np.float64),
            "pose_valid": np.ones(len(rgb_timestamps), dtype=bool),
            "rgb_observation_frame_id": np.asarray(
                rgb.frames["frame_id"], dtype=np.int64
            )[rgb_indices],
            "rgb_observation_index": rgb_indices,
            "rgb_corrected_source_clock_timestamp_ns": query_source_clock,
            "depth_source_index": depth_indices,
            "depth_source_frame_id": source_frames["frame_id"][depth_indices],
            "depth_source_frame_timestamp_ns": source_frames["timestamp_ns"][
                depth_indices
            ],
            "depth_source_timestamp_ns": selected_depth_timestamps,
            "depth_sync_residual_ns": depth_residual,
            "corrected_rgb_minus_depth_sync_residual_ns": (
                rgb_minus_depth_residual
            ),
            "depth_source_valid_pixels": np.asarray(source_valid_counts, dtype=np.int64),
            "depth_scattered_valid_pixels": np.asarray(
                scattered_valid_counts, dtype=np.int64
            ),
            "depth_projected_valid_pixels": np.asarray(
                projected_valid_counts, dtype=np.int64
            ),
            "pose_interpolation_lower_frame_id": source_frames["frame_id"][lower],
            "pose_interpolation_upper_frame_id": source_frames["frame_id"][upper],
            "pose_interpolation_alpha": alpha,
            "depth_pose_interpolation_lower_frame_id": source_frames["frame_id"][
                depth_lower
            ],
            "depth_pose_interpolation_upper_frame_id": source_frames["frame_id"][
                depth_upper
            ],
            "depth_pose_interpolation_alpha": depth_alpha,
        }
        if "log_timestamp_ns" in rgb.frames:
            frames["left_log_timestamp_ns"] = np.asarray(
                rgb.frames["log_timestamp_ns"], dtype=np.int64
            )[rgb_indices]
        camera_frame_id = str(rgb.calibration["camera_frame_id"])
        source_world_frame_id = str(
            source.meta["coordinate_frame"]["world_frame_id"]
        )
        calibration = {
            "contract_version": CONTRACT_VERSION,
            "cameras": {
                "left": {
                    "model": "PINHOLE",
                    "width": int(output_camera["width"]),
                    "height": int(output_camera["height"]),
                    "K": output_k.tolist(),
                    "distortion": [0.0, 0.0, 0.0, 0.0],
                }
            },
            "image_geometry": "rectified",
            "rectification": copy.deepcopy(rgb.calibration["rectification"]),
            "sensor_frames": {
                "world": source_world_frame_id,
                "camera": camera_frame_id,
                "reference_position": camera_frame_id,
            },
            "transform_conventions": {
                "T_rgb_source_camera": "rgb_from_source_camera",
            },
            "camera_transfer": {
                "T_rgb_source_camera": T_rgb_source.tolist(),
                "extrinsic_translation_sigma_m": rgb_sigma.tolist(),
                "extrinsic_translation_sigma_frame_id": rgb.calibration[
                    "extrinsic_translation_sigma_frame_id"
                ],
                "provenance": copy.deepcopy(rgb.calibration["extrinsic_provenance"]),
            },
        }

        meta = copy.deepcopy(source.meta)
        meta["adapter"] = "derived_rgbd_transfer"
        meta["acquisition_id"] = source_acquisition_id
        meta["n_frames"] = len(rgb_timestamps)
        meta["capabilities"] = {
            "stereo": False,
            "rgbd": True,
            "single_rtk": False,
            "dual_rtk": False,
            "depth_recorded": False,
            "depth_computed": True,
            "imu_present": False,
            "images_raw": False,
            "images_rectified": True,
        }
        meta["timebase"] = {
            "frame_timestamp_source": rgb.meta["timestamp_source"],
            "observation_timestamp_source": (
                "derived reference at selected source depth timestamp"
            ),
            "unit": "ns",
            "association_clock_offset_ns": clock_offset,
        }
        meta["clock_alignment"] = {
            "model": "one_global_constant_rgb_to_source_offset",
            "rgb_to_source_clock_offset_ns": clock_offset,
            "association_method": association["method"],
            "depth_sync_tolerance_ns": config.max_depth_sync_residual_ns,
            "rolling_shutter_correction_applied": False,
        }
        meta["sensor_frames"] = {
            "world": source_world_frame_id,
            "camera": camera_frame_id,
            "reference_position": camera_frame_id,
        }
        meta["position_observation"] = {
            "type": "derived_reference_camera_position",
            "quantity": "rgb_camera_center",
            "sensor_frame_id": camera_frame_id,
            "coordinates": "ENU_m",
            "covariance_frame": "ENU_m2",
            "validity_field": "position_valid",
            "quality_field": "position_quality",
            "quality_vocabulary": list(POSITION_QUALITY_VOCABULARY),
        }
        meta.pop("heading_observation", None)
        meta["initial_pose"] = {
            **copy.deepcopy(source.meta["initial_pose"]),
            "camera_frame_id": camera_frame_id,
            "position_quantity": "left_camera_center",
            "source": (
                "derived from the immutable source segment's RTK-initial camera "
                "poses plus the calibrated source-camera-to-RGB transform; no "
                "target RTK observation was acquired"
            ),
            "extrinsic_translation_sigma_m": np.sqrt(
                np.diag(sigma_rgb_covariance)
            ).tolist(),
            "extrinsic_translation_sigma_frame_id": camera_frame_id,
        }
        meta["position_evidence"] = {
            "quantity": "derived RGB camera optical-centre position",
            "frame_id": camera_frame_id,
            "coordinates": "local ENU metres [east, north, up]",
            "role": (
                "derived reference only; no target RTK observation was acquired"
            ),
            "source": (
                "immutable source segment RTK evidence, accepted source pose "
                "artifact, and sealed RGB extrinsic"
            ),
        }
        meta.pop("heading_evidence", None)
        meta["initial_pose_semantics"] = {
            "camera_frame_id": camera_frame_id,
            "camera_centres": (
                "interpolated source RTK-initial camera centres transformed into "
                "the calibrated RGB optical frame"
            ),
            "not_direct_gnss": True,
            "target_rtk_acquired": False,
            "source_segment": str(source.root.resolve()),
            "source_pose_artifact": str(source_pose),
            "extrinsic_provenance": copy.deepcopy(
                rgb.calibration["extrinsic_provenance"]
            ),
        }
        meta["image_formats"] = {"left": ["png"]}
        meta["depth_observation"] = {
            "format": "npz_depth_valid",
            "units": "m",
            "quantity": "optical_z",
            "aligned_to": "rgb",
            "invalid_convention": "valid=false and depth=0",
        }
        meta["derived_segment"] = {
            "operation": "calibrated_motion_compensated_rgbd_transfer_v1",
            "source_segment": str(source.root.resolve()),
            "source_contract_sha256": _source_hashes(source),
            "source_pose_artifact": str(source_pose),
            "source_pose_manifest_sha256": sha256_file(source_pose / "manifest.json"),
            "source_pose_georeferencing_sha256": sha256_file(
                source_pose / "georeferencing.json"
            ),
            "source_pose_segment_binding": source_pose_binding,
            "rgb_observations": str(rgb.root),
            "rgb_observations_manifest_sha256": sha256_file(rgb.root / "manifest.json"),
            "config": asdict(config),
            "clock": copy.deepcopy(rgb.meta["clock"]),
            "clock_model": {
                "model": "one_global_constant_offset",
                "offset_ns": clock_offset,
                "limitations": [
                    "a global offset may be unobservable or insufficient",
                    "no rolling-shutter correction is claimed or applied",
                ],
            },
            "rgb_depth_association": association,
            "projected_depth_quality": projected_depth_quality,
            "held_out_registration_quality": registration_quality,
            "artifact_status": dict(artifact_status),
            "no_imu_consumed": True,
            "source_rtk_mode": (
                "dual_rtk" if capabilities["dual_rtk"] else "single_rtk"
            ),
            "raw_rtk_evidence_ownership": "immutable source segment only",
        }
        meta["provenance"] = {
            "operation": "calibrated_motion_compensated_rgbd_transfer_v1",
            "acquisition_id": source_acquisition_id,
            "source_segment": str(source.root.resolve()),
            "source_pose_artifact": str(source_pose),
            "rgb_observations": str(rgb.root),
            "target_rtk_acquired": False,
            "raw_rtk_evidence_ownership": "immutable source segment only",
        }
        split_by_source = np.empty(len(source_frames["frame_id"]), dtype="<U5")
        for split in ("train", "val", "test"):
            split_by_source[np.asarray(source.manifest[split], dtype=np.int64)] = split
        target_splits = split_by_source[depth_indices]
        manifest = {
            split: np.flatnonzero(target_splits == split).astype(int).tolist()
            for split in ("train", "val", "test")
        }
        manifest["policy"] = (
            "inherit source split through nearest depth-frame association"
        )
        writer.write_frames(frames)
        writer.write_calibration(calibration)
        writer.write_meta(meta)
        writer.write_manifest(manifest)
        writer.write_observations(
            "gnss",
            {
                "frame_id": frames["frame_id"],
                "frame_timestamp_ns": rgb_timestamps,
                "association_query_timestamp_ns": query_source_clock,
                "source_index": depth_indices,
                "source_timestamp_ns": selected_depth_timestamps,
                "source_residual_ns": depth_residual,
                "enu_m": rgb_initial_centers.astype(np.float64),
                "covariance_enu_m2": effective_covariance,
                "fix_status": np.full(len(rgb_timestamps), -1, dtype=np.int16),
                "carrier_status": np.full(len(rgb_timestamps), -1, dtype=np.int16),
                "position_valid": np.ones(len(rgb_timestamps), dtype=bool),
                "position_quality": np.asarray(
                    ["unknown_valid"] * len(rgb_timestamps), dtype=np.str_
                ),
            },
        )

        transfer_evaluation = {
            "schema_version": 1,
            "method": "no_fit_camera_transfer_consistency_against_interpolated_rtk",
            "n_frames": len(rgb_timestamps),
            "rgb_depth_association": association,
            "projected_depth_quality": projected_depth_quality,
            "held_out_registration_quality": registration_quality,
            "artifact_status": dict(artifact_status),
            "source_gate_policy": asdict(mapper_config),
            "source_gate_fields_defaulted": defaulted_gate_fields,
            "per_frame_evidence": "transfer_evidence.npz",
            "quality": transfer_quality,
            "pixel_registration_gates_passed": bool(
                registration_quality["passed"]
            ),
            "passed": False,
            "clock_model": {
                "model": "one_global_constant_offset",
                "offset_ns": clock_offset,
                "association_direction": (
                    "source depth frame to nearest corrected RGB observation"
                ),
                "selected_source_timestamp_field": (
                    "frames.depth_source_timestamp_ns"
                ),
                "signed_residual_field": "frames.depth_sync_residual_ns",
                "signed_residual_convention": (
                    "source_depth_timestamp_minus_corrected_rgb_timestamp"
                ),
                "opposite_signed_residual_field": (
                    "frames.corrected_rgb_minus_depth_sync_residual_ns"
                ),
                "rolling_shutter_correction_applied": False,
            },
            "covariance_limitations": [
                "source visual-pose uncertainty is not modelled",
                "clock-offset uncertainty is not modelled",
                "a single global clock offset may not explain motion-dependent residuals",
                "rolling-shutter readout is not corrected",
                "extrinsic rotation uncertainty is not modelled",
                "no parameters were fitted to target RTK observations",
            ],
        }
        pose_provenance = {
            "acquisition_id": source_acquisition_id,
            "source_segment": str(source.root.resolve()),
            "source_segment_hashes": _source_hashes(source),
            "source_pose_artifact": str(source_pose),
            "source_pose_manifest_sha256": sha256_file(source_pose / "manifest.json"),
            "source_pose_georeferencing_sha256": sha256_file(
                source_pose / "georeferencing.json"
            ),
            "source_pose_segment_binding": source_pose_binding,
            "rgb_observations": str(rgb.root),
            "rgb_observations_manifest_sha256": sha256_file(rgb.root / "manifest.json"),
            "backend": "calibrated_rgbd_transfer",
            "backend_config": asdict(config),
            "backend_config_effective": asdict(mapper_config),
            "no_mapper_or_bundle_adjustment_rerun": True,
            "no_imu_consumed": True,
            "T_rgb_source_camera": T_rgb_source.tolist(),
            "clock": copy.deepcopy(rgb.meta["clock"]),
            "rgb_depth_association": association,
            "projected_depth_quality": projected_depth_quality,
            "held_out_registration_quality": registration_quality,
            "selected_original_rgb_indices_field": (
                "transfer_evidence.npz:selected_rgb_indices"
            ),
        }
        destination_pose.parent.mkdir(parents=True, exist_ok=True)
        _write_pose_artifact(
            pose_staging,
            destination_pose.name,
            rgb_viewmats,
            rgb_centers,
            rgb_timestamps,
            np.asarray([Path(path).name for path in image_paths]),
            transfer_quality,
            pose_provenance,
            transfer_evaluation,
            {
                "residual_vectors_m": residual_vectors,
                "thresholds_m": thresholds,
                "effective_covariance_m2": effective_covariance,
                "selected_source_depth_indices": depth_indices,
                "selected_source_depth_frame_ids": np.asarray(
                    source_frames["frame_id"], dtype=np.int64
                )[depth_indices],
                "selected_source_depth_timestamps_ns": selected_depth_timestamps,
                "selected_rgb_indices": rgb_indices,
                "selected_rgb_frame_ids": np.asarray(
                    rgb.frames["frame_id"], dtype=np.int64
                )[rgb_indices],
                "corrected_rgb_minus_source_depth_residual_ns": (
                    rgb_minus_depth_residual
                ),
                "source_depth_valid_pixel_counts": np.asarray(
                    source_valid_counts, dtype=np.int64
                ),
                "projected_depth_valid_pixel_counts": np.asarray(
                    projected_valid_counts, dtype=np.int64
                ),
                "depth_pose_interpolation_lower_indices": depth_lower,
                "depth_pose_interpolation_upper_indices": depth_upper,
                "depth_pose_interpolation_alpha": depth_alpha,
                "missing_outside_gate_source_indices": np.asarray(
                    association["missing_outside_gate_source_indices"],
                    dtype=np.int64,
                ),
                "missing_duplicate_source_indices": np.asarray(
                    association["missing_duplicate_source_indices"],
                    dtype=np.int64,
                ),
                "unused_rgb_indices": np.asarray(
                    association["unused_rgb_indices"], dtype=np.int64
                ),
            },
            artifact_status,
        )
        verify_pose_georeferencing_artifact(
            pose_staging, expected_name=destination_pose.name
        )
        SegmentReader(writer.staging_dir).validate()
        result = writer.finalize()
        segment_published = True
        destination_pose.parent.mkdir(parents=True, exist_ok=True)
        publish_directory_noreplace(pose_staging, destination_pose)
        pose_published = True
        verify_pose_georeferencing_artifact(
            destination_pose, expected_name=destination_pose.name
        )
        return result, destination_pose
    except Exception:
        writer.abort()
        shutil.rmtree(pose_staging, ignore_errors=True)
        # Both paths were absent on entry and are owned by this invocation.
        # Roll back the first atomic publication if the second one fails.
        if pose_published:
            shutil.rmtree(destination_pose, ignore_errors=True)
        if segment_published:
            shutil.rmtree(destination_segment, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive georeferenced RGB-D from accepted stereo poses"
    )
    parser.add_argument("--source-segment", required=True, type=Path)
    parser.add_argument("--source-pose-artifact", required=True, type=Path)
    parser.add_argument("--rgb-observations", required=True, type=Path)
    parser.add_argument("--destination-segment", required=True, type=Path)
    parser.add_argument("--destination-pose-artifact", required=True, type=Path)
    parser.add_argument("--max-depth-sync-residual-ns", required=True, type=int)
    parser.add_argument("--max-pose-interpolation-gap-ns", required=True, type=int)
    parser.add_argument(
        "--min-depth-rgb-association-fraction", required=True, type=float
    )
    parser.add_argument(
        "--min-projected-depth-coverage-fraction", required=True, type=float
    )
    parser.add_argument(
        "--min-projected-depth-retained-fraction", required=True, type=float
    )
    parser.add_argument("--position-sigma-floor-m", type=float, default=0.01)
    parser.add_argument("--depth-fill-min-valid-neighbors", type=int, default=3)
    parser.add_argument("--depth-fill-max-spread-abs-m", type=float, default=0.05)
    parser.add_argument("--depth-fill-max-spread-rel", type=float, default=0.02)
    parser.add_argument("--held-out-registration-evidence", type=Path)
    parser.add_argument("--min-held-out-correspondences", type=int, default=20)
    parser.add_argument(
        "--max-held-out-median-reprojection-error-px", type=float, default=2.0
    )
    parser.add_argument(
        "--max-held-out-p95-reprojection-error-px", type=float, default=5.0
    )
    parser.add_argument(
        "--clock-observability-probe-ns", type=int, default=10_000_000
    )
    parser.add_argument(
        "--min-clock-observability-px-per-s", type=float, default=0.1
    )
    parser.add_argument(
        "--max-extrinsic-observability-condition-number",
        type=float,
        default=1_000_000.0,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    segment, pose = derive_rgbd_segment(
        args.source_segment,
        args.source_pose_artifact,
        args.rgb_observations,
        args.destination_segment,
        args.destination_pose_artifact,
        config=RgbdTransferConfig(
            max_depth_sync_residual_ns=args.max_depth_sync_residual_ns,
            max_pose_interpolation_gap_ns=args.max_pose_interpolation_gap_ns,
            min_depth_rgb_association_fraction=(
                args.min_depth_rgb_association_fraction
            ),
            min_projected_depth_coverage_fraction=(
                args.min_projected_depth_coverage_fraction
            ),
            min_projected_depth_retained_fraction=(
                args.min_projected_depth_retained_fraction
            ),
            position_sigma_floor_m=args.position_sigma_floor_m,
            depth_fill_min_valid_neighbors=args.depth_fill_min_valid_neighbors,
            depth_fill_max_spread_abs_m=args.depth_fill_max_spread_abs_m,
            depth_fill_max_spread_rel=args.depth_fill_max_spread_rel,
            held_out_registration_evidence=(
                str(args.held_out_registration_evidence)
                if args.held_out_registration_evidence is not None
                else None
            ),
            min_held_out_correspondences=args.min_held_out_correspondences,
            max_held_out_median_reprojection_error_px=(
                args.max_held_out_median_reprojection_error_px
            ),
            max_held_out_p95_reprojection_error_px=(
                args.max_held_out_p95_reprojection_error_px
            ),
            clock_observability_probe_ns=args.clock_observability_probe_ns,
            min_clock_observability_px_per_s=(
                args.min_clock_observability_px_per_s
            ),
            max_extrinsic_observability_condition_number=(
                args.max_extrinsic_observability_condition_number
            ),
        ),
    )
    print(json.dumps({"segment": str(segment.root), "pose_artifact": str(pose)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
