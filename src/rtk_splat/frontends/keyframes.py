"""Deterministic metric keyframe selection with auditable reasons."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class KeyframeConfig:
    select_all_frames: bool = False
    translation_m: float = 0.10
    rotation_deg: float = 2.0
    max_elapsed_s: float = 1.5
    min_blur_score: float = -np.inf
    min_exposure_quality: float = -np.inf
    max_stereo_sync_residual_s: float = np.inf
    max_rtk_covariance_m2: float = np.inf
    acceptable_rtk_status: tuple[int, ...] | None = None
    revisit_distance_m: float = 0.75
    revisit_min_separation_s: float = 10.0
    revisit_force_spacing_s: float = 2.0
    revisit_points_per_cell: int = 16

    def __post_init__(self):
        if type(self.select_all_frames) is not bool:
            raise ValueError("select_all_frames must be boolean")
        positive = {
            "translation_m": self.translation_m,
            "rotation_deg": self.rotation_deg,
            "max_elapsed_s": self.max_elapsed_s,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        nonnegative = {
            "revisit_distance_m": self.revisit_distance_m,
            "revisit_min_separation_s": self.revisit_min_separation_s,
            "revisit_force_spacing_s": self.revisit_force_spacing_s,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if np.isnan(self.min_blur_score) or np.isnan(self.min_exposure_quality):
            raise ValueError("quality minimums must not be NaN")
        for name in (
            "max_stereo_sync_residual_s",
            "max_rtk_covariance_m2",
        ):
            value = getattr(self, name)
            if np.isnan(value) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            isinstance(self.revisit_points_per_cell, bool)
            or not isinstance(self.revisit_points_per_cell, (int, np.integer))
            or self.revisit_points_per_cell <= 0
        ):
            raise ValueError("revisit_points_per_cell must be positive")
        if self.acceptable_rtk_status is not None and any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in self.acceptable_rtk_status
        ):
            raise ValueError("acceptable_rtk_status must contain integers")


KEYFRAME_PRESETS = {
    "all": KeyframeConfig(select_all_frames=True),
    "dense": KeyframeConfig(translation_m=0.08, rotation_deg=1.0),
    "balanced": KeyframeConfig(translation_m=0.10, rotation_deg=2.0),
    "sparse": KeyframeConfig(translation_m=0.15, rotation_deg=3.0),
}


@dataclass(frozen=True)
class FrameQuality:
    blur_score: float | None
    exposure_quality: float | None
    stereo_sync_residual_s: float | None
    rtk_covariance_m2: float | None
    rtk_status: int | None
    acceptable: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class Keyframe:
    frame_index: int
    timestamp_s: float
    reasons: tuple[str, ...]
    quality: FrameQuality
    revisit_source_index: int | None


@dataclass(frozen=True)
class KeyframeSelection:
    frame_count: int
    keyframes: tuple[Keyframe, ...]
    frame_quality: tuple[FrameQuality, ...]
    revisit_source_indices: np.ndarray

    @property
    def indices(self) -> np.ndarray:
        return np.asarray(
            [item.frame_index for item in self.keyframes], dtype=np.int64
        )

    @property
    def solve_mask(self) -> np.ndarray:
        mask = np.zeros(self.frame_count, dtype=bool)
        mask[self.indices] = True
        return mask


_REASON_ORDER = (
    "all_frames",
    "endpoint",
    "turn_entry",
    "turn_peak",
    "turn_exit",
    "gnss_before_change",
    "gnss_state_change",
    "spatial_revisit_source",
    "spatial_revisit",
    "translation",
    "rotation",
    "max_elapsed",
    "quality_override",
)


def _timestamps(values: Sequence[float] | np.ndarray) -> np.ndarray:
    timestamps = np.asarray(values, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError("timestamps_s must be a non-empty vector")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps_s must be finite and strictly increasing")
    return timestamps


def _positions(values, n: int) -> np.ndarray:
    positions = np.asarray(values, dtype=np.float64)
    if positions.shape != (n, 3) or not np.isfinite(positions).all():
        raise ValueError(f"positions_m must have finite shape ({n}, 3)")
    return positions


def _orientations(values, n: int) -> np.ndarray:
    orientations = np.asarray(values, dtype=np.float64)
    if orientations.shape not in {(n, 3, 3), (n, 4)}:
        raise ValueError(
            f"orientations must have shape ({n}, 3, 3) or ({n}, 4)"
        )
    if not np.isfinite(orientations).all():
        raise ValueError("orientations must be finite")
    if orientations.shape == (n, 4):
        norms = np.linalg.norm(orientations, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("orientation quaternions must be non-zero")
        orientations = orientations / norms[:, None]
    return orientations


def _rotation_deg(orientations: np.ndarray, first: int, second: int) -> float:
    if orientations.ndim == 2:
        cosine = abs(float(np.dot(orientations[first], orientations[second])))
        return float(np.degrees(2.0 * np.arccos(np.clip(cosine, 0.0, 1.0))))
    relative = orientations[first].T @ orientations[second]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _optional(values, n: int, name: str) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},)")
    return array


def _status(values, n: int) -> np.ndarray | None:
    status = _optional(values, n, "rtk_status")
    if status is not None and status.dtype.kind not in "iu":
        raise ValueError("rtk_status must contain integers")
    return status


def _covariance_score(values, n: int) -> np.ndarray | None:
    if values is None:
        return None
    covariance = np.asarray(values, dtype=np.float64)
    if covariance.shape == (n,):
        return covariance
    if covariance.shape == (n, 3, 3):
        return np.max(np.diagonal(covariance, axis1=1, axis2=2), axis=1)
    raise ValueError(
        f"rtk_covariance_m2 must have shape ({n},) or ({n}, 3, 3)"
    )


def _quality_records(
    n: int,
    config: KeyframeConfig,
    blur_score,
    exposure_quality,
    stereo_sync_residual_s,
    rtk_covariance_m2,
    rtk_status,
) -> tuple[FrameQuality, ...]:
    blur = _optional(blur_score, n, "blur_score")
    exposure = _optional(exposure_quality, n, "exposure_quality")
    sync = _optional(stereo_sync_residual_s, n, "stereo_sync_residual_s")
    covariance = _covariance_score(rtk_covariance_m2, n)
    status = _status(rtk_status, n)
    output = []
    for index in range(n):
        issues = []
        if blur is not None and (
            not np.isfinite(blur[index])
            or blur[index] < config.min_blur_score
        ):
            issues.append("blur")
        if exposure is not None and (
            not np.isfinite(exposure[index])
            or exposure[index] < config.min_exposure_quality
        ):
            issues.append("exposure")
        if sync is not None and (
            not np.isfinite(sync[index])
            or abs(sync[index]) > config.max_stereo_sync_residual_s
        ):
            issues.append("stereo_sync")
        if covariance is not None and (
            not np.isfinite(covariance[index])
            or covariance[index] > config.max_rtk_covariance_m2
        ):
            issues.append("rtk_covariance")
        state = None if status is None else int(status[index])
        if (
            state is not None
            and config.acceptable_rtk_status is not None
            and state not in config.acceptable_rtk_status
        ):
            issues.append("rtk_status")
        output.append(
            FrameQuality(
                blur_score=None if blur is None else float(blur[index]),
                exposure_quality=(
                    None if exposure is None else float(exposure[index])
                ),
                stereo_sync_residual_s=(
                    None if sync is None else float(sync[index])
                ),
                rtk_covariance_m2=(
                    None if covariance is None else float(covariance[index])
                ),
                rtk_status=state,
                acceptable=not issues,
                issues=tuple(issues),
            )
        )
    return tuple(output)


def _add_reason(reasons: dict[int, set[str]], index: int, reason: str) -> None:
    reasons.setdefault(index, set()).add(reason)


def _reached(value: float, threshold: float) -> bool:
    tolerance = 1e-12 * max(1.0, abs(threshold))
    return value + tolerance >= threshold


def _turn_reasons(
    turn_region: np.ndarray, orientations: np.ndarray, reasons: dict[int, set[str]]
) -> None:
    indices = np.flatnonzero(turn_region)
    if indices.size == 0:
        return
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    for run in np.split(indices, splits):
        start, end = int(run[0]), int(run[-1])
        step_angles = [
            _rotation_deg(orientations, max(0, index - 1), index)
            for index in run
        ]
        peak = int(run[int(np.argmax(step_angles))])
        _add_reason(reasons, start, "turn_entry")
        _add_reason(reasons, peak, "turn_peak")
        _add_reason(reasons, end, "turn_exit")


def _revisit_sources(
    timestamps: np.ndarray,
    positions: np.ndarray,
    config: KeyframeConfig,
) -> np.ndarray:
    n = len(timestamps)
    sources = np.full(n, -1, dtype=np.int64)
    if config.revisit_distance_m == 0:
        return sources
    cell_width = config.revisit_distance_m
    cells: dict[tuple[int, int, int], list[int]] = {}
    eligible = 0
    offsets = tuple(product((-1, 0, 1), repeat=3))
    for index in range(n):
        while (
            eligible < index
            and _reached(
                float(timestamps[index] - timestamps[eligible]),
                config.revisit_min_separation_s,
            )
        ):
            cell = tuple(np.floor(positions[eligible] / cell_width).astype(int))
            bucket = cells.setdefault(cell, [])
            if len(bucket) < config.revisit_points_per_cell:
                bucket.append(eligible)
            else:
                bucket[-1] = eligible
            eligible += 1
        cell = tuple(np.floor(positions[index] / cell_width).astype(int))
        candidates = []
        for offset in offsets:
            neighbour = tuple(cell[axis] + offset[axis] for axis in range(3))
            candidates.extend(cells.get(neighbour, ()))
        if candidates:
            candidate_array = np.asarray(candidates, dtype=np.int64)
            distance = np.linalg.norm(
                positions[candidate_array] - positions[index], axis=1
            )
            tolerance = 1e-12 * max(1.0, config.revisit_distance_m)
            valid = distance <= config.revisit_distance_m + tolerance
            if np.any(valid):
                valid_indices = candidate_array[valid]
                valid_distance = distance[valid]
                order = np.lexsort((valid_indices, valid_distance))
                sources[index] = valid_indices[order[0]]
    return sources


def select_keyframes(
    timestamps_s,
    positions_m,
    orientations,
    *,
    config: KeyframeConfig = KEYFRAME_PRESETS["balanced"],
    blur_score=None,
    exposure_quality=None,
    stereo_sync_residual_s=None,
    rtk_covariance_m2=None,
    rtk_status=None,
    turn_region=None,
) -> KeyframeSelection:
    """Select solve keyframes while retaining an audit trail per frame.

    Orientations may be rotation matrices or normalized-on-input XYZW
    quaternions. Quality thresholds gate ordinary motion candidates; endpoints,
    elapsed-time bounds, turns, revisits, and GNSS transitions are retained
    even when their quality is poor and receive a ``quality_override`` reason.
    """
    timestamps = _timestamps(timestamps_s)
    n = len(timestamps)
    positions = _positions(positions_m, n)
    rotations = _orientations(orientations, n)
    quality = _quality_records(
        n,
        config,
        blur_score,
        exposure_quality,
        stereo_sync_residual_s,
        rtk_covariance_m2,
        rtk_status,
    )
    if config.select_all_frames:
        return KeyframeSelection(
            frame_count=n,
            keyframes=tuple(
                Keyframe(
                    frame_index=index,
                    timestamp_s=float(timestamps[index]),
                    reasons=(
                        ("all_frames",)
                        if quality[index].acceptable
                        else ("all_frames", "quality_override")
                    ),
                    quality=quality[index],
                    revisit_source_index=None,
                )
                for index in range(n)
            ),
            frame_quality=quality,
            revisit_source_indices=np.full(n, -1, dtype=np.int64),
        )
    turns = (
        np.zeros(n, dtype=bool)
        if turn_region is None
        else np.asarray(turn_region, dtype=bool)
    )
    if turns.shape != (n,):
        raise ValueError(f"turn_region must have shape ({n},)")

    forced: dict[int, set[str]] = {}
    _add_reason(forced, 0, "endpoint")
    _add_reason(forced, n - 1, "endpoint")
    _turn_reasons(turns, rotations, forced)

    states = _status(rtk_status, n)
    if states is not None:
        for index in np.flatnonzero(states[1:] != states[:-1]) + 1:
            _add_reason(forced, int(index - 1), "gnss_before_change")
            _add_reason(forced, int(index), "gnss_state_change")

    revisit_sources = _revisit_sources(timestamps, positions, config)
    last_revisit_time = -np.inf
    for index in np.flatnonzero(revisit_sources >= 0):
        if (
            timestamps[index] - last_revisit_time
            < config.revisit_force_spacing_s
        ):
            continue
        source = int(revisit_sources[index])
        _add_reason(forced, source, "spatial_revisit_source")
        _add_reason(forced, int(index), "spatial_revisit")
        last_revisit_time = timestamps[index]

    selected = []
    last_selected = None
    hard_reasons = {
        "endpoint",
        "turn_entry",
        "turn_peak",
        "turn_exit",
        "gnss_before_change",
        "gnss_state_change",
        "spatial_revisit_source",
        "spatial_revisit",
        "max_elapsed",
    }
    for index in range(n):
        reasons = set(forced.get(index, ()))
        if last_selected is not None:
            translation = float(
                np.linalg.norm(positions[index] - positions[last_selected])
            )
            rotation = _rotation_deg(rotations, last_selected, index)
            elapsed = timestamps[index] - timestamps[last_selected]
            if _reached(translation, config.translation_m):
                reasons.add("translation")
            if _reached(rotation, config.rotation_deg):
                reasons.add("rotation")
            if _reached(float(elapsed), config.max_elapsed_s):
                reasons.add("max_elapsed")
        if not reasons:
            continue
        is_forced = bool(reasons.intersection(hard_reasons))
        if not quality[index].acceptable and not is_forced:
            continue
        if not quality[index].acceptable:
            reasons.add("quality_override")
        ordered = tuple(reason for reason in _REASON_ORDER if reason in reasons)
        source = int(revisit_sources[index])
        selected.append(
            Keyframe(
                frame_index=index,
                timestamp_s=float(timestamps[index]),
                reasons=ordered,
                quality=quality[index],
                revisit_source_index=None if source < 0 else source,
            )
        )
        last_selected = index

    return KeyframeSelection(
        frame_count=n,
        keyframes=tuple(selected),
        frame_quality=quality,
        revisit_source_indices=revisit_sources,
    )
