"""Adapter-side resolution of data-dependent frame sampling controls."""

import math
from typing import Any

import numpy as np

from rtk_splat.core.runtime_resolution import (
    record_override,
    record_runtime_resolution,
    runtime_control_value,
)


def _auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def resolve_frame_stride(
    cfg: Any,
    *,
    measured_camera_rate_hz: float,
    measured_median_speed_m_s: float,
) -> int:
    current = runtime_control_value(
        cfg,
        "frame_stride",
        getattr(cfg.segment, "frame_stride", None),
        config_path="segment.frame_stride",
    )
    if not _auto(current):
        if isinstance(current, bool) or not isinstance(current, (int, np.integer)):
            raise ValueError("segment.frame_stride must be positive or 'auto'")
        stride = int(current)
        if stride <= 0:
            raise ValueError("segment.frame_stride must be positive or 'auto'")
        record_override(cfg, "frame_stride", stride)
        return stride
    sampling = cfg.derivation.frame_sampling
    spacing = _positive(
        sampling.target_spacing_m, "derivation.frame_sampling.target_spacing_m"
    )
    rate = _positive(measured_camera_rate_hz, "measured_camera_rate_hz")
    speed = _positive(measured_median_speed_m_s, "measured_median_speed_m_s")
    stride = max(1, int(math.floor(spacing * rate / speed + 0.5)))
    record_runtime_resolution(
        cfg,
        "frame_stride",
        source="derived",
        formula="round(target_spacing_m * measured_camera_rate_hz / measured_median_speed_m_s)",
        inputs={
            "measured_camera_rate_hz": rate,
            "measured_median_speed_m_s": speed,
        },
        policy={"target_spacing_m": spacing, "minimum_stride": 1},
        chosen=stride,
    )
    cfg.segment.frame_stride = stride
    return stride


def resolve_metric_frame_spacing(cfg: Any) -> float:
    current = runtime_control_value(
        cfg,
        "frame_spacing_m",
        getattr(cfg.segment, "frame_spacing_m", None),
        config_path="segment.frame_spacing_m",
    )
    if not _auto(current):
        spacing = _positive(current, "segment.frame_spacing_m")
        record_override(cfg, "frame_spacing_m", spacing)
        return spacing
    spacing = _positive(
        cfg.derivation.frame_sampling.target_spacing_m,
        "derivation.frame_sampling.target_spacing_m",
    )
    record_runtime_resolution(
        cfg,
        "frame_spacing_m",
        source="derived",
        formula="metric sampler target equals quality-policy target_spacing_m",
        inputs={"sampler_quantity": "GNSS trajectory arc length"},
        policy={"target_spacing_m": spacing},
        chosen=spacing,
    )
    cfg.segment.frame_spacing_m = spacing
    return spacing
