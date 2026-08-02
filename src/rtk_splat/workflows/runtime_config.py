"""Workflow resolution of data-dependent depth and GS training controls.

These rules are named *quality policy*, not physical truth.  They translate
measured properties of one segment and machine into bounded operational
values.  A numeric authored value always wins and is recorded as an override;
automatic resolution occurs only for the literal ``auto``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rtk_splat.core.runtime_resolution import (
    commit_runtime_resolution_records,
    make_runtime_resolution_record,
    runtime_control_value,
    record_runtime_resolution,
)


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


def _section(owner: Any, name: str) -> Any:
    value = getattr(owner, name, None)
    if value is None:
        raise ValueError(f"automatic runtime resolution requires '{name}:' policy")
    return value


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def resolve_depth_max_z(cfg: Any, calibration: Mapping[str, Any]) -> float:
    """Resolve maximum stereo range from fB and a bounded disparity policy."""
    current = runtime_control_value(
        cfg,
        "depth_max_z_m",
        getattr(cfg.depth, "max_z_m", None),
        config_path="depth.max_z_m",
    )
    if not _is_auto(current):
        maximum = _positive(current, "depth.max_z_m")
        record_runtime_resolution(
            cfg,
            "depth_max_z_m",
            source="override",
            formula="authored or CLI numeric value; no runtime formula applied",
            inputs={},
            policy={},
            chosen=maximum,
        )
        return maximum
    policy = _section(_section(cfg, "derivation"), "depth")
    disparity = _positive(
        policy.min_reliable_disparity_px,
        "derivation.depth.min_reliable_disparity_px",
    )
    cap = _positive(policy.max_z_cap_m, "derivation.depth.max_z_cap_m")
    cameras = calibration.get("cameras")
    if not isinstance(cameras, Mapping) or not isinstance(
        cameras.get("left"), Mapping
    ):
        raise ValueError("depth range derivation needs calibration.cameras.left")
    k = np.asarray(cameras["left"].get("K"), dtype=np.float64)
    transform = np.asarray(calibration.get("T_right_left"), dtype=np.float64)
    if k.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("depth range derivation needs valid K and T_right_left")
    fx = _positive(k[0, 0], "calibration left fx")
    baseline = _positive(np.linalg.norm(transform[:3, 3]), "stereo baseline")
    physics_range = fx * baseline / disparity
    maximum = min(physics_range, cap)
    minimum = _positive(cfg.depth.min_z_m, "depth.min_z_m")
    if maximum <= minimum:
        raise ValueError(
            "derived depth.max_z_m does not exceed depth.min_z_m; adjust the "
            "reliable-disparity or maximum-range policy"
        )
    record_runtime_resolution(
        cfg,
        "depth_max_z_m",
        source="derived",
        formula="min(fx_px * stereo_baseline_m / min_reliable_disparity_px, max_z_cap_m)",
        inputs={
            "fx_px": fx,
            "stereo_baseline_m": baseline,
            "f_times_baseline_px_m": fx * baseline,
        },
        policy={
            "min_reliable_disparity_px": disparity,
            "max_z_cap_m": cap,
            "min_z_m": minimum,
        },
        chosen=maximum,
    )
    cfg.depth.max_z_m = maximum
    return maximum


def detect_cuda_total_memory_gib() -> float:
    """Read total CUDA memory lazily; never import Torch during config load."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "automatic max_gaussians needs PyTorch CUDA or "
            "derivation.training.vram_gib_override"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "automatic max_gaussians needs CUDA or "
            "derivation.training.vram_gib_override"
        )
    return float(torch.cuda.get_device_properties(0).total_memory / 2**30)


def _training_view_count(segment: Any, use_right_camera: bool) -> tuple[int, int]:
    manifest = segment.manifest
    train_pairs = len(manifest["train"])
    if train_pairs <= 0:
        raise ValueError("training derivation needs at least one training frame")
    return train_pairs, train_pairs * (2 if use_right_camera else 1)


def resolve_training_controls(
    cfg: Any,
    segment: Any,
    initial_cloud: str | Path,
    *,
    cuda_total_memory_gib: float | None = None,
) -> tuple[int, int]:
    """Resolve GS iterations and capacity after segment/cloud publication."""
    iteration_value = runtime_control_value(
        cfg,
        "train_iterations",
        getattr(cfg.train, "iterations", None),
        config_path="train.iterations",
    )
    gaussian_value = runtime_control_value(
        cfg,
        "train_max_gaussians",
        getattr(cfg.train, "max_gaussians", None),
        config_path="train.max_gaussians",
    )
    needs_policy = _is_auto(iteration_value) or _is_auto(gaussian_value)
    policy = (
        _section(_section(cfg, "derivation"), "training")
        if needs_policy
        else None
    )
    records = []
    if _is_auto(iteration_value):
        train_pairs, train_views = _training_view_count(
            segment, bool(getattr(cfg.train, "use_right_camera", False))
        )
        presentations = _positive(
            policy.image_presentations_per_view,
            "derivation.training.image_presentations_per_view",
        )
        minimum_iterations = _positive_int(
            policy.min_iterations, "derivation.training.min_iterations"
        )
        maximum_iterations = _positive_int(
            policy.max_iterations, "derivation.training.max_iterations"
        )
        if maximum_iterations < minimum_iterations:
            raise ValueError("training maximum iterations is below its minimum")
        raw_iterations = int(math.ceil(train_views * presentations))
        iterations = max(
            minimum_iterations, min(maximum_iterations, raw_iterations)
        )
        records.append(make_runtime_resolution_record(
            cfg,
            "train_iterations",
            source="derived",
            formula="clamp(ceil(training_views * image_presentations_per_view), min_iterations, max_iterations)",
            inputs={
                "training_pairs": train_pairs,
                "training_views": train_views,
                "right_camera_supervision": bool(
                    getattr(cfg.train, "use_right_camera", False)
                ),
            },
            policy={
                "image_presentations_per_view": presentations,
                "min_iterations": minimum_iterations,
                "max_iterations": maximum_iterations,
            },
            chosen=iterations,
        ))
    else:
        iterations = _positive_int(iteration_value, "train.iterations")
        records.append(make_runtime_resolution_record(
            cfg,
            "train_iterations",
            source="override",
            formula="authored or CLI numeric value; no runtime formula applied",
            inputs={},
            policy={},
            chosen=iterations,
        ))

    if _is_auto(gaussian_value):
        cloud_path = Path(initial_cloud)
        try:
            with np.load(cloud_path, allow_pickle=False) as archive:
                initial_points = int(len(archive["xyz"]))
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(
                f"cannot inspect initial cloud for max_gaussians: {cloud_path}"
            ) from exc
        if initial_points <= 0:
            raise ValueError("initial cloud contains no points")
        configured_vram = getattr(policy, "vram_gib_override", None)
        if configured_vram is not None:
            total_vram = _positive(
                configured_vram, "derivation.training.vram_gib_override"
            )
            vram_source = "policy_override"
        elif cuda_total_memory_gib is not None:
            total_vram = _positive(
                cuda_total_memory_gib, "cuda_total_memory_gib"
            )
            vram_source = "injected_measurement"
        else:
            total_vram = detect_cuda_total_memory_gib()
            vram_source = "torch_cuda_device_total_memory"
        reserve = _positive(
            policy.reserve_vram_gib,
            "derivation.training.reserve_vram_gib",
        )
        per_gib = _positive(
            policy.gaussians_per_gib,
            "derivation.training.gaussians_per_gib",
        )
        growth = _positive(
            policy.initial_cloud_growth_factor,
            "derivation.training.initial_cloud_growth_factor",
        )
        minimum_gaussians = _positive_int(
            policy.min_gaussians, "derivation.training.min_gaussians"
        )
        maximum_gaussians = _positive_int(
            policy.max_gaussians, "derivation.training.max_gaussians"
        )
        usable_vram = total_vram - reserve
        if usable_vram <= 0:
            raise ValueError("reserved VRAM leaves no memory for Gaussian training")
        vram_budget = int(math.floor(usable_vram * per_gib))
        if vram_budget < minimum_gaussians:
            raise ValueError(
                "measured VRAM budget is below quality_v1 min_gaussians; "
                "choose a smaller explicit cap or a different policy"
            )
        scene_demand = max(
            minimum_gaussians,
            int(math.ceil(initial_points * growth)),
        )
        gaussians = min(maximum_gaussians, vram_budget, scene_demand)
        records.append(make_runtime_resolution_record(
            cfg,
            "train_max_gaussians",
            source="derived",
            formula="min(policy_max, floor((total_vram_gib - reserve_vram_gib) * gaussians_per_gib), max(policy_min, ceil(initial_cloud_points * growth_factor)))",
            inputs={
                "initial_cloud_points": initial_points,
                "total_vram_gib": total_vram,
                "vram_measurement_source": vram_source,
                "vram_budget_gaussians": vram_budget,
                "scene_demand_gaussians": scene_demand,
            },
            policy={
                "reserve_vram_gib": reserve,
                "gaussians_per_gib": per_gib,
                "initial_cloud_growth_factor": growth,
                "min_gaussians": minimum_gaussians,
                "max_gaussians": maximum_gaussians,
            },
            chosen=gaussians,
        ))
    else:
        gaussians = _positive_int(gaussian_value, "train.max_gaussians")
        records.append(make_runtime_resolution_record(
            cfg,
            "train_max_gaussians",
            source="override",
            formula="authored or CLI numeric value; no runtime formula applied",
            inputs={},
            policy={},
            chosen=gaussians,
        ))
    # Commit both evidence records and operational values only after every
    # measurement and policy gate succeeds. A failed Gaussian budget can never
    # relabel or partially apply the iteration decision on a later retry.
    commit_runtime_resolution_records(cfg, records)
    cfg.train.iterations = iterations
    cfg.train.max_gaussians = gaussians
    return iterations, gaussians
