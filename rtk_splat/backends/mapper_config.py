"""Validated configuration for COLMAP mapper backends."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


MapperKind = Literal["global", "incremental"]
CovarianceGateMode = Literal["diagnostic_only", "enforce"]


@dataclass(frozen=True)
class MapperConfig:
    """Shared controls for one isolated mapper attempt."""

    backend: MapperKind = "global"
    num_threads: int = 8
    random_seed: int = 7
    min_num_matches: int = 15
    ba_num_iterations: int = 3
    keep_max_num_tracks: int = 60_000
    track_required_tracks_per_view: int = 1_000
    skip_retriangulation: bool = True
    gp_use_gpu: bool = False
    ba_ceres_use_gpu: bool = False
    process_nice: int = 10
    resource_sample_interval_s: float = 2.0
    minimum_available_memory_gb: float = 4.0
    low_memory_consecutive_samples: int = 2
    minimum_free_space_gb: float = 15.0
    minimum_runtime_free_space_gb: float = 5.0
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
    rtk_chi2_inlier_probability: float = 0.95
    max_rtk_median_mahalanobis_sq: float = 4.108344935632312
    min_rtk_chi2_inlier_fraction: float = 0.80
    rtk_covariance_gate_mode: CovarianceGateMode = "diagnostic_only"

    def __post_init__(self) -> None:
        if self.backend not in ("global", "incremental"):
            raise ValueError("backend must be 'global' or 'incremental'")
        for name in (
            "num_threads",
            "min_num_matches",
            "ba_num_iterations",
            "keep_max_num_tracks",
            "track_required_tracks_per_view",
            "low_memory_consecutive_samples",
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
        if (
            isinstance(self.process_nice, bool)
            or not isinstance(self.process_nice, int)
            or not 0 <= self.process_nice <= 19
        ):
            raise ValueError("process_nice must be an integer in [0, 19]")
        for name in (
            "skip_retriangulation",
            "gp_use_gpu",
            "ba_ceres_use_gpu",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in (
            "max_reprojection_error_px",
            "min_mean_track_length",
            "alignment_ransac_threshold_m",
            "max_rtk_median_error_m",
            "max_rtk_p95_inlier_error_m",
            "resource_sample_interval_s",
            "minimum_available_memory_gb",
            "minimum_free_space_gb",
            "minimum_runtime_free_space_gb",
            "max_rtk_median_mahalanobis_sq",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_runtime_free_space_gb > self.minimum_free_space_gb:
            raise ValueError(
                "minimum_runtime_free_space_gb cannot exceed "
                "minimum_free_space_gb"
            )
        if not 0 < self.min_rtk_inlier_fraction <= 1:
            raise ValueError("min_rtk_inlier_fraction must be in (0, 1]")
        if not 0 < self.rtk_chi2_inlier_probability < 1:
            raise ValueError("rtk_chi2_inlier_probability must be in (0, 1)")
        if not 0 < self.min_rtk_chi2_inlier_fraction <= 1:
            raise ValueError("min_rtk_chi2_inlier_fraction must be in (0, 1]")
        if self.rtk_covariance_gate_mode not in ("diagnostic_only", "enforce"):
            raise ValueError(
                "rtk_covariance_gate_mode must be 'diagnostic_only' or 'enforce'"
            )


# These controls affect only post-solve RTK evaluation.  A backend prepared
# before they existed may safely use their recorded defaults without changing
# its visual solve, database, or registered model.  No mapper/solve option is
# eligible for this compatibility path.
_LEGACY_EVALUATION_DEFAULTS = {
    "rtk_chi2_inlier_probability": MapperConfig.rtk_chi2_inlier_probability,
    "max_rtk_median_mahalanobis_sq": (
        MapperConfig.max_rtk_median_mahalanobis_sq
    ),
    "min_rtk_chi2_inlier_fraction": (
        MapperConfig.min_rtk_chi2_inlier_fraction
    ),
    "rtk_covariance_gate_mode": MapperConfig.rtk_covariance_gate_mode,
}
