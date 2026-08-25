"""Pure raw-GNSS eligibility rules for bounded geodetic submaps.

The policy in this module deliberately has no access to a reconstructed
visual model, visual-model residuals, or held-out evaluation results.  It can
therefore be used while a private optimizer database is being planned without
creating a circular acceptance test.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


_POSITION_QUALITIES = (
    "unknown_valid",
    "standalone",
    "differential",
    "rtk_float",
    "rtk_fixed",
    "oracle",
)


@dataclass(frozen=True)
class GeodeticPairPolicy:
    """Conservative physical bounds for nonlocal visual pair evidence."""

    nearby_frame_gap: int = 2
    nearby_time_s: float = 2.0
    revisit_distance_m: float = 1.0
    revisit_physical_cap_m: float = 1.5
    covariance_sigma_multiplier: float = 3.0
    max_endpoint_position_std_m: float = 0.5
    strong_verified_matches: int = 30

    def __post_init__(self) -> None:
        if (
            isinstance(self.nearby_frame_gap, bool)
            or not isinstance(self.nearby_frame_gap, int)
            or self.nearby_frame_gap < 1
        ):
            raise ValueError("nearby_frame_gap must be a positive integer")
        if (
            isinstance(self.strong_verified_matches, bool)
            or not isinstance(self.strong_verified_matches, int)
            or self.strong_verified_matches < 1
        ):
            raise ValueError(
                "strong_verified_matches must be a positive integer"
            )
        for name in (
            "nearby_time_s",
            "revisit_distance_m",
            "revisit_physical_cap_m",
            "covariance_sigma_multiplier",
            "max_endpoint_position_std_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.revisit_distance_m > self.revisit_physical_cap_m:
            raise ValueError(
                "revisit_distance_m cannot exceed revisit_physical_cap_m"
            )


@dataclass(frozen=True)
class RawGnssEndpoint:
    """Raw position evidence attached to one acquisition frame."""

    position_m: tuple[float, float, float] | None
    covariance_m2: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None
    position_valid: bool
    position_quality: str
    fix_status: int
    carrier_status: int

    def trusted_std_m(self) -> float | None:
        if (
            not self.position_valid
            or self.position_quality not in _POSITION_QUALITIES
            or self.position_m is None
            or self.covariance_m2 is None
        ):
            return None
        position = np.asarray(self.position_m, dtype=np.float64)
        covariance = np.asarray(self.covariance_m2, dtype=np.float64)
        if (
            position.shape != (3,)
            or covariance.shape != (3, 3)
            or not np.isfinite(position).all()
            or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, atol=1e-10)
        ):
            return None
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues[0] < -1e-10:
            return None
        return float(math.sqrt(max(float(eigenvalues[-1]), 0.0)))

    def record(self) -> dict[str, Any]:
        return {
            "position_m": (
                list(self.position_m) if self.position_m is not None else None
            ),
            "covariance_m2": (
                [list(row) for row in self.covariance_m2]
                if self.covariance_m2 is not None
                else None
            ),
            "position_valid": self.position_valid,
            "position_quality": self.position_quality,
            "fix_status": self.fix_status,
            "carrier_status": self.carrier_status,
            "maximum_position_std_m": self.trusted_std_m(),
        }


@dataclass(frozen=True)
class PairCandidate:
    """Model-independent evidence for one existing COLMAP image pair."""

    pair_id: int
    first_image_id: int
    second_image_id: int
    first_image_name: str
    second_image_name: str
    first_frame_id: int
    second_frame_id: int
    first_frame_index: int
    second_frame_index: int
    first_camera: str
    second_camera: str
    first_timestamp_ns: int
    second_timestamp_ns: int
    raw_matches: int | None
    verified_matches: int | None
    first_gnss: RawGnssEndpoint
    second_gnss: RawGnssEndpoint

    def __post_init__(self) -> None:
        for name in (
            "pair_id",
            "first_image_id",
            "second_image_id",
            "first_frame_id",
            "second_frame_id",
            "first_frame_index",
            "second_frame_index",
            "first_timestamp_ns",
            "second_timestamp_ns",
        ):
            if isinstance(getattr(self, name), bool) or not isinstance(
                getattr(self, name), int
            ):
                raise ValueError(f"{name} must be an integer")
        if self.first_image_id >= self.second_image_id:
            raise ValueError("pair image IDs must use ascending COLMAP order")
        if not self.first_image_name or not self.second_image_name:
            raise ValueError("pair image names must be non-empty")
        for name in ("raw_matches", "verified_matches"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be non-negative or None")


@dataclass(frozen=True)
class PairDecision:
    """One auditable pair disposition."""

    retained: bool
    reason: str
    evidence: dict[str, Any]

    def record(self, candidate: PairCandidate) -> dict[str, Any]:
        return {
            "pair_id": candidate.pair_id,
            "images": [candidate.first_image_name, candidate.second_image_name],
            "image_ids": [candidate.first_image_id, candidate.second_image_id],
            "frame_ids": [candidate.first_frame_id, candidate.second_frame_id],
            "frame_indices": [
                candidate.first_frame_index,
                candidate.second_frame_index,
            ],
            "cameras": [candidate.first_camera, candidate.second_camera],
            "timestamps_ns": [
                candidate.first_timestamp_ns,
                candidate.second_timestamp_ns,
            ],
            "raw_matches": candidate.raw_matches,
            "verified_matches": candidate.verified_matches,
            "retained": self.retained,
            "reason": self.reason,
            "evidence": self.evidence,
        }


def decide_pair(
    candidate: PairCandidate,
    policy: GeodeticPairPolicy = GeodeticPairPolicy(),
) -> PairDecision:
    """Classify a pair without consulting any finished visual trajectory."""
    same_frame = candidate.first_frame_id == candidate.second_frame_id
    same_frame_stereo = same_frame and (
        candidate.first_camera != candidate.second_camera
    )
    frame_gap = abs(
        candidate.second_frame_index - candidate.first_frame_index
    )
    time_gap_s = (
        abs(candidate.second_timestamp_ns - candidate.first_timestamp_ns)
        * 1.0e-9
    )
    common = {
        "same_frame_stereo": same_frame_stereo,
        "frame_index_gap": frame_gap,
        "time_gap_s": time_gap_s,
        "policy": asdict(policy),
        "gnss": [
            candidate.first_gnss.record(),
            candidate.second_gnss.record(),
        ],
        "visual_model_residual_used": False,
        "heldout_result_used": False,
    }
    if same_frame_stereo:
        return PairDecision(True, "same_frame_stereo", common)
    if frame_gap <= policy.nearby_frame_gap or time_gap_s <= policy.nearby_time_s:
        return PairDecision(True, "adjacent_or_nearby_temporal", common)

    first_std = candidate.first_gnss.trusted_std_m()
    second_std = candidate.second_gnss.trusted_std_m()
    if first_std is None or second_std is None:
        return PairDecision(False, "nonlocal_gnss_untrusted", common)
    if max(first_std, second_std) > policy.max_endpoint_position_std_m:
        return PairDecision(False, "nonlocal_gnss_too_uncertain", common)

    first_position = np.asarray(candidate.first_gnss.position_m, dtype=np.float64)
    second_position = np.asarray(candidate.second_gnss.position_m, dtype=np.float64)
    first_covariance = np.asarray(
        candidate.first_gnss.covariance_m2, dtype=np.float64
    )
    second_covariance = np.asarray(
        candidate.second_gnss.covariance_m2, dtype=np.float64
    )
    displacement = float(np.linalg.norm(second_position - first_position))
    combined = 0.5 * (
        first_covariance
        + second_covariance
        + (first_covariance + second_covariance).T
    )
    combined_sigma = float(
        math.sqrt(max(float(np.linalg.eigvalsh(combined)[-1]), 0.0))
    )
    covariance_allowance = policy.covariance_sigma_multiplier * combined_sigma
    allowed = min(
        policy.revisit_physical_cap_m,
        policy.revisit_distance_m + covariance_allowance,
    )
    evidence = {
        **common,
        "raw_gnss_displacement_m": displacement,
        "combined_position_std_m": combined_sigma,
        "covariance_allowance_m": covariance_allowance,
        "allowed_displacement_m": allowed,
        "physical_cap_applied": (
            allowed == policy.revisit_physical_cap_m
        ),
    }
    if displacement > allowed:
        return PairDecision(False, "nonlocal_gnss_incompatible", evidence)
    reason = (
        "strong_locally_consistent_track"
        if (candidate.verified_matches or 0) >= policy.strong_verified_matches
        else "gnss_consistent_revisit"
    )
    return PairDecision(True, reason, evidence)


__all__ = [
    "GeodeticPairPolicy",
    "PairCandidate",
    "PairDecision",
    "RawGnssEndpoint",
    "decide_pair",
]
