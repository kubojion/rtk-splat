"""Pure temporal screening for raw GNSS position observations.

The filter in this module deliberately has no access to images, reconstructed
poses, visual residuals, optimizer outputs, or held-out evaluation results.
It detects a conservative class of receiver failures: a discontinuous
position jump followed by an opposing discontinuity that returns to the
surrounding trajectory within a bounded time.  Raw observations are never
changed; the returned mask only controls their eligibility downstream.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GeodeticGnssTemporalPolicy:
    """Physical and statistical bounds for paired GNSS discontinuities."""

    support_epochs: int = 5
    minimum_boundary_jump_m: float = 0.075
    covariance_sigma_multiplier: float = 3.0
    maximum_acceleration_m_s2: float = 2.0
    maximum_raw_interval_s: float = 0.5
    maximum_excursion_duration_s: float = 60.0
    minimum_return_opposition_cosine: float = 0.90
    maximum_return_closure_fraction: float = 0.55
    minimum_return_closure_m: float = 0.03
    maximum_rejected_raw_fraction: float = 0.05
    maximum_unpaired_boundaries: int = 0
    minimum_carrier_status_for_jump_detection: int = 2

    def __post_init__(self) -> None:
        if (
            isinstance(self.support_epochs, bool)
            or not isinstance(self.support_epochs, int)
            or self.support_epochs < 2
        ):
            raise ValueError("support_epochs must be an integer >= 2")
        if (
            isinstance(self.maximum_unpaired_boundaries, bool)
            or not isinstance(self.maximum_unpaired_boundaries, int)
            or self.maximum_unpaired_boundaries < 0
        ):
            raise ValueError(
                "maximum_unpaired_boundaries must be a non-negative integer"
            )
        if (
            isinstance(self.minimum_carrier_status_for_jump_detection, bool)
            or not isinstance(
                self.minimum_carrier_status_for_jump_detection, int
            )
            or self.minimum_carrier_status_for_jump_detection < 0
        ):
            raise ValueError(
                "minimum_carrier_status_for_jump_detection must be a "
                "non-negative integer"
            )
        for name in (
            "minimum_boundary_jump_m",
            "covariance_sigma_multiplier",
            "maximum_acceleration_m_s2",
            "maximum_raw_interval_s",
            "maximum_excursion_duration_s",
            "minimum_return_opposition_cosine",
            "maximum_return_closure_fraction",
            "minimum_return_closure_m",
            "maximum_rejected_raw_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_return_opposition_cosine > 1.0:
            raise ValueError(
                "minimum_return_opposition_cosine cannot exceed one"
            )
        if self.maximum_return_closure_fraction > 1.0:
            raise ValueError(
                "maximum_return_closure_fraction cannot exceed one"
            )
        if self.maximum_rejected_raw_fraction > 1.0:
            raise ValueError(
                "maximum_rejected_raw_fraction cannot exceed one"
            )


def _maximum_position_std_m(covariance: np.ndarray) -> float | None:
    symmetric = 0.5 * (covariance + covariance.T)
    if not np.isfinite(symmetric).all():
        return None
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if eigenvalues[0] < -1.0e-10:
        return None
    return float(math.sqrt(max(float(eigenvalues[-1]), 0.0)))


def _validated_inputs(
    timestamps_ns: Any,
    positions_m: Any,
    covariance_m2: Any,
    fix_status: Any,
    carrier_status: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    timestamps = np.asarray(timestamps_ns)
    positions = np.asarray(positions_m, dtype=np.float64)
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    fix = np.asarray(fix_status)
    carrier = np.asarray(carrier_status)
    if (
        timestamps.ndim != 1
        or len(timestamps) < 3
        or not np.issubdtype(timestamps.dtype, np.integer)
        or positions.shape != (len(timestamps), 3)
        or covariance.shape != (len(timestamps), 3, 3)
        or fix.shape != (len(timestamps),)
        or carrier.shape != (len(timestamps),)
        or not np.issubdtype(fix.dtype, np.integer)
        or not np.issubdtype(carrier.dtype, np.integer)
    ):
        raise ValueError("raw GNSS arrays have incompatible shapes or dtypes")
    timestamps = timestamps.astype(np.int64, copy=False)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("raw GNSS timestamps must be strictly increasing")
    symmetric = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))
    finite = np.isfinite(positions).all(axis=1) & np.isfinite(symmetric).all(
        axis=(1, 2)
    )
    covariance_valid = np.zeros(len(timestamps), dtype=bool)
    finite_indices = np.flatnonzero(finite)
    if finite_indices.size:
        eigenvalues = np.linalg.eigvalsh(symmetric[finite_indices])
        covariance_valid[finite_indices] = eigenvalues[:, 0] >= -1.0e-10
    valid = finite & covariance_valid & (fix.astype(np.int64) >= 0)
    return timestamps, positions, symmetric, fix, carrier, valid


def _array_binding(value: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "sha256": digest.hexdigest(),
    }


def raw_gnss_temporal_filter(
    timestamps_ns: Any,
    positions_m: Any,
    covariance_m2: Any,
    fix_status: Any,
    carrier_status: Any,
    *,
    policy: GeodeticGnssTemporalPolicy = GeodeticGnssTemporalPolicy(),
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Return an eligibility mask and a complete raw-only decision audit.

    A candidate boundary is measured against robust velocities on both sides.
    It must exceed an absolute jump floor, a covariance-derived allowance, and
    an acceleration-derived allowance.  Rejection occurs only when the next
    candidate boundary is timely, strongly opposing, and closes the first
    displacement.  Unpaired discontinuities are surfaced fail-closed through
    the audit rather than silently choosing one side of an absolute shift.
    """

    (
        timestamps,
        positions,
        covariance,
        fix,
        carrier,
        valid,
    ) = _validated_inputs(
        timestamps_ns,
        positions_m,
        covariance_m2,
        fix_status,
        carrier_status,
    )
    n = len(timestamps)
    retained = valid.copy()
    reasons = np.full(n, "raw_gnss_temporally_consistent", dtype="<U48")
    reasons[~valid] = "raw_gnss_stream_invalid"
    time_s = timestamps.astype(np.float64) * 1.0e-9
    intervals_s = np.diff(time_s)
    velocities = np.diff(positions, axis=0) / intervals_s[:, None]
    support = policy.support_epochs
    boundaries: list[dict[str, Any]] = []
    first_boundary = support + 1
    last_boundary_exclusive = n - support
    for index in range(first_boundary, last_boundary_exclusive):
        before_edges = slice(index - support - 1, index - 1)
        after_edges = slice(index, index + support)
        support_positions = slice(index - support - 1, index + support + 1)
        if (
            not valid[support_positions].all()
            or max(int(carrier[index - 1]), int(carrier[index]))
            < policy.minimum_carrier_status_for_jump_detection
            or np.any(
                intervals_s[before_edges] > policy.maximum_raw_interval_s
            )
            or np.any(intervals_s[after_edges] > policy.maximum_raw_interval_s)
            or intervals_s[index - 1] > policy.maximum_raw_interval_s
        ):
            continue
        before_velocity = np.median(velocities[before_edges], axis=0)
        after_velocity = np.median(velocities[after_edges], axis=0)
        interval_s = float(intervals_s[index - 1])
        expected_displacement = (
            0.5 * (before_velocity + after_velocity) * interval_s
        )
        observed_displacement = positions[index] - positions[index - 1]
        residual = observed_displacement - expected_displacement
        residual_norm = float(np.linalg.norm(residual))
        combined_covariance = covariance[index - 1] + covariance[index]
        combined_std = _maximum_position_std_m(combined_covariance)
        if combined_std is None:
            continue
        covariance_allowance = (
            policy.covariance_sigma_multiplier * combined_std
        )
        acceleration_allowance = (
            0.5 * policy.maximum_acceleration_m_s2 * interval_s**2
        )
        threshold = max(
            policy.minimum_boundary_jump_m,
            covariance_allowance,
            acceleration_allowance,
        )
        if residual_norm <= threshold:
            continue
        boundaries.append(
            {
                "boundary_id": len(boundaries),
                "raw_index_before": index - 1,
                "raw_index_after": index,
                "timestamps_ns": [
                    int(timestamps[index - 1]),
                    int(timestamps[index]),
                ],
                "interval_s": interval_s,
                "observed_displacement_m": observed_displacement.tolist(),
                "expected_displacement_m": expected_displacement.tolist(),
                "residual_displacement_m": residual.tolist(),
                "residual_norm_m": residual_norm,
                "combined_position_std_m": combined_std,
                "covariance_allowance_m": covariance_allowance,
                "acceleration_allowance_m": acceleration_allowance,
                "threshold_m": threshold,
                "fix_status": [
                    int(fix[index - 1]),
                    int(fix[index]),
                ],
                "carrier_status": [
                    int(carrier[index - 1]),
                    int(carrier[index]),
                ],
            }
        )

    excursions: list[dict[str, Any]] = []
    unpaired: list[dict[str, Any]] = []
    boundary_index = 0
    while boundary_index < len(boundaries):
        first = boundaries[boundary_index]
        if boundary_index + 1 >= len(boundaries):
            unpaired.append(first)
            break
        second = boundaries[boundary_index + 1]
        first_vector = np.asarray(
            first["residual_displacement_m"], dtype=np.float64
        )
        second_vector = np.asarray(
            second["residual_displacement_m"], dtype=np.float64
        )
        first_norm = float(first["residual_norm_m"])
        second_norm = float(second["residual_norm_m"])
        opposition_cosine = float(
            np.dot(first_vector, second_vector) / (first_norm * second_norm)
        )
        closure = float(np.linalg.norm(first_vector + second_vector))
        closure_limit = max(
            policy.minimum_return_closure_m,
            policy.maximum_return_closure_fraction
            * max(first_norm, second_norm),
        )
        start = int(first["raw_index_after"])
        stop = int(second["raw_index_after"])
        duration_s = float(
            (timestamps[stop] - timestamps[start]) * 1.0e-9
        )
        paired = bool(
            stop > start
            and duration_s <= policy.maximum_excursion_duration_s
            and opposition_cosine
            <= -policy.minimum_return_opposition_cosine
            and closure <= closure_limit
            and valid[start : stop + 1].all()
        )
        if not paired:
            unpaired.append(first)
            boundary_index += 1
            continue
        excursion_id = len(excursions)
        retained[start:stop] = False
        reasons[start:stop] = "raw_gnss_paired_discontinuity_excursion"
        excursions.append(
            {
                "excursion_id": excursion_id,
                "entry_boundary_id": int(first["boundary_id"]),
                "exit_boundary_id": int(second["boundary_id"]),
                "rejected_raw_index_start": start,
                "rejected_raw_index_stop_exclusive": stop,
                "rejected_raw_epoch_count": stop - start,
                "timestamps_ns": [
                    int(timestamps[start]),
                    int(timestamps[stop]),
                ],
                "duration_s": duration_s,
                "opposition_cosine": opposition_cosine,
                "return_closure_m": closure,
                "return_closure_limit_m": closure_limit,
            }
        )
        boundary_index += 2

    rejected_temporal = valid & ~retained
    rejected_fraction = float(rejected_temporal.sum() / max(int(valid.sum()), 1))
    checks = {
        "unpaired_discontinuity_boundaries": {
            "value": len(unpaired),
            "maximum": policy.maximum_unpaired_boundaries,
            "passed": len(unpaired) <= policy.maximum_unpaired_boundaries,
        },
        "rejected_raw_gnss_fraction": {
            "value": rejected_fraction,
            "maximum": policy.maximum_rejected_raw_fraction,
            "passed": rejected_fraction
            <= policy.maximum_rejected_raw_fraction,
        },
    }
    audit = {
        "schema_version": 1,
        "method": "raw_gnss_paired_discontinuity_filter_v1",
        "policy": asdict(policy),
        "decision_inputs": [
            "raw_timestamp_ns",
            "raw_enu_m",
            "raw_effective_covariance_enu_m2",
            "raw_fix_status",
            "raw_carrier_status",
        ],
        "decision_inputs_exclude": [
            "images",
            "visual_poses",
            "finished_visual_model_residual",
            "optimizer_output",
            "heldout_evaluation_result",
        ],
        "raw_input_bindings": {
            "raw_timestamp_ns": _array_binding(timestamps),
            "raw_enu_m": _array_binding(positions),
            "raw_effective_covariance_enu_m2": _array_binding(covariance),
            "raw_fix_status": _array_binding(fix),
            "raw_carrier_status": _array_binding(carrier),
        },
        "raw_epoch_count": n,
        "raw_valid_count": int(valid.sum()),
        "candidate_boundary_count": len(boundaries),
        "paired_excursion_count": len(excursions),
        "rejected_raw_epoch_count": int(rejected_temporal.sum()),
        "rejected_raw_fraction": rejected_fraction,
        "retained_raw_epoch_count": int(retained.sum()),
        "unpaired_boundary_ids": [
            int(item["boundary_id"]) for item in unpaired
        ],
        "boundaries": boundaries,
        "excursions": excursions,
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }
    return retained, tuple(str(value) for value in reasons), audit


__all__ = [
    "GeodeticGnssTemporalPolicy",
    "raw_gnss_temporal_filter",
]
