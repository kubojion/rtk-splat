"""Deterministic timestamp association for dataset adapters.

Timestamps are signed integer nanoseconds throughout.  Keeping this boundary
integer-valued avoids losing sub-microsecond information in large Unix epoch
timestamps.  The caller chooses the tolerance; this module makes no frame-rate
or sensor-rate assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


class TimestampMatchError(ValueError):
    """Raised when a timestamp stream or matching option is invalid."""


@dataclass(frozen=True)
class TimestampMatches:
    """Sparse association from a reference stream to a sampled stream.

    ``residual_ns`` is signed and follows ``sample - reference``.  Arrays are
    ordered by ``reference_indices``.  A reference omitted from the match is
    listed in ``unmatched_reference_indices``.
    """

    reference_count: int
    sample_count: int
    reference_indices: np.ndarray
    sample_indices: np.ndarray
    residual_ns: np.ndarray
    unmatched_reference_indices: np.ndarray
    tolerance_ns: int
    method: str

    @property
    def match_count(self) -> int:
        return int(self.reference_indices.size)

    @property
    def match_fraction(self) -> float:
        if self.reference_count == 0:
            return 1.0
        return self.match_count / self.reference_count

    def residual_summary(self) -> dict[str, int | float | None]:
        """Return JSON-friendly signed/absolute residual statistics."""
        if self.residual_ns.size == 0:
            return {
                "count": 0,
                "median_abs_ns": None,
                "max_abs_ns": None,
                "mean_ns": None,
            }
        residual = self.residual_ns
        return {
            "count": self.match_count,
            "median_abs_ns": float(np.median(np.abs(residual))),
            "max_abs_ns": int(np.max(np.abs(residual))),
            "mean_ns": float(np.mean(residual)),
        }


def _timestamp_array(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise TimestampMatchError(f"{name} must be a one-dimensional stream")
    if raw.size == 0:
        return np.empty(0, dtype=np.int64)
    if raw.dtype.kind not in "iu":
        raise TimestampMatchError(
            f"{name} must contain integer nanoseconds, got {raw.dtype}"
        )
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise TimestampMatchError(f"{name} contains values outside int64")
    timestamps = raw.astype(np.int64, copy=False)
    if np.any(timestamps[1:] <= timestamps[:-1]):
        raise TimestampMatchError(f"{name} must be strictly increasing")
    return timestamps


def _tolerance(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TimestampMatchError("tolerance_ns must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise TimestampMatchError("tolerance_ns must be non-negative")
    return result


def _validate_delta_range(
    reference_ns: np.ndarray, sample_ns: np.ndarray
) -> None:
    if reference_ns.size == 0 or sample_ns.size == 0:
        return
    low = min(int(reference_ns[0]), int(sample_ns[0]))
    high = max(int(reference_ns[-1]), int(sample_ns[-1]))
    if high - low > np.iinfo(np.int64).max:
        raise TimestampMatchError("timestamp span is too large for int64 residuals")


def _result(
    *,
    reference_count: int,
    sample_count: int,
    reference_indices: list[int] | np.ndarray,
    sample_indices: list[int] | np.ndarray,
    residual_ns: list[int] | np.ndarray,
    tolerance_ns: int,
    method: str,
) -> TimestampMatches:
    references = np.asarray(reference_indices, dtype=np.int64)
    samples = np.asarray(sample_indices, dtype=np.int64)
    residuals = np.asarray(residual_ns, dtype=np.int64)
    matched = np.zeros(reference_count, dtype=bool)
    matched[references] = True
    return TimestampMatches(
        reference_count=reference_count,
        sample_count=sample_count,
        reference_indices=references,
        sample_indices=samples,
        residual_ns=residuals,
        unmatched_reference_indices=np.flatnonzero(~matched).astype(
            np.int64, copy=False
        ),
        tolerance_ns=tolerance_ns,
        method=method,
    )


def nearest_matches(
    reference_ns: Sequence[int] | np.ndarray,
    sample_ns: Sequence[int] | np.ndarray,
    *,
    tolerance_ns: int,
) -> TimestampMatches:
    """Associate each reference with its nearest sample within tolerance.

    Samples may be reused, which is appropriate when associating a fast frame
    stream with slower GNSS, heading, or IMU observations.  Equal-distance
    ties select the earlier sample.  For sorted streams the selected indices
    are non-decreasing.
    """
    references = _timestamp_array(reference_ns, "reference_ns")
    samples = _timestamp_array(sample_ns, "sample_ns")
    tolerance = _tolerance(tolerance_ns)
    _validate_delta_range(references, samples)

    if references.size == 0 or samples.size == 0:
        return _result(
            reference_count=references.size,
            sample_count=samples.size,
            reference_indices=[],
            sample_indices=[],
            residual_ns=[],
            tolerance_ns=tolerance,
            method="nearest",
        )

    insertion = np.searchsorted(samples, references, side="left")
    before = np.clip(insertion - 1, 0, samples.size - 1)
    after = np.clip(insertion, 0, samples.size - 1)
    before_delta = np.abs(samples[before] - references)
    after_delta = np.abs(samples[after] - references)
    selected = np.where(before_delta <= after_delta, before, after)
    residual = samples[selected] - references
    accepted = np.abs(residual) <= tolerance
    reference_indices = np.flatnonzero(accepted)
    return _result(
        reference_count=references.size,
        sample_count=samples.size,
        reference_indices=reference_indices,
        sample_indices=selected[accepted],
        residual_ns=residual[accepted],
        tolerance_ns=tolerance,
        method="nearest",
    )


def monotonic_matches(
    reference_ns: Sequence[int] | np.ndarray,
    sample_ns: Sequence[int] | np.ndarray,
    *,
    tolerance_ns: int,
) -> TimestampMatches:
    """Create a chronological one-to-one association within tolerance.

    The earliest feasible reference/sample pair is emitted at each step.  This
    policy is deterministic and maximizes cardinality for ordered streams.
    Unlike :func:`nearest_matches`, a sample is never reused.  It is intended
    for triggered stereo streams and similar one-message-per-frame sensors.
    """
    references = _timestamp_array(reference_ns, "reference_ns")
    samples = _timestamp_array(sample_ns, "sample_ns")
    tolerance = _tolerance(tolerance_ns)
    _validate_delta_range(references, samples)

    reference_indices: list[int] = []
    sample_indices: list[int] = []
    residuals: list[int] = []
    reference_i = 0
    sample_i = 0
    while reference_i < references.size and sample_i < samples.size:
        delta = int(samples[sample_i]) - int(references[reference_i])
        if delta < -tolerance:
            sample_i += 1
        elif delta > tolerance:
            reference_i += 1
        else:
            reference_indices.append(reference_i)
            sample_indices.append(sample_i)
            residuals.append(delta)
            reference_i += 1
            sample_i += 1

    return _result(
        reference_count=references.size,
        sample_count=samples.size,
        reference_indices=reference_indices,
        sample_indices=sample_indices,
        residual_ns=residuals,
        tolerance_ns=tolerance,
        method="monotonic",
    )


def associate_timestamps(
    reference_ns: Sequence[int] | np.ndarray,
    sample_ns: Sequence[int] | np.ndarray,
    *,
    tolerance_ns: int,
    method: Literal["nearest", "monotonic"] = "nearest",
) -> TimestampMatches:
    """Dispatch to a named timestamp association policy."""
    if method == "nearest":
        return nearest_matches(
            reference_ns, sample_ns, tolerance_ns=tolerance_ns
        )
    if method == "monotonic":
        return monotonic_matches(
            reference_ns, sample_ns, tolerance_ns=tolerance_ns
        )
    raise TimestampMatchError(
        f"unknown timestamp association method {method!r}; "
        "expected 'nearest' or 'monotonic'"
    )
