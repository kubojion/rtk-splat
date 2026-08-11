"""ROS-free observation records shared by dataset adapters.

Concrete adapters own message decoding and bag traversal.  This module owns
only the plain, NumPy-backed records passed between those adapters and the
contract publisher or optional diagnostics.  Keeping these records here
prevents one robot adapter from depending on another robot's implementation
and prevents ingestion code from importing the diagnostics package.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


RELPOS_FLAG_NAMES = (
    "gnss_fix_ok",
    "diff_soln",
    "rel_pos_valid",
    "is_moving",
    "ref_pos_miss",
    "ref_obs_miss",
    "rel_pos_heading_valid",
    "rel_pos_normalized",
)


def _typed_vector(value: Any, *, name: str, dtype: np.dtype) -> np.ndarray:
    """Return a canonical numeric vector without accepting lossy casts."""
    array = np.asarray(value)
    expected = np.dtype(dtype)
    if expected.kind in "iu" and array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    if expected.kind == "b" and array.dtype.kind != "b":
        raise ValueError(f"{name} must contain booleans")
    if expected.kind == "f" and array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain numeric values")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    return np.asarray(array, dtype=expected)


def _typed_matrix_series(
    value: Any,
    *,
    name: str,
    trailing_shape: tuple[int, ...],
) -> np.ndarray:
    """Return canonical floating-point matrix evidence with a sample axis."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain numeric values")
    if array.ndim != len(trailing_shape) + 1 or array.shape[1:] != trailing_shape:
        suffix = ", ".join(str(item) for item in trailing_shape)
        raise ValueError(f"{name} must have shape (N, {suffix})")
    return np.asarray(array, dtype=np.float64)


@dataclass(frozen=True)
class SecondaryGnssEvidence:
    """Complete raw/effective fix stream for a secondary GNSS antenna.

    This is deliberately one record rather than a collection of optional
    ``RtkTrack`` attributes: a partial secondary stream must never be
    publishable as if it were complete evidence.
    """

    header_ns: np.ndarray
    log_ns: np.ndarray
    enu_m: np.ndarray
    geodetic_deg_m: np.ndarray
    raw_covariance_enu_m2: np.ndarray
    effective_covariance_enu_m2: np.ndarray
    fix_status: np.ndarray
    carrier_status: np.ndarray
    covariance_type: np.ndarray
    service: np.ndarray

    def __post_init__(self) -> None:
        header = _typed_vector(
            self.header_ns, name="secondary GNSS header_ns", dtype=np.int64
        )
        log = _typed_vector(
            self.log_ns, name="secondary GNSS log_ns", dtype=np.int64
        )
        enu = _typed_matrix_series(
            self.enu_m, name="secondary GNSS enu_m", trailing_shape=(3,)
        )
        geodetic = _typed_matrix_series(
            self.geodetic_deg_m,
            name="secondary GNSS geodetic_deg_m",
            trailing_shape=(3,),
        )
        raw_covariance = _typed_matrix_series(
            self.raw_covariance_enu_m2,
            name="secondary GNSS raw_covariance_enu_m2",
            trailing_shape=(3, 3),
        )
        effective_covariance = _typed_matrix_series(
            self.effective_covariance_enu_m2,
            name="secondary GNSS effective_covariance_enu_m2",
            trailing_shape=(3, 3),
        )
        fix_status = _typed_vector(
            self.fix_status, name="secondary GNSS fix_status", dtype=np.int16
        )
        carrier_status = _typed_vector(
            self.carrier_status,
            name="secondary GNSS carrier_status",
            dtype=np.int8,
        )
        covariance_type = _typed_vector(
            self.covariance_type,
            name="secondary GNSS covariance_type",
            dtype=np.int16,
        )
        service = _typed_vector(
            self.service, name="secondary GNSS service", dtype=np.int16
        )
        count = len(header)
        if count == 0:
            raise ValueError("secondary GNSS evidence must not be empty")
        fields = {
            "log_ns": log,
            "enu_m": enu,
            "geodetic_deg_m": geodetic,
            "raw_covariance_enu_m2": raw_covariance,
            "effective_covariance_enu_m2": effective_covariance,
            "fix_status": fix_status,
            "carrier_status": carrier_status,
            "covariance_type": covariance_type,
            "service": service,
        }
        bad = [name for name, value in fields.items() if len(value) != count]
        if bad:
            raise ValueError(
                "secondary GNSS evidence arrays must have equal sample count: "
                + ", ".join(sorted(bad))
            )
        if np.any(np.diff(header) <= 0):
            raise ValueError("secondary GNSS header_ns must be strictly increasing")
        if np.any(np.diff(log) <= 0):
            raise ValueError("secondary GNSS log_ns must be strictly increasing")
        for name, value in (
            ("enu_m", enu),
            ("geodetic_deg_m", geodetic),
            ("effective_covariance_enu_m2", effective_covariance),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"secondary GNSS {name} must be finite")
        for name, covariance in (
            ("raw_covariance_enu_m2", raw_covariance),
            ("effective_covariance_enu_m2", effective_covariance),
        ):
            if not np.isfinite(covariance).all():
                raise ValueError(f"secondary GNSS {name} must be finite")
            if not np.allclose(covariance, np.swapaxes(covariance, 1, 2), atol=1e-9):
                raise ValueError(f"secondary GNSS {name} must be symmetric")
            if np.any(np.diagonal(covariance, axis1=1, axis2=2) < 0):
                raise ValueError(
                    f"secondary GNSS {name} must have non-negative diagonal"
                )
        canonical = {
            "header_ns": header,
            "log_ns": log,
            "enu_m": enu,
            "geodetic_deg_m": geodetic,
            "raw_covariance_enu_m2": raw_covariance,
            "effective_covariance_enu_m2": effective_covariance,
            "fix_status": fix_status,
            "carrier_status": carrier_status,
            "covariance_type": covariance_type,
            "service": service,
        }
        for name, value in canonical.items():
            object.__setattr__(self, name, value)


@dataclass
class RtkTrack:
    """RTK fixes and optional heading evidence over one time window."""

    fix_t: np.ndarray
    fix_lat: np.ndarray
    fix_lon: np.ndarray
    fix_alt: np.ndarray
    fix_status: np.ndarray
    fix_cov_max: np.ndarray
    relpos_t: np.ndarray
    relpos_yaw: np.ndarray
    relpos_carr: np.ndarray
    fix_header_ns: np.ndarray | None = None
    fix_log_ns: np.ndarray | None = None
    fix_covariance_enu_m2: np.ndarray | None = None
    fix_covariance_type: np.ndarray | None = None
    fix_carrier_status: np.ndarray | None = None
    relpos_header_ns: np.ndarray | None = None
    relpos_log_ns: np.ndarray | None = None
    relpos_ned_m: np.ndarray | None = None
    relpos_acc_heading_rad: np.ndarray | None = None
    relpos_flags: np.ndarray | None = None
    heading_valid: np.ndarray | None = None
    heading_quality_kind: str = "carrier"
    pvt_header_ns: np.ndarray | None = None
    pvt_log_ns: np.ndarray | None = None
    pvt_carrier_status: np.ndarray | None = None
    raw_message_covariance_enu_m2: np.ndarray | None = None
    effective_covariance_policy: Mapping[str, Any] | None = None
    fix_service: np.ndarray | None = None
    fix_position_valid: np.ndarray | None = None
    fix_position_quality: np.ndarray | None = None
    secondary_gnss: SecondaryGnssEvidence | None = None
    fix_timestamps_exact: bool = field(init=False)
    relpos_timestamps_exact: bool = field(init=False)

    def __post_init__(self) -> None:
        self.fix_timestamps_exact = (
            self.fix_header_ns is not None and self.fix_log_ns is not None
        )
        self.relpos_timestamps_exact = (
            self.relpos_header_ns is not None
            and self.relpos_log_ns is not None
        )
        n_fix = len(self.fix_t)
        n_relpos = len(self.relpos_t)
        if self.fix_header_ns is None:
            self.fix_header_ns = np.rint(
                np.asarray(self.fix_t, dtype=np.float64) * 1e9
            ).astype(np.int64)
        if self.fix_log_ns is None:
            self.fix_log_ns = np.full(n_fix, -1, dtype=np.int64)
        if self.fix_covariance_enu_m2 is None:
            self.fix_covariance_enu_m2 = np.full((n_fix, 3, 3), np.nan)
        if self.fix_covariance_type is None:
            self.fix_covariance_type = np.full(n_fix, -1, dtype=np.int16)
        if self.fix_carrier_status is None:
            self.fix_carrier_status = np.full(n_fix, -1, dtype=np.int8)
        if self.raw_message_covariance_enu_m2 is not None:
            raw_covariance = _typed_matrix_series(
                self.raw_message_covariance_enu_m2,
                name="raw_message_covariance_enu_m2",
                trailing_shape=(3, 3),
            )
            if raw_covariance.shape[0] != n_fix:
                raise ValueError(
                    "raw_message_covariance_enu_m2 must have one matrix per fix"
                )
            self.raw_message_covariance_enu_m2 = raw_covariance
        if self.effective_covariance_policy is not None and not isinstance(
            self.effective_covariance_policy, Mapping
        ):
            raise ValueError("effective_covariance_policy must be a mapping")
        if self.fix_service is not None:
            self.fix_service = _typed_vector(
                self.fix_service, name="fix_service", dtype=np.int16
            )
            if self.fix_service.shape != (n_fix,):
                raise ValueError("fix_service must have one entry per fix")
        if (self.fix_position_valid is None) != (self.fix_position_quality is None):
            raise ValueError(
                "fix_position_valid and fix_position_quality must be supplied together"
            )
        if self.fix_position_valid is not None:
            self.fix_position_valid = _typed_vector(
                self.fix_position_valid, name="fix_position_valid", dtype=np.bool_
            )
            quality = np.asarray(self.fix_position_quality)
            if quality.dtype.kind not in "US":
                raise ValueError("fix_position_quality must contain strings")
            self.fix_position_quality = np.asarray(quality, dtype=np.str_)
            valid_shape = self.fix_position_valid.shape == (n_fix,)
            quality_shape = self.fix_position_quality.shape == (n_fix,)
            if not valid_shape or not quality_shape:
                raise ValueError(
                    "fix position quality fields must have one entry per fix"
                )
        if self.secondary_gnss is not None and not isinstance(
            self.secondary_gnss, SecondaryGnssEvidence
        ):
            raise ValueError("secondary_gnss must be SecondaryGnssEvidence")
        if self.relpos_header_ns is None:
            self.relpos_header_ns = np.rint(
                np.asarray(self.relpos_t, dtype=np.float64) * 1e9
            ).astype(np.int64)
        if self.relpos_log_ns is None:
            self.relpos_log_ns = np.full(n_relpos, -1, dtype=np.int64)
        if self.relpos_ned_m is None:
            self.relpos_ned_m = np.full((n_relpos, 3), np.nan)
        if self.relpos_acc_heading_rad is None:
            self.relpos_acc_heading_rad = np.full(n_relpos, np.nan)
        if self.relpos_flags is None:
            self.relpos_flags = np.zeros(
                (n_relpos, len(RELPOS_FLAG_NAMES)), dtype=bool
            )
        if self.heading_valid is None:
            self.heading_valid = np.asarray(self.relpos_carr) >= 2
        else:
            self.heading_valid = np.asarray(self.heading_valid, dtype=bool)
        if self.heading_valid.shape != (n_relpos,):
            raise ValueError("heading_valid must have one entry per heading sample")
        if self.heading_quality_kind not in {
            "carrier",
            "course",
            "trajectory",
            "dual_position",
        }:
            raise ValueError("unknown heading_quality_kind")
        if self.pvt_header_ns is None:
            self.pvt_header_ns = np.empty(0, dtype=np.int64)
        if self.pvt_log_ns is None:
            self.pvt_log_ns = np.empty(0, dtype=np.int64)
        if self.pvt_carrier_status is None:
            self.pvt_carrier_status = np.empty(0, dtype=np.int8)
        pvt_count = len(self.pvt_header_ns)
        if (
            np.asarray(self.pvt_header_ns).shape != (pvt_count,)
            or np.asarray(self.pvt_log_ns).shape != (pvt_count,)
            or np.asarray(self.pvt_carrier_status).shape != (pvt_count,)
        ):
            raise ValueError("NavPVT evidence arrays must have equal vector shape")

    def apply_effective_covariance(
        self,
        covariance_enu_m2: np.ndarray,
        *,
        policy: Mapping[str, Any],
    ) -> None:
        """Atomically retain receiver covariance and install an external prior."""
        if self.raw_message_covariance_enu_m2 is not None:
            raise ValueError("effective covariance has already been applied")
        if not isinstance(policy, Mapping) or not policy:
            raise ValueError("effective covariance policy must be a non-empty mapping")
        raw = _typed_matrix_series(
            self.fix_covariance_enu_m2,
            name="fix_covariance_enu_m2",
            trailing_shape=(3, 3),
        )
        effective = _typed_matrix_series(
            covariance_enu_m2,
            name="effective fix covariance",
            trailing_shape=(3, 3),
        )
        expected = (len(self.fix_t), 3, 3)
        if raw.shape != expected or effective.shape != expected:
            raise ValueError(
                f"raw and effective fix covariance must have shape {expected}"
            )
        for name, value in (("raw", raw), ("effective", effective)):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} fix covariance must be finite")
            if not np.allclose(value, np.swapaxes(value, 1, 2), atol=1e-9):
                raise ValueError(f"{name} fix covariance must be symmetric")
            if np.any(np.diagonal(value, axis1=1, axis2=2) < 0):
                raise ValueError(
                    f"{name} fix covariance must have non-negative diagonal"
                )
        self.raw_message_covariance_enu_m2 = raw.copy()
        self.fix_covariance_enu_m2 = effective.copy()
        self.fix_cov_max = np.max(
            np.diagonal(effective, axis1=1, axis2=2), axis=1
        )
        self.effective_covariance_policy = dict(policy)


@dataclass(frozen=True)
class StagedPayload:
    """One disposable file owned by an adapter publication operation.

    Unlike a plain :class:`~pathlib.Path`, this path may be moved into the
    immutable segment rather than copied.  Adapters must only use it for files
    in their own temporary staging directory; callers retaining source files
    should continue to pass ``Path``.
    """

    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass
class FrameRecord:
    """One selected stereo frame with exact stamps and image payloads."""

    t: float
    t_right: float
    left_jpeg: bytes | Path | StagedPayload
    right_jpeg: bytes | Path | StagedPayload
    left_header_ns: int | None = None
    right_header_ns: int | None = None
    header_timestamps_exact: bool = field(init=False)

    def __post_init__(self) -> None:
        self.header_timestamps_exact = (
            self.left_header_ns is not None
            and self.right_header_ns is not None
        )
        if self.left_header_ns is None:
            self.left_header_ns = int(round(self.t * 1e9))
        if self.right_header_ns is None:
            self.right_header_ns = int(round(self.t_right * 1e9))


@dataclass(frozen=True)
class StereoFrameObservation:
    frame_id: int
    left_header_ns: int
    left_log_ns: int
    right_header_ns: int
    right_log_ns: int
    left_sha256: str
    right_sha256: str

    @property
    def stereo_delta_ns(self) -> int:
        return self.right_header_ns - self.left_header_ns


@dataclass(frozen=True)
class NavSatFixObservation:
    header_ns: int
    log_ns: int
    frame_id: str
    latitude_deg: float
    longitude_deg: float
    altitude_ellipsoid_m: float
    status: int
    service: int
    covariance_enu_m2: tuple[float, ...]
    covariance_type: int

    def __post_init__(self) -> None:
        if len(self.covariance_enu_m2) != 9:
            raise ValueError("NavSatFix covariance must contain nine values")

    @property
    def covariance_matrix_enu_m2(self) -> np.ndarray:
        return np.asarray(self.covariance_enu_m2, dtype=float).reshape(3, 3)


@dataclass(frozen=True)
class RelPosObservation:
    header_ns: int
    log_ns: int
    frame_id: str
    version: int
    ref_station_id: int
    itow_ms: int
    rel_pos_ned_cm: tuple[int, int, int]
    rel_pos_hp_ned_0p1mm: tuple[int, int, int]
    rel_pos_ned_m: tuple[float, float, float]
    rel_pos_enu_m: tuple[float, float, float]
    rel_pos_length_cm: int
    rel_pos_hp_length_0p1mm: int
    rel_pos_length_m: float
    rel_pos_heading_1e5_deg: int
    rel_pos_heading_deg: float
    accuracy_ned_0p1mm: tuple[int, int, int]
    accuracy_ned_m: tuple[float, float, float]
    accuracy_length_0p1mm: int
    accuracy_length_m: float
    accuracy_heading_1e5_deg: int
    accuracy_heading_deg: float
    carrier_solution: int
    gnss_fix_ok: bool
    diff_soln: bool
    rel_pos_valid: bool
    is_moving: bool
    ref_pos_miss: bool
    ref_obs_miss: bool
    rel_pos_heading_valid: bool
    rel_pos_normalized: bool


@dataclass(frozen=True)
class NavPvtStatusObservation:
    header_ns: int
    log_ns: int
    frame_id: str
    itow_ms: int
    gps_fix_type: int
    gnss_fix_ok: bool
    diff_soln: bool
    carrier_solution: int
    invalid_llh: bool
    num_sv: int
    horizontal_accuracy_m: float
    vertical_accuracy_m: float
    position_dop: float
    valid_date: bool
    valid_time: bool
    fully_resolved: bool
    time_accuracy_ns: int
    utc_nano_ns: int


@dataclass(frozen=True)
class CalibrationBagObservations:
    stereo_frames: tuple[StereoFrameObservation, ...]
    fixes: tuple[NavSatFixObservation, ...]
    relpos: tuple[RelPosObservation, ...]
    moving_base_pvt: tuple[NavPvtStatusObservation, ...]
    topic_bags: tuple[tuple[str, str], ...]
    header_window_ns: tuple[int, int]
    rtk_window_ns: tuple[int, int]
    log_window_ns: tuple[int, int]

    def to_npz_payload(self) -> dict[str, np.ndarray]:
        """Flatten all retained evidence into non-pickled NumPy arrays."""
        stereo = self.stereo_frames
        fixes = self.fixes
        relpos = self.relpos
        pvt = self.moving_base_pvt
        return {
            "stereo_frame_id": _field_array(stereo, "frame_id", np.int64),
            "stereo_left_header_ns": _field_array(stereo, "left_header_ns", np.int64),
            "stereo_left_log_ns": _field_array(stereo, "left_log_ns", np.int64),
            "stereo_right_header_ns": _field_array(stereo, "right_header_ns", np.int64),
            "stereo_right_log_ns": _field_array(stereo, "right_log_ns", np.int64),
            "stereo_delta_ns": np.asarray(
                [item.stereo_delta_ns for item in stereo], dtype=np.int64
            ),
            "stereo_left_sha256": _field_array(stereo, "left_sha256", "<U64"),
            "stereo_right_sha256": _field_array(stereo, "right_sha256", "<U64"),
            "fix_header_ns": _field_array(fixes, "header_ns", np.int64),
            "fix_log_ns": _field_array(fixes, "log_ns", np.int64),
            "fix_geodetic": np.asarray(
                [
                    (item.latitude_deg, item.longitude_deg, item.altitude_ellipsoid_m)
                    for item in fixes
                ],
                dtype=np.float64,
            ).reshape(-1, 3),
            "fix_status": _field_array(fixes, "status", np.int16),
            "fix_service": _field_array(fixes, "service", np.uint16),
            "fix_frame_id": _field_array(fixes, "frame_id", "<U256"),
            "fix_covariance_enu_m2": np.asarray(
                [item.covariance_enu_m2 for item in fixes], dtype=np.float64
            ).reshape(-1, 3, 3),
            "fix_covariance_type": _field_array(fixes, "covariance_type", np.uint8),
            "relpos_header_ns": _field_array(relpos, "header_ns", np.int64),
            "relpos_log_ns": _field_array(relpos, "log_ns", np.int64),
            "relpos_frame_id": _field_array(relpos, "frame_id", "<U256"),
            "relpos_version": _field_array(relpos, "version", np.uint8),
            "relpos_ref_station_id": _field_array(relpos, "ref_station_id", np.uint16),
            "relpos_itow_ms": _field_array(relpos, "itow_ms", np.uint32),
            "relpos_ned_cm": _tuple_field_array(relpos, "rel_pos_ned_cm", np.int32, 3),
            "relpos_hp_ned_0p1mm": _tuple_field_array(
                relpos, "rel_pos_hp_ned_0p1mm", np.int8, 3
            ),
            "relpos_ned_m": _tuple_field_array(relpos, "rel_pos_ned_m", np.float64, 3),
            "relpos_enu_m": _tuple_field_array(relpos, "rel_pos_enu_m", np.float64, 3),
            "relpos_length_cm": _field_array(relpos, "rel_pos_length_cm", np.int32),
            "relpos_hp_length_0p1mm": _field_array(
                relpos, "rel_pos_hp_length_0p1mm", np.int8
            ),
            "relpos_accuracy_ned_m": _tuple_field_array(
                relpos, "accuracy_ned_m", np.float64, 3
            ),
            "relpos_accuracy_ned_0p1mm": _tuple_field_array(
                relpos, "accuracy_ned_0p1mm", np.uint32, 3
            ),
            "relpos_length_m": _field_array(relpos, "rel_pos_length_m", np.float64),
            "relpos_accuracy_length_0p1mm": _field_array(
                relpos, "accuracy_length_0p1mm", np.uint32
            ),
            "relpos_accuracy_length_m": _field_array(
                relpos, "accuracy_length_m", np.float64
            ),
            "relpos_heading_1e5_deg": _field_array(
                relpos, "rel_pos_heading_1e5_deg", np.int32
            ),
            "relpos_heading_deg": _field_array(
                relpos, "rel_pos_heading_deg", np.float64
            ),
            "relpos_accuracy_heading_1e5_deg": _field_array(
                relpos, "accuracy_heading_1e5_deg", np.uint32
            ),
            "relpos_accuracy_heading_deg": _field_array(
                relpos, "accuracy_heading_deg", np.float64
            ),
            "relpos_carrier_solution": _field_array(
                relpos, "carrier_solution", np.uint8
            ),
            "relpos_flags": np.asarray(
                [
                    (
                        item.gnss_fix_ok,
                        item.diff_soln,
                        item.rel_pos_valid,
                        item.is_moving,
                        item.ref_pos_miss,
                        item.ref_obs_miss,
                        item.rel_pos_heading_valid,
                        item.rel_pos_normalized,
                    )
                    for item in relpos
                ],
                dtype=np.bool_,
            ).reshape(-1, 8),
            "pvt_header_ns": _field_array(pvt, "header_ns", np.int64),
            "pvt_log_ns": _field_array(pvt, "log_ns", np.int64),
            "pvt_frame_id": _field_array(pvt, "frame_id", "<U256"),
            "pvt_itow_ms": _field_array(pvt, "itow_ms", np.uint32),
            "pvt_gps_fix_type": _field_array(pvt, "gps_fix_type", np.uint8),
            "pvt_carrier_solution": _field_array(
                pvt, "carrier_solution", np.uint8
            ),
            "pvt_status_flags": np.asarray(
                [
                    (
                        item.gnss_fix_ok,
                        item.diff_soln,
                        item.invalid_llh,
                        item.valid_date,
                        item.valid_time,
                        item.fully_resolved,
                    )
                    for item in pvt
                ],
                dtype=np.bool_,
            ).reshape(-1, 6),
            "pvt_num_sv": _field_array(pvt, "num_sv", np.uint8),
            "pvt_accuracy_m": np.asarray(
                [
                    (item.horizontal_accuracy_m, item.vertical_accuracy_m)
                    for item in pvt
                ],
                dtype=np.float64,
            ).reshape(-1, 2),
            "pvt_position_dop": _field_array(pvt, "position_dop", np.float64),
            "pvt_time_accuracy_ns": _field_array(
                pvt, "time_accuracy_ns", np.uint32
            ),
            "pvt_utc_nano_ns": _field_array(pvt, "utc_nano_ns", np.int32),
        }


def _field_array(records, name: str, dtype) -> np.ndarray:
    return np.asarray([getattr(record, name) for record in records], dtype=dtype)


def _tuple_field_array(records, name: str, dtype, width: int) -> np.ndarray:
    return np.asarray(
        [getattr(record, name) for record in records], dtype=dtype
    ).reshape(-1, width)
