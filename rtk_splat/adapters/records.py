"""ROS-free observation records shared by dataset adapters.

Concrete adapters own message decoding and bag traversal.  This module owns
only the plain, NumPy-backed records passed between those adapters and the
contract publisher or optional diagnostics.  Keeping these records here
prevents one robot adapter from depending on another robot's implementation
and prevents ingestion code from importing the diagnostics package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
        if self.heading_quality_kind not in {"carrier", "course", "trajectory"}:
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


@dataclass
class FrameRecord:
    """One selected stereo frame with exact stamps and image payloads."""

    t: float
    t_right: float
    left_jpeg: bytes | Path
    right_jpeg: bytes | Path
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
