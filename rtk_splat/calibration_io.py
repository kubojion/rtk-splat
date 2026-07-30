"""Read-only inputs and fail-closed artifacts for RTK--stereo calibration.

This module deliberately does not publish pose artifacts.  It provides:

* exact, integer-nanosecond observations recovered from a bounded rosbag pass;
* the historical stereo-selection policy used by the existing extracted
  segment, with byte-for-byte JPEG verification;
* raw chronological left-camera poses from a COLMAP rig model without reading
  the potentially multi-gigabyte ``images.txt``; and
* atomic, fail-if-exists calibration artifact publication.

Coordinate conventions
----------------------
RELPOS is retained in its native NED convention and also converted to ENU as
``[east, north, -down]``.  COLMAP poses are returned as camera-from-visual-
world matrices.  No RTK alignment or scale is applied here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Iterator, Mapping, Sequence
from urllib.parse import quote

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore


_SAFE_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_NS_PER_S = 1_000_000_000


def validate_calibration_artifact_name(name: str) -> str:
    """Return a safe artifact name or reject traversal/shell-like names."""
    value = str(name)
    if not _SAFE_ARTIFACT_NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"invalid calibration artifact name {value!r}")
    return value


def calibration_artifact_path(segment_dir: Path, name: str) -> Path:
    """Canonical directory for a diagnostic calibration artifact."""
    return (Path(segment_dir) / "calibration_artifacts"
            / validate_calibration_artifact_name(name))


@dataclass(frozen=True)
class CameraAntennaGeometry:
    """Direct geometry observable by one camera and a dual-antenna RTK rig.

    Both vectors are expressed in the camera optical coordinate frame:

    ``camera_to_primary_antenna_in_camera_m``
        Camera origin to the position-producing primary antenna.
    ``primary_to_secondary_antenna_in_camera_m``
        Primary antenna to the secondary antenna used by RELPOS.
    """

    camera_to_primary_antenna_in_camera_m: tuple[float, float, float]
    primary_to_secondary_antenna_in_camera_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        for item in fields(self):
            value = tuple(float(v) for v in getattr(self, item.name))
            if len(value) != 3 or not all(math.isfinite(v) for v in value):
                raise ValueError(f"{item.name} must contain three finite values")
            object.__setattr__(self, item.name, value)
        baseline = np.asarray(
            self.primary_to_secondary_antenna_in_camera_m, dtype=float)
        if np.linalg.norm(baseline) < 1.0e-3:
            raise ValueError("dual-antenna baseline must be non-zero")

    @classmethod
    def from_body_geometry(
            cls,
            body_from_camera: np.ndarray,
            primary_antenna_position_body_m: Sequence[float],
            secondary_antenna_position_body_m: Sequence[float],
    ) -> "CameraAntennaGeometry":
        """Convert explicit body-frame geometry to the direct camera model.

        ``body_from_camera`` is a homogeneous transform mapping camera-frame
        points into the body frame.  Antenna arguments are phase-centre
        positions expressed in that same body frame.
        """
        transform = np.asarray(body_from_camera, dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("body_from_camera must be a finite 4x4 matrix")
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1.0e-10):
            raise ValueError("body_from_camera has an invalid last row")
        rotation = transform[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8)
                or not np.isclose(np.linalg.det(rotation), 1.0,
                                  atol=1.0e-8)):
            raise ValueError("body_from_camera rotation is not proper")
        primary = _finite_vector3(
            primary_antenna_position_body_m,
            "primary_antenna_position_body_m")
        secondary = _finite_vector3(
            secondary_antenna_position_body_m,
            "secondary_antenna_position_body_m")
        camera_origin_body = transform[:3, 3]
        camera_from_body_rotation = rotation.T
        lever_camera = camera_from_body_rotation @ (
            primary - camera_origin_body)
        baseline_camera = camera_from_body_rotation @ (secondary - primary)
        return cls(tuple(lever_camera), tuple(baseline_camera))


def _finite_vector3(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite three-vector")
    return array


@dataclass(frozen=True)
class CalibrationTopics:
    left_image: str
    right_image: str
    fix: str
    relpos: str
    moving_base_pvt: str

    def __post_init__(self) -> None:
        values = [str(getattr(self, item.name)) for item in fields(self)]
        if any(not value.startswith("/") for value in values):
            raise ValueError("calibration topics must be absolute ROS names")
        if len(set(values)) != len(values):
            raise ValueError("calibration topics must be distinct")


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
        payload: dict[str, np.ndarray] = {
            "stereo_frame_id": _field_array(stereo, "frame_id", np.int64),
            "stereo_left_header_ns":
                _field_array(stereo, "left_header_ns", np.int64),
            "stereo_left_log_ns":
                _field_array(stereo, "left_log_ns", np.int64),
            "stereo_right_header_ns":
                _field_array(stereo, "right_header_ns", np.int64),
            "stereo_right_log_ns":
                _field_array(stereo, "right_log_ns", np.int64),
            "stereo_delta_ns":
                np.asarray([item.stereo_delta_ns for item in stereo],
                           dtype=np.int64),
            "stereo_left_sha256":
                _field_array(stereo, "left_sha256", "<U64"),
            "stereo_right_sha256":
                _field_array(stereo, "right_sha256", "<U64"),
            "fix_header_ns": _field_array(fixes, "header_ns", np.int64),
            "fix_log_ns": _field_array(fixes, "log_ns", np.int64),
            "fix_geodetic":
                np.asarray([(item.latitude_deg, item.longitude_deg,
                             item.altitude_ellipsoid_m) for item in fixes],
                           dtype=np.float64).reshape(-1, 3),
            "fix_status": _field_array(fixes, "status", np.int16),
            "fix_service": _field_array(fixes, "service", np.uint16),
            "fix_frame_id": _field_array(fixes, "frame_id", "<U256"),
            "fix_covariance_enu_m2":
                np.asarray([item.covariance_enu_m2 for item in fixes],
                           dtype=np.float64).reshape(-1, 3, 3),
            "fix_covariance_type":
                _field_array(fixes, "covariance_type", np.uint8),
            "relpos_header_ns":
                _field_array(relpos, "header_ns", np.int64),
            "relpos_log_ns": _field_array(relpos, "log_ns", np.int64),
            "relpos_frame_id": _field_array(relpos, "frame_id", "<U256"),
            "relpos_version": _field_array(relpos, "version", np.uint8),
            "relpos_ref_station_id":
                _field_array(relpos, "ref_station_id", np.uint16),
            "relpos_itow_ms": _field_array(relpos, "itow_ms", np.uint32),
            "relpos_ned_cm":
                _tuple_field_array(relpos, "rel_pos_ned_cm", np.int32, 3),
            "relpos_hp_ned_0p1mm":
                _tuple_field_array(
                    relpos, "rel_pos_hp_ned_0p1mm", np.int8, 3),
            "relpos_ned_m":
                _tuple_field_array(relpos, "rel_pos_ned_m", np.float64, 3),
            "relpos_enu_m":
                _tuple_field_array(relpos, "rel_pos_enu_m", np.float64, 3),
            "relpos_length_cm":
                _field_array(relpos, "rel_pos_length_cm", np.int32),
            "relpos_hp_length_0p1mm":
                _field_array(relpos, "rel_pos_hp_length_0p1mm", np.int8),
            "relpos_accuracy_ned_m":
                _tuple_field_array(relpos, "accuracy_ned_m", np.float64, 3),
            "relpos_accuracy_ned_0p1mm":
                _tuple_field_array(
                    relpos, "accuracy_ned_0p1mm", np.uint32, 3),
            "relpos_length_m":
                _field_array(relpos, "rel_pos_length_m", np.float64),
            "relpos_accuracy_length_0p1mm":
                _field_array(
                    relpos, "accuracy_length_0p1mm", np.uint32),
            "relpos_accuracy_length_m":
                _field_array(relpos, "accuracy_length_m", np.float64),
            "relpos_heading_1e5_deg":
                _field_array(
                    relpos, "rel_pos_heading_1e5_deg", np.int32),
            "relpos_heading_deg":
                _field_array(relpos, "rel_pos_heading_deg", np.float64),
            "relpos_accuracy_heading_1e5_deg":
                _field_array(
                    relpos, "accuracy_heading_1e5_deg", np.uint32),
            "relpos_accuracy_heading_deg":
                _field_array(relpos, "accuracy_heading_deg", np.float64),
            "relpos_carrier_solution":
                _field_array(relpos, "carrier_solution", np.uint8),
            "relpos_flags":
                np.asarray([
                    (item.gnss_fix_ok, item.diff_soln, item.rel_pos_valid,
                     item.is_moving, item.ref_pos_miss, item.ref_obs_miss,
                     item.rel_pos_heading_valid, item.rel_pos_normalized)
                    for item in relpos], dtype=np.bool_).reshape(-1, 8),
            "pvt_header_ns": _field_array(pvt, "header_ns", np.int64),
            "pvt_log_ns": _field_array(pvt, "log_ns", np.int64),
            "pvt_frame_id": _field_array(pvt, "frame_id", "<U256"),
            "pvt_itow_ms": _field_array(pvt, "itow_ms", np.uint32),
            "pvt_gps_fix_type":
                _field_array(pvt, "gps_fix_type", np.uint8),
            "pvt_carrier_solution":
                _field_array(pvt, "carrier_solution", np.uint8),
            "pvt_status_flags":
                np.asarray([(item.gnss_fix_ok, item.diff_soln,
                             item.invalid_llh, item.valid_date,
                             item.valid_time, item.fully_resolved)
                            for item in pvt], dtype=np.bool_).reshape(-1, 6),
            "pvt_num_sv": _field_array(pvt, "num_sv", np.uint8),
            "pvt_accuracy_m":
                np.asarray([(item.horizontal_accuracy_m,
                             item.vertical_accuracy_m) for item in pvt],
                           dtype=np.float64).reshape(-1, 2),
            "pvt_position_dop":
                _field_array(pvt, "position_dop", np.float64),
            "pvt_time_accuracy_ns":
                _field_array(pvt, "time_accuracy_ns", np.uint32),
            "pvt_utc_nano_ns":
                _field_array(pvt, "utc_nano_ns", np.int32),
        }
        return payload


def _field_array(records, name: str, dtype) -> np.ndarray:
    return np.asarray([getattr(record, name) for record in records],
                      dtype=dtype)


def _tuple_field_array(records, name: str, dtype, width: int) -> np.ndarray:
    return np.asarray([getattr(record, name) for record in records],
                      dtype=dtype).reshape(-1, width)


_CALIBRATION_UBLOX_TYPES = (
    "CarrSoln",
    "GpsFix",
    "PSMPVT",
    "UBXNavRelPosNED",
    "UBXNavPVT",
)


def build_calibration_typestore(ublox_msgs_dir: Path):
    """Build the minimal Humble typestore for the calibration topics."""
    directory = Path(ublox_msgs_dir)
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    custom_types = {}
    for stem in _CALIBRATION_UBLOX_TYPES:
        source = directory / f"{stem}.msg"
        if not source.is_file():
            raise FileNotFoundError(f"required u-blox message is missing: {source}")
        custom_types.update(get_types_from_msg(
            source.read_text(), f"ublox_ubx_msgs/msg/{stem}"))
    typestore.register(custom_types)
    return typestore


def header_stamp_ns(message) -> int:
    """Exact ROS header timestamp as a signed Python integer."""
    stamp = message.header.stamp
    return int(stamp.sec) * _NS_PER_S + int(stamp.nanosec)


def historical_stereo_bucket(header_ns: int) -> int:
    """Reproduce the original ``round(header_seconds * 30)`` pairing key."""
    seconds, nanoseconds = divmod(int(header_ns), _NS_PER_S)
    header_seconds = seconds + nanoseconds * 1.0e-9
    return round(header_seconds * 30)


def navsat_fix_observation(message, log_ns: int) -> NavSatFixObservation:
    covariance = tuple(float(v) for v in message.position_covariance)
    if len(covariance) != 9 or not all(math.isfinite(v) for v in covariance):
        raise ValueError("NavSatFix contains an invalid covariance")
    return NavSatFixObservation(
        header_ns=header_stamp_ns(message),
        log_ns=int(log_ns),
        frame_id=str(message.header.frame_id),
        latitude_deg=float(message.latitude),
        longitude_deg=float(message.longitude),
        altitude_ellipsoid_m=float(message.altitude),
        status=int(message.status.status),
        service=int(message.status.service),
        covariance_enu_m2=covariance,
        covariance_type=int(message.position_covariance_type),
    )


def relpos_observation(message, log_ns: int) -> RelPosObservation:
    raw_ned = (int(message.rel_pos_n), int(message.rel_pos_e),
               int(message.rel_pos_d))
    hp_ned = (int(message.rel_pos_hp_n), int(message.rel_pos_hp_e),
              int(message.rel_pos_hp_d))
    ned_m = tuple(raw * 0.01 + hp * 1.0e-4
                  for raw, hp in zip(raw_ned, hp_ned))
    enu_m = (ned_m[1], ned_m[0], -ned_m[2])
    raw_acc = (int(message.acc_n), int(message.acc_e), int(message.acc_d))
    acc_m = tuple(value * 1.0e-4 for value in raw_acc)
    raw_length = int(message.rel_pos_length)
    hp_length = int(message.rel_pos_hp_length)
    raw_acc_length = int(message.acc_length)
    raw_heading = int(message.rel_pos_heading)
    raw_acc_heading = int(message.acc_heading)
    return RelPosObservation(
        header_ns=header_stamp_ns(message),
        log_ns=int(log_ns),
        frame_id=str(message.header.frame_id),
        version=int(message.version),
        ref_station_id=int(message.ref_station_id),
        itow_ms=int(message.itow),
        rel_pos_ned_cm=raw_ned,
        rel_pos_hp_ned_0p1mm=hp_ned,
        rel_pos_ned_m=ned_m,
        rel_pos_enu_m=enu_m,
        rel_pos_length_cm=raw_length,
        rel_pos_hp_length_0p1mm=hp_length,
        rel_pos_length_m=raw_length * 0.01 + hp_length * 1.0e-4,
        rel_pos_heading_1e5_deg=raw_heading,
        rel_pos_heading_deg=raw_heading * 1.0e-5,
        accuracy_ned_0p1mm=raw_acc,
        accuracy_ned_m=acc_m,
        accuracy_length_0p1mm=raw_acc_length,
        accuracy_length_m=raw_acc_length * 1.0e-4,
        accuracy_heading_1e5_deg=raw_acc_heading,
        accuracy_heading_deg=raw_acc_heading * 1.0e-5,
        carrier_solution=int(message.carr_soln.status),
        gnss_fix_ok=bool(message.gnss_fix_ok),
        diff_soln=bool(message.diff_soln),
        rel_pos_valid=bool(message.rel_pos_valid),
        is_moving=bool(message.is_moving),
        ref_pos_miss=bool(message.ref_pos_miss),
        ref_obs_miss=bool(message.ref_obs_miss),
        rel_pos_heading_valid=bool(message.rel_pos_heading_valid),
        rel_pos_normalized=bool(message.rel_pos_normalized),
    )


def navpvt_status_observation(message, log_ns: int) \
        -> NavPvtStatusObservation:
    return NavPvtStatusObservation(
        header_ns=header_stamp_ns(message),
        log_ns=int(log_ns),
        frame_id=str(message.header.frame_id),
        itow_ms=int(message.itow),
        gps_fix_type=int(message.gps_fix.fix_type),
        gnss_fix_ok=bool(message.gnss_fix_ok),
        diff_soln=bool(message.diff_soln),
        carrier_solution=int(message.carr_soln.status),
        invalid_llh=bool(message.invalid_llh),
        num_sv=int(message.num_sv),
        horizontal_accuracy_m=float(message.h_acc) * 1.0e-3,
        vertical_accuracy_m=float(message.v_acc) * 1.0e-3,
        position_dop=float(message.p_dop) * 0.01,
        valid_date=bool(message.valid_date),
        valid_time=bool(message.valid_time),
        fully_resolved=bool(message.fully_resolved),
        time_accuracy_ns=int(message.t_acc),
        utc_nano_ns=int(message.nano),
    )


@dataclass(frozen=True)
class _ImageDigest:
    header_ns: int
    log_ns: int
    sha256: str


def _jpeg_bytes(message) -> bytes:
    data = message.data
    return data.tobytes() if hasattr(data, "tobytes") else bytes(data)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_topic_bags(
        bags: Sequence[Path],
        requested_topics: Sequence[str],
) -> dict[str, Path]:
    remaining = set(requested_topics)
    resolved: dict[str, Path] = {}
    normalized_bags = [Path(bag).expanduser() for bag in bags]
    for bag in normalized_bags:
        with Reader(bag) as reader:
            available = {connection.topic for connection in reader.connections}
        for topic in list(remaining):
            if topic in available:
                resolved[topic] = bag
                remaining.remove(topic)
        if not remaining:
            break
    if remaining:
        raise RuntimeError("calibration topics are missing: "
                           + ", ".join(sorted(remaining)))
    return resolved


def _bounded_reader_messages(
        reader,
        connections,
        start_ns: int,
        stop_ns: int,
):
    """Yield a bounded bag interval without MCAP's all-chunk fan-out.

    The rosbags MCAP index reader constructs one generator for every matching
    chunk before yielding its first message.  On a very large, highly
    fragmented exFAT file this turns an otherwise contiguous time interval
    into thousands of pathological random seeks.  A single-file,
    uncompressed rosbag2 MCAP can instead be scanned exactly over the
    contiguous indexed chunk range.

    Other storage layouts retain the library's normal reader path.  Internal
    bounds are restored in ``finally`` and the source file is always opened
    read-only by rosbags.
    """
    directory_storage = getattr(reader, "storage", None)
    storages = getattr(directory_storage, "storages", None)
    metadata = getattr(directory_storage, "metadata", None)
    if not storages or len(storages) != 1 \
            or getattr(metadata, "compression_mode", None) == "message":
        yield from reader.messages(
            connections=connections, start=start_ns, stop=stop_ns)
        return

    storage = storages[0]
    chunks = getattr(storage, "chunks", None)
    scan = getattr(storage, "messages_scan", None)
    if not chunks or scan is None:
        yield from reader.messages(
            connections=connections, start=start_ns, stop=stop_ns)
        return

    requested_topics = {connection.topic for connection in connections}
    storage_connections = [
        connection for connection in storage.connections
        if connection.topic in requested_topics]
    if not storage_connections:
        return
    connection_by_storage_id = {
        connection.id: next(
            requested for requested in connections
            if requested.topic == connection.topic)
        for connection in storage_connections
    }
    channel_ids = set(connection_by_storage_id)
    ordered = sorted(chunks, key=lambda chunk: chunk.chunk_start_offset)
    matching_positions = [
        index for index, chunk in enumerate(ordered)
        if start_ns < chunk.message_end_time
        and chunk.message_start_time < stop_ns
        and any(chunk.channel_count.get(channel_id, 0)
                for channel_id in channel_ids)
    ]
    if not matching_positions:
        return
    first = min(matching_positions)
    last = max(matching_positions)
    bounded_start = ordered[first].chunk_start_offset
    bounded_stop = (
        ordered[last + 1].chunk_start_offset
        if last + 1 < len(ordered) else storage.data_end)
    original_start, original_stop = storage.data_start, storage.data_end
    storage.data_start, storage.data_end = bounded_start, bounded_stop
    try:
        for connection, timestamp, data in scan(
                storage_connections, start_ns, stop_ns):
            yield (
                connection_by_storage_id[connection.id],
                timestamp,
                data,
            )
    finally:
        storage.data_start, storage.data_end = original_start, original_stop


def _validate_windows(start_ns: int, stop_ns: int, label: str) -> tuple[int, int]:
    start, stop = int(start_ns), int(stop_ns)
    if stop < start:
        raise ValueError(f"{label} stop precedes start")
    return start, stop


def read_calibration_bag_observations(
        bags: Sequence[Path],
        topics: CalibrationTopics,
        typestore,
        image_start_header_ns: int,
        image_stop_header_ns: int,
        frame_stride: int,
        extracted_images_dir: Path,
        expected_frame_count: int | None = None,
        *,
        rtk_start_header_ns: int | None = None,
        rtk_stop_header_ns: int | None = None,
        log_start_ns: int | None = None,
        log_stop_ns: int | None = None,
        default_log_margin_ns: int = 5 * _NS_PER_S,
) -> CalibrationBagObservations:
    """Recover calibration evidence in one bounded message pass per bag.

    Topics are resolved to the first listed bag containing each topic, matching
    the existing pipeline contract.  A bag containing several requested topics
    is opened for one grouped ``messages`` pass.

    The log-time bounds are explicit because ROS header clocks and bag log
    clocks need not be identical.  If omitted, the union of the image/RTK
    header windows is expanded by ``default_log_margin_ns``.
    """
    image_window = _validate_windows(
        image_start_header_ns, image_stop_header_ns, "image window")
    rtk_window = _validate_windows(
        image_window[0] if rtk_start_header_ns is None else rtk_start_header_ns,
        image_window[1] if rtk_stop_header_ns is None else rtk_stop_header_ns,
        "RTK window")
    stride = int(frame_stride)
    if stride < 1:
        raise ValueError("frame_stride must be at least one")
    margin = int(default_log_margin_ns)
    if margin < 0:
        raise ValueError("default_log_margin_ns cannot be negative")
    union_start = min(image_window[0], rtk_window[0])
    union_stop = max(image_window[1], rtk_window[1])
    log_window = _validate_windows(
        union_start - margin if log_start_ns is None else log_start_ns,
        union_stop + margin if log_stop_ns is None else log_stop_ns,
        "log window")

    topic_values = tuple(str(getattr(topics, item.name))
                         for item in fields(topics))
    topic_bags = _resolve_topic_bags(bags, topic_values)
    grouped: dict[Path, set[str]] = {}
    for topic, bag in topic_bags.items():
        grouped.setdefault(bag, set()).add(topic)

    selected_left: list[tuple[int, _ImageDigest]] = []
    right_by_bucket: dict[int, _ImageDigest] = {}
    fixes: list[NavSatFixObservation] = []
    relpos: list[RelPosObservation] = []
    pvt: list[NavPvtStatusObservation] = []
    seen_left = 0
    selected_left_buckets: set[int] = set()

    for bag, bag_topics in grouped.items():
        with Reader(bag) as reader:
            connections = [
                connection for connection in reader.connections
                if connection.topic in bag_topics]
            for connection, log_ns_value, raw in _bounded_reader_messages(
                    reader, connections, log_window[0], log_window[1]):
                message = typestore.deserialize_cdr(raw, connection.msgtype)
                stamp_ns = header_stamp_ns(message)
                topic = connection.topic
                if topic in {topics.left_image, topics.right_image}:
                    if not image_window[0] <= stamp_ns <= image_window[1]:
                        continue
                    bucket = historical_stereo_bucket(stamp_ns)
                    if topic == topics.left_image:
                        take = seen_left % stride == 0
                        seen_left += 1
                        if not take:
                            continue
                        if bucket in selected_left_buckets:
                            raise RuntimeError(
                                f"ambiguous selected-left stereo bucket {bucket}")
                        selected_left_buckets.add(bucket)
                        selected_left.append((
                            bucket,
                            _ImageDigest(
                                header_ns=stamp_ns,
                                log_ns=int(log_ns_value),
                                sha256=_sha256_bytes(_jpeg_bytes(message)))))
                    else:
                        if bucket in right_by_bucket:
                            raise RuntimeError(
                                f"ambiguous right stereo bucket {bucket}")
                        right_by_bucket[bucket] = _ImageDigest(
                            header_ns=stamp_ns,
                            log_ns=int(log_ns_value),
                            sha256=_sha256_bytes(_jpeg_bytes(message)))
                elif rtk_window[0] <= stamp_ns <= rtk_window[1]:
                    if topic == topics.fix:
                        fixes.append(navsat_fix_observation(
                            message, log_ns_value))
                    elif topic == topics.relpos:
                        relpos.append(relpos_observation(
                            message, log_ns_value))
                    elif topic == topics.moving_base_pvt:
                        pvt.append(navpvt_status_observation(
                            message, log_ns_value))

    if not selected_left:
        raise RuntimeError("bounded pass found no selected left images")
    if not fixes or not relpos or not pvt:
        raise RuntimeError(
            "bounded pass did not recover all RTK streams "
            f"(fix={len(fixes)}, relpos={len(relpos)}, pvt={len(pvt)})")

    frames_out: list[StereoFrameObservation] = []
    image_dir = Path(extracted_images_dir)
    for frame_id, (bucket, left) in enumerate(selected_left):
        right = right_by_bucket.get(bucket)
        if right is None:
            raise RuntimeError(
                f"selected left frame {frame_id} has no right image in "
                f"historical bucket {bucket}")
        left_path = image_dir / f"left_{frame_id:06d}.jpg"
        right_path = image_dir / f"right_{frame_id:06d}.jpg"
        if not left_path.is_file() or not right_path.is_file():
            raise FileNotFoundError(
                f"extracted stereo pair {frame_id} is missing in {image_dir}")
        extracted_left_hash = _sha256_file(left_path)
        extracted_right_hash = _sha256_file(right_path)
        if extracted_left_hash != left.sha256:
            raise ValueError(
                f"left JPEG hash mismatch at extracted frame {frame_id}")
        if extracted_right_hash != right.sha256:
            raise ValueError(
                f"right JPEG hash mismatch at extracted frame {frame_id}")
        frames_out.append(StereoFrameObservation(
            frame_id=frame_id,
            left_header_ns=left.header_ns,
            left_log_ns=left.log_ns,
            right_header_ns=right.header_ns,
            right_log_ns=right.log_ns,
            left_sha256=left.sha256,
            right_sha256=right.sha256,
        ))

    if expected_frame_count is not None \
            and len(frames_out) != int(expected_frame_count):
        raise RuntimeError(
            f"recovered {len(frames_out)} stereo pairs, expected "
            f"{int(expected_frame_count)}")
    extra_left = image_dir / f"left_{len(frames_out):06d}.jpg"
    extra_right = image_dir / f"right_{len(frames_out):06d}.jpg"
    if extra_left.exists() or extra_right.exists():
        raise RuntimeError(
            "extracted image directory contains more frames than recovered")

    fixes.sort(key=lambda item: (item.header_ns, item.log_ns))
    relpos.sort(key=lambda item: (item.header_ns, item.log_ns))
    pvt.sort(key=lambda item: (item.header_ns, item.log_ns))
    return CalibrationBagObservations(
        stereo_frames=tuple(frames_out),
        fixes=tuple(fixes),
        relpos=tuple(relpos),
        moving_base_pvt=tuple(pvt),
        topic_bags=tuple(sorted(
            (topic, str(path)) for topic, path in topic_bags.items())),
        header_window_ns=image_window,
        rtk_window_ns=rtk_window,
        log_window_ns=log_window,
    )


@dataclass(frozen=True)
class RawColmapRigTrajectory:
    """Chronological raw left-camera trajectory in the visual world."""

    frame_indices: np.ndarray
    colmap_frame_ids: np.ndarray
    image_ids: np.ndarray
    image_names: tuple[str, ...]
    camera_from_visual_world: np.ndarray
    camera_centers_visual: np.ndarray


@dataclass(frozen=True)
class _RigDefinition:
    reference_sensor: tuple[str, int]
    sensor_from_rig: Mapping[tuple[str, int], np.ndarray]


def _qvec_to_rotmat(qvec: Sequence[float]) -> np.ndarray:
    q = np.asarray(qvec, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("invalid COLMAP quaternion")
    norm = np.linalg.norm(q)
    if norm < 1.0e-12:
        raise ValueError("zero COLMAP quaternion")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def _pose_matrix(qvec: Sequence[float], translation: Sequence[float]) \
        -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = _qvec_to_rotmat(qvec)
    vector = np.asarray(translation, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("invalid COLMAP translation")
    matrix[:3, 3] = vector
    return matrix


def _data_lines(path: Path) -> Iterator[list[str]]:
    with path.open() as stream:
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                yield stripped.split()


def _parse_rigs(path: Path) -> dict[int, _RigDefinition]:
    rigs: dict[int, _RigDefinition] = {}
    for tokens in _data_lines(path):
        if len(tokens) < 4:
            raise ValueError(f"malformed COLMAP rig row in {path}")
        rig_id, sensor_count = int(tokens[0]), int(tokens[1])
        reference = (tokens[2], int(tokens[3]))
        poses: dict[tuple[str, int], np.ndarray] = {reference: np.eye(4)}
        cursor = 4
        while cursor < len(tokens):
            if cursor + 2 >= len(tokens):
                raise ValueError(f"truncated COLMAP rig sensor in {path}")
            key = (tokens[cursor], int(tokens[cursor + 1]))
            has_pose = int(tokens[cursor + 2])
            cursor += 3
            if key in poses:
                raise ValueError(f"duplicate COLMAP rig sensor {key}")
            if not has_pose:
                raise ValueError(
                    f"non-reference COLMAP rig sensor {key} has no pose")
            if cursor + 7 > len(tokens):
                raise ValueError(f"truncated COLMAP rig pose for {key}")
            poses[key] = _pose_matrix(
                [float(value) for value in tokens[cursor:cursor + 4]],
                [float(value) for value in tokens[cursor + 4:cursor + 7]])
            cursor += 7
        if len(poses) != sensor_count:
            raise ValueError(
                f"rig {rig_id} declares {sensor_count} sensors, parsed "
                f"{len(poses)}")
        if rig_id in rigs:
            raise ValueError(f"duplicate COLMAP rig id {rig_id}")
        rigs[rig_id] = _RigDefinition(reference, poses)
    if not rigs:
        raise ValueError(f"no COLMAP rigs parsed from {path}")
    return rigs


def _read_database_images(database_path: Path) \
        -> dict[int, tuple[str, int]]:
    path = Path(database_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"COLMAP database is missing: {path}")
    uri = "file:" + quote(str(path), safe="/") + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            "SELECT image_id, name, camera_id FROM images").fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError(f"COLMAP database has no images: {path}")
    return {int(image_id): (str(name).replace("\\", "/"), int(camera_id))
            for image_id, name, camera_id in rows}


def _frame_index_from_name(name: str, left_prefix: str) -> int:
    normalized_prefix = left_prefix.replace("\\", "/")
    if not name.startswith(normalized_prefix):
        raise ValueError(f"image {name!r} does not use prefix {left_prefix!r}")
    path = Path(name)
    if path.suffix.lower() not in _IMAGE_SUFFIXES or not path.stem.isdigit():
        raise ValueError(f"left image name has no numeric frame id: {name}")
    return int(path.stem)


def load_raw_colmap_rig_trajectory(
        model_dir: Path,
        database_path: Path,
        *,
        left_prefix: str = "zed/left/",
        expected_frame_count: int | None = None,
) -> RawColmapRigTrajectory:
    """Load raw chronological left poses from ``frames.txt`` and the database.

    ``images.txt`` is intentionally never opened.  The database is opened in
    immutable read-only mode and is used only to map image IDs to names.
    """
    model = Path(model_dir)
    rigs = _parse_rigs(model / "rigs.txt")
    images = _read_database_images(database_path)
    records = []
    for tokens in _data_lines(model / "frames.txt"):
        if len(tokens) < 10:
            raise ValueError("malformed COLMAP frame row")
        frame_id, rig_id = int(tokens[0]), int(tokens[1])
        if rig_id not in rigs:
            raise ValueError(f"frame {frame_id} references unknown rig {rig_id}")
        rig_from_world = _pose_matrix(
            [float(value) for value in tokens[2:6]],
            [float(value) for value in tokens[6:9]])
        count = int(tokens[9])
        if len(tokens) != 10 + 3 * count:
            raise ValueError(f"frame {frame_id} has malformed data IDs")
        data = []
        for cursor in range(10, len(tokens), 3):
            data.append((tokens[cursor], int(tokens[cursor + 1]),
                         int(tokens[cursor + 2])))
        left_candidates = []
        for sensor_type, sensor_id, image_id in data:
            if image_id not in images:
                raise ValueError(
                    f"frame {frame_id} references missing image {image_id}")
            name, camera_id = images[image_id]
            if sensor_type == "CAMERA" and name.startswith(left_prefix):
                if camera_id != sensor_id:
                    raise ValueError(
                        f"image {image_id} camera id {camera_id} disagrees "
                        f"with rig sensor id {sensor_id}")
                left_candidates.append((sensor_type, sensor_id,
                                        image_id, name))
        if len(left_candidates) != 1:
            raise ValueError(
                f"frame {frame_id} has {len(left_candidates)} left images")
        sensor_type, sensor_id, image_id, name = left_candidates[0]
        sensor_key = (sensor_type, sensor_id)
        rig = rigs[rig_id]
        if sensor_key not in rig.sensor_from_rig:
            raise ValueError(
                f"frame {frame_id} uses sensor {sensor_key} absent from rig")
        camera_from_world = (
            rig.sensor_from_rig[sensor_key] @ rig_from_world)
        index = _frame_index_from_name(name, left_prefix)
        rotation = camera_from_world[:3, :3]
        center = -rotation.T @ camera_from_world[:3, 3]
        records.append((index, frame_id, image_id, name,
                        camera_from_world, center))

    if not records:
        raise ValueError(f"no COLMAP frames parsed from {model/'frames.txt'}")
    records.sort(key=lambda item: item[0])
    indices = np.asarray([item[0] for item in records], dtype=np.int64)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("COLMAP model contains duplicate left frame indices")
    if expected_frame_count is not None:
        expected = int(expected_frame_count)
        if not np.array_equal(indices, np.arange(expected, dtype=np.int64)):
            raise ValueError(
                "COLMAP left frame indices are not the expected contiguous "
                f"range 0..{expected - 1}")
    viewmats = np.stack([item[4] for item in records]).astype(np.float64)
    rotations = viewmats[:, :3, :3]
    if (not np.allclose(
            rotations @ np.swapaxes(rotations, 1, 2), np.eye(3),
            atol=1.0e-8)
            or not np.allclose(np.linalg.det(rotations), 1.0,
                               atol=1.0e-8)):
        raise ValueError("raw COLMAP trajectory contains invalid rotations")
    return RawColmapRigTrajectory(
        frame_indices=indices,
        colmap_frame_ids=np.asarray(
            [item[1] for item in records], dtype=np.int64),
        image_ids=np.asarray([item[2] for item in records], dtype=np.int64),
        image_names=tuple(item[3] for item in records),
        camera_from_visual_world=viewmats,
        camera_centers_visual=np.stack(
            [item[5] for item in records]).astype(np.float64),
    )


def _fsync_tree(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def atomic_calibration_artifact(
        segment_dir: Path,
        name: str,
) -> Iterator[Path]:
    """Yield a temporary directory and atomically publish it on success.

    A lock created with ``O_EXCL`` serializes cooperating writers.  The final
    directory is never reused or overwritten.  Exceptions remove the temporary
    directory and leave no published artifact.
    """
    safe_name = validate_calibration_artifact_name(name)
    root = Path(segment_dir) / "calibration_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    final = root / safe_name
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"calibration artifact already exists: {final}")
    lock = root / f".{safe_name}.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"calibration artifact writer is already active: {lock}") from exc
    temp: Path | None = None
    try:
        os.write(lock_fd, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(lock_fd)
        temp = Path(tempfile.mkdtemp(
            prefix=f".{safe_name}.tmp-", dir=root))
        yield temp
        if not any(temp.iterdir()):
            raise RuntimeError("refusing to publish an empty calibration artifact")
        _fsync_tree(temp)
        if final.exists() or final.is_symlink():
            raise FileExistsError(
                f"calibration artifact appeared during write: {final}")
        os.rename(temp, final)
        root_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
    finally:
        os.close(lock_fd)
        if temp is not None and temp.exists():
            shutil.rmtree(temp)
        lock.unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    """Deterministic JSON helper intended for an artifact temporary directory."""
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
