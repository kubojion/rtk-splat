"""ROS1 chained-bag adapter for RTK-anchored rectified stereo datasets.

The module is named for the first supported dataset, but its implementation is
not tied to CitrusFarm filenames or coordinates.  Dataset facts live in the
robot/sequence configuration.  The adapter reads ROS1 bags without a ROS
installation, retains exact sensor and bag-log stamps, derives observable
single-antenna course heading, and samples frames by travelled distance.

RGB and RTK are the primary method inputs.  Recorded depth/confidence topics
are inventoried explicitly but are not silently enabled as training inputs;
the first CitrusFarm reproduction derives stereo depth in a separate stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from rtk_splat.adapters.image_decode import decode_raw_image
from rtk_splat.adapters.ros2_zed_ublox import (
    FrameRecord,
    RELPOS_FLAG_NAMES,
    RtkTrack,
    publish_segment_v2,
)
from rtk_splat.adapters.synchronization import nearest_matches
from rtk_splat.core.segment import normalize_navsat_position_quality
from rtk_splat.core.poses import LocalEnu, pose_frames_from_extrinsic


NANOSECONDS = 1_000_000_000


@dataclass(frozen=True)
class BagChunk:
    path: str
    start_log_ns: int
    stop_log_ns: int
    topics: tuple[str, ...]


@dataclass(frozen=True)
class BagChainReport:
    chunks: tuple[BagChunk, ...]
    gaps_ns: tuple[int, ...]
    maximum_gap_ns: int
    maximum_overlap_ns: int

    @property
    def start_log_ns(self) -> int:
        return self.chunks[0].start_log_ns

    @property
    def stop_log_ns(self) -> int:
        return self.chunks[-1].stop_log_ns

    def as_json(self) -> dict[str, Any]:
        return {
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "gaps_ns": list(self.gaps_ns),
            "maximum_gap_ns": self.maximum_gap_ns,
            "maximum_overlap_ns": self.maximum_overlap_ns,
            "start_log_ns": self.start_log_ns,
            "stop_log_ns": self.stop_log_ns,
        }


@dataclass(frozen=True)
class ClockEstimate:
    camera_to_gnss_header_offset_ns: int
    configured_offset_ns: int
    configured_error_ns: int
    gnss_header_minus_log_median_ns: int
    camera_header_minus_log_median_ns: int
    camera_to_gnss_start_ns: int
    camera_to_gnss_stop_ns: int
    drift_ns: int
    sample_count: int

    def as_json(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ReceiverStateEvidence:
    """Complete decoded receiver-state stream retained beside NavSatFix.

    ``fix_mode`` is the receiver's own solution classification.  It is kept
    separately from ``NavSatStatus`` because ROS status 2 only says that a
    ground-based augmentation service was used; it cannot distinguish RTK
    float from RTK fixed.
    """

    topic: str
    message_type: str
    message_definition_sha256: str
    header_ns: np.ndarray
    log_ns: np.ndarray
    fix_mode: np.ndarray
    rtk_mode_fix: np.ndarray
    num_sat: np.ndarray
    system_error: np.ndarray
    io_error: np.ndarray
    swift_nap_error: np.ndarray
    external_antenna_present: np.ndarray

    def summary(self) -> dict[str, Any]:
        modes, counts = np.unique(self.fix_mode.astype(str), return_counts=True)
        return {
            "topic": self.topic,
            "message_type": self.message_type,
            "message_definition_sha256": self.message_definition_sha256,
            "sample_count": int(len(self.header_ns)),
            "fix_mode_counts": {
                str(mode): int(count) for mode, count in zip(modes, counts)
            },
            "rtk_mode_fix_count": int(np.count_nonzero(self.rtk_mode_fix)),
            "receiver_error_count": int(
                np.count_nonzero(
                    self.system_error | self.io_error | self.swift_nap_error
                )
            ),
            "minimum_num_sat": int(np.min(self.num_sat)),
            "maximum_num_sat": int(np.max(self.num_sat)),
        }


@dataclass
class _TimedImage:
    header_ns: int
    log_ns: int
    message: Any


class _DistanceSampler:
    """Residual-carry arc-length sampler for irregular frame rates."""

    def __init__(self, spacing_m: float):
        if not math.isfinite(spacing_m) or spacing_m <= 0:
            raise ValueError("segment.frame_spacing_m must be positive and finite")
        self.spacing_m = float(spacing_m)
        self.previous: np.ndarray | None = None
        self.residual_m = 0.0
        self.path_length_m = 0.0
        self.selected = 0

    def accept(self, position: Sequence[float]) -> bool:
        value = np.asarray(position, dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("metric sampler position must be finite xyz")
        if self.previous is None:
            self.previous = value
            self.selected = 1
            return True
        distance = float(np.linalg.norm(value - self.previous))
        self.previous = value
        self.path_length_m += distance
        self.residual_m += distance
        if self.residual_m + 1.0e-12 < self.spacing_m:
            return False
        self.residual_m %= self.spacing_m
        self.selected += 1
        return True


def _reader_type():
    try:
        from rosbags.rosbag1 import Reader
    except ImportError as exc:  # pragma: no cover - optional dependency gate
        raise RuntimeError(
            "ROS1 bag ingestion requires the optional 'rosbags' package"
        ) from exc
    return Reader


def build_typestore():
    try:
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:  # pragma: no cover - optional dependency gate
        raise RuntimeError(
            "ROS1 bag ingestion requires the optional 'rosbags' package"
        ) from exc
    return get_typestore(Stores.ROS1_NOETIC)


def _stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * NANOSECONDS + int(stamp.nanosec)


def _field(message: Any, lower: str, upper: str | None = None):
    if hasattr(message, lower):
        return getattr(message, lower)
    alternate = lower.upper() if upper is None else upper
    if hasattr(message, alternate):
        return getattr(message, alternate)
    raise AttributeError(f"message has neither {lower!r} nor {alternate!r}")


def _plain_metadata(value: Any) -> Any:
    """Recursively remove YAML config namespaces before publication."""
    if isinstance(value, Mapping):
        return {str(key): _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_metadata(item) for item in value]
    if hasattr(value, "__dict__"):
        return _plain_metadata(vars(value))
    return value


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", path.name)
    )


def _configured_paths(cfg: Any, name: str) -> list[Path]:
    value = getattr(cfg.paths, name, None)
    glob_value = getattr(cfg.paths, f"{name[:-1]}_glob", None)
    if value is not None and glob_value is not None:
        raise ValueError(
            f"declare paths.{name} or paths.{name[:-1]}_glob, not both"
        )
    if glob_value is not None:
        expression = Path(str(glob_value)).expanduser()
        paths = sorted(expression.parent.glob(expression.name), key=_natural_key)
    else:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"paths.{name} must be an explicit ordered list")
        paths = [Path(str(item)).expanduser() for item in value]
    if not paths:
        raise ValueError(f"paths.{name} resolves to no bags")
    duplicates = [str(path) for path in paths if paths.count(path) > 1]
    if duplicates:
        raise ValueError(f"paths.{name} contains duplicate paths")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing ROS1 bag: {missing[0]}")
    return paths


def validate_bag_chain(
    paths: Sequence[Path],
    *,
    required_each: Iterable[str],
    expected_count: int | None,
    maximum_gap_ns: int,
    maximum_overlap_ns: int,
) -> BagChainReport:
    """Read only ROS1 indexes and validate declared chunk order/coverage."""
    if expected_count is not None and len(paths) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} bag chunks, found {len(paths)}"
        )
    Reader = _reader_type()
    required = set(required_each)
    chunks: list[BagChunk] = []
    for path in paths:
        with Reader(path) as reader:
            topics = {connection.topic for connection in reader.connections}
            missing = required - topics
            if missing:
                raise RuntimeError(
                    f"{path} is missing required topics: "
                    + ", ".join(sorted(missing))
                )
            if int(reader.end_time) <= int(reader.start_time):
                raise RuntimeError(f"{path} has an empty/invalid ROS time range")
            chunks.append(
                BagChunk(
                    path=str(path),
                    start_log_ns=int(reader.start_time),
                    stop_log_ns=int(reader.end_time),
                    topics=tuple(sorted(topics)),
                )
            )
    gaps = tuple(
        chunks[index + 1].start_log_ns - chunks[index].stop_log_ns
        for index in range(len(chunks) - 1)
    )
    if any(gap > maximum_gap_ns for gap in gaps):
        index = next(i for i, gap in enumerate(gaps) if gap > maximum_gap_ns)
        raise RuntimeError(
            f"bag-chain gap exceeds {maximum_gap_ns} ns between "
            f"{chunks[index].path} and {chunks[index + 1].path}: {gaps[index]} ns"
        )
    if any(gap < -maximum_overlap_ns for gap in gaps):
        index = next(i for i, gap in enumerate(gaps) if gap < -maximum_overlap_ns)
        raise RuntimeError(
            f"bag-chain overlap exceeds {maximum_overlap_ns} ns between "
            f"{chunks[index].path} and {chunks[index + 1].path}: {-gaps[index]} ns"
        )
    starts = [chunk.start_log_ns for chunk in chunks]
    if any(right <= left for left, right in zip(starts, starts[1:])):
        raise RuntimeError("bag chunks are not in increasing ROS-log order")
    return BagChainReport(
        chunks=tuple(chunks),
        gaps_ns=gaps,
        maximum_gap_ns=maximum_gap_ns,
        maximum_overlap_ns=maximum_overlap_ns,
    )


def _deserialize(typestore: Any, raw: bytes, msgtype: str):
    return typestore.deserialize_ros1(raw, msgtype)


def read_navsat_track(
    bags: Sequence[Path], topic: str, typestore: Any
) -> RtkTrack:
    """Read a chained ROS1 NavSatFix stream without dropping evidence."""
    Reader = _reader_type()
    header_ns: list[int] = []
    log_ns: list[int] = []
    latitude: list[float] = []
    longitude: list[float] = []
    altitude: list[float] = []
    status: list[int] = []
    service: list[int] = []
    covariance: list[np.ndarray] = []
    covariance_type: list[int] = []
    for bag in bags:
        with Reader(bag) as reader:
            connections = [c for c in reader.connections if c.topic == topic]
            if not connections:
                raise RuntimeError(f"{bag} has no NavSatFix topic {topic}")
            for connection, timestamp, raw in reader.messages(connections=connections):
                message = _deserialize(typestore, raw, connection.msgtype)
                header_ns.append(_stamp_ns(message))
                log_ns.append(int(timestamp))
                latitude.append(float(message.latitude))
                longitude.append(float(message.longitude))
                altitude.append(float(message.altitude))
                status.append(int(message.status.status))
                service.append(int(message.status.service))
                matrix = np.asarray(
                    message.position_covariance, dtype=np.float64
                ).reshape(3, 3)
                covariance.append(matrix)
                covariance_type.append(int(message.position_covariance_type))
    if len(header_ns) < 3:
        raise RuntimeError(f"fewer than three NavSatFix messages on {topic}")
    header = np.asarray(header_ns, dtype=np.int64)
    log = np.asarray(log_ns, dtype=np.int64)
    if np.any(np.diff(header) <= 0):
        raise RuntimeError("GNSS chunks overlap or duplicate header timestamps")
    if np.any(np.diff(log) <= 0):
        raise RuntimeError("GNSS chunks overlap or duplicate bag-log timestamps")
    cov = np.asarray(covariance, dtype=np.float64)
    valid_fix = np.asarray(status, dtype=np.int16) >= 0
    valid_covariance = (
        np.isfinite(cov).all(axis=(1, 2))
        & (np.diagonal(cov, axis1=1, axis2=2) >= 0).all(axis=1)
        & np.isclose(cov, np.swapaxes(cov, 1, 2), atol=1e-9).all(axis=(1, 2))
    )
    if np.any(valid_fix & ~valid_covariance):
        index = int(np.flatnonzero(valid_fix & ~valid_covariance)[0])
        raise RuntimeError(
            f"valid NavSatFix sample {index} has invalid ENU covariance"
        )
    placeholder_ns = np.array([-1], dtype=np.int64)
    track = RtkTrack(
        fix_t=header.astype(np.float64) * 1.0e-9,
        fix_lat=np.asarray(latitude, dtype=np.float64),
        fix_lon=np.asarray(longitude, dtype=np.float64),
        fix_alt=np.asarray(altitude, dtype=np.float64),
        fix_status=np.asarray(status, dtype=np.int16),
        fix_cov_max=np.max(np.diagonal(cov, axis1=1, axis2=2), axis=1),
        relpos_t=placeholder_ns.astype(np.float64),
        relpos_yaw=np.zeros(1, dtype=np.float64),
        relpos_carr=np.full(1, -1, dtype=np.int8),
        fix_header_ns=header,
        fix_log_ns=log,
        fix_covariance_enu_m2=cov,
        fix_covariance_type=np.asarray(covariance_type, dtype=np.int16),
        fix_carrier_status=np.full(len(header), -1, dtype=np.int8),
        relpos_header_ns=placeholder_ns,
        relpos_log_ns=placeholder_ns,
    )
    track.fix_service = np.asarray(service, dtype=np.int16)
    origin_index = next(
        (
            index
            for index in range(len(header))
            if status[index] >= 0
            and np.isfinite([latitude[index], longitude[index], altitude[index]]).all()
        ),
        None,
    )
    if origin_index is None:
        raise RuntimeError("NavSatFix stream has no valid finite origin")
    enu = LocalEnu(
        latitude[origin_index], longitude[origin_index], altitude[origin_index]
    )
    track.enu_xyz = enu.to_enu(track.fix_lat, track.fix_lon, track.fix_alt)
    track.enu = enu
    track.origin = {
        "lat0": latitude[origin_index],
        "lon0": longitude[origin_index],
        "alt0": altitude[origin_index],
    }
    return track


_RECEIVER_MODE_ALIASES = {
    "INVALID": "INVALID",
    "SPP": "SPP",
    "SINGLE_POINT_POSITION": "SPP",
    "DGNSS": "DGNSS",
    "DIFFERENTIAL_GNSS": "DGNSS",
    "FLOAT_RTK": "FLOAT_RTK",
    "RTK_FLOAT": "FLOAT_RTK",
    "FIXED_RTK": "FIXED_RTK",
    "RTK_FIXED": "FIXED_RTK",
    "DEAD_RECKONING": "DEAD_RECKONING",
    "SBAS": "SBAS",
    "UNKNOWN": "UNKNOWN",
}


def _canonical_receiver_fix_mode(value: Any) -> str:
    text = re.sub(r"[^A-Z0-9]+", "_", str(value).strip().upper()).strip("_")
    return _RECEIVER_MODE_ALIASES.get(text, f"UNRECOGNIZED:{text}")


def _register_ros1_connection_type(typestore: Any, connection: Any) -> None:
    """Register an embedded ROS1 custom definition without a ROS install."""
    if connection.msgtype in typestore.types:
        return
    try:
        from rosbags.typesys import get_types_from_msg
    except ImportError as exc:  # pragma: no cover - optional dependency gate
        raise RuntimeError(
            "ROS1 custom-message decoding requires the optional 'rosbags' package"
        ) from exc
    definition = getattr(connection, "msgdef", None)
    data = getattr(definition, "data", None)
    if not isinstance(data, str) or not data.strip():
        raise RuntimeError(
            f"ROS1 bag does not embed a definition for {connection.msgtype}"
        )
    typestore.register(get_types_from_msg(data, connection.msgtype))


def read_receiver_state(
    bags: Sequence[Path], topic: str, typestore: Any
) -> ReceiverStateEvidence:
    """Decode a Piksi-style receiver state while retaining classification evidence.

    The message type is discovered from the bag's embedded definition.  This
    keeps the adapter independent of a local ROS/piksi_rtk_msgs installation,
    while required fields are validated explicitly so a look-alike message
    cannot silently be interpreted as carrier state.
    """
    Reader = _reader_type()
    headers: list[int] = []
    logs: list[int] = []
    modes: list[str] = []
    rtk_fixed: list[bool] = []
    num_sat: list[int] = []
    system_error: list[int] = []
    io_error: list[int] = []
    swift_nap_error: list[int] = []
    external_antenna: list[int] = []
    schema: tuple[str, str] | None = None
    required_fields = (
        "header",
        "fix_mode",
        "rtk_mode_fix",
        "num_sat",
        "system_error",
        "io_error",
        "swift_nap_error",
        "external_antenna_present",
    )
    for bag in bags:
        with Reader(bag) as reader:
            connections = [c for c in reader.connections if c.topic == topic]
            for connection in connections:
                definition = getattr(getattr(connection, "msgdef", None), "data", "")
                digest = hashlib.sha256(definition.encode("utf-8")).hexdigest()
                current_schema = (str(connection.msgtype), digest)
                if schema is None:
                    schema = current_schema
                elif current_schema != schema:
                    raise RuntimeError(
                        f"receiver-state schema changes across bag chunks on {topic}"
                    )
                _register_ros1_connection_type(typestore, connection)
            for connection, timestamp, raw in reader.messages(connections=connections):
                message = _deserialize(typestore, raw, connection.msgtype)
                missing = [name for name in required_fields if not hasattr(message, name)]
                if missing:
                    raise RuntimeError(
                        f"receiver-state message {connection.msgtype} lacks: "
                        + ", ".join(missing)
                    )
                mode = _canonical_receiver_fix_mode(message.fix_mode)
                fixed_flag = bool(message.rtk_mode_fix)
                if (mode == "FIXED_RTK") != fixed_flag:
                    raise RuntimeError(
                        "receiver-state fix_mode and rtk_mode_fix disagree at "
                        f"header timestamp {_stamp_ns(message)}"
                    )
                headers.append(_stamp_ns(message))
                logs.append(int(timestamp))
                modes.append(mode)
                rtk_fixed.append(fixed_flag)
                num_sat.append(int(message.num_sat))
                system_error.append(int(message.system_error))
                io_error.append(int(message.io_error))
                swift_nap_error.append(int(message.swift_nap_error))
                external_antenna.append(int(message.external_antenna_present))
    if not headers or schema is None:
        raise RuntimeError(f"no receiver-state messages on {topic}")
    header = np.asarray(headers, dtype=np.int64)
    log = np.asarray(logs, dtype=np.int64)
    if np.any(np.diff(header) <= 0):
        raise RuntimeError(
            "receiver-state chunks overlap or duplicate header timestamps"
        )
    if np.any(np.diff(log) <= 0):
        raise RuntimeError(
            "receiver-state chunks overlap or duplicate bag-log timestamps"
        )
    return ReceiverStateEvidence(
        topic=str(topic),
        message_type=schema[0],
        message_definition_sha256=schema[1],
        header_ns=header,
        log_ns=log,
        fix_mode=np.asarray(modes, dtype=np.str_),
        rtk_mode_fix=np.asarray(rtk_fixed, dtype=bool),
        num_sat=np.asarray(num_sat, dtype=np.int16),
        system_error=np.asarray(system_error, dtype=np.int16),
        io_error=np.asarray(io_error, dtype=np.int16),
        swift_nap_error=np.asarray(swift_nap_error, dtype=np.int16),
        external_antenna_present=np.asarray(external_antenna, dtype=np.int16),
    )


def associate_receiver_state(
    track: RtkTrack,
    evidence: ReceiverStateEvidence,
    *,
    tolerance_ns: int,
    required: bool,
) -> dict[str, Any]:
    """Associate receiver states to every raw NavSatFix and classify quality."""
    fixed_mode = np.asarray(evidence.fix_mode, dtype=str) == "FIXED_RTK"
    if not np.array_equal(fixed_mode, np.asarray(evidence.rtk_mode_fix, dtype=bool)):
        raise RuntimeError("receiver-state fix_mode and rtk_mode_fix disagree")
    fix_header = np.asarray(track.fix_header_ns, dtype=np.int64)
    matches = nearest_matches(
        fix_header, evidence.header_ns, tolerance_ns=int(tolerance_ns)
    )
    if required and matches.match_count != len(fix_header):
        first = int(matches.unmatched_reference_indices[0])
        raise RuntimeError(
            "required receiver-state association exceeds "
            f"{tolerance_ns} ns at NavSatFix sample {first}"
        )

    count = len(fix_header)
    source_index = np.full(count, -1, dtype=np.int64)
    source_header = np.full(count, -1, dtype=np.int64)
    source_log = np.full(count, -1, dtype=np.int64)
    residual = np.zeros(count, dtype=np.int64)
    matched = np.zeros(count, dtype=bool)
    mode = np.full(count, "UNAVAILABLE", dtype="<U32")
    rtk_mode_fix = np.zeros(count, dtype=bool)
    num_sat = np.full(count, -1, dtype=np.int16)
    system_error = np.full(count, -1, dtype=np.int16)
    io_error = np.full(count, -1, dtype=np.int16)
    swift_nap_error = np.full(count, -1, dtype=np.int16)
    external_antenna = np.full(count, -1, dtype=np.int16)
    carrier = np.full(count, -1, dtype=np.int8)

    references = matches.reference_indices
    sources = matches.sample_indices
    source_index[references] = sources
    source_header[references] = evidence.header_ns[sources]
    source_log[references] = evidence.log_ns[sources]
    residual[references] = matches.residual_ns
    matched[references] = True
    mode[references] = evidence.fix_mode[sources]
    rtk_mode_fix[references] = evidence.rtk_mode_fix[sources]
    num_sat[references] = evidence.num_sat[sources]
    system_error[references] = evidence.system_error[sources]
    io_error[references] = evidence.io_error[sources]
    swift_nap_error[references] = evidence.swift_nap_error[sources]
    external_antenna[references] = evidence.external_antenna_present[sources]

    carrier[mode == "DGNSS"] = 0
    carrier[mode == "SBAS"] = 0
    carrier[mode == "SPP"] = 0
    carrier[mode == "FLOAT_RTK"] = 1
    carrier[mode == "FIXED_RTK"] = 2
    receiver_error = (system_error > 0) | (io_error > 0) | (swift_nap_error > 0)
    carrier[receiver_error] = -1

    finite = np.isfinite(np.asarray(track.enu_xyz, dtype=np.float64)).all(axis=1)
    fallback_valid, fallback_quality = normalize_navsat_position_quality(
        np.asarray(track.fix_status, dtype=np.int16), carrier, finite
    )
    position_valid = fallback_valid.copy()
    position_quality = fallback_quality.astype("<U13", copy=True)
    state_valid = matched & ~receiver_error & (
        np.asarray(track.fix_status, dtype=np.int16) >= 0
    ) & finite
    recognized_valid = np.isin(
        mode, ["SPP", "DGNSS", "SBAS", "FLOAT_RTK", "FIXED_RTK"]
    )
    position_valid[matched] = state_valid[matched] & recognized_valid[matched]
    position_quality[matched] = "invalid"
    position_quality[state_valid & (mode == "SPP")] = "standalone"
    position_quality[state_valid & np.isin(mode, ["DGNSS", "SBAS"])] = (
        "differential"
    )
    position_quality[state_valid & (mode == "FLOAT_RTK")] = "rtk_float"
    position_quality[state_valid & (mode == "FIXED_RTK")] = "rtk_fixed"

    track.fix_carrier_status = carrier
    track.fix_position_valid = position_valid
    track.fix_position_quality = position_quality
    track.fix_receiver_state_index = source_index
    track.fix_receiver_state_matched = matched
    track.fix_receiver_state_header_ns = source_header
    track.fix_receiver_state_log_ns = source_log
    track.fix_receiver_state_residual_ns = residual
    track.fix_receiver_mode = mode
    track.fix_receiver_rtk_mode_fix = rtk_mode_fix
    track.fix_receiver_num_sat = num_sat
    track.fix_receiver_system_error = system_error
    track.fix_receiver_io_error = io_error
    track.fix_receiver_swift_nap_error = swift_nap_error
    track.fix_receiver_external_antenna_present = external_antenna
    track.receiver_state_evidence = evidence

    result = evidence.summary()
    result.update(
        {
            "required": bool(required),
            "association_method": matches.method,
            "association_tolerance_ns": int(matches.tolerance_ns),
            "associated_navsatfix_count": int(matches.match_count),
            "navsatfix_count": int(matches.reference_count),
            "association_fraction": float(matches.match_fraction),
            "association_residual": matches.residual_summary(),
            "associated_position_quality_counts": {
                str(value): int(number)
                for value, number in zip(
                    *np.unique(position_quality.astype(str), return_counts=True)
                )
            },
        }
    )
    return result


def derive_gnss_course(track: RtkTrack, pose_cfg: Any) -> RtkTrack:
    """Populate the common heading surface from a single-antenna trajectory."""
    window = max(3, int(pose_cfg.yaw_smooth_window))
    kernel = np.ones(window, dtype=np.float64) / window
    padding = (window // 2, window - 1 - window // 2)
    east = np.convolve(
        np.pad(track.enu_xyz[:, 0], padding, mode="edge"), kernel, mode="valid"
    )
    north = np.convolve(
        np.pad(track.enu_xyz[:, 1], padding, mode="edge"), kernel, mode="valid"
    )
    east_rate = np.gradient(east, track.fix_t)
    north_rate = np.gradient(north, track.fix_t)
    speed = np.hypot(east_rate, north_rate)
    minimum_speed = float(getattr(pose_cfg, "min_course_speed_ms", 0.1))
    if not math.isfinite(minimum_speed) or minimum_speed <= 0:
        raise ValueError("pose.min_course_speed_ms must be positive")
    track.relpos_t = track.fix_t.copy()
    track.relpos_yaw = np.unwrap(np.arctan2(north_rate, east_rate))
    track.relpos_carr = np.full(len(speed), -1, dtype=np.int8)
    track.heading_valid = speed >= minimum_speed
    track.heading_quality_kind = "course"
    track.relpos_header_ns = np.asarray(track.fix_header_ns, dtype=np.int64).copy()
    track.relpos_log_ns = np.asarray(track.fix_log_ns, dtype=np.int64).copy()
    track.relpos_ned_m = np.full((len(speed), 3), np.nan)
    track.relpos_acc_heading_rad = np.full(len(speed), np.nan)
    track.relpos_flags = np.zeros((len(speed), len(RELPOS_FLAG_NAMES)), dtype=bool)
    track.course_speed_ms = speed
    return track


def resolve_relative_window(
    track: RtkTrack, window_s: Sequence[float], epoch_source: str
) -> tuple[int, int, int]:
    if epoch_source != "first_gnss_log":
        raise ValueError(
            "segment.window_epoch_source must explicitly be 'first_gnss_log'"
        )
    if len(window_s) != 2:
        raise ValueError("segment.window_s must contain [start_s, stop_s]")
    start_s, stop_s = (float(value) for value in window_s)
    if not np.isfinite([start_s, stop_s]).all() or start_s < 0 or stop_s <= start_s:
        raise ValueError("segment.window_s must be finite, non-negative and increasing")
    epoch_ns = int(np.asarray(track.fix_log_ns, dtype=np.int64)[0])
    start_ns = epoch_ns + int(round(start_s * NANOSECONDS))
    stop_ns = epoch_ns + int(round(stop_s * NANOSECONDS))
    fix_log = np.asarray(track.fix_log_ns, dtype=np.int64)
    if start_ns < int(fix_log[0]) or stop_ns > int(fix_log[-1]):
        raise RuntimeError("relative window falls outside the GNSS log-time coverage")
    return epoch_ns, start_ns, stop_ns


def _bag_for_log_time(report: BagChainReport, timestamp_ns: int) -> Path:
    candidates = [
        chunk
        for chunk in report.chunks
        if chunk.start_log_ns <= timestamp_ns < chunk.stop_log_ns
    ]
    if not candidates:
        candidates = sorted(
            report.chunks,
            key=lambda chunk: min(
                abs(timestamp_ns - chunk.start_log_ns),
                abs(timestamp_ns - chunk.stop_log_ns),
            ),
        )[:1]
    return Path(candidates[0].path)


def sample_header_log_offsets(
    report: BagChainReport,
    topic: str,
    typestore: Any,
    start_log_ns: int,
    stop_log_ns: int,
    *,
    samples_per_probe: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample beginning/middle/end offsets without scanning the full window."""
    Reader = _reader_type()
    probes = np.linspace(start_log_ns, stop_log_ns, 3, dtype=np.int64)
    log_values: list[int] = []
    offsets: list[int] = []
    for probe in probes:
        bag = _bag_for_log_time(report, int(probe))
        with Reader(bag) as reader:
            connections = [c for c in reader.connections if c.topic == topic]
            if not connections:
                raise RuntimeError(f"{bag} lacks timing probe topic {topic}")
            count = 0
            for connection, log_ns, raw in reader.messages(
                connections=connections,
                start=max(int(probe) - NANOSECONDS, int(reader.start_time)),
                stop=min(int(probe) + 2 * NANOSECONDS, int(reader.end_time)),
            ):
                message = _deserialize(typestore, raw, connection.msgtype)
                log_values.append(int(log_ns))
                offsets.append(_stamp_ns(message) - int(log_ns))
                count += 1
                if count >= samples_per_probe:
                    break
            if count == 0:
                raise RuntimeError(f"no {topic} timing samples near log time {probe}")
    order = np.argsort(np.asarray(log_values, dtype=np.int64), kind="stable")
    return (
        np.asarray(log_values, dtype=np.int64)[order],
        np.asarray(offsets, dtype=np.int64)[order],
    )


def estimate_clock_offset(
    track: RtkTrack,
    camera_log_ns: np.ndarray,
    camera_header_minus_log_ns: np.ndarray,
    *,
    window_start_log_ns: int,
    window_stop_log_ns: int,
    configured_offset_ns: int,
    maximum_error_ns: int,
    maximum_drift_ns: int,
) -> ClockEstimate:
    """Estimate camera-header -> GNSS-header from their shared ROS log clock."""
    fix_log = np.asarray(track.fix_log_ns, dtype=np.int64)
    fix_offsets = np.asarray(track.fix_header_ns, dtype=np.int64) - fix_log
    in_window = (fix_log >= window_start_log_ns) & (fix_log <= window_stop_log_ns)
    if np.count_nonzero(in_window) < 3 or len(camera_log_ns) < 3:
        raise RuntimeError("insufficient clock samples in selected window")
    gnss_offset = int(np.rint(np.median(fix_offsets[in_window])))
    camera_offset = int(np.rint(np.median(camera_header_minus_log_ns)))
    estimate = gnss_offset - camera_offset

    interpolation_origin = int(fix_log[0])
    gnss_at_camera = np.interp(
        (camera_log_ns - interpolation_origin).astype(np.float64),
        (fix_log - interpolation_origin).astype(np.float64),
        fix_offsets.astype(np.float64),
    )
    instantaneous = gnss_at_camera - camera_header_minus_log_ns
    midpoint = (window_start_log_ns + window_stop_log_ns) / 2.0
    early = instantaneous[camera_log_ns <= midpoint]
    late = instantaneous[camera_log_ns > midpoint]
    if not len(early) or not len(late):
        raise RuntimeError("clock probes do not span the selected window")
    start_offset = int(np.rint(np.median(early)))
    stop_offset = int(np.rint(np.median(late)))
    drift = stop_offset - start_offset
    error = int(configured_offset_ns) - estimate
    if abs(error) > maximum_error_ns:
        raise RuntimeError(
            "configured camera-to-GNSS clock offset disagrees with ROS-log "
            f"estimate by {error} ns (limit {maximum_error_ns} ns)"
        )
    if abs(drift) > maximum_drift_ns:
        raise RuntimeError(
            f"camera/GNSS clock drift {drift} ns exceeds {maximum_drift_ns} ns"
        )
    return ClockEstimate(
        camera_to_gnss_header_offset_ns=estimate,
        configured_offset_ns=int(configured_offset_ns),
        configured_error_ns=error,
        gnss_header_minus_log_median_ns=gnss_offset,
        camera_header_minus_log_median_ns=camera_offset,
        camera_to_gnss_start_ns=start_offset,
        camera_to_gnss_stop_ns=stop_offset,
        drift_ns=drift,
        sample_count=len(camera_log_ns),
    )


def _camera_info(message: Any) -> dict[str, Any]:
    return {
        "frame_id": str(message.header.frame_id),
        "width": int(message.width),
        "height": int(message.height),
        "k": np.asarray(_field(message, "k", "K"), dtype=float)
        .reshape(3, 3)
        .tolist(),
        "d": np.asarray(_field(message, "d", "D"), dtype=float).tolist(),
        "r": np.asarray(_field(message, "r", "R"), dtype=float)
        .reshape(3, 3)
        .tolist(),
        "p": np.asarray(_field(message, "p", "P"), dtype=float)
        .reshape(3, 4)
        .tolist(),
        "distortion_model": str(message.distortion_model),
    }


def read_camera_calibration(
    bags: Sequence[Path], topic: str, typestore: Any
) -> dict[str, Any]:
    Reader = _reader_type()
    for bag in bags:
        with Reader(bag) as reader:
            connections = [c for c in reader.connections if c.topic == topic]
            for connection, _, raw in reader.messages(connections=connections):
                return _camera_info(_deserialize(typestore, raw, connection.msgtype))
    raise RuntimeError(f"no CameraInfo messages on {topic}")


def _rotation_deviation_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((float(np.trace(rotation)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _validated_rotation(value: Any, label: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise RuntimeError(f"{label} CameraInfo.R is not a valid rotation")
    return rotation


def _validate_rectified_stereo(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    left_rotation = _validated_rotation(left["r"], "left")
    right_rotation = _validated_rotation(right["r"], "right")
    if (int(left["width"]), int(left["height"])) != (
        int(right["width"]),
        int(right["height"]),
    ):
        raise RuntimeError("left/right CameraInfo dimensions differ")
    left_projection = np.asarray(left["p"], dtype=np.float64)
    right_projection = np.asarray(right["p"], dtype=np.float64)
    if (
        left_projection.shape != (3, 4)
        or right_projection.shape != (3, 4)
        or not np.isfinite(left_projection).all()
        or not np.isfinite(right_projection).all()
        or not np.allclose(
            left_projection[:, :3], right_projection[:, :3], rtol=1e-6, atol=1e-6
        )
        or not np.allclose(left_projection[:, 3], 0.0, atol=1e-7)
        or not np.allclose(right_projection[1:, 3], 0.0, atol=1e-7)
    ):
        raise RuntimeError(
            "CameraInfo.P does not describe a standard horizontal rectified rig"
        )
    fx = float(right_projection[0, 0])
    baseline = abs(float(right_projection[0, 3] / fx))
    if not math.isfinite(baseline) or baseline <= 0:
        raise RuntimeError("CameraInfo does not contain a valid stereo baseline")
    return {
        "baseline_m": baseline,
        "left_rotation": left_rotation,
        "right_rotation": right_rotation,
    }


def _resolved_camera_extrinsic(
    geometry: Any, left_camera_info: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve an explicitly declared raw/rectified camera transform target."""
    source_geometry = getattr(geometry, "extrinsic_camera_geometry", None)
    if source_geometry not in {"raw_left", "rectified_left"}:
        raise ValueError(
            "sensor_geometry.extrinsic_camera_geometry must explicitly be "
            "'raw_left' or 'rectified_left'"
        )
    transform = np.asarray(geometry.T_camera_primary_antenna, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("T_camera_primary_antenna must be 4x4")
    rectification = _validated_rotation(left_camera_info["r"], "left")
    composed = source_geometry == "raw_left"
    if composed:
        raw_to_rectified = np.eye(4, dtype=np.float64)
        raw_to_rectified[:3, :3] = rectification
        transform = raw_to_rectified @ transform
    return transform, {
        "configured_geometry": source_geometry,
        "published_geometry": "rectified_left",
        "raw_to_rectified_composed": composed,
        "left_rectification_rotation": rectification.tolist(),
        "left_rectification_deviation_deg": _rotation_deviation_deg(rectification),
    }


def _encode_image(message: Any, destination: Path, encoding: str) -> None:
    source_encoding = str(message.encoding).lower()
    image = decode_raw_image(
        message.data,
        width=int(message.width),
        height=int(message.height),
        encoding=source_encoding,
        step=int(message.step),
        is_bigendian=int(message.is_bigendian),
    )
    if source_encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif source_encoding == "rgba8":
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif source_encoding == "bgra8":
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding != "png_lossless":
        raise ValueError("segment.image_encoding must be 'png_lossless'")
    ok, payload = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("OpenCV failed to encode a lossless PNG")
    destination.write_bytes(payload.tobytes())


def _interpolate_enu(track: RtkTrack, timestamp_ns: int) -> np.ndarray | None:
    times = np.asarray(track.fix_header_ns, dtype=np.int64)
    if timestamp_ns < int(times[0]) or timestamp_ns > int(times[-1]):
        return None
    origin = int(times[0])
    relative_times = (times - origin).astype(np.float64)
    query = float(timestamp_ns - origin)
    value = np.array(
        [
            np.interp(query, relative_times, track.enu_xyz[:, axis])
            for axis in range(3)
        ],
        dtype=np.float64,
    )
    return value if np.isfinite(value).all() else None


def _drain_pairs(
    left: deque[_TimedImage], right: deque[_TimedImage], tolerance_ns: int
):
    while left and right:
        residual = right[0].header_ns - left[0].header_ns
        if residual < -tolerance_ns:
            right.popleft()
        elif residual > tolerance_ns:
            left.popleft()
        else:
            yield left.popleft(), right.popleft()


def stage_metric_stereo_frames(
    bags: Sequence[Path],
    topics: Any,
    typestore: Any,
    track: RtkTrack,
    *,
    start_log_ns: int,
    stop_log_ns: int,
    camera_to_gnss_offset_ns: int,
    spacing_m: float,
    stereo_tolerance_ns: int,
    output_dir: Path,
    image_encoding: str,
) -> tuple[list[FrameRecord], dict[str, Any]]:
    """Stream chained raw stereo, encode only metric-selected pairs to disk."""
    Reader = _reader_type()
    sampler = _DistanceSampler(spacing_m)
    pending_left: deque[_TimedImage] = deque()
    pending_right: deque[_TimedImage] = deque()
    frames: list[FrameRecord] = []
    last_header = {topics.left_image: None, topics.right_image: None}
    header_log_offsets: list[int] = []
    left_source_encodings: set[str] = set()
    right_source_encodings: set[str] = set()
    candidate_pairs = 0
    for bag in bags:
        with Reader(bag) as reader:
            connections = [
                connection
                for connection in reader.connections
                if connection.topic in {topics.left_image, topics.right_image}
            ]
            local_start = max(start_log_ns - NANOSECONDS, int(reader.start_time))
            local_stop = min(stop_log_ns + NANOSECONDS, int(reader.end_time))
            if local_stop <= local_start:
                continue
            for connection, log_ns, raw in reader.messages(
                connections=connections, start=local_start, stop=local_stop
            ):
                message = _deserialize(typestore, raw, connection.msgtype)
                header_ns = _stamp_ns(message)
                previous = last_header[connection.topic]
                if previous is not None and header_ns <= previous:
                    raise RuntimeError(
                        f"{connection.topic} overlaps or duplicates across chunks"
                    )
                last_header[connection.topic] = header_ns
                timed = _TimedImage(header_ns, int(log_ns), message)
                if connection.topic == topics.left_image:
                    pending_left.append(timed)
                else:
                    pending_right.append(timed)
                for left, right in _drain_pairs(
                    pending_left, pending_right, stereo_tolerance_ns
                ):
                    if not start_log_ns <= left.log_ns <= stop_log_ns:
                        continue
                    candidate_pairs += 1
                    header_log_offsets.append(left.header_ns - left.log_ns)
                    query_ns = left.header_ns + camera_to_gnss_offset_ns
                    position = _interpolate_enu(track, query_ns)
                    if position is None or not sampler.accept(position):
                        continue
                    index = len(frames)
                    left_source_encodings.add(str(left.message.encoding))
                    right_source_encodings.add(str(right.message.encoding))
                    left_path = output_dir / f"left_{index:06d}.png"
                    right_path = output_dir / f"right_{index:06d}.png"
                    _encode_image(left.message, left_path, image_encoding)
                    _encode_image(right.message, right_path, image_encoding)
                    record = FrameRecord(
                        t=left.header_ns * 1.0e-9,
                        t_right=right.header_ns * 1.0e-9,
                        left_jpeg=left_path,
                        right_jpeg=right_path,
                        left_header_ns=left.header_ns,
                        right_header_ns=right.header_ns,
                    )
                    record.left_log_ns = left.log_ns
                    record.right_log_ns = right.log_ns
                    frames.append(record)
    if not frames:
        raise RuntimeError("selected log-time window produced no stereo frames")
    if candidate_pairs < len(frames):
        raise RuntimeError("internal metric sampling count is inconsistent")
    offsets = np.asarray(header_log_offsets, dtype=np.int64)
    return frames, {
        "candidate_pairs": candidate_pairs,
        "selected_pairs": len(frames),
        "path_length_m": sampler.path_length_m,
        "requested_spacing_m": spacing_m,
        "effective_spacing_m": (
            sampler.path_length_m / max(1, len(frames) - 1)
        ),
        "camera_header_minus_log_median_ns": int(np.rint(np.median(offsets))),
        "camera_header_minus_log_min_ns": int(np.min(offsets)),
        "camera_header_minus_log_max_ns": int(np.max(offsets)),
        "left_source_encodings": sorted(left_source_encodings),
        "right_source_encodings": sorted(right_source_encodings),
    }


def _integer_ns(seconds: float, label: str) -> int:
    value = float(seconds) * NANOSECONDS
    rounded = int(round(value))
    if not math.isfinite(value) or not np.isclose(value, rounded, atol=1.0e-6):
        raise ValueError(f"{label} must be exactly representable in nanoseconds")
    return rounded


def _timing_options(cfg: Any) -> tuple[int, int, int]:
    if not hasattr(cfg.pose, "time_offset_s"):
        raise ValueError(
            "pose.time_offset_s is required (camera header -> GNSS header)"
        )
    offset = _integer_ns(cfg.pose.time_offset_s, "pose.time_offset_s")
    timing = getattr(cfg, "timing", None)
    if timing is None or getattr(timing, "clock_model", None) != "ros_log_constant":
        raise ValueError("timing.clock_model must explicitly be 'ros_log_constant'")
    maximum_error = _integer_ns(
        getattr(timing, "maximum_offset_error_s", 0.015),
        "timing.maximum_offset_error_s",
    )
    maximum_drift = _integer_ns(
        getattr(timing, "maximum_window_drift_s", 0.015),
        "timing.maximum_window_drift_s",
    )
    if maximum_error < 0 or maximum_drift < 0:
        raise ValueError("timing tolerances must be non-negative")
    return offset, maximum_error, maximum_drift


def _chain_options(cfg: Any) -> tuple[int, int]:
    segment = cfg.segment
    maximum_gap = _integer_ns(
        getattr(segment, "maximum_bag_gap_s", 0.1),
        "segment.maximum_bag_gap_s",
    )
    maximum_overlap = _integer_ns(
        getattr(segment, "maximum_bag_overlap_s", 0.01),
        "segment.maximum_bag_overlap_s",
    )
    return maximum_gap, maximum_overlap


def _gnss_quality_options(cfg: Any) -> tuple[bool, int, str, bool | None]:
    quality = getattr(cfg, "gnss_quality", None)
    required_value = (
        getattr(quality, "receiver_state_required", False)
        if quality is not None
        else False
    )
    if type(required_value) is not bool:
        raise ValueError("gnss_quality.receiver_state_required must be boolean")
    required = required_value
    tolerance = _integer_ns(
        getattr(quality, "receiver_state_association_tolerance_s", 0.15)
        if quality is not None
        else 0.15,
        "gnss_quality.receiver_state_association_tolerance_s",
    )
    if tolerance < 0:
        raise ValueError(
            "gnss_quality.receiver_state_association_tolerance_s must be non-negative"
        )
    provenance = str(
        getattr(quality, "covariance_provenance", "unspecified")
        if quality is not None
        else "unspecified"
    )
    allowed = {
        "unspecified",
        "receiver_reported_per_epoch",
        "driver_static_nominal",
    }
    if provenance not in allowed:
        raise ValueError(
            "gnss_quality.covariance_provenance must be one of: "
            + ", ".join(sorted(allowed))
        )
    live_value = (
        getattr(quality, "covariance_is_live_per_epoch", None)
        if quality is not None
        else None
    )
    if live_value is not None and type(live_value) is not bool:
        raise ValueError(
            "gnss_quality.covariance_is_live_per_epoch must be boolean or omitted"
        )
    live = live_value
    if provenance == "driver_static_nominal" and live is not False:
        raise ValueError(
            "driver_static_nominal covariance must declare "
            "gnss_quality.covariance_is_live_per_epoch: false"
        )
    if provenance == "receiver_reported_per_epoch" and live is not True:
        raise ValueError(
            "receiver_reported_per_epoch covariance must declare "
            "gnss_quality.covariance_is_live_per_epoch: true"
        )
    return required, tolerance, provenance, live


_NAVSAT_COVARIANCE_TYPE = {
    0: "UNKNOWN",
    1: "APPROXIMATED",
    2: "DIAGONAL_KNOWN",
    3: "KNOWN",
}


def _covariance_provenance_report(
    track: RtkTrack, configured: str, live_per_epoch: bool | None
) -> dict[str, Any]:
    covariance_type = np.asarray(track.fix_covariance_type, dtype=np.int16)
    codes, counts = np.unique(covariance_type, return_counts=True)
    covariance = np.asarray(track.fix_covariance_enu_m2, dtype=np.float64)
    static = bool(
        len(covariance) > 0
        and np.allclose(covariance, covariance[0], rtol=0.0, atol=1.0e-15)
    )
    return {
        "configured_provenance": configured,
        "is_live_per_epoch": live_per_epoch,
        "observed_covariance_type_counts": {
            _NAVSAT_COVARIANCE_TYPE.get(int(code), f"UNKNOWN_CODE_{int(code)}"): int(
                count
            )
            for code, count in zip(codes, counts)
        },
        "observed_static_across_stream": static,
        "interpretation": (
            "driver-configured static nominal covariance; not measured "
            "per-epoch receiver accuracy"
            if configured == "driver_static_nominal"
            else "receiver-reported per-epoch covariance"
            if configured == "receiver_reported_per_epoch"
            else "covariance source not declared"
        ),
    }


def _prepare_config(
    cfg: Any,
) -> tuple[dict[str, Any], RtkTrack, list[Path], list[Path], Any]:
    """Prepare one audited ingest without reading the GNSS stream twice."""
    if getattr(cfg, "adapter", None) != "ros1_citrusfarm":
        raise ValueError("preflight_config requires adapter: ros1_citrusfarm")
    if getattr(cfg.pose, "source", None) != "gnss_course":
        raise ValueError("ROS1 single-antenna ingestion requires pose.source=gnss_course")
    camera_bags = _configured_paths(cfg, "camera_bags")
    gnss_bags = _configured_paths(cfg, "gnss_bags")
    maximum_gap, maximum_overlap = _chain_options(cfg)
    (
        receiver_state_required,
        receiver_state_tolerance_ns,
        covariance_provenance,
        covariance_is_live,
    ) = _gnss_quality_options(cfg)
    receiver_state_topic = getattr(cfg.topics, "receiver_state", None)
    if receiver_state_required and not receiver_state_topic:
        raise ValueError(
            "topics.receiver_state is required when "
            "gnss_quality.receiver_state_required is true"
        )
    camera_report = validate_bag_chain(
        camera_bags,
        required_each={cfg.topics.left_image, cfg.topics.right_image},
        expected_count=int(getattr(cfg.segment, "expected_camera_bag_count", len(camera_bags))),
        maximum_gap_ns=maximum_gap,
        maximum_overlap_ns=maximum_overlap,
    )
    required_gnss_topics = {cfg.topics.fix}
    if receiver_state_required:
        required_gnss_topics.add(str(receiver_state_topic))
    gnss_report = validate_bag_chain(
        gnss_bags,
        required_each=required_gnss_topics,
        expected_count=int(getattr(cfg.segment, "expected_gnss_bag_count", len(gnss_bags))),
        maximum_gap_ns=maximum_gap,
        maximum_overlap_ns=maximum_overlap,
    )
    typestore = build_typestore()
    track = read_navsat_track(gnss_bags, cfg.topics.fix, typestore)
    receiver_state_report: dict[str, Any] | None = None
    if receiver_state_topic:
        try:
            receiver_state = read_receiver_state(
                gnss_bags, str(receiver_state_topic), typestore
            )
        except RuntimeError as exc:
            if receiver_state_required or not str(exc).startswith(
                "no receiver-state messages"
            ):
                raise
        else:
            receiver_state_report = associate_receiver_state(
                track,
                receiver_state,
                tolerance_ns=receiver_state_tolerance_ns,
                required=receiver_state_required,
            )
    if receiver_state_required and receiver_state_report is None:
        raise RuntimeError("required receiver-state evidence is unavailable")
    track = derive_gnss_course(track, cfg.pose)
    epoch_ns, start_ns, stop_ns = resolve_relative_window(
        track, cfg.segment.window_s, cfg.segment.window_epoch_source
    )
    if start_ns < camera_report.start_log_ns or stop_ns > camera_report.stop_log_ns:
        raise RuntimeError("selected window falls outside camera bag coverage")
    active_camera_bags = [
        Path(chunk.path)
        for chunk in camera_report.chunks
        if chunk.stop_log_ns > start_ns - NANOSECONDS
        and chunk.start_log_ns < stop_ns + NANOSECONDS
    ]
    camera_log, camera_offsets = sample_header_log_offsets(
        camera_report,
        cfg.topics.left_image,
        typestore,
        start_ns,
        stop_ns,
    )
    configured, maximum_error, maximum_drift = _timing_options(cfg)
    clock = estimate_clock_offset(
        track,
        camera_log,
        camera_offsets,
        window_start_log_ns=start_ns,
        window_stop_log_ns=stop_ns,
        configured_offset_ns=configured,
        maximum_error_ns=maximum_error,
        maximum_drift_ns=maximum_drift,
    )
    left_info = read_camera_calibration(
        active_camera_bags, cfg.topics.left_info, typestore
    )
    right_info = read_camera_calibration(
        active_camera_bags, cfg.topics.right_info, typestore
    )
    rig = _validate_rectified_stereo(left_info, right_info)
    configured_camera_frame = str(cfg.sensor_geometry.frames.camera)
    recorded_camera_frame = str(left_info["frame_id"])
    if configured_camera_frame != recorded_camera_frame:
        raise RuntimeError(
            "sensor_geometry.frames.camera does not match left CameraInfo.frame_id: "
            f"{configured_camera_frame!r} != {recorded_camera_frame!r}"
        )
    _, extrinsic_resolution = _resolved_camera_extrinsic(
        cfg.sensor_geometry, left_info
    )
    spacing = float(cfg.segment.frame_spacing_m)
    fix_log = np.asarray(track.fix_log_ns, dtype=np.int64)
    mask = (fix_log >= start_ns) & (fix_log <= stop_ns)
    trajectory = track.enu_xyz[mask]
    if len(trajectory) < 2:
        raise RuntimeError("selected window contains too few GNSS fixes")
    path_length = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
    estimated_frames = int(math.floor(path_length / spacing)) + 1
    available_topics = set().union(*(set(chunk.topics) for chunk in camera_report.chunks))
    depth_topic = getattr(cfg.topics, "depth", None)
    confidence_topic = getattr(cfg.topics, "confidence", None)
    report = {
        "passed": True,
        "adapter": "ros1_citrusfarm",
        "camera_chain": camera_report.as_json(),
        "gnss_chain": gnss_report.as_json(),
        "window": {
            "epoch_source": "first_gnss_log",
            "epoch_ns": epoch_ns,
            "start_log_ns": start_ns,
            "stop_log_ns": stop_ns,
            "duration_s": (stop_ns - start_ns) * 1.0e-9,
        },
        "clock": clock.as_json(),
        "gnss_quality": {
            "position_classification_source": (
                "receiver_state"
                if receiver_state_report is not None
                else "navsat_status_only"
            ),
            "receiver_state": receiver_state_report,
            "covariance": _covariance_provenance_report(
                track, covariance_provenance, covariance_is_live
            ),
        },
        "stereo": {
            "baseline_m": rig["baseline_m"],
            "width": int(left_info["width"]),
            "height": int(left_info["height"]),
            "left_frame_id": left_info["frame_id"],
            "right_frame_id": right_info["frame_id"],
            "left_model": left_info["distortion_model"],
            "right_model": right_info["distortion_model"],
            "left_rectification_rotation": left_info["r"],
            "right_rectification_rotation": right_info["r"],
            "left_rectification_deviation_deg": extrinsic_resolution[
                "left_rectification_deviation_deg"
            ],
            "right_rectification_deviation_deg": _rotation_deviation_deg(
                rig["right_rotation"]
            ),
        },
        "extrinsic_resolution": extrinsic_resolution,
        "sampling": {
            "frame_spacing_m": spacing,
            "gnss_path_length_m": path_length,
            "estimated_frame_count": estimated_frames,
            "note": "estimate from GNSS arc length; ingest reports exact stereo count",
        },
        "optional_recorded_inputs": {
            "depth_topic": depth_topic,
            "depth_available": bool(depth_topic and depth_topic in available_topics),
            "confidence_topic": confidence_topic,
            "confidence_available": bool(
                confidence_topic and confidence_topic in available_topics
            ),
            "used_by_primary_ingest": False,
        },
    }
    return report, track, camera_bags, gnss_bags, typestore


def preflight_config(cfg: Any) -> dict[str, Any]:
    """Perform index, timing, calibration and selection-size checks only."""
    report, _, _, _, _ = _prepare_config(cfg)
    return report


def _overlapping_chunk_paths(
    report: Mapping[str, Any], start_ns: int, stop_ns: int, *, padding_ns: int
) -> list[Path]:
    paths = [
        Path(chunk["path"])
        for chunk in report["chunks"]
        if int(chunk["stop_log_ns"]) > start_ns - padding_ns
        and int(chunk["start_log_ns"]) < stop_ns + padding_ns
    ]
    if not paths:
        raise RuntimeError("no camera chunks overlap the selected window")
    return paths


def ingest_config_v2(
    cfg: Any,
    destination: str | Path,
    *,
    window: Mapping[str, Any] | None = None,
):
    """Publish one immutable contract-v2 segment from chained ROS1 bags."""
    if window is not None:
        raise ValueError(
            "ros1_citrusfarm resolves its explicit relative log-time window; "
            "external window artifacts are not accepted"
        )
    report, track, _, _, typestore = _prepare_config(cfg)
    start_log_ns = int(report["window"]["start_log_ns"])
    stop_log_ns = int(report["window"]["stop_log_ns"])
    camera_bags = _overlapping_chunk_paths(
        report["camera_chain"],
        start_log_ns,
        stop_log_ns,
        padding_ns=NANOSECONDS,
    )
    clock_offset_ns = int(report["clock"]["configured_offset_ns"])
    stereo_tolerance_ns = _integer_ns(
        cfg.segment.stereo_tolerance_s, "segment.stereo_tolerance_s"
    )
    spacing_m = float(cfg.segment.frame_spacing_m)
    image_encoding = str(getattr(cfg.segment, "image_encoding", "png_lossless"))
    geometry = getattr(cfg, "sensor_geometry", None)
    if geometry is None:
        raise ValueError("sensor_geometry with explicit camera/GNSS extrinsic is required")
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination_path.parent, prefix=".ros1-images-"
    ) as temporary:
        frames, sampling = stage_metric_stereo_frames(
            camera_bags,
            cfg.topics,
            typestore,
            track,
            start_log_ns=start_log_ns,
            stop_log_ns=stop_log_ns,
            camera_to_gnss_offset_ns=clock_offset_ns,
            spacing_m=spacing_m,
            stereo_tolerance_ns=stereo_tolerance_ns,
            output_dir=Path(temporary),
            image_encoding=image_encoding,
        )
        pose_stamps = [
            (int(frame.left_header_ns) + clock_offset_ns) * 1.0e-9
            for frame in frames
        ]
        left_info = read_camera_calibration(
            camera_bags, cfg.topics.left_info, typestore
        )
        right_info = read_camera_calibration(
            camera_bags, cfg.topics.right_info, typestore
        )
        resolved_extrinsic, extrinsic_resolution = _resolved_camera_extrinsic(
            geometry, left_info
        )
        poses = pose_frames_from_extrinsic(
            track,
            pose_stamps,
            cfg.pose,
            resolved_extrinsic,
        )
        keep = [index for index, pose in enumerate(poses) if pose is not None]
        if not keep:
            raise RuntimeError("course/RTK quality gates reject every selected frame")
        selected_frames = [frames[index] for index in keep]
        selected_poses = [poses[index] for index in keep]
        count = len(selected_frames)
        validation_every = int(getattr(cfg.segment, "validation_every", 8))
        if validation_every <= 1:
            raise ValueError("segment.validation_every must be greater than one")
        validation = list(range(validation_every - 1, count, validation_every))
        validation_set = set(validation)
        splits = {
            "train": [i for i in range(count) if i not in validation_set],
            "val": validation,
            "test": [],
        }
        frames_cfg = geometry.frames
        enu = track.enu.crs()
        available = report["optional_recorded_inputs"]
        return publish_segment_v2(
            destination_path,
            selected_frames,
            selected_poses,
            track,
            left_info,
            right_info,
            camera_frame_id=str(frames_cfg.camera),
            primary_antenna_frame_id=str(frames_cfg.primary_antenna),
            secondary_antenna_frame_id=None,
            enu_definition={
                "origin_lat_deg": float(enu["origin_lat"]),
                "origin_lon_deg": float(enu["origin_lon"]),
                "origin_alt_ellipsoidal_m": float(enu["origin_alt_ellipsoidal"]),
                "ellipsoid": str(enu["ellipsoid"]),
                "vertical_datum": str(enu["vertical_datum"]),
                "world_frame_id": str(getattr(frames_cfg, "world", "map")),
            },
            T_camera_primary_antenna=resolved_extrinsic,
            extrinsic_translation_sigma_m=geometry.extrinsic_translation_sigma_m,
            extrinsic_provenance={
                **_plain_metadata(geometry.extrinsic_provenance),
                "rectification_resolution": extrinsic_resolution,
            },
            clock_offset_ns=clock_offset_ns,
            association_tolerance_ns=_integer_ns(
                cfg.segment.association_tolerance_s,
                "segment.association_tolerance_s",
            ),
            stereo_tolerance_ns=stereo_tolerance_ns,
            capabilities={"single_rtk": True, "dual_rtk": False},
            splits=splits,
            adapter_name="ros1_citrusfarm",
            provenance={
                "adapter": "ros1_citrusfarm",
                "window": report["window"],
                "clock": report["clock"],
                "camera_chain": report["camera_chain"],
                "gnss_chain": report["gnss_chain"],
                "sampling": sampling,
                "dropped_unposeable": len(frames) - count,
                "input_image_transport": "sensor_msgs/Image raw",
                "stored_image_encoding": image_encoding,
                "optional_recorded_inputs": available,
                "gnss_quality": report["gnss_quality"],
                "extrinsic_resolution": extrinsic_resolution,
                "recorded_depth_policy": (
                    "inventoried but withheld from the primary RGB+RTK method; "
                    "stereo depth is derived as a separate immutable segment"
                ),
            },
        )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "preflight":
        from rtk_splat.core.configio import load_config

        print(json.dumps(preflight_config(load_config(arguments.config)), indent=2))
        return 0
    raise AssertionError(arguments.command)


if __name__ == "__main__":  # pragma: no cover - exercised by shell preflight
    raise SystemExit(_main())
