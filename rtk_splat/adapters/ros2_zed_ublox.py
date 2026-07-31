"""Read images, RTK fixes and dual-antenna heading from a rosbag2 bag.

Uses the optional `rosbags` package (no ROS runtime). Humble sqlite3 bags do
not embed custom message definitions, so ublox_ubx_msgs types are registered
from their .msg sources (read-only). Importing this adapter does not require
rosbags; an actionable error is raised only when bag I/O is requested.

Legacy pose code still receives timestamps in seconds, while every observation
also retains its exact integer header and bag-log timestamps for contract-v2
evidence export.
"""

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rtk_splat.adapters.image_decode import detect_compressed_format
from rtk_splat.adapters.synchronization import (
    TimestampMatches,
    monotonic_matches,
    nearest_matches,
)
from rtk_splat.core.segment import (
    CAPABILITIES,
    CONTRACT_VERSION,
    POSITION_QUALITY_VOCABULARY,
    SegmentReader,
    SegmentWriter,
    normalize_navsat_position_quality,
)


@dataclass
class RtkTrack:
    """RTK fixes and dual-antenna evidence over a time window.

    The first nine fields are the legacy pose-estimation surface.  The
    remaining arrays preserve the unabridged evidence needed by contract-v2
    writers.  Defaults keep synthetic and trajectory-file construction
    backwards compatible without pretending unavailable values are measured.
    """

    fix_t: np.ndarray       # (N,) s
    fix_lat: np.ndarray     # (N,) deg
    fix_lon: np.ndarray     # (N,) deg
    fix_alt: np.ndarray     # (N,) m
    fix_status: np.ndarray  # (N,) NavSatFix status.status
    fix_cov_max: np.ndarray  # (N,) max diagonal of position covariance
    relpos_t: np.ndarray    # (M,) s
    relpos_yaw: np.ndarray  # (M,) rad, ENU yaw of moving_base->rover, unwrapped
    relpos_carr: np.ndarray  # (M,) int carrier solution status (2 = RTK fixed)
    fix_header_ns: np.ndarray | None = None       # (N,) exact sensor stamp
    fix_log_ns: np.ndarray | None = None          # (N,) exact bag log time
    fix_covariance_enu_m2: np.ndarray | None = None  # (N,3,3)
    fix_covariance_type: np.ndarray | None = None  # (N,)
    fix_carrier_status: np.ndarray | None = None   # (N,), -1 = unavailable
    relpos_header_ns: np.ndarray | None = None     # (M,) exact sensor stamp
    relpos_log_ns: np.ndarray | None = None        # (M,) exact bag log time
    relpos_ned_m: np.ndarray | None = None         # (M,3), North/East/Down
    relpos_acc_heading_rad: np.ndarray | None = None  # (M,)
    relpos_flags: np.ndarray | None = None         # (M,8), see names below
    heading_valid: np.ndarray | None = None         # (M,), source observability
    heading_quality_kind: str = "carrier"          # carrier | course | trajectory
    pvt_header_ns: np.ndarray | None = None         # (P,), optional NavPVT
    pvt_log_ns: np.ndarray | None = None            # (P,)
    pvt_carrier_status: np.ndarray | None = None    # (P,)
    fix_timestamps_exact: bool = field(init=False)
    relpos_timestamps_exact: bool = field(init=False)

    def __post_init__(self):
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
    """One selected stereo frame with exact stamps and legacy float seconds."""

    t: float
    t_right: float
    left_jpeg: bytes
    right_jpeg: bytes
    left_header_ns: int | None = None
    right_header_ns: int | None = None
    header_timestamps_exact: bool = field(init=False)

    def __post_init__(self):
        self.header_timestamps_exact = (
            self.left_header_ns is not None
            and self.right_header_ns is not None
        )
        if self.left_header_ns is None:
            self.left_header_ns = int(round(self.t * 1e9))
        if self.right_header_ns is None:
            self.right_header_ns = int(round(self.t_right * 1e9))


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


# Only the types this pipeline reads (plus their nested deps). Registering the
# whole ublox_ubx_msgs directory pulls in ESF messages whose binary-literal
# constants some rosbags versions cannot parse.
_UBLOX_TYPES = (
    "CarrSoln",
    "GpsFix",
    "PSMPVT",
    "UBXNavRelPosNED",
    "UBXNavPVT",
)


def _reader_type():
    try:
        from rosbags.rosbag2 import Reader
    except ImportError as exc:
        raise RuntimeError(
            "ROS2 bag ingestion requires the optional 'rosbags' package"
        ) from exc
    return Reader


def build_typestore(ublox_msgs_dir: Path | None):
    """Humble typestore; ublox types registered only when a directory is
    given (only the rtk_dual_antenna source needs them)."""
    try:
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError as exc:
        raise RuntimeError(
            "ROS2 bag ingestion requires the optional 'rosbags' package"
        ) from exc
    ts = get_typestore(Stores.ROS2_HUMBLE)
    if ublox_msgs_dir is not None:
        types = {}
        for stem in _UBLOX_TYPES:
            msg_file = ublox_msgs_dir / f"{stem}.msg"
            name = f"ublox_ubx_msgs/msg/{stem}"
            types.update(get_types_from_msg(msg_file.read_text(), name))
        ts.register(types)
    return ts


def _stamp_s(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _stamp_ns(msg) -> int:
    return (
        int(msg.header.stamp.sec) * 1_000_000_000
        + int(msg.header.stamp.nanosec)
    )


def bag_for_topic(bags: list[Path], topic: str) -> Path:
    """First bag that contains `topic` (topics may be split across bags)."""
    Reader = _reader_type()
    for bag in bags:
        with Reader(bag) as reader:
            if any(c.topic == topic for c in reader.connections):
                return bag
    raise RuntimeError(f"topic {topic} not found in any of: "
                       + ", ".join(str(b) for b in bags))


def bags_for_topics(bags: list[Path], topics: set[str]) -> list[Path]:
    """Return input-order bag chunks containing every requested topic."""
    Reader = _reader_type()
    matches = []
    for bag in bags:
        with Reader(bag) as reader:
            available = {connection.topic for connection in reader.connections}
        if topics <= available:
            matches.append(bag)
    if not matches:
        raise RuntimeError(
            "no bag contains all topics "
            + ", ".join(sorted(topics))
            + " in "
            + ", ".join(str(path) for path in bags)
        )
    return matches


def read_rtk_track(bags: list[Path], topics, typestore,
                   need_relpos: bool = True) -> RtkTrack:
    """Linear pass(es) over the small RTK topics, resolved across bags."""
    Reader = _reader_type()
    fix_header_ns, fix_log_ns = [], []
    fix_lat, fix_lon, fix_alt = [], [], []
    fix_status, fix_cov_max = [], []
    fix_covariance, fix_covariance_type = [], []
    relpos_header_ns, relpos_log_ns = [], []
    relpos_yaw, relpos_carr = [], []
    relpos_ned_m, relpos_acc_heading_rad, relpos_flags = [], [], []
    pvt_header_ns, pvt_log_ns, pvt_carrier = [], [], []
    pvt_topic = getattr(
        topics, "pvt", getattr(topics, "moving_base_pvt", None)
    )
    wanted = {topics.fix}
    if need_relpos:
        wanted.add(topics.relpos)
    if pvt_topic:
        wanted.add(pvt_topic)
    found_topics: set[str] = set()
    for bag in bags:
        with Reader(bag) as reader:
            conns = [c for c in reader.connections if c.topic in wanted]
            found_topics.update(connection.topic for connection in conns)
            for conn, log_ns, raw in reader.messages(connections=conns):
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
                header_ns = _stamp_ns(msg)
                if conn.topic == topics.fix:
                    cov = np.asarray(
                        msg.position_covariance, dtype=np.float64
                    ).reshape(3, 3)
                    fix_header_ns.append(header_ns)
                    fix_log_ns.append(int(log_ns))
                    fix_lat.append(float(msg.latitude))
                    fix_lon.append(float(msg.longitude))
                    fix_alt.append(float(msg.altitude))
                    fix_status.append(int(msg.status.status))
                    fix_cov_max.append(float(np.max(np.diag(cov))))
                    fix_covariance.append(cov)
                    fix_covariance_type.append(
                        int(getattr(msg, "position_covariance_type", -1))
                    )
                elif need_relpos and conn.topic == topics.relpos:
                    n = msg.rel_pos_n * 0.01 + msg.rel_pos_hp_n * 1e-4
                    e = msg.rel_pos_e * 0.01 + msg.rel_pos_hp_e * 1e-4
                    d = msg.rel_pos_d * 0.01 + msg.rel_pos_hp_d * 1e-4
                    yaw = np.arctan2(n, e)  # ENU yaw, same as crop-row node
                    relpos_header_ns.append(header_ns)
                    relpos_log_ns.append(int(log_ns))
                    relpos_yaw.append(float(yaw))
                    relpos_carr.append(int(msg.carr_soln.status))
                    relpos_ned_m.append((float(n), float(e), float(d)))
                    relpos_acc_heading_rad.append(
                        float(np.deg2rad(msg.acc_heading * 1e-5))
                    )
                    relpos_flags.append(
                        tuple(bool(getattr(msg, name)) for name in RELPOS_FLAG_NAMES)
                    )
                elif pvt_topic and conn.topic == pvt_topic:
                    pvt_header_ns.append(header_ns)
                    pvt_log_ns.append(int(log_ns))
                    pvt_carrier.append(int(msg.carr_soln.status))
    missing_topics = wanted - found_topics
    if missing_topics:
        raise RuntimeError(
            "missing required topics: " + ", ".join(sorted(missing_topics))
        )
    if not fix_header_ns or (need_relpos and not relpos_header_ns):
        raise RuntimeError(
            f"no messages on {topics.fix}"
            + (f" or {topics.relpos}" if need_relpos else ""))
    if not relpos_header_ns:
        # Placeholder used only until GnssCourse derives its own heading track.
        relpos_header_ns = [-1]
        relpos_log_ns = [-1]
        relpos_yaw = [0.0]
        relpos_carr = [-1]
        relpos_ned_m = [(np.nan, np.nan, np.nan)]
        relpos_acc_heading_rad = [np.nan]
        relpos_flags = [(False,) * len(RELPOS_FLAG_NAMES)]
    fix_order = np.argsort(np.asarray(fix_header_ns, dtype=np.int64), kind="stable")
    fix_header = np.asarray(fix_header_ns, dtype=np.int64)[fix_order]
    if len(fix_header) > 1 and np.any(np.diff(fix_header) <= 0):
        raise RuntimeError("GNSS chunks overlap or contain duplicate header timestamps")
    fix_lat = np.asarray(fix_lat, dtype=np.float64)[fix_order]
    fix_lon = np.asarray(fix_lon, dtype=np.float64)[fix_order]
    fix_alt = np.asarray(fix_alt, dtype=np.float64)[fix_order]
    fix_status = np.asarray(fix_status, dtype=np.int16)[fix_order]
    fix_cov_max = np.asarray(fix_cov_max, dtype=np.float64)[fix_order]
    fix_log = np.asarray(fix_log_ns, dtype=np.int64)[fix_order]
    fix_covariance = np.asarray(fix_covariance, dtype=np.float64)[fix_order]
    fix_covariance_type = np.asarray(
        fix_covariance_type, dtype=np.int16
    )[fix_order]

    relpos_order = np.argsort(
        np.asarray(relpos_header_ns, dtype=np.int64), kind="stable"
    )
    relpos_header = np.asarray(relpos_header_ns, dtype=np.int64)[relpos_order]
    if len(relpos_header) > 1 and np.any(np.diff(relpos_header) <= 0):
        raise RuntimeError(
            "heading chunks overlap or contain duplicate header timestamps"
        )
    relpos_log = np.asarray(relpos_log_ns, dtype=np.int64)[relpos_order]
    relpos_yaw = np.asarray(relpos_yaw, dtype=np.float64)[relpos_order]
    relpos_carr = np.asarray(relpos_carr, dtype=np.int8)[relpos_order]
    relpos_ned_m = np.asarray(relpos_ned_m, dtype=np.float64)[relpos_order]
    relpos_acc_heading_rad = np.asarray(
        relpos_acc_heading_rad, dtype=np.float64
    )[relpos_order]
    relpos_flags = np.asarray(relpos_flags, dtype=bool)[relpos_order]

    pvt_header = np.asarray(pvt_header_ns, dtype=np.int64)
    pvt_log = np.asarray(pvt_log_ns, dtype=np.int64)
    pvt_carrier_array = np.asarray(pvt_carrier, dtype=np.int8)
    fix_carrier = np.full(len(fix_header), -1, dtype=np.int8)
    if len(pvt_header):
        order = np.argsort(pvt_header, kind="stable")
        pvt_header = pvt_header[order]
        pvt_log = pvt_log[order]
        pvt_carrier_array = pvt_carrier_array[order]
        if len(pvt_header) > 1 and np.any(np.diff(pvt_header) <= 0):
            raise RuntimeError(
                "NavPVT chunks overlap or contain duplicate header timestamps"
            )
        association = nearest_matches(
            fix_header, pvt_header, tolerance_ns=200_000_000
        )
        fix_carrier[association.reference_indices] = pvt_carrier_array[
            association.sample_indices
        ]
    relpos_flags_array = np.asarray(relpos_flags, dtype=bool)
    relpos_ned_array = np.asarray(relpos_ned_m, dtype=np.float64)
    heading_valid = (
        relpos_flags_array[:, 0]
        & relpos_flags_array[:, 1]
        & relpos_flags_array[:, 2]
        & relpos_flags_array[:, 3]
        & ~relpos_flags_array[:, 4]
        & ~relpos_flags_array[:, 5]
        & relpos_flags_array[:, 6]
        & np.isfinite(relpos_ned_array).all(axis=1)
    )
    return RtkTrack(
        fix_t=fix_header.astype(np.float64) * 1e-9,
        fix_lat=fix_lat,
        fix_lon=fix_lon,
        fix_alt=fix_alt,
        fix_status=fix_status,
        fix_cov_max=fix_cov_max,
        relpos_t=relpos_header.astype(np.float64) * 1e-9,
        relpos_yaw=np.unwrap(relpos_yaw),
        relpos_carr=relpos_carr,
        fix_header_ns=fix_header,
        fix_log_ns=fix_log,
        fix_covariance_enu_m2=fix_covariance,
        fix_covariance_type=fix_covariance_type,
        fix_carrier_status=fix_carrier,
        relpos_header_ns=relpos_header,
        relpos_log_ns=relpos_log,
        relpos_ned_m=relpos_ned_array,
        relpos_acc_heading_rad=relpos_acc_heading_rad,
        relpos_flags=relpos_flags_array,
        heading_valid=heading_valid,
        heading_quality_kind="carrier",
        pvt_header_ns=pvt_header,
        pvt_log_ns=pvt_log,
        pvt_carrier_status=pvt_carrier_array,
    )


def read_camera_calibration(bags: list[Path], topic: str, typestore) -> dict:
    """First CameraInfo message, including the rectification matrices."""
    Reader = _reader_type()
    with Reader(bag_for_topic(bags, topic)) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for conn, _, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            k = np.asarray(msg.k, dtype=float)
            return {"width": int(msg.width), "height": int(msg.height),
                    "fx": float(k[0]), "fy": float(k[4]),
                    "cx": float(k[2]), "cy": float(k[5]),
                    "k": k.reshape(3, 3).tolist(),
                    "distortion_model": str(msg.distortion_model),
                    "d": np.asarray(msg.d, dtype=float).tolist(),
                    "r": np.asarray(msg.r, dtype=float).reshape(3, 3).tolist(),
                    "p": np.asarray(msg.p, dtype=float).reshape(3, 4).tolist()}
    raise RuntimeError(f"no camera_info on {topic}")


def read_camera_info(bags: list[Path], topics, typestore) -> dict:
    """First left camera_info message -> calibration/intrinsics dict."""
    return read_camera_calibration(bags, topics.left_info, typestore)


@dataclass(frozen=True)
class _TimedPayload:
    stamp_ns: int
    payload: bytes


def pair_stereo_timestamps(
    left_ns: Iterable[int],
    right_ns: Iterable[int],
    *,
    tolerance_ns: int,
) -> TimestampMatches:
    """Pure timing primitive used to validate a stereo synchronization setup."""
    return monotonic_matches(
        list(left_ns), list(right_ns), tolerance_ns=tolerance_ns
    )


def _drain_stereo_pairs(
    left: deque[_TimedPayload],
    right: deque[_TimedPayload],
    tolerance_ns: int,
):
    """Yield all currently decidable earliest-feasible one-to-one pairs."""
    while left and right:
        residual_ns = right[0].stamp_ns - left[0].stamp_ns
        if residual_ns < -tolerance_ns:
            right.popleft()
        elif residual_ns > tolerance_ns:
            left.popleft()
        else:
            yield left.popleft(), right.popleft()


def _message_bytes(data) -> bytes:
    if hasattr(data, "tobytes"):
        return data.tobytes()
    return bytes(data)


def read_stereo_frames(
    bags: list[Path],
    topics,
    typestore,
    t0: float,
    t1: float,
    stride: int,
    *,
    timestamp_tolerance_s: float = 0.02,
    read_ahead_s: float = 2.0,
):
    """Yield every `stride`-th stereo pair with header stamp in [t0, t1].

    Both image topics must share each bag chunk. Chronologically chained chunks
    are consumed in the order supplied. Pairing is deterministic one-to-one
    association using the configured timestamp tolerance; it makes no
    frame-rate assumption. Pending payloads older than the tolerance are
    discarded so memory stays bounded regardless of selected window length.
    """
    if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)):
        raise ValueError("stride must be a positive integer")
    if stride <= 0:
        raise ValueError("stride must be a positive integer")
    if not np.isfinite(timestamp_tolerance_s) or timestamp_tolerance_s < 0:
        raise ValueError("timestamp_tolerance_s must be finite and non-negative")
    if not np.isfinite(read_ahead_s) or read_ahead_s < 0:
        raise ValueError("read_ahead_s must be finite and non-negative")

    tolerance_ns = int(round(timestamp_tolerance_s * 1_000_000_000))
    t0_ns = int(round(t0 * 1_000_000_000))
    t1_ns = int(round(t1 * 1_000_000_000))
    pending_left: deque[_TimedPayload] = deque()
    pending_right: deque[_TimedPayload] = deque()
    kept = 0
    seen_left = 0
    Reader = _reader_type()
    camera_topics = {topics.left_image, topics.right_image}
    last_header = {topic: None for topic in camera_topics}
    for bag in bags_for_topics(bags, camera_topics):
        with Reader(bag) as reader:
            conns = [
                connection
                for connection in reader.connections
                if connection.topic in camera_topics
            ]
            for conn, _, raw in reader.messages(connections=conns):
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
                stamp_ns = _stamp_ns(msg)
                previous = last_header[conn.topic]
                if previous is not None and stamp_ns <= previous:
                    raise RuntimeError(
                        "camera bag chunks overlap or have non-increasing "
                        f"{conn.topic} header timestamps"
                    )
                last_header[conn.topic] = stamp_ns
                if stamp_ns > t1_ns + int(round(read_ahead_s * 1_000_000_000)):
                    break
                if stamp_ns < t0_ns or stamp_ns > t1_ns:
                    continue
                if conn.topic == topics.left_image:
                    seen_left += 1
                    if (seen_left - 1) % stride != 0:
                        continue
                    pending_left.append(
                        _TimedPayload(stamp_ns, _message_bytes(msg.data))
                    )
                else:
                    pending_right.append(
                        _TimedPayload(stamp_ns, _message_bytes(msg.data))
                    )
                for left, right in _drain_stereo_pairs(
                    pending_left, pending_right, tolerance_ns
                ):
                    kept += 1
                    yield FrameRecord(
                        t=left.stamp_ns * 1e-9,
                        t_right=right.stamp_ns * 1e-9,
                        left_jpeg=left.payload,
                        right_jpeg=right.payload,
                        left_header_ns=left.stamp_ns,
                        right_header_ns=right.stamp_ns,
                    )
                cutoff_ns = stamp_ns - tolerance_ns
                while pending_left and pending_left[0].stamp_ns < cutoff_ns:
                    pending_left.popleft()
                while pending_right and pending_right[0].stamp_ns < cutoff_ns:
                    pending_right.popleft()
    if kept == 0:
        raise RuntimeError(f"no stereo pairs found in [{t0:.1f}, {t1:.1f}]")


def _exact_ns(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{label} must be an exact integer nanosecond value")
    result = int(value)
    if not np.iinfo(np.int64).min <= result <= np.iinfo(np.int64).max:
        raise ValueError(f"{label} is outside int64")
    return result


def _nonnegative_ns(value: Any, label: str) -> int:
    result = _exact_ns(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("JSON metadata must not contain NaN or infinity")
        return value
    raise TypeError(f"unsupported metadata value: {type(value).__name__}")


def _camera_matrix(
    camera_info: Mapping[str, Any],
    lower_name: str,
    upper_name: str,
    shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    value = camera_info.get(lower_name, camera_info.get(upper_name))
    if value is None and lower_name == "k":
        required = ("fx", "fy", "cx", "cy")
        if all(name in camera_info for name in required):
            value = [
                [camera_info["fx"], 0.0, camera_info["cx"]],
                [0.0, camera_info["fy"], camera_info["cy"]],
                [0.0, 0.0, 1.0],
            ]
    if value is None:
        raise ValueError(f"{label} CameraInfo is missing {upper_name}")
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size != shape[0] * shape[1]:
        raise ValueError(f"{label} CameraInfo {upper_name} has invalid size")
    matrix = matrix.reshape(shape)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} CameraInfo {upper_name} is not finite")
    return matrix


def _camera_dimension(camera_info: Mapping[str, Any], name: str, label: str) -> int:
    value = camera_info.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} CameraInfo {name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} CameraInfo {name} must be a positive integer")
    return result


def _rectified_calibration(
    left_info: Mapping[str, Any], right_info: Mapping[str, Any]
) -> dict[str, Any]:
    left_p = _camera_matrix(left_info, "p", "P", (3, 4), "left")
    right_p = _camera_matrix(right_info, "p", "P", (3, 4), "right")
    left_k = left_p[:, :3]
    right_k = right_p[:, :3]
    for label, matrix in (("left", left_k), ("right", right_k)):
        if (
            matrix[0, 0] <= 0
            or matrix[1, 1] <= 0
            or abs(np.linalg.det(matrix)) <= 1e-12
        ):
            raise ValueError(f"{label} rectified projection K is invalid")
    left_projection_t = np.linalg.solve(left_k, left_p[:, 3])
    right_projection_t = np.linalg.solve(right_k, right_p[:, 3])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = right_projection_t - left_projection_t
    if np.linalg.norm(transform[:3, 3]) <= 1e-9:
        raise ValueError("CameraInfo P matrices encode a zero stereo baseline")

    cameras = {}
    for label, info, matrix in (
        ("left", left_info, left_k),
        ("right", right_info, right_k),
    ):
        cameras[label] = {
            "model": "PINHOLE",
            "width": _camera_dimension(info, "width", label),
            "height": _camera_dimension(info, "height", label),
            "K": matrix.tolist(),
            "distortion": [],
        }
    return {
        "contract_version": CONTRACT_VERSION,
        "cameras": cameras,
        "T_right_left": transform.tolist(),
        "rectification": {
            "stored_images": "rectified",
            "intrinsics_source": "CameraInfo.P",
            "distortion_applied_to_stored_images": False,
            "source_camera_info": {
                "left": _json_value(left_info),
                "right": _json_value(right_info),
            },
        },
    }


def _capability_record(
    requested: Mapping[str, bool] | None, dual_available: bool
) -> dict[str, bool]:
    capabilities = {
        "stereo": True,
        "rgbd": False,
        "single_rtk": not dual_available,
        "dual_rtk": dual_available,
        "depth_recorded": False,
        "depth_computed": False,
        "imu_present": False,
        "images_raw": False,
        "images_rectified": True,
    }
    if requested is not None:
        unknown = set(requested).difference(CAPABILITIES)
        if unknown:
            raise ValueError(
                f"unknown capabilities: {', '.join(sorted(unknown))}"
            )
        if any(type(value) is not bool for value in requested.values()):
            raise ValueError("capability options must be booleans")
        capabilities.update(requested)
    fixed = {
        "stereo": True,
        "rgbd": False,
        "depth_recorded": False,
        "depth_computed": False,
        "imu_present": False,
        "images_raw": False,
        "images_rectified": True,
    }
    for name, expected in fixed.items():
        if capabilities[name] is not expected:
            raise ValueError(
                f"ROS2 rectified stereo writer requires {name}={expected}"
            )
    if capabilities["single_rtk"] == capabilities["dual_rtk"]:
        raise ValueError("declare exactly one of single_rtk or dual_rtk")
    if capabilities["dual_rtk"] and not dual_available:
        raise ValueError("dual_rtk requested without complete heading evidence")
    return capabilities


def _associate_all(
    query_ns: np.ndarray,
    source_ns: np.ndarray,
    tolerance_ns: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    matches = nearest_matches(
        query_ns, source_ns, tolerance_ns=tolerance_ns
    )
    if matches.match_count != len(query_ns):
        first = int(matches.unmatched_reference_indices[0])
        raise ValueError(
            f"{label} association exceeds {tolerance_ns} ns at frame {first}"
        )
    if not np.array_equal(
        matches.reference_indices, np.arange(len(query_ns), dtype=np.int64)
    ):
        raise RuntimeError(f"{label} association returned incomplete ordering")
    return matches.sample_indices, matches.residual_ns


def _split_manifest(
    splits: Mapping[str, Sequence[int]] | None, n_frames: int
) -> dict[str, list[int]]:
    if splits is None:
        return {"train": list(range(n_frames)), "val": [], "test": []}
    unknown = set(splits).difference({"train", "val", "test"})
    if unknown:
        raise ValueError(f"unknown data splits: {', '.join(sorted(unknown))}")
    output = {}
    for name in ("train", "val", "test"):
        values = splits.get(name, ())
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"{name} split must be a sequence of frame IDs")
        output[name] = []
        for value in values:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(
                    f"{name} split must contain integer frame IDs"
                )
            output[name].append(int(value))
    return output


def _frame_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty frame ID")
    return value


def _sensor_semantics(
    *,
    camera_frame_id: str,
    primary_antenna_frame_id: str,
    secondary_antenna_frame_id: str | None,
    enu_definition: Mapping[str, Any],
    T_camera_primary_antenna: Any,
    extrinsic_translation_sigma_m: Any,
    extrinsic_provenance: Mapping[str, Any],
    dual_rtk: bool,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    camera_frame = _frame_name(camera_frame_id, "camera_frame_id")
    primary_frame = _frame_name(
        primary_antenna_frame_id, "primary_antenna_frame_id"
    )
    secondary_frame = (
        None
        if secondary_antenna_frame_id is None
        else _frame_name(
            secondary_antenna_frame_id, "secondary_antenna_frame_id"
        )
    )
    if dual_rtk and secondary_frame is None:
        raise ValueError("dual_rtk requires secondary_antenna_frame_id")
    if secondary_frame == primary_frame:
        raise ValueError("primary and secondary antenna frame IDs must differ")

    required_enu = {
        "origin_lat_deg",
        "origin_lon_deg",
        "origin_alt_ellipsoidal_m",
        "ellipsoid",
        "vertical_datum",
        "world_frame_id",
    }
    missing = required_enu.difference(enu_definition)
    if missing:
        raise ValueError(
            f"enu_definition missing: {', '.join(sorted(missing))}"
        )
    latitude = float(enu_definition["origin_lat_deg"])
    longitude = float(enu_definition["origin_lon_deg"])
    altitude = float(enu_definition["origin_alt_ellipsoidal_m"])
    ellipsoid = enu_definition["ellipsoid"]
    vertical_datum = enu_definition["vertical_datum"]
    world_frame = _frame_name(
        enu_definition["world_frame_id"], "enu_definition.world_frame_id"
    )
    if (
        not np.isfinite([latitude, longitude, altitude]).all()
        or not -90 <= latitude <= 90
        or not -180 <= longitude <= 180
    ):
        raise ValueError("enu_definition origin is invalid")
    if not isinstance(ellipsoid, str) or not ellipsoid:
        raise ValueError("enu_definition ellipsoid must be non-empty")
    if not isinstance(vertical_datum, str) or not vertical_datum:
        raise ValueError("enu_definition vertical_datum must be non-empty")
    crs = {
        "type": "local_ENU",
        "world_frame_id": world_frame,
        "axis_order": ["east", "north", "up"],
        "origin_lat": latitude,
        "origin_lon": longitude,
        "origin_alt_ellipsoidal": altitude,
        "ellipsoid": ellipsoid,
        "vertical_datum": vertical_datum,
    }

    transform = np.asarray(T_camera_primary_antenna, dtype=np.float64)
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(
            transform[:3, :3].T @ transform[:3, :3],
            np.eye(3),
            atol=1e-5,
        )
        or np.linalg.det(transform[:3, :3]) <= 0
    ):
        raise ValueError(
            "T_camera_primary_antenna must be a finite rigid 4x4 transform"
        )
    normalized_provenance = _json_value(extrinsic_provenance)
    if (
        not isinstance(normalized_provenance, dict)
        or not isinstance(normalized_provenance.get("method"), str)
        or not normalized_provenance["method"]
    ):
        raise ValueError(
            "extrinsic_provenance must include a non-empty method"
        )
    translation_sigma = np.asarray(
        extrinsic_translation_sigma_m, dtype=np.float64
    )
    if (
        translation_sigma.shape != (3,)
        or not np.isfinite(translation_sigma).all()
        or np.any(translation_sigma < 0)
    ):
        raise ValueError(
            "extrinsic_translation_sigma_m must be finite non-negative [x,y,z]"
        )
    frames = {
        "world": world_frame,
        "camera": camera_frame,
        "primary_antenna": primary_frame,
        "secondary_antenna": secondary_frame,
    }
    return frames, transform, {
        "crs": crs,
        "provenance": normalized_provenance,
        "translation_sigma_m": translation_sigma.tolist(),
    }


def _dual_evidence_available(track: RtkTrack) -> bool:
    header = np.asarray(track.relpos_header_ns)
    baseline = np.asarray(track.relpos_ned_m)
    return (
        track.relpos_timestamps_exact
        and header.ndim == 1
        and header.size > 0
        and np.all(header >= 0)
        and baseline.shape == (len(header), 3)
    )


def _heading_validity(
    baseline: np.ndarray,
    accuracy: np.ndarray,
    carrier: np.ndarray,
    flags: np.ndarray,
) -> np.ndarray:
    return (
        flags[:, 0]
        & flags[:, 1]
        & flags[:, 2]
        & flags[:, 3]
        & ~flags[:, 4]
        & ~flags[:, 5]
        & flags[:, 6]
        & (carrier >= 2)
        & np.isfinite(baseline).all(axis=1)
        & np.isfinite(accuracy)
        & (accuracy >= 0)
    )


def publish_segment_v2(
    destination: str | Path,
    frames: Sequence[FrameRecord],
    posed_frames: Sequence[Any | None],
    track: RtkTrack,
    left_camera_info: Mapping[str, Any],
    right_camera_info: Mapping[str, Any],
    *,
    camera_frame_id: str,
    primary_antenna_frame_id: str,
    secondary_antenna_frame_id: str | None,
    enu_definition: Mapping[str, Any],
    T_camera_primary_antenna: Any,
    extrinsic_translation_sigma_m: Any,
    extrinsic_provenance: Mapping[str, Any],
    clock_offset_ns: int,
    association_tolerance_ns: int,
    stereo_tolerance_ns: int,
    capabilities: Mapping[str, bool] | None = None,
    splits: Mapping[str, Sequence[int]] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> SegmentReader:
    """Atomically publish selected ROS2 evidence as segment contract v2.

    ``clock_offset_ns`` maps each camera header timestamp into the RTK clock
    before nearest-neighbour association. Stored frame timestamps remain the
    exact camera header timestamps; each observation preserves its exact source
    index, header/log timestamp, and signed source-minus-query residual.
    """
    clock_offset = _exact_ns(clock_offset_ns, "clock_offset_ns")
    association_tolerance = _nonnegative_ns(
        association_tolerance_ns, "association_tolerance_ns"
    )
    stereo_tolerance = _nonnegative_ns(
        stereo_tolerance_ns, "stereo_tolerance_ns"
    )
    frame_records = tuple(frames)
    poses = tuple(posed_frames)
    if not frame_records or len(poses) != len(frame_records):
        raise ValueError("frames and posed_frames must have equal non-zero length")
    if not track.fix_timestamps_exact:
        raise ValueError("RtkTrack lacks exact fix header/log timestamps")

    writer = SegmentWriter(destination)
    try:
        n_frames = len(frame_records)
        left_ns_values = []
        right_ns_values = []
        left_payloads = []
        right_payloads = []
        left_formats = []
        right_formats = []
        for index, frame in enumerate(frame_records):
            if not frame.header_timestamps_exact:
                raise ValueError(
                    f"frame {index} lacks exact left/right header timestamps"
                )
            left_ns = _exact_ns(frame.left_header_ns, f"frame {index} left_header_ns")
            right_ns = _exact_ns(
                frame.right_header_ns, f"frame {index} right_header_ns"
            )
            residual = right_ns - left_ns
            if abs(residual) > stereo_tolerance:
                raise ValueError(
                    f"stereo association exceeds {stereo_tolerance} ns "
                    f"at frame {index}"
                )
            left_payload = bytes(frame.left_jpeg)
            right_payload = bytes(frame.right_jpeg)
            left_ns_values.append(left_ns)
            right_ns_values.append(right_ns)
            left_payloads.append(left_payload)
            right_payloads.append(right_payload)
            left_formats.append(detect_compressed_format(left_payload))
            right_formats.append(detect_compressed_format(right_payload))
        left_ns_array = np.asarray(left_ns_values, dtype=np.int64)
        right_ns_array = np.asarray(right_ns_values, dtype=np.int64)
        if np.any(np.diff(left_ns_array) <= 0) or np.any(
            np.diff(right_ns_array) <= 0
        ):
            raise ValueError("selected stereo timestamps must be strictly increasing")

        pose_query_values = [
            _exact_ns(
                int(timestamp) + clock_offset,
                f"frame {index} clock-adjusted timestamp",
            )
            for index, timestamp in enumerate(left_ns_array)
        ]
        pose_query_ns = np.asarray(pose_query_values, dtype=np.int64)
        fix_header_ns = np.asarray(track.fix_header_ns, dtype=np.int64)
        fix_indices, fix_residual_ns = _associate_all(
            pose_query_ns,
            fix_header_ns,
            association_tolerance,
            "GNSS",
        )

        frame_ids = np.arange(n_frames, dtype=np.int64)
        image_dir = writer.directory("images")
        left_paths = []
        right_paths = []
        extension = {"jpeg": ".jpg", "png": ".png"}
        for index, (left_payload, right_payload, left_format, right_format) in enumerate(
            zip(left_payloads, right_payloads, left_formats, right_formats)
        ):
            left_name = f"left_{index:06d}{extension[left_format]}"
            right_name = f"right_{index:06d}{extension[right_format]}"
            (image_dir / left_name).write_bytes(left_payload)
            (image_dir / right_name).write_bytes(right_payload)
            left_paths.append(f"images/{left_name}")
            right_paths.append(f"images/{right_name}")

        viewmats = np.full((n_frames, 4, 4), np.nan, dtype=np.float64)
        centers = np.full((n_frames, 3), np.nan, dtype=np.float64)
        pose_valid = np.zeros(n_frames, dtype=bool)
        for index, pose in enumerate(poses):
            if pose is None:
                continue
            viewmats[index] = np.asarray(pose.viewmat, dtype=np.float64)
            centers[index] = np.asarray(pose.cam_center, dtype=np.float64)
            pose_valid[index] = True
        frame_values = {
            "frame_id": frame_ids,
            "timestamp_ns": left_ns_array,
            "right_timestamp_ns": right_ns_array,
            "stereo_sync_residual_ns": right_ns_array - left_ns_array,
            "pose_query_timestamp_ns": pose_query_ns,
            "left_image_path": np.asarray(left_paths, dtype=np.str_),
            "right_image_path": np.asarray(right_paths, dtype=np.str_),
            "initial_viewmat": viewmats,
            "initial_camera_center_m": centers,
            "pose_valid": pose_valid,
        }

        fix_count = len(fix_header_ns)
        fix_enu = np.asarray(getattr(track, "enu_xyz", None), dtype=np.float64)
        fix_covariance = np.asarray(
            track.fix_covariance_enu_m2, dtype=np.float64
        )
        fix_status = np.asarray(track.fix_status, dtype=np.int16)
        fix_carrier = np.asarray(track.fix_carrier_status, dtype=np.int16)
        fix_log_ns = np.asarray(track.fix_log_ns, dtype=np.int64)
        fix_covariance_type = np.asarray(
            track.fix_covariance_type, dtype=np.int16
        )
        fix_geodetic = np.column_stack(
            (
                np.asarray(track.fix_lat, dtype=np.float64),
                np.asarray(track.fix_lon, dtype=np.float64),
                np.asarray(track.fix_alt, dtype=np.float64),
            )
        )
        expected_fix_shapes = {
            "enu_xyz": (fix_count, 3),
            "fix_covariance_enu_m2": (fix_count, 3, 3),
            "fix_status": (fix_count,),
            "fix_carrier_status": (fix_count,),
            "fix_log_ns": (fix_count,),
            "fix_covariance_type": (fix_count,),
            "fix_geodetic": (fix_count, 3),
        }
        actual_fix = {
            "enu_xyz": fix_enu,
            "fix_covariance_enu_m2": fix_covariance,
            "fix_status": fix_status,
            "fix_carrier_status": fix_carrier,
            "fix_log_ns": fix_log_ns,
            "fix_covariance_type": fix_covariance_type,
            "fix_geodetic": fix_geodetic,
        }
        for name, shape in expected_fix_shapes.items():
            if actual_fix[name].shape != shape:
                raise ValueError(f"RtkTrack {name} must have shape {shape}")
        position_valid, position_quality = normalize_navsat_position_quality(
            fix_status,
            fix_carrier,
            np.isfinite(fix_enu).all(axis=1)
            & np.isfinite(fix_covariance).all(axis=(1, 2)),
        )
        gnss = {
            "frame_id": frame_ids,
            "frame_timestamp_ns": left_ns_array,
            "association_query_timestamp_ns": pose_query_ns,
            "source_index": fix_indices,
            "source_timestamp_ns": fix_header_ns[fix_indices],
            "source_header_timestamp_ns": fix_header_ns[fix_indices],
            "source_log_timestamp_ns": fix_log_ns[fix_indices],
            "source_residual_ns": fix_residual_ns,
            "enu_m": fix_enu[fix_indices],
            "covariance_enu_m2": fix_covariance[fix_indices],
            "fix_status": fix_status[fix_indices],
            "carrier_status": fix_carrier[fix_indices],
            "position_valid": position_valid[fix_indices],
            "position_quality": position_quality[fix_indices],
            "covariance_type": fix_covariance_type[fix_indices],
            "raw_timestamp_ns": fix_header_ns,
            "raw_header_timestamp_ns": fix_header_ns,
            "raw_log_timestamp_ns": fix_log_ns,
            "raw_enu_m": fix_enu,
            "raw_geodetic_deg_m": fix_geodetic,
            "raw_covariance_enu_m2": fix_covariance,
            "raw_fix_status": fix_status,
            "raw_carrier_status": fix_carrier,
            "raw_covariance_type": fix_covariance_type,
            "pvt_header_ns": np.asarray(track.pvt_header_ns, dtype=np.int64),
            "pvt_log_ns": np.asarray(track.pvt_log_ns, dtype=np.int64),
            "pvt_carrier_status": np.asarray(
                track.pvt_carrier_status, dtype=np.int16
            ),
        }

        dual_available = _dual_evidence_available(track)
        capability_record = _capability_record(capabilities, dual_available)
        sensor_frames, camera_primary_transform, semantics = _sensor_semantics(
            camera_frame_id=camera_frame_id,
            primary_antenna_frame_id=primary_antenna_frame_id,
            secondary_antenna_frame_id=secondary_antenna_frame_id,
            enu_definition=enu_definition,
            T_camera_primary_antenna=T_camera_primary_antenna,
            extrinsic_translation_sigma_m=extrinsic_translation_sigma_m,
            extrinsic_provenance=extrinsic_provenance,
            dual_rtk=capability_record["dual_rtk"],
        )
        heading = None
        if capability_record["dual_rtk"]:
            relpos_header_ns = np.asarray(track.relpos_header_ns, dtype=np.int64)
            relpos_log_ns = np.asarray(track.relpos_log_ns, dtype=np.int64)
            baseline = np.asarray(track.relpos_ned_m, dtype=np.float64)
            accuracy = np.asarray(
                track.relpos_acc_heading_rad, dtype=np.float64
            )
            carrier = np.asarray(track.relpos_carr, dtype=np.int16)
            flags = np.asarray(track.relpos_flags, dtype=bool)
            relpos_count = len(relpos_header_ns)
            expected = {
                "relpos_log_ns": (relpos_count,),
                "relpos_ned_m": (relpos_count, 3),
                "relpos_acc_heading_rad": (relpos_count,),
                "relpos_carr": (relpos_count,),
                "relpos_flags": (relpos_count, len(RELPOS_FLAG_NAMES)),
            }
            values = {
                "relpos_log_ns": relpos_log_ns,
                "relpos_ned_m": baseline,
                "relpos_acc_heading_rad": accuracy,
                "relpos_carr": carrier,
                "relpos_flags": flags,
            }
            for name, shape in expected.items():
                if values[name].shape != shape:
                    raise ValueError(f"RtkTrack {name} must have shape {shape}")
            heading_indices, heading_residual_ns = _associate_all(
                pose_query_ns,
                relpos_header_ns,
                association_tolerance,
                "heading",
            )
            raw_valid = _heading_validity(
                baseline, accuracy, carrier, flags
            )
            heading = {
                "frame_id": frame_ids,
                "frame_timestamp_ns": left_ns_array,
                "association_query_timestamp_ns": pose_query_ns,
                "source_index": heading_indices,
                "source_timestamp_ns": relpos_header_ns[heading_indices],
                "source_header_timestamp_ns": relpos_header_ns[heading_indices],
                "source_log_timestamp_ns": relpos_log_ns[heading_indices],
                "source_residual_ns": heading_residual_ns,
                "baseline_ned_m": baseline[heading_indices],
                "acc_heading_rad": accuracy[heading_indices],
                "carrier_status": carrier[heading_indices],
                "flags": flags[heading_indices],
                "valid": raw_valid[heading_indices],
                "raw_timestamp_ns": relpos_header_ns,
                "raw_header_timestamp_ns": relpos_header_ns,
                "raw_log_timestamp_ns": relpos_log_ns,
                "raw_baseline_ned_m": baseline,
                "raw_acc_heading_rad": accuracy,
                "raw_carrier_status": carrier,
                "raw_flags": flags,
                "raw_valid": raw_valid,
            }

        calibration = _rectified_calibration(
            left_camera_info, right_camera_info
        )
        calibration["sensor_frames"] = sensor_frames
        calibration["transform_conventions"] = {
            "T_right_left": "right_from_left",
            "T_camera_primary_antenna": "camera_from_primary_antenna",
        }
        calibration["rough_extrinsics"] = {
            "convention": "T_camera_primary_antenna maps primary-antenna "
                          "frame coordinates into the camera optical frame",
            "T_camera_primary_antenna": camera_primary_transform.tolist(),
            "provenance": semantics["provenance"],
        }
        crs = semantics["crs"]
        meta = {
            "contract_version": CONTRACT_VERSION,
            "n_frames": n_frames,
            "adapter": "ros2_zed_ublox",
            "capabilities": capability_record,
            "clock_alignment": {
                "camera_to_rtk_offset_ns": clock_offset,
                "association_tolerance_ns": association_tolerance,
                "stereo_tolerance_ns": stereo_tolerance,
                "frame_timestamp_semantics": "camera header clock",
                "association_query_semantics":
                    "frames.timestamp_ns + camera_to_rtk_offset_ns",
                "source_timestamp_semantics": "sensor header clock",
                "source_log_timestamp_semantics": "rosbag log timestamp",
            },
            "world_origin": {
                "lat0": crs["origin_lat"],
                "lon0": crs["origin_lon"],
                "alt0_ellipsoidal_m": crs["origin_alt_ellipsoidal"],
            },
            "crs": crs,
            "coordinate_frame": {
                "type": "local_enu",
                "world_frame_id": crs["world_frame_id"],
                "units": "m",
                "origin_wgs84": {
                    "latitude_deg": crs["origin_lat"],
                    "longitude_deg": crs["origin_lon"],
                    "ellipsoidal_altitude_m":
                        crs["origin_alt_ellipsoidal"],
                    "ellipsoid": crs["ellipsoid"],
                    "vertical_datum": crs["vertical_datum"],
                },
            },
            "timebase": {
                "frame_timestamp_source": "left_camera_header",
                "observation_timestamp_source": "sensor_header",
                "unit": "ns",
                "association_clock_offset_ns": clock_offset,
            },
            "sensor_frames": sensor_frames,
            "position_observation": {
                "type": "gnss",
                "quantity": "antenna_phase_center",
                "sensor_frame_id": sensor_frames["primary_antenna"],
                "coordinates": "ENU_m",
                "covariance_frame": "ENU_m2",
                "validity_field": "position_valid",
                "quality_field": "position_quality",
                "quality_vocabulary": list(POSITION_QUALITY_VOCABULARY),
            },
            "heading_observation": (
                {
                    "vector": "primary_to_secondary",
                    "components": ["north", "east", "down"],
                    "primary_frame_id": sensor_frames["primary_antenna"],
                    "secondary_frame_id": sensor_frames["secondary_antenna"],
                }
                if capability_record["dual_rtk"]
                else None
            ),
            "initial_pose": {
                "camera_frame_id": sensor_frames["camera"],
                "position_quantity": "left_camera_center",
                "source": "RTK position/heading plus configured rough "
                          "antenna-to-camera extrinsic",
                "lever_arm_applied": True,
                "extrinsic_translation_sigma_m":
                    semantics["translation_sigma_m"],
                "extrinsic_translation_sigma_frame_id":
                    sensor_frames["camera"],
            },
            "position_evidence": {
                "quantity": "primary antenna phase-center position",
                "frame_id": sensor_frames["primary_antenna"],
                "coordinates": "local ENU metres [east, north, up]",
                "role": "absolute RTK observation; not a camera position",
            },
            "heading_evidence": (
                {
                    "quantity": "primary-to-secondary antenna baseline",
                    "primary_frame_id": sensor_frames["primary_antenna"],
                    "secondary_frame_id": sensor_frames["secondary_antenna"],
                    "coordinates": "N/E/D metres [north, east, down]",
                }
                if capability_record["dual_rtk"]
                else None
            ),
            "initial_pose_semantics": {
                "camera_frame_id": sensor_frames["camera"],
                "camera_centres": "lever-arm-corrected rough-prior camera "
                                  "optical centres in local ENU",
                "not_direct_gnss": True,
                "rough_extrinsic_convention":
                    calibration["rough_extrinsics"]["convention"],
                "extrinsic_provenance": semantics["provenance"],
            },
            "image_formats": {
                "left": sorted(set(left_formats)),
                "right": sorted(set(right_formats)),
            },
            "provenance": _json_value({} if provenance is None else provenance),
        }
        manifest = _split_manifest(splits, n_frames)
        writer.write_frames(frame_values)
        writer.write_calibration(calibration)
        writer.write_meta(meta)
        writer.write_manifest(manifest)
        writer.write_observations("gnss", gnss)
        if heading is not None:
            writer.write_observations("heading", heading)
        return writer.finalize()
    except Exception:
        writer.abort()
        raise


def ingest_config_v2(
    cfg,
    destination: str | Path,
    *,
    window: Mapping[str, Any] | None = None,
) -> SegmentReader:
    """Run the ROS2 adapter from a resolved robot/sequence configuration."""
    from rtk_splat.adapters.pose_sources import make_pose_source
    from rtk_splat.core.poses import (
        pose_frames_from_extrinsic,
        tilt_deviations,
    )

    typestore, source = make_pose_source(cfg)
    geometry = getattr(cfg, "sensor_geometry", None)
    if geometry is None:
        raise ValueError(
            "robot profile must declare sensor_geometry with frames and "
            "T_camera_primary_antenna"
        )
    if window is None:
        relative = getattr(cfg.segment, "window_s", None)
        if relative is None or len(relative) != 2:
            raise ValueError(
                "ROS2 ingestion needs segment.window_s or a selected window artifact"
            )
        start, end = (float(value) for value in relative)
        window = {
            "t0": float(source.track.fix_t[0] + start),
            "t1": float(source.track.fix_t[0] + end),
            "t0_rel_s": start,
        }
    t0, t1 = float(window["t0"]), float(window["t1"])
    if not np.isfinite([t0, t1]).all() or t1 <= t0:
        raise ValueError("selected window must contain finite increasing t0/t1")

    stride = int(getattr(cfg.segment, "frame_stride", 1))
    stereo_tolerance_s = float(
        getattr(cfg.segment, "stereo_tolerance_s", 0.02)
    )
    clock_offset_float = float(cfg.pose.time_offset_s) * 1.0e9
    clock_offset_ns = int(round(clock_offset_float))
    if not np.isclose(clock_offset_float, clock_offset_ns, atol=1.0e-6):
        raise ValueError("pose.time_offset_s must be exactly representable in ns")
    offset_s = clock_offset_ns * 1.0e-9
    frames = list(
        read_stereo_frames(
            cfg.paths.bags,
            cfg.topics,
            typestore,
            t0 - offset_s,
            t1 - offset_s,
            stride,
            timestamp_tolerance_s=stereo_tolerance_s,
        )
    )
    pose_stamps = [
        (int(frame.left_header_ns) + clock_offset_ns) * 1.0e-9
        for frame in frames
    ]
    tilts = None
    tilt_provenance = None
    if bool(getattr(cfg.pose, "use_imu_tilt", False)):
        imu_t, imu_q, _ = read_imu(
            cfg.paths.bags,
            cfg.topics.imu,
            typestore,
            frames[0].t - 3.0,
            frames[-1].t + 3.0,
        )
        tilts, tilt_provenance = tilt_deviations(
            imu_t,
            imu_q,
            [frame.t for frame in frames],
            float(cfg.pose.imu_lp_window_s),
        )
    poses = pose_frames_from_extrinsic(
        source.track,
        pose_stamps,
        cfg.pose,
        geometry.T_camera_primary_antenna,
        tilts,
    )
    keep = [index for index, pose in enumerate(poses) if pose is not None]
    if not keep:
        raise RuntimeError("the selected window has no trustworthy RTK camera poses")
    selected_frames = [frames[index] for index in keep]
    selected_poses = [poses[index] for index in keep]

    left_info = read_camera_calibration(
        cfg.paths.bags, cfg.topics.left_info, typestore
    )
    right_info = read_camera_calibration(
        cfg.paths.bags, cfg.topics.right_info, typestore
    )
    frame_ids = geometry.frames
    enu = source.enu.crs()
    enu_definition = {
        "origin_lat_deg": float(enu["origin_lat"]),
        "origin_lon_deg": float(enu["origin_lon"]),
        "origin_alt_ellipsoidal_m": float(enu["origin_alt_ellipsoidal"]),
        "ellipsoid": str(enu["ellipsoid"]),
        "vertical_datum": str(enu["vertical_datum"]),
        "world_frame_id": str(getattr(frame_ids, "world", "map")),
    }
    n_frames = len(selected_frames)
    validation_every = int(
        getattr(cfg.segment, "validation_every", 8)
    )
    if validation_every <= 1:
        raise ValueError("segment.validation_every must be greater than one")
    val = list(range(validation_every - 1, n_frames, validation_every))
    val_set = set(val)
    splits = {
        "train": [index for index in range(n_frames) if index not in val_set],
        "val": val,
        "test": [],
    }
    tolerance_ns = int(
        round(float(getattr(cfg.segment, "association_tolerance_s", 0.15)) * 1.0e9)
    )
    return publish_segment_v2(
        destination,
        selected_frames,
        selected_poses,
        source.track,
        left_info,
        right_info,
        camera_frame_id=str(frame_ids.camera),
        primary_antenna_frame_id=str(frame_ids.primary_antenna),
        secondary_antenna_frame_id=(
            str(frame_ids.secondary_antenna)
            if hasattr(frame_ids, "secondary_antenna")
            else None
        ),
        enu_definition=enu_definition,
        T_camera_primary_antenna=geometry.T_camera_primary_antenna,
        extrinsic_translation_sigma_m=(
            geometry.extrinsic_translation_sigma_m
        ),
        extrinsic_provenance=vars(geometry.extrinsic_provenance),
        clock_offset_ns=clock_offset_ns,
        association_tolerance_ns=tolerance_ns,
        stereo_tolerance_ns=int(round(stereo_tolerance_s * 1.0e9)),
        splits=splits,
        provenance={
            "adapter": "ros2_zed_ublox",
            "window": dict(window),
            "input_frame_count": len(frames),
            "dropped_unposeable": len(frames) - n_frames,
            "frame_stride": stride,
            "imu_tilt": tilt_provenance,
        },
    )


def read_imu(bags: list[Path], topic: str, typestore, t0: float, t1: float):
    """(t, quat_xyzw, gyro_xyz) arrays from sensor_msgs/Imu within [t0, t1]."""
    Reader = _reader_type()
    ts, quats, gyros = [], [], []
    with Reader(bag_for_topic(bags, topic)) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for conn, _, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            t = _stamp_s(msg)
            if t > t1 + 2.0:
                break
            if t < t0 - 2.0:
                continue
            q, w = msg.orientation, msg.angular_velocity
            ts.append(t)
            quats.append((q.x, q.y, q.z, q.w))
            gyros.append((w.x, w.y, w.z))
    if not ts:
        raise RuntimeError(f"no IMU messages on {topic} in window")
    return np.array(ts), np.array(quats), np.array(gyros)


def read_fix_with_reception(bag: Path, topic: str, typestore):
    """(header_stamp_ns, reception_ns) per message -- for cross-bag clock
    offset estimation on a topic recorded by two machines."""
    Reader = _reader_type()
    out = []
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for conn, t_recv, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            hdr_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            out.append((hdr_ns, t_recv))
    if not out:
        raise RuntimeError(f"no messages on {topic} in {bag}")
    return np.array(out, dtype=np.int64)
