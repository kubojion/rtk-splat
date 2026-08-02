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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rtk_splat.adapters.records import FrameRecord, RtkTrack
from rtk_splat.adapters.ros2_rtk_io import (
    build_typestore,
    read_rtk_track as _read_rtk_track,
)
from rtk_splat.adapters.sampling import resolve_frame_stride
from rtk_splat.adapters.synchronization import (
    TimestampMatches,
    monotonic_matches,
)
from rtk_splat.core.segment import SegmentReader
from rtk_splat.core.runtime_resolution import (
    configuration_evidence,
    record_override,
    runtime_resolution_plain,
)

def _reader_type():
    try:
        from rosbags.rosbag2 import Reader
    except ImportError as exc:
        raise RuntimeError(
            "ROS2 bag ingestion requires the optional 'rosbags' package"
        ) from exc
    return Reader

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


def read_rtk_track(
    bags: list[Path], topics, typestore, need_relpos: bool = True
) -> RtkTrack:
    """Compatibility facade for the shared ROS2 RTK bag reader."""
    return _read_rtk_track(
        bags,
        topics,
        typestore,
        need_relpos,
        reader_type=_reader_type(),
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


def measure_camera_header_rate(
    bags: list[Path],
    topic: str,
    typestore,
    t0: float,
    t1: float,
    *,
    max_samples: int = 64,
) -> float:
    """Measure image rate from a bounded header sample in the chosen window."""
    if max_samples < 3:
        raise ValueError("max_samples must be at least three")
    start_ns = int(round((t0 - 2.0) * 1.0e9))
    stop_ns = int(round((t1 + 2.0) * 1.0e9))
    header_ns: list[int] = []
    Reader = _reader_type()
    for bag in bags_for_topics(bags, {topic}):
        with Reader(bag) as reader:
            connections = [c for c in reader.connections if c.topic == topic]
            for connection, _, raw in reader.messages(
                connections=connections, start=start_ns, stop=stop_ns
            ):
                message = typestore.deserialize_cdr(raw, connection.msgtype)
                stamp = _stamp_ns(message)
                if int(round(t0 * 1.0e9)) <= stamp <= int(round(t1 * 1.0e9)):
                    header_ns.append(stamp)
                    if len(header_ns) >= max_samples:
                        break
        if len(header_ns) >= max_samples:
            break
    if len(header_ns) < 3:
        raise RuntimeError(
            f"cannot measure camera rate from {topic} in [{t0:.3f}, {t1:.3f}]"
        )
    delta_s = np.diff(np.asarray(header_ns, dtype=np.int64)) * 1.0e-9
    if np.any(delta_s <= 0) or not np.isfinite(delta_s).all():
        raise RuntimeError("camera header sample is not strictly increasing")
    return float(1.0 / np.median(delta_s))


def measure_track_speed(track: RtkTrack, t0: float, t1: float) -> float:
    """Median moving speed from the GNSS trajectory within one window."""
    timestamps = np.asarray(track.fix_t, dtype=np.float64)
    positions = np.asarray(track.enu_xyz, dtype=np.float64)
    mask = (timestamps >= t0) & (timestamps <= t1)
    timestamps = timestamps[mask]
    positions = positions[mask]
    if len(timestamps) < 3:
        raise RuntimeError("cannot measure speed: fewer than three GNSS fixes")
    delta_t = np.diff(timestamps)
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / delta_t
    moving = speed[np.isfinite(speed) & (speed > 0.01)]
    if len(moving) < 2:
        raise RuntimeError("cannot derive frame stride from a stationary window")
    return float(np.median(moving))


from rtk_splat.adapters.publication import publish_segment_v2


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

    stereo_tolerance_s = float(
        getattr(cfg.segment, "stereo_tolerance_s", 0.02)
    )
    clock_offset_float = float(cfg.pose.time_offset_s) * 1.0e9
    clock_offset_ns = int(round(clock_offset_float))
    if not np.isclose(clock_offset_float, clock_offset_ns, atol=1.0e-6):
        raise ValueError("pose.time_offset_s must be exactly representable in ns")
    offset_s = clock_offset_ns * 1.0e-9
    configured_stride = getattr(cfg.segment, "frame_stride", 1)
    if isinstance(configured_stride, str) and configured_stride.strip().lower() == "auto":
        camera_rate_hz = measure_camera_header_rate(
            cfg.paths.bags,
            cfg.topics.left_image,
            typestore,
            t0 - offset_s,
            t1 - offset_s,
        )
        median_speed_m_s = measure_track_speed(source.track, t0, t1)
        stride = resolve_frame_stride(
            cfg,
            measured_camera_rate_hz=camera_rate_hz,
            measured_median_speed_m_s=median_speed_m_s,
        )
    else:
        stride = int(configured_stride)
        if stride <= 0:
            raise ValueError("segment.frame_stride must be positive or 'auto'")
        record_override(cfg, "frame_stride", stride)
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
            "runtime_resolution": runtime_resolution_plain(cfg),
            "configuration": configuration_evidence(cfg),
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
