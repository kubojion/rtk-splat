"""Read images, RTK fixes and dual-antenna heading from a rosbag2 bag.

Uses `rosbags` (no ROS runtime). Humble sqlite3 bags do not embed custom
message definitions, so ublox_ubx_msgs types are registered from their .msg
sources (read-only).

All timestamps are message header stamps in seconds (float), matching the
convention of the validated crop-row pipeline.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore


@dataclass
class RtkTrack:
    """RTK antenna fixes and heading samples over a time window."""

    fix_t: np.ndarray       # (N,) s
    fix_lat: np.ndarray     # (N,) deg
    fix_lon: np.ndarray     # (N,) deg
    fix_alt: np.ndarray     # (N,) m
    fix_status: np.ndarray  # (N,) NavSatFix status.status
    fix_cov_max: np.ndarray  # (N,) max diagonal of position covariance
    relpos_t: np.ndarray    # (M,) s
    relpos_yaw: np.ndarray  # (M,) rad, ENU yaw of moving_base->rover, unwrapped
    relpos_carr: np.ndarray  # (M,) int carrier solution status (2 = RTK fixed)


@dataclass
class FrameRecord:
    """One selected stereo frame: header stamp + jpeg payloads."""

    t: float
    t_right: float
    left_jpeg: bytes
    right_jpeg: bytes


# Only the types this pipeline reads (plus their nested deps). Registering the
# whole ublox_ubx_msgs directory pulls in ESF messages whose binary-literal
# constants some rosbags versions cannot parse.
_UBLOX_TYPES = ("CarrSoln", "UBXNavRelPosNED")


def build_typestore(ublox_msgs_dir: Path | None):
    """Humble typestore; ublox types registered only when a directory is
    given (only the rtk_dual_antenna source needs them)."""
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


def bag_for_topic(bags: list[Path], topic: str) -> Path:
    """First bag that contains `topic` (topics may be split across bags)."""
    for bag in bags:
        with Reader(bag) as reader:
            if any(c.topic == topic for c in reader.connections):
                return bag
    raise RuntimeError(f"topic {topic} not found in any of: "
                       + ", ".join(str(b) for b in bags))


def read_rtk_track(bags: list[Path], topics, typestore,
                   need_relpos: bool = True) -> RtkTrack:
    """Linear pass(es) over the small RTK topics, resolved across bags."""
    fixes, relpos = [], []
    wanted = {topics.fix: bag_for_topic(bags, topics.fix)}
    if need_relpos:
        wanted[topics.relpos] = bag_for_topic(bags, topics.relpos)
    for bag in set(wanted.values()):
        with Reader(bag) as reader:
            conns = [c for c in reader.connections
                     if c.topic in wanted and wanted[c.topic] == bag]
            for conn, _, raw in reader.messages(connections=conns):
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
                if conn.topic == topics.fix:
                    cov = np.asarray(msg.position_covariance, dtype=float)
                    fixes.append((_stamp_s(msg), msg.latitude, msg.longitude,
                                  msg.altitude, int(msg.status.status),
                                  float(max(cov[0], cov[4], cov[8]))))
                else:
                    n = msg.rel_pos_n * 0.01 + msg.rel_pos_hp_n * 1e-4
                    e = msg.rel_pos_e * 0.01 + msg.rel_pos_hp_e * 1e-4
                    yaw = np.arctan2(n, e)  # ENU yaw, same as crop-row node
                    relpos.append((_stamp_s(msg), yaw,
                                   int(msg.carr_soln.status)))
    if not fixes or (need_relpos and not relpos):
        raise RuntimeError(
            f"no messages on {topics.fix}"
            + (f" or {topics.relpos}" if need_relpos else ""))
    if not relpos:
        relpos = [(0.0, 0.0, 0)]  # placeholder; unused without relpos
    fx = np.array(fixes)
    rp = np.array(relpos)
    return RtkTrack(
        fix_t=fx[:, 0], fix_lat=fx[:, 1], fix_lon=fx[:, 2], fix_alt=fx[:, 3],
        fix_status=fx[:, 4].astype(int), fix_cov_max=fx[:, 5],
        relpos_t=rp[:, 0], relpos_yaw=np.unwrap(rp[:, 1]),
        relpos_carr=rp[:, 2].astype(int),
    )


def read_camera_calibration(bags: list[Path], topic: str, typestore) -> dict:
    """First CameraInfo message, including the rectification matrices."""
    with Reader(bag_for_topic(bags, topic)) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for conn, _, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            k = np.asarray(msg.k, dtype=float)
            return {"width": int(msg.width), "height": int(msg.height),
                    "fx": float(k[0]), "fy": float(k[4]),
                    "cx": float(k[2]), "cy": float(k[5]),
                    "distortion_model": str(msg.distortion_model),
                    "d": np.asarray(msg.d, dtype=float).tolist(),
                    "r": np.asarray(msg.r, dtype=float).reshape(3, 3).tolist(),
                    "p": np.asarray(msg.p, dtype=float).reshape(3, 4).tolist()}
    raise RuntimeError(f"no camera_info on {topic}")


def read_camera_info(bags: list[Path], topics, typestore) -> dict:
    """First left camera_info message -> calibration/intrinsics dict."""
    return read_camera_calibration(bags, topics.left_info, typestore)


def read_stereo_frames(bags: list[Path], topics, typestore, t0: float,
                       t1: float, stride: int):
    """Yield every `stride`-th stereo pair with header stamp in [t0, t1].

    This adapter currently assumes both image topics share the bag containing
    the left stream and use the project's 15 Hz ZED trigger. They are paired by
    header stamp bucketed at half a frame period. Pending dicts are pruned so
    memory stays bounded regardless of window length. Other camera rates or
    split image bags need a dataset adapter rather than a silent guess here.
    """
    pending_left: dict[int, tuple[float, bytes]] = {}
    pending_right: dict[int, tuple[float, bytes]] = {}
    kept = 0
    seen_left = 0
    with Reader(bag_for_topic(bags, topics.left_image)) as reader:
        conns = [c for c in reader.connections
                 if c.topic in (topics.left_image, topics.right_image)]
        for conn, _, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            t = _stamp_s(msg)
            if t > t1 + 2.0:
                break  # bag is time-ordered; nothing further can match
            if t < t0 or t > t1:
                continue
            key = round(t * 30)  # bucket = half a frame period at 15 Hz
            if conn.topic == topics.left_image:
                seen_left += 1
                if (seen_left - 1) % stride != 0:
                    continue
                pending_left[key] = (t, msg.data.tobytes())
            else:
                pending_right[key] = (t, msg.data.tobytes())
            if key in pending_left and key in pending_right:
                tl, left = pending_left.pop(key)
                tr, right = pending_right.pop(key)
                kept += 1
                yield FrameRecord(t=tl, t_right=tr, left_jpeg=left,
                                  right_jpeg=right)
            # prune anything older than ~2 s that never found its partner
            stale = key - 60
            for d in (pending_left, pending_right):
                for k in [k for k in d if k < stale]:
                    del d[k]
    if kept == 0:
        raise RuntimeError(f"no stereo pairs found in [{t0:.1f}, {t1:.1f}]")


def read_imu(bags: list[Path], topic: str, typestore, t0: float, t1: float):
    """(t, quat_xyzw, gyro_xyz) arrays from sensor_msgs/Imu within [t0, t1]."""
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
