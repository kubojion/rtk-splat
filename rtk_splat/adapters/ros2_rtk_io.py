"""ROS2 bag I/O for GNSS/RTK observations.

This module is independent of pose-source policy and dataset publication.  It
provides the optional rosbags typestore and converts ROS messages into the
plain records consumed by pose sources and concrete adapters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rtk_splat.adapters.records import RELPOS_FLAG_NAMES, RtkTrack
from rtk_splat.adapters.synchronization import nearest_matches


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
    """Build a minimal Humble typestore, optionally adding u-blox types."""
    try:
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError as exc:
        raise RuntimeError(
            "ROS2 bag ingestion requires the optional 'rosbags' package"
        ) from exc
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    if ublox_msgs_dir is not None:
        types = {}
        for stem in _UBLOX_TYPES:
            source = Path(ublox_msgs_dir) / f"{stem}.msg"
            name = f"ublox_ubx_msgs/msg/{stem}"
            types.update(get_types_from_msg(source.read_text(), name))
        typestore.register(types)
    return typestore


def _stamp_ns(message) -> int:
    return (
        int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec)
    )


def read_rtk_track(
    bags: list[Path],
    topics,
    typestore,
    need_relpos: bool = True,
    *,
    reader_type=None,
) -> RtkTrack:
    """Linear pass(es) over the small RTK topics, resolved across bags."""
    Reader = _reader_type() if reader_type is None else reader_type
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
