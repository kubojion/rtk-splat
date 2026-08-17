"""ROS 2 bag reader for the optional RTK--stereo integrity audit.

This module is the only owner of calibration-specific ROS topics, message
decoding, typestores, and rosbag traversal.  It returns plain observation
records consumed by the ROS-free diagnostics layer.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from .records import (
    CalibrationBagObservations,
    NavPvtStatusObservation,
    NavSatFixObservation,
    RelPosObservation,
    StereoFrameObservation,
)


_NS_PER_S = 1_000_000_000
_CALIBRATION_UBLOX_TYPES = (
    "CarrSoln",
    "GpsFix",
    "PSMPVT",
    "UBXNavRelPosNED",
    "UBXNavPVT",
)


@dataclass(frozen=True)
class CalibrationTopics:
    """Absolute, distinct ROS topics needed by the integrity audit."""

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


def build_calibration_typestore(ublox_msgs_dir: Path):
    """Build the minimal ROS 2 Humble typestore for the audit topics."""
    directory = Path(ublox_msgs_dir)
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    custom_types = {}
    for stem in _CALIBRATION_UBLOX_TYPES:
        source = directory / f"{stem}.msg"
        if not source.is_file():
            raise FileNotFoundError(
                f"required u-blox message is missing: {source}"
            )
        custom_types.update(
            get_types_from_msg(
                source.read_text(), f"ublox_ubx_msgs/msg/{stem}"
            )
        )
    typestore.register(custom_types)
    return typestore


def header_stamp_ns(message) -> int:
    """Return an exact ROS header timestamp as a Python integer."""
    stamp = message.header.stamp
    return int(stamp.sec) * _NS_PER_S + int(stamp.nanosec)


def historical_stereo_bucket(header_ns: int) -> int:
    """Reproduce the original ``round(header_seconds * 30)`` pairing key."""
    seconds, nanoseconds = divmod(int(header_ns), _NS_PER_S)
    header_seconds = seconds + nanoseconds * 1.0e-9
    return round(header_seconds * 30)


def navsat_fix_observation(
    message, log_ns: int
) -> NavSatFixObservation:
    covariance = tuple(float(value) for value in message.position_covariance)
    if len(covariance) != 9 or not all(
        math.isfinite(value) for value in covariance
    ):
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
    raw_ned = (
        int(message.rel_pos_n),
        int(message.rel_pos_e),
        int(message.rel_pos_d),
    )
    hp_ned = (
        int(message.rel_pos_hp_n),
        int(message.rel_pos_hp_e),
        int(message.rel_pos_hp_d),
    )
    ned_m = tuple(
        raw * 0.01 + high_precision * 1.0e-4
        for raw, high_precision in zip(raw_ned, hp_ned)
    )
    enu_m = (ned_m[1], ned_m[0], -ned_m[2])
    raw_accuracy = (
        int(message.acc_n),
        int(message.acc_e),
        int(message.acc_d),
    )
    accuracy_m = tuple(value * 1.0e-4 for value in raw_accuracy)
    raw_length = int(message.rel_pos_length)
    hp_length = int(message.rel_pos_hp_length)
    raw_accuracy_length = int(message.acc_length)
    raw_heading = int(message.rel_pos_heading)
    raw_accuracy_heading = int(message.acc_heading)
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
        accuracy_ned_0p1mm=raw_accuracy,
        accuracy_ned_m=accuracy_m,
        accuracy_length_0p1mm=raw_accuracy_length,
        accuracy_length_m=raw_accuracy_length * 1.0e-4,
        accuracy_heading_1e5_deg=raw_accuracy_heading,
        accuracy_heading_deg=raw_accuracy_heading * 1.0e-5,
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


def navpvt_status_observation(
    message, log_ns: int
) -> NavPvtStatusObservation:
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
            available = {
                connection.topic for connection in reader.connections
            }
        for topic in list(remaining):
            if topic in available:
                resolved[topic] = bag
                remaining.remove(topic)
        if not remaining:
            break
    if remaining:
        raise RuntimeError(
            "calibration topics are missing: " + ", ".join(sorted(remaining))
        )
    return resolved


def mcap_index_integrity(directory: Path) -> dict[str, int | bool]:
    """Open and validate one indexed, single-file MCAP bag read-only."""
    root = Path(directory).expanduser()
    try:
        with Reader(root) as reader:
            storage_container = getattr(reader, "storage", None)
            storages = getattr(storage_container, "storages", None)
            if not storages or len(storages) != 1:
                raise RuntimeError(
                    "MCAP audit requires one indexed storage file"
                )
            storage = storages[0]
            chunks = list(getattr(storage, "chunks", ()))
            data_start = int(getattr(storage, "data_start"))
            data_end = int(getattr(storage, "data_end"))
            connection_count = len(reader.connections)
    except RuntimeError:
        raise
    except BaseException as exc:
        raise RuntimeError(
            f"cannot parse MCAP indexes in {root}: {exc}"
        ) from exc
    if not chunks or data_start < 0 or data_end <= data_start:
        raise RuntimeError(f"MCAP has no valid indexed data range: {root}")
    offsets = [int(item.chunk_start_offset) for item in chunks]
    starts = [int(item.message_start_time) for item in chunks]
    stops = [int(item.message_end_time) for item in chunks]
    if offsets != sorted(offsets) or any(
        stop < start for start, stop in zip(starts, stops)
    ):
        raise RuntimeError(f"MCAP chunk index is inconsistent: {root}")
    return {
        "reader_opened": True,
        "connection_count": connection_count,
        "chunk_count": len(chunks),
        "data_start_offset": data_start,
        "data_end_offset": data_end,
        "first_chunk_message_start_ns": min(starts),
        "last_chunk_message_stop_ns": max(stops),
        "chunk_offsets_monotonic": True,
    }


def _bounded_reader_messages(
    reader,
    connections,
    start_ns: int,
    stop_ns: int,
):
    """Yield a bounded bag interval without MCAP's all-chunk fan-out.

    The rosbags MCAP index reader constructs one generator for every matching
    chunk before yielding its first message.  On a large, fragmented exFAT
    file this turns a contiguous interval into pathological random seeks.
    A single-file, uncompressed rosbag2 MCAP can instead be scanned over the
    exact contiguous indexed chunk range.

    Other layouts retain the library reader path. Internal bounds are restored
    in ``finally`` and the source file remains read-only.
    """
    directory_storage = getattr(reader, "storage", None)
    storages = getattr(directory_storage, "storages", None)
    metadata = getattr(directory_storage, "metadata", None)
    if (
        not storages
        or len(storages) != 1
        or getattr(metadata, "compression_mode", None) == "message"
    ):
        yield from reader.messages(
            connections=connections, start=start_ns, stop=stop_ns
        )
        return

    storage = storages[0]
    chunks = getattr(storage, "chunks", None)
    scan = getattr(storage, "messages_scan", None)
    if not chunks or scan is None:
        yield from reader.messages(
            connections=connections, start=start_ns, stop=stop_ns
        )
        return

    requested_topics = {connection.topic for connection in connections}
    storage_connections = [
        connection
        for connection in storage.connections
        if connection.topic in requested_topics
    ]
    if not storage_connections:
        return
    connection_by_storage_id = {
        connection.id: next(
            requested
            for requested in connections
            if requested.topic == connection.topic
        )
        for connection in storage_connections
    }
    channel_ids = set(connection_by_storage_id)
    ordered = sorted(chunks, key=lambda chunk: chunk.chunk_start_offset)
    matching_positions = [
        index
        for index, chunk in enumerate(ordered)
        if start_ns < chunk.message_end_time
        and chunk.message_start_time < stop_ns
        and any(
            chunk.channel_count.get(channel_id, 0)
            for channel_id in channel_ids
        )
    ]
    if not matching_positions:
        return
    first = min(matching_positions)
    last = max(matching_positions)
    bounded_start = ordered[first].chunk_start_offset
    bounded_stop = (
        ordered[last + 1].chunk_start_offset
        if last + 1 < len(ordered)
        else storage.data_end
    )
    original_start, original_stop = storage.data_start, storage.data_end
    storage.data_start, storage.data_end = bounded_start, bounded_stop
    try:
        for connection, timestamp, data in scan(
            storage_connections, start_ns, stop_ns
        ):
            yield (
                connection_by_storage_id[connection.id],
                timestamp,
                data,
            )
    finally:
        storage.data_start, storage.data_end = original_start, original_stop


def _validate_windows(
    start_ns: int, stop_ns: int, label: str
) -> tuple[int, int]:
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
    """Recover calibration evidence in one bounded message pass per bag."""
    image_window = _validate_windows(
        image_start_header_ns, image_stop_header_ns, "image window"
    )
    rtk_window = _validate_windows(
        image_window[0]
        if rtk_start_header_ns is None
        else rtk_start_header_ns,
        image_window[1] if rtk_stop_header_ns is None else rtk_stop_header_ns,
        "RTK window",
    )
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
        "log window",
    )

    topic_values = tuple(
        str(getattr(topics, item.name)) for item in fields(topics)
    )
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
                connection
                for connection in reader.connections
                if connection.topic in bag_topics
            ]
            for connection, log_ns_value, raw in _bounded_reader_messages(
                reader, connections, log_window[0], log_window[1]
            ):
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
                                "ambiguous selected-left stereo bucket "
                                f"{bucket}"
                            )
                        selected_left_buckets.add(bucket)
                        selected_left.append(
                            (
                                bucket,
                                _ImageDigest(
                                    header_ns=stamp_ns,
                                    log_ns=int(log_ns_value),
                                    sha256=_sha256_bytes(
                                        _jpeg_bytes(message)
                                    ),
                                ),
                            )
                        )
                    else:
                        if bucket in right_by_bucket:
                            raise RuntimeError(
                                f"ambiguous right stereo bucket {bucket}"
                            )
                        right_by_bucket[bucket] = _ImageDigest(
                            header_ns=stamp_ns,
                            log_ns=int(log_ns_value),
                            sha256=_sha256_bytes(_jpeg_bytes(message)),
                        )
                elif rtk_window[0] <= stamp_ns <= rtk_window[1]:
                    if topic == topics.fix:
                        fixes.append(
                            navsat_fix_observation(message, log_ns_value)
                        )
                    elif topic == topics.relpos:
                        relpos.append(
                            relpos_observation(message, log_ns_value)
                        )
                    elif topic == topics.moving_base_pvt:
                        pvt.append(
                            navpvt_status_observation(message, log_ns_value)
                        )

    if not selected_left:
        raise RuntimeError("bounded pass found no selected left images")
    if not fixes or not relpos or not pvt:
        raise RuntimeError(
            "bounded pass did not recover all RTK streams "
            f"(fix={len(fixes)}, relpos={len(relpos)}, pvt={len(pvt)})"
        )

    frames_out: list[StereoFrameObservation] = []
    image_dir = Path(extracted_images_dir)
    for frame_id, (bucket, left) in enumerate(selected_left):
        right = right_by_bucket.get(bucket)
        if right is None:
            raise RuntimeError(
                f"selected left frame {frame_id} has no right image in "
                f"historical bucket {bucket}"
            )
        left_path = image_dir / f"left_{frame_id:06d}.jpg"
        right_path = image_dir / f"right_{frame_id:06d}.jpg"
        if not left_path.is_file() or not right_path.is_file():
            raise FileNotFoundError(
                f"extracted stereo pair {frame_id} is missing in {image_dir}"
            )
        if _sha256_file(left_path) != left.sha256:
            raise ValueError(
                f"left JPEG hash mismatch at extracted frame {frame_id}"
            )
        if _sha256_file(right_path) != right.sha256:
            raise ValueError(
                f"right JPEG hash mismatch at extracted frame {frame_id}"
            )
        frames_out.append(
            StereoFrameObservation(
                frame_id=frame_id,
                left_header_ns=left.header_ns,
                left_log_ns=left.log_ns,
                right_header_ns=right.header_ns,
                right_log_ns=right.log_ns,
                left_sha256=left.sha256,
                right_sha256=right.sha256,
            )
        )

    if (
        expected_frame_count is not None
        and len(frames_out) != int(expected_frame_count)
    ):
        raise RuntimeError(
            f"recovered {len(frames_out)} stereo pairs, expected "
            f"{int(expected_frame_count)}"
        )
    extra_left = image_dir / f"left_{len(frames_out):06d}.jpg"
    extra_right = image_dir / f"right_{len(frames_out):06d}.jpg"
    if extra_left.exists() or extra_right.exists():
        raise RuntimeError(
            "extracted image directory contains more frames than recovered"
        )

    fixes.sort(key=lambda item: (item.header_ns, item.log_ns))
    relpos.sort(key=lambda item: (item.header_ns, item.log_ns))
    pvt.sort(key=lambda item: (item.header_ns, item.log_ns))
    return CalibrationBagObservations(
        stereo_frames=tuple(frames_out),
        fixes=tuple(fixes),
        relpos=tuple(relpos),
        moving_base_pvt=tuple(pvt),
        topic_bags=tuple(
            sorted((topic, str(path)) for topic, path in topic_bags.items())
        ),
        header_window_ns=image_window,
        rtk_window_ns=rtk_window,
        log_window_ns=log_window,
    )


def read_selected_calibration_bag_observations(
    *,
    camera_bag: Path,
    navigation_bag: Path,
    topics: CalibrationTopics,
    typestore,
    frame_ids: Sequence[int],
    left_header_ns: Sequence[int],
    right_header_ns: Sequence[int],
    left_image_paths: Sequence[Path],
    right_image_paths: Sequence[Path],
    rtk_start_header_ns: int,
    rtk_stop_header_ns: int,
    log_margin_ns: int = 5 * _NS_PER_S,
) -> CalibrationBagObservations:
    """Recover an explicit immutable stereo selection and nearby RTK data.

    Unlike :func:`read_calibration_bag_observations`, this entry point does
    not reproduce an historical stride or infer frame IDs from bag order.
    Every selected camera header stamp and extracted JPEG path is provided by
    the caller.  Camera evidence is read only from ``camera_bag`` and RTK
    evidence only from ``navigation_bag`` even when a recording contains
    duplicate topics.  This makes the source of each observation explicit and
    prevents bag-order precedence from silently changing an audit.
    """
    ids = np.asarray(frame_ids)
    left_ns = np.asarray(left_header_ns)
    right_ns = np.asarray(right_header_ns)
    left_paths = tuple(Path(path) for path in left_image_paths)
    right_paths = tuple(Path(path) for path in right_image_paths)
    n = len(ids)
    if not (
        ids.ndim == left_ns.ndim == right_ns.ndim == 1
        and len(left_ns) == len(right_ns) == len(left_paths)
        == len(right_paths) == n
    ):
        raise ValueError("selected stereo inputs must be equal-length vectors")
    if n == 0:
        raise ValueError("selected stereo inputs cannot be empty")
    if (
        not np.issubdtype(ids.dtype, np.integer)
        or not np.issubdtype(left_ns.dtype, np.integer)
        or not np.issubdtype(right_ns.dtype, np.integer)
    ):
        raise ValueError("selected frame IDs and timestamps must be integers")
    ids = ids.astype(np.int64, copy=False)
    left_ns = left_ns.astype(np.int64, copy=False)
    right_ns = right_ns.astype(np.int64, copy=False)
    if (
        np.any(ids < 0)
        or len(np.unique(ids)) != n
        or len(np.unique(left_ns)) != n
        or len(np.unique(right_ns)) != n
        or np.any(np.diff(ids) <= 0)
        or np.any(np.diff(left_ns) <= 0)
        or np.any(np.diff(right_ns) <= 0)
    ):
        raise ValueError(
            "selected frame IDs and timestamps must be unique and increasing"
        )
    if len(set(left_paths)) != n or len(set(right_paths)) != n:
        raise ValueError("selected extracted image paths must be unique")
    if any(not path.is_file() for path in left_paths + right_paths):
        raise FileNotFoundError("a selected extracted stereo image is missing")

    camera = Path(camera_bag).expanduser()
    navigation = Path(navigation_bag).expanduser()
    if camera.resolve() == navigation.resolve():
        raise ValueError("camera and navigation bags must be distinct inputs")
    margin = int(log_margin_ns)
    if margin < 0:
        raise ValueError("log_margin_ns cannot be negative")
    rtk_window = _validate_windows(
        rtk_start_header_ns, rtk_stop_header_ns, "RTK window"
    )
    image_window = (int(min(left_ns[0], right_ns[0])),
                    int(max(left_ns[-1], right_ns[-1])))
    log_window = (
        min(image_window[0], rtk_window[0]) - margin,
        max(image_window[1], rtk_window[1]) + margin,
    )

    expected_by_topic = {
        topics.left_image: camera,
        topics.right_image: camera,
        topics.fix: navigation,
        topics.relpos: navigation,
        topics.moving_base_pvt: navigation,
    }
    grouped: dict[Path, set[str]] = {}
    for topic, bag in expected_by_topic.items():
        grouped.setdefault(bag, set()).add(topic)
    for bag, required_topics in grouped.items():
        with Reader(bag) as reader:
            available = {item.topic for item in reader.connections}
        missing = required_topics - available
        if missing:
            raise RuntimeError(
                f"required topics are missing from {bag}: "
                + ", ".join(sorted(missing))
            )

    wanted_left = {int(value) for value in left_ns}
    wanted_right = {int(value) for value in right_ns}
    left_digests: dict[int, _ImageDigest] = {}
    right_digests: dict[int, _ImageDigest] = {}
    fixes: list[NavSatFixObservation] = []
    relpos: list[RelPosObservation] = []
    pvt: list[NavPvtStatusObservation] = []

    for bag, bag_topics in grouped.items():
        with Reader(bag) as reader:
            connections = [
                connection
                for connection in reader.connections
                if connection.topic in bag_topics
            ]
            for connection, log_ns_value, raw in _bounded_reader_messages(
                reader, connections, log_window[0], log_window[1]
            ):
                message = typestore.deserialize_cdr(raw, connection.msgtype)
                stamp_ns = header_stamp_ns(message)
                topic = connection.topic
                if topic == topics.left_image and stamp_ns in wanted_left:
                    if stamp_ns in left_digests:
                        raise RuntimeError(
                            f"duplicate selected left header stamp {stamp_ns}"
                        )
                    left_digests[stamp_ns] = _ImageDigest(
                        header_ns=stamp_ns,
                        log_ns=int(log_ns_value),
                        sha256=_sha256_bytes(_jpeg_bytes(message)),
                    )
                elif topic == topics.right_image and stamp_ns in wanted_right:
                    if stamp_ns in right_digests:
                        raise RuntimeError(
                            f"duplicate selected right header stamp {stamp_ns}"
                        )
                    right_digests[stamp_ns] = _ImageDigest(
                        header_ns=stamp_ns,
                        log_ns=int(log_ns_value),
                        sha256=_sha256_bytes(_jpeg_bytes(message)),
                    )
                elif rtk_window[0] <= stamp_ns <= rtk_window[1]:
                    if topic == topics.fix:
                        fixes.append(navsat_fix_observation(message, log_ns_value))
                    elif topic == topics.relpos:
                        relpos.append(relpos_observation(message, log_ns_value))
                    elif topic == topics.moving_base_pvt:
                        pvt.append(
                            navpvt_status_observation(message, log_ns_value)
                        )

    missing_left = wanted_left - set(left_digests)
    missing_right = wanted_right - set(right_digests)
    if missing_left or missing_right:
        raise RuntimeError(
            "selected stereo header stamps are missing from the camera bag "
            f"(left={len(missing_left)}, right={len(missing_right)})"
        )
    if not fixes or not relpos or not pvt:
        raise RuntimeError(
            "bounded navigation pass did not recover all RTK streams "
            f"(fix={len(fixes)}, relpos={len(relpos)}, pvt={len(pvt)})"
        )

    frames_out: list[StereoFrameObservation] = []
    for frame_id, left_stamp, right_stamp, left_path, right_path in zip(
        ids, left_ns, right_ns, left_paths, right_paths
    ):
        left = left_digests[int(left_stamp)]
        right = right_digests[int(right_stamp)]
        if _sha256_file(left_path) != left.sha256:
            raise ValueError(
                f"left JPEG hash mismatch at selected frame {int(frame_id)}"
            )
        if _sha256_file(right_path) != right.sha256:
            raise ValueError(
                f"right JPEG hash mismatch at selected frame {int(frame_id)}"
            )
        frames_out.append(
            StereoFrameObservation(
                frame_id=int(frame_id),
                left_header_ns=left.header_ns,
                left_log_ns=left.log_ns,
                right_header_ns=right.header_ns,
                right_log_ns=right.log_ns,
                left_sha256=left.sha256,
                right_sha256=right.sha256,
            )
        )

    fixes.sort(key=lambda item: (item.header_ns, item.log_ns))
    relpos.sort(key=lambda item: (item.header_ns, item.log_ns))
    pvt.sort(key=lambda item: (item.header_ns, item.log_ns))
    return CalibrationBagObservations(
        stereo_frames=tuple(frames_out),
        fixes=tuple(fixes),
        relpos=tuple(relpos),
        moving_base_pvt=tuple(pvt),
        topic_bags=tuple(
            sorted(
                (topic, str(path.resolve()))
                for topic, path in expected_by_topic.items()
            )
        ),
        header_window_ns=image_window,
        rtk_window_ns=rtk_window,
        log_window_ns=log_window,
    )
