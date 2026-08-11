"""Publish adapter-neutral observations as immutable segment-v2 artifacts.

This module knows the segment contract but nothing about ROS versions, bag
formats, or dataset names.  Concrete adapters provide plain records from
``rtk_splat.adapters.records`` and call ``publish_segment_v2``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from rtk_splat.adapters.image_decode import detect_compressed_format
from rtk_splat.adapters.records import (
    FrameRecord,
    RELPOS_FLAG_NAMES,
    RtkTrack,
    StagedPayload,
)
from rtk_splat.adapters.synchronization import nearest_matches
from rtk_splat.core.segment import (
    CAPABILITIES,
    CONTRACT_VERSION,
    POSITION_QUALITY_VOCABULARY,
    SegmentReader,
    SegmentWriter,
    normalize_navsat_position_quality,
)


def _payload_format(payload: bytes | Path | StagedPayload) -> str:
    """Detect an in-memory payload or staged image without loading it."""
    if isinstance(payload, StagedPayload):
        payload = payload.path
    if isinstance(payload, Path):
        if not payload.is_file():
            raise ValueError(f"staged image does not exist: {payload}")
        with payload.open("rb") as stream:
            return detect_compressed_format(stream.read(8))
    return detect_compressed_format(payload)


def _publish_image_payload(
    payload: bytes | Path | StagedPayload, destination: Path
) -> None:
    """Write one payload, consuming only explicitly disposable staged files."""
    if isinstance(payload, StagedPayload):
        source = payload.path
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"staged image must be a regular file: {source}")
        # The ROS2 spool is created beside the destination, so rename is both
        # atomic and zero-copy.  An unexpected cross-filesystem setup fails
        # loudly instead of silently doubling a large ingest's disk usage.
        source.replace(destination)
    elif isinstance(payload, Path):
        shutil.copyfile(payload, destination)
    else:
        destination.write_bytes(payload)


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
    requested: Mapping[str, bool] | None,
    dual_available: bool,
    recorded_depth_available: bool,
) -> dict[str, bool]:
    capabilities = {
        "stereo": True,
        "rgbd": False,
        "single_rtk": not dual_available,
        "dual_rtk": dual_available,
        "depth_recorded": recorded_depth_available,
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
        "depth_recorded": recorded_depth_available,
        "depth_computed": False,
        "imu_present": False,
        "images_raw": False,
        "images_rectified": True,
    }
    for name, expected in fixed.items():
        if capabilities[name] is not expected:
            raise ValueError(
                f"rectified stereo writer requires {name}={expected}"
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


def _associate_nearest_with_validity(
    query_ns: np.ndarray,
    source_ns: np.ndarray,
    tolerance_ns: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Associate every query while marking, rather than hiding, stale rows.

    This is deliberately separate from :func:`_associate_all`: absolute GNSS
    position remains fail-closed, while a dual-antenna heading stream may have
    a small, explicitly budgeted dropout.  The nearest source row and its exact
    signed residual are retained even when it is outside the validity
    tolerance; consumers must use ``association_valid``/``valid`` before
    treating that row as a heading observation.
    """
    query = np.asarray(query_ns)
    source = np.asarray(source_ns)
    if query.ndim != 1 or source.ndim != 1 or source.size == 0:
        raise ValueError(f"{label} association needs non-empty 1-D timestamps")

    # Compute this in Python integers so the bound itself cannot overflow.
    maximum_residual = max(
        abs(int(source[0]) - int(query[-1])),
        abs(int(source[-1]) - int(query[0])),
    )
    if maximum_residual > np.iinfo(np.int64).max:
        raise ValueError(f"{label} timestamp span is outside int64")
    matches = nearest_matches(
        query,
        source,
        tolerance_ns=maximum_residual,
    )
    if matches.match_count != len(query) or not np.array_equal(
        matches.reference_indices, np.arange(len(query), dtype=np.int64)
    ):
        raise RuntimeError(f"{label} association returned incomplete ordering")
    association_valid = np.abs(matches.residual_ns) <= tolerance_ns
    return matches.sample_indices, matches.residual_ns, association_valid


def _fraction(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite fraction in [0, 1]")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be a finite fraction in [0, 1]")
    return result


def _nonnegative_count(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{label} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


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
    maximum_heading_association_invalid_fraction: float = 0.0,
    maximum_heading_association_invalid_run_frames: int = 0,
    maximum_heading_association_invalid_residual_ns: int = 0,
    capabilities: Mapping[str, bool] | None = None,
    splits: Mapping[str, Sequence[int]] | None = None,
    provenance: Mapping[str, Any] | None = None,
    adapter_name: str = "ros2_zed_ublox",
    recorded_depth: Sequence[Mapping[str, Any]] | None = None,
    recorded_depth_semantics: Mapping[str, Any] | None = None,
    acquisition_id: str | None = None,
) -> SegmentReader:
    """Atomically publish adapter-selected evidence as segment contract v2.

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
    maximum_heading_invalid_fraction = _fraction(
        maximum_heading_association_invalid_fraction,
        "maximum_heading_association_invalid_fraction",
    )
    maximum_heading_invalid_run = _nonnegative_count(
        maximum_heading_association_invalid_run_frames,
        "maximum_heading_association_invalid_run_frames",
    )
    maximum_heading_invalid_residual = _nonnegative_ns(
        maximum_heading_association_invalid_residual_ns,
        "maximum_heading_association_invalid_residual_ns",
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
        has_frame_logs = [
            hasattr(frame, "left_log_ns") and hasattr(frame, "right_log_ns")
            for frame in frame_records
        ]
        if any(has_frame_logs) and not all(has_frame_logs):
            raise ValueError("frame bag-log timestamps must be present for every frame")
        left_log_values = []
        right_log_values = []
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
            left_payload = frame.left_jpeg
            right_payload = frame.right_jpeg
            if not isinstance(left_payload, (Path, StagedPayload)):
                left_payload = bytes(left_payload)
            if not isinstance(right_payload, (Path, StagedPayload)):
                right_payload = bytes(right_payload)
            left_ns_values.append(left_ns)
            right_ns_values.append(right_ns)
            left_payloads.append(left_payload)
            right_payloads.append(right_payload)
            left_formats.append(_payload_format(left_payload))
            right_formats.append(_payload_format(right_payload))
            if has_frame_logs[index]:
                left_log_values.append(
                    _exact_ns(frame.left_log_ns, f"frame {index} left_log_ns")
                )
                right_log_values.append(
                    _exact_ns(frame.right_log_ns, f"frame {index} right_log_ns")
                )
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
            _publish_image_payload(left_payload, image_dir / left_name)
            _publish_image_payload(right_payload, image_dir / right_name)
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
        if all(has_frame_logs):
            frame_values["left_log_timestamp_ns"] = np.asarray(
                left_log_values, dtype=np.int64
            )
            frame_values["right_log_timestamp_ns"] = np.asarray(
                right_log_values, dtype=np.int64
            )

        depth_available = recorded_depth is not None
        if depth_available:
            depth_records = tuple(recorded_depth)
            if len(depth_records) != n_frames:
                raise ValueError(
                    "recorded_depth must contain exactly one record per frame"
                )
            if recorded_depth_semantics is None:
                raise ValueError(
                    "recorded_depth_semantics is required with recorded_depth"
                )
            depth_directory = writer.directory("depth")
            depth_paths: list[str] = []
            depth_header_ns: list[int] = []
            depth_log_ns: list[int] = []
            depth_source_encoding: list[str] = []
            for index, record in enumerate(depth_records):
                if not isinstance(record, Mapping):
                    raise ValueError("recorded_depth records must be mappings")
                payload = record.get("payload")
                if not isinstance(payload, Path) or not payload.is_file():
                    raise ValueError(
                        f"recorded_depth[{index}].payload must be a staged file"
                    )
                header_ns = _exact_ns(
                    record.get("header_ns"),
                    f"recorded_depth[{index}].header_ns",
                )
                log_ns = _exact_ns(
                    record.get("log_ns"), f"recorded_depth[{index}].log_ns"
                )
                name = f"{index:06d}.npz"
                shutil.copyfile(payload, depth_directory / name)
                depth_paths.append(f"depth/{name}")
                depth_header_ns.append(header_ns)
                depth_log_ns.append(log_ns)
                depth_source_encoding.append(str(record.get("source_encoding", "")))
            depth_header_array = np.asarray(depth_header_ns, dtype=np.int64)
            if np.any(np.diff(depth_header_array) <= 0):
                raise ValueError("recorded depth timestamps must be strictly increasing")
            frame_values["depth_path"] = np.asarray(depth_paths, dtype=np.str_)
            frame_values["depth_timestamp_ns"] = depth_header_array
            frame_values["depth_log_timestamp_ns"] = np.asarray(
                depth_log_ns, dtype=np.int64
            )
            frame_values["depth_sync_residual_ns"] = (
                depth_header_array - left_ns_array
            )
            frame_values["depth_source_encoding"] = np.asarray(
                depth_source_encoding, dtype=np.str_
            )

        fix_count = len(fix_header_ns)
        fix_enu = np.asarray(getattr(track, "enu_xyz", None), dtype=np.float64)
        fix_covariance = np.asarray(
            track.fix_covariance_enu_m2, dtype=np.float64
        )
        raw_message_covariance = track.raw_message_covariance_enu_m2
        message_fix_covariance = np.asarray(
            fix_covariance
            if raw_message_covariance is None
            else raw_message_covariance,
            dtype=np.float64,
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
            "message_fix_covariance_enu_m2": (fix_count, 3, 3),
            "fix_status": (fix_count,),
            "fix_carrier_status": (fix_count,),
            "fix_log_ns": (fix_count,),
            "fix_covariance_type": (fix_count,),
            "fix_geodetic": (fix_count, 3),
        }
        actual_fix = {
            "enu_xyz": fix_enu,
            "fix_covariance_enu_m2": fix_covariance,
            "message_fix_covariance_enu_m2": message_fix_covariance,
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
        supplied_valid = getattr(track, "fix_position_valid", None)
        supplied_quality = getattr(track, "fix_position_quality", None)
        if (supplied_valid is None) != (supplied_quality is None):
            raise ValueError(
                "RtkTrack must supply fix_position_valid and "
                "fix_position_quality together"
            )
        if supplied_valid is not None:
            position_valid = np.asarray(supplied_valid, dtype=bool)
            position_quality = np.asarray(supplied_quality, dtype=np.str_)
            if position_valid.shape != (fix_count,) or position_quality.shape != (
                fix_count,
            ):
                raise ValueError(
                    "RtkTrack receiver-derived position quality must have one "
                    "entry per fix"
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
            "raw_covariance_enu_m2": message_fix_covariance,
            "raw_effective_covariance_enu_m2": fix_covariance,
            "raw_fix_status": fix_status,
            "raw_carrier_status": fix_carrier,
            "raw_covariance_type": fix_covariance_type,
            "pvt_header_ns": np.asarray(track.pvt_header_ns, dtype=np.int64),
            "pvt_log_ns": np.asarray(track.pvt_log_ns, dtype=np.int64),
            "pvt_carrier_status": np.asarray(
                track.pvt_carrier_status, dtype=np.int16
            ),
        }
        if track.fix_service is not None:
            service = np.asarray(track.fix_service, dtype=np.int16)
            if service.shape != (fix_count,):
                raise ValueError(
                    f"RtkTrack fix_service must have shape ({fix_count},)"
                )
            gnss["service"] = service[fix_indices]
            gnss["raw_service"] = service

        receiver_state = getattr(track, "receiver_state_evidence", None)
        if receiver_state is not None:
            fix_receiver_fields = {
                "state_index": np.asarray(
                    track.fix_receiver_state_index, dtype=np.int64
                ),
                "state_matched": np.asarray(
                    track.fix_receiver_state_matched, dtype=bool
                ),
                "state_header_timestamp_ns": np.asarray(
                    track.fix_receiver_state_header_ns, dtype=np.int64
                ),
                "state_log_timestamp_ns": np.asarray(
                    track.fix_receiver_state_log_ns, dtype=np.int64
                ),
                "state_residual_ns": np.asarray(
                    track.fix_receiver_state_residual_ns, dtype=np.int64
                ),
                "fix_mode": np.asarray(track.fix_receiver_mode, dtype=np.str_),
                "rtk_mode_fix": np.asarray(
                    track.fix_receiver_rtk_mode_fix, dtype=bool
                ),
                "num_sat": np.asarray(track.fix_receiver_num_sat, dtype=np.int16),
                "system_error": np.asarray(
                    track.fix_receiver_system_error, dtype=np.int16
                ),
                "io_error": np.asarray(
                    track.fix_receiver_io_error, dtype=np.int16
                ),
                "swift_nap_error": np.asarray(
                    track.fix_receiver_swift_nap_error, dtype=np.int16
                ),
                "external_antenna_present": np.asarray(
                    track.fix_receiver_external_antenna_present, dtype=np.int16
                ),
            }
            bad_shape = [
                name
                for name, value in fix_receiver_fields.items()
                if value.shape != (fix_count,)
            ]
            if bad_shape:
                raise ValueError(
                    "RtkTrack receiver-state fix associations have wrong shape: "
                    + ", ".join(sorted(bad_shape))
                )
            for name, value in fix_receiver_fields.items():
                gnss[f"receiver_{name}"] = value[fix_indices]
                gnss[f"raw_receiver_{name}"] = value

            state_fields = {
                "header_timestamp_ns": np.asarray(
                    receiver_state.header_ns, dtype=np.int64
                ),
                "log_timestamp_ns": np.asarray(
                    receiver_state.log_ns, dtype=np.int64
                ),
                "fix_mode": np.asarray(receiver_state.fix_mode, dtype=np.str_),
                "rtk_mode_fix": np.asarray(
                    receiver_state.rtk_mode_fix, dtype=bool
                ),
                "num_sat": np.asarray(receiver_state.num_sat, dtype=np.int16),
                "system_error": np.asarray(
                    receiver_state.system_error, dtype=np.int16
                ),
                "io_error": np.asarray(receiver_state.io_error, dtype=np.int16),
                "swift_nap_error": np.asarray(
                    receiver_state.swift_nap_error, dtype=np.int16
                ),
                "external_antenna_present": np.asarray(
                    receiver_state.external_antenna_present, dtype=np.int16
                ),
            }
            state_count = len(state_fields["header_timestamp_ns"])
            bad_state_shape = [
                name
                for name, value in state_fields.items()
                if value.shape != (state_count,)
            ]
            if bad_state_shape:
                raise ValueError(
                    "receiver-state raw evidence has inconsistent shape: "
                    + ", ".join(sorted(bad_state_shape))
                )
            for name, value in state_fields.items():
                # The prefix deliberately differs from ``raw_*``: this is a
                # second raw stream with its own timestamps and sample count.
                gnss[f"receiver_state_raw_{name}"] = value

        dual_available = _dual_evidence_available(track)
        capability_record = _capability_record(
            capabilities, dual_available, depth_available
        )
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
        heading_association_summary = None
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
            (
                heading_indices,
                heading_residual_ns,
                heading_association_valid,
            ) = _associate_nearest_with_validity(
                pose_query_ns,
                relpos_header_ns,
                association_tolerance,
                "heading",
            )
            invalid_indices = np.flatnonzero(~heading_association_valid)
            invalid_count = int(invalid_indices.size)
            invalid_fraction = invalid_count / n_frames
            longest_invalid_run = _longest_true_run(
                ~heading_association_valid
            )
            invalid_max_abs_residual = (
                int(np.max(np.abs(heading_residual_ns[invalid_indices])))
                if invalid_count
                else None
            )
            heading_association_summary = {
                "method": "nearest",
                "validity_field": "association_valid",
                "tolerance_ns": association_tolerance,
                "maximum_invalid_fraction": maximum_heading_invalid_fraction,
                "maximum_consecutive_invalid_frames": (
                    maximum_heading_invalid_run
                ),
                "maximum_invalid_abs_residual_ns": (
                    maximum_heading_invalid_residual
                ),
                "frame_count": n_frames,
                "valid_count": n_frames - invalid_count,
                "invalid_count": invalid_count,
                "invalid_fraction": invalid_fraction,
                "longest_consecutive_invalid_frames": longest_invalid_run,
                "max_abs_residual_ns": int(
                    np.max(np.abs(heading_residual_ns))
                ),
                "invalid_max_abs_residual_ns": invalid_max_abs_residual,
                "first_invalid_frame_id": (
                    int(invalid_indices[0]) if invalid_count else None
                ),
            }
            if invalid_fraction > maximum_heading_invalid_fraction:
                raise ValueError(
                    "heading association invalid fraction "
                    f"{invalid_fraction:.9f} exceeds configured maximum "
                    f"{maximum_heading_invalid_fraction:.9f} "
                    f"({invalid_count}/{n_frames} frames; tolerance "
                    f"{association_tolerance} ns; first invalid frame "
                    f"{int(invalid_indices[0])})"
                )
            if longest_invalid_run > maximum_heading_invalid_run:
                raise ValueError(
                    "heading association longest invalid run "
                    f"{longest_invalid_run} frames exceeds configured maximum "
                    f"{maximum_heading_invalid_run} frames"
                )
            if (
                invalid_max_abs_residual is not None
                and invalid_max_abs_residual
                > maximum_heading_invalid_residual
            ):
                raise ValueError(
                    "heading association invalid residual "
                    f"{invalid_max_abs_residual} ns exceeds configured maximum "
                    f"{maximum_heading_invalid_residual} ns"
                )
            if track.heading_quality_kind == "dual_position":
                raw_valid = np.asarray(track.heading_valid, dtype=bool)
                if raw_valid.shape != (relpos_count,):
                    raise ValueError(
                        "dual-position heading_valid must have one value per sample"
                    )
            else:
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
                "association_valid": heading_association_valid,
                "baseline_ned_m": baseline[heading_indices],
                "acc_heading_rad": accuracy[heading_indices],
                "carrier_status": carrier[heading_indices],
                "flags": flags[heading_indices],
                "valid": (
                    raw_valid[heading_indices] & heading_association_valid
                ),
                "raw_timestamp_ns": relpos_header_ns,
                "raw_header_timestamp_ns": relpos_header_ns,
                "raw_log_timestamp_ns": relpos_log_ns,
                "raw_baseline_ned_m": baseline,
                "raw_acc_heading_rad": accuracy,
                "raw_carrier_status": carrier,
                "raw_flags": flags,
                "raw_valid": raw_valid,
            }
            secondary = track.secondary_gnss
            if secondary is not None:
                secondary_fields = {
                    "timestamp_ns": secondary.header_ns,
                    "header_timestamp_ns": secondary.header_ns,
                    "log_timestamp_ns": secondary.log_ns,
                    "enu_m": secondary.enu_m,
                    "geodetic_deg_m": secondary.geodetic_deg_m,
                    "covariance_enu_m2": secondary.raw_covariance_enu_m2,
                    "effective_covariance_enu_m2": (
                        secondary.effective_covariance_enu_m2
                    ),
                    "fix_status": secondary.fix_status,
                    "carrier_status": secondary.carrier_status,
                    "covariance_type": secondary.covariance_type,
                    "service": secondary.service,
                }
                for name, value in secondary_fields.items():
                    heading[f"secondary_raw_{name}"] = value

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
        normalized_provenance = _json_value({} if provenance is None else provenance)
        if acquisition_id is not None:
            if (
                not isinstance(acquisition_id, str)
                or not acquisition_id
                or len(acquisition_id) > 256
                or acquisition_id.strip() != acquisition_id
                or any(character.isspace() for character in acquisition_id)
            ):
                raise ValueError(
                    "acquisition_id must be a non-empty opaque identifier"
                )
            existing_acquisition_id = normalized_provenance.get("acquisition_id")
            if existing_acquisition_id not in (None, acquisition_id):
                raise ValueError(
                    "segment provenance acquisition_id conflicts with publication"
                )
            normalized_provenance["acquisition_id"] = acquisition_id
        receiver_state_metadata = None
        if receiver_state is not None:
            receiver_state_metadata = {
                "topic": str(receiver_state.topic),
                "message_type": str(receiver_state.message_type),
                "message_definition_sha256": str(
                    receiver_state.message_definition_sha256
                ),
                "classification_field": "fix_mode",
                "fixed_cross_check_field": "rtk_mode_fix",
                "raw_stream_prefix": "receiver_state_raw_",
            }
        meta = {
            "contract_version": CONTRACT_VERSION,
            "n_frames": n_frames,
            "adapter": str(adapter_name),
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
                "frame_log_timestamps_preserved": bool(all(has_frame_logs)),
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
                "quality_source": (
                    "receiver_state.fix_mode"
                    if receiver_state is not None
                    else "NavSatStatus plus optional carrier status"
                ),
                "receiver_state_evidence": receiver_state_metadata,
                "covariance_provenance": normalized_provenance.get(
                    "gnss_quality", {}
                ).get("covariance"),
            },
            "heading_observation": (
                {
                    "vector": "primary_to_secondary",
                    "components": ["north", "east", "down"],
                    "primary_frame_id": sensor_frames["primary_antenna"],
                    "secondary_frame_id": sensor_frames["secondary_antenna"],
                    "quality_source": str(track.heading_quality_kind),
                    "association": heading_association_summary,
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
            "provenance": normalized_provenance,
        }
        if acquisition_id is not None:
            meta["acquisition_id"] = acquisition_id
        if heading_association_summary is not None:
            # Also place the measured result in provenance so downstream
            # reports that carry only provenance do not lose the dropout gate.
            meta["provenance"]["heading_association"] = (
                heading_association_summary
            )
        if depth_available:
            normalized_depth = _json_value(recorded_depth_semantics)
            if not isinstance(normalized_depth, dict):
                raise ValueError("recorded_depth_semantics must be a mapping")
            meta["depth_observation"] = normalized_depth
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
