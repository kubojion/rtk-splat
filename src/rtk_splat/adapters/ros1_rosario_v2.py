"""Bounded ROS1 adapter for Rosario Dataset v2.

The GNSS evidence source is explicit: the quality pilot uses the separately
recorded dual-M2 PPK solution, while the online differential stream is a
degraded-GNSS ablation.  The adapter never opens PGT, conventional-GNSS, IMU,
wheel-odometry, or ground-truth inputs.  The canonical segment contains the
rectified infrared stereo pair, Reach M2 position evidence from the declared
source, a synchronized two-position antenna baseline, and
recorded metric optical-z depth.  Raw RGB is sealed in a separate immutable
observation artifact for an explicit calibrated transfer stage.

The implementation deliberately lives at the adapter boundary.  No Rosario
topic name, calibration quirk, or dataset convention enters core, frontends,
or backends.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import shutil
import tempfile
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rtk_splat.adapters.image_decode import decode_raw_image
from rtk_splat.adapters.publication import publish_segment_v2
from rtk_splat.adapters.records import (
    FrameRecord,
    RELPOS_FLAG_NAMES,
    RtkTrack,
    SecondaryGnssEvidence,
)
from rtk_splat.adapters.rgb_observations import publish_rectified_rgb_observations
from rtk_splat.adapters.ros1_citrusfarm import (
    _DistanceSampler,
    _camera_info,
    _deserialize,
    _integer_ns,
    _plain_metadata,
    _reader_type,
    _resolved_camera_extrinsic,
    _register_ros1_connection_type,
    _stamp_ns,
    _validate_rectified_stereo,
    build_typestore,
    sample_header_log_offsets,
    validate_bag_chain,
)
from rtk_splat.adapters.sampling import resolve_metric_frame_spacing
from rtk_splat.core.poses import LocalEnu, pose_frames_from_extrinsic
from rtk_splat.core.segment import normalize_navsat_position_quality
from rtk_splat.core.runtime_resolution import (
    configuration_evidence,
    runtime_resolution_plain,
)


NANOSECONDS = 1_000_000_000
_ADAPTER = "ros1_rosario_v2"
_RGB_SUFFIX = ".rgb_observations"


@dataclass
class _TimedImage:
    header_ns: int
    log_ns: int
    message: Any


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise ValueError(f"{label} must be a mapping")


def validate_adapter_options(value: Any) -> dict[str, Any]:
    """Validate every Rosario-only option and reject silent spellings."""
    options = _mapping(value, "adapter_options")
    required = {
        "body_yaw_from_baseline_deg",
        "gnss_source",
        "ir_geometry_mode",
        "rgb_geometry_mode",
        "expected_baseline_m",
        "maximum_baseline_error_m",
        "minimum_valid_baseline_fraction",
        "heading_velocity_min_speed_m_s",
        "heading_velocity_max_turn_rate_deg_s",
        "maximum_heading_course_disagreement_deg",
        "gnss_window_padding_s",
        "offline_ppk_effective_sigma_m",
        "offline_ppk_effective_covariance_provenance",
        "offline_ppk_covariance_override",
        "calibration_acceptance",
        "recorded_depth_scale_m_per_unit",
        "recorded_depth_min_z_m",
        "recorded_depth_max_z_m",
        "depth_sync_tolerance_s",
        "kalibr_ir",
        "kalibr_rgb",
        "recorded_tf",
    }
    unknown = set(options) - required
    missing = required - set(options)
    if unknown or missing:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise ValueError("invalid Rosario adapter_options: " + "; ".join(details))

    finite_positive = (
        "expected_baseline_m",
        "maximum_baseline_error_m",
        "minimum_valid_baseline_fraction",
        "heading_velocity_min_speed_m_s",
        "heading_velocity_max_turn_rate_deg_s",
        "maximum_heading_course_disagreement_deg",
        "gnss_window_padding_s",
        "recorded_depth_scale_m_per_unit",
        "recorded_depth_min_z_m",
        "recorded_depth_max_z_m",
        "depth_sync_tolerance_s",
    )
    for name in finite_positive:
        number = float(options[name])
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"adapter_options.{name} must be positive and finite")
        options[name] = number
    effective_sigma = np.asarray(
        options["offline_ppk_effective_sigma_m"], dtype=np.float64
    )
    if (
        effective_sigma.shape != (3,)
        or not np.isfinite(effective_sigma).all()
        or np.any(effective_sigma <= 0)
    ):
        raise ValueError(
            "offline_ppk_effective_sigma_m must be finite positive [east,north,up]"
        )
    options["offline_ppk_effective_sigma_m"] = effective_sigma.tolist()
    if not isinstance(options["offline_ppk_effective_covariance_provenance"], str) or not options[
        "offline_ppk_effective_covariance_provenance"
    ].strip():
        raise ValueError(
            "offline_ppk_effective_covariance_provenance must be non-empty"
        )
    covariance_override = _mapping(
        options["offline_ppk_covariance_override"],
        "adapter_options.offline_ppk_covariance_override",
    )
    if set(covariance_override) != {
        "placeholder_diagonal_m2",
        "absolute_tolerance_m2",
        "on_mismatch",
    }:
        raise ValueError(
            "offline_ppk_covariance_override must contain exactly "
            "placeholder_diagonal_m2, absolute_tolerance_m2, and on_mismatch"
        )
    placeholder = np.asarray(
        covariance_override["placeholder_diagonal_m2"], dtype=np.float64
    )
    if (
        placeholder.shape != (3,)
        or not np.isfinite(placeholder).all()
        or np.any(placeholder < 0)
    ):
        raise ValueError(
            "offline PPK placeholder covariance must be finite non-negative ENU"
        )
    tolerance = float(covariance_override["absolute_tolerance_m2"])
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(
            "offline PPK placeholder covariance tolerance must be finite and non-negative"
        )
    if covariance_override["on_mismatch"] != "retain_recorded":
        raise ValueError(
            "offline PPK covariance mismatch policy must be retain_recorded"
        )
    covariance_override["placeholder_diagonal_m2"] = placeholder.tolist()
    covariance_override["absolute_tolerance_m2"] = tolerance
    options["offline_ppk_covariance_override"] = covariance_override
    acceptance = _mapping(
        options["calibration_acceptance"],
        "adapter_options.calibration_acceptance",
    )
    if set(acceptance) != {
        "recorded_stereo",
        "recorded_depth",
        "depth_left_camera_info",
        "depth_left_static_tf",
    }:
        raise ValueError(
            "calibration_acceptance must contain exactly recorded_stereo, "
            "recorded_depth, depth_left_camera_info, and depth_left_static_tf"
        )
    stereo_acceptance = _mapping(
        acceptance["recorded_stereo"],
        "adapter_options.calibration_acceptance.recorded_stereo",
    )
    stereo_keys = {
        "maximum_median_sample_p95_vertical_residual_px",
        "minimum_fraction_under_1px_per_sample",
        "minimum_fraction_under_1px_aggregate",
        "minimum_geometric_matches_per_sample",
    }
    if set(stereo_acceptance) != stereo_keys:
        raise ValueError(
            "recorded_stereo acceptance must contain exactly "
            + ", ".join(sorted(stereo_keys))
        )
    maximum_p95 = float(
        stereo_acceptance[
            "maximum_median_sample_p95_vertical_residual_px"
        ]
    )
    if not math.isfinite(maximum_p95) or maximum_p95 <= 0:
        raise ValueError("recorded stereo p95 gate must be positive and finite")
    stereo_acceptance[
        "maximum_median_sample_p95_vertical_residual_px"
    ] = maximum_p95
    for name in (
        "minimum_fraction_under_1px_per_sample",
        "minimum_fraction_under_1px_aggregate",
    ):
        fraction = float(stereo_acceptance[name])
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError(f"recorded stereo {name} must be in (0,1]")
        stereo_acceptance[name] = fraction
    minimum_matches = stereo_acceptance[
        "minimum_geometric_matches_per_sample"
    ]
    if isinstance(minimum_matches, bool) or not isinstance(
        minimum_matches, (int, np.integer)
    ):
        raise ValueError("recorded stereo minimum matches must be an integer")
    if int(minimum_matches) < 20:
        raise ValueError("recorded stereo minimum matches must be at least 20")
    stereo_acceptance["minimum_geometric_matches_per_sample"] = int(
        minimum_matches
    )
    depth_acceptance = _mapping(
        acceptance["depth_left_camera_info"],
        "adapter_options.calibration_acceptance.depth_left_camera_info",
    )
    if set(depth_acceptance) != {"maximum_absolute_parameter_difference"}:
        raise ValueError(
            "depth_left_camera_info acceptance must contain exactly "
            "maximum_absolute_parameter_difference"
        )
    geometry_tolerance = float(
        depth_acceptance["maximum_absolute_parameter_difference"]
    )
    if not math.isfinite(geometry_tolerance) or geometry_tolerance < 0:
        raise ValueError(
            "depth/left CameraInfo tolerance must be finite and non-negative"
        )
    depth_acceptance[
        "maximum_absolute_parameter_difference"
    ] = geometry_tolerance
    recorded_depth_acceptance = _mapping(
        acceptance["recorded_depth"],
        "adapter_options.calibration_acceptance.recorded_depth",
    )
    if set(recorded_depth_acceptance) != {
        "maximum_disparity_error_p95_px_per_sample",
        "minimum_valid_pixels_per_sample",
    }:
        raise ValueError(
            "recorded_depth acceptance must contain exactly "
            "maximum_disparity_error_p95_px_per_sample and "
            "minimum_valid_pixels_per_sample"
        )
    depth_p95 = float(
        recorded_depth_acceptance[
            "maximum_disparity_error_p95_px_per_sample"
        ]
    )
    if not math.isfinite(depth_p95) or depth_p95 <= 0:
        raise ValueError("recorded depth disparity p95 gate must be positive")
    minimum_pixels = recorded_depth_acceptance[
        "minimum_valid_pixels_per_sample"
    ]
    if isinstance(minimum_pixels, bool) or not isinstance(
        minimum_pixels, (int, np.integer)
    ):
        raise ValueError("recorded depth minimum valid pixels must be an integer")
    if int(minimum_pixels) < 1000:
        raise ValueError("recorded depth minimum valid pixels must be at least 1000")
    recorded_depth_acceptance[
        "maximum_disparity_error_p95_px_per_sample"
    ] = depth_p95
    recorded_depth_acceptance["minimum_valid_pixels_per_sample"] = int(
        minimum_pixels
    )
    static_tf_acceptance = _mapping(
        acceptance["depth_left_static_tf"],
        "adapter_options.calibration_acceptance.depth_left_static_tf",
    )
    if set(static_tf_acceptance) != {
        "maximum_translation_m",
        "maximum_rotation_deg",
        "on_missing",
    }:
        raise ValueError(
            "depth_left_static_tf acceptance must contain exactly "
            "maximum_translation_m, maximum_rotation_deg, and on_missing"
        )
    for name in ("maximum_translation_m", "maximum_rotation_deg"):
        tolerance = float(static_tf_acceptance[name])
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"depth/left static TF {name} must be non-negative")
        static_tf_acceptance[name] = tolerance
    if static_tf_acceptance["on_missing"] != "fail":
        raise ValueError("depth/left static TF on_missing policy must be fail")
    acceptance["recorded_stereo"] = stereo_acceptance
    acceptance["recorded_depth"] = recorded_depth_acceptance
    acceptance["depth_left_camera_info"] = depth_acceptance
    acceptance["depth_left_static_tf"] = static_tf_acceptance
    options["calibration_acceptance"] = acceptance
    options["body_yaw_from_baseline_deg"] = float(
        options["body_yaw_from_baseline_deg"]
    )
    if not math.isfinite(options["body_yaw_from_baseline_deg"]):
        raise ValueError("body_yaw_from_baseline_deg must be finite")
    if options["gnss_source"] not in {"online_differential", "offline_ppk"}:
        raise ValueError(
            "adapter_options.gnss_source must be online_differential or offline_ppk"
        )
    if options["ir_geometry_mode"] != "recorded_rectified_p":
        raise ValueError(
            "initial Rosario support requires ir_geometry_mode=recorded_rectified_p"
        )
    if options["rgb_geometry_mode"] != "recorded_camera_info":
        raise ValueError(
            "initial Rosario support requires rgb_geometry_mode=recorded_camera_info"
        )
    if not 0 < options["minimum_valid_baseline_fraction"] <= 1:
        raise ValueError("minimum_valid_baseline_fraction must be in (0,1]")
    if options["recorded_depth_max_z_m"] <= options["recorded_depth_min_z_m"]:
        raise ValueError("recorded depth max_z must exceed min_z")

    ir = _mapping(options["kalibr_ir"], "adapter_options.kalibr_ir")
    ir_required = {
        "left_intrinsics",
        "left_distortion",
        "right_intrinsics",
        "right_distortion",
        "T_right_left",
        "source_url",
        "sha256",
    }
    if set(ir) != ir_required:
        raise ValueError("kalibr_ir must contain exactly " + ", ".join(sorted(ir_required)))
    rgb = _mapping(options["kalibr_rgb"], "adapter_options.kalibr_rgb")
    rgb_required = {
        "intrinsics",
        "distortion",
        "T_rgb_left",
        "source_url",
        "sha256",
        "extrinsic_translation_sigma_m",
    }
    if set(rgb) != rgb_required:
        raise ValueError(
            "kalibr_rgb must contain exactly " + ", ".join(sorted(rgb_required))
        )
    recorded_tf = _mapping(options["recorded_tf"], "adapter_options.recorded_tf")
    if set(recorded_tf) != {
        "topic",
        "static_topic",
        "comparison_child_frame",
        "primary_antenna_in_base_m",
    }:
        raise ValueError(
            "recorded_tf must contain topic, static_topic, comparison_child_frame, "
            "and primary_antenna_in_base_m"
        )
    base_antenna = np.asarray(recorded_tf["primary_antenna_in_base_m"], dtype=float)
    if base_antenna.shape != (3,) or not np.isfinite(base_antenna).all():
        raise ValueError("recorded_tf.primary_antenna_in_base_m must be finite xyz")

    def _intrinsics(candidate: Any, label: str) -> list[float]:
        array = np.asarray(candidate, dtype=np.float64)
        if array.shape != (4,) or not np.isfinite(array).all() or np.any(array[:2] <= 0):
            raise ValueError(f"{label} must be finite [fx,fy,cx,cy]")
        return array.tolist()

    ir["left_intrinsics"] = _intrinsics(ir["left_intrinsics"], "kalibr_ir.left_intrinsics")
    ir["right_intrinsics"] = _intrinsics(ir["right_intrinsics"], "kalibr_ir.right_intrinsics")
    rgb["intrinsics"] = _intrinsics(rgb["intrinsics"], "kalibr_rgb.intrinsics")
    for candidate, label in (
        (ir["left_distortion"], "kalibr_ir.left_distortion"),
        (ir["right_distortion"], "kalibr_ir.right_distortion"),
        (rgb["distortion"], "kalibr_rgb.distortion"),
    ):
        array = np.asarray(candidate, dtype=np.float64)
        if array.shape not in {(4,), (5,)} or not np.isfinite(array).all():
            raise ValueError(f"{label} must contain four or five finite coefficients")
    for owner, key, label in (
        (ir, "T_right_left", "kalibr_ir.T_right_left"),
        (rgb, "T_rgb_left", "kalibr_rgb.T_rgb_left"),
    ):
        transform = np.asarray(owner[key], dtype=np.float64)
        if (
            transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-5)
        ):
            raise ValueError(f"{label} must be a rigid 4x4 transform")
    sigma = np.asarray(rgb["extrinsic_translation_sigma_m"], dtype=np.float64)
    if sigma.shape != (3,) or not np.isfinite(sigma).all() or np.any(sigma < 0):
        raise ValueError("kalibr_rgb.extrinsic_translation_sigma_m is invalid")
    for source, label in ((ir, "kalibr_ir"), (rgb, "kalibr_rgb")):
        if not isinstance(source["source_url"], str) or not source["source_url"]:
            raise ValueError(f"{label}.source_url must be non-empty")
        digest = str(source["sha256"])
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError(f"{label}.sha256 must be a lowercase SHA-256")
    options["kalibr_ir"] = ir
    options["kalibr_rgb"] = rgb
    options["recorded_tf"] = recorded_tf
    return options


def _configured_bag(cfg: Any) -> Path:
    values = getattr(cfg.paths, "bags", None)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("paths.bags must be an explicit one-item list")
    if len(values) != 1:
        raise ValueError("Rosario v2 adapter currently requires exactly one main bag")
    bag = Path(str(values[0])).expanduser()
    if not bag.is_file():
        raise ValueError(f"missing Rosario ROS1 bag: {bag}")
    lowered = bag.name.lower()
    forbidden = ("ppk", "pgt", "ground", "conventional")
    if any(token in lowered for token in forbidden):
        raise ValueError("Rosario ingest refuses evaluation/oracle GNSS bag paths")
    return bag


def _configured_gnss_bag(cfg: Any, options: Mapping[str, Any], main_bag: Path) -> Path:
    if options["gnss_source"] == "online_differential":
        return main_bag
    values = getattr(cfg.paths, "gnss_bags", None)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 1:
        raise ValueError("offline_ppk requires paths.gnss_bags as a one-item list")
    bag = Path(str(values[0])).expanduser()
    if not bag.is_file():
        raise ValueError(f"missing Rosario PPK bag: {bag}")
    lower = bag.name.lower()
    if "ppk" not in lower or any(token in lower for token in ("pgt", "conventional")):
        raise ValueError("offline_ppk accepts only the Rosario *_ppk_gnss.bag add-on")
    return bag


def _navsat_position_quality(
    status: np.ndarray,
    finite_evidence: np.ndarray,
    source_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize online status while keeping offline PPK quality conservative."""
    status_array = np.asarray(status)
    finite = np.asarray(finite_evidence)
    if source_mode == "online_differential":
        return normalize_navsat_position_quality(
            status_array,
            np.full(status_array.shape, -1, dtype=np.int8),
            finite,
        )
    if source_mode != "offline_ppk":
        raise ValueError("unknown Rosario GNSS source mode")
    valid = finite & (status_array >= 0)
    quality = np.full(status_array.shape, "invalid", dtype="<U13")
    quality[valid] = "unknown_valid"
    return valid, quality


def _read_navsat_window(
    bag: Path,
    topic: str,
    typestore: Any,
    *,
    start_log_ns: int,
    stop_log_ns: int,
    source_mode: str,
) -> RtkTrack:
    Reader = _reader_type()
    header: list[int] = []
    logs: list[int] = []
    geodetic: list[tuple[float, float, float]] = []
    status: list[int] = []
    service: list[int] = []
    covariance: list[np.ndarray] = []
    covariance_type: list[int] = []
    with Reader(bag) as reader:
        connections = [item for item in reader.connections if item.topic == topic]
        if not connections:
            raise RuntimeError(f"GNSS bag has no NavSatFix topic {topic}")
        for connection, log_ns, raw in reader.messages(
            connections=connections, start=start_log_ns, stop=stop_log_ns
        ):
            message = _deserialize(typestore, raw, connection.msgtype)
            header.append(_stamp_ns(message))
            logs.append(int(log_ns))
            geodetic.append(
                (float(message.latitude), float(message.longitude), float(message.altitude))
            )
            status.append(int(message.status.status))
            service.append(int(message.status.service))
            covariance.append(
                np.asarray(message.position_covariance, dtype=np.float64).reshape(3, 3)
            )
            covariance_type.append(int(message.position_covariance_type))
    if len(header) < 5:
        raise RuntimeError(f"fewer than five online fixes on {topic} in the window")
    header_array = np.asarray(header, dtype=np.int64)
    log_array = np.asarray(logs, dtype=np.int64)
    if np.any(np.diff(header_array) <= 0) or np.any(np.diff(log_array) <= 0):
        raise RuntimeError(f"{topic} timestamps are not strictly increasing")
    geo = np.asarray(geodetic, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    valid_cov = (
        np.isfinite(cov).all(axis=(1, 2))
        & (np.diagonal(cov, axis1=1, axis2=2) >= 0).all(axis=1)
        & np.isclose(cov, np.swapaxes(cov, 1, 2), atol=1e-9).all(axis=(1, 2))
    )
    if not valid_cov.all():
        raise RuntimeError(f"{topic} contains invalid covariance evidence")
    placeholder = np.array([-1], dtype=np.int64)
    status_array = np.asarray(status, dtype=np.int16)
    position_valid, position_quality = _navsat_position_quality(
        status_array,
        np.isfinite(geo).all(axis=1) & valid_cov,
        source_mode,
    )
    # Rosario has no receiver carrier-state field. Preserve that absence and
    # never promote a PPK fix to the portable ``rtk_fixed`` label.
    track = RtkTrack(
        fix_t=header_array.astype(np.float64) * 1e-9,
        fix_lat=geo[:, 0],
        fix_lon=geo[:, 1],
        fix_alt=geo[:, 2],
        fix_status=status_array,
        fix_cov_max=np.max(np.diagonal(cov, axis1=1, axis2=2), axis=1),
        relpos_t=placeholder.astype(np.float64),
        relpos_yaw=np.zeros(1, dtype=np.float64),
        relpos_carr=np.full(1, -1, dtype=np.int8),
        fix_header_ns=header_array,
        fix_log_ns=log_array,
        fix_covariance_enu_m2=cov,
        fix_covariance_type=np.asarray(covariance_type, dtype=np.int16),
        fix_carrier_status=np.full(len(header_array), -1, dtype=np.int8),
        fix_service=np.asarray(service, dtype=np.int16),
        fix_position_valid=position_valid,
        fix_position_quality=position_quality,
        relpos_header_ns=placeholder,
        relpos_log_ns=placeholder,
    )
    return track


def _circular_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def _attach_dual_position_heading(
    primary: RtkTrack, secondary: RtkTrack, options: Mapping[str, Any]
) -> dict[str, Any]:
    origin_index = int(np.flatnonzero(primary.fix_position_valid)[0])
    enu = LocalEnu(
        primary.fix_lat[origin_index],
        primary.fix_lon[origin_index],
        primary.fix_alt[origin_index],
    )
    primary.enu_xyz = enu.to_enu(primary.fix_lat, primary.fix_lon, primary.fix_alt)
    secondary.enu_xyz = enu.to_enu(
        secondary.fix_lat, secondary.fix_lon, secondary.fix_alt
    )
    primary.enu = enu
    primary.origin = {
        "lat0": float(primary.fix_lat[origin_index]),
        "lon0": float(primary.fix_lon[origin_index]),
        "alt0": float(primary.fix_alt[origin_index]),
    }
    p_ns = np.asarray(primary.fix_header_ns, dtype=np.int64)
    s_ns = np.asarray(secondary.fix_header_ns, dtype=np.int64)
    primary_position_valid = np.asarray(primary.fix_position_valid, dtype=bool)
    secondary_position_valid = np.asarray(secondary.fix_position_valid, dtype=bool)
    if primary_position_valid.shape != p_ns.shape:
        raise RuntimeError("primary GNSS validity does not match its fix stream")
    if secondary_position_valid.shape != s_ns.shape:
        raise RuntimeError("secondary GNSS validity does not match its fix stream")
    primary_status = np.asarray(primary.fix_status, dtype=np.int16)
    secondary_status = np.asarray(secondary.fix_status, dtype=np.int16)
    inside = (p_ns >= s_ns[0]) & (p_ns <= s_ns[-1])
    if np.count_nonzero(inside) < 5:
        raise RuntimeError("Reach1/Reach2 fixes have no useful overlap")
    p_index = np.flatnonzero(inside)
    query_ns = p_ns[p_index]
    upper = np.searchsorted(s_ns, query_ns, side="left")
    exact = s_ns[upper] == query_ns
    lower = np.where(exact, upper, upper - 1)
    alpha = np.zeros(len(query_ns), dtype=np.float64)
    between = ~exact
    alpha[between] = (
        (query_ns[between] - s_ns[lower[between]])
        / (s_ns[upper[between]] - s_ns[lower[between]])
    )
    secondary_brackets_valid = (
        secondary_position_valid[lower]
        & secondary_position_valid[upper]
        & (secondary_status[lower] >= 0)
        & (secondary_status[upper] >= 0)
    )
    source_valid = (
        primary_position_valid[p_index]
        & (primary_status[p_index] >= 0)
        & secondary_brackets_valid
    )
    secondary_xyz = np.asarray(secondary.enu_xyz, dtype=np.float64)
    secondary_at_primary = (
        secondary_xyz[lower] * (1.0 - alpha[:, None])
        + secondary_xyz[upper] * alpha[:, None]
    )
    # An invalid source bracket is never allowed to produce a baseline, even
    # if its numeric latitude/longitude fields happen to be finite.
    secondary_at_primary[~source_valid] = np.nan
    baseline_enu = secondary_at_primary - primary.enu_xyz[p_index]
    baseline_ned = np.column_stack(
        (baseline_enu[:, 1], baseline_enu[:, 0], -baseline_enu[:, 2])
    )
    length = np.linalg.norm(baseline_ned, axis=1)
    expected = float(options["expected_baseline_m"])
    valid = (
        source_valid
        & np.isfinite(baseline_ned).all(axis=1)
        & (np.abs(length - expected) <= float(options["maximum_baseline_error_m"]))
    )
    valid_fraction = float(np.mean(valid))
    if valid_fraction < float(options["minimum_valid_baseline_fraction"]):
        raise RuntimeError(
            f"Reach1/Reach2 baseline valid fraction {valid_fraction:.3f} "
            "is below the configured minimum"
        )
    baseline_yaw = np.arctan2(baseline_enu[:, 1], baseline_enu[:, 0])
    yaw_offset = np.deg2rad(float(options["body_yaw_from_baseline_deg"]))
    valid_index = np.flatnonzero(valid)
    if len(valid_index) < 5:
        raise RuntimeError("fewer than five valid dual-position headings")
    valid_body_yaw = np.unwrap(baseline_yaw[valid] + yaw_offset)
    # Invalid entries receive only a numerical fill for smoothing. Their
    # heading_valid=false gate prevents these values from producing a pose.
    body_yaw = np.interp(np.arange(len(valid)), valid_index, valid_body_yaw)

    position = primary.enu_xyz[p_index]
    times = primary.fix_t[p_index]
    east_rate = np.gradient(position[:, 0], times)
    north_rate = np.gradient(position[:, 1], times)
    speed = np.hypot(east_rate, north_rate)
    course = np.unwrap(np.arctan2(north_rate, east_rate))
    course_rate = np.abs(np.gradient(course, times))
    straight = (
        valid
        & (speed >= float(options["heading_velocity_min_speed_m_s"]))
        & (
            course_rate
            <= np.deg2rad(float(options["heading_velocity_max_turn_rate_deg_s"]))
        )
    )
    if np.count_nonzero(straight) < 5:
        raise RuntimeError("too few straight/high-speed samples to verify heading sign")
    disagreement = np.abs(_circular_difference(body_yaw[straight], course[straight]))
    median_disagreement_deg = float(np.degrees(np.median(disagreement)))
    wrong_sign_yaw = baseline_yaw - yaw_offset
    wrong_disagreement_deg = float(
        np.degrees(
            np.median(np.abs(_circular_difference(wrong_sign_yaw[straight], course[straight])))
        )
    )
    limit = float(options["maximum_heading_course_disagreement_deg"])
    if median_disagreement_deg > limit:
        hint = (
            " (approximately 180 degrees: likely baseline sign/frame reversal)"
            if median_disagreement_deg > 150
            else ""
        )
        raise RuntimeError(
            f"dual-position heading disagrees with straight-line velocity by "
            f"{median_disagreement_deg:.2f} deg (limit {limit:.2f}){hint}"
        )
    choose_lower = (
        np.abs(s_ns[lower] - query_ns)
        <= np.abs(s_ns[upper] - query_ns)
    )
    nearest_secondary = np.where(choose_lower, lower, upper)
    raw_receiver_skew = s_ns[nearest_secondary] - query_ns

    primary.relpos_header_ns = p_ns[p_index]
    primary.relpos_log_ns = np.asarray(primary.fix_log_ns, dtype=np.int64)[p_index]
    primary.relpos_t = primary.relpos_header_ns.astype(np.float64) * 1e-9
    primary.relpos_ned_m = baseline_ned
    primary.relpos_yaw = body_yaw
    primary.relpos_carr = np.full(len(p_index), -1, dtype=np.int8)
    # No receiver-provided relative-heading covariance exists. Propagate the
    # two effective horizontal position variances conservatively and label it.
    p_cov = np.asarray(primary.fix_covariance_enu_m2)[p_index]
    s_cov = np.asarray(secondary.fix_covariance_enu_m2, dtype=np.float64)
    secondary_covariance = (
        s_cov[lower] * (1.0 - alpha[:, None, None])
        + s_cov[upper] * alpha[:, None, None]
    )
    s_var_e = secondary_covariance[:, 0, 0]
    s_var_n = secondary_covariance[:, 1, 1]
    perpendicular_var = np.maximum(p_cov[:, 0, 0] + s_var_e, p_cov[:, 1, 1] + s_var_n)
    primary.relpos_acc_heading_rad = np.sqrt(perpendicular_var) / np.maximum(length, 1e-6)
    primary.relpos_acc_heading_rad[~valid] = np.nan
    primary.relpos_flags = np.zeros((len(p_index), len(RELPOS_FLAG_NAMES)), dtype=bool)
    primary.heading_valid = valid
    primary.heading_quality_kind = "dual_position"

    return {
        "association": "linear interpolation of Reach2 ENU at exact Reach1 headers",
        "primary_sample_count": int(len(primary.fix_t)),
        "secondary_sample_count": int(len(secondary.fix_t)),
        "heading_sample_count": int(len(p_index)),
        "valid_heading_sample_count": int(np.count_nonzero(valid)),
        "validity_rejections": {
            "primary_fix": int(
                np.count_nonzero(~primary_position_valid[p_index])
            ),
            "secondary_bracket": int(
                np.count_nonzero(~secondary_brackets_valid)
            ),
            "baseline_geometry": int(
                np.count_nonzero(source_valid & ~valid)
            ),
        },
        "receiver_header_skew_ns": {
            "median": int(np.rint(np.median(raw_receiver_skew))),
            "minimum": int(np.min(raw_receiver_skew)),
            "maximum": int(np.max(raw_receiver_skew)),
            "semantics": "nearest raw Reach2 header minus Reach1 header; interpolation does not erase this skew",
        },
        "baseline_length_m": {
            "median": float(np.median(length[source_valid])),
            "p05": float(np.percentile(length[source_valid], 5)),
            "p95": float(np.percentile(length[source_valid], 95)),
            "expected": expected,
        },
        "valid_fraction": valid_fraction,
        "heading_convention": {
            "baseline": "primary Reach1 to secondary Reach2",
            "body_yaw_from_baseline_deg": float(options["body_yaw_from_baseline_deg"]),
            "verified_straight_sample_count": int(np.count_nonzero(straight)),
            "median_course_disagreement_deg": median_disagreement_deg,
            "opposite_sign_disagreement_deg": wrong_disagreement_deg,
        },
        "carrier_state": "unavailable; raw carrier_status=-1",
        "heading_accuracy": "conservative effective-covariance propagation; receiver cross-covariance unavailable",
    }


def _probe_messages(
    bag: Path,
    topics: Sequence[str],
    typestore: Any,
    probe_log_ns: int,
    *,
    reference_topic: str,
    required_header_gates_ns: Mapping[str, int],
) -> dict[str, Any]:
    """Build one calibration bundle around a real reference-camera header.

    ROS bag write order is not a synchronization contract: at an arbitrary
    log-time boundary the first depth message can belong to the following
    camera frame.  First select a reference image, then search a small bounded
    log-time neighbourhood and choose every other topic by nearest *sensor
    header*.  Only streams that are expected to be synchronous are gated;
    asynchronous RGB is retained so its measured residual remains evidence.
    """
    Reader = _reader_type()
    if reference_topic not in topics:
        raise ValueError("calibration probe reference topic is not requested")

    reference: tuple[int, Any] | None = None
    with Reader(bag) as reader:
        connections = [
            item for item in reader.connections if item.topic == reference_topic
        ]
        for connection, log_ns, raw in reader.messages(
            connections=connections,
            start=probe_log_ns,
            stop=probe_log_ns + 2 * NANOSECONDS,
        ):
            reference = (
                int(log_ns),
                _deserialize(typestore, raw, connection.msgtype),
            )
            break
    if reference is None:
        raise RuntimeError(f"probe lacks reference topic: {reference_topic}")

    reference_log_ns, reference_message = reference
    reference_header_ns = _stamp_ns(reference_message)
    result: dict[str, Any] = {reference_topic: reference}
    residuals: dict[str, int] = {reference_topic: 0}
    search_radius_ns = 250_000_000
    requested = set(topics) - {reference_topic}
    with Reader(bag) as reader:
        connections = [item for item in reader.connections if item.topic in requested]
        for connection, log_ns, raw in reader.messages(
            connections=connections,
            start=reference_log_ns - search_radius_ns,
            stop=reference_log_ns + search_radius_ns,
        ):
            message = _deserialize(typestore, raw, connection.msgtype)
            residual = abs(_stamp_ns(message) - reference_header_ns)
            if connection.topic not in residuals or residual < residuals[connection.topic]:
                residuals[connection.topic] = residual
                result[connection.topic] = (int(log_ns), message)

    missing = set(topics) - set(result)
    if missing:
        raise RuntimeError("probe lacks topics: " + ", ".join(sorted(missing)))
    for topic, gate_ns in required_header_gates_ns.items():
        if topic not in result:
            raise RuntimeError(f"probe gate names an unavailable topic: {topic}")
        residual = abs(_stamp_ns(result[topic][1]) - reference_header_ns)
        if residual > int(gate_ns):
            raise RuntimeError(
                f"calibration probe {topic} header residual {residual} ns exceeds "
                f"its {int(gate_ns)} ns gate"
            )
    return result


def _decode(message: Any) -> np.ndarray:
    return decode_raw_image(
        message.data,
        width=int(message.width),
        height=int(message.height),
        encoding=str(message.encoding),
        step=int(message.step),
        is_bigendian=int(message.is_bigendian),
    )


def _camera_matrix(intrinsics: Sequence[float]) -> np.ndarray:
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=np.float64)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _stereo_epipolar_score(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    detector = cv2.SIFT_create(nfeatures=3000)
    left_keys, left_desc = detector.detectAndCompute(left, None)
    right_keys, right_desc = detector.detectAndCompute(right, None)
    if left_desc is None or right_desc is None:
        raise RuntimeError("stereo calibration probe has no SIFT descriptors")
    matches = [
        first
        for first, second in cv2.BFMatcher(cv2.NORM_L2).knnMatch(
            left_desc, right_desc, k=2
        )
        if first.distance < 0.7 * second.distance
    ]
    if len(matches) < 30:
        raise RuntimeError("too few stereo feature matches for calibration audit")
    p_left = np.float32([left_keys[item.queryIdx].pt for item in matches])
    p_right = np.float32([right_keys[item.trainIdx].pt for item in matches])
    _, mask = cv2.findFundamentalMat(
        p_left, p_right, cv2.USAC_MAGSAC, 1.0, 0.999, 10_000
    )
    if mask is None:
        raise RuntimeError("stereo calibration fundamental fit failed")
    disparity = p_left[:, 0] - p_right[:, 0]
    keep = mask.ravel().astype(bool) & (disparity > 0) & (disparity < 256)
    if np.count_nonzero(keep) < 20:
        raise RuntimeError("too few geometrically valid stereo matches")
    residual = np.abs(p_left[keep, 1] - p_right[keep, 1])
    return {
        "match_count": int(np.count_nonzero(keep)),
        "vertical_residual_median_px": float(np.median(residual)),
        "vertical_residual_p95_px": float(np.percentile(residual, 95)),
        "fraction_under_1px": float(np.mean(residual <= 1.0)),
    }


def _enforce_recorded_stereo_acceptance(
    scores: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed when real-image rectification evidence misses its gates."""
    if not scores:
        raise RuntimeError("recorded stereo acceptance has no real-image samples")
    counts = np.asarray([item["match_count"] for item in scores], dtype=np.int64)
    p95 = np.asarray(
        [item["vertical_residual_p95_px"] for item in scores], dtype=np.float64
    )
    fractions = np.asarray(
        [item["fraction_under_1px"] for item in scores], dtype=np.float64
    )
    if (
        np.any(counts < 0)
        or not np.isfinite(p95).all()
        or not np.isfinite(fractions).all()
        or np.any((fractions < 0) | (fractions > 1))
    ):
        raise RuntimeError("recorded stereo acceptance evidence is malformed")
    total_matches = int(np.sum(counts))
    aggregate_fraction = (
        float(np.sum(counts * fractions) / total_matches)
        if total_matches > 0
        else 0.0
    )
    observed = {
        "sample_count": len(scores),
        "minimum_geometric_matches_per_sample": int(np.min(counts)),
        "total_geometric_matches": total_matches,
        "median_sample_p95_vertical_residual_px": float(np.median(p95)),
        "minimum_fraction_under_1px_per_sample": float(np.min(fractions)),
        "fraction_under_1px_aggregate": aggregate_fraction,
    }
    thresholds = {
        name: (
            int(value)
            if name == "minimum_geometric_matches_per_sample"
            else float(value)
        )
        for name, value in gates.items()
    }
    failures = []
    if observed["minimum_geometric_matches_per_sample"] < thresholds[
        "minimum_geometric_matches_per_sample"
    ]:
        failures.append("insufficient geometric matches")
    if observed["median_sample_p95_vertical_residual_px"] > thresholds[
        "maximum_median_sample_p95_vertical_residual_px"
    ]:
        failures.append("median sample p95 vertical residual")
    if observed["minimum_fraction_under_1px_per_sample"] < thresholds[
        "minimum_fraction_under_1px_per_sample"
    ]:
        failures.append("per-sample fraction under 1 px")
    if observed["fraction_under_1px_aggregate"] < thresholds[
        "minimum_fraction_under_1px_aggregate"
    ]:
        failures.append("aggregate fraction under 1 px")
    if failures:
        raise RuntimeError(
            "recorded stereo real-image acceptance failed: "
            + ", ".join(failures)
        )
    return {"passed": True, "thresholds": thresholds, "observed": observed}


def _enforce_depth_left_camera_info_alignment(
    left_info: Mapping[str, Any],
    depth_info: Mapping[str, Any],
    maximum_absolute_difference: float,
) -> dict[str, Any]:
    """Verify that recorded optical-z depth uses the selected left geometry."""
    left_dimensions = [int(left_info["width"]), int(left_info["height"])]
    depth_dimensions = [int(depth_info["width"]), int(depth_info["height"])]
    dimensions_match = left_dimensions == depth_dimensions
    maximum_differences: dict[str, float | None] = {}
    shape_matches: dict[str, bool] = {}
    failed_fields = []
    for label, key in (("K", "k"), ("P", "p"), ("R", "r"), ("D", "d")):
        left = np.asarray(left_info[key], dtype=np.float64)
        depth = np.asarray(depth_info[key], dtype=np.float64)
        shape_matches[label] = left.shape == depth.shape
        if not shape_matches[label]:
            maximum_differences[label] = None
            failed_fields.append(label)
            continue
        difference = float(np.max(np.abs(left - depth))) if left.size else 0.0
        maximum_differences[label] = difference
        if (
            not np.isfinite(left).all()
            or not np.isfinite(depth).all()
            or difference > maximum_absolute_difference
        ):
            failed_fields.append(label)
    if not dimensions_match:
        failed_fields.append("dimensions")
    report = {
        "passed": not failed_fields,
        "left_dimensions": left_dimensions,
        "depth_dimensions": depth_dimensions,
        "dimensions_match": dimensions_match,
        "shape_matches": shape_matches,
        "maximum_absolute_differences": maximum_differences,
        "maximum_allowed_absolute_difference": float(
            maximum_absolute_difference
        ),
    }
    if failed_fields:
        raise RuntimeError(
            "recorded depth CameraInfo is not aligned to selected infra1 geometry: "
            + ", ".join(failed_fields)
        )
    return report


def _kalibr_rectification(options: Mapping[str, Any], size: tuple[int, int]):
    ir = options["kalibr_ir"]
    left_k = _camera_matrix(ir["left_intrinsics"])
    right_k = _camera_matrix(ir["right_intrinsics"])
    transform = np.asarray(ir["T_right_left"], dtype=np.float64)
    left_r, right_r, left_p, right_p, _, _, _ = cv2.stereoRectify(
        left_k,
        np.asarray(ir["left_distortion"], dtype=np.float64),
        right_k,
        np.asarray(ir["right_distortion"], dtype=np.float64),
        size,
        transform[:3, :3],
        transform[:3, 3].reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    left_maps = cv2.initUndistortRectifyMap(
        left_k,
        np.asarray(ir["left_distortion"], dtype=np.float64),
        left_r,
        left_p,
        size,
        cv2.CV_32FC1,
    )
    right_maps = cv2.initUndistortRectifyMap(
        right_k,
        np.asarray(ir["right_distortion"], dtype=np.float64),
        right_r,
        right_p,
        size,
        cv2.CV_32FC1,
    )
    return left_maps, right_maps


def _depth_stereo_score(
    left: np.ndarray,
    right: np.ndarray,
    raw_depth: np.ndarray,
    *,
    fx: float,
    baseline_m: float,
    scale_m: float,
) -> dict[str, Any]:
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        P1=8 * 25,
        P2=32 * 25,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
    )
    observed = matcher.compute(left, right).astype(np.float32) / 16.0
    depth_m = raw_depth.astype(np.float32) * float(scale_m)
    expected = np.divide(
        fx * baseline_m,
        depth_m,
        out=np.zeros_like(depth_m),
        where=depth_m > 0,
    )
    valid = (
        (raw_depth != 0)
        & (raw_depth != np.iinfo(np.uint16).max)
        & (observed > 1)
        & (observed < 128)
        & (depth_m > 0.2)
        & (depth_m < 20.0)
    )
    if np.count_nonzero(valid) < 1000:
        raise RuntimeError("too few valid stereo/depth pixels for scale audit")
    error = np.abs(observed[valid] - expected[valid])
    inferred_scale_ratio = expected[valid] / observed[valid]
    return {
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "source_encoding": "16UC1",
        "invalid_values": [0, 65535],
        "configured_scale_m_per_unit": float(scale_m),
        "disparity_error_median_px": float(np.median(error)),
        "disparity_error_p95_px": float(np.percentile(error, 95)),
        "inferred_scale_ratio_median": float(np.median(inferred_scale_ratio)),
        "scale_supported": bool(
            0.98 <= float(np.median(inferred_scale_ratio)) <= 1.02
        ),
    }


def _enforce_recorded_depth_acceptance(
    scores: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Gate recorded depth using its measured agreement with stereo."""
    if not scores:
        raise RuntimeError("recorded depth acceptance has no real-image samples")
    valid_pixels = np.asarray(
        [item["valid_pixel_count"] for item in scores], dtype=np.int64
    )
    p95 = np.asarray(
        [item["disparity_error_p95_px"] for item in scores], dtype=np.float64
    )
    scale_supported = np.asarray(
        [item["scale_supported"] for item in scores], dtype=bool
    )
    if (
        np.any(valid_pixels < 0)
        or not np.isfinite(p95).all()
        or np.any(p95 < 0)
    ):
        raise RuntimeError("recorded depth acceptance evidence is malformed")
    observed = {
        "sample_count": len(scores),
        "minimum_valid_pixels_per_sample": int(np.min(valid_pixels)),
        "maximum_disparity_error_p95_px_per_sample": float(np.max(p95)),
        "all_scale_ratios_supported": bool(np.all(scale_supported)),
    }
    thresholds = {
        "minimum_valid_pixels_per_sample": int(
            gates["minimum_valid_pixels_per_sample"]
        ),
        "maximum_disparity_error_p95_px_per_sample": float(
            gates["maximum_disparity_error_p95_px_per_sample"]
        ),
    }
    failures = []
    if observed["minimum_valid_pixels_per_sample"] < thresholds[
        "minimum_valid_pixels_per_sample"
    ]:
        failures.append("insufficient valid stereo/depth pixels")
    if observed["maximum_disparity_error_p95_px_per_sample"] > thresholds[
        "maximum_disparity_error_p95_px_per_sample"
    ]:
        failures.append("stereo/depth p95 disparity disagreement")
    if not observed["all_scale_ratios_supported"]:
        failures.append("configured metric depth scale")
    if failures:
        raise RuntimeError(
            "recorded depth real-image acceptance failed: "
            + ", ".join(failures)
        )
    return {"passed": True, "thresholds": thresholds, "observed": observed}


def _calibration_audit(
    bag: Path,
    cfg: Any,
    typestore: Any,
    start_ns: int,
    stop_ns: int,
    options: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    topics = cfg.topics
    names = [
        topics.left_image,
        topics.right_image,
        topics.left_info,
        topics.right_info,
        topics.color_image,
        topics.color_info,
        topics.depth,
        topics.depth_info,
    ]
    duration = stop_ns - start_ns
    margin = min(20 * NANOSECONDS, duration // 5)
    probes = np.asarray(
        [start_ns + margin, (start_ns + stop_ns) // 2, stop_ns - margin],
        dtype=np.int64,
    )
    bundles = [
        _probe_messages(
            bag,
            names,
            typestore,
            int(probe),
            reference_topic=topics.left_image,
            required_header_gates_ns={
                topics.right_image: _integer_ns(
                    cfg.segment.stereo_tolerance_s, "segment.stereo_tolerance_s"
                ),
                topics.depth: _integer_ns(
                    options["depth_sync_tolerance_s"], "depth_sync_tolerance_s"
                ),
            },
        )
        for probe in probes
    ]
    info_by_topic: dict[str, list[dict[str, Any]]] = {
        name: [_camera_info(bundle[name][1]) for bundle in bundles]
        for name in (topics.left_info, topics.right_info, topics.color_info, topics.depth_info)
    }
    constancy = {}
    for topic, values in info_by_topic.items():
        canonical = json.dumps(values[0], sort_keys=True)
        constancy[topic] = {
            "sample_count": len(values),
            "constant": all(json.dumps(item, sort_keys=True) == canonical for item in values),
        }
        if not constancy[topic]["constant"]:
            raise RuntimeError(f"CameraInfo changes across the selected window: {topic}")
    left_info = info_by_topic[topics.left_info][0]
    right_info = info_by_topic[topics.right_info][0]
    rig = _validate_rectified_stereo(left_info, right_info)
    kalibr_baseline = float(
        np.linalg.norm(np.asarray(options["kalibr_ir"]["T_right_left"], dtype=float)[:3, 3])
    )
    width, height = int(left_info["width"]), int(left_info["height"])
    left_maps, right_maps = _kalibr_rectification(options, (width, height))
    stored_scores = []
    kalibr_scores = []
    depth_scores = []
    sync = []
    for bundle in bundles:
        left_message = bundle[topics.left_image][1]
        right_message = bundle[topics.right_image][1]
        color_message = bundle[topics.color_image][1]
        depth_message = bundle[topics.depth][1]
        left = _decode(left_message)
        right = _decode(right_message)
        depth = _decode(depth_message)
        if left.ndim != 2 or right.ndim != 2:
            raise RuntimeError("Rosario IR topics must decode as mono images")
        if str(depth_message.encoding).lower() != "16uc1" or depth.dtype != np.uint16:
            raise RuntimeError("Rosario recorded depth must be 16UC1")
        if str(color_message.encoding).lower() != "rgb8":
            raise RuntimeError("Rosario raw color must be rgb8")
        stored_scores.append(_stereo_epipolar_score(left, right))
        kalibr_scores.append(
            _stereo_epipolar_score(
                cv2.remap(left, left_maps[0], left_maps[1], cv2.INTER_LINEAR),
                cv2.remap(right, right_maps[0], right_maps[1], cv2.INTER_LINEAR),
            )
        )
        depth_scores.append(
            _depth_stereo_score(
                left,
                right,
                depth,
                fx=float(np.asarray(left_info["p"])[0, 0]),
                baseline_m=float(rig["baseline_m"]),
                scale_m=float(options["recorded_depth_scale_m_per_unit"]),
            )
        )
        sync.append(
            {
                "right_minus_left_ns": _stamp_ns(right_message) - _stamp_ns(left_message),
                "depth_minus_left_ns": _stamp_ns(depth_message) - _stamp_ns(left_message),
                "rgb_minus_left_ns": _stamp_ns(color_message) - _stamp_ns(left_message),
            }
        )
    acceptance_options = options["calibration_acceptance"]
    stereo_acceptance = _enforce_recorded_stereo_acceptance(
        stored_scores, acceptance_options["recorded_stereo"]
    )
    depth_acceptance = _enforce_recorded_depth_acceptance(
        depth_scores, acceptance_options["recorded_depth"]
    )
    depth_info = info_by_topic[topics.depth_info][0]
    depth_camera_alignment = _enforce_depth_left_camera_info_alignment(
        left_info,
        depth_info,
        float(
            acceptance_options["depth_left_camera_info"][
                "maximum_absolute_parameter_difference"
            ]
        ),
    )
    recorded_p95 = float(np.median([item["vertical_residual_p95_px"] for item in stored_scores]))
    kalibr_p95 = float(np.median([item["vertical_residual_p95_px"] for item in kalibr_scores]))
    # Both are reported.  Recorded P remains the simple initial path because
    # it preserves exact depth alignment; re-rectification is an explicit A/B.
    calibration = {
        "camera_info_constancy": constancy,
        "recorded_stereo": {
            "baseline_m": float(rig["baseline_m"]),
            "samples": stored_scores,
            "median_p95_vertical_residual_px": recorded_p95,
            "acceptance": stereo_acceptance,
        },
        "official_kalibr_stereo": {
            "baseline_m": kalibr_baseline,
            "samples_after_re_rectifying_stored_pixels": kalibr_scores,
            "median_p95_vertical_residual_px": kalibr_p95,
            "source_url": options["kalibr_ir"]["source_url"],
            "sha256": options["kalibr_ir"]["sha256"],
        },
        "baseline_difference_m": float(rig["baseline_m"] - kalibr_baseline),
        "selected_geometry": "stored rectified pixels plus recorded CameraInfo.P",
        "selection_reason": (
            "preserves exact recorded-depth alignment; Kalibr re-rectification is "
            "reported as an unresolved photogrammetric A/B, not silently mixed"
        ),
        "frame_id_defect": {
            "left_frame_id": left_info["frame_id"],
            "right_frame_id": right_info["frame_id"],
            "detected": left_info["frame_id"] == right_info["frame_id"],
            "waiver": (
                "Rosario records the infra1 optical frame ID on both CameraInfo "
                "topics; right identity comes from the infra2 topic and P/T baseline"
            ),
        },
    }
    color_info = info_by_topic[topics.color_info][0]
    auxiliary = {
        "color": {
            "topic": topics.color_image,
            "recorded_camera_info": color_info,
            "recorded_zero_distortion_conflicts_with_official_raw_kalibr": bool(
                np.allclose(color_info["d"], 0.0)
                and not np.allclose(options["kalibr_rgb"]["distortion"], 0.0)
            ),
            "official_kalibr": options["kalibr_rgb"],
        },
        "depth": {
            "topic": topics.depth,
            "camera_info": depth_info,
            "encoding": "16UC1",
            "invalid_values": [0, 65535],
            "quantity": "infra1 optical-z",
            "scale_provenance": (
                "sensor_msgs 16UC1 millimetre convention, empirically cross-checked "
                "against recorded stereo disparity"
            ),
            "samples": depth_scores,
            "acceptance": depth_acceptance,
            "camera_info_alignment": depth_camera_alignment,
        },
        "synchronization_samples": sync,
    }
    return calibration, auxiliary, {"left": left_info, "right": right_info}


def _covariance_report(track: RtkTrack) -> dict[str, Any]:
    covariance = np.asarray(track.fix_covariance_enu_m2, dtype=np.float64)
    raw_message_covariance = track.raw_message_covariance_enu_m2
    raw_covariance = np.asarray(
        covariance
        if raw_message_covariance is None
        else raw_message_covariance,
        dtype=np.float64,
    )
    codes, counts = np.unique(track.fix_covariance_type, return_counts=True)
    recorded_static = bool(
        np.allclose(raw_covariance, raw_covariance[0], atol=0, rtol=0)
    )
    return {
        "provenance": (
            "recorded_static_navsatfix"
            if recorded_static
            else "recorded_per_epoch_navsatfix"
        ),
        "is_live_per_epoch": not recorded_static,
        "static_across_window": recorded_static,
        "raw_message_first_diagonal_m2": np.diag(raw_covariance[0]).tolist(),
        "effective_first_diagonal_m2": np.diag(covariance[0]).tolist(),
        "covariance_type_counts": {
            str(int(code)): int(count) for code, count in zip(codes, counts)
        },
        "optimization_policy": (
            "effective externally declared covariance used; raw message covariance retained"
            if raw_message_covariance is not None
            else "recorded covariance used directly; no manufacturer accuracy substituted"
        ),
        "effective_policy": (
            track.effective_covariance_policy
            if track.effective_covariance_policy is not None
            else "raw message covariance used directly"
        ),
    }


def _apply_ppk_effective_covariance(
    track: RtkTrack, options: Mapping[str, Any]
) -> dict[str, Any]:
    raw = np.asarray(track.fix_covariance_enu_m2, dtype=np.float64)
    override = options["offline_ppk_covariance_override"]
    expected_diagonal = np.asarray(
        override["placeholder_diagonal_m2"], dtype=np.float64
    )
    expected = np.broadcast_to(np.diag(expected_diagonal), raw.shape)
    tolerance = float(override["absolute_tolerance_m2"])
    placeholder_verified = bool(
        raw.shape == expected.shape
        and np.allclose(raw, expected, rtol=0.0, atol=tolerance)
    )
    if not placeholder_verified:
        decision = {
            "applied": False,
            "decision": "retained_recorded_covariance",
            "reason": "recorded covariance does not match the declared static placeholder",
            "on_mismatch": override["on_mismatch"],
            "expected_placeholder_diagonal_m2": expected_diagonal.tolist(),
            "absolute_tolerance_m2": tolerance,
            "recorded_static_across_window": bool(
                np.allclose(raw, raw[0], rtol=0.0, atol=tolerance)
            ),
        }
        track.effective_covariance_policy = decision
        return decision

    sigma = np.asarray(options["offline_ppk_effective_sigma_m"], dtype=np.float64)
    effective = np.broadcast_to(
        np.diag(np.square(sigma)),
        np.asarray(track.fix_covariance_enu_m2).shape,
    ).copy()
    policy = {
        "source": "externally_declared_conservative_offline_ppk_prior",
        "sigma_enu_m": sigma.tolist(),
        "provenance": options["offline_ppk_effective_covariance_provenance"],
        "status": "not independently surveyed; raw NavSatFix covariance retained separately",
        "placeholder_verification": {
            "matched": True,
            "expected_diagonal_m2": expected_diagonal.tolist(),
            "absolute_tolerance_m2": tolerance,
        },
    }
    track.apply_effective_covariance(
        effective,
        policy=policy,
    )
    return {
        "applied": True,
        "decision": "replaced_verified_static_placeholder",
        **policy["placeholder_verification"],
    }


def _bounded_gnss_log_window(
    gnss_report: Any,
    requested_start_ns: int,
    requested_stop_ns: int,
    padding_ns: int,
) -> tuple[int, int]:
    """Clamp a requested method window to the GNSS bag that is read."""
    start = max(
        int(gnss_report.start_log_ns), int(requested_start_ns) - int(padding_ns)
    )
    stop = min(
        int(gnss_report.stop_log_ns), int(requested_stop_ns) + int(padding_ns)
    )
    if stop <= start:
        raise RuntimeError("selected camera window does not overlap the GNSS bag")
    return start, stop


def _quaternion_transform(message: Any) -> np.ndarray:
    translation = message.transform.translation
    quaternion = message.transform.rotation
    x, y, z, w = (float(quaternion.x), float(quaternion.y), float(quaternion.z), float(quaternion.w))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0:
        raise RuntimeError("recorded TF contains a zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [translation.x, translation.y, translation.z]
    return transform


def _rotation_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    delta = left[:3, :3].T @ right[:3, :3]
    cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _enforce_depth_left_static_tf_alignment(
    depth_from_left: np.ndarray | None,
    gates: Mapping[str, Any],
    *,
    depth_frame_id: str,
    left_frame_id: str,
) -> dict[str, Any]:
    """Require a recorded static identity transform for aligned depth."""
    if depth_from_left is None:
        if gates["on_missing"] == "fail":
            raise RuntimeError(
                "recorded /tf_static has no depth-optical to infra1-optical path"
            )
        raise ValueError("unsupported depth/left static TF missing policy")
    transform = np.asarray(depth_from_left, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise RuntimeError("recorded depth/left static TF is not a finite 4x4")
    translation_m = float(np.linalg.norm(transform[:3, 3]))
    rotation_deg = _rotation_error_deg(transform, np.eye(4))
    report = {
        "passed": (
            translation_m <= float(gates["maximum_translation_m"])
            and rotation_deg <= float(gates["maximum_rotation_deg"])
        ),
        "source": "recorded /tf_static graph",
        "left_frame_id": left_frame_id,
        "depth_frame_id": depth_frame_id,
        "transform_convention": "depth_from_left",
        "depth_from_left": transform.tolist(),
        "observed_translation_m": translation_m,
        "observed_rotation_deg": rotation_deg,
        "thresholds": {
            "maximum_translation_m": float(gates["maximum_translation_m"]),
            "maximum_rotation_deg": float(gates["maximum_rotation_deg"]),
            "on_missing": gates["on_missing"],
        },
    }
    if not report["passed"]:
        raise RuntimeError(
            "recorded depth optical frame is not identity-aligned with infra1"
        )
    return report


def _tf_audit(
    bag: Path,
    cfg: Any,
    typestore: Any,
    options: Mapping[str, Any],
    resolved_extrinsic: np.ndarray,
    *,
    left_frame_id: str,
    depth_frame_id: str,
) -> dict[str, Any]:
    Reader = _reader_type()
    tf_options = options["recorded_tf"]
    transforms: dict[tuple[str, str], np.ndarray] = {}
    static_transforms: dict[tuple[str, str], np.ndarray] = {}
    with Reader(bag) as reader:
        connections = [
            item
            for item in reader.connections
            if item.topic in {tf_options["topic"], tf_options["static_topic"]}
        ]
        for connection in connections:
            _register_ros1_connection_type(typestore, connection)
        for connection, _, raw in reader.messages(connections=connections):
            message = _deserialize(typestore, raw, connection.msgtype)
            for stamped in message.transforms:
                key = (str(stamped.header.frame_id), str(stamped.child_frame_id))
                transform = _quaternion_transform(stamped)
                transforms[key] = transform
                if connection.topic == tf_options["static_topic"]:
                    static_transforms[key] = transform

    def relative(
        target: str,
        source: str,
        evidence: Mapping[tuple[str, str], np.ndarray] = transforms,
    ) -> np.ndarray:
        if target == source:
            return np.eye(4)
        graph: dict[str, list[tuple[str, np.ndarray]]] = {}
        for (parent, child), parent_from_child in evidence.items():
            graph.setdefault(child, []).append((parent, parent_from_child))
            graph.setdefault(parent, []).append((child, np.linalg.inv(parent_from_child)))
        queue = deque([(source, np.eye(4))])
        seen = {source}
        while queue:
            frame, frame_from_source = queue.popleft()
            for neighbor, neighbor_from_frame in graph.get(frame, []):
                if neighbor in seen:
                    continue
                candidate = neighbor_from_frame @ frame_from_source
                if neighbor == target:
                    return candidate
                seen.add(neighbor)
                queue.append((neighbor, candidate))
        raise RuntimeError(f"recorded TF has no path {source} -> {target}")

    def discrepancy(recorded: np.ndarray, official: np.ndarray) -> dict[str, float]:
        return {
            "translation_m": float(np.linalg.norm(recorded[:3, 3] - official[:3, 3])),
            "rotation_deg": _rotation_error_deg(recorded, official),
        }

    recorded_right_from_left = relative(
        "realsense_infra2_optical_frame", "realsense_infra1_optical_frame"
    )
    official_right_from_left = np.asarray(options["kalibr_ir"]["T_right_left"], dtype=float)
    recorded_rgb_from_left = relative(
        "realsense_color_optical_frame", "realsense_infra1_optical_frame"
    )
    try:
        recorded_depth_from_left = relative(
            depth_frame_id, left_frame_id, static_transforms
        )
    except RuntimeError:
        recorded_depth_from_left = None
    depth_left_alignment = _enforce_depth_left_static_tf_alignment(
        recorded_depth_from_left,
        options["calibration_acceptance"]["depth_left_static_tf"],
        depth_frame_id=depth_frame_id,
        left_frame_id=left_frame_id,
    )
    official_rgb_from_left = np.asarray(options["kalibr_rgb"]["T_rgb_left"], dtype=float)
    rgb_discrepancy = discrepancy(recorded_rgb_from_left, official_rgb_from_left)
    if rgb_discrepancy["translation_m"] > 0.02 or rgb_discrepancy["rotation_deg"] > 2.0:
        raise RuntimeError(
            "recorded factory IR/RGB TF disagrees materially with official Kalibr prior"
        )
    comparison_child = str(tf_options["comparison_child_frame"])
    recorded_base_from_camera = transforms.get(("base_link", comparison_child))
    base_from_primary = np.eye(4)
    base_from_primary[:3, 3] = np.asarray(tf_options["primary_antenna_in_base_m"], dtype=float)
    official_base_from_camera = base_from_primary @ np.linalg.inv(resolved_extrinsic)

    return {
        "static_ir_stereo_vs_kalibr": discrepancy(
            recorded_right_from_left, official_right_from_left
        ),
        "static_ir_rgb_vs_kalibr": rgb_discrepancy,
        "static_depth_left_alignment": depth_left_alignment,
        "selected_recorded_rgb_from_left": recorded_rgb_from_left.tolist(),
        "selected_transform_source": "bag /tf_static graph",
        "alternative_kalibr_rgb_from_left": official_rgb_from_left.tolist(),
        "base_camera_comparison": (
            {
                "recorded_child_frame": comparison_child,
                "configured_camera_frame": str(cfg.sensor_geometry.frames.camera),
                "frame_semantics_match": comparison_child
                == str(cfg.sensor_geometry.frames.camera),
                "discrepancy": discrepancy(
                    recorded_base_from_camera, official_base_from_camera
                ),
                "policy": "reported only; URDF camera-to-Reach transform remains a rough prior",
            }
            if recorded_base_from_camera is not None
            else {"available": False}
        ),
    }


def _prepare_config(cfg: Any):
    if getattr(cfg, "adapter", None) != _ADAPTER:
        raise ValueError(f"Rosario preflight requires adapter: {_ADAPTER}")
    if getattr(cfg.pose, "source", None) != "dual_position":
        raise ValueError("Rosario dual-M2 ingestion requires pose.source=dual_position")
    if getattr(cfg.segment, "image_encoding", None) != "png_lossless":
        raise ValueError("Rosario adapter requires segment.image_encoding=png_lossless")
    options = validate_adapter_options(cfg.adapter_options)
    bag = _configured_bag(cfg)
    gnss_bag = _configured_gnss_bag(cfg, options, bag)
    primary_topic = (
        cfg.topics.fix
        if options["gnss_source"] == "online_differential"
        else cfg.topics.ppk_fix
    )
    secondary_topic = (
        cfg.topics.secondary_fix
        if options["gnss_source"] == "online_differential"
        else cfg.topics.ppk_secondary_fix
    )
    required = {
        cfg.topics.left_image,
        cfg.topics.right_image,
        cfg.topics.left_info,
        cfg.topics.right_info,
        cfg.topics.color_image,
        cfg.topics.color_info,
        cfg.topics.depth,
        cfg.topics.depth_info,
        cfg.topics.tf,
        cfg.topics.tf_static,
    }
    report = validate_bag_chain(
        [bag],
        required_each=required,
        expected_count=1,
        maximum_gap_ns=0,
        maximum_overlap_ns=0,
    )
    gnss_report = validate_bag_chain(
        [gnss_bag],
        required_each={primary_topic, secondary_topic},
        expected_count=1,
        maximum_gap_ns=0,
        maximum_overlap_ns=0,
    )
    if getattr(cfg.segment, "window_epoch_source", None) != "first_bag_log":
        raise ValueError("Rosario window_epoch_source must be first_bag_log")
    window = np.asarray(cfg.segment.window_s, dtype=np.float64)
    if window.shape != (2,) or not np.isfinite(window).all() or window[0] < 0 or window[1] <= window[0]:
        raise ValueError("segment.window_s must be finite increasing [start,stop]")
    epoch_ns = int(report.start_log_ns)
    start_ns = epoch_ns + int(round(window[0] * NANOSECONDS))
    stop_ns = epoch_ns + int(round(window[1] * NANOSECONDS))
    if start_ns < report.start_log_ns or stop_ns > report.stop_log_ns:
        raise RuntimeError("selected Rosario window is outside the main bag")
    padding_ns = _integer_ns(options["gnss_window_padding_s"], "gnss_window_padding_s")
    gnss_start_ns, gnss_stop_ns = _bounded_gnss_log_window(
        gnss_report, start_ns, stop_ns, padding_ns
    )
    typestore = build_typestore()
    primary = _read_navsat_window(
        gnss_bag,
        primary_topic,
        typestore,
        start_log_ns=gnss_start_ns,
        stop_log_ns=gnss_stop_ns,
        source_mode=options["gnss_source"],
    )
    secondary = _read_navsat_window(
        gnss_bag,
        secondary_topic,
        typestore,
        start_log_ns=gnss_start_ns,
        stop_log_ns=gnss_stop_ns,
        source_mode=options["gnss_source"],
    )
    covariance_decisions = None
    if options["gnss_source"] == "offline_ppk":
        covariance_decisions = {
            "primary": _apply_ppk_effective_covariance(primary, options),
            "secondary": _apply_ppk_effective_covariance(secondary, options),
        }
    # Heading accuracy must be propagated from the exact covariance that the
    # optimizer will consume, whether externally overridden or retained raw.
    dual = _attach_dual_position_heading(primary, secondary, options)
    secondary_raw_covariance = secondary.raw_message_covariance_enu_m2
    primary.secondary_gnss = SecondaryGnssEvidence(
        header_ns=np.asarray(secondary.fix_header_ns, dtype=np.int64),
        log_ns=np.asarray(secondary.fix_log_ns, dtype=np.int64),
        enu_m=np.asarray(secondary.enu_xyz, dtype=np.float64),
        geodetic_deg_m=np.column_stack(
            (secondary.fix_lat, secondary.fix_lon, secondary.fix_alt)
        ),
        raw_covariance_enu_m2=np.asarray(
            secondary.fix_covariance_enu_m2
            if secondary_raw_covariance is None
            else secondary_raw_covariance,
            dtype=np.float64,
        ),
        effective_covariance_enu_m2=np.asarray(
            secondary.fix_covariance_enu_m2, dtype=np.float64
        ),
        fix_status=np.asarray(secondary.fix_status, dtype=np.int16),
        carrier_status=np.asarray(secondary.fix_carrier_status, dtype=np.int8),
        covariance_type=np.asarray(
            secondary.fix_covariance_type, dtype=np.int16
        ),
        service=np.asarray(secondary.fix_service, dtype=np.int16),
    )
    camera_logs, camera_offsets = sample_header_log_offsets(
        report,
        cfg.topics.left_image,
        typestore,
        start_ns,
        stop_ns,
    )
    fix_log = np.asarray(primary.fix_log_ns, dtype=np.int64)
    fix_offsets = np.asarray(primary.fix_header_ns, dtype=np.int64) - fix_log
    in_window = (fix_log >= start_ns) & (fix_log <= stop_ns)
    clock_estimate = int(
        np.rint(np.median(fix_offsets[in_window]) - np.median(camera_offsets))
    )
    configured_offset = _integer_ns(cfg.pose.time_offset_s, "pose.time_offset_s")
    maximum_error = _integer_ns(
        cfg.timing.maximum_offset_error_s, "timing.maximum_offset_error_s"
    )
    if abs(configured_offset - clock_estimate) > maximum_error:
        raise RuntimeError(
            f"camera-to-Reach clock estimate {clock_estimate} ns disagrees with "
            f"configured {configured_offset} ns"
        )
    calibration, auxiliary, camera_info = _calibration_audit(
        bag, cfg, typestore, start_ns, stop_ns, options
    )
    configured_frame = str(cfg.sensor_geometry.frames.camera)
    if configured_frame != camera_info["left"]["frame_id"]:
        raise RuntimeError("configured left camera frame does not match CameraInfo")
    resolved_extrinsic, extrinsic_resolution = _resolved_camera_extrinsic(
        cfg.sensor_geometry, camera_info["left"]
    )
    tf_audit = _tf_audit(
        bag,
        cfg,
        typestore,
        options,
        resolved_extrinsic,
        left_frame_id=str(camera_info["left"]["frame_id"]),
        depth_frame_id=str(auxiliary["depth"]["camera_info"]["frame_id"]),
    )
    auxiliary["depth"]["static_tf_alignment"] = tf_audit[
        "static_depth_left_alignment"
    ]
    spacing = resolve_metric_frame_spacing(cfg)
    mask = (fix_log >= start_ns) & (fix_log <= stop_ns)
    trajectory = primary.enu_xyz[mask]
    if len(trajectory) < 2:
        raise RuntimeError("selected window contains too few online GNSS fixes")
    path_length = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
    preflight = {
        "passed": True,
        "adapter": _ADAPTER,
        "input_policy": {
            "method_bag": str(bag),
            "gnss_source": options["gnss_source"],
            "gnss_bag": str(gnss_bag),
            "postprocessed_input": options["gnss_source"] == "offline_ppk",
            "excluded": ["PGT", "conventional GNSS", "IMU", "wheel odometry"],
        },
        "bag": report.as_json(),
        "gnss_bag": gnss_report.as_json(),
        "window": {
            "epoch_source": "first_bag_log",
            "epoch_ns": epoch_ns,
            "start_log_ns": start_ns,
            "stop_log_ns": stop_ns,
            "duration_s": (stop_ns - start_ns) * 1e-9,
        },
        "clock": {
            "configured_camera_to_reach_ns": configured_offset,
            "ros_log_estimate_ns": clock_estimate,
            "error_ns": configured_offset - clock_estimate,
            "camera_probe_log_ns": camera_logs.tolist(),
        },
        "gnss": {
            "source_mode": options["gnss_source"],
            "primary_topic": primary_topic,
            "secondary_topic": secondary_topic,
            "quality_label": (
                "online NavSatStatus per epoch: status 0 standalone, status >=1 differential"
                if options["gnss_source"] == "online_differential"
                else "offline PPK / unknown_valid (carrier state unavailable)"
            ),
            "primary_quality_counts": {
                str(label): int(count)
                for label, count in zip(
                    *np.unique(primary.fix_position_quality, return_counts=True)
                )
            },
            "primary_covariance": _covariance_report(primary),
            "secondary_covariance": _covariance_report(secondary),
            "dual_baseline": dual,
            "offline_ppk_covariance_decisions": covariance_decisions,
        },
        "calibration": calibration,
        "auxiliary_observations": auxiliary,
        "extrinsic": {
            "status": "rough prior pending independent image/TF audit",
            "resolution": extrinsic_resolution,
            "recorded_tf_audit": tf_audit,
        },
        "sampling": {
            "frame_spacing_m": float(spacing),
            "gnss_path_length_m": path_length,
            "estimated_frame_count": int(math.floor(path_length / spacing)) + 1,
        },
    }
    return preflight, primary, typestore, options, camera_info, resolved_extrinsic


def preflight_config(cfg: Any) -> dict[str, Any]:
    """Run bounded index, sensor, timing, geometry, and scale checks only."""
    return _prepare_config(cfg)[0]


def _interpolate_position(track: RtkTrack, timestamp_ns: int) -> np.ndarray | None:
    times = np.asarray(track.fix_header_ns, dtype=np.int64)
    if timestamp_ns < times[0] or timestamp_ns > times[-1]:
        return None
    origin = int(times[0])
    x = (times - origin).astype(np.float64)
    query = float(timestamp_ns - origin)
    return np.asarray(
        [np.interp(query, x, track.enu_xyz[:, axis]) for axis in range(3)],
        dtype=np.float64,
    )


def _encode_png(message: Any, destination: Path) -> None:
    image = _decode(message)
    source_encoding = str(message.encoding).lower()
    if source_encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, payload = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("OpenCV failed to encode lossless PNG")
    destination.write_bytes(payload.tobytes())


def _drain_stereo(
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


def _stage_ir_frames(
    bag: Path,
    cfg: Any,
    typestore: Any,
    track: RtkTrack,
    *,
    start_ns: int,
    stop_ns: int,
    spacing_m: float,
    clock_offset_ns: int,
    output_dir: Path,
) -> tuple[list[FrameRecord], dict[str, Any]]:
    Reader = _reader_type()
    tolerance_ns = _integer_ns(
        cfg.segment.stereo_tolerance_s, "segment.stereo_tolerance_s"
    )
    sampler = _DistanceSampler(spacing_m)
    left_queue: deque[_TimedImage] = deque()
    right_queue: deque[_TimedImage] = deque()
    frames: list[FrameRecord] = []
    candidates = 0
    with Reader(bag) as reader:
        connections = [
            item
            for item in reader.connections
            if item.topic in {cfg.topics.left_image, cfg.topics.right_image}
        ]
        for connection, log_ns, raw in reader.messages(
            connections=connections,
            start=start_ns - NANOSECONDS,
            stop=stop_ns + NANOSECONDS,
        ):
            message = _deserialize(typestore, raw, connection.msgtype)
            timed = _TimedImage(_stamp_ns(message), int(log_ns), message)
            (left_queue if connection.topic == cfg.topics.left_image else right_queue).append(timed)
            for left, right in _drain_stereo(left_queue, right_queue, tolerance_ns):
                if not start_ns <= left.log_ns <= stop_ns:
                    continue
                candidates += 1
                position = _interpolate_position(track, left.header_ns + clock_offset_ns)
                if position is None or not sampler.accept(position):
                    continue
                index = len(frames)
                left_path = output_dir / f"left_{index:06d}.png"
                right_path = output_dir / f"right_{index:06d}.png"
                _encode_png(left.message, left_path)
                _encode_png(right.message, right_path)
                record = FrameRecord(
                    t=left.header_ns * 1e-9,
                    t_right=right.header_ns * 1e-9,
                    left_jpeg=left_path,
                    right_jpeg=right_path,
                    left_header_ns=left.header_ns,
                    right_header_ns=right.header_ns,
                )
                record.left_log_ns = left.log_ns
                record.right_log_ns = right.log_ns
                frames.append(record)
    if not frames:
        raise RuntimeError("selected Rosario window produced no IR stereo frames")
    return frames, {
        "candidate_pairs": candidates,
        "selected_pairs": len(frames),
        "requested_spacing_m": float(spacing_m),
        "path_length_m": float(sampler.path_length_m),
        "effective_spacing_m": float(sampler.path_length_m / max(1, len(frames) - 1)),
        "stored_geometry": "recorded rectified pixels and CameraInfo.P",
    }


def _nearest_stream_records(
    bag: Path,
    topic: str,
    typestore: Any,
    query_ns: np.ndarray,
    *,
    start_log_ns: int,
    stop_log_ns: int,
    tolerance_ns: int,
    require_all: bool = True,
) -> list[_TimedImage | None]:
    Reader = _reader_type()
    output: list[_TimedImage | None] = [None] * len(query_ns)
    query_index = 0
    previous: _TimedImage | None = None
    with Reader(bag) as reader:
        connections = [item for item in reader.connections if item.topic == topic]
        for connection, log_ns, raw in reader.messages(
            connections=connections,
            start=start_log_ns - NANOSECONDS,
            stop=stop_log_ns + NANOSECONDS,
        ):
            message = _deserialize(typestore, raw, connection.msgtype)
            current = _TimedImage(_stamp_ns(message), int(log_ns), message)
            while query_index < len(query_ns) and query_ns[query_index] <= current.header_ns:
                candidates = [item for item in (previous, current) if item is not None]
                chosen = min(candidates, key=lambda item: abs(item.header_ns - int(query_ns[query_index])))
                if abs(chosen.header_ns - int(query_ns[query_index])) <= tolerance_ns:
                    output[query_index] = chosen
                query_index += 1
            previous = current
    while query_index < len(query_ns) and previous is not None:
        if abs(previous.header_ns - int(query_ns[query_index])) > tolerance_ns:
            break
        output[query_index] = previous
        query_index += 1
    if require_all and any(item is None for item in output):
        first = next(index for index, item in enumerate(output) if item is None)
        raise RuntimeError(f"{topic} has no bounded association for frame {first}")
    return output


def _drop_missing_depth_frames(
    frames: Sequence[FrameRecord],
    depth_records: Sequence[Mapping[str, Any] | None],
) -> tuple[list[FrameRecord], list[Mapping[str, Any]], list[int]]:
    """Drop only stereo samples lacking exact recorded-depth evidence."""
    if len(frames) != len(depth_records):
        raise RuntimeError("depth association count does not match selected stereo")
    keep = [index for index, record in enumerate(depth_records) if record is not None]
    dropped = [index for index, record in enumerate(depth_records) if record is None]
    if not keep:
        raise RuntimeError("recorded-depth availability rejects every selected frame")
    return (
        [frames[index] for index in keep],
        [depth_records[index] for index in keep if depth_records[index] is not None],
        dropped,
    )


def _publish_rgb_observations(
    destination: Path,
    bag: Path,
    topic: str,
    typestore: Any,
    options: Mapping[str, Any],
    *,
    start_log_ns: int,
    stop_log_ns: int,
    source_start_header_ns: int,
    source_stop_header_ns: int,
    source_camera_frame: str,
    T_rgb_source_camera: Sequence[Sequence[float]],
    recorded_camera_info: Mapping[str, Any],
    provenance: Mapping[str, Any],
    acquisition_id: str,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=".rosario-rgb-stage-"
    ) as temporary:
        image_dir = Path(temporary)
        paths = []
        headers = []
        logs = []
        discarded_before = 0
        discarded_after = 0
        Reader = _reader_type()
        with Reader(bag) as reader:
            connections = [item for item in reader.connections if item.topic == topic]
            for connection, log_ns, raw in reader.messages(
                connections=connections,
                start=start_log_ns,
                stop=stop_log_ns,
            ):
                message = _deserialize(typestore, raw, connection.msgtype)
                header_ns = _stamp_ns(message)
                if header_ns < source_start_header_ns:
                    discarded_before += 1
                    continue
                if header_ns > source_stop_header_ns:
                    discarded_after += 1
                    continue
                index = len(headers)
                name = f"rgb_{index:06d}.png"
                _encode_png(message, image_dir / name)
                paths.append(image_dir / name)
                headers.append(header_ns)
                logs.append(int(log_ns))
        if not headers or np.any(np.diff(np.asarray(headers, dtype=np.int64)) <= 0):
            raise RuntimeError("bounded RGB stream is empty or not strictly increasing")
        rgb = options["kalibr_rgb"]
        width = int(recorded_camera_info["width"])
        height = int(recorded_camera_info["height"])
        recorded_k = np.asarray(recorded_camera_info["p"], dtype=np.float64)[:, :3]
        recorded_d = np.asarray(recorded_camera_info["d"], dtype=np.float64)
        if not np.allclose(recorded_d, 0.0):
            raise RuntimeError("recorded_camera_info mode requires explicit zero D")
        rgb_frame = str(recorded_camera_info["frame_id"])
        calibration = {
            "schema_version": 1,
            "camera_frame_id": rgb_frame,
            "source_camera_frame_id": source_camera_frame,
            "source_camera_geometry": "source_segment_left_rectified",
            "rgb_camera_geometry": "recorded_factory_pinhole_direct",
            "image_geometry": "rectified",
            "camera": {
                "model": "PINHOLE",
                "width": width,
                "height": height,
                "K": recorded_k.tolist(),
                "distortion": [0.0, 0.0, 0.0, 0.0],
            },
            "rectification": {
                "input_camera": {
                    "model": "PINHOLE",
                    "width": width,
                    "height": height,
                    "K": recorded_k.tolist(),
                    "distortion_model": "none",
                    "distortion": [0.0, 0.0, 0.0, 0.0],
                },
                "method": "identity_copy_recorded_factory_pinhole",
                "provenance": {
                    "selection": (
                        "preliminary recorded factory representation; requires "
                        "held-out pose-motion-compensated RGB/depth validation"
                    ),
                    "source_url": rgb["source_url"],
                    "sha256": rgb["sha256"],
                    "official_kalibr_raw_model_retained_for_future_A_B": {
                        "K": _camera_matrix(rgb["intrinsics"]).tolist(),
                        "distortion": list(rgb["distortion"]),
                    },
                }
            },
            "T_rgb_source_camera": _plain_metadata(T_rgb_source_camera),
            "transform_convention": "rgb_from_source_camera",
            "extrinsic_translation_sigma_m": rgb["extrinsic_translation_sigma_m"],
            "extrinsic_translation_sigma_frame_id": rgb_frame,
            "extrinsic_provenance": {
                "method": "selected recorded factory transform from bag /tf_static graph",
                "source_url": rgb["source_url"],
                "sha256": rgb["sha256"],
                "status": "recorded factory transform selected; official Kalibr retained as bounded alternative prior",
                "alternative_kalibr_T_rgb_source_camera": rgb["T_rgb_left"],
            },
        }
        artifact_provenance = _plain_metadata(provenance)
        artifact_provenance["coverage_crop"] = {
            "source_start_header_ns": int(source_start_header_ns),
            "source_stop_header_ns": int(source_stop_header_ns),
            "discarded_before": int(discarded_before),
            "discarded_after": int(discarded_after),
            "reason": "transfer requires every RGB observation inside accepted source-pose coverage",
        }
        return publish_rectified_rgb_observations(
            destination,
            paths,
            np.asarray(headers, dtype=np.int64),
            log_timestamp_ns=np.asarray(logs, dtype=np.int64),
            calibration=calibration,
            timestamp_source="raw RGB sensor header",
            clock={
                "rgb_to_source_clock_offset_ns": 0,
                "provenance": {
                    "method": "zero prior only; bounded IR/RGB timing A/B is not observable enough to replace it",
                    "status": "unresolved; all bounded RGB timestamps retained for transfer-time association",
                },
            },
            provenance=artifact_provenance,
            acquisition_id=acquisition_id,
        )


@contextmanager
def _rollback_new_rgb_on_failure(path: Path):
    """Remove only this invocation's new sibling if segment publish fails."""
    try:
        yield
    except Exception:
        if path.is_dir() and path.name.endswith(_RGB_SUFFIX):
            shutil.rmtree(path)
        raise


def _stage_auxiliary(
    bag: Path,
    cfg: Any,
    typestore: Any,
    frames: Sequence[FrameRecord],
    options: Mapping[str, Any],
    *,
    start_ns: int,
    stop_ns: int,
    output_dir: Path,
) -> tuple[list[dict[str, Any] | None], dict[str, Any]]:
    queries = np.asarray([frame.left_header_ns for frame in frames], dtype=np.int64)
    depth = _nearest_stream_records(
        bag,
        cfg.topics.depth,
        typestore,
        queries,
        start_log_ns=start_ns,
        stop_log_ns=stop_ns,
        tolerance_ns=_integer_ns(options["depth_sync_tolerance_s"], "depth_sync_tolerance_s"),
        require_all=False,
    )
    depth_records = []
    invalid_counts = {"zero": 0, "saturated_65535": 0, "range": 0}
    for index, record in enumerate(depth):
        if record is None:
            depth_records.append(None)
            continue
        raw = _decode(record.message)
        if raw.dtype != np.uint16 or str(record.message.encoding).lower() != "16uc1":
            raise RuntimeError("recorded Rosario depth changed from 16UC1")
        depth_m = raw.astype(np.float32) * float(options["recorded_depth_scale_m_per_unit"])
        zero = raw == 0
        saturated = raw == np.iinfo(np.uint16).max
        in_range = (
            (depth_m >= float(options["recorded_depth_min_z_m"]))
            & (depth_m <= float(options["recorded_depth_max_z_m"]))
        )
        valid = ~zero & ~saturated & in_range
        invalid_counts["zero"] += int(np.count_nonzero(zero))
        invalid_counts["saturated_65535"] += int(np.count_nonzero(saturated))
        invalid_counts["range"] += int(np.count_nonzero(~zero & ~saturated & ~in_range))
        depth_m[~valid] = 0.0
        path = output_dir / f"depth_{index:06d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                depth=depth_m.astype(np.float32),
                valid=valid.astype(bool),
                raw_depth_units=raw,
            )
        depth_records.append(
            {
                "payload": path,
                "header_ns": record.header_ns,
                "log_ns": record.log_ns,
                "source_encoding": str(record.message.encoding),
            }
        )
    residuals = np.asarray(
        [
            record.header_ns - query
            for record, query in zip(depth, queries)
            if record is not None
        ],
        dtype=np.int64,
    )
    if not len(residuals):
        raise RuntimeError("no selected stereo frame has exact recorded depth")
    return depth_records, {
        "rgb_timing": {
            "prior_offset_ns": 0,
            "status": "unresolved; full bounded stream retained for calibrated transfer A/B",
        },
        "depth_residual_ns": {
            "median": int(np.rint(np.median(residuals))),
            "maximum_absolute": int(np.max(np.abs(residuals))),
        },
        "missing_depth_source_frame_indices": [
            index for index, record in enumerate(depth) if record is None
        ],
        "depth_invalid_pixel_counts": invalid_counts,
    }


def ingest_config_v2(
    cfg: Any,
    destination: str | Path,
    *,
    window: Mapping[str, Any] | None = None,
):
    """Publish bounded Rosario IR/depth segment plus sealed RGB observations."""
    if window is not None:
        raise ValueError("Rosario adapter uses its explicit first-bag-log window")
    preflight, track, typestore, options, camera_info, resolved_extrinsic = _prepare_config(cfg)
    bag = _configured_bag(cfg)
    start_ns = int(preflight["window"]["start_log_ns"])
    stop_ns = int(preflight["window"]["stop_log_ns"])
    destination_path = Path(destination)
    rgb_destination = destination_path.with_name(destination_path.name + _RGB_SUFFIX)
    if destination_path.exists() or rgb_destination.exists():
        raise FileExistsError("refusing to overwrite an existing Rosario artifact")
    acquisition_id = uuid.uuid4().hex
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    spacing = float(preflight["sampling"]["frame_spacing_m"])
    clock_offset_ns = int(preflight["clock"]["configured_camera_to_reach_ns"])
    with tempfile.TemporaryDirectory(
        dir=destination_path.parent, prefix=".rosario-ingest-"
    ) as temporary:
        temporary_path = Path(temporary)
        frames, sampling = _stage_ir_frames(
            bag,
            cfg,
            typestore,
            track,
            start_ns=start_ns,
            stop_ns=stop_ns,
            spacing_m=spacing,
            clock_offset_ns=clock_offset_ns,
            output_dir=temporary_path,
        )
        depth_records, auxiliary_staging = _stage_auxiliary(
            bag,
            cfg,
            typestore,
            frames,
            options,
            start_ns=start_ns,
            stop_ns=stop_ns,
            output_dir=temporary_path,
        )
        frames, depth_records, dropped_depth = _drop_missing_depth_frames(
            frames, depth_records
        )
        sampling["dropped_missing_recorded_depth_source_frame_indices"] = dropped_depth
        sampling["published_pairs"] = len(frames)
        pose_stamps = [
            (int(frame.left_header_ns) + clock_offset_ns) * 1e-9 for frame in frames
        ]
        poses = pose_frames_from_extrinsic(
            track, pose_stamps, cfg.pose, resolved_extrinsic
        )
        keep = [index for index, pose in enumerate(poses) if pose is not None]
        if len(keep) != len(frames):
            # Keeping RGB/depth correspondence auditable is more important than
            # silently dropping mismatched rows after auxiliary association.
            raise RuntimeError(
                f"pose gates reject {len(frames) - len(keep)} selected Rosario frames"
            )
        validation_every = int(getattr(cfg.segment, "validation_every", 8))
        validation = list(range(validation_every - 1, len(frames), validation_every))
        validation_set = set(validation)
        splits = {
            "train": [index for index in range(len(frames)) if index not in validation_set],
            "val": validation,
            "test": [],
        }
        rgb_artifact = _publish_rgb_observations(
            rgb_destination,
            bag,
            cfg.topics.color_image,
            typestore,
            options,
            start_log_ns=start_ns,
            stop_log_ns=stop_ns,
            source_start_header_ns=int(frames[0].left_header_ns),
            source_stop_header_ns=int(frames[-1].left_header_ns),
            source_camera_frame=str(cfg.sensor_geometry.frames.camera),
            T_rgb_source_camera=preflight["extrinsic"]["recorded_tf_audit"][
                "selected_recorded_rgb_from_left"
            ],
            recorded_camera_info=preflight["auxiliary_observations"]["color"]["recorded_camera_info"],
            provenance={
                "adapter": _ADAPTER,
                "acquisition_id": acquisition_id,
                "source_bag": str(bag),
                "source_topic": cfg.topics.color_image,
                "window": preflight["window"],
                "timing": auxiliary_staging["rgb_timing"],
                "policy": "full bounded RGB stream retained outside canonical IR stereo contract",
            },
            acquisition_id=acquisition_id,
        )
        enu = track.enu.crs()
        geometry = cfg.sensor_geometry
        with _rollback_new_rgb_on_failure(rgb_artifact):
            return publish_segment_v2(
                destination_path,
                frames,
                poses,
                track,
                camera_info["left"],
                camera_info["right"],
                camera_frame_id=str(geometry.frames.camera),
                primary_antenna_frame_id=str(geometry.frames.primary_antenna),
                secondary_antenna_frame_id=str(geometry.frames.secondary_antenna),
                enu_definition={
                    "origin_lat_deg": float(enu["origin_lat"]),
                    "origin_lon_deg": float(enu["origin_lon"]),
                    "origin_alt_ellipsoidal_m": float(
                        enu["origin_alt_ellipsoidal"]
                    ),
                    "ellipsoid": str(enu["ellipsoid"]),
                    "vertical_datum": str(enu["vertical_datum"]),
                    "world_frame_id": str(geometry.frames.world),
                },
                T_camera_primary_antenna=resolved_extrinsic,
                extrinsic_translation_sigma_m=(
                    geometry.extrinsic_translation_sigma_m
                ),
                extrinsic_provenance={
                    **_plain_metadata(geometry.extrinsic_provenance),
                    "preflight_status": preflight["extrinsic"],
                },
                clock_offset_ns=clock_offset_ns,
                association_tolerance_ns=_integer_ns(
                    cfg.segment.association_tolerance_s,
                    "segment.association_tolerance_s",
                ),
                stereo_tolerance_ns=_integer_ns(
                    cfg.segment.stereo_tolerance_s,
                    "segment.stereo_tolerance_s",
                ),
                capabilities={
                    "single_rtk": False,
                    "dual_rtk": True,
                    "depth_recorded": True,
                },
                splits=splits,
                adapter_name=_ADAPTER,
                recorded_depth=depth_records,
                recorded_depth_semantics={
                    "format": "npz_depth_valid",
                    "units": "m",
                    "quantity": "optical_z",
                    "aligned_to": "left",
                    "invalid_convention": (
                        "raw 0 and 65535 invalid; configured range invalid; "
                        "published depth=0 and valid=false"
                    ),
                    "source_encoding": "16UC1",
                    "raw_units_preserved_field": "raw_depth_units",
                    "scale_m_per_unit": float(
                        options["recorded_depth_scale_m_per_unit"]
                    ),
                    "scale_validation": preflight["auxiliary_observations"][
                        "depth"
                    ],
                    "alignment_validation": {
                        "camera_info": preflight[
                            "auxiliary_observations"
                        ]["depth"]["camera_info_alignment"],
                        "static_tf": preflight[
                            "auxiliary_observations"
                        ]["depth"]["static_tf_alignment"],
                    },
                },
                provenance={
                    "adapter": _ADAPTER,
                    "acquisition_id": acquisition_id,
                    "input_policy": preflight["input_policy"],
                    "window": preflight["window"],
                    "clock": preflight["clock"],
                    "gnss": preflight["gnss"],
                    "gnss_quality": {
                        "covariance": preflight["gnss"]["primary_covariance"]
                    },
                    "calibration_audit": preflight["calibration"],
                    "sampling": sampling,
                    "auxiliary_staging": auxiliary_staging,
                    "rgb_observations_artifact": str(rgb_artifact.resolve()),
                    "runtime_resolution": runtime_resolution_plain(cfg),
                    "configuration": configuration_evidence(cfg),
                    "imu_used": False,
                    "wheel_odometry_used": False,
                    "oracle_used": False,
                    "postprocessed_gnss_used": options["gnss_source"] == "offline_ppk",
                },
                acquisition_id=acquisition_id,
            )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "preflight":
        from rtk_splat.workflows.configio import load_config

        print(json.dumps(preflight_config(load_config(arguments.config)), indent=2))
        return 0
    raise AssertionError(arguments.command)


if __name__ == "__main__":
    raise SystemExit(_main())
