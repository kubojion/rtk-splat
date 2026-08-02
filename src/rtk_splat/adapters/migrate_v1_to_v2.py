"""One-time, fail-closed migration from the validated v1 headland layout.

This is deliberately an adapter, not a compatibility path in the mapping
core.  It reads a v1 segment plus the metric-integrity observation archive and
publishes a new canonical v2 segment without modifying or copying bulk source
data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rtk_splat.adapters.synchronization import TimestampMatchError, nearest_matches
from rtk_splat.core.segment import (
    POSITION_QUALITY_VOCABULARY,
    SegmentReader,
    SegmentWriter,
    normalize_navsat_position_quality,
)


DEFAULT_ASSOCIATION_TOLERANCE_NS = 150_000_000


class MigrationError(ValueError):
    """The v1 evidence cannot be migrated without guessing."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MigrationError(f"{path} must contain a JSON object")
    return value


def _npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise MigrationError(f"cannot read {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp_array(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise MigrationError(f"{label} must contain integer nanoseconds")
    converted = array.astype(np.int64)
    if not np.array_equal(array, converted):
        raise MigrationError(f"{label} cannot be represented exactly as int64")
    return converted


def _require(values: dict[str, Any], names: set[str], label: str) -> None:
    missing = sorted(names.difference(values))
    if missing:
        raise MigrationError(f"{label} missing: {', '.join(missing)}")


def _stream_count(
    values: dict[str, np.ndarray], prefix: str, required: set[str]
) -> int:
    _require(values, required, "observation archive")
    count = len(values[next(iter(required))])
    for name, value in values.items():
        if name.startswith(prefix) and (value.ndim == 0 or value.shape[0] != count):
            raise MigrationError(
                f"{name} count disagrees with {prefix.rstrip('_')} stream"
            )
    return count


def _nearest_all(
    reference_ns: np.ndarray,
    sample_ns: np.ndarray,
    tolerance_ns: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        matches = nearest_matches(
            reference_ns, sample_ns, tolerance_ns=tolerance_ns
        )
    except TimestampMatchError as exc:
        raise MigrationError(f"{label}: {exc}") from exc
    if matches.match_count != len(reference_ns):
        worst = matches.unmatched_reference_indices[:5].tolist()
        raise MigrationError(
            f"{label}: {len(reference_ns) - matches.match_count} samples exceed "
            f"{tolerance_ns} ns tolerance; first frame indices: {worst}"
        )
    if not np.array_equal(
        matches.reference_indices, np.arange(len(reference_ns), dtype=np.int64)
    ):
        raise MigrationError(f"{label}: internal association order mismatch")
    return matches.sample_indices, matches.residual_ns


def _camera(value: dict[str, Any], label: str) -> dict[str, Any]:
    _require(
        value,
        {"width", "height", "fx", "fy", "cx", "cy", "d", "r", "p"},
        f"{label} calibration",
    )
    projection = np.asarray(value["p"], dtype=float).reshape(3, 4)
    return {
        # The migrated JPEGs are already rectified. Using the raw distortion
        # model here would apply lens correction twice in downstream tools.
        "model": "PINHOLE",
        "width": int(value["width"]),
        "height": int(value["height"]),
        "K": projection[:, :3].tolist(),
        "distortion": [],
        "rectification": np.asarray(value["r"], dtype=float).reshape(3, 3).tolist(),
        "projection": projection.tolist(),
        "source_camera_info": {
            "distortion_model": str(value.get("distortion_model", "UNKNOWN")),
            "K": [
                [float(value["fx"]), 0.0, float(value["cx"])],
                [0.0, float(value["fy"]), float(value["cy"])],
                [0.0, 0.0, 1.0],
            ],
            "distortion": np.asarray(value["d"], dtype=float).tolist(),
        },
    }


def _calibration(
    sidecar: dict[str, Any],
    sidecar_path: Path,
    config_path: Path | None,
) -> dict[str, Any]:
    _require(
        sidecar,
        {"left_calibration", "right_calibration"},
        "stereo sidecar",
    )
    left_raw = sidecar["left_calibration"]
    right_raw = sidecar["right_calibration"]
    left = _camera(left_raw, "left")
    right = _camera(right_raw, "right")
    left_p = np.asarray(left["projection"], dtype=float)
    right_p = np.asarray(right["projection"], dtype=float)
    if left_p[0, 0] == 0 or right_p[0, 0] == 0:
        raise MigrationError("stereo projection has zero focal length")
    translation_x = (
        right_p[0, 3] / right_p[0, 0]
        - left_p[0, 3] / left_p[0, 0]
    )
    if not np.isfinite(translation_x) or abs(translation_x) <= 0:
        raise MigrationError("cannot recover a non-zero rectified stereo baseline")
    declared = sidecar.get("baseline_m_camera_info")
    if declared is not None and not np.isclose(
        abs(translation_x), float(declared), atol=1e-6, rtol=1e-6
    ):
        raise MigrationError(
            "sidecar baseline disagrees with right-camera projection"
        )
    transform = np.eye(4)
    transform[0, 3] = translation_x
    provenance: dict[str, Any] = {
        "source": "v1 colmap_stereo sidecar",
        "sidecar_path": str(sidecar_path.resolve()),
        "sidecar_sha256": _sha256(sidecar_path),
        "source_schema_version": sidecar.get("schema_version"),
        "coordinate_conventions": sidecar.get("coordinate_conventions"),
    }
    rough_geometry = _rough_geometry(config_path)
    if rough_geometry is not None:
        provenance.update(rough_geometry["provenance"])
    result: dict[str, Any] = {
        "contract_version": 2,
        "cameras": {"left": left, "right": right},
        "T_right_left": transform.tolist(),
        "transform_conventions": {"T_right_left": "right_from_left"},
        "image_geometry": "rectified",
        "provenance": provenance,
    }
    if rough_geometry is not None:
        result["rough_sensor_geometry"] = rough_geometry["geometry"]
    return result


def _rough_geometry(config_path: Path | None) -> dict[str, Any] | None:
    if config_path is None:
        return None
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MigrationError(f"cannot read config {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise MigrationError(f"{config_path} must contain a YAML mapping")
    pose = config.get("pose", {})
    if not isinstance(pose, dict):
        raise MigrationError("config.pose must be a mapping")
    pose_keys = (
        "antenna_forward_m",
        "cam_forward_m",
        "cam_left_m",
        "cam_up_m",
        "pitch_down_deg",
        "yaw_offset_deg",
        "center_up_m",
        "left_eye_forward_m",
        "left_eye_left_m",
        "baseline_m",
    )
    geometry: dict[str, Any] = {
        "pose_parameters": {
            name: pose[name] for name in pose_keys if name in pose
        }
    }
    calibration = config.get("calibration", {})
    if isinstance(calibration, dict) and isinstance(
        calibration.get("physical_geometry"), dict
    ):
        geometry["physical_geometry"] = calibration["physical_geometry"]
    return {
        "geometry": geometry,
        "provenance": {
            "config_path": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
        },
    }


def _check_bulk(source: Path, n: int) -> None:
    for frame_id in range(n):
        required = (
            source / "images" / f"left_{frame_id:06d}.jpg",
            source / "images" / f"right_{frame_id:06d}.jpg",
            source / "depth" / f"{frame_id:06d}.npz",
        )
        for path in required:
            if not path.is_file():
                raise MigrationError(f"missing v1 frame artifact: {path}")


def _verify_stereo_hashes(
    source: Path, observations: dict[str, np.ndarray], n: int
) -> None:
    for side in ("left", "right"):
        name = f"stereo_{side}_sha256"
        if name not in observations:
            continue
        expected = np.asarray(observations[name])
        if expected.shape != (n,) or not np.issubdtype(expected.dtype, np.str_):
            raise MigrationError(f"{name} must contain one hexadecimal string per frame")
        for frame_id, digest in enumerate(expected.astype(str)):
            path = source / "images" / f"{side}_{frame_id:06d}.jpg"
            if len(digest) != 64 or _sha256(path) != digest.lower():
                raise MigrationError(f"{name} mismatch for {path}")


def _sensor_semantics(
    source_meta: dict[str, Any],
    observations: dict[str, np.ndarray],
    rough_geometry: dict[str, Any] | None,
) -> dict[str, Any]:
    origin = source_meta.get("world_origin")
    if not isinstance(origin, dict):
        raise MigrationError("v1 metadata lacks world_origin")
    try:
        latitude = float(origin["lat0"])
        longitude = float(origin["lon0"])
        altitude = float(
            origin.get("alt0_ellipsoidal_m", origin.get("alt0"))
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MigrationError("v1 world_origin is incomplete") from exc
    if not np.isfinite([latitude, longitude, altitude]).all():
        raise MigrationError("v1 world_origin must be finite")

    fix_frames = observations.get("fix_frame_id")
    if fix_frames is None:
        primary_frame = "primary_gnss_antenna"
    else:
        unique_frames = sorted(set(np.asarray(fix_frames).astype(str).tolist()))
        if len(unique_frames) != 1 or not unique_frames[0]:
            raise MigrationError("fix_frame_id must identify one GNSS antenna frame")
        primary_frame = unique_frames[0]

    physical: dict[str, Any] = {}
    pose_parameters: dict[str, Any] = {}
    if rough_geometry is not None:
        physical = rough_geometry["geometry"].get("physical_geometry", {})
        pose_parameters = rough_geometry["geometry"].get("pose_parameters", {})
    camera_frame = str(physical.get("camera_frame", "left_camera"))
    primary_frame = str(physical.get("primary_antenna_frame", primary_frame))
    secondary_frame = str(
        physical.get("secondary_antenna_frame", "secondary_gnss_antenna")
    )

    sigma = [0.30, 0.30, 0.30]
    if rough_geometry is not None:
        # The migration config provenance is retained separately. These are
        # uncertainty labels, not corrections applied by the migration.
        config_path = rough_geometry["provenance"].get("config_path")
        if config_path:
            try:
                config_data = yaml.safe_load(
                    Path(config_path).read_text(encoding="utf-8")
                )
                candidate = (
                    config_data.get("calibration", {})
                    .get("optimization", {})
                    .get("lever_prior_sigma_m")
                )
                if candidate is not None:
                    values = np.asarray(candidate, dtype=float)
                    if values.shape == (3,) and np.isfinite(values).all():
                        sigma = values.tolist()
            except (OSError, TypeError, ValueError, yaml.YAMLError):
                pass

    offset_ns = _clock_offset_ns(source_meta)

    crs = source_meta.get("crs")
    vertical_datum = "WGS84 ellipsoid"
    if isinstance(crs, dict):
        vertical_datum = str(crs.get("vertical_datum", vertical_datum))
    return {
        "coordinate_frame": {
            "type": "local_enu",
            "world_frame_id": "segment_local_enu",
            "units": "m",
            "origin_wgs84": {
                "latitude_deg": latitude,
                "longitude_deg": longitude,
                "ellipsoidal_altitude_m": altitude,
                "ellipsoid": "WGS84",
                "vertical_datum": vertical_datum,
            },
        },
        "timebase": {
            "frame_timestamp_source": "left_camera_header",
            "observation_timestamp_source": "GNSS/heading sensor_header",
            "unit": "ns",
            "association_clock_offset_ns": offset_ns,
        },
        "position_observation": {
            "type": "gnss",
            "quantity": "antenna_phase_center",
            "sensor_frame_id": primary_frame,
            "coordinates": "ENU_m",
            "covariance_frame": "ENU_m2",
            "validity_field": "position_valid",
            "quality_field": "position_quality",
            "quality_vocabulary": list(POSITION_QUALITY_VOCABULARY),
        },
        "heading_observation": {
            "vector": "primary_to_secondary",
            "components": ["north", "east", "down"],
            "primary_frame_id": primary_frame,
            "secondary_frame_id": secondary_frame,
        },
        "initial_pose": {
            "camera_frame_id": camera_frame,
            "position_quantity": "left_camera_center",
            "source": str(source_meta.get("pose_source", "v1 rough RTK pose")),
            "lever_arm_applied": True,
            "extrinsic_translation_sigma_m": sigma,
            "extrinsic_translation_sigma_frame_id": camera_frame,
            "pose_parameters": pose_parameters,
        },
        "depth_observation": {
            "format": "npz_depth_valid",
            "units": "m",
            "quantity": "optical_z",
            "aligned_to": "left",
            "invalid_convention": "valid=false and depth=0",
        },
    }


def _clock_offset_ns(source_meta: dict[str, Any]) -> int:
    offset_s = source_meta.get("time_offset_s", 0.0)
    try:
        offset_ns_float = float(offset_s) * 1.0e9
    except (TypeError, ValueError) as exc:
        raise MigrationError("v1 time_offset_s must be numeric") from exc
    offset_ns = int(round(offset_ns_float))
    if not np.isfinite(offset_ns_float) or not np.isclose(
        offset_ns_float, offset_ns, atol=1.0e-6
    ):
        raise MigrationError("v1 time_offset_s is not exactly representable in ns")
    return offset_ns


def _raw_prefixed(
    observations: dict[str, np.ndarray], prefix: str
) -> dict[str, np.ndarray]:
    return {
        f"raw_{name[len(prefix):]}": np.array(value, copy=True)
        for name, value in observations.items()
        if name.startswith(prefix)
    }


def _gnss(
    observations: dict[str, np.ndarray],
    frame_id: np.ndarray,
    frame_ns: np.ndarray,
    association_query_ns: np.ndarray,
    tolerance_ns: int,
) -> dict[str, np.ndarray]:
    required = {
        "fix_header_ns",
        "fix_log_ns",
        "fix_geodetic",
        "fix_enu_m",
        "fix_covariance_enu_m2",
        "fix_status",
        "fix_service",
        "fix_covariance_type",
    }
    n_fix = _stream_count(observations, "fix_", required)
    if n_fix == 0:
        raise MigrationError("GNSS fix stream is empty")
    fix_ns = _timestamp_array(observations["fix_header_ns"], "fix_header_ns")
    indices, residual = _nearest_all(
        association_query_ns, fix_ns, tolerance_ns, "frame-to-GNSS association"
    )

    raw_carrier = np.full(n_fix, -1, dtype=np.int16)
    pvt_names = [name for name in observations if name.startswith("pvt_")]
    if pvt_names:
        _stream_count(
            observations,
            "pvt_",
            {"pvt_header_ns", "pvt_carrier_solution"},
        )
        pvt_indices, _ = _nearest_all(
            fix_ns,
            _timestamp_array(observations["pvt_header_ns"], "pvt_header_ns"),
            tolerance_ns,
            "GNSS-to-PVT status association",
        )
        raw_carrier = observations["pvt_carrier_solution"][pvt_indices].astype(
            np.int16
        )
    raw_fix_status = observations["fix_status"].astype(np.int16)
    raw_enu = observations["fix_enu_m"].astype(np.float64)
    raw_covariance = observations["fix_covariance_enu_m2"].astype(np.float64)
    position_valid, position_quality = normalize_navsat_position_quality(
        raw_fix_status,
        raw_carrier,
        np.isfinite(raw_enu).all(axis=1)
        & np.isfinite(raw_covariance).all(axis=(1, 2)),
    )

    result = _raw_prefixed(observations, "fix_")
    result.update(
        {
            "frame_id": frame_id,
            "frame_timestamp_ns": frame_ns,
            "association_query_timestamp_ns": association_query_ns,
            "source_index": indices,
            "source_timestamp_ns": fix_ns[indices],
            "source_residual_ns": residual,
            "enu_m": raw_enu[indices],
            "covariance_enu_m2": raw_covariance[indices],
            "fix_status": raw_fix_status[indices],
            "carrier_status": raw_carrier[indices],
            "position_valid": position_valid[indices],
            "position_quality": position_quality[indices],
            "raw_timestamp_ns": fix_ns,
            "raw_log_timestamp_ns": _timestamp_array(
                observations["fix_log_ns"], "fix_log_ns"
            ),
            "raw_geodetic_deg_m": observations["fix_geodetic"].astype(np.float64),
            "raw_enu_m": raw_enu,
            "raw_covariance_enu_m2": raw_covariance,
            "raw_fix_status": raw_fix_status,
            "raw_carrier_status": raw_carrier,
        }
    )
    for name in pvt_names:
        result[name] = np.array(observations[name], copy=True)
    return result


def _heading(
    observations: dict[str, np.ndarray],
    frame_id: np.ndarray,
    frame_ns: np.ndarray,
    association_query_ns: np.ndarray,
    tolerance_ns: int,
) -> dict[str, np.ndarray]:
    required = {
        "relpos_header_ns",
        "relpos_log_ns",
        "relpos_ned_m",
        "relpos_accuracy_heading_deg",
        "relpos_carrier_solution",
        "relpos_flags",
    }
    n_relpos = _stream_count(observations, "relpos_", required)
    if n_relpos == 0:
        raise MigrationError("dual-antenna heading stream is empty")
    relpos_ns = _timestamp_array(
        observations["relpos_header_ns"], "relpos_header_ns"
    )
    indices, residual = _nearest_all(
        association_query_ns,
        relpos_ns,
        tolerance_ns,
        "frame-to-heading association",
    )
    flags = observations["relpos_flags"].astype(bool)
    if flags.shape != (n_relpos, 8):
        raise MigrationError("relpos_flags must have shape (M, 8)")
    valid = (
        flags[:, 0]
        & flags[:, 1]
        & flags[:, 2]
        & flags[:, 3]
        & ~flags[:, 4]
        & ~flags[:, 5]
        & flags[:, 6]
        & (observations["relpos_carrier_solution"].astype(int) >= 2)
    )
    if "baseline_good_for_calibration" in observations:
        supplied = observations["baseline_good_for_calibration"].astype(bool)
        if supplied.shape != (n_relpos,) or not np.array_equal(supplied, valid):
            raise MigrationError(
                "derived heading validity disagrees with audit validity"
            )
    result = _raw_prefixed(observations, "relpos_")
    result.update(
        {
            "frame_id": frame_id,
            "frame_timestamp_ns": frame_ns,
            "association_query_timestamp_ns": association_query_ns,
            "source_index": indices,
            "source_timestamp_ns": relpos_ns[indices],
            "source_residual_ns": residual,
            "baseline_ned_m": observations["relpos_ned_m"][indices].astype(
                np.float64
            ),
            "acc_heading_rad": np.deg2rad(
                observations["relpos_accuracy_heading_deg"][indices]
            ).astype(np.float64),
            "carrier_status": observations["relpos_carrier_solution"][
                indices
            ].astype(np.int16),
            "flags": flags[indices],
            "valid": valid[indices],
            "raw_timestamp_ns": relpos_ns,
            "raw_log_timestamp_ns": _timestamp_array(
                observations["relpos_log_ns"], "relpos_log_ns"
            ),
            "raw_baseline_ned_m": observations["relpos_ned_m"].astype(
                np.float64
            ),
            "raw_acc_heading_rad": np.deg2rad(
                observations["relpos_accuracy_heading_deg"]
            ).astype(np.float64),
            "raw_carrier_status": observations[
                "relpos_carrier_solution"
            ].astype(np.int16),
            "raw_flags": flags,
            "raw_valid": valid,
        }
    )
    return result


def migrate(
    source_segment: str | Path,
    destination_segment: str | Path,
    observations_archive: str | Path,
    *,
    config: str | Path | None = None,
    association_tolerance_ns: int = DEFAULT_ASSOCIATION_TOLERANCE_NS,
) -> SegmentReader:
    """Publish a new v2 segment from immutable v1 inputs."""
    source = Path(source_segment).expanduser().resolve()
    destination = Path(destination_segment).expanduser().resolve()
    observations_path = Path(observations_archive).expanduser().resolve()
    config_path = Path(config).expanduser().resolve() if config else None
    if not source.is_dir():
        raise MigrationError(f"source segment does not exist: {source}")
    if source == destination or source in destination.parents:
        raise MigrationError("destination must not be inside the source segment")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    if (
        isinstance(association_tolerance_ns, bool)
        or not isinstance(association_tolerance_ns, int)
        or association_tolerance_ns < 0
    ):
        raise MigrationError("association_tolerance_ns must be non-negative")

    meta_path = source / "segment_meta.json"
    manifest_path = source / "manifest.json"
    viewmats_path = source / "viewmats.npy"
    centers_path = source / "cam_centers.npy"
    sidecar_path = (
        source / "pose_artifacts" / "colmap_stereo" / "sidecar_config.json"
    )
    source_meta = _json(meta_path)
    manifest = _json(manifest_path)
    observations = _npz(observations_path)
    sidecar = _json(sidecar_path)
    _require(source_meta, {"n_frames"}, "v1 segment metadata")
    n = int(source_meta["n_frames"])
    if n <= 0:
        raise MigrationError("v1 n_frames must be positive")
    _check_bulk(source, n)
    _verify_stereo_hashes(source, observations, n)

    try:
        viewmats = np.load(viewmats_path, allow_pickle=False)
        centers = np.load(centers_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise MigrationError(f"cannot read v1 initial poses: {exc}") from exc
    if viewmats.shape != (n, 4, 4) or centers.shape != (n, 3):
        raise MigrationError("v1 pose arrays disagree with n_frames")
    stereo_required = {
        "stereo_frame_id",
        "stereo_left_header_ns",
        "stereo_left_log_ns",
        "stereo_right_header_ns",
        "stereo_right_log_ns",
        "stereo_delta_ns",
    }
    _stream_count(observations, "stereo_", stereo_required)
    raw_frame_id = np.asarray(observations["stereo_frame_id"])
    if not np.issubdtype(raw_frame_id.dtype, np.integer):
        raise MigrationError("stereo_frame_id must be integer")
    frame_id = raw_frame_id.astype(np.int64)
    if len(frame_id) != n or not np.array_equal(frame_id, np.arange(n)):
        raise MigrationError("stereo frame IDs/count disagree with v1 segment")
    left_ns = _timestamp_array(
        observations["stereo_left_header_ns"], "stereo_left_header_ns"
    )
    right_ns = _timestamp_array(
        observations["stereo_right_header_ns"], "stereo_right_header_ns"
    )
    clock_offset_ns = _clock_offset_ns(source_meta)
    pose_query_ns = left_ns + np.int64(clock_offset_ns)
    if np.any(
        pose_query_ns.astype(object)
        != left_ns.astype(object) + clock_offset_ns
    ):
        raise MigrationError("camera-to-GNSS clock offset overflows int64")
    residual_ns = right_ns - left_ns
    if not np.array_equal(residual_ns, observations["stereo_delta_ns"]):
        raise MigrationError("stored stereo_delta_ns disagrees with exact headers")

    frames: dict[str, np.ndarray] = {
        "frame_id": frame_id,
        "timestamp_ns": left_ns,
        "left_image_path": np.asarray(
            [f"images/left_{index:06d}.jpg" for index in range(n)]
        ),
        "right_image_path": np.asarray(
            [f"images/right_{index:06d}.jpg" for index in range(n)]
        ),
        "right_timestamp_ns": right_ns,
        "stereo_sync_residual_ns": residual_ns,
        "pose_query_timestamp_ns": pose_query_ns,
        "left_log_timestamp_ns": _timestamp_array(
            observations["stereo_left_log_ns"], "stereo_left_log_ns"
        ),
        "right_log_timestamp_ns": _timestamp_array(
            observations["stereo_right_log_ns"], "stereo_right_log_ns"
        ),
        "depth_path": np.asarray(
            [f"depth/{index:06d}.npz" for index in range(n)]
        ),
        "initial_viewmat": viewmats,
        "initial_camera_center_m": centers,
        "pose_valid": (
            np.isfinite(viewmats).all(axis=(1, 2))
            & np.isfinite(centers).all(axis=1)
        ),
    }
    for name in ("stereo_left_sha256", "stereo_right_sha256"):
        if name in observations:
            frames[name] = observations[name]

    gnss = _gnss(
        observations,
        frame_id,
        left_ns,
        pose_query_ns,
        association_tolerance_ns,
    )
    heading = _heading(
        observations,
        frame_id,
        left_ns,
        pose_query_ns,
        association_tolerance_ns,
    )
    calibration = _calibration(sidecar, sidecar_path, config_path)
    rough_geometry = _rough_geometry(config_path)
    semantics = _sensor_semantics(source_meta, observations, rough_geometry)
    meta = dict(source_meta)
    meta.update(
        {
            "contract_version": 2,
            "n_frames": n,
            "capabilities": {
                "stereo": True,
                "rgbd": False,
                "single_rtk": False,
                "dual_rtk": True,
                "depth_recorded": False,
                "depth_computed": True,
                "imu_present": False,
                "images_raw": False,
                "images_rectified": True,
            },
            **semantics,
            "migration": {
                "source_contract": 1,
                "source_segment": str(source),
                "observations_archive": str(observations_path),
                "association_tolerance_ns": association_tolerance_ns,
                "input_sha256": {
                    "segment_meta.json": _sha256(meta_path),
                    "manifest.json": _sha256(manifest_path),
                    "viewmats.npy": _sha256(viewmats_path),
                    "cam_centers.npy": _sha256(centers_path),
                    "observations.npz": _sha256(observations_path),
                    "sidecar_config.json": _sha256(sidecar_path),
                },
            },
        }
    )

    writer = SegmentWriter(destination)
    try:
        os.symlink(source / "images", writer.staging_dir / "images")
        os.symlink(source / "depth", writer.staging_dir / "depth")
        writer.write_frames(frames)
        writer.write_calibration(calibration)
        writer.write_meta(meta)
        writer.write_manifest(manifest)
        writer.write_observations("gnss", gnss)
        writer.write_observations("heading", heading)
        return writer.finalize()
    except Exception:
        writer.abort()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate one immutable v1 segment to contract v2"
    )
    parser.add_argument("--source-segment", type=Path, required=True)
    parser.add_argument("--destination-segment", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--association-tolerance-ms",
        type=float,
        default=DEFAULT_ASSOCIATION_TOLERANCE_NS / 1e6,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not np.isfinite(args.association_tolerance_ms):
        raise MigrationError("association tolerance must be finite")
    tolerance_ns = int(round(args.association_tolerance_ms * 1e6))
    migrated = migrate(
        args.source_segment,
        args.destination_segment,
        args.observations,
        config=args.config,
        association_tolerance_ns=tolerance_ns,
    )
    print(f"contract-v2 segment ready: {migrated.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
