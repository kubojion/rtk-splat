"""Canonical, dataset-independent segment contract (version 2).

Adapters write this contract; mapping backends only read it.  Timestamps are
integer nanoseconds so associations retain the source sensor time exactly.
Observation files contain one associated row per frame and may additionally
carry the complete source stream in ``raw_*`` arrays.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
import ctypes
import errno
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


CONTRACT_VERSION = 2
CAPABILITIES = (
    "stereo",
    "rgbd",
    "single_rtk",
    "dual_rtk",
    "depth_recorded",
    "depth_computed",
    "imu_present",
    "images_raw",
    "images_rectified",
)
OBSERVATION_KINDS = ("gnss", "heading", "imu")
POSITION_QUALITY_VOCABULARY = (
    "invalid",
    "unknown_valid",
    "standalone",
    "differential",
    "rtk_float",
    "rtk_fixed",
    "oracle",
)
JsonObject = dict[str, Any]
Arrays = dict[str, NDArray[Any]]


class SegmentContractError(ValueError):
    """A segment is incomplete, inconsistent, or unsafe to consume."""


def normalize_navsat_position_quality(
    fix_status: ArrayLike,
    carrier_status: ArrayLike,
    finite_evidence: ArrayLike,
) -> tuple[NDArray[np.bool_], NDArray[np.str_]]:
    """Normalize ROS NavSat/PVT status without discarding the raw status codes.

    ROS ``NavSatStatus.status == 0`` is a valid standalone fix.  A missing PVT
    carrier state is represented by ``-1`` and does not invalidate that fix.
    Adapters retain both raw integer arrays alongside these portable fields.
    """
    fix = np.asarray(fix_status)
    carrier = np.asarray(carrier_status)
    finite = np.asarray(finite_evidence)
    if (
        fix.ndim != 1
        or carrier.shape != fix.shape
        or finite.shape != fix.shape
        or not np.issubdtype(fix.dtype, np.integer)
        or not np.issubdtype(carrier.dtype, np.integer)
        or not np.issubdtype(finite.dtype, np.bool_)
    ):
        raise SegmentContractError(
            "NavSat normalization needs equal 1-D integer status arrays and "
            "a boolean finite-evidence mask"
        )
    valid = finite & (fix.astype(np.int64) >= 0)
    quality = np.full(fix.shape, "invalid", dtype="<U13")
    quality[valid] = "standalone"
    quality[valid & (fix.astype(np.int64) >= 1)] = "differential"
    quality[valid & (carrier.astype(np.int64) == 1)] = "rtk_float"
    quality[valid & (carrier.astype(np.int64) >= 2)] = "rtk_fixed"
    return valid.astype(bool, copy=False), quality


def _read_json(path: Path) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SegmentContractError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise SegmentContractError(f"{path.name} must contain a JSON object")
    return value


def _read_npz(path: Path) -> Arrays:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise SegmentContractError(f"cannot read {path.name}: {exc}") from exc


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_bytes(path, payload.encode("utf-8"))


def _atomic_npz(path: Path, values: Mapping[str, ArrayLike]) -> None:
    arrays = {name: np.asarray(value) for name, value in values.items()}
    bad = [name for name, value in arrays.items() if value.dtype.hasobject]
    if bad:
        raise SegmentContractError(
            f"{path.name} contains unsafe object arrays: {', '.join(bad)}"
        )
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _require_keys(values: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required.difference(values))
    if missing:
        raise SegmentContractError(f"{label} missing: {', '.join(missing)}")


def _vector(values: Arrays, name: str, n: int, label: str) -> NDArray[Any]:
    value = values[name]
    if value.shape != (n,):
        raise SegmentContractError(
            f"{label}.{name} must have shape ({n},), got {value.shape}"
        )
    return value


def _matrix(
    values: Arrays, name: str, shape: tuple[int, ...], label: str
) -> NDArray[Any]:
    value = values[name]
    if value.shape != shape:
        raise SegmentContractError(
            f"{label}.{name} must have shape {shape}, got {value.shape}"
        )
    return value


def _integer(values: NDArray[Any], label: str) -> None:
    if not np.issubdtype(values.dtype, np.integer):
        raise SegmentContractError(f"{label} must use an integer dtype")


def _timestamp(values: NDArray[Any], label: str) -> None:
    if values.dtype != np.dtype(np.int64):
        raise SegmentContractError(f"{label} must use int64 nanoseconds")


def _strictly_increasing(values: NDArray[Any], label: str) -> None:
    _integer(values, label)
    if len(values) > 1 and np.any(np.diff(values.astype(np.int64)) <= 0):
        raise SegmentContractError(f"{label} must be strictly increasing")


def _nondecreasing(values: NDArray[Any], label: str) -> None:
    _integer(values, label)
    if len(values) > 1 and np.any(np.diff(values.astype(np.int64)) < 0):
        raise SegmentContractError(f"{label} must be nondecreasing")


def _safe_relative_path(value: Any, label: str, *, allow_empty: bool) -> str:
    text = str(value)
    if allow_empty and not text:
        return text
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != text
    ):
        raise SegmentContractError(f"{label} must be a normalized relative path")
    return text


def _validate_camera(camera: Any, label: str) -> None:
    if not isinstance(camera, dict):
        raise SegmentContractError(f"calibration.cameras.{label} must be an object")
    _require_keys(
        camera, {"model", "width", "height", "K", "distortion"}, f"camera {label}"
    )
    if not isinstance(camera["model"], str) or not camera["model"]:
        raise SegmentContractError(f"camera {label}.model must be non-empty")
    if not isinstance(camera["width"], int) or camera["width"] <= 0:
        raise SegmentContractError(f"camera {label}.width must be positive")
    if not isinstance(camera["height"], int) or camera["height"] <= 0:
        raise SegmentContractError(f"camera {label}.height must be positive")
    try:
        k = np.asarray(camera["K"], dtype=float)
        distortion = np.asarray(camera["distortion"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise SegmentContractError(
            f"camera {label} calibration must be numeric"
        ) from exc
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise SegmentContractError(f"camera {label}.K must be a valid 3x3 matrix")
    if distortion.ndim != 1 or not np.isfinite(distortion).all():
        raise SegmentContractError(f"camera {label}.distortion must be a 1-D array")


def _validate_transform(value: Any) -> None:
    try:
        transform = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SegmentContractError(
            "calibration.T_right_left must be numeric"
        ) from exc
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise SegmentContractError("calibration.T_right_left must be finite 4x4")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise SegmentContractError("calibration.T_right_left has invalid last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise SegmentContractError("calibration.T_right_left rotation is invalid")
    if np.linalg.det(rotation) <= 0 or np.linalg.norm(transform[:3, 3]) <= 0:
        raise SegmentContractError("calibration.T_right_left baseline is invalid")


def _validate_raw_stream(values: Arrays, label: str) -> None:
    raw_names = [name for name in values if name.startswith("raw_")]
    if not raw_names:
        return
    if "raw_timestamp_ns" not in values:
        raise SegmentContractError(f"{label} raw stream needs raw_timestamp_ns")
    timestamps = values["raw_timestamp_ns"]
    if timestamps.ndim != 1:
        raise SegmentContractError(f"{label}.raw_timestamp_ns must be one-dimensional")
    _timestamp(timestamps, f"{label}.raw_timestamp_ns")
    # Some sensors legitimately repeat a header stamp. Preserve those samples
    # and their bag-log order instead of silently dropping evidence.
    _nondecreasing(timestamps, f"{label}.raw_timestamp_ns")
    count = len(timestamps)
    for name in raw_names:
        if values[name].ndim == 0 or values[name].shape[0] != count:
            raise SegmentContractError(
                f"{label}.{name} first dimension must equal raw sample count {count}"
            )


def _validate_associations(
    values: Arrays, frames: Arrays, label: str, extra_required: set[str]
) -> None:
    required = {
        "frame_id",
        "frame_timestamp_ns",
        "source_index",
        "source_timestamp_ns",
    } | extra_required
    _require_keys(values, required, label)
    n = len(frames["frame_id"])
    for name in required:
        if values[name].ndim == 0 or values[name].shape[0] != n:
            raise SegmentContractError(
                f"{label}.{name} first dimension must equal frame count {n}"
            )
    for name in ("frame_id", "frame_timestamp_ns", "source_index", "source_timestamp_ns"):
        _integer(values[name], f"{label}.{name}")
    for name in ("frame_timestamp_ns", "source_timestamp_ns"):
        _timestamp(values[name], f"{label}.{name}")
    if not np.array_equal(values["frame_id"], frames["frame_id"]):
        raise SegmentContractError(f"{label}.frame_id does not match frames.npz")
    if not np.array_equal(values["frame_timestamp_ns"], frames["timestamp_ns"]):
        raise SegmentContractError(
            f"{label}.frame_timestamp_ns does not match frames.npz"
        )
    _validate_raw_stream(values, label)
    if "raw_timestamp_ns" in values:
        source_index = values["source_index"].astype(np.int64)
        raw_time = values["raw_timestamp_ns"].astype(np.int64)
        if np.any(source_index < 0) or np.any(source_index >= len(raw_time)):
            raise SegmentContractError(f"{label}.source_index is outside raw stream")
        if not np.array_equal(
            values["source_timestamp_ns"].astype(np.int64), raw_time[source_index]
        ):
            raise SegmentContractError(
                f"{label}.source_timestamp_ns does not match raw source_index"
            )


@dataclass(frozen=True)
class SegmentReader:
    """Typed access to a validated version-2 segment."""

    root: Path

    def __init__(self, root: str | Path):
        object.__setattr__(self, "root", Path(root))

    @property
    def frames(self) -> Arrays:
        return _read_npz(self.root / "frames.npz")

    @property
    def calibration(self) -> JsonObject:
        return _read_json(self.root / "calibration.json")

    @property
    def meta(self) -> JsonObject:
        return _read_json(self.root / "segment_meta.json")

    @property
    def manifest(self) -> JsonObject:
        return _read_json(self.root / "manifest.json")

    def observations(self, kind: str, *, required: bool = True) -> Arrays | None:
        if kind not in OBSERVATION_KINDS:
            raise ValueError(f"unknown observation kind: {kind}")
        path = self.root / "observations" / f"{kind}.npz"
        if not path.exists() and not required:
            return None
        return _read_npz(path)

    def validate(self, *, check_files: bool = True) -> "SegmentReader":
        if not self.root.is_dir():
            raise SegmentContractError(f"segment does not exist: {self.root}")
        frames = self.frames
        meta = self.meta
        calibration = self.calibration
        manifest = self.manifest

        _require_keys(
            frames, {"frame_id", "timestamp_ns", "left_image_path"}, "frames.npz"
        )
        frame_ids = frames["frame_id"]
        timestamps = frames["timestamp_ns"]
        if frame_ids.ndim != 1 or timestamps.shape != frame_ids.shape:
            raise SegmentContractError("frame IDs and timestamps must be 1-D and equal")
        _strictly_increasing(frame_ids, "frames.frame_id")
        _timestamp(timestamps, "frames.timestamp_ns")
        _strictly_increasing(timestamps, "frames.timestamp_ns")
        n = len(frame_ids)
        if n == 0:
            raise SegmentContractError("a segment must contain at least one frame")
        if not np.array_equal(frame_ids, np.arange(n, dtype=frame_ids.dtype)):
            raise SegmentContractError("frames.frame_id must be contiguous from zero")

        _require_keys(meta, {"contract_version", "n_frames", "capabilities"}, "meta")
        if meta["contract_version"] != CONTRACT_VERSION:
            raise SegmentContractError(
                f"contract_version must be {CONTRACT_VERSION}"
            )
        if meta["n_frames"] != n:
            raise SegmentContractError("segment_meta.n_frames does not match frames")
        capabilities = meta["capabilities"]
        if not isinstance(capabilities, dict):
            raise SegmentContractError("segment_meta.capabilities must be an object")
        _require_keys(capabilities, set(CAPABILITIES), "capabilities")
        if any(type(capabilities[name]) is not bool for name in CAPABILITIES):
            raise SegmentContractError("all declared capabilities must be booleans")
        if not (capabilities["stereo"] or capabilities["rgbd"]):
            raise SegmentContractError("segment must declare stereo and/or rgbd")
        if not (capabilities["images_raw"] or capabilities["images_rectified"]):
            raise SegmentContractError(
                "declare images_raw and/or images_rectified"
            )
        if capabilities["images_raw"] == capabilities["images_rectified"]:
            raise SegmentContractError(
                "exactly one of images_raw/images_rectified must be true"
            )
        if capabilities["single_rtk"] and capabilities["dual_rtk"]:
            raise SegmentContractError(
                "single_rtk and dual_rtk are mutually exclusive acquisition modes"
            )
        if capabilities["depth_recorded"] and capabilities["depth_computed"]:
            raise SegmentContractError(
                "depth_recorded and depth_computed are mutually exclusive"
            )
        _validate_coordinate_semantics(meta, capabilities)

        cameras = calibration.get("cameras")
        if calibration.get("contract_version") != CONTRACT_VERSION:
            raise SegmentContractError("calibration.contract_version must be 2")
        if not isinstance(cameras, dict) or "left" not in cameras:
            raise SegmentContractError("calibration.cameras.left is required")
        _validate_camera(cameras["left"], "left")
        if capabilities["stereo"]:
            if "right" not in cameras or "T_right_left" not in calibration:
                raise SegmentContractError(
                    "stereo calibration needs right camera and T_right_left"
                )
            _validate_camera(cameras["right"], "right")
            _validate_transform(calibration["T_right_left"])
            conventions = calibration.get("transform_conventions")
            if not isinstance(conventions, dict) or (
                conventions.get("T_right_left") != "right_from_left"
            ):
                raise SegmentContractError(
                    "calibration must declare T_right_left as right_from_left"
                )

        path_fields = ("left_image_path", "right_image_path", "depth_path")
        for field in path_fields:
            if field not in frames:
                continue
            paths = _vector(frames, field, n, "frames")
            if not np.issubdtype(paths.dtype, np.str_):
                raise SegmentContractError(f"frames.{field} must contain strings")
            for index, value in enumerate(paths):
                relative = _safe_relative_path(
                    value, f"frames.{field}[{index}]", allow_empty=field != "left_image_path"
                )
                if relative and check_files and not (self.root / relative).is_file():
                    raise SegmentContractError(f"missing frame file: {relative}")
        if capabilities["stereo"]:
            _require_keys(
                frames,
                {
                    "right_image_path",
                    "right_timestamp_ns",
                    "stereo_sync_residual_ns",
                },
                "stereo frames",
            )
            right_time = _vector(frames, "right_timestamp_ns", n, "frames")
            residual = _vector(frames, "stereo_sync_residual_ns", n, "frames")
            _integer(right_time, "frames.right_timestamp_ns")
            _integer(residual, "frames.stereo_sync_residual_ns")
            _timestamp(right_time, "frames.right_timestamp_ns")
            _timestamp(residual, "frames.stereo_sync_residual_ns")
            _strictly_increasing(right_time, "frames.right_timestamp_ns")
            if not np.array_equal(
                residual.astype(np.int64),
                right_time.astype(np.int64) - timestamps.astype(np.int64),
            ):
                raise SegmentContractError(
                    "stereo_sync_residual_ns must equal right minus left timestamp"
                )
        if capabilities["stereo"] and np.any(frames["right_image_path"].astype(str) == ""):
            raise SegmentContractError("stereo segment has empty right_image_path")
        if (
            capabilities["rgbd"]
            or capabilities["depth_recorded"]
            or capabilities["depth_computed"]
        ) and "depth_path" not in frames:
            raise SegmentContractError("RGB-D/recorded-depth segment needs depth_path")
        if (
            capabilities["rgbd"]
            or capabilities["depth_recorded"]
            or capabilities["depth_computed"]
        ) and np.any(frames["depth_path"].astype(str) == ""):
            raise SegmentContractError("RGB-D/recorded-depth segment has empty depth_path")
        if (
            capabilities["rgbd"]
            or capabilities["depth_recorded"]
            or capabilities["depth_computed"]
        ):
            _validate_depth_semantics(meta)
        _validate_initial_poses(frames, n)
        if {"initial_viewmat", "initial_camera_center_m", "pose_valid"} <= set(frames):
            _validate_initial_pose_semantics(meta, capabilities)

        _validate_manifest(manifest, frame_ids)
        gnss = self.observations("gnss")
        assert gnss is not None
        _validate_gnss(gnss, frames, capabilities)
        heading = self.observations("heading", required=capabilities["dual_rtk"])
        if heading is not None:
            _validate_heading(heading, frames)
        imu = self.observations("imu", required=capabilities["imu_present"])
        if imu is not None:
            _validate_imu(imu)
        return self


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SegmentContractError(f"{label} must be a non-empty string")
    return value


def _validate_coordinate_semantics(meta: JsonObject, capabilities: JsonObject) -> None:
    frame = meta.get("coordinate_frame")
    if not isinstance(frame, dict):
        raise SegmentContractError("segment_meta.coordinate_frame is required")
    _require_keys(frame, {"type", "world_frame_id", "units"}, "coordinate_frame")
    kind = frame["type"]
    if kind not in {"local_enu", "cartesian_metric"}:
        raise SegmentContractError(
            "coordinate_frame.type must be local_enu or cartesian_metric"
        )
    _nonempty_string(frame["world_frame_id"], "coordinate_frame.world_frame_id")
    if frame["units"] != "m":
        raise SegmentContractError("coordinate_frame.units must be 'm'")
    if kind == "local_enu":
        origin = frame.get("origin_wgs84")
        if not isinstance(origin, dict):
            raise SegmentContractError("local_enu needs coordinate_frame.origin_wgs84")
        _require_keys(
            origin,
            {
                "latitude_deg",
                "longitude_deg",
                "ellipsoidal_altitude_m",
                "ellipsoid",
                "vertical_datum",
            },
            "coordinate_frame.origin_wgs84",
        )
        numeric = np.asarray(
            [
                origin["latitude_deg"],
                origin["longitude_deg"],
                origin["ellipsoidal_altitude_m"],
            ],
            dtype=float,
        )
        if not np.isfinite(numeric).all():
            raise SegmentContractError("coordinate_frame origin must be finite")
        if not -90 <= numeric[0] <= 90 or not -180 <= numeric[1] <= 180:
            raise SegmentContractError("coordinate_frame origin latitude/longitude invalid")
        _nonempty_string(origin["ellipsoid"], "coordinate_frame origin ellipsoid")
        _nonempty_string(
            origin["vertical_datum"], "coordinate_frame origin vertical_datum"
        )

    timebase = meta.get("timebase")
    if not isinstance(timebase, dict):
        raise SegmentContractError("segment_meta.timebase is required")
    _require_keys(
        timebase,
        {
            "frame_timestamp_source",
            "observation_timestamp_source",
            "unit",
            "association_clock_offset_ns",
        },
        "timebase",
    )
    _nonempty_string(
        timebase["frame_timestamp_source"], "timebase.frame_timestamp_source"
    )
    _nonempty_string(
        timebase["observation_timestamp_source"],
        "timebase.observation_timestamp_source",
    )
    if timebase["unit"] != "ns":
        raise SegmentContractError("timebase.unit must be 'ns'")
    if type(timebase["association_clock_offset_ns"]) is not int:
        raise SegmentContractError(
            "timebase.association_clock_offset_ns must be an integer"
        )

    observation = meta.get("position_observation")
    if not isinstance(observation, dict):
        raise SegmentContractError("segment_meta.position_observation is required")
    _require_keys(
        observation,
        {
            "type",
            "quantity",
            "sensor_frame_id",
            "coordinates",
            "covariance_frame",
            "validity_field",
            "quality_field",
            "quality_vocabulary",
        },
        "position_observation",
    )
    for name in ("type", "quantity", "sensor_frame_id"):
        _nonempty_string(observation[name], f"position_observation.{name}")
    if observation["coordinates"] != "ENU_m":
        raise SegmentContractError("position_observation.coordinates must be ENU_m")
    if observation["covariance_frame"] != "ENU_m2":
        raise SegmentContractError(
            "position_observation.covariance_frame must be ENU_m2"
        )
    if observation["validity_field"] != "position_valid":
        raise SegmentContractError(
            "position_observation.validity_field must be position_valid"
        )
    if observation["quality_field"] != "position_quality":
        raise SegmentContractError(
            "position_observation.quality_field must be position_quality"
        )
    if observation["quality_vocabulary"] != list(POSITION_QUALITY_VOCABULARY):
        raise SegmentContractError(
            "position_observation.quality_vocabulary is not the contract "
            "v2 vocabulary"
        )
    is_rtk = capabilities["single_rtk"] or capabilities["dual_rtk"]
    if is_rtk and (
        observation["type"] != "gnss"
        or observation["quantity"] != "antenna_phase_center"
    ):
        raise SegmentContractError(
            "RTK position observations must identify a GNSS antenna phase center"
        )
    if capabilities["dual_rtk"]:
        heading = meta.get("heading_observation")
        if not isinstance(heading, dict):
            raise SegmentContractError(
                "dual_rtk needs segment_meta.heading_observation"
            )
        _require_keys(
            heading,
            {
                "vector",
                "components",
                "primary_frame_id",
                "secondary_frame_id",
            },
            "heading_observation",
        )
        if heading["vector"] != "primary_to_secondary":
            raise SegmentContractError(
                "heading_observation.vector must be primary_to_secondary"
            )
        if heading["components"] != ["north", "east", "down"]:
            raise SegmentContractError(
                "heading_observation.components must be north/east/down"
            )
        _nonempty_string(
            heading["primary_frame_id"], "heading_observation.primary_frame_id"
        )
        _nonempty_string(
            heading["secondary_frame_id"], "heading_observation.secondary_frame_id"
        )


def _validate_depth_semantics(meta: JsonObject) -> None:
    depth = meta.get("depth_observation")
    if not isinstance(depth, dict):
        raise SegmentContractError("depth data needs segment_meta.depth_observation")
    _require_keys(
        depth,
        {"format", "units", "quantity", "aligned_to", "invalid_convention"},
        "depth_observation",
    )
    if depth["format"] != "npz_depth_valid":
        raise SegmentContractError(
            "depth_observation.format must be npz_depth_valid"
        )
    if depth["units"] != "m" or depth["quantity"] != "optical_z":
        raise SegmentContractError("depth must be optical_z in metres")
    if depth["aligned_to"] not in {"left", "rgb"}:
        raise SegmentContractError("depth_observation.aligned_to must be left or rgb")
    _nonempty_string(
        depth["invalid_convention"], "depth_observation.invalid_convention"
    )


def _validate_initial_pose_semantics(
    meta: JsonObject, capabilities: JsonObject
) -> None:
    pose = meta.get("initial_pose")
    if not isinstance(pose, dict):
        raise SegmentContractError("initial pose arrays need segment_meta.initial_pose")
    _require_keys(
        pose,
        {
            "camera_frame_id",
            "position_quantity",
            "source",
            "lever_arm_applied",
            "extrinsic_translation_sigma_m",
            "extrinsic_translation_sigma_frame_id",
        },
        "initial_pose",
    )
    _nonempty_string(pose["camera_frame_id"], "initial_pose.camera_frame_id")
    _nonempty_string(pose["source"], "initial_pose.source")
    if pose["position_quantity"] != "left_camera_center":
        raise SegmentContractError(
            "initial_pose.position_quantity must be left_camera_center"
        )
    if type(pose["lever_arm_applied"]) is not bool:
        raise SegmentContractError("initial_pose.lever_arm_applied must be boolean")
    sigma_frame = _nonempty_string(
        pose["extrinsic_translation_sigma_frame_id"],
        "initial_pose.extrinsic_translation_sigma_frame_id",
    )
    if sigma_frame != pose["camera_frame_id"]:
        raise SegmentContractError(
            "extrinsic translation sigma must be expressed in the declared "
            "camera frame"
        )
    sigma = np.asarray(pose["extrinsic_translation_sigma_m"], dtype=float)
    if sigma.shape != (3,) or np.any(~np.isfinite(sigma)) or np.any(sigma < 0):
        raise SegmentContractError(
            "initial_pose.extrinsic_translation_sigma_m must be finite [x,y,z]"
        )
    if (
        capabilities["single_rtk"] or capabilities["dual_rtk"]
    ) and not pose["lever_arm_applied"]:
        raise SegmentContractError(
            "RTK-derived camera centres must explicitly apply the lever arm"
        )


def _validate_manifest(manifest: JsonObject, frame_ids: NDArray[Any]) -> None:
    _require_keys(manifest, {"train", "val", "test"}, "manifest")
    splits: list[int] = []
    for name in ("train", "val", "test"):
        values = manifest[name]
        if not isinstance(values, list) or any(type(item) is not int for item in values):
            raise SegmentContractError(f"manifest.{name} must be a list of frame IDs")
        if len(values) != len(set(values)):
            raise SegmentContractError(f"manifest.{name} contains duplicate IDs")
        splits.extend(values)
    if len(splits) != len(set(splits)):
        raise SegmentContractError("manifest splits overlap")
    if set(splits) != set(frame_ids.astype(int).tolist()):
        raise SegmentContractError("manifest splits must cover every segment frame")


def _validate_initial_poses(frames: Arrays, n: int) -> None:
    names = {"initial_viewmat", "initial_camera_center_m", "pose_valid"}
    present = names.intersection(frames)
    if present and present != names:
        raise SegmentContractError(
            "initial poses require initial_viewmat, initial_camera_center_m, and pose_valid"
        )
    if not present:
        return
    viewmats = _matrix(frames, "initial_viewmat", (n, 4, 4), "frames")
    centers = _matrix(
        frames, "initial_camera_center_m", (n, 3), "frames"
    )
    valid = _vector(frames, "pose_valid", n, "frames")
    if not np.issubdtype(valid.dtype, np.bool_):
        raise SegmentContractError("frames.pose_valid must be boolean")
    for index in np.flatnonzero(valid):
        viewmat = viewmats[index]
        center = centers[index]
        if not np.isfinite(viewmat).all() or not np.isfinite(center).all():
            raise SegmentContractError("valid initial poses must be finite")
        if not np.allclose(viewmat[3], [0, 0, 0, 1], atol=1e-8):
            raise SegmentContractError("initial_viewmat has invalid last row")
        rotation = viewmat[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise SegmentContractError("initial_viewmat rotation is invalid")
        derived_center = -(rotation.T @ viewmat[:3, 3])
        if not np.allclose(center, derived_center, atol=1e-5, rtol=1e-6):
            raise SegmentContractError(
                "initial_camera_center_m disagrees with initial_viewmat"
            )


def _validate_gnss(
    values: Arrays, frames: Arrays, capabilities: JsonObject
) -> None:
    label = "observations/gnss.npz"
    _validate_associations(
        values,
        frames,
        label,
        {
            "enu_m",
            "covariance_enu_m2",
            "fix_status",
            "carrier_status",
            "position_valid",
            "position_quality",
        },
    )
    n = len(frames["frame_id"])
    enu = _matrix(values, "enu_m", (n, 3), label)
    covariance = _matrix(values, "covariance_enu_m2", (n, 3, 3), label)
    _vector(values, "fix_status", n, label)
    _vector(values, "carrier_status", n, label)
    position_valid = _vector(values, "position_valid", n, label)
    position_quality = _vector(values, "position_quality", n, label)
    if not np.issubdtype(enu.dtype, np.floating):
        raise SegmentContractError(f"{label}.enu_m must be floating point")
    if not np.issubdtype(covariance.dtype, np.floating):
        raise SegmentContractError(
            f"{label}.covariance_enu_m2 must be floating point"
        )
    enu_finite = np.isfinite(enu).all(axis=1)
    enu_unknown = np.isnan(enu).all(axis=1)
    if not np.all(enu_finite | enu_unknown):
        raise SegmentContractError("GNSS ENU rows must be finite or all NaN")
    for name in ("fix_status", "carrier_status"):
        _integer(values[name], f"{label}.{name}")
    if not np.issubdtype(position_valid.dtype, np.bool_):
        raise SegmentContractError(f"{label}.position_valid must be boolean")
    if not np.issubdtype(position_quality.dtype, np.str_):
        raise SegmentContractError(f"{label}.position_quality must contain strings")
    unknown_quality = sorted(
        set(position_quality.astype(str)).difference(POSITION_QUALITY_VOCABULARY)
    )
    if unknown_quality:
        raise SegmentContractError(
            f"{label}.position_quality has unknown values: "
            f"{', '.join(unknown_quality)}"
        )
    if np.any(position_valid & (position_quality == "invalid")) or np.any(
        ~position_valid & (position_quality != "invalid")
    ):
        raise SegmentContractError(
            "position_valid and position_quality validity disagree"
        )
    if np.any(position_valid & ~enu_finite):
        raise SegmentContractError("valid GNSS positions must contain finite ENU")
    finite = np.isfinite(covariance).all(axis=(1, 2))
    unknown = np.isnan(covariance).all(axis=(1, 2))
    if not np.all(finite | unknown):
        raise SegmentContractError("GNSS covariance rows must be finite or all NaN")
    for matrix in covariance[finite]:
        if not np.allclose(matrix, matrix.T, atol=1e-10):
            raise SegmentContractError("GNSS covariance must be symmetric")
        if np.linalg.eigvalsh(matrix).min() < -1e-10:
            raise SegmentContractError("GNSS covariance must be positive semidefinite")
    if capabilities["single_rtk"] or capabilities["dual_rtk"]:
        if np.any(position_valid & ~finite):
            raise SegmentContractError(
                "valid RTK positions require full finite covariance"
            )


def _validate_heading(values: Arrays, frames: Arrays) -> None:
    label = "observations/heading.npz"
    _validate_associations(
        values,
        frames,
        label,
        {"baseline_ned_m", "acc_heading_rad", "valid"},
    )
    n = len(frames["frame_id"])
    _matrix(values, "baseline_ned_m", (n, 3), label)
    _vector(values, "acc_heading_rad", n, label)
    valid = _vector(values, "valid", n, label)
    if not np.issubdtype(values["baseline_ned_m"].dtype, np.floating):
        raise SegmentContractError(f"{label}.baseline_ned_m must be floating point")
    if not np.issubdtype(values["acc_heading_rad"].dtype, np.floating):
        raise SegmentContractError(f"{label}.acc_heading_rad must be floating point")
    if not np.issubdtype(valid.dtype, np.bool_):
        raise SegmentContractError(f"{label}.valid must be boolean")
    if np.any(~np.isfinite(values["baseline_ned_m"][valid])):
        raise SegmentContractError("valid heading baselines must be finite")
    accuracy = values["acc_heading_rad"][valid]
    if np.any(~np.isfinite(accuracy)) or np.any(accuracy < 0):
        raise SegmentContractError("valid heading accuracy must be finite and nonnegative")


def _validate_imu(values: Arrays) -> None:
    label = "observations/imu.npz"
    _require_keys(values, {"timestamp_ns", "accel_mps2", "gyro_radps"}, label)
    timestamps = values["timestamp_ns"]
    if timestamps.ndim != 1:
        raise SegmentContractError(f"{label}.timestamp_ns must be one-dimensional")
    _timestamp(timestamps, f"{label}.timestamp_ns")
    _strictly_increasing(timestamps, f"{label}.timestamp_ns")
    n = len(timestamps)
    _matrix(values, "accel_mps2", (n, 3), label)
    _matrix(values, "gyro_radps", (n, 3), label)
    if "orientation_xyzw" in values:
        _matrix(values, "orientation_xyzw", (n, 4), label)


class SegmentWriter:
    """Build a new segment in staging and atomically publish it once validated."""

    def __init__(self, destination: str | Path):
        self.destination = Path(destination)
        if self.destination.exists():
            raise FileExistsError(f"refusing to modify existing segment: {destination}")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.staging_dir = self.destination.with_name(
            f".{self.destination.name}.writing-{uuid.uuid4().hex}"
        )
        self.staging_dir.mkdir()
        (self.staging_dir / "observations").mkdir()
        self._finished = False

    def directory(self, relative: str) -> Path:
        """Create and return a safe directory inside this writer's staging area."""
        path = _safe_relative_path(relative, "directory", allow_empty=False)
        target = self.staging_dir / path
        target.mkdir(parents=True, exist_ok=True)
        return target

    def write_frames(self, values: Mapping[str, ArrayLike]) -> None:
        _atomic_npz(self.staging_dir / "frames.npz", values)

    def write_calibration(self, value: Mapping[str, Any]) -> None:
        _atomic_json(self.staging_dir / "calibration.json", value)

    def write_meta(self, value: Mapping[str, Any]) -> None:
        _atomic_json(self.staging_dir / "segment_meta.json", value)

    def write_manifest(self, value: Mapping[str, Any]) -> None:
        _atomic_json(self.staging_dir / "manifest.json", value)

    def write_observations(
        self, kind: str, values: Mapping[str, ArrayLike]
    ) -> None:
        if kind not in OBSERVATION_KINDS:
            raise ValueError(f"unknown observation kind: {kind}")
        _atomic_npz(self.staging_dir / "observations" / f"{kind}.npz", values)

    def finalize(self, *, check_files: bool = True) -> SegmentReader:
        if self._finished:
            raise RuntimeError("segment writer is already finished")
        SegmentReader(self.staging_dir).validate(check_files=check_files)
        publish_directory_noreplace(self.staging_dir, self.destination)
        self._finished = True
        return SegmentReader(self.destination)

    def abort(self) -> None:
        """Remove only this writer's unpublished staging directory."""
        if not self._finished:
            shutil.rmtree(self.staging_dir, ignore_errors=True)
            self._finished = True


def publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing target."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:  # pragma: no cover - current Linux/glibc has renameat2
        if destination.exists():
            raise FileExistsError(f"refusing to modify existing segment: {destination}")
        os.rename(source, destination)
        return
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(f"refusing to modify existing segment: {destination}")
    raise OSError(error, os.strerror(error), str(destination))
