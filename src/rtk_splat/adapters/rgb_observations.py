"""Immutable adapter-side artifact for calibrated rectified RGB streams."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from rtk_splat.core.segment import publish_directory_noreplace
from rtk_splat.frontends.artifact import ArtifactError, sha256_file


@dataclass(frozen=True)
class RgbObservations:
    root: Path
    frames: dict[str, np.ndarray]
    meta: dict[str, Any]
    calibration: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read RGB artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"RGB artifact is not an object: {path}")
    return value


def _safe_relative(value: Any, label: str) -> str:
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != text
    ):
        raise ArtifactError(f"{label} must be a normalized relative path")
    return text


def _acquisition_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or value.strip() != value
        or any(character.isspace() for character in value)
    ):
        raise ArtifactError("acquisition_id must be a non-empty opaque identifier")
    return value


def _validate_calibration(value: dict[str, Any]) -> None:
    camera = value.get("camera")
    if not isinstance(camera, dict):
        raise ArtifactError("RGB camera calibration must be an object")
    try:
        matrix = np.asarray(camera["K"], dtype=np.float64)
        distortion = np.asarray(camera["distortion"], dtype=np.float64)
        width, height = int(camera["width"]), int(camera["height"])
        transform = np.asarray(value["T_rgb_source_camera"], dtype=np.float64)
        sigma = np.asarray(value["extrinsic_translation_sigma_m"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError("RGB camera/extrinsic calibration is incomplete") from exc
    camera_frame = value.get("camera_frame_id")
    rectification = value.get("rectification")
    if (
        camera.get("model") != "PINHOLE"
        or distortion.ndim != 1
        or np.any(np.abs(distortion) > 1e-12)
    ):
        raise ArtifactError(
            "RGB pixels must declare an explicit zero-distortion PINHOLE model"
        )
    if (
        value.get("schema_version") != 1
        or value.get("image_geometry") != "rectified"
        or width <= 0
        or height <= 0
        or matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or min(matrix[0, 0], matrix[1, 1]) <= 0
        or not isinstance(rectification, dict)
        or not isinstance(rectification.get("input_camera"), dict)
        or not isinstance(rectification.get("method"), str)
        or not isinstance(rectification.get("provenance"), dict)
        or not rectification["provenance"]
        or not isinstance(camera_frame, str)
        or not camera_frame
        or value.get("source_camera_geometry")
        != "source_segment_left_rectified"
        or value.get("rgb_camera_geometry")
        not in {"output_camera_rectified", "recorded_factory_pinhole_direct"}
        or value.get("transform_convention") != "rgb_from_source_camera"
        or transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5
        )
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-5)
        or sigma.shape != (3,)
        or not np.isfinite(sigma).all()
        or np.any(sigma < 0)
        or value.get("extrinsic_translation_sigma_frame_id") != camera_frame
        or not isinstance(value.get("extrinsic_provenance"), dict)
        or not value["extrinsic_provenance"]
    ):
        raise ArtifactError("RGB geometry/calibration provenance is invalid")


def load_rectified_rgb_observations(root: str | Path) -> RgbObservations:
    """Verify and load one sealed, adapter-selected RGB representation."""
    root = Path(root).expanduser().resolve()
    required = {"rgb_observations_meta.json", "frames.npz", "calibration.json"}
    manifest = _read_json(root / "manifest.json")
    sealed = manifest.get("files")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != "calibrated_rgb_observations"
        or not isinstance(sealed, dict)
        or not required <= set(sealed)
    ):
        raise ArtifactError(f"incomplete RGB observation artifact: {root}")
    for relative, evidence in sealed.items():
        safe = _safe_relative(relative, "RGB manifest path")
        path = (root / safe).resolve()
        if root not in path.parents or not isinstance(evidence, dict) or (
            not path.is_file()
            or evidence.get("sha256") != sha256_file(path)
            or evidence.get("size_bytes") != path.stat().st_size
        ):
            raise ArtifactError(f"RGB observation file changed: {relative}")
    try:
        with np.load(root / "frames.npz", allow_pickle=False) as archive:
            frames = {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"cannot read RGB frames: {exc}") from exc
    if not {"frame_id", "timestamp_ns", "image_path"} <= set(frames):
        raise ArtifactError("RGB frames are incomplete")
    ids, timestamps, paths = (
        frames["frame_id"],
        frames["timestamp_ns"],
        frames["image_path"],
    )
    n = len(ids) if ids.ndim == 1 else 0
    if (
        n == 0
        or ids.dtype.kind not in "iu"
        or not np.array_equal(ids, np.arange(n))
        or timestamps.dtype != np.dtype(np.int64)
        or timestamps.shape != (n,)
        or np.any(np.diff(timestamps) <= 0)
        or paths.shape != (n,)
        or not np.issubdtype(paths.dtype, np.str_)
    ):
        raise ArtifactError("RGB IDs, timestamps, or paths are invalid")
    if "log_timestamp_ns" in frames and (
        frames["log_timestamp_ns"].dtype != np.dtype(np.int64)
        or frames["log_timestamp_ns"].shape != (n,)
    ):
        raise ArtifactError("RGB log timestamps are invalid")
    for index, value in enumerate(paths):
        relative = _safe_relative(value, f"RGB image_path[{index}]")
        if relative not in sealed or Path(relative).suffix.lower() != ".png":
            raise ArtifactError(f"RGB image must be a sealed lossless PNG: {relative}")
    meta = _read_json(root / "rgb_observations_meta.json")
    clock = meta.get("clock")
    acquisition_id = meta.get("acquisition_id")
    provenance = meta.get("provenance")
    if (
        meta.get("schema_version") != 1
        or meta.get("artifact_type") != "calibrated_rgb_observations"
        or meta.get("n_frames") != n
        or meta.get("timestamp_unit") != "ns"
        or not isinstance(meta.get("timestamp_source"), str)
        or not isinstance(clock, dict)
        or type(clock.get("rgb_to_source_clock_offset_ns")) is not int
        or not isinstance(clock.get("provenance"), dict)
        or not clock["provenance"]
        or not isinstance(provenance, dict)
        or provenance.get("acquisition_id") != acquisition_id
    ):
        raise ArtifactError("RGB metadata or clock provenance is invalid")
    _acquisition_id(acquisition_id)
    calibration = _read_json(root / "calibration.json")
    _validate_calibration(calibration)
    return RgbObservations(root, frames, meta, calibration)


def publish_rectified_rgb_observations(
    destination: str | Path,
    image_paths: Sequence[str | Path],
    timestamp_ns: Sequence[int] | np.ndarray,
    *,
    calibration: Mapping[str, Any],
    timestamp_source: str,
    clock: Mapping[str, Any],
    provenance: Mapping[str, Any],
    acquisition_id: str,
    log_timestamp_ns: Sequence[int] | np.ndarray | None = None,
) -> Path:
    """Copy staged PNG observations and atomically publish schema version 1."""
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to modify existing RGB artifact: {destination}")
    images = [Path(path) for path in image_paths]
    timestamps = np.asarray(timestamp_ns)
    if (
        not images
        or timestamps.dtype != np.dtype(np.int64)
        or timestamps.shape != (len(images),)
        or np.any(np.diff(timestamps) <= 0)
        or any(path.suffix.lower() != ".png" or not path.is_file() for path in images)
    ):
        raise ValueError("RGB publisher requires ordered int64 timestamps and PNG files")
    logs = None if log_timestamp_ns is None else np.asarray(log_timestamp_ns)
    if logs is not None and (
        logs.dtype != np.dtype(np.int64) or logs.shape != timestamps.shape
    ):
        raise ValueError("RGB log timestamps must be int64 with one row per image")
    try:
        normalized_acquisition_id = _acquisition_id(acquisition_id)
    except ArtifactError as exc:
        raise ValueError(str(exc)) from exc
    normalized_provenance = dict(provenance)
    existing_acquisition_id = normalized_provenance.get("acquisition_id")
    if existing_acquisition_id not in (None, normalized_acquisition_id):
        raise ValueError("RGB provenance acquisition_id conflicts with publication")
    normalized_provenance["acquisition_id"] = normalized_acquisition_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.writing-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        image_dir = staging / "images"
        image_dir.mkdir()
        relative_paths = []
        for index, source in enumerate(images):
            name = f"rgb_{index:06d}.png"
            shutil.copyfile(source, image_dir / name)
            relative_paths.append(f"images/{name}")
        frames = {
            "frame_id": np.arange(len(images), dtype=np.int64),
            "timestamp_ns": timestamps,
            "image_path": np.asarray(relative_paths, dtype=np.str_),
        }
        if logs is not None:
            frames["log_timestamp_ns"] = logs
        with (staging / "frames.npz").open("xb") as stream:
            np.savez_compressed(stream, **frames)
        (staging / "calibration.json").write_text(
            json.dumps(
                dict(calibration), indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        meta = {
            "schema_version": 1,
            "artifact_type": "calibrated_rgb_observations",
            "n_frames": len(images),
            "acquisition_id": normalized_acquisition_id,
            "timestamp_unit": "ns",
            "timestamp_source": timestamp_source,
            "clock": dict(clock),
            "provenance": normalized_provenance,
        }
        (staging / "rgb_observations_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        files = {
            path.relative_to(staging).as_posix(): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        (staging / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "calibrated_rgb_observations",
                    "files": files,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        load_rectified_rgb_observations(staging)
        publish_directory_noreplace(staging, destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
