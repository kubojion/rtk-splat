"""ROS-free records and fail-closed RTK--stereo diagnostic artifacts.

This module deliberately does not publish pose artifacts.  It provides:

* plain observation records returned by dataset adapters;
* raw chronological left-camera poses from a COLMAP rig model without reading
  the potentially multi-gigabyte ``images.txt``; and
* atomic, fail-if-exists calibration artifact publication.

ROS topic selection, message decoding, and bag traversal live exclusively in
``rtk_splat.adapters.calibration_bag``.

Coordinate conventions
----------------------
RELPOS is retained in its native NED convention and also converted to ENU as
``[east, north, -down]``.  COLMAP poses are returned as camera-from-visual-
world matrices.  No RTK alignment or scale is applied here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Iterator, Mapping, Sequence
from urllib.parse import quote

import numpy as np

from rtk_splat.adapters.records import (
    CalibrationBagObservations,
    NavPvtStatusObservation,
    NavSatFixObservation,
    RelPosObservation,
    StereoFrameObservation,
)


_SAFE_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def validate_calibration_artifact_name(name: str) -> str:
    """Return a safe artifact name or reject traversal/shell-like names."""
    value = str(name)
    if not _SAFE_ARTIFACT_NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"invalid calibration artifact name {value!r}")
    return value


def calibration_artifact_path(segment_dir: Path, name: str) -> Path:
    """Canonical directory for a diagnostic calibration artifact."""
    return (Path(segment_dir) / "calibration_artifacts"
            / validate_calibration_artifact_name(name))


@dataclass(frozen=True)
class CameraAntennaGeometry:
    """Direct geometry observable by one camera and a dual-antenna RTK rig.

    Both vectors are expressed in the camera optical coordinate frame:

    ``camera_to_primary_antenna_in_camera_m``
        Camera origin to the position-producing primary antenna.
    ``primary_to_secondary_antenna_in_camera_m``
        Primary antenna to the secondary antenna used by RELPOS.
    """

    camera_to_primary_antenna_in_camera_m: tuple[float, float, float]
    primary_to_secondary_antenna_in_camera_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        for item in fields(self):
            value = tuple(float(v) for v in getattr(self, item.name))
            if len(value) != 3 or not all(math.isfinite(v) for v in value):
                raise ValueError(f"{item.name} must contain three finite values")
            object.__setattr__(self, item.name, value)
        baseline = np.asarray(
            self.primary_to_secondary_antenna_in_camera_m, dtype=float)
        if np.linalg.norm(baseline) < 1.0e-3:
            raise ValueError("dual-antenna baseline must be non-zero")

    @classmethod
    def from_body_geometry(
            cls,
            body_from_camera: np.ndarray,
            primary_antenna_position_body_m: Sequence[float],
            secondary_antenna_position_body_m: Sequence[float],
    ) -> "CameraAntennaGeometry":
        """Convert explicit body-frame geometry to the direct camera model.

        ``body_from_camera`` is a homogeneous transform mapping camera-frame
        points into the body frame.  Antenna arguments are phase-centre
        positions expressed in that same body frame.
        """
        transform = np.asarray(body_from_camera, dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("body_from_camera must be a finite 4x4 matrix")
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1.0e-10):
            raise ValueError("body_from_camera has an invalid last row")
        rotation = transform[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8)
                or not np.isclose(np.linalg.det(rotation), 1.0,
                                  atol=1.0e-8)):
            raise ValueError("body_from_camera rotation is not proper")
        primary = _finite_vector3(
            primary_antenna_position_body_m,
            "primary_antenna_position_body_m")
        secondary = _finite_vector3(
            secondary_antenna_position_body_m,
            "secondary_antenna_position_body_m")
        camera_origin_body = transform[:3, 3]
        camera_from_body_rotation = rotation.T
        lever_camera = camera_from_body_rotation @ (
            primary - camera_origin_body)
        baseline_camera = camera_from_body_rotation @ (secondary - primary)
        return cls(tuple(lever_camera), tuple(baseline_camera))


def _finite_vector3(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite three-vector")
    return array

@dataclass(frozen=True)
class RawColmapRigTrajectory:
    """Chronological raw left-camera trajectory in the visual world."""

    frame_indices: np.ndarray
    colmap_frame_ids: np.ndarray
    image_ids: np.ndarray
    image_names: tuple[str, ...]
    camera_from_visual_world: np.ndarray
    camera_centers_visual: np.ndarray


@dataclass(frozen=True)
class _RigDefinition:
    reference_sensor: tuple[str, int]
    sensor_from_rig: Mapping[tuple[str, int], np.ndarray]


def _qvec_to_rotmat(qvec: Sequence[float]) -> np.ndarray:
    q = np.asarray(qvec, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("invalid COLMAP quaternion")
    norm = np.linalg.norm(q)
    if norm < 1.0e-12:
        raise ValueError("zero COLMAP quaternion")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def _pose_matrix(qvec: Sequence[float], translation: Sequence[float]) \
        -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = _qvec_to_rotmat(qvec)
    vector = np.asarray(translation, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("invalid COLMAP translation")
    matrix[:3, 3] = vector
    return matrix


def _data_lines(path: Path) -> Iterator[list[str]]:
    with path.open() as stream:
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                yield stripped.split()


def _parse_rigs(path: Path) -> dict[int, _RigDefinition]:
    rigs: dict[int, _RigDefinition] = {}
    for tokens in _data_lines(path):
        if len(tokens) < 4:
            raise ValueError(f"malformed COLMAP rig row in {path}")
        rig_id, sensor_count = int(tokens[0]), int(tokens[1])
        reference = (tokens[2], int(tokens[3]))
        poses: dict[tuple[str, int], np.ndarray] = {reference: np.eye(4)}
        cursor = 4
        while cursor < len(tokens):
            if cursor + 2 >= len(tokens):
                raise ValueError(f"truncated COLMAP rig sensor in {path}")
            key = (tokens[cursor], int(tokens[cursor + 1]))
            has_pose = int(tokens[cursor + 2])
            cursor += 3
            if key in poses:
                raise ValueError(f"duplicate COLMAP rig sensor {key}")
            if not has_pose:
                raise ValueError(
                    f"non-reference COLMAP rig sensor {key} has no pose")
            if cursor + 7 > len(tokens):
                raise ValueError(f"truncated COLMAP rig pose for {key}")
            poses[key] = _pose_matrix(
                [float(value) for value in tokens[cursor:cursor + 4]],
                [float(value) for value in tokens[cursor + 4:cursor + 7]])
            cursor += 7
        if len(poses) != sensor_count:
            raise ValueError(
                f"rig {rig_id} declares {sensor_count} sensors, parsed "
                f"{len(poses)}")
        if rig_id in rigs:
            raise ValueError(f"duplicate COLMAP rig id {rig_id}")
        rigs[rig_id] = _RigDefinition(reference, poses)
    if not rigs:
        raise ValueError(f"no COLMAP rigs parsed from {path}")
    return rigs


def _read_database_images(database_path: Path) \
        -> dict[int, tuple[str, int]]:
    path = Path(database_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"COLMAP database is missing: {path}")
    uri = "file:" + quote(str(path), safe="/") + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            "SELECT image_id, name, camera_id FROM images").fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError(f"COLMAP database has no images: {path}")
    return {int(image_id): (str(name).replace("\\", "/"), int(camera_id))
            for image_id, name, camera_id in rows}


def _frame_index_from_name(name: str, left_prefix: str) -> int:
    normalized_prefix = left_prefix.replace("\\", "/")
    if not name.startswith(normalized_prefix):
        raise ValueError(f"image {name!r} does not use prefix {left_prefix!r}")
    path = Path(name)
    if path.suffix.lower() not in _IMAGE_SUFFIXES or not path.stem.isdigit():
        raise ValueError(f"left image name has no numeric frame id: {name}")
    return int(path.stem)


def load_raw_colmap_rig_trajectory(
        model_dir: Path,
        database_path: Path,
        *,
        left_prefix: str = "zed/left/",
        left_image_frame_ids: Mapping[str, int] | None = None,
        expected_frame_count: int | None = None,
) -> RawColmapRigTrajectory:
    """Load raw chronological left poses from ``frames.txt`` and the database.

    ``images.txt`` is intentionally never opened.  The database is opened in
    immutable read-only mode and is used only to map image IDs to names.
    """
    model = Path(model_dir)
    explicit_left: dict[str, int] | None = None
    if left_image_frame_ids is not None:
        explicit_left = {}
        for raw_name, raw_index in left_image_frame_ids.items():
            name = str(raw_name).replace("\\", "/")
            if not name or name.startswith("/") or name in explicit_left:
                raise ValueError("left image assignment contains an invalid name")
            if isinstance(raw_index, bool) or not isinstance(
                    raw_index, (int, np.integer)):
                raise ValueError(
                    "left image assignment frame IDs must be integers")
            index = int(raw_index)
            if index < 0:
                raise ValueError(
                    "left image assignment frame IDs cannot be negative")
            explicit_left[name] = index
        if not explicit_left:
            raise ValueError("left image assignment cannot be empty")
        if len(set(explicit_left.values())) != len(explicit_left):
            raise ValueError("left image assignment contains duplicate frame IDs")
    rigs = _parse_rigs(model / "rigs.txt")
    images = _read_database_images(database_path)
    records = []
    for tokens in _data_lines(model / "frames.txt"):
        if len(tokens) < 10:
            raise ValueError("malformed COLMAP frame row")
        frame_id, rig_id = int(tokens[0]), int(tokens[1])
        if rig_id not in rigs:
            raise ValueError(f"frame {frame_id} references unknown rig {rig_id}")
        rig_from_world = _pose_matrix(
            [float(value) for value in tokens[2:6]],
            [float(value) for value in tokens[6:9]])
        count = int(tokens[9])
        if len(tokens) != 10 + 3 * count:
            raise ValueError(f"frame {frame_id} has malformed data IDs")
        data = []
        for cursor in range(10, len(tokens), 3):
            data.append((tokens[cursor], int(tokens[cursor + 1]),
                         int(tokens[cursor + 2])))
        left_candidates = []
        for sensor_type, sensor_id, image_id in data:
            if image_id not in images:
                raise ValueError(
                    f"frame {frame_id} references missing image {image_id}")
            name, camera_id = images[image_id]
            is_left = (
                name in explicit_left
                if explicit_left is not None
                else name.startswith(left_prefix)
            )
            if sensor_type == "CAMERA" and is_left:
                if camera_id != sensor_id:
                    raise ValueError(
                        f"image {image_id} camera id {camera_id} disagrees "
                        f"with rig sensor id {sensor_id}")
                left_candidates.append((sensor_type, sensor_id,
                                        image_id, name))
        if len(left_candidates) != 1:
            raise ValueError(
                f"frame {frame_id} has {len(left_candidates)} left images")
        sensor_type, sensor_id, image_id, name = left_candidates[0]
        sensor_key = (sensor_type, sensor_id)
        rig = rigs[rig_id]
        if sensor_key not in rig.sensor_from_rig:
            raise ValueError(
                f"frame {frame_id} uses sensor {sensor_key} absent from rig")
        camera_from_world = (
            rig.sensor_from_rig[sensor_key] @ rig_from_world)
        index = (
            explicit_left[name]
            if explicit_left is not None
            else _frame_index_from_name(name, left_prefix)
        )
        rotation = camera_from_world[:3, :3]
        center = -rotation.T @ camera_from_world[:3, 3]
        records.append((index, frame_id, image_id, name,
                        camera_from_world, center))

    if not records:
        raise ValueError(f"no COLMAP frames parsed from {model/'frames.txt'}")
    records.sort(key=lambda item: item[0])
    indices = np.asarray([item[0] for item in records], dtype=np.int64)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("COLMAP model contains duplicate left frame indices")
    if expected_frame_count is not None:
        expected = int(expected_frame_count)
        if not np.array_equal(indices, np.arange(expected, dtype=np.int64)):
            raise ValueError(
                "COLMAP left frame indices are not the expected contiguous "
                f"range 0..{expected - 1}")
    viewmats = np.stack([item[4] for item in records]).astype(np.float64)
    rotations = viewmats[:, :3, :3]
    if (not np.allclose(
            rotations @ np.swapaxes(rotations, 1, 2), np.eye(3),
            atol=1.0e-8)
            or not np.allclose(np.linalg.det(rotations), 1.0,
                               atol=1.0e-8)):
        raise ValueError("raw COLMAP trajectory contains invalid rotations")
    return RawColmapRigTrajectory(
        frame_indices=indices,
        colmap_frame_ids=np.asarray(
            [item[1] for item in records], dtype=np.int64),
        image_ids=np.asarray([item[2] for item in records], dtype=np.int64),
        image_names=tuple(item[3] for item in records),
        camera_from_visual_world=viewmats,
        camera_centers_visual=np.stack(
            [item[5] for item in records]).astype(np.float64),
    )


def _fsync_tree(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def atomic_calibration_artifact(
        segment_dir: Path,
        name: str,
) -> Iterator[Path]:
    """Yield a temporary directory and atomically publish it on success.

    A lock created with ``O_EXCL`` serializes cooperating writers.  The final
    directory is never reused or overwritten.  Exceptions remove the temporary
    directory and leave no published artifact.
    """
    safe_name = validate_calibration_artifact_name(name)
    root = Path(segment_dir) / "calibration_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    final = root / safe_name
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"calibration artifact already exists: {final}")
    lock = root / f".{safe_name}.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"calibration artifact writer is already active: {lock}") from exc
    temp: Path | None = None
    try:
        os.write(lock_fd, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(lock_fd)
        temp = Path(tempfile.mkdtemp(
            prefix=f".{safe_name}.tmp-", dir=root))
        yield temp
        if not any(temp.iterdir()):
            raise RuntimeError("refusing to publish an empty calibration artifact")
        _fsync_tree(temp)
        if final.exists() or final.is_symlink():
            raise FileExistsError(
                f"calibration artifact appeared during write: {final}")
        os.rename(temp, final)
        root_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
    finally:
        os.close(lock_fd)
        if temp is not None and temp.exists():
            shutil.rmtree(temp)
        lock.unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    """Deterministic JSON helper intended for an artifact temporary directory."""
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
