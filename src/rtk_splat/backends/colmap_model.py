"""COLMAP model parsing, validation and pose-prior I/O."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np

from rtk_splat.backends.execution import Runner, _execute
from rtk_splat.frontends.artifact import ArtifactError


_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
_STAT_NAMES = {
    "Rigs": "rigs",
    "Cameras": "cameras",
    "Frames": "frames",
    "Registered frames": "registered_frames",
    "Images": "images",
    "Registered images": "registered_images",
    "Points": "points",
    "Observations": "observations",
    "Mean track length": "mean_track_length",
    "Mean observations per image": "mean_observations_per_image",
    "Mean reprojection error": "mean_reprojection_error_px",
}
_STAT_PATTERN = re.compile(
    r"(Rigs|Cameras|Frames|Registered frames|Images|Registered images|"
    r"Points|Observations|Mean track length|Mean observations per image|"
    r"Mean reprojection error):\s*([-+0-9.eE]+)"
)
_INTEGER_STATS = {
    "rigs",
    "cameras",
    "frames",
    "registered_frames",
    "images",
    "registered_images",
    "points",
    "observations",
}


def build_model_analyzer_command(
    model: str | Path, executable: str | Path
) -> tuple[str, ...]:
    return (
        str(executable),
        "model_analyzer",
        "--log_target",
        "stdout",
        "--path",
        str(Path(model).resolve()),
    )


def parse_model_analyzer(output: str) -> dict[str, int | float]:
    """Parse COLMAP's stable model-analyzer labels."""
    result: dict[str, int | float] = {}
    for label, raw in _STAT_PATTERN.findall(output):
        name = _STAT_NAMES[label]
        value = float(raw)
        result[name] = int(round(value)) if name in _INTEGER_STATS else value
    required = {
        "registered_images",
        "points",
        "observations",
        "mean_track_length",
        "mean_reprojection_error_px",
    }
    missing = required - set(result)
    if missing:
        raise ArtifactError(
            "model_analyzer output is missing: " + ", ".join(sorted(missing))
        )
    return result


def _analyze_model(
    model: Path, executable: str | Path, runner: Runner
) -> dict[str, int | float]:
    result = _execute(
        build_model_analyzer_command(model, executable), runner, capture=True
    )
    output = getattr(result, "stdout", None)
    if not isinstance(output, str):
        raise ArtifactError("model_analyzer runner returned no text output")
    return parse_model_analyzer(output)


def _model_candidates(root: Path) -> list[Path]:
    candidates = []
    if all((root / name).is_file() for name in _MODEL_FILES):
        candidates.append(root)
    if root.is_dir():
        candidates.extend(
            child
            for child in sorted(root.iterdir())
            if child.is_dir() and all((child / name).is_file() for name in _MODEL_FILES)
        )
    return candidates


def registered_names_from_images_txt(path: str | Path) -> tuple[str, ...]:
    """Read registered image names from a COLMAP text model."""
    names: list[str] = []
    expect_pose = True
    with Path(path).open(encoding="utf-8") as stream:
        for raw in stream:
            stripped = raw.strip()
            if stripped.startswith("#"):
                continue
            if expect_pose:
                if not stripped:
                    continue
                fields = stripped.split()
                if len(fields) < 10:
                    raise ArtifactError("malformed COLMAP images.txt pose row")
                names.append(fields[9])
                expect_pose = False
            else:
                # The observation row may legitimately be empty.
                expect_pose = True
    if not expect_pose:
        raise ArtifactError("COLMAP images.txt ends before an observation row")
    if len(names) != len(set(names)):
        raise ArtifactError("COLMAP text model has duplicate image names")
    return tuple(names)


def _quaternion_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    quaternion = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ArtifactError("COLMAP pose contains an invalid quaternion")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _poses_from_images_txt(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    expect_pose = True
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            stripped = raw.strip()
            if stripped.startswith("#"):
                continue
            if not expect_pose:
                expect_pose = True
                continue
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) < 10:
                raise ArtifactError("malformed COLMAP images.txt pose row")
            try:
                quaternion = [float(value) for value in fields[1:5]]
                translation = np.asarray(
                    [float(value) for value in fields[5:8]], dtype=np.float64
                )
            except ValueError as exc:
                raise ArtifactError(
                    "COLMAP pose row contains non-numeric values"
                ) from exc
            name = fields[9]
            if name in poses:
                raise ArtifactError(f"duplicate COLMAP pose for {name}")
            rotation = _quaternion_rotation(*quaternion)
            viewmat = np.eye(4, dtype=np.float64)
            viewmat[:3, :3] = rotation
            viewmat[:3, 3] = translation
            center = -(rotation.T @ translation)
            poses[name] = (viewmat, center)
            expect_pose = False
    if not expect_pose:
        raise ArtifactError("COLMAP images.txt ends before an observation row")
    return poses


def _cartesian_camera_priors(
    database: Path, allowed_names: set[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    uri = f"file:{database.resolve()}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
        rows = connection.execute(
            """
            SELECT i.name, p.position, p.position_covariance, p.coordinate_system
            FROM pose_priors AS p
            JOIN images AS i ON i.image_id = p.corr_data_id
            WHERE p.corr_sensor_type = 0
            """
        ).fetchall()
    except sqlite3.Error as exc:
        raise ArtifactError(f"cannot read sealed Cartesian pose priors: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, position_blob, covariance_blob, coordinate_system in rows:
        name = str(name)
        if name not in allowed_names:
            continue
        if int(coordinate_system) != 1:
            raise ArtifactError(f"pose prior for {name} is not Cartesian")
        if position_blob is None or covariance_blob is None:
            raise ArtifactError(f"pose prior for {name} lacks covariance or position")
        position = np.frombuffer(position_blob, dtype="<f8").copy()
        covariance = np.frombuffer(covariance_blob, dtype="<f8").copy()
        if position.shape != (3,) or covariance.shape != (9,):
            raise ArtifactError(f"pose prior for {name} has invalid blob dimensions")
        covariance = covariance.reshape(3, 3, order="F")
        covariance = 0.5 * (covariance + covariance.T)
        if (
            not np.isfinite(position).all()
            or not np.isfinite(covariance).all()
            or np.linalg.eigvalsh(covariance).min() < -1e-10
        ):
            raise ArtifactError(f"pose prior for {name} is numerically invalid")
        if name in result:
            raise ArtifactError(f"duplicate Cartesian pose prior for {name}")
        result[name] = (position, covariance)
    if len(result) < 3:
        raise ArtifactError("fewer than three trusted Cartesian camera-centre priors")
    return result
