"""Explicit COLMAP stages for a sealed mapper-neutral frontend artifact.

No mapper or reconstruction is run here.  Each short stage is fingerprinted,
validates its semantic database output, and writes a stage-specific report so
later stages may safely extend the shared database.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rtk_splat.core.segment import SegmentReader

from .artifact import (
    ArtifactError,
    FRONTEND_SEAL_FILE,
    StageLedger,
    canonical_hash,
    create_frontend_seal,
    frontend_image_record,
    sha256_file,
    verify_frontend_seal,
)


FeatureProfile = Literal["gpu", "cpu_reference"]
Runner = Callable[..., Any]
_PAIR_ID_BASE = 2_147_483_647
_MINIMUM_TRUSTED_PRIORS = 3
_POSE_PRIOR_COLUMNS = (
    "pose_prior_id",
    "corr_data_id",
    "corr_sensor_id",
    "corr_sensor_type",
    "position",
    "position_covariance",
    "gravity",
    "coordinate_system",
)


def build_feature_extractor_command(
    artifact: str | Path,
    executable: str | Path,
    *,
    profile: FeatureProfile,
    max_image_size: int = -1,
    num_threads: int = 8,
    max_num_features: int = 8192,
    gpu_index: int = -1,
    seed: int = 0,
) -> tuple[str, ...]:
    """Build all-image SIFT extraction for the controlled GPU/CPU A/B."""
    root = Path(artifact).resolve()
    if profile not in ("gpu", "cpu_reference"):
        raise ValueError("profile must be 'gpu' or 'cpu_reference'")
    if num_threads == 0 or max_num_features <= 0:
        raise ValueError("num_threads must be non-zero and features positive")
    command = [
        str(executable),
        "feature_extractor",
        "--database_path",
        str(root / "database.db"),
        "--image_path",
        str(root / "images"),
        "--default_random_seed",
        str(seed),
        # Flat, deterministic left_/right_ names are grouped by the following
        # rig stage. Per-image cameras prevent the two sensors being conflated.
        "--ImageReader.single_camera_per_image",
        "1",
        "--FeatureExtraction.max_image_size",
        str(max_image_size),
        "--FeatureExtraction.num_threads",
        str(num_threads),
        "--FeatureExtraction.use_gpu",
        "1" if profile == "gpu" else "0",
        "--FeatureExtraction.gpu_index",
        str(gpu_index),
        "--SiftExtraction.max_num_features",
        str(max_num_features),
    ]
    if profile == "cpu_reference":
        command.extend(
            [
                "--SiftExtraction.estimate_affine_shape",
                "1",
                "--SiftExtraction.domain_size_pooling",
                "1",
            ]
        )
    return tuple(command)


def build_rig_configurator_command(
    artifact: str | Path, executable: str | Path
) -> tuple[str, ...]:
    root = Path(artifact).resolve()
    return (
        str(executable),
        "rig_configurator",
        "--database_path",
        str(root / "database.db"),
        "--rig_config_path",
        str(root / "rig_config.json"),
    )


def build_matches_importer_command(
    artifact: str | Path,
    executable: str | Path,
    *,
    use_gpu: bool = True,
    gpu_index: int = -1,
    num_threads: int = 8,
    seed: int = 0,
) -> tuple[str, ...]:
    """Build explicit-pair matching with geometric and rig verification."""
    root = Path(artifact).resolve()
    return (
        str(executable),
        "matches_importer",
        "--database_path",
        str(root / "database.db"),
        "--match_list_path",
        str(root / "pairs.txt"),
        "--match_type",
        "pairs",
        "--default_random_seed",
        str(seed),
        "--FeatureMatching.num_threads",
        str(num_threads),
        "--FeatureMatching.use_gpu",
        "1" if use_gpu else "0",
        "--FeatureMatching.gpu_index",
        str(gpu_index),
        "--FeatureMatching.guided_matching",
        "1",
        "--FeatureMatching.rig_verification",
        "1",
        "--FeatureMatching.skip_geometric_verification",
        "0",
    )


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid frontend file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{path} must contain a JSON object")
    return value


def _sealed_context(
    artifact: str | Path,
) -> tuple[Path, dict[str, Any], SegmentReader]:
    root = Path(artifact).resolve()
    required = (
        "frame_manifest.json",
        "rig_config.json",
        "keyframes.json",
        "pairs.txt",
        "provenance.json",
        "quality.json",
    )
    if not root.is_dir() or any(not (root / name).is_file() for name in required):
        raise ArtifactError(f"incomplete sealed frontend artifact: {root}")
    images = root / "images"
    if not images.is_dir():
        raise ArtifactError("sealed frontend has no images directory")
    image_entries = list(images.iterdir())
    if not image_entries or any(not entry.is_symlink() for entry in image_entries):
        raise ArtifactError("frontend images must contain symlinks only")
    provenance = _json(root / "provenance.json")
    contract = provenance.get("contract_inputs")
    if not isinstance(contract, dict) or not isinstance(contract.get("files"), dict):
        raise ArtifactError("frontend provenance has no contract input evidence")
    segment = Path(str(contract.get("segment_root", ""))).resolve()
    for relative, expected in contract["files"].items():
        path = segment / relative
        if (
            not isinstance(expected, dict)
            or not path.is_file()
            or sha256_file(path) != expected.get("sha256")
            or path.stat().st_size != expected.get("size_bytes")
        ):
            raise ArtifactError(f"sealed contract input changed: {relative}")
    reader = SegmentReader(segment).validate()
    manifest = _json(root / "frame_manifest.json")
    rows = manifest.get("frames")
    if not isinstance(rows, list) or len(rows) != reader.meta["n_frames"]:
        raise ArtifactError("frontend frame manifest count is invalid")
    names = {
        image["name"]
        for row in rows
        for key, image in row.items()
        if key.endswith("_image") and isinstance(image, dict) and "name" in image
    }
    if names != {entry.name for entry in image_entries}:
        raise ArtifactError("frontend image symlinks disagree with frame manifest")
    # Do not trust the symlink path alone: validate the bytes consumed by
    # COLMAP against the immutable frame manifest at every stage boundary.
    frontend_image_record(root)
    matching_marker = root / "stages" / "matching.json"
    if matching_marker.is_file():
        marker = _json(matching_marker)
        if marker.get("state") == "complete":
            verify_frontend_seal(root)
    return root, manifest, reader


def _execute(command: Sequence[str], runner: Runner) -> None:
    runner(list(command), check=True)


def _connect_readonly(database: Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        return connection
    except sqlite3.Error as exc:
        raise ArtifactError(f"cannot open COLMAP database {database}: {exc}") from exc


def _require_tables(connection: sqlite3.Connection, names: set[str]) -> None:
    actual = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    missing = sorted(names - actual)
    if missing:
        raise ArtifactError(f"partial COLMAP database; missing: {', '.join(missing)}")


def _manifest_image_names(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    names = []
    for row in manifest["frames"]:
        for side in ("left_image", "right_image"):
            if side in row:
                names.append(str(row[side]["name"]))
    return tuple(names)


def _feature_quality(
    database: Path, manifest: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not database.exists():
        return None
    with _connect_readonly(database) as connection:
        _require_tables(connection, {"images", "keypoints", "descriptors"})
        expected = set(_manifest_image_names(manifest))
        images = {
            str(name): int(image_id)
            for image_id, name in connection.execute(
                "SELECT image_id, name FROM images"
            )
        }
        keypoints = {
            int(row[0])
            for row in connection.execute("SELECT image_id FROM keypoints")
        }
        descriptors = {
            int(row[0])
            for row in connection.execute("SELECT image_id FROM descriptors")
        }
        if set(images) != expected:
            raise ArtifactError("partial/stale feature database image inventory")
        image_ids = set(images.values())
        if keypoints != image_ids or descriptors != image_ids:
            raise ArtifactError("partial feature extraction output")
        keypoint_count = int(
            connection.execute("SELECT COALESCE(SUM(rows), 0) FROM keypoints").fetchone()[0]
        )
        descriptor_count = int(
            connection.execute(
                "SELECT COALESCE(SUM(rows), 0) FROM descriptors"
            ).fetchone()[0]
        )
    return {
        "n_images": len(expected),
        "n_keypoints": keypoint_count,
        "n_descriptors": descriptor_count,
        "image_assignment_sha256": canonical_hash(sorted(images.items())),
    }


def _stage_marker_digest(root: Path, stage: str) -> str:
    path = root / "stages" / f"{stage}.json"
    marker = _json(path)
    if marker.get("state") != "complete":
        raise ArtifactError(f"required upstream stage is incomplete: {stage}")
    outputs = marker.get("outputs")
    if not isinstance(outputs, dict):
        raise ArtifactError(f"upstream stage has no output evidence: {stage}")
    for relative, expected in outputs.items():
        output = root / relative
        if not output.is_file() or sha256_file(output) != expected:
            raise ArtifactError(f"upstream stage output changed: {relative}")
    return sha256_file(path)


def _has_stage_marker(root: Path, stage: str) -> bool:
    return (root / "stages" / f"{stage}.json").exists()


def _write_report(root: Path, stage: str, value: Mapping[str, Any]) -> Path:
    path = root / "stage_reports" / f"{stage}.json"
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ArtifactError(f"refusing to overwrite stale stage report: {path}")
        return path
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _finish_stage(
    ledger: StageLedger,
    root: Path,
    stage: str,
    inputs: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    path = _write_report(root, stage, report)
    ledger.complete(stage, inputs, [path])
    return dict(report)


def _completed_report(root: Path, stage: str) -> dict[str, Any]:
    return _json(root / "stage_reports" / f"{stage}.json")


def run_feature_extraction(
    artifact: str | Path,
    executable: str | Path,
    *,
    profile: FeatureProfile,
    max_image_size: int = -1,
    num_threads: int = 8,
    max_num_features: int = 8192,
    gpu_index: int = -1,
    seed: int = 0,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    root, manifest, _ = _sealed_context(artifact)
    command = build_feature_extractor_command(
        root,
        executable,
        profile=profile,
        max_image_size=max_image_size,
        num_threads=num_threads,
        max_num_features=max_num_features,
        gpu_index=gpu_index,
        seed=seed,
    )
    inputs = {
        "command": command,
        "frame_manifest_sha256": sha256_file(root / "frame_manifest.json"),
        "image_inventory_sha256": manifest["image_inventory_sha256"],
    }
    database = root / "database.db"
    if not _has_stage_marker(root, "features") and database.exists():
        raise ArtifactError("refusing unmarked/pre-existing feature database")
    ledger = StageLedger(root)
    status = ledger.begin("features", inputs)
    if status == "complete":
        if _feature_quality(database, manifest) is None:
            raise ArtifactError("completed feature stage has no database")
        return _completed_report(root, "features")
    quality = _feature_quality(database, manifest) if database.exists() else None
    if status == "resume" and database.exists() and quality is None:
        raise ArtifactError("partial feature extraction cannot be resumed")
    if quality is None:
        _execute(command, runner)
        quality = _feature_quality(database, manifest)
        if quality is None:
            raise ArtifactError("feature extractor produced no database")
    report = {
        "schema_version": 1,
        "stage": "features",
        "profile": profile,
        "command": list(command),
        **quality,
    }
    return _finish_stage(ledger, root, "features", inputs, report)


def _rig_assignments(
    database: Path, manifest: Mapping[str, Any]
) -> tuple[dict[int, tuple[int, int]], dict[str, Any]] | None:
    with _connect_readonly(database) as connection:
        _require_tables(connection, {"images", "rigs", "frames", "frame_data"})
        counts = [
            int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("rigs", "frames", "frame_data")
        ]
        if counts == [0, 0, 0]:
            return None
        rows = list(
            connection.execute(
                """
                SELECT i.name, i.image_id, i.camera_id, fd.frame_id,
                       fd.sensor_id, fd.sensor_type,
                       r.ref_sensor_id, r.ref_sensor_type
                FROM images AS i
                JOIN frame_data AS fd
                  ON fd.data_id=i.image_id AND fd.sensor_type=0
                JOIN frames AS f ON f.frame_id=fd.frame_id
                JOIN rigs AS r ON r.rig_id=f.rig_id
                """
            )
        )
    by_name: dict[str, tuple[int, int, int, int, int, int, int]] = {}
    for row in rows:
        name = str(row[0])
        if name in by_name:
            raise ArtifactError(f"duplicate rig assignment for image {name}")
        by_name[name] = tuple(int(value) for value in row[1:])
    expected_names = set(_manifest_image_names(manifest))
    if set(by_name) != expected_names:
        raise ArtifactError("partial/stale rig image assignments")
    if counts != [1, len(manifest["frames"]), len(expected_names)]:
        raise ArtifactError("partial/stale rig/frame counts")
    left: dict[int, tuple[int, int]] = {}
    db_frames: set[int] = set()
    for frame in manifest["frames"]:
        frame_id = int(frame["frame_id"])
        left_name = str(frame["left_image"]["name"])
        left_row = by_name[left_name]
        image_id, camera_id, db_frame, sensor_id, sensor_type, ref_id, ref_type = left_row
        if sensor_type != 0 or ref_type != 0 or sensor_id != camera_id:
            raise ArtifactError("invalid reference-camera sensor assignment")
        if ref_id != camera_id:
            raise ArtifactError("left camera is not the rig reference sensor")
        if db_frame in db_frames:
            raise ArtifactError("multiple contract frames share a COLMAP frame")
        db_frames.add(db_frame)
        left[frame_id] = (image_id, camera_id)
        if "right_image" in frame:
            right_row = by_name[str(frame["right_image"]["name"])]
            if right_row[2] != db_frame:
                raise ArtifactError("stereo images are assigned to different frames")
            if right_row[4] != 0 or right_row[3] != right_row[1]:
                raise ArtifactError("invalid right-camera sensor assignment")
    quality = {
        "n_rigs": counts[0],
        "n_frames": counts[1],
        "n_frame_data": counts[2],
        "left_assignment_sha256": canonical_hash(sorted(left.items())),
    }
    return left, quality


def run_rig_configurator(
    artifact: str | Path,
    executable: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    root, manifest, _ = _sealed_context(artifact)
    database = root / "database.db"
    _feature_quality(database, manifest)
    command = build_rig_configurator_command(root, executable)
    inputs = {
        "command": command,
        "feature_stage_marker_sha256": _stage_marker_digest(root, "features"),
        "rig_config_sha256": sha256_file(root / "rig_config.json"),
    }
    assignment = _rig_assignments(database, manifest)
    if not _has_stage_marker(root, "rig") and assignment is not None:
        raise ArtifactError("refusing unmarked/pre-existing rig assignments")
    ledger = StageLedger(root)
    status = ledger.begin("rig", inputs)
    if status == "complete":
        if assignment is None:
            raise ArtifactError("completed rig stage has no assignments")
        return _completed_report(root, "rig")
    if assignment is None:
        _execute(command, runner)
        assignment = _rig_assignments(database, manifest)
        if assignment is None:
            raise ArtifactError("rig configurator produced no assignments")
    _, quality = assignment
    report = {
        "schema_version": 1,
        "stage": "rig",
        "command": list(command),
        **quality,
    }
    return _finish_stage(ledger, root, "rig", inputs, report)


def _pose_prior_schema(connection: sqlite3.Connection) -> None:
    columns = tuple(
        str(row[1]) for row in connection.execute("PRAGMA table_info(pose_priors)")
    )
    if columns != _POSE_PRIOR_COLUMNS:
        raise ArtifactError(
            f"unsupported COLMAP pose_priors schema: {columns}"
        )
    unique_assignment = False
    for row in connection.execute("PRAGMA index_list(pose_priors)"):
        if not bool(row[2]):
            continue
        name = str(row[1]).replace('"', '""')
        indexed = tuple(
            str(item[2])
            for item in connection.execute(f'PRAGMA index_info("{name}")')
        )
        if indexed == ("corr_data_id", "corr_sensor_id", "corr_sensor_type"):
            unique_assignment = True
            break
    if not unique_assignment:
        raise ArtifactError("pose_priors lacks a unique sensor assignment index")


def _trusted_priors(
    reader: SegmentReader,
    *,
    covariance_floor_m: float,
    max_covariance_m2: float | None,
    min_fix_status: int,
    min_carrier_status: int,
    max_source_residual_s: float | None,
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[str, int],
    dict[str, int],
    dict[str, Any],
]:
    if not np.isfinite(covariance_floor_m) or covariance_floor_m <= 0:
        raise ValueError("covariance_floor_m must be finite and positive")
    if max_covariance_m2 is not None and (
        not np.isfinite(max_covariance_m2) or max_covariance_m2 <= 0
    ):
        raise ValueError("max_covariance_m2 must be positive or None")
    if (
        max_covariance_m2 is not None
        and covariance_floor_m**2 > max_covariance_m2
    ):
        raise ValueError("covariance floor cannot exceed the maximum covariance")
    if max_source_residual_s is not None and (
        not np.isfinite(max_source_residual_s) or max_source_residual_s < 0
    ):
        raise ValueError("max_source_residual_s must be non-negative or None")
    frames = reader.frames
    required = {
        "initial_camera_center_m",
        "initial_viewmat",
        "pose_valid",
    }
    if not required <= set(frames):
        raise ArtifactError("pose priors need the complete initial-pose triplet")
    gnss = reader.observations("gnss")
    assert gnss is not None
    pose_meta = reader.meta["initial_pose"]
    sigma_camera_m = np.asarray(
        pose_meta["extrinsic_translation_sigma_m"], dtype=np.float64
    )
    sigma_frame_id = str(
        pose_meta["extrinsic_translation_sigma_frame_id"]
    )
    camera_frame_id = str(pose_meta["camera_frame_id"])
    if sigma_frame_id != camera_frame_id:
        raise ArtifactError(
            "extrinsic translation sigma is not expressed in the camera frame"
        )
    extrinsic_covariance_camera = np.diag(np.square(sigma_camera_m))
    source_residual = gnss.get("source_residual_ns", np.zeros(len(frames["frame_id"])))
    trusted: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    skipped: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter(
        str(value) for value in gnss["position_quality"]
    )
    floor_variance = covariance_floor_m**2
    for index, frame_id in enumerate(frames["frame_id"].astype(int)):
        center = frames["initial_camera_center_m"][index].astype(np.float64)
        gnss_covariance = gnss["covariance_enu_m2"][index].astype(np.float64)
        reason = None
        if not bool(frames["pose_valid"][index]) or not np.isfinite(center).all():
            reason = "invalid_initial_pose"
        elif not bool(gnss["position_valid"][index]):
            reason = "position_invalid"
        elif int(gnss["fix_status"][index]) < min_fix_status:
            reason = "fix_status"
        elif int(gnss["carrier_status"][index]) < min_carrier_status:
            reason = "carrier_status"
        elif not np.isfinite(gnss_covariance).all():
            reason = "missing_covariance"
        elif max_source_residual_s is not None and (
            abs(int(source_residual[index])) * 1e-9 > max_source_residual_s
        ):
            reason = "source_age"
        else:
            rotation_world_to_camera = np.asarray(
                frames["initial_viewmat"][index][:3, :3], dtype=np.float64
            )
            rotation_camera_to_world = rotation_world_to_camera.T
            covariance = (
                gnss_covariance
                + rotation_camera_to_world
                @ extrinsic_covariance_camera
                @ rotation_camera_to_world.T
            )
            covariance = 0.5 * (covariance + covariance.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            if eigenvalues.min() < -1e-10:
                reason = "invalid_covariance"
            elif (
                max_covariance_m2 is not None
                and eigenvalues.max() > max_covariance_m2
            ):
                reason = "covariance_too_large"
            else:
                eigenvalues = np.maximum(eigenvalues, floor_variance)
                covariance = (
                    eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                )
        if reason is not None:
            skipped[reason] += 1
        else:
            trusted[frame_id] = (center, covariance)
    covariance_model = {
        "output_quantity": "left_camera_center",
        "output_frame": reader.meta["coordinate_frame"]["world_frame_id"],
        "formula": "Sigma_center_world = Sigma_gnss_ENU + "
                   "R_camera_to_world diag(sigma_extrinsic_camera^2) "
                   "R_camera_to_world^T",
        "rotation_source": "frames.initial_viewmat",
        "extrinsic_translation_sigma_m": sigma_camera_m.tolist(),
        "extrinsic_translation_sigma_frame_id": sigma_frame_id,
        "omitted_terms": [
            "heading uncertainty and the lever-arm rotation Jacobian are not "
            "included; no unsupported covariance term is invented"
        ],
    }
    return (
        trusted,
        dict(sorted(skipped.items())),
        dict(sorted(quality_counts.items())),
        covariance_model,
    )


def _record_prior_preflight_rejection(
    root: Path, report: Mapping[str, Any]
) -> Path:
    """Persist a deterministic rejection audit without overwriting evidence."""
    payload = (
        json.dumps(
            dict(report),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    fingerprint = canonical_hash(report)[:16]
    destination = root / "stages" / f"pose_priors_rejected_{fingerprint}.json"
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ArtifactError("pose-prior rejection audit hash collision")
        return destination
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != payload:
                raise ArtifactError("conflicting pose-prior rejection audit")
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _prior_rows(
    connection: sqlite3.Connection,
    assignments: Mapping[int, tuple[int, int]],
    priors: Mapping[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[str, dict[int, tuple[Any, ...]]]:
    _pose_prior_schema(connection)
    expected: dict[int, tuple[Any, ...]] = {}
    for frame_id, (position, covariance) in priors.items():
        image_id, camera_id = assignments[frame_id]
        expected[frame_id] = (
            image_id,
            camera_id,
            0,
            np.asarray(position, dtype="<f8").tobytes(order="C"),
            np.asarray(covariance, dtype="<f8").tobytes(order="F"),
            None,
            1,
        )
    rows = list(
        connection.execute(
            """
            SELECT corr_data_id, corr_sensor_id, corr_sensor_type,
                   position, position_covariance, gravity, coordinate_system
            FROM pose_priors
            """
        )
    )
    target = {
        (int(row[0]), int(row[1]), int(row[2])): row
        for row in expected.values()
    }
    intended_data_ids = {key[0] for key in target}
    conflicting = [
        row
        for row in rows
        if int(row[0]) in intended_data_ids
        and (int(row[0]), int(row[1]), int(row[2])) not in target
    ]
    if conflicting:
        raise ArtifactError("conflicting pose prior already targets a left image")
    actual = {
        (int(row[0]), int(row[1]), int(row[2])): tuple(row)
        for row in rows
        if (int(row[0]), int(row[1]), int(row[2])) in target
    }
    if not actual:
        return "absent", expected
    if actual != target:
        raise ArtifactError("partial/stale/conflicting Cartesian pose priors")
    return "complete", expected


def _insert_prior_rows(
    database: Path,
    assignments: Mapping[int, tuple[int, int]],
    priors: Mapping[int, tuple[np.ndarray, np.ndarray]],
) -> None:
    try:
        connection = sqlite3.connect(database)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        for image_id, camera_id in assignments.values():
            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM images AS i
                    JOIN frame_data AS fd
                      ON fd.data_id=i.image_id
                     AND fd.sensor_id=i.camera_id
                     AND fd.sensor_type=0
                    JOIN frames AS f ON f.frame_id=fd.frame_id
                    JOIN rigs AS r
                      ON r.rig_id=f.rig_id
                     AND r.ref_sensor_id=i.camera_id
                     AND r.ref_sensor_type=0
                    WHERE i.image_id=? AND i.camera_id=?
                    """,
                    (image_id, camera_id),
                ).fetchone()[0]
            )
            if count != 1:
                raise ArtifactError(
                    "left/reference image assignment changed before insertion"
                )
        state, expected = _prior_rows(connection, assignments, priors)
        if state != "absent":
            raise ArtifactError("refusing to overwrite existing pose priors")
        for row in expected.values():
            connection.execute(
                """
                INSERT INTO pose_priors(
                    corr_data_id, corr_sensor_id, corr_sensor_type,
                    position, position_covariance, gravity, coordinate_system
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row[0],
                    row[1],
                    row[2],
                    sqlite3.Binary(row[3]),
                    sqlite3.Binary(row[4]),
                    row[5],
                    row[6],
                ),
            )
        state, _ = _prior_rows(connection, assignments, priors)
        if expected and state != "complete":
            raise ArtifactError("pose-prior verification failed before commit")
        connection.commit()
    except Exception:
        if "connection" in locals():
            connection.rollback()
        raise
    finally:
        if "connection" in locals():
            connection.close()


def insert_pose_priors(
    artifact: str | Path,
    *,
    covariance_floor_m: float = 0.03,
    max_covariance_m2: float | None = 0.04,
    min_fix_status: int = 0,
    min_carrier_status: int = -1,
    max_source_residual_s: float | None = 0.15,
) -> dict[str, Any]:
    root, manifest, reader = _sealed_context(artifact)
    database = root / "database.db"
    assignment = _rig_assignments(database, manifest)
    if assignment is None:
        raise ArtifactError("pose-prior insertion requires configured rigs")
    assignments, _ = assignment
    priors, skipped, quality_counts, covariance_model = _trusted_priors(
        reader,
        covariance_floor_m=covariance_floor_m,
        max_covariance_m2=max_covariance_m2,
        min_fix_status=min_fix_status,
        min_carrier_status=min_carrier_status,
        max_source_residual_s=max_source_residual_s,
    )
    if len(priors) < _MINIMUM_TRUSTED_PRIORS:
        rejection = {
            "schema_version": 1,
            "stage": "pose_priors_preflight",
            "status": "rejected",
            "minimum_trusted_priors": _MINIMUM_TRUSTED_PRIORS,
            "n_candidate_frames": len(manifest["frames"]),
            "n_trusted_priors": len(priors),
            "n_rejected": len(manifest["frames"]) - len(priors),
            "skipped_by_reason": skipped,
            "position_quality_counts": quality_counts,
            "raw_status_gates": {
                "min_fix_status": min_fix_status,
                "min_carrier_status": min_carrier_status,
            },
            "max_covariance_m2": max_covariance_m2,
            "max_source_residual_s": max_source_residual_s,
            "camera_center_covariance_model": covariance_model,
        }
        audit = _record_prior_preflight_rejection(root, rejection)
        raise ArtifactError(
            "pose-prior preflight rejected the frontend: "
            f"{len(priors)} trusted priors, at least "
            f"{_MINIMUM_TRUSTED_PRIORS} required; reasons={skipped}; "
            f"audit={audit}"
        )
    inputs = {
        "rig_stage_marker_sha256": _stage_marker_digest(root, "rig"),
        "frames_sha256": sha256_file(reader.root / "frames.npz"),
        "gnss_sha256": sha256_file(reader.root / "observations" / "gnss.npz"),
        "segment_meta_sha256": sha256_file(
            reader.root / "segment_meta.json"
        ),
        "covariance_floor_m": covariance_floor_m,
        "max_covariance_m2": max_covariance_m2,
        "min_fix_status": min_fix_status,
        "min_carrier_status": min_carrier_status,
        "max_source_residual_s": max_source_residual_s,
        "assignment_sha256": canonical_hash(sorted(assignments.items())),
    }
    with _connect_readonly(database) as connection:
        state, _ = _prior_rows(connection, assignments, priors)
    if not _has_stage_marker(root, "pose_priors") and state == "complete":
        raise ArtifactError("refusing unmarked/pre-existing pose priors")
    ledger = StageLedger(root)
    status = ledger.begin("pose_priors", inputs)
    if status == "complete":
        if priors and state != "complete":
            raise ArtifactError("completed pose-prior stage is stale")
        return _completed_report(root, "pose_priors")
    if state == "absent":
        _insert_prior_rows(database, assignments, priors)
    report = {
        "schema_version": 1,
        "stage": "pose_priors",
        "coordinate_system": "CARTESIAN",
        "covariance_floor_m": covariance_floor_m,
        "n_candidate_frames": len(manifest["frames"]),
        "n_inserted": len(priors),
        "n_skipped": len(manifest["frames"]) - len(priors),
        "skipped_by_reason": skipped,
        "position_quality_counts": quality_counts,
        "minimum_trusted_priors": _MINIMUM_TRUSTED_PRIORS,
        "raw_status_gates": {
            "min_fix_status": min_fix_status,
            "min_carrier_status": min_carrier_status,
        },
        "camera_center_covariance_model": covariance_model,
        "assignment_sha256": canonical_hash(sorted(assignments.items())),
    }
    return _finish_stage(ledger, root, "pose_priors", inputs, report)


def _pair_ids(
    database: Path, pairs_path: Path
) -> tuple[tuple[int, ...], dict[int, tuple[str, str]]]:
    pairs: list[tuple[str, str]] = []
    for line_number, line in enumerate(
        pairs_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split()
        if len(parts) != 2 or parts[0] == parts[1]:
            raise ArtifactError(f"invalid pairs.txt line {line_number}")
        pairs.append((parts[0], parts[1]))
    if not pairs or len(pairs) != len(set(tuple(sorted(pair)) for pair in pairs)):
        raise ArtifactError("pairs.txt must contain unique non-empty pairs")
    with _connect_readonly(database) as connection:
        images = {
            str(name): int(image_id)
            for image_id, name in connection.execute(
                "SELECT image_id, name FROM images"
            )
        }
    result: dict[int, tuple[str, str]] = {}
    for first, second in pairs:
        if first not in images or second not in images:
            raise ArtifactError(f"pair references unknown image: {(first, second)}")
        low, high = sorted((images[first], images[second]))
        pair_id = low * _PAIR_ID_BASE + high
        if pair_id in result:
            raise ArtifactError("multiple pair lines resolve to the same image pair")
        result[pair_id] = (first, second)
    return tuple(sorted(result)), result


def _matching_quality(
    database: Path, pair_ids: Sequence[int]
) -> dict[str, Any] | None:
    with _connect_readonly(database) as connection:
        _require_tables(connection, {"matches", "two_view_geometries"})
        requested = set(pair_ids)
        matched = {
            int(pair_id): int(rows)
            for pair_id, rows in connection.execute(
                "SELECT pair_id, rows FROM matches"
            )
            if int(pair_id) in requested
        }
        verified = {
            int(pair_id): int(rows)
            for pair_id, rows in connection.execute(
                "SELECT pair_id, rows FROM two_view_geometries"
            )
            if int(pair_id) in requested
        }
    if not (set(matched) or set(verified)):
        return None
    if set(matched) != requested or set(verified) != requested:
        raise ArtifactError("partial requested-pair matching output")
    return {
        "n_requested_pairs": len(pair_ids),
        "n_matched_pairs": len(matched),
        "n_verified_pairs": len(verified),
        "n_raw_matches": sum(matched.values()),
        "n_verified_matches": sum(verified.values()),
    }


def run_matches_importer(
    artifact: str | Path,
    executable: str | Path,
    *,
    use_gpu: bool = True,
    gpu_index: int = -1,
    num_threads: int = 8,
    seed: int = 0,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    root, _, _ = _sealed_context(artifact)
    database = root / "database.db"
    pair_ids, _ = _pair_ids(database, root / "pairs.txt")
    command = build_matches_importer_command(
        root,
        executable,
        use_gpu=use_gpu,
        gpu_index=gpu_index,
        num_threads=num_threads,
        seed=seed,
    )
    inputs = {
        "command": command,
        "pose_prior_stage_marker_sha256": _stage_marker_digest(
            root, "pose_priors"
        ),
        "pairs_sha256": sha256_file(root / "pairs.txt"),
    }
    quality = _matching_quality(database, pair_ids)
    if not _has_stage_marker(root, "matching") and quality is not None:
        raise ArtifactError("refusing unmarked/pre-existing requested matches")
    ledger = StageLedger(root)
    status = ledger.begin("matching", inputs)
    if status == "complete":
        if quality is None:
            raise ArtifactError("completed matching stage has no requested pairs")
        verify_frontend_seal(root)
        return _completed_report(root, "matching")
    if quality is None:
        _execute(command, runner)
        quality = _matching_quality(database, pair_ids)
        if quality is None:
            raise ArtifactError("matches importer produced no requested pairs")
    report = {
        "schema_version": 1,
        "stage": "matching",
        "command": list(command),
        **quality,
    }
    report_path = _write_report(root, "matching", report)
    create_frontend_seal(root)
    ledger.complete(
        "matching",
        inputs,
        [report_path, root / "database.db", root / FRONTEND_SEAL_FILE],
    )
    verify_frontend_seal(root)
    return report
