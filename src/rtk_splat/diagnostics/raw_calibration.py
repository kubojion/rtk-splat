"""Sealed, explicit raw-bag calibration for contract-v2 field segments.

This module adapts the existing :mod:`calibration_sidecar` solver to a caller-
selected set of frames from a completed COLMAP rig trajectory.  It does not
implement a new optimizer, publish production poses, or use final evaluation
observations.  Selection, physical priors, raw recordings, and the reserved
production roles are explicit hash-bound inputs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from rtk_splat.backends.geodetic_submap import _input_context
from rtk_splat.core.segment import (
    SegmentReader,
    SegmentWriter,
    publish_directory_noreplace,
)
from rtk_splat.diagnostics.calibration_io import (
    RawColmapRigTrajectory,
    load_raw_colmap_rig_trajectory,
)
from rtk_splat.diagnostics.calibration_sidecar import (
    AuditConfig,
    _build_numerical_data,
    _human_report,
    _model_from_report,
)
from rtk_splat.diagnostics.metric_calibration import (
    CALIBRATION_PARAMETER_NAMES,
    CalibrationProblem,
    calibrate,
)
from rtk_splat.frontends.artifact import (
    ArtifactError,
    _atomic_json,
    canonical_hash,
    sha256_file,
)
from rtk_splat.workflows.configio import load_config


_PLAN_KIND = "rtk_splat_raw_calibration_plan"
_RESULT_KIND = "rtk_splat_raw_calibration_result"
_DERIVED_SEGMENT_KIND = "rtk_splat_fixed_calibration_segment"
_PLAN_FILES = (
    "bag_audit.json",
    "calibration_plan.json",
    "input_manifest.json",
    "selection.json",
)
_RESULT_FILES = (
    "diagnostics.json",
    "observations.npz",
    "observability.npz",
    "provenance.json",
    "raw_visual_poses.npz",
    "recommendation.json",
    "residuals.npz",
    "result.json",
    "temporal_folds.json",
    "REPORT.md",
)
_UBLOX_MESSAGE_STEMS = (
    "CarrSoln",
    "GpsFix",
    "PSMPVT",
    "UBXNavRelPosNED",
    "UBXNavPVT",
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON artifact must contain an object: {path}")
    return value


def _file_record(path: Path, *, include_mtime: bool = False) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.is_file() or target.is_symlink():
        raise ArtifactError(f"immutable input file is missing or unsafe: {target}")
    stat = target.stat()
    record: dict[str, Any] = {
        "path": str(target),
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(target),
    }
    if include_mtime:
        record["mtime_ns"] = int(stat.st_mtime_ns)
    return record


def _verify_file_record(record: Mapping[str, Any], *, hash_file: bool) -> Path:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ArtifactError(f"sealed immutable input is missing or unsafe: {path}")
    stat = path.stat()
    if int(record.get("size_bytes", -1)) != stat.st_size:
        raise ArtifactError(f"sealed immutable input size changed: {path}")
    if "mtime_ns" in record and int(record["mtime_ns"]) != stat.st_mtime_ns:
        raise ArtifactError(f"sealed immutable input timestamp changed: {path}")
    if hash_file and record.get("sha256") != sha256_file(path):
        raise ArtifactError(f"sealed immutable input content changed: {path}")
    return path


def _seal_files(root: Path, names: Sequence[str]) -> dict[str, Any]:
    files = {
        name: {
            "sha256": sha256_file(root / name),
            "size_bytes": (root / name).stat().st_size,
        }
        for name in sorted(names)
    }
    body = {"schema_version": 1, "files": files}
    return {**body, "seal_sha256": canonical_hash(body)}


def _verify_seal(root: Path, seal_name: str, names: Sequence[str]) -> dict[str, Any]:
    seal = _json(root / seal_name)
    body = dict(seal)
    digest = body.pop("seal_sha256", None)
    if digest != canonical_hash(body) or body != {
        "schema_version": 1,
        "files": {
            name: {
                "sha256": sha256_file(root / name),
                "size_bytes": (root / name).stat().st_size,
            }
            for name in sorted(names)
        },
    }:
        raise ArtifactError(f"sealed artifact files changed: {root}")
    return seal


def _fsync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def atomic_raw_calibration_artifact(destination: str | Path):
    """Stage and publish one directory atomically without replacement."""
    final = Path(destination).expanduser().resolve()
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite sealed artifact: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.writing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        yield staging
        if not any(staging.iterdir()):
            raise ArtifactError("refusing to publish an empty sealed artifact")
        _fsync_tree(staging)
        publish_directory_noreplace(staging, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _bag_metadata(directory: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = Path(directory).expanduser().resolve()
    metadata_path = root / "metadata.yaml"
    if not root.is_dir() or root.is_symlink() or not metadata_path.is_file():
        raise ArtifactError(f"ROS bag directory is missing or unsafe: {root}")
    try:
        raw = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        info = raw["rosbag2_bagfile_information"]
        storage = str(info["storage_identifier"])
        start_ns = int(info["starting_time"]["nanoseconds_since_epoch"])
        duration_ns = int(info["duration"]["nanoseconds"])
        count = int(info["message_count"])
        relative_paths = [str(value) for value in info["relative_file_paths"]]
        topics = {
            str(item["topic_metadata"]["name"]): {
                "type": str(item["topic_metadata"]["type"]),
                "serialization_format": str(
                    item["topic_metadata"]["serialization_format"]
                ),
                "message_count": int(item["message_count"]),
                "qos_sha256": canonical_hash(
                    str(item["topic_metadata"].get("offered_qos_profiles", ""))
                ),
            }
            for item in info["topics_with_message_count"]
        }
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ArtifactError(f"invalid rosbag metadata: {metadata_path}") from exc
    if duration_ns <= 0 or count <= 0 or not relative_paths or not topics:
        raise ArtifactError(f"incomplete rosbag metadata: {metadata_path}")
    files = [root / value for value in relative_paths]
    if any(not path.is_file() or path.is_symlink() for path in files):
        raise ArtifactError(f"rosbag storage inventory is incomplete: {root}")
    return root, {
        "storage_identifier": storage,
        "start_ns": start_ns,
        "stop_ns": start_ns + duration_ns,
        "duration_ns": duration_ns,
        "message_count": count,
        "topics": topics,
        "relative_file_paths": relative_paths,
    }, {"metadata": metadata_path, "storage": files}


def _sqlite_integrity(path: Path) -> dict[str, Any]:
    uri = f"file:{path}?mode=ro&immutable=1"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            connection.execute("PRAGMA query_only=ON")
            quick = [str(row[0]) for row in connection.execute("PRAGMA quick_check")]
            schema = [
                {
                    "type": str(kind),
                    "name": str(name),
                    "table": str(table),
                    "sql": None if sql is None else str(sql),
                }
                for kind, name, table, sql in connection.execute(
                    "SELECT type,name,tbl_name,sql FROM sqlite_schema "
                    "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name,tbl_name"
                )
            ]
            count, minimum, maximum = connection.execute(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM messages"
            ).fetchone()
            topic_count = int(
                connection.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
            )
    except sqlite3.Error as exc:
        raise ArtifactError(f"cannot audit immutable SQLite bag {path}: {exc}") from exc
    if quick != ["ok"] or minimum is None or maximum is None:
        raise ArtifactError(f"SQLite bag integrity check failed: {path}")
    return {
        "quick_check": "ok",
        "schema_sha256": canonical_hash(schema),
        "message_count": int(count),
        "minimum_log_timestamp_ns": int(minimum),
        "maximum_log_timestamp_ns": int(maximum),
        "topic_count": topic_count,
    }


def _mcap_integrity(directory: Path) -> dict[str, Any]:
    # Keep ROS bag runtime imports in the adapter layer. Opening the reader
    # forces footer, summary, channel, and chunk-index parsing.
    from rtk_splat.adapters.calibration_bag import mcap_index_integrity

    try:
        return mcap_index_integrity(directory)
    except BaseException as exc:
        raise ArtifactError(f"cannot parse MCAP indexes in {directory}: {exc}") from exc


def audit_raw_bag_inputs(
    navigation_bag: str | Path,
    camera_bag: str | Path,
    ublox_msgs_dir: str | Path,
    *,
    camera_topics: Sequence[str],
    navigation_topics: Sequence[str],
) -> dict[str, Any]:
    """Hash and structurally audit two immutable simultaneous ROS bags."""
    nav_root, nav_meta, nav_files = _bag_metadata(Path(navigation_bag))
    camera_root, camera_meta, camera_files = _bag_metadata(Path(camera_bag))
    if nav_root == camera_root:
        raise ArtifactError("navigation and camera bags must be distinct")
    if nav_meta["storage_identifier"] != "sqlite3":
        raise ArtifactError("navigation bag must use sqlite3 storage")
    if camera_meta["storage_identifier"] != "mcap":
        raise ArtifactError("camera bag must use MCAP storage")
    missing_camera = set(camera_topics) - set(camera_meta["topics"])
    missing_navigation = set(navigation_topics) - set(nav_meta["topics"])
    if missing_camera or missing_navigation:
        raise ArtifactError(
            "required raw topics are missing "
            f"(camera={sorted(missing_camera)}, navigation={sorted(missing_navigation)})"
        )
    overlap_start = max(nav_meta["start_ns"], camera_meta["start_ns"])
    overlap_stop = min(nav_meta["stop_ns"], camera_meta["stop_ns"])
    if overlap_stop <= overlap_start:
        raise ArtifactError("raw navigation and camera bags have no common time span")
    message_dir = Path(ublox_msgs_dir).expanduser().resolve()
    message_files = [message_dir / f"{stem}.msg" for stem in _UBLOX_MESSAGE_STEMS]
    if any(not path.is_file() for path in message_files):
        raise ArtifactError("required u-blox message definitions are incomplete")

    nav_storage = nav_files["storage"]
    camera_storage = camera_files["storage"]
    if len(nav_storage) != 1 or len(camera_storage) != 1:
        raise ArtifactError("raw bag audit requires one storage file per bag")
    records = {
        "navigation_metadata": _file_record(nav_files["metadata"], include_mtime=True),
        "navigation_storage": _file_record(nav_storage[0], include_mtime=True),
        "camera_metadata": _file_record(camera_files["metadata"], include_mtime=True),
        "camera_storage": _file_record(camera_storage[0], include_mtime=True),
        "ublox_message_definitions": [
            _file_record(path, include_mtime=True) for path in message_files
        ],
    }
    nav_integrity = _sqlite_integrity(nav_storage[0])
    if nav_integrity["message_count"] != nav_meta["message_count"]:
        raise ArtifactError("navigation metadata and SQLite message counts disagree")
    camera_integrity = _mcap_integrity(camera_root)
    return {
        "schema_version": 1,
        "read_only": True,
        "navigation_bag": str(nav_root),
        "camera_bag": str(camera_root),
        "navigation": {**nav_meta, "integrity": nav_integrity},
        "camera": {**camera_meta, "integrity": camera_integrity},
        "common_time_span_ns": {
            "start": overlap_start,
            "stop": overlap_stop,
            "duration": overlap_stop - overlap_start,
        },
        "required_topics": {
            "camera": list(camera_topics),
            "navigation": list(navigation_topics),
        },
        "files": records,
        "files_sha256": canonical_hash(records),
    }


def _audit_config_record(audit: AuditConfig) -> dict[str, Any]:
    value = asdict(audit)
    value["body_from_camera"] = audit.body_from_camera.tolist()
    value["primary_antenna_body_m"] = audit.primary_antenna_body_m.tolist()
    value["secondary_antenna_body_m"] = audit.secondary_antenna_body_m.tolist()
    value["solver"] = asdict(audit.solver)
    for key, item in list(value["solver"].items()):
        if isinstance(item, np.ndarray):
            value["solver"][key] = item.tolist()
    # Plans are audited after a JSON round trip, where tuples necessarily
    # become lists.  Normalize the in-memory record to that same representation
    # before sealing or comparing it.
    return json.loads(json.dumps(value, allow_nan=False))


def _role_names(value: Mapping[str, Any], name: str) -> tuple[str, ...]:
    raw = value.get(name)
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ArtifactError(f"evaluation roles have no valid {name}")
    result = tuple(raw)
    if len(result) != len(set(result)):
        raise ArtifactError(f"evaluation role {name} contains duplicates")
    return result


def prepare_raw_calibration_plan(
    frontend_artifact: str | Path,
    completed_backend: str | Path,
    segment: str | Path,
    navigation_bag: str | Path,
    camera_bag: str | Path,
    ublox_msgs_dir: str | Path,
    calibration_config: str | Path,
    evaluation_roles: str | Path,
    destination: str | Path,
    *,
    window_start_s: float,
    window_stop_s: float,
    sim3_scale_range: tuple[float, float] = (0.98, 1.02),
) -> Path:
    """Create a sealed, leakage-free raw calibration plan."""
    if not (
        math.isfinite(window_start_s)
        and math.isfinite(window_stop_s)
        and 0.0 <= window_start_s < window_stop_s
    ):
        raise ValueError("calibration window seconds must be finite and ordered")
    scale_low, scale_high = map(float, sim3_scale_range)
    if not (0.0 < scale_low <= 1.0 <= scale_high):
        raise ValueError("Sim(3) diagnostic scale range must contain 1.0")
    output = Path(destination).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite raw calibration plan: {output}")

    (
        frontend,
        manifest,
        backend,
        _source_plan,
        _mapper_config,
        source_quality,
        segment_path,
        reader,
        source_evidence,
    ) = _input_context(frontend_artifact, completed_backend, segment)
    if not source_quality.get("passed"):
        raise ArtifactError("completed visual model did not pass its quality gates")
    config_path = Path(calibration_config).expanduser().resolve()
    cfg = load_config(config_path)
    audit = AuditConfig.from_config(cfg)
    topics = {
        "left_image": str(cfg.topics.left_image),
        "right_image": str(cfg.topics.right_image),
        "fix": str(cfg.topics.fix),
        "relpos": str(cfg.topics.relpos),
        "moving_base_pvt": audit.moving_base_pvt_topic,
    }
    bag_audit = audit_raw_bag_inputs(
        navigation_bag,
        camera_bag,
        ublox_msgs_dir,
        camera_topics=[topics["left_image"], topics["right_image"]],
        navigation_topics=[
            topics["fix"], topics["relpos"], topics["moving_base_pvt"]
        ],
    )

    role_path = Path(evaluation_roles).expanduser().resolve()
    roles = _json(role_path)
    calibration_names = set(_role_names(roles, "calibration_names"))
    holdout_names = set(_role_names(roles, "holdout_names"))
    excluded_names = set(_role_names(roles, "excluded_names"))
    if (
        calibration_names & holdout_names
        or calibration_names & excluded_names
        or holdout_names & excluded_names
    ):
        raise ArtifactError("production evaluation roles overlap")

    frame_rows = manifest.get("frames")
    if not isinstance(frame_rows, list) or len(frame_rows) != len(reader.frames["frame_id"]):
        raise ArtifactError("frontend frame manifest does not match the segment")
    frame_by_name = {
        str(item["left_image"]["name"]): item for item in frame_rows
    }
    if len(frame_by_name) != len(frame_rows):
        raise ArtifactError("frontend frame manifest has duplicate left names")
    all_role_names = calibration_names | holdout_names | excluded_names
    if not all_role_names <= set(frame_by_name):
        raise ArtifactError("production evaluation roles contain unknown images")

    camera_start_ns = int(bag_audit["camera"]["start_ns"])
    absolute_start_ns = camera_start_ns + int(round(window_start_s * 1.0e9))
    absolute_stop_ns = camera_start_ns + int(round(window_stop_s * 1.0e9))
    selection = []
    for item in frame_rows:
        name = str(item["left_image"]["name"])
        timestamp_ns = int(item["timestamp_ns"])
        if name not in calibration_names or not (
            absolute_start_ns <= timestamp_ns <= absolute_stop_ns
        ):
            continue
        frame_id = int(item["frame_id"])
        selection.append(
            {
                "frame_id": frame_id,
                "left_name": name,
                "right_name": str(item["right_image"]["name"]),
                "left_header_ns": timestamp_ns,
                "right_header_ns": int(item["right_timestamp_ns"]),
                "left_relative_path": str(item["left_image"]["source_relative_path"]),
                "right_relative_path": str(item["right_image"]["source_relative_path"]),
                "left_sha256": str(item["left_image"]["content_sha256"]),
                "right_sha256": str(item["right_image"]["content_sha256"]),
            }
        )
    if len(selection) < 40:
        raise ArtifactError("predeclared calibration window has too few calibration frames")
    frame_ids = [item["frame_id"] for item in selection]
    if frame_ids != sorted(frame_ids) or len(frame_ids) != len(set(frame_ids)):
        raise ArtifactError("calibration selection is not unique and ordered")
    selected_names = {item["left_name"] for item in selection}
    if selected_names & (holdout_names | excluded_names):
        raise ArtifactError("calibration selection leaked production evaluation roles")

    input_manifest = {
        "schema_version": 1,
        "frontend_artifact": str(frontend),
        "completed_backend": str(backend),
        "segment": str(segment_path),
        "source_evidence": source_evidence,
        "calibration_config": _file_record(config_path),
        "evaluation_roles": _file_record(role_path),
        "raw_bag_files": bag_audit["files"],
    }
    selection_record = {
        "schema_version": 1,
        "window": {
            "origin": "camera_bag_metadata_start",
            "start_s": float(window_start_s),
            "stop_s": float(window_stop_s),
            "absolute_start_ns": absolute_start_ns,
            "absolute_stop_ns": absolute_stop_ns,
        },
        "production_roles": {
            "n_calibration": len(calibration_names),
            "n_holdout": len(holdout_names),
            "n_excluded": len(excluded_names),
            "membership_sha256": canonical_hash(
                {
                    "calibration": sorted(calibration_names),
                    "holdout": sorted(holdout_names),
                    "excluded": sorted(excluded_names),
                }
            ),
        },
        "selection_method": (
            "intersection of externally reserved production-calibration role "
            "and caller-predeclared absolute camera-time window"
        ),
        "heldout_used_for_selection": False,
        "frames": selection,
        "frame_ids_sha256": canonical_hash(frame_ids),
    }
    plan_body = {
        "schema_version": 1,
        "kind": _PLAN_KIND,
        "experimental": True,
        "input_manifest_sha256": canonical_hash(input_manifest),
        "bag_audit_sha256": canonical_hash(bag_audit),
        "selection_sha256": canonical_hash(selection_record),
        "selected_frame_count": len(selection),
        "selected_frame_ids_sha256": canonical_hash(frame_ids),
        "audit_config": _audit_config_record(audit),
        "topics": topics,
        "sim3_diagnostic_scale_range": [scale_low, scale_high],
        "fixed_scale": 1.0,
        "uses_imu": False,
        "uses_production_holdout": False,
        "publishes_poses": False,
    }
    plan = {**plan_body, "plan_sha256": canonical_hash(plan_body)}
    with atomic_raw_calibration_artifact(output) as staging:
        _atomic_json(staging / "input_manifest.json", input_manifest)
        _atomic_json(staging / "bag_audit.json", bag_audit)
        _atomic_json(staging / "selection.json", selection_record)
        _atomic_json(staging / "calibration_plan.json", plan)
        _atomic_json(staging / "plan_seal.json", _seal_files(staging, _PLAN_FILES))
    return output


def audited_raw_calibration_plan(
    artifact: str | Path,
    *,
    verify_runtime_inputs: bool = True,
) -> dict[str, Any]:
    root = Path(artifact).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ArtifactError("raw calibration plan is missing or unsafe")
    expected = set(_PLAN_FILES) | {"plan_seal.json"}
    if {path.name for path in root.iterdir()} != expected:
        raise ArtifactError("raw calibration plan inventory changed")
    seal = _verify_seal(root, "plan_seal.json", _PLAN_FILES)
    manifest = _json(root / "input_manifest.json")
    bag_audit = _json(root / "bag_audit.json")
    selection = _json(root / "selection.json")
    plan = _json(root / "calibration_plan.json")
    body = dict(plan)
    digest = body.pop("plan_sha256", None)
    frames = selection.get("frames")
    if not isinstance(frames, list):
        raise ArtifactError("raw calibration selection has no frames")
    frame_ids = [int(item["frame_id"]) for item in frames]
    checks = (
        plan.get("kind") == _PLAN_KIND,
        plan.get("schema_version") == 1,
        digest == canonical_hash(body),
        plan.get("input_manifest_sha256") == canonical_hash(manifest),
        plan.get("bag_audit_sha256") == canonical_hash(bag_audit),
        plan.get("selection_sha256") == canonical_hash(selection),
        plan.get("selected_frame_count") == len(frames),
        plan.get("selected_frame_ids_sha256") == canonical_hash(frame_ids),
        selection.get("frame_ids_sha256") == canonical_hash(frame_ids),
        plan.get("fixed_scale") == 1.0,
        plan.get("uses_imu") is False,
        plan.get("uses_production_holdout") is False,
        plan.get("publishes_poses") is False,
    )
    if not all(checks):
        raise ArtifactError("raw calibration plan binding changed")
    runtime = None
    if verify_runtime_inputs:
        config_path = _verify_file_record(manifest["calibration_config"], hash_file=True)
        role_path = _verify_file_record(manifest["evaluation_roles"], hash_file=True)
        # Full raw hashes are rechecked after the bag pass.  Before that pass,
        # size and mtime provide a cheap fail-fast guard.
        for key in (
            "navigation_metadata",
            "navigation_storage",
            "camera_metadata",
            "camera_storage",
        ):
            _verify_file_record(bag_audit["files"][key], hash_file=False)
        for record in bag_audit["files"]["ublox_message_definitions"]:
            _verify_file_record(record, hash_file=True)
        runtime = _input_context(
            manifest["frontend_artifact"],
            manifest["completed_backend"],
            manifest["segment"],
        )
        if runtime[-1] != manifest["source_evidence"]:
            raise ArtifactError("sealed visual/segment source evidence changed")
        cfg = load_config(config_path)
        if _audit_config_record(AuditConfig.from_config(cfg)) != plan["audit_config"]:
            raise ArtifactError("sealed calibration configuration changed")
        roles = _json(role_path)
        role_membership = canonical_hash(
            {
                "calibration": sorted(_role_names(roles, "calibration_names")),
                "holdout": sorted(_role_names(roles, "holdout_names")),
                "excluded": sorted(_role_names(roles, "excluded_names")),
            }
        )
        if role_membership != selection["production_roles"]["membership_sha256"]:
            raise ArtifactError("sealed production role membership changed")
    return {
        **plan,
        "artifact": str(root),
        "seal": seal,
        "plan_seal_sha256": sha256_file(root / "plan_seal.json"),
        "input_manifest": manifest,
        "bag_audit": bag_audit,
        "selection": selection,
        "_runtime_context": runtime,
    }


def _subset_trajectory(
    trajectory: RawColmapRigTrajectory, indices: Sequence[int]
) -> RawColmapRigTrajectory:
    positions = np.asarray(indices, dtype=np.int64)
    if positions.ndim != 1 or len(positions) == 0:
        raise ValueError("trajectory subset indices must be non-empty")
    by_frame = {int(value): index for index, value in enumerate(trajectory.frame_indices)}
    try:
        selected = np.asarray([by_frame[int(value)] for value in positions], dtype=np.int64)
    except KeyError as exc:
        raise ArtifactError(f"selected frame is absent from visual model: {exc}") from exc
    return RawColmapRigTrajectory(
        frame_indices=trajectory.frame_indices[selected],
        colmap_frame_ids=trajectory.colmap_frame_ids[selected],
        image_ids=trajectory.image_ids[selected],
        image_names=tuple(trajectory.image_names[index] for index in selected),
        camera_from_visual_world=trajectory.camera_from_visual_world[selected],
        camera_centers_visual=trajectory.camera_centers_visual[selected],
    )


def baseline_initial_rotation(
    data,
    baseline_camera_m: np.ndarray,
    *,
    maximum_age_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate only an initial SE(3) rotation from raw dual-antenna vectors."""
    query = np.asarray(data.camera_t_s, dtype=float)
    reference = np.asarray(data.baseline_t_s, dtype=float)
    right = np.clip(np.searchsorted(reference, query), 0, len(reference) - 1)
    left = np.clip(right - 1, 0, len(reference) - 1)
    use_left = np.abs(query - reference[left]) <= np.abs(reference[right] - query)
    nearest = np.where(use_left, left, right)
    age = np.abs(query - reference[nearest])
    good = np.asarray(data.baseline_good, dtype=bool)[nearest] & (age <= maximum_age_s)
    if int(good.sum()) < 20:
        raise ArtifactError("too few raw baseline vectors for rotation initialization")
    visual_from_camera = np.asarray(data.visual_from_camera_rotation)[good]
    predicted_visual = np.einsum(
        "nij,j->ni", visual_from_camera, np.asarray(baseline_camera_m, dtype=float)
    )
    measured_enu = np.asarray(data.baseline_enu_m)[nearest[good]]
    covariance = np.asarray(data.baseline_cov_enu_m2)[nearest[good]]
    weights = 1.0 / np.maximum(np.trace(covariance, axis1=1, axis2=2), 1.0e-8)
    cross = np.einsum("n,ni,nj->ij", weights, measured_enu, predicted_visual)
    u, singular, vt = np.linalg.svd(cross)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    if singular[0] <= 0 or singular[1] / singular[0] < 0.01:
        raise ArtifactError("raw baseline heading excitation is insufficient")
    mapped = np.einsum("ij,nj->ni", rotation, predicted_visual)
    cosine = np.sum(mapped * measured_enu, axis=1) / np.maximum(
        np.linalg.norm(mapped, axis=1) * np.linalg.norm(measured_enu, axis=1),
        1.0e-12,
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return rotation, {
        "method": "weighted Kabsch on raw dual-antenna vectors only",
        "n_vectors": int(good.sum()),
        "maximum_match_age_ms": float(maximum_age_s * 1000.0),
        "actual_match_age_p95_ms": float(np.percentile(age[good], 95) * 1000.0),
        "cross_covariance_singular_values": singular.tolist(),
        "second_to_first_singular_ratio": float(singular[1] / singular[0]),
        "post_alignment_angle_median_deg": float(np.median(angles)),
        "post_alignment_angle_p95_deg": float(np.percentile(angles, 95)),
        "uses_positions": False,
        "uses_production_holdout": False,
        "fixed_scale": 1.0,
    }


def _source_hashes_after_run(plan: Mapping[str, Any]) -> None:
    bag_files = plan["bag_audit"]["files"]
    for key in (
        "navigation_metadata",
        "navigation_storage",
        "camera_metadata",
        "camera_storage",
    ):
        _verify_file_record(bag_files[key], hash_file=True)
    for record in bag_files["ublox_message_definitions"]:
        _verify_file_record(record, hash_file=True)


def _corrected_recommendation(
    problem: CalibrationProblem,
    correction: np.ndarray,
    audit: AuditConfig,
) -> dict[str, Any]:
    corrected_body_camera = problem.corrected_body_from_camera(correction)
    body_from_primary = np.eye(4)
    body_from_primary[:3, 3] = audit.primary_antenna_body_m
    camera_from_primary = np.linalg.inv(corrected_body_camera) @ body_from_primary
    _, corrected_baseline_camera, clock_s = problem.corrected_geometry(correction)
    return {
        "schema_version": 1,
        "accepted_fixed_calibration": True,
        "T_body_left_camera": corrected_body_camera.tolist(),
        "T_camera_primary_antenna": camera_from_primary.tolist(),
        "T_camera_primary_antenna_convention": "camera_from_primary_antenna",
        "primary_to_secondary_baseline_camera_m": corrected_baseline_camera.tolist(),
        "baseline_length_m": float(np.linalg.norm(corrected_baseline_camera)),
        "camera_to_rtk_offset_ns": int(round(clock_s * 1.0e9)),
        "clock_model": "GNSS header time = camera header time + offset",
        "metric_scale": 1.0,
        "lever_arm_estimated_inside_mapper": False,
        "scale_estimated_inside_mapper": False,
        "uses_imu": False,
    }


def run_raw_calibration_plan(
    plan_artifact: str | Path,
    destination: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Run and atomically publish one accepted fixed-scale calibration."""
    final = Path(destination).expanduser().resolve()
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite raw calibration result: {final}")
    plan = audited_raw_calibration_plan(plan_artifact, verify_runtime_inputs=True)
    runtime = plan["_runtime_context"]
    assert runtime is not None
    frontend, manifest, backend, _, _, _, segment, reader, _ = runtime
    config_path = Path(plan["input_manifest"]["calibration_config"]["path"])
    cfg = load_config(config_path)
    audit = AuditConfig.from_config(cfg)
    selection = plan["selection"]["frames"]
    frame_ids = np.asarray([item["frame_id"] for item in selection], dtype=np.int64)
    left_assignment = {
        str(item["left_image"]["name"]): int(item["frame_id"])
        for item in manifest["frames"]
    }
    trajectory_all = load_raw_colmap_rig_trajectory(
        backend / "registered_text",
        frontend / "database.db",
        left_image_frame_ids=left_assignment,
        expected_frame_count=len(manifest["frames"]),
    )
    trajectory = _subset_trajectory(trajectory_all, frame_ids)

    from rtk_splat.adapters.calibration_bag import (
        CalibrationTopics,
        build_calibration_typestore,
        read_selected_calibration_bag_observations,
    )

    topics = CalibrationTopics(**plan["topics"])
    bag_files = plan["bag_audit"]["files"]
    camera_bag = Path(plan["bag_audit"]["camera_bag"])
    navigation_bag = Path(plan["bag_audit"]["navigation_bag"])
    message_definitions = bag_files["ublox_message_definitions"]
    ublox_dir = Path(message_definitions[0]["path"]).parent
    typestore = build_calibration_typestore(ublox_dir)
    clock_low, clock_high = audit.solver.clock_offset_bounds_s
    image_min = min(int(item["left_header_ns"]) for item in selection)
    image_max = max(int(item["left_header_ns"]) for item in selection)
    padding_ns = 2_000_000_000
    rtk_start = image_min + int(math.floor(clock_low * 1.0e9)) - padding_ns
    rtk_stop = image_max + int(math.ceil(clock_high * 1.0e9)) + padding_ns
    observations = read_selected_calibration_bag_observations(
        camera_bag=camera_bag,
        navigation_bag=navigation_bag,
        topics=topics,
        typestore=typestore,
        frame_ids=frame_ids,
        left_header_ns=[item["left_header_ns"] for item in selection],
        right_header_ns=[item["right_header_ns"] for item in selection],
        left_image_paths=[segment / item["left_relative_path"] for item in selection],
        right_image_paths=[segment / item["right_relative_path"] for item in selection],
        rtk_start_header_ns=rtk_start,
        rtk_stop_header_ns=rtk_stop,
        log_margin_ns=int(round(audit.log_time_margin_s * 1.0e9)),
    )
    meta = dict(reader.meta)
    data, diagnostics, derived = _build_numerical_data(
        observations, trajectory, meta, audit
    )
    initial_rotation, rotation_report = baseline_initial_rotation(
        data,
        audit.rig_prior.baseline_camera_m,
        maximum_age_s=audit.status_max_age_s,
    )
    diagnostics["initial_visual_to_enu_rotation"] = rotation_report
    problem = CalibrationProblem(
        data,
        audit.rig_prior,
        audit.solver,
        initial_global_rotation=initial_rotation,
    )
    result = calibrate(problem)
    final_section = result["final_retained_prior_safe"]
    scale = float(
        result["sim3_diagnostic_fixed_physical_prior"]["scale_visual_to_enu"]
    )
    scale_low, scale_high = plan["sim3_diagnostic_scale_range"]
    accepted = bool(
        final_section["calibration_accepted"]
        and scale_low <= scale <= scale_high
    )
    if not accepted:
        raise ArtifactError(
            "raw calibration failed prior-safe or fixed-metric-scale gates"
        )
    correction = np.asarray(final_section["correction"], dtype=float)
    recommendation = _corrected_recommendation(problem, correction, audit)

    frame_to_index = {int(frame): index for index, frame in enumerate(data.frame_ids)}
    eligible = np.asarray(
        [frame_to_index[int(frame)] for frame in result["eligible_frame_ids"]],
        dtype=np.int64,
    )
    fit_set = set(result["fit_frame_ids"])
    fit_mask = np.asarray(
        [int(frame) in fit_set for frame in result["eligible_frame_ids"]],
        dtype=bool,
    )
    baseline_result = _model_from_report(
        result["baseline_fixed_scale"], np.zeros(len(CALIBRATION_PARAMETER_NAMES))
    )
    retained_result = _model_from_report(final_section, correction)
    before_pos, before_base, before_angle = problem.raw_errors(eligible, baseline_result)
    after_pos, after_base, after_angle = problem.raw_errors(eligible, retained_result)

    # Rehash all 150+ GB raw inputs only after the bounded pass and before any
    # result can appear.  A transfer/edit during processing therefore fails
    # closed and leaves no published artifact.
    _source_hashes_after_run(plan)
    provenance = {
        "schema_version": 1,
        "kind": _RESULT_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(Path(plan_artifact).expanduser().resolve()),
        "plan_seal_sha256": plan["plan_seal_sha256"],
        "source_inputs_sha256": plan["input_manifest_sha256"],
        "raw_bag_audit_sha256": plan["bag_audit_sha256"],
        "production_heldout_used_for_fitting": False,
        "production_excluded_used_for_fitting": False,
        "visual_residual_used_for_raw_gnss_filtering": False,
        "fixed_scale": 1.0,
        "runs_colmap": False,
        "runs_gaussian_training": False,
        "publishes_poses": False,
        "selected_frame_count": len(selection),
        "selected_frame_ids_sha256": plan["selected_frame_ids_sha256"],
        "raw_topics": plan["topics"],
    }
    diagnostics["sim3_diagnostic_gate"] = {
        "value": scale,
        "minimum": scale_low,
        "maximum": scale_high,
        "passed": True,
        "diagnostic_only": True,
        "applied_scale": 1.0,
    }
    with atomic_raw_calibration_artifact(final) as staging:
        payload = observations.to_npz_payload()
        payload.update(derived)
        np.savez_compressed(staging / "observations.npz", **payload)
        np.savez_compressed(
            staging / "raw_visual_poses.npz",
            frame_indices=trajectory.frame_indices,
            colmap_frame_ids=trajectory.colmap_frame_ids,
            image_ids=trajectory.image_ids,
            image_names=np.asarray(trajectory.image_names),
            camera_from_visual_world=trajectory.camera_from_visual_world,
            camera_centers_visual=trajectory.camera_centers_visual,
        )
        np.savez_compressed(
            staging / "residuals.npz",
            frame_ids=data.frame_ids[eligible],
            fit_mask=fit_mask,
            before_position_enu_m=before_pos,
            after_position_enu_m=after_pos,
            before_baseline_enu_m=before_base,
            after_baseline_enu_m=after_base,
            before_baseline_angle_deg=before_angle,
            after_baseline_angle_deg=after_angle,
        )
        observability = result["observability"]
        np.savez_compressed(
            staging / "observability.npz",
            global_alignment_singular_values=np.asarray(
                observability["global_alignment_singular_values"]
            ),
            singular_values=np.asarray(observability["singular_values_dimensionless"]),
            right_singular_vectors=np.asarray(observability["right_singular_vectors"]),
            data_information_matrix=np.asarray(
                observability["data_information_matrix_dimensionless"]
            ),
            prior_sigma=np.asarray(observability["prior_sigma"]),
            posterior_sigma=np.asarray(observability["posterior_sigma_linearized"]),
            std_reduction_fraction=np.asarray(observability["std_reduction_fraction"]),
        )
        _atomic_json(staging / "diagnostics.json", diagnostics)
        _atomic_json(staging / "result.json", result)
        _atomic_json(staging / "recommendation.json", recommendation)
        _atomic_json(staging / "provenance.json", provenance)
        _atomic_json(
            staging / "temporal_folds.json",
            {
                "fit_frame_ids": result["fit_frame_ids"],
                "heldout_frame_ids": result["heldout_frame_ids"],
                "heldout_role": "internal calibration cross-validation only",
                "production_evaluation_holdout_used": False,
                "eligible_temporal_block_ids": result["eligible_temporal_block_ids"],
                "heldout_block_results": final_section["heldout_blocks"],
            },
        )
        (staging / "REPORT.md").write_text(
            _human_report(result, diagnostics, audit), encoding="utf-8"
        )
        _atomic_json(staging / "result_seal.json", _seal_files(staging, _RESULT_FILES))
    return final, result


def audited_raw_calibration_result(artifact: str | Path) -> dict[str, Any]:
    root = Path(artifact).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ArtifactError("raw calibration result is missing or unsafe")
    expected = set(_RESULT_FILES) | {"result_seal.json"}
    if {path.name for path in root.iterdir()} != expected:
        raise ArtifactError("raw calibration result inventory changed")
    seal = _verify_seal(root, "result_seal.json", _RESULT_FILES)
    provenance = _json(root / "provenance.json")
    result = _json(root / "result.json")
    recommendation = _json(root / "recommendation.json")
    if (
        provenance.get("kind") != _RESULT_KIND
        or provenance.get("production_heldout_used_for_fitting") is not False
        or provenance.get("fixed_scale") != 1.0
        or recommendation.get("accepted_fixed_calibration") is not True
        or recommendation.get("metric_scale") != 1.0
        or result.get("final_retained_prior_safe", {}).get("calibration_accepted")
        is not True
    ):
        raise ArtifactError("raw calibration result contract failed")
    plan = audited_raw_calibration_plan(
        provenance.get("plan", ""), verify_runtime_inputs=False
    )
    if provenance.get("plan_seal_sha256") != plan["plan_seal_sha256"]:
        raise ArtifactError("raw calibration result plan binding changed")
    return {
        "artifact": str(root),
        "passed": True,
        "seal": seal,
        "result_seal_sha256": sha256_file(root / "result_seal.json"),
        "provenance": provenance,
        "result": result,
        "recommendation": recommendation,
    }


def _segment_source_records(root: Path) -> dict[str, Any]:
    names = (
        "calibration.json",
        "frames.npz",
        "manifest.json",
        "observations/gnss.npz",
        "observations/heading.npz",
        "segment_meta.json",
    )
    return {
        name: {
            "sha256": sha256_file(root / name),
            "size_bytes": (root / name).stat().st_size,
        }
        for name in names
    }


def _derived_calibration(
    source: Mapping[str, Any],
    recommendation: Mapping[str, Any],
    result_artifact: Path,
    result_seal_sha256: str,
) -> dict[str, Any]:
    value = json.loads(json.dumps(source))
    rough = value.get("rough_extrinsics")
    if not isinstance(rough, dict):
        raise ArtifactError("source segment has no antenna-camera extrinsic")
    rough["T_camera_primary_antenna"] = recommendation[
        "T_camera_primary_antenna"
    ]
    rough["convention"] = (
        "T_camera_primary_antenna maps primary-antenna frame coordinates "
        "into the camera optical frame"
    )
    rough["provenance"] = {
        "method": "sealed raw camera/RTK/dual-antenna calibration sidecar",
        "status": "accepted fixed calibration; not estimated inside mapper",
        "result_artifact": str(result_artifact),
        "result_seal_sha256": result_seal_sha256,
        "metric_scale": 1.0,
        "camera_to_rtk_offset_ns": int(
            recommendation["camera_to_rtk_offset_ns"]
        ),
    }
    return value


def _derived_meta(
    source: Mapping[str, Any],
    recommendation: Mapping[str, Any],
    derivation: Mapping[str, Any],
) -> dict[str, Any]:
    value = json.loads(json.dumps(source))
    offset_ns = int(recommendation["camera_to_rtk_offset_ns"])
    value["timebase"]["association_clock_offset_ns"] = offset_ns
    value["clock_alignment"]["camera_to_rtk_offset_ns"] = offset_ns
    value["initial_pose"]["source"] = (
        "RTK position/dual-antenna heading plus sealed fixed "
        "antenna-to-camera calibration"
    )
    semantics = value.get("initial_pose_semantics")
    if isinstance(semantics, dict):
        semantics["camera_centres"] = (
            "lever-arm-corrected fixed-calibration camera optical centres "
            "in local ENU"
        )
        semantics["extrinsic_provenance"] = {
            "method": "sealed raw camera/RTK/dual-antenna calibration sidecar",
            "status": "accepted fixed calibration",
            "result_seal_sha256": derivation["calibration_result_seal_sha256"],
        }
        semantics["rough_extrinsic_convention"] = (
            "T_camera_primary_antenna maps primary-antenna frame coordinates "
            "into the camera optical frame"
        )
    value["fixed_calibration_derivation"] = dict(derivation)
    return value


def _hardlink_payloads(
    writer: SegmentWriter,
    source: Path,
    frames: Mapping[str, np.ndarray],
) -> int:
    relative_paths: set[str] = set()
    for field in ("left_image_path", "right_image_path", "depth_path"):
        if field in frames:
            relative_paths.update(
                str(item) for item in frames[field] if str(item)
            )
    count = 0
    for relative in sorted(relative_paths):
        source_path = source / relative
        destination = writer.staging_dir / relative
        if not source_path.is_file() or source_path.is_symlink():
            raise ArtifactError(f"source segment payload is missing or unsafe: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source_path, destination)
        count += 1
    return count


def publish_calibrated_segment(
    source_segment: str | Path,
    calibration_result: str | Path,
    destination: str | Path,
) -> Path:
    """Publish a new segment using an accepted fixed calibration.

    The raw GNSS/heading observations, camera timestamps, images, depth, and
    data splits are retained byte-for-byte.  Only the fixed camera/antenna
    transform and the camera poses derived from it change.  A non-zero clock
    recommendation is rejected here because it would require a separately
    audited observation reassociation rather than a metadata-only edit.
    """
    source = Path(source_segment).expanduser().resolve()
    source_reader = SegmentReader(source).validate()
    result = audited_raw_calibration_result(calibration_result)
    result_root = Path(result["artifact"])
    recommendation = result["recommendation"]
    offset_ns = int(recommendation["camera_to_rtk_offset_ns"])
    if offset_ns != 0:
        raise ArtifactError(
            "non-zero accepted clock offset requires a sealed full observation "
            "reassociation; refusing metadata-only derivation"
        )
    output = Path(destination).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite calibrated segment: {output}")
    source_records_before = _segment_source_records(source)

    provenance = result["provenance"]
    plan = audited_raw_calibration_plan(
        provenance["plan"], verify_runtime_inputs=False
    )
    if Path(plan["input_manifest"]["segment"]).resolve() != source:
        raise ArtifactError("calibration result was not derived from this segment")
    navigation_bag = Path(plan["bag_audit"]["navigation_bag"])
    definitions = plan["bag_audit"]["files"]["ublox_message_definitions"]
    ublox_dir = Path(definitions[0]["path"]).parent

    from rtk_splat.adapters.calibration_bag import build_calibration_typestore
    from rtk_splat.adapters.ros2_rtk_io import read_rtk_track
    from rtk_splat.core.poses import LocalEnu, pose_frames_from_extrinsic

    typestore = build_calibration_typestore(ublox_dir)
    topics = SimpleNamespace(
        fix=plan["topics"]["fix"],
        relpos=plan["topics"]["relpos"],
        moving_base_pvt=plan["topics"]["moving_base_pvt"],
    )
    track = read_rtk_track([navigation_bag], topics, typestore)
    origin = source_reader.meta["world_origin"]
    altitude = origin.get("alt0_ellipsoidal_m", origin.get("alt0"))
    if altitude is None:
        raise ArtifactError("source segment has no ellipsoidal ENU altitude")
    enu = LocalEnu(origin["lat0"], origin["lon0"], altitude)
    track.enu_xyz = enu.to_enu(track.fix_lat, track.fix_lon, track.fix_alt)
    pose_record = source_reader.meta["provenance"]["configuration"][
        "effective_config"
    ]["pose"]
    pose_cfg = SimpleNamespace(**pose_record)

    frames = {key: value.copy() for key, value in source_reader.frames.items()}
    query_ns = frames["timestamp_ns"].astype(np.int64) + offset_ns
    poses = pose_frames_from_extrinsic(
        track,
        query_ns.astype(np.float64) * 1.0e-9,
        pose_cfg,
        recommendation["T_camera_primary_antenna"],
        primary_to_secondary_baseline_camera_m=recommendation[
            "primary_to_secondary_baseline_camera_m"
        ],
    )
    if any(item is None for item in poses):
        missing = sum(item is None for item in poses)
        raise ArtifactError(
            f"accepted calibration cannot pose all source frames ({missing} missing)"
        )
    new_viewmats = np.stack([item.viewmat for item in poses]).astype(np.float64)
    new_centers = np.stack([item.cam_center for item in poses]).astype(np.float64)
    if not np.array_equal(frames["pose_valid"], np.ones(len(poses), dtype=bool)):
        raise ArtifactError("source segment pose-valid inventory is not complete")
    frames["pose_query_timestamp_ns"] = query_ns
    frames["initial_viewmat"] = new_viewmats
    frames["initial_camera_center_m"] = new_centers

    result_seal_sha256 = result["result_seal_sha256"]
    derivation = {
        "schema_version": 1,
        "kind": _DERIVED_SEGMENT_KIND,
        "source_segment": str(source),
        "source_files": source_records_before,
        "source_files_sha256": canonical_hash(source_records_before),
        "calibration_result": str(result_root),
        "calibration_result_seal_sha256": result_seal_sha256,
        "recommendation_sha256": canonical_hash(recommendation),
        "fixed_scale": 1.0,
        "camera_to_rtk_offset_ns": offset_ns,
        "dual_antenna_orientation_method": (
            "measured_horizontal_baseline_yaw_minus_sealed_calibrated_"
            "baseline_yaw_in_primary_antenna_frame"
        ),
        "raw_observations_changed": False,
        "frame_inventory_changed": False,
        "payload_method": "same-filesystem hardlinks to immutable source",
        "allowed_frame_changes": [
            "initial_viewmat",
            "initial_camera_center_m",
            "pose_query_timestamp_ns",
        ],
    }
    calibration = _derived_calibration(
        source_reader.calibration,
        recommendation,
        result_root,
        result_seal_sha256,
    )
    meta = _derived_meta(source_reader.meta, recommendation, derivation)
    writer = SegmentWriter(output)
    try:
        payload_count = _hardlink_payloads(writer, source, frames)
        writer.write_frames(frames)
        writer.write_calibration(calibration)
        writer.write_meta(meta)
        # These three contract inputs must remain byte-identical.
        os.link(source / "manifest.json", writer.staging_dir / "manifest.json")
        for name in ("gnss", "heading"):
            os.link(
                source / "observations" / f"{name}.npz",
                writer.staging_dir / "observations" / f"{name}.npz",
            )
        seal_files = (
            "calibration.json",
            "frames.npz",
            "manifest.json",
            "observations/gnss.npz",
            "observations/heading.npz",
            "segment_meta.json",
        )
        seal = {
            "schema_version": 1,
            "kind": _DERIVED_SEGMENT_KIND,
            "derivation_sha256": canonical_hash(derivation),
            "payload_hardlink_count": payload_count,
            "files": {
                name: {
                    "sha256": sha256_file(writer.staging_dir / name),
                    "size_bytes": (writer.staging_dir / name).stat().st_size,
                }
                for name in seal_files
            },
        }
        seal["seal_sha256"] = canonical_hash(seal)
        _atomic_json(
            writer.staging_dir / "fixed_calibration_segment_seal.json", seal
        )
        writer.finalize(check_files=True)
    except BaseException:
        writer.abort()
        raise
    if _segment_source_records(source) != source_records_before:
        raise ArtifactError("source segment changed during calibrated derivation")
    audited_calibrated_segment(output)
    return output


def audited_calibrated_segment(
    artifact: str | Path,
    *,
    expected_source_segment: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a calibrated derived segment and its exact source delta."""
    root = Path(artifact).expanduser().resolve()
    reader = SegmentReader(root).validate()
    seal_path = root / "fixed_calibration_segment_seal.json"
    seal = _json(seal_path)
    body = dict(seal)
    digest = body.pop("seal_sha256", None)
    if (
        body.get("kind") != _DERIVED_SEGMENT_KIND
        or digest != canonical_hash(body)
    ):
        raise ArtifactError("calibrated segment seal changed")
    for name, record in body.get("files", {}).items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("size_bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise ArtifactError(f"calibrated segment file changed: {name}")
    derivation = reader.meta.get("fixed_calibration_derivation")
    if not isinstance(derivation, dict):
        raise ArtifactError("calibrated segment has no derivation record")
    if (
        derivation.get("kind") != _DERIVED_SEGMENT_KIND
        or body.get("derivation_sha256") != canonical_hash(derivation)
        or derivation.get("fixed_scale") != 1.0
        or derivation.get("raw_observations_changed") is not False
        or derivation.get("frame_inventory_changed") is not False
    ):
        raise ArtifactError("calibrated segment derivation contract changed")
    source = Path(derivation["source_segment"]).expanduser().resolve()
    if expected_source_segment is not None and source != Path(
        expected_source_segment
    ).expanduser().resolve():
        raise ArtifactError("calibrated segment source is not the expected segment")
    source_reader = SegmentReader(source).validate()
    source_records = _segment_source_records(source)
    if (
        source_records != derivation.get("source_files")
        or canonical_hash(source_records) != derivation.get("source_files_sha256")
    ):
        raise ArtifactError("calibrated segment source evidence changed")
    result = audited_raw_calibration_result(derivation["calibration_result"])
    recommendation = result["recommendation"]
    if (
        result["result_seal_sha256"]
        != derivation["calibration_result_seal_sha256"]
        or canonical_hash(recommendation) != derivation["recommendation_sha256"]
    ):
        raise ArtifactError("calibrated segment result binding changed")
    expected_calibration = _derived_calibration(
        source_reader.calibration,
        recommendation,
        Path(result["artifact"]),
        result["result_seal_sha256"],
    )
    expected_meta = _derived_meta(
        source_reader.meta, recommendation, derivation
    )
    if reader.calibration != expected_calibration or reader.meta != expected_meta:
        raise ArtifactError("calibrated segment metadata delta changed")

    source_frames = source_reader.frames
    derived_frames = reader.frames
    allowed = set(derivation["allowed_frame_changes"])
    if set(source_frames) != set(derived_frames):
        raise ArtifactError("calibrated segment frame-array inventory changed")
    for name in set(source_frames) - allowed:
        if not np.array_equal(source_frames[name], derived_frames[name]):
            raise ArtifactError(f"uncertified calibrated frame field changed: {name}")
    if not np.array_equal(source_frames["pose_valid"], derived_frames["pose_valid"]):
        raise ArtifactError("calibrated segment pose-valid inventory changed")
    for name in ("manifest.json", "observations/gnss.npz", "observations/heading.npz"):
        if sha256_file(root / name) != sha256_file(source / name):
            raise ArtifactError(f"calibrated segment changed raw contract input: {name}")

    relative_paths: set[str] = set()
    for field in ("left_image_path", "right_image_path", "depth_path"):
        if field in derived_frames:
            relative_paths.update(str(item) for item in derived_frames[field] if str(item))
    for relative in relative_paths:
        source_stat = (source / relative).stat()
        derived_stat = (root / relative).stat()
        if (source_stat.st_dev, source_stat.st_ino) != (
            derived_stat.st_dev,
            derived_stat.st_ino,
        ):
            raise ArtifactError(f"calibrated payload is not source-bound: {relative}")
    center_delta = np.linalg.norm(
        derived_frames["initial_camera_center_m"]
        - source_frames["initial_camera_center_m"],
        axis=1,
    )
    return {
        "artifact": str(root),
        "passed": True,
        "source_segment": str(source),
        "seal_sha256": sha256_file(seal_path),
        "calibration_result": result["artifact"],
        "frame_count": len(derived_frames["frame_id"]),
        "payload_hardlink_count": len(relative_paths),
        "camera_center_delta_m": {
            "median": float(np.median(center_delta)),
            "p95": float(np.percentile(center_delta, 95)),
            "maximum": float(np.max(center_delta)),
        },
        "fixed_scale": 1.0,
        "camera_to_rtk_offset_ns": int(
            recommendation["camera_to_rtk_offset_ns"]
        ),
    }
