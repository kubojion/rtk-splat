"""Sealed bounded geodetic-submap sidecars for experimental local BA.

This module materializes a database containing only a caller-sealed frame
selection and raw-GNSS-compatible pair evidence.  It delegates the actual
solve command and fixed-calibration controls to the existing
``rtk_refinement`` pose-prior-mapper implementation.  No optimizer is
implemented here and no Python Ceres or pycolmap binding is used.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import sqlite3
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rtk_splat.backends.artifact_io import (
    _atomic_json,
    _atomic_write,
    _json,
    _tree_manifest,
    _verify_tree,
)
from rtk_splat.backends.colmap_model import (
    _analyze_model,
    _cartesian_camera_priors,
    _model_candidates,
    _poses_from_images_txt,
    registered_names_from_images_txt,
)
from rtk_splat.backends.execution import (
    Runner,
    _execute,
    _next_solve_attempt,
    _run_injected_mapper,
    _run_monitored_mapper,
)
from rtk_splat.backends.geodetic_pairs import (
    GeodeticPairPolicy,
    PairCandidate,
    RawGnssEndpoint,
    decide_pair,
)
from rtk_splat.backends.mapper_config import MapperConfig
from rtk_splat.backends.quality import quality_summary, temporal_block_split
from rtk_splat.backends.rtk_refinement import (
    RtkRefinementConfig,
    _calibration_difference,
    _heldout_evaluation,
    _model_converter_command,
    _rtk_refinement_command,
    _similarity_scale,
    _source_context,
    _source_evidence,
    _stereo_baselines,
)
from rtk_splat.backends.workspace import _frontend_context
from rtk_splat.core.segment import SegmentReader, publish_directory_noreplace
from rtk_splat.frontends.artifact import (
    ArtifactError,
    FRONTEND_SEAL_FILE,
    _atomic_json as _atomic_json_noreplace,
    canonical_hash,
    sha256_file,
    sqlite_logical_record,
    verify_frontend_seal,
)


_PAIR_ID_BASE = 2_147_483_647
_SELECTION_KIND = "rtk_splat_geodetic_submap_frame_selection"
_PLAN_KIND = "rtk_splat_geodetic_submap_plan"
_RESULT_KIND = "rtk_splat_geodetic_submap_result"
_PLAN_FILES = (
    "all_images.txt",
    "calibration_prior_names.txt",
    "constant_cameras.txt",
    "constant_rigs.txt",
    "database.db",
    "database_inventory.json",
    "geodetic_submap_plan.json",
    "holdout_prior_names.txt",
    "pair_audit.json",
    "prior_split.json",
    "source_database_evidence.json",
)
_DATABASE_TABLES = (
    "cameras",
    "descriptors",
    "frame_data",
    "frames",
    "images",
    "keypoints",
    "matches",
    "pose_priors",
    "rig_sensors",
    "rigs",
    "two_view_geometries",
)
_QUALITY_WEIGHTS = (
    ("unknown_valid", 0.05),
    ("standalone", 0.10),
    ("differential", 0.25),
    ("rtk_float", 0.50),
    ("rtk_fixed", 1.00),
    ("oracle", 0.00),
)


def _fresh_refinement_config() -> RtkRefinementConfig:
    return RtkRefinementConfig(initialization_mode="fresh")


@dataclass(frozen=True)
class GeodeticSubmapConfig:
    """All generic planning and existing mapper controls for one sidecar."""

    pair_policy: GeodeticPairPolicy = field(default_factory=GeodeticPairPolicy)
    refinement: RtkRefinementConfig = field(
        default_factory=_fresh_refinement_config
    )
    position_quality_weights: tuple[tuple[str, float], ...] = _QUALITY_WEIGHTS

    def __post_init__(self) -> None:
        if self.refinement.initialization_mode != "fresh":
            raise ValueError(
                "geodetic submap BA requires fresh pose-prior mapping"
            )
        pairs = tuple(self.position_quality_weights)
        names = [name for name, _ in pairs]
        if names != [name for name, _ in _QUALITY_WEIGHTS]:
            raise ValueError(
                "position_quality_weights must cover the canonical qualities "
                "in canonical order"
            )
        for name, value in pairs:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"invalid prior weight for {name}")


def _config_record(config: GeodeticSubmapConfig) -> dict[str, Any]:
    return asdict(config)


def _config_from_record(value: Any) -> GeodeticSubmapConfig:
    if not isinstance(value, Mapping):
        raise ArtifactError("geodetic submap config must be an object")
    try:
        pair = GeodeticPairPolicy(**dict(value["pair_policy"]))
        refinement = RtkRefinementConfig(**dict(value["refinement"]))
        raw_weights = value["position_quality_weights"]
        if not isinstance(raw_weights, (list, tuple)):
            raise TypeError("position quality weights must be a sequence")
        weights = tuple((str(item[0]), float(item[1])) for item in raw_weights)
        config = GeodeticSubmapConfig(
            pair_policy=pair,
            refinement=refinement,
            position_quality_weights=weights,
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ArtifactError("invalid geodetic submap config") from exc
    # ``asdict`` retains tuples in memory while JSON loads them as lists.
    normalized = _config_record(config)
    normalized["position_quality_weights"] = [
        list(item) for item in config.position_quality_weights
    ]
    if normalized != dict(value):
        raise ArtifactError("geodetic submap config schema changed")
    return config


def _read_lines(path: Path) -> list[str]:
    try:
        values = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ArtifactError(f"cannot read {path.name}: {exc}") from exc
    if not values or any(not value for value in values):
        raise ArtifactError(f"{path.name} must contain non-empty lines")
    return values


def _write_lines(path: Path, values: Sequence[str | int]) -> None:
    strings = [str(value) for value in values]
    if not strings or len(strings) != len(set(strings)):
        raise ArtifactError(f"{path.name} must be non-empty and unique")
    _atomic_write(path, "".join(f"{value}\n" for value in strings).encode())


def _segment_binding(
    frontend: Path, segment: Path
) -> tuple[SegmentReader, dict[str, Any]]:
    reader = SegmentReader(segment).validate()
    provenance = _json(frontend / "provenance.json")
    contract = provenance.get("contract_inputs")
    if not isinstance(contract, Mapping):
        raise ArtifactError("frontend provenance has no segment contract")
    if Path(str(contract.get("segment_root", ""))).resolve() != segment:
        raise ArtifactError("supplied segment differs from the sealed frontend")
    files = contract.get("files")
    if not isinstance(files, Mapping):
        raise ArtifactError("frontend segment contract has no file evidence")
    current: dict[str, dict[str, Any]] = {}
    for relative, expected in files.items():
        path = segment / str(relative)
        if not isinstance(expected, Mapping) or not path.is_file():
            raise ArtifactError(f"sealed segment input is missing: {relative}")
        record = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if record != dict(expected):
            raise ArtifactError(f"sealed segment input changed: {relative}")
        current[str(relative)] = record
    return reader, {
        "segment_root": str(segment),
        "contract_inputs_sha256": canonical_hash(contract),
        "files": current,
    }


def _input_context(
    frontend_artifact: str | Path,
    completed_backend: str | Path,
    segment: str | Path,
) -> tuple[
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    MapperConfig,
    dict[str, Any],
    Path,
    SegmentReader,
    dict[str, Any],
]:
    frontend, manifest, _ = _frontend_context(frontend_artifact)
    seal = verify_frontend_seal(frontend)
    source, source_plan, mapper_config, source_quality = _source_context(
        completed_backend
    )
    if Path(str(source_plan["frontend_artifact"])).resolve() != frontend:
        raise ArtifactError("completed model was not built from this frontend")
    segment_path = Path(segment).expanduser().resolve()
    reader, segment_evidence = _segment_binding(frontend, segment_path)
    return (
        frontend,
        manifest,
        source,
        source_plan,
        mapper_config,
        source_quality,
        segment_path,
        reader,
        {
            "frontend_seal_sha256": sha256_file(
                frontend / FRONTEND_SEAL_FILE
            ),
            "frontend_database_raw_sha256": seal["database"]["raw_sha256"],
            "frontend_database_committed_sha256": seal["database"][
                "committed_view"
            ]["sha256"],
            "source_backend": _source_evidence(source, source_plan),
            "segment": segment_evidence,
        },
    )


def _selection_payload(
    frontend: Path,
    segment: Path,
    frame_ids: Sequence[int],
) -> dict[str, Any]:
    values = list(frame_ids)
    body = {
        "schema_version": 1,
        "kind": _SELECTION_KIND,
        "frontend_artifact": str(frontend),
        "frontend_seal_sha256": sha256_file(frontend / FRONTEND_SEAL_FILE),
        "segment": str(segment),
        "segment_frames_sha256": sha256_file(segment / "frames.npz"),
        "frame_ids": values,
        "frame_ids_sha256": canonical_hash(values),
    }
    return {**body, "selection_sha256": canonical_hash(body)}


def create_geodetic_frame_selection(
    frontend_artifact: str | Path,
    segment: str | Path,
    frame_ids: Sequence[int],
    destination: str | Path,
) -> Path:
    """Atomically seal a sorted, unique frame list against immutable inputs."""
    frontend, manifest, _ = _frontend_context(frontend_artifact)
    segment_path = Path(segment).expanduser().resolve()
    reader, _ = _segment_binding(frontend, segment_path)
    values = list(frame_ids)
    if (
        not values
        or any(type(value) is not int for value in values)
        or values != sorted(set(values))
    ):
        raise ArtifactError("selected frame IDs must be sorted, unique integers")
    manifest_ids = [int(row["frame_id"]) for row in manifest["frames"]]
    segment_ids = reader.frames["frame_id"].astype(int).tolist()
    if not set(values) <= set(manifest_ids) or manifest_ids != segment_ids:
        raise ArtifactError("selected frame IDs are outside the bound inputs")
    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frame selection: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_noreplace(
        output, _selection_payload(frontend, segment_path, values)
    )
    return output


def _load_frame_selection(
    selection: str | Path,
    frontend: Path,
    segment: Path,
    manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path = Path(selection).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ArtifactError("sealed frame selection is missing or unsafe")
    value = _json(path)
    frame_ids = value.get("frame_ids")
    if (
        value.get("kind") != _SELECTION_KIND
        or not isinstance(frame_ids, list)
        or not frame_ids
        or any(type(item) is not int for item in frame_ids)
        or frame_ids != sorted(set(frame_ids))
    ):
        raise ArtifactError("invalid sealed geodetic frame selection")
    expected = _selection_payload(frontend, segment, frame_ids)
    if value != expected:
        raise ArtifactError("geodetic frame selection seal verification failed")
    available = {int(row["frame_id"]) for row in manifest["frames"]}
    if not set(frame_ids) <= available:
        raise ArtifactError("frame selection references an unknown frame")
    return path, value


def _raw_gnss_endpoints(reader: SegmentReader) -> dict[int, RawGnssEndpoint]:
    frames = reader.frames
    gnss = reader.observations("gnss")
    assert gnss is not None
    association = gnss.get(
        "association_valid", np.ones(len(frames["frame_id"]), dtype=bool)
    )
    result: dict[int, RawGnssEndpoint] = {}
    for index, frame_id in enumerate(frames["frame_id"].astype(int)):
        valid = bool(gnss["position_valid"][index]) and bool(
            association[index]
        )
        position = np.asarray(gnss["enu_m"][index], dtype=np.float64)
        covariance = np.asarray(
            gnss["covariance_enu_m2"][index], dtype=np.float64
        )
        result[int(frame_id)] = RawGnssEndpoint(
            position_m=(
                tuple(float(value) for value in position)
                if np.isfinite(position).all()
                else None
            ),
            covariance_m2=(
                tuple(
                    tuple(float(value) for value in row)
                    for row in covariance
                )
                if np.isfinite(covariance).all()
                else None
            ),
            position_valid=valid,
            position_quality=str(gnss["position_quality"][index]),
            fix_status=int(gnss["fix_status"][index]),
            carrier_status=int(gnss["carrier_status"][index]),
        )
    return result


def _selected_rows(
    manifest: Mapping[str, Any], frame_ids: Sequence[int]
) -> list[dict[str, Any]]:
    selected = set(frame_ids)
    rows = [dict(row) for row in manifest["frames"] if row["frame_id"] in selected]
    if [int(row["frame_id"]) for row in rows] != list(frame_ids):
        raise ArtifactError("frame selection order differs from the frontend")
    return rows


def _image_metadata(
    rows: Sequence[Mapping[str, Any]],
    all_manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    indices = {
        int(row["frame_id"]): index
        for index, row in enumerate(all_manifest_rows)
    }
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        frame_id = int(row["frame_id"])
        for camera, field_name in (
            ("left", "left_image"),
            ("right", "right_image"),
        ):
            name = str(row[field_name]["name"])
            result[name] = {
                "frame_id": frame_id,
                "frame_index": indices[frame_id],
                "camera": camera,
                "timestamp_ns": int(row["timestamp_ns"]),
            }
    return result


def _readonly_database(
    database: Path, *, immutable: bool = False
) -> sqlite3.Connection:
    try:
        uri = f"file:{database.resolve()}?mode=ro"
        if immutable:
            uri += "&immutable=1"
        connection = sqlite3.connect(
            uri, uri=True
        )
        connection.execute("PRAGMA query_only=ON")
        return connection
    except sqlite3.Error as exc:
        raise ArtifactError(f"cannot open immutable COLMAP database: {exc}") from exc


def _sqlite_metadata(
    connection: sqlite3.Connection, schema: str = "main"
) -> dict[str, Any]:
    if schema not in {"main", "source_db"}:
        raise ValueError("unsupported SQLite schema name")
    try:
        user_version = int(
            connection.execute(f"PRAGMA {schema}.user_version").fetchone()[0]
        )
        application_id = int(
            connection.execute(f"PRAGMA {schema}.application_id").fetchone()[0]
        )
        journal_mode = str(
            connection.execute(f"PRAGMA {schema}.journal_mode").fetchone()[0]
        ).lower()
    except (sqlite3.Error, TypeError, ValueError, IndexError) as exc:
        raise ArtifactError("cannot read COLMAP SQLite metadata") from exc
    return {
        "user_version": user_version,
        "application_id": application_id,
        "journal_mode": journal_mode,
    }


def _database_sqlite_metadata(database: Path) -> dict[str, Any]:
    """Read stable SQLite header metadata without opening the sealed database."""
    try:
        with database.open("rb") as stream:
            header = stream.read(100)
    except OSError as exc:
        raise ArtifactError("cannot read COLMAP SQLite metadata") from exc
    if len(header) != 100 or header[:16] != b"SQLite format 3\x00":
        raise ArtifactError("private COLMAP database has an invalid SQLite header")
    write_version, read_version = header[18], header[19]
    if (write_version, read_version) == (1, 1):
        journal_mode = "delete"
    elif (write_version, read_version) == (2, 2):
        journal_mode = "wal"
    else:
        raise ArtifactError("private COLMAP database has mixed journal metadata")
    return {
        "user_version": int.from_bytes(header[60:64], "big"),
        "application_id": int.from_bytes(header[68:72], "big"),
        "journal_mode": journal_mode,
    }


def _normalize_private_database_metadata(
    database: Path, source_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Make the private copy metadata-stable before the pinned COLMAP opens it."""
    try:
        user_version = int(source_metadata["user_version"])
        application_id = int(source_metadata["application_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError("source COLMAP SQLite metadata is invalid") from exc
    if user_version <= 0:
        raise ArtifactError(
            "source COLMAP database has no positive format version marker"
        )
    try:
        with sqlite3.connect(database) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            journal_mode = str(
                connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            ).lower()
            if journal_mode != "wal":
                raise ArtifactError("private COLMAP database did not enter WAL mode")
            connection.execute(f"PRAGMA user_version={user_version}")
            connection.execute(f"PRAGMA application_id={application_id}")
            connection.commit()
            checkpoint = tuple(
                int(value)
                for value in connection.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                ).fetchone()
            )
            if len(checkpoint) != 3 or checkpoint[0] != 0:
                raise ArtifactError(
                    "private COLMAP database WAL checkpoint did not complete"
                )
            integrity = [
                str(row[0])
                for row in connection.execute("PRAGMA integrity_check")
            ]
            if integrity != ["ok"]:
                raise ArtifactError(
                    "private COLMAP database integrity failed after metadata "
                    "normalization"
                )
            metadata = _sqlite_metadata(connection)
    except (sqlite3.Error, OSError) as exc:
        raise ArtifactError(
            f"cannot normalize private COLMAP database metadata: {exc}"
        ) from exc
    expected = {
        "user_version": user_version,
        "application_id": application_id,
        "journal_mode": "wal",
    }
    if metadata != expected:
        raise ArtifactError("private COLMAP SQLite metadata normalization failed")
    # A successful TRUNCATE checkpoint makes empty sidecars disposable. Keeping
    # them out of the sealed plan also makes copy and inventory semantics exact.
    _remove_checkpointed_database_sidecars(database)
    return metadata


def _remove_checkpointed_database_sidecars(database: Path) -> None:
    wal = database.with_name(database.name + "-wal")
    shm = database.with_name(database.name + "-shm")
    if wal.exists() and wal.stat().st_size:
        raise ArtifactError(
            "private COLMAP database has an uncheckpointed WAL before sealing"
        )
    wal.unlink(missing_ok=True)
    shm.unlink(missing_ok=True)


def _pair_candidates(
    database: Path,
    metadata: Mapping[str, Mapping[str, Any]],
    gnss: Mapping[int, RawGnssEndpoint],
) -> tuple[list[PairCandidate], dict[str, Any]]:
    try:
        with _readonly_database(database) as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table'"
                )
            }
            if not {"images", "matches", "two_view_geometries"} <= tables:
                raise ArtifactError("COLMAP database lacks pair evidence tables")
            images = {
                int(image_id): str(name)
                for image_id, name in connection.execute(
                    "SELECT image_id, name FROM images"
                )
                if str(name) in metadata
            }
            matches = {
                int(pair_id): int(rows)
                for pair_id, rows in connection.execute(
                    "SELECT pair_id, rows FROM matches"
                )
            }
            verified = {
                int(pair_id): int(rows)
                for pair_id, rows in connection.execute(
                    "SELECT pair_id, rows FROM two_view_geometries"
                )
            }
    except sqlite3.Error as exc:
        raise ArtifactError(f"cannot enumerate COLMAP pair evidence: {exc}") from exc
    candidates: list[PairCandidate] = []
    for pair_id in sorted(set(matches) | set(verified)):
        first_id, second_id = divmod(pair_id, _PAIR_ID_BASE)
        if first_id not in images or second_id not in images:
            continue
        first_name, second_name = images[first_id], images[second_id]
        first, second = metadata[first_name], metadata[second_name]
        candidates.append(
            PairCandidate(
                pair_id=pair_id,
                first_image_id=first_id,
                second_image_id=second_id,
                first_image_name=first_name,
                second_image_name=second_name,
                first_frame_id=int(first["frame_id"]),
                second_frame_id=int(second["frame_id"]),
                first_frame_index=int(first["frame_index"]),
                second_frame_index=int(second["frame_index"]),
                first_camera=str(first["camera"]),
                second_camera=str(second["camera"]),
                first_timestamp_ns=int(first["timestamp_ns"]),
                second_timestamp_ns=int(second["timestamp_ns"]),
                raw_matches=matches.get(pair_id),
                verified_matches=verified.get(pair_id),
                first_gnss=gnss[int(first["frame_id"])],
                second_gnss=gnss[int(second["frame_id"])],
            )
        )
    if not candidates:
        raise ArtifactError("selected frames have no existing pair evidence")
    selected_pair_ids = {candidate.pair_id for candidate in candidates}
    return candidates, {
        "source_match_pair_ids": sorted(
            pair_id for pair_id in matches if pair_id in selected_pair_ids
        ),
        "source_geometry_pair_ids": sorted(
            pair_id for pair_id in verified if pair_id in selected_pair_ids
        ),
    }


def _pair_audit(
    candidates: Sequence[PairCandidate], policy: GeodeticPairPolicy
) -> tuple[dict[str, Any], set[int]]:
    records = []
    retained: set[int] = set()
    for candidate in candidates:
        decision = decide_pair(candidate, policy)
        records.append(decision.record(candidate))
        if decision.retained:
            retained.add(candidate.pair_id)
    if not retained:
        raise ArtifactError("raw-GNSS pair policy rejected every selected pair")
    return (
        {
            "schema_version": 1,
            "method": "raw_gnss_covariance_physical_gate_v1",
            "policy": asdict(policy),
            "pair_count": len(records),
            "retained_count": len(retained),
            "rejected_count": len(records) - len(retained),
            "retained_pair_ids_sha256": canonical_hash(sorted(retained)),
            "decision_inputs_exclude": [
                "finished_visual_model_residual",
                "heldout_evaluation_result",
            ],
            "records": records,
        },
        retained,
    )


def _quoted(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _insert_temp_ids(
    connection: sqlite3.Connection, table: str, values: Sequence[int]
) -> None:
    connection.execute(
        f"CREATE TEMP TABLE {_quoted(table)}(value INTEGER PRIMARY KEY)"
    )
    connection.executemany(
        f"INSERT INTO {_quoted(table)}(value) VALUES (?)",
        ((int(value),) for value in values),
    )


def _schema_rows(
    connection: sqlite3.Connection,
) -> tuple[set[str], dict[str, str], list[str]]:
    rows = list(
        connection.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM source_db.sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        )
    )
    tables = {
        str(name)
        for kind, name, _, _ in rows
        if str(kind) == "table"
    }
    missing = sorted(set(_DATABASE_TABLES) - tables)
    unexpected = sorted(tables - set(_DATABASE_TABLES))
    if missing:
        raise ArtifactError(
            "COLMAP database lacks required tables: " + ", ".join(missing)
        )
    if unexpected:
        raise ArtifactError(
            "COLMAP database has unsupported tables that cannot be safely "
            "subsetted: " + ", ".join(unexpected)
        )
    selected = set(_DATABASE_TABLES)
    table_sql = {
        str(name): str(sql)
        for kind, name, _, sql in rows
        if str(kind) == "table" and str(name) in selected and sql is not None
    }
    auxiliary_sql = [
        str(sql)
        for kind, _, table, sql in rows
        if str(kind) in {"index", "trigger"}
        and str(table) in selected
        and sql is not None
    ]
    return selected, table_sql, auxiliary_sql


def _source_assignments(
    database: Path,
    selected_names: Sequence[str],
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    names = set(selected_names)
    try:
        with _readonly_database(database) as connection:
            image_rows = [
                (int(image_id), str(name), int(camera_id))
                for image_id, name, camera_id in connection.execute(
                    "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
                )
                if str(name) in names
            ]
            if {name for _, name, _ in image_rows} != names:
                raise ArtifactError("selected frontend images are absent from SQLite")
            image_ids = {image_id for image_id, _, _ in image_rows}
            frame_rows = [
                tuple(int(value) for value in row)
                for row in connection.execute(
                    """
                    SELECT fd.frame_id, fd.data_id, fd.sensor_id,
                           fd.sensor_type, f.rig_id
                    FROM frame_data AS fd
                    JOIN frames AS f ON f.frame_id=fd.frame_id
                    ORDER BY fd.frame_id, fd.data_id
                    """
                )
                if int(row[1]) in image_ids
            ]
            rig_rows = [
                tuple(int(value) for value in row)
                for row in connection.execute(
                    "SELECT rig_id, ref_sensor_id, ref_sensor_type "
                    "FROM rigs ORDER BY rig_id"
                )
            ]
            rig_sensor_rows = list(
                connection.execute(
                    "SELECT rig_id, sensor_id, sensor_type, sensor_from_rig "
                    "FROM rig_sensors ORDER BY rig_id, sensor_id, sensor_type"
                )
            )
    except sqlite3.Error as exc:
        raise ArtifactError(f"cannot read source rig assignments: {exc}") from exc
    if len(frame_rows) != len(image_rows):
        raise ArtifactError("selected images lack exact rig frame assignments")
    by_data = {row[1]: row for row in frame_rows}
    images_by_name = {
        name: (image_id, camera_id)
        for image_id, name, camera_id in image_rows
    }
    side_cameras: dict[str, set[int]] = {"left": set(), "right": set()}
    contract_to_database_frame: dict[int, int] = {}
    database_frame_counts: dict[int, int] = {}
    for name, item in metadata.items():
        image_id, camera_id = images_by_name[name]
        assignment = by_data.get(image_id)
        if (
            assignment is None
            or assignment[2] != camera_id
            or assignment[3] != 0
        ):
            raise ArtifactError("image/camera rig sensor assignment changed")
        side = str(item["camera"])
        if side not in side_cameras:
            raise ArtifactError("geodetic submap supports one stereo pair only")
        side_cameras[side].add(camera_id)
        contract_frame = int(item["frame_id"])
        previous = contract_to_database_frame.setdefault(
            contract_frame, assignment[0]
        )
        if previous != assignment[0]:
            raise ArtifactError("stereo images are assigned to different frames")
        database_frame_counts[assignment[0]] = (
            database_frame_counts.get(assignment[0], 0) + 1
        )
    if set(by_data) != image_ids or any(row[3] != 0 for row in frame_rows):
        raise ArtifactError("selected images have unsupported sensor assignments")
    frame_ids = sorted({row[0] for row in frame_rows})
    rig_ids = sorted({row[4] for row in frame_rows})
    camera_ids = sorted({row[2] for row in image_rows})
    if (
        len(frame_ids) * 2 != len(image_rows)
        or len(rig_ids) != 1
        or any(count != 2 for count in database_frame_counts.values())
        or any(len(values) != 1 for values in side_cameras.values())
        or side_cameras["left"] == side_cameras["right"]
        or len(camera_ids) != 2
    ):
        raise ArtifactError("geodetic submap requires one complete stereo rig")
    left_camera = next(iter(side_cameras["left"]))
    right_camera = next(iter(side_cameras["right"]))
    used_rig_rows = [row for row in rig_rows if row[0] in rig_ids]
    used_sensor_rows = [row for row in rig_sensor_rows if int(row[0]) in rig_ids]
    if (
        used_rig_rows != [(rig_ids[0], left_camera, 0)]
        or len(used_sensor_rows) != 1
        or tuple(int(value) for value in used_sensor_rows[0][:3])
        != (rig_ids[0], right_camera, 0)
        or used_sensor_rows[0][3] is None
    ):
        raise ArtifactError("database stereo rig calibration is incomplete")
    return {
        "image_rows": image_rows,
        "image_ids": sorted(image_ids),
        "camera_ids": camera_ids,
        "frame_rows": frame_rows,
        "database_frame_ids": frame_ids,
        "rig_ids": rig_ids,
        "side_camera_ids": {
            "left": left_camera,
            "right": right_camera,
        },
        "rig_rows": [list(row) for row in used_rig_rows],
    }


def _copy_subset_database(
    source: Path,
    destination: Path,
    assignments: Mapping[str, Any],
    retained_pair_ids: set[int],
    selected_left_image_ids: set[int],
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite private database: {destination}")
    source_raw_before = sha256_file(source)
    source_uri = f"{source.resolve().as_uri()}?mode=ro"
    connection: sqlite3.Connection | None = None
    source_sqlite_metadata: dict[str, Any] | None = None
    succeeded = False
    try:
        connection = sqlite3.connect(destination, uri=True)
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("ATTACH DATABASE ? AS source_db", (source_uri,))
        source_sqlite_metadata = _sqlite_metadata(connection, "source_db")
        if source_sqlite_metadata["user_version"] <= 0:
            raise ArtifactError(
                "source COLMAP database has no positive format version marker"
            )
        tables, table_sql, auxiliary_sql = _schema_rows(connection)
        connection.execute("BEGIN")
        for name in _DATABASE_TABLES:
            connection.execute(table_sql[name])
        _insert_temp_ids(connection, "keep_images", assignments["image_ids"])
        _insert_temp_ids(connection, "keep_cameras", assignments["camera_ids"])
        _insert_temp_ids(
            connection, "keep_frames", assignments["database_frame_ids"]
        )
        _insert_temp_ids(connection, "keep_rigs", assignments["rig_ids"])
        _insert_temp_ids(connection, "keep_pairs", sorted(retained_pair_ids))
        _insert_temp_ids(
            connection, "keep_prior_images", sorted(selected_left_image_ids)
        )

        predicates = {
            "cameras": "camera_id IN (SELECT value FROM keep_cameras)",
            "images": "image_id IN (SELECT value FROM keep_images)",
            "keypoints": "image_id IN (SELECT value FROM keep_images)",
            "descriptors": "image_id IN (SELECT value FROM keep_images)",
            "rigs": "rig_id IN (SELECT value FROM keep_rigs)",
            "frames": "frame_id IN (SELECT value FROM keep_frames)",
            "frame_data": (
                "frame_id IN (SELECT value FROM keep_frames) AND "
                "data_id IN (SELECT value FROM keep_images)"
            ),
            "matches": "pair_id IN (SELECT value FROM keep_pairs)",
            "two_view_geometries": "pair_id IN (SELECT value FROM keep_pairs)",
            "pose_priors": (
                "corr_sensor_type=0 AND corr_data_id IN "
                "(SELECT value FROM keep_prior_images)"
            ),
            "rig_sensors": "rig_id IN (SELECT value FROM keep_rigs)",
        }
        for table in (
            "cameras",
            "rigs",
            "rig_sensors",
            "frames",
            "images",
            "frame_data",
            "keypoints",
            "descriptors",
            "matches",
            "two_view_geometries",
            "pose_priors",
        ):
            if table not in tables:
                continue
            connection.execute(
                f"INSERT INTO main.{_quoted(table)} SELECT * FROM "
                f"source_db.{_quoted(table)} WHERE {predicates[table]}"
            )
        for sql in auxiliary_sql:
            connection.execute(sql)
        violations = list(connection.execute("PRAGMA foreign_key_check"))
        if violations:
            raise ArtifactError(
                f"private COLMAP database has foreign-key violations: "
                f"{violations[:4]}"
            )
        connection.commit()
        connection.execute("DETACH DATABASE source_db")
        connection.execute("PRAGMA foreign_keys=ON")
        integrity = [
            str(row[0]) for row in connection.execute("PRAGMA integrity_check")
        ]
        if integrity != ["ok"]:
            raise ArtifactError(
                "private COLMAP database integrity failed: "
                + "; ".join(integrity)
            )
        connection.close()
        connection = None
        succeeded = True
    except (sqlite3.Error, OSError) as exc:
        raise ArtifactError(f"cannot materialize private COLMAP database: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
        if not succeeded:
            destination.unlink(missing_ok=True)
            destination.with_name(destination.name + "-wal").unlink(
                missing_ok=True
            )
            destination.with_name(destination.name + "-shm").unlink(
                missing_ok=True
            )
    with destination.open("rb") as stream:
        os.fsync(stream.fileno())
    source_raw_after = sha256_file(source)
    if source_raw_before != source_raw_after:
        destination.unlink(missing_ok=True)
        raise ArtifactError("source SQLite bytes changed during submap copy")
    if source_sqlite_metadata is None:
        raise ArtifactError("source COLMAP SQLite metadata was not captured")
    return {
        "schema_version": 1,
        "method": "read_only_attached_selective_copy_v1",
        "source_database": str(source),
        "source_raw_sha256_before": source_raw_before,
        "source_raw_sha256_after": source_raw_after,
        "source_bytes_unchanged": True,
        "source_sqlite_metadata": source_sqlite_metadata,
        "selected_tables": sorted(tables),
    }


def _decode_vector(blob: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise ArtifactError(f"{label} is not a binary float64 array")
    values = np.frombuffer(bytes(blob), dtype="<f8")
    if values.size != int(np.prod(shape)):
        raise ArtifactError(f"{label} has the wrong element count")
    order = "F" if shape == (3, 3) else "C"
    result = values.reshape(shape, order=order).copy()
    if not np.isfinite(result).all():
        raise ArtifactError(f"{label} is not finite")
    return result


def _weight_and_split_priors(
    database: Path,
    rows: Sequence[Mapping[str, Any]],
    reader: SegmentReader,
    config: GeodeticSubmapConfig,
    temporal_blocks: int,
) -> dict[str, Any]:
    gnss = _raw_gnss_endpoints(reader)
    weights = dict(config.position_quality_weights)
    frames = reader.frames
    centers = (
        {
            int(frame_id): np.asarray(center, dtype=np.float64)
            for frame_id, center in zip(
                frames["frame_id"], frames["initial_camera_center_m"], strict=True
            )
        }
        if "initial_camera_center_m" in frames
        else {}
    )
    by_left = {
        str(row["left_image"]["name"]): (
            int(row["frame_id"]),
            int(row["timestamp_ns"]),
        )
        for row in rows
    }
    try:
        with sqlite3.connect(database) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            source_rows = list(
                connection.execute(
                    """
                    SELECT p.pose_prior_id, p.corr_data_id, p.corr_sensor_id,
                           p.corr_sensor_type, p.position,
                           p.position_covariance, p.gravity,
                           p.coordinate_system, i.name, i.camera_id
                    FROM pose_priors AS p
                    JOIN images AS i ON i.image_id=p.corr_data_id
                    ORDER BY i.image_id
                    """
                )
            )
            records: list[dict[str, Any]] = []
            eligible: list[dict[str, Any]] = []
            remove_ids: list[int] = []
            for row in source_rows:
                (
                    prior_id,
                    image_id,
                    sensor_id,
                    sensor_type,
                    position_blob,
                    covariance_blob,
                    gravity,
                    coordinate_system,
                    name,
                    camera_id,
                ) = row
                name = str(name)
                if name not in by_left:
                    raise ArtifactError("private DB contains a non-left pose prior")
                frame_id, timestamp_ns = by_left[name]
                endpoint = gnss[frame_id]
                weight = float(weights.get(endpoint.position_quality, 0.0))
                base = {
                    "name": name,
                    "frame_id": frame_id,
                    "timestamp_ns": timestamp_ns,
                    "position_quality": endpoint.position_quality,
                    "fix_status": endpoint.fix_status,
                    "carrier_status": endpoint.carrier_status,
                    "status_weight": weight,
                }
                if not endpoint.position_valid or weight <= 0.0:
                    remove_ids.append(int(prior_id))
                    records.append(
                        {**base, "role": "excluded", "reason": "raw_gnss_status"}
                    )
                    continue
                if (
                    int(sensor_type) != 0
                    or int(sensor_id) != int(camera_id)
                    or int(coordinate_system) != 1
                    or gravity is not None
                ):
                    raise ArtifactError(
                        "geodetic submap priors must be Cartesian camera-position "
                        "rows without gravity/IMU data"
                    )
                position = _decode_vector(position_blob, (3,), f"prior {name} position")
                covariance = _decode_vector(
                    covariance_blob, (3, 3), f"prior {name} covariance"
                )
                covariance = 0.5 * (covariance + covariance.T)
                try:
                    np.linalg.cholesky(covariance)
                except np.linalg.LinAlgError as exc:
                    raise ArtifactError(
                        f"pose prior covariance is not positive definite: {name}"
                    ) from exc
                if frame_id not in centers or not np.allclose(
                    position, centers[frame_id], atol=1e-6, rtol=1e-9
                ):
                    raise ArtifactError(
                        "pose prior is not the segment's lever-arm-applied raw-GNSS "
                        f"camera centre: {name}"
                    )
                weighted = covariance / weight
                eligible.append(
                    {
                        **base,
                        "prior_id": int(prior_id),
                        "image_id": int(image_id),
                        "position_m": position.tolist(),
                        "source_covariance_m2": covariance.tolist(),
                        "optimizer_covariance_m2": weighted.tolist(),
                        "optimizer_covariance_blob": np.asarray(
                            weighted, dtype="<f8"
                        ).tobytes(order="F"),
                    }
                )
            eligible.sort(key=lambda item: item["timestamp_ns"])
            if len(eligible) < 4:
                raise ArtifactError(
                    "geodetic submap needs at least four trusted position priors"
                )
            split = temporal_block_split(
                np.asarray(
                    [item["timestamp_ns"] for item in eligible], dtype=np.int64
                ),
                temporal_blocks=temporal_blocks,
            )
            calibration_names: list[str] = []
            holdout_names: list[str] = []
            for index, item in enumerate(eligible):
                calibration = bool(split.calibration_mask[index])
                role = "calibration" if calibration else "holdout"
                item["role"] = role
                item["block_id"] = int(split.block_ids[index])
                records.append(
                    {key: value for key, value in item.items() if not key.endswith("blob")}
                )
                if calibration:
                    calibration_names.append(str(item["name"]))
                    connection.execute(
                        "UPDATE pose_priors SET position_covariance=? "
                        "WHERE pose_prior_id=?",
                        (item["optimizer_covariance_blob"], item["prior_id"]),
                    )
                else:
                    holdout_names.append(str(item["name"]))
                    remove_ids.append(int(item["prior_id"]))
            connection.executemany(
                "DELETE FROM pose_priors WHERE pose_prior_id=?",
                ((value,) for value in remove_ids),
            )
            remaining = [
                str(name)
                for (name,) in connection.execute(
                    """
                    SELECT i.name FROM pose_priors AS p
                    JOIN images AS i ON i.image_id=p.corr_data_id
                    ORDER BY i.image_id
                    """
                )
            ]
            connection.commit()
    except sqlite3.Error as exc:
        raise ArtifactError(f"cannot seal geodetic prior split: {exc}") from exc
    if set(remaining) != set(calibration_names) or set(remaining) & set(
        holdout_names
    ):
        raise ArtifactError("held-out position priors remain in the optimizer DB")
    return {
        "schema_version": 1,
        "method": "alternating_temporal_blocks_status_covariance_weighted_v1",
        "calibration_block_ids": list(split.calibration_block_ids),
        "holdout_block_ids": list(split.holdout_block_ids),
        "calibration_names": calibration_names,
        "holdout_names": holdout_names,
        "n_calibration": len(calibration_names),
        "n_holdout": len(holdout_names),
        "n_excluded": sum(item["role"] == "excluded" for item in records),
        "position_quality_weights": [
            list(item) for item in config.position_quality_weights
        ],
        "optimizer_uses": [
            "raw_gnss_derived_camera_position",
            "source_covariance",
            "position_quality_status_weight",
        ],
        "optimizer_excludes": [
            "heldout_position_priors",
            "imu",
            "gravity",
            "lever_arm_estimation",
        ],
        "records": sorted(records, key=lambda item: item["timestamp_ns"]),
    }


def _digest_sql_value(digest: Any, value: Any) -> None:
    if value is None:
        digest.update(b"N\0")
    elif isinstance(value, bytes):
        digest.update(b"B")
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value)
    elif isinstance(value, int):
        digest.update(b"I")
        digest.update(str(value).encode())
        digest.update(b"\0")
    elif isinstance(value, float):
        digest.update(b"F")
        digest.update(repr(value).encode())
        digest.update(b"\0")
    else:
        payload = str(value).encode("utf-8", errors="surrogateescape")
        digest.update(b"T")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)


def _table_rows_digest(
    database: Path,
    table: str,
    where: str = "",
    parameters: Sequence[Any] = (),
    *,
    immutable: bool = False,
) -> dict[str, Any]:
    if table not in set(_DATABASE_TABLES):
        raise ValueError("unsupported COLMAP table digest")
    query = f"SELECT * FROM {_quoted(table)}"
    if where:
        query += f" WHERE {where}"
    query += " ORDER BY rowid"
    digest = hashlib.sha256()
    count = 0
    try:
        with _readonly_database(database, immutable=immutable) as connection:
            for row in connection.execute(query, tuple(parameters)):
                count += 1
                digest.update(b"R\0")
                for value in row:
                    _digest_sql_value(digest, value)
                    digest.update(b"\0")
    except sqlite3.Error as exc:
        raise ArtifactError(f"cannot hash {table} rows: {exc}") from exc
    return {"rows": count, "sha256": digest.hexdigest()}


def _pair_table_digest(
    database: Path, table: str, pair_ids: Sequence[int]
) -> dict[str, Any]:
    if not pair_ids:
        return {"rows": 0, "sha256": hashlib.sha256().hexdigest()}
    records: list[tuple[Any, ...]] = []
    requested = set(pair_ids)
    try:
        with _readonly_database(database) as connection:
            records = [
                tuple(row)
                for row in connection.execute(
                    f"SELECT * FROM {_quoted(table)} ORDER BY pair_id"
                )
                if int(row[0]) in requested
            ]
    except sqlite3.Error as exc:
        raise ArtifactError(f"cannot hash retained {table}: {exc}") from exc
    digest = hashlib.sha256()
    for row in records:
        digest.update(b"R\0")
        for value in row:
            _digest_sql_value(digest, value)
            digest.update(b"\0")
    return {"rows": len(records), "sha256": digest.hexdigest()}


def _database_inventory(
    database: Path,
    source_database: Path,
    assignments: Mapping[str, Any],
    selected_names: Sequence[str],
    retained_pair_ids: set[int],
    pair_sources: Mapping[str, Any],
    calibration_names: Sequence[str],
) -> dict[str, Any]:
    try:
        with _readonly_database(database) as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table'"
                )
            }
            images = [
                (int(image_id), str(name), int(camera_id))
                for image_id, name, camera_id in connection.execute(
                    "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
                )
            ]
            cameras = [
                int(row[0])
                for row in connection.execute(
                    "SELECT camera_id FROM cameras ORDER BY camera_id"
                )
            ]
            rigs = [
                int(row[0])
                for row in connection.execute(
                    "SELECT rig_id FROM rigs ORDER BY rig_id"
                )
            ]
            frames = [
                int(row[0])
                for row in connection.execute(
                    "SELECT frame_id FROM frames ORDER BY frame_id"
                )
            ]
            frame_rows = [
                tuple(int(value) for value in row)
                for row in connection.execute(
                    """
                    SELECT fd.frame_id, fd.data_id, fd.sensor_id,
                           fd.sensor_type, f.rig_id
                    FROM frame_data AS fd
                    JOIN frames AS f ON f.frame_id=fd.frame_id
                    ORDER BY fd.frame_id, fd.data_id
                    """
                )
            ]
            keypoint_ids = [
                int(row[0])
                for row in connection.execute(
                    "SELECT image_id FROM keypoints ORDER BY image_id"
                )
            ]
            descriptor_ids = [
                int(row[0])
                for row in connection.execute(
                    "SELECT image_id FROM descriptors ORDER BY image_id"
                )
            ]
            match_ids = [
                int(row[0])
                for row in connection.execute(
                    "SELECT pair_id FROM matches ORDER BY pair_id"
                )
            ]
            geometry_ids = [
                int(row[0])
                for row in connection.execute(
                    "SELECT pair_id FROM two_view_geometries ORDER BY pair_id"
                )
            ]
            prior_names = [
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT i.name FROM pose_priors AS p
                    JOIN images AS i ON i.image_id=p.corr_data_id
                    ORDER BY i.image_id
                    """
                )
            ]
            gravity_rows = int(
                connection.execute(
                    "SELECT COUNT(*) FROM pose_priors WHERE gravity IS NOT NULL"
                ).fetchone()[0]
            )
    except sqlite3.Error as exc:
        raise ArtifactError(f"cannot inventory private database: {exc}") from exc

    expected_images = list(assignments["image_rows"])
    expected_ids = list(assignments["image_ids"])
    expected_matches = sorted(
        retained_pair_ids & set(pair_sources["source_match_pair_ids"])
    )
    expected_geometries = sorted(
        retained_pair_ids & set(pair_sources["source_geometry_pair_ids"])
    )
    if (
        images != expected_images
        or [name for _, name, _ in images] != [
            name for _, name, _ in expected_images
        ]
        or set(selected_names) != {name for _, name, _ in images}
        or cameras != list(assignments["camera_ids"])
        or rigs != list(assignments["rig_ids"])
        or frames != list(assignments["database_frame_ids"])
        or frame_rows != list(assignments["frame_rows"])
        or keypoint_ids != expected_ids
        or descriptor_ids != expected_ids
        or match_ids != expected_matches
        or geometry_ids != expected_geometries
        or set(prior_names) != set(calibration_names)
        or gravity_rows != 0
    ):
        raise ArtifactError("private database selected inventory is not exact")

    source_pair_digests = {
        table: _pair_table_digest(source_database, table, values)
        for table, values in (
            ("matches", expected_matches),
            ("two_view_geometries", expected_geometries),
        )
    }
    private_pair_digests = {
        table: _pair_table_digest(database, table, values)
        for table, values in (
            ("matches", expected_matches),
            ("two_view_geometries", expected_geometries),
        )
    }
    if source_pair_digests != private_pair_digests:
        raise ArtifactError("retained pair evidence changed during materialization")
    return {
        "schema_version": 1,
        "tables": sorted(tables),
        "images": [
            {"image_id": image_id, "name": name, "camera_id": camera_id}
            for image_id, name, camera_id in images
        ],
        "camera_ids": cameras,
        "rig_ids": rigs,
        "database_frame_ids": frames,
        "frame_assignments": [list(row) for row in frame_rows],
        "feature_image_ids": expected_ids,
        "match_pair_ids": match_ids,
        "geometry_pair_ids": geometry_ids,
        "calibration_prior_names": sorted(prior_names),
        "heldout_priors_physically_absent": True,
        "gravity_or_imu_prior_rows": gravity_rows,
        "retained_pair_evidence": private_pair_digests,
        "source_retained_pair_evidence": source_pair_digests,
        "pair_evidence_byte_exact": True,
        "committed_view": sqlite_logical_record(database, immutable=True),
    }


def _calibration_contract(
    database: Path,
    source: Path,
    rows: Sequence[Mapping[str, Any]],
    reader: SegmentReader,
    assignments: Mapping[str, Any],
    config: GeodeticSubmapConfig,
) -> dict[str, Any]:
    source_poses = _poses_from_images_txt(
        source / "registered_text" / "images.txt"
    )
    selected_baselines = _stereo_baselines(source_poses, rows)
    transform = np.asarray(
        reader.calibration["T_right_left"], dtype=np.float64
    )
    contract_baseline = float(np.linalg.norm(transform[:3, 3]))
    maximum_error = float(
        np.max(np.abs(selected_baselines - contract_baseline))
    )
    if maximum_error > config.refinement.max_stereo_baseline_change_m:
        raise ArtifactError(
            "completed model violates the sealed metric stereo baseline"
        )
    row_hashes = {
        table: _table_rows_digest(database, table)
        for table in ("cameras", "rigs")
    }
    try:
        with _readonly_database(database) as connection:
            has_rig_sensors = bool(
                connection.execute(
                    "SELECT COUNT(*) FROM sqlite_schema "
                    "WHERE type='table' AND name='rig_sensors'"
                ).fetchone()[0]
            )
    except sqlite3.Error as exc:
        raise ArtifactError(f"cannot inspect rig calibration: {exc}") from exc
    if has_rig_sensors:
        row_hashes["rig_sensors"] = _table_rows_digest(
            database, "rig_sensors"
        )
    return {
        "schema_version": 1,
        "camera_ids": list(assignments["camera_ids"]),
        "rig_ids": list(assignments["rig_ids"]),
        "database_calibration_rows": row_hashes,
        "segment_stereo_baseline_m": contract_baseline,
        "source_model_stereo_baseline_m": {
            "minimum": float(selected_baselines.min()),
            "median": float(np.median(selected_baselines)),
            "maximum": float(selected_baselines.max()),
            "maximum_abs_contract_error_m": maximum_error,
        },
        "maximum_source_baseline_error_m": (
            config.refinement.max_stereo_baseline_change_m
        ),
        "mapper_contract": {
            "initialization_mode": "fresh",
            "fixed_intrinsics": True,
            "fixed_rig_extrinsics": True,
            "fixed_metric_baseline": True,
            "free_scale_estimation": False,
            "lever_arm_estimation": False,
            "imu_used": False,
        },
    }


def _colmap_binding(frontend: Path) -> dict[str, Any]:
    provenance = _json(frontend / "provenance.json")
    value = provenance.get("colmap")
    if not isinstance(value, Mapping):
        raise ArtifactError("frontend provenance has no pinned COLMAP identity")
    executable = value.get("executable")
    digest = value.get("executable_sha256")
    version = value.get("version")
    if (
        not isinstance(executable, str)
        or not executable
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not isinstance(version, str)
        or "COLMAP" not in version.upper()
    ):
        raise ArtifactError("frontend COLMAP identity is not hash-pinned")
    return {
        "executable": str(Path(executable).expanduser().resolve()),
        "executable_sha256": digest,
        "version": version,
    }


def _seal_files(root: Path, names: Sequence[str]) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ArtifactError(f"cannot seal missing or unsafe file: {name}")
        files[name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    body = {"schema_version": 1, "files": files}
    return {**body, "seal_sha256": canonical_hash(body)}


def _verify_file_seal(
    root: Path, seal_name: str, required_names: Sequence[str]
) -> dict[str, Any]:
    seal_path = root / seal_name
    seal = _json(seal_path)
    body = {"schema_version": seal.get("schema_version"), "files": seal.get("files")}
    if (
        seal.get("schema_version") != 1
        or seal.get("seal_sha256") != canonical_hash(body)
        or not isinstance(seal.get("files"), Mapping)
        or set(seal["files"]) != set(required_names)
    ):
        raise ArtifactError(f"invalid {seal_name}")
    for name, expected in seal["files"].items():
        path = root / str(name)
        current = {
            "sha256": sha256_file(path) if path.is_file() else None,
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
        if path.is_symlink() or current != dict(expected):
            raise ArtifactError(f"sealed artifact file changed: {name}")
    return seal


def prepare_geodetic_submap_plan(
    frontend_artifact: str | Path,
    completed_backend: str | Path,
    segment: str | Path,
    frame_selection: str | Path,
    destination: str | Path,
    *,
    config: GeodeticSubmapConfig = GeodeticSubmapConfig(),
) -> Path:
    """Atomically publish one immutable private-database execution plan."""
    (
        frontend,
        manifest,
        source,
        source_plan,
        mapper_config,
        source_quality,
        segment_path,
        reader,
        source_evidence,
    ) = _input_context(frontend_artifact, completed_backend, segment)
    selection_path, selection = _load_frame_selection(
        frame_selection, frontend, segment_path, manifest
    )
    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite geodetic plan: {output}")
    immutable = (frontend, source, segment_path)
    if any(output == item or item in output.parents for item in immutable):
        raise ArtifactError("geodetic plan must be outside immutable inputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.writing-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        rows = _selected_rows(manifest, selection["frame_ids"])
        metadata = _image_metadata(rows, manifest["frames"])
        selected_names = list(metadata)
        selected_left_names = [
            str(row["left_image"]["name"]) for row in rows
        ]
        source_database = frontend / "database.db"
        assignments = _source_assignments(
            source_database, selected_names, metadata
        )
        id_by_name = {
            name: image_id
            for image_id, name, _ in assignments["image_rows"]
        }
        gnss = _raw_gnss_endpoints(reader)
        candidates, pair_sources = _pair_candidates(
            source_database, metadata, gnss
        )
        pair_audit, retained_pair_ids = _pair_audit(
            candidates, config.pair_policy
        )
        source_database_evidence = _copy_subset_database(
            source_database,
            staging / "database.db",
            assignments,
            retained_pair_ids,
            {id_by_name[name] for name in selected_left_names},
        )
        if (
            source_database_evidence["source_raw_sha256_before"]
            != source_evidence["frontend_database_raw_sha256"]
        ):
            raise ArtifactError("source database differs from its frontend seal")
        prior_split = _weight_and_split_priors(
            staging / "database.db",
            rows,
            reader,
            config,
            mapper_config.alignment_temporal_blocks,
        )
        private_sqlite_metadata = _normalize_private_database_metadata(
            staging / "database.db",
            source_database_evidence["source_sqlite_metadata"],
        )
        source_database_evidence["private_sqlite_metadata"] = (
            private_sqlite_metadata
        )
        inventory = _database_inventory(
            staging / "database.db",
            source_database,
            assignments,
            selected_names,
            retained_pair_ids,
            pair_sources,
            prior_split["calibration_names"],
        )
        calibration = _calibration_contract(
            staging / "database.db",
            source,
            rows,
            reader,
            assignments,
            config,
        )
        _remove_checkpointed_database_sidecars(staging / "database.db")
        _write_lines(staging / "all_images.txt", selected_names)
        _write_lines(
            staging / "constant_cameras.txt", assignments["camera_ids"]
        )
        _write_lines(staging / "constant_rigs.txt", assignments["rig_ids"])
        _write_lines(
            staging / "calibration_prior_names.txt",
            prior_split["calibration_names"],
        )
        _write_lines(
            staging / "holdout_prior_names.txt", prior_split["holdout_names"]
        )
        _atomic_json(staging / "pair_audit.json", pair_audit)
        _atomic_json(staging / "prior_split.json", prior_split)
        _atomic_json(staging / "database_inventory.json", inventory)
        _atomic_json(
            staging / "source_database_evidence.json",
            source_database_evidence,
        )
        plan = {
            "schema_version": 1,
            "kind": _PLAN_KIND,
            "frontend_artifact": str(frontend),
            "completed_backend": str(source),
            "completed_registered_model": str(source / "registered_model"),
            "segment": str(segment_path),
            "frame_selection": str(selection_path),
            "frame_selection_sha256": sha256_file(selection_path),
            "selection_sha256": selection["selection_sha256"],
            "selected_frame_ids": list(selection["frame_ids"]),
            "selected_image_names": selected_names,
            "selected_left_image_names": selected_left_names,
            "n_frames": len(rows),
            "n_images": len(selected_names),
            "image_path": str(frontend / "images"),
            "input_model": str(source / "registered_model"),
            "config": _config_record(config),
            "mapper_evaluation_config": asdict(mapper_config),
            "source_quality_passed": bool(source_quality["passed"]),
            "source_evidence": source_evidence,
            "private_database_committed_view": inventory["committed_view"],
            "private_database_sqlite_metadata": private_sqlite_metadata,
            "pair_audit_sha256": sha256_file(staging / "pair_audit.json"),
            "prior_split_sha256": sha256_file(staging / "prior_split.json"),
            "database_inventory_sha256": sha256_file(
                staging / "database_inventory.json"
            ),
            "calibration_contract": calibration,
            "colmap": _colmap_binding(frontend),
            "optimizer_contract": {
                "implementation": "existing_rtk_refinement_pose_prior_mapper",
                "initialization_mode": "fresh",
                "position_priors": "raw_gnss_covariance_status_weighted",
                "heldout_priors_physically_absent": True,
                "fixed_stereo_rig": True,
                "fixed_intrinsics": True,
                "fixed_extrinsics": True,
                "fixed_metric_baseline": True,
                "deterministic_seed": config.refinement.random_seed,
                "imu": False,
                "free_scale": False,
                "lever_arm_estimation": False,
                "stock_colmap_strict_se3_only": False,
                "internal_alignment": "Sim3_then_restore_database_rig_scale",
                "scale_restored_from_fixed_rig": True,
            },
        }
        _atomic_json(staging / "geodetic_submap_plan.json", plan)
        _atomic_json(staging / "plan_seal.json", _seal_files(staging, _PLAN_FILES))
        # Close the copy/verification race before publishing the immutable plan.
        _input_context(frontend, source, segment_path)
        _load_frame_selection(selection_path, frontend, segment_path, manifest)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite geodetic plan: {output}")
        publish_directory_noreplace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _plan_context(output)
    return output


def _plan_context(
    artifact: str | Path,
) -> tuple[
    Path,
    dict[str, Any],
    GeodeticSubmapConfig,
    MapperConfig,
    dict[str, Any],
]:
    root = Path(artifact).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ArtifactError("geodetic submap plan is missing or unsafe")
    expected_entries = set(_PLAN_FILES) | {"plan_seal.json"}
    if {path.name for path in root.iterdir()} != expected_entries:
        raise ArtifactError("geodetic submap plan file inventory changed")
    seal = _verify_file_seal(root, "plan_seal.json", _PLAN_FILES)
    plan = _json(root / "geodetic_submap_plan.json")
    if plan.get("kind") != _PLAN_KIND or plan.get("schema_version") != 1:
        raise ArtifactError("invalid geodetic submap plan")
    config = _config_from_record(plan.get("config"))
    (
        frontend,
        manifest,
        source,
        source_plan,
        mapper_config,
        source_quality,
        segment,
        reader,
        evidence,
    ) = _input_context(
        plan.get("frontend_artifact", ""),
        plan.get("completed_backend", ""),
        plan.get("segment", ""),
    )
    selection_path, selection = _load_frame_selection(
        plan.get("frame_selection", ""), frontend, segment, manifest
    )
    rows = _selected_rows(manifest, selection["frame_ids"])
    metadata = _image_metadata(rows, manifest["frames"])
    expected_names = list(metadata)
    expected_left = [str(row["left_image"]["name"]) for row in rows]
    if (
        plan.get("source_evidence") != evidence
        or Path(str(plan.get("completed_registered_model", ""))).resolve()
        != source / "registered_model"
        or Path(str(plan.get("image_path", ""))).resolve()
        != frontend / "images"
        or Path(str(plan.get("input_model", ""))).resolve()
        != source / "registered_model"
        or plan.get("selected_frame_ids") != selection["frame_ids"]
        or plan.get("selected_image_names") != expected_names
        or plan.get("selected_left_image_names") != expected_left
        or plan.get("n_frames") != len(rows)
        or plan.get("n_images") != len(expected_names)
        or plan.get("frame_selection_sha256") != sha256_file(selection_path)
        or plan.get("selection_sha256") != selection["selection_sha256"]
        or plan.get("mapper_evaluation_config") != asdict(mapper_config)
        or not source_quality.get("passed")
        or plan.get("colmap") != _colmap_binding(frontend)
    ):
        raise ArtifactError("geodetic submap plan input binding changed")
    for filename, expected in (
        ("pair_audit.json", plan.get("pair_audit_sha256")),
        ("prior_split.json", plan.get("prior_split_sha256")),
        ("database_inventory.json", plan.get("database_inventory_sha256")),
    ):
        if sha256_file(root / filename) != expected:
            raise ArtifactError(f"geodetic plan hash binding changed: {filename}")
    committed = sqlite_logical_record(root / "database.db", immutable=True)
    inventory = _json(root / "database_inventory.json")
    source_database_evidence = _json(
        root / "source_database_evidence.json"
    )
    sqlite_metadata = _database_sqlite_metadata(root / "database.db")
    if (
        committed != plan.get("private_database_committed_view")
        or committed != inventory.get("committed_view")
        or sqlite_metadata
        != plan.get("private_database_sqlite_metadata")
        or sqlite_metadata
        != source_database_evidence.get("private_sqlite_metadata")
        or not inventory.get("pair_evidence_byte_exact")
        or not inventory.get("heldout_priors_physically_absent")
        or inventory.get("gravity_or_imu_prior_rows") != 0
    ):
        raise ArtifactError("private optimizer database changed")
    prior_split = _json(root / "prior_split.json")
    calibration_names = _read_lines(root / "calibration_prior_names.txt")
    holdout_names = _read_lines(root / "holdout_prior_names.txt")
    if (
        calibration_names != prior_split.get("calibration_names")
        or holdout_names != prior_split.get("holdout_names")
        or set(calibration_names) & set(holdout_names)
        or _read_lines(root / "all_images.txt") != expected_names
        or [int(value) for value in _read_lines(root / "constant_cameras.txt")]
        != inventory.get("camera_ids")
        or [int(value) for value in _read_lines(root / "constant_rigs.txt")]
        != inventory.get("rig_ids")
    ):
        raise ArtifactError("geodetic plan optimizer inventory changed")
    calibration = plan.get("calibration_contract")
    if not isinstance(calibration, Mapping):
        raise ArtifactError("geodetic plan has no calibration contract")
    for table, expected in calibration.get(
        "database_calibration_rows", {}
    ).items():
        if _table_rows_digest(
            root / "database.db", str(table), immutable=True
        ) != expected:
            raise ArtifactError("private calibration rows changed")
    contract = calibration.get("mapper_contract")
    if contract != {
        "initialization_mode": "fresh",
        "fixed_intrinsics": True,
        "fixed_rig_extrinsics": True,
        "fixed_metric_baseline": True,
        "free_scale_estimation": False,
        "lever_arm_estimation": False,
        "imu_used": False,
    }:
        raise ArtifactError("fixed calibration/baseline contract changed")
    optimizer = plan.get("optimizer_contract")
    if not isinstance(optimizer, Mapping) or not all(
        (
            optimizer.get("heldout_priors_physically_absent"),
            optimizer.get("fixed_stereo_rig"),
            optimizer.get("fixed_intrinsics"),
            optimizer.get("fixed_extrinsics"),
            optimizer.get("fixed_metric_baseline"),
        )
    ) or any(
        (
            optimizer.get("imu"),
            optimizer.get("free_scale"),
            optimizer.get("lever_arm_estimation"),
        )
    ):
        raise ArtifactError("optimizer contract changed")
    # The seal is returned so result artifacts can bind the exact plan seal.
    return root, plan, config, mapper_config, {
        "source_quality": source_quality,
        "manifest": manifest,
        "rows": rows,
        "reader": reader,
        "seal": seal,
    }


def _verify_pinned_colmap(
    executable: str | Path, expected: Mapping[str, Any]
) -> Path:
    path = Path(executable).expanduser().resolve()
    if not path.is_file():
        raise ArtifactError(f"pinned COLMAP executable does not exist: {path}")
    if (
        path != Path(str(expected.get("executable", ""))).resolve()
        or sha256_file(path) != expected.get("executable_sha256")
    ):
        raise ArtifactError("COLMAP executable differs from the frontend pin")
    return path


def _option(command: Sequence[str], name: str) -> str:
    try:
        index = command.index(name)
        return str(command[index + 1])
    except (ValueError, IndexError) as exc:
        raise ArtifactError(f"pose-prior mapper command lacks {name}") from exc


def _verify_mapper_command(
    command: Sequence[str], config: GeodeticSubmapConfig
) -> None:
    if len(command) < 2 or command[1] != "pose_prior_mapper":
        raise ArtifactError("geodetic solve is not pose_prior_mapper")
    if "--input_path" in command:
        raise ArtifactError("geodetic submap mapper must initialize fresh")
    expected = {
        "--default_random_seed": str(config.refinement.random_seed),
        "--overwrite_priors_covariance": "0",
        "--Mapper.random_seed": str(config.refinement.random_seed),
        "--Mapper.multiple_models": "0",
        "--Mapper.extract_colors": "0",
        "--Mapper.ba_refine_focal_length": "0",
        "--Mapper.ba_refine_principal_point": "0",
        "--Mapper.ba_refine_extra_params": "0",
        "--Mapper.ba_refine_sensor_from_rig": "0",
        "--Mapper.ba_use_gpu": "0",
    }
    for name, value in expected.items():
        if _option(command, name) != value:
            raise ArtifactError(f"fixed mapper contract violated by {name}")
    for required in (
        "--Mapper.image_list_path",
        "--Mapper.constant_camera_list_path",
        "--Mapper.constant_rig_list_path",
    ):
        _option(command, required)
    forbidden_fragments = ("imu", "gravity", "lever_arm")
    if any(
        fragment in value.lower()
        for value in command
        for fragment in forbidden_fragments
    ):
        raise ArtifactError("mapper command unexpectedly consumes IMU/lever data")


def build_geodetic_submap_command(
    plan_artifact: str | Path,
    execution_workspace: str | Path,
    executable: str | Path,
) -> tuple[str, ...]:
    """Build and audit the existing fresh pose-prior-mapper command."""
    plan_root, plan, config, _, _ = _plan_context(plan_artifact)
    colmap = _verify_pinned_colmap(executable, plan["colmap"])
    execution = Path(execution_workspace).expanduser().resolve()
    immutable = (
        plan_root,
        Path(str(plan["frontend_artifact"])).resolve(),
        Path(str(plan["completed_backend"])).resolve(),
        Path(str(plan["segment"])).resolve(),
    )
    if any(
        execution == item or item in execution.parents for item in immutable
    ):
        raise ArtifactError("execution workspace must be outside immutable inputs")
    command = _rtk_refinement_command(
        execution, plan, config.refinement, colmap
    )
    _verify_mapper_command(command, config)
    return command


def _copy_execution_input(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite execution input: {destination}")
    shutil.copyfile(source, destination)
    with destination.open("rb") as stream:
        os.fsync(stream.fileno())


def _quality_report(
    execution: Path,
    plan_root: Path,
    plan: Mapping[str, Any],
    config: GeodeticSubmapConfig,
    mapper_config: MapperConfig,
    context: Mapping[str, Any],
    executable: str | Path,
    runner: Runner,
) -> dict[str, Any]:
    rows = context["rows"]
    source_quality = context["source_quality"]
    refined_stats = _analyze_model(
        execution / "refined_model", executable, runner
    )
    registered = registered_names_from_images_txt(
        execution / "refined_text" / "images.txt"
    )
    visual = quality_summary(
        plan["selected_image_names"],
        registered,
        refined_stats,
        max_reprojection_error_px=mapper_config.max_reprojection_error_px,
        min_mean_track_length=mapper_config.min_mean_track_length,
    )
    source = Path(str(plan["completed_backend"])).resolve()
    source_poses = _poses_from_images_txt(
        source / "registered_text" / "images.txt"
    )
    refined_poses = _poses_from_images_txt(
        execution / "refined_text" / "images.txt"
    )
    left_names = [str(row["left_image"]["name"]) for row in rows]
    source_centers = np.stack([source_poses[name][1] for name in left_names])
    refined_centers = np.stack([refined_poses[name][1] for name in left_names])
    scale = _similarity_scale(source_centers, refined_centers)
    source_baselines = _stereo_baselines(source_poses, rows)
    refined_baselines = _stereo_baselines(refined_poses, rows)
    baseline_change = float(
        np.max(np.abs(refined_baselines - source_baselines))
    )
    calibration = _calibration_difference(
        source / "registered_text", execution / "refined_text"
    )
    split = _json(plan_root / "prior_split.json")
    evaluation_names = {
        str(item["name"])
        for item in split["records"]
        if item.get("role") in {"calibration", "holdout"}
    }
    source_priors = _cartesian_camera_priors(
        Path(str(plan["frontend_artifact"])) / "database.db",
        set(left_names),
    )
    priors = {
        name: value
        for name, value in source_priors.items()
        if name in evaluation_names
    }
    source_rtk, source_evaluation = _heldout_evaluation(
        source_poses, rows, priors, mapper_config
    )
    refined_rtk, refined_evaluation = _heldout_evaluation(
        refined_poses, rows, priors, mapper_config
    )
    if not np.array_equal(
        source_evaluation.temporal_block_ids,
        refined_evaluation.temporal_block_ids,
    ):
        raise ArtifactError("source/refined temporal GNSS splits differ")
    ordered_records = [
        item
        for item in split["records"]
        if item.get("role") in {"calibration", "holdout"}
    ]
    recorded_roles = [str(item["role"]) for item in ordered_records]
    expected_roles = [
        "calibration" if value else "holdout"
        for value in source_evaluation.calibration_mask
    ]
    if recorded_roles != expected_roles:
        raise ArtifactError("optimizer and evaluator GNSS splits differ")
    source_median = float(source_rtk["residual_m"]["median"])
    refined_median = float(refined_rtk["residual_m"]["median"])
    source_rtk_passed = all(
        check["passed"]
        for check in source_rtk["checks"].values()
        if check["authoritative"]
    )
    refined_rtk_passed = all(
        check["passed"]
        for check in refined_rtk["checks"].values()
        if check["authoritative"]
    )
    source_reprojection = float(
        source_quality["checks"]["mean_reprojection_error_px"]["value"]
    )
    inventory = _json(plan_root / "database_inventory.json")
    checks = {
        "source_visual_quality": {
            "value": bool(source_quality["passed"]),
            "expected": True,
            "passed": bool(source_quality["passed"]),
        },
        "selected_visual_quality": {
            "value": bool(visual["passed"]),
            "expected": True,
            "passed": bool(visual["passed"]),
        },
        "selected_image_inventory_exact": {
            "value": sorted(registered) == sorted(plan["selected_image_names"]),
            "expected": True,
            "passed": sorted(registered) == sorted(plan["selected_image_names"]),
        },
        "reprojection_regression_px": {
            "value": float(refined_stats["mean_reprojection_error_px"])
            - source_reprojection,
            "maximum": config.refinement.max_reprojection_regression_px,
            "passed": float(refined_stats["mean_reprojection_error_px"])
            <= source_reprojection
            + config.refinement.max_reprojection_regression_px,
        },
        "fresh_mean_track_length_absolute": {
            "value": float(refined_stats["mean_track_length"]),
            "minimum": config.refinement.fresh_min_mean_track_length,
            "passed": float(refined_stats["mean_track_length"])
            >= config.refinement.fresh_min_mean_track_length,
        },
        "fresh_mean_observations_per_image_absolute": {
            "value": float(refined_stats["mean_observations_per_image"]),
            "minimum": config.refinement.fresh_min_mean_observations_per_image,
            "passed": float(refined_stats["mean_observations_per_image"])
            >= config.refinement.fresh_min_mean_observations_per_image,
        },
        "eligible_pair_track_evidence_retention": {
            "value": 1.0 if inventory["pair_evidence_byte_exact"] else 0.0,
            "minimum": 1.0,
            "passed": bool(inventory["pair_evidence_byte_exact"]),
        },
        "trajectory_similarity_scale": {
            "value": scale,
            "expected": 1.0,
            "maximum_abs_deviation": (
                config.refinement.max_trajectory_scale_deviation
            ),
            "passed": abs(scale - 1.0)
            <= config.refinement.max_trajectory_scale_deviation,
        },
        "stereo_baseline_max_change_m": {
            "value": baseline_change,
            "maximum": config.refinement.max_stereo_baseline_change_m,
            "passed": baseline_change
            <= config.refinement.max_stereo_baseline_change_m,
        },
        "calibration_parameter_max_change": {
            "value": calibration["maximum_numeric_abs_change"],
            "maximum": config.refinement.max_calibration_parameter_change,
            "passed": bool(calibration["same_structure"])
            and calibration["maximum_numeric_abs_change"]
            <= config.refinement.max_calibration_parameter_change,
        },
        "heldout_rtk_absolute_gates": {
            "value": refined_rtk_passed,
            "expected": True,
            "passed": refined_rtk_passed,
        },
        "heldout_median_not_worse_m": {
            "value": refined_median - source_median,
            "maximum": config.refinement.max_holdout_median_regression_m,
            "passed": refined_median
            <= source_median
            + config.refinement.max_holdout_median_regression_m,
        },
        "heldout_improvement_when_source_failed_m": {
            "authoritative": not source_rtk_passed,
            "value": source_median - refined_median,
            "minimum": (
                config.refinement.min_holdout_median_improvement_m
                if not source_rtk_passed
                else None
            ),
            "passed": source_rtk_passed
            or source_median - refined_median
            >= config.refinement.min_holdout_median_improvement_m,
        },
    }
    passed = all(bool(check["passed"]) for check in checks.values())
    return {
        "schema_version": 1,
        "stage": "quality",
        "method": "bounded_geodetic_submap_pose_prior_mapping",
        "experimental": True,
        "passed": passed,
        "publication_eligible": passed,
        "production_poses_published": False,
        "checks": checks,
        "visual_quality": visual,
        "source_model_stats": {
            "mean_reprojection_error_px": source_reprojection,
        },
        "refined_model_stats": refined_stats,
        "metric_integrity": {
            "stock_colmap_strict_se3_only": False,
            "scale_restored_from_fixed_rig": True,
            "trajectory_similarity_scale": scale,
            "source_stereo_baseline_m": {
                "minimum": float(source_baselines.min()),
                "median": float(np.median(source_baselines)),
                "maximum": float(source_baselines.max()),
            },
            "refined_stereo_baseline_m": {
                "minimum": float(refined_baselines.min()),
                "median": float(np.median(refined_baselines)),
                "maximum": float(refined_baselines.max()),
            },
            "calibration": calibration,
        },
        "rtk_holdout": {
            "same_temporal_split": True,
            "n_calibration": int(source_evaluation.calibration_mask.sum()),
            "n_holdout": int(source_evaluation.holdout_mask.sum()),
            "source_passed": source_rtk_passed,
            "refined_passed": refined_rtk_passed,
            "source": source_rtk,
            "refined": refined_rtk,
        },
    }


def run_geodetic_submap_plan(
    plan_artifact: str | Path,
    result_destination: str | Path,
    executable: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Run one plan in staging and atomically publish an experimental result."""
    plan_root, plan, config, mapper_config, context = _plan_context(
        plan_artifact
    )
    output = Path(result_destination).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite geodetic result: {output}")
    immutable = (
        plan_root,
        Path(str(plan["frontend_artifact"])).resolve(),
        Path(str(plan["completed_backend"])).resolve(),
        Path(str(plan["segment"])).resolve(),
    )
    if any(output == item or item in output.parents for item in immutable):
        raise ArtifactError("geodetic result must be outside immutable inputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.writing-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        for name in (
            "database.db",
            "all_images.txt",
            "constant_cameras.txt",
            "constant_rigs.txt",
            "calibration_prior_names.txt",
            "holdout_prior_names.txt",
            "pair_audit.json",
            "prior_split.json",
            "database_inventory.json",
        ):
            _copy_execution_input(plan_root / name, staging / name)
        if sqlite_logical_record(
            staging / "database.db", immutable=True
        ) != plan[
            "private_database_committed_view"
        ]:
            raise ArtifactError("execution database copy differs from the plan")
        command = build_geodetic_submap_command(
            plan_root, staging, executable
        )
        incomplete = staging / "refined_model.incomplete"
        incomplete.mkdir()
        attempt = _next_solve_attempt(staging)
        resources = (
            _run_monitored_mapper(command, attempt, config.refinement)
            if runner is subprocess.run
            else _run_injected_mapper(command, attempt, runner)
        )
        candidates = _model_candidates(incomplete)
        if not candidates:
            raise ArtifactError("pose-prior mapper produced no valid submap model")
        analyses = [
            {
                "relative_path": candidate.relative_to(incomplete).as_posix(),
                "stats": _analyze_model(candidate, executable, runner),
            }
            for candidate in candidates
        ]
        selected = max(
            analyses,
            key=lambda item: (
                item["stats"]["registered_images"],
                item["stats"]["observations"],
                -item["stats"]["mean_reprojection_error_px"],
            ),
        )
        if int(selected["stats"]["registered_images"]) != int(plan["n_images"]):
            raise ArtifactError("geodetic submap lost selected stereo images")
        selected_path = incomplete / str(selected["relative_path"])
        published_model = staging / "refined_model"
        if selected_path == incomplete:
            os.rename(incomplete, published_model)
        else:
            os.rename(selected_path, published_model)
            shutil.rmtree(incomplete)
        _atomic_json(
            staging / "refined_model_manifest.json",
            _tree_manifest(published_model),
        )

        refined_text_incomplete = staging / "refined_text.incomplete"
        refined_text_incomplete.mkdir()
        converter = _model_converter_command(
            published_model, refined_text_incomplete, executable
        )
        _execute(converter, runner)
        if any(
            not (refined_text_incomplete / name).is_file()
            for name in ("images.txt", "cameras.txt", "rigs.txt")
        ):
            raise ArtifactError("refined geodetic text model is incomplete")
        os.rename(refined_text_incomplete, staging / "refined_text")
        _atomic_json(
            staging / "refined_text_manifest.json",
            _tree_manifest(staging / "refined_text"),
        )
        # The existing mapper path is required to leave its private DB intact.
        if _database_sqlite_metadata(staging / "database.db") != plan[
            "private_database_sqlite_metadata"
        ]:
            raise ArtifactError(
                "pose-prior mapper changed private database metadata"
            )
        if sqlite_logical_record(staging / "database.db") != plan[
            "private_database_committed_view"
        ]:
            raise ArtifactError(
                "pose-prior mapper changed private database contents"
            )
        _normalize_private_database_metadata(
            staging / "database.db",
            plan["private_database_sqlite_metadata"],
        )
        _plan_context(plan_root)
        quality = _quality_report(
            staging,
            plan_root,
            plan,
            config,
            mapper_config,
            context,
            executable,
            runner,
        )
        solve = {
            "schema_version": 1,
            "stage": "solve",
            "method": "existing_rtk_refinement_pose_prior_mapper_fresh",
            "command": list(command),
            "selected_stats": selected["stats"],
            "candidates": analyses,
            "all_selected_images_retained": quality["checks"][
                "selected_image_inventory_exact"
            ]["passed"],
            "resources": resources,
        }
        (staging / "reports").mkdir()
        _atomic_json(staging / "reports" / "solve.json", solve)
        _atomic_json(staging / "reports" / "quality.json", quality)
        result = {
            "schema_version": 1,
            "kind": _RESULT_KIND,
            "experimental": True,
            "plan_artifact": str(plan_root),
            "plan_seal_sha256": sha256_file(plan_root / "plan_seal.json"),
            "plan_sha256": sha256_file(
                plan_root / "geodetic_submap_plan.json"
            ),
            "solve_report_sha256": sha256_file(
                staging / "reports" / "solve.json"
            ),
            "quality_report_sha256": sha256_file(
                staging / "reports" / "quality.json"
            ),
            "passed": quality["passed"],
            "publication_eligible": quality["publication_eligible"],
            "production_poses_published": False,
        }
        _atomic_json(staging / "geodetic_submap_result.json", result)
        result_files = sorted(
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file() and path.name != "result_seal.json"
        )
        _atomic_json(
            staging / "result_seal.json", _seal_files(staging, result_files)
        )
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite geodetic result: {output}"
            )
        publish_directory_noreplace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return audited_geodetic_submap_result(output)


def audited_geodetic_submap_result(
    result_artifact: str | Path,
) -> dict[str, Any]:
    """Verify and load an atomically published experimental result."""
    root = Path(result_artifact).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ArtifactError("geodetic result is missing or unsafe")
    seal = _json(root / "result_seal.json")
    files = seal.get("files")
    if not isinstance(files, Mapping):
        raise ArtifactError("geodetic result has no file seal")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "result_seal.json"
    }
    if actual != set(files):
        raise ArtifactError("geodetic result file inventory changed")
    _verify_file_seal(root, "result_seal.json", sorted(actual))
    result = _json(root / "geodetic_submap_result.json")
    if result.get("kind") != _RESULT_KIND:
        raise ArtifactError("invalid geodetic submap result")
    plan_root, _, _, _, _ = _plan_context(result.get("plan_artifact", ""))
    if (
        result.get("plan_seal_sha256")
        != sha256_file(plan_root / "plan_seal.json")
        or result.get("plan_sha256")
        != sha256_file(plan_root / "geodetic_submap_plan.json")
        or result.get("solve_report_sha256")
        != sha256_file(root / "reports" / "solve.json")
        or result.get("quality_report_sha256")
        != sha256_file(root / "reports" / "quality.json")
    ):
        raise ArtifactError("geodetic result hash binding changed")
    _verify_tree(
        root / "refined_model", _json(root / "refined_model_manifest.json")
    )
    _verify_tree(
        root / "refined_text", _json(root / "refined_text_manifest.json")
    )
    quality = _json(root / "reports" / "quality.json")
    if (
        bool(result.get("passed")) != bool(quality.get("passed"))
        or bool(result.get("publication_eligible"))
        != bool(quality.get("publication_eligible"))
        or result.get("production_poses_published") is not False
    ):
        raise ArtifactError("geodetic result publication state changed")
    return {
        **result,
        "artifact": str(root),
        "quality": quality,
        "solve": _json(root / "reports" / "solve.json"),
        "result_seal_sha256": sha256_file(root / "result_seal.json"),
    }


def geodetic_submap_export_context(
    result_artifact: str | Path,
) -> dict[str, Any]:
    """Expose candidate poses only after every sealed acceptance gate passes."""
    audited = audited_geodetic_submap_result(result_artifact)
    if not audited["publication_eligible"] or not audited["quality"]["passed"]:
        raise ArtifactError(
            "geodetic candidate failed acceptance gates; production poses "
            "remain unpublished"
        )
    root = Path(audited["artifact"])
    return {
        "experimental_result": root,
        "text_model": root / "refined_text",
        "text_model_manifest": root / "refined_text_manifest.json",
        "quality_report": root / "reports" / "quality.json",
        "result_seal": root / "result_seal.json",
        "production_publication_allowed": True,
    }


__all__ = [
    "GeodeticSubmapConfig",
    "audited_geodetic_submap_result",
    "build_geodetic_submap_command",
    "create_geodetic_frame_selection",
    "geodetic_submap_export_context",
    "prepare_geodetic_submap_plan",
    "run_geodetic_submap_plan",
]
