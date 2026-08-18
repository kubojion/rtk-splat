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
from collections import defaultdict
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
from rtk_splat.backends.geodetic_gnss import (
    GeodeticGnssTemporalPolicy,
    raw_gnss_temporal_filter,
)
from rtk_splat.backends.geodetic_pairs import (
    GeodeticPairPolicy,
    PairCandidate,
    RawGnssEndpoint,
    decide_pair,
)
from rtk_splat.backends.mapper_config import MapperConfig
from rtk_splat.backends.quality import (
    estimate_rigid_alignment,
    quality_summary,
    rtk_residual_quality,
    temporal_block_split,
)
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
from rtk_splat.frontends.colmap import _trusted_priors


_PAIR_ID_BASE = 2_147_483_647
_SELECTION_KIND = "rtk_splat_geodetic_submap_frame_selection"
_PLAN_KIND = "rtk_splat_geodetic_submap_plan"
_RESULT_KIND = "rtk_splat_geodetic_submap_result"
_HARDENED_PLAN_SCHEMA_VERSION = 4
_SUPPORTED_PLAN_SCHEMA_VERSIONS = {1, 2, 3, 4}
_SEALED_EVALUATION_PRIOR_SCHEMA_VERSION = 4
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
class GeodeticInitialPairPolicy:
    """Model-independent controls for a deterministic mapper seed pair."""

    boundary_fraction: float = 0.10
    minimum_boundary_frames: int = 1
    minimum_verified_matches: int = 30
    minimum_raw_gnss_displacement_m: float = 0.15

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_boundary_frames, bool)
            or not isinstance(self.minimum_boundary_frames, int)
            or self.minimum_boundary_frames < 1
        ):
            raise ValueError(
                "minimum_boundary_frames must be a positive integer"
            )
        if (
            isinstance(self.minimum_verified_matches, bool)
            or not isinstance(self.minimum_verified_matches, int)
            or self.minimum_verified_matches < 1
        ):
            raise ValueError(
                "minimum_verified_matches must be a positive integer"
            )
        if (
            not math.isfinite(float(self.boundary_fraction))
            or not 0.0 < float(self.boundary_fraction) < 0.5
        ):
            raise ValueError("boundary_fraction must be in (0, 0.5)")
        if (
            not math.isfinite(float(self.minimum_raw_gnss_displacement_m))
            or self.minimum_raw_gnss_displacement_m <= 0.0
        ):
            raise ValueError(
                "minimum_raw_gnss_displacement_m must be finite and positive"
            )


@dataclass(frozen=True)
class GeodeticTrajectoryPolicy:
    """Absolute physical acceptance gates for the complete selected path."""

    max_consecutive_calibration_outliers: int = 5
    max_adjacent_displacement_error_m: float = 0.25
    local_window_frames: int = 31
    max_local_window_path_error_m: float = 0.25
    max_local_window_path_relative_error: float = 0.15

    def __post_init__(self) -> None:
        for name, minimum in (
            ("max_consecutive_calibration_outliers", 0),
            ("local_window_frames", 3),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                relation = "non-negative" if minimum == 0 else "at least 3"
                raise ValueError(f"{name} must be an integer {relation}")
        for name in (
            "max_adjacent_displacement_error_m",
            "max_local_window_path_error_m",
            "max_local_window_path_relative_error",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class GeodeticSubmapConfig:
    """All generic planning and existing mapper controls for one sidecar."""

    pair_policy: GeodeticPairPolicy = field(default_factory=GeodeticPairPolicy)
    initial_pair_policy: GeodeticInitialPairPolicy = field(
        default_factory=GeodeticInitialPairPolicy
    )
    trajectory_policy: GeodeticTrajectoryPolicy = field(
        default_factory=GeodeticTrajectoryPolicy
    )
    gnss_temporal_policy: GeodeticGnssTemporalPolicy = field(
        default_factory=GeodeticGnssTemporalPolicy
    )
    refinement: RtkRefinementConfig = field(
        default_factory=_fresh_refinement_config
    )
    position_quality_weights: tuple[tuple[str, float], ...] = _QUALITY_WEIGHTS

    def __post_init__(self) -> None:
        if self.refinement.initialization_mode != "fresh":
            raise ValueError(
                "geodetic submap BA requires fresh pose-prior mapping"
            )
        if (
            self.initial_pair_policy.minimum_raw_gnss_displacement_m
            > self.pair_policy.revisit_physical_cap_m
        ):
            raise ValueError(
                "initial-pair minimum displacement cannot exceed the pair "
                "policy physical cap"
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


def _config_from_record(
    value: Any,
    *,
    legacy_schema: bool = False,
    legacy_temporal_policy: bool = False,
) -> GeodeticSubmapConfig:
    if not isinstance(value, Mapping):
        raise ArtifactError("geodetic submap config must be an object")
    try:
        pair = GeodeticPairPolicy(**dict(value["pair_policy"]))
        initial_pair = (
            GeodeticInitialPairPolicy()
            if legacy_schema
            else GeodeticInitialPairPolicy(
                **dict(value["initial_pair_policy"])
            )
        )
        trajectory = (
            GeodeticTrajectoryPolicy()
            if legacy_schema
            else GeodeticTrajectoryPolicy(**dict(value["trajectory_policy"]))
        )
        temporal = (
            GeodeticGnssTemporalPolicy()
            if legacy_temporal_policy
            else GeodeticGnssTemporalPolicy(
                **dict(value["gnss_temporal_policy"])
            )
        )
        refinement = RtkRefinementConfig(**dict(value["refinement"]))
        raw_weights = value["position_quality_weights"]
        if not isinstance(raw_weights, (list, tuple)):
            raise TypeError("position quality weights must be a sequence")
        weights = tuple((str(item[0]), float(item[1])) for item in raw_weights)
        config = GeodeticSubmapConfig(
            pair_policy=pair,
            initial_pair_policy=initial_pair,
            trajectory_policy=trajectory,
            gnss_temporal_policy=temporal,
            refinement=refinement,
            position_quality_weights=weights,
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ArtifactError("invalid geodetic submap config") from exc
    # ``asdict`` retains tuples in memory while JSON loads them as lists.
    normalized = _config_record(config)
    if legacy_schema:
        normalized.pop("initial_pair_policy")
        normalized.pop("trajectory_policy")
    if legacy_temporal_policy:
        normalized.pop("gnss_temporal_policy")
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
    sealed_source = Path(str(contract.get("segment_root", ""))).resolve()
    if not sealed_source.is_dir():
        raise ArtifactError("frontend sealed source segment is missing")
    files = contract.get("files")
    if not isinstance(files, Mapping):
        raise ArtifactError("frontend segment contract has no file evidence")
    current: dict[str, dict[str, Any]] = {}
    for relative, expected in files.items():
        path = sealed_source / str(relative)
        if not isinstance(expected, Mapping) or not path.is_file():
            raise ArtifactError(f"sealed segment input is missing: {relative}")
        record = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if record != dict(expected):
            raise ArtifactError(f"sealed segment input changed: {relative}")
        current[str(relative)] = record
    if segment == sealed_source:
        return reader, {
            "segment_root": str(segment),
            "contract_inputs_sha256": canonical_hash(contract),
            "files": current,
        }
    else:
        # A completed frontend remains bound to the exact immutable source
        # segment from which its images/features were built.  A later physical
        # calibration may change only the camera priors in a separately sealed
        # segment.  Accept that narrow case after recomputing its source delta;
        # arbitrary replacement segments remain rejected.
        from rtk_splat.diagnostics.raw_calibration import (
            audited_calibrated_segment,
        )

        derived_evidence = audited_calibrated_segment(
            segment, expected_source_segment=sealed_source
        )
    return reader, {
        "segment_root": str(segment),
        "frontend_source_segment_root": str(sealed_source),
        "contract_inputs_sha256": canonical_hash(contract),
        "files": current,
        "calibrated_derivation": derived_evidence,
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


def _temporally_filtered_gnss(
    reader: SegmentReader,
    policy: GeodeticGnssTemporalPolicy,
    frame_ids: Sequence[int],
) -> tuple[
    dict[int, RawGnssEndpoint],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    """Bind raw-only temporal decisions to acquisition-frame observations."""
    gnss = reader.observations("gnss")
    assert gnss is not None
    required = (
        "raw_timestamp_ns",
        "raw_enu_m",
        "raw_effective_covariance_enu_m2",
        "raw_fix_status",
        "raw_carrier_status",
        "source_index",
    )
    missing = [name for name in required if name not in gnss]
    if missing:
        raise ArtifactError(
            "timestamp-aware GNSS filtering requires immutable raw fields: "
            + ", ".join(missing)
        )
    raw_retained, raw_reasons, raw_audit = raw_gnss_temporal_filter(
        gnss["raw_timestamp_ns"],
        gnss["raw_enu_m"],
        gnss["raw_effective_covariance_enu_m2"],
        gnss["raw_fix_status"],
        gnss["raw_carrier_status"],
        policy=policy,
    )
    if not raw_audit["passed"]:
        failed = [
            name
            for name, check in raw_audit["checks"].items()
            if not check["passed"]
        ]
        raise ArtifactError(
            "raw GNSS temporal filter failed closed: " + ", ".join(failed)
        )

    endpoints = _raw_gnss_endpoints(reader)
    frames = reader.frames
    all_frame_ids = frames["frame_id"].astype(int)
    index_by_frame = {
        int(frame_id): index for index, frame_id in enumerate(all_frame_ids)
    }
    if not set(int(value) for value in frame_ids) <= set(index_by_frame):
        raise ArtifactError("GNSS temporal audit references an unknown frame")
    association_valid = np.asarray(
        gnss.get("association_valid", np.ones(len(all_frame_ids), dtype=bool)),
        dtype=bool,
    )
    source_indices = np.asarray(gnss["source_index"])
    if (
        association_valid.shape != (len(all_frame_ids),)
        or source_indices.shape != (len(all_frame_ids),)
        or not np.issubdtype(source_indices.dtype, np.integer)
    ):
        raise ArtifactError("GNSS raw association arrays are invalid")

    raw_timestamps = np.asarray(gnss["raw_timestamp_ns"], dtype=np.int64)
    raw_positions = np.asarray(gnss["raw_enu_m"], dtype=np.float64)
    raw_covariance = np.asarray(
        gnss["raw_effective_covariance_enu_m2"], dtype=np.float64
    )
    raw_fix = np.asarray(gnss["raw_fix_status"])
    raw_carrier = np.asarray(gnss["raw_carrier_status"])
    filtered: dict[int, RawGnssEndpoint] = {}
    records_by_frame: dict[int, dict[str, Any]] = {}
    for frame_id in all_frame_ids:
        frame_id = int(frame_id)
        index = index_by_frame[frame_id]
        raw_index = int(source_indices[index])
        raw_index_valid = 0 <= raw_index < len(raw_retained)
        associated = bool(association_valid[index]) and raw_index_valid
        temporal_retained = bool(associated and raw_retained[raw_index])
        endpoint = endpoints[frame_id]
        filtered_endpoint = RawGnssEndpoint(
            position_m=endpoint.position_m,
            covariance_m2=endpoint.covariance_m2,
            position_valid=bool(endpoint.position_valid and temporal_retained),
            position_quality=endpoint.position_quality,
            fix_status=endpoint.fix_status,
            carrier_status=endpoint.carrier_status,
        )
        filtered[frame_id] = filtered_endpoint
        if not association_valid[index]:
            reason = "raw_gnss_association_invalid"
        elif not raw_index_valid:
            reason = "raw_gnss_source_index_invalid"
        elif not raw_retained[raw_index]:
            reason = raw_reasons[raw_index]
        elif not endpoint.position_valid:
            reason = "associated_gnss_status_invalid"
        else:
            reason = "raw_gnss_temporally_consistent"
        raw_covariance_record = (
            raw_covariance[raw_index].tolist() if raw_index_valid else None
        )
        raw_position_record = (
            raw_positions[raw_index].tolist() if raw_index_valid else None
        )
        records_by_frame[frame_id] = {
            "frame_id": frame_id,
            "frame_timestamp_ns": int(frames["timestamp_ns"][index]),
            "association_valid": bool(association_valid[index]),
            "source_timestamp_ns": (
                int(gnss["source_timestamp_ns"][index])
                if "source_timestamp_ns" in gnss
                else None
            ),
            "source_residual_ns": (
                int(gnss["source_residual_ns"][index])
                if "source_residual_ns" in gnss
                else None
            ),
            "raw_source_index": raw_index if raw_index_valid else None,
            "raw_timestamp_ns": (
                int(raw_timestamps[raw_index]) if raw_index_valid else None
            ),
            "raw_position_m": raw_position_record,
            "raw_effective_covariance_m2": raw_covariance_record,
            "raw_fix_status": (
                int(raw_fix[raw_index]) if raw_index_valid else None
            ),
            "raw_carrier_status": (
                int(raw_carrier[raw_index]) if raw_index_valid else None
            ),
            "associated_endpoint": endpoint.record(),
            "retained_by_temporal_filter": temporal_retained,
            "trusted_for_nonlocal_pair_decisions": (
                filtered_endpoint.trusted_std_m() is not None
            ),
            "reason": str(reason),
        }

    selected_records = [
        records_by_frame[int(frame_id)] for frame_id in frame_ids
    ]
    selected_rejected = [
        item for item in selected_records if not item["retained_by_temporal_filter"]
    ]
    audit = {
        "schema_version": 1,
        "method": "selected_frame_raw_gnss_temporal_audit_v1",
        "policy": asdict(policy),
        "raw_stream_audit": raw_audit,
        "selected_frame_count": len(selected_records),
        "selected_retained_count": len(selected_records) - len(selected_rejected),
        "selected_rejected_count": len(selected_rejected),
        "decision_inputs_exclude": [
            "visual_poses",
            "finished_visual_model_residual",
            "optimizer_output",
            "heldout_evaluation_result",
        ],
        "observation_records": selected_records,
        "passed": True,
    }
    return filtered, records_by_frame, audit


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
            verified: dict[int, int] = {}
            initial_geometry: dict[int, dict[str, Any]] = {}
            for pair_id, rows, configuration, qvec, tvec in connection.execute(
                "SELECT pair_id, rows, config, qvec, tvec "
                "FROM two_view_geometries"
            ):
                pair_id = int(pair_id)
                verified[pair_id] = int(rows)
                if qvec is not None and tvec is not None:
                    initial_geometry[pair_id] = {
                        "configuration": int(configuration),
                        "qvec": bytes(qvec),
                        "tvec": bytes(tvec),
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
        "_database_path": str(database.resolve()),
        "source_match_pair_ids": sorted(
            pair_id for pair_id in matches if pair_id in selected_pair_ids
        ),
        "source_geometry_pair_ids": sorted(
            pair_id for pair_id in verified if pair_id in selected_pair_ids
        ),
        "initial_geometry_by_pair_id": {
            pair_id: initial_geometry[pair_id]
            for pair_id in sorted(selected_pair_ids & set(initial_geometry))
        },
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


def _initial_pair_gnss_evidence(
    candidate: PairCandidate,
    config: GeodeticSubmapConfig,
) -> dict[str, Any] | None:
    first_std = candidate.first_gnss.trusted_std_m()
    second_std = candidate.second_gnss.trusted_std_m()
    if first_std is None or second_std is None:
        return None
    if max(first_std, second_std) > config.pair_policy.max_endpoint_position_std_m:
        return None
    first_position = np.asarray(
        candidate.first_gnss.position_m, dtype=np.float64
    )
    second_position = np.asarray(
        candidate.second_gnss.position_m, dtype=np.float64
    )
    first_covariance = np.asarray(
        candidate.first_gnss.covariance_m2, dtype=np.float64
    )
    second_covariance = np.asarray(
        candidate.second_gnss.covariance_m2, dtype=np.float64
    )
    displacement = float(np.linalg.norm(second_position - first_position))
    combined = 0.5 * (
        first_covariance
        + second_covariance
        + (first_covariance + second_covariance).T
    )
    combined_sigma = float(
        math.sqrt(max(float(np.linalg.eigvalsh(combined)[-1]), 0.0))
    )
    covariance_allowance = (
        config.pair_policy.covariance_sigma_multiplier * combined_sigma
    )
    allowed = min(
        config.pair_policy.revisit_physical_cap_m,
        config.pair_policy.revisit_distance_m + covariance_allowance,
    )
    if (
        displacement
        < config.initial_pair_policy.minimum_raw_gnss_displacement_m
        or displacement > allowed
    ):
        return None
    return {
        "raw_gnss_displacement_m": displacement,
        "minimum_raw_gnss_displacement_m": (
            config.initial_pair_policy.minimum_raw_gnss_displacement_m
        ),
        "maximum_allowed_displacement_m": allowed,
        "combined_position_std_m": combined_sigma,
        "covariance_allowance_m": covariance_allowance,
        "endpoints": [
            candidate.first_gnss.record(),
            candidate.second_gnss.record(),
        ],
    }


def _select_initial_pair(
    candidates: Sequence[PairCandidate],
    selected_frame_ids: Sequence[int],
    calibration_names: Sequence[str],
    config: GeodeticSubmapConfig,
) -> dict[str, Any]:
    """Choose a deterministic interior seed without consulting model poses."""
    frame_order = {
        int(frame_id): index
        for index, frame_id in enumerate(selected_frame_ids)
    }
    calibration = set(calibration_names)
    required_margin = max(
        config.initial_pair_policy.minimum_boundary_frames,
        int(
            math.ceil(
                (len(selected_frame_ids) - 1)
                * config.initial_pair_policy.boundary_fraction
            )
        ),
    )
    required_verified_matches = max(
        config.initial_pair_policy.minimum_verified_matches,
        config.pair_policy.strong_verified_matches,
    )
    eligible: list[
        tuple[tuple[Any, ...], PairCandidate, dict[str, Any], str]
    ] = []
    for candidate in candidates:
        if (
            candidate.first_image_name not in calibration
            or candidate.second_image_name not in calibration
            or candidate.first_frame_id == candidate.second_frame_id
            or candidate.first_camera != candidate.second_camera
            or (candidate.verified_matches or 0)
            < required_verified_matches
        ):
            continue
        decision = decide_pair(candidate, config.pair_policy)
        if not decision.retained:
            continue
        selection_indices = (
            frame_order[candidate.first_frame_id],
            frame_order[candidate.second_frame_id],
        )
        boundary_margin = min(
            *selection_indices,
            *(len(selected_frame_ids) - 1 - value for value in selection_indices),
        )
        if boundary_margin < required_margin:
            continue
        gnss_evidence = _initial_pair_gnss_evidence(candidate, config)
        if gnss_evidence is None:
            continue
        # Verified geometry dominates; the remaining fields provide stable,
        # physically useful tie-breakers before the ascending pair ID.
        score = (
            int(candidate.verified_matches or 0),
            int(candidate.raw_matches or 0),
            float(gnss_evidence["raw_gnss_displacement_m"]),
            int(boundary_margin),
            -candidate.pair_id,
        )
        eligible.append((score, candidate, gnss_evidence, decision.reason))
    if not eligible:
        raise ArtifactError(
            "no interior calibration-prior image pair satisfies the sealed "
            "visual-evidence and raw-GNSS initialization policy"
        )
    score, selected, gnss_evidence, retention_reason = max(
        eligible, key=lambda item: item[0]
    )
    selection_indices = [
        frame_order[selected.first_frame_id],
        frame_order[selected.second_frame_id],
    ]
    boundary_margin = min(
        *selection_indices,
        *(len(selected_frame_ids) - 1 - value for value in selection_indices),
    )
    return {
        "schema_version": 1,
        "method": "deterministic_interior_raw_gnss_seed_v1",
        "pair_id": selected.pair_id,
        "image_ids": [selected.first_image_id, selected.second_image_id],
        "image_names": [
            selected.first_image_name,
            selected.second_image_name,
        ],
        "frame_ids": [selected.first_frame_id, selected.second_frame_id],
        "frame_indices": [
            selected.first_frame_index,
            selected.second_frame_index,
        ],
        "selection_indices": selection_indices,
        "cameras": [selected.first_camera, selected.second_camera],
        "timestamps_ns": [
            selected.first_timestamp_ns,
            selected.second_timestamp_ns,
        ],
        "raw_matches": selected.raw_matches,
        "verified_matches": selected.verified_matches,
        "required_verified_matches": required_verified_matches,
        "pair_retention_reason": retention_reason,
        "required_boundary_margin_frames": required_margin,
        "actual_boundary_margin_frames": boundary_margin,
        "eligible_candidate_count": len(eligible),
        "score": {
            "verified_matches": score[0],
            "raw_matches": score[1],
            "raw_gnss_displacement_m": score[2],
            "boundary_margin_frames": score[3],
            "pair_id_tiebreak": -score[4],
        },
        "raw_gnss_evidence": gnss_evidence,
        "decision_inputs_exclude": [
            "finished_visual_model_residual",
            "heldout_evaluation_result",
        ],
    }


def _initial_geometry_evidence(
    candidate: PairCandidate,
    geometry_by_pair_id: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any] | None:
    record = geometry_by_pair_id.get(candidate.pair_id)
    if not isinstance(record, Mapping):
        return None
    try:
        configuration = int(record["configuration"])
        qvec = np.frombuffer(record["qvec"], dtype="<f8")
        tvec = np.frombuffer(record["tvec"], dtype="<f8")
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(
            "invalid stored two-view initialization geometry"
        ) from exc
    if (
        configuration not in {2, 9}
        or qvec.shape != (4,)
        or tvec.shape != (3,)
        or not np.isfinite(qvec).all()
        or not np.isfinite(tvec).all()
    ):
        return None
    qvec_norm = float(np.linalg.norm(qvec))
    translation_norm = float(np.linalg.norm(tvec))
    if qvec_norm <= 1.0e-12 or translation_norm <= 1.0e-12:
        return None
    absolute_forward_motion = float(abs(tvec[2]) / translation_norm)
    if absolute_forward_motion >= 0.95:
        return None
    return {
        "source": "sealed_two_view_geometries_relative_pose",
        "configuration": configuration,
        "qvec": qvec.tolist(),
        "tvec": tvec.tolist(),
        "qvec_norm": qvec_norm,
        "translation_norm": translation_norm,
        "absolute_forward_motion": absolute_forward_motion,
        "maximum_absolute_forward_motion": 0.95,
    }


def _eligible_initial_pair_geometry_candidates(
    candidates: Sequence[PairCandidate],
    selected_frame_ids: Sequence[int],
    calibration_names: Sequence[str],
    config: GeodeticSubmapConfig,
    geometry_by_pair_id: Mapping[int, Mapping[str, Any]],
) -> tuple[
    list[tuple[PairCandidate, dict[str, Any], dict[str, Any], str, int]],
    dict[int, int],
    int,
    int,
]:
    frame_order = {
        int(frame_id): index
        for index, frame_id in enumerate(selected_frame_ids)
    }
    calibration = set(calibration_names)
    required_margin = max(
        config.initial_pair_policy.minimum_boundary_frames,
        int(
            math.ceil(
                (len(selected_frame_ids) - 1)
                * config.initial_pair_policy.boundary_fraction
            )
        ),
    )
    required_verified_matches = max(
        config.initial_pair_policy.minimum_verified_matches,
        config.pair_policy.strong_verified_matches,
    )
    eligible: list[
        tuple[PairCandidate, dict[str, Any], dict[str, Any], str, int]
    ] = []
    for candidate in candidates:
        if (
            candidate.first_image_name not in calibration
            or candidate.second_image_name not in calibration
            or candidate.first_frame_id == candidate.second_frame_id
            or candidate.first_camera != candidate.second_camera
            or (candidate.verified_matches or 0)
            < required_verified_matches
        ):
            continue
        decision = decide_pair(candidate, config.pair_policy)
        if not decision.retained:
            continue
        selection_indices = (
            frame_order[candidate.first_frame_id],
            frame_order[candidate.second_frame_id],
        )
        boundary_margin = min(
            *selection_indices,
            *(
                len(selected_frame_ids) - 1 - value
                for value in selection_indices
            ),
        )
        if boundary_margin < required_margin:
            continue
        gnss_evidence = _initial_pair_gnss_evidence(candidate, config)
        geometry_evidence = _initial_geometry_evidence(
            candidate, geometry_by_pair_id
        )
        if gnss_evidence is None or geometry_evidence is None:
            continue
        eligible.append(
            (
                candidate,
                gnss_evidence,
                geometry_evidence,
                decision.reason,
                boundary_margin,
            )
        )
    if not eligible:
        raise ArtifactError(
            "no interior calibration-prior image pair satisfies the sealed "
            "raw-GNSS and two-view initialization policy"
        )
    return eligible, frame_order, required_margin, required_verified_matches


def _select_initial_pair_v2(
    candidates: Sequence[PairCandidate],
    selected_frame_ids: Sequence[int],
    calibration_names: Sequence[str],
    config: GeodeticSubmapConfig,
    geometry_by_pair_id: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose a strong interior seed using only sealed raw pair geometry."""
    (
        eligible,
        frame_order,
        required_margin,
        required_verified_matches,
    ) = _eligible_initial_pair_geometry_candidates(
        candidates,
        selected_frame_ids,
        calibration_names,
        config,
        geometry_by_pair_id,
    )
    maximum_verified_matches = max(
        int(item[0].verified_matches or 0) for item in eligible
    )
    robust_match_floor = max(
        required_verified_matches,
        int(math.ceil(0.25 * maximum_verified_matches)),
    )
    robust = [
        item
        for item in eligible
        if int(item[0].verified_matches or 0) >= robust_match_floor
    ]

    def score(item: tuple[Any, ...]) -> tuple[Any, ...]:
        candidate, gnss, geometry, _reason, boundary_margin = item
        # COLMAP explicitly rejects forward-motion seeds. Among candidates
        # with a robust fraction of the best verified-match support, prefer
        # the sealed two-view pose with the largest lateral component.
        return (
            -float(geometry["absolute_forward_motion"]),
            float(gnss["raw_gnss_displacement_m"]),
            int(candidate.verified_matches or 0),
            int(candidate.raw_matches or 0),
            int(boundary_margin),
            -candidate.pair_id,
        )

    (
        selected,
        gnss_evidence,
        geometry_evidence,
        retention_reason,
        boundary_margin,
    ) = max(robust, key=score)
    selected_score = score(
        (
            selected,
            gnss_evidence,
            geometry_evidence,
            retention_reason,
            boundary_margin,
        )
    )
    selection_indices = [
        frame_order[selected.first_frame_id],
        frame_order[selected.second_frame_id],
    ]
    return {
        "schema_version": 2,
        "method": "deterministic_interior_raw_gnss_two_view_seed_v2",
        "pair_id": selected.pair_id,
        "image_ids": [selected.first_image_id, selected.second_image_id],
        "image_names": [
            selected.first_image_name,
            selected.second_image_name,
        ],
        "frame_ids": [selected.first_frame_id, selected.second_frame_id],
        "frame_indices": [
            selected.first_frame_index,
            selected.second_frame_index,
        ],
        "selection_indices": selection_indices,
        "cameras": [selected.first_camera, selected.second_camera],
        "timestamps_ns": [
            selected.first_timestamp_ns,
            selected.second_timestamp_ns,
        ],
        "raw_matches": selected.raw_matches,
        "verified_matches": selected.verified_matches,
        "required_verified_matches": required_verified_matches,
        "robust_verified_match_floor": robust_match_floor,
        "maximum_eligible_verified_matches": maximum_verified_matches,
        "pair_retention_reason": retention_reason,
        "required_boundary_margin_frames": required_margin,
        "actual_boundary_margin_frames": boundary_margin,
        "eligible_candidate_count": len(eligible),
        "robust_candidate_count": len(robust),
        "score": {
            "negative_absolute_forward_motion": selected_score[0],
            "raw_gnss_displacement_m": selected_score[1],
            "verified_matches": selected_score[2],
            "raw_matches": selected_score[3],
            "boundary_margin_frames": selected_score[4],
            "pair_id_tiebreak": -selected_score[5],
        },
        "raw_gnss_evidence": gnss_evidence,
        "sealed_two_view_geometry": geometry_evidence,
        "decision_inputs_exclude": [
            "finished_visual_model_pose",
            "finished_visual_model_residual",
            "heldout_evaluation_result",
        ],
    }


def _rotation_matrix_from_qvec(qvec: np.ndarray) -> np.ndarray:
    normalized = qvec / np.linalg.norm(qvec)
    w, x, y, z = normalized.tolist()
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _pinhole_keypoint_rays(
    connection: sqlite3.Connection,
    image_id: int,
    ray_cache: dict[int, np.ndarray],
    camera_cache: dict[int, tuple[float, float, float, float]],
) -> np.ndarray:
    cached = ray_cache.get(image_id)
    if cached is not None:
        return cached
    image = connection.execute(
        "SELECT camera_id FROM images WHERE image_id = ?", (image_id,)
    ).fetchone()
    if image is None:
        raise ArtifactError("initial-pair image is absent from source database")
    camera_id = int(image[0])
    calibration = camera_cache.get(camera_id)
    if calibration is None:
        camera = connection.execute(
            "SELECT model, params FROM cameras WHERE camera_id = ?",
            (camera_id,),
        ).fetchone()
        if camera is None or camera[1] is None:
            raise ArtifactError("initial-pair camera calibration is missing")
        model = int(camera[0])
        parameters = np.frombuffer(camera[1], dtype="<f8")
        if model == 0 and parameters.shape == (3,):
            focal, cx, cy = parameters.tolist()
            calibration = (focal, focal, cx, cy)
        elif model == 1 and parameters.shape == (4,):
            calibration = tuple(float(value) for value in parameters)
        else:
            raise ArtifactError(
                "parallax-ranked initialization requires a sealed "
                "zero-distortion SIMPLE_PINHOLE or PINHOLE camera"
            )
        if (
            not np.isfinite(calibration).all()
            or calibration[0] <= 0.0
            or calibration[1] <= 0.0
        ):
            raise ArtifactError("initial-pair camera calibration is invalid")
        camera_cache[camera_id] = calibration
    keypoints = connection.execute(
        "SELECT rows, cols, data FROM keypoints WHERE image_id = ?",
        (image_id,),
    ).fetchone()
    if keypoints is None or keypoints[2] is None:
        raise ArtifactError("initial-pair keypoints are missing")
    rows, columns = int(keypoints[0]), int(keypoints[1])
    values = np.frombuffer(keypoints[2], dtype="<f4")
    if rows < 1 or columns < 2 or values.size != rows * columns:
        raise ArtifactError("initial-pair keypoint storage is invalid")
    points = values.reshape(rows, columns)[:, :2].astype(np.float64)
    fx, fy, cx, cy = calibration
    rays = np.column_stack(
        (
            (points[:, 0] - cx) / fx,
            (points[:, 1] - cy) / fy,
            np.ones(rows, dtype=np.float64),
        )
    )
    norms = np.linalg.norm(rays, axis=1)
    if (
        not np.isfinite(rays).all()
        or not np.isfinite(norms).all()
        or bool((norms <= 1.0e-12).any())
    ):
        raise ArtifactError("initial-pair keypoint rays are invalid")
    rays /= norms[:, None]
    ray_cache[image_id] = rays
    return rays


def _sealed_pair_parallax_evidence(
    connection: sqlite3.Connection,
    candidate: PairCandidate,
    geometry: Mapping[str, Any],
    ray_cache: dict[int, np.ndarray],
    camera_cache: dict[int, tuple[float, float, float, float]],
) -> dict[str, Any]:
    row = connection.execute(
        "SELECT rows, cols, data, config, qvec, tvec "
        "FROM two_view_geometries WHERE pair_id = ?",
        (candidate.pair_id,),
    ).fetchone()
    if row is None or row[2] is None or row[4] is None or row[5] is None:
        raise ArtifactError("sealed initial-pair geometry is missing")
    rows, columns = int(row[0]), int(row[1])
    matches = np.frombuffer(row[2], dtype="<u4")
    qvec = np.frombuffer(row[4], dtype="<f8")
    tvec = np.frombuffer(row[5], dtype="<f8")
    if (
        rows != int(candidate.verified_matches or -1)
        or rows < 1
        or columns != 2
        or matches.size != rows * columns
        or int(row[3]) != int(geometry["configuration"])
        or qvec.shape != (4,)
        or tvec.shape != (3,)
        or not np.allclose(qvec, geometry["qvec"], atol=0.0, rtol=0.0)
        or not np.allclose(tvec, geometry["tvec"], atol=0.0, rtol=0.0)
    ):
        raise ArtifactError("sealed initial-pair geometry storage changed")
    pairs = matches.reshape(rows, columns).astype(np.int64)
    first_rays = _pinhole_keypoint_rays(
        connection, candidate.first_image_id, ray_cache, camera_cache
    )
    second_rays = _pinhole_keypoint_rays(
        connection, candidate.second_image_id, ray_cache, camera_cache
    )
    if (
        bool((pairs < 0).any())
        or bool((pairs[:, 0] >= len(first_rays)).any())
        or bool((pairs[:, 1] >= len(second_rays)).any())
    ):
        raise ArtifactError("initial-pair match index is out of range")
    rotation_second_from_first = _rotation_matrix_from_qvec(qvec)
    second_rays_in_first = (
        second_rays[pairs[:, 1]] @ rotation_second_from_first
    )
    cosines = np.einsum(
        "ij,ij->i", first_rays[pairs[:, 0]], second_rays_in_first
    )
    angles_deg = np.rad2deg(np.arccos(np.clip(cosines, -1.0, 1.0)))
    if not np.isfinite(angles_deg).all():
        raise ArtifactError("initial-pair parallax evidence is non-finite")
    return {
        "source": "sealed_two_view_inlier_keypoint_rays",
        "correspondence_count": rows,
        "minimum_angle_deg": float(np.min(angles_deg)),
        "median_angle_deg": float(np.median(angles_deg)),
        "maximum_angle_deg": float(np.max(angles_deg)),
        "camera_models": "zero_distortion_pinhole",
        "rotation_convention": "cam2_from_cam1",
    }


def _select_initial_pair_v3(
    candidates: Sequence[PairCandidate],
    selected_frame_ids: Sequence[int],
    calibration_names: Sequence[str],
    config: GeodeticSubmapConfig,
    pair_sources: Mapping[str, Any],
) -> dict[str, Any]:
    """Choose the strongest sealed raw-GNSS-consistent parallax seed."""
    geometry_by_pair_id = pair_sources.get(
        "initial_geometry_by_pair_id", {}
    )
    (
        eligible,
        frame_order,
        required_margin,
        required_verified_matches,
    ) = _eligible_initial_pair_geometry_candidates(
        candidates,
        selected_frame_ids,
        calibration_names,
        config,
        geometry_by_pair_id,
    )
    maximum_verified_matches = max(
        int(item[0].verified_matches or 0) for item in eligible
    )
    robust_match_floor = max(
        required_verified_matches,
        int(math.ceil(0.10 * maximum_verified_matches)),
    )
    robust = [
        item
        for item in eligible
        if int(item[0].verified_matches or 0) >= robust_match_floor
    ]
    database_value = pair_sources.get("_database_path")
    if not isinstance(database_value, str):
        raise ArtifactError("source database is missing for parallax audit")
    database = Path(database_value).expanduser().resolve()
    parallax_by_pair_id: dict[int, dict[str, Any]] = {}
    ray_cache: dict[int, np.ndarray] = {}
    camera_cache: dict[int, tuple[float, float, float, float]] = {}
    try:
        with _readonly_database(database) as connection:
            for candidate, _gnss, geometry, _reason, _margin in robust:
                parallax_by_pair_id[candidate.pair_id] = (
                    _sealed_pair_parallax_evidence(
                        connection,
                        candidate,
                        geometry,
                        ray_cache,
                        camera_cache,
                    )
                )
    except sqlite3.Error as exc:
        raise ArtifactError(
            f"cannot audit sealed initial-pair parallax: {exc}"
        ) from exc

    def score(item: tuple[Any, ...]) -> tuple[Any, ...]:
        candidate, gnss, geometry, _reason, boundary_margin = item
        parallax = parallax_by_pair_id[candidate.pair_id]
        return (
            float(parallax["median_angle_deg"]),
            -float(geometry["absolute_forward_motion"]),
            float(gnss["raw_gnss_displacement_m"]),
            int(candidate.verified_matches or 0),
            int(candidate.raw_matches or 0),
            int(boundary_margin),
            -candidate.pair_id,
        )

    (
        selected,
        gnss_evidence,
        geometry_evidence,
        retention_reason,
        boundary_margin,
    ) = max(robust, key=score)
    selected_score = score(
        (
            selected,
            gnss_evidence,
            geometry_evidence,
            retention_reason,
            boundary_margin,
        )
    )
    selection_indices = [
        frame_order[selected.first_frame_id],
        frame_order[selected.second_frame_id],
    ]
    return {
        "schema_version": 3,
        "method": (
            "deterministic_interior_raw_gnss_two_view_parallax_seed_v3"
        ),
        "pair_id": selected.pair_id,
        "image_ids": [selected.first_image_id, selected.second_image_id],
        "image_names": [
            selected.first_image_name,
            selected.second_image_name,
        ],
        "frame_ids": [selected.first_frame_id, selected.second_frame_id],
        "frame_indices": [
            selected.first_frame_index,
            selected.second_frame_index,
        ],
        "selection_indices": selection_indices,
        "cameras": [selected.first_camera, selected.second_camera],
        "timestamps_ns": [
            selected.first_timestamp_ns,
            selected.second_timestamp_ns,
        ],
        "raw_matches": selected.raw_matches,
        "verified_matches": selected.verified_matches,
        "required_verified_matches": required_verified_matches,
        "robust_verified_match_floor": robust_match_floor,
        "maximum_eligible_verified_matches": maximum_verified_matches,
        "pair_retention_reason": retention_reason,
        "required_boundary_margin_frames": required_margin,
        "actual_boundary_margin_frames": boundary_margin,
        "eligible_candidate_count": len(eligible),
        "robust_candidate_count": len(robust),
        "score": {
            "median_parallax_angle_deg": selected_score[0],
            "negative_absolute_forward_motion": selected_score[1],
            "raw_gnss_displacement_m": selected_score[2],
            "verified_matches": selected_score[3],
            "raw_matches": selected_score[4],
            "boundary_margin_frames": selected_score[5],
            "pair_id_tiebreak": -selected_score[6],
        },
        "raw_gnss_evidence": gnss_evidence,
        "sealed_two_view_geometry": geometry_evidence,
        "sealed_parallax_evidence": parallax_by_pair_id[selected.pair_id],
        "decision_inputs_exclude": [
            "finished_visual_model_pose",
            "finished_visual_model_residual",
            "heldout_evaluation_result",
        ],
    }


def _select_initial_pair_v4(
    candidates: Sequence[PairCandidate],
    config: GeodeticSubmapConfig,
) -> dict[str, Any]:
    """Seal deterministic COLMAP auto-initialization over the filtered DB."""
    required_verified_matches = max(
        100,
        config.initial_pair_policy.minimum_verified_matches,
        config.pair_policy.strong_verified_matches,
    )
    candidate_records = []
    for candidate in candidates:
        decision = decide_pair(candidate, config.pair_policy)
        if (
            not decision.retained
            or candidate.first_frame_id == candidate.second_frame_id
            or (candidate.verified_matches or 0) < required_verified_matches
        ):
            continue
        candidate_records.append(
            {
                "pair_id": candidate.pair_id,
                "image_ids": [
                    candidate.first_image_id,
                    candidate.second_image_id,
                ],
                "frame_ids": [
                    candidate.first_frame_id,
                    candidate.second_frame_id,
                ],
                "verified_matches": int(candidate.verified_matches or 0),
                "retention_reason": decision.reason,
            }
        )
    candidate_records.sort(key=lambda item: int(item["pair_id"]))
    if not candidate_records:
        raise ArtifactError(
            "filtered private database has no strong distinct-frame "
            "initialization candidates"
        )
    return {
        "schema_version": 4,
        "method": "deterministic_colmap_auto_filtered_database_v4",
        "selection_owner": "pinned_colmap_pose_prior_mapper",
        "explicit_image_ids": False,
        "random_seed": config.refinement.random_seed,
        "required_verified_matches": required_verified_matches,
        "candidate_pair_count": len(candidate_records),
        "candidate_pair_records_sha256": canonical_hash(candidate_records),
        "candidate_pool": (
            "distinct-frame pair evidence physically retained by the sealed "
            "raw-GNSS consistency policy"
        ),
        "heldout_position_priors_available_to_selector": False,
        "decision_inputs_exclude": [
            "finished_visual_model_pose",
            "finished_visual_model_residual",
            "heldout_evaluation_result",
        ],
    }


def _select_initial_pair_v5(
    candidates: Sequence[PairCandidate],
    selected_frame_ids: Sequence[int],
    calibration_names: Sequence[str],
    config: GeodeticSubmapConfig,
    pair_sources: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal one trusted anchor and delegate only its partner to COLMAP.

    Fully automatic initialization can begin inside a held-out temporal block,
    where the optimizer intentionally has no position priors.  A fully pinned
    pair is also brittle because COLMAP recomputes its initialization geometry.
    This policy therefore seals the earlier endpoint of the deterministic v3
    calibration pair as ``init_image_id1`` and lets the pinned mapper select a
    compatible second image from the already filtered private database.
    """
    source_pair = _select_initial_pair_v3(
        candidates,
        selected_frame_ids,
        calibration_names,
        config,
        pair_sources,
    )
    partner_pool = _select_initial_pair_v4(candidates, config)
    anchor_index = min(
        range(len(source_pair["image_ids"])),
        key=lambda index: (
            int(source_pair["selection_indices"][index]),
            int(source_pair["image_ids"][index]),
        ),
    )
    anchor_name = str(source_pair["image_names"][anchor_index])
    if anchor_name not in set(calibration_names):
        raise ArtifactError("sealed initialization anchor is not calibration")
    return {
        "schema_version": 5,
        "method": "deterministic_calibration_anchor_colmap_partner_v5",
        "selection_owner": {
            "anchor": "sealed_raw_gnss_two_view_parallax_policy",
            "second_image": "pinned_colmap_pose_prior_mapper",
        },
        "explicit_anchor_image_id": True,
        "explicit_second_image_id": False,
        "image_ids": [int(source_pair["image_ids"][anchor_index])],
        "image_names": [anchor_name],
        "frame_ids": [int(source_pair["frame_ids"][anchor_index])],
        "frame_indices": [int(source_pair["frame_indices"][anchor_index])],
        "selection_indices": [
            int(source_pair["selection_indices"][anchor_index])
        ],
        "cameras": [str(source_pair["cameras"][anchor_index])],
        "timestamps_ns": [
            int(source_pair["timestamps_ns"][anchor_index])
        ],
        "anchor_position_prior_role": "calibration",
        "anchor_position_prior_physically_present": True,
        "pair_id": int(source_pair["pair_id"]),
        "source_pair_image_ids": list(source_pair["image_ids"]),
        "source_pair_image_names": list(source_pair["image_names"]),
        "source_pair_frame_ids": list(source_pair["frame_ids"]),
        "source_pair_verified_matches": int(source_pair["verified_matches"]),
        "source_pair_retention_reason": str(
            source_pair["pair_retention_reason"]
        ),
        "source_pair_raw_gnss_evidence": source_pair["raw_gnss_evidence"],
        "source_pair_two_view_geometry": source_pair[
            "sealed_two_view_geometry"
        ],
        "source_pair_parallax_evidence": source_pair[
            "sealed_parallax_evidence"
        ],
        "source_pair_score": source_pair["score"],
        "required_boundary_margin_frames": int(
            source_pair["required_boundary_margin_frames"]
        ),
        "actual_boundary_margin_frames": int(
            source_pair["actual_boundary_margin_frames"]
        ),
        "eligible_anchor_source_pair_count": int(
            source_pair["eligible_candidate_count"]
        ),
        "robust_anchor_source_pair_count": int(
            source_pair["robust_candidate_count"]
        ),
        "required_verified_matches": int(
            partner_pool["required_verified_matches"]
        ),
        "candidate_pair_count": int(partner_pool["candidate_pair_count"]),
        "candidate_pair_records_sha256": str(
            partner_pool["candidate_pair_records_sha256"]
        ),
        "candidate_pool": partner_pool["candidate_pool"],
        "random_seed": config.refinement.random_seed,
        "heldout_position_priors_available_to_selector": False,
        "decision_inputs_exclude": [
            "finished_visual_model_pose",
            "finished_visual_model_residual",
            "heldout_evaluation_result",
        ],
    }


def _select_initial_pair_for_method(
    method: str,
    candidates: Sequence[PairCandidate],
    selected_frame_ids: Sequence[int],
    calibration_names: Sequence[str],
    config: GeodeticSubmapConfig,
    pair_sources: Mapping[str, Any],
) -> dict[str, Any]:
    if method == "v1":
        return _select_initial_pair(
            candidates, selected_frame_ids, calibration_names, config
        )
    if method == "v2":
        return _select_initial_pair_v2(
            candidates,
            selected_frame_ids,
            calibration_names,
            config,
            pair_sources.get("initial_geometry_by_pair_id", {}),
        )
    if method == "v3":
        return _select_initial_pair_v3(
            candidates,
            selected_frame_ids,
            calibration_names,
            config,
            pair_sources,
        )
    if method == "v4":
        return _select_initial_pair_v4(candidates, config)
    if method == "v5":
        return _select_initial_pair_v5(
            candidates,
            selected_frame_ids,
            calibration_names,
            config,
            pair_sources,
        )
    raise ArtifactError(f"unsupported initial-pair method: {method}")


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


def _resolved_private_prior_position(
    source_position: np.ndarray,
    segment_center: np.ndarray,
    *,
    calibrated_derivation: bool,
) -> tuple[np.ndarray, bool]:
    """Resolve one private prior target without touching the source database."""
    source = np.asarray(source_position, dtype=np.float64)
    center = np.asarray(segment_center, dtype=np.float64)
    if source.shape != (3,) or center.shape != (3,) or not (
        np.isfinite(source).all() and np.isfinite(center).all()
    ):
        raise ArtifactError("pose-prior positions must be finite three-vectors")
    if np.allclose(source, center, atol=1e-6, rtol=1e-9):
        return source.copy(), False
    if not calibrated_derivation:
        raise ArtifactError(
            "pose prior is not the segment's lever-arm-applied raw-GNSS "
            "camera centre"
        )
    return center.copy(), True


def _resolved_private_prior_covariance(
    source_covariance: np.ndarray,
    segment_covariance: np.ndarray,
    *,
    calibrated_derivation: bool,
) -> tuple[np.ndarray, bool]:
    """Resolve the covariance paired with a private calibrated target.

    A fixed-calibration segment may replace the rough camera/antenna lever-arm
    uncertainty that was used when the immutable frontend database was built.
    The private optimizer copy must use that same rederived camera-centre
    covariance.  The source frontend remains byte-for-byte unchanged.
    """

    source = np.asarray(source_covariance, dtype=np.float64)
    derived = np.asarray(segment_covariance, dtype=np.float64)
    for label, value in (("source", source), ("segment", derived)):
        if value.shape != (3, 3) or not np.isfinite(value).all():
            raise ArtifactError(f"{label} pose-prior covariance is invalid")
        value = 0.5 * (value + value.T)
        try:
            np.linalg.cholesky(value)
        except np.linalg.LinAlgError as exc:
            raise ArtifactError(
                f"{label} pose-prior covariance is not positive definite"
            ) from exc
        if label == "source":
            source = value
        else:
            derived = value
    if np.allclose(source, derived, atol=1.0e-12, rtol=1.0e-12):
        return source.copy(), False
    if not calibrated_derivation:
        raise ArtifactError(
            "pose-prior covariance differs without a sealed fixed calibration"
        )
    return derived.copy(), True


def _fixed_calibration_pose_priors(
    frontend: Path,
    reader: SegmentReader,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Recompute camera targets using the frontend's sealed prior policy.

    This deliberately reuses the normal frontend covariance implementation.
    Only the immutable segment's accepted fixed calibration changes; the raw
    GNSS observations and the frontend's status/floor policy stay fixed.
    """

    marker_path = frontend / "stages" / "pose_priors.json"
    if not marker_path.is_file():
        raise ArtifactError(
            "fixed-calibration prior rederivation needs the frontend "
            "pose-prior stage marker"
        )
    marker = _json(marker_path)
    inputs = marker.get("inputs")
    if (
        marker.get("schema_version") != 1
        or marker.get("stage") != "pose_priors"
        or marker.get("state") != "complete"
        or not isinstance(inputs, Mapping)
    ):
        raise ArtifactError("frontend pose-prior stage marker is invalid")
    try:
        settings = {
            "covariance_floor_m": float(inputs["covariance_floor_m"]),
            "max_covariance_m2": (
                None
                if inputs["max_covariance_m2"] is None
                else float(inputs["max_covariance_m2"])
            ),
            "min_fix_status": int(inputs["min_fix_status"]),
            "min_carrier_status": int(inputs["min_carrier_status"]),
            "max_source_residual_s": (
                None
                if inputs["max_source_residual_s"] is None
                else float(inputs["max_source_residual_s"])
            ),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(
            "frontend pose-prior stage settings are invalid"
        ) from exc
    priors, skipped, quality_counts, covariance_model = _trusted_priors(
        reader, **settings
    )
    return priors, {
        "schema_version": 1,
        "method": "normal_frontend_prior_rederivation_on_fixed_segment_v1",
        "source_pose_prior_stage_sha256": sha256_file(marker_path),
        "source_pose_prior_settings": settings,
        "derived_segment_meta_sha256": sha256_file(
            reader.root / "segment_meta.json"
        ),
        "derived_frames_sha256": sha256_file(reader.root / "frames.npz"),
        "derived_gnss_sha256": sha256_file(
            reader.root / "observations" / "gnss.npz"
        ),
        "camera_center_covariance_model": covariance_model,
        "n_rederived_priors": len(priors),
        "skipped_by_reason": skipped,
        "position_quality_counts": quality_counts,
    }


def _weight_and_split_priors(
    database: Path,
    frontend: Path,
    rows: Sequence[Mapping[str, Any]],
    reader: SegmentReader,
    gnss: Mapping[int, RawGnssEndpoint],
    gnss_records: Mapping[int, Mapping[str, Any]],
    gnss_temporal_audit: Mapping[str, Any],
    config: GeodeticSubmapConfig,
    temporal_blocks: int,
    predeclared_role_windows: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
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
    calibrated_derivation = reader.meta.get("fixed_calibration_derivation")
    calibrated_priors = bool(
        isinstance(calibrated_derivation, Mapping)
        and calibrated_derivation.get("kind")
        == "rtk_splat_fixed_calibration_segment"
        and calibrated_derivation.get("fixed_scale") == 1.0
        and calibrated_derivation.get("raw_observations_changed") is False
        and calibrated_derivation.get("frame_inventory_changed") is False
    )
    fixed_priors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    fixed_prior_derivation: dict[str, Any] | None = None
    if calibrated_priors:
        fixed_priors, fixed_prior_derivation = _fixed_calibration_pose_priors(
            frontend, reader
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
                    "gnss_temporal_filter": dict(gnss_records[frame_id]),
                }
                if endpoint.trusted_std_m() is None or weight <= 0.0:
                    remove_ids.append(int(prior_id))
                    temporal_reason = str(
                        gnss_records[frame_id].get("reason", "")
                    )
                    reason = (
                        temporal_reason
                        if not gnss_records[frame_id].get(
                            "retained_by_temporal_filter", False
                        )
                        else "raw_gnss_status_or_covariance"
                    )
                    records.append(
                        {**base, "role": "excluded", "reason": reason}
                    )
                    continue
                if calibrated_priors and frame_id not in fixed_priors:
                    remove_ids.append(int(prior_id))
                    records.append(
                        {
                            **base,
                            "role": "excluded",
                            "reason": "fixed_calibration_prior_preflight",
                        }
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
                source_covariance = _decode_vector(
                    covariance_blob, (3, 3), f"prior {name} covariance"
                )
                source_covariance = 0.5 * (
                    source_covariance + source_covariance.T
                )
                try:
                    np.linalg.cholesky(source_covariance)
                except np.linalg.LinAlgError as exc:
                    raise ArtifactError(
                        f"pose prior covariance is not positive definite: {name}"
                    ) from exc
                if frame_id not in centers:
                    raise ArtifactError(
                        "pose prior has no segment camera centre: " + name
                    )
                source_position = position.copy()
                segment_position = (
                    np.asarray(fixed_priors[frame_id][0], dtype=np.float64)
                    if calibrated_priors
                    else centers[frame_id]
                )
                position, position_replaced = _resolved_private_prior_position(
                    source_position,
                    segment_position,
                    calibrated_derivation=calibrated_priors,
                )
                covariance, covariance_replaced = (
                    _resolved_private_prior_covariance(
                        source_covariance,
                        np.asarray(
                            fixed_priors[frame_id][1], dtype=np.float64
                        ),
                        calibrated_derivation=True,
                    )
                    if calibrated_priors
                    else (source_covariance.copy(), False)
                )
                if position_replaced:
                    # The completed frontend remains immutable and retains its
                    # rough-extrinsic priors.  Only its already-private submap
                    # copy may receive the camera centres from an audited
                    # fixed-calibration segment.
                    connection.execute(
                        "UPDATE pose_priors SET position=? "
                        "WHERE pose_prior_id=?",
                        (
                            np.asarray(position, dtype="<f8").tobytes(),
                            int(prior_id),
                        ),
                    )
                weighted = covariance / weight
                eligible.append(
                    {
                        **base,
                        "prior_id": int(prior_id),
                        "image_id": int(image_id),
                        "position_m": position.tolist(),
                        "source_frontend_position_m": source_position.tolist(),
                        "position_replaced_from_sealed_fixed_calibration": bool(
                            position_replaced
                        ),
                        "source_covariance_m2": source_covariance.tolist(),
                        "calibrated_covariance_m2": covariance.tolist(),
                        "covariance_replaced_from_sealed_fixed_calibration": bool(
                            covariance_replaced
                        ),
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
            role_assignment = _prior_role_assignment(
                eligible,
                temporal_blocks,
                predeclared_role_windows,
            )
            calibration_names: list[str] = []
            holdout_names: list[str] = []
            for item in eligible:
                name = str(item["name"])
                role = role_assignment["role_by_name"][name]
                calibration = role == "calibration"
                item["role"] = role
                item["block_id"] = role_assignment["block_id_by_name"][name]
                if predeclared_role_windows is not None:
                    item["predeclared_local_role_evidence"] = (
                        role_assignment["local_role_evidence_by_name"][name]
                    )
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
        "schema_version": 4 if predeclared_role_windows is not None else 3,
        "method": (
            "raw_temporal_filter_then_predeclared_window_role_union_status_"
            "fixed_calibration_covariance_weighted_v4"
            if predeclared_role_windows is not None
            else "raw_temporal_filter_then_alternating_blocks_status_"
            "fixed_calibration_covariance_weighted_v3"
        ),
        "calibration_block_ids": role_assignment["calibration_block_ids"],
        "holdout_block_ids": role_assignment["holdout_block_ids"],
        "calibration_names": calibration_names,
        "holdout_names": holdout_names,
        "n_calibration": len(calibration_names),
        "n_holdout": len(holdout_names),
        "n_excluded": sum(item["role"] == "excluded" for item in records),
        "n_temporally_rejected": sum(
            not item["retained_by_temporal_filter"]
            for item in gnss_temporal_audit["observation_records"]
        ),
        "gnss_temporal_audit": dict(gnss_temporal_audit),
        "position_quality_weights": [
            list(item) for item in config.position_quality_weights
        ],
        "fixed_calibration_prior_derivation": fixed_prior_derivation,
        **(
            {
                "predeclared_prior_role_contract": role_assignment[
                    "contract"
                ]
            }
            if predeclared_role_windows is not None
            else {}
        ),
        "optimizer_uses": [
            "raw_gnss_derived_camera_position",
            "sealed_camera_center_covariance",
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


def _normalized_prior_role_windows(
    selected_frame_ids: Sequence[int],
    value: Sequence[Sequence[int]] | None,
) -> list[list[int]] | None:
    if value is None:
        return None
    selected = [int(frame_id) for frame_id in selected_frame_ids]
    selected_set = set(selected)
    windows: list[list[int]] = []
    for raw_window in value:
        window = [int(frame_id) for frame_id in raw_window]
        if (
            len(window) < 4
            or window != sorted(set(window))
            or not set(window) <= selected_set
        ):
            raise ArtifactError("invalid predeclared prior-role window")
        windows.append(window)
    if (
        not windows
        or len(windows) != len({tuple(window) for window in windows})
        or windows != sorted(windows, key=lambda item: (item[0], item[-1]))
        or any(
            not set(left) & set(right)
            for left, right in zip(windows, windows[1:])
        )
        or set().union(*(set(window) for window in windows)) != selected_set
    ):
        raise ArtifactError("predeclared prior-role windows do not cover selection")
    return windows


def _prior_role_assignment(
    eligible: Sequence[Mapping[str, Any]],
    temporal_blocks: int,
    predeclared_role_windows: Sequence[Sequence[int]] | None,
) -> dict[str, Any]:
    """Derive roles without consulting visual poses or evaluation residuals."""

    ordered = list(eligible)
    union_split = temporal_block_split(
        np.asarray(
            [int(item["timestamp_ns"]) for item in ordered], dtype=np.int64
        ),
        temporal_blocks=temporal_blocks,
    )
    block_id_by_name = {
        str(item["name"]): int(union_split.block_ids[index])
        for index, item in enumerate(ordered)
    }
    if predeclared_role_windows is None:
        role_by_name = {
            str(item["name"]): (
                "calibration"
                if bool(union_split.calibration_mask[index])
                else "holdout"
            )
            for index, item in enumerate(ordered)
        }
        return {
            "role_by_name": role_by_name,
            "block_id_by_name": block_id_by_name,
            "local_role_evidence_by_name": {},
            "calibration_block_ids": list(union_split.calibration_block_ids),
            "holdout_block_ids": list(union_split.holdout_block_ids),
            "contract": None,
        }

    by_frame = {int(item["frame_id"]): item for item in ordered}
    evidence_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    membership: dict[str, list[str]] = defaultdict(list)
    window_evidence: list[dict[str, Any]] = []
    for ordinal, frame_ids in enumerate(predeclared_role_windows):
        local = [
            by_frame[int(frame_id)]
            for frame_id in frame_ids
            if int(frame_id) in by_frame
        ]
        if len(local) < 4:
            raise ArtifactError(
                "predeclared prior-role window has fewer than four trusted priors"
            )
        local_split = temporal_block_split(
            np.asarray(
                [int(item["timestamp_ns"]) for item in local], dtype=np.int64
            ),
            temporal_blocks=temporal_blocks,
        )
        local_names: list[str] = []
        for index, item in enumerate(local):
            name = str(item["name"])
            role = (
                "calibration"
                if bool(local_split.calibration_mask[index])
                else "holdout"
            )
            membership[name].append(role)
            evidence_by_name[name].append(
                {
                    "window_ordinal": ordinal,
                    "role": role,
                    "block_id": int(local_split.block_ids[index]),
                }
            )
            local_names.append(name)
        window_evidence.append(
            {
                "window_ordinal": ordinal,
                "frame_ids_sha256": canonical_hash(
                    [int(frame_id) for frame_id in frame_ids]
                ),
                "eligible_names_sha256": canonical_hash(local_names),
                "eligible_count": len(local_names),
            }
        )
    names = [str(item["name"]) for item in ordered]
    if set(membership) != set(names):
        raise ArtifactError("predeclared prior-role membership is incomplete")
    role_by_name = {
        name: "calibration" if "calibration" in membership[name] else "holdout"
        for name in names
    }
    calibration_blocks = sorted(
        {
            block_id_by_name[name]
            for name, role in role_by_name.items()
            if role == "calibration"
        }
    )
    holdout_blocks = sorted(
        {
            block_id_by_name[name]
            for name, role in role_by_name.items()
            if role == "holdout"
        }
    )
    contract_body = {
        "schema_version": 1,
        "method": "calibration_in_any_window_holdout_in_every_window_v1",
        "decision_inputs": [
            "sealed_selected_frame_ids",
            "predeclared_window_frame_membership",
            "immutable_raw_gnss_status_covariance_and_timestamps",
        ],
        "decision_inputs_exclude": [
            "visual_pose",
            "finished_visual_model_residual",
            "heldout_evaluation_result",
        ],
        "physical_nonleakage_rule": (
            "an observation is held out only when it is held out in every "
            "predeclared optimizer window containing it"
        ),
        "window_count": len(predeclared_role_windows),
        "windows": window_evidence,
        "role_membership_sha256": canonical_hash(
            {name: membership[name] for name in names}
        ),
        "calibration_names_sha256": canonical_hash(
            [name for name in names if role_by_name[name] == "calibration"]
        ),
        "holdout_names_sha256": canonical_hash(
            [name for name in names if role_by_name[name] == "holdout"]
        ),
    }
    return {
        "role_by_name": role_by_name,
        "block_id_by_name": block_id_by_name,
        "local_role_evidence_by_name": evidence_by_name,
        "calibration_block_ids": calibration_blocks,
        "holdout_block_ids": holdout_blocks,
        "contract": {
            **contract_body,
            "contract_sha256": canonical_hash(contract_body),
        },
    }


def _evaluation_prior_contract(prior_split: Mapping[str, Any]) -> dict[str, Any]:
    derivation = prior_split.get("fixed_calibration_prior_derivation")
    return {
        "schema_version": 1,
        "target_quantity": "left_camera_center_in_sealed_local_enu_m",
        "target_source": "sealed_prior_split_records",
        "position_field": "position_m",
        "covariance_field": "optimizer_covariance_m2",
        "alignment_role": "calibration",
        "final_evaluation_role": "holdout",
        "optimizer_evaluator_target_identity_required": True,
        "heldout_priors_physically_absent": True,
        "raw_gnss_observations_mutated": False,
        "fixed_calibration_prior_derivation_sha256": (
            canonical_hash(derivation) if derivation is not None else None
        ),
    }


def _recomputed_evaluation_prior_records(
    frontend: Path,
    reader: SegmentReader,
    rows: Sequence[Mapping[str, Any]],
    gnss: Mapping[int, RawGnssEndpoint],
    gnss_records: Mapping[int, Mapping[str, Any]],
    config: GeodeticSubmapConfig,
    temporal_blocks: int,
    predeclared_role_windows: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Recompute sealed evaluation targets from immutable physical inputs."""

    left_names = {str(row["left_image"]["name"]) for row in rows}
    source_priors = _cartesian_camera_priors(
        frontend / "database.db", left_names
    )
    derivation = reader.meta.get("fixed_calibration_derivation")
    calibrated = bool(
        isinstance(derivation, Mapping)
        and derivation.get("kind") == "rtk_splat_fixed_calibration_segment"
        and derivation.get("fixed_scale") == 1.0
        and derivation.get("raw_observations_changed") is False
        and derivation.get("frame_inventory_changed") is False
    )
    fixed_priors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    fixed_derivation: dict[str, Any] | None = None
    if calibrated:
        fixed_priors, fixed_derivation = _fixed_calibration_pose_priors(
            frontend, reader
        )
    weights = dict(config.position_quality_weights)
    eligible: list[dict[str, Any]] = []
    for row in rows:
        name = str(row["left_image"]["name"])
        frame_id = int(row["frame_id"])
        endpoint = gnss[frame_id]
        weight = float(weights.get(endpoint.position_quality, 0.0))
        if (
            name not in source_priors
            or endpoint.trusted_std_m() is None
            or weight <= 0.0
            or (calibrated and frame_id not in fixed_priors)
        ):
            continue
        source_position, source_covariance = source_priors[name]
        if calibrated:
            position, position_replaced = _resolved_private_prior_position(
                source_position,
                fixed_priors[frame_id][0],
                calibrated_derivation=True,
            )
            covariance, covariance_replaced = (
                _resolved_private_prior_covariance(
                    source_covariance,
                    fixed_priors[frame_id][1],
                    calibrated_derivation=True,
                )
            )
        else:
            position = np.asarray(source_position, dtype=np.float64).copy()
            covariance = np.asarray(source_covariance, dtype=np.float64).copy()
            position_replaced = False
            covariance_replaced = False
        eligible.append(
            {
                "name": name,
                "frame_id": frame_id,
                "timestamp_ns": int(row["timestamp_ns"]),
                "position_quality": endpoint.position_quality,
                "fix_status": endpoint.fix_status,
                "carrier_status": endpoint.carrier_status,
                "status_weight": weight,
                "gnss_temporal_filter": dict(gnss_records[frame_id]),
                "position_m": position.tolist(),
                "source_frontend_position_m": np.asarray(
                    source_position, dtype=np.float64
                ).tolist(),
                "position_replaced_from_sealed_fixed_calibration": bool(
                    position_replaced
                ),
                "source_covariance_m2": np.asarray(
                    source_covariance, dtype=np.float64
                ).tolist(),
                "calibrated_covariance_m2": covariance.tolist(),
                "covariance_replaced_from_sealed_fixed_calibration": bool(
                    covariance_replaced
                ),
                "optimizer_covariance_m2": (covariance / weight).tolist(),
            }
        )
    eligible.sort(key=lambda item: item["timestamp_ns"])
    if len(eligible) < 4:
        raise ArtifactError(
            "fewer than four recomputed sealed evaluation priors"
        )
    role_assignment = _prior_role_assignment(
        eligible,
        temporal_blocks,
        predeclared_role_windows,
    )
    calibration_names: list[str] = []
    holdout_names: list[str] = []
    for item in eligible:
        name = str(item["name"])
        role = role_assignment["role_by_name"][name]
        item["role"] = role
        item["block_id"] = role_assignment["block_id_by_name"][name]
        if predeclared_role_windows is not None:
            item["predeclared_local_role_evidence"] = role_assignment[
                "local_role_evidence_by_name"
            ][name]
        (calibration_names if role == "calibration" else holdout_names).append(
            name
        )
    return {
        "calibration_names": calibration_names,
        "holdout_names": holdout_names,
        "fixed_calibration_prior_derivation": fixed_derivation,
        "predeclared_prior_role_contract": role_assignment["contract"],
        "records": eligible,
    }


def _verify_recomputed_evaluation_prior_records(
    prior_split: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> None:
    if (
        prior_split.get("calibration_names")
        != recomputed.get("calibration_names")
        or prior_split.get("holdout_names")
        != recomputed.get("holdout_names")
        or prior_split.get("fixed_calibration_prior_derivation")
        != recomputed.get("fixed_calibration_prior_derivation")
        or prior_split.get("predeclared_prior_role_contract")
        != recomputed.get("predeclared_prior_role_contract")
    ):
        raise ArtifactError("sealed evaluation prior derivation changed")
    recorded_by_name = {
        str(record.get("name")): record
        for record in prior_split.get("records", ())
        if isinstance(record, Mapping)
        and record.get("role") in {"calibration", "holdout"}
    }
    expected_records = list(recomputed.get("records", ()))
    if set(recorded_by_name) != {
        str(record["name"]) for record in expected_records
    }:
        raise ArtifactError("sealed evaluation prior inventory changed")
    for expected in expected_records:
        recorded = recorded_by_name[str(expected["name"])]
        if any(recorded.get(key) != value for key, value in expected.items()):
            raise ArtifactError("sealed evaluation prior evidence changed")


def _sealed_evaluation_priors(
    plan_root: Path,
    plan: Mapping[str, Any],
    allowed_names: set[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load the exact sealed targets shared by optimization and evaluation."""

    if int(plan.get("schema_version", 1)) < _SEALED_EVALUATION_PRIOR_SCHEMA_VERSION:
        # Preserve byte-equivalent audits for already-published legacy plans.
        return _cartesian_camera_priors(
            Path(str(plan["frontend_artifact"])) / "database.db",
            allowed_names,
        )
    split = _json(plan_root / "prior_split.json")
    if split.get("schema_version") not in {3, 4}:
        raise ArtifactError("sealed evaluation prior split schema changed")
    expected_contract = _evaluation_prior_contract(split)
    if plan.get("evaluation_prior_contract") != expected_contract:
        raise ArtifactError("sealed evaluation prior contract changed")
    records = split.get("records")
    if not isinstance(records, list):
        raise ArtifactError("sealed evaluation prior records are missing")
    expected_names = set(split.get("calibration_names", ())) | set(
        split.get("holdout_names", ())
    )
    priors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    roles: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ArtifactError("sealed evaluation prior record is invalid")
        role = str(record.get("role", ""))
        if role not in {"calibration", "holdout"}:
            continue
        name = str(record.get("name", ""))
        if not name or name in priors or name not in allowed_names:
            raise ArtifactError("sealed evaluation prior inventory changed")
        position = np.asarray(record.get("position_m"), dtype=np.float64)
        covariance = np.asarray(
            record.get("optimizer_covariance_m2"), dtype=np.float64
        )
        if (
            position.shape != (3,)
            or covariance.shape != (3, 3)
            or not np.isfinite(position).all()
            or not np.isfinite(covariance).all()
        ):
            raise ArtifactError("sealed evaluation prior is numerically invalid")
        covariance = 0.5 * (covariance + covariance.T)
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ArtifactError(
                "sealed evaluation prior covariance is not positive definite"
            ) from exc
        priors[name] = (position, covariance)
        roles[name] = role
    if (
        set(priors) != expected_names
        or set(split.get("calibration_names", ()))
        != {name for name, role in roles.items() if role == "calibration"}
        or set(split.get("holdout_names", ()))
        != {name for name, role in roles.items() if role == "holdout"}
    ):
        raise ArtifactError("sealed evaluation prior roles changed")
    if len(priors) < 3:
        raise ArtifactError("fewer than three sealed evaluation priors")
    return priors


def _verify_private_evaluation_prior_identity(
    database: Path,
    priors: Mapping[str, tuple[np.ndarray, np.ndarray]],
    calibration_names: Sequence[str],
    holdout_names: Sequence[str],
) -> None:
    """Prove calibration rows equal the sealed evaluator target exactly."""

    try:
        with _readonly_database(database, immutable=True) as connection:
            rows = list(
                connection.execute(
                    """
                    SELECT i.name, p.position, p.position_covariance
                    FROM pose_priors AS p
                    JOIN images AS i ON i.image_id=p.corr_data_id
                    ORDER BY i.image_id
                    """
                )
            )
    except sqlite3.Error as exc:
        raise ArtifactError(
            f"cannot verify optimizer/evaluator prior identity: {exc}"
        ) from exc
    actual_names = {str(name) for name, _, _ in rows}
    if actual_names != set(calibration_names) or actual_names & set(holdout_names):
        raise ArtifactError("optimizer/evaluator prior role inventory changed")
    for name, position_blob, covariance_blob in rows:
        name = str(name)
        expected_position, expected_covariance = priors[name]
        actual_position = _decode_vector(
            position_blob, (3,), f"private prior {name} position"
        )
        actual_covariance = _decode_vector(
            covariance_blob, (3, 3), f"private prior {name} covariance"
        )
        if not (
            np.array_equal(actual_position, expected_position)
            and np.array_equal(actual_covariance, expected_covariance)
        ):
            raise ArtifactError("optimizer/evaluator prior targets differ")


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
    _verified_input_context: tuple[Any, ...] | None = None,
    _defer_full_reaudit_until_execution: bool = False,
    _initial_pair_method: str = "v1",
    _predeclared_prior_role_windows: Sequence[Sequence[int]] | None = None,
) -> Path:
    """Atomically publish one immutable private-database execution plan."""
    input_context = (
        tuple(_verified_input_context)
        if _verified_input_context is not None
        else _input_context(frontend_artifact, completed_backend, segment)
    )
    if len(input_context) != 9:
        raise ArtifactError("invalid cached geodetic input context")
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
    ) = input_context
    if (
        Path(frontend_artifact).expanduser().resolve() != frontend
        or Path(completed_backend).expanduser().resolve() != source
        or Path(segment).expanduser().resolve() != segment_path
    ):
        raise ArtifactError("cached geodetic input context path changed")
    if _defer_full_reaudit_until_execution and _verified_input_context is None:
        raise ArtifactError("deferred re-audit requires a verified input context")
    if _initial_pair_method not in {"v1", "v2", "v3", "v4", "v5"}:
        raise ArtifactError("unsupported initial-pair method")
    selection_path, selection = _load_frame_selection(
        frame_selection, frontend, segment_path, manifest
    )
    predeclared_role_windows = _normalized_prior_role_windows(
        selection["frame_ids"], _predeclared_prior_role_windows
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
        gnss, gnss_records, gnss_temporal_audit = _temporally_filtered_gnss(
            reader,
            config.gnss_temporal_policy,
            selection["frame_ids"],
        )
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
            frontend,
            rows,
            reader,
            gnss,
            gnss_records,
            gnss_temporal_audit,
            config,
            mapper_config.alignment_temporal_blocks,
            predeclared_role_windows,
        )
        initial_pair = _select_initial_pair_for_method(
            _initial_pair_method,
            candidates,
            selection["frame_ids"],
            prior_split["calibration_names"],
            config,
            pair_sources,
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
        if (
            _initial_pair_method != "v4"
            and initial_pair["pair_id"] not in inventory["geometry_pair_ids"]
        ):
            raise ArtifactError(
                "sealed initial pair lacks private verified geometry evidence"
            )
        if (
            _initial_pair_method == "v5"
            and initial_pair["image_names"][0]
            not in inventory["calibration_prior_names"]
        ):
            raise ArtifactError(
                "sealed initialization anchor lacks a private position prior"
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
            "schema_version": _HARDENED_PLAN_SCHEMA_VERSION,
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
            "predeclared_prior_role_windows": predeclared_role_windows,
            "predeclared_prior_role_windows_sha256": (
                canonical_hash(predeclared_role_windows)
                if predeclared_role_windows is not None
                else None
            ),
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
            "evaluation_prior_contract": _evaluation_prior_contract(
                prior_split
            ),
            "gnss_temporal_audit_sha256": canonical_hash(
                gnss_temporal_audit
            ),
            "database_inventory_sha256": sha256_file(
                staging / "database_inventory.json"
            ),
            "initial_pair": initial_pair,
            "calibration_contract": calibration,
            "colmap": _colmap_binding(frontend),
            "optimizer_contract": {
                "implementation": "existing_rtk_refinement_pose_prior_mapper",
                "initialization_mode": "fresh",
                "sealed_initial_pair": _initial_pair_method
                in {"v1", "v2", "v3"},
                "sealed_initial_anchor": _initial_pair_method == "v5",
                "sealed_initialization_policy": True,
                "colmap_auto_initial_pair": _initial_pair_method == "v4",
                "colmap_auto_second_image": _initial_pair_method == "v5",
                "position_priors": "raw_gnss_covariance_status_weighted",
                "optimizer_evaluator_prior_identity": True,
                "evaluation_targets": (
                    "sealed_prior_split_position_and_optimizer_covariance"
                ),
                "heldout_priors_physically_absent": True,
                "temporally_rejected_priors_physically_absent": True,
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
        if not _defer_full_reaudit_until_execution:
            _input_context(frontend, source, segment_path)
        _load_frame_selection(selection_path, frontend, segment_path, manifest)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite geodetic plan: {output}")
        publish_directory_noreplace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if _defer_full_reaudit_until_execution:
        _verify_file_seal(output, "plan_seal.json", _PLAN_FILES)
    else:
        _plan_context(output)
    return output


def _plan_context(
    artifact: str | Path,
    *,
    require_hardened: bool = False,
    _verified_input_context: tuple[Any, ...] | None = None,
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
    schema_version = plan.get("schema_version")
    if (
        plan.get("kind") != _PLAN_KIND
        or schema_version not in _SUPPORTED_PLAN_SCHEMA_VERSIONS
    ):
        raise ArtifactError("invalid geodetic submap plan")
    legacy_schema = schema_version == 1
    if require_hardened and legacy_schema:
        raise ArtifactError(
            "legacy geodetic plan lacks a sealed initial pair and full-path "
            "acceptance policy; prepare a new immutable plan"
        )
    config = _config_from_record(
        plan.get("config"),
        legacy_schema=legacy_schema,
        legacy_temporal_policy=schema_version < 3,
    )
    input_context = (
        _input_context(
            plan.get("frontend_artifact", ""),
            plan.get("completed_backend", ""),
            plan.get("segment", ""),
        )
        if _verified_input_context is None
        else _verified_input_context
    )
    if len(input_context) != 9:
        raise ArtifactError("invalid cached geodetic input context")
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
    ) = input_context
    if (
        Path(str(plan.get("frontend_artifact", ""))).resolve() != frontend
        or Path(str(plan.get("completed_backend", ""))).resolve() != source
        or Path(str(plan.get("segment", ""))).resolve() != segment
    ):
        raise ArtifactError("cached input context refers to another plan")
    selection_path, selection = _load_frame_selection(
        plan.get("frame_selection", ""), frontend, segment, manifest
    )
    predeclared_role_windows = _normalized_prior_role_windows(
        selection["frame_ids"], plan.get("predeclared_prior_role_windows")
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
        or plan.get("predeclared_prior_role_windows_sha256")
        != (
            canonical_hash(predeclared_role_windows)
            if predeclared_role_windows is not None
            else None
        )
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
    pair_audit = _json(root / "pair_audit.json")
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
    if schema_version >= _SEALED_EVALUATION_PRIOR_SCHEMA_VERSION:
        evaluation_priors = _sealed_evaluation_priors(
            root, plan, set(expected_left)
        )
        _verify_private_evaluation_prior_identity(
            root / "database.db",
            evaluation_priors,
            calibration_names,
            holdout_names,
        )
    if schema_version >= 3:
        gnss, gnss_records, expected_gnss_audit = (
            _temporally_filtered_gnss(
                reader,
                config.gnss_temporal_policy,
                selection["frame_ids"],
            )
        )
        if (
            prior_split.get("gnss_temporal_audit") != expected_gnss_audit
            or plan.get("gnss_temporal_audit_sha256")
            != canonical_hash(expected_gnss_audit)
        ):
            raise ArtifactError("sealed raw-GNSS temporal audit changed")
        if schema_version >= _SEALED_EVALUATION_PRIOR_SCHEMA_VERSION:
            recomputed_priors = _recomputed_evaluation_prior_records(
                frontend,
                reader,
                rows,
                gnss,
                gnss_records,
                config,
                mapper_config.alignment_temporal_blocks,
                predeclared_role_windows,
            )
            _verify_recomputed_evaluation_prior_records(
                prior_split, recomputed_priors
            )
    else:
        gnss = _raw_gnss_endpoints(reader)
    candidates, pair_sources = _pair_candidates(
        frontend / "database.db", metadata, gnss
    )
    expected_pair_audit, _ = _pair_audit(candidates, config.pair_policy)
    if pair_audit != expected_pair_audit:
        raise ArtifactError("sealed raw-GNSS pair audit changed")
    if not legacy_schema:
        initial_pair_method = {
            "deterministic_interior_raw_gnss_seed_v1": "v1",
            "deterministic_interior_raw_gnss_two_view_seed_v2": "v2",
            "deterministic_interior_raw_gnss_two_view_parallax_seed_v3": "v3",
            "deterministic_colmap_auto_filtered_database_v4": "v4",
            "deterministic_calibration_anchor_colmap_partner_v5": "v5",
        }.get(str(plan.get("initial_pair", {}).get("method", "")))
        if initial_pair_method is None:
            raise ArtifactError("optimizer initial-pair method changed")
        expected_initial_pair = _select_initial_pair_for_method(
            initial_pair_method,
            candidates,
            selection["frame_ids"],
            calibration_names,
            config,
            pair_sources,
        )
        if plan.get("initial_pair") != expected_initial_pair:
            raise ArtifactError("sealed mapper initial-pair contract changed")
        if (
            initial_pair_method != "v4"
            and expected_initial_pair["pair_id"]
            not in inventory.get("geometry_pair_ids", ())
        ):
            raise ArtifactError("sealed mapper initial pair has no geometry")
        if (
            initial_pair_method == "v5"
            and expected_initial_pair["image_names"][0]
            not in inventory.get("calibration_prior_names", ())
        ):
            raise ArtifactError(
                "sealed initialization anchor has no private position prior"
            )
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
    if schema_version >= 3 and optimizer.get(
        "temporally_rejected_priors_physically_absent"
    ) is not True:
        raise ArtifactError("temporal GNSS exclusion contract changed")
    if schema_version >= _SEALED_EVALUATION_PRIOR_SCHEMA_VERSION and (
        optimizer.get("optimizer_evaluator_prior_identity") is not True
        or optimizer.get("evaluation_targets")
        != "sealed_prior_split_position_and_optimizer_covariance"
    ):
        raise ArtifactError("optimizer/evaluator prior contract changed")
    if not legacy_schema:
        auto_initialization = initial_pair_method == "v4"
        anchored_initialization = initial_pair_method == "v5"
        if auto_initialization and (
            optimizer.get("sealed_initialization_policy") is not True
            or optimizer.get("sealed_initial_pair") is not False
            or optimizer.get("colmap_auto_initial_pair") is not True
            or optimizer.get("sealed_initial_anchor") not in {None, False}
            or optimizer.get("colmap_auto_second_image") not in {None, False}
        ):
            raise ArtifactError("optimizer initial-pair contract changed")
        if anchored_initialization and (
            optimizer.get("sealed_initial_pair") is not False
            or optimizer.get("sealed_initial_anchor") is not True
            or optimizer.get("colmap_auto_initial_pair") is not False
            or optimizer.get("colmap_auto_second_image") is not True
            or optimizer.get("sealed_initialization_policy") is not True
        ):
            raise ArtifactError("optimizer initial-pair contract changed")
        if not auto_initialization and not anchored_initialization and (
            optimizer.get("sealed_initial_pair") is not True
            or optimizer.get("colmap_auto_initial_pair") not in {None, False}
            or optimizer.get("sealed_initial_anchor") not in {None, False}
            or optimizer.get("colmap_auto_second_image") not in {None, False}
            or optimizer.get("sealed_initialization_policy") not in {None, True}
        ):
            raise ArtifactError("optimizer initial-pair contract changed")
    # The seal is returned so result artifacts can bind the exact plan seal.
    return root, plan, config, mapper_config, {
        "source_quality": source_quality,
        "manifest": manifest,
        "rows": rows,
        "reader": reader,
        "seal": seal,
        "hardened_plan": not legacy_schema,
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
        if command.count(name) != 1:
            raise ValueError
        index = command.index(name)
        return str(command[index + 1])
    except (ValueError, IndexError) as exc:
        raise ArtifactError(f"pose-prior mapper command lacks {name}") from exc


def _verify_mapper_command(
    command: Sequence[str],
    config: GeodeticSubmapConfig,
    initial_pair: Mapping[str, Any],
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
    method = initial_pair.get("method")
    auto_initialization = (
        method == "deterministic_colmap_auto_filtered_database_v4"
    )
    anchored_initialization = (
        method == "deterministic_calibration_anchor_colmap_partner_v5"
    )
    if auto_initialization:
        if any(
            option in command
            for option in (
                "--Mapper.init_image_id1",
                "--Mapper.init_image_id2",
            )
        ):
            raise ArtifactError(
                "COLMAP auto-initialization command pins an image pair"
            )
    elif anchored_initialization:
        if "--Mapper.init_image_id2" in command:
            raise ArtifactError(
                "calibration-anchored command pins COLMAP's second image"
            )
        expected["--Mapper.init_image_id1"] = str(
            initial_pair["image_ids"][0]
        )
    else:
        expected.update(
            {
                "--Mapper.init_image_id1": str(initial_pair["image_ids"][0]),
                "--Mapper.init_image_id2": str(initial_pair["image_ids"][1]),
            }
        )
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


def _build_geodetic_submap_command_from_context(
    plan_root: Path,
    plan: Mapping[str, Any],
    config: GeodeticSubmapConfig,
    execution_workspace: str | Path,
    executable: str | Path,
) -> tuple[str, ...]:
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
    initial_method = plan["initial_pair"].get("method")
    if initial_method == "deterministic_calibration_anchor_colmap_partner_v5":
        command += (
            "--Mapper.init_image_id1",
            str(plan["initial_pair"]["image_ids"][0]),
        )
    elif initial_method != "deterministic_colmap_auto_filtered_database_v4":
        command += (
            "--Mapper.init_image_id1",
            str(plan["initial_pair"]["image_ids"][0]),
            "--Mapper.init_image_id2",
            str(plan["initial_pair"]["image_ids"][1]),
        )
    _verify_mapper_command(command, config, plan["initial_pair"])
    return command


def build_geodetic_submap_command(
    plan_artifact: str | Path,
    execution_workspace: str | Path,
    executable: str | Path,
) -> tuple[str, ...]:
    """Build and audit the existing fresh pose-prior-mapper command."""
    plan_root, plan, config, _, _ = _plan_context(
        plan_artifact, require_hardened=True
    )
    return _build_geodetic_submap_command_from_context(
        plan_root, plan, config, execution_workspace, executable
    )


def _copy_execution_input(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite execution input: {destination}")
    shutil.copyfile(source, destination)
    with destination.open("rb") as stream:
        os.fsync(stream.fileno())


def _maximum_true_run(values: np.ndarray) -> int:
    maximum = 0
    current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def _authoritative_checks_pass(checks: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        bool(check.get("passed"))
        for check in checks.values()
        if check.get("authoritative", True)
    )


def _metric_specific_source_repairs(
    source_rtk: Mapping[str, Any], refined_rtk: Mapping[str, Any]
) -> dict[str, Any]:
    """Require each failed source RTK metric to pass in the refined result."""
    source_checks = source_rtk.get("checks")
    refined_checks = refined_rtk.get("checks")
    if not isinstance(source_checks, Mapping) or not isinstance(
        refined_checks, Mapping
    ):
        raise ArtifactError("RTK quality records have no checks")
    failed_names = [
        str(name)
        for name, check in source_checks.items()
        if isinstance(check, Mapping)
        and check.get("authoritative", True)
        and not check.get("passed")
    ]
    repairs: list[dict[str, Any]] = []
    for name in failed_names:
        source = source_checks[name]
        refined = refined_checks.get(name)
        if not isinstance(refined, Mapping):
            raise ArtifactError(
                f"refined RTK quality lacks source metric: {name}"
            )
        source_value = source.get("value")
        refined_value = refined.get("value")
        if "maximum" in source:
            direction = "decrease"
            improved = bool(
                refined_value is not None
                and (
                    source_value is None
                    or float(refined_value) < float(source_value)
                )
            )
            limit = {"maximum": source.get("maximum")}
        elif "minimum" in source:
            direction = "increase"
            improved = bool(
                refined_value is not None
                and source_value is not None
                and float(refined_value) > float(source_value)
            )
            limit = {"minimum": source.get("minimum")}
        else:
            direction = "pass_same_metric"
            improved = bool(refined.get("passed"))
            limit = {}
        repaired = bool(refined.get("passed") and improved)
        repairs.append(
            {
                "metric": name,
                "direction": direction,
                "source_value": source_value,
                "refined_value": refined_value,
                **limit,
                "refined_same_metric_passed": bool(refined.get("passed")),
                "improved_in_required_direction": improved,
                "repaired": repaired,
            }
        )
    return {
        "source_failed_metrics": failed_names,
        "repairs": repairs,
        "passed": all(item["repaired"] for item in repairs),
    }


def _full_trajectory_quality(
    aligned_refined_centers: np.ndarray,
    camera_prior_centers: np.ndarray,
    raw_gnss_centers: np.ndarray,
    roles: Sequence[str],
    calibration_inlier_mask: np.ndarray,
    names: Sequence[str],
    frame_ids: Sequence[int],
    policy: GeodeticTrajectoryPolicy,
) -> dict[str, Any]:
    """Evaluate absolute local motion over every selected acquisition step."""
    refined = np.asarray(aligned_refined_centers, dtype=np.float64)
    camera = np.asarray(camera_prior_centers, dtype=np.float64)
    raw = np.asarray(raw_gnss_centers, dtype=np.float64)
    inliers = np.asarray(calibration_inlier_mask, dtype=bool)
    n = len(refined)
    if (
        n < 3
        or refined.shape != (n, 3)
        or camera.shape != (n, 3)
        or raw.shape != (n, 3)
        or inliers.shape != (n,)
        or len(roles) != n
        or len(names) != n
        or len(frame_ids) != n
        or not np.isfinite(refined).all()
    ):
        raise ArtifactError("full-trajectory gate inputs are inconsistent")

    eligible = np.asarray(
        [role in {"calibration", "holdout"} for role in roles], dtype=bool
    )
    camera_valid = np.isfinite(camera).all(axis=1) & eligible
    raw_valid = np.isfinite(raw).all(axis=1) & eligible
    eligible_coverage = bool(
        eligible.any()
        and camera_valid[eligible].all()
        and raw_valid[eligible].all()
    )
    calibration_outliers = np.asarray(
        [role == "calibration" for role in roles], dtype=bool
    ) & ~inliers
    maximum_outlier_run = _maximum_true_run(calibration_outliers)

    refined_steps = np.diff(refined, axis=0)
    refined_lengths = np.linalg.norm(refined_steps, axis=1)
    adjacent_valid = (
        eligible[:-1]
        & eligible[1:]
        & camera_valid[:-1]
        & camera_valid[1:]
        & raw_valid[:-1]
        & raw_valid[1:]
    )
    camera_steps = np.diff(camera, axis=0)
    raw_steps = np.diff(raw, axis=0)
    camera_vector_errors = np.linalg.norm(
        refined_steps - camera_steps, axis=1
    )
    raw_length_errors = np.abs(
        refined_lengths - np.linalg.norm(raw_steps, axis=1)
    )
    camera_vector_errors[~adjacent_valid] = np.nan
    raw_length_errors[~adjacent_valid] = np.nan

    def worst_pair(values: np.ndarray) -> dict[str, Any] | None:
        finite = np.flatnonzero(np.isfinite(values))
        if not finite.size:
            return None
        index = int(finite[np.argmax(values[finite])])
        return {
            "selection_indices": [index, index + 1],
            "frame_ids": [int(frame_ids[index]), int(frame_ids[index + 1])],
            "image_names": [str(names[index]), str(names[index + 1])],
            "error_m": float(values[index]),
            "refined_displacement_m": float(refined_lengths[index]),
            "camera_prior_displacement_m": float(
                np.linalg.norm(camera_steps[index])
            ),
            "raw_gnss_displacement_m": float(
                np.linalg.norm(raw_steps[index])
            ),
        }

    worst_camera_pair = worst_pair(camera_vector_errors)
    worst_raw_pair = worst_pair(raw_length_errors)
    maximum_camera_error = (
        float(np.nanmax(camera_vector_errors))
        if np.isfinite(camera_vector_errors).any()
        else None
    )
    maximum_raw_error = (
        float(np.nanmax(raw_length_errors))
        if np.isfinite(raw_length_errors).any()
        else None
    )

    window_frames = min(policy.local_window_frames, n)
    windows: list[dict[str, Any]] = []
    for start in range(n - window_frames + 1):
        stop = start + window_frames
        if not (
            camera_valid[start:stop].all()
            and raw_valid[start:stop].all()
        ):
            continue
        prior_path = float(
            np.linalg.norm(camera_steps[start : stop - 1], axis=1).sum()
        )
        refined_path = float(refined_lengths[start : stop - 1].sum())
        error = abs(refined_path - prior_path)
        allowed = max(
            policy.max_local_window_path_error_m,
            policy.max_local_window_path_relative_error * prior_path,
        )
        windows.append(
            {
                "selection_indices": [start, stop - 1],
                "frame_ids": [
                    int(frame_ids[start]),
                    int(frame_ids[stop - 1]),
                ],
                "image_names": [str(names[start]), str(names[stop - 1])],
                "prior_path_m": prior_path,
                "refined_path_m": refined_path,
                "absolute_error_m": error,
                "relative_error": (
                    error / prior_path if prior_path > 1.0e-12 else None
                ),
                "allowed_error_m": allowed,
                "normalized_error": error / allowed,
            }
        )
    worst_window = (
        max(windows, key=lambda item: item["normalized_error"])
        if windows
        else None
    )
    window_passed = (
        eligible_coverage
        and worst_window is not None
        and worst_window["normalized_error"] <= 1.0
    )
    checks = {
        "full_trajectory_raw_gnss_coverage": {
            "authoritative": True,
            "value": int((camera_valid & raw_valid).sum()),
            "expected": int(eligible.sum()),
            "excluded_observation_count": int((~eligible).sum()),
            "passed": eligible_coverage,
        },
        "consecutive_calibration_prior_outliers": {
            "authoritative": True,
            "value": maximum_outlier_run,
            "maximum": policy.max_consecutive_calibration_outliers,
            "passed": maximum_outlier_run
            <= policy.max_consecutive_calibration_outliers,
        },
        "adjacent_camera_prior_displacement_error_m": {
            "authoritative": True,
            "value": maximum_camera_error,
            "maximum": policy.max_adjacent_displacement_error_m,
            "passed": eligible_coverage
            and maximum_camera_error is not None
            and maximum_camera_error
            <= policy.max_adjacent_displacement_error_m,
        },
        "adjacent_raw_gnss_step_error_m": {
            "authoritative": True,
            "value": maximum_raw_error,
            "maximum": policy.max_adjacent_displacement_error_m,
            "passed": eligible_coverage
            and maximum_raw_error is not None
            and maximum_raw_error
            <= policy.max_adjacent_displacement_error_m,
        },
        "sliding_local_window_path_consistency": {
            "authoritative": True,
            "value": (
                worst_window["normalized_error"]
                if worst_window is not None
                else None
            ),
            "maximum": 1.0,
            "window_frames": window_frames,
            "passed": window_passed,
        },
    }
    return {
        "schema_version": 2,
        "method": "aligned_eligible_raw_gnss_path_gates_v2",
        "selected_frame_count": n,
        "eligible_observation_count": int(eligible.sum()),
        "excluded_observation_count": int((~eligible).sum()),
        "adjacent_step_count": n - 1,
        "evaluated_adjacent_step_count": int(adjacent_valid.sum()),
        "calibration_outlier_indices": np.flatnonzero(
            calibration_outliers
        ).astype(int).tolist(),
        "worst_adjacent_camera_prior_pair": worst_camera_pair,
        "worst_adjacent_raw_gnss_pair": worst_raw_pair,
        "worst_local_window": worst_window,
        "checks": checks,
    }


@dataclass(frozen=True)
class _SealedRoleEvaluation:
    """Calibration-only SE(3) fit for an already sealed role inventory."""

    alignment: Any
    calibration_mask: np.ndarray
    holdout_mask: np.ndarray
    residual_vectors_m: np.ndarray
    residuals_m: np.ndarray
    thresholds_m: np.ndarray
    calibration_inlier_mask: np.ndarray
    holdout_inlier_mask: np.ndarray


def _sealed_role_heldout_evaluation(
    poses: Mapping[str, tuple[np.ndarray, np.ndarray]],
    priors: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ordered_records: Sequence[Mapping[str, Any]],
    mapper_config: MapperConfig,
) -> tuple[dict[str, Any], _SealedRoleEvaluation]:
    """Evaluate the exact sealed roles without deriving a second split.

    A predeclared multi-window plan has already assigned each observation a
    calibration or holdout role using immutable timestamps and window
    membership.  Re-running ``temporal_block_split`` across the union would
    create a different experiment.  This path fits only the sealed
    calibration observations and touches the sealed holdout targets only
    after that transform is fixed.
    """

    names = [str(record.get("name", "")) for record in ordered_records]
    roles = [str(record.get("role", "")) for record in ordered_records]
    if (
        not names
        or len(set(names)) != len(names)
        or set(names) != set(priors)
        or any(name not in poses for name in names)
        or any(role not in {"calibration", "holdout"} for role in roles)
    ):
        raise ArtifactError("sealed evaluator role inventory changed")
    calibration_mask = np.asarray(
        [role == "calibration" for role in roles], dtype=bool
    )
    holdout_mask = np.asarray(
        [role == "holdout" for role in roles], dtype=bool
    )
    if calibration_mask.sum() < 3 or not holdout_mask.any():
        raise ArtifactError(
            "sealed holdout evaluation requires at least three calibration "
            "priors and one untouched holdout prior"
        )

    source = np.stack([poses[name][1] for name in names])
    target = np.stack([priors[name][0] for name in names])
    covariance = np.stack([priors[name][1] for name in names])
    alignment = estimate_rigid_alignment(
        source[calibration_mask],
        target[calibration_mask],
        covariance[calibration_mask],
        ransac_threshold_m=mapper_config.alignment_ransac_threshold_m,
        ransac_iterations=mapper_config.alignment_ransac_iterations,
        random_seed=mapper_config.random_seed,
    )
    residual_vectors = (
        source @ alignment.rotation.T + alignment.translation - target
    )
    residuals = np.linalg.norm(residual_vectors, axis=1)
    maximum_sigma = np.sqrt(
        np.maximum(np.linalg.eigvalsh(covariance)[:, -1], 1.0e-8)
    )
    thresholds = np.maximum(
        mapper_config.alignment_ransac_threshold_m, 3.0 * maximum_sigma
    )
    calibration_inliers = np.zeros(len(source), dtype=bool)
    calibration_inliers[calibration_mask] = alignment.inlier_mask
    holdout_inliers = holdout_mask & (residuals <= thresholds)
    evaluation = _SealedRoleEvaluation(
        alignment=alignment,
        calibration_mask=calibration_mask,
        holdout_mask=holdout_mask,
        residual_vectors_m=residual_vectors,
        residuals_m=residuals,
        thresholds_m=thresholds,
        calibration_inlier_mask=calibration_inliers,
        holdout_inlier_mask=holdout_inliers,
    )
    quality = rtk_residual_quality(
        residual_vectors[holdout_mask],
        covariance[holdout_mask],
        holdout_inliers[holdout_mask],
        config=mapper_config,
    )
    return quality, evaluation


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
    try:
        scale: float | None = _similarity_scale(
            source_centers, refined_centers
        )
    except ArtifactError:
        # This source-relative number is deliberately diagnostic-only. The
        # absolute raw-GNSS path gates below remain authoritative.
        scale = None
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
    evaluation_priors = _sealed_evaluation_priors(
        plan_root, plan, set(left_names)
    )
    priors = {
        name: value
        for name, value in evaluation_priors.items()
        if name in evaluation_names
    }
    ordered_records = [
        item
        for item in split["records"]
        if item.get("role") in {"calibration", "holdout"}
    ]
    role_contract = split.get("predeclared_prior_role_contract")
    if role_contract is None:
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
        evaluation_role_method = "single_window_temporal_block_split"
    else:
        source_rtk, source_evaluation = _sealed_role_heldout_evaluation(
            source_poses, priors, ordered_records, mapper_config
        )
        refined_rtk, refined_evaluation = _sealed_role_heldout_evaluation(
            refined_poses, priors, ordered_records, mapper_config
        )
        evaluation_role_method = str(role_contract["method"])
    if not (
        np.array_equal(
            source_evaluation.calibration_mask,
            refined_evaluation.calibration_mask,
        )
        and np.array_equal(
            source_evaluation.holdout_mask,
            refined_evaluation.holdout_mask,
        )
    ):
        raise ArtifactError("source/refined sealed GNSS roles differ")
    recorded_roles = [str(item["role"]) for item in ordered_records]
    expected_roles = [
        "calibration" if value else "holdout"
        for value in source_evaluation.calibration_mask
    ]
    if recorded_roles != expected_roles:
        raise ArtifactError("optimizer and evaluator GNSS splits differ")
    evaluation_inliers = {
        str(item["name"]): bool(
            refined_evaluation.calibration_inlier_mask[index]
        )
        for index, item in enumerate(ordered_records)
    }
    record_by_name = {
        str(item["name"]): item for item in split["records"]
    }
    raw_endpoints = _raw_gnss_endpoints(context["reader"])
    nan_position = np.full(3, np.nan, dtype=np.float64)
    camera_prior_centers = np.stack(
        [
            np.asarray(record_by_name[name]["position_m"], dtype=np.float64)
            if record_by_name.get(name, {}).get("role")
            in {"calibration", "holdout"}
            else nan_position
            for name in left_names
        ]
    )
    raw_gnss_centers = np.stack(
        [
            np.asarray(
                raw_endpoints[int(row["frame_id"])].position_m,
                dtype=np.float64,
            )
            if record_by_name.get(str(row["left_image"]["name"]), {}).get(
                "role"
            )
            in {"calibration", "holdout"}
            and raw_endpoints[int(row["frame_id"])].trusted_std_m()
            is not None
            else nan_position
            for row in rows
        ]
    )
    aligned_refined_centers = (
        refined_centers @ refined_evaluation.alignment.rotation.T
        + refined_evaluation.alignment.translation
    )
    trajectory = _full_trajectory_quality(
        aligned_refined_centers,
        camera_prior_centers,
        raw_gnss_centers,
        [
            str(record_by_name.get(name, {}).get("role", "excluded"))
            for name in left_names
        ],
        np.asarray(
            [evaluation_inliers.get(name, False) for name in left_names],
            dtype=bool,
        ),
        left_names,
        [int(row["frame_id"]) for row in rows],
        config.trajectory_policy,
    )
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
    source_repairs = _metric_specific_source_repairs(
        source_rtk, refined_rtk
    )
    temporal_audit = split.get("gnss_temporal_audit")
    temporal_audit_passed = bool(
        plan.get("schema_version", 1) < 3
        or (
            isinstance(temporal_audit, Mapping)
            and temporal_audit.get("passed")
        )
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
        "raw_gnss_temporal_filter_audit": {
            "authoritative": plan.get("schema_version", 1) >= 3,
            "value": temporal_audit_passed,
            "expected": True,
            "passed": temporal_audit_passed,
        },
        "rejected_and_heldout_priors_physically_absent": {
            "value": set(inventory["calibration_prior_names"])
            == set(split["calibration_names"]),
            "expected": True,
            "passed": set(inventory["calibration_prior_names"])
            == set(split["calibration_names"]),
        },
        "trajectory_similarity_scale": {
            "authoritative": False,
            "kind": "diagnostic_only_source_comparison",
            "value": scale,
            "expected": 1.0,
            "maximum_abs_deviation": (
                config.refinement.max_trajectory_scale_deviation
            ),
            "passed": scale is not None
            and abs(scale - 1.0)
            <= config.refinement.max_trajectory_scale_deviation,
        },
        **trajectory["checks"],
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
        "heldout_source_failed_metrics_repaired": {
            "authoritative": not source_rtk_passed,
            "value": source_repairs["repairs"],
            "source_failed_metrics": source_repairs[
                "source_failed_metrics"
            ],
            "expected": "each failed source metric passes after refinement",
            "passed": source_repairs["passed"],
        },
    }
    passed = _authoritative_checks_pass(checks)
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
            "trajectory_similarity_scale_authoritative": False,
            "full_trajectory": trajectory,
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
            "evaluation_role_method": evaluation_role_method,
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
    _verified_input_context: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    """Run one plan in staging and atomically publish an experimental result."""
    plan_root, plan, config, mapper_config, context = _plan_context(
        plan_artifact,
        require_hardened=True,
        _verified_input_context=_verified_input_context,
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
        # Reuse the fully audited context above. Calling the public builder here
        # would repeat the immutable source-database snapshot while the first
        # snapshot remains live for result evaluation.
        command = _build_geodetic_submap_command_from_context(
            plan_root, plan, config, staging, executable
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
        # Re-audit the immutable sources after COLMAP completes. This closes
        # the input-mutation race even when an assembly launcher supplied its
        # immediately preceding verified context for the preflight.
        post_solve_plan_context = _plan_context(plan_root)
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
            "sealed_initial_pair": plan["initial_pair"],
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
            "schema_version": _HARDENED_PLAN_SCHEMA_VERSION,
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
    return audited_geodetic_submap_result(
        output, _verified_plan_context=post_solve_plan_context
    )


def audited_geodetic_submap_result(
    result_artifact: str | Path,
    *,
    _include_internal_plan_context: bool = False,
    _verified_plan_context: tuple[Any, ...] | None = None,
    _verified_input_context: tuple[Any, ...] | None = None,
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
    plan_artifact = Path(str(result.get("plan_artifact", ""))).resolve()
    if (
        _verified_plan_context is not None
        and _verified_input_context is not None
    ):
        raise ArtifactError("result audit received conflicting cached contexts")
    plan_context = (
        _plan_context(
            plan_artifact,
            _verified_input_context=_verified_input_context,
        )
        if _verified_plan_context is None
        else _verified_plan_context
    )
    if len(plan_context) != 5 or Path(plan_context[0]).resolve() != plan_artifact:
        raise ArtifactError("cached result plan context refers to another plan")
    plan_root, _, _, _, _ = plan_context
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
    audited = {
        **result,
        "artifact": str(root),
        "quality": quality,
        "solve": _json(root / "reports" / "solve.json"),
        "result_seal_sha256": sha256_file(root / "result_seal.json"),
    }
    if _include_internal_plan_context:
        audited["_internal_plan_context"] = plan_context
    return audited


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
    "GeodeticInitialPairPolicy",
    "GeodeticSubmapConfig",
    "GeodeticTrajectoryPolicy",
    "audited_geodetic_submap_result",
    "build_geodetic_submap_command",
    "create_geodetic_frame_selection",
    "geodetic_submap_export_context",
    "prepare_geodetic_submap_plan",
    "run_geodetic_submap_plan",
]
