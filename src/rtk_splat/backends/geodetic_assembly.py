"""Sealed overlap validation and full-trajectory geodetic assembly.

This module composes accepted :mod:`geodetic_submap` results.  It does not
implement an optimizer: every local solve remains the pinned COLMAP
``pose_prior_mapper`` path already used by ``rtk_refinement``.  Assembly uses
only fixed-scale SE(3)-aligned local poses, deterministic overlap weights, and
GNSS observations that were physically absent from every contributing local
optimizer when they are used as final holdouts.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import shutil
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rtk_splat.backends.artifact_io import _atomic_json, _atomic_write, _json
from rtk_splat.backends.colmap_model import (
    _cartesian_camera_priors,
    _poses_from_images_txt,
)
from rtk_splat.backends.geodetic_submap import (
    GeodeticSubmapConfig,
    GeodeticTrajectoryPolicy,
    _authoritative_checks_pass,
    _config_from_record,
    _config_record,
    _full_trajectory_quality,
    _input_context,
    _load_frame_selection,
    _pair_candidates,
    _plan_context,
    _raw_gnss_endpoints,
    _selection_payload,
    _select_initial_pair_for_method,
    _selected_rows,
    audited_geodetic_submap_result,
    create_geodetic_frame_selection,
    prepare_geodetic_submap_plan,
)
from rtk_splat.backends.mapper import (
    _apply_world_alignment,
    _verify_pose_artifact,
)
from rtk_splat.backends.quality import (
    rtk_residual_quality,
    temporal_block_split,
)
from rtk_splat.backends.rtk_refinement import _heldout_evaluation
from rtk_splat.backends.pose_evidence import (
    verify_pose_georeferencing_artifact,
)
from rtk_splat.core.segment import SegmentReader, publish_directory_noreplace
from rtk_splat.frontends.artifact import (
    ArtifactError,
    FRONTEND_SEAL_FILE,
    canonical_hash,
    sha256_file,
)


_PLAN_KIND = "rtk_splat_geodetic_assembly_plan"
_EXPORT_KIND = "rtk_splat_geodetic_submap_pose_export"
_OVERLAP_KIND = "rtk_splat_geodetic_overlap_report"
_PLAN_FILES = (
    "frame_selection.json",
    "geodetic_assembly_plan.json",
    "global_holdout.json",
    "windows.json",
)


@dataclass(frozen=True)
class GeodeticOverlapPolicy:
    """Predeclared agreement gates for independently solved overlaps."""

    minimum_overlap_frames: int = 100
    max_median_center_disagreement_m: float = 0.10
    max_p95_center_disagreement_m: float = 0.20
    max_center_disagreement_m: float = 0.35
    max_median_rotation_disagreement_deg: float = 2.0
    max_p95_rotation_disagreement_deg: float = 5.0
    max_rotation_disagreement_deg: float = 10.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_overlap_frames, bool)
            or not isinstance(self.minimum_overlap_frames, int)
            or self.minimum_overlap_frames < 2
        ):
            raise ValueError("minimum_overlap_frames must be at least two")
        for name in (
            "max_median_center_disagreement_m",
            "max_p95_center_disagreement_m",
            "max_center_disagreement_m",
            "max_median_rotation_disagreement_deg",
            "max_p95_rotation_disagreement_deg",
            "max_rotation_disagreement_deg",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not (
            self.max_median_center_disagreement_m
            <= self.max_p95_center_disagreement_m
            <= self.max_center_disagreement_m
        ):
            raise ValueError("centre disagreement limits must be ordered")
        if not (
            self.max_median_rotation_disagreement_deg
            <= self.max_p95_rotation_disagreement_deg
            <= self.max_rotation_disagreement_deg
        ):
            raise ValueError("rotation disagreement limits must be ordered")


@dataclass(frozen=True)
class GeodeticAssemblyConfig:
    """Generic windowing, overlap, and final path policy."""

    submap_frames: int = 801
    overlap_frames: int = 201
    probe_submaps: int = 3
    blend_boundary_power: float = 2.0
    overlap_policy: GeodeticOverlapPolicy = field(
        default_factory=GeodeticOverlapPolicy
    )
    trajectory_policy: GeodeticTrajectoryPolicy = field(
        default_factory=GeodeticTrajectoryPolicy
    )
    submap_config: GeodeticSubmapConfig = field(
        default_factory=GeodeticSubmapConfig
    )

    def __post_init__(self) -> None:
        for name, minimum in (
            ("submap_frames", 5),
            ("overlap_frames", 2),
            ("probe_submaps", 2),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.overlap_frames >= self.submap_frames:
            raise ValueError("overlap_frames must be smaller than submap_frames")
        if self.overlap_policy.minimum_overlap_frames > self.overlap_frames:
            raise ValueError(
                "minimum overlap gate cannot exceed the requested overlap"
            )
        if (
            not math.isfinite(float(self.blend_boundary_power))
            or self.blend_boundary_power <= 0.0
        ):
            raise ValueError("blend_boundary_power must be finite and positive")


def _config_record_v1(config: GeodeticAssemblyConfig) -> dict[str, Any]:
    return asdict(config)


def _config_from_record_v1(value: Any) -> GeodeticAssemblyConfig:
    if not isinstance(value, Mapping):
        raise ArtifactError("geodetic assembly config must be an object")
    try:
        config = GeodeticAssemblyConfig(
            submap_frames=int(value["submap_frames"]),
            overlap_frames=int(value["overlap_frames"]),
            probe_submaps=int(value["probe_submaps"]),
            blend_boundary_power=float(value["blend_boundary_power"]),
            overlap_policy=GeodeticOverlapPolicy(
                **dict(value["overlap_policy"])
            ),
            trajectory_policy=GeodeticTrajectoryPolicy(
                **dict(value["trajectory_policy"])
            ),
            submap_config=_config_from_record(value["submap_config"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError("invalid geodetic assembly config") from exc
    normalized = _config_record_v1(config)
    normalized["submap_config"]["position_quality_weights"] = [
        list(item) for item in config.submap_config.position_quality_weights
    ]
    if normalized != dict(value):
        raise ArtifactError("geodetic assembly config schema changed")
    return config


def _window_starts(count: int, window: int, overlap: int) -> list[int]:
    if count < window:
        raise ArtifactError(
            f"selection has {count} frames but submaps require {window}"
        )
    stride = window - overlap
    starts = list(range(0, count - window + 1, stride))
    final = count - window
    if starts[-1] != final:
        starts.append(final)
    if starts != sorted(set(starts)):
        raise ArtifactError("submap window schedule is not unique and ordered")
    if starts[0] != 0 or starts[-1] + window != count:
        raise ArtifactError("submap schedule does not cover both boundaries")
    if any(second - first >= window for first, second in zip(starts, starts[1:])):
        raise ArtifactError("submap schedule contains a coverage gap")
    return starts


def _eligible_roles(
    rows: Sequence[Mapping[str, Any]],
    source_priors: Mapping[str, Any],
    endpoints: Mapping[int, Any],
    config: GeodeticSubmapConfig,
    temporal_blocks: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    weights = dict(config.position_quality_weights)
    eligible: list[tuple[str, int]] = []
    for row in rows:
        name = str(row["left_image"]["name"])
        endpoint = endpoints[int(row["frame_id"])]
        if (
            name in source_priors
            and endpoint.trusted_std_m() is not None
            and float(weights.get(endpoint.position_quality, 0.0)) > 0.0
        ):
            eligible.append((name, int(row["timestamp_ns"])))
    if len(eligible) < 4:
        raise ArtifactError("assembly window has fewer than four trusted priors")
    split = temporal_block_split(
        np.asarray([item[1] for item in eligible], dtype=np.int64),
        temporal_blocks=temporal_blocks,
    )
    calibration = [
        name
        for index, (name, _) in enumerate(eligible)
        if bool(split.calibration_mask[index])
    ]
    holdout = [
        name
        for index, (name, _) in enumerate(eligible)
        if bool(split.holdout_mask[index])
    ]
    block_by_name = {
        name: int(split.block_ids[index])
        for index, (name, _) in enumerate(eligible)
    }
    return calibration, holdout, block_by_name


def _build_window_records(
    rows: Sequence[Mapping[str, Any]],
    source_database: Path,
    reader: SegmentReader,
    mapper_config: Any,
    config: GeodeticAssemblyConfig,
    *,
    initial_pair_method: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    left_names = [str(row["left_image"]["name"]) for row in rows]
    source_priors = _cartesian_camera_priors(source_database, set(left_names))
    endpoints = _raw_gnss_endpoints(reader)
    frame_index = {
        int(frame_id): index
        for index, frame_id in enumerate(reader.frames["frame_id"].astype(int))
    }
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        frame_id = int(row["frame_id"])
        for camera, field_name in (
            ("left", "left_image"),
            ("right", "right_image"),
        ):
            metadata[str(row[field_name]["name"])] = {
                "frame_id": frame_id,
                "frame_index": frame_index[frame_id],
                "camera": camera,
                "timestamp_ns": int(row["timestamp_ns"]),
            }
    all_candidates, pair_sources = _pair_candidates(
        source_database, metadata, endpoints
    )
    starts = _window_starts(
        len(rows), config.submap_frames, config.overlap_frames
    )
    roles_by_name: dict[str, list[str]] = defaultdict(list)
    windows: list[dict[str, Any]] = []
    for ordinal, start in enumerate(starts):
        stop = start + config.submap_frames
        selected = rows[start:stop]
        calibration, holdout, block_by_name = _eligible_roles(
            selected,
            source_priors,
            endpoints,
            config.submap_config,
            mapper_config.alignment_temporal_blocks,
        )
        frame_id_set = {int(row["frame_id"]) for row in selected}
        window_candidates = [
            candidate
            for candidate in all_candidates
            if candidate.first_frame_id in frame_id_set
            and candidate.second_frame_id in frame_id_set
        ]
        initial_pair = _select_initial_pair_for_method(
            initial_pair_method,
            window_candidates,
            [int(row["frame_id"]) for row in selected],
            calibration,
            config.submap_config,
            pair_sources,
        )
        for name in calibration:
            roles_by_name[name].append("calibration")
        for name in holdout:
            roles_by_name[name].append("holdout")
        frame_ids = [int(row["frame_id"]) for row in selected]
        names = [str(row["left_image"]["name"]) for row in selected]
        windows.append(
            {
                "schema_version": 1,
                "window_id": f"submap-{ordinal:04d}",
                "ordinal": ordinal,
                "selection_start": start,
                "selection_stop_exclusive": stop,
                "frame_ids": frame_ids,
                "frame_ids_sha256": canonical_hash(frame_ids),
                "left_image_names_sha256": canonical_hash(names),
                "first_timestamp_ns": int(selected[0]["timestamp_ns"]),
                "last_timestamp_ns": int(selected[-1]["timestamp_ns"]),
                "calibration_names": calibration,
                "holdout_names": holdout,
                "block_id_by_eligible_name": block_by_name,
                "n_calibration": len(calibration),
                "n_holdout": len(holdout),
                "initial_pair": initial_pair,
            }
        )
    global_holdout = [
        name
        for name in left_names
        if roles_by_name.get(name)
        and all(role == "holdout" for role in roles_by_name[name])
    ]
    global_calibration = [
        name
        for name in left_names
        if "calibration" in roles_by_name.get(name, ())
    ]
    excluded = [
        name
        for name in left_names
        if name not in global_holdout and name not in global_calibration
    ]
    if excluded:
        raise ArtifactError(
            "full assembly requires a trusted raw-GNSS position prior for "
            f"every selected frame; missing {len(excluded)}"
        )
    global_record = {
        "schema_version": 1,
        "method": "intersection_of_local_temporal_holdouts_v1",
        "decision_inputs_exclude": [
            "finished_visual_model_residual",
            "submap_overlap_result",
            "heldout_evaluation_result",
        ],
        "physical_absence_rule": (
            "a global holdout must be a local holdout in every optimizer "
            "window containing that image"
        ),
        "holdout_names": global_holdout,
        "calibration_names": global_calibration,
        "excluded_names": excluded,
        "n_holdout": len(global_holdout),
        "n_calibration": len(global_calibration),
        "n_excluded": len(excluded),
        "role_membership_sha256": canonical_hash(
            {name: roles_by_name.get(name, []) for name in left_names}
        ),
    }
    if len(global_holdout) < 1 or len(global_calibration) < 3:
        raise ArtifactError("assembly split lacks calibration or holdout support")
    return windows, global_record


def _file_evidence(root: Path, names: Sequence[str]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for name in sorted(names):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ArtifactError(f"cannot seal missing or unsafe file: {name}")
        evidence[name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    body = {"schema_version": 1, "files": evidence}
    return {**body, "seal_sha256": canonical_hash(body)}


def _verify_file_evidence(
    root: Path, seal_name: str, expected_names: Sequence[str]
) -> dict[str, Any]:
    seal = _json(root / seal_name)
    expected = _file_evidence(root, expected_names)
    if seal != expected:
        raise ArtifactError(f"artifact file seal changed: {root}")
    return seal


def prepare_geodetic_assembly_plan(
    frontend_artifact: str | Path,
    completed_backend: str | Path,
    segment: str | Path,
    frame_selection: str | Path,
    destination: str | Path,
    *,
    config: GeodeticAssemblyConfig = GeodeticAssemblyConfig(),
) -> Path:
    """Seal deterministic overlapping windows and a leakage-free holdout."""
    (
        frontend,
        manifest,
        source,
        _source_plan,
        mapper_config,
        source_quality,
        segment_path,
        reader,
        source_evidence,
    ) = _input_context(frontend_artifact, completed_backend, segment)
    selection_path, selection = _load_frame_selection(
        frame_selection, frontend, segment_path, manifest
    )
    rows = _selected_rows(manifest, selection["frame_ids"])
    windows, global_holdout = _build_window_records(
        rows,
        frontend / "database.db",
        reader,
        mapper_config,
        config,
        initial_pair_method="v3",
    )
    if config.probe_submaps > len(windows):
        raise ArtifactError("probe_submaps exceeds the planned window count")
    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite assembly plan: {output}")
    for immutable in (frontend, source, segment_path):
        if output == immutable or immutable in output.parents:
            raise ArtifactError("assembly plan must be outside immutable inputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.writing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        shutil.copyfile(selection_path, staging / "frame_selection.json")
        _atomic_json(
            staging / "windows.json",
            {"schema_version": 1, "windows": windows},
        )
        _atomic_json(staging / "global_holdout.json", global_holdout)
        plan_body = {
            "schema_version": 3,
            "kind": _PLAN_KIND,
            "experimental": True,
            "frontend_artifact": str(frontend),
            "completed_backend": str(source),
            "segment": str(segment_path),
            "source_evidence": source_evidence,
            "source_quality_passed": bool(source_quality.get("passed")),
            "frontend_seal_sha256": sha256_file(
                frontend / FRONTEND_SEAL_FILE
            ),
            "frame_selection_sha256": sha256_file(
                staging / "frame_selection.json"
            ),
            "windows_sha256": sha256_file(staging / "windows.json"),
            "global_holdout_sha256": sha256_file(
                staging / "global_holdout.json"
            ),
            "selected_frame_count": len(rows),
            "selected_frame_ids_sha256": canonical_hash(
                [int(row["frame_id"]) for row in rows]
            ),
            "window_count": len(windows),
            "config": _config_record_v1(config),
        }
        plan_body["config"]["submap_config"][
            "position_quality_weights"
        ] = [list(item) for item in config.submap_config.position_quality_weights]
        plan = {**plan_body, "plan_sha256": canonical_hash(plan_body)}
        _atomic_json(staging / "geodetic_assembly_plan.json", plan)
        _atomic_json(
            staging / "plan_seal.json", _file_evidence(staging, _PLAN_FILES)
        )
        publish_directory_noreplace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _verify_file_evidence(output, "plan_seal.json", _PLAN_FILES)
    return output


def audited_geodetic_assembly_plan(
    artifact: str | Path,
    *,
    _include_runtime_context: bool = False,
) -> dict[str, Any]:
    """Recompute and verify an immutable assembly plan."""
    root = Path(artifact).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ArtifactError("geodetic assembly plan is missing or unsafe")
    expected = set(_PLAN_FILES) | {"plan_seal.json"}
    if {path.name for path in root.iterdir()} != expected:
        raise ArtifactError("geodetic assembly plan inventory changed")
    seal = _verify_file_evidence(root, "plan_seal.json", _PLAN_FILES)
    plan = _json(root / "geodetic_assembly_plan.json")
    schema_version = plan.get("schema_version")
    if plan.get("kind") != _PLAN_KIND or schema_version not in {1, 2, 3}:
        raise ArtifactError("invalid geodetic assembly plan")
    config = _config_from_record_v1(plan.get("config"))
    (
        frontend,
        manifest,
        source,
        _source_plan,
        mapper_config,
        source_quality,
        segment,
        reader,
        source_evidence,
    ) = _input_context(
        plan.get("frontend_artifact", ""),
        plan.get("completed_backend", ""),
        plan.get("segment", ""),
    )
    _, selection = _load_frame_selection(
        root / "frame_selection.json", frontend, segment, manifest
    )
    rows = _selected_rows(manifest, selection["frame_ids"])
    windows, global_holdout = _build_window_records(
        rows,
        frontend / "database.db",
        reader,
        mapper_config,
        config,
        initial_pair_method={1: "v1", 2: "v2", 3: "v3"}[schema_version],
    )
    recorded_windows = _json(root / "windows.json")
    recorded_holdout = _json(root / "global_holdout.json")
    body = dict(plan)
    digest = body.pop("plan_sha256", None)
    checks = (
        plan.get("source_evidence") == source_evidence,
        plan.get("source_quality_passed") is bool(source_quality.get("passed")),
        Path(str(plan.get("completed_backend", ""))).resolve() == source,
        plan.get("frame_selection_sha256")
        == sha256_file(root / "frame_selection.json"),
        plan.get("windows_sha256") == sha256_file(root / "windows.json"),
        plan.get("global_holdout_sha256")
        == sha256_file(root / "global_holdout.json"),
        plan.get("selected_frame_count") == len(rows),
        plan.get("window_count") == len(windows),
        recorded_windows == {"schema_version": 1, "windows": windows},
        recorded_holdout == global_holdout,
        digest == canonical_hash(body),
    )
    if not all(checks):
        raise ArtifactError("geodetic assembly plan binding changed")
    audited = {
        **plan,
        "artifact": str(root),
        "config_object": config,
        "windows": windows,
        "global_holdout": global_holdout,
        "plan_seal_sha256": sha256_file(root / "plan_seal.json"),
        "seal": seal,
    }
    if _include_runtime_context:
        audited["_runtime_context"] = (
            frontend,
            manifest,
            source,
            _source_plan,
            mapper_config,
            source_quality,
            segment,
            reader,
            source_evidence,
        )
    return audited


def _window_record(plan: Mapping[str, Any], window_id: str) -> dict[str, Any]:
    matches = [
        item for item in plan["windows"] if item["window_id"] == window_id
    ]
    if len(matches) != 1:
        raise ArtifactError(f"unknown assembly window: {window_id}")
    return matches[0]


def prepare_geodetic_assembly_window(
    assembly_plan: str | Path,
    window_id: str,
    workspace: str | Path,
    *,
    _audited_plan: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Prepare or verify one Stage-1-compatible local plan for a window."""
    assembly_path = Path(assembly_plan).expanduser().resolve()
    assembly = (
        dict(_audited_plan)
        if _audited_plan is not None
        else audited_geodetic_assembly_plan(assembly_path)
    )
    if Path(str(assembly.get("artifact", ""))).resolve() != assembly_path:
        raise ArtifactError("cached assembly plan refers to another artifact")
    window = _window_record(assembly, window_id)
    root = Path(workspace).expanduser().resolve()
    for immutable in (
        Path(assembly["frontend_artifact"]),
        Path(assembly["completed_backend"]),
        Path(assembly["segment"]),
    ):
        if root == immutable or immutable in root.parents:
            raise ArtifactError("assembly workspace must be outside immutable inputs")
    root.mkdir(parents=True, exist_ok=True)
    selection_path = root / "selection.json"
    plan_path = root / "plan"
    runtime_context = assembly.get("_runtime_context")
    if selection_path.exists():
        if runtime_context is not None:
            frontend, manifest = runtime_context[:2]
        else:
            frontend, manifest, _ = _input_context(
                assembly["frontend_artifact"],
                assembly["completed_backend"],
                assembly["segment"],
            )[:3]
        _load_frame_selection(
            selection_path,
            frontend,
            Path(assembly["segment"]),
            manifest,
        )
        if _json(selection_path).get("frame_ids") != window["frame_ids"]:
            raise ArtifactError("existing window selection differs from plan")
    else:
        _atomic_json(
            selection_path,
            _selection_payload(
                Path(assembly["frontend_artifact"]),
                Path(assembly["segment"]),
                window["frame_ids"],
            ),
        )
    if plan_path.exists():
        _plan_context(plan_path, require_hardened=True)
    else:
        prepare_geodetic_submap_plan(
            assembly["frontend_artifact"],
            assembly["completed_backend"],
            assembly["segment"],
            selection_path,
            plan_path,
            config=assembly["config_object"].submap_config,
            _verified_input_context=runtime_context,
            _defer_full_reaudit_until_execution=runtime_context is not None,
            _initial_pair_method=(
                {
                    1: "v1",
                    2: "v2",
                    3: "v3",
                }[assembly.get("schema_version")]
            ),
        )
    split = _json(plan_path / "prior_split.json")
    local_plan = _json(plan_path / "geodetic_submap_plan.json")
    if (
        split.get("calibration_names") != window["calibration_names"]
        or split.get("holdout_names") != window["holdout_names"]
        or local_plan.get("initial_pair") != window["initial_pair"]
    ):
        raise ArtifactError(
            "prepared local split or initialization differs from assembly plan"
        )
    binding = {
        "schema_version": 1,
        "assembly_plan": assembly["artifact"],
        "assembly_plan_seal_sha256": assembly["plan_seal_sha256"],
        "window_id": window_id,
        "frame_ids_sha256": window["frame_ids_sha256"],
        "selection_sha256": sha256_file(selection_path),
        "submap_plan_seal_sha256": sha256_file(plan_path / "plan_seal.json"),
    }
    binding_path = root / "window_binding.json"
    if binding_path.exists():
        if _json(binding_path) != binding:
            raise ArtifactError("existing window binding changed")
    else:
        _atomic_json(binding_path, binding)
    return {
        "workspace": str(root),
        "selection": str(selection_path),
        "plan": str(plan_path),
        "result": str(root / "result"),
        "window_id": window_id,
    }


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _aligned_submap_candidate(
    result_artifact: str | Path,
    *,
    _verified_input_context: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    """Load one accepted result and reproduce its calibration-only SE(3)."""
    result = audited_geodetic_submap_result(
        result_artifact,
        _include_internal_plan_context=True,
        _verified_input_context=_verified_input_context,
    )
    if not result.get("passed") or not result.get("publication_eligible"):
        raise ArtifactError("submap result did not pass its acceptance gates")
    (
        plan_root,
        plan,
        _config,
        mapper_config,
        context,
    ) = result.pop("_internal_plan_context")
    if plan.get("schema_version") == 1:
        raise ArtifactError("assembly refuses a legacy unhardened submap plan")
    rows = context["rows"]
    left_names = [str(row["left_image"]["name"]) for row in rows]
    refined_poses = _poses_from_images_txt(
        Path(result["artifact"]) / "refined_text" / "images.txt"
    )
    source_poses = _poses_from_images_txt(
        Path(plan["completed_backend"]) / "registered_text" / "images.txt"
    )
    missing = sorted(
        set(left_names) - set(refined_poses)
        | (set(left_names) - set(source_poses))
    )
    if missing:
        raise ArtifactError(f"accepted submap lacks left poses: {missing[:8]}")
    split = _json(plan_root / "prior_split.json")
    evaluation_names = {
        str(item["name"])
        for item in split["records"]
        if item.get("role") in {"calibration", "holdout"}
    }
    priors = _cartesian_camera_priors(
        Path(plan["frontend_artifact"]) / "database.db",
        set(left_names),
    )
    priors = {
        name: value for name, value in priors.items() if name in evaluation_names
    }
    refined_quality, refined_evaluation = _heldout_evaluation(
        refined_poses, rows, priors, mapper_config
    )
    source_quality, source_evaluation = _heldout_evaluation(
        source_poses, rows, priors, mapper_config
    )
    ordered_records = [
        item
        for item in split["records"]
        if item.get("role") in {"calibration", "holdout"}
    ]
    expected_roles = [
        "calibration" if value else "holdout"
        for value in refined_evaluation.calibration_mask
    ]
    if [str(item["role"]) for item in ordered_records] != expected_roles:
        raise ArtifactError("submap optimizer/evaluator prior roles changed")
    recorded_refined = result["quality"]["rtk_holdout"]["refined"]
    recorded_source = result["quality"]["rtk_holdout"]["source"]
    for calculated, recorded in (
        (refined_quality, recorded_refined),
        (source_quality, recorded_source),
    ):
        if not math.isclose(
            float(calculated["residual_m"]["median"]),
            float(recorded["residual_m"]["median"]),
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        ):
            raise ArtifactError("recomputed submap RTK evaluation changed")
    raw_viewmats = np.stack([refined_poses[name][0] for name in left_names])
    raw_centers = np.stack([refined_poses[name][1] for name in left_names])
    viewmats, centers = _apply_world_alignment(
        raw_viewmats,
        raw_centers,
        refined_evaluation.alignment.rotation,
        refined_evaluation.alignment.translation,
    )
    source_viewmats = np.stack([source_poses[name][0] for name in left_names])
    source_centers = np.stack([source_poses[name][1] for name in left_names])
    aligned_source_viewmats, aligned_source_centers = _apply_world_alignment(
        source_viewmats,
        source_centers,
        source_evaluation.alignment.rotation,
        source_evaluation.alignment.translation,
    )
    segment_reader = context["reader"]
    frame_index = {
        int(frame_id): index
        for index, frame_id in enumerate(
            segment_reader.frames["frame_id"].astype(int)
        )
    }
    raw_gnss_centers = np.stack(
        [
            np.asarray(
                segment_reader.frames["initial_camera_center_m"][
                    frame_index[int(row["frame_id"])]
                ],
                dtype=np.float64,
            )
            for row in rows
        ]
    )
    roles_by_name = {
        str(item["name"]): str(item["role"]) for item in split["records"]
    }
    roles = [roles_by_name.get(name, "excluded") for name in left_names]
    frame_ids = np.asarray(
        [int(row["frame_id"]) for row in rows], dtype=np.int64
    )
    timestamps = np.asarray(
        [int(row["timestamp_ns"]) for row in rows], dtype=np.int64
    )
    if (
        np.any(np.diff(frame_ids) <= 0)
        or np.any(np.diff(timestamps) <= 0)
        or not np.isfinite(viewmats).all()
        or not np.isfinite(centers).all()
    ):
        raise ArtifactError("aligned submap pose arrays are invalid")
    return {
        "result": result,
        "plan_root": plan_root,
        "plan": plan,
        "mapper_config": mapper_config,
        "frame_ids": frame_ids,
        "timestamps_ns": timestamps,
        "left_image_names": left_names,
        "viewmats": viewmats,
        "centers": centers,
        "source_viewmats": aligned_source_viewmats,
        "source_centers": aligned_source_centers,
        "raw_gnss_centers": raw_gnss_centers,
        "roles": roles,
        "calibration_names": list(split["calibration_names"]),
        "holdout_names": list(split["holdout_names"]),
        "refined_alignment": refined_evaluation.alignment,
        "source_alignment": source_evaluation.alignment,
    }


def _trajectory_csv(
    frame_ids: Sequence[int],
    timestamps_ns: Sequence[int],
    names: Sequence[str],
    roles: Sequence[str],
    trajectories: Mapping[str, np.ndarray],
) -> bytes:
    labels = list(trajectories)
    header = ["frame_id", "timestamp_ns", "left_image_name", "prior_role"]
    for label in labels:
        header.extend((f"{label}_x_m", f"{label}_y_m", f"{label}_z_m"))
    lines = [",".join(header)]
    for index, frame_id in enumerate(frame_ids):
        values = [
            str(int(frame_id)),
            str(int(timestamps_ns[index])),
            json.dumps(str(names[index])),
            str(roles[index]),
        ]
        for label in labels:
            values.extend(
                f"{float(value):.12g}"
                for value in trajectories[label][index]
            )
        lines.append(",".join(values))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _trajectory_svg(
    trajectories: Mapping[str, np.ndarray],
    *,
    title: str,
) -> bytes:
    colors = ("#111827", "#dc2626", "#2563eb", "#059669", "#7c3aed")
    arrays = [np.asarray(value, dtype=np.float64) for value in trajectories.values()]
    if not arrays or any(value.ndim != 2 or value.shape[1] != 3 for value in arrays):
        raise ArtifactError("trajectory plot inputs must have shape (N,3)")
    points = np.concatenate([value[:, :2] for value in arrays], axis=0)
    if not np.isfinite(points).all():
        raise ArtifactError("trajectory plot contains non-finite coordinates")
    low = points.min(axis=0)
    high = points.max(axis=0)
    span = np.maximum(high - low, 1.0e-6)
    width, height, margin = 1200.0, 800.0, 70.0
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])

    def project(values: np.ndarray) -> str:
        x = margin + (values[:, 0] - low[0]) * scale
        y = height - margin - (values[:, 1] - low[1]) * scale
        return " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(x, y, strict=True))

    body = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" '
        f'height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="35" font-family="sans-serif" font-size="22">'
        f'{html.escape(title)}</text>',
        f'<text x="{margin}" y="{height - 18}" font-family="sans-serif" '
        'font-size="14">local ENU east (m)</text>',
        f'<text x="18" y="{margin}" font-family="sans-serif" font-size="14" '
        'transform="rotate(-90 18 70)">local ENU north (m)</text>',
    ]
    for index, (label, values) in enumerate(trajectories.items()):
        color = colors[index % len(colors)]
        body.append(
            f'<polyline points="{project(values)}" fill="none" stroke="{color}" '
            'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        body.append(
            f'<line x1="{width - 260}" y1="{55 + 24 * index}" '
            f'x2="{width - 220}" y2="{55 + 24 * index}" stroke="{color}" '
            'stroke-width="3"/>'
        )
        body.append(
            f'<text x="{width - 210}" y="{60 + 24 * index}" '
            f'font-family="sans-serif" font-size="14">{html.escape(label)}</text>'
        )
    body.append("</svg>")
    return ("\n".join(body) + "\n").encode("utf-8")


def export_geodetic_submap_poses(
    result_artifact: str | Path,
    destination: str | Path,
) -> Path:
    """Publish aligned accepted submap poses and a sealed trajectory plot."""
    candidate = _aligned_submap_candidate(result_artifact)
    output = Path(destination).expanduser().resolve()
    for immutable in (
        Path(candidate["result"]["artifact"]),
        Path(candidate["plan_root"]),
    ):
        if output == immutable or immutable in output.parents:
            raise ArtifactError("submap export must be outside immutable inputs")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite submap export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.writing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        _atomic_save_npy(staging / "viewmats.npy", candidate["viewmats"])
        _atomic_save_npy(staging / "cam_centers.npy", candidate["centers"])
        _atomic_save_npy(staging / "frame_ids.npy", candidate["frame_ids"])
        _atomic_save_npy(staging / "timestamps_ns.npy", candidate["timestamps_ns"])
        _atomic_save_npy(
            staging / "left_image_names.npy",
            np.asarray(candidate["left_image_names"]),
        )
        trajectories = {
            "raw GNSS": candidate["raw_gnss_centers"],
            "source visual": candidate["source_centers"],
            "accepted refined": candidate["centers"],
        }
        _atomic_write(
            staging / "trajectory.csv",
            _trajectory_csv(
                candidate["frame_ids"],
                candidate["timestamps_ns"],
                candidate["left_image_names"],
                candidate["roles"],
                trajectories,
            ),
        )
        _atomic_write(
            staging / "trajectory.svg",
            _trajectory_svg(trajectories, title="Accepted geodetic submap"),
        )
        record = {
            "schema_version": 1,
            "kind": _EXPORT_KIND,
            "experimental": True,
            "publication_eligible": True,
            "production_poses_published": False,
            "source_result": candidate["result"]["artifact"],
            "source_result_seal_sha256": candidate["result"][
                "result_seal_sha256"
            ],
            "source_plan_seal_sha256": candidate["result"][
                "plan_seal_sha256"
            ],
            "n_frames": len(candidate["frame_ids"]),
            "coordinate_frame": "local_enu_from_calibration_only_fixed_se3",
            "alignment": {
                "production_scale": 1.0,
                "sim3_scale_applied": False,
                "source_rank": int(
                    candidate["refined_alignment"].source_rank
                ),
                "rotation_target_source": candidate[
                    "refined_alignment"
                ].rotation.tolist(),
                "translation_target_source_m": candidate[
                    "refined_alignment"
                ].translation.tolist(),
            },
        }
        _atomic_json(staging / "submap_pose.json", record)
        files = [path.name for path in staging.iterdir() if path.is_file()]
        _atomic_json(staging / "manifest.json", _file_evidence(staging, files))
        publish_directory_noreplace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _verify_submap_export_against_candidate(output, candidate)
    return output


def _verify_submap_export_against_candidate(
    root: Path, expected_candidate: Mapping[str, Any]
) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise ArtifactError("geodetic submap pose export is missing or unsafe")
    expected = {
        "viewmats.npy",
        "cam_centers.npy",
        "frame_ids.npy",
        "timestamps_ns.npy",
        "left_image_names.npy",
        "trajectory.csv",
        "trajectory.svg",
        "submap_pose.json",
    }
    if {path.name for path in root.iterdir()} != expected | {"manifest.json"}:
        raise ArtifactError("geodetic submap pose export inventory changed")
    _verify_file_evidence(root, "manifest.json", sorted(expected))
    record = _json(root / "submap_pose.json")
    if (
        record.get("kind") != _EXPORT_KIND
        or record.get("publication_eligible") is not True
        or record.get("production_poses_published") is not False
    ):
        raise ArtifactError("invalid geodetic submap pose export")
    source = expected_candidate["result"]
    if Path(str(record.get("source_result", ""))).resolve() != Path(
        source["artifact"]
    ):
        raise ArtifactError("submap pose export names another result")
    if source["result_seal_sha256"] != record.get("source_result_seal_sha256"):
        raise ArtifactError("submap pose export source binding changed")
    viewmats = np.load(root / "viewmats.npy", allow_pickle=False)
    centers = np.load(root / "cam_centers.npy", allow_pickle=False)
    if (
        viewmats.shape != (record["n_frames"], 4, 4)
        or centers.shape != (record["n_frames"], 3)
        or not np.allclose(np.linalg.inv(viewmats)[:, :3, 3], centers, atol=1e-8)
        or not np.allclose(viewmats, expected_candidate["viewmats"], atol=1e-12)
        or not np.allclose(centers, expected_candidate["centers"], atol=1e-12)
        or not np.array_equal(
            np.load(root / "frame_ids.npy", allow_pickle=False),
            expected_candidate["frame_ids"],
        )
        or not np.array_equal(
            np.load(root / "timestamps_ns.npy", allow_pickle=False),
            expected_candidate["timestamps_ns"],
        )
        or np.load(root / "left_image_names.npy", allow_pickle=False).tolist()
        != expected_candidate["left_image_names"]
    ):
        raise ArtifactError("submap export pose arrays disagree")
    return {**record, "artifact": str(root)}


def audited_geodetic_submap_pose_export(
    artifact: str | Path,
) -> dict[str, Any]:
    root = Path(artifact).expanduser().resolve()
    record = _json(root / "submap_pose.json")
    candidate = _aligned_submap_candidate(record.get("source_result", ""))
    return _verify_submap_export_against_candidate(root, candidate)


def _rotation_disagreement_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    relative = np.einsum("nij,nkj->nik", first, second)
    cosine = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cosine))


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ArtifactError("overlap metric values are empty or non-finite")
    return {
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "maximum": float(values.max()),
    }


def _evaluate_overlap_candidates(
    candidates: Sequence[Mapping[str, Any]],
    policy: GeodeticOverlapPolicy,
) -> dict[str, Any]:
    if len(candidates) < 2:
        raise ArtifactError("overlap evaluation needs at least two submaps")
    ordered = sorted(
        candidates,
        key=lambda item: (
            int(item["timestamps_ns"][0]),
            int(item["frame_ids"][0]),
            str(item["result"]["artifact"]),
        ),
    )
    if len({str(item["result"]["artifact"]) for item in ordered}) != len(ordered):
        raise ArtifactError("overlap evaluation contains a duplicate result")
    pairs: list[dict[str, Any]] = []
    for first, second in zip(ordered, ordered[1:]):
        first_by_id = {
            int(frame_id): index
            for index, frame_id in enumerate(first["frame_ids"])
        }
        second_by_id = {
            int(frame_id): index
            for index, frame_id in enumerate(second["frame_ids"])
        }
        shared = sorted(set(first_by_id) & set(second_by_id))
        if len(shared) < policy.minimum_overlap_frames:
            raise ArtifactError(
                "adjacent submaps do not satisfy the minimum overlap: "
                f"{len(shared)} < {policy.minimum_overlap_frames}"
            )
        first_indices = np.asarray([first_by_id[value] for value in shared])
        second_indices = np.asarray([second_by_id[value] for value in shared])
        first_names = [first["left_image_names"][index] for index in first_indices]
        second_names = [second["left_image_names"][index] for index in second_indices]
        if first_names != second_names or not np.array_equal(
            first["timestamps_ns"][first_indices],
            second["timestamps_ns"][second_indices],
        ):
            raise ArtifactError("overlap frame identity differs between submaps")
        center_errors = np.linalg.norm(
            first["centers"][first_indices] - second["centers"][second_indices],
            axis=1,
        )
        rotation_errors = _rotation_disagreement_deg(
            first["viewmats"][first_indices, :3, :3],
            second["viewmats"][second_indices, :3, :3],
        )
        center = _summary(center_errors)
        rotation = _summary(rotation_errors)
        checks = {
            "overlap_frame_count": {
                "value": len(shared),
                "minimum": policy.minimum_overlap_frames,
                "passed": len(shared) >= policy.minimum_overlap_frames,
            },
            "median_center_disagreement_m": {
                "value": center["median"],
                "maximum": policy.max_median_center_disagreement_m,
                "passed": center["median"]
                <= policy.max_median_center_disagreement_m,
            },
            "p95_center_disagreement_m": {
                "value": center["p95"],
                "maximum": policy.max_p95_center_disagreement_m,
                "passed": center["p95"] <= policy.max_p95_center_disagreement_m,
            },
            "maximum_center_disagreement_m": {
                "value": center["maximum"],
                "maximum": policy.max_center_disagreement_m,
                "passed": center["maximum"] <= policy.max_center_disagreement_m,
            },
            "median_rotation_disagreement_deg": {
                "value": rotation["median"],
                "maximum": policy.max_median_rotation_disagreement_deg,
                "passed": rotation["median"]
                <= policy.max_median_rotation_disagreement_deg,
            },
            "p95_rotation_disagreement_deg": {
                "value": rotation["p95"],
                "maximum": policy.max_p95_rotation_disagreement_deg,
                "passed": rotation["p95"]
                <= policy.max_p95_rotation_disagreement_deg,
            },
            "maximum_rotation_disagreement_deg": {
                "value": rotation["maximum"],
                "maximum": policy.max_rotation_disagreement_deg,
                "passed": rotation["maximum"]
                <= policy.max_rotation_disagreement_deg,
            },
        }
        pairs.append(
            {
                "first_result": str(first["result"]["artifact"]),
                "second_result": str(second["result"]["artifact"]),
                "first_result_seal_sha256": first["result"][
                    "result_seal_sha256"
                ],
                "second_result_seal_sha256": second["result"][
                    "result_seal_sha256"
                ],
                "frame_ids": shared,
                "center_disagreement_m": center,
                "rotation_disagreement_deg": rotation,
                "per_frame_center_disagreement_m": center_errors.tolist(),
                "per_frame_rotation_disagreement_deg": rotation_errors.tolist(),
                "checks": checks,
                "passed": _authoritative_checks_pass(checks),
            }
        )
    return {
        "schema_version": 1,
        "kind": _OVERLAP_KIND,
        "policy": asdict(policy),
        "result_count": len(ordered),
        "pair_count": len(pairs),
        "pairs": pairs,
        "passed": all(item["passed"] for item in pairs),
    }


def _overlap_csv(report: Mapping[str, Any]) -> bytes:
    lines = [
        "first_result,second_result,frame_id,center_disagreement_m,"
        "rotation_disagreement_deg"
    ]
    for pair in report["pairs"]:
        first = json.dumps(Path(pair["first_result"]).name)
        second = json.dumps(Path(pair["second_result"]).name)
        for frame_id, center, rotation in zip(
            pair["frame_ids"],
            pair["per_frame_center_disagreement_m"],
            pair["per_frame_rotation_disagreement_deg"],
            strict=True,
        ):
            lines.append(
                f"{first},{second},{int(frame_id)},{float(center):.12g},"
                f"{float(rotation):.12g}"
            )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _overlap_svg(report: Mapping[str, Any]) -> bytes:
    trajectories: dict[str, np.ndarray] = {}
    for index, pair in enumerate(report["pairs"]):
        frame = np.asarray(pair["frame_ids"], dtype=np.float64)
        center = np.asarray(
            pair["per_frame_center_disagreement_m"], dtype=np.float64
        )
        trajectories[f"overlap {index + 1} disagreement"] = np.column_stack(
            (frame, center, np.zeros_like(frame))
        )
    return _trajectory_svg(
        trajectories,
        title="Geodetic submap overlap centre disagreement",
    )


def publish_geodetic_overlap_report(
    result_artifacts: Sequence[str | Path],
    destination: str | Path,
    *,
    policy: GeodeticOverlapPolicy = GeodeticOverlapPolicy(),
) -> Path:
    if not result_artifacts:
        raise ArtifactError("overlap report requires submap results")
    first_result = Path(result_artifacts[0]).expanduser().resolve()
    first_record = _json(first_result / "geodetic_submap_result.json")
    first_plan = Path(str(first_record.get("plan_artifact", ""))).resolve()
    first_plan_record = _json(first_plan / "geodetic_submap_plan.json")
    shared_input_context = _input_context(
        first_plan_record.get("frontend_artifact", ""),
        first_plan_record.get("completed_backend", ""),
        first_plan_record.get("segment", ""),
    )
    candidates = [
        _aligned_submap_candidate(
            path, _verified_input_context=shared_input_context
        )
        for path in result_artifacts
    ]
    report = _evaluate_overlap_candidates(candidates, policy)
    output = Path(destination).expanduser().resolve()
    for candidate in candidates:
        immutable = Path(candidate["result"]["artifact"])
        if output == immutable or immutable in output.parents:
            raise ArtifactError("overlap report must be outside result inputs")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite overlap report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.writing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        _atomic_json(staging / "overlap.json", report)
        _atomic_write(staging / "overlap.csv", _overlap_csv(report))
        _atomic_write(staging / "overlap.svg", _overlap_svg(report))
        files = ["overlap.json", "overlap.csv", "overlap.svg"]
        _atomic_json(staging / "manifest.json", _file_evidence(staging, files))
        publish_directory_noreplace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _verify_overlap_report_files(output, report)
    return output


def _verify_overlap_report_files(
    root: Path, expected_report: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {"overlap.json", "overlap.csv", "overlap.svg"}
    if (
        not root.is_dir()
        or root.is_symlink()
        or {path.name for path in root.iterdir()} != expected | {"manifest.json"}
    ):
        raise ArtifactError("geodetic overlap report is missing or changed")
    _verify_file_evidence(root, "manifest.json", sorted(expected))
    report = _json(root / "overlap.json")
    if report.get("kind") != _OVERLAP_KIND:
        raise ArtifactError("invalid geodetic overlap report")
    if report != dict(expected_report):
        raise ArtifactError("geodetic overlap report content changed")
    return {**report, "artifact": str(root)}


def audited_geodetic_overlap_report(artifact: str | Path) -> dict[str, Any]:
    root = Path(artifact).expanduser().resolve()
    report = _json(root / "overlap.json")
    policy = GeodeticOverlapPolicy(**dict(report.get("policy", {})))
    candidates: list[dict[str, Any]] = []
    paths: list[str] = []
    for pair in report.get("pairs", []):
        for name in ("first_result", "second_result"):
            path = str(pair[name])
            if path not in paths:
                paths.append(path)
    candidates = [_aligned_submap_candidate(path) for path in paths]
    expected = _evaluate_overlap_candidates(candidates, policy)
    return _verify_overlap_report_files(root, expected)


def _project_rotation(value: np.ndarray) -> np.ndarray:
    u, _singular, vt = np.linalg.svd(value)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-9):
        raise ArtifactError("assembled rotation projection failed")
    return rotation


def _blend_candidates(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    boundary_power: float,
) -> tuple[np.ndarray, np.ndarray, list[list[str]], np.ndarray]:
    by_frame: dict[int, list[tuple[Mapping[str, Any], int]]] = defaultdict(list)
    for candidate in candidates:
        for local_index, frame_id in enumerate(candidate["frame_ids"]):
            by_frame[int(frame_id)].append((candidate, local_index))
    centers: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    contributors: list[list[str]] = []
    counts: list[int] = []
    for row in rows:
        frame_id = int(row["frame_id"])
        entries = by_frame.get(frame_id, [])
        if not entries:
            raise ArtifactError(f"assembled pose has no contributor for {frame_id}")
        weights: list[float] = []
        entry_centers: list[np.ndarray] = []
        entry_rotations: list[np.ndarray] = []
        labels: list[str] = []
        for candidate, local_index in entries:
            count = len(candidate["frame_ids"])
            boundary_distance = min(local_index + 1, count - local_index)
            weights.append(float(boundary_distance) ** boundary_power)
            entry_centers.append(candidate["centers"][local_index])
            entry_rotations.append(candidate["viewmats"][local_index, :3, :3])
            labels.append(str(candidate["result"]["artifact"]))
        normalized = np.asarray(weights, dtype=np.float64)
        normalized /= normalized.sum()
        centers.append(
            np.sum(np.stack(entry_centers) * normalized[:, None], axis=0)
        )
        rotations.append(
            _project_rotation(
                np.sum(
                    np.stack(entry_rotations) * normalized[:, None, None],
                    axis=0,
                )
            )
        )
        contributors.append(labels)
        counts.append(len(entries))
    center_array = np.stack(centers)
    rotation_array = np.stack(rotations)
    viewmats = np.repeat(np.eye(4)[None], len(rows), axis=0)
    viewmats[:, :3, :3] = rotation_array
    viewmats[:, :3, 3] = -np.einsum(
        "nij,nj->ni", rotation_array, center_array
    )
    recovered = np.linalg.inv(viewmats)[:, :3, 3]
    if not np.allclose(recovered, center_array, atol=1.0e-9, rtol=1.0e-9):
        raise ArtifactError("assembled view matrices and centres disagree")
    return viewmats, center_array, contributors, np.asarray(counts, dtype=np.int64)


def _load_assembly_candidates(
    assembly: Mapping[str, Any],
    submaps_root: Path,
    *,
    _verified_input_context: tuple[Any, ...] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    global_holdout = set(assembly["global_holdout"]["holdout_names"])
    for window in assembly["windows"]:
        workspace = submaps_root / window["window_id"]
        result_path = workspace / "result"
        candidate = _aligned_submap_candidate(
            result_path,
            _verified_input_context=_verified_input_context,
        )
        if candidate["frame_ids"].astype(int).tolist() != window["frame_ids"]:
            raise ArtifactError(
                f"{window['window_id']} result has the wrong frame inventory"
            )
        if (
            candidate["calibration_names"] != window["calibration_names"]
            or candidate["holdout_names"] != window["holdout_names"]
        ):
            raise ArtifactError(
                f"{window['window_id']} prior split differs from assembly plan"
            )
        names = set(candidate["left_image_names"])
        required_absent = global_holdout & names
        if (
            not required_absent <= set(candidate["holdout_names"])
            or required_absent & set(candidate["calibration_names"])
        ):
            raise ArtifactError(
                f"{window['window_id']} consumed a global held-out prior"
            )
        binding = _json(workspace / "window_binding.json")
        if (
            binding.get("assembly_plan_seal_sha256")
            != assembly["plan_seal_sha256"]
            or binding.get("window_id") != window["window_id"]
            or binding.get("frame_ids_sha256") != window["frame_ids_sha256"]
        ):
            raise ArtifactError(f"{window['window_id']} binding changed")
        candidate["window"] = window
        candidates.append(candidate)
    return candidates


def _direct_rtk_evaluation(
    centers: np.ndarray,
    names: Sequence[str],
    priors: Mapping[str, tuple[np.ndarray, np.ndarray]],
    holdout_names: Sequence[str],
    mapper_config: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    index_by_name = {name: index for index, name in enumerate(names)}
    if len(index_by_name) != len(names):
        raise ArtifactError("assembled image names are not unique")
    missing = sorted(set(holdout_names) - set(priors))
    if missing:
        raise ArtifactError(f"global holdout lacks source priors: {missing[:8]}")
    indices = np.asarray([index_by_name[name] for name in holdout_names])
    target = np.stack([priors[name][0] for name in holdout_names])
    covariance = np.stack([priors[name][1] for name in holdout_names])
    residual_vectors = centers[indices] - target
    sigma = np.sqrt(np.maximum(np.linalg.eigvalsh(covariance)[:, -1], 1.0e-8))
    thresholds = np.maximum(
        float(mapper_config.alignment_ransac_threshold_m), 3.0 * sigma
    )
    residuals = np.linalg.norm(residual_vectors, axis=1)
    inliers = residuals <= thresholds
    quality = rtk_residual_quality(
        residual_vectors, covariance, inliers, config=mapper_config
    )
    return quality, {
        "holdout_names": list(holdout_names),
        "residual_vectors_m": residual_vectors.tolist(),
        "residuals_m": residuals.tolist(),
        "thresholds_m": thresholds.tolist(),
        "euclidean_inlier_mask": inliers.tolist(),
    }


def _status_record() -> dict[str, Any]:
    return {
        "artifact_class": "production",
        "georeferencing_status": "PASSED",
        "metric_georeferencing_claim_eligible": True,
        "diagnostic_export_requested": False,
        "diagnostic_export_override_used": False,
    }


def publish_geodetic_full_pose_artifact(
    assembly_plan: str | Path,
    submaps_root: str | Path,
    output_root: str | Path,
    name: str,
) -> Path:
    """Blend every accepted window and publish one modern production pose."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"invalid pose artifact name: {name!r}")
    assembly = audited_geodetic_assembly_plan(
        assembly_plan, _include_runtime_context=True
    )
    root = Path(submaps_root).expanduser().resolve()
    candidates = _load_assembly_candidates(
        assembly,
        root,
        _verified_input_context=assembly["_runtime_context"],
    )
    overlap = _evaluate_overlap_candidates(
        candidates, assembly["config_object"].overlap_policy
    )
    (
        frontend,
        manifest,
        source,
        _source_plan,
        mapper_config,
        source_quality,
        segment,
        reader,
        source_evidence,
    ) = assembly["_runtime_context"]
    _, selection = _load_frame_selection(
        Path(assembly["artifact"]) / "frame_selection.json",
        frontend,
        segment,
        manifest,
    )
    rows = _selected_rows(manifest, selection["frame_ids"])
    viewmats, centers, contributors, contributor_counts = _blend_candidates(
        rows,
        candidates,
        boundary_power=assembly["config_object"].blend_boundary_power,
    )
    frame_ids = np.asarray(
        [int(row["frame_id"]) for row in rows], dtype=np.int64
    )
    timestamps = np.asarray(
        [int(row["timestamp_ns"]) for row in rows], dtype=np.int64
    )
    names = [str(row["left_image"]["name"]) for row in rows]
    priors = _cartesian_camera_priors(frontend / "database.db", set(names))
    holdout_names = list(assembly["global_holdout"]["holdout_names"])
    holdout_quality, holdout_detail = _direct_rtk_evaluation(
        centers, names, priors, holdout_names, mapper_config
    )
    holdout_passed = all(
        check["passed"]
        for check in holdout_quality["checks"].values()
        if check.get("authoritative", True)
    )
    frame_index = {
        int(frame_id): index
        for index, frame_id in enumerate(reader.frames["frame_id"].astype(int))
    }
    prior_centers = np.stack(
        [np.asarray(priors[name][0], dtype=np.float64) for name in names]
    )
    raw_centers = np.stack(
        [
            np.asarray(
                reader.frames["initial_camera_center_m"][
                    frame_index[int(row["frame_id"])]
                ],
                dtype=np.float64,
            )
            for row in rows
        ]
    )
    all_covariance = np.stack([priors[name][1] for name in names])
    all_sigma = np.sqrt(
        np.maximum(np.linalg.eigvalsh(all_covariance)[:, -1], 1.0e-8)
    )
    all_thresholds = np.maximum(
        float(mapper_config.alignment_ransac_threshold_m), 3.0 * all_sigma
    )
    all_inliers = np.linalg.norm(centers - prior_centers, axis=1) <= all_thresholds
    holdout_set = set(holdout_names)
    roles = ["holdout" if name in holdout_set else "calibration" for name in names]
    trajectory = _full_trajectory_quality(
        centers,
        prior_centers,
        raw_centers,
        roles,
        all_inliers,
        names,
        frame_ids.astype(int).tolist(),
        assembly["config_object"].trajectory_policy,
    )
    submap_calibration_passed = all(
        candidate["result"]["quality"]["checks"][
            "calibration_parameter_max_change"
        ]["passed"]
        and candidate["result"]["quality"]["checks"][
            "stereo_baseline_max_change_m"
        ]["passed"]
        for candidate in candidates
    )
    inventory_exact = (
        len(frame_ids) == assembly["selected_frame_count"]
        and frame_ids.astype(int).tolist() == selection["frame_ids"]
        and len(contributors) == len(frame_ids)
        and bool((contributor_counts >= 1).all())
    )
    rotations = viewmats[:, :3, :3]
    pose_integrity = bool(
        np.isfinite(viewmats).all()
        and np.isfinite(centers).all()
        and np.allclose(
            rotations @ np.swapaxes(rotations, 1, 2), np.eye(3), atol=2.0e-4
        )
        and np.allclose(np.linalg.det(rotations), 1.0, atol=2.0e-4)
    )
    global_nonleakage = all(
        set(holdout_names) & set(candidate["left_image_names"])
        <= set(candidate["holdout_names"])
        for candidate in candidates
    )
    checks = {
        "all_submaps_passed": {
            "value": all(candidate["result"]["passed"] for candidate in candidates),
            "expected": True,
            "passed": all(candidate["result"]["passed"] for candidate in candidates),
        },
        "all_adjacent_overlaps_passed": {
            "value": bool(overlap["passed"]),
            "expected": True,
            "passed": bool(overlap["passed"]),
        },
        "selected_frame_inventory_exact": {
            "value": inventory_exact,
            "expected": True,
            "passed": inventory_exact,
        },
        "global_holdout_physically_absent_from_every_optimizer": {
            "value": global_nonleakage,
            "expected": True,
            "passed": global_nonleakage,
        },
        "submap_fixed_calibration_and_baseline": {
            "value": submap_calibration_passed,
            "expected": True,
            "passed": submap_calibration_passed,
        },
        "assembled_pose_integrity": {
            "value": pose_integrity,
            "expected": True,
            "passed": pose_integrity,
        },
        "global_heldout_rtk_absolute_gates": {
            "value": holdout_passed,
            "expected": True,
            "passed": holdout_passed,
        },
        **trajectory["checks"],
    }
    passed = _authoritative_checks_pass(checks)
    if not passed:
        failed = [name for name, check in checks.items() if not check["passed"]]
        raise ArtifactError(
            "full geodetic pose failed acceptance gates: " + ", ".join(failed)
        )
    status = _status_record()
    output = Path(output_root).expanduser().resolve() / name
    for immutable in (frontend, source, segment):
        if output == immutable or immutable in output.parents:
            raise ArtifactError("pose artifact must be outside immutable inputs")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite pose artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{name}.writing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        _atomic_save_npy(staging / "viewmats.npy", viewmats)
        _atomic_save_npy(staging / "cam_centers.npy", centers)
        _atomic_save_npy(staging / "frame_ids.npy", frame_ids)
        _atomic_save_npy(staging / "timestamps_ns.npy", timestamps)
        _atomic_save_npy(staging / "left_image_names.npy", np.asarray(names))
        trajectories = {"raw GNSS": raw_centers, "assembled geodetic": centers}
        _atomic_write(
            staging / "trajectory.csv",
            _trajectory_csv(
                frame_ids, timestamps, names, roles, trajectories
            ),
        )
        _atomic_write(
            staging / "trajectory.svg",
            _trajectory_svg(trajectories, title="Full-field geodetic trajectory"),
        )
        _atomic_json(staging / "overlap.json", overlap)
        quality = {
            "schema_version": 2,
            **status,
            "stage": "geodetic_submap_assembly_quality",
            "passed": True,
            "rtk_alignment_passed": True,
            "n_frames": len(frame_ids),
            "n_submaps": len(candidates),
            "n_global_holdout": len(holdout_names),
            "source_visual_quality_passed": bool(source_quality.get("passed")),
            "checks": checks,
            "global_holdout_quality": holdout_quality,
            "global_holdout_detail": holdout_detail,
            "full_trajectory": trajectory,
            "overlap": overlap,
            "contributor_count": {
                "minimum": int(contributor_counts.min()),
                "median": float(np.median(contributor_counts)),
                "maximum": int(contributor_counts.max()),
            },
        }
        alignment = {
            "schema_version": 1,
            **status,
            "method": "calibration_only_fixed_se3_submaps_boundary_weighted_v1",
            "source_frame": "independent_colmap_submap_worlds",
            "target_frame": "local_enu_cartesian_pose_priors",
            "production_scale": 1.0,
            "sim3_scale_applied": False,
            "blend_boundary_power": assembly["config_object"].blend_boundary_power,
            "global_holdout_used_for_alignment_or_blending": False,
        }
        provenance = {
            "schema_version": 1,
            **status,
            "method": "sealed_overlapping_geodetic_submap_assembly_v1",
            "assembly_plan": assembly["artifact"],
            "assembly_plan_seal_sha256": assembly["plan_seal_sha256"],
            "frontend_artifact": str(frontend),
            "completed_backend": str(source),
            "segment": str(segment),
            "source_evidence": source_evidence,
            "submap_results": [
                {
                    "window_id": candidate["window"]["window_id"],
                    "artifact": candidate["result"]["artifact"],
                    "result_seal_sha256": candidate["result"][
                        "result_seal_sha256"
                    ],
                    "plan_seal_sha256": candidate["result"][
                        "plan_seal_sha256"
                    ],
                }
                for candidate in candidates
            ],
            "pose_convention": {
                "viewmats": "world_to_left_camera",
                "viewmat_camera_axes": "OpenCV_x_right_y_down_z_forward",
                "cam_centers": "left_camera_center_in_local_enu_m",
                "world_alignment": "fixed_scale_SE3_per_submap_only",
            },
        }
        georeferencing = {
            "schema_version": 1,
            **status,
            "fixed_scale_se3_applied": True,
            "rtk_alignment_passed": True,
            "global_holdout_physically_absent": True,
            "checks": checks,
            "warning": None,
        }
        _atomic_json(staging / "quality.json", quality)
        _atomic_json(staging / "alignment.json", alignment)
        _atomic_json(staging / "provenance.json", provenance)
        _atomic_json(staging / "georeferencing.json", georeferencing)
        file_evidence = {
            path.name: {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        _atomic_json(
            staging / "manifest.json",
            {
                "schema_version": 2,
                "name": name,
                "n_frames": len(frame_ids),
                **status,
                "files": file_evidence,
            },
        )
        publish_directory_noreplace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _verify_pose_artifact(output)
    verify_pose_georeferencing_artifact(output, expected_name=name)
    return output


def audited_geodetic_full_pose_artifact(
    artifact: str | Path,
) -> dict[str, Any]:
    root = Path(artifact).expanduser().resolve()
    manifest = _verify_pose_artifact(root)
    evidence = verify_pose_georeferencing_artifact(
        root, expected_name=root.name
    )
    provenance = _json(root / "provenance.json")
    assembly = audited_geodetic_assembly_plan(
        provenance["assembly_plan"], _include_runtime_context=True
    )
    if (
        assembly["plan_seal_sha256"]
        != provenance.get("assembly_plan_seal_sha256")
    ):
        raise ArtifactError("full pose assembly-plan binding changed")
    (
        frontend,
        manifest_source,
        _source,
        _source_plan,
        _mapper_config,
        _source_quality,
        segment,
        _reader,
        _source_evidence,
    ) = assembly["_runtime_context"]
    _, selection = _load_frame_selection(
        Path(assembly["artifact"]) / "frame_selection.json",
        frontend,
        segment,
        manifest_source,
    )
    rows = _selected_rows(manifest_source, selection["frame_ids"])
    if not provenance.get("submap_results"):
        raise ArtifactError("full pose provenance has no submap results")
    submaps_root = Path(provenance["submap_results"][0]["artifact"]).parent.parent
    candidates = _load_assembly_candidates(
        assembly,
        submaps_root,
        _verified_input_context=assembly["_runtime_context"],
    )
    recorded_results = {
        str(item["artifact"]): item
        for item in provenance.get("submap_results", [])
    }
    if len(recorded_results) != len(candidates):
        raise ArtifactError("full pose submap provenance inventory changed")
    for candidate in candidates:
        result = candidate["result"]
        recorded = recorded_results.get(str(result["artifact"]))
        if (
            recorded is None
            or result["result_seal_sha256"]
            != recorded.get("result_seal_sha256")
            or result["plan_seal_sha256"]
            != recorded.get("plan_seal_sha256")
        ):
            raise ArtifactError("full pose submap-result binding changed")
    expected_viewmats, expected_centers, _contributors, _counts = (
        _blend_candidates(
            rows,
            candidates,
            boundary_power=assembly["config_object"].blend_boundary_power,
        )
    )
    viewmats = np.load(root / "viewmats.npy", allow_pickle=False)
    centers = np.load(root / "cam_centers.npy", allow_pickle=False)
    quality = _json(root / "quality.json")
    expected_overlap = _evaluate_overlap_candidates(
        candidates, assembly["config_object"].overlap_policy
    )
    if (
        not np.allclose(viewmats, expected_viewmats, atol=1.0e-12)
        or not np.allclose(centers, expected_centers, atol=1.0e-12)
        or quality.get("overlap") != expected_overlap
        or _json(root / "overlap.json") != expected_overlap
    ):
        raise ArtifactError("full pose no longer matches its sealed submaps")
    return {
        "artifact": str(root),
        "manifest": manifest,
        "georeferencing": evidence,
        "quality": quality,
    }


__all__ = [
    "GeodeticAssemblyConfig",
    "GeodeticOverlapPolicy",
    "audited_geodetic_assembly_plan",
    "audited_geodetic_full_pose_artifact",
    "audited_geodetic_overlap_report",
    "audited_geodetic_submap_pose_export",
    "export_geodetic_submap_poses",
    "prepare_geodetic_assembly_plan",
    "prepare_geodetic_assembly_window",
    "publish_geodetic_full_pose_artifact",
    "publish_geodetic_overlap_report",
]
