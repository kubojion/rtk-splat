"""Optional RTK-constrained refinement of a completed stereo SfM model.

Only calibration temporal blocks are present in the private optimizer
database.  Alternating blocks remain untouched until quality evaluation.
The source mapper workspace is always treated as immutable.

COLMAP's pose-prior mapper internally uses a Sim(3) initialization and then
restores scale from the configured rig.  Consequently this sidecar freezes
intrinsics and rig calibration and rejects any output whose stereo baseline,
calibration, or independently measured trajectory scale changes beyond the
configured tolerances.  It must not be described as an SE(3)-only optimizer.
"""

from __future__ import annotations

import math
import os
import shutil
import sqlite3
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rtk_splat.backends.mapper import (
    MapperConfig,
    Runner,
    _analyze_model,
    _atomic_json,
    _atomic_write,
    _cartesian_camera_priors,
    _complete_stage,
    _execute,
    _json,
    _model_candidates,
    _next_solve_attempt,
    _poses_from_images_txt,
    _report_path,
    _run_injected_mapper,
    _run_monitored_mapper,
    _solve_resource_paths,
    _tree_manifest,
    _verify_tree,
    _workspace_context,
    estimate_temporal_heldout_alignment,
    quality_summary,
    registered_names_from_images_txt,
    rtk_residual_quality,
    temporal_block_split,
)
from rtk_splat.core.segment import publish_directory_noreplace
from rtk_splat.frontends.artifact import (
    ArtifactError,
    StageLedger,
    canonical_hash,
    create_database_snapshot,
    sha256_file,
    sqlite_logical_record,
)


_PREPARE_CONTROL_OUTPUTS = (
    "refinement_plan.json",
    "source_database_snapshot.json",
    "optimizer_database.json",
    "prior_split.json",
    "all_images.txt",
    "constant_cameras.txt",
    "constant_rigs.txt",
    "calibration_prior_names.txt",
    "holdout_prior_names.txt",
)


def _verify_stage_output_hashes(
    root: Path, stage: str, required_outputs: Sequence[str]
) -> dict[str, Any]:
    """Verify selected sealed outputs without rehashing a multi-GB database."""
    marker = _json(root / "stages" / f"{stage}.json")
    outputs = marker.get("outputs")
    if marker.get("state") != "complete" or not isinstance(outputs, Mapping):
        raise ArtifactError(f"RTK refinement {stage} stage is incomplete")
    for relative in required_outputs:
        expected = outputs.get(relative)
        path = root / relative
        if (
            not isinstance(expected, str)
            or not path.is_file()
            or sha256_file(path) != expected
        ):
            raise ArtifactError(
                f"RTK refinement sealed {stage} output changed: {relative}"
            )
    return marker


@dataclass(frozen=True)
class RtkRefinementConfig:
    """Bounded controls for one mapper-neutral RTK position-prior solve."""

    num_threads: int = 8
    random_seed: int = 7
    ba_global_max_num_iterations: int = 50
    ba_global_max_refinements: int = 3
    initialization_mode: str = "continuation"
    prior_position_loss: str = "cauchy"
    prior_position_loss_scale: float = 2.795
    process_nice: int = 10
    resource_sample_interval_s: float = 2.0
    minimum_available_memory_gb: float = 4.0
    low_memory_consecutive_samples: int = 2
    minimum_free_space_gb: float = 10.0
    minimum_runtime_free_space_gb: float = 5.0
    max_reprojection_regression_px: float = 0.10
    min_track_length_retention: float = 0.90
    fresh_min_mean_track_length: float = 3.0
    fresh_min_mean_observations_per_image: float = 500.0
    max_trajectory_scale_deviation: float = 0.005
    max_stereo_baseline_change_m: float = 5.0e-5
    max_calibration_parameter_change: float = 1.0e-9
    max_holdout_median_regression_m: float = 0.01
    min_holdout_median_improvement_m: float = 0.005

    def __post_init__(self) -> None:
        if self.initialization_mode not in {"continuation", "fresh"}:
            raise ValueError(
                "initialization_mode must be 'continuation' or 'fresh'"
            )
        if self.prior_position_loss not in {"cauchy", "trivial"}:
            raise ValueError(
                "prior_position_loss must be 'cauchy' or 'trivial'"
            )
        for name in (
            "num_threads",
            "ba_global_max_num_iterations",
            "ba_global_max_refinements",
            "low_memory_consecutive_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.random_seed, bool) or not isinstance(
            self.random_seed, int
        ):
            raise ValueError("random_seed must be an integer")
        if (
            isinstance(self.process_nice, bool)
            or not isinstance(self.process_nice, int)
            or not 0 <= self.process_nice <= 19
        ):
            raise ValueError("process_nice must be an integer in [0, 19]")
        positive = (
            "prior_position_loss_scale",
            "resource_sample_interval_s",
            "minimum_available_memory_gb",
            "minimum_free_space_gb",
            "minimum_runtime_free_space_gb",
            "max_reprojection_regression_px",
            "fresh_min_mean_track_length",
            "fresh_min_mean_observations_per_image",
            "max_trajectory_scale_deviation",
            "max_stereo_baseline_change_m",
            "max_calibration_parameter_change",
            "max_holdout_median_regression_m",
            "min_holdout_median_improvement_m",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_runtime_free_space_gb > self.minimum_free_space_gb:
            raise ValueError(
                "minimum_runtime_free_space_gb cannot exceed "
                "minimum_free_space_gb"
            )
        if not 0 < self.min_track_length_retention <= 1:
            raise ValueError("min_track_length_retention must be in (0, 1]")


def _normalized_refinement_config_record(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Hydrate legacy defaults without rewriting a sealed plan."""
    record = dict(value)
    record.setdefault("prior_position_loss", "cauchy")
    record.setdefault("initialization_mode", "continuation")
    record.setdefault("fresh_min_mean_track_length", 3.0)
    record.setdefault("fresh_min_mean_observations_per_image", 500.0)
    return record


def _source_context(
    source_backend: str | Path,
) -> tuple[Path, dict[str, Any], MapperConfig, dict[str, Any]]:
    root, plan, mapper_config = _workspace_context(source_backend)
    quality_marker = root / "stages" / "quality.json"
    if not quality_marker.is_file():
        raise ArtifactError("RTK refinement requires completed backend quality")
    quality = _json(_report_path(root, "quality"))
    if not quality.get("passed") or quality.get("registration_fraction") != 1.0:
        raise ArtifactError(
            "RTK refinement requires a fully registered source model that "
            "passed visual quality"
        )
    for directory, manifest_name in (
        ("registered_model", "registered_model_manifest.json"),
        ("registered_text", "text_model_manifest.json"),
    ):
        manifest = _json(root / manifest_name)
        _verify_tree(root / directory, manifest)
    return root, plan, mapper_config, quality


def _source_evidence(
    root: Path, source_plan: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "backend_plan_sha256": sha256_file(root / "backend_plan.json"),
        # _workspace_context has already verified this committed digest against
        # the backend snapshot; avoid hashing a multi-gigabyte DB twice.
        "database_committed_sha256": source_plan["database_sha256"],
        "quality_marker_sha256": sha256_file(root / "stages" / "quality.json"),
        "quality_report_sha256": sha256_file(_report_path(root, "quality")),
        "registered_model_manifest_sha256": sha256_file(
            root / "registered_model_manifest.json"
        ),
        "text_model_manifest_sha256": sha256_file(
            root / "text_model_manifest.json"
        ),
    }


def _manifest_rows(plan: Mapping[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    frontend = Path(str(plan["frontend_artifact"])).resolve()
    manifest = _json(frontend / "frame_manifest.json")
    rows = manifest.get("frames")
    if not isinstance(rows, list) or not rows:
        raise ArtifactError("source frontend has no frame manifest rows")
    return frontend, rows


def _prior_records(
    database: Path,
    rows: Sequence[Mapping[str, Any]],
    temporal_blocks: int,
) -> tuple[list[dict[str, Any]], set[str], set[str], set[str]]:
    left_names = [str(row["left_image"]["name"]) for row in rows]
    priors = _cartesian_camera_priors(database, set(left_names))
    ordered_names = [name for name in left_names if name in priors]
    timestamps = np.asarray(
        [
            int(row["timestamp_ns"])
            for row in rows
            if str(row["left_image"]["name"]) in priors
        ],
        dtype=np.int64,
    )
    split = temporal_block_split(timestamps, temporal_blocks=temporal_blocks)
    rows_by_left_name = {
        str(row["left_image"]["name"]): row for row in rows
    }
    frame_role: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for index, name in enumerate(ordered_names):
        role = "calibration" if split.calibration_mask[index] else "holdout"
        row = rows_by_left_name[name]
        for side in ("left_image", "right_image"):
            frame_role[str(row[side]["name"])] = role
        position, covariance = priors[name]
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ArtifactError(
                f"pose prior covariance for {name} is not positive definite"
            ) from exc
        records.append(
            {
                "name": name,
                "timestamp_ns": int(timestamps[index]),
                "block_id": int(split.block_ids[index]),
                "role": role,
                "position_m": position.tolist(),
                "covariance_m2": covariance.tolist(),
            }
        )

    uri = f"file:{database.resolve()}?mode=ro&immutable=1"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            database_prior_names = {
                str(name)
                for (name,) in connection.execute(
                    """
                    SELECT i.name
                    FROM pose_priors AS p
                    JOIN images AS i ON i.image_id = p.corr_data_id
                    WHERE p.corr_sensor_type = 0
                    """
                )
            }
    except sqlite3.Error as exc:
        raise ArtifactError(f"cannot enumerate source pose priors: {exc}") from exc
    unknown = database_prior_names - set(frame_role)
    calibration = {
        name for name in database_prior_names if frame_role.get(name) == "calibration"
    }
    holdout = {
        name for name in database_prior_names if frame_role.get(name) == "holdout"
    }
    if unknown:
        raise ArtifactError(
            "pose priors attached outside the stereo frame manifest are not "
            f"safe for refinement: {sorted(unknown)[:8]}"
        )
    if not calibration or not holdout:
        raise ArtifactError("RTK refinement needs calibration and holdout priors")
    return records, calibration, holdout, database_prior_names


def _database_ids(database: Path, table: str, column: str) -> list[int]:
    if table not in {"cameras", "rigs"}:
        raise ValueError("unsupported constant-parameter table")
    try:
        with sqlite3.connect(database) as connection:
            values = [
                int(row[0])
                for row in connection.execute(
                    f"SELECT {column} FROM {table} ORDER BY {column}"
                )
            ]
    except sqlite3.Error as exc:
        raise ArtifactError(f"cannot read COLMAP {table}: {exc}") from exc
    if not values:
        raise ArtifactError(f"COLMAP database contains no {table}")
    return values


def _filter_optimizer_database(
    database: Path, holdout_names: set[str]
) -> dict[str, Any]:
    try:
        with sqlite3.connect(database) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            rows = connection.execute(
                """
                SELECT p.pose_prior_id, i.name
                FROM pose_priors AS p
                JOIN images AS i ON i.image_id = p.corr_data_id
                WHERE p.corr_sensor_type = 0
                """
            ).fetchall()
            remove_ids = [
                int(prior_id)
                for prior_id, name in rows
                if str(name) in holdout_names
            ]
            connection.executemany(
                "DELETE FROM pose_priors WHERE pose_prior_id = ?",
                ((value,) for value in remove_ids),
            )
            remaining = connection.execute(
                """
                SELECT i.name
                FROM pose_priors AS p
                JOIN images AS i ON i.image_id = p.corr_data_id
                WHERE p.corr_sensor_type = 0
                """
            ).fetchall()
            connection.commit()
    except sqlite3.Error as exc:
        raise ArtifactError(f"cannot create optimizer prior split: {exc}") from exc
    remaining_names = {str(row[0]) for row in remaining}
    leaked = sorted(holdout_names & remaining_names)
    if leaked:
        raise ArtifactError(f"held-out pose priors leaked into optimizer DB: {leaked[:8]}")
    record = sqlite_logical_record(database)
    return {
        "schema_version": 1,
        "n_source_camera_priors": len(rows),
        "n_removed_holdout_priors": len(remove_ids),
        "n_optimizer_camera_priors": len(remaining_names),
        "remaining_prior_names_sha256": canonical_hash(sorted(remaining_names)),
        "heldout_prior_names_sha256": canonical_hash(sorted(holdout_names)),
        "committed_view": record,
    }


def prepare_rtk_refinement(
    source_backend: str | Path,
    destination: str | Path,
    *,
    config: RtkRefinementConfig = RtkRefinementConfig(),
) -> Path:
    """Publish a private optimizer DB with temporal holdouts removed."""
    source, source_plan, mapper_config, _ = _source_context(source_backend)
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        root, plan, existing, _, _ = _refinement_context(destination)
        if (
            Path(str(plan["source_backend"])).resolve() != source
            or existing != config
        ):
            raise ArtifactError("existing RTK refinement has different inputs")
        return root
    frontend, rows = _manifest_rows(source_plan)
    for immutable in (source, frontend):
        if destination == immutable or immutable in destination.parents:
            raise ArtifactError(
                f"refinement workspace must be outside immutable input {immutable}"
            )
    evidence = _source_evidence(source, source_plan)
    records, calibration_names, holdout_names, all_prior_names = _prior_records(
        source / "database.db", rows, mapper_config.alignment_temporal_blocks
    )
    staging = destination.with_name(
        f".{destination.name}.prepare-{uuid.uuid4().hex}"
    )
    try:
        snapshot = create_database_snapshot(
            source / "database.db",
            staging,
            backend="rtk-prior-refinement",
            expected_committed_sha256=evidence["database_committed_sha256"],
        )
        os.rename(
            staging / "database_snapshot.json",
            staging / "source_database_snapshot.json",
        )
        optimizer = _filter_optimizer_database(
            staging / "database.db", holdout_names
        )
        camera_ids = _database_ids(staging / "database.db", "cameras", "camera_id")
        rig_ids = _database_ids(staging / "database.db", "rigs", "rig_id")
        all_image_names = list(source_plan["all_image_names"])
        _atomic_write(
            staging / "all_images.txt",
            "".join(f"{name}\n" for name in all_image_names).encode(),
        )
        _atomic_write(
            staging / "constant_cameras.txt",
            "".join(f"{value}\n" for value in camera_ids).encode(),
        )
        _atomic_write(
            staging / "constant_rigs.txt",
            "".join(f"{value}\n" for value in rig_ids).encode(),
        )
        _atomic_write(
            staging / "calibration_prior_names.txt",
            "".join(f"{name}\n" for name in sorted(calibration_names)).encode(),
        )
        _atomic_write(
            staging / "holdout_prior_names.txt",
            "".join(f"{name}\n" for name in sorted(holdout_names)).encode(),
        )
        _atomic_json(
            staging / "prior_split.json",
            {
                "schema_version": 1,
                "method": "contiguous_rank_blocks_alternating_v1",
                "n_evaluation_priors": len(records),
                "n_database_camera_priors": len(all_prior_names),
                "calibration_block_ids": sorted(
                    {row["block_id"] for row in records if row["role"] == "calibration"}
                ),
                "holdout_block_ids": sorted(
                    {row["block_id"] for row in records if row["role"] == "holdout"}
                ),
                "records": records,
            },
        )
        _atomic_json(staging / "optimizer_database.json", optimizer)
        plan = {
            "schema_version": 1,
            "source_backend": str(source),
            "frontend_artifact": str(frontend),
            "image_path": str(frontend / "images"),
            "input_model": str(source / "registered_model"),
            "config": asdict(config),
            "mapper_evaluation_config": asdict(mapper_config),
            "source_evidence": evidence,
            "source_snapshot_evidence": snapshot,
            "optimizer_database_committed_sha256": optimizer["committed_view"][
                "sha256"
            ],
            "n_images": len(all_image_names),
            "n_frames": len(rows),
            "n_evaluation_priors": len(records),
            "n_calibration_database_priors": len(calibration_names),
            "n_holdout_database_priors": len(holdout_names),
            "camera_ids": camera_ids,
            "rig_ids": rig_ids,
            "colmap_pose_prior_model": {
                "initialization_mode": config.initialization_mode,
                "input_model_supplied_to_mapper": (
                    config.initialization_mode == "continuation"
                ),
                "constraint": "camera_position_only",
                "position_loss": config.prior_position_loss,
                "use_robust_loss": config.prior_position_loss == "cauchy",
                "loss_description": (
                    "Cauchy_on_whitened_position_residual"
                    if config.prior_position_loss == "cauchy"
                    else "covariance_weighted_quadratic_position_residual"
                ),
                "prior_position_loss_scale": config.prior_position_loss_scale,
                "prior_position_loss_scale_active": (
                    config.prior_position_loss == "cauchy"
                ),
                "internal_alignment": "Sim3_then_restore_database_rig_scale",
                "strict_se3_only": False,
                "dual_antenna_heading_factor": False,
            },
        }
        inputs = {
            "source_backend": str(source),
            "source_evidence": evidence,
            "config": asdict(config),
            "prior_split_sha256": sha256_file(staging / "prior_split.json"),
        }
        ledger = StageLedger(staging)
        ledger.begin("prepare", inputs)
        _atomic_json(staging / "refinement_plan.json", plan)
        ledger.complete(
            "prepare",
            inputs,
            [
                "refinement_plan.json",
                "source_database_snapshot.json",
                "optimizer_database.json",
                "database.db",
                "prior_split.json",
                "all_images.txt",
                "constant_cameras.txt",
                "constant_rigs.txt",
                "calibration_prior_names.txt",
                "holdout_prior_names.txt",
            ],
        )
        publish_directory_noreplace(staging, destination)
    # Signals such as Ctrl-C may arrive while the multi-gigabyte private
    # database snapshot is being copied. Clean the unpublished staging tree
    # for every interruption before propagating it.
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    root, _, _, _, _ = _refinement_context(destination)
    return root


def _refinement_context(
    workspace: str | Path,
) -> tuple[Path, dict[str, Any], RtkRefinementConfig, MapperConfig, dict[str, Any]]:
    root = Path(workspace).expanduser().resolve()
    plan = _json(root / "refinement_plan.json")
    _verify_stage_output_hashes(root, "prepare", _PREPARE_CONTROL_OUTPUTS)
    try:
        normalized_config = _normalized_refinement_config_record(plan["config"])
        config = RtkRefinementConfig(**normalized_config)
        mapper_config = MapperConfig(**dict(plan["mapper_evaluation_config"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError("invalid RTK refinement plan") from exc
    if asdict(config) != normalized_config:
        raise ArtifactError("RTK refinement config schema changed")
    source, source_plan, source_mapper_config, source_quality = _source_context(
        plan.get("source_backend", "")
    )
    if asdict(source_mapper_config) != dict(plan["mapper_evaluation_config"]):
        raise ArtifactError(
            "RTK refinement evaluation config differs from source backend"
        )
    if _source_evidence(source, source_plan) != plan.get("source_evidence"):
        raise ArtifactError("source backend changed after RTK refinement preparation")
    if Path(str(plan.get("frontend_artifact", ""))).resolve() != Path(
        source_plan["frontend_artifact"]
    ).resolve():
        raise ArtifactError("RTK refinement frontend binding changed")
    optimizer = _json(root / "optimizer_database.json")
    committed = sqlite_logical_record(root / "database.db")
    if (
        committed != optimizer.get("committed_view")
        or committed.get("sha256")
        != plan.get("optimizer_database_committed_sha256")
    ):
        raise ArtifactError("RTK refinement optimizer database changed")
    split = _json(root / "prior_split.json")
    calibration = (root / "calibration_prior_names.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    holdout = (root / "holdout_prior_names.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    split_records = split.get("records")
    if not isinstance(split_records, list):
        raise ArtifactError("RTK refinement prior split has no records")
    expected_calibration = sorted(
        str(item["name"])
        for item in split_records
        if item.get("role") == "calibration"
    )
    expected_holdout = sorted(
        str(item["name"])
        for item in split_records
        if item.get("role") == "holdout"
    )
    all_images = (root / "all_images.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    try:
        camera_ids = [
            int(value)
            for value in (root / "constant_cameras.txt").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        rig_ids = [
            int(value)
            for value in (root / "constant_rigs.txt").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
    except ValueError as exc:
        raise ArtifactError("RTK refinement constant ID list changed") from exc
    if (
        calibration != expected_calibration
        or holdout != expected_holdout
        or len(calibration) != plan.get("n_calibration_database_priors")
        or len(holdout) != plan.get("n_holdout_database_priors")
        or set(calibration) & set(holdout)
        or all_images != list(source_plan["all_image_names"])
        or camera_ids != list(plan.get("camera_ids", ()))
        or rig_ids != list(plan.get("rig_ids", ()))
        or canonical_hash(calibration)
        != optimizer.get("remaining_prior_names_sha256")
        or canonical_hash(holdout)
        != optimizer.get("heldout_prior_names_sha256")
    ):
        raise ArtifactError("RTK refinement prior split evidence changed")
    return root, plan, config, mapper_config, source_quality


def _rtk_refinement_command(
    root: Path,
    plan: Mapping[str, Any],
    config: RtkRefinementConfig,
    executable: str | Path,
) -> tuple[str, ...]:
    prefix = (
        str(executable),
        "pose_prior_mapper",
        "--database_path",
        str(root / "database.db"),
        "--image_path",
        str(plan["image_path"]),
    )
    input_model = (
        ("--input_path", str(plan["input_model"]))
        if config.initialization_mode == "continuation"
        else ()
    )
    # Keep the continuation command byte-for-byte compatible with sealed
    # sidecars.  Fresh mode changes exactly one thing: the existing model is
    # not supplied, so pose priors participate while tracks are first built.
    return prefix + input_model + (
        "--output_path",
        str(root / "refined_model.incomplete"),
        "--default_random_seed",
        str(config.random_seed),
        "--overwrite_priors_covariance",
        "0",
        "--use_robust_loss_on_prior_position",
        "1" if config.prior_position_loss == "cauchy" else "0",
        "--prior_position_loss_scale",
        str(config.prior_position_loss_scale),
        "--Mapper.image_list_path",
        str(root / "all_images.txt"),
        "--Mapper.constant_camera_list_path",
        str(root / "constant_cameras.txt"),
        "--Mapper.constant_rig_list_path",
        str(root / "constant_rigs.txt"),
        "--Mapper.multiple_models",
        "0",
        "--Mapper.num_threads",
        str(config.num_threads),
        "--Mapper.random_seed",
        str(config.random_seed),
        "--Mapper.extract_colors",
        "0",
        "--Mapper.ba_refine_focal_length",
        "0",
        "--Mapper.ba_refine_principal_point",
        "0",
        "--Mapper.ba_refine_extra_params",
        "0",
        "--Mapper.ba_refine_sensor_from_rig",
        "0",
        "--Mapper.fix_existing_frames",
        "0",
        "--Mapper.ba_global_backend",
        "CERES",
        "--Mapper.ba_use_gpu",
        "0",
        "--Mapper.ba_global_max_num_iterations",
        str(config.ba_global_max_num_iterations),
        "--Mapper.ba_global_max_refinements",
        str(config.ba_global_max_refinements),
    )


def build_rtk_refinement_command(
    workspace: str | Path, executable: str | Path
) -> tuple[str, ...]:
    """Build the exact command after verifying a prepared sidecar."""
    root, plan, config, _, _ = _refinement_context(workspace)
    return _rtk_refinement_command(root, plan, config, executable)


def run_rtk_refinement_solve(
    workspace: str | Path,
    executable: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Run a continuation or fresh prior-aware solve on calibration priors."""
    root, plan, config, _, _ = _refinement_context(workspace)
    command = _rtk_refinement_command(root, plan, config, executable)
    inputs = {
        "command": command,
        "prepare_marker_sha256": sha256_file(root / "stages" / "prepare.json"),
        "optimizer_database_committed_sha256": plan[
            "optimizer_database_committed_sha256"
        ],
        "source_registered_model_manifest_sha256": plan["source_evidence"][
            "registered_model_manifest_sha256"
        ],
    }
    ledger = StageLedger(root)
    status = ledger.begin("refine", inputs)
    published = root / "refined_model"
    if status == "complete":
        _verify_tree(published, _json(root / "refined_model_manifest.json"))
        return _json(_report_path(root, "refine"))

    incomplete = root / "refined_model.incomplete"
    resources: dict[str, Any] | None = None
    if not published.exists():
        if incomplete.exists():
            shutil.rmtree(incomplete)
        incomplete.mkdir()
        try:
            attempt = _next_solve_attempt(root)
            resources = (
                _run_monitored_mapper(command, attempt, config)
                if runner is subprocess.run
                else _run_injected_mapper(command, attempt, runner)
            )
            candidates = _model_candidates(incomplete)
            if not candidates:
                raise ArtifactError("pose-prior mapper produced no valid model")
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
                raise ArtifactError("RTK refinement lost registered stereo images")
            selected_path = incomplete / str(selected["relative_path"])
            if selected_path == incomplete:
                os.rename(incomplete, published)
            else:
                os.rename(selected_path, published)
                shutil.rmtree(incomplete)
        except Exception:
            shutil.rmtree(incomplete, ignore_errors=True)
            raise
    else:
        resources = None
        stats = _analyze_model(published, executable, runner)
        analyses = [{"relative_path": ".", "stats": stats}]
        selected = analyses[0]
    # Revalidating the context after COLMAP proves it did not mutate the DB or
    # any sealed source evidence.
    _refinement_context(root)
    report = {
        "schema_version": 1,
        "stage": "refine",
        "method": (
            "colmap_pose_prior_mapper_fresh_calibration_blocks_only"
            if config.initialization_mode == "fresh"
            else "colmap_pose_prior_mapper_calibration_blocks_only"
        ),
        "initialization_mode": config.initialization_mode,
        "input_model_supplied_to_mapper": (
            config.initialization_mode == "continuation"
        ),
        "strict_se3_only": False,
        "position_prior_loss": config.prior_position_loss,
        "prior_position_loss_scale": config.prior_position_loss_scale,
        "prior_position_loss_scale_active": (
            config.prior_position_loss == "cauchy"
        ),
        "command": list(command),
        "n_expected_images": plan["n_images"],
        "selected_stats": selected["stats"],
        "candidates": analyses,
        "all_images_retained": int(selected["stats"]["registered_images"])
        == int(plan["n_images"]),
        **({"resources": resources} if resources is not None else {}),
    }
    resource_outputs: tuple[Path, ...] = ()
    if resources is not None:
        resource_outputs = _solve_resource_paths(root, resources)
    return _complete_stage(
        root,
        ledger,
        "refine",
        inputs,
        report,
        "refined_model_manifest.json",
        _tree_manifest(published),
        resource_outputs,
    )


def _model_converter_command(
    model: Path, output: Path, executable: str | Path
) -> tuple[str, ...]:
    return (
        str(executable),
        "model_converter",
        "--input_path",
        str(model),
        "--output_path",
        str(output),
        "--output_type",
        "TXT",
    )


def _data_lines(path: Path) -> list[list[str]]:
    return [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _calibration_difference(source: Path, refined: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    maximum = 0.0
    structure_equal = True
    for filename in ("cameras.txt", "rigs.txt"):
        before = _data_lines(source / filename)
        after = _data_lines(refined / filename)
        if len(before) != len(after):
            structure_equal = False
            results[filename] = {"same_row_count": False}
            continue
        file_max = 0.0
        file_structure = True
        for left, right in zip(before, after):
            if len(left) != len(right):
                file_structure = False
                break
            for first, second in zip(left, right):
                try:
                    difference = abs(float(first) - float(second))
                except ValueError:
                    if first != second:
                        file_structure = False
                        break
                else:
                    file_max = max(file_max, difference)
            if not file_structure:
                break
        structure_equal &= file_structure
        maximum = max(maximum, file_max)
        results[filename] = {
            "same_structure": file_structure,
            "maximum_numeric_abs_change": file_max,
        }
    return {
        "same_structure": structure_equal,
        "maximum_numeric_abs_change": maximum,
        "files": results,
    }


def _stereo_baselines(
    poses: Mapping[str, tuple[np.ndarray, np.ndarray]],
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    values = []
    for row in rows:
        left = str(row["left_image"]["name"])
        right = str(row["right_image"]["name"])
        if left not in poses or right not in poses:
            raise ArtifactError("stereo baseline audit lacks a registered image")
        values.append(float(np.linalg.norm(poses[right][1] - poses[left][1])))
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ArtifactError("refined stereo baselines are invalid")
    return result


def _similarity_scale(source: np.ndarray, target: np.ndarray) -> float:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("trajectory scale inputs must have matching shape (N, 3)")
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    denominator = float(np.einsum("ni,ni->", source_centered, source_centered))
    if denominator <= 1e-12 or np.linalg.matrix_rank(source_centered) < 2:
        raise ArtifactError("trajectory is too degenerate for a scale audit")
    u, singular, vt = np.linalg.svd(source_centered.T @ target_centered)
    sign = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    scale = float((singular[:-1].sum() + sign * singular[-1]) / denominator)
    if not math.isfinite(scale) or scale <= 0:
        raise ArtifactError("trajectory scale audit produced an invalid scale")
    return scale


def _heldout_evaluation(
    poses: Mapping[str, tuple[np.ndarray, np.ndarray]],
    rows: Sequence[Mapping[str, Any]],
    priors: Mapping[str, tuple[np.ndarray, np.ndarray]],
    mapper_config: MapperConfig,
) -> tuple[dict[str, Any], Any]:
    left_names = [str(row["left_image"]["name"]) for row in rows]
    prior_names = [name for name in left_names if name in priors]
    source = np.stack([poses[name][1] for name in prior_names])
    target = np.stack([priors[name][0] for name in prior_names])
    covariance = np.stack([priors[name][1] for name in prior_names])
    timestamps = np.asarray(
        [
            int(row["timestamp_ns"])
            for row in rows
            if str(row["left_image"]["name"]) in priors
        ],
        dtype=np.int64,
    )
    evaluation = estimate_temporal_heldout_alignment(
        source,
        target,
        timestamps,
        covariance,
        temporal_blocks=mapper_config.alignment_temporal_blocks,
        ransac_threshold_m=mapper_config.alignment_ransac_threshold_m,
        ransac_iterations=mapper_config.alignment_ransac_iterations,
        random_seed=mapper_config.random_seed,
    )
    quality = rtk_residual_quality(
        evaluation.residual_vectors_m[evaluation.holdout_mask],
        covariance[evaluation.holdout_mask],
        evaluation.holdout_inlier_mask[evaluation.holdout_mask],
        config=mapper_config,
    )
    return quality, evaluation


def run_rtk_refinement_quality(
    workspace: str | Path,
    executable: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Re-evaluate visual, metric, rig, and untouched RTK quality."""
    root, plan, config, mapper_config, source_quality = _refinement_context(
        workspace
    )
    refine_marker = root / "stages" / "refine.json"
    if not refine_marker.is_file():
        raise ArtifactError("run RTK refinement before its quality audit")
    _verify_tree(root / "refined_model", _json(root / "refined_model_manifest.json"))
    converter = _model_converter_command(
        root / "refined_model", root / "refined_text.incomplete", executable
    )
    analyzer = (
        str(executable),
        "model_analyzer",
        "--path",
        str(root / "refined_model"),
    )
    inputs = {
        "converter_command": converter,
        "analyzer_command": analyzer,
        "refine_marker_sha256": sha256_file(refine_marker),
        "source_quality_report_sha256": plan["source_evidence"][
            "quality_report_sha256"
        ],
        "prior_split_sha256": sha256_file(root / "prior_split.json"),
        # Keep legacy completed-stage fingerprints byte-equivalent. New plans
        # contain prior_position_loss explicitly; sealed v1 plans do not.
        "acceptance_config": plan["config"],
    }
    ledger = StageLedger(root)
    status = ledger.begin("quality", inputs)
    if status == "complete":
        _verify_tree(root / "refined_text", _json(root / "refined_text_manifest.json"))
        return _json(_report_path(root, "quality"))

    text = root / "refined_text"
    incomplete = root / "refined_text.incomplete"
    if not text.exists():
        if incomplete.exists():
            shutil.rmtree(incomplete)
        incomplete.mkdir()
        try:
            _execute(converter, runner)
            required = ("images.txt", "cameras.txt", "rigs.txt")
            if any(not (incomplete / name).is_file() for name in required):
                raise ArtifactError("refined text model is incomplete")
            os.rename(incomplete, text)
        except Exception:
            shutil.rmtree(incomplete, ignore_errors=True)
            raise

    source = Path(str(plan["source_backend"])).resolve()
    # _refinement_context already verified the bound source in this call.
    source_plan = _json(source / "backend_plan.json")
    _, rows = _manifest_rows(source_plan)
    refined_stats = _analyze_model(root / "refined_model", executable, runner)
    registered = registered_names_from_images_txt(text / "images.txt")
    visual = quality_summary(
        source_plan["all_image_names"],
        registered,
        refined_stats,
        max_reprojection_error_px=mapper_config.max_reprojection_error_px,
        min_mean_track_length=mapper_config.min_mean_track_length,
    )
    source_poses = _poses_from_images_txt(source / "registered_text" / "images.txt")
    refined_poses = _poses_from_images_txt(text / "images.txt")
    left_names = [str(row["left_image"]["name"]) for row in rows]
    source_centers = np.stack([source_poses[name][1] for name in left_names])
    refined_centers = np.stack([refined_poses[name][1] for name in left_names])
    scale = _similarity_scale(source_centers, refined_centers)
    source_baselines = _stereo_baselines(source_poses, rows)
    refined_baselines = _stereo_baselines(refined_poses, rows)
    baseline_change = float(np.max(np.abs(refined_baselines - source_baselines)))
    calibration = _calibration_difference(source / "registered_text", text)

    priors = _cartesian_camera_priors(source / "database.db", set(left_names))
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
        raise ArtifactError("source and refined RTK holdout splits differ")
    split = _json(root / "prior_split.json")
    recorded_roles = [str(item["role"]) for item in split["records"]]
    expected_roles = [
        "calibration" if value else "holdout"
        for value in source_evaluation.calibration_mask
    ]
    if recorded_roles != expected_roles:
        raise ArtifactError("optimizer and evaluator temporal splits differ")

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
    source_track = float(source_quality["checks"]["mean_track_length"]["value"])
    graph_checks = (
        {
            "fresh_mean_track_length_absolute": {
                "value": float(refined_stats["mean_track_length"]),
                "minimum": config.fresh_min_mean_track_length,
                "passed": float(refined_stats["mean_track_length"])
                >= config.fresh_min_mean_track_length,
            },
            "fresh_mean_observations_per_image_absolute": {
                "value": float(
                    refined_stats["mean_observations_per_image"]
                ),
                "minimum": config.fresh_min_mean_observations_per_image,
                "passed": float(
                    refined_stats["mean_observations_per_image"]
                )
                >= config.fresh_min_mean_observations_per_image,
            },
        }
        if config.initialization_mode == "fresh"
        else {
            "track_length_retention": {
                "value": float(refined_stats["mean_track_length"])
                / source_track,
                "minimum": config.min_track_length_retention,
                "passed": float(refined_stats["mean_track_length"])
                >= source_track * config.min_track_length_retention,
            }
        }
    )
    checks = {
        "visual_quality": {
            "value": bool(visual["passed"]),
            "expected": True,
            "passed": bool(visual["passed"]),
        },
        "reprojection_regression_px": {
            "value": float(refined_stats["mean_reprojection_error_px"])
            - source_reprojection,
            "maximum": config.max_reprojection_regression_px,
            "passed": float(refined_stats["mean_reprojection_error_px"])
            <= source_reprojection + config.max_reprojection_regression_px,
        },
        **graph_checks,
        "trajectory_similarity_scale": {
            "value": scale,
            "expected": 1.0,
            "maximum_abs_deviation": config.max_trajectory_scale_deviation,
            "passed": abs(scale - 1.0) <= config.max_trajectory_scale_deviation,
        },
        "stereo_baseline_max_change_m": {
            "value": baseline_change,
            "maximum": config.max_stereo_baseline_change_m,
            "passed": baseline_change <= config.max_stereo_baseline_change_m,
        },
        "calibration_parameter_max_change": {
            "value": calibration["maximum_numeric_abs_change"],
            "maximum": config.max_calibration_parameter_change,
            "passed": bool(calibration["same_structure"])
            and calibration["maximum_numeric_abs_change"]
            <= config.max_calibration_parameter_change,
        },
        "heldout_rtk_absolute_gates": {
            "value": refined_rtk_passed,
            "expected": True,
            "passed": refined_rtk_passed,
        },
        "heldout_median_not_worse_m": {
            "value": refined_median - source_median,
            "maximum": config.max_holdout_median_regression_m,
            "passed": refined_median
            <= source_median + config.max_holdout_median_regression_m,
        },
        "heldout_improvement_when_source_failed_m": {
            "authoritative": not source_rtk_passed,
            "value": source_median - refined_median,
            "minimum": (
                config.min_holdout_median_improvement_m
                if not source_rtk_passed
                else None
            ),
            "passed": source_rtk_passed
            or source_median - refined_median
            >= config.min_holdout_median_improvement_m,
        },
    }
    passed = all(check["passed"] for check in checks.values())
    report = {
        "schema_version": 1,
        "stage": "quality",
        "method": (
            "rtk_position_constrained_stereo_pose_prior_mapping"
            if config.initialization_mode == "fresh"
            else "rtk_position_constrained_stereo_pose_refinement"
        ),
        "initialization_mode": config.initialization_mode,
        "position_prior_loss": config.prior_position_loss,
        "prior_position_loss_scale": config.prior_position_loss_scale,
        "prior_position_loss_scale_active": (
            config.prior_position_loss == "cauchy"
        ),
        "holdout_scope": {
            "independent_of": [
                "rtk_position_refinement_factors",
                "refinement_alignment_fit",
            ],
            "not_independent_of": [
                "upstream_rtk_guided_keyframe_and_pair_planning"
            ],
        },
        "passed": passed,
        "registration_fraction": visual["registration_fraction"],
        "missing_images": visual["missing_images"],
        "unexpected_images": visual["unexpected_images"],
        "checks": checks,
        "visual_quality": visual,
        "source_model_stats": {
            "mean_reprojection_error_px": source_reprojection,
            "mean_track_length": source_track,
        },
        "refined_model_stats": refined_stats,
        "metric_integrity": {
            "stock_colmap_strict_se3_only": False,
            "trajectory_similarity_scale": scale,
            "source_stereo_baseline_m": {
                "median": float(np.median(source_baselines)),
                "minimum": float(source_baselines.min()),
                "maximum": float(source_baselines.max()),
            },
            "refined_stereo_baseline_m": {
                "median": float(np.median(refined_baselines)),
                "minimum": float(refined_baselines.min()),
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
    return _complete_stage(
        root,
        ledger,
        "quality",
        inputs,
        report,
        "refined_text_manifest.json",
        _tree_manifest(text),
    )


def run_rtk_refinement(
    source_backend: str | Path,
    destination: str | Path,
    executable: str | Path,
    *,
    config: RtkRefinementConfig = RtkRefinementConfig(),
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Prepare, solve, and audit one resumable RTK refinement sidecar."""
    destination_path = Path(destination).expanduser().resolve()
    if destination_path.exists():
        existing_plan = _json(destination_path / "refinement_plan.json")
        try:
            existing_config = _normalized_refinement_config_record(
                existing_plan["config"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError("invalid existing RTK refinement plan") from exc
        if (
            Path(str(existing_plan.get("source_backend", ""))).resolve()
            != Path(source_backend).expanduser().resolve()
            or existing_config != asdict(config)
        ):
            raise ArtifactError("existing RTK refinement has different inputs")
        workspace = destination_path
    else:
        workspace = prepare_rtk_refinement(
            source_backend, destination_path, config=config
        )
    run_rtk_refinement_solve(workspace, executable, runner=runner)
    return run_rtk_refinement_quality(workspace, executable, runner=runner)


def audited_rtk_refinement_result(workspace: str | Path) -> dict[str, Any]:
    """Load one completed arm after verifying all sealed result evidence."""
    root, plan, config, _, _ = _refinement_context(workspace)
    _verify_stage_output_hashes(
        root,
        "refine",
        ("reports/refine.json", "refined_model_manifest.json"),
    )
    _verify_stage_output_hashes(
        root,
        "quality",
        ("reports/quality.json", "refined_text_manifest.json"),
    )
    _verify_tree(root / "refined_model", _json(root / "refined_model_manifest.json"))
    _verify_tree(root / "refined_text", _json(root / "refined_text_manifest.json"))
    optimizer = _json(root / "optimizer_database.json")
    split = _json(root / "prior_split.json")
    refine = _json(_report_path(root, "refine"))
    quality = _json(_report_path(root, "quality"))
    return {
        "schema_version": 1,
        "workspace": str(root),
        "source_backend": str(Path(str(plan["source_backend"])).resolve()),
        "frontend_artifact": str(
            Path(str(plan["frontend_artifact"])).resolve()
        ),
        "image_path": str(Path(str(plan["image_path"])).resolve()),
        "input_model": str(Path(str(plan["input_model"])).resolve()),
        "config": asdict(config),
        "mapper_evaluation_config": plan["mapper_evaluation_config"],
        "source_evidence": plan["source_evidence"],
        "optimizer_database_committed_sha256": plan[
            "optimizer_database_committed_sha256"
        ],
        "optimizer_database": optimizer,
        "counts": {
            "images": plan["n_images"],
            "frames": plan["n_frames"],
            "evaluation_priors": plan["n_evaluation_priors"],
            "calibration_priors": plan["n_calibration_database_priors"],
            "holdout_priors": plan["n_holdout_database_priors"],
        },
        "camera_ids": plan["camera_ids"],
        "rig_ids": plan["rig_ids"],
        "split": {
            "method": split["method"],
            "calibration_block_ids": split["calibration_block_ids"],
            "holdout_block_ids": split["holdout_block_ids"],
            "prior_split_sha256": sha256_file(root / "prior_split.json"),
            "calibration_names_sha256": sha256_file(
                root / "calibration_prior_names.txt"
            ),
            "holdout_names_sha256": sha256_file(
                root / "holdout_prior_names.txt"
            ),
            "all_images_sha256": sha256_file(root / "all_images.txt"),
        },
        "refinement_plan_sha256": sha256_file(root / "refinement_plan.json"),
        "refine_marker_sha256": sha256_file(root / "stages" / "refine.json"),
        "quality_marker_sha256": sha256_file(
            root / "stages" / "quality.json"
        ),
        "command": refine["command"],
        "quality": quality,
    }


def refinement_export_context(workspace: str | Path) -> dict[str, Any]:
    """Return sealed refined-model inputs for the shared pose exporter."""
    root, plan, config, _, _ = _refinement_context(workspace)
    quality_marker = root / "stages" / "quality.json"
    _verify_stage_output_hashes(
        root,
        "quality",
        ("reports/quality.json", "refined_text_manifest.json"),
    )
    quality = _json(_report_path(root, "quality"))
    if not quality.get("passed"):
        raise ArtifactError(
            "RTK refinement failed acceptance gates; source poses remain untouched"
        )
    _verify_tree(root / "refined_text", _json(root / "refined_text_manifest.json"))
    return {
        "workspace": root,
        "source_backend": Path(str(plan["source_backend"])).resolve(),
        "text_model": root / "refined_text",
        "text_model_manifest": root / "refined_text_manifest.json",
        "binary_model_manifest": root / "refined_model_manifest.json",
        "quality_marker": quality_marker,
        "quality_report": _report_path(root, "quality"),
        "quality": quality,
        "pose_prior_role": (
            "calibration_blocks_constrain_incremental_mapping; "
            "holdout_blocks_absent_from_mapper_factors_but_may_influence_"
            "upstream_rtk_guided_selection"
            if config.initialization_mode == "fresh"
            else (
                "calibration_blocks_constrain_position_refinement; "
                "holdout_blocks_absent_from_refinement_factors_but_may_"
                "influence_upstream_rtk_guided_selection"
            )
        ),
        "pose_priors_constrain_mapper": config.initialization_mode == "fresh",
        "pose_priors_constrain_refinement": (
            config.initialization_mode == "continuation"
        ),
        "provenance": {
            "initialization_mode": config.initialization_mode,
            "rtk_refinement_workspace": str(root),
            "rtk_refinement_plan_sha256": sha256_file(
                root / "refinement_plan.json"
            ),
            "rtk_refinement_marker_sha256": sha256_file(quality_marker),
            "rtk_refinement_quality_sha256": sha256_file(
                _report_path(root, "quality")
            ),
            "optimizer_database_committed_sha256": plan[
                "optimizer_database_committed_sha256"
            ],
            "heldout_priors_removed_from_optimizer": True,
            "stock_colmap_strict_se3_only": False,
        },
    }
