"""Mapper-neutral COLMAP reconstruction backend.

The backend reads only a sealed frontend artifact.  Each mapper receives a
private, SHA-256-verified database snapshot, solves the selected stereo
keyframes, and then registers every remaining image against that model.
Global mapping is the default; the incremental mapper is an independent
fallback using the same frontend evidence.

Cartesian pose priors are used only for post-solve fixed-scale
georeferencing and held-out evaluation. They do not constrain GlobalMapper.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rtk_splat.backends.artifact_io import (
    _atomic_json,
    _atomic_write,
    _complete_stage,
    _json,
    _report_path,
    _tree_manifest,
    _verify_tree,
)
from rtk_splat.backends.colmap_model import (
    _MODEL_FILES,
    _analyze_model,
    _cartesian_camera_priors,
    _model_candidates,
    _poses_from_images_txt,
    build_model_analyzer_command,
    parse_model_analyzer,
    registered_names_from_images_txt,
)
from rtk_splat.backends.execution import (
    Runner,
    _MonitoredProcessInterrupted,
    _execute,
    _next_solve_attempt,
    _run_injected_mapper,
    _run_monitored_mapper,
    _solve_resource_paths,
)
from rtk_splat.backends.mapper_config import (
    MapperConfig,
    _LEGACY_EVALUATION_DEFAULTS,
)
from rtk_splat.backends.quality import (
    AlignmentResult,
    TemporalAlignmentResult,
    TemporalBlockSplit,
    estimate_rigid_alignment,
    estimate_temporal_heldout_alignment,
    quality_summary,
    rtk_residual_quality,
    squared_mahalanobis_residuals,
    temporal_block_split,
)
from rtk_splat.backends.workspace import (
    _frontend_context,
    _frontend_hashes,
    _image_sets,
    _immutable_input_roots,
    _marker_hash,
    _workspace_context,
    _write_lines,
)
from rtk_splat.frontends.artifact import (
    ArtifactError,
    FRONTEND_SEAL_FILE,
    StageLedger,
    canonical_hash,
    create_database_snapshot,
    sha256_file,
    verify_frontend_seal,
)


def prepare_mapper_backend(
    frontend_artifact: str | Path,
    destination: str | Path,
    *,
    config: MapperConfig = MapperConfig(),
) -> Path:
    """Create or verify one isolated backend workspace.

    Calling this again with exactly the same inputs verifies and returns the
    existing workspace; changed inputs require a new destination.
    """
    frontend, manifest, keyframes = _frontend_context(frontend_artifact)
    workspace = Path(destination).expanduser().resolve()
    for immutable in _immutable_input_roots(frontend):
        if workspace == immutable or immutable in workspace.parents:
            raise ArtifactError(
                f"backend workspace must be outside immutable input {immutable}"
            )
    all_names, solve_names, registration_names = _image_sets(manifest, keyframes)
    frontend_inputs = _frontend_hashes(frontend)
    seal = verify_frontend_seal(frontend)
    matching_stage_sha256 = _marker_hash(frontend, "matching")
    if matching_stage_sha256 is None:
        raise ArtifactError("sealed frontend matching stage is incomplete or absent")
    inputs = {
        "frontend_artifact": str(frontend),
        "frontend_inputs": frontend_inputs,
        "frontend_seal_sha256": sha256_file(frontend / FRONTEND_SEAL_FILE),
        "frontend_database_raw_sha256": sha256_file(frontend / "database.db"),
        "frontend_database_committed_sha256": seal["database"]["committed_view"][
            "sha256"
        ],
        "matching_stage_sha256": matching_stage_sha256,
        "config": asdict(config),
        "all_image_names": all_names,
        "solve_image_names": solve_names,
        "registration_image_names": registration_names,
    }
    if not workspace.exists():
        snapshot = create_database_snapshot(
            frontend / "database.db",
            workspace,
            backend=config.backend,
            expected_committed_sha256=inputs[
                "frontend_database_committed_sha256"
            ],
        )
        # Close the narrow verification/copy race before publishing a plan.
        verify_frontend_seal(frontend)
    else:
        allowed = {
            "database.db",
            "database_snapshot.json",
            "stages",
            "backend_plan.json",
            "solve_images.txt",
            "registration_images.txt",
        }
        unexpected = {path.name for path in workspace.iterdir()} - allowed
        if unexpected and not (workspace / "backend_plan.json").is_file():
            raise ArtifactError(
                "existing backend prepare workspace has unexpected files: "
                + ", ".join(sorted(unexpected))
            )
        snapshot = _json(workspace / "database_snapshot.json")
        snapshot_digest = sha256_file(workspace / "database.db")
        if (
            snapshot.get("backend") != config.backend
            or Path(str(snapshot.get("source_database", ""))).resolve()
            != (frontend / "database.db").resolve()
            or snapshot.get("snapshot_sha256") != snapshot_digest
            or snapshot.get("source_sha256_before")
            != inputs["frontend_database_raw_sha256"]
            or snapshot.get("source_sha256_after")
            != inputs["frontend_database_raw_sha256"]
            or snapshot.get("source_committed_sha256")
            != inputs["frontend_database_committed_sha256"]
        ):
            raise ArtifactError("existing backend database snapshot is incompatible")
    ledger = StageLedger(workspace)
    plan = {
        "schema_version": 1,
        **inputs,
        "database_sha256": snapshot["snapshot_sha256"],
        "image_path": str(frontend / "images"),
        "n_frames": len(manifest["frames"]),
        "n_images": len(all_names),
        "n_solve_images": len(solve_names),
        "n_registration_images": len(registration_names),
        "all_frames_retained": True,
    }
    if (workspace / "backend_plan.json").exists():
        if _json(workspace / "backend_plan.json") != plan:
            raise ArtifactError("existing backend workspace has different inputs")
    status = ledger.begin("prepare", inputs)
    if status == "complete":
        root, _, _ = _workspace_context(workspace)
        return root
    _atomic_json(workspace / "backend_plan.json", plan)
    _write_lines(workspace / "solve_images.txt", solve_names)
    if registration_names:
        _write_lines(workspace / "registration_images.txt", registration_names)
    else:
        _atomic_write(workspace / "registration_images.txt", b"")
    ledger.complete(
        "prepare",
        inputs,
        [
            "backend_plan.json",
            "database_snapshot.json",
            "solve_images.txt",
            "registration_images.txt",
        ],
    )
    root, _, _ = _workspace_context(workspace)
    return root


def build_mapper_command(
    workspace: str | Path,
    executable: str | Path,
) -> tuple[str, ...]:
    root, plan, config = _workspace_context(workspace)
    shared = [
        str(executable),
        "global_mapper" if config.backend == "global" else "mapper",
        "--database_path",
        str(root / "database.db"),
        "--image_path",
        plan["image_path"],
        "--output_path",
        str(root / "solve.incomplete"),
    ]
    if config.backend == "global":
        shared.extend(
            [
                "--log_target",
                "stdout",
                "--GlobalMapper.image_list_path",
                str(root / "solve_images.txt"),
                "--GlobalMapper.num_threads",
                str(config.num_threads),
                "--GlobalMapper.random_seed",
                str(config.random_seed),
                "--GlobalMapper.min_num_matches",
                str(config.min_num_matches),
                "--GlobalMapper.decompose_relative_pose",
                "1",
                "--GlobalMapper.ba_num_iterations",
                str(config.ba_num_iterations),
                "--GlobalMapper.gp_optimize_positions",
                "1",
                "--GlobalMapper.gp_optimize_points",
                "1",
                "--GlobalMapper.gp_optimize_scales",
                "1",
                "--GlobalMapper.gp_use_gpu",
                "1" if config.gp_use_gpu else "0",
                "--GlobalMapper.ba_refine_focal_length",
                "0",
                "--GlobalMapper.ba_refine_principal_point",
                "0",
                "--GlobalMapper.ba_refine_extra_params",
                "0",
                "--GlobalMapper.refine_sensor_from_rig",
                "0",
                "--GlobalMapper.ba_refine_rig_from_world",
                "1",
                "--GlobalMapper.ba_refine_points3D",
                "1",
                "--GlobalMapper.ba_ceres_use_gpu",
                "1" if config.ba_ceres_use_gpu else "0",
                "--GlobalMapper.keep_max_num_tracks",
                str(config.keep_max_num_tracks),
                "--GlobalMapper.track_required_tracks_per_view",
                str(config.track_required_tracks_per_view),
                "--GlobalMapper.skip_retriangulation",
                "1" if config.skip_retriangulation else "0",
            ]
        )
    else:
        shared.extend(
            [
                "--Mapper.image_list_path",
                str(root / "solve_images.txt"),
                "--Mapper.num_threads",
                str(config.num_threads),
                "--Mapper.random_seed",
                str(config.random_seed),
                "--Mapper.min_num_matches",
                str(config.min_num_matches),
                "--Mapper.multiple_models",
                "0",
                "--Mapper.ba_refine_focal_length",
                "0",
                "--Mapper.ba_refine_principal_point",
                "0",
                "--Mapper.ba_refine_extra_params",
                "0",
                "--Mapper.ba_refine_sensor_from_rig",
                "0",
            ]
        )
    return tuple(shared)


def build_registration_command(
    workspace: str | Path,
    executable: str | Path,
) -> tuple[str, ...]:
    root, plan, config = _workspace_context(workspace)
    selected = _selected_model(root)
    return (
        str(executable),
        "image_registrator",
        "--database_path",
        str(root / "database.db"),
        "--input_path",
        str(selected),
        "--output_path",
        str(root / "registration.incomplete"),
        "--Mapper.min_num_matches",
        str(config.min_num_matches),
    )

def build_model_converter_command(
    workspace: str | Path, executable: str | Path
) -> tuple[str, ...]:
    root, _, _ = _workspace_context(workspace)
    return (
        str(executable),
        "model_converter",
        "--input_path",
        str(root / "registered_model"),
        "--output_path",
        str(root / "registered_text.incomplete"),
        "--output_type",
        "TXT",
    )

def _latest_solve_resources(root: Path) -> dict[str, Any]:
    candidates = sorted(
        (root / "attempts" / "solve").glob("attempt-*/resource_usage.json")
    )
    if not candidates:
        raise ArtifactError(
            "published solve model has no resource evidence; prepare a new "
            "backend artifact"
        )
    usage = _json(candidates[-1])
    if int(usage.get("returncode", -1)) != 0:
        raise ArtifactError("published solve model has no successful resource record")
    samples, usage_path = _solve_resource_paths(root, usage)
    if not samples.is_file() or usage_path != candidates[-1]:
        raise ArtifactError("mapper resource evidence is incomplete")
    return usage

def _selected_model(workspace: Path) -> Path:
    record = _json(_report_path(workspace, "solve"))
    relative = record.get("selected_model_relative")
    if not isinstance(relative, str):
        raise ArtifactError("solve report has no selected model")
    candidate = (workspace / "solve_models" / relative).resolve()
    try:
        candidate.relative_to((workspace / "solve_models").resolve())
    except ValueError as exc:
        raise ArtifactError("selected model escapes solve_models") from exc
    if not all((candidate / name).is_file() for name in _MODEL_FILES):
        raise ArtifactError("selected solve model is incomplete")
    return candidate

def run_mapper_solve(
    workspace: str | Path,
    executable: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Run or verify the keyframe-only mapper stage."""
    root, plan, config = _workspace_context(workspace)
    command = build_mapper_command(root, executable)
    inputs = {
        "command": command,
        "prepare_marker_sha256": sha256_file(root / "stages" / "prepare.json"),
        "database_sha256": plan["database_sha256"],
    }
    ledger = StageLedger(root)
    status = ledger.begin("solve", inputs)
    published = root / "solve_models"
    if status == "complete":
        manifest = _json(root / "solve_model_manifest.json")
        _verify_tree(published, manifest)
        _selected_model(root)
        return _json(_report_path(root, "solve"))

    incomplete = root / "solve.incomplete"
    resources: dict[str, Any] | None = None
    if not published.exists():
        if incomplete.exists():
            shutil.rmtree(incomplete)
        incomplete.mkdir()
        try:
            if config.backend == "global":
                attempt = _next_solve_attempt(root)
                resources = (
                    _run_monitored_mapper(command, attempt, config)
                    if runner is subprocess.run
                    else _run_injected_mapper(command, attempt, runner)
                )
            else:
                # The fallback command and execution semantics intentionally
                # remain unchanged by the Global Mapper resource policy.
                _execute(command, runner)
            candidates = _model_candidates(incomplete)
            if not candidates:
                raise ArtifactError("mapper produced no valid COLMAP model")
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
            if config.require_all_keyframes and (
                selected["stats"]["registered_images"] != plan["n_solve_images"]
            ):
                raise ArtifactError(
                    "mapper did not register every selected stereo keyframe image"
                )
            os.rename(incomplete, published)
        except Exception:
            shutil.rmtree(incomplete, ignore_errors=True)
            raise
    else:
        if config.backend == "global":
            resources = _latest_solve_resources(root)
        candidates = _model_candidates(published)
        analyses = [
            {
                "relative_path": candidate.relative_to(published).as_posix(),
                "stats": _analyze_model(candidate, executable, runner),
            }
            for candidate in candidates
        ]
        if not analyses:
            raise ArtifactError("published solve model is incomplete")
        selected = max(
            analyses,
            key=lambda item: (
                item["stats"]["registered_images"],
                item["stats"]["observations"],
                -item["stats"]["mean_reprojection_error_px"],
            ),
        )
    report = {
        "schema_version": 1,
        "stage": "solve",
        "backend": config.backend,
        "command": list(command),
        "n_expected_solve_images": plan["n_solve_images"],
        "selected_model_relative": selected["relative_path"],
        "selected_stats": selected["stats"],
        "candidates": analyses,
        "all_keyframes_registered": (
            selected["stats"]["registered_images"] == plan["n_solve_images"]
        ),
        **({"resources": resources} if resources is not None else {}),
    }
    resource_outputs: tuple[Path, ...] = ()
    if resources is not None:
        resource_outputs = _solve_resource_paths(root, resources)
    return _complete_stage(
        root,
        ledger,
        "solve",
        inputs,
        report,
        "solve_model_manifest.json",
        _tree_manifest(published),
        resource_outputs,
    )


def run_image_registration(
    workspace: str | Path,
    executable: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Register all non-keyframes from the full private frontend database."""
    root, plan, config = _workspace_context(workspace)
    solve_marker = root / "stages" / "solve.json"
    if not solve_marker.is_file():
        raise ArtifactError("run the solve stage before image registration")
    _verify_tree(root / "solve_models", _json(root / "solve_model_manifest.json"))
    command = build_registration_command(root, executable)
    inputs = {
        "command": command,
        "solve_marker_sha256": sha256_file(solve_marker),
        "database_sha256": plan["database_sha256"],
        "registration_image_names_sha256": canonical_hash(
            plan["registration_image_names"]
        ),
    }
    ledger = StageLedger(root)
    status = ledger.begin("register", inputs)
    published = root / "registered_model"
    if status == "complete":
        _verify_tree(published, _json(root / "registered_model_manifest.json"))
        return _json(_report_path(root, "register"))

    incomplete = root / "registration.incomplete"
    if not published.exists():
        if incomplete.exists():
            shutil.rmtree(incomplete)
        incomplete.mkdir()
        try:
            _execute(command, runner)
            if not all((incomplete / name).is_file() for name in _MODEL_FILES):
                raise ArtifactError("image_registrator produced no valid model")
            stats = _analyze_model(incomplete, executable, runner)
            if config.require_all_frames and stats["registered_images"] != plan["n_images"]:
                raise ArtifactError("image_registrator did not register every image")
            os.rename(incomplete, published)
        except Exception:
            shutil.rmtree(incomplete, ignore_errors=True)
            raise
    else:
        stats = _analyze_model(published, executable, runner)
    solve_stats = _json(_report_path(root, "solve"))["selected_stats"]
    added = int(stats["registered_images"]) - int(solve_stats["registered_images"])
    report = {
        "schema_version": 1,
        "stage": "register",
        "command": list(command),
        "n_expected_images": plan["n_images"],
        "n_expected_non_keyframe_images": plan["n_registration_images"],
        "n_newly_registered_images": added,
        "stats": stats,
        "all_frames_registered_by_count": stats["registered_images"] == plan["n_images"],
    }
    return _complete_stage(
        root,
        ledger,
        "register",
        inputs,
        report,
        "registered_model_manifest.json",
        _tree_manifest(published),
    )

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


def _verify_pose_artifact(path: Path) -> dict[str, Any]:
    manifest = _json(path / "manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ArtifactError("pose artifact manifest has no file evidence")
    for relative, evidence in files.items():
        candidate = path / relative
        if (
            not isinstance(evidence, dict)
            or not candidate.is_file()
            or sha256_file(candidate) != evidence.get("sha256")
            or candidate.stat().st_size != evidence.get("size_bytes")
        ):
            raise ArtifactError(f"pose artifact file changed: {relative}")
    return manifest

def _apply_world_alignment(
    viewmats: np.ndarray,
    centers: np.ndarray,
    rotation_target_source: np.ndarray,
    translation_target_source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a unit-scale world transform while preserving OpenCV w2c poses."""
    aligned_centers = centers @ rotation_target_source.T + translation_target_source
    aligned = np.repeat(np.eye(4, dtype=np.float64)[None], len(viewmats), axis=0)
    aligned[:, :3, :3] = (
        viewmats[:, :3, :3] @ rotation_target_source.T
    )
    aligned[:, :3, 3] = -np.einsum(
        "nij,nj->ni", aligned[:, :3, :3], aligned_centers
    )
    recovered = -np.einsum(
        "nji,nj->ni", aligned[:, :3, :3], aligned[:, :3, 3]
    )
    if not np.allclose(recovered, aligned_centers, atol=1e-9, rtol=1e-9):
        raise ArtifactError("aligned view matrices and camera centres disagree")
    return aligned, aligned_centers


def export_pose_artifact(
    workspace: str | Path,
    name: str,
    *,
    output_root: str | Path,
    refinement_workspace: str | Path | None = None,
) -> Path:
    """Publish one named, provenance-complete left-camera pose artifact."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"invalid pose artifact name: {name!r}")
    root, plan, config = _workspace_context(workspace)
    quality_marker = root / "stages" / "quality.json"
    quality_report_path = _report_path(root, "quality")
    text_model = root / "registered_text"
    text_manifest_path = root / "text_model_manifest.json"
    binary_manifest_path = root / "registered_model_manifest.json"
    diagnostic_root = root
    pose_prior_role = (
        "post_solve_georeferencing_and_heldout_evaluation_only"
    )
    pose_priors_constrain_mapper = False
    pose_priors_constrain_refinement = False
    refinement_provenance: dict[str, Any] = {}
    if refinement_workspace is not None:
        from rtk_splat.backends.rtk_refinement import refinement_export_context

        refined = refinement_export_context(refinement_workspace)
        if refined["source_backend"] != root:
            raise ArtifactError(
                "RTK refinement was prepared from a different mapper backend"
            )
        quality_marker = refined["quality_marker"]
        quality_report_path = refined["quality_report"]
        quality_report = refined["quality"]["visual_quality"]
        text_model = refined["text_model"]
        text_manifest_path = refined["text_model_manifest"]
        binary_manifest_path = refined["binary_model_manifest"]
        diagnostic_root = refined["workspace"]
        pose_prior_role = refined["pose_prior_role"]
        pose_priors_constrain_mapper = refined[
            "pose_priors_constrain_mapper"
        ]
        pose_priors_constrain_refinement = refined[
            "pose_priors_constrain_refinement"
        ]
        refinement_provenance = refined["provenance"]
    else:
        quality_report = _json(quality_report_path)
    if not quality_marker.is_file():
        raise ArtifactError("run the exact-registration quality stage before export")
    if (
        quality_report.get("missing_images")
        or quality_report.get("unexpected_images")
        or quality_report.get("registration_fraction") != 1.0
    ):
        raise ArtifactError("pose export requires exact left+right registration")
    if not quality_report.get("passed"):
        raise ArtifactError("refusing to export a pose artifact that failed quality")
    text_manifest = _json(text_manifest_path)
    _verify_tree(text_model, text_manifest)
    frontend = Path(plan["frontend_artifact"])
    output = Path(output_root).expanduser().resolve()
    destination = output / name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite pose artifact: {destination}")
    for immutable in _immutable_input_roots(frontend):
        if destination == immutable or immutable in destination.parents:
            raise ArtifactError(
                f"pose artifact output must be outside immutable input {immutable}"
            )
    manifest = _json(frontend / "frame_manifest.json")
    rows = manifest["frames"]
    poses = _poses_from_images_txt(text_model / "images.txt")
    left_names = [row["left_image"]["name"] for row in rows]
    missing = sorted(set(left_names) - set(poses))
    if missing:
        raise ArtifactError(f"final model lacks left-camera poses: {missing[:8]}")
    frame_ids = np.asarray([row["frame_id"] for row in rows], dtype=np.int64)
    timestamps = np.asarray([row["timestamp_ns"] for row in rows], dtype=np.int64)
    raw_viewmats = np.stack([poses[name][0] for name in left_names])
    raw_centers = np.stack([poses[name][1] for name in left_names])
    priors = _cartesian_camera_priors(root / "database.db", set(left_names))
    prior_names = [name for name in left_names if name in priors]
    source = np.stack([poses[name][1] for name in prior_names])
    target = np.stack([priors[name][0] for name in prior_names])
    covariance = np.stack([priors[name][1] for name in prior_names])
    prior_timestamps = np.asarray(
        [
            rows[index]["timestamp_ns"]
            for index, name in enumerate(left_names)
            if name in priors
        ],
        dtype=np.int64,
    )
    evaluation = estimate_temporal_heldout_alignment(
        source,
        target,
        prior_timestamps,
        covariance,
        temporal_blocks=config.alignment_temporal_blocks,
        ransac_threshold_m=config.alignment_ransac_threshold_m,
        ransac_iterations=config.alignment_ransac_iterations,
        random_seed=config.random_seed,
    )
    alignment = evaluation.alignment
    viewmats, centers = _apply_world_alignment(
        raw_viewmats,
        raw_centers,
        alignment.rotation,
        alignment.translation,
    )
    inputs = {
        "name": name,
        "quality_marker_sha256": sha256_file(quality_marker),
        "text_model_manifest_sha256": sha256_file(text_manifest_path),
        "frame_manifest_sha256": sha256_file(frontend / "frame_manifest.json"),
        "backend_plan_sha256": sha256_file(root / "backend_plan.json"),
        "pose_model_source": (
            "rtk_refinement" if refinement_workspace is not None else "mapper"
        ),
        **refinement_provenance,
        "pose_prior_assignment_sha256": canonical_hash(
            [
                {
                    "name": prior_name,
                    "position_m": priors[prior_name][0],
                    "covariance_m2": priors[prior_name][1],
                }
                for prior_name in prior_names
            ]
        ),
    }
    holdout_quality = rtk_residual_quality(
        evaluation.residual_vectors_m[evaluation.holdout_mask],
        covariance[evaluation.holdout_mask],
        evaluation.holdout_inlier_mask[evaluation.holdout_mask],
        config=config,
    )
    rtk_checks = {
        "fixed_scale_se3_applied": {
            "kind": "metric_integrity",
            "authoritative": True,
            "value": 1.0,
            "expected": 1.0,
            "passed": True,
        },
        **holdout_quality["checks"],
    }
    rtk_alignment_passed = all(
        check["passed"]
        for check in rtk_checks.values()
        if check["authoritative"]
    )
    backfilled_fields = sorted(
        set(_LEGACY_EVALUATION_DEFAULTS) - set(plan["config"])
    )
    gate_policy = {
        "schema_version": 1,
        "normalized_model": "chi_square_3d_camera_centre_residual",
        "covariance": "stored_position_covariance_in_target_frame",
        "covariance_gate_mode": config.rtk_covariance_gate_mode,
        "covariance_provenance_requirement": (
            "enforce only when upstream evidence establishes calibrated "
            "effective camera-centre covariance"
        ),
        "effective_config": {
            field: asdict(config)[field]
            for field in (
                "max_rtk_median_error_m",
                "max_rtk_p95_inlier_error_m",
                "min_rtk_inlier_fraction",
                "rtk_chi2_inlier_probability",
                "max_rtk_median_mahalanobis_sq",
                "min_rtk_chi2_inlier_fraction",
                "rtk_covariance_gate_mode",
            )
        },
        "legacy_backfilled_evaluation_fields": backfilled_fields,
        "absolute_caps_are_independent": True,
    }
    holdout_block_ids = evaluation.temporal_block_ids[
        evaluation.holdout_mask
    ]
    holdout_residuals = evaluation.residuals_m[evaluation.holdout_mask]
    holdout_mahalanobis_sq = np.asarray(
        holdout_quality["mahalanobis_sq"]["values"], dtype=np.float64
    )
    block_summaries = []
    for block_id in evaluation.holdout_block_ids:
        block_mask = holdout_block_ids == block_id
        block_summaries.append(
            {
                "block_id": block_id,
                "n_residuals": int(block_mask.sum()),
                "median_residual_m": float(
                    np.median(holdout_residuals[block_mask])
                ),
                "median_mahalanobis_sq": float(
                    np.median(holdout_mahalanobis_sq[block_mask])
                ),
            }
        )
    diagnostic_report = {
        "schema_version": 1,
        "stage": "pose_export_rtk_alignment_evaluation",
        "pose_artifact_name": name,
        "passed": rtk_alignment_passed,
        "inputs": inputs,
        "gate_policy": gate_policy,
        "n_calibration_priors": int(evaluation.calibration_mask.sum()),
        "n_holdout_priors": int(evaluation.holdout_mask.sum()),
        "holdout_block_summaries": block_summaries,
        "holdout_residual_vectors_m": evaluation.residual_vectors_m[
            evaluation.holdout_mask
        ].tolist(),
        "holdout_quality": holdout_quality,
        "checks": rtk_checks,
        "sim3_scale_diagnostic": alignment.sim3_scale_diagnostic,
    }
    diagnostic_identity = canonical_hash(
        {
            "inputs": inputs,
            "gate_policy": gate_policy,
            "temporal_block_ids": evaluation.temporal_block_ids,
        }
    )[:12]
    diagnostic_path = _report_path(
        diagnostic_root, f"pose_export_alignment_{name}_{diagnostic_identity}"
    )
    _atomic_json(diagnostic_path, diagnostic_report)
    if not rtk_alignment_passed:
        raise ArtifactError(
            "fixed-scale ENU alignment failed RTK residual gates; "
            f"diagnostics: {diagnostic_path}"
        )
    if (
        np.any(np.diff(frame_ids) <= 0)
        or np.any(np.diff(timestamps) <= 0)
        or not np.isfinite(viewmats).all()
        or not np.isfinite(centers).all()
    ):
        raise ArtifactError("exported pose arrays failed integrity checks")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{name}.writing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        _atomic_save_npy(staging / "viewmats.npy", viewmats)
        _atomic_save_npy(staging / "cam_centers.npy", centers)
        _atomic_save_npy(staging / "frame_ids.npy", frame_ids)
        _atomic_save_npy(staging / "timestamps_ns.npy", timestamps)
        _atomic_save_npy(staging / "left_image_names.npy", np.asarray(left_names))
        export_quality = {
            "schema_version": 1,
            "source_quality_report_sha256": sha256_file(
                quality_report_path
            ),
            "n_frames": len(frame_ids),
            "all_left_camera_poses_present": True,
            "all_frontend_images_registered": bool(quality_report["passed"]),
            "registration_fraction": quality_report["registration_fraction"],
            "mean_reprojection_error_px": quality_report["checks"][
                "mean_reprojection_error_px"
            ]["value"],
            "mean_track_length": quality_report["checks"]["mean_track_length"][
                "value"
            ],
            "coordinate_frame": "local_enu_from_cartesian_pose_priors",
            "n_trusted_rtk_priors": len(prior_names),
            "n_calibration_rtk_priors": int(evaluation.calibration_mask.sum()),
            "n_calibration_rtk_inliers": int(
                evaluation.calibration_inlier_mask.sum()
            ),
            "n_holdout_rtk_priors": int(evaluation.holdout_mask.sum()),
            "n_holdout_rtk_inliers": int(
                evaluation.holdout_inlier_mask.sum()
            ),
            "holdout_rtk_residual_m": holdout_quality["residual_m"],
            "holdout_rtk_mahalanobis_sq": holdout_quality[
                "mahalanobis_sq"
            ],
            "rtk_gate_policy": gate_policy,
            "rtk_alignment_checks": rtk_checks,
            "rtk_alignment_passed": rtk_alignment_passed,
            "rtk_alignment_diagnostic_report": str(diagnostic_path),
            "sim3_scale_diagnostic": alignment.sim3_scale_diagnostic,
            "sim3_scale_applied": False,
        }
        alignment_record = {
            "schema_version": 1,
            "method": (
                "covariance_weighted_ransac_fixed_scale_se3_"
                "with_temporal_holdout"
            ),
            "source_frame": "colmap_sfm_world",
            "target_frame": "local_enu_cartesian_pose_priors",
            "pose_prior_role": pose_prior_role,
            "pose_priors_constrain_mapper": pose_priors_constrain_mapper,
            "pose_priors_constrain_refinement": (
                pose_priors_constrain_refinement
            ),
            "production_scale": 1.0,
            "sim3_scale_diagnostic": alignment.sim3_scale_diagnostic,
            "sim3_scale_fit_membership": "calibration_blocks_only",
            "sim3_scale_applied": False,
            "rotation_target_source": alignment.rotation.tolist(),
            "translation_target_source_m": alignment.translation.tolist(),
            "source_rank": alignment.source_rank,
            "prior_image_names": prior_names,
            "prior_timestamps_ns": prior_timestamps.tolist(),
            "temporal_split": {
                "method": "contiguous_rank_blocks_alternating_v1",
                "requested_block_count": config.alignment_temporal_blocks,
                "realized_block_count": int(
                    len(np.unique(evaluation.temporal_block_ids))
                ),
                "block_id_by_prior": evaluation.temporal_block_ids.tolist(),
                "calibration_block_ids": list(
                    evaluation.calibration_block_ids
                ),
                "holdout_block_ids": list(evaluation.holdout_block_ids),
                "calibration_prior_image_names": [
                    prior_names[index]
                    for index in np.flatnonzero(evaluation.calibration_mask)
                ],
                "holdout_prior_image_names": [
                    prior_names[index]
                    for index in np.flatnonzero(evaluation.holdout_mask)
                ],
            },
            "calibration_inlier_mask": (
                evaluation.calibration_inlier_mask.tolist()
            ),
            "holdout_inlier_mask": evaluation.holdout_inlier_mask.tolist(),
            "residuals_m": evaluation.residuals_m.tolist(),
            "residual_vectors_m": evaluation.residual_vectors_m.tolist(),
            "ransac_thresholds_m": evaluation.thresholds_m.tolist(),
            "gate_policy": gate_policy,
            "holdout_quality": holdout_quality,
            "diagnostic_report": str(diagnostic_path),
            "checks": rtk_checks,
        }
        provenance = {
            "schema_version": 1,
            "frontend_artifact": str(frontend),
            "source_backend_workspace": str(root),
            "frontend_inputs": plan["frontend_inputs"],
            "backend": config.backend,
            "backend_config": plan["config"],
            "backend_config_effective": asdict(config),
            "legacy_backfilled_evaluation_fields": backfilled_fields,
            "database_snapshot_sha256": plan["database_sha256"],
            "solve_marker_sha256": sha256_file(root / "stages" / "solve.json"),
            "register_marker_sha256": sha256_file(
                root / "stages" / "register.json"
            ),
            "quality_marker_sha256": sha256_file(quality_marker),
            "backend_plan_sha256": inputs["backend_plan_sha256"],
            "frame_manifest_sha256": inputs["frame_manifest_sha256"],
            "pose_prior_assignment_sha256": inputs[
                "pose_prior_assignment_sha256"
            ],
            "registered_model_manifest_sha256": sha256_file(
                binary_manifest_path
            ),
            "text_model_manifest_sha256": inputs["text_model_manifest_sha256"],
            **refinement_provenance,
            "rtk_alignment_diagnostic_report": str(diagnostic_path),
            "rtk_alignment_diagnostic_sha256": sha256_file(diagnostic_path),
            "pose_convention": {
                "viewmats": "world_to_left_camera",
                "viewmat_camera_axes": "OpenCV_x_right_y_down_z_forward",
                "cam_centers": "left_camera_center_in_local_enu_m",
                "quaternion_input": "COLMAP_QW_QX_QY_QZ",
                "world_alignment": "fixed_scale_SE3_only",
            },
        }
        _atomic_json(staging / "quality.json", export_quality)
        _atomic_json(staging / "alignment.json", alignment_record)
        _atomic_json(staging / "provenance.json", provenance)
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
                "schema_version": 1,
                "name": name,
                "n_frames": len(frame_ids),
                "files": file_evidence,
            },
        )
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _verify_pose_artifact(destination)
    return destination




def run_quality_summary(
    workspace: str | Path,
    executable: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Convert the final model to text and verify exact image registration."""
    root, plan, config = _workspace_context(workspace)
    register_marker = root / "stages" / "register.json"
    if not register_marker.is_file():
        raise ArtifactError("run image registration before quality summary")
    _verify_tree(
        root / "registered_model", _json(root / "registered_model_manifest.json")
    )
    converter = build_model_converter_command(root, executable)
    analyzer = build_model_analyzer_command(root / "registered_model", executable)
    inputs = {
        "converter_command": converter,
        "analyzer_command": analyzer,
        "register_marker_sha256": sha256_file(register_marker),
        "expected_image_names_sha256": canonical_hash(plan["all_image_names"]),
        "max_reprojection_error_px": config.max_reprojection_error_px,
        "min_mean_track_length": config.min_mean_track_length,
    }
    ledger = StageLedger(root)
    status = ledger.begin("quality", inputs)
    if status == "complete":
        _verify_tree(root / "registered_text", _json(root / "text_model_manifest.json"))
        return _json(_report_path(root, "quality"))

    text_model = root / "registered_text"
    incomplete = root / "registered_text.incomplete"
    if not text_model.exists():
        if incomplete.exists():
            shutil.rmtree(incomplete)
        incomplete.mkdir()
        try:
            _execute(converter, runner)
            if not (incomplete / "images.txt").is_file():
                raise ArtifactError("model_converter produced no images.txt")
            os.rename(incomplete, text_model)
        except Exception:
            shutil.rmtree(incomplete, ignore_errors=True)
            raise
    names = registered_names_from_images_txt(text_model / "images.txt")
    stats = _analyze_model(root / "registered_model", executable, runner)
    report = quality_summary(
        plan["all_image_names"],
        names,
        stats,
        max_reprojection_error_px=config.max_reprojection_error_px,
        min_mean_track_length=config.min_mean_track_length,
    )
    report.update(
        {
            "stage": "quality",
            "converter_command": list(converter),
            "analyzer_command": list(analyzer),
        }
    )
    if config.require_all_frames and (
        report["missing_images"] or report["unexpected_images"]
    ):
        raise ArtifactError("final model failed exact all-image registration")
    return _complete_stage(
        root,
        ledger,
        "quality",
        inputs,
        report,
        "text_model_manifest.json",
        _tree_manifest(text_model),
    )
