"""Orchestration for the explicit, non-destructive metric-integrity audit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .calibration_io import (
    atomic_calibration_artifact,
    calibration_artifact_path,
    load_raw_colmap_rig_trajectory,
    validate_calibration_artifact_name,
    write_json,
)
from .metric_calibration import (
    CALIBRATION_PARAMETER_NAMES,
    CalibrationData,
    CalibrationProblem,
    LocalWgs84Enu,
    RigPrior,
    SolveResult,
    SolverSettings,
    calibrate,
)


def _required(namespace, name: str):
    if not hasattr(namespace, name):
        raise ValueError(f"calibration.{name} is required")
    return getattr(namespace, name)


def _vector(namespace, name: str, length: int) -> np.ndarray:
    value = np.asarray(_required(namespace, name), dtype=float)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(
            f"calibration.{name} must contain {length} finite values")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(paths: list[Path]) -> dict[str, str]:
    result = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required immutable input is missing: {path}")
        result[str(path.resolve())] = _sha256(path)
    return result


@dataclass(frozen=True)
class AuditConfig:
    source_pose_artifact: str
    output_artifact: str
    moving_base_pvt_topic: str
    body_frame: str
    camera_frame: str
    primary_antenna_frame: str
    secondary_antenna_frame: str
    body_from_camera: np.ndarray
    primary_antenna_body_m: np.ndarray
    secondary_antenna_body_m: np.ndarray
    solver: SolverSettings
    log_time_margin_s: float
    status_max_age_s: float
    minimum_navsat_status: int
    minimum_position_carrier_solution: int
    minimum_baseline_carrier_solution: int
    baseline_length_tolerance_fraction: float

    def __post_init__(self) -> None:
        frames = (
            self.body_frame,
            self.camera_frame,
            self.primary_antenna_frame,
            self.secondary_antenna_frame,
        )
        if any(not value.strip() for value in frames):
            raise ValueError("calibration frame names cannot be empty")
        if len(set(frames)) != len(frames):
            raise ValueError("calibration frame names must be distinct")
        if self.body_from_camera.shape != (4, 4) \
                or not np.isfinite(self.body_from_camera).all():
            raise ValueError("body_from_camera must be a finite 4x4 transform")
        if not np.allclose(
                self.body_from_camera[3], [0, 0, 0, 1], atol=1e-12):
            raise ValueError("body_from_camera has an invalid last row")
        if self.log_time_margin_s < 0:
            raise ValueError("log_time_margin_s cannot be negative")
        if self.status_max_age_s <= 0:
            raise ValueError("status_max_age_s must be positive")
        if not 0 < self.baseline_length_tolerance_fraction < 1:
            raise ValueError(
                "baseline_length_tolerance_fraction must lie in (0,1)")

    @classmethod
    def from_config(cls, cfg) -> "AuditConfig":
        if not hasattr(cfg, "calibration"):
            raise ValueError(
                "this explicit stage requires a calibration: config section")
        section = cfg.calibration
        source = validate_calibration_artifact_name(
            str(_required(section, "source_pose_artifact")))
        output = validate_calibration_artifact_name(
            str(_required(section, "output_artifact")))
        if source == output:
            raise ValueError("source and output artifact names must differ")

        geometry = _required(section, "physical_geometry")
        transform = _required(geometry, "body_from_left_camera")
        translation = _vector(transform, "translation_m", 3)
        quaternion = _vector(transform, "quaternion_xyzw", 4)
        quaternion_norm = float(np.linalg.norm(quaternion))
        if abs(quaternion_norm - 1.0) > 1e-3:
            raise ValueError(
                "calibration body-from-camera quaternion must be unit length")
        body_from_camera = np.eye(4)
        body_from_camera[:3, :3] = Rotation.from_quat(
            quaternion).as_matrix()
        body_from_camera[:3, 3] = translation

        optimization = _required(section, "optimization")
        lever_bounds = _vector(
            optimization, "lever_correction_bounds_m", 3)
        angle_bounds_rad = np.radians(_vector(
            optimization, "baseline_angle_correction_bounds_deg", 2))
        lever_prior = _vector(
            optimization, "lever_prior_sigma_m", 3)
        angle_prior_rad = np.radians(_vector(
            optimization, "baseline_angle_prior_sigma_deg", 2))
        prior_sigma = np.concatenate([
            lever_prior, angle_prior_rad,
            [float(_required(optimization, "clock_prior_sigma_s"))],
        ])
        solver = SolverSettings(
            initial_clock_offset_s=float(
                _required(optimization, "initial_clock_offset_s")),
            clock_offset_bounds_s=tuple(float(v) for v in _vector(
                optimization, "clock_offset_bounds_s", 2)),
            lever_bounds_m=lever_bounds,
            baseline_angle_bounds_rad=angle_bounds_rad,
            prior_sigma=prior_sigma,
            position_sigma_floor_m=float(
                _required(optimization, "position_sigma_floor_m")),
            baseline_sigma_floor_m=float(
                _required(optimization, "baseline_sigma_floor_m")),
            sample_interval_s=float(
                _required(optimization, "sample_interval_s")),
            temporal_blocks=int(
                _required(optimization, "temporal_blocks")),
            holdout_block_stride=int(
                _required(optimization, "holdout_block_stride")),
            holdout_block_offset=int(
                _required(optimization, "holdout_block_offset")),
            robust_loss=str(getattr(
                optimization, "robust_loss", "soft_l1")),
            robust_f_scale=float(getattr(
                optimization, "robust_f_scale", 1.0)),
            max_nfev=int(getattr(optimization, "max_nfev", 400)),
            clock_grid_steps=int(getattr(
                optimization, "clock_grid_steps", 21)),
            minimum_std_reduction=float(getattr(
                optimization, "minimum_std_reduction", 0.20)),
            maximum_bound_fraction=float(getattr(
                optimization, "maximum_bound_fraction", 0.90)),
            maximum_start_std_prior_fraction=float(getattr(
                optimization, "maximum_start_std_prior_fraction", 0.35)),
            maximum_fold_std_prior_fraction=float(getattr(
                optimization, "maximum_fold_std_prior_fraction", 0.50)),
            minimum_correction_to_fold_std_ratio=float(getattr(
                optimization,
                "minimum_correction_to_fold_std_ratio", 2.0)),
            maximum_heldout_regression_fraction=float(getattr(
                optimization, "maximum_heldout_regression_fraction", 0.02)),
            maximum_heldout_position_rms_regression_fraction=float(getattr(
                optimization,
                "maximum_heldout_position_rms_regression_fraction", 0.02)),
            maximum_heldout_position_p95_regression_fraction=float(getattr(
                optimization,
                "maximum_heldout_position_p95_regression_fraction", 0.05)),
            minimum_heldout_improvement_fraction=float(getattr(
                optimization, "minimum_heldout_improvement_fraction", 0.005)),
            minimum_heldout_baseline_improvement_fraction=float(getattr(
                optimization,
                "minimum_heldout_baseline_improvement_fraction", 0.005)),
        )
        return cls(
            source_pose_artifact=source,
            output_artifact=output,
            moving_base_pvt_topic=str(
                _required(section, "moving_base_pvt_topic")),
            body_frame=str(_required(geometry, "body_frame")),
            camera_frame=str(_required(geometry, "camera_frame")),
            primary_antenna_frame=str(
                _required(geometry, "primary_antenna_frame")),
            secondary_antenna_frame=str(
                _required(geometry, "secondary_antenna_frame")),
            body_from_camera=body_from_camera,
            primary_antenna_body_m=_vector(
                geometry, "primary_antenna_position_body_m", 3),
            secondary_antenna_body_m=_vector(
                geometry, "secondary_antenna_position_body_m", 3),
            solver=solver,
            log_time_margin_s=float(getattr(
                section, "log_time_margin_s", 5.0)),
            status_max_age_s=float(getattr(
                section, "status_max_age_s", 0.15)),
            minimum_navsat_status=int(getattr(
                section, "minimum_navsat_status", 0)),
            minimum_position_carrier_solution=int(getattr(
                section, "minimum_position_carrier_solution", 2)),
            minimum_baseline_carrier_solution=int(getattr(
                section, "minimum_baseline_carrier_solution", 2)),
            baseline_length_tolerance_fraction=float(getattr(
                section, "baseline_length_tolerance_fraction", 0.03)),
        )

    @property
    def rig_prior(self) -> RigPrior:
        return RigPrior(
            body_from_camera_rotation=self.body_from_camera[:3, :3],
            camera_in_body_m=self.body_from_camera[:3, 3],
            position_antenna_in_body_m=self.primary_antenna_body_m,
            baseline_body_m=(
                self.secondary_antenna_body_m
                - self.primary_antenna_body_m),
        )


@dataclass(frozen=True)
class RawVisualSource:
    """Backend-neutral location of one immutable raw COLMAP trajectory."""

    model: Path
    database: Path
    initial_visual_to_enu_rotation: np.ndarray
    backend: str


def resolve_raw_visual_source(
        pose_artifact: Path, quality: dict) -> RawVisualSource:
    """Resolve raw metric inputs from a supported pose-artifact schema.

    Absolute paths recorded in quality reports are treated as provenance only.
    Resolution is relative to the artifact so copied work directories remain
    auditable.
    """
    source = Path(pose_artifact)
    if "selected_model" in quality and "alignment" in quality:
        model_name = Path(str(quality["selected_model"])).name
        model = source / "colmap" / "models_text" / model_name
        database = source / "colmap" / "database.db"
        rotation_value = quality["alignment"].get(
            "rotation_visual_world_to_enu")
        backend = "incremental_stereo"
    elif "model_stats" in quality and "fixed_scale_alignment" in quality:
        model_name = Path(str(quality["model_stats"].get("path", ""))).name
        model = source / "global" / "models_text" / model_name
        database = source / "global" / "database.db"
        rotation_value = quality["fixed_scale_alignment"].get(
            "rotation_visual_world_to_enu")
        backend = "global_mapper"
    else:
        raise ValueError(
            "pose artifact does not expose a supported raw COLMAP trajectory")

    if not model_name or model_name in {".", ".."}:
        raise ValueError("pose artifact has no selected COLMAP model name")
    for required in (model / "frames.txt", model / "rigs.txt", database):
        if not required.is_file():
            raise FileNotFoundError(
                f"required raw visual input is missing: {required}")

    rotation = np.asarray(rotation_value, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(
            "pose artifact has no finite 3x3 visual-to-ENU rotation")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-6) \
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
        raise ValueError(
            "pose artifact visual-to-ENU rotation is not a valid rotation")
    return RawVisualSource(
        model=model,
        database=database,
        initial_visual_to_enu_rotation=rotation,
        backend=backend,
    )


def _nearest_indices(reference_ns: np.ndarray,
                     query_ns: np.ndarray) -> np.ndarray:
    if len(reference_ns) == 0:
        raise ValueError("cannot match against an empty timestamp stream")
    right = np.searchsorted(reference_ns, query_ns, side="left")
    right = np.clip(right, 0, len(reference_ns) - 1)
    left = np.clip(right - 1, 0, len(reference_ns) - 1)
    choose_left = (
        np.abs(query_ns - reference_ns[left])
        <= np.abs(reference_ns[right] - query_ns))
    return np.where(choose_left, left, right)


def _unique_time_rows(times: np.ndarray, *arrays):
    _, first = np.unique(times, return_index=True)
    first.sort()
    return (times[first],) + tuple(np.asarray(array)[first] for array in arrays)


def _build_numerical_data(observations, trajectory, meta: dict,
                          audit: AuditConfig) \
        -> tuple[CalibrationData, dict, dict[str, np.ndarray]]:
    payload = observations.to_npz_payload()
    camera_ns = payload["stereo_left_header_ns"].astype(np.int64)
    frame_ids = payload["stereo_frame_id"].astype(np.int64)
    if not np.array_equal(frame_ids, trajectory.frame_indices):
        raise ValueError(
            "recovered bag frame IDs disagree with raw COLMAP trajectory")

    fix_ns = payload["fix_header_ns"].astype(np.int64)
    geodetic = payload["fix_geodetic"].astype(float)
    fix_cov = payload["fix_covariance_enu_m2"].astype(float)
    fix_status = payload["fix_status"].astype(int)
    fix_cov_type = payload["fix_covariance_type"].astype(int)

    pvt_ns = payload["pvt_header_ns"].astype(np.int64)
    pvt_nearest = _nearest_indices(pvt_ns, fix_ns)
    pvt_age_s = np.abs(fix_ns - pvt_ns[pvt_nearest]) * 1e-9
    pvt_flags = payload["pvt_status_flags"].astype(bool)
    pvt_good_all = (
        pvt_flags[:, 0]       # gnss_fix_ok
        & pvt_flags[:, 1]     # differential solution
        & ~pvt_flags[:, 2]    # not invalid_llh
        & (payload["pvt_gps_fix_type"].astype(int) >= 3)
        & (payload["pvt_carrier_solution"].astype(int)
           >= audit.minimum_position_carrier_solution)
    )
    fix_cov_finite = np.isfinite(fix_cov).all(axis=(1, 2))
    fix_good = (
        (fix_status >= audit.minimum_navsat_status)
        & (fix_cov_type > 0)
        & fix_cov_finite
        & (pvt_age_s <= audit.status_max_age_s)
        & pvt_good_all[pvt_nearest]
    )

    rel_ns = payload["relpos_header_ns"].astype(np.int64)
    baseline_enu = payload["relpos_enu_m"].astype(float)
    rel_accuracy_ned = payload["relpos_accuracy_ned_m"].astype(float)
    # Variance is insensitive to the sign flip on Down->Up.
    rel_accuracy_enu = rel_accuracy_ned[:, [1, 0, 2]]
    baseline_cov = np.zeros((len(rel_ns), 3, 3), dtype=float)
    baseline_cov[:, range(3), range(3)] = rel_accuracy_enu ** 2
    rel_flags = payload["relpos_flags"].astype(bool)
    baseline_good = (
        rel_flags[:, 0]       # gnss_fix_ok
        & rel_flags[:, 1]     # differential solution
        & rel_flags[:, 2]     # vector valid
        & rel_flags[:, 3]     # moving-base mode
        & ~rel_flags[:, 4]    # no missing reference position
        & ~rel_flags[:, 5]    # no missing reference observations
        & rel_flags[:, 6]     # heading/vector direction valid
        & (payload["relpos_carrier_solution"].astype(int)
           >= audit.minimum_baseline_carrier_solution)
    )

    origin = meta["world_origin"]
    altitude = origin.get("alt0")
    if altitude is None:
        altitude = origin.get("alt0_ellipsoidal_m")
    if altitude is None:
        raise ValueError(
            "world_origin must provide ellipsoidal altitude as alt0 or "
            "alt0_ellipsoidal_m"
        )
    enu = LocalWgs84Enu(
        origin["lat0"], origin["lon0"], altitude)
    raw_fix_enu = enu.to_enu(
        geodetic[:, 0], geodetic[:, 1], geodetic[:, 2])

    raw_fix_ns = fix_ns.copy()
    raw_rel_ns = rel_ns.copy()
    raw_fix_good = fix_good.copy()
    raw_baseline_cov = baseline_cov.copy()
    raw_baseline_good = baseline_good.copy()
    fix_ns, fix_enu, fix_cov, fix_good, solver_pvt_age_s = _unique_time_rows(
        fix_ns, raw_fix_enu, fix_cov, fix_good, pvt_age_s)
    rel_ns, baseline_enu, baseline_cov, baseline_good = _unique_time_rows(
        rel_ns, baseline_enu, baseline_cov, baseline_good)
    time_origin_ns = int(min(camera_ns[0], fix_ns[0], rel_ns[0]))
    seconds = lambda value: (value - time_origin_ns).astype(float) * 1e-9
    rotations_visual_camera = np.swapaxes(
        trajectory.camera_from_visual_world[:, :3, :3], 1, 2)
    data = CalibrationData(
        camera_t_s=seconds(camera_ns),
        camera_centers_visual_m=trajectory.camera_centers_visual,
        visual_from_camera_rotation=rotations_visual_camera,
        frame_ids=frame_ids,
        fix_t_s=seconds(fix_ns),
        antenna_enu_m=fix_enu,
        antenna_cov_enu_m2=fix_cov,
        fix_good=fix_good,
        baseline_t_s=seconds(rel_ns),
        baseline_enu_m=baseline_enu,
        baseline_cov_enu_m2=baseline_cov,
        baseline_good=baseline_good,
    )

    configured_length = float(np.linalg.norm(
        audit.rig_prior.baseline_body_m))
    observed_lengths = np.linalg.norm(baseline_enu, axis=1)
    observed_median = float(np.median(observed_lengths))
    length_fraction = abs(observed_median - configured_length) \
        / configured_length
    if length_fraction > audit.baseline_length_tolerance_fraction:
        raise ValueError(
            f"configured baseline {configured_length:.6f} m disagrees with "
            f"observed median {observed_median:.6f} m by "
            f"{length_fraction:.1%}")
    derived = {
        # Raw-derived arrays remain one-to-one with the exact observations.
        "fix_enu_m": raw_fix_enu,
        "fix_good_for_calibration": raw_fix_good,
        "fix_pvt_nearest_age_s": pvt_age_s,
        "baseline_covariance_enu_m2": raw_baseline_cov,
        "baseline_good_for_calibration": raw_baseline_good,
        # Solver inputs are named separately in case duplicate header stamps
        # need deterministic reduction.  Raw evidence is never discarded.
        "solver_fix_header_ns": fix_ns,
        "solver_fix_enu_m": fix_enu,
        "solver_fix_covariance_enu_m2": fix_cov,
        "solver_fix_good": fix_good,
        "solver_fix_pvt_nearest_age_s": solver_pvt_age_s,
        "solver_relpos_header_ns": rel_ns,
        "solver_relpos_enu_m": baseline_enu,
        "solver_relpos_covariance_enu_m2": baseline_cov,
        "solver_relpos_good": baseline_good,
    }
    diagnostics = {
        "time_origin_ns": time_origin_ns,
        "n_camera_frames": int(len(camera_ns)),
        "n_raw_fixes": int(len(raw_fix_ns)),
        "n_solver_fixes": int(len(fix_ns)),
        "n_duplicate_fix_header_stamps":
            int(len(raw_fix_ns) - len(fix_ns)),
        "n_good_fixes": int(fix_good.sum()),
        "n_raw_baselines": int(len(raw_rel_ns)),
        "n_solver_baselines": int(len(rel_ns)),
        "n_duplicate_relpos_header_stamps":
            int(len(raw_rel_ns) - len(rel_ns)),
        "n_good_baselines": int(baseline_good.sum()),
        "stereo_sync_abs_max_ms": float(
            np.max(np.abs(payload["stereo_delta_ns"])) * 1e-6),
        "pvt_match_age_p95_ms": float(
            np.percentile(pvt_age_s, 95) * 1000),
        "configured_baseline_length_m": configured_length,
        "observed_baseline_length_median": observed_median,
        "observed_baseline_length_min":
            float(observed_lengths.min()),
        "observed_baseline_length_max":
            float(observed_lengths.max()),
        "configured_vs_observed_length_fraction": float(length_fraction),
        "local_enu": enu.crs(),
    }
    return data, diagnostics, derived


def _model_from_report(section: dict, correction: np.ndarray) -> SolveResult:
    return SolveResult(
        global_rotation=np.asarray(
            section["global_rotation_visual_to_enu"], dtype=float),
        global_translation_m=np.asarray(
            section["global_translation_enu_m"], dtype=float),
        calibration=np.asarray(correction, dtype=float),
        scale=1.0,
        success=True,
        cost=0.0,
        optimality=0.0,
        nfev=0,
        message="reconstructed from immutable report",
    )


def _human_report(result: dict, diagnostics: dict,
                  audit: AuditConfig) -> str:
    final = result["final_retained_prior_safe"]
    baseline = result["baseline_fixed_scale"]["heldout"]
    after = final["heldout"]
    lines = [
        "# RTK–Stereo Metric-Integrity Audit",
        "",
        f"- Calibration accepted: **{final['calibration_accepted']}**",
        f"- Source visual artifact: `{audit.source_pose_artifact}`",
        "- Stereo scale: fixed at 1.0 in calibration",
        (f"- Sim(3)-only diagnostic scale: "
         f"{result['sim3_diagnostic_fixed_physical_prior']['scale_visual_to_enu']:.9f}"),
        (f"- Held-out antenna median: "
         f"{baseline['position_median_m']*100:.2f} cm → "
         f"{after['position_median_m']*100:.2f} cm"),
        (f"- Held-out antenna p95: "
         f"{baseline['position_p95_m']*100:.2f} cm → "
         f"{after['position_p95_m']*100:.2f} cm"),
        (f"- Held-out antenna RMS: "
         f"{baseline['position_rms_m']*100:.2f} cm → "
         f"{after['position_rms_m']*100:.2f} cm"),
        (f"- Held-out baseline-angle median: "
         f"{baseline['baseline_angle_median_deg']:.3f}° → "
         f"{after['baseline_angle_median_deg']:.3f}°"),
        (f"- Baseline length configured/observed: "
         f"{diagnostics['configured_baseline_length_m']:.6f} / "
         f"{diagnostics['observed_baseline_length_median']:.6f} m"),
        "",
        "## Parameter trust",
        "",
        "| Parameter | Joint candidate | One-mode diagnostic | Retained | "
        "Fold SNR | Trusted | Reason |",
        "|---|---:|---:|---:|---:|:---:|---|",
    ]
    for name in CALIBRATION_PARAMETER_NAMES:
        parameter = result["parameters"][name]
        reason = "; ".join(parameter["reasons"]) or "all gates passed"
        lines.append(
            f"| `{name}` | {parameter['candidate_correction']:+.7f} | "
            f"{parameter['independent_correction']:+.7f} | "
            f"{parameter['retained_correction']:+.7f} | "
            f"{parameter['correction_to_temporal_refit_std_ratio']:.2f} | "
            f"{'yes' if parameter['trusted'] else 'no'} | {reason} |")
    lines.extend([
        "",
        "## Held-out temporal blocks",
        "",
        "| Block | Frames | Position median | Position RMS | Position p95 | "
        "Baseline angle | Passed |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for block in final["heldout_blocks"]:
        before = block["baseline"]
        retained = block["retained"]
        lines.append(
            f"| {block['block_id']} | {len(block['frame_ids'])} | "
            f"{before['position_median_m']*100:.2f}→"
            f"{retained['position_median_m']*100:.2f} cm | "
            f"{before['position_rms_m']*100:.2f}→"
            f"{retained['position_rms_m']*100:.2f} cm | "
            f"{before['position_p95_m']*100:.2f}→"
            f"{retained['position_p95_m']*100:.2f} cm | "
            f"{before['baseline_angle_median_deg']:.3f}→"
            f"{retained['baseline_angle_median_deg']:.3f}° | "
            f"{'yes' if block['passed'] else 'no'} |")
    lines.extend([
        "",
        "Rotation about the physical dual-antenna baseline is structurally "
        "unobservable with one baseline and remains exactly at its configured "
        "prior.",
        "",
        "This artifact contains no `viewmats.npy` and is not consumed by "
        "normal cloud or GS training.",
        "",
    ])
    return "\n".join(lines)


def run_integrity_audit(segment_dir: Path, cfg) -> tuple[Path, dict]:
    """Run one bounded audit and atomically publish a diagnostic artifact."""
    # Keep the diagnostics package importable without the optional ROS stack.
    from ..adapters.calibration_bag import (
        CalibrationTopics,
        build_calibration_typestore,
        read_calibration_bag_observations,
    )

    seg = Path(segment_dir)
    audit = AuditConfig.from_config(cfg)
    final_path = calibration_artifact_path(seg, audit.output_artifact)
    if final_path.exists() or final_path.is_symlink():
        raise FileExistsError(
            f"calibration artifact already exists: {final_path}")
    meta = json.loads((seg / "segment_meta.json").read_text())
    n_frames = int(meta["n_frames"])
    source = seg / "pose_artifacts" / audit.source_pose_artifact
    quality_path = source / "quality.json"
    quality = json.loads(quality_path.read_text())
    raw_source = resolve_raw_visual_source(source, quality)
    model = raw_source.model
    database = raw_source.database

    immutable_paths = [
        seg / "viewmats.npy",
        seg / "cam_centers.npy",
        source / "viewmats.npy",
        source / "cam_centers.npy",
        quality_path,
        model / "frames.txt",
        model / "rigs.txt",
    ]
    print(
        f"integrity audit: verifying immutable source "
        f"{audit.source_pose_artifact}", flush=True)
    hashes_before = _source_hashes(immutable_paths)
    database_stat_before = (
        database.stat().st_size, database.stat().st_mtime_ns)

    print("integrity audit: loading raw fixed-scale COLMAP rig poses",
          flush=True)
    trajectory = load_raw_colmap_rig_trajectory(
        model, database, expected_frame_count=n_frames)
    window = meta["window"]
    # Reproduce the original image interval in the camera header clock.
    # Extraction maps camera stamps into the pose clock by adding this value.
    historical_pose_offset_s = float(meta.get("time_offset_s", 0.0))
    image_start_ns = int(round(
        (float(window["t0"]) - historical_pose_offset_s) * 1e9))
    image_stop_ns = int(round(
        (float(window["t1"]) - historical_pose_offset_s) * 1e9))
    clock_low, clock_high = audit.solver.clock_offset_bounds_s
    rtk_padding_ns = int(2.0e9)
    rtk_start_ns = image_start_ns + int(np.floor(clock_low * 1e9)) \
        - rtk_padding_ns
    rtk_stop_ns = image_stop_ns + int(np.ceil(clock_high * 1e9)) \
        + rtk_padding_ns
    topics = CalibrationTopics(
        left_image=str(cfg.topics.left_image),
        right_image=str(cfg.topics.right_image),
        fix=str(cfg.topics.fix),
        relpos=str(cfg.topics.relpos),
        moving_base_pvt=audit.moving_base_pvt_topic,
    )
    typestore = build_calibration_typestore(
        Path(cfg.paths.ublox_msgs_dir).expanduser()
    )
    print(
        "integrity audit: bounded bag pass for exact stereo/RTK evidence",
        flush=True)
    observations = read_calibration_bag_observations(
        bags=cfg.paths.bags,
        topics=topics,
        typestore=typestore,
        image_start_header_ns=image_start_ns,
        image_stop_header_ns=image_stop_ns,
        frame_stride=int(cfg.segment.frame_stride),
        extracted_images_dir=seg / "images",
        expected_frame_count=n_frames,
        rtk_start_header_ns=rtk_start_ns,
        rtk_stop_header_ns=rtk_stop_ns,
        default_log_margin_ns=int(round(audit.log_time_margin_s * 1e9)),
    )
    data, diagnostics, derived = _build_numerical_data(
        observations, trajectory, meta, audit)
    print(
        f"integrity audit: recovered {len(observations.stereo_frames)} stereo "
        f"pairs, {len(observations.fixes)} fixes, "
        f"{len(observations.relpos)} baseline vectors", flush=True)
    initial_rotation = raw_source.initial_visual_to_enu_rotation
    problem = CalibrationProblem(
        data, audit.rig_prior, audit.solver,
        initial_global_rotation=initial_rotation)
    zero = np.zeros(len(CALIBRATION_PARAMETER_NAMES))
    all_camera = np.arange(len(data.camera_t_s))
    initial_measured_baseline = problem.measurements(
        all_camera, audit.solver.initial_clock_offset_s)[2]
    initial_predicted_baseline = problem.predictions(
        all_camera, initial_rotation, np.zeros(3), zero)[1]
    dot = np.sum(
        initial_measured_baseline * initial_predicted_baseline, axis=1)
    denominator = (
        np.linalg.norm(initial_measured_baseline, axis=1)
        * np.linalg.norm(initial_predicted_baseline, axis=1))
    cosine = np.clip(dot / np.maximum(denominator, 1e-12), -1.0, 1.0)
    reversed_fraction = float(np.mean(cosine < 0.0))
    diagnostics["baseline_direction_prior_check"] = {
        "median_angle_deg": float(
            np.degrees(np.arccos(np.median(cosine)))),
        "reversed_dot_fraction": reversed_fraction,
        "uses_quality_alignment_only_as_initialization": True,
    }
    if reversed_fraction > 0.5:
        raise ValueError(
            "configured primary-to-secondary baseline direction is reversed "
            "relative to the RTK vector and visual trajectory")
    print(
        "integrity audit: bounded fixed-scale solve, observability, and "
        "held-out temporal validation", flush=True)
    result = calibrate(problem)

    # Preserve per-frame unaligned residuals for later plots and independent
    # review without needing to rerun the optimizer.
    frame_to_index = {
        int(frame): i for i, frame in enumerate(data.frame_ids)}
    eligible = np.asarray([
        frame_to_index[int(frame)]
        for frame in result["eligible_frame_ids"]], dtype=int)
    fit_set = set(result["fit_frame_ids"])
    fit_mask = np.asarray([
        int(frame) in fit_set for frame in result["eligible_frame_ids"]],
        dtype=bool)
    baseline_result = _model_from_report(
        result["baseline_fixed_scale"], np.zeros(6))
    final_section = result["final_retained_prior_safe"]
    final_result = _model_from_report(
        final_section, np.asarray(final_section["correction"]))
    before_pos, before_base, before_angle = problem.raw_errors(
        eligible, baseline_result)
    after_pos, after_base, after_angle = problem.raw_errors(
        eligible, final_result)

    provenance = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose":
            "read-only RTK-stereo metric integrity and physical extrinsic audit",
        "source_pose_artifact": audit.source_pose_artifact,
        "source_pose_backend": raw_source.backend,
        "output_artifact": audit.output_artifact,
        "selected_raw_colmap_model": str(model.resolve()),
        "raw_colmap_input":
            "frames.txt + rigs.txt + read-only database; exported Sim(3) "
            "poses were not used as metric input",
        "bag_read_strategy":
            "single forward scan over the indexed bounded MCAP chunk range; "
            "no full-bag scan and no source writes",
        "normal_pipeline_enabled": False,
        "publishes_pose_artifact": False,
        "runs_colmap": False,
        "runs_gaussian_training": False,
        "source_sha256": hashes_before,
        "source_database_stat": {
            "size_bytes": database_stat_before[0],
            "mtime_ns": database_stat_before[1],
        },
        "topic_bags": dict(observations.topic_bags),
        "timestamp_note":
            "exact header and MCAP log nanoseconds preserved; u-blox header "
            "timestamps are host receive time, while iTOW preserves GNSS epoch",
        "historical_pose_time_offset_s": historical_pose_offset_s,
        "coordinate_conventions": {
            "visual_world": "raw metric COLMAP stereo world",
            "global_world": "local WGS84 ENU from segment origin",
            "camera": audit.camera_frame,
            "body": audit.body_frame,
            "baseline_from": audit.primary_antenna_frame,
            "baseline_to": audit.secondary_antenna_frame,
            "relpos_conversion": "NED -> ENU [E,N,-D]",
            "clock_model":
                "GNSS header time = camera header time + clock_offset_s",
        },
        "physical_prior": {
            "body_from_left_camera": audit.body_from_camera.tolist(),
            "primary_antenna_position_body_m":
                audit.primary_antenna_body_m.tolist(),
            "secondary_antenna_position_body_m":
                audit.secondary_antenna_body_m.tolist(),
            "camera_to_primary_antenna_camera_m":
                audit.rig_prior.lever_camera_m.tolist(),
            "primary_to_secondary_baseline_camera_m":
                audit.rig_prior.baseline_camera_m.tolist(),
            "legacy_pose_yaw_offset_used": False,
        },
    }

    print(
        f"integrity audit: atomically publishing new artifact "
        f"{audit.output_artifact}", flush=True)
    with atomic_calibration_artifact(seg, audit.output_artifact) as work:
        observation_payload = observations.to_npz_payload()
        observation_payload.update(derived)
        np.savez_compressed(
            work / "observations.npz", **observation_payload)
        np.savez_compressed(
            work / "raw_visual_poses.npz",
            frame_indices=trajectory.frame_indices,
            colmap_frame_ids=trajectory.colmap_frame_ids,
            image_ids=trajectory.image_ids,
            image_names=np.asarray(trajectory.image_names),
            camera_from_visual_world=
                trajectory.camera_from_visual_world,
            camera_centers_visual=
                trajectory.camera_centers_visual)
        np.savez_compressed(
            work / "residuals.npz",
            frame_ids=data.frame_ids[eligible],
            fit_mask=fit_mask,
            before_position_enu_m=before_pos,
            after_position_enu_m=after_pos,
            before_baseline_enu_m=before_base,
            after_baseline_enu_m=after_base,
            before_baseline_angle_deg=before_angle,
            after_baseline_angle_deg=after_angle)
        observability_report = result["observability"]
        np.savez_compressed(
            work / "observability.npz",
            global_alignment_singular_values=np.asarray(
                observability_report[
                    "global_alignment_singular_values"]),
            singular_values=np.asarray(
                observability_report[
                    "singular_values_dimensionless"]),
            right_singular_vectors=np.asarray(
                observability_report["right_singular_vectors"]),
            data_information_matrix=np.asarray(
                observability_report[
                    "data_information_matrix_dimensionless"]),
            prior_sigma=np.asarray(
                observability_report["prior_sigma"]),
            posterior_sigma=np.asarray(
                observability_report[
                    "posterior_sigma_linearized"]),
            std_reduction_fraction=np.asarray(
                observability_report[
                    "std_reduction_fraction"]))
        write_json(work / "diagnostics.json", diagnostics)
        write_json(work / "result.json", result)
        write_json(work / "provenance.json", provenance)
        write_json(work / "temporal_folds.json", {
            "fit_frame_ids": result["fit_frame_ids"],
            "heldout_frame_ids": result["heldout_frame_ids"],
            "eligible_temporal_block_ids":
                result["eligible_temporal_block_ids"],
            "heldout_block_results":
                result["final_retained_prior_safe"]["heldout_blocks"],
        })
        (work / "REPORT.md").write_text(
            _human_report(result, diagnostics, audit))

        hashes_after = _source_hashes(immutable_paths)
        database_stat_after = (
            database.stat().st_size, database.stat().st_mtime_ns)
        if hashes_after != hashes_before \
                or database_stat_after != database_stat_before:
            raise RuntimeError(
                "immutable source changed during audit; refusing to publish")

    return final_path, result
