"""Strict structural schema for authored RTK-Splat workflow configuration.

The runtime still exposes configuration as ``SimpleNamespace`` for backwards
compatibility.  This module closes the dangerous part of that interface:
misspelled or obsolete YAML keys are rejected before any long-running stage is
started.  Leaves are validated by their owning component, where the numerical
requirements are known; this schema owns mapping structure and key names.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


_LEAF = None


def _leaves(*names: str) -> dict[str, None]:
    return {name: _LEAF for name in names}


CONFIG_SCHEMA: dict[str, Any] = {
    "profile": _LEAF,
    "robot": _LEAF,
    "adapter": _LEAF,
    # Third-party adapters put dataset-specific options in this one explicit
    # namespace and validate it themselves. They never expand the core schema
    # or gain permission to add arbitrary top-level keys.
    "adapter_options": _LEAF,
    "paths": _leaves(
        "workdir",
        "segment",
        "bags",
        "camera_bags",
        "gnss_bags",
        "dataset_root",
        "ground_truth_csv",
        "ublox_msgs_dir",
    ),
    "topics": _leaves(
        "left_image",
        "right_image",
        "left_info",
        "right_info",
        "fix",
        "pvt",
        "relpos",
        "imu",
        "depth",
        "depth_info",
        "confidence",
        "receiver_state",
    ),
    "pose": _leaves(
        "source",
        "artifact",
        "artifact_root",
        "trajectory_file",
        "clockcheck_topic",
        "time_offset_s",
        "antenna_forward_m",
        "min_carr_soln",
        "yaw_smooth_window",
        "min_course_speed_ms",
        "minimum_navsat_status",
        "minimum_position_carrier_status",
        "maximum_position_covariance_m2",
        "use_imu_tilt",
        "imu_lp_window_s",
        # Legacy mount fields remain accepted for frozen reproduction files.
        "cam_forward_m",
        "cam_left_m",
        "cam_up_m",
        "pitch_down_deg",
        "yaw_offset_deg",
        "center_up_m",
        "left_eye_forward_m",
        "left_eye_left_m",
        "baseline_m",
    ),
    "segment": _leaves(
        "window_s",
        "window_epoch_source",
        "search_t_start_s",
        "heading_std_max_deg",
        "speed_range_ms",
        "length_s",
        "frame_stride",
        "frame_spacing_m",
        "stereo_tolerance_s",
        "association_tolerance_s",
        "validation_every",
        "expected_camera_bag_count",
        "expected_gnss_bag_count",
        "maximum_bag_gap_s",
        "maximum_bag_overlap_s",
        "image_encoding",
    ),
    "timing": _leaves(
        "clock_model",
        "maximum_offset_error_s",
        "maximum_window_drift_s",
    ),
    "gnss_quality": _leaves(
        "receiver_state_required",
        "receiver_state_association_tolerance_s",
        "covariance_provenance",
        "covariance_is_live_per_epoch",
    ),
    "sensor_geometry": {
        **_leaves(
            "T_camera_primary_antenna",
            "extrinsic_translation_sigma_m",
            "extrinsic_camera_geometry",
        ),
        "frames": _leaves(
            "world", "camera", "primary_antenna", "secondary_antenna"
        ),
        "extrinsic_provenance": {
            **_leaves("method", "status"),
            # Source records are intentionally opaque citation metadata.
            "sources": _LEAF,
        },
    },
    "depth": {
        **_leaves("backend", "min_z_m", "max_z_m"),
        "sgbm": _leaves(
            "num_disparities",
            "block_size",
            "uniqueness_ratio",
            "speckle_window",
            "speckle_range",
        ),
    },
    "frontend": {
        **_leaves("name", "seed", "turn_rate_deg_s"),
        "keyframes": _leaves(
            "preset",
            "select_all_frames",
            "translation_m",
            "rotation_deg",
            "max_elapsed_s",
            "min_blur_score",
            "min_exposure_quality",
            "max_stereo_sync_residual_s",
            "max_rtk_covariance_m2",
            "acceptable_rtk_status",
            "revisit_distance_m",
            "revisit_min_separation_s",
            "revisit_force_spacing_s",
            "revisit_points_per_cell",
        ),
        "features": _leaves(
            "profile",
            "max_image_size",
            "num_threads",
            "max_num_features",
            "gpu_index",
        ),
        "matching": _leaves("use_gpu", "gpu_index", "num_threads"),
        "pose_priors": _leaves(
            "covariance_floor_m",
            "max_covariance_m2",
            "min_fix_status",
            "min_carrier_status",
            "max_source_residual_s",
        ),
        "pairs": _leaves(
            "temporal_max_distance_m",
            "temporal_max_seconds",
            "max_temporal_neighbors",
            "revisit_distance_m",
            "revisit_min_separation_s",
            "max_revisit_neighbors",
            "max_cross_frame_degree",
            "max_view_angle_deg",
            "registration_max_distance_m",
            "registration_max_seconds",
            "max_registration_neighbors",
            "match_right_camera_temporally",
        ),
    },
    "mapper": _leaves(
        "name",
        "pose_artifact_name",
        "backend",
        "num_threads",
        "random_seed",
        "min_num_matches",
        "ba_num_iterations",
        "keep_max_num_tracks",
        "track_required_tracks_per_view",
        "skip_retriangulation",
        "gp_use_gpu",
        "ba_ceres_use_gpu",
        "process_nice",
        "resource_sample_interval_s",
        "minimum_available_memory_gb",
        "low_memory_consecutive_samples",
        "minimum_free_space_gb",
        "minimum_runtime_free_space_gb",
        "require_all_keyframes",
        "require_all_frames",
        "max_reprojection_error_px",
        "min_mean_track_length",
        "alignment_ransac_threshold_m",
        "alignment_ransac_iterations",
        "alignment_temporal_blocks",
        "max_rtk_median_error_m",
        "max_rtk_p95_inlier_error_m",
        "min_rtk_inlier_fraction",
        "rtk_chi2_inlier_probability",
        "max_rtk_median_mahalanobis_sq",
        "min_rtk_chi2_inlier_fraction",
        "rtk_covariance_gate_mode",
    ),
    "rtk_refinement": _leaves(
        "name",
        "num_threads",
        "random_seed",
        "ba_global_max_num_iterations",
        "ba_global_max_refinements",
        "initialization_mode",
        "prior_position_loss",
        "prior_position_loss_scale",
        "process_nice",
        "resource_sample_interval_s",
        "minimum_available_memory_gb",
        "low_memory_consecutive_samples",
        "minimum_free_space_gb",
        "minimum_runtime_free_space_gb",
        "max_reprojection_regression_px",
        "min_track_length_retention",
        "fresh_min_mean_track_length",
        "fresh_min_mean_observations_per_image",
        "max_trajectory_scale_deviation",
        "max_stereo_baseline_change_m",
        "max_calibration_parameter_change",
        "max_holdout_median_regression_m",
        "min_holdout_median_improvement_m",
    ),
    "cloud": _leaves(
        "artifact_root", "pixel_stride", "voxel_m", "max_points"
    ),
    "colmap": _leaves(
        "executable",
        # Legacy diagnostic-sidecar controls remain schema-valid.
        "max_image_size",
        "max_num_features",
        "feature_num_threads",
        "sequential_overlap",
        "alignment_max_error_m",
        "alignment_ransac_iterations",
        "min_registered_fraction",
        "scale_range",
    ),
    "train": {
        **_leaves(
            "run_name",
            "seed",
            "max_gaussians",
            "iterations",
            "ssim_lambda",
            "depth_lambda",
            "opacity_reg",
            "scale_reg",
            "holdout_every",
            "eval_every",
            "sh_degree",
            "init_opacity",
            "init_scale_mult",
            "mask_dilate_px",
            "min_scale_m",
            "max_scale_m",
            "max_anisotropy",
            "export_prune_opacity",
            "export_crop",
            "rasterize_mode",
            "refine_stop_frac",
            "means_lr_final_mult",
            "color_finetune_frac",
            "use_right_camera",
            "eval_pose_align_steps",
            "eval_pose_align_lr",
        ),
        "pose_opt": _leaves(
            "enabled",
            "start_iter",
            "lr",
            "weight_decay",
            "max_trans_m",
            "max_rot_deg",
            "trans_penalty",
            "rot_penalty",
        ),
        "exposure_opt": _leaves("enabled", "lr", "weight_decay"),
        "lr": _leaves("means", "scales", "quats", "opacities", "sh0", "shN"),
    },
    "agrigs": _leaves("dataset_dir", "camera", "intrinsic", "distortion"),
    "calibration": {
        **_leaves(
            "source_pose_artifact",
            "output_artifact",
            "moving_base_pvt_topic",
            "log_time_margin_s",
            "status_max_age_s",
            "minimum_navsat_status",
            "minimum_position_carrier_solution",
            "minimum_baseline_carrier_solution",
            "baseline_length_tolerance_fraction",
        ),
        "physical_geometry": {
            **_leaves(
                "body_frame",
                "camera_frame",
                "primary_antenna_frame",
                "secondary_antenna_frame",
                "primary_antenna_position_body_m",
                "secondary_antenna_position_body_m",
            ),
            "body_from_left_camera": _leaves(
                "translation_m", "quaternion_xyzw"
            ),
        },
        "optimization": _leaves(
            "initial_clock_offset_s",
            "clock_offset_bounds_s",
            "lever_correction_bounds_m",
            "baseline_angle_correction_bounds_deg",
            "lever_prior_sigma_m",
            "baseline_angle_prior_sigma_deg",
            "clock_prior_sigma_s",
            "position_sigma_floor_m",
            "baseline_sigma_floor_m",
            "sample_interval_s",
            "temporal_blocks",
            "holdout_block_stride",
            "holdout_block_offset",
            "robust_loss",
            "robust_f_scale",
            "max_nfev",
            "clock_grid_steps",
            "minimum_std_reduction",
            "maximum_bound_fraction",
            "maximum_start_std_prior_fraction",
            "maximum_fold_std_prior_fraction",
            "minimum_correction_to_fold_std_ratio",
            "maximum_heldout_regression_fraction",
            "maximum_heldout_position_rms_regression_fraction",
            "maximum_heldout_position_p95_regression_fraction",
            "minimum_heldout_improvement_fraction",
            "minimum_heldout_baseline_improvement_fraction",
        ),
    },
    # Policies are authored; the resolved values/report are runtime output and
    # deliberately cannot be supplied in YAML.
    "derivation": {
        "frame_sampling": _leaves("target_spacing_m"),
        "depth": _leaves("min_reliable_disparity_px", "max_z_cap_m"),
        "training": _leaves(
            "image_presentations_per_view",
            "min_iterations",
            "max_iterations",
            "gaussians_per_gib",
            "reserve_vram_gib",
            "initial_cloud_growth_factor",
            "min_gaussians",
            "max_gaussians",
            "vram_gib_override",
        ),
    },
}


def _subset(name: str, *keys: str) -> dict[str, Any]:
    section = CONFIG_SCHEMA[name]
    return {key: section[key] for key in keys}


# Ownership is intentionally narrower than the structural schema. Profiles
# own method policy, robots own stable sensor facts, and sequences own inputs
# plus a short list of evidence-based run overrides. Frozen reproductions and
# standalone legacy configurations remain self-contained by design.
LAYER_SCHEMAS: dict[str, dict[str, Any]] = {
    "profile": {
        "profile": _LEAF,
        "segment": _subset(
            "segment",
            "stereo_tolerance_s",
            "association_tolerance_s",
            "validation_every",
        ),
        "depth": CONFIG_SCHEMA["depth"],
        "frontend": {
            key: value
            for key, value in CONFIG_SCHEMA["frontend"].items()
            if key != "name"
        },
        "mapper": {
            key: value
            for key, value in CONFIG_SCHEMA["mapper"].items()
            if key not in {"name", "pose_artifact_name"}
        },
        "rtk_refinement": {
            key: value
            for key, value in CONFIG_SCHEMA["rtk_refinement"].items()
            if key != "name"
        },
        "cloud": _subset("cloud", "pixel_stride", "voxel_m", "max_points"),
        "colmap": CONFIG_SCHEMA["colmap"],
        "train": {
            key: value
            for key, value in CONFIG_SCHEMA["train"].items()
            if key != "run_name"
        },
        "derivation": CONFIG_SCHEMA["derivation"],
    },
    "robot": {
        "robot": _LEAF,
        "adapter": _LEAF,
        "adapter_options": _LEAF,
        "paths": _subset("paths", "ublox_msgs_dir"),
        "topics": CONFIG_SCHEMA["topics"],
        "pose": CONFIG_SCHEMA["pose"],
        "segment": _subset(
            "segment",
            "maximum_bag_gap_s",
            "maximum_bag_overlap_s",
            "image_encoding",
        ),
        "timing": CONFIG_SCHEMA["timing"],
        "gnss_quality": CONFIG_SCHEMA["gnss_quality"],
        "sensor_geometry": CONFIG_SCHEMA["sensor_geometry"],
        "frontend": {
            "pose_priors": CONFIG_SCHEMA["frontend"]["pose_priors"],
        },
    },
    "sequence": {
        "profile": _LEAF,
        "robot": _LEAF,
        "adapter_options": _LEAF,
        "paths": _subset(
            "paths",
            "workdir",
            "segment",
            "bags",
            "camera_bags",
            "gnss_bags",
        ),
        "pose": _subset(
            "pose",
            "artifact",
            "artifact_root",
            "trajectory_file",
            "time_offset_s",
        ),
        "segment": _subset(
            "segment",
            "window_s",
            "window_epoch_source",
            "search_t_start_s",
            "heading_std_max_deg",
            "speed_range_ms",
            "length_s",
            "frame_stride",
            "frame_spacing_m",
            "maximum_bag_gap_s",
            "maximum_bag_overlap_s",
        ),
        # Robot profiles provide the normal clock contract, but a recording
        # may override its validation model/gates with explicit evidence.
        "timing": CONFIG_SCHEMA["timing"],
        # Scene range can be an explicit measured override of profile auto.
        "depth": _subset("depth", "min_z_m", "max_z_m"),
        # Motion/topology changes justify these frontend overrides; feature
        # extraction and matching implementation remain profile-owned.
        "frontend": {
            "name": _LEAF,
            "turn_rate_deg_s": _LEAF,
            "keyframes": CONFIG_SCHEMA["frontend"]["keyframes"],
            "pairs": CONFIG_SCHEMA["frontend"]["pairs"],
        },
        # Artifact names and independent acceptance caps are run-specific.
        "mapper": _subset(
            "mapper",
            "name",
            "pose_artifact_name",
            "alignment_ransac_threshold_m",
            "alignment_temporal_blocks",
            "max_rtk_median_error_m",
            "max_rtk_p95_inlier_error_m",
            "min_rtk_inlier_fraction",
            "rtk_chi2_inlier_probability",
            "max_rtk_median_mahalanobis_sq",
            "min_rtk_chi2_inlier_fraction",
            "rtk_covariance_gate_mode",
        ),
        # This explicitly invoked experiment is configured per run and never
        # becomes part of normal mapping merely by appearing here.
        "rtk_refinement": CONFIG_SCHEMA["rtk_refinement"],
        "train": _subset("train", "run_name", "iterations", "max_gaussians"),
    },
    "reproduction": CONFIG_SCHEMA,
}


def _validate_node(value: Any, schema: Any, location: str, owner: Path) -> None:
    if schema is _LEAF:
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"{owner}: {location} must be a mapping")
    unknown = sorted(set(value) - set(schema))
    if unknown:
        names = ", ".join(f"{location}.{name}" for name in unknown)
        raise ValueError(f"{owner}: unknown configuration option(s): {names}")
    for key, item in value.items():
        _validate_node(item, schema[key], f"{location}.{key}", owner)


def validate_config_mapping(
    value: Mapping[str, Any],
    owner: str | Path,
    *,
    layer: str = "reproduction",
) -> None:
    """Reject unknown structural keys with their complete dotted paths."""
    path = Path(owner)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")
    if layer not in LAYER_SCHEMAS:
        raise ValueError(f"unknown configuration layer: {layer}")
    schema = LAYER_SCHEMAS[layer]
    unknown = sorted(set(value) - set(schema))
    if unknown:
        raise ValueError(
            f"{path}: {layer} layer cannot own option(s): " + ", ".join(unknown)
        )
    for key, item in value.items():
        _validate_node(item, schema[key], key, path)
