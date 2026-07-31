"""Synthetic recovery and observability tests for the calibration sidecar."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from diagnostics.metric_calibration import (
    CalibrationData,
    CalibrationProblem,
    LocalWgs84Enu,
    RigPrior,
    SolverSettings,
    _solve,
    _tangent_basis,
    calibrate,
    eligible_camera_indices,
    ned_covariance_to_enu,
    ned_to_enu,
    observability,
)


def _rich_pose(time_s):
    time_s = np.asarray(time_s)
    centers = np.stack([
        0.25 * time_s + 0.8 * np.sin(0.35 * time_s),
        1.5 * np.sin(0.22 * time_s) + 0.02 * time_s ** 2,
        0.4 * np.sin(0.41 * time_s) + 0.015 * time_s,
    ], axis=-1)
    euler_zyx = np.stack([
        0.12 * time_s + 0.15 * np.sin(0.17 * time_s),
        0.18 * np.sin(0.31 * time_s),
        0.12 * np.cos(0.27 * time_s),
    ], axis=-1)
    return centers, Rotation.from_euler("ZYX", euler_zyx).as_matrix()


def _settings(**overrides):
    values = dict(
        initial_clock_offset_s=0.0,
        clock_offset_bounds_s=(-0.2, 0.2),
        lever_bounds_m=np.array([0.20, 0.20, 0.20]),
        baseline_angle_bounds_rad=np.radians([6.0, 6.0]),
        prior_sigma=np.array([
            0.12, 0.12, 0.12,
            np.radians(4.0), np.radians(4.0), 0.12,
        ]),
        position_sigma_floor_m=0.02,
        baseline_sigma_floor_m=0.01,
        sample_interval_s=0.5,
        temporal_blocks=6,
        holdout_block_stride=3,
        holdout_block_offset=1,
        max_nfev=150,
        clock_grid_steps=9,
        minimum_std_reduction=0.05,
        maximum_start_std_prior_fraction=1.0,
        maximum_fold_std_prior_fraction=2.0,
        minimum_heldout_improvement_fraction=0.001,
        maximum_heldout_regression_fraction=0.10,
    )
    values.update(overrides)
    return SolverSettings(**values)


def _rich_problem(true_correction=None, settings=None):
    rig = RigPrior(
        body_from_camera_rotation=Rotation.from_euler(
            "ZYX", [0.20, -0.15, 0.08]).as_matrix(),
        camera_in_body_m=np.array([0.30, -0.10, 0.20]),
        position_antenna_in_body_m=np.array([0.90, 0.10, 0.50]),
        baseline_body_m=np.array([1.35, 0.12, 0.08]),
    )
    truth = np.array([
        0.05, -0.04, 0.03,
        np.radians(2.0), np.radians(-1.5), 0.065,
    ]) if true_correction is None else np.asarray(true_correction, dtype=float)
    tangent = _tangent_basis(rig.baseline_camera_m)
    true_baseline = Rotation.from_rotvec(
        tangent @ truth[3:5]).as_matrix() @ rig.baseline_camera_m
    true_lever = rig.lever_camera_m + truth[:3]
    global_rotation = Rotation.from_euler(
        "ZYX", [0.35, -0.08, 0.12]).as_matrix()
    global_translation = np.array([4.0, -2.0, 1.2])

    camera_t = np.arange(0.0, 30.0001, 0.25)
    centers, rotations = _rich_pose(camera_t)
    gnss_t = np.arange(-0.5, 30.5001, 0.025)
    measurement_centers, measurement_rotations = _rich_pose(
        gnss_t - truth[5])
    antenna = np.einsum(
        "ij,nj->ni", global_rotation,
        measurement_centers + np.einsum(
            "nij,j->ni", measurement_rotations, true_lever),
    ) + global_translation
    baseline = np.einsum(
        "ij,nj->ni", global_rotation,
        np.einsum(
            "nij,j->ni", measurement_rotations, true_baseline),
    )
    random = np.random.default_rng(7)
    antenna += random.normal(0.0, 0.002, antenna.shape)
    baseline += random.normal(0.0, 0.001, baseline.shape)
    position_covariance = np.tile(
        np.eye(3) * 0.02 ** 2, (len(gnss_t), 1, 1))
    baseline_covariance = np.tile(
        np.eye(3) * 0.01 ** 2, (len(gnss_t), 1, 1))
    data = CalibrationData(
        camera_t_s=camera_t,
        camera_centers_visual_m=centers,
        visual_from_camera_rotation=rotations,
        frame_ids=np.arange(len(camera_t)),
        fix_t_s=gnss_t,
        antenna_enu_m=antenna,
        antenna_cov_enu_m2=position_covariance,
        fix_good=np.ones(len(gnss_t), dtype=bool),
        baseline_t_s=gnss_t,
        baseline_enu_m=baseline,
        baseline_cov_enu_m2=baseline_covariance,
        baseline_good=np.ones(len(gnss_t), dtype=bool),
    )
    initial_rotation = Rotation.from_euler(
        "ZYX", [0.30, -0.05, 0.09]).as_matrix()
    return (
        CalibrationProblem(
            data, rig, settings or _settings(),
            initial_global_rotation=initial_rotation),
        truth,
    )


def _degenerate_problem(stationary: bool):
    camera_t = np.arange(0.0, 30.0001, 0.25)
    gnss_t = np.arange(-0.5, 30.5001, 0.05)
    if stationary:
        camera_centers = np.zeros((len(camera_t), 3))
        antenna = np.tile([1.0, 2.0, 3.0], (len(gnss_t), 1))
    else:
        camera_centers = np.stack([
            0.2 * camera_t,
            np.zeros_like(camera_t),
            np.zeros_like(camera_t),
        ], axis=1)
        antenna = np.stack([
            0.2 * gnss_t + 1.0,
            np.zeros_like(gnss_t),
            np.zeros_like(gnss_t),
        ], axis=1)
    camera_rotations = np.tile(
        np.eye(3), (len(camera_t), 1, 1))
    baseline_vector = np.array([1.40, 0.10, 0.05])
    baseline = np.tile(baseline_vector, (len(gnss_t), 1))
    covariance = np.tile(
        np.eye(3) * 0.01 ** 2, (len(gnss_t), 1, 1))
    data = CalibrationData(
        camera_t_s=camera_t,
        camera_centers_visual_m=camera_centers,
        visual_from_camera_rotation=camera_rotations,
        frame_ids=np.arange(len(camera_t)),
        fix_t_s=gnss_t,
        antenna_enu_m=antenna,
        antenna_cov_enu_m2=covariance,
        fix_good=np.ones(len(gnss_t), dtype=bool),
        baseline_t_s=gnss_t,
        baseline_enu_m=baseline,
        baseline_cov_enu_m2=covariance,
        baseline_good=np.ones(len(gnss_t), dtype=bool),
    )
    rig = RigPrior(
        body_from_camera_rotation=np.eye(3),
        camera_in_body_m=np.zeros(3),
        position_antenna_in_body_m=np.array([1.0, 0.0, 0.0]),
        baseline_body_m=baseline_vector,
    )
    return CalibrationProblem(data, rig, _settings(max_nfev=50))


class MetricCalibrationTests(unittest.TestCase):
    def test_ned_to_enu_vector_and_covariance(self):
        vector = np.array([3.0, 4.0, -2.0])
        self.assertTrue(np.array_equal(
            ned_to_enu(vector), [4.0, 3.0, 2.0]))
        covariance_ned = np.diag([1.0, 4.0, 9.0])
        covariance_enu = ned_covariance_to_enu(covariance_ned)
        self.assertTrue(np.array_equal(
            np.diag(covariance_enu), [4.0, 1.0, 9.0]))

    def test_wgs84_local_enu_matches_known_equatorial_displacements(self):
        converter = LocalWgs84Enu(0.0, 0.0, 0.0)
        coordinates = converter.to_enu(
            [0.0, 1.0e-5, 0.0],
            [1.0e-5, 0.0, 0.0],
            [0.0, 0.0, 2.0],
        )
        expected = np.array([
            [1.1131949079327301, 0.0, -9.685754776000977e-08],
            [0.0, 1.1057427582159383, -9.592622518539429e-08],
            [0.0, 0.0, 2.0],
        ])
        self.assertTrue(np.allclose(coordinates, expected, atol=1e-8))

    def test_rich_six_dof_motion_recovers_tf_and_clock_on_heldout_blocks(self):
        problem, truth = _rich_problem()
        result = calibrate(problem)
        candidate = np.asarray(result["candidate"]["correction"])

        self.assertTrue(np.allclose(candidate[:3], truth[:3], atol=0.02))
        self.assertTrue(np.allclose(
            candidate[3:5], truth[3:5], atol=np.radians(0.2)))
        self.assertLess(abs(candidate[5] - truth[5]), 0.02)
        self.assertTrue(all(
            item["trusted"] for item in result["parameters"].values()))
        self.assertLess(
            result["candidate"]["heldout"]["position_median_m"],
            result["baseline_fixed_scale"]["heldout"]["position_median_m"],
        )

        retained = np.asarray(
            result["final_retained_prior_safe"]["correction"])
        transform = problem.corrected_body_from_camera(retained)
        lever, baseline_camera, _ = problem.corrected_geometry(retained)
        self.assertTrue(np.allclose(
            transform[:3, :3] @ baseline_camera,
            problem.rig.baseline_body_m, atol=1e-10))
        self.assertTrue(np.allclose(
            transform[:3, 3] + transform[:3, :3] @ lever,
            problem.rig.position_antenna_in_body_m, atol=1e-10))
        self.assertEqual(
            result["final_retained_prior_safe"]["tf_correction"]
            ["baseline_axis_twist_delta_deg"],
            0.0,
        )

    def test_straight_constant_speed_detects_lever_and_clock_degeneracy(self):
        problem = _degenerate_problem(stationary=False)
        indices = eligible_camera_indices(problem.data, problem.settings)
        aligned = _solve(problem, indices, active_calibration=[])
        report = observability(problem, indices, aligned)
        reduction = np.asarray(report["std_reduction_fraction"])

        self.assertLess(np.max(reduction[:3]), 1e-6)
        self.assertLess(reduction[5], 1e-6)
        self.assertIn(
            "rotation_about_dual_antenna_baseline",
            report["structurally_retained_prior"],
        )

    def test_stationary_no_rotation_rejects_every_calibration_mode(self):
        problem = _degenerate_problem(stationary=True)
        indices = eligible_camera_indices(problem.data, problem.settings)
        aligned = _solve(problem, indices, active_calibration=[])
        report = observability(problem, indices, aligned)

        self.assertLess(np.max(report["std_reduction_fraction"]), 1e-6)
        self.assertEqual(report["projected_jacobian_rank"], 0)

    def test_true_correction_beyond_bound_is_not_reported_trustworthy(self):
        truth = np.array([
            0.27, -0.04, 0.03,
            np.radians(2.0), np.radians(-1.5), 0.065,
        ])
        problem, _ = _rich_problem(
            true_correction=truth,
            settings=_settings(
                lever_bounds_m=np.array([0.08, 0.20, 0.20])))
        result = calibrate(problem)
        parameter = result["parameters"]["lever_camera_x_m"]

        self.assertGreater(parameter["bound_fraction"], 0.90)
        self.assertFalse(parameter["trusted"])
        self.assertEqual(parameter["retained_correction"], 0.0)
        self.assertTrue(any(
            "bound" in reason for reason in parameter["reasons"]))


if __name__ == "__main__":
    unittest.main()
