"""Observable RTK--stereo metric-integrity calibration.

This module deliberately does *not* publish camera poses.  It calibrates and
audits the rigid relationship between an already-solved, metric stereo camera
trajectory and:

* the primary GNSS antenna position, and
* the full primary-to-secondary dual-antenna baseline vector.

The visual trajectory remains fixed.  Stereo scale is fixed in the primary
calibration and is estimated only in a separate diagnostic.  With one antenna
baseline, rotation about that baseline is structurally unobservable; it is not
an optimization variable and therefore remains at the supplied physical
prior.

All optimization timestamps are relative seconds for numerical conditioning.
The I/O sidecar preserves the corresponding integer nanosecond timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


CALIBRATION_PARAMETER_NAMES = (
    "lever_camera_x_m",
    "lever_camera_y_m",
    "lever_camera_z_m",
    "baseline_tangent_0_rad",
    "baseline_tangent_1_rad",
    "clock_offset_delta_s",
)


@dataclass(frozen=True)
class LocalWgs84Enu:
    """Dependency-free WGS84 geodetic to local ENU conversion.

    Coordinates are first converted to WGS84 Earth-centred Earth-fixed (ECEF)
    and then rotated into the tangent frame at the configured ellipsoidal
    origin.  This is the same ellipsoidal construction used by standard
    geodesy libraries; it is not a small-angle/equirectangular approximation.
    """

    latitude_deg: float
    longitude_deg: float
    altitude_ellipsoid_m: float

    def __post_init__(self) -> None:
        values = np.asarray([
            self.latitude_deg,
            self.longitude_deg,
            self.altitude_ellipsoid_m,
        ], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("local ENU origin must be finite")
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("local ENU origin latitude is invalid")

    @staticmethod
    def _ecef(latitude_deg, longitude_deg, altitude_m) -> np.ndarray:
        latitude = np.radians(np.asarray(latitude_deg, dtype=float))
        longitude = np.radians(np.asarray(longitude_deg, dtype=float))
        altitude = np.asarray(altitude_m, dtype=float)
        latitude, longitude, altitude = np.broadcast_arrays(
            latitude, longitude, altitude)
        semi_major_m = 6_378_137.0
        flattening = 1.0 / 298.257_223_563
        eccentricity_squared = flattening * (2.0 - flattening)
        sin_latitude = np.sin(latitude)
        prime_vertical_radius = semi_major_m / np.sqrt(
            1.0 - eccentricity_squared * sin_latitude ** 2)
        x = (
            prime_vertical_radius + altitude
        ) * np.cos(latitude) * np.cos(longitude)
        y = (
            prime_vertical_radius + altitude
        ) * np.cos(latitude) * np.sin(longitude)
        z = (
            prime_vertical_radius * (1.0 - eccentricity_squared)
            + altitude
        ) * sin_latitude
        return np.stack([x, y, z], axis=-1)

    def to_enu(self, latitude_deg, longitude_deg,
               altitude_ellipsoid_m) -> np.ndarray:
        origin = self._ecef(
            self.latitude_deg,
            self.longitude_deg,
            self.altitude_ellipsoid_m)
        delta = self._ecef(
            latitude_deg, longitude_deg, altitude_ellipsoid_m) - origin
        latitude = np.radians(self.latitude_deg)
        longitude = np.radians(self.longitude_deg)
        rotation = np.array([
            [-np.sin(longitude), np.cos(longitude), 0.0],
            [-np.sin(latitude) * np.cos(longitude),
             -np.sin(latitude) * np.sin(longitude),
             np.cos(latitude)],
            [np.cos(latitude) * np.cos(longitude),
             np.cos(latitude) * np.sin(longitude),
             np.sin(latitude)],
        ])
        return np.einsum("ij,...j->...i", rotation, delta)

    def crs(self) -> dict:
        return {
            "type": "local_ENU",
            "ellipsoid": "WGS84",
            "origin_lat": float(self.latitude_deg),
            "origin_lon": float(self.longitude_deg),
            "origin_alt_ellipsoidal":
                float(self.altitude_ellipsoid_m),
            "vertical_datum": "WGS84 ellipsoid (not orthometric)",
            "implementation": "WGS84 geodetic -> ECEF -> tangent ENU",
        }


def ned_to_enu(values: np.ndarray) -> np.ndarray:
    """Convert NED vectors to ENU vectors without changing their magnitude."""
    a = np.asarray(values, dtype=float)
    if a.shape[-1] != 3:
        raise ValueError(f"NED vectors must end in 3 components, got {a.shape}")
    return np.stack([a[..., 1], a[..., 0], -a[..., 2]], axis=-1)


def ned_covariance_to_enu(covariance: np.ndarray) -> np.ndarray:
    """Convert one or more 3x3 covariance matrices from NED to ENU."""
    c = np.asarray(covariance, dtype=float)
    if c.shape[-2:] != (3, 3):
        raise ValueError(f"NED covariance must end in (3,3), got {c.shape}")
    p = np.array([[0.0, 1.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0]])
    return np.einsum("ij,...jk,lk->...il", p, c, p)


def _as_vector(value, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _as_rotation(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(result @ result.T, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} is not orthonormal")
    if not np.isclose(np.linalg.det(result), 1.0, atol=1e-6):
        raise ValueError(f"{name} is not a proper rotation")
    return result


def _tangent_basis(unit_vector: np.ndarray) -> np.ndarray:
    """Two rotation axes perpendicular to ``unit_vector``."""
    u = _as_vector(unit_vector, 3, "unit_vector")
    norm = np.linalg.norm(u)
    if norm < 1e-9:
        raise ValueError("cannot construct a tangent basis for a zero vector")
    u = u / norm
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(seed, u))) > 0.85:
        seed = np.array([0.0, 1.0, 0.0])
    axis0 = np.cross(u, seed)
    axis0 /= np.linalg.norm(axis0)
    axis1 = np.cross(u, axis0)
    axis1 /= np.linalg.norm(axis1)
    return np.stack([axis0, axis1], axis=1)


@dataclass(frozen=True)
class RigPrior:
    """Unambiguous physical rig geometry.

    ``body_from_camera_rotation`` maps left optical-camera vectors into the
    configured body frame.  Translations are origins expressed in that body
    frame.  ``baseline_body_m`` points from the position antenna to the second
    antenna.
    """

    body_from_camera_rotation: np.ndarray
    camera_in_body_m: np.ndarray
    position_antenna_in_body_m: np.ndarray
    baseline_body_m: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "body_from_camera_rotation",
            _as_rotation(self.body_from_camera_rotation,
                         "body_from_camera_rotation"))
        object.__setattr__(
            self, "camera_in_body_m",
            _as_vector(self.camera_in_body_m, 3, "camera_in_body_m"))
        object.__setattr__(
            self, "position_antenna_in_body_m",
            _as_vector(self.position_antenna_in_body_m, 3,
                       "position_antenna_in_body_m"))
        baseline = _as_vector(self.baseline_body_m, 3, "baseline_body_m")
        if np.linalg.norm(baseline) < 0.1:
            raise ValueError("dual-antenna baseline is implausibly short")
        object.__setattr__(self, "baseline_body_m", baseline)

    @property
    def lever_camera_m(self) -> np.ndarray:
        """Vector from left camera to position antenna, in camera axes."""
        r_cb = self.body_from_camera_rotation.T
        return r_cb @ (
            self.position_antenna_in_body_m - self.camera_in_body_m)

    @property
    def baseline_camera_m(self) -> np.ndarray:
        """Position-to-second-antenna baseline, in camera axes."""
        return self.body_from_camera_rotation.T @ self.baseline_body_m


@dataclass(frozen=True)
class CalibrationData:
    """Numerical observations needed by the estimator.

    Camera rotations map camera vectors into the raw visual world.  All time
    arrays use relative seconds in one common numerical origin.
    """

    camera_t_s: np.ndarray
    camera_centers_visual_m: np.ndarray
    visual_from_camera_rotation: np.ndarray
    frame_ids: np.ndarray
    fix_t_s: np.ndarray
    antenna_enu_m: np.ndarray
    antenna_cov_enu_m2: np.ndarray
    fix_good: np.ndarray
    baseline_t_s: np.ndarray
    baseline_enu_m: np.ndarray
    baseline_cov_enu_m2: np.ndarray
    baseline_good: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "camera_t_s": (self.camera_t_s, 1),
            "camera_centers_visual_m": (self.camera_centers_visual_m, 2),
            "visual_from_camera_rotation":
                (self.visual_from_camera_rotation, 3),
            "frame_ids": (self.frame_ids, 1),
            "fix_t_s": (self.fix_t_s, 1),
            "antenna_enu_m": (self.antenna_enu_m, 2),
            "antenna_cov_enu_m2": (self.antenna_cov_enu_m2, 3),
            "fix_good": (self.fix_good, 1),
            "baseline_t_s": (self.baseline_t_s, 1),
            "baseline_enu_m": (self.baseline_enu_m, 2),
            "baseline_cov_enu_m2": (self.baseline_cov_enu_m2, 3),
            "baseline_good": (self.baseline_good, 1),
        }
        for name, (value, ndim) in arrays.items():
            a = np.asarray(value)
            if a.ndim != ndim:
                raise ValueError(f"{name} must have {ndim} dimensions")
            if not np.isfinite(a.astype(float, copy=False)).all():
                raise ValueError(f"{name} contains NaN/Inf")
            object.__setattr__(self, name, a)
        n = len(self.camera_t_s)
        if self.camera_centers_visual_m.shape != (n, 3):
            raise ValueError("camera centres must have shape (N,3)")
        if self.visual_from_camera_rotation.shape != (n, 3, 3):
            raise ValueError("camera rotations must have shape (N,3,3)")
        if self.frame_ids.shape != (n,):
            raise ValueError("frame_ids must match camera samples")
        if self.antenna_enu_m.shape != (len(self.fix_t_s), 3):
            raise ValueError("antenna positions do not match fix timestamps")
        if self.antenna_cov_enu_m2.shape != (len(self.fix_t_s), 3, 3):
            raise ValueError("antenna covariances do not match fixes")
        if self.fix_good.shape != (len(self.fix_t_s),):
            raise ValueError("fix_good does not match fixes")
        if self.baseline_enu_m.shape != (len(self.baseline_t_s), 3):
            raise ValueError("baseline vectors do not match timestamps")
        if self.baseline_cov_enu_m2.shape != (
                len(self.baseline_t_s), 3, 3):
            raise ValueError("baseline covariances do not match timestamps")
        if self.baseline_good.shape != (len(self.baseline_t_s),):
            raise ValueError("baseline_good does not match baselines")
        for name, times in (
                ("camera_t_s", self.camera_t_s),
                ("fix_t_s", self.fix_t_s),
                ("baseline_t_s", self.baseline_t_s)):
            if len(times) < 2 or np.any(np.diff(times) <= 0):
                raise ValueError(f"{name} must be strictly increasing")
        rotations = self.visual_from_camera_rotation
        if not np.allclose(
                rotations @ np.swapaxes(rotations, 1, 2),
                np.eye(3), atol=2e-4):
            raise ValueError("visual camera rotations are not orthonormal")


@dataclass(frozen=True)
class SolverSettings:
    initial_clock_offset_s: float
    clock_offset_bounds_s: tuple[float, float]
    lever_bounds_m: np.ndarray
    baseline_angle_bounds_rad: np.ndarray
    prior_sigma: np.ndarray
    position_sigma_floor_m: float = 0.03
    baseline_sigma_floor_m: float = 0.01
    sample_interval_s: float = 1.0
    temporal_blocks: int = 10
    holdout_block_stride: int = 5
    holdout_block_offset: int = 2
    robust_loss: str = "soft_l1"
    robust_f_scale: float = 1.0
    max_nfev: int = 400
    clock_grid_steps: int = 21
    minimum_std_reduction: float = 0.20
    maximum_bound_fraction: float = 0.90
    maximum_start_std_prior_fraction: float = 0.35
    maximum_fold_std_prior_fraction: float = 0.50
    minimum_correction_to_fold_std_ratio: float = 2.0
    maximum_heldout_regression_fraction: float = 0.02
    maximum_heldout_position_rms_regression_fraction: float = 0.02
    maximum_heldout_position_p95_regression_fraction: float = 0.05
    minimum_heldout_improvement_fraction: float = 0.005
    minimum_heldout_baseline_improvement_fraction: float = 0.005

    def __post_init__(self) -> None:
        lo, hi = (float(v) for v in self.clock_offset_bounds_s)
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError("clock_offset_bounds_s must be finite and ordered")
        if not lo <= self.initial_clock_offset_s <= hi:
            raise ValueError("initial clock offset lies outside its bounds")
        object.__setattr__(self, "clock_offset_bounds_s", (lo, hi))
        lever = _as_vector(self.lever_bounds_m, 3, "lever_bounds_m")
        angles = _as_vector(
            self.baseline_angle_bounds_rad, 2,
            "baseline_angle_bounds_rad")
        prior = _as_vector(self.prior_sigma, 6, "prior_sigma")
        if np.any(lever <= 0) or np.any(angles <= 0) or np.any(prior <= 0):
            raise ValueError("bounds and prior sigmas must be positive")
        object.__setattr__(self, "lever_bounds_m", lever)
        object.__setattr__(self, "baseline_angle_bounds_rad", angles)
        object.__setattr__(self, "prior_sigma", prior)
        if self.position_sigma_floor_m <= 0 \
                or self.baseline_sigma_floor_m <= 0:
            raise ValueError("measurement sigma floors must be positive")
        if self.sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be positive")
        if self.temporal_blocks < 4:
            raise ValueError("at least four temporal blocks are required")
        if self.holdout_block_stride < 2:
            raise ValueError("holdout_block_stride must be at least two")
        if self.clock_grid_steps < 5:
            raise ValueError("clock_grid_steps must be at least five")
        fractions = (
            self.minimum_std_reduction,
            self.maximum_bound_fraction,
            self.maximum_start_std_prior_fraction,
            self.maximum_fold_std_prior_fraction,
            self.maximum_heldout_regression_fraction,
            self.maximum_heldout_position_rms_regression_fraction,
            self.maximum_heldout_position_p95_regression_fraction,
            self.minimum_heldout_improvement_fraction,
            self.minimum_heldout_baseline_improvement_fraction,
        )
        if any(value < 0 for value in fractions):
            raise ValueError("solver trust thresholds cannot be negative")
        if self.minimum_correction_to_fold_std_ratio < 0:
            raise ValueError(
                "minimum_correction_to_fold_std_ratio cannot be negative")

    @property
    def lower_calibration_bounds(self) -> np.ndarray:
        clock_lo = self.clock_offset_bounds_s[0] \
            - self.initial_clock_offset_s
        return np.concatenate([
            -self.lever_bounds_m,
            -self.baseline_angle_bounds_rad,
            [clock_lo],
        ])

    @property
    def upper_calibration_bounds(self) -> np.ndarray:
        clock_hi = self.clock_offset_bounds_s[1] \
            - self.initial_clock_offset_s
        return np.concatenate([
            self.lever_bounds_m,
            self.baseline_angle_bounds_rad,
            [clock_hi],
        ])


@dataclass
class SolveResult:
    global_rotation: np.ndarray
    global_translation_m: np.ndarray
    calibration: np.ndarray
    scale: float
    success: bool
    cost: float
    optimality: float
    nfev: int
    message: str


def _interpolate(times: np.ndarray, values: np.ndarray,
                 query: np.ndarray) -> np.ndarray:
    flat = values.reshape(len(values), -1)
    out = np.stack([
        np.interp(query, times, flat[:, column])
        for column in range(flat.shape[1])
    ], axis=1)
    return out.reshape((len(query),) + values.shape[1:])


def _support_is_good(times: np.ndarray, good: np.ndarray,
                     query_low: float, query_high: float) -> bool:
    lo = int(np.searchsorted(times, query_low, side="right")) - 1
    hi = int(np.searchsorted(times, query_high, side="left"))
    if lo < 0 or hi >= len(times):
        return False
    return bool(np.asarray(good[lo:hi + 1], dtype=bool).all())


def eligible_camera_indices(data: CalibrationData,
                            settings: SolverSettings) -> np.ndarray:
    """Quality-gated, temporally decimated camera indices.

    The complete configured clock interval must be supported by good position
    and baseline samples, keeping the least-squares residual dimension fixed
    while clock offset changes.
    """
    clock_lo, clock_hi = settings.clock_offset_bounds_s
    candidates: list[int] = []
    last_time = -np.inf
    for i, t in enumerate(data.camera_t_s):
        if t - last_time < settings.sample_interval_s:
            continue
        if not _support_is_good(
                data.fix_t_s, data.fix_good, t + clock_lo, t + clock_hi):
            continue
        if not _support_is_good(
                data.baseline_t_s, data.baseline_good,
                t + clock_lo, t + clock_hi):
            continue
        candidates.append(i)
        last_time = float(t)
    if len(candidates) < 30:
        raise RuntimeError(
            f"only {len(candidates)} quality-gated calibration samples; "
            "need at least 30")
    return np.asarray(candidates, dtype=int)


def temporal_split(camera_times: np.ndarray, indices: np.ndarray,
                   settings: SolverSettings) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fit indices, held-out indices, and per-sample block IDs."""
    times = camera_times[indices]
    edges = np.linspace(times[0] - 1e-9, times[-1] + 1e-9,
                        settings.temporal_blocks + 1)
    block = np.clip(np.searchsorted(edges, times, side="right") - 1,
                    0, settings.temporal_blocks - 1)
    held_blocks = {
        b for b in range(settings.temporal_blocks)
        if (b - settings.holdout_block_offset)
        % settings.holdout_block_stride == 0
    }
    held_mask = np.isin(block, sorted(held_blocks))
    fit, held = indices[~held_mask], indices[held_mask]
    if len(fit) < 20 or len(held) < 5:
        raise RuntimeError(
            f"invalid temporal split: {len(fit)} fit / {len(held)} held out")
    return fit, held, block


class CalibrationProblem:
    """Fixed visual trajectory with continuous-time RTK interpolation."""

    def __init__(self, data: CalibrationData, rig: RigPrior,
                 settings: SolverSettings,
                 initial_global_rotation: np.ndarray | None = None):
        self.data = data
        self.rig = rig
        self.settings = settings
        self.initial_lever = rig.lever_camera_m
        self.initial_baseline = rig.baseline_camera_m
        self.baseline_tangent_basis = _tangent_basis(
            self.initial_baseline)
        self.initial_global_rotation = (
            np.eye(3) if initial_global_rotation is None
            else _as_rotation(initial_global_rotation,
                              "initial_global_rotation"))

    def corrected_geometry(self, calibration: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray, float]:
        c = _as_vector(calibration, 6, "calibration")
        lever = self.initial_lever + c[:3]
        rotation_vector = self.baseline_tangent_basis @ c[3:5]
        baseline = Rotation.from_rotvec(
            rotation_vector).as_matrix() @ self.initial_baseline
        clock = self.settings.initial_clock_offset_s + c[5]
        return lever, baseline, float(clock)

    def corrected_body_from_camera(self, calibration: np.ndarray) -> np.ndarray:
        """Recover the prior-safe physical camera TF implied by calibration.

        The two baseline tangent parameters supply the only two camera
        rotation modes observable from a single antenna baseline.  Rotation
        about that baseline is therefore inherited exactly from the supplied
        body-from-camera prior.  The corrected direct camera-to-primary lever
        then determines the camera origin in the body frame.
        """
        c = _as_vector(calibration, 6, "calibration")
        lever, _, _ = self.corrected_geometry(c)
        camera_rotation = Rotation.from_rotvec(
            self.baseline_tangent_basis @ c[3:5]).as_matrix()
        body_from_camera_rotation = (
            self.rig.body_from_camera_rotation @ camera_rotation.T)
        camera_in_body = (
            self.rig.position_antenna_in_body_m
            - body_from_camera_rotation @ lever)
        transform = np.eye(4)
        transform[:3, :3] = body_from_camera_rotation
        transform[:3, 3] = camera_in_body
        return transform

    def tf_correction(self, calibration: np.ndarray) -> dict:
        """Describe the bounded TF correction relative to the rough prior."""
        prior = np.eye(4)
        prior[:3, :3] = self.rig.body_from_camera_rotation
        prior[:3, 3] = self.rig.camera_in_body_m
        corrected = self.corrected_body_from_camera(calibration)
        body_rotation_delta = (
            corrected[:3, :3] @ prior[:3, :3].T)
        return {
            "body_from_camera": corrected.tolist(),
            "translation_delta_body_m":
                (corrected[:3, 3] - prior[:3, 3]).tolist(),
            "rotation_delta_rotvec_body_deg": np.degrees(
                Rotation.from_matrix(body_rotation_delta).as_rotvec()).tolist(),
            "rotation_delta_angle_deg": float(np.degrees(
                Rotation.from_matrix(body_rotation_delta).magnitude())),
            "baseline_axis_twist_delta_deg": 0.0,
        }

    def measurements(self, indices: np.ndarray, clock: float) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        query = self.data.camera_t_s[indices] + clock
        antenna = _interpolate(
            self.data.fix_t_s, self.data.antenna_enu_m, query)
        antenna_cov = _interpolate(
            self.data.fix_t_s, self.data.antenna_cov_enu_m2, query)
        baseline = _interpolate(
            self.data.baseline_t_s, self.data.baseline_enu_m, query)
        baseline_cov = _interpolate(
            self.data.baseline_t_s,
            self.data.baseline_cov_enu_m2, query)
        return antenna, antenna_cov, baseline, baseline_cov

    def predictions(self, indices: np.ndarray, global_rotation: np.ndarray,
                    global_translation: np.ndarray,
                    calibration: np.ndarray, scale: float = 1.0) \
            -> tuple[np.ndarray, np.ndarray]:
        lever, baseline_camera, _ = self.corrected_geometry(calibration)
        r_vc = self.data.visual_from_camera_rotation[indices]
        centers = self.data.camera_centers_visual_m[indices]
        lever_visual = np.einsum("nij,j->ni", r_vc, lever)
        baseline_visual = np.einsum("nij,j->ni", r_vc, baseline_camera)
        antenna = (
            scale * np.einsum("ij,nj->ni", global_rotation, centers)
            + np.einsum("ij,nj->ni", global_rotation, lever_visual)
            + global_translation)
        baseline = np.einsum(
            "ij,nj->ni", global_rotation, baseline_visual)
        return antenna, baseline

    @staticmethod
    def _whiten(errors: np.ndarray, covariance: np.ndarray,
                sigma_floor: float) -> np.ndarray:
        floor_var = float(sigma_floor) ** 2
        covariance = np.asarray(covariance, dtype=float)
        sym = 0.5 * (
            covariance + np.swapaxes(covariance, 1, 2))
        values, vectors = np.linalg.eigh(sym)
        inv_sqrt_values = 1.0 / np.sqrt(np.maximum(values, floor_var))
        # V diag(lambda^-1/2) V^T e, evaluated as a batched operation.
        local_error = np.einsum("nji,nj->ni", vectors, errors)
        return np.einsum(
            "nij,nj->ni", vectors, inv_sqrt_values * local_error)

    def measurement_residual(
            self, indices: np.ndarray, global_rotation: np.ndarray,
            global_translation: np.ndarray, calibration: np.ndarray,
            scale: float = 1.0) -> np.ndarray:
        _, _, clock = self.corrected_geometry(calibration)
        antenna, antenna_cov, baseline, baseline_cov = self.measurements(
            indices, clock)
        pred_ant, pred_baseline = self.predictions(
            indices, global_rotation, global_translation,
            calibration, scale)
        ant_white = self._whiten(
            pred_ant - antenna, antenna_cov,
            self.settings.position_sigma_floor_m)
        baseline_white = self._whiten(
            pred_baseline - baseline, baseline_cov,
            self.settings.baseline_sigma_floor_m)
        return np.concatenate([ant_white.ravel(), baseline_white.ravel()])

    def raw_errors(
            self, indices: np.ndarray, result: SolveResult) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, _, clock = self.corrected_geometry(result.calibration)
        antenna, _, baseline, _ = self.measurements(indices, clock)
        pred_ant, pred_baseline = self.predictions(
            indices, result.global_rotation, result.global_translation_m,
            result.calibration, result.scale)
        position_error = pred_ant - antenna
        baseline_error = pred_baseline - baseline
        dot = np.sum(pred_baseline * baseline, axis=1)
        denom = (np.linalg.norm(pred_baseline, axis=1)
                 * np.linalg.norm(baseline, axis=1))
        angle = np.degrees(np.arccos(
            np.clip(dot / np.maximum(denom, 1e-12), -1.0, 1.0)))
        return position_error, baseline_error, angle


def _initial_translation(problem: CalibrationProblem, indices: np.ndarray,
                         global_rotation: np.ndarray,
                         calibration: np.ndarray, scale: float) -> np.ndarray:
    _, _, clock = problem.corrected_geometry(calibration)
    antenna, _, _, _ = problem.measurements(indices, clock)
    predicted, _ = problem.predictions(
        indices, global_rotation, np.zeros(3), calibration, scale)
    return np.median(antenna - predicted, axis=0)


def _solve(problem: CalibrationProblem, indices: np.ndarray,
           active_calibration: Iterable[int],
           initial_calibration: np.ndarray | None = None,
           fixed_calibration: np.ndarray | None = None,
           estimate_scale: bool = False,
           scale_bounds: tuple[float, float] = (0.95, 1.05),
           initial_global_rotation: np.ndarray | None = None) -> SolveResult:
    active = np.asarray(list(active_calibration), dtype=int)
    if np.any((active < 0) | (active >= 6)) or len(np.unique(active)) != len(active):
        raise ValueError("active calibration indices are invalid")
    fixed = (np.zeros(6) if fixed_calibration is None
             else _as_vector(fixed_calibration, 6, "fixed_calibration").copy())
    initial = fixed.copy()
    if initial_calibration is not None:
        supplied = _as_vector(
            initial_calibration, 6, "initial_calibration")
        initial[active] = supplied[active]
    r0 = (problem.initial_global_rotation
          if initial_global_rotation is None
          else _as_rotation(initial_global_rotation,
                            "initial_global_rotation"))
    scale0 = 1.0
    t0 = _initial_translation(problem, indices, r0, initial, scale0)
    z0 = np.concatenate([
        Rotation.from_matrix(r0).as_rotvec(),
        t0,
        initial[active],
        [0.0] if estimate_scale else [],
    ])
    lower = np.concatenate([
        np.full(6, -np.inf),
        problem.settings.lower_calibration_bounds[active],
        [np.log(scale_bounds[0])] if estimate_scale else [],
    ])
    upper = np.concatenate([
        np.full(6, np.inf),
        problem.settings.upper_calibration_bounds[active],
        [np.log(scale_bounds[1])] if estimate_scale else [],
    ])
    z0 = np.minimum(np.maximum(z0, lower + 1e-12), upper - 1e-12)

    def decode(z):
        rotation = Rotation.from_rotvec(z[:3]).as_matrix()
        translation = z[3:6]
        calibration = fixed.copy()
        calibration[active] = z[6:6 + len(active)]
        scale = float(np.exp(z[-1])) if estimate_scale else 1.0
        return rotation, translation, calibration, scale

    def residual(z):
        rotation, translation, calibration, scale = decode(z)
        measurement = problem.measurement_residual(
            indices, rotation, translation, calibration, scale)
        priors = calibration[active] \
            / problem.settings.prior_sigma[active]
        return np.concatenate([measurement, priors])

    solution = least_squares(
        residual, z0, bounds=(lower, upper),
        loss=problem.settings.robust_loss,
        f_scale=problem.settings.robust_f_scale,
        max_nfev=problem.settings.max_nfev,
        x_scale="jac")
    rotation, translation, calibration, scale = decode(solution.x)
    return SolveResult(
        global_rotation=rotation,
        global_translation_m=np.asarray(translation),
        calibration=calibration,
        scale=scale,
        success=bool(solution.success),
        cost=float(solution.cost),
        optimality=float(solution.optimality),
        nfev=int(solution.nfev),
        message=str(solution.message),
    )


def score(problem: CalibrationProblem, indices: np.ndarray,
          result: SolveResult) -> dict:
    position, baseline, angle = problem.raw_errors(indices, result)
    pos_norm = np.linalg.norm(position, axis=1)
    base_norm = np.linalg.norm(baseline, axis=1)
    return {
        "n": int(len(indices)),
        "position_median_m": float(np.median(pos_norm)),
        "position_rms_m": float(np.sqrt(np.mean(pos_norm ** 2))),
        "position_p95_m": float(np.percentile(pos_norm, 95)),
        "position_max_m": float(pos_norm.max()),
        "position_axis_rms_m":
            np.sqrt(np.mean(position ** 2, axis=0)).tolist(),
        "baseline_error_median_m": float(np.median(base_norm)),
        "baseline_error_p95_m": float(np.percentile(base_norm, 95)),
        "baseline_angle_median_deg": float(np.median(angle)),
        "baseline_angle_p95_deg": float(np.percentile(angle, 95)),
    }


def _finite_difference_jacobian(function, x: np.ndarray,
                                steps: np.ndarray) -> np.ndarray:
    base = np.asarray(function(x), dtype=float)
    jacobian = np.empty((len(base), len(x)), dtype=float)
    for column, step in enumerate(steps):
        plus, minus = x.copy(), x.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (
            np.asarray(function(plus)) - np.asarray(function(minus))
        ) / (2.0 * step)
    return jacobian


def observability(problem: CalibrationProblem, indices: np.ndarray,
                  result: SolveResult) -> dict:
    """Project global SE(3) nuisance directions out of the data Jacobian."""
    x = np.concatenate([
        Rotation.from_matrix(result.global_rotation).as_rotvec(),
        result.global_translation_m,
        result.calibration,
    ])

    def residual(value):
        rotation = Rotation.from_rotvec(value[:3]).as_matrix()
        return problem.measurement_residual(
            indices, rotation, value[3:6], value[6:12], scale=1.0)

    steps = np.array([
        1e-6, 1e-6, 1e-6,
        1e-5, 1e-5, 1e-5,
        1e-5, 1e-5, 1e-5,
        1e-6, 1e-6, 1e-4,
    ])
    jac = _finite_difference_jacobian(residual, x, steps)
    j_global, j_cal = jac[:, :6], jac[:, 6:]
    u_global, singular_global, _ = np.linalg.svd(
        j_global, full_matrices=False)
    global_tolerance = max(
        1e-8,
        1e-8 * (singular_global[0] if len(singular_global) else 1.0))
    global_rank = int(np.sum(singular_global > global_tolerance))
    q = u_global[:, :global_rank]
    projected = j_cal - q @ (q.T @ j_cal)
    normalized = projected @ np.diag(problem.settings.prior_sigma)
    _, singular, vt = np.linalg.svd(normalized, full_matrices=False)
    information = normalized.T @ normalized
    posterior_norm = np.linalg.pinv(
        information + np.eye(len(CALIBRATION_PARAMETER_NAMES)),
        rcond=1e-12)
    posterior_std = problem.settings.prior_sigma * np.sqrt(
        np.maximum(np.diag(posterior_norm), 0.0))
    reduction = np.clip(
        1.0 - posterior_std / problem.settings.prior_sigma, 0.0, 1.0)
    projected_tolerance = max(
        1e-6, 1e-8 * (singular[0] if len(singular) else 1.0))
    return {
        "global_alignment_singular_values": singular_global.tolist(),
        "singular_values_dimensionless": singular.tolist(),
        "right_singular_vectors": vt.tolist(),
        "prior_sigma": problem.settings.prior_sigma.tolist(),
        "posterior_sigma_linearized": posterior_std.tolist(),
        "std_reduction_fraction": reduction.tolist(),
        "data_information_matrix_dimensionless": information.tolist(),
        "global_alignment_jacobian_rank": global_rank,
        "projected_jacobian_rank":
            int(np.sum(singular > projected_tolerance)),
        "projected_rank_tolerance": float(projected_tolerance),
        "structurally_retained_prior": {
            "rotation_about_dual_antenna_baseline":
                "not parameterized: one baseline cannot observe this twist",
        },
    }


def _clock_profile(problem: CalibrationProblem, fit_indices: np.ndarray) -> dict:
    settings = problem.settings
    offsets = np.linspace(
        settings.clock_offset_bounds_s[0],
        settings.clock_offset_bounds_s[1],
        settings.clock_grid_steps)
    costs, medians = [], []
    solutions = []
    for offset in offsets:
        calibration = np.zeros(6)
        calibration[5] = offset - settings.initial_clock_offset_s
        result = _solve(
            problem, fit_indices, active_calibration=[],
            fixed_calibration=calibration)
        solutions.append(result)
        raw = problem.measurement_residual(
            fit_indices, result.global_rotation,
            result.global_translation_m, calibration)
        costs.append(float(np.mean(raw ** 2)))
        medians.append(score(problem, fit_indices, result)[
            "position_median_m"])
    best = int(np.argmin(costs))
    interior = 0 < best < len(offsets) - 1
    endpoint_reference = min(costs[0], costs[-1])
    relative_drop = (
        (endpoint_reference - costs[best])
        / max(endpoint_reference, 1e-12))
    cost_array = np.asarray(costs)
    sorted_costs = np.sort(cost_array)
    second_best_separation = (
        (sorted_costs[1] - sorted_costs[0])
        / max(sorted_costs[0], 1e-12))
    if interior:
        neighbor_reference = min(costs[best - 1], costs[best + 1])
        local_prominence = (
            (neighbor_reference - costs[best])
            / max(costs[best], 1e-12))
    else:
        local_prominence = 0.0
    return {
        "offset_s": offsets.tolist(),
        "mean_whitened_squared_residual": costs,
        "position_median_m": medians,
        "best_index": best,
        "best_offset_s": float(offsets[best]),
        "best_is_interior": bool(interior),
        "relative_cost_drop_from_best_endpoint": float(relative_drop),
        "relative_separation_from_second_grid_point":
            float(second_best_separation),
        "relative_local_grid_prominence": float(local_prominence),
        "grid_step_s": float(offsets[1] - offsets[0]),
        "_best_solution": solutions[best],
    }


def _calibration_starts(problem: CalibrationProblem,
                        clock_offset: float) -> list[np.ndarray]:
    base = np.zeros(6)
    base[5] = clock_offset - problem.settings.initial_clock_offset_s
    perturb = 0.25 * problem.settings.prior_sigma
    plus, minus = base.copy(), base.copy()
    plus[:5] += perturb[:5] * np.array([1, -1, 1, -1, 1])
    minus[:5] -= perturb[:5] * np.array([1, -1, 1, -1, 1])
    lo, hi = (problem.settings.lower_calibration_bounds,
              problem.settings.upper_calibration_bounds)
    return [np.clip(value, lo + 1e-9, hi - 1e-9)
            for value in (base, plus, minus)]


def _heldout_gate(before: dict, after: dict,
                  settings: SolverSettings) -> tuple[bool, dict]:
    position_median_improvement = (
        (before["position_median_m"] - after["position_median_m"])
        / max(before["position_median_m"], 1e-12))
    position_rms_regression = (
        (after["position_rms_m"] - before["position_rms_m"])
        / max(before["position_rms_m"], 1e-12))
    position_p95_regression = (
        (after["position_p95_m"] - before["position_p95_m"])
        / max(before["position_p95_m"], 1e-12))
    baseline_angle_regression = (
        (after["baseline_angle_median_deg"]
         - before["baseline_angle_median_deg"])
        / max(before["baseline_angle_median_deg"], 1e-6))
    acceptable = (
        position_median_improvement
        >= settings.minimum_heldout_improvement_fraction
        and position_rms_regression
        <= settings.maximum_heldout_position_rms_regression_fraction
        and position_p95_regression
        <= settings.maximum_heldout_position_p95_regression_fraction
        and baseline_angle_regression
        <= settings.maximum_heldout_regression_fraction)
    return bool(acceptable), {
        "position_median_improvement_fraction":
            float(position_median_improvement),
        "position_rms_regression_fraction":
            float(position_rms_regression),
        "position_p95_regression_fraction":
            float(position_p95_regression),
        "baseline_angle_regression_fraction":
            float(baseline_angle_regression),
        "required_position_improvement_fraction":
            settings.minimum_heldout_improvement_fraction,
        "allowed_position_rms_regression_fraction":
            settings.maximum_heldout_position_rms_regression_fraction,
        "allowed_position_p95_regression_fraction":
            settings.maximum_heldout_position_p95_regression_fraction,
        "allowed_baseline_regression_fraction":
            settings.maximum_heldout_regression_fraction,
    }


def _mode_heldout_gate(parameter_index: int, before: dict, after: dict,
                       settings: SolverSettings) -> tuple[bool, dict]:
    """Gate one independently fitted calibration mode on held-out data."""
    _, common = _heldout_gate(before, after, settings)
    baseline_improvement = -common["baseline_angle_regression_fraction"]
    if parameter_index in (3, 4):
        # A baseline-direction mode is directly supported by the vector
        # residual, but may not buy position error by itself.  It must improve
        # baseline direction and may not degrade any position summary by more
        # than the explicit tail ceiling.
        median_regression = -common[
            "position_median_improvement_fraction"]
        acceptable = (
            baseline_improvement
            >= settings.minimum_heldout_baseline_improvement_fraction
            and median_regression
            <= settings.maximum_heldout_position_p95_regression_fraction
            and common["position_rms_regression_fraction"]
            <= settings.maximum_heldout_position_p95_regression_fraction
            and common["position_p95_regression_fraction"]
            <= settings.maximum_heldout_position_p95_regression_fraction)
    else:
        acceptable = (
            common["position_median_improvement_fraction"]
            >= settings.minimum_heldout_improvement_fraction
            and common["position_rms_regression_fraction"]
            <= settings.maximum_heldout_position_rms_regression_fraction
            and common["position_p95_regression_fraction"]
            <= settings.maximum_heldout_position_p95_regression_fraction
            and common["baseline_angle_regression_fraction"]
            <= settings.maximum_heldout_regression_fraction)
    common["baseline_angle_improvement_fraction"] = float(
        baseline_improvement)
    common["required_baseline_angle_improvement_fraction"] = (
        settings.minimum_heldout_baseline_improvement_fraction)
    common["parameter_gate_kind"] = (
        "baseline_direction" if parameter_index in (3, 4)
        else "position_or_clock")
    return bool(acceptable), common


def _four_contiguous_refit_subsets(indices: np.ndarray) -> list[np.ndarray]:
    fold_label = np.empty(len(indices), dtype=int)
    for fold, positions in enumerate(
            np.array_split(np.arange(len(indices)), 4)):
        fold_label[positions] = fold
    return [indices[fold_label != fold] for fold in range(4)]


def calibrate(problem: CalibrationProblem) -> dict:
    """Run the fixed-scale audit, bounded calibration, and trust decision."""
    eligible = eligible_camera_indices(problem.data, problem.settings)
    fit, heldout, block_for_eligible = temporal_split(
        problem.data.camera_t_s, eligible, problem.settings)

    baseline = _solve(
        problem, fit, active_calibration=[],
        fixed_calibration=np.zeros(6))
    baseline_fit = score(problem, fit, baseline)
    baseline_held = score(problem, heldout, baseline)

    scale_diag = _solve(
        problem, fit, active_calibration=[],
        fixed_calibration=np.zeros(6), estimate_scale=True)

    profile = _clock_profile(problem, fit)
    profile_solution = profile.pop("_best_solution")
    starts = _calibration_starts(problem, profile["best_offset_s"])
    start_results = [
        _solve(
            problem, fit, active_calibration=range(6),
            initial_calibration=start,
            initial_global_rotation=profile_solution.global_rotation)
        for start in starts
    ]
    successful = [result for result in start_results if result.success]
    if not successful:
        raise RuntimeError("all bounded calibration starts failed")
    candidate = min(successful, key=lambda result: result.cost)
    joint_start_values = np.stack([
        result.calibration for result in successful])

    obs = observability(problem, fit, candidate)
    reduction = np.asarray(obs["std_reduction_fraction"])

    # Joint temporal refits remain a diagnostic.  Parameter acceptance below
    # uses independent mode refits so one bound-hitting/correlated mode cannot
    # make another correction appear stable.
    refit_subsets = _four_contiguous_refit_subsets(fit)
    joint_fold_results = []
    for subset in refit_subsets:
        if len(subset) < 20:
            continue
        fold_result = _solve(
            problem, subset, active_calibration=range(6),
            initial_calibration=candidate.calibration,
            initial_global_rotation=candidate.global_rotation)
        if fold_result.success:
            joint_fold_results.append(fold_result)
    joint_fold_values = (
        np.stack([
            result.calibration for result in joint_fold_results])
        if joint_fold_results else np.empty((0, 6)))
    joint_start_std = np.std(joint_start_values, axis=0)
    joint_fold_std = (
        np.std(joint_fold_values, axis=0)
        if len(joint_fold_values) else np.full(6, np.inf))

    candidate_fit = score(problem, fit, candidate)
    candidate_held = score(problem, heldout, candidate)
    heldout_ok, heldout_gate = _heldout_gate(
        baseline_held, candidate_held, problem.settings)

    clock_unique = bool(
        profile["best_is_interior"]
        and profile["relative_cost_drop_from_best_endpoint"] >= 0.005
        and profile["relative_local_grid_prominence"] >= 1e-4)

    lower = problem.settings.lower_calibration_bounds
    upper = problem.settings.upper_calibration_bounds
    span_to_bound = np.maximum(np.abs(lower), np.abs(upper))
    joint_bound_fraction = (
        np.abs(candidate.calibration) / span_to_bound)
    joint_start_fraction = (
        joint_start_std / problem.settings.prior_sigma)
    joint_fold_fraction = (
        joint_fold_std / problem.settings.prior_sigma)
    joint_signal_ratio = (
        np.abs(candidate.calibration)
        / np.maximum(joint_fold_std, 1e-12))
    acceptance_reasons: list[list[str]] = [[] for _ in range(6)]
    for parameter_index in range(6):
        if reduction[parameter_index] \
                < problem.settings.minimum_std_reduction:
            acceptance_reasons[parameter_index].append(
                "insufficient data information after removing global SE3")
        if joint_bound_fraction[parameter_index] \
                > problem.settings.maximum_bound_fraction:
            acceptance_reasons[parameter_index].append(
                "joint candidate is too close to a configured bound")
        if joint_start_fraction[parameter_index] \
                > problem.settings.maximum_start_std_prior_fraction:
            acceptance_reasons[parameter_index].append(
                "joint multiple starts do not converge consistently")
        if joint_fold_fraction[parameter_index] \
                > problem.settings.maximum_fold_std_prior_fraction:
            acceptance_reasons[parameter_index].append(
                "joint temporal refits are unstable")
        if joint_signal_ratio[parameter_index] \
                < problem.settings.minimum_correction_to_fold_std_ratio:
            acceptance_reasons[parameter_index].append(
                "joint correction is not distinguishable from temporal "
                "refit spread")
        if parameter_index == 5 and not clock_unique:
            acceptance_reasons[parameter_index].append(
                "clock profile is not unique and interior")

    independent = []
    for parameter_index in range(6):
        seed = np.zeros(6)
        seed[parameter_index] = candidate.calibration[parameter_index]
        perturbation = 0.25 * problem.settings.prior_sigma[parameter_index]
        mode_starts = []
        for delta in (0.0, perturbation, -perturbation):
            initial = seed.copy()
            initial[parameter_index] = np.clip(
                seed[parameter_index] + delta,
                lower[parameter_index] + 1e-9,
                upper[parameter_index] - 1e-9)
            result = _solve(
                problem, fit, active_calibration=[parameter_index],
                initial_calibration=initial,
                initial_global_rotation=candidate.global_rotation)
            if result.success:
                mode_starts.append(result)
        if not mode_starts:
            raise RuntimeError(
                f"all starts failed for calibration parameter "
                f"{CALIBRATION_PARAMETER_NAMES[parameter_index]}")
        mode_result = min(mode_starts, key=lambda result: result.cost)
        mode_start_values = np.asarray([
            result.calibration[parameter_index]
            for result in mode_starts])
        mode_start_std = float(np.std(mode_start_values))

        mode_folds = []
        for subset in refit_subsets:
            if len(subset) < 20:
                continue
            fold_result = _solve(
                problem, subset,
                active_calibration=[parameter_index],
                initial_calibration=mode_result.calibration,
                initial_global_rotation=mode_result.global_rotation)
            if fold_result.success:
                mode_folds.append(fold_result)
        mode_fold_values = np.asarray([
            result.calibration[parameter_index]
            for result in mode_folds])
        mode_fold_std = (
            float(np.std(mode_fold_values))
            if len(mode_fold_values) else float("inf"))
        correction = float(mode_result.calibration[parameter_index])
        mode_bound_fraction = (
            abs(correction) / span_to_bound[parameter_index])
        mode_start_fraction = (
            mode_start_std
            / problem.settings.prior_sigma[parameter_index])
        mode_fold_fraction = (
            mode_fold_std
            / problem.settings.prior_sigma[parameter_index])
        correction_to_fold_std = (
            abs(correction) / max(mode_fold_std, 1e-12))
        mode_fit = score(problem, fit, mode_result)
        mode_held = score(problem, heldout, mode_result)
        mode_heldout_ok, mode_gate = _mode_heldout_gate(
            parameter_index, baseline_held, mode_held, problem.settings)

        independent.append({
            "result": mode_result,
            "fit": mode_fit,
            "heldout": mode_held,
            "heldout_gate": mode_gate,
            "heldout_ok": mode_heldout_ok,
            "multi_start_values": mode_start_values,
            "multi_start_std": mode_start_std,
            "temporal_refit_values": mode_fold_values,
            "temporal_refit_std": mode_fold_std,
            "bound_fraction": float(mode_bound_fraction),
            "start_fraction": float(mode_start_fraction),
            "fold_fraction": float(mode_fold_fraction),
            "correction_to_fold_std_ratio":
                float(correction_to_fold_std),
            "diagnostic_flags": [
                *([] if mode_bound_fraction
                   <= problem.settings.maximum_bound_fraction else [
                       "independent candidate is near a bound"]),
                *([] if mode_heldout_ok else [
                    "independent one-mode held-out gate failed"]),
            ],
        })

    # Remove modes that fail the joint candidate gates, refit the surviving
    # subset, and repeat temporal stability tests in that reduced model.  This
    # lets a bound-hitting nuisance mode be dropped without preserving the
    # biased values it induced in correlated parameters.
    trusted = np.asarray(
        [not reasons for reasons in acceptance_reasons], dtype=bool)
    acceptance_refit_std = joint_fold_std.copy()
    acceptance_signal_ratio = joint_signal_ratio.copy()
    subset_refit_history = []
    initial_subset_calibration = candidate.calibration.copy()
    while True:
        if np.any(trusted):
            initial = np.zeros(6)
            initial[trusted] = initial_subset_calibration[trusted]
            final_result = _solve(
                problem, fit,
                active_calibration=np.flatnonzero(trusted),
                initial_calibration=initial,
                fixed_calibration=np.zeros(6),
                initial_global_rotation=candidate.global_rotation)
        else:
            final_result = baseline
        subset_fold_results = []
        for subset in refit_subsets:
            if not np.any(trusted) or len(subset) < 20:
                continue
            fold_result = _solve(
                problem, subset,
                active_calibration=np.flatnonzero(trusted),
                initial_calibration=final_result.calibration,
                initial_global_rotation=final_result.global_rotation)
            if fold_result.success:
                subset_fold_results.append(fold_result)
        subset_values = (
            np.stack([
                result.calibration for result in subset_fold_results])
            if subset_fold_results else np.empty((0, 6)))
        subset_std = (
            np.std(subset_values, axis=0)
            if len(subset_values) else np.full(6, np.inf))
        subset_fraction = (
            subset_std / problem.settings.prior_sigma)
        subset_signal = (
            np.abs(final_result.calibration)
            / np.maximum(subset_std, 1e-12))
        active_now = np.flatnonzero(trusted)
        acceptance_refit_std[active_now] = subset_std[active_now]
        acceptance_signal_ratio[active_now] = subset_signal[active_now]
        rejected = []
        for parameter_index in active_now:
            if subset_fraction[parameter_index] \
                    > problem.settings.maximum_fold_std_prior_fraction:
                acceptance_reasons[parameter_index].append(
                    "reduced-model temporal refits are unstable")
                rejected.append(parameter_index)
            elif subset_signal[parameter_index] \
                    < problem.settings.minimum_correction_to_fold_std_ratio:
                acceptance_reasons[parameter_index].append(
                    "reduced-model correction is not distinguishable from "
                    "temporal refit spread")
                rejected.append(parameter_index)
        subset_refit_history.append({
            "active_parameters": [
                CALIBRATION_PARAMETER_NAMES[index]
                for index in active_now],
            "correction": final_result.calibration.tolist(),
            "temporal_refit_corrections":
                subset_values.tolist(),
            "temporal_refit_std": subset_std.tolist(),
            "correction_to_temporal_refit_std_ratio":
                subset_signal.tolist(),
            "rejected_after_refit": [
                CALIBRATION_PARAMETER_NAMES[index]
                for index in rejected],
        })
        if not rejected:
            break
        for parameter_index in rejected:
            trusted[parameter_index] = False
        initial_subset_calibration = final_result.calibration.copy()

    final_fit = score(problem, fit, final_result)
    final_held = score(problem, heldout, final_result)
    final_heldout_ok, final_gate = _heldout_gate(
        baseline_held, final_held, problem.settings)

    block_lookup = {
        int(index): int(block)
        for index, block in zip(eligible, block_for_eligible)}
    heldout_blocks = []
    for block in sorted({block_lookup[int(index)] for index in heldout}):
        block_indices = np.asarray([
            index for index in heldout
            if block_lookup[int(index)] == block], dtype=int)
        before = score(problem, block_indices, baseline)
        after = score(problem, block_indices, final_result)
        block_ok, block_gate = _heldout_gate(
            before, after, problem.settings)
        heldout_blocks.append({
            "block_id": int(block),
            "frame_ids": problem.data.frame_ids[
                block_indices].astype(int).tolist(),
            "baseline": before,
            "retained": after,
            "gate": block_gate,
            "passed": bool(block_ok),
        })
    all_blocks_ok = bool(
        heldout_blocks and all(block["passed"] for block in heldout_blocks))
    retained_set_ok = bool(
        np.any(trusted) and final_heldout_ok and all_blocks_ok)
    if not retained_set_ok:
        # Fail closed: a subset correction that does not generalize in the
        # aggregate and in every held-out temporal block cannot survive.
        for parameter_index in np.flatnonzero(trusted):
            acceptance_reasons[parameter_index].append(
                "joint retained set fails aggregate or per-block held-out gate")
        trusted[:] = False
        final_result = baseline
        final_fit = baseline_fit
        final_held = baseline_held
        final_gate = {
            **final_gate,
            "failed_closed_to_physical_prior": True,
        }
        for block in heldout_blocks:
            block["attempted_retained"] = block["retained"]
            block["retained"] = block["baseline"]

    parameter_report = {}
    for i, name in enumerate(CALIBRATION_PARAMETER_NAMES):
        mode = independent[i]
        reasons = list(acceptance_reasons[i])
        parameter_report[name] = {
            "candidate_correction": float(candidate.calibration[i]),
            "independent_correction":
                float(mode["result"].calibration[i]),
            "retained_correction": float(final_result.calibration[i]),
            "trusted": bool(trusted[i] and retained_set_ok),
            "reasons": reasons,
            "linearized_std_reduction_fraction": float(reduction[i]),
            "bound_fraction": float(joint_bound_fraction[i]),
            "multi_start_std": float(joint_start_std[i]),
            "temporal_refit_std":
                float(acceptance_refit_std[i]),
            "correction_to_temporal_refit_std_ratio":
                float(acceptance_signal_ratio[i]),
            "independent_fit": mode["fit"],
            "independent_heldout": mode["heldout"],
            "independent_heldout_gate": mode["heldout_gate"],
            "independent_diagnostic_flags":
                mode["diagnostic_flags"],
            "candidate_to_retained_shift":
                float(abs(
                    candidate.calibration[i]
                    - final_result.calibration[i])),
        }

    lever, baseline_camera, clock = problem.corrected_geometry(
        final_result.calibration)
    candidate_tf = problem.tf_correction(candidate.calibration)
    retained_tf = problem.tf_correction(final_result.calibration)
    return {
        "eligible_frame_ids":
            problem.data.frame_ids[eligible].astype(int).tolist(),
        "fit_frame_ids": problem.data.frame_ids[fit].astype(int).tolist(),
        "heldout_frame_ids":
            problem.data.frame_ids[heldout].astype(int).tolist(),
        "eligible_temporal_block_ids": block_for_eligible.astype(int).tolist(),
        "baseline_fixed_scale": {
            "fit": baseline_fit,
            "heldout": baseline_held,
            "global_rotation_visual_to_enu":
                baseline.global_rotation.tolist(),
            "global_translation_enu_m":
                baseline.global_translation_m.tolist(),
        },
        "sim3_diagnostic_fixed_physical_prior": {
            "scale_visual_to_enu": float(scale_diag.scale),
            "fit": score(problem, fit, scale_diag),
            "warning":
                "diagnostic only; scale is never combined with calibration",
        },
        "clock_profile": profile,
        "candidate": {
            "fit": candidate_fit,
            "heldout": candidate_held,
            "correction": candidate.calibration.tolist(),
            "success": candidate.success,
            "cost": candidate.cost,
            "nfev": candidate.nfev,
            "tf_correction": candidate_tf,
        },
        "observability": obs,
        "heldout_candidate_gate": heldout_gate,
        "parameters": parameter_report,
        "final_retained_prior_safe": {
            "fit": final_fit,
            "heldout": final_held,
            "heldout_gate": final_gate,
            "heldout_blocks": heldout_blocks,
            "calibration_accepted": bool(retained_set_ok),
            "correction": final_result.calibration.tolist(),
            "camera_to_position_antenna_m": lever.tolist(),
            "primary_to_secondary_baseline_camera_m":
                baseline_camera.tolist(),
            "clock_offset_s": float(clock),
            "tf_correction": retained_tf,
            "global_rotation_visual_to_enu":
                final_result.global_rotation.tolist(),
            "global_translation_enu_m":
                final_result.global_translation_m.tolist(),
            "structurally_unobservable_retained_at_prior": [
                "rotation about the dual-antenna baseline",
            ],
        },
        "stability": {
            "successful_multi_starts": int(len(successful)),
            "multi_start_corrections":
                [result.calibration.tolist() for result in successful],
            "successful_temporal_refits":
                int(len(joint_fold_results)),
            "temporal_refit_corrections":
                [result.calibration.tolist()
                 for result in joint_fold_results],
            "independent_parameter_refits": {
                name: {
                    "multi_start_values":
                        independent[index][
                            "multi_start_values"].tolist(),
                    "temporal_refit_values":
                        independent[index][
                            "temporal_refit_values"].tolist(),
                }
                for index, name in enumerate(
                    CALIBRATION_PARAMETER_NAMES)
            },
            "reduced_model_refit_history": subset_refit_history,
        },
    }
