"""RTK track -> per-frame camera poses in a local ENU world frame.

Conventions (replicated from the validated crop-row pipeline, 2-4 cm RMS):
- World: WGS84 geodetic -> local ENU from an origin fix (pymap3d). x=east,
  y=north, z=up (ellipsoidal height relative to origin).
- Base yaw: ENU yaw of the moving_base->rover antenna vector, atan2(n, e),
  unwrapped, box-smoothed, linearly interpolated to frame stamps. Frames whose
  bracketing heading samples are not RTK-fixed are rejected.
- base_link (rear axle, CAR mode): antenna position pulled back by
  antenna_forward_m along the yaw direction.
- Camera chain: base -> mount (forward/left/up + yaw_offset + pitch_down)
  -> ZED center (up) -> left eye (forward/left in the mount frame).
- Camera model: OpenCV/COLMAP optical frame (x right, y down, z forward).
  Output per frame is the world-to-camera matrix `viewmat` expected by gsplat.

The only rotations involved are yaw (about world z) and mount pitch (about the
camera's lateral axis); roll is assumed zero for E1 (documented decision).
"""

from dataclasses import dataclass

import numpy as np

class LocalEnu:
    """WGS84 geodetic -> local ENU tangent frame (proper ellipsoidal math via
    pymap3d; the earlier equirectangular approximation differed by <1 mm over
    a tile but is not defensible in print). z is ellipsoidal-height relative;
    the CRS description is stored in segment metadata."""

    def __init__(self, lat0: float, lon0: float, alt0: float):
        self.lat0, self.lon0, self.alt0 = float(lat0), float(lon0), float(alt0)

    def to_enu(self, lat, lon, alt):
        import pymap3d
        e, n, u = pymap3d.geodetic2enu(np.asarray(lat), np.asarray(lon),
                                       np.asarray(alt),
                                       self.lat0, self.lon0, self.alt0)
        return np.stack([e, n, u], axis=-1)

    def crs(self) -> dict:
        return {"type": "local_ENU", "ellipsoid": "WGS84",
                "origin_lat": self.lat0, "origin_lon": self.lon0,
                "origin_alt_ellipsoidal": self.alt0,
                "vertical_datum": "WGS84 ellipsoid (not orthometric)"}


@dataclass
class PosedFrame:
    t: float
    viewmat: np.ndarray      # (4,4) world-to-camera, OpenCV convention
    cam_center: np.ndarray   # (3,) camera position in ENU world


def smooth_yaw(track, window: int) -> np.ndarray:
    """Box smoothing of the unwrapped yaw, same window as the crop-row node."""
    yaw = track.relpos_yaw
    if window <= 1 or len(yaw) < 2:
        return yaw
    kernel = np.ones(window) / window
    # 'same' convolution with edge padding so ends are not biased toward zero
    padded = np.pad(yaw, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def antenna_pose_at(track, yaw_smoothed: np.ndarray, t: float, pose_cfg):
    """Interpolated primary-antenna centre and yaw, or ``None``.

    Position/heading quality gates are applied before interpolation. The
    returned position is the measured phase centre; no robot-specific lever
    arm is hidden in this function.
    """
    ft, rt = track.fix_t, track.relpos_t
    if not (ft[0] <= t <= ft[-1] and rt[0] <= t <= rt[-1]):
        return None
    j = int(np.searchsorted(rt, t))
    j0, j1 = max(0, j - 1), min(len(rt) - 1, j)
    # The smoothed yaw at j0/j1 averages window//2 raw neighbors on each side,
    # so its source-specific observability gate must cover the whole support.
    half = pose_cfg.yaw_smooth_window // 2
    lo = max(0, j0 - half)
    hi = min(len(rt) - 1, j1 + half)
    heading_valid = getattr(track, "heading_valid", None)
    if heading_valid is not None and not np.asarray(
        heading_valid, dtype=bool
    )[lo:hi + 1].all():
        return None
    heading_kind = getattr(track, "heading_quality_kind", "carrier")
    if heading_kind == "carrier" and (
        track.relpos_carr[lo:hi + 1].min() < pose_cfg.min_carr_soln
    ):
        return None

    k = int(np.searchsorted(ft, t))
    k0, k1 = max(0, k - 1), min(len(ft) - 1, k)
    position_valid = getattr(track, "fix_position_valid", None)
    if position_valid is not None and not np.asarray(
        position_valid, dtype=bool
    )[[k0, k1]].all():
        return None
    minimum_fix_status = int(getattr(pose_cfg, "minimum_navsat_status", -1))
    if np.min(np.asarray(track.fix_status)[[k0, k1]]) < minimum_fix_status:
        return None
    minimum_position_carrier = int(
        getattr(pose_cfg, "minimum_position_carrier_status", -1)
    )
    position_carrier = np.asarray(
        getattr(track, "fix_carrier_status", np.full(len(ft), -1))
    )
    if np.min(position_carrier[[k0, k1]]) < minimum_position_carrier:
        return None
    maximum_covariance = float(
        getattr(pose_cfg, "maximum_position_covariance_m2", np.inf)
    )
    covariance = np.asarray(
        getattr(track, "fix_cov_max", np.full(len(ft), np.nan)),
        dtype=float,
    )[[k0, k1]]
    if np.isfinite(maximum_covariance) and (
        not np.isfinite(covariance).all()
        or np.max(covariance) > maximum_covariance
    ):
        return None
    yaw = float(np.interp(t, rt, yaw_smoothed))
    ant_x = float(np.interp(t, ft, track.enu_xyz[:, 0]))
    ant_y = float(np.interp(t, ft, track.enu_xyz[:, 1]))
    ant_z = float(np.interp(t, ft, track.enu_xyz[:, 2]))
    return np.array([ant_x, ant_y, ant_z]), yaw


def base_pose_at(track, yaw_smoothed: np.ndarray, t: float, pose_cfg):
    """Legacy scalar-chain base pose retained for historical diagnostics."""
    result = antenna_pose_at(track, yaw_smoothed, t, pose_cfg)
    if result is None:
        return None
    antenna, yaw = result
    base = antenna - np.array(
        [
            pose_cfg.antenna_forward_m * np.cos(yaw),
            pose_cfg.antenna_forward_m * np.sin(yaw),
            0.0,
        ]
    )
    return base, yaw


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# ROS body frame (x fwd, y left, z up) -> OpenCV optical (x right, y down,
# z fwd). Columns are the optical axes in body coordinates:
#   x_opt(right) = -y_body, y_opt(down) = -z_body, z_opt(fwd) = +x_body.
_BODY_TO_OPTICAL = np.array([[0.0, 0.0, 1.0],
                             [-1.0, 0.0, 0.0],
                             [0.0, -1.0, 0.0]])


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def camera_pose(base_xyz: np.ndarray, yaw: float, pose_cfg,
                tilt: tuple[float, float] = (0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """(R_world_cam, cam_center_world) for the LEFT eye.

    Chain: world <- Rz(yaw)Ry(pitch_dev)Rx(roll_dev) <- base;
    base <- mount(t_mount, Rz(yaw_off)Ry(pitch)); mount <- center/left-eye
    translations; body axes -> optical axes. `tilt` = (roll_dev, pitch_dev)
    dynamic deviations of the body from horizontal (rad), zero by default.
    """
    p = pose_cfg
    roll_dev, pitch_dev = tilt
    # base -> world rotation: robot body yaw about world z (ENU yaw measures
    # the body x axis from east, which is exactly what atan2(n,e) yields),
    # then the dynamic terrain tilt.
    r_wb = _rot_z(yaw) @ _rot_y(pitch_dev) @ _rot_x(roll_dev)
    # mount rotation within the body frame: calibrated yaw offset, then pitch
    # down about the mount's y (left) axis. Ry(+pitch) tilts body-x downward.
    r_bm = _rot_z(np.radians(p.yaw_offset_deg)) @ _rot_y(np.radians(p.pitch_down_deg))
    # translations: mount origin in base frame, then ZED internals in the
    # (rotated) mount frame.
    t_bm = np.array([p.cam_forward_m, p.cam_left_m, p.cam_up_m])
    t_internal = np.array([p.left_eye_forward_m, p.left_eye_left_m, p.center_up_m])
    cam_center = base_xyz + r_wb @ (t_bm + r_bm @ t_internal)
    r_wc = r_wb @ r_bm @ _BODY_TO_OPTICAL
    return r_wc, cam_center


def viewmat_from(r_wc: np.ndarray, cam_center: np.ndarray) -> np.ndarray:
    """World-to-camera 4x4 from camera rotation and center."""
    m = np.eye(4)
    m[:3, :3] = r_wc.T
    m[:3, 3] = -r_wc.T @ cam_center
    return m


def pose_frames(track, frame_stamps, pose_cfg,
                tilts=None) -> list[PosedFrame | None]:
    """Camera pose per frame stamp; None entries mark unposeable frames.

    `tilts`: optional (n_frames, 2) array of (roll_dev, pitch_dev) rad.
    """
    yaw_s = smooth_yaw(track, pose_cfg.yaw_smooth_window)
    out = []
    for k, t in enumerate(frame_stamps):
        bp = base_pose_at(track, yaw_s, t, pose_cfg)
        if bp is None:
            out.append(None)
            continue
        base, yaw = bp
        tilt = tuple(tilts[k]) if tilts is not None else (0.0, 0.0)
        r_wc, center = camera_pose(base, yaw, pose_cfg, tilt)
        out.append(PosedFrame(t=t, viewmat=viewmat_from(r_wc, center),
                              cam_center=center))
    return out


def pose_frames_from_extrinsic(
    track,
    frame_stamps,
    pose_cfg,
    T_camera_primary_antenna,
    tilts=None,
    *,
    primary_to_secondary_baseline_camera_m=None,
) -> list[PosedFrame | None]:
    """Construct camera poses from the one declared antenna-camera transform.

    ``T_camera_primary_antenna`` maps coordinates in the primary-antenna
    frame into the left optical-camera frame. The antenna frame is assumed
    body-aligned (x forward, y left, z up).  When the calibrated
    primary-to-secondary vector is supplied, the measured dual-antenna yaw is
    converted from *baseline yaw* to antenna-frame yaw before the fixed
    extrinsic is applied.  Omitting the vector preserves the historical
    assumption that the measured baseline is exactly antenna-frame +X.

    A single antenna baseline does not observe twist about itself, so this
    function deliberately continues to use only its horizontal direction;
    optional terrain tilt remains an explicit, separate input.
    """
    camera_from_antenna = np.asarray(
        T_camera_primary_antenna, dtype=np.float64
    )
    if (
        camera_from_antenna.shape != (4, 4)
        or not np.isfinite(camera_from_antenna).all()
        or not np.allclose(camera_from_antenna[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(
            camera_from_antenna[:3, :3].T
            @ camera_from_antenna[:3, :3],
            np.eye(3),
            atol=1e-5,
        )
        or not np.isclose(
            np.linalg.det(camera_from_antenna[:3, :3]), 1.0, atol=1e-5
        )
    ):
        raise ValueError(
            "T_camera_primary_antenna must be a finite rigid transform"
        )
    antenna_from_camera = np.linalg.inv(camera_from_antenna)
    r_ac = antenna_from_camera[:3, :3]
    t_ac = antenna_from_camera[:3, 3]
    baseline_yaw_in_antenna = 0.0
    if primary_to_secondary_baseline_camera_m is not None:
        baseline_camera = np.asarray(
            primary_to_secondary_baseline_camera_m, dtype=np.float64
        )
        if (
            baseline_camera.shape != (3,)
            or not np.isfinite(baseline_camera).all()
        ):
            raise ValueError(
                "primary_to_secondary_baseline_camera_m must be a finite "
                "three-vector"
            )
        baseline_antenna = camera_from_antenna[:3, :3].T @ baseline_camera
        horizontal_norm = float(np.linalg.norm(baseline_antenna[:2]))
        if not np.isfinite(horizontal_norm) or horizontal_norm <= 1.0e-8:
            raise ValueError(
                "declared antenna baseline has no observable horizontal "
                "direction"
            )
        baseline_yaw_in_antenna = float(
            np.arctan2(baseline_antenna[1], baseline_antenna[0])
        )
    yaw_smoothed = smooth_yaw(track, pose_cfg.yaw_smooth_window)
    output = []
    for index, stamp in enumerate(frame_stamps):
        measured = antenna_pose_at(track, yaw_smoothed, stamp, pose_cfg)
        if measured is None:
            output.append(None)
            continue
        antenna_center, measured_baseline_yaw = measured
        yaw = measured_baseline_yaw - baseline_yaw_in_antenna
        roll_dev, pitch_dev = (
            tuple(tilts[index]) if tilts is not None else (0.0, 0.0)
        )
        r_wa = (
            _rot_z(yaw)
            @ _rot_y(pitch_dev)
            @ _rot_x(roll_dev)
        )
        r_wc = r_wa @ r_ac
        camera_center = antenna_center + r_wa @ t_ac
        output.append(
            PosedFrame(
                t=stamp,
                viewmat=viewmat_from(r_wc, camera_center),
                cam_center=camera_center,
            )
        )
    return output


def tilt_deviations(imu_t: np.ndarray, imu_quat_xyzw: np.ndarray,
                    frame_stamps, lp_window_s: float):
    """Per-frame (roll_dev, pitch_dev) of the body from the fused IMU.

    Robust by construction: euler roll/pitch are extracted per sample
    (ZYX order, so yaw -- which the IMU cannot know absolutely -- is
    discarded), low-passed, and MEDIAN-SUBTRACTED, so any static mount
    angle, calibration residual or wrong-frame constant drops out; only
    time-varying terrain tilt remains. Returns (n,2) array and a stats dict.
    """
    from scipy.spatial.transform import Rotation

    eul = Rotation.from_quat(imu_quat_xyzw).as_euler("ZYX")  # yaw, pitch, roll
    pitch, roll = np.unwrap(eul[:, 1]), np.unwrap(eul[:, 2])
    dt = np.median(np.diff(imu_t))
    n_lp = max(1, int(round(lp_window_s / dt)))
    kernel = np.ones(n_lp) / n_lp
    pad = (n_lp // 2, n_lp - 1 - n_lp // 2)
    pitch_lp = np.convolve(np.pad(pitch, pad, mode="edge"), kernel, "valid")
    roll_lp = np.convolve(np.pad(roll, pad, mode="edge"), kernel, "valid")
    pitch_dev = pitch_lp - np.median(pitch_lp)
    roll_dev = roll_lp - np.median(roll_lp)
    out = np.stack([np.interp(frame_stamps, imu_t, roll_dev),
                    np.interp(frame_stamps, imu_t, pitch_dev)], axis=1)
    stats = {"imu_pitch_median_deg": float(np.degrees(np.median(pitch_lp))),
             "imu_roll_median_deg": float(np.degrees(np.median(roll_lp))),
             "roll_dev_std_deg": float(np.degrees(np.std(roll_dev))),
             "pitch_dev_std_deg": float(np.degrees(np.std(pitch_dev))),
             "roll_dev_p95_deg": float(np.degrees(np.percentile(np.abs(roll_dev), 95))),
             "lp_samples": int(n_lp)}
    return out, stats
