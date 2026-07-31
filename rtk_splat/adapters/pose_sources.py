"""Interchangeable pose sources, selected by `pose.source` in config.

Every source exposes the same minimal surface used by the CLI stages:

  .track        RtkTrack-like arrays (fix_t, enu_xyz, relpos_t, relpos_yaw,
                relpos_carr) -- the yaw arrays are the source's best heading
                series, whatever it derives them from
  .origin       dict describing the world frame (for georeferencing metadata)
  .pose_frames(stamps, tilts=None) -> list[PosedFrame | None]

Sources:
  rtk_dual_antenna  GNSS fix + dual-antenna heading topic (u-blox moving base)
  gnss_course       GNSS fix only; yaw = smoothed track direction. Works on
                    any dataset with cm-grade positions; yaw is undefined when
                    (nearly) stationary, so such samples are marked invalid.
  trajectory_file   poses from a TUM-format text file (t x y z qx qy qz qw),
                    interpreted directly as CAMERA poses in a metric world
                    frame (e.g. published ground-truth trajectories).
"""

from pathlib import Path

import numpy as np

from rtk_splat.adapters.ros2_zed_ublox import (
    RELPOS_FLAG_NAMES,
    RtkTrack,
    build_typestore,
    read_rtk_track,
)
from rtk_splat.core.poses import LocalEnu, PosedFrame, pose_frames, viewmat_from


def _attach_enu(track: RtkTrack):
    enu = LocalEnu(track.fix_lat[0], track.fix_lon[0], track.fix_alt[0])
    track.enu_xyz = enu.to_enu(track.fix_lat, track.fix_lon, track.fix_alt)
    track.enu = enu
    track.origin = {"lat0": track.fix_lat[0], "lon0": track.fix_lon[0],
                    "alt0": track.fix_alt[0]}
    return track


class RtkDualAntenna:
    """Position from the GNSS fix topic, yaw from the dual-antenna vector."""

    def __init__(self, cfg, typestore):
        self.cfg = cfg
        self.track = _attach_enu(
            read_rtk_track(cfg.paths.bags, cfg.topics, typestore))
        self.origin = self.track.origin
        self.enu = self.track.enu

    def pose_frames(self, stamps, tilts=None):
        return pose_frames(self.track, stamps, self.cfg.pose, tilts)


class GnssCourse:
    """Position from the GNSS fix topic, yaw from the track direction.

    Synthesizes the heading arrays of RtkTrack from the smoothed ENU track
    tangent, so all downstream machinery (selection, gating, interpolation)
    is shared with the dual-antenna source. Samples slower than
    `min_course_speed_ms` get quality 0 (course undefined when stationary).
    """

    def __init__(self, cfg, typestore):
        self.cfg = cfg
        tr = _attach_enu(read_rtk_track(cfg.paths.bags, cfg.topics, typestore,
                                        need_relpos=False))
        w = max(3, int(cfg.pose.yaw_smooth_window))
        kernel = np.ones(w) / w
        pad = (w // 2, w - 1 - w // 2)
        xs = np.convolve(np.pad(tr.enu_xyz[:, 0], pad, "edge"), kernel, "valid")
        ys = np.convolve(np.pad(tr.enu_xyz[:, 1], pad, "edge"), kernel, "valid")
        dx, dy = np.gradient(xs, tr.fix_t), np.gradient(ys, tr.fix_t)
        speed = np.hypot(dx, dy)
        minimum_speed = float(
            getattr(cfg.pose, "min_course_speed_ms", 0.05)
        )
        if not np.isfinite(minimum_speed) or minimum_speed <= 0:
            raise ValueError("pose.min_course_speed_ms must be positive")
        tr.relpos_t = tr.fix_t.copy()
        tr.relpos_yaw = np.unwrap(np.arctan2(dy, dx))
        # Course observability is not a carrier-phase solution. Keep the
        # carrier status unknown and gate heading with a separate boolean.
        tr.relpos_carr = np.full(len(speed), -1, dtype=np.int8)
        tr.heading_valid = speed >= minimum_speed
        tr.heading_quality_kind = "course"
        tr.relpos_header_ns = tr.fix_header_ns.copy()
        tr.relpos_log_ns = tr.fix_log_ns.copy()
        tr.relpos_ned_m = np.full((len(tr.fix_t), 3), np.nan)
        tr.relpos_acc_heading_rad = np.full(len(tr.fix_t), np.nan)
        tr.relpos_flags = np.zeros(
            (len(tr.fix_t), len(RELPOS_FLAG_NAMES)), dtype=bool
        )
        self.track = tr
        self.origin = tr.origin
        self.enu = tr.enu

    def pose_frames(self, stamps, tilts=None):
        return pose_frames(self.track, stamps, self.cfg.pose, tilts)


class TrajectoryFile:
    """Camera poses straight from a TUM-format file; lever arms do not apply.

    The file's poses are camera-to-world (TUM convention); output viewmats are
    their inverses, position lerped and orientation slerped at frame stamps.
    """

    def __init__(self, cfg, typestore=None):
        from scipy.spatial.transform import Rotation, Slerp

        path = Path(cfg.pose.trajectory_file).expanduser()
        data = np.loadtxt(path, comments="#")
        if data.ndim != 2 or data.shape[1] != 8:
            raise RuntimeError(f"{path}: expected TUM rows 't x y z qx qy qz qw'")
        self.t = data[:, 0]
        if len(self.t) < 2 or not np.all(np.diff(self.t) > 0):
            raise RuntimeError(f"{path}: timestamps must be strictly increasing")
        if not np.isfinite(data).all():
            raise RuntimeError(f"{path}: trajectory contains NaN/Inf")
        self.xyz = data[:, 1:4]
        self.rots = Rotation.from_quat(data[:, 4:8])
        self._slerp = Slerp(self.t, self.rots)
        self.origin = {"trajectory_file": str(path)}
        self.enu = None
        # minimal track surface for the selection stage
        yaw = self.rots.as_euler("ZYX")[:, 0]
        self.track = RtkTrack(
            fix_t=self.t, fix_lat=np.zeros_like(self.t),
            fix_lon=np.zeros_like(self.t), fix_alt=self.xyz[:, 2],
            fix_status=np.full(len(self.t), -1, dtype=int),
            fix_cov_max=np.full(len(self.t), np.nan),
            relpos_t=self.t, relpos_yaw=np.unwrap(yaw),
            relpos_carr=np.full(len(self.t), -1),
            heading_valid=np.ones(len(self.t), dtype=bool),
            heading_quality_kind="trajectory")
        self.track.enu_xyz = self.xyz
        self.track.origin = self.origin

    def pose_frames(self, stamps, tilts=None):
        out = []
        for t in stamps:
            if not (self.t[0] <= t <= self.t[-1]):
                out.append(None)
                continue
            pos = np.array([np.interp(t, self.t, self.xyz[:, k]) for k in range(3)])
            r_wc = self._slerp([t]).as_matrix()[0]
            out.append(PosedFrame(t=t, viewmat=viewmat_from(r_wc, pos),
                                  cam_center=pos))
        return out


_SOURCES = {"rtk_dual_antenna": RtkDualAntenna, "gnss_course": GnssCourse,
            "trajectory_file": TrajectoryFile}


def make_pose_source(cfg):
    """(typestore, source). The ublox types are registered only when the
    selected source actually needs them, so generic datasets need no .msg dir."""
    name = cfg.pose.source
    if name not in _SOURCES:
        raise RuntimeError(f"unknown pose.source '{name}'; "
                           f"choose from {sorted(_SOURCES)}")
    with_ublox = name == "rtk_dual_antenna"
    if with_ublox and not hasattr(cfg.paths, "ublox_msgs_dir"):
        raise ValueError(
            "pose.source=rtk_dual_antenna requires paths.ublox_msgs_dir"
        )
    uses_bag = name in {"rtk_dual_antenna", "gnss_course"}
    ublox_dir = (
        Path(cfg.paths.ublox_msgs_dir).expanduser()
        if with_ublox
        else None
    )
    typestore = build_typestore(ublox_dir) if uses_bag else None
    return typestore, _SOURCES[name](cfg, typestore)
