"""Resolve and validate immutable pose artifacts.

The original RTK poses stay at ``segment/viewmats.npy``.  A refined pose
source lives in ``segment/pose_artifacts/<name>/`` and gets its own initial
cloud, preventing accidental mixing of geometry built in different frames.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np


_RAW_NAMES = {"", "rtk", "raw_rtk"}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def pose_artifact_name(cfg) -> str:
    name = str(getattr(cfg.pose, "artifact", "rtk"))
    if name in _RAW_NAMES:
        return "rtk"
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError(f"invalid pose artifact name {name!r}")
    return name


def pose_artifact_dir(seg_dir: Path, cfg) -> Path:
    name = pose_artifact_name(cfg)
    return seg_dir if name == "rtk" else seg_dir / "pose_artifacts" / name


def pose_paths(seg_dir: Path, cfg) -> tuple[Path, Path]:
    root = pose_artifact_dir(seg_dir, cfg)
    return root / "viewmats.npy", root / "cam_centers.npy"


def cloud_path(seg_dir: Path, cfg) -> Path:
    return pose_artifact_dir(seg_dir, cfg) / "init_cloud.npz"


def pose_fingerprint(viewmats: np.ndarray) -> str:
    a = np.ascontiguousarray(viewmats.astype("<f4", copy=False))
    h = hashlib.sha256()
    h.update(str(a.shape).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def _validate_viewmats(viewmats: np.ndarray, expected_n: int | None) -> None:
    if viewmats.ndim != 3 or viewmats.shape[1:] != (4, 4):
        raise ValueError(f"pose array must have shape (N,4,4), got {viewmats.shape}")
    if expected_n is not None and len(viewmats) != expected_n:
        raise ValueError(f"pose count {len(viewmats)} != segment frame count {expected_n}")
    if not np.isfinite(viewmats).all():
        raise ValueError("pose array contains NaN/Inf")
    if not np.allclose(viewmats[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("pose matrices have invalid homogeneous last row")
    rotations = viewmats[:, :3, :3]
    eye = np.eye(3)
    if not np.allclose(rotations @ np.swapaxes(rotations, 1, 2), eye,
                       atol=2e-4):
        raise ValueError("pose matrices contain non-orthonormal rotations")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=2e-4):
        raise ValueError("pose matrices contain improper rotations")


def load_pose_artifact(seg_dir: Path, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Return validated OpenCV world-to-camera matrices and camera centres."""
    view_path, center_path = pose_paths(seg_dir, cfg)
    if not view_path.exists():
        raise FileNotFoundError(
            f"pose artifact '{pose_artifact_name(cfg)}' is missing {view_path}; "
            "run stereo-export first" if pose_artifact_name(cfg) != "rtk"
            else f"raw RTK poses are missing: {view_path}")
    meta_path = seg_dir / "segment_meta.json"
    expected_n = json.loads(meta_path.read_text())["n_frames"] \
        if meta_path.exists() else None
    viewmats = np.load(view_path)
    _validate_viewmats(viewmats, expected_n)
    derived = np.linalg.inv(viewmats)[:, :3, 3]
    if center_path.exists():
        centers = np.load(center_path)
        if centers.shape != (len(viewmats), 3) or not np.isfinite(centers).all():
            raise ValueError(f"invalid camera-centre array {center_path}")
        if not np.allclose(centers, derived, atol=2e-4):
            raise ValueError(f"camera centres disagree with poses in {view_path}")
    else:
        centers = derived
    return viewmats.astype(np.float32, copy=False), centers.astype(np.float32,
                                                                   copy=False)


def verify_cloud_matches_poses(cloud_file: Path, viewmats: np.ndarray,
                               require_fingerprint: bool) -> None:
    if not cloud_file.exists():
        raise FileNotFoundError(
            f"pose-specific initial cloud is missing: {cloud_file}; run cloud first")
    with np.load(cloud_file) as cloud:
        if "pose_fingerprint" not in cloud:
            if require_fingerprint:
                raise ValueError(f"{cloud_file} predates pose provenance; rebuild it")
            return
        stored = str(cloud["pose_fingerprint"].item())
    current = pose_fingerprint(viewmats)
    if stored != current:
        raise ValueError(
            f"{cloud_file} was built from different poses; rebuild the cloud")
