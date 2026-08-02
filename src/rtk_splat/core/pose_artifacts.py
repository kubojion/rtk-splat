"""Resolve immutable contract-v2 initial poses and named refined artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .segment import SegmentReader


def pose_artifact_name(cfg) -> str:
    name = str(getattr(cfg.pose, "artifact", "rtk"))
    if name in {"", "rtk", "raw_rtk"}:
        return "rtk"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"invalid pose artifact name {name!r}")
    return name


def pose_artifact_dir(seg_dir: Path, cfg) -> Path:
    name = pose_artifact_name(cfg)
    if name == "rtk":
        raise ValueError("raw RTK poses live in frames.npz, not a sidecar directory")
    root = getattr(cfg.pose, "artifact_root", None)
    if root is None:
        paths = getattr(cfg, "paths", None)
        workdir = getattr(paths, "workdir", None)
        if workdir is None:
            raise ValueError(
                "named poses require pose.artifact_root or paths.workdir"
            )
        root = Path(workdir) / "pose_artifacts"
    return Path(root).expanduser() / name


def pose_paths(seg_dir: Path, cfg) -> tuple[Path, Path]:
    root = pose_artifact_dir(seg_dir, cfg)
    return root / "viewmats.npy", root / "cam_centers.npy"


def cloud_path(seg_dir: Path, cfg) -> Path:
    paths = getattr(cfg, "paths", None)
    workdir = getattr(paths, "workdir", None)
    root = getattr(getattr(cfg, "cloud", None), "artifact_root", None)
    if root is None:
        if workdir is None:
            raise ValueError(
                "cloud artifacts require cloud.artifact_root or paths.workdir"
            )
        root = Path(workdir) / "cloud_artifacts"
    return Path(root).expanduser() / pose_artifact_name(cfg) / "init_cloud.npz"


def pose_fingerprint(viewmats: np.ndarray) -> str:
    a = np.ascontiguousarray(viewmats.astype("<f4", copy=False))
    h = hashlib.sha256()
    h.update(str(a.shape).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def _requires_render_only_permission(root: Path) -> bool:
    declaration = root / "georeferencing.json"
    if not declaration.is_file():
        return False
    value = json.loads(declaration.read_text(encoding="utf-8"))
    return (
        value.get("artifact_class") == "diagnostic_render_only"
        or value.get("georeferencing_status") == "FAILED"
    )


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
    if not np.allclose(
        rotations @ np.swapaxes(rotations, 1, 2), np.eye(3), atol=2e-4
    ):
        raise ValueError("pose matrices contain non-orthonormal rotations")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=2e-4):
        raise ValueError("pose matrices contain improper rotations")


def load_pose_artifact(
    seg_dir: Path,
    cfg,
    *,
    allow_failed_georeferencing_for_render: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated OpenCV world-to-camera matrices and camera centres."""
    reader = SegmentReader(seg_dir)
    expected_n = int(reader.meta["n_frames"])
    if pose_artifact_name(cfg) == "rtk":
        frames = reader.frames
        required = {
            "initial_viewmat",
            "initial_camera_center_m",
            "pose_valid",
        }
        if not required <= set(frames):
            raise FileNotFoundError(
                "contract-v2 segment has no complete initial pose triplet"
            )
        valid = frames["pose_valid"].astype(bool)
        if not valid.all():
            raise ValueError(
                f"initial pose source covers {int(valid.sum())}/{len(valid)} frames"
            )
        viewmats = frames["initial_viewmat"]
        centers = frames["initial_camera_center_m"]
        _validate_viewmats(viewmats, expected_n)
        derived = np.linalg.inv(viewmats)[:, :3, 3]
        if centers.shape != (expected_n, 3) or not np.isfinite(centers).all():
            raise ValueError("invalid contract-v2 initial camera centres")
        if not np.allclose(centers, derived, atol=2e-4):
            raise ValueError("initial camera centres disagree with initial poses")
        return (
            viewmats.astype(np.float32, copy=False),
            centers.astype(np.float32, copy=False),
        )

    root = pose_artifact_dir(seg_dir, cfg)
    if (
        _requires_render_only_permission(root)
        and not allow_failed_georeferencing_for_render
    ):
        raise ValueError("pose artifact requires explicit render-only permission")
    view_path, center_path = pose_paths(seg_dir, cfg)
    if not view_path.exists():
        raise FileNotFoundError(
            f"pose artifact '{pose_artifact_name(cfg)}' is missing {view_path}; "
            "run the selected pose backend first"
        )
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
    return (
        viewmats.astype(np.float32, copy=False),
        centers.astype(np.float32, copy=False),
    )


def verify_cloud_matches_poses(
    cloud_file: Path,
    viewmats: np.ndarray,
    require_fingerprint: bool,
) -> None:
    if not cloud_file.exists():
        raise FileNotFoundError(
            f"pose-specific initial cloud is missing: {cloud_file}; run cloud first"
        )
    with np.load(cloud_file, allow_pickle=False) as cloud:
        if "pose_fingerprint" not in cloud:
            if require_fingerprint:
                raise ValueError(
                    f"{cloud_file} predates pose provenance; rebuild it"
                )
            return
        stored = str(cloud["pose_fingerprint"].item())
    if stored != pose_fingerprint(viewmats):
        raise ValueError(
            f"{cloud_file} was built from different poses; rebuild the cloud"
        )
