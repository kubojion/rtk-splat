"""Back-project depth maps into a fused, voxel-downsampled ENU point cloud."""

import numpy as np


def backproject(depth: np.ndarray, valid: np.ndarray, rgb: np.ndarray,
                intr: dict, stride: int):
    """Camera-frame points (M,3) + colors (M,3 uint8) from one depth map."""
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[vs, us]
    ok = valid[vs, us]
    us, vs, d = us[ok], vs[ok], d[ok]
    x = (us - intr["cx"]) / intr["fx"] * d
    y = (vs - intr["cy"]) / intr["fy"] * d
    pts = np.stack([x, y, d], axis=-1).astype(np.float32)
    cols = rgb[vs, us]
    return pts, cols


def to_world(pts_cam: np.ndarray, viewmat: np.ndarray) -> np.ndarray:
    """Camera-frame points -> ENU world via the inverse of world-to-camera."""
    r = viewmat[:3, :3]
    t = viewmat[:3, 3]
    return (pts_cam - t) @ r  # == R^T (p - t)


def voxel_downsample(pts: np.ndarray, cols: np.ndarray, voxel: float,
                     max_points: int):
    """Keep one point per voxel (first hit). Plain numpy, no Open3D."""
    keys = np.floor(pts / voxel).astype(np.int64)
    # hash 3 ints to 1 for np.unique
    h = keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791
    _, idx = np.unique(h, return_index=True)
    if len(idx) > max_points:
        idx = np.random.default_rng(0).choice(idx, max_points, replace=False)
    return pts[idx], cols[idx]
