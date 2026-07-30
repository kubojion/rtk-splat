"""Built-in OpenCV SGBM depth for calibrated rectified stereo pairs.

Learned depth is not selected through this module. A future backend or dataset
adapter must publish the same metric depth/validity artifacts and pass a
controlled benchmark before it becomes an active configuration option.
"""

import cv2
import numpy as np


def make_sgbm(cfg):
    s = cfg.depth.sgbm
    block = s.block_size
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=s.num_disparities,
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=s.uniqueness_ratio,
        speckleWindowSize=s.speckle_window,
        speckleRange=s.speckle_range,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def depth_from_pair(matcher, left_bgr: np.ndarray, right_bgr: np.ndarray,
                    fx: float, baseline_m: float, cfg):
    """(depth_m float32, valid bool) at full resolution. Invalid -> 0."""
    left_g = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_g = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    disp = matcher.compute(left_g, right_g).astype(np.float32) / 16.0
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = fx * baseline_m / disp
    valid = ((disp > 0.5)
             & (depth > cfg.depth.min_z_m)
             & (depth < cfg.depth.max_z_m))
    depth[~valid] = 0.0
    return depth, valid
