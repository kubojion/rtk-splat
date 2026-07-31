import unittest
from types import SimpleNamespace

import numpy as np

from rtk_splat.poses import base_pose_at, pose_frames_from_extrinsic


def _config(**overrides):
    values = {
        "yaw_smooth_window": 1,
        "min_carr_soln": 2,
        "minimum_navsat_status": 1,
        "minimum_position_carrier_status": -1,
        "maximum_position_covariance_m2": 0.01,
        "antenna_forward_m": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _track(*, heading_kind="course"):
    return SimpleNamespace(
        fix_t=np.array([0.0, 1.0]),
        relpos_t=np.array([0.0, 1.0]),
        relpos_yaw=np.zeros(2),
        relpos_carr=np.full(2, -1),
        heading_valid=np.ones(2, dtype=bool),
        heading_quality_kind=heading_kind,
        fix_status=np.ones(2, dtype=np.int16),
        fix_carrier_status=np.full(2, -1, dtype=np.int16),
        fix_cov_max=np.full(2, 0.0004),
        enu_xyz=np.column_stack((np.arange(2), np.zeros((2, 2)))),
    )


class PoseQualityGateTests(unittest.TestCase):
    def test_declared_camera_from_antenna_transform_is_actually_applied(self):
        track = _track()
        transform = np.eye(4)
        # Camera is one metre forward of the antenna, therefore
        # camera-from-antenna translates antenna coordinates by -1 m.
        transform[0, 3] = -1.0
        poses = pose_frames_from_extrinsic(
            track, [0.5], _config(), transform
        )
        self.assertIsNotNone(poses[0])
        np.testing.assert_allclose(poses[0].cam_center, [1.5, 0.0, 0.0])
        np.testing.assert_allclose(
            np.linalg.inv(poses[0].viewmat)[:3, :3], np.eye(3)
        )

    def test_course_heading_is_not_mislabeled_as_carrier_fixed(self):
        track = _track(heading_kind="course")
        result = base_pose_at(track, track.relpos_yaw, 0.5, _config())
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(track.relpos_carr, [-1, -1])

    def test_course_observability_and_gnss_quality_are_independent_gates(self):
        track = _track()
        track.heading_valid[1] = False
        self.assertIsNone(base_pose_at(track, track.relpos_yaw, 0.5, _config()))

        track = _track()
        track.fix_cov_max[1] = 0.1
        self.assertIsNone(base_pose_at(track, track.relpos_yaw, 0.5, _config()))

    def test_dual_heading_still_requires_configured_carrier_quality(self):
        track = _track(heading_kind="carrier")
        track.relpos_carr[:] = [2, 1]
        self.assertIsNone(base_pose_at(track, track.relpos_yaw, 0.5, _config()))


if __name__ == "__main__":
    unittest.main()
