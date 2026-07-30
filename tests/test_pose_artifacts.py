import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rtk_splat.pose_artifacts import (
    cloud_path,
    load_pose_artifact,
    pose_fingerprint,
    verify_cloud_matches_poses,
)


def cfg(name=None):
    pose = SimpleNamespace()
    if name is not None:
        pose.artifact = name
    return SimpleNamespace(pose=pose)


class PoseArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.seg = Path(self.tmp.name)
        (self.seg / "segment_meta.json").write_text(
            json.dumps({"n_frames": 3}))
        self.viewmats = np.repeat(np.eye(4)[None], 3, axis=0)
        self.viewmats[:, 0, 3] = [0.0, -1.0, -2.0]
        self.centers = np.linalg.inv(self.viewmats)[:, :3, 3]
        np.save(self.seg / "viewmats.npy", self.viewmats)
        np.save(self.seg / "cam_centers.npy", self.centers)

    def tearDown(self):
        self.tmp.cleanup()

    def test_legacy_config_defaults_to_raw_rtk(self):
        viewmats, centers = load_pose_artifact(self.seg, cfg())
        np.testing.assert_allclose(viewmats, self.viewmats)
        np.testing.assert_allclose(centers, self.centers)
        self.assertEqual(cloud_path(self.seg, cfg()), self.seg / "init_cloud.npz")

    def test_named_sidecar_never_falls_back_to_raw(self):
        with self.assertRaisesRegex(FileNotFoundError, "stereo-export"):
            load_pose_artifact(self.seg, cfg("colmap_stereo"))

    def test_named_sidecar_and_cloud_are_isolated(self):
        sidecar = self.seg / "pose_artifacts" / "colmap_stereo"
        sidecar.mkdir(parents=True)
        refined = self.viewmats.copy()
        refined[:, 1, 3] = -0.1
        centres = np.linalg.inv(refined)[:, :3, 3]
        np.save(sidecar / "viewmats.npy", refined)
        np.save(sidecar / "cam_centers.npy", centres)
        loaded, _ = load_pose_artifact(self.seg, cfg("colmap_stereo"))
        cloud = cloud_path(self.seg, cfg("colmap_stereo"))
        np.savez(cloud, xyz=np.zeros((1, 3)), rgb=np.zeros((1, 3)),
                 pose_fingerprint=np.asarray(pose_fingerprint(loaded)))
        verify_cloud_matches_poses(cloud, loaded, require_fingerprint=True)
        np.testing.assert_allclose(np.load(self.seg / "viewmats.npy"),
                                   self.viewmats)

    def test_cloud_pose_mismatch_fails(self):
        cloud = self.seg / "init_cloud.npz"
        np.savez(cloud, xyz=np.zeros((1, 3)), rgb=np.zeros((1, 3)),
                 pose_fingerprint=np.asarray("wrong"))
        with self.assertRaisesRegex(ValueError, "different poses"):
            verify_cloud_matches_poses(
                cloud, self.viewmats, require_fingerprint=False)

    def test_invalid_rotation_fails(self):
        bad = self.viewmats.copy()
        bad[1, 0, 0] = 2.0
        np.save(self.seg / "viewmats.npy", bad)
        with self.assertRaisesRegex(ValueError, "non-orthonormal"):
            load_pose_artifact(self.seg, cfg())


if __name__ == "__main__":
    unittest.main()
