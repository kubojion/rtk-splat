import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rtk_splat.core.pose_artifacts import (
    cloud_path,
    load_pose_artifact,
    pose_fingerprint,
    verify_cloud_matches_poses,
)


def cfg(name=None, workdir=None):
    pose = SimpleNamespace()
    if name is not None:
        pose.artifact = name
    paths = SimpleNamespace(workdir=workdir) if workdir is not None else None
    return SimpleNamespace(pose=pose, paths=paths)


class PoseArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.seg = self.root / "segment"
        self.work = self.root / "work"
        self.seg.mkdir()
        (self.seg / "segment_meta.json").write_text(
            '{"contract_version": 2, "n_frames": 3}\n')
        self.viewmats = np.repeat(np.eye(4)[None], 3, axis=0)
        self.viewmats[:, 0, 3] = [0.0, -1.0, -2.0]
        self.centers = np.linalg.inv(self.viewmats)[:, :3, 3]
        self._write_initial_poses(self.viewmats)

    def _write_initial_poses(self, viewmats):
        np.savez_compressed(
            self.seg / "frames.npz",
            frame_id=np.arange(3, dtype=np.int64),
            timestamp_ns=np.arange(3, dtype=np.int64),
            left_image_path=np.asarray(
                [f"images/left_{index:06d}.jpg" for index in range(3)]
            ),
            initial_viewmat=viewmats,
            initial_camera_center_m=np.linalg.inv(viewmats)[:, :3, 3],
            pose_valid=np.ones(3, dtype=bool),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_config_without_artifact_uses_contract_initial_poses(self):
        viewmats, centers = load_pose_artifact(self.seg, cfg(workdir=self.work))
        np.testing.assert_allclose(viewmats, self.viewmats)
        np.testing.assert_allclose(centers, self.centers)
        self.assertEqual(
            cloud_path(self.seg, cfg(workdir=self.work)),
            self.work / "cloud_artifacts" / "rtk" / "init_cloud.npz",
        )

    def test_named_sidecar_never_falls_back_to_raw(self):
        with self.assertRaisesRegex(FileNotFoundError, "selected pose backend"):
            load_pose_artifact(
                self.seg, cfg("colmap_stereo", workdir=self.work)
            )

    def test_named_sidecar_and_cloud_are_isolated(self):
        sidecar = self.work / "pose_artifacts" / "colmap_stereo"
        sidecar.mkdir(parents=True)
        refined = self.viewmats.copy()
        refined[:, 1, 3] = -0.1
        centres = np.linalg.inv(refined)[:, :3, 3]
        np.save(sidecar / "viewmats.npy", refined)
        np.save(sidecar / "cam_centers.npy", centres)
        selected = cfg("colmap_stereo", workdir=self.work)
        loaded, _ = load_pose_artifact(self.seg, selected)
        cloud = cloud_path(self.seg, selected)
        cloud.parent.mkdir(parents=True)
        np.savez(cloud, xyz=np.zeros((1, 3)), rgb=np.zeros((1, 3)),
                 pose_fingerprint=np.asarray(pose_fingerprint(loaded)))
        verify_cloud_matches_poses(cloud, loaded, require_fingerprint=True)
        np.testing.assert_allclose(
            np.load(self.seg / "frames.npz")["initial_viewmat"], self.viewmats
        )

    def test_cloud_pose_mismatch_fails(self):
        cloud = self.work / "cloud_artifacts" / "rtk" / "init_cloud.npz"
        cloud.parent.mkdir(parents=True)
        np.savez(cloud, xyz=np.zeros((1, 3)), rgb=np.zeros((1, 3)),
                 pose_fingerprint=np.asarray("wrong"))
        with self.assertRaisesRegex(ValueError, "different poses"):
            verify_cloud_matches_poses(
                cloud, self.viewmats, require_fingerprint=False)

    def test_invalid_rotation_fails(self):
        bad = self.viewmats.copy()
        bad[1, 0, 0] = 2.0
        self._write_initial_poses(bad)
        with self.assertRaisesRegex(ValueError, "non-orthonormal"):
            load_pose_artifact(self.seg, cfg(workdir=self.work))


if __name__ == "__main__":
    unittest.main()
