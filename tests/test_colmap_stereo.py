import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rtk_splat.colmap_stereo import (
    _validate_rectified_stereo,
    aligned_viewmat,
    colmap_commands,
    make_rig_config,
    parse_colmap_images,
    qvec_to_rotmat,
    robust_similarity,
    umeyama,
)


def calibration(tx=0.0):
    fx, fy, cx, cy = 1000.0, 1001.0, 960.0, 540.0
    return {
        "width": 1920, "height": 1080,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "d": [0.0] * 8, "r": np.eye(3).tolist(),
        "p": [[fx, 0, cx, tx], [0, fy, cy, 0], [0, 0, 1, 0]],
    }


class RigTests(unittest.TestCase):
    def test_rig_sign_principal_point_and_disparity(self):
        baseline = 0.12
        left, right = calibration(), calibration(-1000.0 * baseline)
        measured = _validate_rectified_stereo(left, right, baseline)
        self.assertAlmostEqual(measured, baseline)
        cameras = make_rig_config(left, right, baseline)[0]["cameras"]
        self.assertTrue(cameras[0]["ref_sensor"])
        self.assertEqual(cameras[1]["cam_from_rig_translation"],
                         [-baseline, 0.0, 0.0])
        self.assertEqual(cameras[0]["camera_params"],
                         [1000.0, 1001.0, 960.5, 540.5])
        point = np.array([0.4, 0.0, 5.0])
        u_left = left["fx"] * point[0] / point[2] + left["cx"]
        u_right = right["fx"] * (point[0] - baseline) / point[2] + right["cx"]
        self.assertGreater(u_left, u_right)
        self.assertAlmostEqual(u_left - u_right,
                               left["fx"] * baseline / point[2])

    def test_wrong_baseline_sign_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "wrong stereo-baseline sign"):
            _validate_rectified_stereo(calibration(),
                                       calibration(120.0), 0.12)


class ModelTests(unittest.TestCase):
    def test_scalar_first_quaternion(self):
        r = qvec_to_rotmat([np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
        np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-7)

    def test_images_parser_uses_name_and_ignores_observations(self):
        text = """# Image list
7 1 0 0 0 1 2 3 9 zed/right/000004.jpg
10.0 20.0 -1 30.0 40.0 5
2 0.7071067812 0 0 0.7071067812 4 5 6 8 zed/left/000004.jpg

"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "images.txt"
            path.write_text(text)
            parsed = parse_colmap_images(path)
        self.assertEqual(set(parsed), {
            "zed/right/000004.jpg", "zed/left/000004.jpg"})
        np.testing.assert_allclose(parsed["zed/right/000004.jpg"][1], [1, 2, 3])

    def test_duplicate_image_name_is_rejected(self):
        line = "1 1 0 0 0 0 0 0 1 zed/left/000000.jpg\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "images.txt"
            path.write_text(line + "\n" + line)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                parse_colmap_images(path)


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.source = rng.normal(size=(40, 3))
        angle = 0.4
        self.rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        self.scale = 1.007
        self.translation = np.array([4.0, -2.0, 0.7])
        self.target = (self.scale * (self.rotation @ self.source.T)).T \
            + self.translation

    def test_umeyama_exact(self):
        scale, rotation, translation = umeyama(self.source, self.target)
        self.assertAlmostEqual(scale, self.scale, places=10)
        np.testing.assert_allclose(rotation, self.rotation, atol=1e-10)
        np.testing.assert_allclose(translation, self.translation, atol=1e-10)

    def test_ransac_rejects_outliers(self):
        target = self.target.copy()
        target[[2, 9, 21]] += np.array([3.0, -2.0, 1.0])
        scale, rotation, translation, inliers, errors = robust_similarity(
            self.source, target, max_error_m=0.05, iterations=1000)
        self.assertEqual(int(inliers.sum()), 37)
        self.assertAlmostEqual(scale, self.scale, places=8)
        np.testing.assert_allclose(rotation, self.rotation, atol=1e-8)
        np.testing.assert_allclose(translation, self.translation, atol=1e-8)
        self.assertGreater(errors[2], 1.0)

    def test_collinear_trajectory_is_rejected(self):
        x = np.stack([np.arange(5), np.zeros(5), np.zeros(5)], axis=1)
        with self.assertRaisesRegex(ValueError, "line"):
            umeyama(x, x)

    def test_pose_transform_is_applied_once(self):
        r_cw = qvec_to_rotmat([1, 0, 0, 0])
        center = np.array([1.0, 2.0, 3.0])
        vm, aligned_center = aligned_viewmat(
            r_cw, center, self.scale, self.rotation, self.translation)
        expected = self.scale * self.rotation @ center + self.translation
        np.testing.assert_allclose(aligned_center, expected)
        np.testing.assert_allclose(np.linalg.inv(vm)[:3, 3], expected)
        np.testing.assert_allclose(np.linalg.inv(vm)[:3, :3], self.rotation)


class CommandTests(unittest.TestCase):
    def test_solver_order_and_fixed_calibration(self):
        cfg = SimpleNamespace(colmap=SimpleNamespace(
            max_image_size=-1, max_num_features=8192,
            feature_num_threads=8, sequential_overlap=10))
        commands = colmap_commands(Path("/tmp/a path"), cfg, "/x/colmap")
        self.assertEqual([name for name, _ in commands], [
            "feature_extractor", "rig_configurator",
            "sequential_matcher", "mapper"])
        mapper = commands[-1][1]
        self.assertEqual(mapper[mapper.index(
            "--Mapper.ba_refine_sensor_from_rig") + 1], "0")
        self.assertEqual(mapper[mapper.index(
            "--Mapper.ba_refine_focal_length") + 1], "0")
        self.assertEqual(mapper[mapper.index("--Mapper.ba_use_gpu") + 1], "0")
        self.assertTrue(all(isinstance(arg, str)
                            for _, command in commands for arg in command))


if __name__ == "__main__":
    unittest.main()
