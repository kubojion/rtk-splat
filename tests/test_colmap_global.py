import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rtk_splat.colmap_global import (
    global_mapper_command,
    inspect_rig_model,
    parse_model_analyzer,
    prepare,
    rigid_alignment,
    robust_rigid_alignment,
)


def global_config(**overrides):
    values = {
        "source_pose_artifact": "colmap_stereo",
        "num_threads": 6,
        "random_seed": 11,
        "ba_num_iterations": 3,
        "max_num_tracks": 0,
        "required_tracks_per_view": 0,
        "skip_retriangulation": False,
    }
    values.update(overrides)
    return SimpleNamespace(
        global_mapper=SimpleNamespace(**values),
        colmap=SimpleNamespace(
            alignment_max_error_m=0.15,
            alignment_ransac_iterations=1000,
            scale_range=[0.98, 1.02]),
    )


class FixedScaleAlignmentTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.source = rng.normal(size=(60, 3))
        angle = 0.31
        self.rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.translation = np.array([2.1, -0.7, 0.4])
        self.target = (self.rotation @ self.source.T).T + self.translation

    def test_exact_metric_recovery(self):
        rotation, translation = rigid_alignment(self.source, self.target)
        np.testing.assert_allclose(rotation, self.rotation, atol=1.0e-12)
        np.testing.assert_allclose(
            translation, self.translation, atol=1.0e-12)

    def test_ransac_rejects_outliers(self):
        target = self.target.copy()
        target[[3, 17, 44]] += np.array([1.0, -2.0, 0.5])
        rotation, translation, inliers, errors = robust_rigid_alignment(
            self.source, target, 0.02, 1000, seed=3)
        self.assertEqual(int(inliers.sum()), 57)
        np.testing.assert_allclose(rotation, self.rotation, atol=1.0e-10)
        np.testing.assert_allclose(
            translation, self.translation, atol=1.0e-10)
        self.assertGreater(errors[3], 1.0)

    def test_scale_is_not_silently_absorbed(self):
        scaled = 1.02 * self.target
        rotation, translation = rigid_alignment(self.source, scaled)
        errors = np.linalg.norm(
            (rotation @ self.source.T).T + translation - scaled, axis=1)
        self.assertGreater(float(np.percentile(errors, 95)), 0.02)

    def test_collinear_trajectory_is_rejected(self):
        line = np.stack([
            np.arange(8, dtype=float), np.zeros(8), np.zeros(8)], axis=1)
        with self.assertRaisesRegex(ValueError, "line"):
            rigid_alignment(line, line)


class CommandTests(unittest.TestCase):
    def test_full_profile_fixes_calibration_and_uses_cpu(self):
        command = global_mapper_command(
            Path("/tmp/global sidecar"), global_config(), "/x/colmap")
        self.assertEqual(command[:2], ["/x/colmap", "global_mapper"])
        expected = {
            "--GlobalMapper.refine_sensor_from_rig": "0",
            "--GlobalMapper.ba_refine_focal_length": "0",
            "--GlobalMapper.ba_refine_principal_point": "0",
            "--GlobalMapper.ba_refine_extra_params": "0",
            "--GlobalMapper.ba_refine_rig_from_world": "1",
            "--GlobalMapper.gp_use_gpu": "0",
            "--GlobalMapper.ba_ceres_use_gpu": "0",
            "--GlobalMapper.num_threads": "6",
            "--GlobalMapper.random_seed": "11",
        }
        for option, value in expected.items():
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertNotIn("--GlobalMapper.keep_max_num_tracks", command)
        self.assertNotIn("--GlobalMapper.skip_retriangulation", command)
        self.assertNotIn("feature_extractor", command)
        self.assertNotIn("sequential_matcher", command)

    def test_reduced_profile_is_explicit(self):
        command = global_mapper_command(
            Path("/tmp/a"), global_config(
                max_num_tracks=60000, required_tracks_per_view=1000,
                skip_retriangulation=True),
            "colmap")
        self.assertEqual(command[command.index(
            "--GlobalMapper.keep_max_num_tracks") + 1], "60000")
        self.assertEqual(command[command.index(
            "--GlobalMapper.track_required_tracks_per_view") + 1], "1000")
        self.assertEqual(command[command.index(
            "--GlobalMapper.skip_retriangulation") + 1], "1")


class AnalyzerTests(unittest.TestCase):
    def test_parser_requires_complete_model_summary(self):
        output = """
I123 model.cc:440] Rigs: 1
I123 model.cc:441] Cameras: 2
I123 model.cc:442] Frames: 2
I123 model.cc:443] Registered frames: 2
I123 model.cc:445] Images: 4
I123 model.cc:446] Registered images: 4
I123 model.cc:448] Points: 101
I123 model.cc:449] Observations: 505
I123 model.cc:451] Mean track length: 5.0
I123 model.cc:453] Mean observations per image: 126.25
I123 model.cc:456] Mean reprojection error: 0.42px
"""
        parsed = parse_model_analyzer(output)
        self.assertEqual(parsed["registered_images"], 4)
        self.assertEqual(parsed["observations"], 505)
        self.assertAlmostEqual(parsed["mean_reprojection_error_px"], 0.42)
        with self.assertRaisesRegex(ValueError, "missing"):
            parse_model_analyzer("I123 model.cc:440] Rigs: 1\n")


class RigModelTests(unittest.TestCase):
    def _write_model(self, root: Path, baseline: float = 0.12):
        root.mkdir()
        (root / "cameras.txt").write_text(
            "1 PINHOLE 1920 1080 1000 1000 960.5 540.5\n"
            "2 PINHOLE 1920 1080 1000 1000 960.5 540.5\n")
        (root / "rigs.txt").write_text(
            "1 2 CAMERA 1 CAMERA 2 1 1 0 0 0 "
            f"{-baseline} 0 0\n")
        (root / "frames.txt").write_text(
            "10 1 1 0 0 0 0 0 0 2 CAMERA 1 1 CAMERA 2 2\n"
            "11 1 1 0 0 0 -1 0 0 2 CAMERA 1 3 CAMERA 2 4\n")
        (root / "images.txt").write_text(
            "1 1 0 0 0 0 0 0 1 zed/left/000000.jpg\n"
            "0 0 1\n"
            "2 1 0 0 0 0 0 0 2 zed/right/000000.jpg\n"
            "0 0 1\n"
            "3 1 0 0 0 1 0 0 1 zed/left/000001.jpg\n"
            "0 0 1\n"
            "4 1 0 0 0 1 0 0 2 zed/right/000001.jpg\n"
            "0 0 1\n")

    def _write_database(self, path: Path):
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE images("
            "image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)")
        connection.executemany(
            "INSERT INTO images VALUES (?, ?, ?)", [
                (1, "zed/left/000000.jpg", 1),
                (2, "zed/right/000000.jpg", 2),
                (3, "zed/left/000001.jpg", 1),
                (4, "zed/right/000001.jpg", 2),
            ])
        connection.commit()
        connection.close()

    def test_complete_fixed_rig_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            database = root / "database.db"
            self._write_model(model)
            self._write_database(database)
            inventory = {"cameras": [
                {"camera_id": 1, "width": 1920, "height": 1080,
                 "params": [1000, 1000, 960.5, 540.5]},
                {"camera_id": 2, "width": 1920, "height": 1080,
                 "params": [1000, 1000, 960.5, 540.5]},
            ]}
            result = inspect_rig_model(
                model, database, inventory, 2, 0.12)
        self.assertEqual(result["n_registered_images"], 4)
        self.assertEqual(
            result["observations_per_image"]["minimum"], 1)
        self.assertAlmostEqual(result["baseline_error_m"], 0.0)
        self.assertAlmostEqual(result["rig_rotation_error_deg"], 0.0)

    def test_baseline_change_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            database = root / "database.db"
            self._write_model(model, baseline=0.125)
            self._write_database(database)
            inventory = {"cameras": [
                {"camera_id": 1, "width": 1920, "height": 1080,
                 "params": [1000, 1000, 960.5, 540.5]},
                {"camera_id": 2, "width": 1920, "height": 1080,
                 "params": [1000, 1000, 960.5, 540.5]},
            ]}
            result = inspect_rig_model(
                model, database, inventory, 2, 0.12)
        self.assertAlmostEqual(result["baseline_error_m"], 0.005)


class PrepareTests(unittest.TestCase):
    @staticmethod
    def _source_database(path: Path):
        connection = sqlite3.connect(path)
        connection.executescript("""
            CREATE TABLE cameras(
                camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER,
                height INTEGER, params BLOB, prior_focal_length INTEGER);
            CREATE TABLE images(
                image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER);
            CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER);
            CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY, rows INTEGER);
            CREATE TABLE matches(pair_id INTEGER PRIMARY KEY, rows INTEGER);
            CREATE TABLE two_view_geometries(
                pair_id INTEGER PRIMARY KEY, rows INTEGER);
            CREATE TABLE pose_priors(image_id INTEGER PRIMARY KEY);
            CREATE TABLE rigs(
                rig_id INTEGER PRIMARY KEY, ref_sensor_id INTEGER,
                ref_sensor_type INTEGER);
            CREATE TABLE frames(frame_id INTEGER PRIMARY KEY, rig_id INTEGER);
            CREATE TABLE frame_data(
                frame_id INTEGER, data_id INTEGER, sensor_id INTEGER,
                sensor_type INTEGER);
        """)
        params = np.asarray(
            [1000.0, 1000.0, 960.5, 540.5], dtype="<f8").tobytes()
        connection.executemany(
            "INSERT INTO cameras VALUES (?, 1, 1920, 1080, ?, 1)",
            [(1, params), (2, params)])
        connection.executemany(
            "INSERT INTO images VALUES (?, ?, ?)", [
                (1, "zed/left/000000.jpg", 1),
                (2, "zed/right/000000.jpg", 2),
            ])
        connection.executemany(
            "INSERT INTO keypoints VALUES (?, 10)", [(1,), (2,)])
        connection.executemany(
            "INSERT INTO descriptors VALUES (?, 10)", [(1,), (2,)])
        connection.execute("INSERT INTO matches VALUES (1, 8)")
        connection.execute("INSERT INTO two_view_geometries VALUES (1, 8)")
        connection.execute("INSERT INTO rigs VALUES (1, 1, 0)")
        connection.execute("INSERT INTO frames VALUES (10, 1)")
        connection.executemany(
            "INSERT INTO frame_data VALUES (10, ?, ?, 0)",
            [(1, 1), (2, 2)])
        connection.commit()
        connection.close()

    def test_prepare_publishes_independent_verified_clone(self):
        with tempfile.TemporaryDirectory() as tmp:
            segment = Path(tmp) / "segment"
            source = segment / "pose_artifacts" / "colmap_stereo"
            work = source / "colmap"
            images = work / "images"
            images.mkdir(parents=True)
            database = work / "database.db"
            self._source_database(database)
            (work / "rig_config.json").write_text("{}")
            (source / "sidecar_config.json").write_text(
                json.dumps({"baseline_m_camera_info": 0.12}))
            (source / "frame_manifest.json").write_text("[]")
            source_model = work / "models_text" / "0"
            source_model.mkdir(parents=True)
            for name in ("cameras.txt", "rigs.txt", "frames.txt"):
                (source_model / name).write_text(f"source {name}\n")
            np.save(source / "viewmats.npy", np.eye(4)[None])
            np.save(source / "cam_centers.npy", np.zeros((1, 3)))
            (source / "quality.json").write_text(json.dumps({
                "pose_fingerprint": "source",
                "selected_model": str(source_model),
            }))
            segment.mkdir(exist_ok=True)
            (segment / "segment_meta.json").write_text(
                json.dumps({"n_frames": 1}))
            np.save(segment / "cam_centers.npy", np.zeros((1, 3)))
            cfg = global_config(minimum_free_space_gb=0)
            cfg.pose = SimpleNamespace(artifact="global_test")

            prepared = prepare(segment, cfg)
            clone = prepared / "database.db"
            source_stat = database.stat()
            clone_stat = clone.stat()
            record = json.loads(
                (segment / "pose_artifacts" / "global_test"
                 / "source.json").read_text())

            self.assertNotEqual(source_stat.st_ino, clone_stat.st_ino)
            self.assertEqual(
                record["database_inventory"]["tables"]["pose_priors"], 0)
            self.assertTrue((prepared / "images").is_symlink())
            self.assertEqual(
                record["rtk_centers_snapshot_sha256"],
                hashlib.sha256(
                    (segment / "pose_artifacts" / "global_test"
                     / "rtk_camera_centers.npy").read_bytes()).hexdigest())
            self.assertEqual(
                database.read_bytes(), clone.read_bytes())


if __name__ == "__main__":
    unittest.main()
