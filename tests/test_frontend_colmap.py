import hashlib
import json
import sqlite3
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rtk_splat.frontends.artifact import (
    ArtifactError,
    FrontendArtifactBuilder,
    collect_provenance,
    sha256_file,
    verify_frontend_seal,
)
from rtk_splat.frontends.colmap import (
    build_feature_extractor_command,
    build_matches_importer_command,
    build_rig_configurator_command,
    insert_pose_priors,
    run_feature_extraction,
    run_matches_importer,
    run_rig_configurator,
)
from rtk_splat.core.segment import POSITION_QUALITY_VOCABULARY, SegmentWriter


PAIR_ID_BASE = 2_147_483_647


def _option(command: list[str] | tuple[str, ...], name: str) -> str:
    return command[command.index(name) + 1]


def _segment(destination: Path) -> Path:
    writer = SegmentWriter(destination)
    images = writer.directory("images")
    payloads = {
        f"{side}_{index:06d}.jpg": f"{side}-{index}".encode()
        for index in range(4)
        for side in ("left", "right")
    }
    for name, payload in payloads.items():
        (images / name).write_bytes(payload)
    left_ns = (
        np.arange(4, dtype=np.int64) * 100_000_000 + 1_000_000_000
    )
    right_ns = left_ns + 2_000
    centers = np.column_stack(
        (np.arange(1.0, 5.0), np.full(4, 2.0), np.full(4, 3.0))
    )
    viewmats = np.repeat(np.eye(4)[None], 4, axis=0)
    viewmats[0, :3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    viewmats[:, :3, 3] = -np.einsum(
        "nij,nj->ni", viewmats[:, :3, :3], centers
    )
    writer.write_frames(
        {
            "frame_id": np.arange(4, dtype=np.int64),
            "timestamp_ns": left_ns,
            "left_image_path": np.array(
                [f"images/left_{index:06d}.jpg" for index in range(4)]
            ),
            "right_image_path": np.array(
                [f"images/right_{index:06d}.jpg" for index in range(4)]
            ),
            "right_timestamp_ns": right_ns,
            "stereo_sync_residual_ns": right_ns - left_ns,
            "stereo_left_sha256": np.array(
                [
                    hashlib.sha256(
                        payloads[f"left_{index:06d}.jpg"]
                    ).hexdigest()
                    for index in range(4)
                ]
            ),
            "stereo_right_sha256": np.array(
                [
                    hashlib.sha256(
                        payloads[f"right_{index:06d}.jpg"]
                    ).hexdigest()
                    for index in range(4)
                ]
            ),
            "initial_viewmat": viewmats,
            "initial_camera_center_m": centers,
            "pose_valid": np.ones(4, dtype=bool),
        }
    )
    camera = {
        "model": "PINHOLE",
        "width": 8,
        "height": 6,
        "K": [[10.0, 0.0, 4.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]],
        "distortion": [],
    }
    writer.write_calibration(
        {
            "contract_version": 2,
            "cameras": {"left": camera, "right": camera},
            "T_right_left": [
                [1.0, 0.0, 0.0, -0.12],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "transform_conventions": {"T_right_left": "right_from_left"},
        }
    )
    writer.write_meta(
        {
            "contract_version": 2,
            "n_frames": 4,
            "capabilities": {
                "stereo": True,
                "rgbd": False,
                "single_rtk": True,
                "dual_rtk": False,
                "depth_recorded": False,
                "depth_computed": False,
                "imu_present": False,
                "images_raw": False,
                "images_rectified": True,
            },
            "coordinate_frame": {
                "type": "local_enu",
                "world_frame_id": "map",
                "units": "m",
                "origin_wgs84": {
                    "latitude_deg": 52.0,
                    "longitude_deg": 16.0,
                    "ellipsoidal_altitude_m": 100.0,
                    "ellipsoid": "WGS84",
                    "vertical_datum": "WGS84 ellipsoid",
                },
            },
            "timebase": {
                "frame_timestamp_source": "left_camera_header",
                "observation_timestamp_source": "gnss_header",
                "unit": "ns",
                "association_clock_offset_ns": 0,
            },
            "position_observation": {
                "type": "gnss",
                "quantity": "antenna_phase_center",
                "sensor_frame_id": "primary_antenna",
                "coordinates": "ENU_m",
                "covariance_frame": "ENU_m2",
                "validity_field": "position_valid",
                "quality_field": "position_quality",
                "quality_vocabulary": list(POSITION_QUALITY_VOCABULARY),
            },
            "initial_pose": {
                "camera_frame_id": "left_camera",
                "position_quantity": "left_camera_center",
                "source": "synthetic_rtk_prior",
                "lever_arm_applied": True,
                "extrinsic_translation_sigma_m": [0.01, 0.02, 0.03],
                "extrinsic_translation_sigma_frame_id": "left_camera",
            },
        }
    )
    writer.write_manifest({"train": [0, 1, 2, 3], "val": [], "test": []})
    writer.write_observations(
        "gnss",
        {
            "frame_id": np.arange(4, dtype=np.int64),
            "frame_timestamp_ns": left_ns,
            "source_index": np.arange(4, dtype=np.int64),
            "source_timestamp_ns": left_ns + 1_000_000,
            "source_residual_ns": np.full(4, 1_000_000, dtype=np.int64),
            "enu_m": centers.copy(),
            "covariance_enu_m2": np.array(
                [
                    np.diag([0.0001, 0.0004, 0.0009]),
                    np.diag([0.0004, 0.0004, 0.0004]),
                    np.diag([0.0004, 0.0004, 0.0004]),
                    np.diag([0.0004, 0.0004, 0.0004]),
                ]
            ),
            "fix_status": np.array([2, 2, 2, 0], dtype=np.int16),
            "carrier_status": np.array([2, 2, 2, -1], dtype=np.int16),
            "position_valid": np.ones(4, dtype=bool),
            "position_quality": np.array(
                ["rtk_fixed", "rtk_fixed", "rtk_fixed", "standalone"],
                dtype=np.str_,
            ),
        },
    )
    return writer.finalize().root


def _git_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"], check=True
    )
    (path / "tracked").write_text("x")
    subprocess.run(["git", "-C", str(path), "add", "tracked"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "initial"], check=True)
    return path


def _artifact(root: Path) -> Path:
    segment = _segment(root / "segment")
    provenance = collect_provenance(
        segment,
        resolved_config={"frontend": "test"},
        colmap={"executable": "colmap", "version": "COLMAP 4.1.1"},
        seed=7,
        repo_root=_git_repo(root / "repo"),
    )
    return FrontendArtifactBuilder(segment, root / "work", "sealed").build(
        rig_config=[
            {
                "cameras": [
                    {
                        "image_prefix": "left_",
                        "ref_sensor": True,
                        "camera_model_name": "PINHOLE",
                        "camera_params": [1118.7, 1118.7, 965.4, 559.8],
                    },
                    {
                        "image_prefix": "right_",
                        "cam_from_rig_rotation": [1.0, 0.0, 0.0, 0.0],
                        "cam_from_rig_translation": [-0.1198, 0.0, 0.0],
                        "camera_model_name": "PINHOLE",
                        "camera_params": [1118.7, 1118.7, 965.4, 559.8],
                    },
                ]
            }
        ],
        keyframes={"frame_ids": [0, 1, 2, 3], "selector": "all"},
        pairs=[
            (f"left_{index:06d}.jpg", f"right_{index:06d}.jpg")
            for index in range(4)
        ],
        provenance=provenance,
        quality={"status": "prepared"},
    )


def _create_colmap_schema(database: Path, names: list[str]) -> None:
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS cameras(
              camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER,
              height INTEGER, params BLOB, prior_focal_length INTEGER);
            CREATE TABLE IF NOT EXISTS images(
              image_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
              camera_id INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS keypoints(
              image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
            CREATE TABLE IF NOT EXISTS descriptors(
              image_id INTEGER PRIMARY KEY, type INTEGER, rows INTEGER,
              cols INTEGER, data BLOB);
            CREATE TABLE IF NOT EXISTS rigs(
              rig_id INTEGER PRIMARY KEY, ref_sensor_id INTEGER,
              ref_sensor_type INTEGER);
            CREATE TABLE IF NOT EXISTS frames(frame_id INTEGER PRIMARY KEY, rig_id INTEGER);
            CREATE TABLE IF NOT EXISTS frame_data(
              frame_id INTEGER, data_id INTEGER, sensor_id INTEGER,
              sensor_type INTEGER);
            CREATE UNIQUE INDEX IF NOT EXISTS frame_sensor_assignment
              ON frame_data(data_id, sensor_type);
            CREATE TABLE IF NOT EXISTS pose_priors(
              pose_prior_id INTEGER PRIMARY KEY NOT NULL,
              corr_data_id INTEGER NOT NULL,
              corr_sensor_id INTEGER NOT NULL,
              corr_sensor_type INTEGER NOT NULL,
              position BLOB,
              position_covariance BLOB,
              gravity BLOB,
              coordinate_system INTEGER NOT NULL);
            CREATE UNIQUE INDEX IF NOT EXISTS pose_prior_data_assignment
              ON pose_priors(corr_data_id, corr_sensor_id, corr_sensor_type);
            CREATE TABLE IF NOT EXISTS matches(
              pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
            CREATE TABLE IF NOT EXISTS two_view_geometries(
              pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB,
              config INTEGER, F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB);
            """
        )
        # Real COLMAP appends to an existing database and, with
        # --ImageReader.single_camera, allocates exactly one camera per
        # invocation. The A/B runs one invocation per rig sensor.
        camera_id = int(
            connection.execute(
                "SELECT COALESCE(MAX(camera_id), 0) + 1 FROM cameras"
            ).fetchone()[0]
        )
        next_image_id = int(
            connection.execute(
                "SELECT COALESCE(MAX(image_id), 0) + 1 FROM images"
            ).fetchone()[0]
        )
        connection.execute(
            "INSERT INTO cameras VALUES (?, 1, 8, 6, NULL, 1)",
            (camera_id,),
        )
        rig_id = int(
            connection.execute(
                "SELECT COALESCE(MAX(rig_id), 0) + 1 FROM rigs"
            ).fetchone()[0]
        )
        connection.execute(
            "INSERT INTO rigs VALUES (?, ?, 0)",
            (rig_id, camera_id),
        )
        next_frame_id = int(
            connection.execute(
                "SELECT COALESCE(MAX(frame_id), 0) + 1 FROM frames"
            ).fetchone()[0]
        )
        for offset, name in enumerate(names):
            image_id = next_image_id + offset
            frame_id = next_frame_id + offset
            connection.execute(
                "INSERT INTO images VALUES (?, ?, ?)",
                (image_id, name, camera_id),
            )
            connection.execute(
                "INSERT INTO keypoints VALUES (?, 10, 4, ?)",
                (image_id, b"keypoints"),
            )
            connection.execute(
                "INSERT INTO descriptors VALUES (?, 0, 10, 128, ?)",
                (image_id, b"descriptors"),
            )
            # COLMAP 4.1 creates this default singleton rig/frame assignment
            # during feature extraction. ``rig_configurator`` replaces it
            # with synchronized multi-sensor frames.
            connection.execute(
                "INSERT INTO frames VALUES (?, ?)", (frame_id, rig_id)
            )
            connection.execute(
                "INSERT INTO frame_data VALUES (?, ?, ?, 0)",
                (frame_id, image_id, camera_id),
            )


def _feature_runner(command: list[str], *, check: bool) -> None:
    assert check
    if "--image_list_path" in command:
        listing = Path(_option(command, "--image_list_path"))
        images = [line for line in listing.read_text().splitlines() if line]
    else:
        images = sorted(
            path.name for path in Path(_option(command, "--image_path")).iterdir()
        )
    _create_colmap_schema(Path(_option(command, "--database_path")), images)


def _rig_runner(command: list[str], *, check: bool) -> None:
    assert check
    database = Path(_option(command, "--database_path"))
    with sqlite3.connect(database) as connection:
        images = {
            name: image_id
            for image_id, name in connection.execute(
                "SELECT image_id, name FROM images"
            )
        }
        connection.execute("DELETE FROM frame_data")
        connection.execute("DELETE FROM frames")
        connection.execute("DELETE FROM rigs")
        connection.execute("INSERT INTO rigs VALUES (1, 1, 0)")
        for frame_id in (1, 2, 3, 4):
            connection.execute("INSERT INTO frames VALUES (?, 1)", (frame_id,))
            left = images[f"left_{frame_id - 1:06d}.jpg"]
            right = images[f"right_{frame_id - 1:06d}.jpg"]
            connection.execute(
                "UPDATE images SET camera_id=1 WHERE image_id=?", (left,)
            )
            connection.execute(
                "UPDATE images SET camera_id=2 WHERE image_id=?", (right,)
            )
            connection.execute(
                "INSERT INTO frame_data VALUES (?, ?, 1, 0)",
                (frame_id, left),
            )
            connection.execute(
                "INSERT INTO frame_data VALUES (?, ?, 2, 0)",
                (frame_id, right),
            )


def _matching_runner(command: list[str], *, check: bool) -> None:
    assert check
    database = Path(_option(command, "--database_path"))
    with sqlite3.connect(database) as connection:
        images = {
            name: image_id
            for image_id, name in connection.execute(
                "SELECT image_id, name FROM images"
            )
        }
        for line in Path(_option(command, "--match_list_path")).read_text().splitlines():
            first, second = line.split()
            low, high = sorted((images[first], images[second]))
            pair_id = low * PAIR_ID_BASE + high
            connection.execute(
                "INSERT INTO matches VALUES (?, 20, 2, ?)",
                (pair_id, b"matches"),
            )
            connection.execute(
                """
                INSERT INTO two_view_geometries
                VALUES (?, 16, 2, ?, 2, NULL, NULL, NULL, NULL, NULL)
                """,
                (pair_id, b"inliers"),
            )


class CommandBuilderTests(unittest.TestCase):
    def test_gpu_and_cpu_feature_profiles_are_an_exact_controlled_ab(self):
        gpu = build_feature_extractor_command(
            "/tmp/artifact", "colmap", profile="gpu"
        )
        cpu = build_feature_extractor_command(
            "/tmp/artifact", "colmap", profile="cpu_reference"
        )
        self.assertEqual(_option(gpu, "--FeatureExtraction.use_gpu"), "1")
        self.assertNotIn("--SiftExtraction.estimate_affine_shape", gpu)
        self.assertNotIn("--SiftExtraction.domain_size_pooling", gpu)
        self.assertEqual(_option(cpu, "--FeatureExtraction.use_gpu"), "0")
        self.assertEqual(_option(cpu, "--SiftExtraction.estimate_affine_shape"), "1")
        self.assertEqual(_option(cpu, "--SiftExtraction.domain_size_pooling"), "1")
        self.assertEqual(_option(gpu, "--image_path"), "/tmp/artifact/images")
        self.assertNotIn("image_list_path", " ".join(gpu))

    def test_extraction_is_one_single_camera_pass_per_rig_sensor(self):
        """The rig stage needs exactly one camera per sensor.

        Per-image cameras leave the rig configurator unable to group the two
        sensors, and a single shared camera conflates them. Both previously
        surfaced only as a late, opaque rig/frame count error.
        """
        with tempfile.TemporaryDirectory() as tmp:
            artifact = _artifact(Path(tmp))
            run_feature_extraction(
                artifact, "colmap", profile="gpu", runner=_feature_runner
            )
            with sqlite3.connect(artifact / "database.db") as connection:
                cameras = connection.execute(
                    "SELECT COUNT(*) FROM cameras"
                ).fetchone()[0]
                rows = list(
                    connection.execute("SELECT name, camera_id FROM images")
                )
        self.assertEqual(cameras, 2)
        for prefix in ("left_", "right_"):
            assigned = {
                camera_id for name, camera_id in rows if name.startswith(prefix)
            }
            self.assertEqual(len(assigned), 1, f"{prefix} must share one camera")
        self.assertEqual(len(rows), 8)

    def test_rig_and_match_commands_use_sealed_inputs_and_verification(self):
        rig = build_rig_configurator_command("/tmp/artifact", "colmap")
        match = build_matches_importer_command("/tmp/artifact", "colmap")
        self.assertEqual(_option(rig, "--rig_config_path"), "/tmp/artifact/rig_config.json")
        self.assertEqual(_option(match, "--match_list_path"), "/tmp/artifact/pairs.txt")
        self.assertEqual(_option(match, "--match_type"), "pairs")
        self.assertEqual(_option(match, "--FeatureMatching.guided_matching"), "1")
        self.assertEqual(_option(match, "--FeatureMatching.rig_verification"), "1")
        self.assertEqual(
            _option(match, "--FeatureMatching.skip_geometric_verification"), "0"
        )


class ColmapStageTests(unittest.TestCase):
    def test_colmap_singleton_pre_rig_state_is_accepted_but_malformed_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = _artifact(Path(tmp))
            run_feature_extraction(
                artifact, "colmap", profile="gpu", runner=_feature_runner
            )
            with sqlite3.connect(artifact / "database.db") as connection:
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM rigs").fetchone()[0],
                    2,
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0],
                    8,
                )
                connection.execute(
                    "UPDATE rigs SET ref_sensor_id=999 WHERE rig_id=1"
                )
            with self.assertRaisesRegex(ArtifactError, "singleton pre-rig"):
                run_rig_configurator(
                    artifact,
                    "colmap",
                    runner=lambda *args, **kwargs: self.fail("must not run"),
                )
            self.assertFalse((artifact / "stages" / "rig.json").exists())

    def test_unmarked_preexisting_configured_rig_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = _artifact(Path(tmp))
            run_feature_extraction(
                artifact, "colmap", profile="gpu", runner=_feature_runner
            )
            _rig_runner(
                list(build_rig_configurator_command(artifact, "colmap")),
                check=True,
            )
            with self.assertRaisesRegex(ArtifactError, "unmarked/pre-existing"):
                run_rig_configurator(
                    artifact,
                    "colmap",
                    runner=lambda *args, **kwargs: self.fail("must not run"),
                )
            self.assertFalse((artifact / "stages" / "rig.json").exists())

    def test_mocked_full_frontend_inserts_exact_cartesian_blob_and_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = _artifact(Path(tmp))
            features = run_feature_extraction(
                artifact, "colmap", profile="gpu", runner=_feature_runner
            )
            self.assertEqual(features["n_images"], 8)
            rig = run_rig_configurator(
                artifact, "colmap", runner=_rig_runner
            )
            self.assertEqual(rig["n_frames"], 4)
            priors = insert_pose_priors(
                artifact,
                covariance_floor_m=0.02,
                min_fix_status=1,
                min_carrier_status=2,
            )
            self.assertEqual(priors["n_inserted"], 3)
            self.assertEqual(priors["n_skipped"], 1)
            self.assertEqual(priors["skipped_by_reason"], {"fix_status": 1})

            with sqlite3.connect(artifact / "database.db") as connection:
                row = connection.execute(
                    """
                    SELECT corr_data_id, corr_sensor_id, corr_sensor_type,
                           position, position_covariance, gravity,
                           coordinate_system
                    FROM pose_priors
                    """
                ).fetchone()
            self.assertEqual(row[:3], (1, 1, 0))
            self.assertEqual(struct.unpack("<3d", row[3]), (1.0, 2.0, 3.0))
            np.testing.assert_allclose(
                np.frombuffer(row[4], dtype="<f8").reshape(3, 3, order="F"),
                np.diag([0.0005, 0.0005, 0.0018]),
            )
            self.assertIsNone(row[5])
            self.assertEqual(row[6], 1)

            matching = run_matches_importer(
                artifact, "colmap", runner=_matching_runner
            )
            self.assertEqual(matching["n_requested_pairs"], 4)
            self.assertEqual(matching["n_verified_pairs"], 4)
            seal = verify_frontend_seal(artifact)
            self.assertEqual(seal["images"]["count"], 8)
            marker = json.loads(
                (artifact / "stages" / "matching.json").read_text()
            )
            self.assertEqual(
                marker["outputs"]["database.db"],
                sha256_file(artifact / "database.db"),
            )
            self.assertEqual(
                marker["outputs"]["frontend_seal.json"],
                sha256_file(artifact / "frontend_seal.json"),
            )

            def must_not_run(*args, **kwargs):
                raise AssertionError("completed stage reran")

            run_feature_extraction(
                artifact, "colmap", profile="gpu", runner=must_not_run
            )
            run_rig_configurator(artifact, "colmap", runner=must_not_run)
            insert_pose_priors(
                artifact,
                covariance_floor_m=0.02,
                min_fix_status=1,
                min_carrier_status=2,
            )
            run_matches_importer(artifact, "colmap", runner=must_not_run)

    def test_too_few_trusted_priors_fail_before_matching_and_report_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = _artifact(Path(tmp))
            run_feature_extraction(
                artifact, "colmap", profile="gpu", runner=_feature_runner
            )
            run_rig_configurator(artifact, "colmap", runner=_rig_runner)
            with self.assertRaisesRegex(
                ArtifactError, "0 trusted priors, at least 3 required"
            ):
                insert_pose_priors(
                    artifact,
                    min_fix_status=3,
                    min_carrier_status=3,
                )
            audits = list(
                (artifact / "stages").glob("pose_priors_rejected_*.json")
            )
            self.assertEqual(len(audits), 1)
            rejection = json.loads(audits[0].read_text())
            self.assertEqual(rejection["n_trusted_priors"], 0)
            self.assertEqual(rejection["skipped_by_reason"], {"fix_status": 4})
            with sqlite3.connect(artifact / "database.db") as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM pose_priors"
                    ).fetchone()[0],
                    0,
                )
            with self.assertRaisesRegex(ArtifactError, "pose_priors"):
                run_matches_importer(
                    artifact,
                    "colmap",
                    runner=lambda *args, **kwargs: self.fail("must not match"),
                )

    def test_fingerprint_change_is_rejected_without_rerunning(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = _artifact(Path(tmp))
            run_feature_extraction(
                artifact, "colmap", profile="gpu", runner=_feature_runner
            )
            with self.assertRaisesRegex(ArtifactError, "inputs differ"):
                run_feature_extraction(
                    artifact,
                    "colmap",
                    profile="cpu_reference",
                    runner=lambda *args, **kwargs: self.fail("must not run"),
                )

    def test_preexisting_unmarked_database_is_refused_without_creating_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = _artifact(Path(tmp))
            (artifact / "database.db").write_bytes(b"do not replace")
            with self.assertRaisesRegex(ArtifactError, "unmarked"):
                run_feature_extraction(
                    artifact, "colmap", profile="gpu", runner=_feature_runner
                )
            self.assertEqual((artifact / "database.db").read_bytes(), b"do not replace")
            self.assertFalse((artifact / "stages" / "features.json").exists())

    def test_partial_matching_output_cannot_be_silently_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = _artifact(Path(tmp))
            run_feature_extraction(
                artifact, "colmap", profile="gpu", runner=_feature_runner
            )
            run_rig_configurator(artifact, "colmap", runner=_rig_runner)
            priors = insert_pose_priors(artifact)
            self.assertEqual(priors["n_inserted"], 4)
            self.assertEqual(
                priors["position_quality_counts"],
                {"rtk_fixed": 3, "standalone": 1},
            )
            calls = 0

            def partial_runner(command, *, check):
                nonlocal calls
                calls += 1
                database = Path(_option(command, "--database_path"))
                with sqlite3.connect(database) as connection:
                    images = {
                        name: image_id
                        for image_id, name in connection.execute(
                            "SELECT image_id, name FROM images"
                        )
                    }
                    first, second = (
                        Path(_option(command, "--match_list_path"))
                        .read_text()
                        .splitlines()[0]
                        .split()
                    )
                    low, high = sorted((images[first], images[second]))
                    pair_id = low * PAIR_ID_BASE + high
                    connection.execute(
                        "INSERT INTO matches VALUES (?, 2, 2, ?)",
                        (pair_id, b"x"),
                    )
                    connection.execute(
                        """
                        INSERT INTO two_view_geometries
                        VALUES (?, 2, 2, ?, 2, NULL, NULL, NULL, NULL, NULL)
                        """,
                        (pair_id, b"x"),
                    )

            with self.assertRaisesRegex(ArtifactError, "partial"):
                run_matches_importer(
                    artifact, "colmap", runner=partial_runner
                )
            with self.assertRaisesRegex(ArtifactError, "partial"):
                run_matches_importer(
                    artifact, "colmap", runner=partial_runner
                )
            self.assertEqual(calls, 1)

    def test_pose_prior_schema_without_unique_assignment_rolls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = _artifact(Path(tmp))
            run_feature_extraction(
                artifact, "colmap", profile="gpu", runner=_feature_runner
            )
            run_rig_configurator(artifact, "colmap", runner=_rig_runner)
            with sqlite3.connect(artifact / "database.db") as connection:
                connection.execute("DROP INDEX pose_prior_data_assignment")
            with self.assertRaisesRegex(ArtifactError, "unique"):
                insert_pose_priors(artifact)
            with sqlite3.connect(artifact / "database.db") as connection:
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0],
                    0,
                )


if __name__ == "__main__":
    unittest.main()
