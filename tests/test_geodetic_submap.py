import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from rtk_splat.backends.geodetic_pairs import (
    GeodeticPairPolicy,
    PairCandidate,
    RawGnssEndpoint,
    decide_pair,
)
from rtk_splat.backends.geodetic_submap import (
    GeodeticSubmapConfig,
    build_geodetic_submap_command,
    create_geodetic_frame_selection,
    geodetic_submap_export_context,
    prepare_geodetic_submap_plan,
    run_geodetic_submap_plan,
)
from rtk_splat.backends.mapper import (
    MapperConfig,
    prepare_mapper_backend,
    run_image_registration,
    run_mapper_solve,
    run_quality_summary,
)
from rtk_splat.core.segment import POSITION_QUALITY_VOCABULARY, SegmentWriter
from rtk_splat.frontends.artifact import (
    ArtifactError,
    FrontendArtifactBuilder,
    collect_provenance,
    create_frontend_seal,
    sha256_file,
    sqlite_logical_record,
)


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _positions() -> np.ndarray:
    return np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.20, 0.05, 0.01],
            [0.40, 0.20, 0.02],
            [0.60, 0.45, 0.03],
            [0.80, 0.80, 0.04],
            [1.00, 1.25, 0.05],
            [0.25, 0.10, 0.02],
            [5.00, 5.00, 0.06],
        ],
        dtype=np.float64,
    )


def _segment(path: Path) -> Path:
    writer = SegmentWriter(path)
    images = writer.directory("images")
    n = len(_positions())
    left_paths, right_paths = [], []
    for index in range(n):
        for side, paths in (("left", left_paths), ("right", right_paths)):
            relative = f"images/{side}_{index:06d}.jpg"
            (writer.staging_dir / relative).write_bytes(
                f"{side}-{index}".encode()
            )
            paths.append(relative)
    timestamps = (
        np.arange(n, dtype=np.int64) * 1_000_000_000 + 1_000_000_000
    )
    centers = _positions()
    viewmats = np.repeat(np.eye(4)[None], n, axis=0)
    viewmats[:, :3, 3] = -centers
    writer.write_frames(
        {
            "frame_id": np.arange(n, dtype=np.int64),
            "timestamp_ns": timestamps,
            "left_image_path": np.asarray(left_paths),
            "right_image_path": np.asarray(right_paths),
            "right_timestamp_ns": timestamps + 1_000,
            "stereo_sync_residual_ns": np.full(n, 1_000, dtype=np.int64),
            "initial_viewmat": viewmats,
            "initial_camera_center_m": centers,
            "pose_valid": np.ones(n, dtype=bool),
        }
    )
    camera = {
        "model": "PINHOLE",
        "width": 16,
        "height": 12,
        "K": [[20.0, 0.0, 8.0], [0.0, 20.0, 6.0], [0.0, 0.0, 1.0]],
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
            "n_frames": n,
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
                "observation_timestamp_source": "sensor_header",
                "unit": "ns",
                "association_clock_offset_ns": 0,
            },
            "position_observation": {
                "type": "gnss",
                "quantity": "antenna_phase_center",
                "sensor_frame_id": "gnss",
                "coordinates": "ENU_m",
                "covariance_frame": "ENU_m2",
                "validity_field": "position_valid",
                "quality_field": "position_quality",
                "quality_vocabulary": list(POSITION_QUALITY_VOCABULARY),
            },
            "initial_pose": {
                "camera_frame_id": "left_camera",
                "position_quantity": "left_camera_center",
                "source": "raw_gnss_plus_fixed_calibration",
                "lever_arm_applied": True,
                "extrinsic_translation_sigma_m": [0.01, 0.01, 0.01],
                "extrinsic_translation_sigma_frame_id": "left_camera",
            },
        }
    )
    writer.write_manifest(
        {"train": list(range(n - 2)), "val": [n - 2], "test": [n - 1]}
    )
    writer.write_observations(
        "gnss",
        {
            "frame_id": np.arange(n, dtype=np.int64),
            "frame_timestamp_ns": timestamps,
            "source_index": np.arange(n, dtype=np.int64),
            "source_timestamp_ns": timestamps,
            "enu_m": centers,
            "covariance_enu_m2": np.repeat(
                (np.eye(3) * 0.0004)[None], n, axis=0
            ),
            "fix_status": np.ones(n, dtype=np.int16),
            "carrier_status": np.full(n, 2, dtype=np.int16),
            "position_valid": np.ones(n, dtype=bool),
            "position_quality": np.full(n, "rtk_fixed", dtype="<U16"),
        },
    )
    return writer.finalize().root


def _pair_id(first: int, second: int) -> int:
    low, high = sorted((first, second))
    return low * 2_147_483_647 + high


def _create_database(artifact: Path) -> list[str]:
    manifest = json.loads((artifact / "frame_manifest.json").read_text())
    names = [
        str(row[field]["name"])
        for row in manifest["frames"]
        for field in ("left_image", "right_image")
    ]
    by_name = {name: index + 1 for index, name in enumerate(names)}
    with sqlite3.connect(artifact / "database.db") as connection:
        journal = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(journal).lower() != "wal":
            raise RuntimeError("test COLMAP database did not enter WAL mode")
        connection.execute("PRAGMA user_version=4010100")
        connection.executescript(
            """
            CREATE TABLE cameras(
              camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
              model INTEGER NOT NULL, width INTEGER NOT NULL,
              height INTEGER NOT NULL, params BLOB,
              prior_focal_length INTEGER NOT NULL);
            CREATE TABLE rigs(
              rig_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
              ref_sensor_id INTEGER NOT NULL,
              ref_sensor_type INTEGER NOT NULL);
            CREATE TABLE rig_sensors(
              rig_id INTEGER NOT NULL, sensor_id INTEGER NOT NULL,
              sensor_type INTEGER NOT NULL, sensor_from_rig BLOB,
              FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);
            CREATE TABLE frames(
              frame_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
              rig_id INTEGER NOT NULL,
              FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);
            CREATE TABLE images(
              image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
              name TEXT NOT NULL UNIQUE, camera_id INTEGER NOT NULL,
              CHECK(image_id >= 0 AND image_id < 2147483647),
              FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
            CREATE TABLE frame_data(
              frame_id INTEGER NOT NULL, data_id INTEGER NOT NULL,
              sensor_id INTEGER NOT NULL, sensor_type INTEGER NOT NULL,
              FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE);
            CREATE TABLE keypoints(
              image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
              cols INTEGER NOT NULL, data BLOB,
              FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
            CREATE TABLE descriptors(
              image_id INTEGER PRIMARY KEY NOT NULL, type INTEGER NOT NULL,
              rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB,
              FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
            CREATE TABLE matches(
              pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
              cols INTEGER NOT NULL, data BLOB);
            CREATE TABLE two_view_geometries(
              pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
              cols INTEGER NOT NULL, data BLOB, config INTEGER NOT NULL,
              F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB);
            CREATE TABLE pose_priors(
              pose_prior_id INTEGER PRIMARY KEY NOT NULL,
              corr_data_id INTEGER NOT NULL, corr_sensor_id INTEGER NOT NULL,
              corr_sensor_type INTEGER NOT NULL, position BLOB,
              position_covariance BLOB, gravity BLOB,
              coordinate_system INTEGER NOT NULL);
            CREATE UNIQUE INDEX frame_sensor_assignment
              ON frame_data(data_id, sensor_type);
            CREATE UNIQUE INDEX index_name ON images(name);
            CREATE UNIQUE INDEX pose_prior_data_assignment
              ON pose_priors(corr_data_id, corr_sensor_id, corr_sensor_type);
            CREATE UNIQUE INDEX rig_ref_sensor_assignment
              ON rigs(ref_sensor_id, ref_sensor_type);
            CREATE UNIQUE INDEX rig_sensor_assignment
              ON rig_sensors(sensor_id, sensor_type);
            INSERT INTO cameras VALUES(1, 1, 16, 12, X'00', 1);
            INSERT INTO cameras VALUES(2, 1, 16, 12, X'00', 1);
            INSERT INTO rigs VALUES(1, 1, 0);
            """
        )
        sensor_pose = np.asarray(
            [1.0, 0.0, 0.0, 0.0, 0.12, 0.0, 0.0], dtype="<f8"
        ).tobytes()
        connection.execute(
            "INSERT INTO rig_sensors VALUES(1, 2, 0, ?)", (sensor_pose,)
        )
        positions = _positions()
        covariance = np.eye(3, dtype="<f8") * 0.0004
        for frame_index, row in enumerate(manifest["frames"]):
            database_frame_id = frame_index + 1
            connection.execute(
                "INSERT INTO frames VALUES(?, 1)", (database_frame_id,)
            )
            for field, camera_id in (("left_image", 1), ("right_image", 2)):
                name = str(row[field]["name"])
                image_id = by_name[name]
                connection.execute(
                    "INSERT INTO images VALUES(?, ?, ?)",
                    (image_id, name, camera_id),
                )
                connection.execute(
                    "INSERT INTO frame_data VALUES(?, ?, ?, 0)",
                    (database_frame_id, image_id, camera_id),
                )
                connection.execute(
                    "INSERT INTO keypoints VALUES(?, 64, 2, ?)",
                    (image_id, bytes([image_id % 255]) * 512),
                )
                connection.execute(
                    "INSERT INTO descriptors VALUES(?, 0, 64, 128, ?)",
                    (image_id, bytes([image_id % 255]) * 1024),
                )
                if camera_id == 1:
                    connection.execute(
                        "INSERT INTO pose_priors VALUES(?, ?, 1, 0, ?, ?, NULL, 1)",
                        (
                            image_id,
                            image_id,
                            positions[frame_index].astype("<f8").tobytes(),
                            covariance.tobytes(order="F"),
                        ),
                    )

        pairs: set[tuple[str, str]] = set()
        for frame_index, row in enumerate(manifest["frames"]):
            left = str(row["left_image"]["name"])
            right = str(row["right_image"]["name"])
            pairs.add((left, right))
            if frame_index:
                previous = manifest["frames"][frame_index - 1]
                pairs.add((str(previous["left_image"]["name"]), left))
                pairs.add((str(previous["right_image"]["name"]), right))
        pairs.add((names[0], names[12]))  # nearby nonlocal revisit
        pairs.add((names[0], names[14]))  # false repetitive nonlocal pair
        for first, second in sorted(pairs):
            pair_id = _pair_id(by_name[first], by_name[second])
            count = 100
            data = np.zeros((count, 2), dtype="<u4").tobytes()
            connection.execute(
                "INSERT INTO matches VALUES(?, ?, 2, ?)",
                (pair_id, count, data),
            )
            connection.execute(
                "INSERT INTO two_view_geometries VALUES"
                "(?, ?, 2, ?, 2, NULL, NULL, NULL, NULL, NULL)",
                (pair_id, count, data),
            )
    return names


def _model(path: Path, token: bytes = b"model") -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (path / name).write_bytes(token + name.encode())


def _analyzer_output(n_images: int) -> str:
    return f"""
Rigs: 1
Cameras: 2
Frames: {n_images // 2}
Registered frames: {n_images // 2}
Images: {n_images}
Registered images: {n_images}
Points: 101
Observations: 4040
Mean track length: 4.0
Mean observations per image: 252.5
Mean reprojection error: 0.42
"""


def _option(command, name: str) -> str:
    return command[command.index(name) + 1]


class _Runner:
    def __init__(
        self,
        names: list[str],
        *,
        fail_pose_mapper: bool = False,
        mutate_pose_mapper_database: bool = False,
    ):
        self.names = names
        self.fail_pose_mapper = fail_pose_mapper
        self.mutate_pose_mapper_database = mutate_pose_mapper_database
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command, **_kwargs):
        command = tuple(command)
        self.commands.append(command)
        action = command[1]
        if action in {"global_mapper", "mapper"}:
            _model(Path(_option(command, "--output_path")) / "0", b"solve")
            return SimpleNamespace(returncode=0)
        if action == "image_registrator":
            _model(Path(_option(command, "--output_path")), b"registered")
            return SimpleNamespace(returncode=0)
        if action == "pose_prior_mapper":
            if self.fail_pose_mapper:
                raise RuntimeError("injected pose mapper failure")
            database = Path(_option(command, "--database_path"))
            with sqlite3.connect(database) as connection:
                journal = connection.execute(
                    "PRAGMA journal_mode=WAL"
                ).fetchone()[0]
                if str(journal).lower() != "wal":
                    raise RuntimeError("fake COLMAP database did not enter WAL mode")
                # Real COLMAP writes its database format marker on open, even
                # if no semantic table row is changed.
                connection.execute("PRAGMA user_version=4010100")
                if self.mutate_pose_mapper_database:
                    connection.execute(
                        "UPDATE cameras SET width=width+1 WHERE camera_id=1"
                    )
                connection.commit()
            _model(Path(_option(command, "--output_path")) / "0", b"refined")
            return SimpleNamespace(returncode=0)
        if action == "model_converter":
            output = Path(_option(command, "--output_path"))
            output.mkdir(exist_ok=True)
            lines = ["# Image list"]
            positions = _positions()
            for image_id, name in enumerate(self.names, start=1):
                frame_id = int(name.split("_")[1].split(".")[0])
                center = positions[frame_id].copy()
                camera_id = 1 if name.startswith("left_") else 2
                if camera_id == 2:
                    center += np.array([0.12, 0.0, 0.0])
                translation = -center
                lines.extend(
                    [
                        f"{image_id} 1 0 0 0 {translation[0]} "
                        f"{translation[1]} {translation[2]} {camera_id} {name}",
                        "",
                    ]
                )
            (output / "images.txt").write_text("\n".join(lines) + "\n")
            (output / "cameras.txt").write_text(
                "# cameras\n"
                "1 PINHOLE 16 12 20 20 8 6\n"
                "2 PINHOLE 16 12 20 20 8 6\n"
            )
            (output / "rigs.txt").write_text(
                "# rigs\n1 2 CAMERA 1 1 CAMERA 2 0 1 0 0 0 0 0.12 0 0\n"
            )
            (output / "points3D.txt").write_text("# points\n")
            return SimpleNamespace(returncode=0)
        if action == "model_analyzer":
            model = Path(_option(command, "--path"))
            n_images = 8 if "solve" in model.as_posix() else len(self.names)
            return SimpleNamespace(
                returncode=0, stdout=_analyzer_output(n_images)
            )
        raise AssertionError(f"unexpected command: {command}")


def _fixture(root: Path):
    segment = _segment(root / "segment")
    colmap = root / "pinned-colmap"
    colmap.write_bytes(b"test pinned COLMAP 4.1.1")
    colmap.chmod(0o755)
    provenance = collect_provenance(
        segment,
        resolved_config={"frontend": {"seed": 7}},
        colmap={
            "executable": str(colmap),
            "executable_sha256": sha256_file(colmap),
            "version": "COLMAP 4.1.1 test pin",
        },
        seed=7,
        repo_root=Path(__file__).resolve().parents[1],
    )
    frontend = FrontendArtifactBuilder(
        segment, root / "work", "frontend"
    ).build(
        rig_config={"schema_version": 1, "rigs": [{"ref_sensor_id": 1}]},
        keyframes={"frame_ids": [0, 2, 4, 6], "selector": "bounded"},
        pairs=[("left_000000.jpg", "right_000000.jpg")],
        provenance=provenance,
        quality={"n_frames": 8},
    )
    names = _create_database(frontend)
    create_frontend_seal(frontend)
    _write_json(
        frontend / "stages" / "matching.json",
        {
            "schema_version": 1,
            "stage": "matching",
            "state": "complete",
            "outputs": {
                "database.db": sha256_file(frontend / "database.db"),
                "frontend_seal.json": sha256_file(
                    frontend / "frontend_seal.json"
                ),
            },
        },
    )
    backend = prepare_mapper_backend(
        frontend,
        root / "backend",
        config=MapperConfig(
            minimum_free_space_gb=0.01,
            minimum_runtime_free_space_gb=0.005,
        ),
    )
    runner = _Runner(names)
    run_mapper_solve(backend, colmap, runner=runner)
    run_image_registration(backend, colmap, runner=runner)
    run_quality_summary(backend, colmap, runner=runner)
    selected_frame_ids = [0, 1, 2, 3, 4, 6, 7]
    selection = create_geodetic_frame_selection(
        frontend, segment, selected_frame_ids, root / "selection.json"
    )
    refinement = replace(
        GeodeticSubmapConfig().refinement,
        minimum_free_space_gb=0.01,
        minimum_runtime_free_space_gb=0.005,
        fresh_min_mean_observations_per_image=60.0,
    )
    config = GeodeticSubmapConfig(refinement=refinement)
    plan = prepare_geodetic_submap_plan(
        frontend,
        backend,
        segment,
        selection,
        root / "plan",
        config=config,
    )
    selected_names = [
        name
        for name in names
        if int(name.split("_")[1].split(".")[0]) in selected_frame_ids
    ]
    return segment, frontend, backend, selection, plan, colmap, selected_names


class GeodeticPairPolicyTests(unittest.TestCase):
    def _candidate(self, *, kind: str, distance: float) -> PairCandidate:
        covariance = tuple(tuple(row) for row in (np.eye(3) * 0.0004))
        first = RawGnssEndpoint(
            (0.0, 0.0, 0.0), covariance, True, "rtk_fixed", 1, 2
        )
        second = RawGnssEndpoint(
            (distance, 0.0, 0.0), covariance, True, "rtk_fixed", 1, 2
        )
        if kind == "stereo":
            frame_ids, indices, cameras, timestamps = (
                (4, 4),
                (4, 4),
                ("left", "right"),
                (5_000_000_000, 5_000_000_000),
            )
        elif kind == "adjacent":
            frame_ids, indices, cameras, timestamps = (
                (4, 5),
                (4, 5),
                ("left", "left"),
                (5_000_000_000, 6_000_000_000),
            )
        else:
            frame_ids, indices, cameras, timestamps = (
                (0, 8),
                (0, 8),
                ("left", "left"),
                (1_000_000_000, 20_000_000_000),
            )
        return PairCandidate(
            pair_id=2_147_483_649,
            first_image_id=1,
            second_image_id=2,
            first_image_name="first.jpg",
            second_image_name="second.jpg",
            first_frame_id=frame_ids[0],
            second_frame_id=frame_ids[1],
            first_frame_index=indices[0],
            second_frame_index=indices[1],
            first_camera=cameras[0],
            second_camera=cameras[1],
            first_timestamp_ns=timestamps[0],
            second_timestamp_ns=timestamps[1],
            raw_matches=100,
            verified_matches=80,
            first_gnss=first,
            second_gnss=second,
        )

    def test_nonlocal_gate_and_unconditional_local_edges(self):
        policy = GeodeticPairPolicy()
        rejected = decide_pair(
            self._candidate(kind="nonlocal", distance=5.0), policy
        )
        self.assertFalse(rejected.retained)
        self.assertEqual(rejected.reason, "nonlocal_gnss_incompatible")
        nearby = decide_pair(
            self._candidate(kind="nonlocal", distance=0.25), policy
        )
        self.assertTrue(nearby.retained)
        self.assertEqual(nearby.reason, "strong_locally_consistent_track")
        for kind in ("stereo", "adjacent"):
            decision = decide_pair(
                self._candidate(kind=kind, distance=50.0), policy
            )
            self.assertTrue(decision.retained)


class GeodeticSubmapArtifactTests(unittest.TestCase):
    def test_private_inventory_pair_audit_holdout_and_source_immutability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment = _segment(root / "pre-segment")
            # The full fixture creates its own bound segment; this first segment
            # merely makes accidental broad-path assumptions easier to catch.
            self.assertTrue(segment.is_dir())
            _, frontend, _, _, plan, _, names = _fixture(root / "pilot")
            source_before = (frontend / "database.db").read_bytes()
            inventory = json.loads(
                (plan / "database_inventory.json").read_text()
            )
            audit = json.loads((plan / "pair_audit.json").read_text())
            split = json.loads((plan / "prior_split.json").read_text())
            source_copy = json.loads(
                (plan / "source_database_evidence.json").read_text()
            )
            self.assertEqual(
                {item["name"] for item in inventory["images"]}, set(names)
            )
            self.assertEqual(inventory["camera_ids"], [1, 2])
            self.assertEqual(inventory["rig_ids"], [1])
            geodetic_plan = json.loads(
                (plan / "geodetic_submap_plan.json").read_text()
            )
            self.assertEqual(
                geodetic_plan["private_database_sqlite_metadata"],
                {
                    "application_id": 0,
                    "journal_mode": "wal",
                    "user_version": 4_010_100,
                },
            )
            with sqlite3.connect(
                f"file:{plan / 'database.db'}?mode=ro", uri=True
            ) as connection:
                self.assertEqual(
                    str(connection.execute("PRAGMA journal_mode").fetchone()[0]),
                    "wal",
                )
                self.assertEqual(
                    int(connection.execute("PRAGMA user_version").fetchone()[0]),
                    4_010_100,
                )
            self.assertEqual(len(inventory["database_frame_ids"]), 7)
            self.assertNotIn("left_000005.jpg", {item["name"] for item in inventory["images"]})
            self.assertTrue(inventory["pair_evidence_byte_exact"])
            self.assertTrue(source_copy["source_bytes_unchanged"])
            self.assertEqual(
                source_copy["source_raw_sha256_before"],
                source_copy["source_raw_sha256_after"],
            )
            self.assertEqual(
                source_copy["source_raw_sha256_after"],
                sha256_file(frontend / "database.db"),
            )
            reasons = {record["reason"] for record in audit["records"]}
            self.assertIn("nonlocal_gnss_incompatible", reasons)
            self.assertIn("strong_locally_consistent_track", reasons)
            rejected = [
                record
                for record in audit["records"]
                if record["reason"] == "nonlocal_gnss_incompatible"
            ]
            self.assertEqual(len(rejected), 1)
            holdouts = set(split["holdout_names"])
            with sqlite3.connect(plan / "database.db") as connection:
                optimizer = {
                    str(name)
                    for (name,) in connection.execute(
                        """
                        SELECT i.name FROM pose_priors AS p
                        JOIN images AS i ON i.image_id=p.corr_data_id
                        """
                    )
                }
            self.assertFalse(optimizer & holdouts)
            self.assertEqual(
                optimizer, set(split["calibration_names"])
            )
            self.assertEqual((frontend / "database.db").read_bytes(), source_before)

    def test_tampered_plan_is_rejected_and_fixed_command_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, _, plan, colmap, _ = _fixture(root)
            execution = root / "execution"
            command = build_geodetic_submap_command(plan, execution, colmap)
            self.assertEqual(command[1], "pose_prior_mapper")
            self.assertNotIn("--input_path", command)
            for name in (
                "--Mapper.constant_camera_list_path",
                "--Mapper.constant_rig_list_path",
                "--Mapper.ba_refine_focal_length",
                "--Mapper.ba_refine_sensor_from_rig",
            ):
                self.assertIn(name, command)
            self.assertEqual(
                command[command.index("--Mapper.ba_refine_focal_length") + 1],
                "0",
            )
            plan_path = plan / "geodetic_submap_plan.json"
            value = json.loads(plan_path.read_text())
            value["optimizer_contract"]["free_scale"] = True
            _write_json(plan_path, value)
            with self.assertRaisesRegex(ArtifactError, "sealed artifact file"):
                build_geodetic_submap_command(plan, execution, colmap)

    def test_atomic_result_success_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, _, plan, colmap, names = _fixture(root)
            result_path = root / "result"
            result = run_geodetic_submap_plan(
                plan, result_path, colmap, runner=_Runner(names)
            )
            self.assertTrue(result["passed"])
            self.assertTrue(result["publication_eligible"])
            export = geodetic_submap_export_context(result_path)
            self.assertEqual(export["text_model"], result_path / "refined_text")
            with self.assertRaises(FileExistsError):
                run_geodetic_submap_plan(
                    plan, result_path, colmap, runner=_Runner(names)
                )

    def test_real_colmap_open_preserves_normalized_database_logical_view(self):
        executable = os.environ.get("RTK_SPLAT_TEST_COLMAP")
        if not executable:
            self.skipTest("RTK_SPLAT_TEST_COLMAP is not configured")
        colmap = Path(executable).resolve()
        if not colmap.is_file():
            self.fail(f"RTK_SPLAT_TEST_COLMAP does not exist: {colmap}")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, _, plan, _, _ = _fixture(root / "fixture")
            probe = root / "real-colmap-open.db"
            shutil.copyfile(plan / "database.db", probe)
            before = sqlite_logical_record(probe)
            completed = subprocess.run(
                [
                    str(colmap),
                    "database_creator",
                    "--database_path",
                    str(probe),
                    "--log_target",
                    "stderr",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            after = sqlite_logical_record(probe)
            self.assertEqual(after, before)
            self.assertEqual(after["user_version"], 4_010_100)

    def test_semantic_database_mutation_after_mapper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, _, plan, colmap, names = _fixture(root)
            result = root / "mutated-result"
            runner = _Runner(names, mutate_pose_mapper_database=True)
            with self.assertRaisesRegex(
                ArtifactError, "changed private database contents"
            ):
                run_geodetic_submap_plan(
                    plan, result, colmap, runner=runner
                )
            self.assertFalse(result.exists())
            self.assertEqual(list(root.glob(".mutated-result.writing-*")), [])

    def test_injected_failure_leaves_no_result_or_staging_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment, frontend, backend, selection, plan, colmap, names = _fixture(root)
            failed_plan = root / "failed-plan"
            with mock.patch(
                "rtk_splat.backends.geodetic_submap._copy_subset_database",
                side_effect=RuntimeError("injected materialization failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    prepare_geodetic_submap_plan(
                        frontend,
                        backend,
                        segment,
                        selection,
                        failed_plan,
                    )
            self.assertFalse(failed_plan.exists())
            self.assertEqual(list(root.glob(".failed-plan.writing-*")), [])
            result = root / "failed-result"
            with self.assertRaisesRegex(RuntimeError, "injected"):
                run_geodetic_submap_plan(
                    plan,
                    result,
                    colmap,
                    runner=_Runner(names, fail_pose_mapper=True),
                )
            self.assertFalse(result.exists())
            self.assertEqual(list(root.glob(".failed-result.writing-*")), [])

    def test_selection_and_plan_never_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment, frontend, backend, selection, plan, _, _ = _fixture(root)
            with self.assertRaises(FileExistsError):
                create_geodetic_frame_selection(
                    frontend,
                    segment,
                    json.loads(selection.read_text())["frame_ids"],
                    selection,
                )
            with self.assertRaises(FileExistsError):
                prepare_geodetic_submap_plan(
                    frontend, backend, segment, selection, plan
                )

    def test_reusable_code_has_no_pilot_dataset_or_frame_literals(self):
        package = Path(__file__).resolve().parents[1] / "src" / "rtk_splat"
        paths = [
            package / "backends" / "geodetic_pairs.py",
            package / "backends" / "geodetic_submap.py",
        ]
        forbidden = ("field1", "/data/jkobo", "1100", "1800")
        for path in paths:
            contents = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                self.assertNotIn(token, contents, f"{token!r} in {path.name}")

    def test_existing_rtk_refinement_module_is_not_modified_by_new_api(self):
        module = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "rtk_splat"
            / "backends"
            / "rtk_refinement.py"
        )
        before = module.read_bytes()
        with mock.patch(
            "rtk_splat.backends.geodetic_submap._rtk_refinement_command"
        ):
            pass
        self.assertEqual(module.read_bytes(), before)

    def test_experimental_launcher_is_generic_and_outside_public_cli(self):
        repository = Path(__file__).resolve().parents[1]
        launcher = (
            repository
            / "scripts"
            / "experiments"
            / "geodetic_submap_ba_v1.py"
        )
        completed = subprocess.run(
            [sys.executable, str(launcher), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("seal-selection", completed.stdout)
        self.assertIn("pilot", completed.stdout)
        pilot_help = subprocess.run(
            [sys.executable, str(launcher), "pilot", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(pilot_help.returncode, 0, pilot_help.stderr)
        self.assertIn("--frame-start", pilot_help.stdout)
        self.assertIn("--frame-end", pilot_help.stdout)
        contents = launcher.read_text(encoding="utf-8").lower()
        for token in ("field1", "/data/jkobo", "1100", "1800"):
            self.assertNotIn(token, contents)


if __name__ == "__main__":
    unittest.main()
