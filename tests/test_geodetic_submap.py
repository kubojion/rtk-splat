import json
import os
import pickle
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

from rtk_splat.backends.geodetic_gnss import GeodeticGnssTemporalPolicy
from rtk_splat.backends.geodetic_pairs import (
    GeodeticPairPolicy,
    PairCandidate,
    RawGnssEndpoint,
    decide_pair,
)
from rtk_splat.backends.geodetic_submap import (
    GeodeticInitialPairPolicy,
    GeodeticSubmapConfig,
    GeodeticTrajectoryPolicy,
    _authoritative_checks_pass,
    _full_trajectory_quality,
    _metric_specific_source_repairs,
    _plan_context,
    _fixed_calibration_pose_priors,
    _resolved_private_prior_covariance,
    _resolved_private_prior_position,
    _sealed_evaluation_priors,
    _seal_files,
    _select_initial_pair_v2,
    _select_initial_pair_v3,
    _select_initial_pair_v5,
    _select_initial_pair_v6,
    _similarity_scale,
    audited_geodetic_submap_failure,
    build_geodetic_submap_command,
    create_geodetic_frame_selection,
    geodetic_submap_export_context,
    prepare_geodetic_submap_plan,
    run_geodetic_submap_plan,
)
from rtk_splat.backends.geodetic_assembly import (
    GeodeticAssemblyConfig,
    GeodeticDiagnosticPolicy,
    GeodeticOverlapPolicy,
    _aligned_submap_candidate,
    _assembly_evaluation_priors,
    _evaluate_overlap_candidates,
    _evaluate_overlap_candidates_v2,
    _file_evidence,
    _legacy_overlap_report,
    _overlap_csv,
    _overlap_svg,
    audited_geodetic_assembly_plan,
    audited_geodetic_diagnostic_full_pose_artifact,
    audited_geodetic_diagnostic_overlap_report,
    audited_geodetic_diagnostic_result_inventory,
    audited_geodetic_diagnostic_submap_result,
    audited_geodetic_diagnostic_initialization_fallback_window,
    audited_geodetic_diagnostic_low_parallax_fallback_window,
    audited_geodetic_full_pose_artifact,
    audited_geodetic_overlap_report,
    audited_geodetic_submap_pose_export,
    export_geodetic_submap_poses,
    prepare_geodetic_assembly_plan,
    prepare_geodetic_assembly_window,
    prepare_geodetic_diagnostic_initialization_fallback_window,
    prepare_geodetic_diagnostic_low_parallax_fallback_window,
    publish_geodetic_diagnostic_full_pose_artifact,
    publish_geodetic_diagnostic_overlap_report,
    publish_geodetic_diagnostic_result_inventory,
    publish_geodetic_full_pose_artifact,
    publish_geodetic_overlap_report,
)
from rtk_splat.backends.mapper import (
    MapperConfig,
    _apply_world_alignment,
    prepare_mapper_backend,
    run_image_registration,
    run_mapper_solve,
    run_quality_summary,
)
from rtk_splat.core.segment import (
    POSITION_QUALITY_VOCABULARY,
    SegmentReader,
    SegmentWriter,
)
from rtk_splat.frontends.artifact import (
    ArtifactError,
    FrontendArtifactBuilder,
    canonical_hash,
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


def _segment(path: Path, positions: np.ndarray | None = None) -> Path:
    writer = SegmentWriter(path)
    images = writer.directory("images")
    centers = _positions() if positions is None else np.asarray(positions)
    n = len(centers)
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
            "raw_timestamp_ns": timestamps,
            "raw_enu_m": centers,
            "raw_effective_covariance_enu_m2": np.repeat(
                (np.eye(3) * 0.0004)[None], n, axis=0
            ),
            "raw_fix_status": np.ones(n, dtype=np.int16),
            "raw_carrier_status": np.full(n, 2, dtype=np.int16),
        },
    )
    return writer.finalize().root


def _pair_id(first: int, second: int) -> int:
    low, high = sorted((first, second))
    return low * 2_147_483_647 + high


def _create_database(
    artifact: Path, positions: np.ndarray | None = None
) -> list[str]:
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
            INSERT INTO rigs VALUES(1, 1, 0);
            """
        )
        camera_parameters = np.asarray(
            [10.0, 10.0, 8.0, 6.0], dtype="<f8"
        ).tobytes()
        connection.execute(
            "INSERT INTO cameras VALUES(1, 1, 16, 12, ?, 1)",
            (camera_parameters,),
        )
        connection.execute(
            "INSERT INTO cameras VALUES(2, 1, 16, 12, ?, 1)",
            (camera_parameters,),
        )
        sensor_pose = np.asarray(
            [1.0, 0.0, 0.0, 0.0, 0.12, 0.0, 0.0], dtype="<f8"
        ).tobytes()
        connection.execute(
            "INSERT INTO rig_sensors VALUES(1, 2, 0, ?)", (sensor_pose,)
        )
        positions = _positions() if positions is None else np.asarray(positions)
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
                    (
                        image_id,
                        np.column_stack(
                            (
                                np.linspace(1.0, 14.0, 64),
                                np.linspace(1.0, 10.0, 64),
                            )
                        )
                        .astype("<f4")
                        .tobytes(),
                    ),
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
            if frame_index >= 2:
                previous = manifest["frames"][frame_index - 2]
                pairs.add((str(previous["left_image"]["name"]), left))
                pairs.add((str(previous["right_image"]["name"]), right))
        pairs.add((names[0], names[12]))  # nearby nonlocal revisit
        pairs.add((names[0], names[14]))  # false repetitive nonlocal pair
        qvec = np.asarray([1.0, 0.0, 0.0, 0.0], dtype="<f8").tobytes()
        tvec = np.asarray([1.0, 0.0, 0.1], dtype="<f8").tobytes()
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
                "(?, ?, 2, ?, 2, NULL, NULL, NULL, ?, ?)",
                (pair_id, count, data, qvec, tvec),
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
        positions: np.ndarray | None = None,
    ):
        self.names = names
        self.fail_pose_mapper = fail_pose_mapper
        self.mutate_pose_mapper_database = mutate_pose_mapper_database
        self.positions = _positions() if positions is None else np.asarray(positions)
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
            positions = self.positions
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


def _fixture(
    root: Path,
    *,
    positions: np.ndarray | None = None,
    config: GeodeticSubmapConfig | None = None,
    initial_pair_method: str = "v1",
):
    segment = _segment(root / "segment", positions)
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
    names = _create_database(frontend, positions)
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
    runner = _Runner(names, positions=positions)
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
    config = (
        GeodeticSubmapConfig(refinement=refinement)
        if config is None
        else replace(config, refinement=refinement)
    )
    plan = prepare_geodetic_submap_plan(
        frontend,
        backend,
        segment,
        selection,
        root / "plan",
        config=config,
        _initial_pair_method=initial_pair_method,
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

    def test_temporally_rejected_endpoint_is_untrusted_only_for_nonlocal_pair(self):
        rejected_endpoint = RawGnssEndpoint(
            (0.0, 0.0, 0.0),
            tuple(tuple(row) for row in (np.eye(3) * 0.0004)),
            False,
            "rtk_fixed",
            1,
            2,
        )
        nonlocal_candidate = replace(
            self._candidate(kind="nonlocal", distance=0.25),
            first_gnss=rejected_endpoint,
        )
        decision = decide_pair(nonlocal_candidate)
        self.assertFalse(decision.retained)
        self.assertEqual(decision.reason, "nonlocal_gnss_untrusted")
        for kind in ("stereo", "adjacent"):
            local = replace(
                self._candidate(kind=kind, distance=50.0),
                first_gnss=rejected_endpoint,
            )
            self.assertTrue(decide_pair(local).retained)

    def test_v2_seed_prefers_stable_two_view_geometry(self):
        high_matches = self._candidate(kind="adjacent", distance=0.25)
        high_matches = replace(
            high_matches,
            raw_matches=1200,
            verified_matches=1000,
        )
        stable = replace(
            high_matches,
            pair_id=6_442_450_945,
            first_image_id=3,
            second_image_id=4,
            first_image_name="stable_first.jpg",
            second_image_name="stable_second.jpg",
            first_frame_id=6,
            second_frame_id=7,
            first_frame_index=6,
            second_frame_index=7,
            first_timestamp_ns=7_000_000_000,
            second_timestamp_ns=8_000_000_000,
            raw_matches=400,
            verified_matches=300,
        )
        qvec = np.asarray([1.0, 0.0, 0.0, 0.0], dtype="<f8").tobytes()
        geometry = {
            high_matches.pair_id: {
                "configuration": 2,
                "qvec": qvec,
                "tvec": np.asarray([0.4, 0.0, 0.9], dtype="<f8").tobytes(),
            },
            stable.pair_id: {
                "configuration": 2,
                "qvec": qvec,
                "tvec": np.asarray([1.0, 0.0, 0.1], dtype="<f8").tobytes(),
            },
        }
        selected = _select_initial_pair_v2(
            [high_matches, stable],
            list(range(12)),
            [
                high_matches.first_image_name,
                high_matches.second_image_name,
                stable.first_image_name,
                stable.second_image_name,
            ],
            GeodeticSubmapConfig(),
            geometry,
        )
        self.assertEqual(selected["pair_id"], stable.pair_id)
        self.assertLess(
            selected["sealed_two_view_geometry"]["absolute_forward_motion"],
            0.2,
        )
        self.assertNotIn("finished_visual_model_pose", selected["score"])

    def test_v3_seed_prefers_sealed_correspondence_parallax(self):
        high_matches = replace(
            self._candidate(kind="adjacent", distance=0.25),
            raw_matches=1200,
            verified_matches=1000,
        )
        stable = replace(
            high_matches,
            pair_id=6_442_450_945,
            first_image_id=3,
            second_image_id=4,
            first_image_name="stable_first.jpg",
            second_image_name="stable_second.jpg",
            first_frame_id=6,
            second_frame_id=7,
            first_frame_index=6,
            second_frame_index=7,
            first_timestamp_ns=7_000_000_000,
            second_timestamp_ns=8_000_000_000,
            raw_matches=400,
            verified_matches=300,
        )
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "database.db"
            qvec = np.asarray(
                [1.0, 0.0, 0.0, 0.0], dtype="<f8"
            ).tobytes()
            geometry = {
                high_matches.pair_id: {
                    "configuration": 2,
                    "qvec": qvec,
                    "tvec": np.asarray(
                        [1.0, 0.0, 0.1], dtype="<f8"
                    ).tobytes(),
                },
                stable.pair_id: {
                    "configuration": 2,
                    "qvec": qvec,
                    "tvec": np.asarray(
                        [0.4, 0.0, 0.9], dtype="<f8"
                    ).tobytes(),
                },
            }
            with sqlite3.connect(database) as connection:
                connection.executescript(
                    """
                    CREATE TABLE cameras(
                      camera_id INTEGER PRIMARY KEY, model INTEGER,
                      width INTEGER, height INTEGER, params BLOB,
                      prior_focal_length INTEGER);
                    CREATE TABLE images(
                      image_id INTEGER PRIMARY KEY, name TEXT,
                      camera_id INTEGER);
                    CREATE TABLE keypoints(
                      image_id INTEGER PRIMARY KEY, rows INTEGER,
                      cols INTEGER, data BLOB);
                    CREATE TABLE two_view_geometries(
                      pair_id INTEGER PRIMARY KEY, rows INTEGER,
                      cols INTEGER, data BLOB, config INTEGER,
                      F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB);
                    """
                )
                parameters = np.asarray(
                    [10.0, 10.0, 8.0, 6.0], dtype="<f8"
                ).tobytes()
                connection.execute(
                    "INSERT INTO cameras VALUES(1, 1, 16, 12, ?, 1)",
                    (parameters,),
                )
                for candidate, second_x in (
                    (high_matches, 4.0),
                    (stable, 10.0),
                ):
                    count = int(candidate.verified_matches or 0)
                    for image_id, name, x in (
                        (
                            candidate.first_image_id,
                            candidate.first_image_name,
                            4.0,
                        ),
                        (
                            candidate.second_image_id,
                            candidate.second_image_name,
                            second_x,
                        ),
                    ):
                        points = np.tile(
                            np.asarray([[x, 6.0]], dtype="<f4"),
                            (count, 1),
                        )
                        connection.execute(
                            "INSERT INTO images VALUES(?, ?, 1)",
                            (image_id, name),
                        )
                        connection.execute(
                            "INSERT INTO keypoints VALUES(?, ?, 2, ?)",
                            (image_id, count, points.tobytes()),
                        )
                    matches = np.column_stack(
                        (
                            np.arange(count, dtype="<u4"),
                            np.arange(count, dtype="<u4"),
                        )
                    ).tobytes()
                    record = geometry[candidate.pair_id]
                    connection.execute(
                        "INSERT INTO two_view_geometries VALUES"
                        "(?, ?, 2, ?, 2, NULL, NULL, NULL, ?, ?)",
                        (
                            candidate.pair_id,
                            count,
                            matches,
                            record["qvec"],
                            record["tvec"],
                        ),
                    )
            pair_sources = {
                "_database_path": str(database),
                "initial_geometry_by_pair_id": geometry,
            }
            selected = _select_initial_pair_v3(
                [high_matches, stable],
                list(range(12)),
                [
                    high_matches.first_image_name,
                    high_matches.second_image_name,
                    stable.first_image_name,
                    stable.second_image_name,
                ],
                GeodeticSubmapConfig(),
                pair_sources,
            )
            anchored = _select_initial_pair_v5(
                [high_matches, stable],
                list(range(12)),
                [
                    high_matches.first_image_name,
                    high_matches.second_image_name,
                    stable.first_image_name,
                    stable.second_image_name,
                ],
                GeodeticSubmapConfig(),
                pair_sources,
            )
            low_parallax = _select_initial_pair_v6(
                [high_matches, stable],
                list(range(12)),
                [
                    high_matches.first_image_name,
                    high_matches.second_image_name,
                    stable.first_image_name,
                    stable.second_image_name,
                ],
                GeodeticSubmapConfig(),
                pair_sources,
            )
        self.assertEqual(selected["pair_id"], stable.pair_id)
        self.assertGreater(
            selected["sealed_parallax_evidence"]["median_angle_deg"], 20.0
        )
        self.assertNotIn("finished_visual_model_pose", selected["score"])
        self.assertEqual(anchored["image_ids"], [stable.first_image_id])
        self.assertEqual(
            anchored["anchor_position_prior_role"], "calibration"
        )
        self.assertTrue(anchored["anchor_position_prior_physically_present"])
        self.assertFalse(anchored["explicit_second_image_id"])
        self.assertEqual(
            anchored["source_pair_image_ids"],
            [stable.first_image_id, stable.second_image_id],
        )
        self.assertEqual(
            low_parallax["image_ids"],
            [stable.first_image_id, stable.second_image_id],
        )
        self.assertEqual(
            low_parallax["method"],
            "deterministic_calibration_pair_low_parallax_v6",
        )
        self.assertEqual(low_parallax["mapper_init_min_tri_angle_deg"], 1.0)
        self.assertTrue(
            low_parallax["calibration_position_priors_physically_present"]
        )
        self.assertFalse(
            low_parallax["heldout_position_priors_available_to_selector"]
        )


class GeodeticTrajectoryGateTests(unittest.TestCase):
    def _inputs(self, count: int = 40):
        index = np.arange(count, dtype=np.float64)
        prior = np.column_stack(
            (0.10 * index, 0.02 * np.sin(index / 3.0), 0.001 * index)
        )
        names = [f"image_{value:06d}.jpg" for value in range(count)]
        frame_ids = list(range(count))
        roles = ["calibration"] * count
        inliers = np.ones(count, dtype=bool)
        policy = GeodeticTrajectoryPolicy(local_window_frames=31)
        return prior, names, frame_ids, roles, inliers, policy

    def test_tail_jump_and_long_calibration_outlier_run_are_rejected(self):
        prior, names, frame_ids, roles, inliers, policy = self._inputs()
        refined = prior.copy()
        refined[-6:] += np.array([1.5, 0.0, 0.0])
        inliers[-6:] = False
        report = _full_trajectory_quality(
            refined,
            prior,
            prior,
            roles,
            inliers,
            names,
            frame_ids,
            policy,
        )
        checks = report["checks"]
        self.assertFalse(
            checks["consecutive_calibration_prior_outliers"]["passed"]
        )
        self.assertFalse(
            checks["adjacent_camera_prior_displacement_error_m"]["passed"]
        )
        self.assertFalse(checks["adjacent_raw_gnss_step_error_m"]["passed"])
        self.assertFalse(
            checks["sliding_local_window_path_consistency"]["passed"]
        )

    def test_sliding_window_rejects_accumulated_local_scale_distortion(self):
        prior, names, frame_ids, roles, inliers, policy = self._inputs()
        refined = prior * 1.20
        report = _full_trajectory_quality(
            refined,
            prior,
            prior,
            roles,
            inliers,
            names,
            frame_ids,
            policy,
        )
        checks = report["checks"]
        self.assertTrue(
            checks["adjacent_camera_prior_displacement_error_m"]["passed"]
        )
        self.assertFalse(
            checks["sliding_local_window_path_consistency"]["passed"]
        )

    def test_gnss_correct_repair_passes_despite_broken_source_scale(self):
        prior, names, frame_ids, roles, inliers, policy = self._inputs()
        report = _full_trajectory_quality(
            prior,
            prior,
            prior,
            roles,
            inliers,
            names,
            frame_ids,
            policy,
        )
        source = prior * 1.50
        source_to_repair_scale = _similarity_scale(source, prior)
        self.assertAlmostEqual(source_to_repair_scale, 2.0 / 3.0)
        self.assertGreater(abs(source_to_repair_scale - 1.0), 0.005)
        checks = {
            "trajectory_similarity_scale": {
                "authoritative": False,
                "passed": False,
            },
            **report["checks"],
        }
        self.assertTrue(_authoritative_checks_pass(checks))

    def test_temporally_rejected_observation_is_not_a_trajectory_gate_input(self):
        prior, names, frame_ids, roles, inliers, policy = self._inputs(80)
        rejected = 40
        roles[rejected] = "excluded"
        camera = prior.copy()
        raw = prior.copy()
        camera[rejected] = np.nan
        raw[rejected] = np.nan

        report = _full_trajectory_quality(
            prior,
            camera,
            raw,
            roles,
            inliers,
            names,
            frame_ids,
            policy,
        )

        self.assertTrue(_authoritative_checks_pass(report["checks"]))
        self.assertEqual(report["excluded_observation_count"], 1)
        self.assertEqual(
            report["checks"]["full_trajectory_raw_gnss_coverage"]["expected"],
            len(prior) - 1,
        )


class GeodeticConditionalRtkGateTests(unittest.TestCase):
    @staticmethod
    def _quality(
        *, inlier_fraction: float, inlier_passed: bool, median: float = 0.06
    ):
        return {
            "checks": {
                "median_holdout_rtk_residual_m": {
                    "authoritative": True,
                    "value": median,
                    "maximum": 0.10,
                    "passed": median <= 0.10,
                },
                "holdout_rtk_inlier_fraction": {
                    "authoritative": True,
                    "value": inlier_fraction,
                    "minimum": 0.80,
                    "passed": inlier_passed,
                },
                "diagnostic": {
                    "authoritative": False,
                    "value": 5.0,
                    "maximum": 1.0,
                    "passed": False,
                },
            }
        }

    def test_failed_inlier_fraction_is_repaired_by_same_metric(self):
        self.assertEqual(
            GeodeticSubmapConfig().refinement.max_holdout_median_regression_m,
            0.01,
        )
        source = self._quality(inlier_fraction=0.7375, inlier_passed=False)
        refined = self._quality(
            inlier_fraction=1.0, inlier_passed=True, median=0.065
        )

        repair = _metric_specific_source_repairs(source, refined)

        self.assertTrue(repair["passed"])
        self.assertEqual(
            repair["source_failed_metrics"], ["holdout_rtk_inlier_fraction"]
        )
        self.assertEqual(repair["repairs"][0]["direction"], "increase")

    def test_median_improvement_cannot_hide_unrepaired_source_metric(self):
        source = self._quality(
            inlier_fraction=0.7375, inlier_passed=False, median=0.09
        )
        refined = self._quality(
            inlier_fraction=0.75, inlier_passed=False, median=0.01
        )

        repair = _metric_specific_source_repairs(source, refined)

        self.assertFalse(repair["passed"])
        self.assertFalse(repair["repairs"][0]["refined_same_metric_passed"])


class GeodeticSubmapArtifactTests(unittest.TestCase):
    def test_fixed_calibration_updates_only_private_prior_target(self):
        source = np.asarray([1.0, 2.0, 3.0])
        corrected = np.asarray([1.04, 1.98, 3.01])
        with self.assertRaisesRegex(ArtifactError, "lever-arm-applied"):
            _resolved_private_prior_position(
                source, corrected, calibrated_derivation=False
            )

        resolved, replaced = _resolved_private_prior_position(
            source, corrected, calibrated_derivation=True
        )

        np.testing.assert_allclose(resolved, corrected)
        np.testing.assert_allclose(source, [1.0, 2.0, 3.0])
        self.assertTrue(replaced)

        source_covariance = np.diag([0.01, 0.01, 0.01])
        corrected_covariance = np.diag([0.0016, 0.01, 0.0016])
        with self.assertRaisesRegex(ArtifactError, "without a sealed"):
            _resolved_private_prior_covariance(
                source_covariance,
                corrected_covariance,
                calibrated_derivation=False,
            )
        resolved_covariance, covariance_replaced = (
            _resolved_private_prior_covariance(
                source_covariance,
                corrected_covariance,
                calibrated_derivation=True,
            )
        )
        np.testing.assert_array_equal(
            resolved_covariance, corrected_covariance
        )
        np.testing.assert_array_equal(
            source_covariance, np.diag([0.01, 0.01, 0.01])
        )
        self.assertTrue(covariance_replaced)

    def test_fixed_calibration_covariance_reuses_frontend_prior_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment = _segment(root / "segment")
            frontend = root / "frontend"
            (frontend / "stages").mkdir(parents=True)
            _write_json(
                frontend / "stages" / "pose_priors.json",
                {
                    "schema_version": 1,
                    "stage": "pose_priors",
                    "state": "complete",
                    "inputs": {
                        "covariance_floor_m": 0.001,
                        "max_covariance_m2": 1.0,
                        "min_fix_status": 1,
                        "min_carrier_status": 2,
                        "max_source_residual_s": 0.15,
                    },
                },
            )

            priors, audit = _fixed_calibration_pose_priors(
                frontend, SegmentReader(segment).validate()
            )

            self.assertEqual(len(priors), 8)
            self.assertEqual(
                audit["method"],
                "normal_frontend_prior_rederivation_on_fixed_segment_v1",
            )
            np.testing.assert_allclose(
                priors[0][1], np.eye(3) * 0.0005, atol=1.0e-12
            )

    def test_sealed_evaluation_prior_identity_and_resealed_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, _, plan, colmap, _ = _fixture(root)
            plan_record = json.loads(
                (plan / "geodetic_submap_plan.json").read_text()
            )
            split_path = plan / "prior_split.json"
            split = json.loads(split_path.read_text())
            priors = _sealed_evaluation_priors(
                plan,
                plan_record,
                set(plan_record["selected_left_image_names"]),
            )
            expected_names = set(split["calibration_names"]) | set(
                split["holdout_names"]
            )
            self.assertEqual(set(priors), expected_names)
            with sqlite3.connect(
                f"file:{plan / 'database.db'}?mode=ro&immutable=1", uri=True
            ) as connection:
                optimizer_names = {
                    str(name)
                    for (name,) in connection.execute(
                        "SELECT i.name FROM pose_priors p "
                        "JOIN images i ON i.image_id=p.corr_data_id"
                    )
                }
            self.assertEqual(optimizer_names, set(split["calibration_names"]))
            self.assertFalse(optimizer_names & set(split["holdout_names"]))

            holdout = next(
                record
                for record in split["records"]
                if record.get("role") == "holdout"
            )
            holdout["position_m"][0] += 1.0
            _write_json(split_path, split)
            plan_path = plan / "geodetic_submap_plan.json"
            plan_record["prior_split_sha256"] = sha256_file(split_path)
            _write_json(plan_path, plan_record)
            seal_path = plan / "plan_seal.json"
            seal = json.loads(seal_path.read_text())
            for path in (split_path, plan_path):
                seal["files"][path.name] = {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            seal["seal_sha256"] = canonical_hash(
                {
                    "schema_version": seal["schema_version"],
                    "files": seal["files"],
                }
            )
            _write_json(seal_path, seal)

            with self.assertRaisesRegex(
                ArtifactError, "evaluation prior evidence changed"
            ):
                build_geodetic_submap_command(
                    plan, root / "execution", colmap
                )

    def test_predeclared_window_roles_are_sealed_physical_and_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (
                segment,
                frontend,
                backend,
                selection,
                default_plan,
                colmap,
                selected_names,
            ) = _fixture(root)
            role_plan = prepare_geodetic_submap_plan(
                frontend,
                backend,
                segment,
                selection,
                root / "role-plan",
                config=_plan_context(default_plan)[2],
                _initial_pair_method="v4",
                _predeclared_prior_role_windows=(
                    (0, 1, 2, 3, 4),
                    (2, 3, 4, 6, 7),
                ),
            )
            _plan_context(role_plan, require_hardened=True)
            split_path = role_plan / "prior_split.json"
            split = json.loads(split_path.read_text())
            contract = split["predeclared_prior_role_contract"]
            self.assertEqual(
                contract["method"],
                "calibration_in_any_window_holdout_in_every_window_v1",
            )
            self.assertEqual(split["n_calibration"], 4)
            self.assertEqual(split["n_holdout"], 3)
            by_frame = {item["frame_id"]: item for item in split["records"]}
            self.assertEqual(by_frame[2]["role"], "calibration")
            self.assertEqual(
                [
                    item["role"]
                    for item in by_frame[2][
                        "predeclared_local_role_evidence"
                    ]
                ],
                ["calibration", "calibration"],
            )
            self.assertEqual(by_frame[3]["role"], "holdout")
            self.assertEqual(
                [
                    item["role"]
                    for item in by_frame[3][
                        "predeclared_local_role_evidence"
                    ]
                ],
                ["holdout", "holdout"],
            )
            with sqlite3.connect(
                f"file:{role_plan / 'database.db'}?mode=ro&immutable=1",
                uri=True,
            ) as connection:
                optimizer_names = {
                    str(name)
                    for (name,) in connection.execute(
                        "SELECT i.name FROM pose_priors p "
                        "JOIN images i ON i.image_id=p.corr_data_id"
                    )
                }
            self.assertEqual(optimizer_names, set(split["calibration_names"]))
            self.assertFalse(optimizer_names & set(split["holdout_names"]))

            with mock.patch(
                "rtk_splat.backends.geodetic_submap._heldout_evaluation",
                side_effect=AssertionError(
                    "predeclared roles must not derive a second split"
                ),
            ):
                result = run_geodetic_submap_plan(
                    role_plan,
                    root / "role-result",
                    colmap,
                    runner=_Runner(selected_names),
                )
            self.assertTrue(result["passed"])
            self.assertEqual(
                result["quality"]["rtk_holdout"]["n_calibration"], 4
            )
            self.assertEqual(
                result["quality"]["rtk_holdout"]["n_holdout"], 3
            )
            self.assertEqual(
                result["quality"]["rtk_holdout"][
                    "evaluation_role_method"
                ],
                contract["method"],
            )

            contract["role_membership_sha256"] = "0" * 64
            _write_json(split_path, split)
            plan_path = role_plan / "geodetic_submap_plan.json"
            plan_record = json.loads(plan_path.read_text())
            plan_record["prior_split_sha256"] = sha256_file(split_path)
            _write_json(plan_path, plan_record)
            seal_path = role_plan / "plan_seal.json"
            seal = json.loads(seal_path.read_text())
            for path in (split_path, plan_path):
                seal["files"][path.name] = {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            seal["seal_sha256"] = canonical_hash(
                {
                    "schema_version": seal["schema_version"],
                    "files": seal["files"],
                }
            )
            _write_json(seal_path, seal)
            with self.assertRaisesRegex(
                ArtifactError, "evaluation prior derivation changed"
            ):
                build_geodetic_submap_command(
                    role_plan, root / "role-execution", colmap
                )

    def test_temporal_rejection_is_physically_absent_and_resealed_tamper_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            positions = np.column_stack(
                (
                    np.arange(8, dtype=np.float64) * 0.05,
                    np.zeros(8),
                    np.zeros(8),
                )
            )
            positions[3, 0] += 0.18
            policy = GeodeticGnssTemporalPolicy(
                support_epochs=2,
                maximum_acceleration_m_s2=0.01,
                maximum_raw_interval_s=2.0,
                maximum_rejected_raw_fraction=0.20,
            )
            _, _, _, _, plan, colmap, _ = _fixture(
                root,
                positions=positions,
                config=GeodeticSubmapConfig(
                    gnss_temporal_policy=policy,
                    initial_pair_policy=GeodeticInitialPairPolicy(
                        minimum_raw_gnss_displacement_m=0.04
                    ),
                ),
                initial_pair_method="v4",
            )
            split_path = plan / "prior_split.json"
            split = json.loads(split_path.read_text())
            audit = split["gnss_temporal_audit"]
            rejected = [
                item
                for item in audit["observation_records"]
                if not item["retained_by_temporal_filter"]
            ]
            self.assertEqual([item["frame_id"] for item in rejected], [3])
            record = next(
                item for item in split["records"] if item["frame_id"] == 3
            )
            self.assertEqual(record["role"], "excluded")
            self.assertEqual(
                record["reason"],
                "raw_gnss_paired_discontinuity_excursion",
            )
            with sqlite3.connect(
                f"file:{plan / 'database.db'}?mode=ro&immutable=1", uri=True
            ) as connection:
                prior_names = {
                    str(name)
                    for (name,) in connection.execute(
                        """
                        SELECT i.name FROM pose_priors AS p
                        JOIN images AS i ON i.image_id=p.corr_data_id
                        """
                    )
                }
            self.assertNotIn("left_000003.jpg", prior_names)
            self.assertIn("left_000003.jpg", (plan / "all_images.txt").read_text())

            rejected[0]["retained_by_temporal_filter"] = True
            rejected[0]["reason"] = "raw_gnss_temporally_consistent"
            _write_json(split_path, split)
            plan_path = plan / "geodetic_submap_plan.json"
            plan_record = json.loads(plan_path.read_text())
            plan_record["prior_split_sha256"] = sha256_file(split_path)
            plan_record["gnss_temporal_audit_sha256"] = canonical_hash(audit)
            _write_json(plan_path, plan_record)
            seal_path = plan / "plan_seal.json"
            seal = json.loads(seal_path.read_text())
            for path in (split_path, plan_path):
                seal["files"][path.name] = {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            seal_body = {
                "schema_version": seal["schema_version"],
                "files": seal["files"],
            }
            seal["seal_sha256"] = canonical_hash(seal_body)
            _write_json(seal_path, seal)

            with self.assertRaisesRegex(
                ArtifactError, "raw-GNSS temporal audit changed"
            ):
                build_geodetic_submap_command(plan, root / "execution", colmap)

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

    def test_initial_pair_is_deterministic_sealed_and_bound_to_mapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment, frontend, backend, selection, plan, colmap, _ = _fixture(
                root
            )
            first = json.loads(
                (plan / "geodetic_submap_plan.json").read_text()
            )["initial_pair"]
            second_plan = prepare_geodetic_submap_plan(
                frontend,
                backend,
                segment,
                selection,
                root / "second-plan",
                config=GeodeticSubmapConfig(
                    refinement=replace(
                        GeodeticSubmapConfig().refinement,
                        minimum_free_space_gb=0.01,
                        minimum_runtime_free_space_gb=0.005,
                        fresh_min_mean_observations_per_image=60.0,
                    )
                ),
            )
            second = json.loads(
                (second_plan / "geodetic_submap_plan.json").read_text()
            )["initial_pair"]
            self.assertEqual(first, second)
            self.assertGreaterEqual(
                first["actual_boundary_margin_frames"],
                first["required_boundary_margin_frames"],
            )
            self.assertGreaterEqual(first["verified_matches"], 30)
            self.assertGreaterEqual(
                first["raw_gnss_evidence"]["raw_gnss_displacement_m"],
                first["raw_gnss_evidence"][
                    "minimum_raw_gnss_displacement_m"
                ],
            )
            command = build_geodetic_submap_command(
                plan, root / "execution", colmap
            )
            self.assertEqual(
                _option(command, "--Mapper.init_image_id1"),
                str(first["image_ids"][0]),
            )
            self.assertEqual(
                _option(command, "--Mapper.init_image_id2"),
                str(first["image_ids"][1]),
            )

    def test_resealed_initial_pair_tamper_is_recomputed_and_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, _, plan, colmap, _ = _fixture(root)
            plan_path = plan / "geodetic_submap_plan.json"
            value = json.loads(plan_path.read_text())
            value["initial_pair"]["image_ids"][0] += 1
            _write_json(plan_path, value)
            seal_path = plan / "plan_seal.json"
            seal = json.loads(seal_path.read_text())
            seal["files"][plan_path.name] = {
                "sha256": sha256_file(plan_path),
                "size_bytes": plan_path.stat().st_size,
            }
            body = {
                "schema_version": seal["schema_version"],
                "files": seal["files"],
            }
            seal["seal_sha256"] = canonical_hash(body)
            _write_json(seal_path, seal)
            with self.assertRaisesRegex(ArtifactError, "initial-pair contract"):
                build_geodetic_submap_command(
                    plan, root / "execution", colmap
                )

    def test_atomic_result_success_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, _, plan, colmap, names = _fixture(root)
            result_path = root / "result"
            with mock.patch(
                "rtk_splat.backends.geodetic_submap._plan_context",
                wraps=_plan_context,
            ) as audited_plan_context:
                result = run_geodetic_submap_plan(
                    plan, result_path, colmap, runner=_Runner(names)
                )
            # One audit precedes execution and one closes the mapper mutation
            # race.  Both the command builder and final publication verifier
            # reuse those audited contexts instead of adding full source-
            # database snapshots.
            self.assertEqual(audited_plan_context.call_count, 2)
            self.assertTrue(result["passed"])
            self.assertTrue(result["publication_eligible"])
            self.assertTrue(
                result["quality"]["checks"]["heldout_rtk_absolute_gates"][
                    "passed"
                ]
            )
            self.assertFalse(
                result["quality"]["checks"]["trajectory_similarity_scale"][
                    "authoritative"
                ]
            )
            for name in (
                "full_trajectory_raw_gnss_coverage",
                "consecutive_calibration_prior_outliers",
                "adjacent_camera_prior_displacement_error_m",
                "adjacent_raw_gnss_step_error_m",
                "sliding_local_window_path_consistency",
            ):
                self.assertTrue(result["quality"]["checks"][name]["passed"])
            self.assertEqual(
                result["solve"]["sealed_initial_pair"]["image_ids"],
                json.loads(
                    (plan / "geodetic_submap_plan.json").read_text()
                )["initial_pair"]["image_ids"],
            )
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

    def test_mapper_failure_evidence_is_sealed_classified_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, _, plan, colmap, _ = _fixture(root)
            result = root / "failed-result"
            failure = root / "failure-evidence"

            def fail_without_sparse_model(command, attempt, _config):
                (attempt / "colmap.log").write_text(
                    "Finding good initial image pair\n"
                    "No good initial image pair found.\n"
                    "Failed to create any sparse model\n",
                    encoding="utf-8",
                )
                (attempt / "resource_samples.csv").write_text(
                    "elapsed_s,pid\n0.0,1\n", encoding="utf-8"
                )
                _write_json(
                    attempt / "resource_usage.json",
                    {
                        "schema_version": 1,
                        "returncode": 1,
                        "safety_aborted": False,
                    },
                )
                raise subprocess.CalledProcessError(1, list(command))

            with mock.patch(
                "rtk_splat.backends.geodetic_submap._run_monitored_mapper",
                side_effect=fail_without_sparse_model,
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_geodetic_submap_plan(
                        plan,
                        result,
                        colmap,
                        failure_destination=failure,
                    )
            self.assertFalse(result.exists())
            self.assertEqual(list(root.glob(".failed-result.writing-*")), [])
            audited = audited_geodetic_submap_failure(failure)
            self.assertEqual(
                audited["failure_classification"],
                "initialization_exhausted_without_sparse_model",
            )
            self.assertFalse(audited["partial_result_published"])
            self.assertTrue(
                (failure / "attempts" / "solve" / "attempt-0001" / "colmap.log").is_file()
            )
            with mock.patch(
                "rtk_splat.backends.geodetic_submap._run_monitored_mapper"
            ) as mapper:
                with self.assertRaises(FileExistsError):
                    run_geodetic_submap_plan(
                        plan,
                        result,
                        colmap,
                        failure_destination=failure,
                    )
                mapper.assert_not_called()

            record_path = failure / "geodetic_submap_failure.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["failure_classification"] = "mapper_execution_failure"
            _write_json(record_path, record)
            names = sorted(
                path.relative_to(failure).as_posix()
                for path in failure.rglob("*")
                if path.is_file() and path.name != "failure_seal.json"
            )
            _write_json(failure / "failure_seal.json", _seal_files(failure, names))
            with self.assertRaisesRegex(ArtifactError, "binding changed"):
                audited_geodetic_submap_failure(failure)

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
            package / "backends" / "geodetic_gnss.py",
            package / "backends" / "geodetic_pairs.py",
            package / "backends" / "geodetic_submap.py",
            package / "backends" / "geodetic_assembly.py",
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

        continuous = (
            repository
            / "scripts"
            / "experiments"
            / "geodetic_continuous_boundary_v1.py"
        )
        continuous_help = subprocess.run(
            [sys.executable, str(continuous), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            continuous_help.returncode, 0, continuous_help.stderr
        )
        self.assertIn("prepare", continuous_help.stdout)
        self.assertIn("run", continuous_help.stdout)
        self.assertIn("audit-calibration", continuous_help.stdout)
        contents = continuous.read_text(encoding="utf-8").lower()
        for token in ("field1", "/data/jkobo", "1100", "1800"):
            self.assertNotIn(token, contents)


class GeodeticAssemblyArtifactTests(unittest.TestCase):
    def _completed_assembly(self, root: Path):
        segment, frontend, backend, _, source_plan, colmap, _ = _fixture(root)
        selection = create_geodetic_frame_selection(
            frontend,
            segment,
            list(range(8)),
            root / "assembly-selection.json",
        )
        submap_config = _plan_context(source_plan)[2]
        config = GeodeticAssemblyConfig(
            submap_frames=6,
            overlap_frames=2,
            probe_submaps=2,
            overlap_policy=GeodeticOverlapPolicy(
                minimum_overlap_frames=2,
                max_median_center_disagreement_m=0.01,
                max_p95_center_disagreement_m=0.01,
                max_center_disagreement_m=0.01,
                max_median_rotation_disagreement_deg=0.01,
                max_p95_rotation_disagreement_deg=0.01,
                max_rotation_disagreement_deg=0.01,
            ),
            submap_config=submap_config,
        )
        assembly_plan = prepare_geodetic_assembly_plan(
            frontend,
            backend,
            segment,
            selection,
            root / "assembly-plan",
            config=config,
        )
        audited = audited_geodetic_assembly_plan(assembly_plan)
        runtime = audited_geodetic_assembly_plan(
            assembly_plan, _include_runtime_context=True
        )
        # run-all uses spawned worker processes because the mapper's resource
        # monitor owns process-level signal handlers.
        pickle.loads(pickle.dumps(runtime["_runtime_context"]))
        self.assertEqual(audited["schema_version"], 6)
        self.assertEqual(len(audited["windows"]), 2)
        self.assertEqual(
            sorted(
                {
                    frame_id
                    for window in audited["windows"]
                    for frame_id in window["frame_ids"]
                }
            ),
            list(range(8)),
        )
        submaps = root / "submaps"
        results = []
        for window in audited["windows"]:
            prepared = prepare_geodetic_assembly_window(
                assembly_plan,
                window["window_id"],
                submaps / window["window_id"],
            )
            local_plan = json.loads(
                (Path(prepared["plan"]) / "geodetic_submap_plan.json").read_text()
            )
            command = build_geodetic_submap_command(
                prepared["plan"], root / "command-audit", colmap
            )
            self.assertEqual(
                _option(command, "--Mapper.init_image_id1"),
                str(local_plan["initial_pair"]["image_ids"][0]),
            )
            self.assertNotIn("--Mapper.init_image_id2", command)
            self.assertNotIn("--Mapper.init_min_tri_angle", command)
            self.assertEqual(
                local_plan["initial_pair"]["method"],
                "deterministic_calibration_anchor_colmap_partner_v5",
            )
            self.assertTrue(
                local_plan["initial_pair"][
                    "anchor_position_prior_physically_present"
                ]
            )
            self.assertIn(
                local_plan["initial_pair"]["image_names"][0],
                window["calibration_names"],
            )
            database_uri = (
                f"file:{Path(prepared['plan']) / 'database.db'}"
                "?mode=ro&immutable=1"
            )
            with sqlite3.connect(database_uri, uri=True) as db:
                self.assertEqual(
                    db.execute(
                        "SELECT COUNT(*) FROM pose_priors WHERE corr_data_id=?",
                        (local_plan["initial_pair"]["image_ids"][0],),
                    ).fetchone()[0],
                    1,
                )
            runner = _Runner(list(local_plan["selected_image_names"]))
            result = run_geodetic_submap_plan(
                prepared["plan"], prepared["result"], colmap, runner=runner
            )
            self.assertTrue(result["passed"])
            results.append(Path(prepared["result"]))
        return segment, assembly_plan, audited, submaps, results

    def _copy_result_with_failed_checks(
        self,
        source: Path,
        destination_workspace: Path,
        failed_checks: tuple[str, ...],
    ) -> Path:
        destination_workspace.mkdir(parents=True)
        shutil.copy2(
            source.parent / "window_binding.json",
            destination_workspace / "window_binding.json",
        )
        destination = destination_workspace / "result"
        shutil.copytree(source, destination)
        quality_path = destination / "reports" / "quality.json"
        quality = json.loads(quality_path.read_text())
        for name in failed_checks:
            self.assertIn(name, quality["checks"])
            quality["checks"][name]["passed"] = False
        quality["passed"] = False
        quality["publication_eligible"] = False
        _write_json(quality_path, quality)
        result_path = destination / "geodetic_submap_result.json"
        result = json.loads(result_path.read_text())
        result["passed"] = False
        result["publication_eligible"] = False
        result["quality_report_sha256"] = sha256_file(quality_path)
        _write_json(result_path, result)
        files = sorted(
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file() and path.name != "result_seal.json"
        )
        _write_json(destination / "result_seal.json", _seal_files(destination, files))
        return destination

    def test_diagnostic_inventory_overlap_and_pose_preserve_rtk_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment, plan, audited, _, results = self._completed_assembly(root)
            failed = self._copy_result_with_failed_checks(
                results[1],
                root / "failed-window",
                ("heldout_rtk_absolute_gates",),
            )
            with self.assertRaisesRegex(ArtifactError, "acceptance gates"):
                _aligned_submap_candidate(failed)
            policy = GeodeticDiagnosticPolicy(median_rtk_warning_m=1.0e-6)
            result_by_window = {
                audited["windows"][0]["window_id"]: results[0],
                audited["windows"][1]["window_id"]: failed,
            }
            inventory = publish_geodetic_diagnostic_result_inventory(
                plan,
                result_by_window,
                root / "diagnostic-inventory",
                policy=policy,
            )
            record = audited_geodetic_diagnostic_result_inventory(inventory)
            self.assertEqual(
                record["production_failed_window_ids"],
                [audited["windows"][1]["window_id"]],
            )
            self.assertIsInstance(record["warning_exceeded_window_ids"], list)
            overlap = publish_geodetic_diagnostic_overlap_report(
                [results[0], failed],
                root / "diagnostic-overlap",
                overlap_policy=audited["config_object"].overlap_policy,
                diagnostic_policy=policy,
            )
            overlap_record = audited_geodetic_diagnostic_overlap_report(overlap)
            self.assertFalse(overlap_record["production_passed"])
            self.assertTrue(
                overlap_record["structurally_safe_for_diagnostic"]
            )
            pose = publish_geodetic_diagnostic_full_pose_artifact(
                inventory, root / "poses", "diagnostic-assembled"
            )
            final = audited_geodetic_diagnostic_full_pose_artifact(pose)
            self.assertEqual(final["manifest"]["n_frames"], 8)
            self.assertEqual(
                final["georeferencing"]["artifact_class"],
                "diagnostic_render_only",
            )
            self.assertEqual(
                final["georeferencing"]["georeferencing_status"], "FAILED"
            )
            self.assertFalse(
                final["georeferencing"][
                    "metric_georeferencing_claim_eligible"
                ]
            )
            self.assertTrue((pose / "GEOREFERENCING_FAILED.json").is_file())
            cfg = SimpleNamespace(
                pose=SimpleNamespace(
                    artifact="diagnostic-assembled", artifact_root=root / "poses"
                )
            )
            from rtk_splat.core.pose_artifacts import load_pose_artifact

            with self.assertRaisesRegex(ValueError, "explicit render-only"):
                load_pose_artifact(segment, cfg)
            loaded, _ = load_pose_artifact(
                segment, cfg, allow_failed_georeferencing_for_render=True
            )
            self.assertEqual(loaded.shape, (8, 4, 4))
            with self.assertRaises(FileExistsError):
                publish_geodetic_diagnostic_full_pose_artifact(
                    inventory, root / "poses", "diagnostic-assembled"
                )

    def test_diagnostic_auto_initialization_fallback_is_sealed_and_propagated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, assembly_plan, audited, _, results = self._completed_assembly(root)
            window = audited["windows"][0]
            primary = prepare_geodetic_assembly_window(
                assembly_plan,
                window["window_id"],
                root / "primary-failed-window",
            )
            primary_plan = json.loads(
                (
                    Path(primary["plan"]) / "geodetic_submap_plan.json"
                ).read_text(encoding="utf-8")
            )
            colmap = Path(primary_plan["colmap"]["executable"])
            failure_path = Path(primary["workspace"]) / "failure"

            def fail_without_sparse_model(command, attempt, _config):
                (attempt / "colmap.log").write_text(
                    "Finding good initial image pair\n"
                    "No good initial image pair found.\n"
                    "Failed to create any sparse model\n",
                    encoding="utf-8",
                )
                (attempt / "resource_samples.csv").write_text(
                    "elapsed_s,pid\n0.0,1\n", encoding="utf-8"
                )
                _write_json(
                    attempt / "resource_usage.json",
                    {
                        "schema_version": 1,
                        "returncode": 1,
                        "safety_aborted": False,
                    },
                )
                raise subprocess.CalledProcessError(1, list(command))

            with mock.patch(
                "rtk_splat.backends.geodetic_submap._run_monitored_mapper",
                side_effect=fail_without_sparse_model,
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_geodetic_submap_plan(
                        primary["plan"],
                        primary["result"],
                        colmap,
                        failure_destination=failure_path,
                    )
            fallback = prepare_geodetic_diagnostic_initialization_fallback_window(
                assembly_plan,
                window["window_id"],
                failure_path,
                root / "fallback-window",
            )
            verified = audited_geodetic_diagnostic_initialization_fallback_window(
                assembly_plan, fallback["workspace"]
            )
            self.assertEqual(verified, fallback)
            fallback_plan = json.loads(
                (Path(fallback["plan"]) / "geodetic_submap_plan.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                fallback_plan["initial_pair"]["method"],
                "deterministic_colmap_auto_filtered_database_v4",
            )
            command = build_geodetic_submap_command(
                fallback["plan"], root / "fallback-command", colmap
            )
            self.assertNotIn("--Mapper.init_image_id1", command)
            self.assertNotIn("--Mapper.init_image_id2", command)
            runner = _Runner(list(fallback_plan["selected_image_names"]))
            run_geodetic_submap_plan(
                fallback["plan"],
                fallback["result"],
                colmap,
                runner=runner,
            )
            with self.assertRaisesRegex(
                ArtifactError, "diagnostic_auto_initialization_fallback"
            ):
                audited_geodetic_diagnostic_submap_result(fallback["result"])
            policy = GeodeticDiagnosticPolicy(
                allow_sealed_auto_initialization_fallback=True
            )
            assessment = audited_geodetic_diagnostic_submap_result(
                fallback["result"], policy=policy
            )
            self.assertFalse(assessment["production_passed"])
            self.assertTrue(assessment["structurally_safe_for_diagnostic"])
            self.assertEqual(
                assessment["allowed_diagnostic_execution_overrides"],
                ["diagnostic_auto_initialization_fallback"],
            )
            inventory = publish_geodetic_diagnostic_result_inventory(
                assembly_plan,
                {
                    window["window_id"]: fallback["result"],
                    audited["windows"][1]["window_id"]: results[1],
                },
                root / "fallback-inventory",
                policy=policy,
            )
            inventory_record = audited_geodetic_diagnostic_result_inventory(
                inventory
            )
            self.assertEqual(
                inventory_record[
                    "diagnostic_initialization_override_window_ids"
                ],
                [window["window_id"]],
            )
            self.assertIn(
                window["window_id"],
                inventory_record["production_failed_window_ids"],
            )

            binding_path = Path(fallback["workspace"]) / "window_binding.json"
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
            binding["fallback_reason"] = "tampered"
            _write_json(binding_path, binding)
            with self.assertRaisesRegex(ArtifactError, "binding changed"):
                audited_geodetic_diagnostic_initialization_fallback_window(
                    assembly_plan, fallback["workspace"]
                )

    def test_diagnostic_low_parallax_fallback_is_calibration_only_and_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, assembly_plan, audited, _, results = self._completed_assembly(root)
            window = audited["windows"][0]
            primary = prepare_geodetic_assembly_window(
                assembly_plan,
                window["window_id"],
                root / "low-parallax-primary-failed-window",
            )
            primary_plan = json.loads(
                (
                    Path(primary["plan"]) / "geodetic_submap_plan.json"
                ).read_text(encoding="utf-8")
            )
            colmap = Path(primary_plan["colmap"]["executable"])
            failure_path = Path(primary["workspace"]) / "failure"

            def fail_without_sparse_model(command, attempt, _config):
                (attempt / "colmap.log").write_text(
                    "Finding good initial image pair\n"
                    "No good initial image pair found.\n"
                    "Failed to create any sparse model\n",
                    encoding="utf-8",
                )
                (attempt / "resource_samples.csv").write_text(
                    "elapsed_s,pid\n0.0,1\n", encoding="utf-8"
                )
                _write_json(
                    attempt / "resource_usage.json",
                    {
                        "schema_version": 1,
                        "returncode": 1,
                        "safety_aborted": False,
                    },
                )
                raise subprocess.CalledProcessError(1, list(command))

            with mock.patch(
                "rtk_splat.backends.geodetic_submap._run_monitored_mapper",
                side_effect=fail_without_sparse_model,
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_geodetic_submap_plan(
                        primary["plan"],
                        primary["result"],
                        colmap,
                        failure_destination=failure_path,
                    )
            fallback = (
                prepare_geodetic_diagnostic_low_parallax_fallback_window(
                    assembly_plan,
                    window["window_id"],
                    failure_path,
                    root / "low-parallax-fallback-window",
                )
            )
            verified = (
                audited_geodetic_diagnostic_low_parallax_fallback_window(
                    assembly_plan, fallback["workspace"]
                )
            )
            self.assertEqual(verified, fallback)
            fallback_plan = json.loads(
                (Path(fallback["plan"]) / "geodetic_submap_plan.json").read_text(
                    encoding="utf-8"
                )
            )
            initial = fallback_plan["initial_pair"]
            self.assertEqual(
                initial["method"],
                "deterministic_calibration_pair_low_parallax_v6",
            )
            self.assertEqual(initial["mapper_init_min_tri_angle_deg"], 1.0)
            self.assertTrue(
                set(initial["image_names"])
                <= set(window["calibration_names"])
            )
            self.assertFalse(
                set(initial["image_names"]) & set(window["holdout_names"])
            )
            command = build_geodetic_submap_command(
                fallback["plan"], root / "low-parallax-command", colmap
            )
            self.assertEqual(
                command[command.index("--Mapper.init_min_tri_angle") + 1],
                "1.0",
            )
            self.assertEqual(
                command[command.index("--Mapper.init_image_id1") + 1],
                str(initial["image_ids"][0]),
            )
            self.assertEqual(
                command[command.index("--Mapper.init_image_id2") + 1],
                str(initial["image_ids"][1]),
            )
            runner = _Runner(list(fallback_plan["selected_image_names"]))
            run_geodetic_submap_plan(
                fallback["plan"],
                fallback["result"],
                colmap,
                runner=runner,
            )
            with self.assertRaisesRegex(
                ArtifactError,
                "diagnostic_low_parallax_initialization_fallback",
            ):
                audited_geodetic_diagnostic_submap_result(fallback["result"])
            policy = GeodeticDiagnosticPolicy(
                allow_sealed_low_parallax_initialization_fallback=True
            )
            assessment = audited_geodetic_diagnostic_submap_result(
                fallback["result"], policy=policy
            )
            self.assertFalse(assessment["production_passed"])
            self.assertTrue(assessment["structurally_safe_for_diagnostic"])
            self.assertEqual(
                assessment["allowed_diagnostic_execution_overrides"],
                ["diagnostic_low_parallax_initialization_fallback"],
            )
            inventory = publish_geodetic_diagnostic_result_inventory(
                assembly_plan,
                {
                    window["window_id"]: fallback["result"],
                    audited["windows"][1]["window_id"]: results[1],
                },
                root / "low-parallax-fallback-inventory",
                policy=policy,
            )
            inventory_record = audited_geodetic_diagnostic_result_inventory(
                inventory
            )
            self.assertEqual(
                inventory_record[
                    "diagnostic_initialization_override_window_ids"
                ],
                [window["window_id"]],
            )

            binding_path = Path(fallback["workspace"]) / "window_binding.json"
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
            binding["fallback_mapper_init_min_tri_angle_deg"] = 2.0
            _write_json(binding_path, binding)
            with self.assertRaisesRegex(ArtifactError, "binding changed"):
                audited_geodetic_diagnostic_low_parallax_fallback_window(
                    assembly_plan, fallback["workspace"]
                )

    def test_diagnostic_inventory_rejects_structural_failure_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, plan, audited, _, results = self._completed_assembly(root)
            unsafe = self._copy_result_with_failed_checks(
                results[1],
                root / "unsafe-window",
                ("selected_visual_quality",),
            )
            result_by_window = {
                audited["windows"][0]["window_id"]: results[0],
                audited["windows"][1]["window_id"]: unsafe,
            }
            with self.assertRaisesRegex(ArtifactError, "non-RTK"):
                publish_geodetic_diagnostic_result_inventory(
                    plan, result_by_window, root / "unsafe-inventory"
                )

            valid_by_window = {
                window["window_id"]: result
                for window, result in zip(
                    audited["windows"], results, strict=True
                )
            }
            inventory = publish_geodetic_diagnostic_result_inventory(
                plan, valid_by_window, root / "valid-inventory"
            )
            record_path = inventory / "diagnostic_result_inventory.json"
            record = json.loads(record_path.read_text())
            record["selected_frame_count"] += 1
            _write_json(record_path, record)
            _write_json(
                inventory / "manifest.json",
                _file_evidence(
                    inventory, ["diagnostic_result_inventory.json"]
                ),
            )
            with self.assertRaisesRegex(ArtifactError, "content changed"):
                audited_geodetic_diagnostic_result_inventory(inventory)

    def test_sealed_export_overlap_and_full_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment, plan, audited, submaps, results = self._completed_assembly(
                root
            )
            export = export_geodetic_submap_poses(
                results[0], root / "accepted-submap-export"
            )
            export_record = audited_geodetic_submap_pose_export(export)
            self.assertTrue(export_record["publication_eligible"])
            self.assertTrue((export / "trajectory.svg").is_file())
            overlap = publish_geodetic_overlap_report(
                results,
                root / "overlap",
                policy=audited["config_object"].overlap_policy,
            )
            self.assertTrue(audited_geodetic_overlap_report(overlap)["passed"])
            pose = publish_geodetic_full_pose_artifact(
                plan, submaps, root / "poses", "assembled"
            )
            final = audited_geodetic_full_pose_artifact(pose)
            self.assertEqual(final["manifest"]["n_frames"], 8)
            self.assertTrue(
                final["georeferencing"][
                    "metric_georeferencing_claim_eligible"
                ]
            )
            viewmats = np.load(pose / "viewmats.npy")
            centers = np.load(pose / "cam_centers.npy")
            self.assertTrue(
                np.allclose(np.linalg.inv(viewmats)[:, :3, 3], centers)
            )
            self.assertEqual(len(viewmats), 8)
            self.assertEqual(
                len(audited["global_holdout"]["holdout_names"]), 2
            )
            with self.assertRaises(FileExistsError):
                publish_geodetic_full_pose_artifact(
                    plan, submaps, root / "poses", "assembled"
                )
            # The standard consumer validates the exact full segment count.
            cfg = SimpleNamespace(
                pose=SimpleNamespace(
                    artifact="assembled", artifact_root=root / "poses"
                )
            )
            from rtk_splat.core.pose_artifacts import load_pose_artifact

            loaded, loaded_centers = load_pose_artifact(segment, cfg)
            self.assertEqual(loaded.shape, (8, 4, 4))
            self.assertEqual(loaded_centers.shape, (8, 3))

    def test_global_holdouts_are_absent_from_every_covering_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, audited, submaps, _ = self._completed_assembly(root)
            global_holdout = set(audited["global_holdout"]["holdout_names"])
            self.assertTrue(global_holdout)
            for window in audited["windows"]:
                database = submaps / window["window_id"] / "result" / "database.db"
                with sqlite3.connect(database) as connection:
                    remaining = {
                        str(name)
                        for (name,) in connection.execute(
                            "SELECT i.name FROM pose_priors p "
                            "JOIN images i ON i.image_id=p.corr_data_id"
                        )
                    }
                local_global = global_holdout & set(
                    window["calibration_names"] + window["holdout_names"]
                )
                self.assertFalse(local_global & remaining)

    def test_overlap_fixed_se3_gauge_is_synchronized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, audited, _, results = self._completed_assembly(root)
            candidates = [_aligned_submap_candidate(path) for path in results]
            candidates[1] = dict(candidates[1])
            angle = np.deg2rad(5.0)
            rotation = np.asarray(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            viewmats, centers = _apply_world_alignment(
                candidates[1]["viewmats"],
                candidates[1]["centers"],
                rotation,
                np.asarray([1.0, -2.0, 0.5]),
            )
            candidates[1]["viewmats"] = viewmats
            candidates[1]["centers"] = centers
            report = _evaluate_overlap_candidates(
                candidates, audited["config_object"].overlap_policy
            )
            self.assertTrue(report["passed"])
            self.assertFalse(
                report["pairs"][0]["pre_synchronization"]["passed"]
            )
            self.assertTrue(report["pairs"][0]["passed"])
            self.assertEqual(
                report["synchronization"]["production_scale"], 1.0
            )
            self.assertFalse(
                report["synchronization"]["sim3_scale_applied"]
            )
            self.assertTrue(
                report["synchronization"]["checks"][
                    "all_synchronized_heldout_rtk_absolute_gates"
                ]["passed"]
            )
            scope = report["synchronization"][
                "synchronized_holdout_scope"
            ]
            self.assertEqual(
                scope["method"],
                "heldout_in_every_covering_optimizer_v1",
            )
            self.assertEqual(scope["safe_holdout_count"], 2)
            self.assertEqual(scope["excluded_local_holdout_count"], 1)
            self.assertEqual(
                [
                    item["n_holdout"]
                    for item in report["synchronization"][
                        "synchronized_holdout"
                    ]
                ],
                [1, 2],
            )

    def test_assembly_requires_identical_sealed_priors_in_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, _, results = self._completed_assembly(root)
            candidates = [_aligned_submap_candidate(path) for path in results]
            priors = _assembly_evaluation_priors(candidates)
            shared = set(candidates[0]["left_image_names"]) & set(
                candidates[1]["left_image_names"]
            )
            shared_prior = next(name for name in shared if name in priors)
            changed = [dict(candidate) for candidate in candidates]
            changed[1]["prior_centers"] = np.asarray(
                changed[1]["prior_centers"]
            ).copy()
            index = changed[1]["left_image_names"].index(shared_prior)
            changed[1]["prior_centers"][index, 0] += 0.001

            with self.assertRaisesRegex(
                ArtifactError, "disagree on a sealed evaluation prior"
            ):
                _assembly_evaluation_priors(changed)

    def test_synchronized_holdout_excludes_cross_optimizer_role_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, audited, _, results = self._completed_assembly(root)
            candidates = [_aligned_submap_candidate(path) for path in results]
            occurrences = {}
            for candidate_index, candidate in enumerate(candidates):
                for index, name in enumerate(candidate["left_image_names"]):
                    occurrences.setdefault(str(name), []).append(
                        (candidate_index, index, candidate["roles"][index])
                    )
            cross_role_name, entries = next(
                (name, entries)
                for name, entries in occurrences.items()
                if {entry[2] for entry in entries}
                == {"calibration", "holdout"}
            )
            baseline = _evaluate_overlap_candidates(
                candidates, audited["config_object"].overlap_policy
            )
            legacy = _evaluate_overlap_candidates_v2(
                candidates, audited["config_object"].overlap_policy
            )
            changed = [dict(candidate) for candidate in candidates]
            holdout_candidate, holdout_index, _ = next(
                entry for entry in entries if entry[2] == "holdout"
            )
            changed[holdout_candidate]["prior_centers"] = np.asarray(
                changed[holdout_candidate]["prior_centers"]
            ).copy()
            changed[holdout_candidate]["prior_centers"][holdout_index] += (
                np.asarray([100.0, -50.0, 25.0])
            )
            repeated = _evaluate_overlap_candidates(
                changed, audited["config_object"].overlap_policy
            )

            self.assertEqual(baseline, repeated)
            calibration_scope = baseline["synchronization"][
                "calibration_scope"
            ]
            self.assertEqual(
                calibration_scope["method"],
                "calibration_in_any_covering_optimizer_v1",
            )
            self.assertEqual(
                calibration_scope["mixed_local_role_calibration_count"], 1
            )
            self.assertEqual(
                calibration_scope[
                    "mixed_local_role_calibration_names_sha256"
                ],
                canonical_hash([cross_role_name]),
            )
            self.assertEqual(
                calibration_scope["deduplicated_calibration_count"],
                legacy["synchronization"][
                    "shared_safe_calibration_count"
                ]
                + 1,
            )
            scope = baseline["synchronization"][
                "synchronized_holdout_scope"
            ]
            self.assertEqual(scope["excluded_local_holdout_count"], 1)
            self.assertEqual(
                scope["excluded_local_holdout_names_sha256"],
                canonical_hash([cross_role_name]),
            )

    def test_overlap_nonrigid_scale_disagreement_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, audited, _, results = self._completed_assembly(root)
            candidates = [_aligned_submap_candidate(path) for path in results]
            candidates[1] = dict(candidates[1])
            centers = np.asarray(candidates[1]["centers"]).copy()
            centers = centers[0] + 1.5 * (centers - centers[0])
            viewmats = np.asarray(candidates[1]["viewmats"]).copy()
            viewmats[:, :3, 3] = -np.einsum(
                "nij,nj->ni", viewmats[:, :3, :3], centers
            )
            candidates[1]["centers"] = centers
            candidates[1]["viewmats"] = viewmats
            report = _evaluate_overlap_candidates(
                candidates, audited["config_object"].overlap_policy
            )
            self.assertFalse(report["passed"])
            self.assertFalse(
                report["pairs"][0]["checks"][
                    "maximum_center_disagreement_m"
                ]["passed"]
            )
            self.assertEqual(
                report["synchronization"]["production_scale"], 1.0
            )
            self.assertFalse(
                report["synchronization"]["sim3_scale_applied"]
            )

    def test_legacy_overlap_audit_and_resealed_sync_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, audited, _, results = self._completed_assembly(root)
            candidates = [_aligned_submap_candidate(path) for path in results]
            policy = audited["config_object"].overlap_policy
            legacy = _legacy_overlap_report(candidates, policy)
            legacy_root = root / "legacy-overlap"
            legacy_root.mkdir()
            _write_json(legacy_root / "overlap.json", legacy)
            (legacy_root / "overlap.csv").write_bytes(_overlap_csv(legacy))
            (legacy_root / "overlap.svg").write_bytes(_overlap_svg(legacy))
            _write_json(
                legacy_root / "manifest.json",
                _file_evidence(
                    legacy_root,
                    ["overlap.json", "overlap.csv", "overlap.svg"],
                ),
            )
            with mock.patch(
                "rtk_splat.backends.geodetic_assembly."
                "_aligned_submap_candidate",
                side_effect=candidates,
            ):
                self.assertEqual(
                    audited_geodetic_overlap_report(legacy_root)[
                        "schema_version"
                    ],
                    1,
                )

            schema_two = _evaluate_overlap_candidates_v2(candidates, policy)
            schema_two_root = root / "schema-two-overlap"
            schema_two_root.mkdir()
            _write_json(schema_two_root / "overlap.json", schema_two)
            (schema_two_root / "overlap.csv").write_bytes(
                _overlap_csv(schema_two)
            )
            (schema_two_root / "overlap.svg").write_bytes(
                _overlap_svg(schema_two)
            )
            _write_json(
                schema_two_root / "manifest.json",
                _file_evidence(
                    schema_two_root,
                    ["overlap.json", "overlap.csv", "overlap.svg"],
                ),
            )
            self.assertEqual(
                audited_geodetic_overlap_report(schema_two_root)[
                    "schema_version"
                ],
                2,
            )

            current_root = publish_geodetic_overlap_report(
                results, root / "current-overlap", policy=policy
            )
            current = json.loads((current_root / "overlap.json").read_text())
            current["synchronization"]["production_scale"] = 1.01
            _write_json(current_root / "overlap.json", current)
            _write_json(
                current_root / "manifest.json",
                _file_evidence(
                    current_root,
                    ["overlap.json", "overlap.csv", "overlap.svg"],
                ),
            )
            with self.assertRaisesRegex(ArtifactError, "content changed"):
                audited_geodetic_overlap_report(current_root)

            holdout_root = publish_geodetic_overlap_report(
                results, root / "holdout-tamper", policy=policy
            )
            holdout = json.loads((holdout_root / "overlap.json").read_text())
            holdout["synchronization"]["synchronized_holdout_scope"][
                "safe_holdout_names_sha256"
            ] = "0" * 64
            _write_json(holdout_root / "overlap.json", holdout)
            _write_json(
                holdout_root / "manifest.json",
                _file_evidence(
                    holdout_root,
                    ["overlap.json", "overlap.csv", "overlap.svg"],
                ),
            )
            with self.assertRaisesRegex(ArtifactError, "content changed"):
                audited_geodetic_overlap_report(holdout_root)

    def test_assembly_code_and_launcher_are_generic(self):
        repository = Path(__file__).resolve().parents[1]
        paths = [
            repository
            / "src"
            / "rtk_splat"
            / "backends"
            / "geodetic_assembly.py",
            repository
            / "scripts"
            / "experiments"
            / "geodetic_submap_assembly_v1.py",
            repository
            / "scripts"
            / "experiments"
            / "geodetic_full_field_diagnostic_v1.py",
        ]
        for path in paths:
            contents = path.read_text(encoding="utf-8").lower()
            for token in ("field1", "/data/jkobo", "10227", "1100", "1800"):
                self.assertNotIn(token, contents, f"{token!r} in {path.name}")
        launcher_help = {}
        for launcher in paths[1:]:
            completed = subprocess.run(
                [sys.executable, str(launcher), "--help"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            launcher_help[launcher] = completed.stdout
        assembly_help = launcher_help[paths[1]]
        self.assertIn("prepare-all", assembly_help)
        self.assertIn("assemble", assembly_help)
        self.assertIn("diagnostic-inventory", assembly_help)
        diagnostic_help = launcher_help[paths[2]]
        self.assertIn("--reuse", diagnostic_help)
        self.assertIn("--resume", diagnostic_help)
        self.assertIn(
            "--allow-sealed-low-parallax-initialization-fallback",
            diagnostic_help,
        )
        launcher = paths[1].read_text(encoding="utf-8")
        self.assertIn("ProcessPoolExecutor", launcher)
        self.assertIn('get_context("spawn")', launcher)
        self.assertNotIn("ThreadPoolExecutor", launcher)


if __name__ == "__main__":
    unittest.main()
