import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backends.mapper import (
    MapperConfig,
    build_mapper_command,
    build_registration_command,
    estimate_rigid_alignment,
    estimate_temporal_heldout_alignment,
    export_pose_artifact,
    parse_model_analyzer,
    prepare_mapper_backend,
    quality_summary,
    registered_names_from_images_txt,
    run_image_registration,
    run_mapper_solve,
    run_quality_summary,
)
from frontends.artifact import (
    ArtifactError,
    create_database_snapshot,
    create_frontend_seal,
    sha256_file,
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _option(command, name: str) -> str:
    return command[command.index(name) + 1]


def _frontend(root: Path) -> tuple[Path, list[str]]:
    artifact = root / "frontend"
    (root / "immutable-segment").mkdir()
    images = artifact / "images"
    stages = artifact / "stages"
    images.mkdir(parents=True)
    stages.mkdir()
    rows = []
    names = []
    for frame_id in range(6):
        row = {"frame_id": frame_id, "timestamp_ns": 1_000_000_000 + frame_id}
        for side in ("left", "right"):
            name = f"{side}_{frame_id:06d}.jpg"
            source = root / f"source-{name}"
            source.write_bytes(f"{side}-{frame_id}".encode())
            os.symlink(source, images / name)
            row[f"{side}_image"] = {
                "name": name,
                "content_sha256": sha256_file(source),
            }
            names.append(name)
        rows.append(row)
    _write_json(
        artifact / "frame_manifest.json",
        {"schema_version": 1, "frames": rows, "image_inventory_sha256": "a" * 64},
    )
    _write_json(
        artifact / "keyframes.json",
        {"schema_version": 1, "frame_ids": [0, 2], "solve_only": True},
    )
    for name, value in (
        ("rig_config.json", {"schema_version": 1, "rigs": []}),
        (
            "provenance.json",
            {
                "schema_version": 1,
                "test": True,
                "contract_inputs": {
                    "segment_root": str((root / "immutable-segment").resolve())
                },
            },
        ),
        ("quality.json", {"schema_version": 1, "n_frames": 6}),
    ):
        _write_json(artifact / name, value)
    (artifact / "pairs.txt").write_text(
        "left_000000.jpg right_000000.jpg\n", encoding="utf-8"
    )
    with sqlite3.connect(artifact / "database.db") as connection:
        connection.executescript(
            """
            CREATE TABLE images(
              image_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
              camera_id INTEGER NOT NULL);
            CREATE TABLE pose_priors(
              pose_prior_id INTEGER PRIMARY KEY,
              corr_data_id INTEGER NOT NULL,
              corr_sensor_id INTEGER NOT NULL,
              corr_sensor_type INTEGER NOT NULL,
              position BLOB,
              position_covariance BLOB,
              gravity BLOB,
              coordinate_system INTEGER NOT NULL);
            """
        )
        for image_id, name in enumerate(names, start=1):
            connection.execute(
                "INSERT INTO images VALUES (?, ?, ?)",
                (image_id, name, 1 if name.startswith("left_") else 2),
            )
            if name.startswith("left_"):
                frame_id = int(name.split("_")[1].split(".")[0])
                source_center = np.array(
                    [float(frame_id), 0.1 * frame_id**2, 0.0],
                    dtype="<f8",
                )
                target_center = source_center + np.array([10.0, 20.0, 3.0])
                covariance = np.diag([1e-4, 1e-4, 4e-4]).astype("<f8")
                connection.execute(
                    """
                    INSERT INTO pose_priors
                    VALUES (?, ?, 1, 0, ?, ?, NULL, 1)
                    """,
                    (
                        image_id,
                        image_id,
                        target_center.astype("<f8").tobytes(),
                        covariance.tobytes(order="F"),
                    ),
                )
    create_frontend_seal(artifact)
    _write_json(
        stages / "matching.json",
        {
            "schema_version": 1,
            "stage": "matching",
            "state": "complete",
            "outputs": {
                "database.db": sha256_file(artifact / "database.db"),
                "frontend_seal.json": sha256_file(
                    artifact / "frontend_seal.json"
                ),
            },
        },
    )
    return artifact, names


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
Observations: 404
Mean track length: 4.0
Mean observations per image: 67.3
Mean reprojection error: 0.42
"""


class _ColmapRunner:
    def __init__(self, all_names: list[str]):
        self.all_names = all_names
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        action = command[1]
        if action in {"global_mapper", "mapper"}:
            _model(Path(_option(command, "--output_path")) / "0", b"solve")
            return SimpleNamespace(returncode=0)
        if action == "image_registrator":
            _model(Path(_option(command, "--output_path")), b"registered")
            return SimpleNamespace(returncode=0)
        if action == "model_converter":
            output = Path(_option(command, "--output_path"))
            output.mkdir(exist_ok=True)
            lines = ["# Image list"]
            for index, name in enumerate(self.all_names, start=1):
                frame_id = int(name.split("_")[1].split(".")[0])
                center = np.array(
                    [float(frame_id), 0.1 * frame_id**2, 0.0]
                )
                if name.startswith("right_"):
                    center[0] += 0.12
                translation = -center
                lines.extend(
                    [
                        f"{index} 1 0 0 0 {translation[0]} "
                        f"{translation[1]} {translation[2]} 1 {name}",
                        "",
                    ]
                )
            (output / "images.txt").write_text("\n".join(lines) + "\n")
            (output / "cameras.txt").write_text("# cameras\n")
            (output / "points3D.txt").write_text("# points\n")
            return SimpleNamespace(returncode=0)
        if action == "model_analyzer":
            model = Path(_option(command, "--path"))
            count = (
                len(self.all_names)
                if model.name in {"registration.incomplete", "registered_model"}
                else 4
            )
            return SimpleNamespace(returncode=0, stdout=_analyzer_output(count))
        raise AssertionError(f"unexpected command: {command}")


class MapperBackendTests(unittest.TestCase):
    def test_global_is_primary_and_incremental_is_an_independent_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontend, _ = _frontend(root)
            global_work = prepare_mapper_backend(
                frontend, root / "global", config=MapperConfig()
            )
            incremental_work = prepare_mapper_backend(
                frontend,
                root / "incremental",
                config=MapperConfig(backend="incremental"),
            )
            global_command = build_mapper_command(global_work, "/x/colmap")
            incremental_command = build_mapper_command(
                incremental_work, "/x/colmap"
            )
            self.assertEqual(global_command[:2], ("/x/colmap", "global_mapper"))
            self.assertEqual(
                _option(global_command, "--GlobalMapper.image_list_path"),
                str(global_work / "solve_images.txt"),
            )
            self.assertEqual(
                incremental_command[:2], ("/x/colmap", "mapper")
            )
            self.assertEqual(
                _option(incremental_command, "--Mapper.image_list_path"),
                str(incremental_work / "solve_images.txt"),
            )
            self.assertEqual(
                (global_work / "database.db").read_bytes(),
                (incremental_work / "database.db").read_bytes(),
            )
            self.assertNotEqual(
                global_work / "database.db", incremental_work / "database.db"
            )
            with self.assertRaisesRegex(ArtifactError, "outside"):
                prepare_mapper_backend(frontend, frontend / "backends" / "bad")
            with self.assertRaisesRegex(ArtifactError, "outside"):
                prepare_mapper_backend(
                    frontend, root / "immutable-segment" / "backends" / "bad"
                )

    def test_prepare_is_idempotent_and_snapshot_isolated_and_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontend, _ = _frontend(root)
            workspace = prepare_mapper_backend(frontend, root / "backend")
            source_original = (frontend / "database.db").read_bytes()
            original = (workspace / "database.db").read_bytes()
            self.assertEqual(
                _write_and_return(frontend / "database.db", b"source changed"),
                b"source changed",
            )
            self.assertEqual((workspace / "database.db").read_bytes(), original)
            with self.assertRaisesRegex(
                ArtifactError, "different inputs|incompatible|consistent SQLite"
            ):
                prepare_mapper_backend(frontend, workspace)
            (frontend / "database.db").write_bytes(source_original)
            self.assertEqual(prepare_mapper_backend(frontend, workspace), workspace)
            (workspace / "database.db").write_bytes(b"tampered")
            with self.assertRaisesRegex(ArtifactError, "snapshot changed"):
                build_mapper_command(workspace, "colmap")

    def test_prepare_recovers_a_verified_snapshot_only_partial_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontend, _ = _frontend(root)
            workspace = root / "backend"
            create_database_snapshot(
                frontend / "database.db", workspace, backend="global"
            )
            self.assertEqual(
                prepare_mapper_backend(frontend, workspace), workspace
            )
            self.assertTrue((workspace / "stages" / "prepare.json").is_file())

    def test_later_backend_stage_rejects_frontend_database_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontend, _ = _frontend(root)
            workspace = prepare_mapper_backend(frontend, root / "backend")
            with sqlite3.connect(frontend / "database.db") as connection:
                connection.execute("CREATE TABLE injected(value INTEGER)")
            with self.assertRaisesRegex(
                ArtifactError, "terminal seal|terminal marker"
            ):
                build_mapper_command(workspace, "colmap")

    def test_later_backend_stage_rejects_image_symlink_target_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontend, names = _frontend(root)
            workspace = prepare_mapper_backend(frontend, root / "backend")
            link = frontend / "images" / names[0]
            replacement = root / "replacement.jpg"
            replacement.write_bytes(b"different-image-content")
            link.unlink()
            os.symlink(replacement, link)
            with self.assertRaisesRegex(
                ArtifactError, "image content changed|terminal seal"
            ):
                build_mapper_command(workspace, "colmap")

    def test_solve_register_and_quality_are_independently_rerunnable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontend, names = _frontend(root)
            workspace = prepare_mapper_backend(frontend, root / "backend")
            runner = _ColmapRunner(names)

            solve = run_mapper_solve(workspace, "colmap", runner=runner)
            self.assertTrue(solve["all_keyframes_registered"])
            self.assertEqual(solve["n_expected_solve_images"], 4)
            self.assertEqual(
                build_registration_command(workspace, "colmap")[1],
                "image_registrator",
            )
            registration = run_image_registration(
                workspace, "colmap", runner=runner
            )
            self.assertTrue(registration["all_frames_registered_by_count"])
            self.assertEqual(registration["n_newly_registered_images"], 8)
            quality = run_quality_summary(workspace, "colmap", runner=runner)
            self.assertTrue(quality["passed"])
            self.assertEqual(quality["registration_fraction"], 1.0)
            pose_root = root / "pose_artifacts"
            pose_artifact = export_pose_artifact(
                workspace, "global-balanced", output_root=pose_root
            )
            np.testing.assert_array_equal(
                np.load(pose_artifact / "frame_ids.npy"), np.arange(6)
            )
            self.assertEqual(
                np.load(pose_artifact / "viewmats.npy").shape, (6, 4, 4)
            )
            self.assertEqual(
                np.load(pose_artifact / "cam_centers.npy").shape, (6, 3)
            )
            np.testing.assert_allclose(
                np.load(pose_artifact / "cam_centers.npy"),
                [
                    [10.0 + index, 20.0 + 0.1 * index**2, 3.0]
                    for index in range(6)
                ],
                atol=1e-8,
            )
            provenance = json.loads(
                (pose_artifact / "provenance.json").read_text()
            )
            self.assertEqual(provenance["backend"], "global")
            alignment = json.loads(
                (pose_artifact / "alignment.json").read_text()
            )
            self.assertEqual(alignment["production_scale"], 1.0)
            self.assertAlmostEqual(alignment["sim3_scale_diagnostic"], 1.0)
            self.assertFalse(alignment["pose_priors_constrain_mapper"])
            self.assertEqual(
                alignment["temporal_split"]["holdout_prior_image_names"],
                ["left_000002.jpg", "left_000004.jpg"],
            )
            self.assertEqual(
                provenance["source_backend_workspace"], str(workspace)
            )
            with self.assertRaises(FileExistsError):
                export_pose_artifact(
                    workspace, "global-balanced", output_root=pose_root
                )
            with self.assertRaisesRegex(ArtifactError, "immutable"):
                export_pose_artifact(
                    workspace,
                    "invalid-location",
                    output_root=frontend / "pose_artifacts",
                )

            commands_before = len(runner.commands)

            def must_not_run(*args, **kwargs):
                raise AssertionError("completed stage was rerun")

            self.assertEqual(
                run_mapper_solve(workspace, "colmap", runner=must_not_run),
                solve,
            )
            self.assertEqual(
                run_image_registration(
                    workspace, "colmap", runner=must_not_run
                ),
                registration,
            )
            self.assertEqual(
                run_quality_summary(workspace, "colmap", runner=must_not_run),
                quality,
            )
            self.assertEqual(len(runner.commands), commands_before)

    def test_failed_solve_cleans_partial_output_and_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontend, names = _frontend(root)
            workspace = prepare_mapper_backend(frontend, root / "backend")

            def fail_after_partial(command, **kwargs):
                if command[1] == "global_mapper":
                    _model(Path(_option(command, "--output_path")) / "0")
                    raise RuntimeError("interrupted")
                raise AssertionError(command)

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_mapper_solve(workspace, "colmap", runner=fail_after_partial)
            self.assertFalse((workspace / "solve.incomplete").exists())
            solve = run_mapper_solve(
                workspace, "colmap", runner=_ColmapRunner(names)
            )
            self.assertTrue(solve["all_keyframes_registered"])

    def test_exact_registration_and_quality_hooks_report_failures(self):
        stats = parse_model_analyzer(_analyzer_output(2))
        summary = quality_summary(
            ["left.jpg", "right.jpg"],
            ["left.jpg"],
            {**stats, "registered_images": 1},
        )
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["missing_images"], ["right.jpg"])

        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images.txt"
            images.write_text(
                "# images\n"
                "1 1 0 0 0 0 0 0 1 left.jpg\n"
                "\n"
                "2 1 0 0 0 0 0 0 2 right.jpg\n"
                "1.0 2.0 -1\n"
            )
            self.assertEqual(
                registered_names_from_images_txt(images),
                ("left.jpg", "right.jpg"),
            )

    def test_robust_fixed_scale_alignment_recovers_transform_with_outliers(self):
        rng = np.random.default_rng(4)
        source = rng.normal(size=(30, 3))
        angle = np.deg2rad(23.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.array([10.0, -3.0, 1.5])
        target = source @ rotation.T + translation
        target += rng.normal(scale=0.002, size=target.shape)
        target[[3, 17]] += np.array([4.0, -2.0, 3.0])
        covariance = np.repeat((np.eye(3) * 1e-4)[None], len(source), axis=0)
        result = estimate_rigid_alignment(
            source,
            target,
            covariance,
            ransac_threshold_m=0.03,
            ransac_iterations=256,
            random_seed=9,
        )
        self.assertEqual(int(result.inlier_mask.sum()), 28)
        np.testing.assert_allclose(result.rotation, rotation, atol=2e-3)
        np.testing.assert_allclose(result.translation, translation, atol=2e-3)
        self.assertAlmostEqual(result.sim3_scale_diagnostic, 1.0, places=3)

    def test_similarity_scale_is_diagnostic_and_never_applied(self):
        rng = np.random.default_rng(8)
        source = rng.normal(size=(25, 3))
        true_scale = 1.014
        target = true_scale * source + np.array([2.0, 4.0, -1.0])
        result = estimate_rigid_alignment(
            source,
            target,
            ransac_threshold_m=0.2,
            ransac_iterations=128,
        )
        self.assertAlmostEqual(result.sim3_scale_diagnostic, true_scale, places=6)
        # The production transform is rigid: its rotation determinant is one
        # and no scale parameter exists to modify camera-centre distances.
        self.assertAlmostEqual(np.linalg.det(result.rotation), 1.0, places=12)
        source_distance = np.linalg.norm(source[0] - source[1])
        rigid_distance = np.linalg.norm(
            (result.rotation @ source[0] + result.translation)
            - (result.rotation @ source[1] + result.translation)
        )
        self.assertAlmostEqual(rigid_distance, source_distance, places=12)

    def test_temporal_holdout_regression_does_not_contaminate_alignment(self):
        count = 40
        parameter = np.linspace(0.0, 4.0, count)
        source = np.column_stack(
            (
                parameter,
                np.sin(parameter * 1.3),
                np.cos(parameter * 0.7),
            )
        )
        angle = np.deg2rad(-17.0)
        rotation = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        translation = np.array([3.0, -2.0, 0.7])
        clean_target = source @ rotation.T + translation
        covariance = np.repeat((np.eye(3) * 1e-4)[None], count, axis=0)
        timestamps = np.arange(count, dtype=np.int64) * 100_000_000
        clean = estimate_temporal_heldout_alignment(
            source,
            clean_target,
            timestamps,
            covariance,
            ransac_threshold_m=0.03,
            random_seed=11,
        )
        contaminated_target = clean_target.copy()
        contaminated_target[clean.holdout_mask] += np.array([2.0, -1.0, 0.5])
        contaminated = estimate_temporal_heldout_alignment(
            source,
            contaminated_target,
            timestamps,
            covariance,
            ransac_threshold_m=0.03,
            random_seed=11,
        )
        np.testing.assert_array_equal(
            contaminated.temporal_block_ids, clean.temporal_block_ids
        )
        np.testing.assert_allclose(
            contaminated.alignment.rotation, clean.alignment.rotation, atol=1e-12
        )
        np.testing.assert_allclose(
            contaminated.alignment.translation,
            clean.alignment.translation,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            contaminated.alignment.rotation, rotation, atol=1e-10
        )
        np.testing.assert_allclose(
            contaminated.alignment.translation, translation, atol=1e-10
        )
        self.assertEqual(int(contaminated.holdout_inlier_mask.sum()), 0)
        self.assertGreater(
            float(np.median(contaminated.residuals_m[contaminated.holdout_mask])),
            2.0,
        )
        self.assertLess(
            float(
                np.max(
                    contaminated.residuals_m[contaminated.calibration_inlier_mask]
                )
            ),
            1e-9,
        )

    def test_temporal_holdout_requires_enough_independent_calibration_data(self):
        with self.assertRaisesRegex(ArtifactError, "temporal holdout"):
            estimate_temporal_heldout_alignment(
                np.eye(3),
                np.eye(3),
                np.arange(3, dtype=np.int64),
            )

    def test_alignment_rejects_unobservable_collinear_camera_centres(self):
        source = np.column_stack(
            (np.arange(8, dtype=float), np.zeros(8), np.zeros(8))
        )
        with self.assertRaisesRegex(ArtifactError, "observable"):
            estimate_rigid_alignment(
                source,
                source + [5.0, 2.0, 0.0],
                ransac_iterations=32,
            )

    def test_module_has_no_ros_dataset_or_segment_dependency(self):
        source = (
            Path(__file__).parents[1] / "backends" / "mapper.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("rosbag", source)
        self.assertNotIn("sensor_msgs", source)
        self.assertNotIn("rtk_splat.segment", source)
        self.assertNotIn("adapters.", source)


def _write_and_return(path: Path, payload: bytes) -> bytes:
    path.write_bytes(payload)
    return payload


if __name__ == "__main__":
    unittest.main()
