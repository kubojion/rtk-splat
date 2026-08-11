import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rtk_splat.backends.mapper import (
    MapperConfig,
    _MonitoredProcessInterrupted,
    _run_monitored_mapper,
    build_mapper_command,
    build_registration_command,
    estimate_rigid_alignment,
    estimate_temporal_heldout_alignment,
    export_pose_artifact,
    parse_model_analyzer,
    prepare_mapper_backend,
    quality_summary,
    registered_names_from_images_txt,
    rtk_residual_quality,
    run_image_registration,
    run_mapper_solve,
    run_quality_summary,
)
from rtk_splat.frontends.artifact import (
    ArtifactError,
    create_database_snapshot,
    create_frontend_seal,
    sha256_file,
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _option(command, name: str) -> str:
    return command[command.index(name) + 1]


def _frontend(
    root: Path,
    *,
    prior_offsets_m: dict[int, np.ndarray] | None = None,
) -> tuple[Path, list[str]]:
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
                if prior_offsets_m and frame_id in prior_offsets_m:
                    target_center += prior_offsets_m[frame_id]
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
            expected_global = {
                "--GlobalMapper.ba_num_iterations": "3",
                "--GlobalMapper.gp_use_gpu": "0",
                "--GlobalMapper.ba_ceres_use_gpu": "0",
                "--GlobalMapper.keep_max_num_tracks": "60000",
                "--GlobalMapper.track_required_tracks_per_view": "1000",
                "--GlobalMapper.skip_retriangulation": "1",
            }
            for option, value in expected_global.items():
                self.assertEqual(_option(global_command, option), value)
            self.assertEqual(
                incremental_command[:2], ("/x/colmap", "mapper")
            )
            self.assertEqual(
                _option(incremental_command, "--Mapper.image_list_path"),
                str(incremental_work / "solve_images.txt"),
            )
            self.assertNotIn("--GlobalMapper.keep_max_num_tracks", incremental_command)
            self.assertNotIn("--log_target", incremental_command)
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

    def test_mapper_configuration_validation_is_strict(self):
        invalid = (
            {"ba_num_iterations": 0},
            {"keep_max_num_tracks": 0},
            {"track_required_tracks_per_view": -1},
            {"skip_retriangulation": 1},
            {"gp_use_gpu": "false"},
            {"ba_ceres_use_gpu": 0},
            {"process_nice": 20},
            {"resource_sample_interval_s": 0.0},
            {"minimum_available_memory_gb": float("nan")},
            {"low_memory_consecutive_samples": 0},
            {"rtk_chi2_inlier_probability": 1.0},
            {"max_rtk_median_mahalanobis_sq": 0.0},
            {"min_rtk_chi2_inlier_fraction": 0.0},
            {"rtk_covariance_gate_mode": "guess"},
            {
                "minimum_free_space_gb": 1.0,
                "minimum_runtime_free_space_gb": 2.0,
            },
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                MapperConfig(**values)

    def test_backend_plan_must_freeze_the_complete_mapper_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontend, _ = _frontend(root)
            workspace = prepare_mapper_backend(frontend, root / "backend")
            plan_path = workspace / "backend_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["config"].pop("keep_max_num_tracks")
            _write_json(plan_path, plan)
            with self.assertRaisesRegex(ArtifactError, "predates.*new backend"):
                build_mapper_command(workspace, "colmap")

    def test_legacy_plan_backfills_only_new_evaluation_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontend, _ = _frontend(root)
            workspace = prepare_mapper_backend(frontend, root / "backend")
            plan_path = workspace / "backend_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            for field in (
                "rtk_chi2_inlier_probability",
                "max_rtk_median_mahalanobis_sq",
                "min_rtk_chi2_inlier_fraction",
                "rtk_covariance_gate_mode",
            ):
                plan["config"].pop(field)
            _write_json(plan_path, plan)
            self.assertEqual(
                build_mapper_command(workspace, "colmap")[:2],
                ("colmap", "global_mapper"),
            )

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
            self.assertEqual(solve["resources"]["monitor"], "injected-runner")
            samples = workspace / solve["resources"]["samples_csv"]
            self.assertTrue(samples.is_file())
            self.assertTrue(samples.with_name("resource_usage.json").is_file())
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
            self.assertEqual(provenance["artifact_class"], "production")
            self.assertEqual(provenance["georeferencing_status"], "PASSED")
            self.assertTrue(
                provenance["metric_georeferencing_claim_eligible"]
            )
            self.assertFalse(provenance["diagnostic_export_requested"])
            self.assertFalse(provenance["diagnostic_export_override_used"])
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
            backend_plan = json.loads(
                (workspace / "backend_plan.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                provenance["source_segment_binding"],
                backend_plan["source_segment_binding"],
            )
            georeferencing = json.loads(
                (pose_artifact / "georeferencing.json").read_text()
            )
            self.assertEqual(georeferencing["artifact_class"], "production")
            self.assertEqual(georeferencing["georeferencing_status"], "PASSED")
            self.assertTrue(
                georeferencing["metric_georeferencing_claim_eligible"]
            )
            self.assertIsNone(georeferencing["warning"])
            self.assertFalse(
                (pose_artifact / "GEOREFERENCING_FAILED.json").exists()
            )

            diagnostic_pass = export_pose_artifact(
                workspace,
                "global-balanced-diagnostic",
                output_root=pose_root,
                allow_failed_georeferencing_for_render=True,
            )
            diagnostic_pass_quality = json.loads(
                (diagnostic_pass / "quality.json").read_text()
            )
            self.assertEqual(
                diagnostic_pass_quality["artifact_class"],
                "diagnostic_render_only",
            )
            self.assertEqual(
                diagnostic_pass_quality["georeferencing_status"], "PASSED"
            )
            self.assertFalse(
                diagnostic_pass_quality[
                    "metric_georeferencing_claim_eligible"
                ]
            )
            self.assertTrue(
                diagnostic_pass_quality["diagnostic_export_requested"]
            )
            self.assertFalse(
                diagnostic_pass_quality["diagnostic_export_override_used"]
            )
            self.assertFalse(
                (diagnostic_pass / "GEOREFERENCING_FAILED.json").exists()
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

    def test_failed_pose_gate_writes_idempotent_diagnostics_without_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontend, names = _frontend(
                root,
                prior_offsets_m={
                    2: np.array([0.5, 0.0, 0.0]),
                    4: np.array([0.5, 0.0, 0.0]),
                },
            )
            workspace = prepare_mapper_backend(frontend, root / "backend")
            runner = _ColmapRunner(names)
            run_mapper_solve(workspace, "colmap", runner=runner)
            run_image_registration(workspace, "colmap", runner=runner)
            run_quality_summary(workspace, "colmap", runner=runner)
            pose_root = root / "pose_artifacts"
            for _ in range(2):
                with self.assertRaisesRegex(ArtifactError, "diagnostics:"):
                    export_pose_artifact(
                        workspace,
                        "rejected",
                        output_root=pose_root,
                    )
            reports = list(
                (workspace / "reports").glob(
                    "pose_export_alignment_rejected_*.json"
                )
            )
            self.assertEqual(len(reports), 1)
            diagnostic = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertFalse(diagnostic["passed"])
            self.assertIsNone(
                diagnostic["checks"][
                    "p95_holdout_inlier_rtk_residual_m"
                ]["value"]
            )
            self.assertFalse((pose_root / "rejected").exists())
            self.assertFalse(list(pose_root.glob(".rejected.writing-*")))

    def test_failed_pose_gate_opt_in_publishes_sealed_diagnostic_only_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontend, names = _frontend(
                root,
                prior_offsets_m={
                    2: np.array([0.5, 0.0, 0.0]),
                    4: np.array([0.5, 0.0, 0.0]),
                },
            )
            workspace = prepare_mapper_backend(frontend, root / "backend")
            runner = _ColmapRunner(names)
            run_mapper_solve(workspace, "colmap", runner=runner)
            run_image_registration(workspace, "colmap", runner=runner)
            run_quality_summary(workspace, "colmap", runner=runner)

            pose = export_pose_artifact(
                workspace,
                "failed-georeferencing-render",
                output_root=root / "pose_artifacts",
                allow_failed_georeferencing_for_render=True,
            )
            expected_status = {
                "artifact_class": "diagnostic_render_only",
                "georeferencing_status": "FAILED",
                "metric_georeferencing_claim_eligible": False,
                "diagnostic_export_requested": True,
                "diagnostic_export_override_used": True,
            }
            records = {
                filename: json.loads((pose / filename).read_text())
                for filename in (
                    "manifest.json",
                    "quality.json",
                    "alignment.json",
                    "provenance.json",
                    "georeferencing.json",
                    "GEOREFERENCING_FAILED.json",
                )
            }
            for filename, record in records.items():
                for field, expected in expected_status.items():
                    self.assertEqual(
                        record[field], expected, f"{field} in {filename}"
                    )
            self.assertFalse(records["quality.json"]["rtk_alignment_passed"])
            self.assertFalse(
                records["georeferencing.json"]["rtk_alignment_passed"]
            )
            self.assertIn(
                "Do not use this artifact for metric georeferencing claims",
                records["GEOREFERENCING_FAILED.json"]["warning"],
            )
            np.testing.assert_array_equal(
                np.load(pose / "frame_ids.npy"), np.arange(6)
            )
            self.assertEqual(np.load(pose / "viewmats.npy").shape, (6, 4, 4))

            manifest = records["manifest.json"]
            self.assertIn("GEOREFERENCING_FAILED.json", manifest["files"])
            self.assertIn("georeferencing.json", manifest["files"])
            for filename, evidence in manifest["files"].items():
                path = pose / filename
                self.assertEqual(sha256_file(path), evidence["sha256"])
                self.assertEqual(path.stat().st_size, evidence["size_bytes"])

            reports = list(
                (workspace / "reports").glob(
                    "pose_export_alignment_failed-georeferencing-render_*.json"
                )
            )
            self.assertEqual(len(reports), 1)
            diagnostic = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertFalse(diagnostic["passed"])
            self.assertTrue(
                diagnostic["inputs"][
                    "allow_failed_georeferencing_for_render"
                ]
            )

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
            failed_usage = json.loads(
                (
                    workspace
                    / "attempts/solve/attempt-0001/resource_usage.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(failed_usage["exception"], "RuntimeError")
            solve = run_mapper_solve(
                workspace, "colmap", runner=_ColmapRunner(names)
            )
            self.assertTrue(solve["all_keyframes_registered"])
            self.assertTrue(
                (
                    workspace
                    / "attempts/solve/attempt-0002/resource_usage.json"
                ).is_file()
            )

    def test_linux_resource_monitor_records_success_nonzero_and_safe_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solve_root = root / "attempts" / "solve"
            solve_root.mkdir(parents=True)
            safe = MapperConfig(
                process_nice=0,
                resource_sample_interval_s=0.01,
                minimum_available_memory_gb=0.001,
                minimum_free_space_gb=0.01,
                minimum_runtime_free_space_gb=0.005,
            )
            success_dir = solve_root / "attempt-0001"
            success_dir.mkdir()
            success = _run_monitored_mapper(
                (sys.executable, "-c", "pass"), success_dir, safe
            )
            self.assertEqual(success["returncode"], 0)
            self.assertFalse(success["safety_aborted"])

            failure_dir = solve_root / "attempt-0002"
            failure_dir.mkdir()
            with self.assertRaises(subprocess.CalledProcessError):
                _run_monitored_mapper(
                    (sys.executable, "-c", "raise SystemExit(7)"),
                    failure_dir,
                    safe,
                )
            failure = json.loads(
                (failure_dir / "resource_usage.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["returncode"], 7)

            abort_dir = solve_root / "attempt-0003"
            abort_dir.mkdir()
            abort_config = MapperConfig(
                process_nice=0,
                resource_sample_interval_s=0.01,
                minimum_available_memory_gb=1.0e9,
                low_memory_consecutive_samples=1,
                minimum_free_space_gb=0.01,
                minimum_runtime_free_space_gb=0.005,
            )
            with self.assertRaises(_MonitoredProcessInterrupted):
                _run_monitored_mapper(
                    (sys.executable, "-c", "import time; time.sleep(60)"),
                    abort_dir,
                    abort_config,
                )
            aborted = json.loads(
                (abort_dir / "resource_usage.json").read_text(encoding="utf-8")
            )
            self.assertTrue(aborted["safety_aborted"])
            self.assertIn("MemAvailable", aborted["safety_abort_reason"])

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

    @staticmethod
    def _consistent_covariance_residuals(scale: float = 1.0):
        covariance = np.array(
            [
                [4.0e-4, 1.2e-4, 0.4e-4],
                [1.2e-4, 9.0e-4, -0.6e-4],
                [0.4e-4, -0.6e-4, 16.0e-4],
            ],
            dtype=np.float64,
        )
        standardized = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.8, 0.8, 0.8],
                [-0.7, 0.4, -0.5],
            ]
            * 8,
            dtype=np.float64,
        )
        covariance_batch = np.repeat(
            (covariance * scale**2)[None], len(standardized), axis=0
        )
        residuals = standardized @ np.linalg.cholesky(covariance).T * scale
        return residuals, covariance_batch

    def test_covariance_gate_accepts_high_quality_consistent_residuals(self):
        residuals, covariance = self._consistent_covariance_residuals()
        quality = rtk_residual_quality(
            residuals,
            covariance,
            np.ones(len(residuals), dtype=bool),
            config=MapperConfig(rtk_covariance_gate_mode="enforce"),
        )
        self.assertTrue(quality["passed"])
        self.assertEqual(quality["degrees_of_freedom"], 3)
        self.assertTrue(
            quality["checks"]["median_holdout_rtk_mahalanobis_sq"][
                "passed"
            ]
        )

    def test_covariance_gate_accepts_degraded_but_consistent_accuracy(self):
        baseline_residuals, baseline_covariance = (
            self._consistent_covariance_residuals()
        )
        residuals, covariance = self._consistent_covariance_residuals(scale=10.0)
        permissive_absolute_caps = MapperConfig(
            max_rtk_median_error_m=0.5,
            max_rtk_p95_inlier_error_m=0.7,
            rtk_covariance_gate_mode="enforce",
        )
        baseline = rtk_residual_quality(
            baseline_residuals,
            baseline_covariance,
            np.ones(len(residuals), dtype=bool),
            config=permissive_absolute_caps,
        )
        degraded = rtk_residual_quality(
            residuals,
            covariance,
            np.ones(len(residuals), dtype=bool),
            config=permissive_absolute_caps,
        )
        self.assertTrue(degraded["passed"])
        np.testing.assert_allclose(
            degraded["mahalanobis_sq"]["values"],
            baseline["mahalanobis_sq"]["values"],
            atol=1e-12,
        )
        self.assertGreater(
            degraded["residual_m"]["median"],
            baseline["residual_m"]["median"] * 9.9,
        )

    def test_covariance_gate_rejects_systematic_mismatch(self):
        _, covariance = self._consistent_covariance_residuals()
        residuals = np.repeat([[0.20, 0.0, 0.0]], len(covariance), axis=0)
        quality = rtk_residual_quality(
            residuals,
            covariance,
            np.ones(len(residuals), dtype=bool),
            config=MapperConfig(
                max_rtk_median_error_m=1.0,
                max_rtk_p95_inlier_error_m=1.0,
                rtk_covariance_gate_mode="enforce",
            ),
        )
        self.assertFalse(quality["passed"])
        self.assertFalse(
            quality["checks"]["median_holdout_rtk_mahalanobis_sq"][
                "passed"
            ]
        )
        self.assertFalse(
            quality["checks"]["holdout_rtk_chi2_inlier_fraction"][
                "passed"
            ]
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
            Path(__file__).parents[1]
            / "src" / "rtk_splat" / "backends" / "mapper.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("rosbag", source)
        self.assertNotIn("sensor_msgs", source)
        self.assertNotIn("rtk_splat.core.segment", source)
        self.assertNotIn("rtk_splat.adapters.", source)


def _write_and_return(path: Path, payload: bytes) -> bytes:
    path.write_bytes(payload)
    return payload


if __name__ == "__main__":
    unittest.main()
