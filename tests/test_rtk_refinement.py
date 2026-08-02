import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from rtk_splat.backends.mapper import (
    MapperConfig,
    prepare_mapper_backend,
    run_image_registration,
    run_mapper_solve,
    run_quality_summary,
)
from rtk_splat.backends.rtk_refinement import (
    RtkRefinementConfig,
    build_rtk_refinement_command,
    prepare_rtk_refinement,
    refinement_export_context,
    run_rtk_refinement_quality,
    run_rtk_refinement_solve,
)
from rtk_splat.workflows.cli import _rtk_refinement_config, build_parser, main
from rtk_splat.frontends.artifact import (
    ArtifactError,
    create_frontend_seal,
    sha256_file,
)
from rtk_splat.diagnostics.refinement_ab import (
    compare_rtk_refinement_arms,
    write_report_content_identical,
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _option(command, name: str) -> str:
    return command[command.index(name) + 1]


def _model(path: Path, token: bytes = b"model") -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (path / name).write_bytes(token + name.encode())


def _analyzer_output(n_images: int, *, reprojection: float = 0.42) -> str:
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
Mean reprojection error: {reprojection}
"""


def _centers(frame_id: int, *, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    left = scale * np.array(
        [float(frame_id), 0.1 * frame_id**2, 0.03 * frame_id], dtype=np.float64
    )
    right = left + scale * np.array([0.12, 0.0, 0.0], dtype=np.float64)
    return left, right


def _frontend(
    root: Path,
    *,
    prior_offsets_m: dict[int, np.ndarray] | None = None,
) -> tuple[Path, list[str]]:
    artifact = root / "frontend"
    images = artifact / "images"
    stages = artifact / "stages"
    immutable_segment = root / "immutable-segment"
    immutable_segment.mkdir()
    images.mkdir(parents=True)
    stages.mkdir()

    rows: list[dict] = []
    names: list[str] = []
    for frame_id in range(6):
        row = {
            "frame_id": frame_id,
            "timestamp_ns": 1_000_000_000 + frame_id * 100_000_000,
        }
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
    _write_json(
        artifact / "rig_config.json",
        {"schema_version": 1, "rigs": [{"ref_sensor_id": 1}]},
    )
    _write_json(
        artifact / "provenance.json",
        {
            "schema_version": 1,
            "test": True,
            "contract_inputs": {"segment_root": str(immutable_segment.resolve())},
        },
    )
    _write_json(artifact / "quality.json", {"schema_version": 1, "n_frames": 6})
    (artifact / "pairs.txt").write_text(
        "left_000000.jpg right_000000.jpg\n", encoding="utf-8"
    )

    with sqlite3.connect(artifact / "database.db") as connection:
        connection.executescript(
            """
            CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY);
            CREATE TABLE rigs(rig_id INTEGER PRIMARY KEY);
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
            INSERT INTO cameras VALUES (1);
            INSERT INTO cameras VALUES (2);
            INSERT INTO rigs VALUES (1);
            """
        )
        for image_id, name in enumerate(names, start=1):
            camera_id = 1 if name.startswith("left_") else 2
            connection.execute(
                "INSERT INTO images VALUES (?, ?, ?)",
                (image_id, name, camera_id),
            )
            if name.startswith("left_"):
                frame_id = int(name.split("_")[1].split(".")[0])
                center, _ = _centers(frame_id)
                position = center + np.array([10.0, 20.0, 3.0])
                if prior_offsets_m and frame_id in prior_offsets_m:
                    position += prior_offsets_m[frame_id]
                covariance = np.diag([1e-4, 1e-4, 4e-4]).astype("<f8")
                connection.execute(
                    """
                    INSERT INTO pose_priors
                    VALUES (?, ?, 1, 0, ?, ?, NULL, 1)
                    """,
                    (
                        image_id,
                        image_id,
                        position.astype("<f8").tobytes(),
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
                "frontend_seal.json": sha256_file(artifact / "frontend_seal.json"),
            },
        },
    )
    return artifact, names


class _ColmapRunner:
    def __init__(self, all_names: list[str], *, refined_scale: float = 1.0):
        self.all_names = all_names
        self.refined_scale = refined_scale
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command, **kwargs):
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
            _model(Path(_option(command, "--output_path")), b"refined")
            return SimpleNamespace(returncode=0)
        if action == "model_converter":
            output = Path(_option(command, "--output_path"))
            input_path = Path(_option(command, "--input_path"))
            output.mkdir(exist_ok=True)
            refined = input_path.name == "refined_model"
            scale = self.refined_scale if refined else 1.0
            lines = ["# Image list"]
            for index, name in enumerate(self.all_names, start=1):
                frame_id = int(name.split("_")[1].split(".")[0])
                left, right = _centers(frame_id, scale=scale)
                center = left if name.startswith("left_") else right
                translation = -center
                camera_id = 1 if name.startswith("left_") else 2
                lines.extend(
                    [
                        f"{index} 1 0 0 0 {translation[0]} "
                        f"{translation[1]} {translation[2]} {camera_id} {name}",
                        "",
                    ]
                )
            (output / "images.txt").write_text("\n".join(lines) + "\n")
            (output / "cameras.txt").write_text(
                "# cameras\n"
                "1 PINHOLE 1000 800 500 500 500 400\n"
                "2 PINHOLE 1000 800 500 500 500 400\n",
                encoding="utf-8",
            )
            (output / "rigs.txt").write_text(
                "# rigs\n1 2 CAMERA 1 1 CAMERA 2 0 1 0 0 0 0 0.12 0 0\n",
                encoding="utf-8",
            )
            (output / "points3D.txt").write_text("# points\n", encoding="utf-8")
            return SimpleNamespace(returncode=0)
        if action == "model_analyzer":
            model = Path(_option(command, "--path"))
            n_images = 4 if "solve" in model.as_posix() else len(self.all_names)
            return SimpleNamespace(
                returncode=0,
                stdout=_analyzer_output(n_images),
            )
        raise AssertionError(f"unexpected command: {command}")


def _source_backend(
    root: Path,
    *,
    prior_offsets_m: dict[int, np.ndarray] | None = None,
) -> tuple[Path, list[str], _ColmapRunner]:
    frontend, names = _frontend(root, prior_offsets_m=prior_offsets_m)
    backend = prepare_mapper_backend(
        frontend,
        root / "backend",
        config=MapperConfig(
            minimum_free_space_gb=0.01,
            minimum_runtime_free_space_gb=0.005,
        ),
    )
    runner = _ColmapRunner(names)
    run_mapper_solve(backend, "colmap", runner=runner)
    run_image_registration(backend, "colmap", runner=runner)
    run_quality_summary(backend, "colmap", runner=runner)
    return backend, names, runner


class RtkRefinementTests(unittest.TestCase):
    def test_loss_configuration_is_strict_and_cli_override_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "cauchy.*trivial"):
            RtkRefinementConfig(prior_position_loss="huber")

        args = build_parser().parse_args(
            [
                "backend-refine-rtk",
                "--config",
                "unused.yaml",
                "--initialization-mode",
                "fresh",
                "--prior-position-loss",
                "trivial",
            ]
        )
        cfg = SimpleNamespace(
            rtk_refinement=SimpleNamespace(
                name="ignored-name", prior_position_loss_scale=2.795
            )
        )
        self.assertEqual(
            _rtk_refinement_config(cfg, args).prior_position_loss, "trivial"
        )
        self.assertEqual(
            _rtk_refinement_config(cfg, args).initialization_mode, "fresh"
        )
        with self.assertRaisesRegex(ValueError, "continuation.*fresh"):
            RtkRefinementConfig(initialization_mode="warm")
        with self.assertRaisesRegex(ValueError, "backend-refine-rtk"):
            main(
                [
                    "validate",
                    "--config",
                    "unused.yaml",
                    "--initialization-mode",
                    "fresh",
                ]
            )

    def test_prepare_removes_deterministic_holdouts_from_private_db_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend, _, _ = _source_backend(root)
            source_database_before = (backend / "database.db").read_bytes()
            source_model_before = {
                path.name: sha256_file(path)
                for path in (backend / "registered_model").iterdir()
            }

            first = prepare_rtk_refinement(backend, root / "refinement-a")
            second = prepare_rtk_refinement(backend, root / "refinement-b")

            self.assertEqual(
                (first / "prior_split.json").read_bytes(),
                (second / "prior_split.json").read_bytes(),
            )
            split = json.loads((first / "prior_split.json").read_text())
            holdouts = {
                row["name"] for row in split["records"] if row["role"] == "holdout"
            }
            calibration = {
                row["name"]
                for row in split["records"]
                if row["role"] == "calibration"
            }
            self.assertEqual(holdouts, {"left_000002.jpg", "left_000004.jpg"})
            self.assertEqual(len(calibration), 4)

            with sqlite3.connect(first / "database.db") as connection:
                optimizer_names = {
                    str(name)
                    for (name,) in connection.execute(
                        """
                        SELECT i.name FROM pose_priors AS p
                        JOIN images AS i ON i.image_id = p.corr_data_id
                        WHERE p.corr_sensor_type = 0
                        """
                    )
                }
            self.assertEqual(optimizer_names, calibration)
            self.assertFalse(optimizer_names & holdouts)
            self.assertEqual(
                (backend / "database.db").read_bytes(), source_database_before
            )
            self.assertEqual(
                {
                    path.name: sha256_file(path)
                    for path in (backend / "registered_model").iterdir()
                },
                source_model_before,
            )

    def test_command_is_exactly_bounded_and_freezes_metric_rig_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend, _, _ = _source_backend(root)
            workspace = prepare_rtk_refinement(backend, root / "refinement")
            command = build_rtk_refinement_command(workspace, "/opt/colmap")
            expected = (
                "/opt/colmap",
                "pose_prior_mapper",
                "--database_path",
                str(workspace / "database.db"),
                "--image_path",
                str(root / "frontend" / "images"),
                "--input_path",
                str(backend / "registered_model"),
                "--output_path",
                str(workspace / "refined_model.incomplete"),
                "--default_random_seed",
                "7",
                "--overwrite_priors_covariance",
                "0",
                "--use_robust_loss_on_prior_position",
                "1",
                "--prior_position_loss_scale",
                "2.795",
                "--Mapper.image_list_path",
                str(workspace / "all_images.txt"),
                "--Mapper.constant_camera_list_path",
                str(workspace / "constant_cameras.txt"),
                "--Mapper.constant_rig_list_path",
                str(workspace / "constant_rigs.txt"),
                "--Mapper.multiple_models",
                "0",
                "--Mapper.num_threads",
                "8",
                "--Mapper.random_seed",
                "7",
                "--Mapper.extract_colors",
                "0",
                "--Mapper.ba_refine_focal_length",
                "0",
                "--Mapper.ba_refine_principal_point",
                "0",
                "--Mapper.ba_refine_extra_params",
                "0",
                "--Mapper.ba_refine_sensor_from_rig",
                "0",
                "--Mapper.fix_existing_frames",
                "0",
                "--Mapper.ba_global_backend",
                "CERES",
                "--Mapper.ba_use_gpu",
                "0",
                "--Mapper.ba_global_max_num_iterations",
                "50",
                "--Mapper.ba_global_max_refinements",
                "3",
            )
            self.assertEqual(command, expected)

    def test_quadratic_arm_differs_by_only_the_robust_loss_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend, _, _ = _source_backend(root)
            control = prepare_rtk_refinement(backend, root / "control")
            candidate = prepare_rtk_refinement(
                backend,
                root / "candidate",
                config=RtkRefinementConfig(prior_position_loss="trivial"),
            )
            control_command = build_rtk_refinement_command(control, "/opt/colmap")
            candidate_command = build_rtk_refinement_command(
                candidate, "/opt/colmap"
            )
            normalized_control = [
                value.replace(str(control), "{workspace}")
                for value in control_command
            ]
            normalized_candidate = [
                value.replace(str(candidate), "{workspace}")
                for value in candidate_command
            ]
            differences = [
                (index, left, right)
                for index, (left, right) in enumerate(
                    zip(normalized_control, normalized_candidate, strict=True)
                )
                if left != right
            ]
            option_index = normalized_control.index(
                "--use_robust_loss_on_prior_position"
            )
            self.assertEqual(
                differences, [(option_index + 1, "1", "0")]
            )
            plan = json.loads(
                (candidate / "refinement_plan.json").read_text()
            )
            self.assertEqual(
                plan["colmap_pose_prior_model"]["position_loss"], "trivial"
            )
            self.assertFalse(
                plan["colmap_pose_prior_model"][
                    "prior_position_loss_scale_active"
                ]
            )

            with self.assertRaisesRegex(ArtifactError, "different inputs"):
                prepare_rtk_refinement(
                    backend,
                    control,
                    config=RtkRefinementConfig(
                        prior_position_loss="trivial"
                    ),
                )

    def test_fresh_command_only_omits_the_existing_input_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend, _, _ = _source_backend(root)
            continuation = prepare_rtk_refinement(
                backend,
                root / "continuation",
                config=RtkRefinementConfig(prior_position_loss="trivial"),
            )
            fresh = prepare_rtk_refinement(
                backend,
                root / "fresh",
                config=RtkRefinementConfig(
                    initialization_mode="fresh",
                    prior_position_loss="trivial",
                ),
            )
            continuation_command = list(
                build_rtk_refinement_command(continuation, "/opt/colmap")
            )
            fresh_command = list(
                build_rtk_refinement_command(fresh, "/opt/colmap")
            )
            normalized_continuation = [
                value.replace(str(continuation), "{workspace}")
                for value in continuation_command
            ]
            normalized_fresh = [
                value.replace(str(fresh), "{workspace}")
                for value in fresh_command
            ]
            input_index = normalized_continuation.index("--input_path")
            del normalized_continuation[input_index : input_index + 2]
            self.assertEqual(normalized_continuation, normalized_fresh)
            self.assertNotIn("--input_path", fresh_command)

    def test_legacy_plan_without_loss_field_remains_resumable_as_cauchy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend, names, _ = _source_backend(root)
            workspace = prepare_rtk_refinement(backend, root / "refinement")
            plan_path = workspace / "refinement_plan.json"
            plan = json.loads(plan_path.read_text())
            for field in (
                "prior_position_loss",
                "initialization_mode",
                "fresh_min_mean_track_length",
                "fresh_min_mean_observations_per_image",
            ):
                plan["config"].pop(field)
            plan["colmap_pose_prior_model"].pop("initialization_mode")
            plan["colmap_pose_prior_model"].pop(
                "input_model_supplied_to_mapper"
            )
            _write_json(plan_path, plan)
            marker_path = workspace / "stages" / "prepare.json"
            marker = json.loads(marker_path.read_text())
            marker["outputs"]["refinement_plan.json"] = sha256_file(plan_path)
            _write_json(marker_path, marker)

            command = build_rtk_refinement_command(workspace, "/opt/colmap")
            option_index = command.index("--use_robust_loss_on_prior_position")
            self.assertEqual(command[option_index + 1], "1")
            runner = _ColmapRunner(names)
            run_rtk_refinement_solve(workspace, "colmap", runner=runner)
            first = run_rtk_refinement_quality(
                workspace, "colmap", runner=runner
            )
            resumed = run_rtk_refinement_quality(
                workspace, "colmap", runner=runner
            )
            self.assertEqual(first, resumed)

    def test_loss_ab_comparator_proves_parity_and_resumes_identically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend, names, _ = _source_backend(root)
            # Deliberately use an arm path that prefixes ``backend``. Command
            # normalization must not rewrite the independent source path.
            control = prepare_rtk_refinement(backend, root / "b")
            candidate = prepare_rtk_refinement(
                backend,
                root / "candidate",
                config=RtkRefinementConfig(prior_position_loss="trivial"),
            )
            for workspace in (control, candidate):
                runner = _ColmapRunner(names)
                run_rtk_refinement_solve(
                    workspace, "/bin/true", runner=runner
                )
                run_rtk_refinement_quality(
                    workspace, "/bin/true", runner=runner
                )

            report = compare_rtk_refinement_arms(
                control,
                candidate,
                expected_frames=6,
                expected_evaluation_priors=6,
                expected_calibration_priors=4,
                expected_holdout_priors=2,
            )
            self.assertTrue(report["integrity"]["passed"])
            self.assertEqual(
                report["integrity"]["command_differences"][0]["control"],
                "1",
            )
            self.assertEqual(
                report["integrity"]["command_differences"][0]["candidate"],
                "0",
            )
            output = root / "reports" / "ab.json"
            first = write_report_content_identical(output, report)
            second = write_report_content_identical(output, report)
            self.assertEqual(first, second)

    def test_loss_ab_rejects_different_evaluation_gate_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend, names, _ = _source_backend(root)
            control = prepare_rtk_refinement(backend, root / "control")
            candidate = prepare_rtk_refinement(
                backend,
                root / "candidate",
                config=RtkRefinementConfig(prior_position_loss="trivial"),
            )
            for workspace in (control, candidate):
                runner = _ColmapRunner(names)
                run_rtk_refinement_solve(
                    workspace, "/bin/true", runner=runner
                )
                run_rtk_refinement_quality(
                    workspace, "/bin/true", runner=runner
                )

            plan_path = candidate / "refinement_plan.json"
            plan = json.loads(plan_path.read_text())
            plan["mapper_evaluation_config"]["max_rtk_median_error_m"] = 999.0
            _write_json(plan_path, plan)
            marker_path = candidate / "stages" / "prepare.json"
            marker = json.loads(marker_path.read_text())
            marker["outputs"]["refinement_plan.json"] = sha256_file(plan_path)
            _write_json(marker_path, marker)

            with self.assertRaisesRegex(
                ArtifactError, "evaluation config differs"
            ):
                compare_rtk_refinement_arms(control, candidate)

    def test_citrus_loss_launcher_runs_only_candidate_refinement(self):
        repository = Path(__file__).resolve().parents[1]
        script = (
            repository
            / "scripts"
            / "experiments"
            / "citrusfarm_rtk_loss_b.sh"
        )
        syntax = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        help_result = subprocess.run(
            ["bash", str(script), "--help"], capture_output=True, text=True
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        contents = script.read_text(encoding="utf-8")
        self.assertIn("backend-refine-rtk", contents)
        self.assertIn("--prior-position-loss trivial", contents)
        self.assertIn('if [[ ! -e "$CANDIDATE" ]]', contents)
        self.assertIn("flock -n 9", contents)
        self.assertIn("CONTROL_COLMAP", contents)
        self.assertNotIn("backend-export", contents)
        self.assertNotIn("workflows.cli cloud", contents)
        self.assertNotIn("workflows.cli train", contents)

    def test_prepare_interruption_removes_unpublished_database_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend, _, _ = _source_backend(root)
            destination = root / "refinement"

            def interrupt_snapshot(_source, staging, **_kwargs):
                staging = Path(staging)
                staging.mkdir(parents=True)
                (staging / "partial-database.db").write_bytes(b"partial")
                raise KeyboardInterrupt

            with mock.patch(
                "rtk_splat.backends.rtk_refinement.create_database_snapshot",
                side_effect=interrupt_snapshot,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    prepare_rtk_refinement(backend, destination)

            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".refinement.prepare-*")), [])

    def test_prepare_control_files_are_hash_verified_before_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend, _, _ = _source_backend(root)
            workspace = prepare_rtk_refinement(backend, root / "refinement")
            (workspace / "all_images.txt").write_text(
                "left_000000.jpg\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ArtifactError, "sealed prepare output"):
                build_rtk_refinement_command(workspace, "/opt/colmap")

    def test_injected_solve_and_quality_accept_metric_clean_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend, names, _ = _source_backend(root)
            workspace = prepare_rtk_refinement(backend, root / "refinement")
            runner = _ColmapRunner(names)

            solve = run_rtk_refinement_solve(workspace, "colmap", runner=runner)
            quality = run_rtk_refinement_quality(workspace, "colmap", runner=runner)

            self.assertTrue(solve["all_images_retained"])
            self.assertTrue(quality["passed"])
            self.assertEqual(quality["registration_fraction"], 1.0)
            self.assertAlmostEqual(
                quality["checks"]["trajectory_similarity_scale"]["value"], 1.0
            )
            self.assertAlmostEqual(
                quality["checks"]["stereo_baseline_max_change_m"]["value"], 0.0
            )
            self.assertEqual(quality["rtk_holdout"]["n_holdout"], 2)
            context = refinement_export_context(workspace)
            self.assertEqual(context["text_model"], workspace / "refined_text")
            self.assertTrue(context["pose_priors_constrain_refinement"])
            self.assertIn(
                "pose_prior_mapper", {command[1] for command in runner.commands}
            )

    def test_fresh_quality_uses_absolute_graph_gates_and_export_is_truthful(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend, names, _ = _source_backend(root)
            workspace = prepare_rtk_refinement(
                backend,
                root / "fresh",
                config=RtkRefinementConfig(
                    initialization_mode="fresh",
                    prior_position_loss="trivial",
                    fresh_min_mean_track_length=3.0,
                    fresh_min_mean_observations_per_image=60.0,
                ),
            )
            runner = _ColmapRunner(names)
            solve = run_rtk_refinement_solve(
                workspace, "colmap", runner=runner
            )
            quality = run_rtk_refinement_quality(
                workspace, "colmap", runner=runner
            )

            self.assertTrue(quality["passed"])
            self.assertEqual(solve["initialization_mode"], "fresh")
            self.assertFalse(solve["input_model_supplied_to_mapper"])
            self.assertNotIn("--input_path", solve["command"])
            self.assertIn(
                "fresh_mean_track_length_absolute", quality["checks"]
            )
            self.assertIn(
                "fresh_mean_observations_per_image_absolute",
                quality["checks"],
            )
            self.assertNotIn("track_length_retention", quality["checks"])
            context = refinement_export_context(workspace)
            self.assertTrue(context["pose_priors_constrain_mapper"])
            self.assertFalse(context["pose_priors_constrain_refinement"])
            self.assertEqual(
                context["provenance"]["initialization_mode"], "fresh"
            )

    def test_fresh_absolute_graph_gate_rejects_weak_track_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend, names, _ = _source_backend(root)
            workspace = prepare_rtk_refinement(
                backend,
                root / "fresh",
                config=RtkRefinementConfig(
                    initialization_mode="fresh",
                    prior_position_loss="trivial",
                ),
            )
            runner = _ColmapRunner(names)
            run_rtk_refinement_solve(workspace, "colmap", runner=runner)
            quality = run_rtk_refinement_quality(
                workspace, "colmap", runner=runner
            )

            self.assertFalse(quality["passed"])
            self.assertFalse(
                quality["checks"][
                    "fresh_mean_observations_per_image_absolute"
                ]["passed"]
            )
            with self.assertRaisesRegex(ArtifactError, "failed acceptance"):
                refinement_export_context(workspace)

    def test_quality_report_tamper_cannot_bypass_export_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend, names, _ = _source_backend(root)
            workspace = prepare_rtk_refinement(backend, root / "refinement")
            runner = _ColmapRunner(names)
            run_rtk_refinement_solve(workspace, "colmap", runner=runner)
            run_rtk_refinement_quality(workspace, "colmap", runner=runner)
            report_path = workspace / "reports" / "quality.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["passed"] = False
            _write_json(report_path, report)

            with self.assertRaisesRegex(ArtifactError, "sealed quality output"):
                refinement_export_context(workspace)

    def test_metric_scale_change_is_rejected_and_cannot_be_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend, names, _ = _source_backend(root)
            workspace = prepare_rtk_refinement(backend, root / "refinement")
            runner = _ColmapRunner(names, refined_scale=1.02)
            run_rtk_refinement_solve(workspace, "colmap", runner=runner)

            quality = run_rtk_refinement_quality(workspace, "colmap", runner=runner)

            self.assertFalse(quality["passed"])
            self.assertFalse(
                quality["checks"]["trajectory_similarity_scale"]["passed"]
            )
            self.assertFalse(
                quality["checks"]["stereo_baseline_max_change_m"]["passed"]
            )
            with self.assertRaisesRegex(ArtifactError, "failed acceptance"):
                refinement_export_context(workspace)

    def test_source_failure_without_heldout_improvement_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend, names, _ = _source_backend(
                root,
                prior_offsets_m={
                    2: np.array([0.5, 0.0, 0.0]),
                    4: np.array([0.5, 0.0, 0.0]),
                },
            )
            workspace = prepare_rtk_refinement(backend, root / "refinement")
            runner = _ColmapRunner(names)
            run_rtk_refinement_solve(workspace, "colmap", runner=runner)

            quality = run_rtk_refinement_quality(workspace, "colmap", runner=runner)

            self.assertFalse(quality["rtk_holdout"]["source_passed"])
            self.assertFalse(quality["rtk_holdout"]["refined_passed"])
            self.assertFalse(
                quality["checks"]["heldout_improvement_when_source_failed_m"][
                    "passed"
                ]
            )
            self.assertFalse(quality["passed"])


if __name__ == "__main__":
    unittest.main()
