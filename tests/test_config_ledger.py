import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rtk_splat.adapters.sampling import resolve_frame_stride
from rtk_splat.core.runtime_resolution import (
    configuration_evidence,
    runtime_resolution_plain,
    set_runtime_cli_override,
)
from rtk_splat.workflows.config_ledger import (
    authored_config_plain,
    commit_config_ledger,
    prepare_config_ledger,
)
from rtk_splat.workflows.configio import load_config
from rtk_splat.workflows import cli


class ConfigLedgerTests(unittest.TestCase):
    def _layout(self, root: Path) -> tuple[Path, Path, Path]:
        profiles = root / "profiles"
        sequences = root / "sequences"
        profiles.mkdir()
        sequences.mkdir()
        profile = profiles / "quality.yaml"
        profile.write_text(
            "derivation:\n"
            "  frame_sampling: {target_spacing_m: 0.10}\n"
        )
        workdir = root / "work"
        sequence = sequences / "field.yaml"
        sequence.write_text(
            f"profile: quality\npaths: {{workdir: {workdir}}}\n"
            "segment: {frame_stride: auto}\n"
        )
        return profile, sequence, workdir

    def test_separate_invocations_accumulate_without_applying_old_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile, sequence, workdir = self._layout(Path(tmp))
            first = load_config(sequence)
            authored = authored_config_plain(first)
            prepare_config_ledger(
                first, stage="ingest", authored_config=authored
            )
            resolve_frame_stride(
                first,
                measured_camera_rate_hz=15.0,
                measured_median_speed_m_s=0.75,
            )
            ledger_path = commit_config_ledger(
                first, authored_config=authored
            )

            second = load_config(sequence)
            second_authored = authored_config_plain(second)
            self.assertEqual(second.segment.frame_stride, "auto")
            prepare_config_ledger(
                second, stage="validate", authored_config=second_authored
            )

            # Hydration carries proof forward but never turns an authored auto
            # control into a stale operational value.
            self.assertEqual(second.segment.frame_stride, "auto")
            record = runtime_resolution_plain(second)["derivations"][
                "frame_stride"
            ]
            self.assertEqual(record["chosen_value"], 2)
            commit_config_ledger(second, authored_config=second_authored)

            ledger = json.loads(ledger_path.read_text())
            self.assertEqual(ledger["authored_config"], authored)
            self.assertEqual(len(ledger["stages"]), 2)
            self.assertIn("frame_stride", ledger["derivations"])

            # A comment-only source change leaves merged values identical but
            # still invalidates the source identity.
            profile.write_text(profile.read_text() + "# changed source\n")
            changed = load_config(sequence)
            with self.assertRaisesRegex(ValueError, "source_files"):
                prepare_config_ledger(
                    changed,
                    stage="validate",
                    authored_config=authored_config_plain(changed),
                )
            self.assertEqual(
                ledger_path,
                workdir / "config_artifacts/resolved_config.json",
            )

    def test_cli_override_is_separate_and_changes_effective_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, sequence, _ = self._layout(Path(tmp))
            cfg = load_config(sequence)
            authored = authored_config_plain(cfg)
            prepare_config_ledger(
                cfg,
                stage="train",
                authored_config=authored,
                stage_overrides={"train_iters": 42000},
            )
            before = configuration_evidence(cfg)["effective_config_sha256"]
            cfg.train = type(cfg)(iterations="auto")
            set_runtime_cli_override(
                cfg,
                "train_iterations",
                cfg.train.iterations,
                42000,
                option="--train-iters",
                config_path="train.iterations",
            )
            cfg.train.iterations = 42000
            evidence = configuration_evidence(cfg)

            self.assertNotEqual(before, evidence["effective_config_sha256"])
            self.assertEqual(
                evidence["stage_overrides"], {"train_iters": 42000}
            )
            self.assertEqual(
                evidence["runtime_resolution"]["cli_overrides"]
                ["train_iterations"]["option"],
                "--train-iters",
            )
            self.assertEqual(authored["segment"]["frame_stride"], "auto")

    def test_failed_georeferencing_render_authorization_is_explicit(self):
        args = cli.build_parser().parse_args(
            [
                "backend-export",
                "--config",
                "/does/not/need/to/exist.yaml",
                "--allow-failed-georeferencing-for-render",
            ]
        )
        self.assertTrue(args.allow_failed_georeferencing_for_render)
        self.assertEqual(
            cli._stage_overrides(args)[
                "allow_failed_georeferencing_for_render"
            ],
            True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "valid only for backend-export, tiles-plan, cloud, and train",
        ):
            cli.main(
                [
                    "validate",
                    "--config",
                    "/does/not/exist.yaml",
                    "--allow-failed-georeferencing-for-render",
                ]
            )

    def test_diagnostic_render_requires_separate_explicit_names(self):
        cfg = SimpleNamespace(
            mapper=SimpleNamespace(
                name="production-backend",
                pose_artifact_name="production-pose",
            ),
            pose=SimpleNamespace(artifact="production-pose"),
            train=SimpleNamespace(run_name="production-run"),
        )

        def args(stage, pose_name=None, run_name=None):
            return cli.build_parser().parse_args([
                stage,
                "--config", "/unused.yaml",
                "--allow-failed-georeferencing-for-render",
                *([] if pose_name is None else ["--pose-name", pose_name]),
                *([] if run_name is None else ["--run-name", run_name]),
            ])

        with self.assertRaisesRegex(ValueError, "explicit --pose-name"):
            cli._validate_diagnostic_render_names(args("cloud"), cfg)
        with self.assertRaisesRegex(ValueError, "must differ"):
            cli._validate_diagnostic_render_names(
                args("cloud", "production-pose"), cfg
            )
        with self.assertRaisesRegex(ValueError, "explicit --run-name"):
            cli._validate_diagnostic_render_names(
                args("train", "diagnostic-pose"), cfg
            )
        with self.assertRaisesRegex(ValueError, "must differ"):
            cli._validate_diagnostic_render_names(
                args("train", "diagnostic-pose", "production-run"), cfg
            )
        cli._validate_diagnostic_render_names(
            args("train", "diagnostic-pose", "diagnostic-run"), cfg
        )

    def test_failed_stage_does_not_create_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, sequence, workdir = self._layout(Path(tmp))
            cfg = load_config(sequence)
            prepare_config_ledger(
                cfg,
                stage="ingest",
                authored_config=authored_config_plain(cfg),
            )
            self.assertFalse(
                (workdir / "config_artifacts/resolved_config.json").exists()
            )

    def test_tile_plan_flags_are_stage_scoped_and_recorded(self):
        args = cli.build_parser().parse_args([
            "tiles-plan", "--config", "/unused.yaml",
            "--pose-artifact-root", "/poses", "--tile-plan-name", "two-v1",
            "--tile-count", "2", "--tile-max-tiles", "128",
            "--tile-min-visibility-fraction", "0.075",
        ])
        self.assertEqual(args.tile_count, 2)
        self.assertEqual(args.tile_max_tiles, 128)
        self.assertEqual(
            cli._stage_overrides(args)["pose_artifact_root"], "/poses"
        )
        self.assertEqual(
            cli._stage_overrides(args)["tile_max_tiles"], 128
        )
        self.assertEqual(
            cli._stage_overrides(args)["tile_min_visibility_fraction"],
            0.075,
        )
        with self.assertRaisesRegex(ValueError, "valid only for tiles-plan"):
            cli.main([
                "validate", "--config", "/does/not/exist.yaml",
                "--tile-count", "2",
            ])
        with self.assertRaisesRegex(ValueError, "must be positive"):
            cli.main([
                "tiles-plan", "--config", "/does/not/exist.yaml",
                "--tile-count", "0",
            ])
        with self.assertRaisesRegex(ValueError, r"must be in \[1, 4096\]"):
            cli.main([
                "tiles-plan", "--config", "/does/not/exist.yaml",
                "--tile-max-tiles", "4097",
            ])
        with self.assertRaisesRegex(ValueError, "valid only for tiles-plan"):
            cli.main([
                "validate", "--config", "/does/not/exist.yaml",
                "--tile-max-tiles", "128",
            ])
        with self.assertRaisesRegex(ValueError, r"must be in \[0, 1\]"):
            cli.main([
                "tiles-plan", "--config", "/does/not/exist.yaml",
                "--tile-min-visibility-fraction", "1.01",
            ])
        with self.assertRaisesRegex(ValueError, "valid only for tiles-plan"):
            cli.main([
                "validate", "--config", "/does/not/exist.yaml",
                "--tile-min-visibility-fraction", "0.075",
            ])
        tiled = cli.build_parser().parse_args([
            "train", "--config", "/unused.yaml",
            "--tile-plan", "/plans/two-v1", "--tile-id", "tile-0000",
            "--pose-artifact-root", "/poses",
        ])
        overrides = cli._stage_overrides(tiled)
        self.assertEqual(overrides["tile_plan"], "/plans/two-v1")
        self.assertEqual(overrides["tile_id"], "tile-0000")
        with self.assertRaisesRegex(ValueError, "supplied together"):
            cli.main([
                "train", "--config", "/does/not/exist.yaml",
                "--tile-id", "tile-0000",
            ])
        with self.assertRaisesRegex(ValueError, "cloud and train"):
            cli.main([
                "validate", "--config", "/does/not/exist.yaml",
                "--tile-plan", "/plans/two-v1", "--tile-id", "tile-0000",
            ])

    def test_tile_plan_policy_overrides_reach_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "tile_plan_artifacts" / "automatic-v1"
            artifact.mkdir(parents=True)
            (artifact / "tile_plan.json").write_text(
                json.dumps({"summary": {"n_tiles": 1}, "tiles": []})
            )
            cfg = SimpleNamespace(
                paths=SimpleNamespace(workdir=root),
                pose=SimpleNamespace(artifact="pose"),
                tiles=SimpleNamespace(name="automatic-v1", max_tiles=64),
            )
            args = SimpleNamespace(
                pose_name=None,
                pose_artifact_root=None,
                tile_max_tiles=128,
                tile_min_visibility_fraction=0.075,
                tile_plan_name=None,
                tile_count=None,
                allow_failed_georeferencing_for_render=False,
            )

            def build(_reader, received_cfg, **_kwargs):
                self.assertEqual(received_cfg.tiles.max_tiles, 128)
                self.assertEqual(
                    received_cfg.tiles.min_visibility_fraction, 0.075
                )
                return artifact

            with mock.patch.object(cli, "_reader", return_value=object()), \
                    mock.patch(
                        "rtk_splat.workflows.tiles.build_tile_plan",
                        side_effect=build,
                    ):
                cli.cmd_tiles_plan(cfg, args)

    def test_cli_commits_only_after_successful_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workdir = root / "successful"
            config = root / "config.yaml"
            config.write_text(f"paths: {{workdir: {workdir}}}\n")
            observed = {}

            def success(cfg, _args):
                path = Path(cfg.config_ledger.path)
                observed["prepared"] = True
                observed["absent_during_stage"] = not path.exists()

            with mock.patch.dict(cli.COMMANDS, {"validate": success}):
                self.assertEqual(
                    cli.main(["validate", "--config", str(config)]), 0
                )
            self.assertTrue(observed["prepared"])
            self.assertTrue(observed["absent_during_stage"])
            self.assertTrue(
                (workdir / "config_artifacts/resolved_config.json").is_file()
            )

            failed_workdir = root / "failed"
            failed = root / "failed.yaml"
            failed.write_text(f"paths: {{workdir: {failed_workdir}}}\n")

            def failure(_cfg, _args):
                raise RuntimeError("synthetic stage failure")

            with mock.patch.dict(cli.COMMANDS, {"validate": failure}):
                with self.assertRaisesRegex(RuntimeError, "synthetic"):
                    cli.main(["validate", "--config", str(failed)])
            self.assertFalse(
                (failed_workdir / "config_artifacts/resolved_config.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
