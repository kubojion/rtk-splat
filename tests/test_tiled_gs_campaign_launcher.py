import argparse
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/experiments/tiled_gs_campaign_v1.py"
SPEC = importlib.util.spec_from_file_location("tiled_gs_campaign_v1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TiledGsCampaignLauncherTests(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            config=Path("/config.yaml"),
            workdir=Path("/work"),
            segment=Path("/segment"),
            expected_frames=123,
            pose_name="diagnostic-poses-v1",
            pose_artifact_root=Path("/poses"),
            tile_plan=Path("/plan"),
            allow_failed_georeferencing_for_render=True,
            train_iters=65_000,
        )

    def test_binding_and_run_name_are_explicit(self):
        self.assertEqual(
            MODULE._binding("tile-a=/runs/a"),
            ("tile-a", Path("/runs/a")),
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            MODULE._binding("tile-a")
        self.assertEqual(
            MODULE._run_name("diagnostic-{tile_id}-v1", "tile-a"),
            "diagnostic-tile-a-v1",
        )
        with self.assertRaisesRegex(ValueError, "must contain"):
            MODULE._run_name("fixed-name", "tile-a")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            MODULE._run_name("../{tile_id}", "tile-a")

    def test_commands_preserve_diagnostic_tile_contract(self):
        args = self._args()
        cloud = MODULE._stage_command(args, "cloud", "tile-a")
        train = MODULE._stage_command(
            args, "train", "tile-a", run_name="diagnostic-tile-a-v1"
        )
        self.assertEqual(cloud[:4], [
            sys.executable,
            "-m",
            "rtk_splat.workflows.cli",
            "cloud",
        ])
        for command in (cloud, train):
            self.assertIn("--tile-plan", command)
            self.assertIn("--tile-id", command)
            self.assertIn("--pose-artifact-root", command)
            self.assertIn("--allow-failed-georeferencing-for-render", command)
        self.assertNotIn("--run-name", cloud)
        self.assertNotIn("--train-iters", train)
        self.assertEqual(
            train[train.index("--run-name") + 1],
            "diagnostic-tile-a-v1",
        )

    def test_run_path_is_tile_isolated(self):
        self.assertEqual(
            MODULE._run_path(
                Path("/work"), "plan-v1", "tile-a", "run-v1"
            ),
            Path("/work/tile_runs/plan-v1/tile-a/run-v1"),
        )

    def test_train_iteration_contract_checks_authored_value_without_override(self):
        cfg = SimpleNamespace(train=SimpleNamespace(iterations=65_000))
        MODULE._verify_authored_train_iterations(cfg, 65_000)
        for value in (64_999, True, "65000", None):
            cfg.train.iterations = value
            with self.assertRaisesRegex(ValueError, "authored numeric config"):
                MODULE._verify_authored_train_iterations(cfg, 65_000)


if __name__ == "__main__":
    unittest.main()
