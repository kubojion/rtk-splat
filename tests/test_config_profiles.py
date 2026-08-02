import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rtk_splat.core.configio import _deep_merge, load_config
from rtk_splat.workflows.cli import _mapper_config


class ConfigProfileTests(unittest.TestCase):
    def _layout(self, root: Path) -> tuple[Path, Path]:
        robots = root / "robots"
        sequences = root / "sequences"
        robots.mkdir()
        sequences.mkdir()
        return robots, sequences

    def test_profile_is_deep_merged_then_sequence_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            robots, sequences = self._layout(root)
            (robots / "field_robot.yaml").write_text(
                "adapter: ros2_zed_ublox\n"
                "paths:\n"
                "  workdir: /profile/default\n"
                "topics:\n"
                "  left_image: /left\n"
                "  right_image: /right\n"
                "pose:\n"
                "  source: rtk_dual_antenna\n"
                "  mount:\n"
                "    forward_m: 3.1\n"
                "    up_m: 1.3\n"
            )
            sequence = sequences / "field.yaml"
            sequence.write_text(
                "robot: field_robot\n"
                "paths:\n"
                "  workdir: ~/runs/field\n"
                "  bags: [~/bags/field]\n"
                "pose:\n"
                "  mount:\n"
                "    up_m: 1.4\n"
                "train:\n"
                "  run_name: field_01\n"
            )

            cfg = load_config(sequence)

            self.assertEqual(cfg.adapter, "ros2_zed_ublox")
            self.assertEqual(cfg.robot, "field_robot")
            self.assertEqual(cfg.topics.left_image, "/left")
            self.assertEqual(cfg.pose.source, "rtk_dual_antenna")
            self.assertEqual(cfg.pose.mount.forward_m, 3.1)
            self.assertEqual(cfg.pose.mount.up_m, 1.4)
            self.assertEqual(cfg.paths.workdir, Path("~/runs/field").expanduser())
            self.assertEqual(
                cfg.paths.bags, [Path("~/bags/field").expanduser()]
            )

    def test_deep_merge_does_not_mutate_either_input(self):
        profile = {"pose": {"mount": {"up_m": 1.3}}, "items": [1]}
        sequence = {"pose": {"mount": {"up_m": 1.4}}}

        merged = _deep_merge(profile, sequence)
        merged["items"].append(2)

        self.assertEqual(profile, {
            "pose": {"mount": {"up_m": 1.3}},
            "items": [1],
        })
        self.assertEqual(sequence, {"pose": {"mount": {"up_m": 1.4}}})

    def test_explicit_config_root_works_outside_sequences_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            robots, _ = self._layout(root)
            (robots / "robot.yaml").write_text(
                "paths: {workdir: /default}\n"
                "pose: {source: gnss_course}\n"
            )
            elsewhere = root / "runs"
            elsewhere.mkdir()
            sequence = elsewhere / "one.yaml"
            sequence.write_text(
                "robot: robot\npaths: {workdir: /run}\n"
            )

            cfg = load_config(sequence, config_root=root)

            self.assertEqual(cfg.pose.source, "gnss_course")
            self.assertEqual(cfg.paths.workdir, Path("/run"))

    def test_missing_profile_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, sequences = self._layout(root)
            sequence = sequences / "missing.yaml"
            sequence.write_text(
                "robot: absent\npaths: {workdir: /run}\n"
            )
            with self.assertRaisesRegex(
                ValueError, "robot profile 'absent' not found"
            ):
                load_config(sequence)

    def test_profile_cycle_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            robots, sequences = self._layout(root)
            (robots / "a.yaml").write_text("robot: b\n")
            (robots / "b.yaml").write_text("robot: a\n")
            sequence = sequences / "cycle.yaml"
            sequence.write_text("robot: a\npaths: {workdir: /run}\n")
            with self.assertRaisesRegex(ValueError, "robot profile cycle"):
                load_config(sequence)

    def test_invalid_profile_yaml_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            robots, sequences = self._layout(root)
            (robots / "bad.yaml").write_text("topics: [\n")
            sequence = sequences / "bad.yaml"
            sequence.write_text("robot: bad\npaths: {workdir: /run}\n")
            with self.assertRaisesRegex(ValueError, "invalid YAML"):
                load_config(sequence)

    def test_monolithic_config_keeps_generic_path_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "legacy.yaml"
            config.write_text(
                "paths:\n"
                "  workdir: ~/legacy-work\n"
                "  bags: [~/legacy-bag]\n"
                "  ublox_msgs_dir: ~/msgs\n"
                "pose:\n"
                "  source: rtk_dual_antenna\n"
            )

            cfg = load_config(config)

            self.assertEqual(
                cfg.paths.workdir, Path("~/legacy-work").expanduser()
            )
            self.assertEqual(
                cfg.paths.bags, [Path("~/legacy-bag").expanduser()]
            )
            # Dataset-specific paths remain adapter-owned. The generic loader
            # preserves their value without knowing their semantics.
            self.assertEqual(cfg.paths.ublox_msgs_dir, "~/msgs")
            self.assertEqual(cfg.pose.source, "rtk_dual_antenna")

    def test_deprecated_global_mapper_section_is_rejected_loudly(self):
        cfg = SimpleNamespace(
            global_mapper=SimpleNamespace(max_num_tracks=60_000)
        )
        with self.assertRaisesRegex(ValueError, "deprecated 'global_mapper:'"):
            _mapper_config(cfg, SimpleNamespace(backend=None))

    def test_headland_reproduction_uses_canonical_bounded_mapper(self):
        config = (
            Path(__file__).resolve().parents[1]
            / "configs/reproductions/headland_stereo_ba.yaml"
        )
        cfg = load_config(config)
        self.assertFalse(hasattr(cfg, "global_mapper"))
        mapper = _mapper_config(cfg, SimpleNamespace(backend=None))
        self.assertEqual(mapper.keep_max_num_tracks, 60_000)
        self.assertEqual(mapper.track_required_tracks_per_view, 1_000)
        self.assertTrue(mapper.skip_retriangulation)
        self.assertFalse(mapper.gp_use_gpu)
        self.assertFalse(mapper.ba_ceres_use_gpu)
        # Read-only replay of the accepted reduced-global model through the
        # current covariance-aware, five-block temporal holdout evaluator.
        golden_holdout = {
            "median_m": 0.09891653002379946,
            "p95_inliers_m": 0.12088889251051756,
            "inlier_fraction": 1.0,
        }
        self.assertLessEqual(
            golden_holdout["median_m"], mapper.max_rtk_median_error_m
        )
        self.assertLessEqual(
            golden_holdout["p95_inliers_m"],
            mapper.max_rtk_p95_inlier_error_m,
        )
        self.assertGreaterEqual(
            golden_holdout["inlier_fraction"], mapper.min_rtk_inlier_fraction
        )
        # Keep the reproduction allowance bounded: these are not permissive
        # fallback settings for a failed or grossly mis-georeferenced solve.
        self.assertLessEqual(mapper.max_rtk_median_error_m, 0.12)
        self.assertLessEqual(mapper.max_rtk_p95_inlier_error_m, 0.15)
        self.assertGreaterEqual(mapper.min_rtk_inlier_fraction, 0.95)


if __name__ == "__main__":
    unittest.main()
