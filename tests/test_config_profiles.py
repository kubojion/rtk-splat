import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rtk_splat.workflows.configio import _deep_merge, load_config
from rtk_splat.workflows.cli import _mapper_config


class ConfigProfileTests(unittest.TestCase):
    def _layout(self, root: Path) -> tuple[Path, Path]:
        robots = root / "robots"
        sequences = root / "sequences"
        robots.mkdir()
        sequences.mkdir()
        return robots, sequences

    def test_quality_profile_robot_sequence_precedence_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            robots, sequences = self._layout(root)
            profiles = root / "profiles"
            profiles.mkdir()
            (profiles / "quality.yaml").write_text(
                "depth: {min_z_m: 0.5, max_z_m: auto}\n"
                "train: {iterations: auto, max_gaussians: auto}\n"
            )
            (robots / "robot.yaml").write_text(
                "adapter: ros2_zed_ublox\n"
                "pose: {source: rtk_dual_antenna}\n"
            )
            sequence = sequences / "run.yaml"
            sequence.write_text(
                "profile: quality\n"
                "robot: robot\n"
                "paths: {workdir: /sequence}\n"
                "depth: {max_z_m: 12.0}\n"
            )

            cfg = load_config(sequence)

            self.assertEqual(cfg.paths.workdir, Path("/sequence"))
            self.assertEqual(cfg.depth.min_z_m, 0.5)
            self.assertEqual(cfg.depth.max_z_m, 12.0)
            self.assertEqual(cfg.train.iterations, "auto")
            sources = vars(cfg.runtime_resolution.config_sources)
            self.assertEqual(Path(sources["profile"]), profiles / "quality.yaml")
            self.assertEqual(Path(sources["robot"]), robots / "robot.yaml")
            self.assertEqual(Path(sources["sequence"]), sequence)
            origins = vars(cfg.runtime_resolution.origins)
            self.assertEqual(origins["depth_max_z_m"].layer, "sequence")
            self.assertEqual(origins["depth_max_z_m"].authored_value, 12.0)
            self.assertEqual(origins["train_iterations"].layer, "profile")
            self.assertEqual(origins["train_iterations"].authored_value, "auto")

    def test_profile_is_deep_merged_then_sequence_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            robots, sequences = self._layout(root)
            (robots / "field_robot.yaml").write_text(
                "adapter: ros2_zed_ublox\n"
                "topics:\n"
                "  left_image: /left\n"
                "  right_image: /right\n"
                "pose:\n"
                "  source: rtk_dual_antenna\n"
                "  cam_forward_m: 3.1\n"
                "  cam_up_m: 1.3\n"
                "  time_offset_s: 0.0\n"
            )
            sequence = sequences / "field.yaml"
            sequence.write_text(
                "robot: field_robot\n"
                "paths:\n"
                "  workdir: ~/runs/field\n"
                "  bags: [~/bags/field]\n"
                "pose:\n"
                "  time_offset_s: 0.04\n"
                "train:\n"
                "  run_name: field_01\n"
            )

            cfg = load_config(sequence)

            self.assertEqual(cfg.adapter, "ros2_zed_ublox")
            self.assertEqual(cfg.robot, "field_robot")
            self.assertEqual(cfg.topics.left_image, "/left")
            self.assertEqual(cfg.pose.source, "rtk_dual_antenna")
            self.assertEqual(cfg.pose.cam_forward_m, 3.1)
            self.assertEqual(cfg.pose.cam_up_m, 1.3)
            self.assertEqual(cfg.pose.time_offset_s, 0.04)
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

    def test_layers_reject_values_owned_elsewhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            robots, sequences = self._layout(root)
            profiles = root / "profiles"
            profiles.mkdir()

            (profiles / "bad.yaml").write_text(
                "topics: {left_image: /camera/left}\n"
            )
            sequence = sequences / "profile_bad.yaml"
            sequence.write_text(
                "profile: bad\npaths: {workdir: /run}\n"
            )
            with self.assertRaisesRegex(
                ValueError, "profile layer cannot own option.*topics"
            ):
                load_config(sequence)

            (robots / "bad.yaml").write_text(
                "train: {iterations: 30000}\n"
            )
            sequence = sequences / "robot_bad.yaml"
            sequence.write_text(
                "robot: bad\npaths: {workdir: /run}\n"
            )
            with self.assertRaisesRegex(
                ValueError, "robot layer cannot own option.*train"
            ):
                load_config(sequence)

            (robots / "good.yaml").write_text(
                "topics: {left_image: /camera/left}\n"
                "pose: {source: gnss_course}\n"
                "segment: {maximum_bag_gap_s: 0.1, image_encoding: png_lossless}\n"
                "timing: {clock_model: ros_log_constant}\n"
            )
            sequence = sequences / "sequence_bad.yaml"
            sequence.write_text(
                "robot: good\n"
                "paths: {workdir: /run}\n"
                "topics: {left_image: /wrong}\n"
            )
            with self.assertRaisesRegex(
                ValueError, r"sequence layer cannot own option.*topics"
            ):
                load_config(sequence)

            forbidden_nested = {
                "sensor geometry": "pose: {cam_up_m: 1.2}\n",
                "feature policy": "frontend: {features: {profile: cpu_reference}}\n",
                "loss policy": "train: {ssim_lambda: 0.4}\n",
                "robot-only path": "paths: {workdir: /run, ublox_msgs_dir: /msgs}\n",
                "unused dataset root": "paths: {workdir: /run, dataset_root: /data}\n",
            }
            for label, body in forbidden_nested.items():
                with self.subTest(layer="sequence", option=label):
                    candidate = sequences / "nested_bad.yaml"
                    prefix = "paths: {workdir: /run}\n" if not body.startswith("paths:") else ""
                    candidate.write_text(prefix + body)
                    with self.assertRaisesRegex(
                        ValueError,
                        "unknown configuration option|sequence layer cannot own option",
                    ):
                        load_config(candidate)

    def test_sequence_allows_only_documented_run_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            robots, sequences = self._layout(root)
            profiles = root / "profiles"
            profiles.mkdir()
            (profiles / "quality.yaml").write_text(
                "depth: {min_z_m: 0.5, max_z_m: auto}\n"
                "frontend:\n"
                "  features: {profile: gpu}\n"
                "  keyframes: {preset: balanced}\n"
                "mapper: {max_rtk_median_error_m: 0.12}\n"
                "train: {iterations: auto, max_gaussians: auto}\n"
            )
            (robots / "robot.yaml").write_text(
                "topics: {left_image: /left}\n"
                "pose: {source: gnss_course, time_offset_s: 0.0}\n"
            )
            sequence = sequences / "allowed.yaml"
            sequence.write_text(
                "profile: quality\n"
                "robot: robot\n"
                "paths: {workdir: /run}\n"
                "pose: {time_offset_s: 0.04}\n"
                "segment: {maximum_bag_gap_s: 0.2}\n"
                "timing: {maximum_window_drift_s: 0.025}\n"
                "depth: {max_z_m: 18.0}\n"
                "frontend:\n"
                "  keyframes: {preset: all}\n"
                "  pairs: {max_view_angle_deg: 110.0}\n"
                "mapper: {max_rtk_median_error_m: 0.20}\n"
                "train: {iterations: 50000, max_gaussians: 2000000}\n"
            )

            cfg = load_config(sequence)

            self.assertEqual(cfg.frontend.features.profile, "gpu")
            self.assertEqual(cfg.frontend.keyframes.preset, "all")
            self.assertEqual(cfg.pose.time_offset_s, 0.04)
            self.assertEqual(cfg.segment.maximum_bag_gap_s, 0.2)
            self.assertEqual(cfg.timing.maximum_window_drift_s, 0.025)
            self.assertEqual(cfg.depth.max_z_m, 18.0)
            self.assertEqual(cfg.mapper.max_rtk_median_error_m, 0.20)
            self.assertEqual(cfg.train.iterations, 50000)

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

    def test_unknown_nested_option_is_rejected_before_a_stage_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "typo.yaml"
            config.write_text(
                "paths: {workdir: /run}\n"
                "train: {iterrations: 30000}\n"
            )
            with self.assertRaisesRegex(
                ValueError, r"unknown configuration option.*train\.iterrations"
            ):
                load_config(config)

    def test_adapter_options_is_the_only_open_extension_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "plugin.yaml"
            config.write_text(
                "paths: {workdir: /run}\n"
                "adapter_options:\n"
                "  vendor_frame_policy: {arbitrary: true}\n"
            )
            cfg = load_config(config)
            self.assertTrue(cfg.adapter_options.vendor_frame_policy.arbitrary)

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
