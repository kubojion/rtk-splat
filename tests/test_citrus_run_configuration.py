import subprocess
import unittest
from pathlib import Path

import numpy as np

from rtk_splat.core.configio import load_config


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "configs/sequences/citrusfarm_05_13d_uturn.yaml"
SCRIPT = REPOSITORY / "scripts/runs/citrusfarm_05_13d_uturn.sh"


class CitrusRunConfigurationTests(unittest.TestCase):
    def test_profile_and_sequence_freeze_the_intended_generic_contract(self):
        cfg = load_config(CONFIG)

        self.assertEqual(cfg.adapter, "ros1_citrusfarm")
        self.assertEqual(cfg.pose.source, "gnss_course")
        self.assertEqual(cfg.pose.time_offset_s, 0.072548749)
        self.assertEqual(cfg.segment.window_s, [543.0, 735.0])
        self.assertEqual(cfg.segment.window_epoch_source, "first_gnss_log")
        self.assertEqual(cfg.segment.frame_spacing_m, 0.15)
        self.assertFalse(hasattr(cfg.segment, "expected_frames"))
        self.assertEqual(len(cfg.paths.camera_bags), 27)
        self.assertEqual(len(cfg.paths.gnss_bags), 2)
        self.assertEqual(cfg.depth.backend, "sgbm")
        self.assertEqual(cfg.segment.image_encoding, "png_lossless")
        self.assertEqual(cfg.timing.clock_model, "ros_log_constant")
        self.assertEqual(
            cfg.topics.receiver_state, "/piksi/debug/receiver_state"
        )
        self.assertTrue(cfg.gnss_quality.receiver_state_required)
        self.assertEqual(
            cfg.gnss_quality.covariance_provenance,
            "driver_static_nominal",
        )
        self.assertFalse(cfg.gnss_quality.covariance_is_live_per_epoch)
        self.assertEqual(cfg.pose.minimum_position_carrier_status, 2)
        self.assertEqual(cfg.frontend.pose_priors.min_carrier_status, 2)
        self.assertEqual(
            cfg.sensor_geometry.extrinsic_camera_geometry, "raw_left"
        )

        transform = np.asarray(
            cfg.sensor_geometry.T_camera_primary_antenna, dtype=float
        )
        self.assertEqual(transform.shape, (4, 4))
        np.testing.assert_allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-9
        )
        self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1.0, places=9)
        self.assertAlmostEqual(np.linalg.norm(transform[:3, 3]), 0.5284961712)

    def test_mapper_profile_is_the_bounded_golden_mechanism(self):
        cfg = load_config(CONFIG)

        self.assertEqual(cfg.mapper.backend, "global")
        self.assertEqual(cfg.mapper.ba_num_iterations, 3)
        self.assertEqual(cfg.mapper.keep_max_num_tracks, 60_000)
        self.assertEqual(cfg.mapper.track_required_tracks_per_view, 1_000)
        self.assertTrue(cfg.mapper.skip_retriangulation)
        self.assertFalse(cfg.mapper.gp_use_gpu)
        self.assertFalse(cfg.mapper.ba_ceres_use_gpu)
        self.assertEqual(cfg.mapper.minimum_available_memory_gb, 4.0)
        self.assertEqual(cfg.mapper.rtk_covariance_gate_mode, "diagnostic_only")
        self.assertEqual(cfg.mapper.rtk_chi2_inlier_probability, 0.95)
        self.assertEqual(cfg.mapper.min_rtk_chi2_inlier_fraction, 0.80)

    def test_launcher_has_valid_shell_and_plan_is_non_mutating(self):
        syntax = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        planned = subprocess.run(
            ["bash", str(SCRIPT), "plan"], capture_output=True, text=True
        )
        self.assertEqual(planned.returncode, 0, planned.stderr)
        self.assertIn("0.15 m metric spacing", planned.stdout)
        self.assertIn("bounded Global Mapper", planned.stdout)
        self.assertIn("Nothing runs from plan", planned.stdout)
        self.assertNotIn("tmux", planned.stdout.split("Stages", 1)[0])


if __name__ == "__main__":
    unittest.main()
