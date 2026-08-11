import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from rtk_splat.adapters.sampling import (
    resolve_frame_stride,
    resolve_metric_frame_spacing,
)
from rtk_splat.core.runtime_resolution import runtime_resolution_plain
from rtk_splat.workflows.runtime_config import (
    resolve_cloud_max_points,
    resolve_depth_max_z,
    resolve_training_controls,
)


def _policy_config():
    return NS(
        segment=NS(frame_stride="auto", frame_spacing_m="auto"),
        depth=NS(min_z_m=0.5, max_z_m="auto"),
        cloud=NS(max_points="auto"),
        train=NS(
            iterations="auto",
            max_gaussians="auto",
            use_right_camera=False,
        ),
        derivation=NS(
            frame_sampling=NS(target_spacing_m=0.10),
            depth=NS(min_reliable_disparity_px=2.0, max_z_cap_m=20.0),
            training=NS(
                image_presentations_per_view=50.0,
                min_iterations=30000,
                max_iterations=65000,
                gaussians_per_gib=400000,
                reserve_vram_gib=1.5,
                initial_cloud_growth_factor=3.0,
                min_gaussians=500000,
                max_gaussians=3400000,
                vram_gib_override=None,
            ),
        ),
    )


class RuntimeConfigTests(unittest.TestCase):
    def test_frame_stride_uses_measured_rate_and_speed(self):
        cfg = _policy_config()

        chosen = resolve_frame_stride(
            cfg,
            measured_camera_rate_hz=15.0,
            measured_median_speed_m_s=0.75,
        )

        self.assertEqual(chosen, 2)
        record = runtime_resolution_plain(cfg)["derivations"]["frame_stride"]
        self.assertEqual(record["source"], "derived")
        self.assertEqual(record["formula_version"], 1)
        self.assertEqual(record["chosen_value"], 2)
        self.assertEqual(record["measured_inputs"]["measured_camera_rate_hz"], 15.0)
        self.assertEqual(record["policy_bounds"]["target_spacing_m"], 0.10)

        # The operational field is now numeric, but the immutable origin is
        # still authored `auto`; a retry remains derived and idempotent.
        self.assertEqual(
            resolve_frame_stride(
                cfg,
                measured_camera_rate_hz=15.0,
                measured_median_speed_m_s=0.75,
            ),
            2,
        )
        repeated = runtime_resolution_plain(cfg)["derivations"]["frame_stride"]
        self.assertEqual(repeated, record)
        self.assertEqual(repeated["source"], "derived")

    def test_explicit_stride_is_untouched_and_recorded(self):
        cfg = _policy_config()
        cfg.segment.frame_stride = 5

        chosen = resolve_frame_stride(
            cfg,
            measured_camera_rate_hz=30.0,
            measured_median_speed_m_s=2.0,
        )

        self.assertEqual(chosen, 5)
        record = runtime_resolution_plain(cfg)["derivations"]["frame_stride"]
        self.assertEqual(record["source"], "override")
        self.assertEqual(record["chosen_value"], 5)

    def test_metric_sampler_uses_named_policy_target(self):
        cfg = _policy_config()

        self.assertEqual(resolve_metric_frame_spacing(cfg), 0.10)
        record = runtime_resolution_plain(cfg)["derivations"]["frame_spacing_m"]
        self.assertEqual(record["source"], "derived")
        self.assertEqual(record["policy_bounds"], {"target_spacing_m": 0.10})

    def test_depth_range_uses_f_times_baseline_and_cap(self):
        cfg = _policy_config()
        calibration = {
            "cameras": {"left": {"K": [[100.0, 0, 0], [0, 100.0, 0], [0, 0, 1]]}},
            "T_right_left": [
                [1, 0, 0, -0.2],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        }

        chosen = resolve_depth_max_z(cfg, calibration)

        self.assertEqual(chosen, 10.0)
        record = runtime_resolution_plain(cfg)["derivations"]["depth_max_z_m"]
        self.assertEqual(record["source"], "derived")
        self.assertEqual(record["measured_inputs"]["f_times_baseline_px_m"], 20.0)
        self.assertEqual(record["policy_bounds"]["max_z_cap_m"], 20.0)

        self.assertEqual(resolve_depth_max_z(cfg, calibration), 10.0)
        self.assertEqual(
            runtime_resolution_plain(cfg)["derivations"]["depth_max_z_m"],
            record,
        )

    def test_training_policy_uses_views_cloud_and_injected_vram(self):
        cfg = _policy_config()
        segment = NS(manifest={"train": list(range(1000))})
        with tempfile.TemporaryDirectory() as tmp:
            cloud = Path(tmp) / "cloud.npz"
            np.savez(cloud, xyz=np.zeros((1_000_000, 3), dtype=np.float32))

            iterations, gaussians = resolve_training_controls(
                cfg,
                segment,
                cloud,
                cuda_total_memory_gib=8.0,
            )

        self.assertEqual(iterations, 50000)
        self.assertEqual(gaussians, 2600000)
        records = runtime_resolution_plain(cfg)["derivations"]
        self.assertEqual(records["train_iterations"]["source"], "derived")
        self.assertEqual(
            records["train_iterations"]["measured_inputs"]["training_views"],
            1000,
        )
        self.assertEqual(records["train_max_gaussians"]["source"], "derived")
        self.assertEqual(
            records["train_max_gaussians"]["measured_inputs"]["initial_cloud_points"],
            1_000_000,
        )

    def test_tile_training_policy_counts_only_selected_training_frames(self):
        cfg = _policy_config()
        segment = NS(manifest={"train": list(range(1000))})
        with tempfile.TemporaryDirectory() as tmp:
            cloud = Path(tmp) / "cloud.npz"
            np.savez(cloud, xyz=np.zeros((10, 3), dtype=np.float32))
            iterations, _ = resolve_training_controls(
                cfg,
                segment,
                cloud,
                cuda_total_memory_gib=8.0,
                training_frame_ids=list(range(800)),
            )
        self.assertEqual(iterations, 40000)
        record = runtime_resolution_plain(cfg)["derivations"]["train_iterations"]
        self.assertEqual(record["measured_inputs"]["training_pairs"], 800)
        with self.assertRaisesRegex(ValueError, "unique members"):
            resolve_training_controls(
                _policy_config(),
                segment,
                cloud,
                cuda_total_memory_gib=8.0,
                training_frame_ids=[0, 0],
            )

    def test_cloud_policy_leaves_growth_headroom_inside_vram_cap(self):
        cfg = _policy_config()

        chosen = resolve_cloud_max_points(cfg, cuda_total_memory_gib=8.0)

        self.assertEqual(chosen, 866_666)
        record = runtime_resolution_plain(cfg)["derivations"][
            "cloud_max_points"
        ]
        self.assertEqual(record["source"], "derived")
        self.assertEqual(
            record["measured_inputs"]["effective_gaussian_capacity"],
            2_600_000,
        )
        self.assertEqual(
            record["policy_bounds"]["initial_cloud_growth_factor"], 3.0
        )
        self.assertEqual(
            resolve_cloud_max_points(cfg, cuda_total_memory_gib=8.0), chosen
        )

    def test_explicit_cloud_limit_is_recorded_without_gpu_policy(self):
        cfg = NS(cloud=NS(max_points=750_000))

        self.assertEqual(resolve_cloud_max_points(cfg), 750_000)
        record = runtime_resolution_plain(cfg)["derivations"][
            "cloud_max_points"
        ]
        self.assertEqual(record["source"], "override")
        self.assertEqual(record["chosen_value"], 750_000)

    def test_explicit_training_values_need_no_policy_gpu_or_cloud(self):
        cfg = NS(train=NS(iterations=65000, max_gaussians=2500000))

        chosen = resolve_training_controls(
            cfg,
            NS(manifest={"train": []}),
            "/does/not/exist.npz",
        )

        self.assertEqual(chosen, (65000, 2500000))
        records = runtime_resolution_plain(cfg)["derivations"]
        self.assertEqual(records["train_iterations"]["source"], "override")
        self.assertEqual(records["train_max_gaussians"]["source"], "override")

    def test_auto_gaussian_budget_fails_cleanly_without_enough_vram(self):
        cfg = _policy_config()
        cfg.train.iterations = 30000
        with tempfile.TemporaryDirectory() as tmp:
            cloud = Path(tmp) / "cloud.npz"
            np.savez(cloud, xyz=np.zeros((10, 3), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "below quality_v1"):
                resolve_training_controls(
                    cfg,
                    NS(manifest={"train": [0]}),
                    cloud,
                    cuda_total_memory_gib=2.0,
                )

    def test_failed_training_resolution_is_transactional_and_retryable(self):
        cfg = _policy_config()
        segment = NS(manifest={"train": list(range(1000))})
        with tempfile.TemporaryDirectory() as tmp:
            cloud = Path(tmp) / "cloud.npz"
            np.savez(cloud, xyz=np.zeros((1_000_000, 3), dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "below quality_v1"):
                resolve_training_controls(
                    cfg, segment, cloud, cuda_total_memory_gib=2.0
                )

            self.assertEqual(cfg.train.iterations, "auto")
            self.assertEqual(cfg.train.max_gaussians, "auto")
            self.assertEqual(
                runtime_resolution_plain(cfg)["derivations"], {}
            )

            chosen = resolve_training_controls(
                cfg, segment, cloud, cuda_total_memory_gib=8.0
            )
            self.assertEqual(chosen, (50000, 2600000))
            first = runtime_resolution_plain(cfg)["derivations"]
            self.assertEqual(first["train_iterations"]["source"], "derived")
            self.assertEqual(first["train_max_gaussians"]["source"], "derived")

            self.assertEqual(
                resolve_training_controls(
                    cfg, segment, cloud, cuda_total_memory_gib=8.0
                ),
                chosen,
            )
            self.assertEqual(
                runtime_resolution_plain(cfg)["derivations"], first
            )

    def test_repeated_resolution_with_different_measurements_fails_closed(self):
        cfg = _policy_config()
        resolve_frame_stride(
            cfg,
            measured_camera_rate_hz=15.0,
            measured_median_speed_m_s=0.75,
        )
        with self.assertRaisesRegex(ValueError, "changed within one run"):
            resolve_frame_stride(
                cfg,
                measured_camera_rate_hz=30.0,
                measured_median_speed_m_s=0.75,
            )


if __name__ == "__main__":
    unittest.main()
