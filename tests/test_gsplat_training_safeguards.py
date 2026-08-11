import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import torch

from rtk_splat.backends import gsplat
from rtk_splat.backends.gsplat import (
    _GAUSSIAN_PARAMETER_NAMES,
    _atomic_save_checkpoint,
    _is_better_model,
    _load_checkpoint_gaussians,
    _step_mcmc_before_cutoff,
)


def gaussian_params(value: float = 1.0):
    shapes = {
        "means": (2, 3),
        "quats": (2, 4),
        "scales": (2, 3),
        "opacities": (2,),
        "sh0": (2, 1, 3),
        "shN": (2, 3, 3),
    }
    return torch.nn.ParameterDict({
        name: torch.nn.Parameter(torch.full(shape, value))
        for name, shape in shapes.items()
    })


class FakeStrategy:
    def __init__(self):
        self.calls = []

    def step_post_backward(
        self, params, optimizers, state, step, info, *, lr
    ):
        self.calls.append((step, lr))

    def check_sanity(self, params, optimizers):
        return None

    def initialize_state(self):
        return {}

    def step_pre_backward(self, params, optimizers, state, step, info):
        return None


class GsplatTrainingSafeguardTests(unittest.TestCase):
    def test_color_correction_subsampling_is_deterministic(self):
        rgb = torch.rand((12, 13, 3))
        target = (rgb * 0.8 + 0.1).clamp(0, 1)
        mask = torch.ones((12, 13), dtype=torch.bool)
        torch.manual_seed(1)
        first = gsplat._color_correct(rgb, target, mask, max_px=17)
        torch.manual_seed(999)
        second = gsplat._color_correct(rgb, target, mask, max_px=17)
        torch.testing.assert_close(first, second)

    def test_context_mask_intersects_depth_and_supervision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.full((3, 4, 3), 100, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(root / "left.png"), image))
            np.savez(
                root / "depth.npz",
                depth=np.ones((3, 4), dtype=np.float32),
                valid=np.ones((3, 4), dtype=bool),
            )
            frames = {
                "left_image_path": np.asarray(["left.png"]),
                "depth_path": np.asarray(["depth.npz"]),
            }
            context = np.zeros((3, 4), dtype=bool)
            context[1, 2] = True
            _, _, valid, supervise = gsplat._load_frame(
                root,
                frames,
                0,
                dilate_px=1,
                device="cpu",
                context_mask=context,
            )
            torch.testing.assert_close(valid, torch.from_numpy(context))
            torch.testing.assert_close(supervise, valid)

    def test_tile_core_mask_is_half_open_at_shared_boundary(self):
        binding = {
            "partition_origin_enu_m": [0.0, 0.0, 0.0],
            "R_enu_from_partition": np.eye(3).tolist(),
            "scene_bounds_uv_m": [[0.0, 0.0], [2.0, 1.0]],
            "core_bounds_uv_m": [[0.0, 0.0], [1.0, 1.0]],
            "boundary_rule": "lower_closed_upper_open_global_max_closed",
            "ownership_tolerance_m": 1e-9,
        }
        points = torch.tensor([
            [0.0, 0.5, 0.0],
            [0.999, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            [2.0, 0.5, 0.0],
        ])
        keep = gsplat._tile_core_keep(points, binding)
        torch.testing.assert_close(
            keep, torch.tensor([True, True, False, False])
        )

    def test_grayscale_png_expands_to_rgb_and_preserves_recorded_depth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grayscale = np.asarray(
                [[0, 64, 255], [12, 128, 240]], dtype=np.uint8
            )
            depth = np.asarray(
                [[0.0, 1.25, 2.5], [3.75, 5.0, 6.25]], dtype=np.float32
            )
            valid = np.asarray(
                [[False, True, True], [True, False, True]], dtype=np.bool_
            )
            self.assertTrue(cv2.imwrite(str(root / "left.png"), grayscale))
            np.savez(root / "depth.npz", depth=depth, valid=valid)
            frames = {
                "left_image_path": np.asarray(["left.png"]),
                "depth_path": np.asarray(["depth.npz"]),
            }

            rgb, loaded_depth, loaded_valid, supervise = gsplat._load_frame(
                root, frames, 0, dilate_px=1, device="cpu"
            )

            self.assertEqual(rgb.shape, (2, 3, 3))
            self.assertEqual(rgb.dtype, torch.float32)
            torch.testing.assert_close(rgb[..., 0], rgb[..., 1])
            torch.testing.assert_close(rgb[..., 1], rgb[..., 2])
            torch.testing.assert_close(
                rgb[..., 0], torch.from_numpy(grayscale).float() / 255.0
            )
            self.assertEqual(loaded_depth.shape, depth.shape)
            self.assertEqual(loaded_depth.dtype, torch.float32)
            self.assertEqual(loaded_valid.shape, valid.shape)
            self.assertEqual(loaded_valid.dtype, torch.bool)
            torch.testing.assert_close(loaded_depth, torch.from_numpy(depth))
            torch.testing.assert_close(loaded_valid, torch.from_numpy(valid))
            torch.testing.assert_close(supervise, loaded_valid)

    def test_best_metric_uses_strict_first_wins_tie_policy(self):
        self.assertTrue(_is_better_model(20.0, None))
        self.assertTrue(_is_better_model(20.1, 20.0))
        self.assertFalse(_is_better_model(20.0, 20.0))
        self.assertFalse(_is_better_model(19.9, 20.0))
        self.assertFalse(_is_better_model(float("nan"), 20.0))
        self.assertTrue(_is_better_model(float("inf"), 20.0))
        self.assertFalse(_is_better_model(float("inf"), float("inf")))

    def test_atomic_checkpoint_is_optimizer_independent_and_restorable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "best_params.checkpoint.pt"
            params = gaussian_params(1.25)

            _atomic_save_checkpoint(checkpoint, params)
            with torch.no_grad():
                for tensor in params.values():
                    tensor.fill_(9.0)

            restored = _load_checkpoint_gaussians(checkpoint, "cpu")
            self.assertEqual(set(restored), set(_GAUSSIAN_PARAMETER_NAMES))
            for name in _GAUSSIAN_PARAMETER_NAMES:
                torch.testing.assert_close(
                    restored[name], torch.full_like(restored[name], 1.25)
                )
            stored = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            self.assertNotIn("optimizer", stored)
            self.assertFalse((root / "params.pt").exists())
            self.assertEqual(list(root.glob("*.tmp-*")), [])

    def test_checkpoint_replacement_keeps_latest_selected_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "best_params.checkpoint.pt"
            _atomic_save_checkpoint(checkpoint, gaussian_params(1.0))
            _atomic_save_checkpoint(checkpoint, gaussian_params(2.0))
            restored = _load_checkpoint_gaussians(checkpoint, "cpu")
            for tensor in restored.values():
                torch.testing.assert_close(
                    tensor, torch.full_like(tensor, 2.0)
                )

    def test_mcmc_callback_is_never_entered_at_or_after_cutoff(self):
        strategy = FakeStrategy()
        for step in (0, 4, 5, 6):
            called = _step_mcmc_before_cutoff(
                strategy,
                params=None,
                optimizers=None,
                state=None,
                step=step,
                info=None,
                lr=0.125,
                refine_stop_iter=5,
            )
            self.assertEqual(called, step < 5)
        self.assertEqual(strategy.calls, [(0, 0.125), (4, 0.125)])

    def test_normal_completion_exports_restored_best_not_last_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment = root / "segment"
            run = root / "run"
            segment.mkdir()
            for name in (
                "manifest.json", "frames.npz", "calibration.json",
                "segment_meta.json",
            ):
                (segment / name).write_bytes(b"synthetic")
            cloud = root / "cloud.npz"
            cloud.write_bytes(b"synthetic cloud")

            train_cfg = SimpleNamespace(
                seed=0,
                max_gaussians=2,
                iterations=3,
                refine_stop_frac=2 / 3,
                means_lr_final_mult=0.1,
                color_finetune_frac=0.0,
                use_right_camera=False,
                pose_opt=SimpleNamespace(enabled=False),
                exposure_opt=SimpleNamespace(enabled=False),
                lr=SimpleNamespace(
                    means=1e-3,
                    scales=1e-3,
                    quats=1e-3,
                    opacities=1e-3,
                    sh0=1e-3,
                    shN=1e-3,
                ),
                sh_degree=1,
                rasterize_mode="classic",
                mask_dilate_px=1,
                ssim_lambda=0.0,
                depth_lambda=0.0,
                opacity_reg=0.0,
                scale_reg=0.0,
                min_scale_m=1e-3,
                max_scale_m=1.0,
                max_anisotropy=8.0,
                eval_every=1,
            )
            cfg = SimpleNamespace(
                train=train_cfg,
                pose=SimpleNamespace(),
                depth=SimpleNamespace(),
                cloud=SimpleNamespace(),
            )
            synthetic_segment = SimpleNamespace(
                frames={},
                calibration={
                    "cameras": {
                        "left": {
                            "K": [[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]],
                            "width": 2,
                            "height": 2,
                        }
                    },
                    "T_right_left": np.eye(4).tolist(),
                },
                meta={"capabilities": {"stereo": True}},
            )
            synthetic_segment.validate = lambda: synthetic_segment
            strategy = FakeStrategy()
            trajectory = []
            final_flags = []
            exported = {}

            def fake_render(params, *args, **kwargs):
                color = torch.sigmoid(params["sh0"][0, 0]).view(1, 1, 3)
                color = color.expand(2, 2, 3)
                depth = (params["means"][0, 0] * 0 + 1).expand(2, 2, 1)
                return torch.cat((color, depth), dim=-1)[None], None, {}

            scores = [10.0, 12.0, 11.0, 12.0]

            def fake_evaluate(params, *args, **kwargs):
                final_flags.append(bool(args[10]))
                call = len(trajectory)
                if call < 3:
                    trajectory.append(params["means"].detach().clone())
                else:
                    torch.testing.assert_close(params["means"], trajectory[1])
                score = scores[call]
                return {
                    "psnr": score,
                    "psnr_masked": score,
                    "psnr_near": score,
                    "psnr_masked_cc": score,
                    "lpips": 0.5,
                    "lpips_cc": 0.5,
                    "ssim": 0.5,
                    "n_eval": 1,
                }

            def fake_export(params, *args, **kwargs):
                exported["means"] = params["means"].detach().clone()

            rgb = torch.full((2, 2, 3), 0.75)
            depth = torch.ones((2, 2))
            valid = torch.ones((2, 2), dtype=torch.bool)
            patches = (
                mock.patch.object(
                    gsplat, "SegmentReader",
                    return_value=synthetic_segment,
                ),
                mock.patch.object(
                    gsplat, "pose_georeferencing_evidence",
                    return_value={"georeferencing_status": "PASSED"},
                ),
                mock.patch.object(gsplat, "require_render_permission"),
                mock.patch.object(
                    gsplat, "load_pose_artifact",
                    return_value=(np.eye(4)[None], np.zeros((1, 3))),
                ),
                mock.patch.object(gsplat, "cloud_path", return_value=cloud),
                mock.patch.object(gsplat, "verify_cloud_matches_poses"),
                mock.patch.object(gsplat, "cloud_georeferencing_evidence"),
                mock.patch.object(
                    gsplat, "pose_artifact_name", return_value="rtk"
                ),
                mock.patch.object(
                    gsplat, "configuration_evidence", return_value={}
                ),
                mock.patch.object(
                    gsplat, "init_params", return_value=gaussian_params(0.1)
                ),
                mock.patch.object(
                    gsplat, "MCMCStrategy", return_value=strategy
                ),
                mock.patch(
                    "rtk_splat.core.manifest.load_manifest",
                    return_value={"train": [0], "val": [0]},
                ),
                mock.patch.object(
                    gsplat, "_load_frame",
                    return_value=(rgb, depth, valid, valid),
                ),
                mock.patch.object(gsplat, "render", side_effect=fake_render),
                mock.patch.object(
                    gsplat, "tm_ssim",
                    return_value=(torch.tensor(1.0),
                                  torch.ones((1, 3, 2, 2))),
                ),
                mock.patch.object(
                    gsplat, "evaluate", side_effect=fake_evaluate
                ),
                mock.patch.object(
                    gsplat, "export_pruned", side_effect=fake_export
                ),
            )
            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                selected = gsplat.train_tile(
                    segment, run, cfg, device="cpu"
                )

            self.assertEqual([step for step, _ in strategy.calls], [0, 1])
            self.assertEqual(final_flags, [False, False, False, True])
            self.assertAlmostEqual(strategy.calls[0][1], 0.001)
            torch.testing.assert_close(exported["means"], trajectory[1])
            saved = torch.load(
                run / "params.pt", map_location="cpu", weights_only=True
            )
            torch.testing.assert_close(saved["means"], trajectory[1])
            metrics = json.loads((run / "metrics.json").read_text())
            self.assertEqual([item["step"] for item in metrics[:3]], [1, 2, 3])
            self.assertEqual(metrics[-1]["step"], 3)
            self.assertEqual(metrics[-1]["model_step"], 2)
            self.assertTrue(metrics[-1]["selected_for_export"])
            self.assertEqual(selected["model_step"], 2)
            provenance = json.loads((run / "run_provenance.json").read_text())
            self.assertEqual(provenance["model_selection"]["best_step"], 2)
            self.assertEqual(provenance["model_selection"]["best_metric"], 12.0)
            self.assertEqual(provenance["model_selection"]["status"], "complete")
            self.assertFalse((run / "best_params.checkpoint.pt").exists())


if __name__ == "__main__":
    unittest.main()
