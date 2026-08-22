import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from rtk_splat.core.pose_artifacts import pose_fingerprint
from rtk_splat.frontends.artifact import canonical_hash
from rtk_splat.workflows.tile_scene import select_geometric_tile_neighbor
from rtk_splat.workflows.tile_seam import (
    _manifest,
    publish_tile_seam_probe,
    verify_tile_seam_probe,
)


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _params() -> dict[str, torch.Tensor]:
    return {
        "means": torch.tensor([[0.5, 0.5, 0], [1.5, 0.5, 0]]),
        "quats": torch.ones((2, 4)),
        "scales": torch.zeros((2, 3)),
        "opacities": torch.zeros(2),
        "sh0": torch.zeros((2, 1, 3)),
        "shN": torch.zeros((2, 3, 3)),
    }


def _plan() -> dict:
    viewmats = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    return {
        "schema_version": 3,
        "name": "sealed-plan-v1",
        "provisional": True,
        "metric_georeferencing_claim_eligible": False,
        "source_binding": {
            "pose_fingerprint": pose_fingerprint(viewmats),
        },
        "coordinate_frame": {
            "partition_origin_enu_m": [0.0, 0.0, 0.0],
            "R_enu_from_partition": np.eye(3).tolist(),
        },
        "partition": {
            "scene_bounds_uv_m": [[0.0, 0.0], [2.0, 1.0]],
            "context_halo_m": 0.25,
            "boundary_rule": "lower_closed_upper_open_global_max_closed",
            "ownership_tolerance_m": 1e-9,
        },
        "tiles": [
            {
                "tile_id": "tile-0000",
                "core_bounds_uv_m": [[0.0, 0.0], [1.0, 1.0]],
                "context_bounds_uv_m": [[-0.25, -0.25], [1.25, 1.25]],
                "frame_ids": {"train": [1], "val": [0, 8], "test": []},
            },
            {
                "tile_id": "tile-0001",
                "core_bounds_uv_m": [[1.0, 0.0], [2.0, 1.0]],
                "context_bounds_uv_m": [[0.75, -0.25], [2.25, 1.25]],
                "frame_ids": {"train": [2], "val": [0, 8], "test": []},
            },
        ],
    }


def _inventory(plan: dict) -> dict:
    return {
        "segment": {
            "contract_files": {
                "manifest.json": {"sha256": "1" * 64, "size_bytes": 1},
                "frames.npz": {"sha256": "2" * 64, "size_bytes": 1},
                "calibration.json": {"sha256": "3" * 64, "size_bytes": 1},
                "segment_meta.json": {"sha256": "4" * 64, "size_bytes": 1},
            },
        },
        "pose": {
            "name": "pose-v1",
            "fingerprint": plan["source_binding"]["pose_fingerprint"],
        },
    }


class TileSeamTests(unittest.TestCase):
    def test_neighbor_selection_uses_longest_shared_edge_only(self):
        plan = _plan()
        plan["partition"]["scene_bounds_uv_m"] = [[0.0, 0.0], [3.0, 3.0]]
        plan["tiles"][0]["core_bounds_uv_m"] = [[0.0, 0.0], [2.0, 2.0]]
        plan["tiles"][1]["core_bounds_uv_m"] = [[2.0, 0.0], [3.0, 2.0]]
        plan["tiles"].append({
            "tile_id": "tile-0002",
            "core_bounds_uv_m": [[0.0, 2.0], [1.0, 3.0]],
            "context_bounds_uv_m": [[-0.25, 1.75], [1.25, 3.25]],
            "frame_ids": {"train": [99], "val": [12345], "test": []},
        })
        selection = select_geometric_tile_neighbor(plan, "tile-0000")
        self.assertEqual(selection["selected_tile_id"], "tile-0001")
        self.assertEqual(selection["selected_segment_uv"], [0, 2.0, 0.0, 2.0])
        self.assertFalse(selection["uses_heldout_evidence"])
        self.assertEqual(
            [item["tile_id"] for item in selection["candidates"]],
            ["tile-0001", "tile-0002"],
        )

    def _fixture(self, root: Path):
        plan = _plan()
        inventory = _inventory(plan)
        segment = root / "segment"
        segment.mkdir()
        plan_root = root / "plan"
        plan_root.mkdir()
        (plan_root / "manifest.json").write_bytes(b"plan manifest")
        (plan_root / "tile_plan.json").write_bytes(b"plan")
        _write_json(plan_root / "source_inventory.json", inventory)
        georeferencing = {
            "artifact_class": "diagnostic_render_only",
            "georeferencing_status": "FAILED",
            "metric_georeferencing_claim_eligible": False,
        }
        _write_json(plan_root / "provenance.json", {
            "georeferencing": georeferencing,
        })
        runs = {}
        for tile in plan["tiles"]:
            run = root / tile["tile_id"]
            run.mkdir()
            (run / "params.pt").write_bytes(tile["tile_id"].encode())
            (run / "run_provenance.json").write_bytes(b"provenance")
            (run / "best_checkpoint.json").write_bytes(b"best")
            (run / "splat.DIAGNOSTIC_ONLY.ply").write_bytes(b"ply")
            runs[tile["tile_id"]] = run
        reader = SimpleNamespace(
            root=segment,
            manifest={"train": [1, 2], "val": [0, 8], "test": []},
        )
        reader.validate = lambda: reader
        cfg = SimpleNamespace(
            pose=SimpleNamespace(artifact="pose-v1"),
            train=SimpleNamespace(max_gaussians=10, max_scale_m=0.05),
        )
        return plan, inventory, georeferencing, runs, reader, cfg

    def test_probe_is_atomic_sealed_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, inventory, georeferencing, runs, reader, cfg = self._fixture(root)
            params = _params()
            ownership = [
                {
                    "tile_id": tile_id,
                    "source_params": str(runs[tile_id] / "params.pt"),
                    "source_params_sha256": _sha(runs[tile_id] / "params.pt"),
                    "source_gaussians": 2,
                    "core_owned_gaussians": 1,
                    "outside_or_other_core_gaussians": 1,
                }
                for tile_id in ("tile-0000", "tile-0001")
            ]
            feather_ownership = [{
                "tile_id": tile_id,
                "source_gaussians": 2,
                "retained_gaussians": 1,
                "strict_core_gaussians": 1,
                "feather_support_gaussians": 0,
            } for tile_id in ("tile-0000", "tile-0001")]

            def completed(run, binding, _georeferencing, **_kwargs):
                run = Path(run)
                contracts = inventory["segment"]["contract_files"]
                provenance = {
                    "model_selection": {"status": "complete"},
                    "training_implementation_sha256": "9" * 64,
                    "effective_training_config": {
                        "train": {"run_name": run.name, "iterations": 10}
                    },
                    "manifest_sha256": contracts["manifest.json"]["sha256"],
                    "frames_sha256": contracts["frames.npz"]["sha256"],
                    "calibration_sha256": contracts["calibration.json"]["sha256"],
                    "segment_meta_sha256": contracts["segment_meta.json"]["sha256"],
                    "pose_artifact": inventory["pose"]["name"],
                    "pose_fingerprint": inventory["pose"]["fingerprint"],
                }
                return provenance, run / "params.pt", _sha(run / "params.pt"), {
                    "initial_cloud_sha256": "e" * 64,
                    "splat_file": "splat.DIAGNOSTIC_ONLY.ply",
                    "splat_sha256": _sha(run / "splat.DIAGNOSTIC_ONLY.ply"),
                }

            candidate = {
                "psnr_masked": 23.5,
                "psnr_masked_cc": 25.0,
                "ssim": 0.40,
                "lpips_cc": 0.30,
                "pose_fingerprint": plan["source_binding"]["pose_fingerprint"],
            }
            first = {
                **candidate,
                "psnr_masked": 23.4,
                "psnr_masked_cc": 24.9,
                "ssim": 0.39,
                "lpips_cc": 0.31,
            }
            second = {
                **candidate,
                "psnr_masked": 23.6,
                "psnr_masked_cc": 25.1,
                "ssim": 0.41,
                "lpips_cc": 0.29,
            }
            seam_mask = np.ones((2, 2), dtype=bool)
            viewmats = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
            with (
                mock.patch(
                    "rtk_splat.workflows.tile_seam.SegmentReader",
                    return_value=reader,
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_seam.verify_tile_plan",
                    return_value=plan,
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_seam._completed_tile_run",
                    side_effect=completed,
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_seam._concatenate_core_params",
                    return_value=(params, ownership),
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_seam._concatenate_feathered_params",
                    return_value=(params, feather_ownership),
                ) as feathered_assembly,
                mock.patch(
                    "rtk_splat.backends.gsplat._load_checkpoint_gaussians",
                    return_value=params,
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_seam._build_seam_masks",
                    return_value=(
                        {0: seam_mask, 8: seam_mask},
                        {
                            "policy": "synthetic",
                            "n_validation_frames": 2,
                            "metric_depth_pixels_in_band": 512,
                            "internal_segments_uv": [[0, 1.0, 0.0, 1.0]],
                            "frames": [
                                {"frame_id": 0, "metric_depth_pixels_in_band": 256},
                                {"frame_id": 8, "metric_depth_pixels_in_band": 256},
                            ],
                        },
                    ),
                ) as mask_builder,
                mock.patch(
                    "rtk_splat.workflows.tile_seam._evaluate_combined",
                    side_effect=[
                        candidate, first, second,
                        candidate, first, second,
                    ],
                ) as evaluator,
                mock.patch(
                    "rtk_splat.workflows.tile_seam.load_pose_artifact",
                    return_value=(viewmats, np.zeros((2, 3))),
                ) as pose_loader,
                mock.patch(
                    "rtk_splat.workflows.tile_seam.collect_package_state",
                    return_value={"python_tree_sha256": "c" * 64},
                ),
            ):
                result = publish_tile_seam_probe(
                    segment=root / "segment",
                    cfg=cfg,
                    tile_plan_root=root / "plan",
                    pose_root=root / "pose",
                    tile_runs=runs,
                    anchor_tile_id="tile-0000",
                    output_root=root / "output",
                    probe_name="probe-v1",
                    maximum_visible_gaussians=10,
                    device="cpu",
                    allow_nonproduction_georeferencing_for_diagnostic=True,
                )
                feather_result = publish_tile_seam_probe(
                    segment=root / "segment",
                    cfg=cfg,
                    tile_plan_root=root / "plan",
                    pose_root=root / "pose",
                    tile_runs=runs,
                    anchor_tile_id="tile-0000",
                    output_root=root / "output",
                    probe_name="probe-feather-v1",
                    maximum_visible_gaussians=10,
                    device="cpu",
                    allow_nonproduction_georeferencing_for_diagnostic=True,
                    assembly_policy="normalized_core_distance_feather_v1",
                )
            self.assertTrue(result["quality_passed"])
            self.assertTrue(feather_result["quality_passed"])
            self.assertEqual(feathered_assembly.call_count, 1)
            feather_probe = Path(feather_result["probe"])
            feather_record = verify_tile_seam_probe(feather_probe)
            self.assertEqual(
                feather_record["assembly_policy"]["name"],
                "normalized_core_distance_feather_v1",
            )
            self.assertAlmostEqual(
                feather_record["assembly_policy"]["feather_width_m"],
                0.15,
            )
            feather_metrics = json.loads(
                (feather_probe / "metrics.json").read_text(encoding="utf-8")
            )
            feather_metrics["assembly_policy"]["feather_width_m"] = 0.14
            _write_json(feather_probe / "metrics.json", feather_metrics)
            (feather_probe / "manifest.json").unlink()
            _write_json(
                feather_probe / "manifest.json", _manifest(feather_probe)
            )
            with self.assertRaisesRegex(ValueError, "status or quality"):
                verify_tile_seam_probe(feather_probe)
            self.assertTrue(
                mask_builder.call_args.kwargs[
                    "allow_failed_georeferencing_for_render"
                ]
            )
            self.assertTrue(all(
                call.kwargs["allow_failed_georeferencing_for_render"]
                for call in evaluator.call_args_list
            ))
            self.assertTrue(
                pose_loader.call_args.kwargs[
                    "allow_failed_georeferencing_for_render"
                ]
            )
            probe = Path(result["probe"])
            self.assertTrue(probe.is_dir())
            self.assertTrue(verify_tile_seam_probe(probe)["provisional"])
            with self.assertRaises(FileExistsError):
                publish_tile_seam_probe(
                    segment=root / "segment",
                    cfg=cfg,
                    tile_plan_root=root / "plan",
                    pose_root=root / "pose",
                    tile_runs=runs,
                    anchor_tile_id="tile-0000",
                    output_root=root / "output",
                    probe_name="probe-v1",
                    maximum_visible_gaussians=10,
                    allow_nonproduction_georeferencing_for_diagnostic=True,
                )
            metrics = json.loads((probe / "metrics.json").read_text())
            metrics["regressions"]["psnr_masked_loss_db"] = -99.0
            _write_json(probe / "metrics.tampered.json", metrics)
            (probe / "metrics.json").unlink()
            (probe / "metrics.tampered.json").rename(probe / "metrics.json")
            (probe / "manifest.json").unlink()
            _write_json(probe / "manifest.json", _manifest(probe))
            with self.assertRaisesRegex(ValueError, "status or quality"):
                verify_tile_seam_probe(probe)

    def test_injected_failure_leaves_no_published_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, inventory, georeferencing, runs, reader, cfg = self._fixture(root)
            params = _params()

            def completed(run, binding, _georeferencing, **_kwargs):
                run = Path(run)
                contracts = inventory["segment"]["contract_files"]
                provenance = {
                    "model_selection": {"status": "complete"},
                    "training_implementation_sha256": "9" * 64,
                    "effective_training_config": {
                        "train": {"run_name": run.name}
                    },
                    "manifest_sha256": contracts["manifest.json"]["sha256"],
                    "frames_sha256": contracts["frames.npz"]["sha256"],
                    "calibration_sha256": contracts["calibration.json"]["sha256"],
                    "segment_meta_sha256": contracts["segment_meta.json"]["sha256"],
                    "pose_artifact": inventory["pose"]["name"],
                    "pose_fingerprint": inventory["pose"]["fingerprint"],
                }
                return provenance, run / "params.pt", _sha(run / "params.pt"), {
                    "initial_cloud_sha256": "e" * 64,
                    "splat_file": "splat.DIAGNOSTIC_ONLY.ply",
                    "splat_sha256": _sha(run / "splat.DIAGNOSTIC_ONLY.ply"),
                }

            ownership = [{
                "tile_id": tile_id,
                "source_gaussians": 2,
                "core_owned_gaussians": 1,
                "outside_or_other_core_gaussians": 1,
            } for tile_id in ("tile-0000", "tile-0001")]
            with (
                mock.patch(
                    "rtk_splat.workflows.tile_seam.SegmentReader",
                    return_value=reader,
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_seam.verify_tile_plan",
                    return_value=plan,
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_seam._completed_tile_run",
                    side_effect=completed,
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_seam._concatenate_core_params",
                    return_value=(params, ownership),
                ),
                mock.patch(
                    "rtk_splat.backends.gsplat._load_checkpoint_gaussians",
                    return_value=params,
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_seam._build_seam_masks",
                    side_effect=RuntimeError("injected"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    publish_tile_seam_probe(
                        segment=root / "segment",
                        cfg=cfg,
                        tile_plan_root=root / "plan",
                        pose_root=root / "pose",
                        tile_runs=runs,
                        anchor_tile_id="tile-0000",
                        output_root=root / "output",
                        probe_name="failed-probe-v1",
                        maximum_visible_gaussians=10,
                        device="cpu",
                        allow_nonproduction_georeferencing_for_diagnostic=True,
                    )
            parent = root / "output" / "seam_probe_artifacts"
            self.assertFalse((parent / "failed-probe-v1").exists())
            self.assertEqual(list(parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
