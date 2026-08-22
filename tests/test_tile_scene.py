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
from rtk_splat.workflows.cli import _scene_tile_runs, build_parser, cmd_scene_publish
from rtk_splat.workflows.tile_scene import (
    _concatenate_core_params,
    _evaluate_combined,
    _expected_tile_binding,
    _reference_run_evidence,
    _training_identity,
    publish_production_tiled_scene,
    publish_tiled_scene,
    verify_tiled_scene,
)


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> dict:
    viewmats = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    fingerprint = pose_fingerprint(viewmats)
    return {
        "schema_version": 3,
        "name": "two-tile-v1",
        "provisional": True,
        "metric_georeferencing_claim_eligible": False,
        "source_binding": {
            "inventory_sha256": "a" * 64,
            "pose_fingerprint": fingerprint,
        },
        "coordinate_frame": {
            "partition_origin_enu_m": [0.0, 0.0, 0.0],
            "R_enu_from_partition": np.eye(3).tolist(),
            "ownership_axes": ["u", "v"],
            "source": {"type": "local_enu", "units": "m"},
        },
        "partition": {
            "scene_bounds_uv_m": [[0.0, 0.0], [2.0, 1.0]],
            "context_halo_m": 0.25,
            "boundary_rule": "lower_closed_upper_open_global_max_closed",
            "ownership_tolerance_m": 1e-9,
            "z_ownership": "unbounded",
        },
        "tiles": [
            {
                "tile_id": "tile-0000",
                "core_bounds_uv_m": [[0.0, 0.0], [1.0, 1.0]],
                "context_bounds_uv_m": [[-0.25, -0.25], [1.25, 1.25]],
                "frame_ids": {"train": [1, 2], "val": [0], "test": []},
            },
            {
                "tile_id": "tile-0001",
                "core_bounds_uv_m": [[1.0, 0.0], [2.0, 1.0]],
                "context_bounds_uv_m": [[0.75, -0.25], [2.25, 1.25]],
                "frame_ids": {"train": [2, 3], "val": [0, 8], "test": []},
            },
        ],
    }


def _inventory(plan: dict) -> dict:
    return {
        "schema_version": 3,
        "segment": {
            "contract_files": {
                "manifest.json": {"sha256": "1" * 64, "size_bytes": 1},
                "frames.npz": {"sha256": "2" * 64, "size_bytes": 1},
                "calibration.json": {"sha256": "3" * 64, "size_bytes": 1},
                "segment_meta.json": {"sha256": "4" * 64, "size_bytes": 1},
            }
        },
        "pose": {
            "name": "pose-v1",
            "fingerprint": plan["source_binding"]["pose_fingerprint"],
        },
    }


def _params(means) -> dict[str, torch.Tensor]:
    means = torch.tensor(means, dtype=torch.float32)
    n = len(means)
    return {
        "means": means,
        "quats": torch.ones((n, 4)),
        "scales": torch.zeros((n, 3)),
        "opacities": torch.zeros(n),
        "sh0": torch.zeros((n, 1, 3)),
        "shN": torch.zeros((n, 3, 3)),
    }


class TileSceneTests(unittest.TestCase):
    def test_training_identity_ignores_only_run_name(self):
        base = {
            "training_implementation_sha256": "a" * 64,
            "effective_training_config": {
                "train": {"run_name": "control", "iterations": 65_000},
                "depth": {"max_z_m": 12.0},
            },
        }
        tile = json.loads(json.dumps(base))
        tile["effective_training_config"]["train"]["run_name"] = "tile"
        self.assertEqual(_training_identity(base), _training_identity(tile))
        tile["effective_training_config"]["train"]["iterations"] = 64_000
        self.assertNotEqual(_training_identity(base), _training_identity(tile))

    def test_training_identity_ignores_checkout_locators_but_not_content(self):
        def provenance(prefix: str):
            return {
                "training_implementation_sha256": "a" * 64,
                "effective_training_config": {
                    "train": {"run_name": prefix, "iterations": 65_000},
                    "runtime_resolution": {
                        "config_sources": {
                            "profile": f"{prefix}/configs/profile.yaml"
                        },
                        "source_files": [{
                            "path": f"{prefix}/configs/profile.yaml",
                            "role": "profile",
                            "sha256": "b" * 64,
                        }],
                        "origins": {"train_iterations": {
                            "source_path": f"{prefix}/configs/profile.yaml",
                            "source_sha256": "b" * 64,
                            "authored_value": 65_000,
                        }},
                        "derivations": {"train_iterations": {
                            "chosen_value": 65_000,
                            "origin": {
                                "source_path": f"{prefix}/configs/profile.yaml",
                                "source_sha256": "b" * 64,
                            },
                        }},
                    },
                },
            }

        first = provenance("/checkout/one")
        second = provenance("/checkout/two")
        self.assertEqual(_training_identity(first), _training_identity(second))
        second["effective_training_config"]["runtime_resolution"][
            "source_files"
        ][0]["sha256"] = "c" * 64
        self.assertNotEqual(_training_identity(first), _training_identity(second))

    def test_training_identity_ignores_optional_plan_only_derivation(self):
        base = {
            "training_implementation_sha256": "a" * 64,
            "effective_training_config": {
                "train": {"run_name": "old", "iterations": 65_000},
                "runtime_resolution": {
                    "origins": {
                        "tile_max_training_frames": {
                            "authored_value": "auto",
                            "source_sha256": "b" * 64,
                        }
                    },
                    "derivations": {
                        "tile_max_training_frames": {
                            "chosen_value": 1300,
                            "formula_version": 1,
                        }
                    },
                },
            },
        }
        fresh = json.loads(json.dumps(base))
        fresh["effective_training_config"]["train"]["run_name"] = "new"
        fresh["effective_training_config"]["runtime_resolution"][
            "origins"
        ].clear()
        fresh["effective_training_config"]["runtime_resolution"][
            "derivations"
        ].clear()
        self.assertEqual(_training_identity(base), _training_identity(fresh))
        fresh["effective_training_config"]["train"]["iterations"] = 64_000
        self.assertNotEqual(_training_identity(base), _training_identity(fresh))

    def test_scene_cli_accepts_explicit_repeatable_tile_runs(self):
        args = build_parser().parse_args([
            "scene-publish", "--config", "/tmp/config.yaml",
            "--tile-plan", "/tmp/plan", "--pose-artifact-root", "/tmp/poses",
            "--scene-tile-run", "tile-0000=/tmp/run-a",
            "--scene-tile-run", "tile-0001=/tmp/run-b",
            "--scene-name", "scene-v1", "--reference-run", "/tmp/reference",
            "--reference-params-sha256", "a" * 64,
            "--reference-metrics-sha256", "b" * 64,
            "--reference-provenance-sha256", "c" * 64,
        ])
        self.assertEqual(
            _scene_tile_runs(args.scene_tile_run),
            {"tile-0000": Path("/tmp/run-a"),
             "tile-0001": Path("/tmp/run-b")},
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _scene_tile_runs([
                "tile-0000=/tmp/run-a", "tile-0000=/tmp/run-b"
            ])

        production = build_parser().parse_args([
            "scene-publish", "--config", "/tmp/config.yaml",
            "--tile-plan", "/tmp/plan", "--pose-artifact-root", "/tmp/poses",
            "--scene-tile-run", "tile-0000=/tmp/run-a",
            "--scene-tile-run", "tile-0001=/tmp/run-b",
            "--scene-name", "scene-v2", "--scene-mode", "production",
            "--diagnostic-scene",
        ])
        self.assertEqual(production.scene_mode, "production")
        self.assertTrue(production.diagnostic_scene)
        self.assertIsNone(production.reference_run)

    def test_scene_cli_dispatches_reference_free_production_mode(self):
        args = build_parser().parse_args([
            "scene-publish", "--config", "/tmp/config.yaml",
            "--tile-plan", "/tmp/plan", "--pose-artifact-root", "/tmp/poses",
            "--scene-tile-run", "tile-0000=/tmp/run-a",
            "--scene-tile-run", "tile-0001=/tmp/run-b",
            "--scene-name", "scene-v2", "--scene-mode", "production",
            "--diagnostic-scene", "--max-combined-gaussians", "123",
            "--scene-device", "cpu",
        ])
        cfg = SimpleNamespace(
            paths=SimpleNamespace(segment=Path("/tmp/segment"), workdir=Path("/tmp/work")),
            pose=SimpleNamespace(artifact="pose-v1"),
        )
        with mock.patch(
            "rtk_splat.workflows.tile_scene.publish_production_tiled_scene",
            return_value={"quality_passed": True},
        ) as publish:
            cmd_scene_publish(cfg, args)
        kwargs = publish.call_args.kwargs
        self.assertNotIn("reference_run", kwargs)
        self.assertNotIn("reference_hashes", kwargs)
        self.assertTrue(
            kwargs["allow_nonproduction_georeferencing_for_diagnostic"]
        )
        self.assertEqual(kwargs["maximum_combined_gaussians"], 123)

    def test_combined_evaluation_keeps_full_scene_on_cpu_and_culls_per_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            viewmats = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
            reader = SimpleNamespace(
                root=root,
                calibration={"cameras": {"left": {
                    "K": [[10, 0, 5], [0, 10, 5], [0, 0, 1]],
                    "width": 10, "height": 10,
                }}},
                frames={},
            )
            cfg = SimpleNamespace(train=SimpleNamespace(
                pose_opt=SimpleNamespace()
            ))
            params = _params([
                [0.0, 0.0, 2.0],
                [100.0, 0.0, 2.0],
                [0.0, 0.0, 30.0],
            ])
            params["scales"][:] = np.log(0.1)

            selected_counts = []

            def evaluate(selected, *_args, **_kwargs):
                selected_counts.append(len(selected["means"]))
                return {
                    "psnr": 20.0, "psnr_masked": 21.0,
                    "psnr_near": 22.0, "psnr_masked_cc": 23.0,
                    "lpips": 0.4, "lpips_cc": 0.3, "ssim": 0.6,
                    "n_eval": 1,
                }

            with (
                mock.patch(
                    "rtk_splat.workflows.tile_scene.load_pose_artifact",
                    return_value=(viewmats, np.zeros((2, 3))),
                ),
                mock.patch(
                    "rtk_splat.backends.gsplat.evaluate",
                    side_effect=evaluate,
                ) as renderer,
            ):
                result = _evaluate_combined(
                    params, reader, cfg, root, [0, 1], device="cpu",
                    maximum_visible_gaussians=1,
                    maximum_render_depth_m=14.0,
                )
                uncapped = _evaluate_combined(
                    params, reader, cfg, root, [0, 1], device="cpu",
                    maximum_visible_gaussians=2,
                    maximum_render_depth_m=None,
                )
            self.assertEqual(renderer.call_count, 4)
            self.assertEqual(selected_counts, [1, 1, 2, 2])
            self.assertEqual(result["n_eval"], 2)
            self.assertEqual(
                result["frustum_culling"]["maximum_selected_gaussians"], 1
            )
            self.assertIsNone(
                uncapped["frustum_culling"]["maximum_render_depth_m"]
            )

    def test_expected_binding_recomputes_every_selection_and_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = _plan()
            inventory = _inventory(plan)
            (root / "manifest.json").write_bytes(b"manifest")
            (root / "tile_plan.json").write_bytes(b"plan")
            binding = _expected_tile_binding(
                plan, root, inventory, plan["tiles"][1], 1
            )
            self.assertEqual(binding["tile_id"], "tile-0001")
            self.assertEqual(binding["tile_index"], 1)
            self.assertEqual(
                binding["train_selection_sha256"], canonical_hash([2, 3])
            )
            self.assertEqual(
                binding["val_selection_sha256"], canonical_hash([0, 8])
            )
            self.assertEqual(
                binding["core_bounds_uv_m"], [[1.0, 0.0], [2.0, 1.0]]
            )

    def test_core_concatenation_uses_exact_half_open_owner_rule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / "a.pt", root / "b.pt"]
            for path in paths:
                path.write_bytes(path.name.encode())
            loaded = {
                paths[0]: _params([[0.5, 0.5, 0], [1.5, 0.5, 0]]),
                paths[1]: _params([
                    [0.5, 0.5, 0], [1.0, 0.5, 0], [2.0, 1.0, 0]
                ]),
            }

            def load(path, _device):
                return loaded[Path(path)]

            completed = [
                (_plan()["tiles"][0], paths[0], _sha(paths[0])),
                (_plan()["tiles"][1], paths[1], _sha(paths[1])),
            ]
            with mock.patch(
                "rtk_splat.backends.gsplat._load_checkpoint_gaussians",
                side_effect=load,
            ):
                combined, records = _concatenate_core_params(_plan(), completed)
            np.testing.assert_allclose(
                combined["means"].numpy(),
                [[0.5, 0.5, 0], [1.0, 0.5, 0], [2.0, 1.0, 0]],
            )
            self.assertEqual(
                [item["core_owned_gaussians"] for item in records], [1, 2]
            )

    def test_core_concatenation_uses_global_indices_for_nonconsecutive_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = _plan()
            plan["partition"]["scene_bounds_uv_m"] = [[0.0, 0.0], [3.0, 1.0]]
            plan["tiles"].append({
                "tile_id": "tile-0002",
                "core_bounds_uv_m": [[2.0, 0.0], [3.0, 1.0]],
                "context_bounds_uv_m": [[1.75, -0.25], [3.25, 1.25]],
                "frame_ids": {"train": [3], "val": [0], "test": []},
            })
            paths = [root / "a.pt", root / "c.pt"]
            for path in paths:
                path.write_bytes(path.name.encode())
            loaded = {
                paths[0]: _params([[0.5, 0.5, 0], [1.5, 0.5, 0]]),
                paths[1]: _params([[1.5, 0.5, 0], [2.5, 0.5, 0]]),
            }
            completed = [
                (plan["tiles"][0], paths[0], _sha(paths[0])),
                (plan["tiles"][2], paths[1], _sha(paths[1])),
            ]
            with mock.patch(
                "rtk_splat.backends.gsplat._load_checkpoint_gaussians",
                side_effect=lambda path, _device: loaded[Path(path)],
            ):
                combined, records = _concatenate_core_params(plan, completed)
            np.testing.assert_allclose(
                combined["means"].numpy(),
                [[0.5, 0.5, 0], [2.5, 0.5, 0]],
            )
            self.assertEqual(
                [item["core_owned_gaussians"] for item in records], [1, 1]
            )

    def test_frozen_reference_requires_hashes_and_exact_source_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "reference"
            run.mkdir()
            (run / "params.pt").write_bytes(b"params")
            provenance = {
                "pose_artifact": "pose-v1",
                "pose_fingerprint": _plan()["source_binding"]["pose_fingerprint"],
                "manifest_sha256": "1" * 64,
                "frames_sha256": "2" * 64,
                "calibration_sha256": "3" * 64,
                "segment_meta_sha256": "4" * 64,
                "training_implementation_sha256": "5" * 64,
                "effective_training_config": {
                    "train": {"run_name": "reference"}
                },
            }
            _write_json(run / "run_provenance.json", provenance)
            metrics = [{
                "step": 65_000,
                "n_eval": 2,
                "psnr_masked": 24.3,
                "psnr_masked_cc": 25.7,
                "ssim": 0.58,
                "lpips_cc": 0.32,
                "per_frame": {"eval_ids": [0, 8]},
            }]
            _write_json(run / "metrics.json", metrics)
            hashes = {
                "params_sha256": _sha(run / "params.pt"),
                "metrics_sha256": _sha(run / "metrics.json"),
                "provenance_sha256": _sha(run / "run_provenance.json"),
            }
            values, evidence = _reference_run_evidence(
                run, hashes, _plan(), _inventory(_plan()), [0, 8]
            )
            self.assertEqual(values["psnr_masked"], 24.3)
            self.assertEqual(evidence["metrics_sha256"], hashes["metrics_sha256"])
            metrics[0]["per_frame"]["eval_ids"] = [0, 16]
            _write_json(run / "metrics.json", metrics)
            hashes["metrics_sha256"] = _sha(run / "metrics.json")
            with self.assertRaisesRegex(ValueError, "validation split"):
                _reference_run_evidence(
                    run, hashes, _plan(), _inventory(_plan()), [0, 8]
                )

    def test_scene_publication_is_atomic_provisional_and_not_averaged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment = root / "segment"
            segment.mkdir()
            plan_root = root / "plan"
            plan_root.mkdir()
            plan = _plan()
            inventory = _inventory(plan)
            georeferencing = {
                "schema_version": 1,
                "pose_artifact": "pose-v1",
                "artifact_class": "legacy_unassessed",
                "georeferencing_status": "legacy_unassessed",
                "metric_georeferencing_claim_eligible": False,
                "pose_quality_sha256": None,
                "pose_manifest_sha256": None,
                "pose_georeferencing_sha256": None,
            }
            _write_json(plan_root / "tile_plan.json", plan)
            _write_json(plan_root / "source_inventory.json", inventory)
            _write_json(
                plan_root / "provenance.json", {"georeferencing": georeferencing}
            )
            (plan_root / "manifest.json").write_bytes(b"plan seal")
            runs = {}
            for tile in plan["tiles"]:
                run = root / tile["tile_id"]
                run.mkdir()
                (run / "params.pt").write_bytes(tile["tile_id"].encode())
                (run / "run_provenance.json").write_bytes(b"provenance")
                (run / "best_checkpoint.json").write_bytes(b"best")
                runs[tile["tile_id"]] = run
            reference = root / "reference"
            reference.mkdir()
            viewmats = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
            reader = SimpleNamespace(
                root=segment,
                manifest={"train": [1], "val": [0, 8], "test": []},
                calibration={
                    "cameras": {"left": {
                        "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "width": 2,
                        "height": 2,
                    }}
                },
                frames={},
            )
            reader.validate = lambda: reader
            cfg = SimpleNamespace(
                pose=SimpleNamespace(artifact="pose-v1"),
                depth=SimpleNamespace(max_z_m=12.0),
                train=SimpleNamespace(pose_opt=SimpleNamespace()),
            )
            params = _params([[0.5, 0.5, 0], [1.5, 0.5, 0]])
            ownership = [
                {
                    "tile_id": "tile-0000", "source_params": "a",
                    "source_params_sha256": "a" * 64,
                    "source_gaussians": 2, "core_owned_gaussians": 1,
                    "outside_or_other_core_gaussians": 1,
                },
                {
                    "tile_id": "tile-0001", "source_params": "b",
                    "source_params_sha256": "b" * 64,
                    "source_gaussians": 2, "core_owned_gaussians": 1,
                    "outside_or_other_core_gaussians": 1,
                },
            ]

            def completed(run, binding, _georef):
                return (
                    {
                        "model_selection": {"status": "complete"},
                        "tile_plan": binding,
                        "effective_training_config_sha256": "d" * 64,
                        "training_implementation_sha256": "9" * 64,
                        "effective_training_config": {
                            "train": {"run_name": Path(run).name}
                        },
                    },
                    Path(run) / "params.pt",
                    _sha(Path(run) / "params.pt"),
                    {
                        "initial_cloud_sha256": "e" * 64,
                        "splat_file": "splat.ply",
                        "splat_sha256": "f" * 64,
                    },
                )

            def export(_params, output, **_kwargs):
                Path(output).write_bytes(b"ply")
                return 2, 2

            candidate = {
                "psnr_masked": 24.2,
                "psnr_masked_cc": 25.5,
                "ssim": 0.57,
                "lpips_cc": 0.33,
                "n_eval": 2,
                "eval_ids": [0, 8],
                "pose_fingerprint": plan["source_binding"]["pose_fingerprint"],
            }
            reference_metrics = {
                "psnr_masked": 24.3,
                "psnr_masked_cc": 25.7,
                "ssim": 0.58,
                "lpips_cc": 0.32,
            }
            training_identity = {
                "training_implementation_sha256": "9" * 64,
                "comparable_training_config_sha256": canonical_hash({
                    "train": {}
                }),
            }
            candidate_seam = {
                **candidate,
                "psnr_masked": 23.9,
                "psnr_masked_cc": 25.0,
                "lpips_cc": 0.35,
            }
            reference_seam = {
                **candidate,
                "psnr_masked": 24.0,
                "psnr_masked_cc": 25.2,
                "lpips_cc": 0.33,
            }
            seam_mask = np.ones((2, 2), dtype=bool)
            with (
                mock.patch(
                    "rtk_splat.workflows.tile_scene.SegmentReader",
                    return_value=reader,
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_scene.verify_tile_plan",
                    return_value=plan,
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_scene._reference_run_evidence",
                    return_value=(
                        reference_metrics,
                        {
                            "run": str(reference),
                            "files": {},
                            "training_identity": training_identity,
                        },
                    ),
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_scene._completed_tile_run",
                    side_effect=completed,
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_scene._concatenate_core_params",
                    return_value=(params, ownership),
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_scene._evaluate_combined",
                    side_effect=[candidate, candidate_seam, reference_seam],
                ) as evaluation,
                mock.patch(
                    "rtk_splat.workflows.tile_scene._build_seam_masks",
                    return_value=(
                        {0: seam_mask, 8: seam_mask},
                        {
                            "policy": "synthetic",
                            "band_m": 1.0,
                            "minimum_pixels_per_frame": 1,
                            "n_validation_frames": 2,
                            "metric_depth_pixels_in_band": 512,
                            "internal_segments_uv": [[0, 1.0, 0.0, 1.0]],
                            "frames": [],
                        },
                    ),
                ),
                mock.patch(
                    "rtk_splat.backends.gsplat._load_checkpoint_gaussians",
                    return_value=params,
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_scene.load_pose_artifact",
                    return_value=(viewmats, np.zeros((2, 3))),
                ),
                mock.patch(
                    "rtk_splat.backends.gsplat.export_splat_tensors",
                    side_effect=export,
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_scene.collect_package_state",
                    return_value={"python_tree_sha256": "c" * 64},
                ),
            ):
                result = publish_tiled_scene(
                    segment=segment,
                    cfg=cfg,
                    tile_plan_root=plan_root,
                    pose_root=root / "pose",
                    tile_runs=runs,
                    reference_run=reference,
                    reference_hashes={
                        "params_sha256": "a" * 64,
                        "metrics_sha256": "b" * 64,
                        "provenance_sha256": "c" * 64,
                    },
                    output_root=root / "output",
                    scene_name="scene-v1",
                    maximum_combined_gaussians=2,
                    device="cpu",
                )
            self.assertTrue(result["quality_passed"])
            self.assertTrue(result["provisional"])
            self.assertEqual(evaluation.call_count, 3)
            scene_root = Path(result["scene"])
            self.assertTrue((scene_root / "scene.PROVISIONAL.ply").is_file())
            scene_json = verify_tiled_scene(scene_root)
            self.assertEqual(scene_json["ownership"]["core_owned_gaussians"], 2)
            metrics = json.loads((scene_root / "metrics.json").read_text())
            self.assertEqual(
                metrics["evaluation"]["n_unique_validation_frames"], 2
            )
            with self.assertRaises(FileExistsError):
                publish_tiled_scene(
                    segment=segment,
                    cfg=cfg,
                    tile_plan_root=plan_root,
                    pose_root=root / "pose",
                    tile_runs=runs,
                    reference_run=reference,
                    reference_hashes={
                        "params_sha256": "a" * 64,
                        "metrics_sha256": "b" * 64,
                        "provenance_sha256": "c" * 64,
                    },
                    output_root=root / "output",
                    scene_name="scene-v1",
                    maximum_combined_gaussians=2,
                    device="cpu",
                )
            (scene_root / "quality.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "terminal seal"):
                verify_tiled_scene(scene_root)

    def test_production_publication_is_absolute_complete_and_reference_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment = root / "segment"
            segment.mkdir()
            plan_root = root / "plan"
            plan_root.mkdir()
            plan = _plan()
            plan["provisional"] = False
            plan["metric_georeferencing_claim_eligible"] = True
            inventory = _inventory(plan)
            georeferencing = {
                "schema_version": 1,
                "pose_artifact": "pose-v1",
                "artifact_class": "production",
                "georeferencing_status": "PASSED",
                "metric_georeferencing_claim_eligible": True,
                "pose_quality_sha256": "6" * 64,
                "pose_manifest_sha256": "7" * 64,
                "pose_georeferencing_sha256": "8" * 64,
            }
            _write_json(plan_root / "tile_plan.json", plan)
            _write_json(plan_root / "source_inventory.json", inventory)
            _write_json(
                plan_root / "provenance.json", {"georeferencing": georeferencing}
            )
            (plan_root / "manifest.json").write_bytes(b"plan seal")
            runs = {}
            for tile in plan["tiles"]:
                run = root / tile["tile_id"]
                run.mkdir()
                (run / "params.pt").write_bytes(tile["tile_id"].encode())
                (run / "run_provenance.json").write_bytes(b"provenance")
                (run / "best_checkpoint.json").write_bytes(b"best")
                (run / "splat.ply").write_bytes(b"source splat")
                runs[tile["tile_id"]] = run
            viewmats = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
            reader = SimpleNamespace(
                root=segment,
                manifest={"train": [1], "val": [0, 8], "test": []},
                calibration={
                    "cameras": {"left": {
                        "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "width": 2,
                        "height": 2,
                    }}
                },
                frames={},
            )
            reader.validate = lambda: reader
            cfg = SimpleNamespace(
                pose=SimpleNamespace(artifact="pose-v1"),
                depth=SimpleNamespace(max_z_m=12.0),
                train=SimpleNamespace(pose_opt=SimpleNamespace()),
            )
            params = _params([[0.5, 0.5, 0], [1.5, 0.5, 0]])
            ownership = [
                {
                    "tile_id": "tile-0000", "source_params": "a",
                    "source_params_sha256": "a" * 64,
                    "source_gaussians": 2, "core_owned_gaussians": 1,
                    "outside_or_other_core_gaussians": 1,
                },
                {
                    "tile_id": "tile-0001", "source_params": "b",
                    "source_params_sha256": "b" * 64,
                    "source_gaussians": 2, "core_owned_gaussians": 1,
                    "outside_or_other_core_gaussians": 1,
                },
            ]

            def completed(run, binding, _georef, **_kwargs):
                contracts = inventory["segment"]["contract_files"]
                provenance = {
                    "model_selection": {"status": "complete"},
                    "tile_plan": binding,
                    "effective_training_config_sha256": "d" * 64,
                    "training_implementation_sha256": "9" * 64,
                    "effective_training_config": {
                        "train": {"run_name": Path(run).name}
                    },
                    "manifest_sha256": contracts["manifest.json"]["sha256"],
                    "frames_sha256": contracts["frames.npz"]["sha256"],
                    "calibration_sha256": contracts["calibration.json"]["sha256"],
                    "segment_meta_sha256": contracts["segment_meta.json"]["sha256"],
                    "pose_artifact": inventory["pose"]["name"],
                    "pose_fingerprint": inventory["pose"]["fingerprint"],
                }
                return (
                    provenance,
                    Path(run) / "params.pt",
                    _sha(Path(run) / "params.pt"),
                    {
                        "initial_cloud_sha256": "e" * 64,
                        "splat_file": "splat.ply",
                        "splat_sha256": _sha(Path(run) / "splat.ply"),
                    },
                )

            absolute = {
                "psnr_masked": 24.8,
                "psnr_masked_cc": 26.1,
                "ssim": 0.61,
                "lpips_cc": 0.29,
                "n_eval": 2,
                "eval_ids": [0, 8],
                "pose_fingerprint": plan["source_binding"]["pose_fingerprint"],
            }
            seam_absolute = {
                **absolute,
                "psnr_masked": 23.6,
                "psnr_masked_cc": 25.1,
                "ssim": 0.58,
                "lpips_cc": 0.11,
            }
            seam_mask = np.ones((2, 2), dtype=bool)

            def export(_params, output, **_kwargs):
                Path(output).write_bytes(b"ply")
                return 2, 2

            with (
                mock.patch(
                    "rtk_splat.workflows.tile_scene.SegmentReader",
                    return_value=reader,
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_scene.verify_tile_plan",
                    return_value=plan,
                ) as plan_verifier,
                mock.patch(
                    "rtk_splat.workflows.tile_scene._completed_tile_run",
                    side_effect=completed,
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_scene._concatenate_core_params",
                    return_value=(params, ownership),
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_scene._evaluate_combined",
                    side_effect=[
                        absolute,
                        seam_absolute,
                        absolute,
                        seam_absolute,
                    ],
                ) as evaluation,
                mock.patch(
                    "rtk_splat.workflows.tile_scene._build_seam_masks",
                    return_value=(
                        {0: seam_mask, 8: seam_mask},
                        {
                            "policy": "synthetic",
                            "band_m": 1.0,
                            "minimum_pixels_per_frame": 1,
                            "n_validation_frames": 2,
                            "metric_depth_pixels_in_band": 512,
                            "internal_segments_uv": [[0, 1.0, 0.0, 1.0]],
                            "frames": [],
                        },
                    ),
                ) as mask_builder,
                mock.patch(
                    "rtk_splat.workflows.tile_scene.load_pose_artifact",
                    return_value=(viewmats, np.zeros((2, 3))),
                ) as pose_loader,
                mock.patch(
                    "rtk_splat.backends.gsplat.export_splat_tensors",
                    side_effect=export,
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_scene.collect_package_state",
                    return_value={"python_tree_sha256": "c" * 64},
                ),
            ):
                result = publish_production_tiled_scene(
                    segment=segment,
                    cfg=cfg,
                    tile_plan_root=plan_root,
                    pose_root=root / "pose",
                    tile_runs=runs,
                    output_root=root / "output",
                    scene_name="production-scene-v1",
                    maximum_combined_gaussians=2,
                    device="cpu",
                )
                plan["provisional"] = True
                plan["metric_georeferencing_claim_eligible"] = False
                diagnostic_georeferencing = {
                    **georeferencing,
                    "artifact_class": "diagnostic_render_only",
                    "georeferencing_status": "FAILED",
                    "metric_georeferencing_claim_eligible": False,
                }
                _write_json(plan_root / "provenance.json", {
                    "georeferencing": diagnostic_georeferencing
                })
                diagnostic_result = publish_production_tiled_scene(
                    segment=segment,
                    cfg=cfg,
                    tile_plan_root=plan_root,
                    pose_root=root / "pose",
                    tile_runs=runs,
                    output_root=root / "output",
                    scene_name="diagnostic-scene-v1",
                    maximum_combined_gaussians=2,
                    device="cpu",
                    allow_nonproduction_georeferencing_for_diagnostic=True,
                )
            self.assertEqual(evaluation.call_count, 4)
            self.assertEqual(plan_verifier.call_count, 4)
            self.assertFalse(
                mask_builder.call_args_list[0].kwargs[
                    "allow_failed_georeferencing_for_render"
                ]
            )
            self.assertTrue(
                mask_builder.call_args_list[1].kwargs[
                    "allow_failed_georeferencing_for_render"
                ]
            )
            self.assertFalse(
                pose_loader.call_args_list[0].kwargs[
                    "allow_failed_georeferencing_for_render"
                ]
            )
            self.assertTrue(
                pose_loader.call_args_list[1].kwargs[
                    "allow_failed_georeferencing_for_render"
                ]
            )
            self.assertTrue(result["quality_passed"])
            self.assertFalse(result["provisional"])
            self.assertEqual(result["n_tiles"], 2)
            scene_root = Path(result["scene"])
            self.assertTrue((scene_root / "scene.ply").is_file())
            verified = verify_tiled_scene(scene_root)
            self.assertEqual(verified["publication_mode"], "production")
            self.assertTrue(diagnostic_result["quality_passed"])
            self.assertTrue(diagnostic_result["provisional"])
            self.assertFalse(
                diagnostic_result["metric_georeferencing_claim_eligible"]
            )
            self.assertTrue(
                (
                    Path(diagnostic_result["scene"])
                    / "scene.DIAGNOSTIC_ONLY.ply"
                ).is_file()
            )
            metrics = json.loads((scene_root / "metrics.json").read_text())
            self.assertEqual(
                metrics["evaluation"]["comparison"],
                "absolute_only_no_monolithic_reference",
            )
            for forbidden in ("reference", "candidate", "loss_db", "regressions"):
                self.assertNotIn(forbidden, metrics)
                self.assertNotIn(forbidden, metrics["seam"])

    def test_production_publication_rejects_unassessed_georeferencing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment = root / "segment"
            segment.mkdir()
            plan_root = root / "plan"
            plan_root.mkdir()
            plan = _plan()
            _write_json(plan_root / "source_inventory.json", _inventory(plan))
            _write_json(plan_root / "provenance.json", {
                "georeferencing": {
                    "artifact_class": "legacy_unassessed",
                    "georeferencing_status": "legacy_unassessed",
                    "metric_georeferencing_claim_eligible": False,
                }
            })
            reader = SimpleNamespace(root=segment)
            reader.validate = lambda: reader
            with (
                mock.patch(
                    "rtk_splat.workflows.tile_scene.SegmentReader",
                    return_value=reader,
                ),
                mock.patch(
                    "rtk_splat.workflows.tile_scene.verify_tile_plan",
                    return_value=plan,
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "requires a PASSED production pose"
                ):
                    publish_production_tiled_scene(
                        segment=segment,
                        cfg=SimpleNamespace(),
                        tile_plan_root=plan_root,
                        pose_root=root / "pose",
                        tile_runs={},
                        output_root=root / "output",
                        scene_name="must-not-publish",
                        maximum_combined_gaussians=2,
                        device="cpu",
                    )
            self.assertFalse((root / "output" / "scene_artifacts").exists())


if __name__ == "__main__":
    unittest.main()
