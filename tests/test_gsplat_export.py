import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from rtk_splat.backends.gsplat import export_splat_tensors
from rtk_splat.workflows.export_splat import (
    build_parser,
    export_completed_splat,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _parameters() -> dict[str, torch.Tensor]:
    return {
        "means": torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]]
        ),
        "scales": torch.zeros(3, 3),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3),
        "opacities": torch.tensor([-10.0, 0.0, 10.0]),
        "sh0": torch.zeros(3, 1, 3),
        "shN": torch.zeros(3, 3, 3),
    }


def _completed_run(root: Path) -> Path:
    run = root / "run"
    run.mkdir()
    torch.save(_parameters(), run / "params.pt")
    params_hash = _sha256(run / "params.pt")
    georeferencing = {
        "schema_version": 1,
        "pose_artifact": "pose-v1",
        "artifact_class": "production",
        "georeferencing_status": "PASSED",
        "metric_georeferencing_claim_eligible": True,
        "diagnostic_export_requested": False,
        "diagnostic_export_override_used": False,
    }
    _json(run / "georeferencing.json", georeferencing)
    (run / "splat.ply").write_bytes(b"sealed-source-ply")
    _json(
        run / "splat.georeferencing.json",
        {
            **georeferencing,
            "splat_file": "splat.ply",
            "splat_sha256": _sha256(run / "splat.ply"),
        },
    )
    selection = {
        "schema_version": 1,
        "status": "complete",
        "criterion": "psnr_masked_cc",
        "direction": "maximize",
        "tie_policy": "earliest_evaluation_wins",
        "best_step": 10,
        "best_metric": 20.0,
        "completed_training_steps": 10,
        "parameter_artifact": "params.pt",
        "parameter_artifact_sha256": params_hash,
        "exported_splat_uses_best_model": True,
        "optimizer_state_included": False,
    }
    _json(run / "best_checkpoint.json", selection)
    _json(
        run / "run_provenance.json",
        {
            "schema_version": 1,
            "iterations": 10,
            "pose_artifact": "pose-v1",
            "georeferencing": georeferencing,
            "model_selection": selection,
        },
    )
    return run


class GsplatExportTests(unittest.TestCase):
    def test_tensor_export_uses_strict_opacity_and_crop_filters(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "filtered.ply"
            total, retained = export_splat_tensors(
                _parameters(),
                output,
                opacity_threshold=0.01,
                crop_bounds=(
                    torch.tensor([-0.1, -0.1, -0.1]),
                    torch.tensor([1.0, 0.1, 0.1]),
                ),
            )
            self.assertEqual((total, retained), (3, 1))
            self.assertGreater(output.stat().st_size, 0)

    def test_completed_export_is_separate_sealed_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _completed_run(Path(temporary))
            protected = {
                path.name: _sha256(path)
                for path in run.iterdir()
                if path.is_file()
            }
            result = export_completed_splat(
                run,
                "full-v1",
                opacity_threshold=0.0,
                crop=False,
            )
            destination = Path(result["export"])
            self.assertEqual(result["retained_gaussians"], 3)
            self.assertTrue((destination / "splat.ply").is_file())
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(
                set(manifest["files"]),
                {
                    "export_provenance.json",
                    "splat.georeferencing.json",
                    "splat.ply",
                },
            )
            for name, digest in protected.items():
                self.assertEqual(_sha256(run / name), digest)
            with self.assertRaises(FileExistsError):
                export_completed_splat(
                    run,
                    "full-v1",
                    opacity_threshold=0.0,
                    crop=False,
                )

    def test_changed_params_are_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _completed_run(Path(temporary))
            with (run / "params.pt").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "params.pt changed"):
                export_completed_splat(
                    run,
                    "must-not-exist",
                    opacity_threshold=0.0,
                    crop=False,
                )
            self.assertFalse((run / "exports" / "must-not-exist").exists())

    def test_parser_requires_explicit_crop_policy(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "--source-run",
                    "/tmp/run",
                    "--export-name",
                    "test",
                    "--opacity-threshold",
                    "0.0",
                ]
            )
        args = build_parser().parse_args(
            [
                "--source-run",
                "/tmp/run",
                "--export-name",
                "test",
                "--opacity-threshold",
                "0.0",
                "--no-crop",
            ]
        )
        self.assertFalse(args.crop)

    def test_export_name_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _completed_run(Path(temporary))
            with self.assertRaisesRegex(ValueError, "invalid export name"):
                export_completed_splat(
                    run,
                    "../escape",
                    opacity_threshold=0.0,
                    crop=False,
                )


if __name__ == "__main__":
    unittest.main()
