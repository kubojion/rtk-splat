import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rtk_splat.frontends.artifact import ArtifactError
from rtk_splat.workflows.export_layered_scene import (
    _ply_evidence,
    export_layered_scene_ply,
    verify_layered_scene_ply_export,
)


_PROPERTIES = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _ply(path: Path, rows, *, extra_property: bool = False) -> None:
    properties = (*_PROPERTIES, "f_rest_0") if extra_property else _PROPERTIES
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(rows)}",
        *(f"property float {name}" for name in properties),
        "end_header",
        "",
    ]
    with path.open("wb") as stream:
        stream.write("\n".join(header).encode("ascii"))
        for value in rows:
            stream.write(
                struct.pack(
                    "<" + "f" * len(properties), *([value] * len(properties))
                )
            )


def _scene(root: Path, *, diagnostic: bool = True, schema_mismatch: bool = False):
    scene = root / "scene"
    scene.mkdir()
    layers = []
    ownership = []
    for index, count in enumerate((2, 3)):
        tile_id = f"tile-{index:04d}"
        run = root / tile_id
        run.mkdir()
        source = run / ("splat.DIAGNOSTIC_ONLY.ply" if diagnostic else "splat.ply")
        _ply(
            source,
            range(index * 10, index * 10 + count),
            extra_property=(schema_mismatch and index == 1),
        )
        digest = _sha(source)
        bounds = [[float(index), 0.0], [float(index + 1), 1.0]]
        layer = {
            "tile_id": tile_id,
            "run": str(run),
            "source_splat_file": source.name,
            "source_splat_sha256": digest,
            "params_sha256": "a" * 64,
            "core_bounds_uv_m": bounds,
            "context_bounds_uv_m": bounds,
        }
        layers.append(layer)
        ownership.append({"tile_id": tile_id, "core_owned_gaussians": count})
        _json(
            run / "splat.georeferencing.json",
            {
                "splat_file": source.name,
                "splat_sha256": digest,
                "tile_plan": {
                    "tile_id": tile_id,
                    "core_bounds_uv_m": bounds,
                    "boundary_rule": "half-open",
                },
            },
        )
    scene_record = {
        "schema_version": 1,
        "name": "scene",
        "representation": "sealed_tile_layers",
        "layer_index_file": "layers.json",
        "metric_georeferencing_claim_eligible": not diagnostic,
        "partition": {"boundary_rule": "half-open"},
        "ownership": {"tiles": ownership},
    }
    _json(scene / "scene.json", scene_record)
    _json(scene / "layers.json", {"schema_version": 1, "layers": layers})
    _json(
        scene / "provenance.json",
        {
            "georeferencing": {
                "artifact_class": "diagnostic_render_only" if diagnostic else "production",
                "georeferencing_status": "FAILED" if diagnostic else "PASSED",
                "metric_georeferencing_claim_eligible": not diagnostic,
            }
        },
    )
    (scene / "manifest.json").write_bytes(b"sealed-scene")
    return scene, scene_record


class LayeredSceneExportTests(unittest.TestCase):
    def test_export_concatenates_core_plys_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene, scene_record = _scene(root)
            destination = root / "export"
            with mock.patch(
                "rtk_splat.workflows.export_layered_scene.verify_tiled_scene",
                return_value=scene_record,
            ):
                result = export_layered_scene_ply(scene, destination)
                self.assertEqual(result["retained_gaussians"], 5)
                self.assertFalse(result["metric_georeferencing_claim_eligible"])
                output = destination / "scene.DIAGNOSTIC_ONLY.ply"
                self.assertEqual(_ply_evidence(output)["vertex_count"], 5)
                self.assertEqual(
                    verify_layered_scene_ply_export(destination)[
                        "retained_gaussians"
                    ],
                    5,
                )
                with self.assertRaises(FileExistsError):
                    export_layered_scene_ply(scene, destination)

    def test_mismatched_source_schema_fails_without_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene, scene_record = _scene(root, schema_mismatch=True)
            destination = root / "must-not-exist"
            with mock.patch(
                "rtk_splat.workflows.export_layered_scene.verify_tiled_scene",
                return_value=scene_record,
            ):
                with self.assertRaisesRegex(ArtifactError, "schemas do not match"):
                    export_layered_scene_ply(scene, destination)
            self.assertFalse(destination.exists())

    def test_source_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene, scene_record = _scene(root)
            source = root / "tile-0001" / "splat.DIAGNOSTIC_ONLY.ply"
            with source.open("ab") as stream:
                stream.write(b"tampered")
            destination = root / "must-not-exist"
            with mock.patch(
                "rtk_splat.workflows.export_layered_scene.verify_tiled_scene",
                return_value=scene_record,
            ):
                with self.assertRaisesRegex(ArtifactError, "sealed source PLY changed"):
                    export_layered_scene_ply(scene, destination)
            self.assertFalse(destination.exists())

    def test_published_payload_tampering_breaks_terminal_seal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene, scene_record = _scene(root)
            destination = root / "export"
            with mock.patch(
                "rtk_splat.workflows.export_layered_scene.verify_tiled_scene",
                return_value=scene_record,
            ):
                export_layered_scene_ply(scene, destination)
            with (destination / "scene.DIAGNOSTIC_ONLY.ply").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ArtifactError, "terminal seal"):
                verify_layered_scene_ply_export(destination)

    def test_injected_copy_failure_leaves_no_published_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene, scene_record = _scene(root)
            destination = root / "must-not-exist"
            with (
                mock.patch(
                    "rtk_splat.workflows.export_layered_scene.verify_tiled_scene",
                    return_value=scene_record,
                ),
                mock.patch(
                    "rtk_splat.workflows.export_layered_scene._copy_body",
                    side_effect=RuntimeError("injected failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    export_layered_scene_ply(scene, destination)
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".must-not-exist.writing-*")), [])

    def test_symlink_source_and_destination_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene, scene_record = _scene(root)
            linked_scene = root / "linked-scene"
            linked_scene.symlink_to(scene, target_is_directory=True)
            with self.assertRaisesRegex(ArtifactError, "source scene cannot"):
                export_layered_scene_ply(linked_scene, root / "output")

            linked_destination = root / "linked-output"
            linked_destination.symlink_to(root / "absent", target_is_directory=True)
            with mock.patch(
                "rtk_splat.workflows.export_layered_scene.verify_tiled_scene",
                return_value=scene_record,
            ):
                with self.assertRaisesRegex(ArtifactError, "destination cannot"):
                    export_layered_scene_ply(scene, linked_destination)


if __name__ == "__main__":
    unittest.main()
