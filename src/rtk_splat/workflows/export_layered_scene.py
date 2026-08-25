"""Export a sealed layered tiled scene as one portable Gaussian PLY.

The layered renderer is the authoritative representation because it blends
complete context models per view.  This exporter instead concatenates the
already core-owned, opacity-pruned viewer PLY from every sealed tile.  The
result is useful in standard Gaussian-splat viewers, but it intentionally does
not claim to reproduce the layered compositor at tile boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from rtk_splat.core.segment import publish_directory_noreplace
from rtk_splat.frontends.artifact import ArtifactError, sha256_file
from rtk_splat.workflows.tile_scene import verify_tiled_scene


SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TYPE_BYTES = {
    "char": 1,
    "uchar": 1,
    "int8": 1,
    "uint8": 1,
    "short": 2,
    "ushort": 2,
    "int16": 2,
    "uint16": 2,
    "int": 4,
    "uint": 4,
    "int32": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}
_REQUIRED_GAUSSIAN_PROPERTIES = {
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
}


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(value, allow_nan=False)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ArtifactError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON artifact must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _read_header(stream: BinaryIO, path: Path) -> tuple[list[bytes], int]:
    lines: list[bytes] = []
    size = 0
    while size <= 1024 * 1024:
        line = stream.readline()
        if not line:
            raise ArtifactError(f"PLY header is incomplete: {path}")
        size += len(line)
        lines.append(line)
        if line.rstrip(b"\r\n") == b"end_header":
            return lines, size
    raise ArtifactError(f"PLY header exceeds 1 MiB: {path}")


def _ply_evidence(path: Path) -> dict[str, Any]:
    """Validate one scalar binary little-endian Gaussian PLY."""
    if not path.is_file() or path.is_symlink():
        raise ArtifactError(f"Gaussian PLY is missing or unsafe: {path}")
    with path.open("rb") as stream:
        lines, header_size = _read_header(stream, path)
    try:
        decoded = [line.decode("ascii").rstrip("\r\n") for line in lines]
    except UnicodeDecodeError as exc:
        raise ArtifactError(f"PLY header is not ASCII: {path}") from exc
    if not decoded or decoded[0] != "ply":
        raise ArtifactError(f"PLY magic is invalid: {path}")
    if decoded.count("format binary_little_endian 1.0") != 1:
        raise ArtifactError(
            f"PLY must use binary_little_endian 1.0: {path}"
        )
    vertex_count = None
    vertex_line_index = None
    current_element = None
    properties: list[tuple[str, str]] = []
    for index, line in enumerate(decoded[1:], start=1):
        fields = line.split()
        if not fields or fields[0] in {"comment", "obj_info", "format"}:
            continue
        if fields[0] == "element":
            if len(fields) != 3:
                raise ArtifactError(f"invalid PLY element declaration: {path}")
            if fields[1] != "vertex" or vertex_count is not None:
                raise ArtifactError(
                    f"PLY must contain exactly one vertex element: {path}"
                )
            try:
                vertex_count = int(fields[2])
            except ValueError as exc:
                raise ArtifactError(f"invalid PLY vertex count: {path}") from exc
            if vertex_count <= 0:
                raise ArtifactError(f"PLY vertex count must be positive: {path}")
            vertex_line_index = index
            current_element = "vertex"
            continue
        if fields[0] == "property":
            if current_element != "vertex" or len(fields) != 3:
                raise ArtifactError(
                    f"PLY supports only scalar vertex properties: {path}"
                )
            scalar_type, name = fields[1], fields[2]
            if scalar_type not in _TYPE_BYTES or not name:
                raise ArtifactError(f"unsupported PLY property: {path}")
            properties.append((scalar_type, name))
            continue
        if fields[0] != "end_header":
            raise ArtifactError(f"unsupported PLY header directive: {path}")
    if vertex_count is None or vertex_line_index is None or not properties:
        raise ArtifactError(f"PLY has no usable vertex schema: {path}")
    names = [name for _, name in properties]
    if len(names) != len(set(names)) or not _REQUIRED_GAUSSIAN_PROPERTIES.issubset(
        names
    ):
        raise ArtifactError(f"PLY is not a supported Gaussian-splat schema: {path}")
    stride = sum(_TYPE_BYTES[scalar_type] for scalar_type, _ in properties)
    expected_size = header_size + vertex_count * stride
    actual_size = path.stat().st_size
    if expected_size != actual_size:
        raise ArtifactError(
            f"PLY body size disagrees with its header: {path} "
            f"({actual_size} != {expected_size})"
        )
    return {
        "path": str(path),
        "header_lines": lines,
        "header_size_bytes": header_size,
        "vertex_line_index": vertex_line_index,
        "vertex_count": vertex_count,
        "vertex_stride_bytes": stride,
        "properties": properties,
        "schema": ("binary_little_endian_1.0", tuple(properties)),
        "size_bytes": actual_size,
    }


def _source_layers(scene_root: Path, scene: Mapping[str, Any]) -> list[dict[str, Any]]:
    if scene.get("representation") != "sealed_tile_layers":
        raise ArtifactError("scene is not a sealed layered-tile representation")
    layers = _json_object(scene_root / str(scene.get("layer_index_file", "")))
    records = layers.get("layers")
    ownership = scene.get("ownership")
    owners = ownership.get("tiles") if isinstance(ownership, Mapping) else None
    if (
        not isinstance(records, list)
        or not records
        or not isinstance(owners, list)
        or [item.get("tile_id") for item in records]
        != [item.get("tile_id") for item in owners]
    ):
        raise ArtifactError("scene layer order disagrees with core ownership")
    result = []
    for layer in records:
        if not isinstance(layer, dict):
            raise ArtifactError("scene layer record is invalid")
        tile_id = layer.get("tile_id")
        run_value = layer.get("run")
        source_name = layer.get("source_splat_file")
        expected_hash = layer.get("source_splat_sha256")
        if (
            not isinstance(tile_id, str)
            or not isinstance(run_value, str)
            or not isinstance(source_name, str)
            or Path(source_name).name != source_name
            or not isinstance(expected_hash, str)
            or not _SHA256.fullmatch(expected_hash)
        ):
            raise ArtifactError("scene layer lacks safe source PLY evidence")
        run = Path(run_value).expanduser().resolve()
        source = run / source_name
        if sha256_file(source) != expected_hash:
            raise ArtifactError(f"sealed source PLY changed: {source}")
        sidecar = _json_object(run / "splat.georeferencing.json")
        binding = sidecar.get("tile_plan")
        if (
            sidecar.get("splat_file") != source_name
            or sidecar.get("splat_sha256") != expected_hash
            or not isinstance(binding, dict)
            or binding.get("tile_id") != tile_id
            or binding.get("core_bounds_uv_m") != layer.get("core_bounds_uv_m")
            or binding.get("boundary_rule")
            != scene.get("partition", {}).get("boundary_rule")
        ):
            raise ArtifactError(f"source PLY is not bound to {tile_id}'s core")
        evidence = _ply_evidence(source)
        result.append({
            "tile_id": tile_id,
            "run": str(run),
            "file": source_name,
            "sha256": expected_hash,
            "vertex_count": evidence["vertex_count"],
            "vertex_stride_bytes": evidence["vertex_stride_bytes"],
            "size_bytes": evidence["size_bytes"],
            "header_size_bytes": evidence["header_size_bytes"],
            "_path": source,
            "_ply": evidence,
        })
    return result


def _copy_body(source: Path, header_size: int, destination: BinaryIO) -> None:
    with source.open("rb") as stream:
        stream.seek(header_size)
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            destination.write(block)


def _export_manifest(root: Path) -> dict[str, Any]:
    files = {
        path.relative_to(root).as_posix(): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "manifest.json" and not path.is_symlink()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "layered_scene_core_ply_export",
        "files": files,
    }


def verify_layered_scene_ply_export(
    root: str | Path, *, rehash_sources: bool = False
) -> dict[str, Any]:
    supplied = Path(root).expanduser()
    if supplied.is_symlink():
        raise ArtifactError(f"PLY export cannot be a symlink: {supplied}")
    root = supplied.resolve()
    if not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise ArtifactError(f"PLY export is missing or contains symlinks: {root}")
    manifest = _json_object(root / "manifest.json")
    if manifest != _export_manifest(root):
        raise ArtifactError("PLY export terminal seal verification failed")
    record = _json_object(root / "export.json")
    output_name = record.get("splat_file")
    if not isinstance(output_name, str) or Path(output_name).name != output_name:
        raise ArtifactError("PLY export has an unsafe output name")
    output = root / output_name
    ply = _ply_evidence(output)
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("artifact_type") != "layered_scene_core_ply_export"
        or record.get("representation") != "hard_half_open_core_union_ply_v1"
        or record.get("splat_sha256") != sha256_file(output)
        or record.get("splat_size_bytes") != output.stat().st_size
        or record.get("retained_gaussians") != ply["vertex_count"]
        or record.get("source_gaussian_sum") != ply["vertex_count"]
        or record.get("metric_georeferencing_claim_eligible")
        is not (output_name == "scene.ply")
    ):
        raise ArtifactError("PLY export claims disagree with its payload")
    sources = record.get("sources")
    if (
        not isinstance(sources, list)
        or not sources
        or not all(isinstance(item, dict) for item in sources)
        or record.get("source_tile_count") != len(sources)
        or any(not isinstance(item.get("tile_id"), str) for item in sources)
        or len({item.get("tile_id") for item in sources}) != len(sources)
        or any(
            not isinstance(item.get("vertex_count"), int)
            or item["vertex_count"] <= 0
            for item in sources
        )
        or sum(item["vertex_count"] for item in sources) != ply["vertex_count"]
    ):
        raise ArtifactError("PLY export source inventory is invalid")
    georeferencing = _json_object(root / "splat.georeferencing.json")
    if (
        georeferencing.get("splat_file") != output_name
        or georeferencing.get("splat_sha256") != record.get("splat_sha256")
        or georeferencing.get("representation") != record.get("representation")
        or georeferencing.get("source_scene_manifest_sha256")
        != record.get("source_scene_manifest_sha256")
        or georeferencing.get("metric_georeferencing_claim_eligible")
        is not record.get("metric_georeferencing_claim_eligible")
    ):
        raise ArtifactError("PLY export georeferencing sidecar is inconsistent")
    if rehash_sources:
        source_scene = Path(str(record.get("source_scene", "")))
        scene = verify_tiled_scene(source_scene)
        layer_index = source_scene / str(scene.get("layer_index_file", ""))
        if (
            sha256_file(source_scene / "manifest.json")
            != record.get("source_scene_manifest_sha256")
            or sha256_file(layer_index) != record.get("source_layer_index_sha256")
        ):
            raise ArtifactError("PLY export source scene binding changed")
        for item in sources:
            source = Path(str(item.get("run", ""))) / str(item.get("file", ""))
            if sha256_file(source) != item.get("sha256"):
                raise ArtifactError(f"PLY export source changed: {source}")
    return record


def export_layered_scene_ply(
    source_scene: str | Path, destination: str | Path
) -> dict[str, Any]:
    """Atomically publish a standard-viewer PLY from sealed tile-core PLYs."""
    supplied_source = Path(source_scene).expanduser()
    supplied_destination = Path(destination).expanduser()
    if supplied_source.is_symlink():
        raise ArtifactError(f"source scene cannot be a symlink: {supplied_source}")
    if supplied_destination.is_symlink():
        raise ArtifactError(
            f"PLY export destination cannot be a symlink: {supplied_destination}"
        )
    source_root = supplied_source.resolve()
    destination = supplied_destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to modify existing export: {destination}")
    scene = verify_tiled_scene(source_root)
    layers = _source_layers(source_root, scene)
    first = layers[0]["_ply"]
    if any(layer["_ply"]["schema"] != first["schema"] for layer in layers[1:]):
        raise ArtifactError("source tile PLY schemas do not match")
    total = sum(int(layer["vertex_count"]) for layer in layers)
    output_name = (
        "scene.ply"
        if scene.get("metric_georeferencing_claim_eligible") is True
        else "scene.DIAGNOSTIC_ONLY.ply"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.writing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        output = staging / output_name
        header = list(first["header_lines"])
        header[int(first["vertex_line_index"])] = f"element vertex {total}\n".encode(
            "ascii"
        )
        with output.open("xb") as stream:
            for line in header:
                stream.write(line)
            for layer in layers:
                _copy_body(
                    layer["_path"], int(layer["header_size_bytes"]), stream
                )
            stream.flush()
            os.fsync(stream.fileno())
        output_evidence = _ply_evidence(output)
        if output_evidence["vertex_count"] != total:
            raise ArtifactError("combined PLY vertex inventory is incomplete")
        for layer in layers:
            if sha256_file(layer["_path"]) != layer["sha256"]:
                raise ArtifactError(f"source PLY changed during export: {layer['_path']}")
        source_manifest_hash = sha256_file(source_root / "manifest.json")
        source_layer_hash = sha256_file(
            source_root / str(scene["layer_index_file"])
        )
        output_hash = sha256_file(output)
        public_sources = [
            {key: value for key, value in layer.items() if not key.startswith("_")}
            for layer in layers
        ]
        export_record = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "layered_scene_core_ply_export",
            "artifact_class": (
                "production"
                if scene.get("metric_georeferencing_claim_eligible") is True
                else "diagnostic_render_only"
            ),
            "representation": "hard_half_open_core_union_ply_v1",
            "source_scene": str(source_root),
            "source_scene_manifest_sha256": source_manifest_hash,
            "source_layer_index_sha256": source_layer_hash,
            "source_scene_representation": scene.get("representation"),
            "source_scene_assembly_policy": scene.get("assembly_policy"),
            "metric_georeferencing_claim_eligible": bool(
                scene.get("metric_georeferencing_claim_eligible") is True
            ),
            "source_tile_count": len(layers),
            "source_gaussian_sum": total,
            "retained_gaussians": total,
            "splat_file": output_name,
            "splat_sha256": output_hash,
            "splat_size_bytes": output.stat().st_size,
            "sources": public_sources,
            "limitations": [
                "viewer PLY contains each tile's opacity-pruned unique core only",
                "standard PLY cannot reproduce per-view depth/alpha context blending",
                "authoritative visual metrics belong to the sealed layered source scene",
            ],
        }
        _write_json(staging / "export.json", export_record)
        georeferencing = _json_object(source_root / "provenance.json").get(
            "georeferencing"
        )
        if not isinstance(georeferencing, dict):
            raise ArtifactError("source scene has no georeferencing evidence")
        _write_json(
            staging / "splat.georeferencing.json",
            {
                **georeferencing,
                "splat_file": output_name,
                "splat_sha256": output_hash,
                "representation": "hard_half_open_core_union_ply_v1",
                "source_scene_manifest_sha256": source_manifest_hash,
            },
        )
        _write_text(
            staging / "README.txt",
            "This is a portable Gaussian-splat viewer export.\n"
            "\n"
            "It concatenates the opacity-pruned, uniquely core-owned PLY from "
            "every sealed tile. It does not contain the source scene's per-view "
            "depth/alpha context blending, so seams and distant unsupported "
            "regions may look worse than the authoritative layered renders.\n"
            "\n"
            f"Open {output_name} in a viewer that explicitly supports 3D "
            "Gaussian Splatting PLY files. A generic mesh/point-cloud importer "
            "does not interpret Gaussian scale, rotation, opacity, or spherical "
            "harmonics correctly.\n",
        )
        _write_json(staging / "manifest.json", _export_manifest(staging))
        verify_layered_scene_ply_export(staging)
        publish_directory_noreplace(staging, destination)
        verified = verify_layered_scene_ply_export(destination)
        return {
            "export": str(destination),
            "splat": str(destination / output_name),
            "splat_sha256": verified["splat_sha256"],
            "splat_size_bytes": verified["splat_size_bytes"],
            "retained_gaussians": verified["retained_gaussians"],
            "source_tile_count": verified["source_tile_count"],
            "metric_georeferencing_claim_eligible": verified[
                "metric_georeferencing_claim_eligible"
            ],
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rtk-splat-scene-export",
        description=(
            "Export one sealed layered scene as a portable core-owned Gaussian PLY"
        ),
    )
    parser.add_argument("--source-scene", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_layered_scene_ply(args.source_scene, args.destination)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
