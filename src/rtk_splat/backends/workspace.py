"""Verified frontend and mapper-workspace context loading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rtk_splat.backends.artifact_io import _atomic_write, _json
from rtk_splat.backends.mapper_config import MapperConfig, _LEGACY_EVALUATION_DEFAULTS
from rtk_splat.frontends.artifact import (
    ArtifactError,
    FRONTEND_SEAL_FILE,
    sha256_file,
    sqlite_logical_record,
    verify_frontend_seal,
)


_REQUIRED_FRONTEND_FILES = (
    "frame_manifest.json",
    "rig_config.json",
    "keyframes.json",
    "pairs.txt",
    "provenance.json",
    "quality.json",
    FRONTEND_SEAL_FILE,
    "database.db",
)

def _frontend_context(
    artifact: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Validate the mapper-facing portion of a frontend without its segment."""
    root = Path(artifact).expanduser().resolve()
    missing = [name for name in _REQUIRED_FRONTEND_FILES if not (root / name).is_file()]
    if not root.is_dir() or missing:
        raise ArtifactError(f"incomplete frontend artifact {root}: {missing}")
    verify_frontend_seal(root)
    images = root / "images"
    if not images.is_dir():
        raise ArtifactError("frontend artifact has no images directory")
    entries = list(images.iterdir())
    if not entries or any(not entry.is_symlink() for entry in entries):
        raise ArtifactError("frontend images must be non-empty and symlink-only")

    manifest = _json(root / "frame_manifest.json")
    rows = manifest.get("frames")
    if not isinstance(rows, list) or not rows:
        raise ArtifactError("frame_manifest.frames must be a non-empty list")
    frame_ids: list[int] = []
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or type(row.get("frame_id")) is not int:
            raise ArtifactError("invalid frontend frame record")
        frame_ids.append(row["frame_id"])
        for field in ("left_image", "right_image"):
            image = row.get(field)
            if not isinstance(image, dict) or not isinstance(image.get("name"), str):
                raise ArtifactError(f"frontend frame lacks {field}")
            name = image["name"]
            if name in names:
                raise ArtifactError(f"duplicate frontend image name: {name}")
            names.add(name)
    if frame_ids != sorted(set(frame_ids)):
        raise ArtifactError("frontend frame IDs must be sorted and unique")
    if names != {entry.name for entry in entries}:
        raise ArtifactError("frontend image directory differs from its manifest")

    keyframes = _json(root / "keyframes.json")
    selected = keyframes.get("frame_ids")
    if (
        not isinstance(selected, list)
        or any(type(value) is not int for value in selected)
        or selected != sorted(set(selected))
        or not selected
        or not set(selected) <= set(frame_ids)
    ):
        raise ArtifactError("invalid keyframes.frame_ids")
    return root, manifest, keyframes


def _immutable_input_roots(frontend: Path) -> tuple[Path, ...]:
    roots = [frontend.resolve()]
    provenance = _json(frontend / "provenance.json")
    contract = provenance.get("contract_inputs")
    if isinstance(contract, dict) and contract.get("segment_root"):
        roots.append(Path(str(contract["segment_root"])).resolve())
    return tuple(dict.fromkeys(roots))


def _image_sets(
    manifest: Mapping[str, Any], keyframes: Mapping[str, Any]
) -> tuple[list[str], list[str], list[str]]:
    selected = set(keyframes["frame_ids"])
    all_names: list[str] = []
    solve_names: list[str] = []
    registration_names: list[str] = []
    for row in manifest["frames"]:
        names = [row["left_image"]["name"], row["right_image"]["name"]]
        all_names.extend(names)
        (solve_names if row["frame_id"] in selected else registration_names).extend(
            names
        )
    return all_names, solve_names, registration_names


def _frontend_hashes(root: Path) -> dict[str, str]:
    return {
        name: sha256_file(root / name)
        for name in _REQUIRED_FRONTEND_FILES
        if name != "database.db"
    }


def _marker_hash(root: Path, name: str) -> str | None:
    path = root / "stages" / f"{name}.json"
    if not path.is_file():
        return None
    marker = _json(path)
    if marker.get("state") != "complete":
        raise ArtifactError(f"frontend stage {name!r} is not complete")
    outputs = marker.get("outputs")
    if not isinstance(outputs, dict):
        raise ArtifactError(f"frontend stage {name!r} has no output evidence")
    for relative, expected in outputs.items():
        output = root / str(relative)
        if not output.is_file() or sha256_file(output) != expected:
            raise ArtifactError(f"frontend stage output changed: {relative}")
    return sha256_file(path)


def _write_lines(path: Path, values: Sequence[str]) -> None:
    if not values or len(values) != len(set(values)):
        raise ArtifactError(f"{path.name} must be non-empty and unique")
    _atomic_write(path, "".join(f"{value}\n" for value in values).encode())


def _verify_snapshot(workspace: Path, plan: Mapping[str, Any]) -> None:
    snapshot = _json(workspace / "database_snapshot.json")
    database = workspace / "database.db"
    try:
        committed = sqlite_logical_record(database)
    except ArtifactError as exc:
        raise ArtifactError("backend database snapshot changed or is invalid") from exc
    if (
        committed["sha256"] != snapshot.get("snapshot_sha256")
        or committed["sha256"] != plan.get("database_sha256")
        or committed["schema_sha256"] != snapshot.get("schema_sha256")
        or committed["integrity_check"] != "ok"
    ):
        raise ArtifactError("backend database snapshot changed")


def _workspace_context(
    workspace: str | Path,
) -> tuple[Path, dict[str, Any], MapperConfig]:
    root = Path(workspace).expanduser().resolve()
    plan = _json(root / "backend_plan.json")
    try:
        raw_config = plan["config"]
        if not isinstance(raw_config, dict):
            raise TypeError("backend config must be an object")
        effective_config = dict(raw_config)
        for field, default in _LEGACY_EVALUATION_DEFAULTS.items():
            effective_config.setdefault(field, default)
        config = MapperConfig(**effective_config)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError("invalid backend plan configuration") from exc
    if effective_config != asdict(config):
        raise ArtifactError(
            "backend plan predates the current mapper configuration schema; "
            "prepare a new backend artifact instead of resuming it"
        )
    frontend = Path(str(plan.get("frontend_artifact", ""))).resolve()
    seal = verify_frontend_seal(frontend)
    current = _frontend_hashes(frontend)
    if current != plan.get("frontend_inputs"):
        raise ArtifactError("sealed frontend inputs changed after backend preparation")
    if (
        sha256_file(frontend / FRONTEND_SEAL_FILE)
        != plan.get("frontend_seal_sha256")
        or seal["database"]["committed_view"]["sha256"]
        != plan.get("frontend_database_committed_sha256")
        or _marker_hash(frontend, "matching")
        != plan.get("matching_stage_sha256")
    ):
        raise ArtifactError("sealed frontend terminal evidence changed")
    _verify_snapshot(root, plan)
    for filename, expected_key in (
        ("solve_images.txt", "solve_image_names"),
        ("registration_images.txt", "registration_image_names"),
    ):
        values = (root / filename).read_text(encoding="utf-8").splitlines()
        if values != plan[expected_key]:
            raise ArtifactError(f"{filename} changed after backend preparation")
    return root, plan, config
