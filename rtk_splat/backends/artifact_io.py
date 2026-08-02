"""Fail-closed JSON and artifact publication helpers for backends."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import uuid

from rtk_splat.frontends.artifact import ArtifactError, StageLedger, sha256_file


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{path} must contain a JSON object")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            if path.is_file() and path.read_bytes() == payload:
                return
            raise FileExistsError(f"refusing to overwrite {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    _atomic_write(path, payload)


def _tree_manifest(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise ArtifactError(f"model directory does not exist: {root}")
    files = {
        path.relative_to(root).as_posix(): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    if not files:
        raise ArtifactError(f"model directory is empty: {root}")
    return {"schema_version": 1, "files": files}


def _verify_tree(root: Path, manifest: Mapping[str, Any]) -> None:
    if _tree_manifest(root) != manifest:
        raise ArtifactError(f"published model changed: {root}")


def _report_path(root: Path, stage: str) -> Path:
    return root / "reports" / f"{stage}.json"


def _complete_stage(
    root: Path,
    ledger: StageLedger,
    stage: str,
    inputs: Mapping[str, Any],
    report: Mapping[str, Any],
    manifest_name: str,
    manifest: Mapping[str, Any],
    extra_outputs: Sequence[str | Path] = (),
) -> dict[str, Any]:
    report_path = _report_path(root, stage)
    _atomic_json(report_path, report)
    _atomic_json(root / manifest_name, manifest)
    ledger.complete(
        stage, inputs, [report_path, manifest_name, *extra_outputs]
    )
    return dict(report)
