"""Durable, fail-closed configuration evidence for multi-command runs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rtk_splat.core.runtime_resolution import (
    configuration_evidence,
    runtime_resolution_plain,
)


SCHEMA_VERSION = 1
LEDGER_RELATIVE_PATH = Path("config_artifacts/resolved_config.json")


def _plain(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {key: _plain(item) for key, item in vars(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value.item() if hasattr(value, "item") else value


def _namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authored_config_plain(cfg: Any) -> dict[str, Any]:
    """Capture merged authored values before command-line overrides."""
    return {
        key: _plain(value)
        for key, value in vars(cfg).items()
        if key not in {"runtime_resolution", "config_ledger"}
    }


def _read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read resolved-config ledger {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported resolved-config ledger: {path}")
    return value


def _identity(cfg: Any, authored: Mapping[str, Any]) -> dict[str, Any]:
    runtime = runtime_resolution_plain(cfg)
    sources = runtime.get("source_files", [])
    for item in sources:
        path = Path(item["path"])
        if _file_sha256(path) != item["sha256"]:
            raise ValueError(f"configuration source changed after load: {path}")
    plain_authored = _plain(authored)
    return {
        "authored_config": plain_authored,
        "authored_config_sha256": _canonical_hash(plain_authored),
        "source_files": sources,
    }


def _validate_identity(
    ledger: Mapping[str, Any], identity: Mapping[str, Any], path: Path
) -> None:
    for key in ("authored_config", "authored_config_sha256", "source_files"):
        if ledger.get(key) != identity[key]:
            raise ValueError(
                f"resolved-config ledger identity mismatch ({key}); use a new "
                f"workdir instead of reusing {path.parent.parent}"
            )


def prepare_config_ledger(
    cfg: Any,
    *,
    stage: str,
    authored_config: Mapping[str, Any],
    stage_overrides: Mapping[str, Any] | None = None,
) -> Path:
    """Validate run identity and hydrate old derivations as evidence only."""
    path = Path(cfg.paths.workdir).expanduser() / LEDGER_RELATIVE_PATH
    identity = _identity(cfg, authored_config)
    ledger = _read(path)
    if ledger is not None:
        _validate_identity(ledger, identity, path)
        current = runtime_resolution_plain(cfg).get("derivations", {})
        for name, record in ledger.get("derivations", {}).items():
            if name in current and current[name] != record:
                raise ValueError(f"conflicting hydrated runtime derivation: {name}")
            setattr(cfg.runtime_resolution.derivations, name, _namespace(record))
    cfg.config_ledger = SimpleNamespace(
        path=str(path),
        stage=stage,
        stage_overrides=_namespace(_plain(stage_overrides or {})),
        authored_config_sha256=identity["authored_config_sha256"],
    )
    return path


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def commit_config_ledger(cfg: Any, *, authored_config: Mapping[str, Any]) -> Path:
    """Atomically accumulate successful stage evidence in the run ledger."""
    context = getattr(cfg, "config_ledger", None)
    if context is None:
        raise ValueError("configuration ledger was not prepared")
    path = Path(context.path)
    identity = _identity(cfg, authored_config)
    ledger = _read(path)
    if ledger is None:
        ledger = {
            "schema_version": SCHEMA_VERSION,
            **identity,
            "derivations": {},
            "stages": [],
        }
    else:
        _validate_identity(ledger, identity, path)
    current = runtime_resolution_plain(cfg).get("derivations", {})
    for name, record in current.items():
        previous = ledger["derivations"].get(name)
        if previous is not None and previous != record:
            raise ValueError(f"runtime derivation conflicts with ledger: {name}")
        ledger["derivations"][name] = record
    evidence = configuration_evidence(cfg)
    stage_record = {
        "stage": context.stage,
        "stage_overrides": _plain(context.stage_overrides),
        "effective_config_sha256": evidence["effective_config_sha256"],
        "derivations": current,
    }
    if stage_record not in ledger["stages"]:
        ledger["stages"].append(stage_record)
    _atomic_write(path, ledger)
    return path
