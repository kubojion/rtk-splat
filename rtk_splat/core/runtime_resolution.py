"""ROS/backend-free contract for auditable runtime-derived controls."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np


POLICY_NAME = "quality_v1_candidate"
FORMULA_VERSION = 1


def _plain(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {key: _plain(item) for key, item in vars(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolution(cfg: Any) -> SimpleNamespace:
    value = getattr(cfg, "runtime_resolution", None)
    if value is None:
        value = SimpleNamespace(schema_version=1, config_sources={}, source_files=[],
                                origins=SimpleNamespace(), cli_overrides=SimpleNamespace(),
                                derivations=SimpleNamespace())
        cfg.runtime_resolution = value
    for name in ("origins", "cli_overrides", "derivations"):
        if not hasattr(value, name):
            setattr(value, name, SimpleNamespace())
    return value


def runtime_resolution_plain(cfg: Any) -> dict[str, Any]:
    value = getattr(cfg, "runtime_resolution", None)
    return {} if value is None else _plain(value)


def runtime_control_origin(
    cfg: Any, name: str, current: Any, *, config_path: str
) -> dict[str, Any]:
    """Return the immutable authored/CLI origin, never a prior derived value."""
    resolution = _resolution(cfg)
    origin = getattr(resolution.origins, name, None)
    if origin is None:
        origin = SimpleNamespace(config_path=config_path, authored_value=_plain(current),
                                 layer="constructed", source_path=None,
                                 source_sha256=None)
        setattr(resolution.origins, name, origin)
    result = {"kind": "authored", **_plain(origin),
              "effective_value": _plain(origin.authored_value)}
    cli = getattr(resolution.cli_overrides, name, None)
    if cli is not None:
        result.update({"kind": "cli_override", "effective_value": _plain(cli.value),
                       "cli_option": cli.option})
    return result


def runtime_control_value(
    cfg: Any, name: str, current: Any, *, config_path: str
) -> Any:
    return runtime_control_origin(
        cfg, name, current, config_path=config_path
    )["effective_value"]


def set_runtime_cli_override(
    cfg: Any, name: str, current: Any, value: Any, *, option: str,
    config_path: str,
) -> None:
    resolution = _resolution(cfg)
    runtime_control_origin(cfg, name, current, config_path=config_path)
    record = {"option": option, "value": _plain(value)}
    previous = getattr(resolution.cli_overrides, name, None)
    if previous is not None and _plain(previous) != record:
        raise ValueError(f"conflicting CLI override for runtime control {name}")
    setattr(resolution.cli_overrides, name, SimpleNamespace(**record))


def make_runtime_resolution_record(
    cfg: Any, name: str, *, source: str, formula: str,
    inputs: Mapping[str, Any], policy: Mapping[str, Any], chosen: int | float,
) -> tuple[str, dict[str, Any]]:
    if source not in {"derived", "override"}:
        raise ValueError(f"invalid runtime-resolution source: {source}")
    resolution = _resolution(cfg)
    origin = getattr(resolution.origins, name, None)
    if origin is None:
        raise ValueError(f"runtime control {name} has no captured origin")
    origin_record = _plain(origin)
    cli = getattr(resolution.cli_overrides, name, None)
    if cli is not None:
        origin_record["cli_override"] = _plain(cli)
    return name, {
        "policy_name": POLICY_NAME, "formula_version": FORMULA_VERSION,
        "formula": formula, "source": source, "origin": origin_record,
        "measured_inputs": _plain(inputs), "policy_bounds": _plain(policy),
        "chosen_value": _plain(chosen),
    }


def commit_runtime_resolution_records(
    cfg: Any, records: list[tuple[str, dict[str, Any]]]
) -> None:
    derivations = _resolution(cfg).derivations
    names = [name for name, _ in records]
    if len(names) != len(set(names)):
        raise ValueError("duplicate runtime-resolution record in atomic commit")
    for name, record in records:
        previous = getattr(derivations, name, None)
        if previous is not None and _plain(previous) != record:
            raise ValueError(f"runtime derivation changed within one run: {name}")
    for name, record in records:
        setattr(derivations, name, SimpleNamespace(**record))


def record_runtime_resolution(cfg: Any, name: str, **values) -> dict[str, Any]:
    item = make_runtime_resolution_record(cfg, name, **values)
    commit_runtime_resolution_records(cfg, [item])
    return item[1]


def record_override(cfg: Any, name: str, value: Any) -> dict[str, Any]:
    return record_runtime_resolution(
        cfg, name, source="override",
        formula="authored or CLI numeric value; no runtime formula applied",
        inputs={}, policy={}, chosen=value,
    )


def configuration_evidence(cfg: Any) -> dict[str, Any]:
    """Snapshot complete operational config plus immutable source evidence."""
    effective = {
        key: _plain(value) for key, value in vars(cfg).items()
        if key not in {"runtime_resolution", "config_ledger"}
    }
    runtime = runtime_resolution_plain(cfg)
    ledger = _plain(getattr(cfg, "config_ledger", {}))
    payload = {
        "effective_config": effective,
        "runtime_resolution": runtime,
        "stage_overrides": ledger.get("stage_overrides", {}),
    }
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return {
        "schema_version": 1,
        "ledger_path": ledger.get("path"),
        "authored_config_sha256": ledger.get("authored_config_sha256"),
        "source_files": runtime.get("source_files", []),
        **payload,
        "effective_config_sha256": digest,
    }
