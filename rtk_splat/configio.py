"""Load a YAML configuration and normalize optional filesystem paths."""

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import yaml


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    config_path = Path(path).expanduser()
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{config_path}: top-level YAML value must be a mapping")
    if not isinstance(raw.get("paths"), Mapping):
        raise ValueError(f"{config_path}: missing required 'paths' mapping")
    if "workdir" not in raw["paths"]:
        raise ValueError(f"{config_path}: paths.workdir is required")
    cfg = _to_ns(raw)
    cfg.paths.bags = [
        Path(b).expanduser() for b in getattr(cfg.paths, "bags", [])
    ]
    cfg.paths.workdir = Path(cfg.paths.workdir).expanduser()
    if hasattr(cfg.paths, "ublox_msgs_dir"):
        cfg.paths.ublox_msgs_dir = Path(
            cfg.paths.ublox_msgs_dir
        ).expanduser()
    return cfg
