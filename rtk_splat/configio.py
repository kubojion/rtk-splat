"""Load config.yaml into a nested namespace with ~ expansion on paths."""

from pathlib import Path
from types import SimpleNamespace

import yaml


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    with open(Path(path).expanduser()) as f:
        raw = yaml.safe_load(f)
    cfg = _to_ns(raw)
    cfg.paths.bags = [Path(b).expanduser() for b in cfg.paths.bags]
    for name in ("ublox_msgs_dir", "workdir"):
        setattr(cfg.paths, name, Path(getattr(cfg.paths, name)).expanduser())
    return cfg
