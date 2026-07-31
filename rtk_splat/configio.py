"""Load generic YAML, optionally merging a reusable robot profile."""

from copy import deepcopy
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import yaml


def _to_ns(obj):
    if isinstance(obj, Mapping):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(value) for value in obj]
    return obj


def _read_mapping(path: Path) -> dict:
    try:
        with path.open() as stream:
            raw = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise ValueError(f"configuration file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")
    return dict(raw)


def _deep_merge(base: Mapping, override: Mapping) -> dict:
    """Return a recursive merge without mutating either input."""
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _config_root(config_path: Path, explicit_root: str | Path | None) -> Path:
    if explicit_root is not None:
        return Path(explicit_root).expanduser()
    if config_path.parent.name == "sequences":
        return config_path.parent.parent
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "robots").is_dir():
            return parent
    return config_path.parent


def _robot_path(reference: object, root: Path, owner: Path) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"{owner}: 'robot' must be a non-empty profile name")
    name = reference.strip()
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(
            f"{owner}: robot profile must be a name, not a path: {name!r}"
        )
    filename = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    return root / "robots" / filename


def _load_with_robot(
    path: Path,
    root: Path,
    loading: tuple[Path, ...] = (),
) -> dict:
    resolved = path.resolve()
    if resolved in loading:
        chain = " -> ".join(str(item) for item in (*loading, resolved))
        raise ValueError(f"robot profile cycle: {chain}")
    raw = _read_mapping(path)
    if "robot" not in raw:
        return raw
    profile_path = _robot_path(raw["robot"], root, path)
    if not profile_path.is_file():
        raise ValueError(
            f"{path}: robot profile {raw['robot']!r} not found at "
            f"{profile_path}"
        )
    profile = _load_with_robot(profile_path, root, (*loading, resolved))
    return _deep_merge(profile, raw)


def load_config(
    path: str | Path,
    *,
    config_root: str | Path | None = None,
) -> SimpleNamespace:
    """Load a config, optionally merging ``robots/<robot>.yaml`` first.

    A file in ``configs/sequences`` automatically uses its parent ``configs``
    directory as the profile root. Callers loading files elsewhere can pass
    ``config_root`` explicitly; otherwise a nearby ``robots`` directory is
    discovered. The sequence mapping recursively overrides the robot profile.
    """
    config_path = Path(path).expanduser()
    raw = _load_with_robot(
        config_path,
        _config_root(config_path, config_root),
    )
    if not isinstance(raw.get("paths"), Mapping):
        raise ValueError(f"{config_path}: missing required 'paths' mapping")
    if "workdir" not in raw["paths"]:
        raise ValueError(f"{config_path}: paths.workdir is required")
    cfg = _to_ns(raw)
    cfg.paths.bags = [
        Path(b).expanduser() for b in getattr(cfg.paths, "bags", [])
    ]
    cfg.paths.workdir = Path(cfg.paths.workdir).expanduser()
    if hasattr(cfg.paths, "segment"):
        cfg.paths.segment = Path(cfg.paths.segment).expanduser()
    return cfg
