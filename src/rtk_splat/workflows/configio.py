"""Load strict, layered RTK-Splat workflow YAML configuration.

New configurations resolve in the order ``profile -> robot -> sequence``.
Frozen monolithic reproductions remain valid and retain their exact values.
Runtime derivation is deliberately not performed here: an authored ``auto``
value is resolved only by the stage that has the required measurements.
"""

from copy import deepcopy
from collections.abc import Mapping
import hashlib
from pathlib import Path
from types import SimpleNamespace

import yaml

from .config_schema import validate_config_mapping


_RUNTIME_TARGETS = {
    "frame_stride": ("segment", "frame_stride"),
    "frame_spacing_m": ("segment", "frame_spacing_m"),
    "depth_max_z_m": ("depth", "max_z_m"),
    "train_iterations": ("train", "iterations"),
    "train_max_gaussians": ("train", "max_gaussians"),
}


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _leaf_owners(
    value: Mapping,
    owner: Path,
    prefix: tuple[str, ...] = (),
) -> dict[tuple[str, ...], Path]:
    result: dict[tuple[str, ...], Path] = {}
    for key, item in value.items():
        path = (*prefix, str(key))
        if isinstance(item, Mapping):
            result.update(_leaf_owners(item, owner, path))
        else:
            result[path] = owner.resolve()
    return result


def _at_path(value: Mapping, path: tuple[str, ...]) -> object:
    current: object = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(path)
        current = current[key]
    return current


def _layer_kind(path: Path, raw: Mapping) -> str:
    parent = path.parent.name
    if parent == "profiles":
        return "profile"
    if parent == "robots":
        return "robot"
    if parent == "sequences":
        return "sequence"
    if parent == "reproductions" or "benchmarks" in path.parts:
        return "reproduction"
    # Explicit external config roots need not reproduce the repository's
    # directory names. A reference to a robot/profile makes this a sequence.
    if "robot" in raw or "profile" in raw:
        return "sequence"
    return "reproduction"


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


def _named_path(
    reference: object,
    root: Path,
    owner: Path,
    *,
    kind: str,
    directory: str,
) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"{owner}: '{kind}' must be a non-empty profile name")
    name = reference.strip()
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(
            f"{owner}: {kind} profile must be a name, not a path: {name!r}"
        )
    filename = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    return root / directory / filename


def _robot_path(reference: object, root: Path, owner: Path) -> Path:
    return _named_path(
        reference, root, owner, kind="robot", directory="robots"
    )


def _profile_path(reference: object, root: Path, owner: Path) -> Path:
    return _named_path(
        reference, root, owner, kind="quality", directory="profiles"
    )


def _load_with_robot(
    path: Path,
    root: Path,
    loading: tuple[Path, ...] = (),
) -> tuple[dict, list[Path], dict[tuple[str, ...], Path]]:
    resolved = path.resolve()
    if resolved in loading:
        chain = " -> ".join(str(item) for item in (*loading, resolved))
        raise ValueError(f"robot profile cycle: {chain}")
    raw = _read_mapping(path)
    validate_config_mapping(raw, path, layer="robot")
    if "robot" not in raw:
        return raw, [path.resolve()], _leaf_owners(raw, path)
    profile_path = _robot_path(raw["robot"], root, path)
    if not profile_path.is_file():
        raise ValueError(
            f"{path}: robot profile {raw['robot']!r} not found at "
            f"{profile_path}"
        )
    parent, files, owners = _load_with_robot(
        profile_path, root, (*loading, resolved)
    )
    return (
        _deep_merge(parent, raw),
        [*files, path.resolve()],
        {**owners, **_leaf_owners(raw, path)},
    )


def _load_with_profile(
    path: Path,
    root: Path,
    loading: tuple[Path, ...] = (),
) -> tuple[dict, list[Path], dict[tuple[str, ...], Path]]:
    """Load a quality profile, allowing explicit profile inheritance."""
    resolved = path.resolve()
    if resolved in loading:
        chain = " -> ".join(str(item) for item in (*loading, resolved))
        raise ValueError(f"quality profile cycle: {chain}")
    raw = _read_mapping(path)
    validate_config_mapping(raw, path, layer="profile")
    if "profile" not in raw:
        return raw, [path.resolve()], _leaf_owners(raw, path)
    parent_path = _profile_path(raw["profile"], root, path)
    if not parent_path.is_file():
        raise ValueError(
            f"{path}: quality profile {raw['profile']!r} not found at "
            f"{parent_path}"
        )
    parent, files, owners = _load_with_profile(
        parent_path, root, (*loading, resolved)
    )
    return (
        _deep_merge(parent, raw),
        [*files, path.resolve()],
        {**owners, **_leaf_owners(raw, path)},
    )


def _load_layered(path: Path, root: Path) -> tuple[dict, dict, dict]:
    """Return resolved data plus auditable source-layer paths."""
    raw = _read_mapping(path)
    layer = _layer_kind(path, raw)
    validate_config_mapping(raw, path, layer=layer)
    merged: dict = {}
    owners: dict[tuple[str, ...], Path] = {}
    source_files: list[tuple[str, Path]] = []
    sources: dict[str, str | None] = {
        "profile": None,
        "robot": None,
        "sequence": str(path.resolve()),
    }
    if "profile" in raw:
        profile_path = _profile_path(raw["profile"], root, path)
        if not profile_path.is_file():
            raise ValueError(
                f"{path}: quality profile {raw['profile']!r} not found at "
                f"{profile_path}"
            )
        merged, files, profile_owners = _load_with_profile(profile_path, root)
        owners.update(profile_owners)
        source_files.extend(("profile", item) for item in files)
        sources["profile"] = str(profile_path.resolve())
    if "robot" in raw:
        robot_path = _robot_path(raw["robot"], root, path)
        if not robot_path.is_file():
            raise ValueError(
                f"{path}: robot profile {raw['robot']!r} not found at "
                f"{robot_path}"
            )
        robot, files, robot_owners = _load_with_robot(robot_path, root)
        merged = _deep_merge(merged, robot)
        owners.update(robot_owners)
        source_files.extend(("robot", item) for item in files)
        sources["robot"] = str(robot_path.resolve())
    merged = _deep_merge(merged, raw)
    owners.update(_leaf_owners(raw, path))
    source_files.append((layer, path.resolve()))
    validate_config_mapping(merged, path, layer="reproduction")
    unique_files = []
    seen: set[Path] = set()
    for role, source in source_files:
        if source not in seen:
            seen.add(source)
            unique_files.append(
                {"role": role, "path": str(source), "sha256": _sha256(source)}
            )
    origins = {}
    for name, target in _RUNTIME_TARGETS.items():
        try:
            value = _at_path(merged, target)
        except KeyError:
            continue
        source = owners[target]
        source_record = next(
            item for item in unique_files if item["path"] == str(source)
        )
        origins[name] = {
            "config_path": ".".join(target),
            "authored_value": deepcopy(value),
            "layer": source_record["role"],
            "source_path": source_record["path"],
            "source_sha256": source_record["sha256"],
        }
    identity = {
        "layer": layer,
        "source_files": unique_files,
        "origins": origins,
    }
    return merged, sources, identity


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
    raw, sources, identity = _load_layered(
        config_path, _config_root(config_path, config_root)
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
    # Runtime-only output. It cannot be authored in YAML because the strict
    # schema rejects this key, so a stale result can never masquerade as a
    # newly measured derivation.
    cfg.runtime_resolution = _to_ns(
        {
            "schema_version": 1,
            "config_sources": sources,
            "source_files": identity["source_files"],
            "origins": identity["origins"],
            "cli_overrides": {},
            "derivations": {},
        }
    )
    return cfg
