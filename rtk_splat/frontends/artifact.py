"""Sealed, mapper-neutral frontend artifact primitives.

This module owns filesystem integrity and provenance only.  Feature
extraction, pair selection, matching, and mapper execution live elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rtk_splat.core.segment import SegmentReader, publish_directory_noreplace


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JSON_FILES = (
    "frame_manifest.json",
    "rig_config.json",
    "keyframes.json",
    "provenance.json",
    "quality.json",
)
FRONTEND_STATIC_FILES = (*_JSON_FILES, "pairs.txt")
FRONTEND_SEAL_FILE = "frontend_seal.json"


class ArtifactError(ValueError):
    """A frontend artifact operation is unsafe or internally inconsistent."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sqlite_schema_record(connection: sqlite3.Connection) -> dict[str, Any]:
    try:
        integrity = [
            str(row[0])
            for row in connection.execute("PRAGMA integrity_check")
        ]
        if integrity != ["ok"]:
            raise ArtifactError(
                "SQLite integrity_check failed: " + "; ".join(integrity)
            )
        schema = [
            {
                "type": str(kind),
                "name": str(name),
                "table": str(table),
                "sql": None if sql is None else str(sql),
            }
            for kind, name, table, sql in connection.execute(
                """
                SELECT type, name, tbl_name, sql
                FROM sqlite_schema
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name, tbl_name
                """
            )
        ]
        return {
            "integrity_check": "ok",
            "schema_sha256": canonical_hash(schema),
            "tables": sorted(
                row["name"] for row in schema if row["type"] == "table"
            ),
            "user_version": int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            ),
            "page_size": int(
                connection.execute("PRAGMA page_size").fetchone()[0]
            ),
        }
    except sqlite3.Error as exc:
        raise ArtifactError(f"invalid SQLite database: {exc}") from exc


def _sqlite_online_backup(source: Path, destination: Path) -> dict[str, Any]:
    """Copy one committed SQLite view, including rows present only in WAL."""
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite SQLite snapshot: {destination}")
    try:
        with sqlite3.connect(
            f"{source.as_uri()}?mode=ro", uri=True
        ) as source_connection:
            source_connection.execute("PRAGMA query_only=ON")
            with sqlite3.connect(destination) as destination_connection:
                source_connection.backup(destination_connection)
                destination_connection.commit()
                record = _sqlite_schema_record(destination_connection)
    except (OSError, sqlite3.Error) as exc:
        destination.unlink(missing_ok=True)
        raise ArtifactError(
            f"cannot create a consistent SQLite snapshot of {source}: {exc}"
        ) from exc
    with destination.open("rb") as stream:
        os.fsync(stream.fileno())
    return {
        **record,
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def sqlite_logical_record(database: str | Path) -> dict[str, Any]:
    """Hash a transaction-consistent committed view without altering the source."""
    source = Path(database).expanduser().resolve()
    if not source.is_file():
        raise ArtifactError(f"SQLite database does not exist: {source}")
    with tempfile.TemporaryDirectory(prefix="rtk-splat-sqlite-seal-") as temporary:
        snapshot = Path(temporary) / "database.db"
        return _sqlite_online_backup(source, snapshot)


def _normal(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normal(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normal(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _normal(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON-serializable: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _normal(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _normal(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise FileExistsError(f"refusing to overwrite {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    _atomic_write(path, _json_bytes(value), replace=replace)


def _run_git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"").decode(errors="replace").strip()
        raise ArtifactError(f"git {' '.join(args)} failed: {detail}") from exc


def collect_git_state(repo_root: str | Path) -> dict[str, Any]:
    """Return committed identity plus a hash covering all local changes."""
    repo = Path(repo_root).resolve()
    commit = _run_git(repo, "rev-parse", "HEAD").decode().strip()
    tree = _run_git(repo, "rev-parse", "HEAD^{tree}").decode().strip()
    tracked_diff = _run_git(
        repo, "diff", "--binary", "--no-ext-diff", "HEAD", "--"
    )
    untracked_raw = _run_git(
        repo, "ls-files", "--others", "--exclude-standard", "-z"
    )
    untracked = sorted(
        item.decode("utf-8", errors="surrogateescape")
        for item in untracked_raw.split(b"\0")
        if item
    )
    state = bytearray(b"tracked-diff\0")
    state.extend(tracked_diff)
    untracked_records = []
    for relative in untracked:
        path = repo / relative
        if not path.is_file():
            continue
        digest = sha256_file(path)
        untracked_records.append({"path": relative, "sha256": digest})
        state.extend(b"\0untracked\0")
        state.extend(relative.encode("utf-8", errors="surrogateescape"))
        state.extend(b"\0")
        state.extend(digest.encode("ascii"))
    return {
        "commit": commit,
        "tree": tree,
        "dirty": bool(tracked_diff or untracked_records),
        "dirty_diff_sha256": hashlib.sha256(state).hexdigest(),
        "untracked": untracked_records,
    }


def probe_colmap_identity(
    executable: str | Path,
    *,
    version_output: str | None = None,
) -> dict[str, Any]:
    """Resolve COLMAP and capture its version identity without running a job."""
    requested = str(executable)
    found = shutil.which(requested)
    path = Path(found if found is not None else requested).expanduser().resolve()
    if not path.is_file():
        raise ArtifactError(f"COLMAP executable does not exist: {path}")
    output = version_output
    if output is None:
        try:
            process = subprocess.run(
                [str(path), "-h"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ArtifactError(f"cannot query COLMAP identity: {path}") from exc
        output = process.stdout
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    version = next((line for line in lines if "COLMAP" in line.upper()), None)
    if version is None:
        raise ArtifactError("COLMAP version output has no identifiable version")
    return {
        "executable": str(path),
        "executable_sha256": sha256_file(path),
        "version": version,
    }


def _contract_checksums(segment: Path) -> dict[str, dict[str, Any]]:
    required = (
        "frames.npz",
        "calibration.json",
        "segment_meta.json",
        "manifest.json",
        "observations/gnss.npz",
    )
    optional = ("observations/heading.npz", "observations/imu.npz")
    result: dict[str, dict[str, Any]] = {}
    for relative in (*required, *optional):
        path = segment / relative
        if not path.exists():
            if relative in required:
                raise ArtifactError(f"contract input is missing: {path}")
            continue
        result[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def _supplied_image_hash(
    frames: dict[str, np.ndarray], side: str, index: int
) -> tuple[str, str] | None:
    candidates = (
        f"{side}_image_sha256",
        f"stereo_{side}_sha256",
        f"{side}_sha256",
    )
    for field in candidates:
        if field not in frames:
            continue
        values = frames[field]
        if values.shape != (len(frames["frame_id"]),):
            raise ArtifactError(f"frames.{field} has an invalid shape")
        digest = str(values[index]).lower()
        if not _SHA256.fullmatch(digest):
            raise ArtifactError(f"frames.{field}[{index}] is not SHA-256")
        return digest, f"frames.npz:{field}"
    return None


def image_inventory(
    reader: SegmentReader,
) -> list[dict[str, Any]]:
    """Create an ordered content inventory, reusing adapter hashes when present."""
    frames = reader.frames
    inventory: list[dict[str, Any]] = []
    sides = [("left", "left_image_path")]
    if reader.meta["capabilities"]["stereo"]:
        sides.append(("right", "right_image_path"))
    for index, frame_id in enumerate(frames["frame_id"].astype(int)):
        for side, field in sides:
            relative = str(frames[field][index])
            source = reader.root / relative
            supplied = _supplied_image_hash(frames, side, index)
            if supplied is None:
                digest = sha256_file(source)
                hash_source = "computed"
            else:
                digest, hash_source = supplied
            inventory.append(
                {
                    "frame_id": frame_id,
                    "camera": side,
                    "source_relative_path": relative,
                    "content_sha256": digest,
                    "hash_source": hash_source,
                    "size_bytes": source.stat().st_size,
                }
            )
    return inventory


def collect_provenance(
    segment: str | Path,
    *,
    resolved_config: Mapping[str, Any],
    colmap: Mapping[str, Any] | str | Path,
    seed: int,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Collect the immutable evidence needed to reproduce one frontend."""
    reader = SegmentReader(segment).validate()
    inventory = image_inventory(reader)
    colmap_record = (
        _normal(colmap)
        if isinstance(colmap, Mapping)
        else probe_colmap_identity(colmap)
    )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ArtifactError("seed must be an integer")
    if not isinstance(colmap_record, dict) or not {
        "executable",
        "version",
    } <= set(colmap_record):
        raise ArtifactError("COLMAP identity needs executable and version")
    return {
        "schema_version": 1,
        "git": collect_git_state(repo_root),
        "resolved_config_sha256": canonical_hash(resolved_config),
        "colmap": colmap_record,
        "seed": seed,
        "contract_inputs": {
            "segment_root": str(reader.root.resolve()),
            "files": _contract_checksums(reader.root),
            "image_inventory_count": len(inventory),
            "image_inventory_sha256": canonical_hash(inventory),
        },
    }


def _safe_name(value: str, label: str) -> str:
    if not _SAFE_NAME.fullmatch(value):
        raise ArtifactError(f"invalid {label}: {value!r}")
    return value


def _validate_provenance(
    reader: SegmentReader,
    provenance: Mapping[str, Any],
    inventory: list[dict[str, Any]],
) -> None:
    expected = provenance.get("contract_inputs")
    if not isinstance(expected, Mapping):
        raise ArtifactError("provenance has no contract_inputs")
    current = {
        "segment_root": str(reader.root.resolve()),
        "files": _contract_checksums(reader.root),
        "image_inventory_count": len(inventory),
        "image_inventory_sha256": canonical_hash(inventory),
    }
    if _normal(expected) != current:
        raise ArtifactError("segment inputs changed after provenance collection")


def _normalize_keyframes(
    value: Mapping[str, Any], frame_ids: set[int]
) -> dict[str, Any]:
    result = dict(_normal(value))
    selected = result.get("frame_ids")
    if not isinstance(selected, list) or any(
        type(frame_id) is not int for frame_id in selected
    ):
        raise ArtifactError("keyframes.frame_ids must be a list of integers")
    if selected != sorted(set(selected)):
        raise ArtifactError("keyframe IDs must be sorted and unique")
    if not set(selected) <= frame_ids:
        raise ArtifactError("keyframes contain IDs outside the segment")
    result.setdefault("schema_version", 1)
    return result


def _normalize_pairs(
    pairs: Iterable[tuple[str, str]], image_names: set[str]
) -> tuple[list[tuple[str, str]], bytes]:
    normalized: set[tuple[str, str]] = set()
    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ArtifactError("each pair must contain exactly two image names")
        left, right = str(pair[0]), str(pair[1])
        if left == right or left not in image_names or right not in image_names:
            raise ArtifactError(f"invalid image pair: {(left, right)}")
        if any(character.isspace() for character in left + right):
            raise ArtifactError("pair image names cannot contain whitespace")
        normalized.add(tuple(sorted((left, right))))
    ordered = sorted(normalized)
    payload = "".join(f"{left} {right}\n" for left, right in ordered).encode()
    return ordered, payload


class FrontendArtifactBuilder:
    """Atomically publish a new sealed frontend foundation."""

    def __init__(
        self,
        segment: str | Path,
        output_root: str | Path,
        name: str,
    ):
        self.segment = Path(segment).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.name = _safe_name(name, "frontend artifact name")
        self.destination = self.output_root / "frontend_artifacts" / self.name
        if self.segment == self.destination or self.segment in self.destination.parents:
            raise ArtifactError("frontend output must not be inside the source segment")

    def build(
        self,
        *,
        rig_config: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        keyframes: Mapping[str, Any],
        pairs: Iterable[tuple[str, str]],
        provenance: Mapping[str, Any],
        quality: Mapping[str, Any],
    ) -> Path:
        if self.destination.exists():
            raise FileExistsError(
                f"refusing to overwrite frontend artifact: {self.destination}"
            )
        reader = SegmentReader(self.segment).validate()
        inventory = image_inventory(reader)
        _validate_provenance(reader, provenance, inventory)
        frames = reader.frames
        frame_ids = frames["frame_id"].astype(int)
        keyframe_record = _normalize_keyframes(keyframes, set(frame_ids.tolist()))
        split_by_frame = {
            int(frame_id): split
            for split in ("train", "val", "test")
            for frame_id in reader.manifest[split]
        }

        parent = self.destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = parent / f".{self.name}.writing-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            images = staging / "images"
            images.mkdir()
            manifest_rows: dict[int, dict[str, Any]] = {
                int(frame_id): {
                    "frame_id": int(frame_id),
                    "timestamp_ns": int(frames["timestamp_ns"][index]),
                    "split": split_by_frame[int(frame_id)],
                    **(
                        {
                            "right_timestamp_ns": int(
                                frames["right_timestamp_ns"][index]
                            ),
                            "stereo_sync_residual_ns": int(
                                frames["stereo_sync_residual_ns"][index]
                            ),
                        }
                        if reader.meta["capabilities"]["stereo"]
                        else {}
                    ),
                }
                for index, frame_id in enumerate(frame_ids)
            }
            image_names: set[str] = set()
            for item in inventory:
                source = reader.root / item["source_relative_path"]
                suffix = source.suffix.lower()
                name = f"{item['camera']}_{item['frame_id']:06d}{suffix}"
                if name in image_names:
                    raise ArtifactError(f"duplicate frontend image name: {name}")
                image_names.add(name)
                os.symlink(source.resolve(), images / name)
                manifest_rows[item["frame_id"]][f"{item['camera']}_image"] = {
                    "name": name,
                    "source_relative_path": item["source_relative_path"],
                    "content_sha256": item["content_sha256"],
                    "hash_source": item["hash_source"],
                }
            _, pairs_payload = _normalize_pairs(pairs, image_names)
            frame_manifest = {
                "schema_version": 1,
                "contract_version": 2,
                "frames": [manifest_rows[int(frame_id)] for frame_id in frame_ids],
                "image_inventory_sha256": canonical_hash(inventory),
            }
            rig_record = _normal(rig_config)
            if not isinstance(rig_record, (dict, list)) or not rig_record:
                raise ArtifactError(
                    "rig_config must be a non-empty COLMAP object or rig list"
                )
            if isinstance(rig_record, dict):
                rig_record.setdefault("schema_version", 1)
            quality_record = dict(_normal(quality))
            quality_record.setdefault("schema_version", 1)
            quality_record.setdefault("n_frames", len(frame_ids))
            records = {
                "frame_manifest.json": frame_manifest,
                "rig_config.json": rig_record,
                "keyframes.json": keyframe_record,
                "provenance.json": dict(_normal(provenance)),
                "quality.json": quality_record,
            }
            for filename in _JSON_FILES:
                _atomic_json(staging / filename, records[filename])
            _atomic_write(staging / "pairs.txt", pairs_payload)
            (staging / "stages").mkdir()
            if self.destination.exists():
                raise FileExistsError(
                    f"refusing to overwrite frontend artifact: {self.destination}"
                )
            publish_directory_noreplace(staging, self.destination)
            return self.destination
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid frontend JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{path} must contain a JSON object")
    return value


def frontend_static_record(artifact: str | Path) -> dict[str, dict[str, Any]]:
    """Hash the foundation and all finalized frontend JSON audit evidence."""
    root = Path(artifact).expanduser().resolve()
    required = {root / name for name in FRONTEND_STATIC_FILES}
    candidates = required | {
        path
        for path in root.rglob("*.json")
        if path != root / FRONTEND_SEAL_FILE
        and path != root / "stages" / "matching.json"
    }
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(candidates):
        relative = path.relative_to(root).as_posix()
        if not path.is_file() or path.is_symlink():
            raise ArtifactError(
                f"frontend static input is missing or unsafe: {relative}"
            )
        result[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def frontend_image_record(artifact: str | Path) -> dict[str, Any]:
    """Hash the bytes reached by every image link and compare its manifest."""
    root = Path(artifact).expanduser().resolve()
    manifest = _load_json_object(root / "frame_manifest.json")
    rows = manifest.get("frames")
    if not isinstance(rows, list) or not rows:
        raise ArtifactError("frame_manifest.frames must be a non-empty list")
    expected: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ArtifactError("frame_manifest contains a non-object frame")
        for field in ("left_image", "right_image"):
            image = row.get(field)
            if image is None:
                continue
            if not isinstance(image, dict):
                raise ArtifactError(f"frame_manifest.{field} must be an object")
            name = image.get("name")
            digest = str(image.get("content_sha256", "")).lower()
            if (
                not isinstance(name, str)
                or not name
                or "/" in name
                or name in expected
                or not _SHA256.fullmatch(digest)
            ):
                raise ArtifactError(f"invalid or duplicate manifest image: {name!r}")
            expected[name] = digest
    images = root / "images"
    if not images.is_dir():
        raise ArtifactError("frontend images directory is missing")
    entries = {entry.name: entry for entry in images.iterdir()}
    if set(entries) != set(expected):
        raise ArtifactError("frontend image links disagree with frame_manifest")
    records: list[dict[str, Any]] = []
    for name in sorted(expected):
        link = entries[name]
        if not link.is_symlink():
            raise ArtifactError(f"frontend image is not a symlink: {name}")
        try:
            resolved = link.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ArtifactError(f"frontend image link is broken: {name}") from exc
        if not resolved.is_file():
            raise ArtifactError(f"frontend image target is not a file: {name}")
        digest = sha256_file(resolved)
        if digest != expected[name]:
            raise ArtifactError(
                f"frontend image content changed from frame_manifest: {name}"
            )
        records.append(
            {
                "name": name,
                "link_target": os.readlink(link),
                "resolved_target": str(resolved),
                "content_sha256": digest,
                "size_bytes": resolved.stat().st_size,
            }
        )
    return {
        "count": len(records),
        "entries_sha256": canonical_hash(records),
        "entries": records,
    }


def _current_frontend_seal_payload(root: Path) -> dict[str, Any]:
    database = root / "database.db"
    if not database.is_file() or database.is_symlink():
        raise ArtifactError("frontend database.db is missing or unsafe")
    return {
        "schema_version": 1,
        "kind": "rtk_splat_mapper_neutral_frontend",
        "static_files": frontend_static_record(root),
        "images": frontend_image_record(root),
        "database": {
            "raw_sha256": sha256_file(database),
            "raw_size_bytes": database.stat().st_size,
            "committed_view": sqlite_logical_record(database),
        },
    }


def create_frontend_seal(artifact: str | Path) -> dict[str, Any]:
    """Create the terminal content seal immediately before matching acceptance."""
    root = Path(artifact).expanduser().resolve()
    payload = _current_frontend_seal_payload(root)
    path = root / FRONTEND_SEAL_FILE
    if path.exists():
        if _load_json_object(path) != payload:
            raise ArtifactError("existing frontend seal disagrees with current inputs")
        return payload
    _atomic_json(path, payload)
    return payload


def verify_frontend_seal(
    artifact: str | Path,
    *,
    require_terminal_marker: bool = True,
) -> dict[str, Any]:
    """Rehash all sealed inputs, including image targets and SQLite WAL state."""
    root = Path(artifact).expanduser().resolve()
    path = root / FRONTEND_SEAL_FILE
    if not path.is_file() or path.is_symlink():
        raise ArtifactError("frontend terminal seal is missing or unsafe")
    sealed = _load_json_object(path)
    current = _current_frontend_seal_payload(root)
    if sealed != current:
        raise ArtifactError("frontend terminal seal verification failed")
    if require_terminal_marker:
        marker_path = root / "stages" / "matching.json"
        marker = _load_json_object(marker_path)
        if marker.get("state") != "complete":
            raise ArtifactError("frontend matching terminal marker is incomplete")
        outputs = marker.get("outputs")
        if not isinstance(outputs, dict):
            raise ArtifactError("frontend matching terminal marker has no hashes")
        for name in ("database.db", FRONTEND_SEAL_FILE):
            expected = outputs.get(name)
            target = root / name
            if (
                not isinstance(expected, str)
                or not _SHA256.fullmatch(expected)
                or sha256_file(target) != expected
            ):
                raise ArtifactError(
                    f"frontend matching terminal marker does not seal {name}"
                )
    return sealed


def stage_fingerprint(stage: str, inputs: Mapping[str, Any]) -> str:
    name = _safe_name(stage, "stage name")
    return canonical_hash(
        {"schema_version": 1, "stage": name, "inputs": _normal(inputs)}
    )


class StageLedger:
    """Atomic stage markers with strict input and output verification."""

    def __init__(self, artifact: str | Path):
        self.artifact = Path(artifact).resolve()
        if not self.artifact.is_dir():
            raise ArtifactError(f"frontend artifact does not exist: {artifact}")
        self.markers = self.artifact / "stages"
        self.markers.mkdir(exist_ok=True)

    def _path(self, stage: str) -> Path:
        return self.markers / f"{_safe_name(stage, 'stage name')}.json"

    def _load(self, stage: str) -> dict[str, Any] | None:
        path = self._path(stage)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"invalid stage marker {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ArtifactError(f"stage marker must contain an object: {path}")
        return value

    def _verify_outputs(self, marker: Mapping[str, Any]) -> None:
        outputs = marker.get("outputs")
        if not isinstance(outputs, Mapping):
            raise ArtifactError("complete stage marker has no outputs")
        for relative, expected in outputs.items():
            path = self.artifact / str(relative)
            if not path.is_file() or sha256_file(path) != expected:
                raise ArtifactError(f"completed stage output changed: {relative}")

    def begin(
        self, stage: str, inputs: Mapping[str, Any]
    ) -> Literal["run", "resume", "complete"]:
        fingerprint = stage_fingerprint(stage, inputs)
        existing = self._load(stage)
        if existing is not None:
            if existing.get("input_fingerprint") != fingerprint:
                raise ArtifactError(
                    f"stage {stage!r} inputs differ from its existing marker"
                )
            if existing.get("state") == "complete":
                self._verify_outputs(existing)
                return "complete"
            if existing.get("state") == "in_progress":
                return "resume"
            raise ArtifactError(f"stage {stage!r} marker has an invalid state")
        _atomic_json(
            self._path(stage),
            {
                "schema_version": 1,
                "stage": stage,
                "state": "in_progress",
                "input_fingerprint": fingerprint,
                "inputs": _normal(inputs),
            },
        )
        return "run"

    def complete(
        self,
        stage: str,
        inputs: Mapping[str, Any],
        outputs: Sequence[str | Path],
    ) -> None:
        fingerprint = stage_fingerprint(stage, inputs)
        marker = self._load(stage)
        if marker is None or marker.get("state") != "in_progress":
            raise ArtifactError(f"stage {stage!r} was not begun")
        if marker.get("input_fingerprint") != fingerprint:
            raise ArtifactError(f"stage {stage!r} inputs changed before completion")
        output_hashes: dict[str, str] = {}
        for output in outputs:
            path = Path(output)
            if not path.is_absolute():
                path = self.artifact / path
            try:
                relative = path.resolve().relative_to(self.artifact)
            except ValueError as exc:
                raise ArtifactError("stage outputs must stay inside the artifact") from exc
            if not path.is_file():
                raise ArtifactError(f"stage output is missing: {path}")
            output_hashes[relative.as_posix()] = sha256_file(path)
        complete = dict(marker)
        complete.update({"state": "complete", "outputs": output_hashes})
        _atomic_json(self._path(stage), complete, replace=True)


def create_database_snapshot(
    source_database: str | Path,
    destination_directory: str | Path,
    *,
    backend: str,
    expected_committed_sha256: str | None = None,
) -> dict[str, Any]:
    """Atomically give one backend a consistent copy, including committed WAL."""
    source = Path(source_database).resolve()
    destination = Path(destination_directory).resolve()
    backend_name = _safe_name(backend, "backend name")
    if not source.is_file():
        raise ArtifactError(f"source database does not exist: {source}")
    if expected_committed_sha256 is not None and not _SHA256.fullmatch(
        expected_committed_sha256
    ):
        raise ArtifactError("expected committed SQLite digest is not SHA-256")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite backend snapshot: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        f".{destination.name}.snapshot-{uuid.uuid4().hex}"
    )
    staging.mkdir()
    try:
        before = {
            "database.db": sha256_file(source),
            **(
                {"database.db-wal": sha256_file(source.with_name(source.name + "-wal"))}
                if source.with_name(source.name + "-wal").is_file()
                else {}
            ),
        }
        snapshot = staging / "database.db"
        consistent = _sqlite_online_backup(source, snapshot)
        after = {
            "database.db": sha256_file(source),
            **(
                {"database.db-wal": sha256_file(source.with_name(source.name + "-wal"))}
                if source.with_name(source.name + "-wal").is_file()
                else {}
            ),
        }
        if before != after:
            raise ArtifactError("source database changed during SQLite snapshot")
        if (
            expected_committed_sha256 is not None
            and consistent["sha256"] != expected_committed_sha256
        ):
            raise ArtifactError("source database differs from its committed-view seal")
        record = {
            "schema_version": 1,
            "backend": backend_name,
            "source_database": str(source),
            "source_files_before": before,
            "source_files_after": after,
            # Retained names keep existing consumers readable while recording
            # that equality to the snapshot is intentionally not assumed in WAL.
            "source_sha256_before": before["database.db"],
            "source_sha256_after": after["database.db"],
            "source_committed_sha256": consistent["sha256"],
            "snapshot_sha256": consistent["sha256"],
            "size_bytes": consistent["size_bytes"],
            "integrity_check": consistent["integrity_check"],
            "schema_sha256": consistent["schema_sha256"],
            "tables": consistent["tables"],
            "user_version": consistent["user_version"],
            "page_size": consistent["page_size"],
        }
        _atomic_json(staging / "database_snapshot.json", record)
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite backend snapshot: {destination}"
            )
        publish_directory_noreplace(staging, destination)
        return record
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
