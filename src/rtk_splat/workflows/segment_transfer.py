"""Create and verify a self-contained segment for machine-to-machine transfer.

Derived segments deliberately refer to immutable source imagery with symlinks.
That is efficient on one machine, but copying such a directory as links can
leave a valid-looking segment with broken absolute targets on another host.
This workflow dereferences every contract asset into a new, atomically
published directory and seals the exact byte inventory in a terminal manifest.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from rtk_splat.core.segment import (
    OBSERVATION_KINDS,
    SegmentContractError,
    SegmentReader,
    SegmentWriter,
)
from rtk_splat.frontends.artifact import canonical_hash, sha256_file


TRANSFER_SCHEMA_VERSION = 1
TRANSFER_MANIFEST = "transfer_manifest.json"
_CONTRACT_FILES = (
    "frames.npz",
    "calibration.json",
    "segment_meta.json",
    "manifest.json",
)
_ASSET_FIELDS = ("left_image_path", "right_image_path", "depth_path")
_LINK_MODES = {"auto", "hardlink", "copy"}


def _safe_relative(value: str) -> str:
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != text
    ):
        raise SegmentContractError(
            f"transfer inventory has unsafe relative path: {text!r}"
        )
    return text


def _source_paths(reader: SegmentReader) -> list[str]:
    paths = list(_CONTRACT_FILES)
    for kind in OBSERVATION_KINDS:
        relative = f"observations/{kind}.npz"
        if (reader.root / relative).is_file():
            paths.append(relative)
    frames = reader.frames
    for field in _ASSET_FIELDS:
        if field not in frames:
            continue
        for value in frames[field].astype(str).tolist():
            if value:
                paths.append(_safe_relative(value))
    unique = sorted(set(paths))
    if TRANSFER_MANIFEST in unique:
        raise SegmentContractError(
            f"source segment must not reserve {TRANSFER_MANIFEST} as a frame asset"
        )
    return unique


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    if not path.is_file():
        raise SegmentContractError(f"segment asset is not a regular file: {path}")
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def _stable_record(root: Path, relative: str) -> tuple[dict[str, Any], Path, tuple[int, int, int, int]]:
    source = (root / relative).resolve(strict=True)
    before = _stat_signature(source)
    digest = sha256_file(source)
    after = _stat_signature(source)
    if before != after:
        raise SegmentContractError(
            f"source changed while transfer inventory was hashed: {relative}"
        )
    return (
        {
            "path": relative,
            "size_bytes": before[2],
            "sha256": digest,
        },
        source,
        after,
    )


def _materialize_file(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite {destination}")
    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError as exc:
            if mode == "hardlink" or exc.errno not in {
                errno.EXDEV,
                errno.EPERM,
                errno.EACCES,
                errno.EMLINK,
                errno.EOPNOTSUPP,
            }:
                raise
    shutil.copy2(source, destination)
    return "copy"


def _write_manifest(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def materialize_portable_segment(
    source_segment: str | Path,
    destination: str | Path,
    *,
    link_mode: str = "auto",
) -> Path:
    """Publish a self-contained segment without symlinks.

    ``auto`` uses hardlinks when source and destination share a filesystem and
    copies otherwise. Hardlinks consume no additional data blocks locally;
    normal transfer tools still send their file contents to another machine.
    The source must be treated as immutable, and both ends should run
    :func:`verify_portable_segment` before/after transfer.
    """
    if link_mode not in _LINK_MODES:
        raise ValueError(
            f"link_mode must be one of {', '.join(sorted(_LINK_MODES))}"
        )
    reader = SegmentReader(source_segment).validate()
    destination = Path(destination).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to modify existing segment: {destination}")

    relative_paths = _source_paths(reader)
    records: list[dict[str, Any]] = []
    sources: dict[str, tuple[Path, tuple[int, int, int, int]]] = {}
    for relative in relative_paths:
        record, source, signature = _stable_record(reader.root, relative)
        records.append(record)
        sources[relative] = (source, signature)

    writer = SegmentWriter(destination)
    methods = {"hardlink": 0, "copy": 0}
    try:
        for record in records:
            relative = str(record["path"])
            source, _ = sources[relative]
            method = _materialize_file(
                source, writer.staging_dir / relative, link_mode
            )
            methods[method] += 1

        # Close the source-mutation window before publication. The content
        # hashes above are authoritative; this final identity/size/mtime pass
        # rejects files changed during the materialization step.
        for relative, (source, expected) in sources.items():
            if _stat_signature(source) != expected:
                raise SegmentContractError(
                    f"source changed while segment was materialized: {relative}"
                )

        manifest = {
            "schema_version": TRANSFER_SCHEMA_VERSION,
            "artifact_type": "rtk_splat_portable_segment",
            "contract_version": int(reader.meta["contract_version"]),
            "n_frames": int(reader.meta["n_frames"]),
            "files": records,
            "inventory_sha256": canonical_hash(records),
            "materialization": {
                "requested_mode": link_mode,
                "hardlink_files": methods["hardlink"],
                "copied_files": methods["copy"],
            },
        }
        _write_manifest(writer.staging_dir / TRANSFER_MANIFEST, manifest)
        published = writer.finalize()
    except Exception:
        writer.abort()
        raise

    verify_portable_segment(published.root)
    return published.root


def verify_portable_segment(segment: str | Path) -> dict[str, Any]:
    """Rehash a portable segment and reject missing, extra, or linked files."""
    root = Path(segment).expanduser()
    manifest_path = root / TRANSFER_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SegmentContractError(
            f"cannot read portable transfer manifest: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise SegmentContractError("portable transfer manifest must be an object")
    if manifest.get("schema_version") != TRANSFER_SCHEMA_VERSION:
        raise SegmentContractError("unsupported portable transfer schema")
    if manifest.get("artifact_type") != "rtk_splat_portable_segment":
        raise SegmentContractError("not an RTK-Splat portable segment")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise SegmentContractError("portable transfer inventory is empty")

    expected: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise SegmentContractError("portable transfer file record is invalid")
        relative = _safe_relative(record["path"])
        if relative == TRANSFER_MANIFEST or relative in expected:
            raise SegmentContractError("portable transfer inventory has duplicate/reserved path")
        if (
            isinstance(record["size_bytes"], bool)
            or not isinstance(record["size_bytes"], int)
            or record["size_bytes"] < 0
        ):
            raise SegmentContractError("portable transfer size is invalid")
        digest = record["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise SegmentContractError("portable transfer SHA-256 is invalid")
        expected[relative] = record
    if list(expected) != sorted(expected):
        raise SegmentContractError("portable transfer inventory must be sorted")
    if manifest.get("inventory_sha256") != canonical_hash(records):
        raise SegmentContractError("portable transfer inventory hash disagrees")

    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SegmentContractError(
                f"portable segment contains a symlink: {path.relative_to(root)}"
            )
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    declared = set(expected) | {TRANSFER_MANIFEST}
    if actual != declared:
        missing = sorted(declared - actual)
        extra = sorted(actual - declared)
        raise SegmentContractError(
            f"portable transfer file set differs; missing={missing}, extra={extra}"
        )

    for relative, record in expected.items():
        path = root / relative
        if path.stat().st_size != record["size_bytes"]:
            raise SegmentContractError(
                f"portable transfer size mismatch: {relative}"
            )
        if sha256_file(path) != record["sha256"]:
            raise SegmentContractError(
                f"portable transfer SHA-256 mismatch: {relative}"
            )

    reader = SegmentReader(root).validate()
    if manifest.get("contract_version") != int(reader.meta["contract_version"]):
        raise SegmentContractError("portable transfer contract version disagrees")
    if manifest.get("n_frames") != int(reader.meta["n_frames"]):
        raise SegmentContractError("portable transfer frame count disagrees")
    return {
        "segment": str(root.resolve()),
        "verified": True,
        "n_frames": int(reader.meta["n_frames"]),
        "n_files": len(records),
        "inventory_sha256": str(manifest["inventory_sha256"]),
        "size_bytes": int(sum(record["size_bytes"] for record in records)),
    }
