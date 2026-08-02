"""Sealed pose-georeferencing evidence propagated to render artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rtk_splat.core.pose_artifacts import (
    pose_artifact_dir,
    pose_artifact_name,
)


_HASH_FIELDS = (
    "pose_quality_sha256",
    "pose_manifest_sha256",
    "pose_georeferencing_sha256",
)
_CLOUD_FIELDS = (
    "artifact_class",
    "georeferencing_status",
    "metric_georeferencing_claim_eligible",
    *_HASH_FIELDS,
)
_STATUS_FIELDS = (
    "artifact_class",
    "georeferencing_status",
    "metric_georeferencing_claim_eligible",
    "diagnostic_export_requested",
    "diagnostic_export_override_used",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _verify_manifest(root: Path) -> tuple[str | None, dict | None]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None, None
    manifest = _json_object(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"pose artifact manifest has no file evidence: {root}")
    resolved_root = root.resolve()
    for relative, record in files.items():
        if (
            not isinstance(relative, str)
            or not isinstance(record, dict)
            or not isinstance(record.get("sha256"), str)
            or not isinstance(record.get("size_bytes"), int)
        ):
            raise ValueError(f"invalid pose manifest entry: {relative!r}")
        candidate = (root / relative).resolve()
        if resolved_root != candidate and resolved_root not in candidate.parents:
            raise ValueError(f"pose manifest path escapes artifact: {relative!r}")
        if not candidate.is_file():
            raise ValueError(f"pose artifact file is missing: {candidate}")
        if (
            candidate.stat().st_size != record["size_bytes"]
            or _sha256_file(candidate) != record["sha256"]
        ):
            raise ValueError(f"pose artifact file changed: {candidate}")
    return _sha256_file(manifest_path), manifest


def canonical_georeferencing_json(evidence: dict[str, Any]) -> str:
    return json.dumps(evidence, sort_keys=True, separators=(",", ":"))


def pose_georeferencing_evidence(seg_dir: Path, cfg) -> dict[str, Any]:
    """Read a modern sealed declaration or classify an older pose honestly."""
    name = pose_artifact_name(cfg)
    if name == "rtk":
        return _legacy_evidence(name)
    root = pose_artifact_dir(seg_dir, cfg)
    return _named_pose_evidence(root, name=name, require_modern=False)


def verify_pose_georeferencing_artifact(
    root: Path,
    *,
    expected_name: str | None = None,
) -> dict[str, Any]:
    """Verify and return one modern manifest-sealed pose declaration."""
    root = Path(root)
    return _named_pose_evidence(
        root,
        name=expected_name or root.name,
        require_modern=True,
    )


def _named_pose_evidence(
    root: Path,
    *,
    name: str,
    require_modern: bool,
) -> dict[str, Any]:
    manifest_sha256, manifest = _verify_manifest(root)
    manifest_name = (manifest or {}).get("name")
    if require_modern and not isinstance(manifest_name, str):
        raise ValueError("pose manifest has no artifact name")
    if manifest_name is not None:
        if manifest_name != name:
            raise ValueError(
                f"pose manifest name {manifest_name!r} does not match {name!r}"
            )
    quality_path = root / "quality.json"
    quality_sha256 = _sha256_file(quality_path) if quality_path.is_file() else None
    declaration_path = root / "georeferencing.json"
    if not declaration_path.is_file():
        if require_modern:
            raise ValueError("pose artifact has no georeferencing.json declaration")
        quality = _json_object(quality_path) if quality_path.is_file() else {}
        return _legacy_evidence(
            name,
            failed=quality.get("rtk_alignment_passed") is False,
            quality_sha256=quality_sha256,
            manifest_sha256=manifest_sha256,
        )
    if manifest is None:
        raise ValueError("georeferencing declaration is not sealed by a pose manifest")
    required = {
        "viewmats.npy",
        "cam_centers.npy",
        "quality.json",
        "alignment.json",
        "provenance.json",
        "georeferencing.json",
    }
    missing = sorted(required - set(manifest["files"]))
    if missing:
        raise ValueError("pose manifest does not seal required files: " + ", ".join(missing))
    declaration = _json_object(declaration_path)
    artifact_class = declaration.get("artifact_class")
    status = declaration.get("georeferencing_status")
    eligible = declaration.get("metric_georeferencing_claim_eligible")
    diagnostic_requested = declaration.get("diagnostic_export_requested")
    diagnostic_override = declaration.get("diagnostic_export_override_used")
    if artifact_class not in ("production", "diagnostic_render_only"):
        raise ValueError("pose georeferencing artifact_class is invalid")
    if status not in ("PASSED", "FAILED"):
        raise ValueError("pose georeferencing_status must be PASSED or FAILED")
    if not isinstance(eligible, bool):
        raise ValueError("metric_georeferencing_claim_eligible must be boolean")
    if artifact_class == "production" and (
        status != "PASSED"
        or not eligible
        or diagnostic_requested is not False
        or diagnostic_override is not False
    ):
        raise ValueError("production georeferencing status tuple is invalid")
    if artifact_class == "diagnostic_render_only" and (
        eligible
        or diagnostic_requested is not True
        or diagnostic_override is not (status == "FAILED")
    ):
        raise ValueError("diagnostic georeferencing status tuple is invalid")
    quality = _json_object(quality_path)
    if quality.get("rtk_alignment_passed") is not (status == "PASSED"):
        raise ValueError("pose quality and georeferencing status disagree")
    records = {
        "manifest": manifest,
        "quality": quality,
        "alignment": _json_object(root / "alignment.json"),
        "provenance": _json_object(root / "provenance.json"),
    }
    for label, record in records.items():
        for key in _STATUS_FIELDS:
            if record.get(key) != declaration.get(key):
                raise ValueError(
                    f"pose {label} and georeferencing {key} disagree"
                )

    marker_path = root / "GEOREFERENCING_FAILED.json"
    if status == "FAILED":
        if "GEOREFERENCING_FAILED.json" not in manifest["files"]:
            raise ValueError("pose manifest does not seal failed marker")
        marker = _json_object(marker_path)
        for key in _STATUS_FIELDS:
            if marker.get(key) != declaration.get(key):
                raise ValueError(f"failed marker and georeferencing {key} disagree")
        for key in ("diagnostic_report", "diagnostic_report_sha256"):
            if marker.get(key) != declaration.get(key):
                raise ValueError(f"failed marker and georeferencing {key} disagree")
    elif marker_path.exists() or "GEOREFERENCING_FAILED.json" in manifest["files"]:
        raise ValueError("non-failed pose artifact contains a failed marker")
    evidence = dict(declaration)
    evidence.update(
        pose_artifact=name,
        pose_quality_sha256=quality_sha256,
        pose_manifest_sha256=manifest_sha256,
        pose_georeferencing_sha256=_sha256_file(declaration_path),
    )
    return evidence


def _legacy_evidence(
    name: str,
    *,
    failed: bool = False,
    quality_sha256: str | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pose_artifact": name,
        "artifact_class": "diagnostic_render_only" if failed else "legacy_unassessed",
        "georeferencing_status": "FAILED" if failed else "legacy_unassessed",
        "metric_georeferencing_claim_eligible": False,
        "pose_quality_sha256": quality_sha256,
        "pose_manifest_sha256": manifest_sha256,
        "pose_georeferencing_sha256": None,
    }


def require_render_permission(
    evidence: dict[str, Any], *, allow_failed_georeferencing_for_render: bool
) -> None:
    diagnostic = (
        evidence["artifact_class"] == "diagnostic_render_only"
        or evidence["georeferencing_status"] == "FAILED"
    )
    if diagnostic and not allow_failed_georeferencing_for_render:
        raise ValueError(
            "pose artifact requires explicit render-only permission and is not "
            "eligible for a metric georeferencing claim"
        )


def splat_output_name(evidence: dict[str, Any]) -> str:
    if evidence.get("artifact_class") == "diagnostic_render_only":
        return "splat.DIAGNOSTIC_ONLY.ply"
    return "splat.ply"


def verify_training_run_georeferencing(
    run_dir: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Verify that a completed GS run cannot shed its pose-use restriction."""
    run_dir = Path(run_dir)
    expected_json = canonical_georeferencing_json(expected)
    run_evidence = _json_object(run_dir / "georeferencing.json")
    if canonical_georeferencing_json(run_evidence) != expected_json:
        raise ValueError("training georeferencing evidence differs from its pose")
    provenance = _json_object(run_dir / "run_provenance.json")
    if canonical_georeferencing_json(provenance.get("georeferencing")) != expected_json:
        raise ValueError("training provenance dropped or changed georeferencing")
    if provenance.get("pose_artifact") != expected.get("pose_artifact"):
        raise ValueError("training provenance names a different pose artifact")

    selected_name = splat_output_name(expected)
    alternate_name = (
        "splat.ply"
        if selected_name == "splat.DIAGNOSTIC_ONLY.ply"
        else "splat.DIAGNOSTIC_ONLY.ply"
    )
    selected = run_dir / selected_name
    if not selected.is_file() or selected.stat().st_size == 0:
        raise ValueError(f"selected Gaussian PLY is missing or empty: {selected}")
    if (run_dir / alternate_name).exists():
        raise ValueError(f"ambiguous alternate Gaussian PLY exists: {alternate_name}")
    splat_evidence = _json_object(run_dir / "splat.georeferencing.json")
    for key, value in expected.items():
        if splat_evidence.get(key) != value:
            raise ValueError(f"splat georeferencing changed field {key}")
    selected_sha256 = _sha256_file(selected)
    if (
        splat_evidence.get("splat_file") != selected_name
        or splat_evidence.get("splat_sha256") != selected_sha256
    ):
        raise ValueError("splat georeferencing file/hash evidence is invalid")

    failed_marker = run_dir / "GEOREFERENCING_FAILED.json"
    if expected["georeferencing_status"] == "FAILED":
        marker = _json_object(failed_marker)
        if canonical_georeferencing_json(marker) != expected_json:
            raise ValueError("failed-georeferencing marker changed its evidence")
    elif failed_marker.exists():
        raise ValueError("non-failed run contains a failed-georeferencing marker")
    return {
        "georeferencing": run_evidence,
        "splat_file": str(selected),
        "splat_sha256": selected_sha256,
    }


def cloud_georeferencing_evidence(
    cloud_file: Path,
    expected: dict[str, Any],
    *,
    allow_failed_georeferencing_for_render: bool,
) -> dict[str, Any] | None:
    """Verify that a pose-bound cloud retained its source declaration exactly."""
    with np.load(cloud_file, allow_pickle=False) as cloud:
        if "georeferencing_json" not in cloud:
            stored = None
            redundant = {}
        else:
            try:
                stored = json.loads(str(cloud["georeferencing_json"].item()))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{cloud_file} has invalid georeferencing provenance") from exc
            redundant = {key: cloud[key].item() if key in cloud else None
                         for key in _CLOUD_FIELDS}
    if stored is None:
        if expected["georeferencing_status"] != "legacy_unassessed":
            raise ValueError(f"{cloud_file} dropped pose georeferencing provenance")
        return None
    if canonical_georeferencing_json(stored) != canonical_georeferencing_json(expected):
        raise ValueError(f"{cloud_file} was built with different georeferencing evidence")
    for key, value in redundant.items():
        if value is None:
            raise ValueError(f"{cloud_file} dropped georeferencing field {key}")
        if key in _HASH_FIELDS and value == "":
            value = None
        if value != expected[key]:
            raise ValueError(f"{cloud_file} has inconsistent georeferencing field {key}")
    require_render_permission(
        stored,
        allow_failed_georeferencing_for_render=allow_failed_georeferencing_for_render,
    )
    return stored
