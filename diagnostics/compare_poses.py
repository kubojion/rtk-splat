"""Read-only pose comparison for controlled frontend/backend experiments."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _pose_arrays(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    viewmats = np.load(root / "viewmats.npy", allow_pickle=False)
    if (
        viewmats.ndim != 3
        or viewmats.shape[1:] != (4, 4)
        or not np.isfinite(viewmats).all()
    ):
        raise ValueError(f"{root}: invalid viewmats.npy")
    frame_path = root / "frame_ids.npy"
    frame_ids = (
        np.load(frame_path, allow_pickle=False)
        if frame_path.is_file()
        else np.arange(len(viewmats), dtype=np.int64)
    )
    if (
        frame_ids.shape != (len(viewmats),)
        or frame_ids.dtype.kind not in "iu"
        or np.any(np.diff(frame_ids) <= 0)
    ):
        raise ValueError(f"{root}: invalid frame_ids.npy")
    centers = np.linalg.inv(viewmats)[:, :3, 3]
    return viewmats.astype(np.float64), centers, frame_ids.astype(np.int64)


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def _project_rotations(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values)
    for index, matrix in enumerate(values):
        u, _, vh = np.linalg.svd(matrix)
        correction = np.eye(3)
        correction[-1, -1] = np.sign(np.linalg.det(u @ vh))
        output[index] = u @ correction @ vh
    return output


def compare_pose_artifacts(
    reference: str | Path, candidate: str | Path
) -> dict[str, Any]:
    """Compare already-georeferenced poses at identical frame IDs."""
    reference_root = Path(reference).expanduser().resolve()
    candidate_root = Path(candidate).expanduser().resolve()
    reference_view, reference_centers, reference_ids = _pose_arrays(reference_root)
    candidate_view, candidate_centers, candidate_ids = _pose_arrays(candidate_root)
    if not np.array_equal(reference_ids, candidate_ids):
        raise ValueError("reference and candidate frame IDs differ")

    translation = np.linalg.norm(candidate_centers - reference_centers, axis=1)
    candidate_rotation = _project_rotations(candidate_view[:, :3, :3])
    reference_rotation = _project_rotations(reference_view[:, :3, :3])
    relative = np.einsum(
        "nij,njk->nik",
        candidate_rotation,
        np.swapaxes(reference_rotation, 1, 2),
    )
    cosine = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0
    )
    rotation_deg = np.degrees(np.arccos(cosine))
    report: dict[str, Any] = {
        "schema_version": 1,
        "reference_pose_artifact": str(reference_root),
        "candidate_pose_artifact": str(candidate_root),
        "n_common_frames": len(reference_ids),
        "frame_ids_identical": True,
        "translation_delta_m": _distribution(translation),
        "rotation_delta_deg": _distribution(rotation_deg),
        "interpretation": (
            "candidate-minus-reference pose delta after each artifact's "
            "declared georeferencing; this is diagnostic, not an accuracy metric"
        ),
    }
    timestamps = candidate_root / "timestamps_ns.npy"
    if timestamps.is_file():
        values = np.load(timestamps, allow_pickle=False)
        if (
            values.shape != (len(candidate_ids),)
            or values.dtype != np.dtype(np.int64)
            or np.any(np.diff(values) <= 0)
        ):
            raise ValueError("candidate timestamps_ns.npy is invalid")
        report["candidate_timestamps"] = {
            "count": len(values),
            "strictly_increasing_int64_ns": True,
            "first": int(values[0]),
            "last": int(values[-1]),
        }
    for label, root in (
        ("reference_quality", reference_root),
        ("candidate_quality", candidate_root),
    ):
        path = root / "quality.json"
        if path.is_file():
            report[label] = _json(path)
    return report


def write_report_noreplace(path: str | Path, report: dict[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a candidate pose artifact with a fixed reference"
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = write_report_noreplace(
        args.output, compare_pose_artifacts(args.reference, args.candidate)
    )
    print(f"pose comparison -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
