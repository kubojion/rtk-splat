"""Read-only verification of a recorded RTK-Splat experiment.

``exact`` mode verifies the compact files of the original artifact byte for
byte. ``acceptance`` mode is for a fresh reproduction, where optimization can
be nondeterministic but pose and image-quality gates must still pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .pose_artifacts import pose_fingerprint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _last_metrics(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: expected a non-empty metrics history")
    return value[-1]


def _check_min(
    checks: list[tuple[bool, str]], name: str, value: float, minimum: float
) -> None:
    checks.append(
        (value >= minimum, f"{name}: {value:.9g} (minimum {minimum:.9g})")
    )


def _check_max(
    checks: list[tuple[bool, str]], name: str, value: float, maximum: float
) -> None:
    checks.append(
        (value <= maximum, f"{name}: {value:.9g} (maximum {maximum:.9g})")
    )


def verify(
    manifest_path: Path,
    workdir: Path | None = None,
    mode: str = "exact",
    pose_artifact: str | None = None,
    run_name: str | None = None,
) -> list[tuple[bool, str]]:
    """Return ordered ``(passed, description)`` checks without writing."""
    manifest = json.loads(manifest_path.read_text())
    layout = manifest["artifact_layout"]
    root = Path(workdir or layout["recorded_workdir"]).expanduser()
    segment = root / "segment"
    pose_name = pose_artifact or layout["pose_artifact"]
    run = run_name or layout["run_name"]
    pose_dir = segment / "pose_artifacts" / pose_name
    run_dir = root / "runs" / run
    expected = manifest["expected"]
    checks: list[tuple[bool, str]] = []

    if mode == "exact":
        for record in manifest["compact_files"]:
            path = root / record["path"]
            if not path.is_file():
                checks.append((False, f"missing compact artifact: {path}"))
                continue
            actual = _sha256(path)
            checks.append(
                (
                    actual == record["sha256"],
                    f"SHA-256 {record['path']}: {actual}",
                )
            )

    meta_path = segment / "segment_meta.json"
    quality_path = pose_dir / "quality.json"
    pose_path = pose_dir / "viewmats.npy"
    cloud_path = pose_dir / "init_cloud.npz"
    metrics_path = run_dir / "metrics.json"
    required = (meta_path, quality_path, pose_path, cloud_path, metrics_path)
    missing = [path for path in required if not path.is_file()]
    checks.extend((False, f"missing required artifact: {path}") for path in missing)
    if missing:
        return checks

    meta = json.loads(meta_path.read_text())
    n_frames = int(meta["n_frames"])
    checks.append(
        (
            n_frames == int(expected["segment"]["n_frames"]),
            f"segment frames: {n_frames}",
        )
    )

    quality = json.loads(quality_path.read_text())
    registered = int(quality["n_registered_left"])
    checks.append(
        (
            registered == n_frames,
            f"registered left frames: {registered}/{n_frames}",
        )
    )
    alignment = quality["alignment"]
    scale = float(alignment["scale"])
    scale_lo, scale_hi = expected["pose"]["scale_range"]
    checks.append(
        (
            scale_lo <= scale <= scale_hi,
            f"metric scale: {scale:.9g} (range {scale_lo}-{scale_hi})",
        )
    )

    viewmats = np.load(pose_path, mmap_mode="r")
    fingerprint = pose_fingerprint(viewmats)
    checks.append(
        (
            fingerprint == quality["pose_fingerprint"],
            f"pose fingerprint: {fingerprint}",
        )
    )
    with np.load(cloud_path) as cloud:
        stored = (
            str(cloud["pose_fingerprint"].item())
            if "pose_fingerprint" in cloud
            else None
        )
    checks.append(
        (
            stored == fingerprint,
            f"cloud/pose fingerprint match: {stored == fingerprint}",
        )
    )

    metrics = _last_metrics(metrics_path)
    gates = expected["acceptance_gates"]
    _check_min(
        checks,
        "masked PSNR",
        float(metrics["psnr_masked"]),
        float(gates["psnr_masked_min"]),
    )
    _check_min(
        checks,
        "color-corrected masked PSNR",
        float(metrics["psnr_masked_cc"]),
        float(gates["psnr_masked_cc_min"]),
    )
    _check_max(
        checks,
        "color-corrected LPIPS",
        float(metrics["lpips_cc"]),
        float(gates["lpips_cc_max"]),
    )
    _check_min(
        checks,
        "SSIM",
        float(metrics["ssim"]),
        float(gates["ssim_min"]),
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only verification of an RTK-Splat golden result"
    )
    default_manifest = (
        Path(__file__).resolve().parent.parent
        / "docs/experiments/golden/headland_stereo_ba.json"
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument(
        "--mode", choices=("exact", "acceptance"), default="exact"
    )
    parser.add_argument("--pose-artifact")
    parser.add_argument("--run")
    args = parser.parse_args()

    checks = verify(
        args.manifest,
        args.workdir,
        args.mode,
        args.pose_artifact,
        args.run,
    )
    for passed, description in checks:
        print(("PASS" if passed else "FAIL") + f"  {description}")
    failures = sum(not passed for passed, _ in checks)
    print(f"{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
