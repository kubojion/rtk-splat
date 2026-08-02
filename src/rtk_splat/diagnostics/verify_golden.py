"""Read-only verification of a recorded RTK-Splat experiment.

``exact`` mode verifies the compact files of the original artifact byte for
byte. ``acceptance`` mode is for a fresh reproduction, where optimization can
be nondeterministic but pose and image-quality gates must still pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rtk_splat.backends.pose_evidence import (
    verify_pose_georeferencing_artifact,
    verify_training_run_georeferencing,
)
from rtk_splat.core.pose_artifacts import pose_fingerprint


_DEFAULT_MANIFEST = Path("docs/experiments/golden/headland_stereo_ba.json")


def discover_default_manifest(source_file: str | Path | None = None) -> Path | None:
    """Return the repository golden manifest only in a source checkout."""
    source = Path(source_file or __file__).resolve()
    for root in source.parents:
        owns_source = any(
            source.is_relative_to(root / item)
            for item in ("src/rtk_splat", "rtk_splat")
        )
        if (
            owns_source
            and (root / "pyproject.toml").is_file()
            and (root / _DEFAULT_MANIFEST).is_file()
        ):
            return root / _DEFAULT_MANIFEST
    return None


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


def _quality_value(quality: dict[str, Any], dotted_path: str) -> Any:
    """Read a manifest-declared value without assuming a mapper schema."""
    value: Any = quality
    for component in dotted_path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise KeyError(
                f"quality field {dotted_path!r} is missing at {component!r}"
            )
        value = value[component]
    return value


def _exact_number(actual: Any, expected: Any) -> bool:
    """Compare persisted JSON scalars exactly, including finite floats."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if not isinstance(actual, (int, float)) or not isinstance(
        expected, (int, float)
    ):
        return actual == expected
    return math.isfinite(float(actual)) and float(actual) == float(expected)


def _check_min(
    checks: list[tuple[bool, str]], name: str, value: float, minimum: float
) -> None:
    message = f"{name}: {value:.9g} (minimum {minimum:.9g})"
    checks.append((value >= minimum, message))


def _check_max(
    checks: list[tuple[bool, str]], name: str, value: float, maximum: float
) -> None:
    checks.append(
        (value <= maximum, f"{name}: {value:.9g} (maximum {maximum:.9g})")
    )


def _modern_georeferencing_check(
    pose_dir: Path,
    run_dir: Path,
    quality: dict[str, Any],
) -> tuple[bool, str] | None:
    """Reject diagnostic/tampered modern outputs without changing legacy checks."""
    provenance_path = run_dir / "run_provenance.json"
    try:
        provenance = (
            json.loads(provenance_path.read_text())
            if provenance_path.is_file()
            else {}
        )
    except (OSError, json.JSONDecodeError):
        provenance = {}
    if not isinstance(provenance, dict):
        provenance = {}
    modern = (
        any(
            key in quality
            for key in (
                "artifact_class",
                "georeferencing_status",
                "metric_georeferencing_claim_eligible",
            )
        )
        or (pose_dir / "georeferencing.json").is_file()
        or (run_dir / "georeferencing.json").is_file()
        or (run_dir / "splat.georeferencing.json").is_file()
        or isinstance(provenance.get("georeferencing"), dict)
        or (pose_dir / "GEOREFERENCING_FAILED.json").exists()
        or (run_dir / "GEOREFERENCING_FAILED.json").exists()
        or (run_dir / "splat.DIAGNOSTIC_ONLY.ply").exists()
    )
    if not modern:
        return None
    try:
        evidence = verify_pose_georeferencing_artifact(
            pose_dir, expected_name=pose_dir.name
        )
        verify_training_run_georeferencing(run_dir, evidence)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return (
            False,
            "modern output is ineligible for golden/publication acceptance: "
            f"{exc}",
        )
    expected = {
        "artifact_class": "production",
        "georeferencing_status": "PASSED",
        "metric_georeferencing_claim_eligible": True,
    }
    errors = [
        f"{key}={evidence.get(key)!r}"
        for key, value in expected.items()
        if evidence.get(key) != value
    ]
    if errors:
        return (
            False,
            "modern output is ineligible for golden/publication acceptance: "
            + "; ".join(errors),
        )
    return True, "modern georeferencing evidence: production/PASSED/eligible"


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
    modern_georeferencing = _modern_georeferencing_check(
        pose_dir, run_dir, quality
    )
    if modern_georeferencing is not None:
        checks.append(modern_georeferencing)
    registered = int(quality["n_registered_left"])
    checks.append(
        (
            registered == n_frames,
            f"registered left frames: {registered}/{n_frames}",
        )
    )
    for scale_check in expected["pose"]["scale_checks"]:
        scale_name = scale_check["name"]
        scale_path = scale_check["quality_path"]
        try:
            scale = float(_quality_value(quality, scale_path))
        except (KeyError, TypeError, ValueError) as exc:
            checks.append((False, f"{scale_name} field {scale_path}: {exc}"))
            continue
        scale_lo, scale_hi = scale_check["range"]
        checks.append(
            (
                scale_lo <= scale <= scale_hi,
                f"{scale_name} ({scale_path}): {scale:.9g} "
                f"(range {scale_lo}-{scale_hi})",
            )
        )
        if mode == "exact":
            expected_scale = scale_check["expected"]
            checks.append(
                (
                    _exact_number(scale, expected_scale),
                    f"exact {scale_name}: {scale:.17g} "
                    f"(expected {float(expected_scale):.17g})",
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
    if mode == "exact":
        expected_fingerprint = expected["pose"]["pose_fingerprint"]
        checks.append(
            (
                fingerprint == expected_fingerprint,
                f"golden pose fingerprint: {fingerprint}",
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
    if mode == "exact":
        for field in expected["exact_metric_fields"]:
            if field not in metrics:
                checks.append((False, f"missing exact metric: {field}"))
                continue
            exact_expected = expected["metrics"][field]
            exact_actual = metrics[field]
            checks.append(
                (
                    _exact_number(exact_actual, exact_expected),
                    f"exact metric {field}: {exact_actual!r} "
                    f"(expected {exact_expected!r})",
                )
            )
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only verification of an RTK-Splat golden result")
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "experiment manifest to verify; the repository headland manifest "
            "is selected automatically only from a source checkout"
        ),
    )
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--mode", choices=("exact", "acceptance"), default="exact")
    parser.add_argument("--pose-artifact")
    parser.add_argument("--run")
    args = parser.parse_args(argv)

    manifest = args.manifest or discover_default_manifest()
    if manifest is None:
        parser.error("--manifest is required outside an RTK-Splat source checkout")
    manifest = manifest.expanduser()
    if not manifest.is_file():
        parser.error(f"manifest does not exist: {manifest}")

    checks = verify(
        manifest,
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
