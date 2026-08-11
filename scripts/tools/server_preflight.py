#!/usr/bin/env python3
"""Fail-closed environment and data check for the full-field server run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED = {
    "torch": "2.4.1+cu121",
    "torchvision": "0.19.1+cu121",
    "gsplat": "1.5.3+pt24cu121",
    "torchmetrics": "1.9.0",
    "numpy": "2.2.6",
    "opencv-python-headless": "5.0.0.93",
    "scipy": "1.15.3",
    "pymap3d": "3.2.0",
    "PyYAML": "6.0.3",
}
REQUIRED_COLMAP_COMMANDS = (
    "feature_extractor",
    "matches_importer",
    "global_mapper",
    "image_registrator",
    "model_analyzer",
    "model_converter",
)
VGG16_SHA256 = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"


def _gib(value: int) -> float:
    return float(value) / (1024.0**3)


def _memory() -> dict[str, float]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    return {
        "total_gib": _gib(values["MemTotal"]),
        "available_gib": _gib(values["MemAvailable"]),
    }


def _cpu() -> dict[str, object]:
    models = set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name") and ":" in line:
                models.add(line.split(":", 1)[1].strip())
    except OSError:
        pass
    affinity_count = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )
    return {
        "model_names": sorted(models),
        "logical_cpu_count": os.cpu_count(),
        "available_affinity_cpu_count": affinity_count,
    }


def _colmap(executable: Path) -> dict[str, object]:
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(f"COLMAP is not executable: {executable}")
    help_text = subprocess.run(
        [str(executable), "-h"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout
    banner = help_text.splitlines()[0] if help_text else ""
    if not banner.startswith("COLMAP 4.1.1") or "with CUDA" not in banner:
        raise RuntimeError(f"expected COLMAP 4.1.1 CUDA, got: {banner}")
    missing = [name for name in REQUIRED_COLMAP_COMMANDS if name not in help_text]
    if missing:
        raise RuntimeError(f"COLMAP is missing commands: {', '.join(missing)}")

    # A CUDA-labelled binary can still fail at its first real GPU call because
    # of an incompatible host driver. Exercise the exact SIFT path used by the
    # frontend before committing hours to the full database.
    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="rtk-splat-colmap-smoke-") as temporary:
        root = Path(temporary)
        images = root / "images"
        images.mkdir()
        rng = np.random.default_rng(7)
        pixels = rng.integers(0, 256, size=(512, 512), dtype=np.uint8)
        if not cv2.imwrite(str(images / "smoke.png"), pixels):
            raise RuntimeError("cannot write COLMAP CUDA smoke image")
        database = root / "database.db"
        command = [
            str(executable),
            "feature_extractor",
            "--database_path", str(database),
            "--image_path", str(images),
            "--default_random_seed", "7",
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.camera_params", "400,400,256,256",
            "--FeatureExtraction.use_gpu", "1",
            "--FeatureExtraction.gpu_index", "-1",
            "--FeatureExtraction.num_threads", "1",
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            output = getattr(exc, "stdout", "") or ""
            raise RuntimeError(
                "COLMAP CUDA feature-extraction smoke failed: "
                + output[-2000:]
            ) from exc
        with sqlite3.connect(database) as connection:
            row = connection.execute(
                "SELECT rows FROM descriptors ORDER BY image_id LIMIT 1"
            ).fetchone()
        if row is None or int(row[0]) <= 0:
            raise RuntimeError("COLMAP CUDA smoke extracted no descriptors")
        descriptor_count = int(row[0])
    return {
        "executable": str(executable.resolve()),
        "banner": banner,
        "cuda_feature_extraction_smoke": "PASSED",
        "cuda_smoke_descriptor_count": descriptor_count,
    }


def _cuda() -> dict[str, object]:
    import gsplat
    import torch
    from rtk_splat.backends.gsplat import render

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info(0)
    supported_markers = ("RTX 3090", "RTX 4090")
    if not any(marker in name for marker in supported_markers) or total < 22 * 1024**3:
        raise RuntimeError(
            "expected a supported 24 GB RTX 3090/4090, got "
            f"{name} ({_gib(total):.1f} GiB)"
        )
    if free < 18 * 1024**3:
        raise RuntimeError(f"only {_gib(free):.1f} GiB GPU memory is free")
    driver_lines = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    drivers = sorted({line.strip() for line in driver_lines if line.strip()})
    if len(drivers) != 1:
        raise RuntimeError(f"cannot identify one NVIDIA driver version: {drivers}")

    device = "cuda"
    params = {
        "means": torch.tensor([[0.0, 0.0, 3.0]], device=device),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        "scales": torch.log(torch.tensor([[0.1, 0.1, 0.1]], device=device)),
        "opacities": torch.tensor([0.0], device=device),
        "sh0": torch.zeros((1, 1, 3), device=device),
        "shN": torch.zeros((1, 0, 3), device=device),
    }
    viewmat = torch.eye(4, device=device)
    intrinsics = torch.tensor(
        [[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    rendered, alpha, _ = render(
        params, viewmat, intrinsics, 16, 16, 0
    )
    torch.cuda.synchronize()
    if rendered.shape != (1, 16, 16, 4) or alpha.shape != (1, 16, 16, 1):
        raise RuntimeError("gsplat CUDA rasterization smoke returned wrong shapes")
    return {
        "device": name,
        "total_gib": _gib(total),
        "free_gib": _gib(free),
        "torch_cuda": str(torch.version.cuda),
        "gsplat": str(gsplat.__version__),
        "nvidia_driver": drivers[0],
        "rasterization_smoke": "PASSED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--colmap", required=True, type=Path)
    parser.add_argument("--minimum-free-gib", type=float, default=500.0)
    parser.add_argument("--minimum-ram-gib", type=float, default=120.0)
    args = parser.parse_args()

    if not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib < 0.0:
        raise RuntimeError("--minimum-free-gib must be a finite non-negative value")
    if not math.isfinite(args.minimum_ram_gib) or args.minimum_ram_gib <= 0.0:
        raise RuntimeError("--minimum-ram-gib must be a finite positive value")

    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"expected Python 3.10, got {platform.python_version()}")
    versions = {
        name: importlib.metadata.version(name) for name in EXPECTED
    }
    mismatch = {
        name: {"expected": EXPECTED[name], "actual": versions[name]}
        for name in EXPECTED
        if versions[name] != EXPECTED[name]
    }
    if mismatch:
        raise RuntimeError(f"Python environment differs: {mismatch}")

    work_root = args.work_root.expanduser().resolve()
    existing = work_root
    while not existing.exists():
        if existing.parent == existing:
            raise RuntimeError(f"cannot resolve storage for {work_root}")
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    if _gib(usage.free) < float(args.minimum_free_gib):
        raise RuntimeError(
            f"only {_gib(usage.free):.1f} GiB free at {existing}; "
            f"need {args.minimum_free_gib:.1f} GiB"
        )
    memory = _memory()
    if memory["total_gib"] < float(args.minimum_ram_gib):
        raise RuntimeError(
            f"only {memory['total_gib']:.1f} GiB RAM; "
            f"need {args.minimum_ram_gib:.1f} GiB"
        )
    memory["recommendation"] = (
        "meets the full-field 128 GB-class target"
        if memory["total_gib"] >= 120.0
        else "below the 128 GB-class target; explicit lower override was used"
    )

    from rtk_splat.frontends.artifact import sha256_file
    from rtk_splat.workflows.segment_transfer import verify_portable_segment

    segment = verify_portable_segment(args.segment)
    vgg = Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth"
    if not vgg.is_file():
        raise RuntimeError(
            "LPIPS VGG weights are not cached; initialize the metric while online"
        )
    if sha256_file(vgg) != VGG16_SHA256:
        raise RuntimeError("cached LPIPS VGG weights failed SHA-256 verification")
    mount = subprocess.run(
        ["findmnt", "-no", "SOURCE,FSTYPE,TARGET", "--target", str(existing)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "schema_version": 1,
        "status": "PASSED",
        "requirements": {
            "minimum_free_gib": float(args.minimum_free_gib),
            "minimum_ram_gib": float(args.minimum_ram_gib),
        },
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": _cpu(),
            "memory": memory,
            "storage": {
                "mount": mount,
                "free_gib": _gib(usage.free),
                "total_gib": _gib(usage.total),
            },
        },
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": platform.python_version(),
            "packages": versions,
        },
        "cuda": _cuda(),
        "colmap": _colmap(args.colmap.expanduser()),
        "segment": segment,
        "lpips_vgg_sha256": VGG16_SHA256,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
