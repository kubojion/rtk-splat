"""Create a new immutable contract-v2 segment with computed stereo depth."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from rtk_splat.core.depth import depth_from_pair, make_sgbm
from rtk_splat.core.runtime_resolution import (
    configuration_evidence,
    runtime_resolution_plain,
)
from rtk_splat.workflows.runtime_config import resolve_depth_max_z
from rtk_splat.core.segment import OBSERVATION_KINDS, SegmentReader, SegmentWriter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_frame_files(
    writer: SegmentWriter, reader: SegmentReader, frames: dict[str, np.ndarray]
) -> None:
    paths = {
        str(relative)
        for field in ("left_image_path", "right_image_path")
        if field in frames
        for relative in frames[field]
    }
    for relative in sorted(paths):
        source = (reader.root / relative).resolve()
        destination = writer.staging_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source, destination)


def derive_sgbm_depth(
    source_segment: str | Path,
    destination: str | Path,
    cfg,
) -> SegmentReader:
    """Publish SGBM depth beside symlinked source images without mutation."""
    reader = SegmentReader(source_segment).validate()
    capabilities = reader.meta["capabilities"]
    if not capabilities["stereo"] or not capabilities["images_rectified"]:
        raise ValueError("SGBM depth requires rectified stereo images")
    if capabilities["depth_recorded"] or capabilities["depth_computed"]:
        raise ValueError("source segment already declares depth")

    transform = np.asarray(reader.calibration["T_right_left"], dtype=float)
    if not np.allclose(transform[:3, :3], np.eye(3), atol=1e-4):
        raise ValueError("SGBM requires a rectified stereo rotation")
    if not np.allclose(transform[1:3, 3], 0.0, atol=1e-4):
        raise ValueError("SGBM requires a horizontal rectified baseline")
    baseline_m = abs(float(transform[0, 3]))
    if baseline_m <= 0:
        raise ValueError("stereo baseline must be non-zero")
    resolve_depth_max_z(cfg, reader.calibration)

    frames = reader.frames
    camera = reader.calibration["cameras"]["left"]
    fx = float(np.asarray(camera["K"], dtype=float)[0, 0])
    writer = SegmentWriter(destination)
    try:
        _link_frame_files(writer, reader, frames)
        depth_directory = writer.directory("depth")
        matcher = make_sgbm(cfg)
        depth_paths = []
        for index in range(len(frames["frame_id"])):
            left = cv2.imread(
                str(reader.root / str(frames["left_image_path"][index]))
            )
            right = cv2.imread(
                str(reader.root / str(frames["right_image_path"][index]))
            )
            if left is None or right is None:
                raise RuntimeError(f"cannot decode stereo frame {index}")
            depth, valid = depth_from_pair(
                matcher, left, right, fx, baseline_m, cfg
            )
            filename = f"{index:06d}.npz"
            np.savez_compressed(
                depth_directory / filename,
                depth=depth.astype(np.float16),
                valid=valid.astype(bool),
            )
            depth_paths.append(f"depth/{filename}")
            if (index + 1) % 100 == 0:
                print(f"depth {index + 1}/{len(frames['frame_id'])}", flush=True)

        output_frames = dict(frames)
        output_frames["depth_path"] = np.asarray(depth_paths, dtype=np.str_)
        meta = json.loads(json.dumps(reader.meta))
        meta["capabilities"]["depth_computed"] = True
        meta["depth_observation"] = {
            "format": "npz_depth_valid",
            "units": "m",
            "quantity": "optical_z",
            "aligned_to": "left",
            "invalid_convention": "depth=0 and valid=false",
        }
        meta["derived_segment"] = {
            "operation": "rectified_stereo_sgbm",
            "source_segment": str(reader.root.resolve()),
            "source_contract_sha256": {
                name: _sha256(reader.root / name)
                for name in (
                    "frames.npz",
                    "calibration.json",
                    "segment_meta.json",
                    "manifest.json",
                    "observations/gnss.npz",
                )
            },
            "baseline_m": baseline_m,
            "min_z_m": float(cfg.depth.min_z_m),
            "max_z_m": float(cfg.depth.max_z_m),
            "runtime_resolution": runtime_resolution_plain(cfg),
            "configuration": configuration_evidence(cfg),
        }
        writer.write_frames(output_frames)
        writer.write_calibration(reader.calibration)
        writer.write_meta(meta)
        writer.write_manifest(reader.manifest)
        for kind in OBSERVATION_KINDS:
            values = reader.observations(kind, required=False)
            if values is not None:
                writer.write_observations(kind, values)
        return writer.finalize()
    except Exception:
        writer.abort()
        raise
