"""Initial point-cloud workflow built from depth and a sealed pose artifact."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from rtk_splat.core.cloud import backproject, to_world, voxel_downsample
from rtk_splat.core.pose_artifacts import (
    cloud_path,
    load_pose_artifact,
    pose_artifact_name,
    pose_fingerprint,
)
from rtk_splat.core.segment import SegmentReader


def construct_initial_cloud(
    reader: SegmentReader, cfg
) -> tuple[Path, int]:
    """Publish a pose-bound cloud without hiding orchestration in the CLI."""
    frames = reader.frames
    viewmats, _ = load_pose_artifact(reader.root, cfg)
    camera = reader.calibration["cameras"]["left"]
    k = np.asarray(camera["K"], dtype=float)
    intrinsics = {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }
    if "depth_path" not in frames or any(
        not str(path) for path in frames["depth_path"]
    ):
        raise RuntimeError("cloud construction requires depth for every frame")
    points, colours = [], []
    for frame_id in reader.manifest["train"]:
        index = int(frame_id)
        with np.load(reader.root / str(frames["depth_path"][index])) as depth:
            image = cv2.imread(
                str(reader.root / str(frames["left_image_path"][index]))
            )
            if image is None:
                raise RuntimeError(f"cannot read frame {index}")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            xyz, colour = backproject(
                depth["depth"].astype(np.float32),
                depth["valid"].astype(bool),
                rgb,
                intrinsics,
                int(cfg.cloud.pixel_stride),
            )
        points.append(to_world(xyz, viewmats[index]))
        colours.append(colour)
    xyz = np.concatenate(points).astype(np.float32)
    rgb = np.concatenate(colours)
    xyz, rgb = voxel_downsample(
        xyz, rgb, float(cfg.cloud.voxel_m), int(cfg.cloud.max_points)
    )
    output = cloud_path(reader.root, cfg)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cloud artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                xyz=xyz,
                rgb=rgb,
                pose_fingerprint=np.asarray(pose_fingerprint(viewmats)),
                pose_artifact=np.asarray(pose_artifact_name(cfg)),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output, len(xyz)
