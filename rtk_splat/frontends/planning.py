"""Build a deterministic, dataset-neutral frontend plan from contract v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from rtk_splat.core.segment import SegmentReader

from .keyframes import KeyframeConfig, KeyframeSelection, select_keyframes
from .pair_graph import PairGraph, PairGraphConfig, build_pair_graph


@dataclass(frozen=True)
class FrontendPlan:
    rig_config: list[dict[str, Any]]
    keyframes: dict[str, Any]
    pairs: tuple[tuple[str, str], ...]
    quality: dict[str, Any]
    selection: KeyframeSelection
    pair_graph: PairGraph


def _camera_params(camera: dict[str, Any]) -> list[float]:
    intrinsics = np.asarray(camera["K"], dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError("camera K must be 3x3")
    # Preserve the convention of the accepted reconstruction: OpenCV pixel
    # centres start at (0,0), while this COLMAP build expects (+0.5,+0.5).
    return [
        float(intrinsics[0, 0]),
        float(intrinsics[1, 1]),
        float(intrinsics[0, 2] + 0.5),
        float(intrinsics[1, 2] + 0.5),
    ]


def make_rig_config(calibration: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate contract-v2 right-from-left calibration into COLMAP rig JSON."""
    cameras = calibration.get("cameras", {})
    if not {"left", "right"} <= set(cameras):
        raise ValueError("a stereo frontend requires left and right calibration")
    transform = np.asarray(calibration.get("T_right_left"), dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("T_right_left must be 4x4")
    quaternion_xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    quaternion_wxyz = [
        float(quaternion_xyzw[3]),
        float(quaternion_xyzw[0]),
        float(quaternion_xyzw[1]),
        float(quaternion_xyzw[2]),
    ]
    return [
        {
            "cameras": [
                {
                    "image_prefix": "left_",
                    "ref_sensor": True,
                    "camera_model_name": "PINHOLE",
                    "camera_params": _camera_params(cameras["left"]),
                },
                {
                    "image_prefix": "right_",
                    "cam_from_rig_rotation": quaternion_wxyz,
                    "cam_from_rig_translation": transform[:3, 3].tolist(),
                    "camera_model_name": "PINHOLE",
                    "camera_params": _camera_params(cameras["right"]),
                },
            ]
        }
    ]


def _image_names(reader: SegmentReader) -> tuple[list[str], list[str]]:
    frames = reader.frames
    frame_ids = frames["frame_id"].astype(int)
    left = [
        f"left_{frame_id:06d}{Path(str(frames['left_image_path'][index])).suffix.lower()}"
        for index, frame_id in enumerate(frame_ids)
    ]
    right = [
        f"right_{frame_id:06d}{Path(str(frames['right_image_path'][index])).suffix.lower()}"
        for index, frame_id in enumerate(frame_ids)
    ]
    return left, right


def _image_quality(reader: SegmentReader) -> tuple[np.ndarray, np.ndarray]:
    frames = reader.frames
    blur = np.empty(len(frames["frame_id"]), dtype=np.float64)
    exposure = np.empty_like(blur)
    for index, relative in enumerate(frames["left_image_path"]):
        image = cv2.imread(str(reader.root / str(relative)), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"cannot read image for quality audit: {relative}")
        blur[index] = float(cv2.Laplacian(image, cv2.CV_64F).var())
        clipped = np.mean((image <= 5) | (image >= 250))
        exposure[index] = float(1.0 - clipped)
    return blur, exposure


def _turn_mask(
    timestamps_s: np.ndarray, rotations_world_from_camera: np.ndarray, threshold: float
) -> np.ndarray:
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("turn_rate_deg_s must be finite and non-negative")
    result = np.zeros(len(timestamps_s), dtype=bool)
    if threshold == 0:
        return result
    relative = np.einsum(
        "nij,njk->nik",
        np.swapaxes(rotations_world_from_camera[:-1], 1, 2),
        rotations_world_from_camera[1:],
    )
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1, 1)
    rate = np.degrees(np.arccos(cosine)) / np.diff(timestamps_s)
    turning = rate >= threshold
    result[:-1] |= turning
    result[1:] |= turning
    return result


def _selection_json(
    selection: KeyframeSelection, config: KeyframeConfig
) -> dict[str, Any]:
    config_record = asdict(config)
    # JSON has no portable infinity literal. Null means that the corresponding
    # optional quality bound is disabled; finite experiment thresholds remain
    # exact numbers.
    for name, value in tuple(config_record.items()):
        if isinstance(value, float) and not np.isfinite(value):
            config_record[name] = None
    return {
        "schema_version": 1,
        "policy": "metric_adaptive_v1",
        "config": config_record,
        "frame_ids": selection.indices.astype(int).tolist(),
        "solve_only": True,
        "all_frames_registered_after_solve": True,
        "records": [
            {
                "frame_id": item.frame_index,
                "timestamp_s": item.timestamp_s,
                "reasons": list(item.reasons),
                "quality": asdict(item.quality),
                "revisit_source_frame_id": item.revisit_source_index,
            }
            for item in selection.keyframes
        ],
    }


def plan_frontend(
    segment: str | Path,
    *,
    keyframe_config: KeyframeConfig = KeyframeConfig(),
    pair_config: PairGraphConfig = PairGraphConfig(),
    compute_image_quality: bool = True,
    turn_rate_deg_s: float = 0.75,
) -> FrontendPlan:
    """Plan keyframes and matches while retaining every frame for registration."""
    reader = SegmentReader(segment).validate()
    capabilities = reader.meta["capabilities"]
    if not capabilities["stereo"]:
        raise ValueError("the current COLMAP frontend requires a stereo segment")
    if not capabilities["images_rectified"]:
        raise ValueError("the current calibrated rig frontend requires rectified images")

    frames = reader.frames
    required = {"initial_viewmat", "initial_camera_center_m", "pose_valid"}
    if not required <= set(frames) or not frames["pose_valid"].astype(bool).all():
        raise ValueError("frontend planning requires a valid initial pose for every frame")
    viewmats = np.asarray(frames["initial_viewmat"], dtype=np.float64)
    camera_to_world = np.linalg.inv(viewmats)
    positions = np.asarray(frames["initial_camera_center_m"], dtype=np.float64)
    rotations = camera_to_world[:, :3, :3]
    directions = rotations[:, :, 2]
    timestamps_s = (
        frames["timestamp_ns"].astype(np.float64)
        - float(frames["timestamp_ns"][0])
    ) * 1e-9

    gnss = reader.observations("gnss")
    assert gnss is not None
    covariance = gnss["covariance_enu_m2"]
    carrier = gnss["carrier_status"].astype(np.int64)
    fix = gnss["fix_status"].astype(np.int64)
    state = np.where(carrier >= 0, carrier, fix)
    blur, exposure = (
        _image_quality(reader)
        if compute_image_quality
        else (None, None)
    )
    turn = _turn_mask(timestamps_s, rotations, turn_rate_deg_s)
    selection = select_keyframes(
        timestamps_s,
        positions,
        rotations,
        config=keyframe_config,
        blur_score=blur,
        exposure_quality=exposure,
        stereo_sync_residual_s=(
            frames["stereo_sync_residual_ns"].astype(np.float64) * 1e-9
        ),
        rtk_covariance_m2=covariance,
        rtk_status=state,
        turn_region=turn,
    )
    left_names, right_names = _image_names(reader)
    graph = build_pair_graph(
        timestamps_s,
        positions,
        directions,
        left_names,
        right_names,
        selection.indices,
        config=pair_config,
    )
    if graph.unattached_frame_indices.size:
        raise RuntimeError(
            "pair graph cannot register all non-keyframes; relax registration "
            f"bounds (first unattached: {graph.unattached_frame_indices[:8].tolist()})"
        )
    pairs = tuple((edge.image_a, edge.image_b) for edge in graph.edges)
    solve_frame_pairs = {
        (edge.frame_a, edge.frame_b)
        for edge in graph.edges_for_scope("solve")
        if edge.frame_a != edge.frame_b
    }
    solve_degree = np.zeros(len(timestamps_s), dtype=np.int64)
    for first, second in solve_frame_pairs:
        solve_degree[first] += 1
        solve_degree[second] += 1
    quality = {
        "schema_version": 1,
        "n_frames": len(timestamps_s),
        "n_keyframes": len(selection.indices),
        "keyframe_fraction": len(selection.indices) / len(timestamps_s),
        "n_pairs": len(pairs),
        "n_solve_pairs": len(graph.edges_for_scope("solve")),
        "n_registration_pairs": len(graph.edges_for_scope("register")),
        "mandatory_stereo_pairs": sum(edge.mandatory for edge in graph.edges),
        "solve_frame_graph_connected": True,
        "solve_frame_graph_edges": len(solve_frame_pairs),
        "solve_frame_graph_max_degree": int(solve_degree.max(initial=0)),
        "all_frames_have_registration_path": True,
        "all_frames_retained_for_gs": True,
        "turn_frame_count": int(turn.sum()),
        "blur_score": (
            None
            if blur is None
            else {
                "minimum": float(blur.min()),
                "median": float(np.median(blur)),
            }
        ),
        "exposure_quality": (
            None
            if exposure is None
            else {
                "minimum": float(exposure.min()),
                "median": float(np.median(exposure)),
            }
        ),
    }
    return FrontendPlan(
        rig_config=make_rig_config(reader.calibration),
        keyframes=_selection_json(selection, keyframe_config),
        pairs=pairs,
        quality=quality,
        selection=selection,
        pair_graph=graph,
    )
