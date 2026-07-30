"""COLMAP stereo bundle-adjustment sidecar for an extracted RTK-Splat segment.

The sidecar is deliberately explicit:

``stereo-prepare`` creates a symlink-only COLMAP image tree and calibrated
rig description, ``stereo-solve`` runs the long external reconstruction, and
``stereo-export`` aligns its visual poses to the immutable RTK ENU frame.
Nothing here overwrites ``segment/viewmats.npy``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .bagio import read_camera_calibration
from .pose_artifacts import (pose_artifact_dir, pose_artifact_name,
                             pose_fingerprint)


def _sidecar_dir(seg_dir: Path, cfg) -> Path:
    if pose_artifact_name(cfg) == "rtk":
        raise ValueError("stereo stages require pose.artifact to name a sidecar")
    return pose_artifact_dir(seg_dir, cfg)


def _colmap_intrinsics(calib: dict) -> list[float]:
    # ROS/OpenCV locates the first pixel centre at (0, 0); COLMAP's convention
    # locates it at (0.5, 0.5).
    return [float(calib["fx"]), float(calib["fy"]),
            float(calib["cx"]) + 0.5, float(calib["cy"]) + 0.5]


def make_rig_config(left: dict, right: dict, baseline_m: float) -> list[dict]:
    return [{
        "cameras": [
            {
                "image_prefix": "zed/left/",
                "ref_sensor": True,
                "camera_model_name": "PINHOLE",
                "camera_params": _colmap_intrinsics(left),
            },
            {
                "image_prefix": "zed/right/",
                "cam_from_rig_rotation": [1.0, 0.0, 0.0, 0.0],
                # cam_from_rig: the right centre is +B on optical x, hence
                # world/rig-to-camera translation is -B.
                "cam_from_rig_translation": [-float(baseline_m), 0.0, 0.0],
                "camera_model_name": "PINHOLE",
                "camera_params": _colmap_intrinsics(right),
            },
        ]
    }]


def _validate_rectified_stereo(left: dict, right: dict,
                               configured_baseline: float) -> float:
    if (left["width"], left["height"]) != (right["width"], right["height"]):
        raise ValueError("left/right CameraInfo image sizes differ")
    for side, calib in (("left", left), ("right", right)):
        if not np.allclose(np.asarray(calib["r"]), np.eye(3), atol=1e-5):
            raise ValueError(f"{side} stream is not rectified (R != identity)")
        if np.max(np.abs(np.asarray(calib["d"]))) > 1e-7:
            raise ValueError(f"{side} rectified stream still reports distortion")
        p = np.asarray(calib["p"])
        if not np.allclose(
                [p[0, 0], p[1, 1], p[0, 2], p[1, 2]],
                [calib["fx"], calib["fy"], calib["cx"], calib["cy"]],
                atol=1e-4):
            raise ValueError(f"{side} K and rectified P intrinsics disagree")
    p_right = np.asarray(right["p"])
    measured = -float(p_right[0, 3]) / float(p_right[0, 0])
    if measured <= 0:
        raise ValueError("right P matrix has the wrong stereo-baseline sign")
    if abs(measured - configured_baseline) > 5e-4:
        raise ValueError(
            f"configured baseline {configured_baseline:.6f} m disagrees with "
            f"CameraInfo {measured:.6f} m")
    return measured


def _ensure_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise FileExistsError(f"{link} points to the wrong source")
        return
    if link.exists():
        raise FileExistsError(f"refusing to replace existing {link}")
    link.symlink_to(target.resolve())


def prepare(seg_dir: Path, cfg, typestore) -> Path:
    """Create a calibrated, zero-copy COLMAP workspace."""
    sidecar = _sidecar_dir(seg_dir, cfg)
    work = sidecar / "colmap"
    left_dir = work / "images" / "zed" / "left"
    right_dir = work / "images" / "zed" / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((seg_dir / "segment_meta.json").read_text())
    n_frames = int(meta["n_frames"])
    if not hasattr(cfg.topics, "right_info"):
        raise ValueError("topics.right_info is required by stereo-prepare")
    left = read_camera_calibration(cfg.paths.bags, cfg.topics.left_info,
                                   typestore)
    right = read_camera_calibration(cfg.paths.bags, cfg.topics.right_info,
                                    typestore)
    baseline = _validate_rectified_stereo(
        left, right, float(cfg.pose.baseline_m))

    frames = []
    for i in range(n_frames):
        src_l = seg_dir / "images" / f"left_{i:06d}.jpg"
        src_r = seg_dir / "images" / f"right_{i:06d}.jpg"
        if not src_l.is_file() or not src_r.is_file():
            raise FileNotFoundError(f"missing extracted stereo pair {i}")
        name = f"{i:06d}.jpg"
        _ensure_symlink(left_dir / name, src_l)
        _ensure_symlink(right_dir / name, src_r)
        frames.append({"frame_id": i, "left": f"zed/left/{name}",
                       "right": f"zed/right/{name}"})

    rig = make_rig_config(left, right, baseline)
    config = {
        "schema_version": 1,
        "pose_artifact": pose_artifact_name(cfg),
        "n_frames": n_frames,
        "coordinate_conventions": {
            "input_images": "rectified OpenCV",
            "colmap_poses": "cam_from_world",
            "output_world": "segment local ENU",
            "output_poses": "OpenCV world-to-camera",
            "principal_point_conversion": "ROS/OpenCV + (0.5, 0.5)",
        },
        "baseline_m_camera_info": baseline,
        "baseline_m_config": float(cfg.pose.baseline_m),
        "left_calibration": left,
        "right_calibration": right,
    }
    config_path = sidecar / "sidecar_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        if (work / "database.db").exists():
            raise RuntimeError(
                "calibration/config changed after COLMAP started; use a new "
                "pose.artifact name")
    (work / "rig_config.json").write_text(json.dumps(rig, indent=2))
    (sidecar / "frame_manifest.json").write_text(json.dumps(frames, indent=2))
    config_path.write_text(json.dumps(config, indent=2))
    np.save(sidecar / "rtk_camera_centers.npy",
            np.load(seg_dir / "cam_centers.npy").astype(np.float64))
    print(f"prepared {n_frames} stereo pairs at {work} (symlinks; no JPEG copies)")
    print(f"validated rectified CameraInfo baseline: {baseline:.9f} m")
    return work


def colmap_commands(work: Path, cfg, executable: str) -> list[tuple[str, list[str]]]:
    c = cfg.colmap
    database = work / "database.db"
    images = work / "images"
    sparse = work / "sparse_visual"
    return [
        ("feature_extractor", [
            executable, "feature_extractor",
            "--database_path", str(database),
            "--image_path", str(images),
            "--ImageReader.single_camera_per_folder", "1",
            "--FeatureExtraction.max_image_size", str(c.max_image_size),
            "--FeatureExtraction.num_threads", str(c.feature_num_threads),
            "--SiftExtraction.max_num_features", str(c.max_num_features),
            "--SiftExtraction.estimate_affine_shape", "1",
            "--SiftExtraction.domain_size_pooling", "1",
        ]),
        ("rig_configurator", [
            executable, "rig_configurator",
            "--database_path", str(database),
            "--rig_config_path", str(work / "rig_config.json"),
        ]),
        ("sequential_matcher", [
            executable, "sequential_matcher",
            "--database_path", str(database),
            "--SequentialMatching.overlap", str(c.sequential_overlap),
            "--SequentialMatching.quadratic_overlap", "1",
            "--SequentialMatching.expand_rig_images", "1",
            "--FeatureMatching.guided_matching", "1",
            "--FeatureMatching.rig_verification", "1",
        ]),
        ("mapper", [
            executable, "mapper",
            "--database_path", str(database),
            "--image_path", str(images),
            "--output_path", str(sparse),
            "--Mapper.ba_refine_sensor_from_rig", "0",
            "--Mapper.ba_refine_focal_length", "0",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "0",
            # CPU sparse BA is slower but avoids 8 GiB GPU-memory failures on
            # this 2,688-image model; CUDA still accelerates matching.
            "--Mapper.ba_use_gpu", "0",
        ]),
    ]


def solve(seg_dir: Path, cfg) -> None:
    """Run the long COLMAP reconstruction, with resumable step sentinels."""
    sidecar = _sidecar_dir(seg_dir, cfg)
    work = sidecar / "colmap"
    if not (work / "rig_config.json").exists():
        raise FileNotFoundError("run stereo-prepare before stereo-solve")
    executable = os.environ.get("COLMAP_BIN", str(cfg.colmap.executable))
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(
            f"COLMAP executable {executable!r} not found. Install COLMAP 4.1.1 "
            "or set COLMAP_BIN to its executable.")
    executable = resolved
    sparse = work / "sparse_visual"
    sparse.mkdir(exist_ok=True)
    commands = colmap_commands(work, cfg, executable)
    (sidecar / "commands.json").write_text(json.dumps(
        [{"step": name, "argv": argv} for name, argv in commands], indent=2))

    log_path = sidecar / "colmap.log"
    with log_path.open("a") as log:
        version = subprocess.run([executable, "-h"], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 check=False).stdout.splitlines()[:2]
        log.write("\nCOLMAP identity: " + " | ".join(version) + "\n")
        log.flush()
        for name, argv in commands:
            done = work / f".{name}.done"
            if done.exists():
                print(f"COLMAP {name}: already complete")
                continue
            print(f"COLMAP {name}: starting", flush=True)
            log.write("\n$ " + " ".join(argv) + "\n")
            log.flush()
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                    text=True, check=False)
            if result.returncode:
                raise RuntimeError(
                    f"COLMAP {name} failed with exit {result.returncode}; "
                    f"see {log_path}")
            done.write_text("ok\n")
            print(f"COLMAP {name}: complete", flush=True)

        text_root = work / "models_text"
        text_root.mkdir(exist_ok=True)
        models = sorted(p for p in sparse.iterdir()
                        if p.is_dir() and (p / "images.bin").exists())
        if not models:
            raise RuntimeError(f"COLMAP mapper produced no model in {sparse}")
        for model in models:
            out = text_root / model.name
            out.mkdir(exist_ok=True)
            argv = [executable, "model_converter",
                    "--input_path", str(model), "--output_path", str(out),
                    "--output_type", "TXT"]
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                    text=True, check=False)
            if result.returncode:
                raise RuntimeError(
                    f"model_converter failed for {model}; see {log_path}")
    print(f"COLMAP solve complete; text models are in {work/'models_text'}")


def qvec_to_rotmat(qvec) -> np.ndarray:
    """COLMAP scalar-first quaternion -> cam-from-world rotation."""
    q = np.asarray(qvec, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("invalid COLMAP quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def parse_colmap_images(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Parse images.txt by filename; observation lines are ignored."""
    result = {}
    lines = path.read_text().splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 10:
            continue
        # Metadata lines end in an image filename at field 10. Observation
        # lines are triples of x/y/point-id and cannot satisfy this path test.
        name = fields[9].replace("\\", "/")
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
            q = np.asarray([float(v) for v in fields[1:5]])
            t = np.asarray([float(v) for v in fields[5:8]])
        except ValueError as exc:
            raise ValueError(f"invalid COLMAP image record: {line}") from exc
        if name in result:
            raise ValueError(f"duplicate COLMAP image name {name}")
        result[name] = (qvec_to_rotmat(q), t)
    if not result:
        raise ValueError(f"no registered images parsed from {path}")
    return result


def umeyama(source: np.ndarray, target: np.ndarray) \
        -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares Sim(3), mapping target ~= scale * R * source + t."""
    x = np.asarray(source, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError("Sim(3) needs matching (N,3) arrays with N >= 3")
    mx, my = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mx, y - my
    if np.linalg.matrix_rank(xc, tol=1e-8) < 2:
        raise ValueError("camera trajectory is too close to a line for Sim(3)")
    cov = (yc.T @ xc) / len(x)
    u, singular, vt = np.linalg.svd(cov)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    variance = np.sum(xc * xc) / len(x)
    scale = float(np.sum(singular * sign) / variance)
    translation = my - scale * rotation @ mx
    return scale, rotation, translation


def robust_similarity(source: np.ndarray, target: np.ndarray,
                      max_error_m: float, iterations: int,
                      seed: int = 7) -> tuple[float, np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray]:
    """RANSAC + inlier refit for visual-centre to RTK-centre alignment."""
    if len(source) < 3:
        raise ValueError("at least three registered pose correspondences required")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(int(iterations)):
        ids = rng.choice(len(source), 3, replace=False)
        try:
            s, r, t = umeyama(source[ids], target[ids])
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = np.linalg.norm((s * (r @ source.T)).T + t - target, axis=1)
        inliers = errors <= max_error_m
        score = (int(inliers.sum()),
                 -float(np.median(errors[inliers])) if inliers.any() else -np.inf)
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None or best[1].sum() < 3:
        raise RuntimeError("RANSAC could not align COLMAP centres to RTK")
    inliers = best[1]
    for _ in range(3):
        s, r, t = umeyama(source[inliers], target[inliers])
        errors = np.linalg.norm((s * (r @ source.T)).T + t - target, axis=1)
        updated = errors <= max_error_m
        if np.array_equal(updated, inliers) or updated.sum() < 3:
            break
        inliers = updated
    s, r, t = umeyama(source[inliers], target[inliers])
    errors = np.linalg.norm((s * (r @ source.T)).T + t - target, axis=1)
    return s, r, t, inliers, errors


def aligned_viewmat(r_cw_visual: np.ndarray, center_visual: np.ndarray,
                    scale: float, r_align: np.ndarray,
                    translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply visual-world -> ENU Sim(3) to one COLMAP camera."""
    center_enu = scale * (r_align @ center_visual) + translation
    r_wc_enu = r_align @ r_cw_visual.T
    r_cw_enu = r_wc_enu.T
    viewmat = np.eye(4)
    viewmat[:3, :3] = r_cw_enu
    viewmat[:3, 3] = -r_cw_enu @ center_enu
    return viewmat, center_enu


def export(seg_dir: Path, cfg) -> dict:
    """Georegister the largest COLMAP model and publish the pose artifact."""
    sidecar = _sidecar_dir(seg_dir, cfg)
    models_root = sidecar / "colmap" / "models_text"
    candidates = []
    for model in sorted(models_root.iterdir()) if models_root.exists() else []:
        images_path = model / "images.txt"
        if images_path.exists():
            parsed = parse_colmap_images(images_path)
            left_count = sum(name.startswith("zed/left/") for name in parsed)
            candidates.append((left_count, model, parsed))
    if not candidates:
        raise FileNotFoundError("no converted COLMAP model; run stereo-solve first")
    _, model, images = max(candidates, key=lambda item: item[0])

    raw_viewmats = np.load(seg_dir / "viewmats.npy").astype(float)
    raw_centers = np.load(seg_dir / "cam_centers.npy").astype(float)
    n_frames = len(raw_viewmats)
    visual_centers, visual_rotations, frame_ids = [], [], []
    for i in range(n_frames):
        name = f"zed/left/{i:06d}.jpg"
        if name not in images:
            continue
        r_cw, t = images[name]
        visual_centers.append(-r_cw.T @ t)
        visual_rotations.append(r_cw)
        frame_ids.append(i)
    frame_ids = np.asarray(frame_ids, dtype=int)
    registered_fraction = len(frame_ids) / n_frames
    minimum = float(cfg.colmap.min_registered_fraction)
    if registered_fraction < minimum:
        raise RuntimeError(
            f"only {len(frame_ids)}/{n_frames} left frames registered "
            f"({registered_fraction:.1%}, require {minimum:.1%})")
    if len(frame_ids) != n_frames:
        raise RuntimeError(
            f"{n_frames-len(frame_ids)} left frames are unregistered. This "
            "first sidecar intentionally refuses to mix/interpolate RTK and "
            "visual poses; improve COLMAP registration and rerun.")

    visual_centers = np.asarray(visual_centers)
    s, r_align, translation, inliers, errors = robust_similarity(
        visual_centers, raw_centers[frame_ids],
        float(cfg.colmap.alignment_max_error_m),
        int(cfg.colmap.alignment_ransac_iterations))
    scale_lo, scale_hi = [float(v) for v in cfg.colmap.scale_range]
    if not scale_lo <= s <= scale_hi:
        raise RuntimeError(
            f"stereo reconstruction scale {s:.6f} is outside calibrated "
            f"range [{scale_lo}, {scale_hi}]")

    viewmats, centers = [], []
    for r_cw, center in zip(visual_rotations, visual_centers):
        vm, ce = aligned_viewmat(r_cw, center, s, r_align, translation)
        viewmats.append(vm)
        centers.append(ce)
    viewmats = np.asarray(viewmats, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    np.save(sidecar / "viewmats.npy", viewmats)
    np.save(sidecar / "cam_centers.npy", centers)
    np.save(sidecar / "frame_ids.npy", frame_ids)
    np.save(sidecar / "registered.npy", np.ones(n_frames, dtype=bool))
    stats = {
        "schema_version": 1,
        "source": "COLMAP calibrated stereo visual bundle adjustment",
        "selected_model": str(model),
        "n_frames": n_frames,
        "n_registered_left": len(frame_ids),
        "registered_fraction": registered_fraction,
        "alignment": {
            "method": "Sim(3) RANSAC to immutable RTK camera centres",
            "scale": s,
            "rotation_visual_world_to_enu": r_align.tolist(),
            "translation_enu_m": translation.tolist(),
            "threshold_m": float(cfg.colmap.alignment_max_error_m),
            "n_inliers": int(inliers.sum()),
            "median_error_m": float(np.median(errors)),
            "p95_error_m": float(np.percentile(errors, 95)),
            "max_error_m": float(errors.max()),
        },
        "pose_fingerprint": pose_fingerprint(viewmats),
        "coordinate_convention": "local ENU world; OpenCV world-to-camera",
        "raw_rtk_poses_modified": False,
    }
    (sidecar / "quality.json").write_text(json.dumps(stats, indent=2))
    print(f"exported {n_frames} georeferenced stereo-BA poses -> {sidecar}")
    print(f"alignment: scale {s:.6f}, RTK residual median "
          f"{stats['alignment']['median_error_m']*100:.1f} cm, p95 "
          f"{stats['alignment']['p95_error_m']*100:.1f} cm, "
          f"{int(inliers.sum())}/{n_frames} inliers")
    return stats
