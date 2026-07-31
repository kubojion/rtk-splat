"""RTK-Splat pipeline CLI.

Stages (each writes its artifact; each re-runs independently):
  clockcheck  cross-machine clock offset from a topic recorded by two bags
  select      find a straight in-motion time window from the pose track
  extract     save stereo jpegs + per-frame camera poses for the window
  depth       stereo depth per pair (SGBM)
  stereo-prepare  build a calibrated, zero-copy COLMAP rig workspace
  stereo-solve    run the long stereo visual bundle adjustment
  stereo-export   robustly georegister and publish the refined pose sidecar
  global-prepare  clone a completed stereo front end into an isolated backend
  global-solve    run COLMAP's integrated Global Mapper with safety monitoring
  global-export   fixed-scale georegistration and strict publication gates
  integrity-audit read-only bounded RTK/stereo TF and clock calibration audit
  cloud       fuse depth into a voxel-downsampled world-frame init cloud
  train       optimize the Gaussian tile + eval on held-out frames

Usage: python -m rtk_splat.cli <stage> --config <config.yaml>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from .bagio import (build_typestore, read_camera_calibration, read_camera_info,
                    read_fix_with_reception, read_imu, read_stereo_frames)
from .cloud import backproject, to_world, voxel_downsample
from .configio import load_config
from .manifest import load_or_create_manifest
from .depth import depth_from_pair, make_sgbm
from .pose_artifacts import (cloud_path, load_pose_artifact,
                             pose_artifact_name, pose_fingerprint)
from .pose_sources import make_pose_source
from .poses import smooth_yaw, tilt_deviations


def _seg_dir(cfg) -> Path:
    return cfg.paths.workdir / "segment"


def cmd_clockcheck(cfg):
    """Offset between the machines that recorded the bags.

    C0: the shared topic (recorded by both machines) is matched message-by-
    message via identical header stamps; the reception-time difference is
    (network latency + clock offset). C1 (validation): RTK yaw rate vs the
    IMU gyro projected to the world vertical, cross-correlated over lags.
    """
    typestore, source = make_pose_source(cfg)
    topic = cfg.pose.clockcheck_topic

    c0 = None
    if len(cfg.paths.bags) >= 2:
        a = read_fix_with_reception(cfg.paths.bags[0], topic, typestore)
        b = read_fix_with_reception(cfg.paths.bags[1], topic, typestore)
        map_b = dict(zip(b[:, 0].tolist(), b[:, 1].tolist()))
        deltas = np.array([a_recv - map_b[h] for h, a_recv in
                           zip(a[:, 0].tolist(), a[:, 1].tolist()) if h in map_b])
        if len(deltas) < 100:
            raise RuntimeError(f"only {len(deltas)} matched messages on {topic}")
        d_ms = deltas / 1e6
        c0 = {"n": int(len(deltas)), "median_ms": float(np.median(d_ms)),
              "iqr_ms": float(np.percentile(d_ms, 75) - np.percentile(d_ms, 25))}
        print(f"C0 shared-topic '{topic}': {c0['n']} matched; reception delta "
              f"bag0-bag1 median {c0['median_ms']:+.2f} ms (upper bound on "
              f"offset: includes one-way network latency), IQR {c0['iqr_ms']:.1f} ms")
    else:
        print("C0 skipped (single bag); C1 only")

    # C1: RTK yaw rate (robot clock) vs projected gyro (camera clock)
    tr = source.track
    yaw_s = smooth_yaw(tr, cfg.pose.yaw_smooth_window)
    t0, t1 = tr.relpos_t[0], tr.relpos_t[-1]
    imu_t, _, gyro = read_imu(cfg.paths.bags, cfg.topics.imu, typestore, t0, t1)
    pitch = np.radians(cfg.pose.pitch_down_deg)
    wz = -np.sin(pitch) * gyro[:, 0] + np.cos(pitch) * gyro[:, 2]
    grid = np.arange(max(t0, imu_t[0]), min(t1, imu_t[-1]), 0.05)
    # derivative belongs at interval MIDPOINTS (right-edge assignment would
    # bias the estimated lag by half a sample period, 100 ms at 5 Hz)
    t_mid = 0.5 * (tr.relpos_t[1:] + tr.relpos_t[:-1])
    a_sig = np.interp(grid, t_mid, np.diff(yaw_s) / np.diff(tr.relpos_t))
    b_sig = np.interp(grid, imu_t, wz)
    a_sig -= a_sig.mean()
    b_sig -= b_sig.mean()
    lags_s, corrs = [], []
    for lag in np.arange(-2.0, 2.0, 0.01):
        shifted = np.interp(grid, grid + lag, b_sig)
        c = float(np.dot(a_sig, shifted)
                  / (np.linalg.norm(a_sig) * np.linalg.norm(shifted) + 1e-12))
        lags_s.append(lag)
        corrs.append(c)
    k = int(np.argmax(np.abs(corrs)))
    print(f"C1 gyro-vs-RTK yaw-rate: best lag {lags_s[k]*1000:+.0f} ms, "
          f"corr {corrs[k]:+.2f}"
          + ("  [WEAK -- treat as inconclusive]" if abs(corrs[k]) < 0.5 else ""))
    out = {"topic": topic, "c0": c0,
           "gyro_lag_ms": float(lags_s[k] * 1000),
           "gyro_corr": float(corrs[k])}
    cfg.paths.workdir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.workdir / "clockcheck.json").write_text(json.dumps(out, indent=2))
    print(f"-> {cfg.paths.workdir/'clockcheck.json'}; if |median| > ~3 ms, set "
          f"pose.time_offset_s accordingly")


def cmd_select(cfg):
    """Find the first straight, quality-gated, in-motion window -- or take an
    explicit `segment.window_s: [t0_rel, t1_rel]` verbatim (e.g. a headland
    turn, which the straightness search rejects by design). The explicit path
    still enforces the RTK carrier gate; heading stats are recorded as-is."""
    _, source = make_pose_source(cfg)
    track = source.track
    s = cfg.segment
    yaw_s = smooth_yaw(track, cfg.pose.yaw_smooth_window)
    t_rel = track.relpos_t - track.fix_t[0]

    fix_rel = track.fix_t - track.fix_t[0]
    speed = np.linalg.norm(np.diff(track.enu_xyz[:, :2], axis=0), axis=1) \
        / np.diff(track.fix_t)

    explicit = getattr(s, "window_s", None)
    if explicit:
        start, t_end = (float(v) for v in explicit)
        win = t_end - start
        mask = (t_rel >= start) & (t_rel <= t_end)
        if mask.sum() < 10:
            raise RuntimeError(f"explicit window {start:.0f}-{t_end:.0f}s "
                               "contains almost no heading samples")
        carr_min = int(track.relpos_carr[mask].min())
        if carr_min < cfg.pose.min_carr_soln:
            n_bad = int((track.relpos_carr[mask] < cfg.pose.min_carr_soln).sum())
            raise RuntimeError(
                f"explicit window has {n_bad} epochs below carr_soln "
                f"{cfg.pose.min_carr_soln} (min {carr_min}); pick a clean window")
        vmask = (fix_rel[:-1] >= start) & (fix_rel[:-1] <= t_end)
        std = float(np.degrees(np.std(yaw_s[mask])))
        v_mean = float(np.mean(speed[vmask]))
    else:
        win = float(s.length_s)
        v_lo, v_hi = (float(v) for v in s.speed_range_ms)
        best = None
        for start in np.arange(float(s.search_t_start_s), t_rel[-1] - win, 10.0):
            mask = (t_rel >= start) & (t_rel <= start + win)
            if mask.sum() < 10:
                continue
            if track.relpos_carr[mask].min() < cfg.pose.min_carr_soln:
                continue
            vmask = (fix_rel[:-1] >= start) & (fix_rel[:-1] <= start + win)
            v_mean = float(np.mean(speed[vmask]))
            if not (v_lo <= v_mean <= v_hi):
                continue  # parked or turning robot, not straight-line driving
            std = np.degrees(np.std(yaw_s[mask]))
            if std <= s.heading_std_max_deg:
                best = (start, std, v_mean)
                break
        if best is None:
            raise RuntimeError("no straight quality-gated window found; loosen "
                               "segment.heading_std_max_deg / speed_range_ms")
        start, std, v_mean = best
    t0 = track.fix_t[0] + start
    seg = _seg_dir(cfg)
    seg.mkdir(parents=True, exist_ok=True)
    window = {"t0": float(t0), "t1": float(t0 + win),
              "heading_std_deg": float(std), "mean_speed_ms": float(v_mean),
              "t0_rel_s": float(start)}
    (seg / "window.json").write_text(json.dumps(window, indent=2))
    print(f"selected window: t0_rel={start:.0f}s len={win:.0f}s "
          f"heading_std={std:.2f} deg speed={v_mean:.3f} m/s "
          f"-> {seg/'window.json'}")


def cmd_extract(cfg):
    """Save stereo jpegs and per-frame camera poses for the window.

    Time bases: image and IMU stamps come from the camera machine; the pose
    track may come from another machine. pose.time_offset_s (see clockcheck)
    maps image stamps into the pose track's time base.
    """
    typestore, source = make_pose_source(cfg)
    seg = _seg_dir(cfg)
    window = json.loads((seg / "window.json").read_text())
    intr = read_camera_info(cfg.paths.bags, cfg.topics, typestore)
    right_intr = None
    if hasattr(cfg.topics, "right_info"):
        right_intr = read_camera_calibration(
            cfg.paths.bags, cfg.topics.right_info, typestore
        )
    print(f"camera_info: {intr}")

    (seg / "images").mkdir(parents=True, exist_ok=True)
    off = float(cfg.pose.time_offset_s)
    frames = list(read_stereo_frames(cfg.paths.bags, cfg.topics, typestore,
                                     window["t0"] - off, window["t1"] - off,
                                     cfg.segment.frame_stride))
    cam_stamps = [fr.t for fr in frames]          # camera-machine clock
    pose_stamps = [t + off for t in cam_stamps]   # pose-track clock

    tilts, tilt_stats = None, None
    if cfg.pose.use_imu_tilt:
        imu_t, imu_q, _ = read_imu(cfg.paths.bags, cfg.topics.imu, typestore,
                                   cam_stamps[0] - 3, cam_stamps[-1] + 3)
        tilts, tilt_stats = tilt_deviations(imu_t, imu_q, cam_stamps,
                                            cfg.pose.imu_lp_window_s)
        print(f"imu tilt: {tilt_stats}")

    posed = source.pose_frames(pose_stamps, tilts)

    viewmats, centers, kept_cam_stamps, kept_pose_stamps, stereo_dt = \
        [], [], [], [], []
    n_kept = 0
    for fr, pose_stamp, pf in zip(frames, pose_stamps, posed):
        if pf is None:
            continue
        i = n_kept
        (seg / "images" / f"left_{i:06d}.jpg").write_bytes(fr.left_jpeg)
        (seg / "images" / f"right_{i:06d}.jpg").write_bytes(fr.right_jpeg)
        viewmats.append(pf.viewmat)
        centers.append(pf.cam_center)
        kept_cam_stamps.append(fr.t)
        kept_pose_stamps.append(pose_stamp)
        stereo_dt.append(fr.t_right - fr.t)
        n_kept += 1
    if n_kept < 50:
        raise RuntimeError(f"only {n_kept} posed frames; expected hundreds")

    np.save(seg / "viewmats.npy", np.stack(viewmats).astype(np.float32))
    np.save(seg / "cam_centers.npy", np.stack(centers).astype(np.float32))
    np.save(seg / "camera_stamps.npy", np.asarray(kept_cam_stamps))
    np.save(seg / "pose_stamps.npy", np.asarray(kept_pose_stamps))
    np.save(seg / "stereo_dt_s.npy", np.asarray(stereo_dt))
    tr = source.track
    quality = None
    if hasattr(tr, "fix_status") and hasattr(tr, "fix_cov_max"):
        m = ((tr.fix_t >= window["t0"]) & (tr.fix_t <= window["t1"])
             & np.isfinite(tr.fix_cov_max))
    else:
        m = np.zeros(len(tr.fix_t), dtype=bool)
    if m.any():
        quality = {"fix_status_min": int(tr.fix_status[m].min()),
                   "fix_cov_max_m2": float(tr.fix_cov_max[m].max()),
                   "fix_cov_median_m2": float(np.median(tr.fix_cov_max[m]))}
    meta = {"intrinsics": intr, "world_origin": source.origin,
            "crs": getattr(getattr(source, "enu", None), "crs", lambda: None)(),
            "gnss_quality": quality,
            "window": window, "n_frames": n_kept,
            "pose_source": cfg.pose.source,
            "time_offset_s": off,
            "imu_tilt": tilt_stats,
            "stereo_sync_abs_max_ms":
                float(np.max(np.abs(stereo_dt)) * 1000),
            "dropped_unposeable": len(frames) - n_kept,
            "path_length_m": float(np.linalg.norm(
                np.diff(np.stack(centers), axis=0), axis=1).sum())}
    if right_intr is not None:
        meta["stereo_calibration"] = {"left": intr, "right": right_intr}
    (seg / "segment_meta.json").write_text(json.dumps(meta, indent=2))
    (seg / "manifest.json").unlink(missing_ok=True)  # split follows re-extract
    manifest = load_or_create_manifest(seg, cfg.train.holdout_every)
    print(f"extracted {n_kept} frames "
          f"({meta['dropped_unposeable']} dropped), "
          f"path {meta['path_length_m']:.1f} m; split "
          f"{len(manifest['train'])}/{len(manifest['val'])}/"
          f"{len(manifest['test'])} train/val/test; gnss {quality}")


def cmd_depth(cfg):
    seg = _seg_dir(cfg)
    backend = str(getattr(cfg.depth, "backend", "sgbm")).lower()
    if backend != "sgbm":
        raise ValueError(
            f"depth.backend={backend!r} is not implemented; use 'sgbm' or "
            "provide depth artifacts through a dataset adapter"
        )
    meta = json.loads((seg / "segment_meta.json").read_text())
    intr = meta["intrinsics"]
    (seg / "depth").mkdir(exist_ok=True)
    matcher = make_sgbm(cfg)
    n = meta["n_frames"]
    for i in range(n):
        left = cv2.imread(str(seg / "images" / f"left_{i:06d}.jpg"))
        right = cv2.imread(str(seg / "images" / f"right_{i:06d}.jpg"))
        d, valid = depth_from_pair(matcher, left, right, intr["fx"],
                                   cfg.pose.baseline_m, cfg)
        np.savez_compressed(seg / "depth" / f"{i:06d}.npz",
                            depth=d.astype(np.float16), valid=valid)
        if (i + 1) % 50 == 0:
            print(f"depth {i+1}/{n} (valid px {valid.mean()*100:.0f}%)",
                  flush=True)
    print("depth done")


def cmd_cloud(cfg):
    seg = _seg_dir(cfg)
    meta = json.loads((seg / "segment_meta.json").read_text())
    intr = meta["intrinsics"]
    viewmats, _ = load_pose_artifact(seg, cfg)
    # LEAKAGE GUARD: initialization is built from TRAINING frames only --
    # val/test images and their depth must never touch the map.
    manifest = load_or_create_manifest(seg, cfg.train.holdout_every)
    pts_all, cols_all = [], []
    for i in manifest["train"]:
        dz = np.load(seg / "depth" / f"{i:06d}.npz")
        left = cv2.imread(str(seg / "images" / f"left_{i:06d}.jpg"))
        rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        pts, cols = backproject(dz["depth"].astype(np.float32), dz["valid"],
                                rgb, intr, cfg.cloud.pixel_stride)
        pts_all.append(to_world(pts, viewmats[i]))
        cols_all.append(cols)
    pts = np.concatenate(pts_all).astype(np.float32)
    cols = np.concatenate(cols_all)
    print(f"fused {len(pts):,} raw points")
    pts, cols = voxel_downsample(pts, cols, cfg.cloud.voxel_m,
                                 cfg.cloud.max_points)
    out = cloud_path(seg, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(
            temporary, xyz=pts, rgb=cols,
            pose_fingerprint=np.asarray(pose_fingerprint(viewmats)))
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    print(f"init cloud [{pose_artifact_name(cfg)}]: {len(pts):,} pts, extent "
          f"{hi[0]-lo[0]:.1f} x {hi[1]-lo[1]:.1f} x "
          f"{hi[2]-lo[2]:.1f} m -> {out}")


def cmd_ingest_agrigs(cfg):
    """Build a segment from the AgriGS-SLAM demo dataset (see ingest_agrigs)."""
    from .ingest_agrigs import ingest, verify
    a = cfg.agrigs
    seg = _seg_dir(cfg)
    ingest(Path(a.dataset_dir).expanduser(), a.camera,
           [float(v) for v in a.intrinsic], [float(v) for v in a.distortion],
           seg, float(cfg.depth.min_z_m), float(cfg.depth.max_z_m))
    verify(seg, cfg.paths.workdir / "pose_convention_check.jpg")


def cmd_evalonly(cfg, split="val", max_frames=0):
    """Re-score a finished run's checkpoint with the CURRENT metrics, without
    retraining. Lets old and new checkpoints be compared under identical
    metrics (e.g. after adding LPIPS). `split` picks the manifest ids to score
    (train-view scores are the fit ceiling AgriGS-style papers report);
    `max_frames` > 0 subsamples evenly. Writes metrics_evalonly[_split].json."""
    import torch
    from .train import evaluate

    run_dir = cfg.paths.workdir / "runs" / cfg.train.run_name
    seg = _seg_dir(cfg)
    meta = json.loads((seg / "segment_meta.json").read_text())
    intr = meta["intrinsics"]
    device = "cuda"
    raw = torch.load(run_dir / "params.pt", map_location=device,
                     weights_only=True)
    params = {k: v for k, v in raw.items() if k != "pose_deltas_raw"}
    viewmats_np, _ = load_pose_artifact(seg, cfg)
    viewmats = torch.tensor(viewmats_np,
                            dtype=torch.float32, device=device)
    c2ws = torch.linalg.inv(viewmats)
    k_mat = torch.tensor([[intr["fx"], 0, intr["cx"]],
                          [0, intr["fy"], intr["cy"]],
                          [0, 0, 1]], dtype=torch.float32, device=device)
    eval_ids = load_or_create_manifest(seg, cfg.train.holdout_every)[split]
    if not eval_ids:
        raise RuntimeError(f"manifest split '{split}' is empty")
    if max_frames and len(eval_ids) > max_frames:
        idx = np.linspace(0, len(eval_ids) - 1, max_frames).astype(int)
        eval_ids = [eval_ids[i] for i in idx]
    # sh degree from the checkpoint, not the config (older runs differ)
    sh_deg = int(np.sqrt(params["shN"].shape[1] + 1)) - 1
    cfg.train.sh_degree = sh_deg
    m = evaluate(params, c2ws, k_mat, intr["width"], intr["height"], seg,
                 run_dir, eval_ids, cfg, device, step=0, final=False)
    suffix = "" if split == "val" else f"_{split}"
    (run_dir / f"metrics_evalonly{suffix}.json").write_text(json.dumps(m, indent=2))
    print(f"{cfg.train.run_name} [{split}, n={len(eval_ids)}]: " + "  ".join(
        f"{k} {v:.3f}" for k, v in m.items() if isinstance(v, float)))


def cmd_diagnose(cfg):
    """Brightness-vs-quality plot from a finished run: convicts or clears the
    changing-light hypothesis (requires the final eval's per_frame data)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = cfg.paths.workdir / "runs" / cfg.train.run_name
    history = json.loads((run_dir / "metrics.json").read_text())
    per = history[-1].get("per_frame")
    if not per:
        raise RuntimeError(f"{run_dir}/metrics.json has no per_frame data "
                           "(only final evals of runs trained after the "
                           "diagnostic was added record it)")
    b = np.array(per["brightness"])
    q = np.array(per["psnr_masked"])
    r = float(np.corrcoef(b, q)[0, 1])
    fig, ax1 = plt.subplots(figsize=(10, 4))
    x = np.array(per["eval_ids"])
    ax1.plot(x, b, "o-", color="tab:orange", label="GT brightness (masked mean)")
    ax1.set_xlabel("frame index (along the drive)")
    ax1.set_ylabel("brightness", color="tab:orange")
    ax2 = ax1.twinx()
    ax2.plot(x, q, "s-", color="tab:blue", label="masked PSNR")
    ax2.set_ylabel("masked PSNR (dB)", color="tab:blue")
    ax1.set_title(f"{cfg.train.run_name}: brightness vs quality "
                  f"(corr {r:+.2f})")
    fig.tight_layout()
    out = run_dir / "brightness_diagnostic.png"
    fig.savefig(out, dpi=140)
    print(f"correlation(brightness, psnr_masked) = {r:+.2f} -> {out}")
    print("|r| >~ 0.5: lighting change is a major quality factor; "
          "<~ 0.3: clouds largely cleared")


def cmd_train(cfg):
    from .train import train_tile  # torch import deferred to this stage
    run_dir = cfg.paths.workdir / "runs" / cfg.train.run_name
    final = train_tile(_seg_dir(cfg), run_dir, cfg)
    print(f"final: {final}")


def cmd_stereo_prepare(cfg):
    from .colmap_stereo import prepare
    prepare(_seg_dir(cfg), cfg, build_typestore(None))


def cmd_stereo_solve(cfg):
    from .colmap_stereo import solve
    solve(_seg_dir(cfg), cfg)


def cmd_stereo_export(cfg):
    from .colmap_stereo import export
    export(_seg_dir(cfg), cfg)


def cmd_global_prepare(cfg):
    from .colmap_global import prepare
    prepare(_seg_dir(cfg), cfg)


def cmd_global_solve(cfg):
    from .colmap_global import solve
    solve(_seg_dir(cfg), cfg)


def cmd_global_export(cfg):
    from .colmap_global import export
    export(_seg_dir(cfg), cfg)


def cmd_integrity_audit(cfg):
    """Explicit diagnostic stage; never part of ``all`` or normal training."""
    from .calibration_sidecar import run_integrity_audit
    artifact, result = run_integrity_audit(_seg_dir(cfg), cfg)
    accepted = result["final_retained_prior_safe"]["calibration_accepted"]
    print(f"metric-integrity audit -> {artifact}")
    print(f"calibration accepted: {accepted}")
    for name, parameter in result["parameters"].items():
        verdict = "TRUSTED" if parameter["trusted"] else "RETAINED PRIOR"
        print(
            f"  {name}: {verdict}; candidate "
            f"{parameter['candidate_correction']:+.8g}, retained "
            f"{parameter['retained_correction']:+.8g}")


def main():
    stages = {"clockcheck": cmd_clockcheck, "select": cmd_select,
              "extract": cmd_extract, "depth": cmd_depth,
              "cloud": cmd_cloud, "train": cmd_train,
              "evalonly": cmd_evalonly, "diagnose": cmd_diagnose,
              "stereo-prepare": cmd_stereo_prepare,
              "stereo-solve": cmd_stereo_solve,
              "stereo-export": cmd_stereo_export,
              "global-prepare": cmd_global_prepare,
              "global-solve": cmd_global_solve,
              "global-export": cmd_global_export,
              "integrity-audit": cmd_integrity_audit,
              "ingest-agrigs": cmd_ingest_agrigs}
    ap = argparse.ArgumentParser(prog="rtk_splat")
    ap.add_argument("stage", choices=list(stages) + ["all"])
    ap.add_argument("--config", required=True,
                    help="YAML configuration (no machine-specific default)")
    ap.add_argument("--run", default=None,
                    help="override train.run_name (evalonly/diagnose)")
    ap.add_argument("--pose-artifact", default=None,
                    help="override pose.artifact (e.g. colmap_stereo)")
    ap.add_argument("--split", default="val", choices=["val", "train", "test"],
                    help="manifest split to score (evalonly only)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="evenly subsample the split to this many frames "
                         "(evalonly only; 0 = all)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.run:
        cfg.train.run_name = args.run
    if args.pose_artifact:
        cfg.pose.artifact = args.pose_artifact
    cfg.paths.workdir.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        for name in ("select", "extract", "depth", "cloud", "train"):
            print(f"=== {name} ===", flush=True)
            stages[name](cfg)
    elif args.stage == "evalonly":
        cmd_evalonly(cfg, args.split, args.max_frames)
    else:
        stages[args.stage](cfg)


if __name__ == "__main__":
    sys.exit(main())
