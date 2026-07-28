"""Ingest the AgriGS-SLAM demo dataset into a standard rtk_splat segment.

Everything dataset-specific lives HERE; the core pipeline (train/evaluate/
export) consumes the produced segment unchanged.

Their layout (per split: train/ val/):
  zed_multi/cam_k/rgb/<sec-nsec>.jpg      1920x1200, mildly distorted
  zed_multi/cam_k/depth/<sec-nsec>.png    uint16 millimeters, 0 = invalid
  groundtruth_cam_k.csv                   timestamp,tx,ty,tz,qx,qy,qz,qw (ECEF)

Produced segment: undistorted left_{i:06d}.jpg + depth npz + viewmats.npy in
a local ENU frame (origin = first pose), manifest with train = training
traversal and val = the SEPARATE reverse-direction traversal (their novel-view
protocol; a genuinely independent evaluation pass).

Pose convention (verified empirically by the `verify` step): the CSV quaternion
is camera-to-ECEF with OpenCV optical axes (x right, y down, z forward).
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pymap3d
from scipy.spatial.transform import Rotation, Slerp

from .poses import viewmat_from


def _read_gt(csv_path: Path):
    """(t[s], ecef[N,3], Rotation[N]) from a groundtruth_cam_k.csv."""
    ts, xyz, quats = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            sec, nsec = row["timestamp"].split("-")
            ts.append(int(sec) + int(nsec) * 1e-9)
            xyz.append([float(row["tx"]), float(row["ty"]), float(row["tz"])])
            quats.append([float(row["qx"]), float(row["qy"]),
                          float(row["qz"]), float(row["qw"])])
    return np.array(ts), np.array(xyz), Rotation.from_quat(np.array(quats))


def _enu_rotation(lat0: float, lon0: float) -> np.ndarray:
    """ECEF-vector -> ENU-vector rotation at the origin."""
    lam, phi = np.radians(lon0), np.radians(lat0)
    sl, cl = np.sin(lam), np.cos(lam)
    sp, cp = np.sin(phi), np.cos(phi)
    return np.array([[-sl, cl, 0.0],
                     [-sp * cl, -sp * sl, cp],
                     [cp * cl, cp * sl, sp]])


def _stamp_of(name: str) -> float:
    sec, nsec = name.split(".")[0].split("-")
    return int(sec) + int(nsec) * 1e-9


def _interp_poses(t_gt, enu_xyz, rots_enu, stamps):
    """(centers[N,3], R_enu_cam[N,3,3]) lerp/slerp'd at image stamps."""
    slerp = Slerp(t_gt, rots_enu)
    centers, mats = [], []
    for t in stamps:
        t = min(max(t, t_gt[0]), t_gt[-1])
        centers.append([np.interp(t, t_gt, enu_xyz[:, k]) for k in range(3)])
        mats.append(slerp([t]).as_matrix()[0])
    return np.array(centers), np.array(mats)


def ingest(dataset_dir: Path, cam: str, intrinsic, distortion, out_seg: Path,
           min_z: float, max_z: float):
    """Build one segment: train split -> manifest.train, val -> manifest.val."""
    fx, fy, cx, cy = intrinsic
    k_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array(distortion)

    out_seg.mkdir(parents=True, exist_ok=True)
    (out_seg / "images").mkdir(exist_ok=True)
    (out_seg / "depth").mkdir(exist_ok=True)

    # ENU frame anchored at the first training pose
    t0_gt, ecef0, _ = _read_gt(dataset_dir / "train" / f"groundtruth_{cam}.csv")
    lat0, lon0, alt0 = pymap3d.ecef2geodetic(*ecef0[0])
    r_enu_ecef = _enu_rotation(lat0, lon0)

    maps = None
    viewmats, centers_all, split_ids = [], [], {"train": [], "val": []}
    idx = 0
    for split in ("train", "val"):
        t_gt, ecef, rots_ecef = _read_gt(
            dataset_dir / split / f"groundtruth_{cam}.csv")
        enu_xyz = np.array(pymap3d.ecef2enu(
            ecef[:, 0], ecef[:, 1], ecef[:, 2], lat0, lon0, alt0)).T
        rots_enu = Rotation.from_matrix(r_enu_ecef @ rots_ecef.as_matrix())

        rgb_dir = dataset_dir / split / "zed_multi" / cam / "rgb"
        names = sorted(p.name for p in rgb_dir.glob("*.jpg"))
        stamps = [_stamp_of(n) for n in names]
        centers, r_wc = _interp_poses(t_gt, enu_xyz, rots_enu, stamps)

        for name, c, r in zip(names, centers, r_wc):
            img = cv2.imread(str(rgb_dir / name))
            if maps is None:
                h, w = img.shape[:2]
                maps = cv2.initUndistortRectifyMap(
                    k_mat, dist, None, k_mat, (w, h), cv2.CV_32FC1)
            und = cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)
            cv2.imwrite(str(out_seg / "images" / f"left_{idx:06d}.jpg"), und,
                        [cv2.IMWRITE_JPEG_QUALITY, 96])
            dimg = cv2.imread(str(dataset_dir / split / "zed_multi" / cam /
                                  "depth" / name.replace(".jpg", ".png")),
                              cv2.IMREAD_UNCHANGED)
            depth = cv2.remap(dimg, maps[0], maps[1],
                              cv2.INTER_NEAREST).astype(np.float32) / 1000.0
            valid = (depth > min_z) & (depth < max_z)
            depth[~valid] = 0.0
            np.savez_compressed(out_seg / "depth" / f"{idx:06d}.npz",
                                depth=depth.astype(np.float16), valid=valid)
            viewmats.append(viewmat_from(r, c))
            centers_all.append(c)
            split_ids[split].append(idx)
            idx += 1
        print(f"{split}: {len(names)} frames ingested")

    np.save(out_seg / "viewmats.npy", np.stack(viewmats).astype(np.float32))
    np.save(out_seg / "cam_centers.npy",
            np.stack(centers_all).astype(np.float32))
    h, w = maps[0].shape[:2]
    meta = {"intrinsics": {"width": w, "height": h, "fx": fx, "fy": fy,
                           "cx": cx, "cy": cy},
            "world_origin": {"lat0": lat0, "lon0": lon0, "alt0": alt0},
            "crs": {"type": "local_ENU", "ellipsoid": "WGS84",
                    "origin_lat": lat0, "origin_lon": lon0,
                    "origin_alt_ellipsoidal": alt0,
                    "vertical_datum": "WGS84 ellipsoid (not orthometric)"},
            "gnss_quality": None,
            "pose_source": "agrigs_groundtruth_csv (oracle condition)",
            "n_frames": idx, "dataset": str(dataset_dir), "camera": cam,
            "window": {"note": "AgriGS demo; train pass + separate "
                               "reverse-direction val pass"},
            "path_length_m": float(np.linalg.norm(
                np.diff(np.stack(centers_all), axis=0), axis=1).sum())}
    (out_seg / "segment_meta.json").write_text(json.dumps(meta, indent=2))
    manifest = {"train": split_ids["train"], "val": split_ids["val"],
                "test": [],
                "policy": "AgriGS protocol: val = independent reverse-"
                          "direction traversal (true novel views)"}
    (out_seg / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"segment ready: {idx} frames "
          f"({len(split_ids['train'])}/{len(split_ids['val'])} train/val), "
          f"origin ({lat0:.6f}, {lon0:.6f})")


def verify(out_seg: Path, out_png: Path, pair_gap: int = 6):
    """Pose-convention verification: back-project frame i's depth, reproject
    into frame i+gap, and save GT-vs-reprojection side by side. If the
    convention is right, the reprojection aligns with the target image."""
    meta = json.loads((out_seg / "segment_meta.json").read_text())
    intr = meta["intrinsics"]
    vms = np.load(out_seg / "viewmats.npy")
    i, j = 0, pair_gap
    img_i = cv2.imread(str(out_seg / "images" / f"left_{i:06d}.jpg"))
    img_j = cv2.imread(str(out_seg / "images" / f"left_{j:06d}.jpg"))
    dz = np.load(out_seg / "depth" / f"{i:06d}.npz")
    depth, valid = dz["depth"].astype(np.float32), dz["valid"]

    h, w = depth.shape
    vs, us = np.mgrid[0:h:2, 0:w:2]
    d = depth[vs, us]
    ok = valid[vs, us]
    us, vs, d = us[ok], vs[ok], d[ok]
    x = (us - intr["cx"]) / intr["fx"] * d
    y = (vs - intr["cy"]) / intr["fy"] * d
    pts_i = np.stack([x, y, d], axis=-1)
    r_i, t_i = vms[i][:3, :3], vms[i][:3, 3]
    world = (pts_i - t_i) @ r_i
    r_j, t_j = vms[j][:3, :3], vms[j][:3, 3]
    cam_j = world @ r_j.T + t_j
    front = cam_j[:, 2] > 0.1
    cam_j, colors = cam_j[front], img_i[vs[front], us[front]]
    u = (cam_j[:, 0] / cam_j[:, 2] * intr["fx"] + intr["cx"]).astype(int)
    v = (cam_j[:, 1] / cam_j[:, 2] * intr["fy"] + intr["cy"]).astype(int)
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    canvas = np.zeros_like(img_j)
    canvas[v[inb], u[inb]] = colors[inb]
    blend = cv2.addWeighted(img_j, 0.5, canvas, 0.9, 0)
    side = np.concatenate([img_j, blend], axis=1)
    cv2.imwrite(str(out_png), side)
    err_px = float(np.mean(inb))
    print(f"verify: frame {i} depth reprojected into frame {j}; "
          f"{err_px*100:.0f}% points landed in-bounds -> {out_png}")
    print("LOOK at the image: right half = target frame with reprojected "
          "colors overlaid. Aligned structure = convention correct; "
          "smeared/rotated = wrong.")
