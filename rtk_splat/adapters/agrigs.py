"""Ingest the AgriGS-SLAM demo dataset into segment contract v2.

Everything dataset-specific lives HERE; the core pipeline (train/evaluate/
export) consumes the produced segment unchanged.

Their layout (per split: train/ val/):
  zed_multi/cam_k/rgb/<sec-nsec>.jpg      1920x1200, mildly distorted
  zed_multi/cam_k/depth/<sec-nsec>.png    uint16 millimeters, 0 = invalid
  groundtruth_cam_k.csv                   timestamp,tx,ty,tz,qx,qy,qz,qw (ECEF)

Produced segment: undistorted images, recorded depth, initial camera poses, and
GNSS-shaped absolute-position evidence in a local ENU frame. The evidence is
derived from the dataset's published ECEF camera ground truth; it is explicitly
labeled as oracle ground truth and is not presented as measured RTK. The
manifest preserves train = training traversal and val = the SEPARATE
reverse-direction traversal (their genuinely independent novel-view protocol).

Pose convention (verified empirically by the `verify` step): the CSV quaternion
is camera-to-ECEF with OpenCV optical axes (x right, y down, z forward).
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from rtk_splat.core.poses import viewmat_from
from rtk_splat.core.segment import (
    CONTRACT_VERSION,
    POSITION_QUALITY_VOCABULARY,
    SegmentReader,
    SegmentWriter,
)

try:
    import pymap3d
except ModuleNotFoundError:  # adapter-only optional dependency
    pymap3d = None


_UNKNOWN_STATUS = -1
_POSITION_PROVENANCE = "oracle groundtruth"


def _read_gt(csv_path: Path):
    """(t[ns], ecef[N,3], Rotation[N]) from groundtruth_cam_k.csv."""
    ts, xyz, quats = [], [], []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            sec, nsec = row["timestamp"].split("-")
            ts.append(int(sec) * 1_000_000_000 + int(nsec))
            xyz.append([float(row["tx"]), float(row["ty"]), float(row["tz"])])
            quats.append([float(row["qx"]), float(row["qy"]),
                          float(row["qz"]), float(row["qw"])])
    timestamps = np.asarray(ts, dtype=np.int64)
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError(
            f"{csv_path}: ground-truth timestamps must be strictly increasing"
        )
    return (
        timestamps,
        np.asarray(xyz, dtype=np.float64),
        Rotation.from_quat(np.asarray(quats, dtype=np.float64)),
    )


def _enu_rotation(lat0: float, lon0: float) -> np.ndarray:
    """ECEF-vector -> ENU-vector rotation at the origin."""
    lam, phi = np.radians(lon0), np.radians(lat0)
    sl, cl = np.sin(lam), np.cos(lam)
    sp, cp = np.sin(phi), np.cos(phi)
    return np.array([[-sl, cl, 0.0],
                     [-sp * cl, -sp * sl, cp],
                     [cp * cl, cp * sl, sp]])


def _stamp_ns_of(name: str) -> int:
    sec, nsec = Path(name).stem.split("-")
    nanoseconds = int(nsec)
    if not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError(f"invalid timestamp in image name: {name}")
    return int(sec) * 1_000_000_000 + nanoseconds


def _interp_poses(t_gt_ns, enu_xyz, rots_enu, stamps_ns):
    """(centers[N,3], R_enu_cam[N,3,3]) lerp/slerp'd at image stamps."""
    t_gt_ns = np.asarray(t_gt_ns)
    stamps_ns = np.asarray(stamps_ns)
    if not np.issubdtype(t_gt_ns.dtype, np.integer) or not np.issubdtype(
        stamps_ns.dtype, np.integer
    ):
        raise ValueError("ground-truth and image timestamps must be integer nanoseconds")
    if np.any(stamps_ns < t_gt_ns[0]) or np.any(stamps_ns > t_gt_ns[-1]):
        raise ValueError(
            "image timestamp lies outside the ground-truth interval; "
            "refusing endpoint clamping"
        )
    origin_ns = int(t_gt_ns[0])
    t_gt = (
        np.asarray(t_gt_ns, dtype=np.int64) - np.int64(origin_ns)
    ).astype(np.float64) * 1.0e-9
    stamps = (
        np.asarray(stamps_ns, dtype=np.int64) - np.int64(origin_ns)
    ).astype(np.float64) * 1.0e-9
    slerp = Slerp(t_gt, rots_enu)
    centers, mats = [], []
    for t in stamps:
        centers.append([np.interp(t, t_gt, enu_xyz[:, k]) for k in range(3)])
        mats.append(slerp([t]).as_matrix()[0])
    return np.array(centers), np.array(mats)


def _interpolation_indices(
    source_ns: np.ndarray, query_ns: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest/lower/upper source rows and interpolation weights."""
    upper = np.searchsorted(source_ns, query_ns, side="left")
    upper = np.minimum(upper, len(source_ns) - 1)
    lower = np.maximum(upper - 1, 0)
    exact = source_ns[upper] == query_ns
    lower[exact] = upper[exact]
    denominator = source_ns[upper] - source_ns[lower]
    alpha = np.zeros(len(query_ns), dtype=np.float64)
    varying = denominator > 0
    alpha[varying] = (
        (query_ns[varying] - source_ns[lower[varying]])
        / denominator[varying]
    )
    lower_delta = np.abs(query_ns - source_ns[lower])
    upper_delta = np.abs(source_ns[upper] - query_ns)
    nearest = np.where(lower_delta <= upper_delta, lower, upper)
    return (
        nearest.astype(np.int64),
        lower.astype(np.int64),
        upper.astype(np.int64),
        alpha,
    )


def ingest(dataset_dir: Path, cam: str, intrinsic, distortion, out_seg: Path,
           min_z: float, max_z: float) -> SegmentReader:
    """Build one immutable v2 segment with independent train/val traversals."""
    dataset_dir = Path(dataset_dir)
    out_seg = Path(out_seg)
    if out_seg.exists():
        raise FileExistsError(f"refusing to modify existing segment: {out_seg}")
    if pymap3d is None:
        raise ModuleNotFoundError(
            "the AgriGS adapter requires pymap3d"
        )
    fx, fy, cx, cy = (float(value) for value in intrinsic)
    k_mat = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.asarray(distortion, dtype=np.float64)

    _, ecef0, _ = _read_gt(
        dataset_dir / "train" / f"groundtruth_{cam}.csv"
    )
    lat0, lon0, alt0 = (
        float(value) for value in pymap3d.ecef2geodetic(*ecef0[0])
    )
    r_enu_ecef = _enu_rotation(lat0, lon0)

    writer = SegmentWriter(out_seg)
    images_dir = writer.directory("images")
    depth_dir = writer.directory("depth")
    maps = None
    width = height = 0
    timestamps: list[int] = []
    left_paths: list[str] = []
    depth_paths: list[str] = []
    viewmats: list[np.ndarray] = []
    centers_all: list[np.ndarray] = []
    split_ids: dict[str, list[int]] = {"train": [], "val": []}
    source_streams: list[dict[str, np.ndarray]] = []
    source_nearest: list[int] = []
    source_lower: list[int] = []
    source_upper: list[int] = []
    source_alpha: list[float] = []
    source_offset = 0
    try:
        for split in ("train", "val"):
            t_gt_ns, ecef, rots_ecef = _read_gt(
                dataset_dir / split / f"groundtruth_{cam}.csv"
            )
            enu_xyz = np.asarray(
                pymap3d.ecef2enu(
                    ecef[:, 0],
                    ecef[:, 1],
                    ecef[:, 2],
                    lat0,
                    lon0,
                    alt0,
                )
            ).T
            rots_enu = Rotation.from_matrix(
                r_enu_ecef @ rots_ecef.as_matrix()
            )

            rgb_dir = dataset_dir / split / "zed_multi" / cam / "rgb"
            names = sorted(
                (path.name for path in rgb_dir.glob("*.jpg")),
                key=_stamp_ns_of,
            )
            stamps_ns = np.asarray(
                [_stamp_ns_of(name) for name in names], dtype=np.int64
            )
            if not len(stamps_ns):
                raise ValueError(f"{rgb_dir}: no JPEG frames")
            centers, rotations = _interp_poses(
                t_gt_ns, enu_xyz, rots_enu, stamps_ns
            )
            nearest, lower, upper, alpha = _interpolation_indices(
                t_gt_ns, stamps_ns
            )
            source_nearest.extend((nearest + source_offset).tolist())
            source_lower.extend((lower + source_offset).tolist())
            source_upper.extend((upper + source_offset).tolist())
            source_alpha.extend(alpha.tolist())
            source_streams.append(
                {
                    "timestamp_ns": t_gt_ns,
                    "enu_m": enu_xyz,
                    "ecef_m": ecef,
                    "orientation_xyzw": rots_enu.as_quat(),
                    "split": np.asarray([split] * len(t_gt_ns)),
                }
            )
            source_offset += len(t_gt_ns)

            for name, stamp_ns, center, rotation in zip(
                names, stamps_ns, centers, rotations
            ):
                image = cv2.imread(str(rgb_dir / name))
                if image is None:
                    raise ValueError(f"cannot decode {rgb_dir / name}")
                if maps is None:
                    height, width = image.shape[:2]
                    maps = cv2.initUndistortRectifyMap(
                        k_mat,
                        dist,
                        None,
                        k_mat,
                        (width, height),
                        cv2.CV_32FC1,
                    )
                elif image.shape[:2] != (height, width):
                    raise ValueError("all AgriGS images must share one size")
                undistorted = cv2.remap(
                    image, maps[0], maps[1], cv2.INTER_LINEAR
                )

                frame_id = len(timestamps)
                image_relative = f"images/left_{frame_id:06d}.jpg"
                if not cv2.imwrite(
                    str(images_dir / f"left_{frame_id:06d}.jpg"),
                    undistorted,
                    [cv2.IMWRITE_JPEG_QUALITY, 96],
                ):
                    raise OSError(f"failed to write {image_relative}")

                source_depth = (
                    dataset_dir
                    / split
                    / "zed_multi"
                    / cam
                    / "depth"
                    / name.replace(".jpg", ".png")
                )
                depth_image = cv2.imread(
                    str(source_depth), cv2.IMREAD_UNCHANGED
                )
                if depth_image is None:
                    raise ValueError(f"cannot decode {source_depth}")
                depth = (
                    cv2.remap(
                        depth_image, maps[0], maps[1], cv2.INTER_NEAREST
                    ).astype(np.float32)
                    / 1000.0
                )
                valid = (depth > min_z) & (depth < max_z)
                depth[~valid] = 0.0
                depth_relative = f"depth/{frame_id:06d}.npz"
                np.savez_compressed(
                    depth_dir / f"{frame_id:06d}.npz",
                    depth=depth.astype(np.float16),
                    valid=valid,
                )

                timestamps.append(int(stamp_ns))
                left_paths.append(image_relative)
                depth_paths.append(depth_relative)
                viewmats.append(viewmat_from(rotation, center))
                centers_all.append(center)
                split_ids[split].append(frame_id)
            print(f"{split}: {len(names)} frames ingested")

        timestamp_ns = np.asarray(timestamps, dtype=np.int64)
        if np.any(np.diff(timestamp_ns) <= 0):
            raise ValueError(
                "combined train/val image timestamps must be strictly increasing"
            )
        frame_id = np.arange(len(timestamp_ns), dtype=np.int64)
        viewmat_array = np.stack(viewmats).astype(np.float64)
        center_array = np.stack(centers_all).astype(np.float64)
        frames = {
            "frame_id": frame_id,
            "timestamp_ns": timestamp_ns,
            "left_image_path": np.asarray(left_paths, dtype=np.str_),
            "depth_path": np.asarray(depth_paths, dtype=np.str_),
            "initial_viewmat": viewmat_array,
            "initial_camera_center_m": center_array,
            "pose_valid": np.ones(len(timestamp_ns), dtype=bool),
        }
        calibration = {
            "contract_version": CONTRACT_VERSION,
            "cameras": {
                "left": {
                    "model": "PINHOLE",
                    "width": width,
                    "height": height,
                    "K": k_mat.tolist(),
                    "distortion": [0.0] * len(dist),
                }
            },
            "rectification": {
                "method": "opencv_initUndistortRectifyMap",
                "source_distortion": dist.tolist(),
                "output_K": k_mat.tolist(),
            },
            "image_geometry": "rectified",
        }
        capabilities = {
            "stereo": False,
            "rgbd": True,
            "single_rtk": False,
            "dual_rtk": False,
            "depth_recorded": True,
            "depth_computed": False,
            "imu_present": False,
            "images_raw": False,
            "images_rectified": True,
        }
        meta = {
            "contract_version": CONTRACT_VERSION,
            "n_frames": len(timestamp_ns),
            "capabilities": capabilities,
            "coordinate_frame": {
                "type": "local_enu",
                "world_frame_id": "agrigs_local_enu",
                "units": "m",
                "origin_wgs84": {
                    "latitude_deg": lat0,
                    "longitude_deg": lon0,
                    "ellipsoidal_altitude_m": alt0,
                    "ellipsoid": "WGS84",
                    "vertical_datum": "WGS84 ellipsoid",
                },
            },
            "timebase": {
                "frame_timestamp_source": "AgriGS RGB filename",
                "observation_timestamp_source": "AgriGS groundtruth CSV",
                "unit": "ns",
                "association_clock_offset_ns": 0,
            },
            "position_observation": {
                "type": "oracle_groundtruth",
                "quantity": "camera_center",
                "sensor_frame_id": cam,
                "coordinates": "ENU_m",
                "covariance_frame": "ENU_m2",
                "validity_field": "position_valid",
                "quality_field": "position_quality",
                "quality_vocabulary": list(POSITION_QUALITY_VOCABULARY),
            },
            "initial_pose": {
                "camera_frame_id": cam,
                "position_quantity": "left_camera_center",
                "source": "interpolated AgriGS oracle camera ground truth",
                "lever_arm_applied": False,
                "extrinsic_translation_sigma_m": [0.0, 0.0, 0.0],
                "extrinsic_translation_sigma_frame_id": cam,
            },
            "depth_observation": {
                "format": "npz_depth_valid",
                "units": "m",
                "quantity": "optical_z",
                "aligned_to": "rgb",
                "invalid_convention": "valid=false and depth=0",
            },
            "world_origin": {"lat0": lat0, "lon0": lon0, "alt0": alt0},
            "crs": {
                "type": "local_ENU",
                "ellipsoid": "WGS84",
                "origin_lat": lat0,
                "origin_lon": lon0,
                "origin_alt_ellipsoidal": alt0,
                "vertical_datum": "WGS84 ellipsoid (not orthometric)",
            },
            "position_evidence": {
                "type": "oracle_groundtruth",
                "position_provenance": _POSITION_PROVENANCE,
                "source": "AgriGS published groundtruth camera ECEF CSV",
                "covariance": "unknown",
                "fix_status": "unknown",
                "carrier_status": "unknown",
            },
            "pose_source": "agrigs_groundtruth_csv (oracle condition)",
            "dataset": str(dataset_dir),
            "camera": cam,
            "window": {
                "note": "AgriGS train pass plus independent reverse-direction "
                        "validation pass"
            },
            "path_length_m": float(
                np.linalg.norm(np.diff(center_array, axis=0), axis=1).sum()
            ),
        }
        manifest = {
            "train": split_ids["train"],
            "val": split_ids["val"],
            "test": [],
            "policy": "AgriGS protocol: val is an independent "
                      "reverse-direction traversal (true novel views)",
        }
        unknown_covariance = np.full(
            (len(timestamp_ns), 3, 3), np.nan, dtype=np.float64
        )
        unknown_status = np.full(
            len(timestamp_ns), _UNKNOWN_STATUS, dtype=np.int16
        )
        raw_timestamp_unsorted = np.concatenate(
            [stream["timestamp_ns"] for stream in source_streams]
        ).astype(np.int64, copy=False)
        order = np.argsort(raw_timestamp_unsorted, kind="stable")
        inverse_order = np.empty(len(order), dtype=np.int64)
        inverse_order[order] = np.arange(len(order), dtype=np.int64)
        nearest_index = inverse_order[np.asarray(source_nearest, dtype=np.int64)]
        lower_index = inverse_order[np.asarray(source_lower, dtype=np.int64)]
        upper_index = inverse_order[np.asarray(source_upper, dtype=np.int64)]
        raw_timestamp_ns = raw_timestamp_unsorted[order]
        absolute_positions = {
            "frame_id": frame_id,
            "frame_timestamp_ns": timestamp_ns,
            "source_index": nearest_index,
            "source_timestamp_ns": raw_timestamp_ns[nearest_index],
            "interpolation_lower_index": lower_index,
            "interpolation_upper_index": upper_index,
            "interpolation_alpha": np.asarray(source_alpha, dtype=np.float64),
            "enu_m": center_array,
            "covariance_enu_m2": unknown_covariance,
            "fix_status": unknown_status,
            "carrier_status": unknown_status.copy(),
            "position_valid": np.ones(len(timestamp_ns), dtype=bool),
            "position_quality": np.asarray(["oracle"] * len(timestamp_ns)),
            "position_provenance": np.asarray(
                [_POSITION_PROVENANCE] * len(timestamp_ns), dtype=np.str_
            ),
            "evidence_type": np.asarray(
                ["oracle_groundtruth"] * len(timestamp_ns), dtype=np.str_
            ),
            "raw_timestamp_ns": raw_timestamp_ns,
            "raw_enu_m": np.concatenate(
                [stream["enu_m"] for stream in source_streams]
            )[order],
            "raw_ecef_m": np.concatenate(
                [stream["ecef_m"] for stream in source_streams]
            )[order],
            "raw_orientation_xyzw": np.concatenate(
                [stream["orientation_xyzw"] for stream in source_streams]
            )[order],
            "raw_split": np.concatenate(
                [stream["split"] for stream in source_streams]
            )[order],
        }
        writer.write_frames(frames)
        writer.write_calibration(calibration)
        writer.write_meta(meta)
        writer.write_manifest(manifest)
        writer.write_observations("gnss", absolute_positions)
        reader = writer.finalize()
    except Exception:
        writer.abort()
        raise

    print(
        f"segment ready: {len(timestamp_ns)} frames "
        f"({len(split_ids['train'])}/{len(split_ids['val'])} train/val), "
        f"origin ({lat0:.6f}, {lon0:.6f})"
    )
    return reader


def ingest_config_v2(cfg, destination: str | Path) -> SegmentReader:
    """Run the AgriGS adapter from a resolved sequence configuration."""
    section = cfg.agrigs
    return ingest(
        Path(section.dataset_dir).expanduser(),
        str(section.camera),
        [float(value) for value in section.intrinsic],
        [float(value) for value in section.distortion],
        Path(destination),
        float(cfg.depth.min_z_m),
        float(cfg.depth.max_z_m),
    )


def verify(out_seg: Path, out_png: Path, pair_gap: int = 6):
    """Pose-convention verification: back-project frame i's depth, reproject
    into frame i+gap, and save GT-vs-reprojection side by side. If the
    convention is right, the reprojection aligns with the target image."""
    reader = SegmentReader(out_seg).validate()
    frames = reader.frames
    camera = reader.calibration["cameras"]["left"]
    k_mat = np.asarray(camera["K"], dtype=float)
    fx, fy, cx, cy = (
        k_mat[0, 0],
        k_mat[1, 1],
        k_mat[0, 2],
        k_mat[1, 2],
    )
    vms = frames["initial_viewmat"]
    i, j = 0, pair_gap
    if j >= len(frames["frame_id"]):
        raise ValueError(
            f"pair_gap {pair_gap} exceeds {len(frames['frame_id'])} frames"
        )
    img_i = cv2.imread(str(out_seg / str(frames["left_image_path"][i])))
    img_j = cv2.imread(str(out_seg / str(frames["left_image_path"][j])))
    with np.load(out_seg / str(frames["depth_path"][i])) as archive:
        depth = archive["depth"].astype(np.float32)
        valid = archive["valid"]

    h, w = depth.shape
    vs, us = np.mgrid[0:h:2, 0:w:2]
    d = depth[vs, us]
    ok = valid[vs, us]
    us, vs, d = us[ok], vs[ok], d[ok]
    x = (us - cx) / fx * d
    y = (vs - cy) / fy * d
    pts_i = np.stack([x, y, d], axis=-1)
    r_i, t_i = vms[i][:3, :3], vms[i][:3, 3]
    world = (pts_i - t_i) @ r_i
    r_j, t_j = vms[j][:3, :3], vms[j][:3, 3]
    cam_j = world @ r_j.T + t_j
    front = cam_j[:, 2] > 0.1
    cam_j, colors = cam_j[front], img_i[vs[front], us[front]]
    u = (cam_j[:, 0] / cam_j[:, 2] * fx + cx).astype(int)
    v = (cam_j[:, 1] / cam_j[:, 2] * fy + cy).astype(int)
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
