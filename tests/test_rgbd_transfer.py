import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from rtk_splat.adapters.rgb_observations import (
    load_rectified_rgb_observations,
    publish_rectified_rgb_observations,
)
from rtk_splat.backends.mapper_config import MapperConfig
from rtk_splat.backends.pose_evidence import verify_pose_georeferencing_artifact
from rtk_splat.core.segment import POSITION_QUALITY_VOCABULARY, SegmentReader, SegmentWriter
from rtk_splat.frontends.artifact import (
    ArtifactError,
    canonical_hash,
    image_inventory,
)
from rtk_splat.workflows.rgbd_transfer import (
    RgbdTransferConfig,
    _fill_projection_lattice,
    _interpolate_viewmats,
    _reproject_optical_z,
    build_parser,
    derive_rgbd_segment,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _viewmats(centers: np.ndarray) -> np.ndarray:
    result = np.repeat(np.eye(4)[None], len(centers), axis=0)
    result[:, :3, 3] = -centers
    return result


def _camera() -> dict:
    return {
        "model": "PINHOLE",
        "width": 5,
        "height": 5,
        "K": [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
        "distortion": [0.0, 0.0, 0.0, 0.0],
    }


def _source_segment(
    root: Path,
    *,
    imu: bool = False,
    acquisition_id: str = "synthetic-acquisition-a",
    include_depth_timing: bool = True,
    stationary: bool = False,
) -> Path:
    destination = root / "source-segment"
    writer = SegmentWriter(destination)
    image_dir = writer.directory("images")
    depth_dir = writer.directory("depth")
    n = 5
    timestamps = np.arange(n, dtype=np.int64) * 1_000_000_000 + 1_000_000_000
    left_paths = []
    right_paths = []
    depth_paths = []
    for index in range(n):
        image = np.full((5, 5, 3), 20 + index, dtype=np.uint8)
        left_name = f"left_{index:06d}.png"
        right_name = f"right_{index:06d}.png"
        assert cv2.imwrite(str(image_dir / left_name), image)
        assert cv2.imwrite(str(image_dir / right_name), image)
        depth_name = f"{index:06d}.npz"
        np.savez_compressed(
            depth_dir / depth_name,
            depth=np.full((5, 5), 2.0, dtype=np.float32),
            valid=np.ones((5, 5), dtype=bool),
        )
        left_paths.append(f"images/{left_name}")
        right_paths.append(f"images/{right_name}")
        depth_paths.append(f"depth/{depth_name}")
    centers = (
        np.zeros((n, 3), dtype=float)
        if stationary
        else np.column_stack((np.arange(n), np.zeros(n), np.zeros(n))).astype(float)
    )
    frames = {
        "frame_id": np.arange(n, dtype=np.int64),
        "timestamp_ns": timestamps,
        "left_image_path": np.asarray(left_paths),
        "right_image_path": np.asarray(right_paths),
        "right_timestamp_ns": timestamps + 1_000,
        "stereo_sync_residual_ns": np.full(n, 1_000, dtype=np.int64),
        "depth_path": np.asarray(depth_paths),
        "initial_viewmat": _viewmats(centers),
        "initial_camera_center_m": centers,
        "pose_valid": np.ones(n, dtype=bool),
    }
    if include_depth_timing:
        frames.update(
            depth_timestamp_ns=timestamps,
            depth_log_timestamp_ns=timestamps + 17,
            depth_sync_residual_ns=np.zeros(n, dtype=np.int64),
        )
    raw_covariance = np.repeat(np.eye(3)[None] * 0.0004, n, axis=0)
    gnss = {
        "frame_id": np.arange(n, dtype=np.int64),
        "frame_timestamp_ns": timestamps,
        "association_query_timestamp_ns": timestamps,
        "source_index": np.arange(n, dtype=np.int64),
        "source_timestamp_ns": timestamps,
        "source_residual_ns": np.zeros(n, dtype=np.int64),
        "enu_m": centers,
        "covariance_enu_m2": raw_covariance,
        "fix_status": np.ones(n, dtype=np.int16),
        "carrier_status": np.full(n, 2, dtype=np.int16),
        "position_valid": np.ones(n, dtype=bool),
        "position_quality": np.asarray(["rtk_fixed"] * n),
        "raw_timestamp_ns": timestamps,
        "raw_enu_m": centers,
        "raw_covariance_enu_m2": raw_covariance,
        "raw_fix_status": np.ones(n, dtype=np.int16),
        "raw_carrier_status": np.full(n, 2, dtype=np.int16),
    }
    camera = _camera()
    calibration = {
        "contract_version": 2,
        "cameras": {"left": camera, "right": camera},
        "T_right_left": [
            [1, 0, 0, -0.1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        "transform_conventions": {
            "T_right_left": "right_from_left",
            "T_camera_primary_antenna": "camera_from_primary_antenna",
        },
        "sensor_frames": {
            "world": "map",
            "camera": "ir1_rectified",
            "primary_antenna": "antenna",
            "secondary_antenna": None,
        },
        "rough_extrinsics": {
            "convention": "camera_from_primary_antenna",
            "T_camera_primary_antenna": np.eye(4).tolist(),
            "provenance": {"method": "synthetic"},
        },
    }
    meta = {
        "contract_version": 2,
        "n_frames": n,
        "acquisition_id": acquisition_id,
        "capabilities": {
            "stereo": True,
            "rgbd": True,
            "single_rtk": True,
            "dual_rtk": False,
            "depth_recorded": True,
            "depth_computed": False,
            "imu_present": imu,
            "images_raw": False,
            "images_rectified": True,
        },
        "coordinate_frame": {
            "type": "local_enu",
            "world_frame_id": "map",
            "units": "m",
            "origin_wgs84": {
                "latitude_deg": 0.0,
                "longitude_deg": 0.0,
                "ellipsoidal_altitude_m": 0.0,
                "ellipsoid": "WGS84",
                "vertical_datum": "WGS84 ellipsoid",
            },
        },
        "timebase": {
            "frame_timestamp_source": "synthetic IR header",
            "observation_timestamp_source": "synthetic GNSS header",
            "unit": "ns",
            "association_clock_offset_ns": 0,
        },
        "position_observation": {
            "type": "gnss",
            "quantity": "antenna_phase_center",
            "sensor_frame_id": "antenna",
            "coordinates": "ENU_m",
            "covariance_frame": "ENU_m2",
            "validity_field": "position_valid",
            "quality_field": "position_quality",
            "quality_vocabulary": list(POSITION_QUALITY_VOCABULARY),
        },
        "initial_pose": {
            "camera_frame_id": "ir1_rectified",
            "position_quantity": "left_camera_center",
            "source": "synthetic RTK",
            "lever_arm_applied": True,
            "extrinsic_translation_sigma_m": [0.005, 0.005, 0.005],
            "extrinsic_translation_sigma_frame_id": "ir1_rectified",
        },
        "depth_observation": {
            "format": "npz_depth_valid",
            "units": "m",
            "quantity": "optical_z",
            "aligned_to": "left",
            "invalid_convention": "valid=false and depth=0",
        },
        "provenance": {"acquisition_id": acquisition_id},
    }
    writer.write_frames(frames)
    writer.write_calibration(calibration)
    writer.write_meta(meta)
    writer.write_manifest({"train": [0, 1, 2], "val": [3], "test": [4]})
    writer.write_observations("gnss", gnss)
    if imu:
        writer.write_observations(
            "imu",
            {
                "timestamp_ns": timestamps,
                "accel_mps2": np.zeros((n, 3)),
                "gyro_radps": np.zeros((n, 3)),
            },
        )
    writer.finalize()
    return destination


def _pose_artifact(root: Path, source: Path, *, offset_m: float = 0.0) -> Path:
    destination = root / "source-pose"
    destination.mkdir()
    source_reader = SegmentReader(source)
    frames = source_reader.frames
    centers = np.asarray(frames["initial_camera_center_m"], dtype=float).copy()
    centers[:, 1] += offset_m
    viewmats = _viewmats(centers)
    np.save(destination / "viewmats.npy", viewmats)
    np.save(destination / "cam_centers.npy", centers)
    np.save(destination / "frame_ids.npy", frames["frame_id"])
    np.save(destination / "timestamps_ns.npy", frames["timestamp_ns"])
    np.save(destination / "left_image_names.npy", frames["left_image_path"])
    status = {
        "artifact_class": "production",
        "georeferencing_status": "PASSED",
        "metric_georeferencing_claim_eligible": True,
        "diagnostic_export_requested": False,
        "diagnostic_export_override_used": False,
    }
    _json(
        destination / "quality.json",
        {"schema_version": 1, **status, "rtk_alignment_passed": True},
    )
    _json(destination / "alignment.json", {"schema_version": 1, **status})
    _json(
        destination / "provenance.json",
        {
            "schema_version": 1,
            **status,
            "backend_config_effective": asdict(MapperConfig()),
            "source_segment_binding": _source_binding(source_reader),
        },
    )
    _json(
        destination / "georeferencing.json",
        {"schema_version": 1, **status, "rtk_alignment_passed": True},
    )
    files = {
        path.name: {"sha256": _sha(path), "size_bytes": path.stat().st_size}
        for path in destination.iterdir()
    }
    _json(
        destination / "manifest.json",
        {"schema_version": 1, "name": destination.name, **status, "files": files},
    )
    verify_pose_georeferencing_artifact(destination)
    return destination


def _source_binding(source: SegmentReader) -> dict:
    relatives = [
        "frames.npz",
        "calibration.json",
        "segment_meta.json",
        "manifest.json",
        "observations/gnss.npz",
    ]
    if source.meta["capabilities"]["dual_rtk"]:
        relatives.append("observations/heading.npz")
    files = {
        relative: {
            "sha256": _sha(source.root / relative),
            "size_bytes": (source.root / relative).stat().st_size,
        }
        for relative in relatives
    }
    inventory = image_inventory(source)
    contract = {
        "segment_root": str(source.root.resolve()),
        "files": files,
        "image_inventory_count": len(inventory),
        "image_inventory_sha256": canonical_hash(inventory),
        "acquisition_id": source.meta["acquisition_id"],
    }
    return {
        "contract_inputs": contract,
        "contract_inputs_sha256": canonical_hash(contract),
        "acquisition_id": source.meta["acquisition_id"],
    }


def _rgb_observations(
    root: Path,
    *,
    distorted_pinhole: bool = False,
    extrinsic_x_m: float = 0.0,
    timestamp_offset_ns: int = 0,
    acquisition_id: str = "synthetic-acquisition-a",
    explicit_timestamps_ns: np.ndarray | None = None,
) -> Path:
    destination = root / "rgb-observations"
    images = destination / "images"
    images.mkdir(parents=True)
    # Deliberately twice the source-depth rate. The transfer must select one
    # unique nearest RGB observation per source depth without relaxing its gate.
    timestamps = (
        np.arange(9, dtype=np.int64) * 500_000_000
        + 1_000_000_000
        + timestamp_offset_ns
        if explicit_timestamps_ns is None
        else np.asarray(explicit_timestamps_ns, dtype=np.int64)
    )
    paths = []
    for index in range(len(timestamps)):
        name = f"rgb_{index:06d}.png"
        assert cv2.imwrite(
            str(images / name), np.full((5, 5, 3), 100 + index, dtype=np.uint8)
        )
        paths.append(f"images/{name}")
    with (destination / "frames.npz").open("wb") as stream:
        np.savez_compressed(
            stream,
            frame_id=np.arange(len(timestamps), dtype=np.int64),
            timestamp_ns=timestamps,
            log_timestamp_ns=timestamps + 123,
            image_path=np.asarray(paths),
        )
    _json(
        destination / "rgb_observations_meta.json",
        {
            "schema_version": 1,
            "artifact_type": "calibrated_rgb_observations",
            "n_frames": len(timestamps),
            "acquisition_id": acquisition_id,
            "timestamp_unit": "ns",
            "timestamp_source": "synthetic RGB header",
            "clock": {
                "rgb_to_source_clock_offset_ns": 0,
                "provenance": {"method": "synthetic common clock"},
            },
            "provenance": {
                "dataset": "synthetic",
                "acquisition_id": acquisition_id,
            },
        },
    )
    camera = _camera()
    if distorted_pinhole:
        camera["distortion"] = [0.1, 0.0, 0.0, 0.0]
    transform = np.eye(4)
    transform[0, 3] = extrinsic_x_m
    _json(
        destination / "calibration.json",
        {
            "schema_version": 1,
            "camera_frame_id": "rgb_rectified",
            "source_camera_frame_id": "ir1_rectified",
            "source_camera_geometry": "source_segment_left_rectified",
            "rgb_camera_geometry": "output_camera_rectified",
            "camera": camera,
            "image_geometry": "rectified",
            "rectification": {
                "input_camera": _camera(),
                "method": "identity recorded factory representation",
                "provenance": {"method": "synthetic identity"},
            },
            "T_rgb_source_camera": transform.tolist(),
            "transform_convention": "rgb_from_source_camera",
            "extrinsic_translation_sigma_m": [0.005, 0.005, 0.005],
            "extrinsic_translation_sigma_frame_id": "rgb_rectified",
            "extrinsic_provenance": {"method": "synthetic identity"},
        },
    )
    sealed = [
        destination / "rgb_observations_meta.json",
        destination / "frames.npz",
        destination / "calibration.json",
        *(destination / path for path in paths),
    ]
    _json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "calibrated_rgb_observations",
            "files": {
                str(path.relative_to(destination)): {
                    "sha256": _sha(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sealed
            },
        },
    )
    return destination


def _registration_evidence(
    root: Path,
    source: Path,
    source_pose: Path,
    rgb: Path,
    *,
    pixel_bias_px: float = 0.0,
    ground_truth_extrinsic_x_m: float = 0.2,
    declared_clock_offset_ns: int | None = None,
) -> Path:
    destination = root / "held-out-registration"
    destination.mkdir()
    calibration = json.loads((rgb / "calibration.json").read_text(encoding="utf-8"))
    rgb_meta = json.loads(
        (rgb / "rgb_observations_meta.json").read_text(encoding="utf-8")
    )
    K = np.asarray(calibration["camera"]["K"], dtype=np.float64)
    ground_truth_transform = np.eye(4)
    ground_truth_transform[0, 3] = ground_truth_extrinsic_x_m
    u, v = np.meshgrid(np.arange(5, dtype=float) + 0.25, np.arange(5) + 0.25)
    pixels = np.column_stack((u.ravel(), v.ravel()))
    z = np.full(len(pixels), 2.0)
    rgb_points = np.column_stack(
        (
            (pixels[:, 0] - K[0, 2]) * z / K[0, 0],
            (pixels[:, 1] - K[1, 2]) * z / K[1, 1],
            z,
            np.ones(len(pixels)),
        )
    )
    source_points = (np.linalg.inv(ground_truth_transform) @ rgb_points.T).T[:, :3]
    np.save(destination / "source_points_xyz_m.npy", source_points)
    np.save(destination / "rgb_pixels_px.npy", pixels + pixel_bias_px)
    source_depth_ns = np.where(
        np.arange(len(pixels)) < len(pixels) // 2,
        4_000_000_000,
        5_000_000_000,
    ).astype(np.int64)
    rgb_timestamp_ns = source_depth_ns.copy()
    np.save(destination / "source_depth_timestamp_ns.npy", source_depth_ns)
    np.save(destination / "rgb_timestamp_ns.npy", rgb_timestamp_ns)
    source_reader = SegmentReader(source)
    declaration = {
        "schema_version": 1,
        "artifact_type": "held_out_rgb_depth_registration_evidence",
        "validation_split": "held_out",
        "independent_of_extrinsic_estimation": True,
        "correspondence_geometry": (
            "source_depth_xyz_at_depth_time_to_rgb_pixels_at_rgb_time"
        ),
        "method": "synthetic independent calibration-target correspondences",
        "source_segment_manifest_sha256": _sha(source / "manifest.json"),
        "rgb_observations_manifest_sha256": _sha(rgb / "manifest.json"),
        "rgb_calibration_sha256": _sha(rgb / "calibration.json"),
        "source_pose_manifest_sha256": _sha(source_pose / "manifest.json"),
        "rgb_to_source_clock_offset_ns": (
            rgb_meta["clock"]["rgb_to_source_clock_offset_ns"]
            if declared_clock_offset_ns is None
            else declared_clock_offset_ns
        ),
        "source_camera_frame_id": source_reader.meta["initial_pose"][
            "camera_frame_id"
        ],
        "rgb_camera_frame_id": calibration["camera_frame_id"],
    }
    _json(destination / "validation.json", declaration)
    files = {
        path.name: {"sha256": _sha(path), "size_bytes": path.stat().st_size}
        for path in destination.iterdir()
    }
    _json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "held_out_rgb_depth_registration_evidence",
            "files": files,
        },
    )
    return destination


def _config(**overrides) -> RgbdTransferConfig:
    values = {
        "max_depth_sync_residual_ns": 600_000_000,
        "max_pose_interpolation_gap_ns": 1_100_000_000,
        "min_depth_rgb_association_fraction": 0.9,
        "min_projected_depth_coverage_fraction": 0.5,
        "min_projected_depth_retained_fraction": 0.5,
    }
    values.update(overrides)
    return RgbdTransferConfig(**values)


class RgbdTransferTests(unittest.TestCase):
    def test_cli_exposes_every_required_transfer_gate(self):
        args = build_parser().parse_args(
            [
                "--source-segment",
                "/source",
                "--source-pose-artifact",
                "/pose",
                "--rgb-observations",
                "/rgb",
                "--destination-segment",
                "/target",
                "--destination-pose-artifact",
                "/target-pose",
                "--max-depth-sync-residual-ns",
                "100",
                "--max-pose-interpolation-gap-ns",
                "200",
                "--min-depth-rgb-association-fraction",
                "0.9",
                "--min-projected-depth-coverage-fraction",
                "0.1",
                "--min-projected-depth-retained-fraction",
                "0.2",
            ]
        )
        self.assertEqual(args.min_depth_rgb_association_fraction, 0.9)
        self.assertEqual(args.min_projected_depth_coverage_fraction, 0.1)
        self.assertEqual(args.min_projected_depth_retained_fraction, 0.2)
        self.assertEqual(args.clock_observability_probe_ns, 10_000_000)
        self.assertEqual(args.min_clock_observability_px_per_s, 0.1)

    def test_shared_rgb_publisher_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            inputs.mkdir()
            paths = []
            for index in range(2):
                path = inputs / f"{index}.png"
                assert cv2.imwrite(path.as_posix(), np.zeros((5, 5, 3), np.uint8))
                paths.append(path)
            camera = _camera()
            artifact = publish_rectified_rgb_observations(
                root / "rgb",
                paths,
                np.asarray([10, 20], dtype=np.int64),
                log_timestamp_ns=np.asarray([11, 21], dtype=np.int64),
                calibration={
                    "schema_version": 1,
                    "camera_frame_id": "rgb",
                    "source_camera_frame_id": "ir",
                    "source_camera_geometry": "source_segment_left_rectified",
                    "rgb_camera_geometry": "recorded_factory_pinhole_direct",
                    "image_geometry": "rectified",
                    "camera": camera,
                    "rectification": {
                        "input_camera": camera,
                        "method": "identity",
                        "provenance": {"method": "synthetic"},
                    },
                    "T_rgb_source_camera": np.eye(4).tolist(),
                    "transform_convention": "rgb_from_source_camera",
                    "extrinsic_translation_sigma_m": [0.01] * 3,
                    "extrinsic_translation_sigma_frame_id": "rgb",
                    "extrinsic_provenance": {"method": "synthetic"},
                },
                timestamp_source="sensor header",
                clock={
                    "rgb_to_source_clock_offset_ns": 0,
                    "provenance": {"method": "common clock"},
                },
                provenance={"dataset": "synthetic"},
                acquisition_id="synthetic-publisher-acquisition",
            )
            loaded = load_rectified_rgb_observations(artifact)
            np.testing.assert_array_equal(loaded.frames["timestamp_ns"], [10, 20])

    def test_pose_interpolation_is_full_se3(self):
        c2w = np.repeat(np.eye(4)[None], 2, axis=0)
        c2w[1, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        c2w[1, 0, 3] = 2.0
        interpolated, lower, upper, alpha = _interpolate_viewmats(
            np.asarray([0, 2_000_000_000], dtype=np.int64),
            np.linalg.inv(c2w),
            np.asarray([1_000_000_000], dtype=np.int64),
            max_gap_ns=2_000_000_000,
        )
        result = np.linalg.inv(interpolated)[0]
        np.testing.assert_allclose(result[:3, 3], [1.0, 0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(
            result[:3, :3],
            Rotation.from_euler("z", 45, degrees=True).as_matrix(),
            atol=1e-8,
        )
        np.testing.assert_array_equal(lower, [0])
        np.testing.assert_array_equal(upper, [1])
        np.testing.assert_allclose(alpha, [0.5])

    def test_zbuffer_keeps_nearest_optical_surface(self):
        depth, valid = _reproject_optical_z(
            np.asarray([[1.0, 2.0]], dtype=np.float32),
            np.ones((1, 2), dtype=bool),
            np.eye(3),
            np.eye(3),
            np.asarray(
                [[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=float,
            ),
            (1, 4),
        )
        self.assertTrue(valid[0, 2])
        self.assertEqual(float(depth[0, 2]), 1.0)
        self.assertEqual(int(valid.sum()), 1)

    def test_lattice_fill_closes_focal_ratio_resampling_holes(self):
        # A longer output focal length (ratio 1.4, the Rosario D435 IR->RGB
        # case) leaves a fixed lattice of never-sampled rows and columns even
        # on a continuously observed surface. The fill must close it and
        # reproduce the surface, not merely mark pixels valid.
        height, width = 160, 200
        source_k = np.asarray(
            [[100.0, 0.0, 99.5], [0.0, 100.0, 79.5], [0.0, 0.0, 1.0]]
        )
        output_k = np.asarray(
            [[140.0, 0.0, 99.5], [0.0, 140.0, 79.5], [0.0, 0.0, 1.0]]
        )
        y, x = np.mgrid[0:height, 0:width]
        plane = (2.0 + 0.002 * (y - 79.5) + 0.001 * (x - 99.5)).astype(np.float32)
        scattered_depth, scattered_valid = _reproject_optical_z(
            plane,
            np.ones_like(plane, dtype=bool),
            source_k,
            output_k,
            np.eye(4),
            (height, width),
        )
        rows = np.nonzero(scattered_valid.any(axis=1))[0]
        cols = np.nonzero(scattered_valid.any(axis=0))[0]
        footprint = np.zeros_like(scattered_valid)
        footprint[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1] = True
        scatter_coverage = scattered_valid.sum() / footprint.sum()
        self.assertLess(scatter_coverage, 0.65)  # the defect this fix targets
        depth, valid, n_filled = _fill_projection_lattice(
            scattered_depth,
            scattered_valid,
            min_valid_neighbors=3,
            max_spread_abs_m=0.05,
            max_spread_rel=0.02,
        )
        self.assertGreater(n_filled, 0)
        self.assertGreater(valid[footprint].mean(), 0.98)
        # Filled depths must agree with the true surface. Verify against an
        # exact analytic reprojection of the plane into the output camera.
        yy, xx = np.nonzero(valid & ~scattered_valid)
        # z is preserved by the identity transform; the plane expressed in
        # output pixels: invert the projection for each filled pixel.
        a, b = 0.002, 0.001
        ray_x = (xx - output_k[0, 2]) / output_k[0, 0]
        ray_y = (yy - output_k[1, 2]) / output_k[1, 1]
        # Rays are depth-independent under the identity transform:
        # x_src - 99.5 = 100 * ray_x and y_src - 79.5 = 100 * ray_y, so the
        # plane z = 2 + a*(y_src - 79.5) + b*(x_src - 99.5) is linear in rays.
        true_z = 2.0 + 100.0 * a * ray_y + 100.0 * b * ray_x
        error = np.abs(depth[yy, xx] - true_z)
        self.assertLess(float(np.max(error)), 0.01)

    def test_lattice_fill_respects_depth_discontinuities(self):
        # An invalid stripe between two surfaces 1 m apart must stay invalid;
        # the same stripe inside one surface must be filled with that surface.
        depth = np.full((9, 12), 2.0, dtype=np.float32)
        depth[:, 6:] = 3.0
        valid = np.ones_like(depth, dtype=bool)
        valid[:, 2] = False  # lattice hole inside the 2.0 m surface
        valid[:, 5] = False  # hole straddling the 2.0/3.0 m boundary
        valid[:, 9] = False  # lattice hole inside the 3.0 m surface
        filled_depth, filled_valid, n_filled = _fill_projection_lattice(
            depth,
            valid,
            min_valid_neighbors=3,
            max_spread_abs_m=0.05,
            max_spread_rel=0.02,
        )
        self.assertTrue(filled_valid[:, 2].all())
        np.testing.assert_allclose(filled_depth[:, 2], 2.0)
        self.assertTrue(filled_valid[:, 9].all())
        np.testing.assert_allclose(filled_depth[:, 9], 3.0)
        self.assertFalse(filled_valid[:, 5].any())
        self.assertEqual(n_filled, 2 * 9)

    def test_lattice_fill_requires_neighbor_support(self):
        # One isolated sample cannot manufacture a surface around itself.
        depth = np.zeros((5, 5), dtype=np.float32)
        depth[2, 2] = 2.0
        valid = np.zeros_like(depth, dtype=bool)
        valid[2, 2] = True
        filled_depth, filled_valid, n_filled = _fill_projection_lattice(
            depth,
            valid,
            min_valid_neighbors=3,
            max_spread_abs_m=0.05,
            max_spread_rel=0.02,
        )
        self.assertEqual(n_filled, 0)
        self.assertEqual(int(filled_valid.sum()), 1)
        np.testing.assert_array_equal(filled_valid, valid)
        np.testing.assert_array_equal(filled_depth, depth)

    def test_end_to_end_transfer_is_immutable_and_georeferenced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source_segment(root)
            source_pose = _pose_artifact(root, source)
            rgb = _rgb_observations(root, extrinsic_x_m=0.2)
            source_hash = _sha(source / "frames.npz")
            pose_hash = _sha(source_pose / "manifest.json")
            destination = root / "rgbd-segment"
            destination_pose = root / "pose-artifacts" / "rgb-transfer-v1"

            segment, pose = derive_rgbd_segment(
                source,
                source_pose,
                rgb,
                destination,
                destination_pose,
                config=_config(),
            )

            segment.validate()
            evidence = verify_pose_georeferencing_artifact(pose)
            self.assertFalse(evidence["metric_georeferencing_claim_eligible"])
            self.assertEqual(evidence["artifact_class"], "diagnostic_render_only")
            self.assertEqual(evidence["georeferencing_status"], "PASSED")
            georeferencing = json.loads(
                (pose / "georeferencing.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("fixed_scale_se3_applied", georeferencing)
            self.assertTrue(
                georeferencing["source_fixed_scale_alignment_inherited"]
            )
            self.assertTrue(georeferencing["source_rtk_consistency_rechecked"])
            self.assertFalse(georeferencing["target_rtk_acquired"])
            self.assertFalse(georeferencing["new_world_alignment_fitted"])
            self.assertTrue(segment.meta["capabilities"]["rgbd"])
            self.assertFalse(segment.meta["capabilities"]["stereo"])
            self.assertFalse(segment.meta["capabilities"]["single_rtk"])
            self.assertFalse(segment.meta["capabilities"]["dual_rtk"])
            self.assertFalse(segment.meta["capabilities"]["imu_present"])
            self.assertEqual(
                segment.meta["position_observation"]["type"],
                "derived_reference_camera_position",
            )
            self.assertFalse(
                segment.meta["initial_pose_semantics"]["target_rtk_acquired"]
            )
            self.assertIn(
                "no target RTK observation",
                segment.meta["position_evidence"]["role"],
            )
            self.assertNotIn("primary_antenna", segment.meta["sensor_frames"])
            self.assertNotIn("heading_evidence", segment.meta)
            reference = segment.observations("gnss")
            assert reference is not None
            self.assertFalse(any(name.startswith("raw_") for name in reference))
            self.assertEqual(segment.meta["depth_observation"]["aligned_to"], "rgb")
            np.testing.assert_array_equal(
                segment.frames["timestamp_ns"],
                np.arange(5, dtype=np.int64) * 1_000_000_000 + 1_000_000_000,
            )
            np.testing.assert_array_equal(
                segment.frames["rgb_observation_index"], [0, 2, 4, 6, 8]
            )
            np.testing.assert_array_equal(
                segment.frames["rgb_observation_frame_id"], [0, 2, 4, 6, 8]
            )
            np.testing.assert_array_equal(
                segment.frames["left_log_timestamp_ns"],
                segment.frames["timestamp_ns"] + 123,
            )
            np.testing.assert_array_equal(
                segment.frames["corrected_rgb_minus_depth_sync_residual_ns"],
                np.zeros(5, dtype=np.int64),
            )
            np.testing.assert_allclose(
                segment.frames["initial_camera_center_m"][:, 0],
                [-0.2, 0.8, 1.8, 2.8, 3.8],
            )
            association = segment.meta["derived_segment"]["rgb_depth_association"]
            self.assertEqual(association["n_source_depth"], 5)
            self.assertEqual(association["n_rgb_observations"], 9)
            self.assertEqual(association["n_matched"], 5)
            self.assertEqual(association["n_missing_source_depth"], 0)
            self.assertEqual(association["n_unused_rgb"], 4)
            with np.load(pose / "transfer_evidence.npz", allow_pickle=False) as archive:
                np.testing.assert_array_equal(
                    archive["selected_rgb_indices"], [0, 2, 4, 6, 8]
                )
                np.testing.assert_array_equal(
                    archive["selected_source_depth_indices"], [0, 1, 2, 3, 4]
                )
            projected = segment.frames["depth_projected_valid_pixels"]
            self.assertTrue(np.all((projected > 0) & (projected <= 25)))
            self.assertEqual(_sha(source / "frames.npz"), source_hash)
            self.assertEqual(_sha(source_pose / "manifest.json"), pose_hash)
            shutil.rmtree(source)
            shutil.rmtree(source_pose)
            shutil.rmtree(rgb)
            segment.validate()
            verify_pose_georeferencing_artifact(pose)

    def test_sealed_held_out_registration_remains_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source_segment(root)
            source_pose = _pose_artifact(root, source)
            rgb = _rgb_observations(root, extrinsic_x_m=0.2)
            registration = _registration_evidence(root, source, source_pose, rgb)
            segment, pose = derive_rgbd_segment(
                source,
                source_pose,
                rgb,
                root / "target",
                root / "poses" / "target",
                config=_config(held_out_registration_evidence=str(registration)),
            )
            segment.validate()
            evidence = verify_pose_georeferencing_artifact(pose)
            self.assertEqual(evidence["artifact_class"], "diagnostic_render_only")
            self.assertFalse(evidence["metric_georeferencing_claim_eligible"])
            transfer = json.loads(
                (pose / "transfer_evaluation.json").read_text(encoding="utf-8")
            )
            registration_quality = transfer["held_out_registration_quality"]
            self.assertTrue(registration_quality["passed"])
            self.assertFalse(registration_quality["production_promotion_allowed"])
            self.assertEqual(
                registration_quality["reason"],
                "joint_extrinsic_clock_observability_not_established",
            )

    def test_unobservable_clock_keeps_held_out_result_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source_segment(root, stationary=True)
            source_pose = _pose_artifact(root, source)
            rgb = _rgb_observations(root, extrinsic_x_m=0.2)
            registration = _registration_evidence(root, source, source_pose, rgb)
            _, pose = derive_rgbd_segment(
                source,
                source_pose,
                rgb,
                root / "target",
                root / "poses" / "target",
                config=_config(held_out_registration_evidence=str(registration)),
            )
            evidence = verify_pose_georeferencing_artifact(pose)
            self.assertEqual(evidence["artifact_class"], "diagnostic_render_only")
            transfer = json.loads(
                (pose / "transfer_evaluation.json").read_text(encoding="utf-8")
            )
            self.assertFalse(
                transfer["held_out_registration_quality"]["clock_observability"][
                    "observable"
                ]
            )

    def test_held_out_registration_rejects_wrong_candidate_extrinsic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source_segment(root)
            source_pose = _pose_artifact(root, source)
            rgb = _rgb_observations(root, extrinsic_x_m=0.4)
            registration = _registration_evidence(
                root,
                source,
                source_pose,
                rgb,
                ground_truth_extrinsic_x_m=0.2,
            )
            with self.assertRaisesRegex(ArtifactError, "registration gate failed"):
                derive_rgbd_segment(
                    source,
                    source_pose,
                    rgb,
                    root / "target",
                    root / "poses" / "target",
                    config=_config(
                        held_out_registration_evidence=str(registration),
                        max_held_out_median_reprojection_error_px=0.1,
                        max_held_out_p95_reprojection_error_px=0.1,
                    ),
                )
            self.assertFalse((root / "target").exists())

    def test_held_out_registration_is_bound_to_declared_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source_segment(root)
            source_pose = _pose_artifact(root, source)
            rgb = _rgb_observations(root, extrinsic_x_m=0.2)
            registration = _registration_evidence(
                root,
                source,
                source_pose,
                rgb,
                declared_clock_offset_ns=1,
            )
            with self.assertRaisesRegex(ArtifactError, "different inputs"):
                derive_rgbd_segment(
                    source,
                    source_pose,
                    rgb,
                    root / "target",
                    root / "poses" / "target",
                    config=_config(held_out_registration_evidence=str(registration)),
                )

    def test_gross_wrong_extrinsic_fails_projected_depth_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source_segment(root)
            source_pose = _pose_artifact(root, source)
            rgb = _rgb_observations(root, extrinsic_x_m=100.0)
            with self.assertRaisesRegex(ArtifactError, "projected depth gate failed"):
                derive_rgbd_segment(
                    source,
                    source_pose,
                    rgb,
                    root / "target",
                    root / "poses" / "target",
                    config=_config(),
                )
            self.assertFalse((root / "target").exists())

    def test_rgb_sidecar_from_another_acquisition_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source_segment(root, acquisition_id="source-acquisition")
            source_pose = _pose_artifact(root, source)
            rgb = _rgb_observations(root, acquisition_id="other-acquisition")
            with self.assertRaisesRegex(ArtifactError, "sealed to this source"):
                derive_rgbd_segment(
                    source,
                    source_pose,
                    rgb,
                    root / "target",
                    root / "poses" / "target",
                    config=_config(),
                )

    def test_pose_artifact_from_another_acquisition_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source_segment(root, acquisition_id="source-acquisition")
            rgb = _rgb_observations(root, acquisition_id="source-acquisition")
            other = root / "other"
            other.mkdir()
            other_source = _source_segment(
                other, acquisition_id="other-acquisition"
            )
            other_pose = _pose_artifact(other, other_source)
            with self.assertRaisesRegex(ArtifactError, "different segment"):
                derive_rgbd_segment(
                    source,
                    other_pose,
                    rgb,
                    root / "target",
                    root / "poses" / "target",
                    config=_config(),
                )

    def test_sync_gate_fails_without_publishing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source_segment(root)
            source_pose = _pose_artifact(root, source)
            rgb = _rgb_observations(root, timestamp_offset_ns=100_000_000)
            destination = root / "rgbd-segment"
            destination_pose = root / "pose-artifacts" / "rgb-transfer-v1"
            with self.assertRaisesRegex(ArtifactError, "depth.*association failed"):
                derive_rgbd_segment(
                    source,
                    source_pose,
                    rgb,
                    destination,
                    destination_pose,
                    config=_config(max_depth_sync_residual_ns=10),
                )
            self.assertFalse(destination.exists())
            self.assertFalse(destination_pose.exists())

    def test_association_fraction_gate_rejects_partial_rgb_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source_segment(root)
            source_pose = _pose_artifact(root, source)
            rgb = _rgb_observations(
                root,
                explicit_timestamps_ns=np.asarray(
                    [1_000_000_000, 2_000_000_000, 3_000_000_000],
                    dtype=np.int64,
                ),
            )
            with self.assertRaisesRegex(ArtifactError, "fraction gate failed"):
                derive_rgbd_segment(
                    source,
                    source_pose,
                    rgb,
                    root / "target",
                    root / "poses" / "target",
                    config=_config(max_depth_sync_residual_ns=0),
                )

    def test_declared_imu_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source_segment(root, imu=True)
            source_pose = _pose_artifact(root, source)
            rgb = _rgb_observations(root)
            with self.assertRaisesRegex(ArtifactError, "IMU-free"):
                derive_rgbd_segment(
                    source,
                    source_pose,
                    rgb,
                    root / "target",
                    root / "poses" / "target",
                    config=_config(),
                )

    def test_missing_explicit_depth_timestamp_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source_segment(root, include_depth_timing=False)
            source_pose = _pose_artifact(root, source)
            rgb = _rgb_observations(root)
            with self.assertRaisesRegex(ArtifactError, "depth timing evidence"):
                derive_rgbd_segment(
                    source,
                    source_pose,
                    rgb,
                    root / "target",
                    root / "poses" / "target",
                    config=_config(),
                )

    def test_target_georeferencing_is_rechecked_not_inherited_blindly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source_segment(root)
            source_pose = _pose_artifact(root, source, offset_m=0.20)
            rgb = _rgb_observations(root)
            destination = root / "rgbd-segment"
            destination_pose = root / "poses" / "target"
            with self.assertRaisesRegex(ArtifactError, "RTK residual gates"):
                derive_rgbd_segment(
                    source,
                    source_pose,
                    rgb,
                    destination,
                    destination_pose,
                    config=_config(),
                )
            self.assertFalse(destination.exists())
            self.assertFalse(destination_pose.exists())

    def test_distorted_pinhole_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            observations = _rgb_observations(Path(tmp), distorted_pinhole=True)
            with self.assertRaisesRegex(ArtifactError, "zero-distortion PINHOLE"):
                load_rectified_rgb_observations(observations)

    def test_existing_destination_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "target"
            destination.mkdir()
            marker = destination / "keep"
            marker.write_text("unchanged", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                derive_rgbd_segment(
                    root / "missing-source",
                    root / "missing-pose",
                    root / "missing-rgb",
                    destination,
                    root / "poses" / "target",
                    config=_config(),
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")


if __name__ == "__main__":
    unittest.main()
