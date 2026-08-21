import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from rtk_splat.core.segment import (
    POSITION_QUALITY_VOCABULARY,
    SegmentReader,
    SegmentWriter,
)
from rtk_splat.workflows.tiles import (
    _complete_core_training_coverage,
    _training_core_support,
    build_tile_plan,
    load_tile_execution,
    owner_tile_indices,
    verify_tile_plan,
)
from rtk_splat.workflows.cloud import (
    construct_initial_cloud,
    load_tile_context_masks,
    verify_tile_cloud,
)


def _viewmat(center, yaw):
    forward = np.asarray([np.cos(yaw), np.sin(yaw), 0.0])
    right = np.asarray([np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.asarray([0.0, 0.0, -1.0])
    rotation_wc = np.column_stack([right, down, forward])
    rotation_cw = rotation_wc.T
    result = np.eye(4)
    result[:3, :3] = rotation_cw
    result[:3, 3] = -rotation_cw @ np.asarray(center)
    return result


def _segment(destination: Path):
    writer = SegmentWriter(destination)
    writer.directory("images")
    writer.directory("depth")
    centers = []
    yaws = []
    for x in np.linspace(0, 6, 7):
        centers.append([x, 0, 1]); yaws.append(0.0)
    centers.extend(([6.5, 1, 1], [6.5, 3, 1]))
    yaws.extend((np.pi / 2, np.pi / 2))
    for x in np.linspace(6, 0, 7):
        centers.append([x, 4, 1]); yaws.append(np.pi)
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    timestamps = np.arange(n, dtype=np.int64) * 100_000_000 + 1_700_000_000_000_000_000
    left, right, depth_paths = [], [], []
    image = np.full((24, 32, 3), 100, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    for index in range(n):
        for side, records in (("left", left), ("right", right)):
            relative = f"images/{side}_{index:06d}.png"
            (writer.staging_dir / relative).write_bytes(encoded.tobytes())
            records.append(relative)
        relative = f"depth/{index:06d}.npz"
        with (writer.staging_dir / relative).open("wb") as stream:
            np.savez_compressed(
                stream,
                depth=np.full((24, 32), 2.0, dtype=np.float32),
                valid=np.ones((24, 32), dtype=bool),
            )
        depth_paths.append(relative)
    viewmats = np.asarray([
        _viewmat(center, yaw) for center, yaw in zip(centers, yaws)
    ])
    frames = {
        "frame_id": np.arange(n, dtype=np.int64),
        "timestamp_ns": timestamps,
        "left_image_path": np.asarray(left),
        "right_image_path": np.asarray(right),
        "right_timestamp_ns": timestamps + 100,
        "stereo_sync_residual_ns": np.full(n, 100, dtype=np.int64),
        "depth_path": np.asarray(depth_paths),
        "initial_viewmat": viewmats,
        "initial_camera_center_m": centers,
        "pose_valid": np.ones(n, dtype=bool),
    }
    camera = {
        "model": "PINHOLE", "width": 32, "height": 24,
        "K": [[28.0, 0, 16.0], [0, 28.0, 12.0], [0, 0, 1]],
        "distortion": [],
    }
    writer.write_frames(frames)
    writer.write_calibration({
        "contract_version": 2,
        "cameras": {"left": camera, "right": camera},
        "T_right_left": [[1, 0, 0, -0.12], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "transform_conventions": {"T_right_left": "right_from_left"},
    })
    writer.write_meta({
        "contract_version": 2,
        "n_frames": n,
        "capabilities": {
            "stereo": True, "rgbd": False, "single_rtk": True,
            "dual_rtk": False, "depth_recorded": False,
            "depth_computed": True, "imu_present": False,
            "images_raw": False, "images_rectified": True,
        },
        "coordinate_frame": {
            "type": "local_enu", "world_frame_id": "map", "units": "m",
            "origin_wgs84": {
                "latitude_deg": 52.0, "longitude_deg": 16.0,
                "ellipsoidal_altitude_m": 100.0, "ellipsoid": "WGS84",
                "vertical_datum": "WGS84 ellipsoid",
            },
        },
        "timebase": {
            "frame_timestamp_source": "left_header",
            "observation_timestamp_source": "gnss_header", "unit": "ns",
            "association_clock_offset_ns": 0,
        },
        "position_observation": {
            "type": "gnss", "quantity": "antenna_phase_center",
            "sensor_frame_id": "antenna", "coordinates": "ENU_m",
            "covariance_frame": "ENU_m2", "validity_field": "position_valid",
            "quality_field": "position_quality",
            "quality_vocabulary": list(POSITION_QUALITY_VOCABULARY),
        },
        "initial_pose": {
            "camera_frame_id": "left_camera",
            "position_quantity": "left_camera_center", "source": "synthetic",
            "lever_arm_applied": True,
            "extrinsic_translation_sigma_m": [0.01, 0.01, 0.01],
            "extrinsic_translation_sigma_frame_id": "left_camera",
        },
        "depth_observation": {
            "format": "npz_depth_valid", "units": "m",
            "quantity": "optical_z", "aligned_to": "left",
            "invalid_convention": "valid=false and depth=0",
        },
    })
    validation = [3, 11]
    writer.write_manifest({
        "train": [index for index in range(n) if index not in validation],
        "val": validation, "test": [],
    })
    common = {
        "frame_id": np.arange(n, dtype=np.int64),
        "frame_timestamp_ns": timestamps,
        "source_index": np.arange(n, dtype=np.int64),
        "source_timestamp_ns": timestamps,
    }
    writer.write_observations("gnss", {
        **common, "enu_m": centers,
        "covariance_enu_m2": np.repeat(np.eye(3)[None] * 0.0004, n, axis=0),
        "fix_status": np.ones(n, dtype=np.int16),
        "carrier_status": np.full(n, 2, dtype=np.int16),
        "position_valid": np.ones(n, dtype=bool),
        "position_quality": np.full(n, "rtk_fixed", dtype="<U13"),
        "raw_timestamp_ns": timestamps,
    })
    return writer.finalize()


def _cfg(workdir: Path, *, maximum=10):
    return SimpleNamespace(
        paths=SimpleNamespace(workdir=workdir),
        pose=SimpleNamespace(artifact="rtk"),
        cloud=SimpleNamespace(
            artifact_root=workdir / "cloud_artifacts",
            pixel_stride=2,
            voxel_m=0.02,
            max_points=100_000,
        ),
        train=SimpleNamespace(use_right_camera=False),
        tiles=SimpleNamespace(
            max_training_frames=maximum, max_tiles=8,
            max_support_samples=20_000, support_cell_m=0.25,
            min_visibility_fraction=0.01, min_visibility_cells=1,
            context_halo_m=0.25, minimum_core_train_frames=2,
            max_depthless_frame_fraction=0.10,
            min_core_train_support_coverage=0.50,
        ),
    )


def _reseal(artifact: Path, *names: str) -> None:
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in names:
        payload = (artifact / name).read_bytes()
        manifest["files"][name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _legacy_pose(reader: SegmentReader, parent: Path, name: str) -> Path:
    root = parent / name
    root.mkdir(parents=True)
    np.save(root / "viewmats.npy", reader.frames["initial_viewmat"])
    np.save(root / "cam_centers.npy", reader.frames["initial_camera_center_m"])
    (root / "quality.json").write_text(
        json.dumps({"rtk_alignment_passed": True}, sort_keys=True) + "\n"
    )
    files = {}
    for filename in ("viewmats.npy", "cam_centers.npy", "quality.json"):
        payload = (root / filename).read_bytes()
        files[filename] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    (root / "manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "name": name, "files": files},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return root


def _diagnostic_pose(reader: SegmentReader, parent: Path, name: str) -> Path:
    root = parent / name
    root.mkdir(parents=True)
    np.save(root / "viewmats.npy", reader.frames["initial_viewmat"])
    np.save(root / "cam_centers.npy", reader.frames["initial_camera_center_m"])
    status = {
        "artifact_class": "diagnostic_render_only",
        "georeferencing_status": "FAILED",
        "metric_georeferencing_claim_eligible": False,
        "diagnostic_export_requested": True,
        "diagnostic_export_override_used": True,
    }
    declaration = {"schema_version": 1, **status}
    (root / "quality.json").write_text(
        json.dumps(
            {**declaration, "rtk_alignment_passed": False},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for filename in (
        "alignment.json",
        "provenance.json",
        "georeferencing.json",
        "GEOREFERENCING_FAILED.json",
    ):
        (root / filename).write_text(
            json.dumps(declaration, indent=2, sort_keys=True) + "\n"
        )
    files = {}
    for path in sorted(root.iterdir()):
        if path.is_file():
            payload = path.read_bytes()
            files[path.name] = {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "name": name,
                **status,
                "files": files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return root


class TilePlanTests(unittest.TestCase):
    def test_training_coverage_excludes_heldout_only_support(self):
        support = [
            np.asarray([[0.5, 0.5]], dtype=np.float64),
            np.asarray([[1.5, 0.5]], dtype=np.float64),
        ]
        core = np.asarray([[0.0, 0.0], [2.0, 1.0]], dtype=np.float64)
        selected = np.asarray([True, True])
        split_codes = np.asarray([0, 1], dtype=np.uint8)
        all_core, train_coverable, covered, coverage = (
            _training_core_support(
                support, core, selected, split_codes
            )
        )
        self.assertEqual(len(all_core), 2)
        self.assertEqual(train_coverable.tolist(), [[0.5, 0.5]])
        self.assertEqual(covered.tolist(), [[0.5, 0.5]])
        self.assertEqual(coverage, 1.0)

    def test_core_coverage_completion_is_bounded_and_deterministic(self):
        support = [
            np.asarray([[0.5, 0.5]], dtype=np.float64),
            np.asarray([[1.5, 0.5]], dtype=np.float64),
            np.asarray([[1.5, 0.5]], dtype=np.float64),
        ]
        core = np.asarray([[0.0, 0.0], [2.0, 1.0]], dtype=np.float64)
        selected = np.asarray([True, False, False])
        split_codes = np.zeros(3, dtype=np.uint8)
        completed, added = _complete_core_training_coverage(
            support,
            core,
            selected,
            split_codes,
            max_training_frames=2,
            minimum_coverage=1.0,
        )
        self.assertEqual(completed.tolist(), [True, True, False])
        self.assertEqual(added.tolist(), [False, True, False])
        capped, capped_added = _complete_core_training_coverage(
            support,
            core,
            selected,
            split_codes,
            max_training_frames=1,
            minimum_coverage=1.0,
        )
        self.assertEqual(capped.tolist(), selected.tolist())
        self.assertFalse(capped_added.any())

    def test_diagnostic_pose_requires_permission_and_status_is_propagated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = _segment(root / "segment")
            pose_parent = root / "poses"
            _diagnostic_pose(reader, pose_parent, "diagnostic")
            cfg = _cfg(root / "work")
            cfg.pose.artifact = "diagnostic"
            cfg.pose.artifact_root = pose_parent
            with self.assertRaisesRegex(ValueError, "explicit render-only"):
                build_tile_plan(
                    reader,
                    cfg,
                    name="refused",
                    output_root=root / "work",
                    tile_count=2,
                )
            artifact = build_tile_plan(
                reader,
                cfg,
                name="diagnostic-plan",
                output_root=root / "work",
                tile_count=2,
                allow_failed_georeferencing_for_render=True,
            )
            plan = verify_tile_plan(
                artifact,
                segment=reader.root,
                pose_root=pose_parent / "diagnostic",
            )
            self.assertTrue(plan["diagnostic_render_only"])
            self.assertTrue(plan["provisional"])
            self.assertFalse(plan["metric_georeferencing_claim_eligible"])
            self.assertEqual(
                plan["evidence_tier"],
                "metric_depth_and_diagnostic_global_pose",
            )

    def test_tile_execution_builds_context_masked_plan_bound_cloud(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = _segment(root / "segment")
            cfg = _cfg(root / "work")
            artifact = build_tile_plan(
                reader,
                cfg,
                name="execution-v1",
                output_root=root / "work",
                tile_count=2,
            )
            execution = load_tile_execution(
                artifact, "tile-0000", reader, cfg
            )
            self.assertEqual(execution.binding["name"], "execution-v1")
            self.assertEqual(execution.binding["tile_id"], "tile-0000")
            self.assertEqual(len(execution.binding["manifest_sha256"]), 64)
            cloud, count = construct_initial_cloud(
                reader, cfg, tile_execution=execution
            )
            self.assertIn("tile_plans/execution-v1/tile-0000", str(cloud))
            self.assertGreater(count, 0)
            viewmats = reader.frames["initial_viewmat"]
            self.assertEqual(
                verify_tile_cloud(cloud, execution, viewmats),
                (count, len(set(execution.train_ids) | set(execution.val_ids))),
            )
            masks = load_tile_context_masks(cloud, execution)
            self.assertEqual(masks[execution.train_ids[0]].shape, (24, 32))
            changed = dict(execution.binding)
            changed["tile_id"] = "tile-0001"
            from rtk_splat.core.pose_artifacts import verify_cloud_tile_binding
            with self.assertRaisesRegex(ValueError, "different tile"):
                verify_cloud_tile_binding(cloud, changed)

    def test_sealed_two_tile_plan_is_deterministic_and_owns_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = _segment(root / "segment")
            first = build_tile_plan(
                reader, _cfg(root / "work-a"), name="synthetic-v1",
                output_root=root / "work-a", tile_count=2,
            )
            second = build_tile_plan(
                reader, _cfg(root / "work-b"), name="synthetic-v1",
                output_root=root / "work-b", tile_count=2,
            )
            self.assertEqual(
                (first / "tile_plan.json").read_bytes(),
                (second / "tile_plan.json").read_bytes(),
            )
            self.assertEqual(
                (first / "visibility.npz").read_bytes(),
                (second / "visibility.npz").read_bytes(),
            )
            plan = verify_tile_plan(first, segment=reader.root, rehash_sources=True)
            self.assertEqual(plan["summary"]["n_tiles"], 2)
            self.assertTrue(plan["provisional"])
            self.assertFalse(plan["metric_georeferencing_claim_eligible"])
            first_bounds = np.asarray(plan["tiles"][0]["core_bounds_uv_m"])
            second_bounds = np.asarray(plan["tiles"][1]["core_bounds_uv_m"])
            shared = None
            for axis in (0, 1):
                if np.isclose(first_bounds[1, axis], second_bounds[0, axis]):
                    shared = (axis, first_bounds[1, axis])
                if np.isclose(second_bounds[1, axis], first_bounds[0, axis]):
                    shared = (axis, second_bounds[1, axis])
            self.assertIsNotNone(shared)
            axis, value = shared
            uv = np.mean(np.asarray(plan["partition"]["scene_bounds_uv_m"]), axis=0)
            uv[axis] = value
            origin = np.asarray(plan["coordinate_frame"]["partition_origin_enu_m"])
            basis = np.asarray(plan["coordinate_frame"]["R_enu_from_partition"])
            enu = origin + basis @ np.asarray([uv[0], uv[1], 0.0])
            owner = owner_tile_indices(plan, enu[None])
            self.assertGreaterEqual(owner[0], 0)
            scene = np.asarray(plan["partition"]["scene_bounds_uv_m"])
            outer_uv = np.asarray([
                scene[0],
                [scene[0, 0], scene[1, 1]],
                [scene[1, 0], scene[0, 1]],
                scene[1],
            ])
            outer_xyz = np.column_stack([outer_uv, np.zeros(4)])
            outer_enu = origin + outer_xyz @ basis.T
            self.assertTrue(
                np.all(owner_tile_indices(plan, outer_enu) >= 0)
            )

    def test_tamper_existing_destination_and_capacity_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = _segment(root / "segment")
            work = root / "work"
            artifact = build_tile_plan(
                reader, _cfg(work), name="sealed", output_root=work,
                tile_count=2,
            )
            with self.assertRaises(FileExistsError):
                build_tile_plan(
                    reader, _cfg(work), name="sealed", output_root=work,
                    tile_count=2,
                )
            original = (artifact / "quality.json").read_bytes()
            (artifact / "quality.json").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "seal"):
                verify_tile_plan(artifact)
            (artifact / "quality.json").write_bytes(original)
            verify_tile_plan(artifact)
            with self.assertRaisesRegex(ValueError, "violates"):
                build_tile_plan(
                    reader, _cfg(root / "small", maximum=5), name="too-small",
                    output_root=root / "small", tile_count=1,
                )
            self.assertFalse((root / "small" / "tile_plan_artifacts" / "too-small").exists())

    def test_resealed_coverage_completion_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = _segment(root / "segment")
            artifact = build_tile_plan(
                reader,
                _cfg(root / "work"),
                name="coverage-selection",
                output_root=root / "work",
                tile_count=2,
            )
            visibility_path = artifact / "visibility.npz"
            with np.load(visibility_path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            self.assertIn("coverage_selected", arrays)
            arrays["coverage_selected"][0] = ~arrays[
                "coverage_selected"
            ][0]
            np.savez_compressed(visibility_path, **arrays)
            _reseal(artifact, "visibility.npz")
            with self.assertRaisesRegex(ValueError, "coverage-completion"):
                verify_tile_plan(artifact)

    def test_resealed_training_coverage_reference_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = _segment(root / "segment")
            artifact = build_tile_plan(
                reader,
                _cfg(root / "work"),
                name="coverage-reference",
                output_root=root / "work",
                tile_count=2,
            )
            plan_path = artifact / "tile_plan.json"
            plan = json.loads(plan_path.read_text())
            plan["tiles"][0]["summary"][
                "n_train_coverable_core_support_cells"
            ] += 1
            plan_path.write_text(
                json.dumps(plan, indent=2, sort_keys=True) + "\n"
            )
            _reseal(artifact, "tile_plan.json")
            with self.assertRaisesRegex(ValueError, "tile summary"):
                verify_tile_plan(artifact)

    def test_live_depth_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = _segment(root / "segment")
            work = root / "work"
            artifact = build_tile_plan(
                reader, _cfg(work), name="source-bound", output_root=work,
                tile_count=2,
            )
            first_depth = reader.root / str(reader.frames["depth_path"][0])
            first_depth.write_bytes(first_depth.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "depth content changed"):
                verify_tile_plan(
                    artifact, segment=reader.root, rehash_sources=True
                )

    def test_live_image_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = _segment(root / "segment")
            work = root / "work"
            artifact = build_tile_plan(
                reader, _cfg(work), name="image-bound", output_root=work,
                tile_count=2,
            )
            first_image = reader.root / str(reader.frames["left_image_path"][0])
            first_image.write_bytes(first_image.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "image content changed"):
                verify_tile_plan(
                    artifact, segment=reader.root, rehash_sources=True
                )

    def test_resealed_false_quality_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = _segment(root / "segment")
            artifact = build_tile_plan(
                reader, _cfg(root / "work"), name="semantic-seal",
                output_root=root / "work", tile_count=2,
            )
            quality_path = artifact / "quality.json"
            quality = json.loads(quality_path.read_text())
            quality["checks"]["all_support_inside_scene"] = False
            quality_path.write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n")
            _reseal(artifact, "quality.json")
            with self.assertRaisesRegex(ValueError, "quality claims"):
                verify_tile_plan(artifact)

    def test_resealed_pose_promotion_and_false_summaries_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = _segment(root / "segment")
            artifact = build_tile_plan(
                reader, _cfg(root / "work"), name="no-promotion",
                output_root=root / "work", tile_count=2,
            )
            plan_path = artifact / "tile_plan.json"
            provenance_path = artifact / "provenance.json"
            plan = json.loads(plan_path.read_text())
            provenance = json.loads(provenance_path.read_text())
            plan["metric_georeferencing_claim_eligible"] = True
            plan["provisional"] = False
            plan["evidence_tier"] = "metric_depth_and_production_global_pose"
            provenance["georeferencing"].update({
                "artifact_class": "production",
                "georeferencing_status": "PASSED",
                "metric_georeferencing_claim_eligible": True,
            })
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
            provenance_path.write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n"
            )
            _reseal(artifact, "tile_plan.json", "provenance.json")
            with self.assertRaisesRegex(ValueError, "cannot be promoted"):
                verify_tile_plan(artifact)

            # Restore a fresh artifact, then falsify only redundant summaries.
            other = build_tile_plan(
                reader, _cfg(root / "other"), name="false-summary",
                output_root=root / "other", tile_count=2,
            )
            other_plan_path = other / "tile_plan.json"
            other_plan = json.loads(other_plan_path.read_text())
            other_plan["summary"]["n_tiles"] = 99
            other_plan["source_binding"]["pose_fingerprint"] = "0" * 64
            other_plan_path.write_text(
                json.dumps(other_plan, indent=2, sort_keys=True) + "\n"
            )
            _reseal(other, "tile_plan.json")
            with self.assertRaisesRegex(ValueError, "pose identity"):
                verify_tile_plan(other)
            inventory = json.loads((other / "source_inventory.json").read_text())
            other_plan["source_binding"]["pose_fingerprint"] = inventory[
                "pose"
            ]["fingerprint"]
            other_plan_path.write_text(
                json.dumps(other_plan, indent=2, sort_keys=True) + "\n"
            )
            _reseal(other, "tile_plan.json")
            with self.assertRaisesRegex(ValueError, "support summary"):
                verify_tile_plan(other)

    def test_resealed_named_legacy_pose_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = _segment(root / "segment")
            pose_name = "legacy-named"
            pose_parent = root / "poses"
            _legacy_pose(reader, pose_parent, pose_name)
            cfg = _cfg(root / "work")
            cfg.pose.artifact = pose_name
            cfg.pose.artifact_root = pose_parent
            artifact = build_tile_plan(
                reader, cfg, name="named-no-promotion",
                output_root=root / "work", tile_count=2,
            )
            plan_path = artifact / "tile_plan.json"
            provenance_path = artifact / "provenance.json"
            plan = json.loads(plan_path.read_text())
            provenance = json.loads(provenance_path.read_text())
            plan.update({
                "metric_georeferencing_claim_eligible": True,
                "provisional": False,
                "evidence_tier": "metric_depth_and_production_global_pose",
            })
            provenance["georeferencing"].update({
                "artifact_class": "production",
                "georeferencing_status": "PASSED",
                "metric_georeferencing_claim_eligible": True,
            })
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
            provenance_path.write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n"
            )
            _reseal(artifact, "tile_plan.json", "provenance.json")
            with self.assertRaisesRegex(ValueError, "cannot be promoted"):
                verify_tile_plan(artifact)

    def test_resealed_support_diagnostics_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = _segment(root / "segment")
            artifact = build_tile_plan(
                reader, _cfg(root / "work"), name="support-diagnostics",
                output_root=root / "work", tile_count=2,
            )
            quality_path = artifact / "quality.json"
            quality = json.loads(quality_path.read_text())
            quality["support"]["n_support_cells"] = 1
            quality["support"]["sampled_depth_m"]["median"] = 999.0
            quality_path.write_text(
                json.dumps(quality, indent=2, sort_keys=True) + "\n"
            )
            _reseal(artifact, "quality.json")
            with self.assertRaisesRegex(ValueError, "support diagnostics"):
                verify_tile_plan(artifact)

    def test_resealed_mixed_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = _segment(root / "segment")
            artifact = build_tile_plan(
                reader, _cfg(root / "work"), name="mixed-schema",
                output_root=root / "work", tile_count=2,
            )
            for filename in ("quality.json", "provenance.json"):
                path = artifact / filename
                value = json.loads(path.read_text())
                value["schema_version"] = 2
                path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            _reseal(artifact, "quality.json", "provenance.json")
            with self.assertRaisesRegex(ValueError, "schema or quality"):
                verify_tile_plan(artifact)

    def test_minimum_core_training_support_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = _segment(root / "segment")
            cfg = _cfg(root / "work", maximum=100)
            cfg.tiles.minimum_core_train_frames = 100
            with self.assertRaisesRegex(ValueError, "minimum_core_training_support"):
                build_tile_plan(
                    reader, cfg, name="too-little-core", output_root=root / "work",
                    tile_count=1,
                )

    def test_bounded_depthless_frames_use_temporal_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = _segment(root / "segment")
            middle = reader.root / str(reader.frames["depth_path"][8])
            with middle.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    depth=np.zeros((24, 32), dtype=np.float32),
                    valid=np.zeros((24, 32), dtype=bool),
                )
            artifact = build_tile_plan(
                SegmentReader(reader.root).validate(),
                _cfg(root / "work"),
                name="depthless-bounded",
                output_root=root / "work",
                tile_count=2,
            )
            plan = verify_tile_plan(
                artifact, segment=reader.root, rehash_sources=True
            )
            quality = json.loads((artifact / "quality.json").read_text())
            self.assertEqual(
                quality["support"]["depth_support"]["depthless_frame_ids"], [8]
            )
            self.assertTrue(
                any(8 in tile["frame_ids"]["train"] for tile in plan["tiles"])
            )

    def test_depthless_fraction_above_gate_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = _segment(root / "segment")
            for frame_id in (7, 8):
                path = reader.root / str(reader.frames["depth_path"][frame_id])
                with path.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        depth=np.zeros((24, 32), dtype=np.float32),
                        valid=np.zeros((24, 32), dtype=bool),
                    )
            with self.assertRaisesRegex(ValueError, "depthless-frame fraction"):
                build_tile_plan(
                    SegmentReader(reader.root).validate(),
                    _cfg(root / "work"),
                    name="too-depthless",
                    output_root=root / "work",
                    tile_count=2,
                )

    def test_numeric_training_iterations_reduce_automatic_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = _segment(root / "segment")
            cfg = _cfg(root / "work", maximum=10)
            cfg.tiles.max_training_frames = "auto"
            cfg.train.iterations = 400
            cfg.derivation = SimpleNamespace(
                training=SimpleNamespace(
                    image_presentations_per_view=50.0,
                    max_iterations=65_000,
                )
            )
            with self.assertRaisesRegex(ValueError, "violates"):
                build_tile_plan(
                    reader,
                    cfg,
                    name="numeric-iteration-cap",
                    output_root=root / "work",
                    tile_count=1,
                )


if __name__ == "__main__":
    unittest.main()
