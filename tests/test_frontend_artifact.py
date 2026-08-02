import hashlib
import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rtk_splat.frontends.artifact import (
    ArtifactError,
    FrontendArtifactBuilder,
    StageLedger,
    collect_git_state,
    collect_provenance,
    create_database_snapshot,
    probe_colmap_identity,
    stage_fingerprint,
)
from rtk_splat.core.segment import POSITION_QUALITY_VOCABULARY, SegmentWriter


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _segment(path: Path) -> Path:
    writer = SegmentWriter(path)
    images = writer.directory("images")
    n = 2
    left_payloads = [b"left-0", b"left-1"]
    right_payloads = [b"right-0", b"right-1"]
    for index in range(n):
        (images / f"left_{index:06d}.jpg").write_bytes(left_payloads[index])
        (images / f"right_{index:06d}.jpg").write_bytes(right_payloads[index])
    left_ns = np.array([1_000_000_000, 1_100_000_000], dtype=np.int64)
    right_ns = left_ns + 2_000
    writer.write_frames(
        {
            "frame_id": np.arange(n, dtype=np.int64),
            "timestamp_ns": left_ns,
            "left_image_path": np.array(
                [f"images/left_{index:06d}.jpg" for index in range(n)]
            ),
            "right_image_path": np.array(
                [f"images/right_{index:06d}.jpg" for index in range(n)]
            ),
            "right_timestamp_ns": right_ns,
            "stereo_sync_residual_ns": right_ns - left_ns,
            "stereo_left_sha256": np.array(
                [_digest(payload) for payload in left_payloads]
            ),
            "stereo_right_sha256": np.array(
                [_digest(payload) for payload in right_payloads]
            ),
        }
    )
    camera = {
        "model": "PINHOLE",
        "width": 8,
        "height": 6,
        "K": [[10.0, 0.0, 4.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]],
        "distortion": [],
    }
    writer.write_calibration(
        {
            "contract_version": 2,
            "cameras": {"left": camera, "right": camera},
            "T_right_left": [
                [1.0, 0.0, 0.0, -0.12],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "transform_conventions": {"T_right_left": "right_from_left"},
        }
    )
    writer.write_meta(
        {
            "contract_version": 2,
            "n_frames": n,
            "capabilities": {
                "stereo": True,
                "rgbd": False,
                "single_rtk": True,
                "dual_rtk": False,
                "depth_recorded": False,
                "depth_computed": False,
                "imu_present": False,
                "images_raw": False,
                "images_rectified": True,
            },
            "coordinate_frame": {
                "type": "local_enu",
                "world_frame_id": "map",
                "units": "m",
                "origin_wgs84": {
                    "latitude_deg": 52.0,
                    "longitude_deg": 16.0,
                    "ellipsoidal_altitude_m": 100.0,
                    "ellipsoid": "WGS84",
                    "vertical_datum": "WGS84 ellipsoid",
                },
            },
            "timebase": {
                "frame_timestamp_source": "left_camera_header",
                "observation_timestamp_source": "sensor_header",
                "unit": "ns",
                "association_clock_offset_ns": 0,
            },
            "position_observation": {
                "type": "gnss",
                "quantity": "antenna_phase_center",
                "sensor_frame_id": "primary_antenna",
                "coordinates": "ENU_m",
                "covariance_frame": "ENU_m2",
                "validity_field": "position_valid",
                "quality_field": "position_quality",
                "quality_vocabulary": list(POSITION_QUALITY_VOCABULARY),
            },
        }
    )
    writer.write_manifest({"train": [0, 1], "val": [], "test": []})
    writer.write_observations(
        "gnss",
        {
            "frame_id": np.arange(n, dtype=np.int64),
            "frame_timestamp_ns": left_ns,
            "source_index": np.arange(n, dtype=np.int64),
            "source_timestamp_ns": left_ns,
            "enu_m": np.zeros((n, 3), dtype=np.float64),
            "covariance_enu_m2": np.repeat(
                (np.eye(3) * 0.0004)[None], n, axis=0
            ),
            "fix_status": np.ones(n, dtype=np.int16),
            "carrier_status": np.full(n, 2, dtype=np.int16),
            "position_valid": np.ones(n, dtype=bool),
            "position_quality": np.array(["rtk_fixed"] * n, dtype=np.str_),
        },
    )
    return writer.finalize().root


def _git_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"], check=True
    )
    tracked = path / "tracked.txt"
    tracked.write_text("committed\n")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-qm", "initial"], check=True
    )
    tracked.write_text("dirty\n")
    (path / "untracked.txt").write_text("also dirty\n")
    return path


def _provenance(segment: Path, repo: Path) -> dict:
    configuration = {
        "schema_version": 1,
        "effective_config": {"frontend": {"seed": 7, "feature": "sift"}},
        "runtime_resolution": {
            "derivations": {
                "frame_stride": {"source": "derived", "chosen_value": 2}
            }
        },
        "effective_config_sha256": "1" * 64,
    }
    return collect_provenance(
        segment,
        resolved_config={"frontend": {"seed": 7, "feature": "sift"}},
        configuration=configuration,
        colmap={"executable": "/opt/colmap", "version": "COLMAP 4.1.1"},
        seed=7,
        repo_root=repo,
    )


def _build(
    segment: Path, output: Path, name: str, provenance: dict
) -> Path:
    return FrontendArtifactBuilder(segment, output, name).build(
        rig_config={"cameras": ["left", "right"], "rig": "stereo"},
        keyframes={"frame_ids": [0, 1], "selector": "all"},
        pairs=[
            ("right_000001.jpg", "left_000001.jpg"),
            ("left_000000.jpg", "right_000000.jpg"),
        ],
        provenance=provenance,
        quality={"status": "prepared"},
    )


class FrontendArtifactTests(unittest.TestCase):
    def test_atomic_builder_emits_deterministic_sealed_foundation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment = _segment(root / "segment")
            repo = _git_repo(root / "repo")
            provenance = _provenance(segment, repo)
            source_entries = sorted(
                item.relative_to(segment).as_posix()
                for item in segment.rglob("*")
            )

            first = _build(segment, root / "output-a", "gpu_all", provenance)
            second = _build(segment, root / "output-b", "gpu_all", provenance)

            for filename in (
                "frame_manifest.json",
                "rig_config.json",
                "keyframes.json",
                "pairs.txt",
                "provenance.json",
                "quality.json",
            ):
                self.assertTrue((first / filename).is_file())
                self.assertEqual(
                    (first / filename).read_bytes(),
                    (second / filename).read_bytes(),
                )
            links = sorted((first / "images").iterdir())
            self.assertEqual(len(links), 4)
            self.assertTrue(all(path.is_symlink() for path in links))
            manifest = json.loads((first / "frame_manifest.json").read_text())
            self.assertEqual(
                manifest["frames"][0]["left_image"]["hash_source"],
                "frames.npz:stereo_left_sha256",
            )
            self.assertEqual(manifest["frames"][0]["right_timestamp_ns"], 1_000_002_000)
            self.assertEqual(manifest["frames"][0]["stereo_sync_residual_ns"], 2_000)
            self.assertEqual(
                (first / "pairs.txt").read_text(),
                "left_000000.jpg right_000000.jpg\n"
                "left_000001.jpg right_000001.jpg\n",
            )
            stored_provenance = json.loads(
                (first / "provenance.json").read_text()
            )
            self.assertEqual(
                stored_provenance["configuration"]["runtime_resolution"]
                ["derivations"]["frame_stride"]["chosen_value"],
                2,
            )
            self.assertEqual(
                stored_provenance["configuration"]
                ["effective_config_sha256"],
                "1" * 64,
            )
            self.assertEqual(
                source_entries,
                sorted(
                    item.relative_to(segment).as_posix()
                    for item in segment.rglob("*")
                ),
            )

    def test_existing_artifact_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment = _segment(root / "segment")
            repo = _git_repo(root / "repo")
            provenance = _provenance(segment, repo)
            artifact = _build(segment, root / "output", "sealed", provenance)
            marker = artifact / "keep"
            marker.write_text("unchanged")
            with self.assertRaises(FileExistsError):
                _build(segment, root / "output", "sealed", provenance)
            self.assertEqual(marker.read_text(), "unchanged")

    def test_stale_provenance_fails_without_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment = _segment(root / "segment")
            repo = _git_repo(root / "repo")
            provenance = _provenance(segment, repo)
            calibration = json.loads((segment / "calibration.json").read_text())
            calibration["provenance"] = {"changed": True}
            (segment / "calibration.json").write_text(
                json.dumps(calibration, sort_keys=True)
            )
            output = root / "output"
            with self.assertRaisesRegex(ArtifactError, "changed"):
                _build(segment, output, "stale", provenance)
            self.assertFalse((output / "frontend_artifacts" / "stale").exists())

    def test_invalid_pair_cleans_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment = _segment(root / "segment")
            repo = _git_repo(root / "repo")
            provenance = _provenance(segment, repo)
            builder = FrontendArtifactBuilder(segment, root / "output", "bad")
            with self.assertRaisesRegex(ArtifactError, "invalid image pair"):
                builder.build(
                    rig_config={},
                    keyframes={"frame_ids": [0]},
                    pairs=[("left_000000.jpg", "missing.jpg")],
                    provenance=provenance,
                    quality={},
                )
            self.assertFalse(builder.destination.exists())
            self.assertEqual(
                list(builder.destination.parent.glob(".bad.writing-*")), []
            )


class ProvenanceTests(unittest.TestCase):
    def test_git_state_covers_tracked_and_untracked_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _git_repo(Path(tmp) / "repo")
            state = collect_git_state(repo)
            self.assertTrue(state["dirty"])
            self.assertEqual(len(state["commit"]), 40)
            self.assertEqual(len(state["tree"]), 40)
            self.assertRegex(state["dirty_diff_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(state["untracked"][0]["path"], "untracked.txt")
            before = state["dirty_diff_sha256"]
            (repo / "untracked.txt").write_text("different\n")
            self.assertNotEqual(
                before, collect_git_state(repo)["dirty_diff_sha256"]
            )

    def test_colmap_identity_records_version_and_binary_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "colmap"
            executable.write_bytes(b"fake-colmap")
            identity = probe_colmap_identity(
                executable, version_output="COLMAP 4.1.1 -- Structure-from-Motion"
            )
            self.assertEqual(identity["version"], "COLMAP 4.1.1 -- Structure-from-Motion")
            self.assertEqual(identity["executable_sha256"], _digest(b"fake-colmap"))

    def test_stage_fingerprint_is_mapping_order_independent(self):
        self.assertEqual(
            stage_fingerprint("features", {"a": 1, "b": [2, 3]}),
            stage_fingerprint("features", {"b": [2, 3], "a": 1}),
        )


class StageAndSnapshotTests(unittest.TestCase):
    def test_stage_resume_requires_identical_inputs_and_verifies_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"
            (artifact / "stages").mkdir(parents=True)
            ledger = StageLedger(artifact)
            inputs = {"config_sha256": "a" * 64, "seed": 7}
            self.assertEqual(ledger.begin("features", inputs), "run")
            self.assertEqual(ledger.begin("features", inputs), "resume")
            with self.assertRaisesRegex(ArtifactError, "inputs differ"):
                ledger.begin("features", {"config_sha256": "b" * 64, "seed": 7})

            output = artifact / "database.db"
            output.write_bytes(b"database")
            ledger.complete("features", inputs, ["database.db"])
            self.assertEqual(ledger.begin("features", inputs), "complete")
            output.write_bytes(b"mutated")
            with self.assertRaisesRegex(ArtifactError, "output changed"):
                ledger.begin("features", inputs)

    def test_backend_database_snapshot_is_atomic_verified_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "frontend.db"
            with sqlite3.connect(source) as connection:
                connection.execute("CREATE TABLE evidence(value TEXT NOT NULL)")
                connection.execute("INSERT INTO evidence VALUES ('sealed')")
            destination = root / "global-input"
            record = create_database_snapshot(
                source, destination, backend="global_mapper"
            )
            expected = _digest(source.read_bytes())
            self.assertEqual(record["source_sha256_before"], expected)
            self.assertEqual(record["source_sha256_after"], expected)
            self.assertEqual(
                record["snapshot_sha256"],
                _digest((destination / "database.db").read_bytes()),
            )
            with sqlite3.connect(destination / "database.db") as connection:
                self.assertEqual(
                    connection.execute("SELECT value FROM evidence").fetchone()[0],
                    "sealed",
                )
            with self.assertRaises(FileExistsError):
                create_database_snapshot(
                    source, destination, backend="global_mapper"
                )

    def test_snapshot_preserves_committed_rows_present_only_in_wal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "frontend.db"
            writer = sqlite3.connect(source)
            try:
                self.assertEqual(
                    writer.execute("PRAGMA journal_mode=WAL").fetchone()[0],
                    "wal",
                )
                writer.execute("PRAGMA wal_autocheckpoint=0")
                writer.execute("CREATE TABLE evidence(value TEXT NOT NULL)")
                writer.commit()
                writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                main_before = _digest(source.read_bytes())
                writer.execute("INSERT INTO evidence VALUES ('wal-only')")
                writer.commit()
                self.assertEqual(_digest(source.read_bytes()), main_before)

                destination = root / "snapshot"
                create_database_snapshot(
                    source, destination, backend="global_mapper"
                )
                with sqlite3.connect(destination / "database.db") as connection:
                    self.assertEqual(
                        connection.execute(
                            "SELECT value FROM evidence"
                        ).fetchall(),
                        [("wal-only",)],
                    )
            finally:
                writer.close()


if __name__ == "__main__":
    unittest.main()
