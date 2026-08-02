import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rtk_splat.backends.pose_evidence import (
    canonical_georeferencing_json,
    cloud_georeferencing_evidence,
    pose_georeferencing_evidence,
    splat_output_name,
    verify_pose_georeferencing_artifact,
    verify_training_run_georeferencing,
)
from rtk_splat.core.pose_artifacts import (
    cloud_path,
    load_pose_artifact,
    pose_fingerprint,
    verify_cloud_matches_poses,
)
from rtk_splat.workflows.cloud import construct_initial_cloud


def cfg(name=None, workdir=None):
    pose = SimpleNamespace()
    if name is not None:
        pose.artifact = name
    paths = SimpleNamespace(workdir=workdir) if workdir is not None else None
    return SimpleNamespace(pose=pose, paths=paths)


class PoseArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.seg = self.root / "segment"
        self.work = self.root / "work"
        self.seg.mkdir()
        (self.seg / "segment_meta.json").write_text(
            '{"contract_version": 2, "n_frames": 3}\n')
        self.viewmats = np.repeat(np.eye(4)[None], 3, axis=0)
        self.viewmats[:, 0, 3] = [0.0, -1.0, -2.0]
        self.centers = np.linalg.inv(self.viewmats)[:, :3, 3]
        self._write_initial_poses(self.viewmats)

    def _write_initial_poses(self, viewmats):
        np.savez_compressed(
            self.seg / "frames.npz",
            frame_id=np.arange(3, dtype=np.int64),
            timestamp_ns=np.arange(3, dtype=np.int64),
            left_image_path=np.asarray(
                [f"images/left_{index:06d}.jpg" for index in range(3)]
            ),
            initial_viewmat=viewmats,
            initial_camera_center_m=np.linalg.inv(viewmats)[:, :3, 3],
            pose_valid=np.ones(3, dtype=bool),
        )

    @staticmethod
    def _sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _reseal(self, sidecar, filename):
        manifest_path = sidecar / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        changed = sidecar / filename
        manifest["files"][filename] = {
            "sha256": self._sha256(changed),
            "size_bytes": changed.stat().st_size,
        }
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    def _write_modern_sidecar(
        self,
        name="diagnostic",
        *,
        status="FAILED",
        artifact_class="diagnostic_render_only",
        eligible=False,
    ):
        sidecar = self.work / "pose_artifacts" / name
        sidecar.mkdir(parents=True)
        np.save(sidecar / "viewmats.npy", self.viewmats)
        np.save(sidecar / "cam_centers.npy", self.centers)
        declaration = {
            "schema_version": 1,
            "artifact_class": artifact_class,
            "georeferencing_status": status,
            "metric_georeferencing_claim_eligible": eligible,
            "diagnostic_export_requested": (
                artifact_class == "diagnostic_render_only"
            ),
            "diagnostic_export_override_used": status == "FAILED",
            "holdout_median_residual_m": 0.23,
            "holdout_median_gate_m": 0.15,
        }
        (sidecar / "quality.json").write_text(
            json.dumps(
                {
                    **declaration,
                    "rtk_alignment_passed": status == "PASSED",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        for filename in ("alignment.json", "provenance.json"):
            (sidecar / filename).write_text(
                json.dumps(declaration, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (sidecar / "georeferencing.json").write_text(
            json.dumps(declaration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if status == "FAILED":
            (sidecar / "GEOREFERENCING_FAILED.json").write_text(
                json.dumps(declaration, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        files = {}
        for path in sorted(sidecar.iterdir()):
            if path.is_file():
                files[path.name] = {
                    "sha256": self._sha256(path),
                    "size_bytes": path.stat().st_size,
                }
        (sidecar / "manifest.json").write_text(
            json.dumps({**declaration, "name": name, "files": files})
            + "\n",
            encoding="utf-8",
        )
        return sidecar, declaration

    def tearDown(self):
        self.tmp.cleanup()

    def test_config_without_artifact_uses_contract_initial_poses(self):
        viewmats, centers = load_pose_artifact(self.seg, cfg(workdir=self.work))
        np.testing.assert_allclose(viewmats, self.viewmats)
        np.testing.assert_allclose(centers, self.centers)
        self.assertEqual(
            cloud_path(self.seg, cfg(workdir=self.work)),
            self.work / "cloud_artifacts" / "rtk" / "init_cloud.npz",
        )

    def test_named_sidecar_never_falls_back_to_raw(self):
        with self.assertRaisesRegex(FileNotFoundError, "selected pose backend"):
            load_pose_artifact(
                self.seg, cfg("colmap_stereo", workdir=self.work)
            )

    def test_named_sidecar_and_cloud_are_isolated(self):
        sidecar = self.work / "pose_artifacts" / "colmap_stereo"
        sidecar.mkdir(parents=True)
        refined = self.viewmats.copy()
        refined[:, 1, 3] = -0.1
        centres = np.linalg.inv(refined)[:, :3, 3]
        np.save(sidecar / "viewmats.npy", refined)
        np.save(sidecar / "cam_centers.npy", centres)
        selected = cfg("colmap_stereo", workdir=self.work)
        loaded, _ = load_pose_artifact(self.seg, selected)
        cloud = cloud_path(self.seg, selected)
        cloud.parent.mkdir(parents=True)
        np.savez(cloud, xyz=np.zeros((1, 3)), rgb=np.zeros((1, 3)),
                 pose_fingerprint=np.asarray(pose_fingerprint(loaded)))
        verify_cloud_matches_poses(cloud, loaded, require_fingerprint=True)
        np.testing.assert_allclose(
            np.load(self.seg / "frames.npz")["initial_viewmat"], self.viewmats
        )

    def test_failed_diagnostic_requires_explicit_render_permission(self):
        self._write_modern_sidecar()
        selected = cfg("diagnostic", workdir=self.work)
        evidence = pose_georeferencing_evidence(self.seg, selected)
        self.assertEqual(evidence["artifact_class"], "diagnostic_render_only")
        self.assertEqual(evidence["georeferencing_status"], "FAILED")
        self.assertFalse(evidence["metric_georeferencing_claim_eligible"])
        self.assertEqual(len(evidence["pose_manifest_sha256"]), 64)
        self.assertEqual(len(evidence["pose_quality_sha256"]), 64)
        with self.assertRaisesRegex(ValueError, "explicit render-only"):
            load_pose_artifact(self.seg, selected)
        loaded, centers = load_pose_artifact(
            self.seg,
            selected,
            allow_failed_georeferencing_for_render=True,
        )
        np.testing.assert_allclose(loaded, self.viewmats)
        np.testing.assert_allclose(centers, self.centers)

    def test_modern_georeferencing_declaration_is_manifest_sealed(self):
        sidecar, _ = self._write_modern_sidecar()
        selected = cfg("diagnostic", workdir=self.work)
        (sidecar / "quality.json").write_text(
            '{"rtk_alignment_passed": true}\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "file changed"):
            pose_georeferencing_evidence(self.seg, selected)

    def test_public_pose_verifier_checks_manifest_name(self):
        sidecar, _ = self._write_modern_sidecar()
        evidence = verify_pose_georeferencing_artifact(
            sidecar, expected_name="diagnostic"
        )
        self.assertEqual(evidence["pose_artifact"], "diagnostic")
        with self.assertRaisesRegex(ValueError, "does not match"):
            verify_pose_georeferencing_artifact(
                sidecar, expected_name="different"
            )

    def test_public_pose_verifier_rejects_sealed_status_mismatches(self):
        sidecar, _ = self._write_modern_sidecar()
        alignment_path = sidecar / "alignment.json"
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
        alignment["georeferencing_status"] = "PASSED"
        alignment_path.write_text(json.dumps(alignment) + "\n", encoding="utf-8")
        self._reseal(sidecar, "alignment.json")
        with self.assertRaisesRegex(ValueError, "alignment.*disagree"):
            verify_pose_georeferencing_artifact(sidecar)

        marker_sidecar, _ = self._write_modern_sidecar(name="marker-mismatch")
        marker_path = marker_sidecar / "GEOREFERENCING_FAILED.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["artifact_class"] = "production"
        marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
        self._reseal(marker_sidecar, "GEOREFERENCING_FAILED.json")
        with self.assertRaisesRegex(ValueError, "failed marker.*disagree"):
            verify_pose_georeferencing_artifact(marker_sidecar)

        impossible, _ = self._write_modern_sidecar(
            name="production-failed",
            status="FAILED",
            artifact_class="production",
            eligible=False,
        )
        with self.assertRaisesRegex(ValueError, "production.*tuple"):
            verify_pose_georeferencing_artifact(impossible)

    def test_passed_diagnostic_still_requires_explicit_permission_and_name(self):
        self._write_modern_sidecar(status="PASSED")
        selected = cfg("diagnostic", workdir=self.work)
        evidence = pose_georeferencing_evidence(self.seg, selected)
        self.assertEqual(evidence["georeferencing_status"], "PASSED")
        self.assertEqual(splat_output_name(evidence), "splat.DIAGNOSTIC_ONLY.ply")
        with self.assertRaisesRegex(ValueError, "explicit render-only"):
            load_pose_artifact(self.seg, selected)
        load_pose_artifact(
            self.seg,
            selected,
            allow_failed_georeferencing_for_render=True,
        )

    def test_legacy_named_sidecar_remains_usable_but_unassessed(self):
        sidecar = self.work / "pose_artifacts" / "legacy"
        sidecar.mkdir(parents=True)
        np.save(sidecar / "viewmats.npy", self.viewmats)
        np.save(sidecar / "cam_centers.npy", self.centers)
        selected = cfg("legacy", workdir=self.work)
        evidence = pose_georeferencing_evidence(self.seg, selected)
        self.assertEqual(evidence["georeferencing_status"], "legacy_unassessed")
        self.assertFalse(evidence["metric_georeferencing_claim_eligible"])
        load_pose_artifact(self.seg, selected)

    def test_legacy_failed_quality_cannot_enter_cloud_without_permission(self):
        sidecar = self.work / "pose_artifacts" / "legacy-failed"
        sidecar.mkdir(parents=True)
        np.save(sidecar / "viewmats.npy", self.viewmats)
        np.save(sidecar / "cam_centers.npy", self.centers)
        (sidecar / "quality.json").write_text(
            '{"rtk_alignment_passed": false}\n', encoding="utf-8"
        )
        selected = cfg("legacy-failed", workdir=self.work)
        reader = SimpleNamespace(root=self.seg, frames={})
        with self.assertRaisesRegex(ValueError, "explicit render-only"):
            construct_initial_cloud(reader, selected)

    def test_cloud_retains_failed_georeferencing_provenance(self):
        self._write_modern_sidecar()
        selected = cfg("diagnostic", workdir=self.work)
        evidence = pose_georeferencing_evidence(self.seg, selected)
        cloud = cloud_path(self.seg, selected)
        cloud.parent.mkdir(parents=True)
        np.savez(
            cloud,
            xyz=np.zeros((1, 3)),
            rgb=np.zeros((1, 3)),
            pose_fingerprint=np.asarray(pose_fingerprint(self.viewmats)),
            georeferencing_json=np.asarray(
                canonical_georeferencing_json(evidence)
            ),
            artifact_class=np.asarray(evidence["artifact_class"]),
            georeferencing_status=np.asarray(evidence["georeferencing_status"]),
            metric_georeferencing_claim_eligible=np.asarray(False),
            pose_quality_sha256=np.asarray(evidence["pose_quality_sha256"]),
            pose_manifest_sha256=np.asarray(evidence["pose_manifest_sha256"]),
            pose_georeferencing_sha256=np.asarray(
                evidence["pose_georeferencing_sha256"]
            ),
        )
        with self.assertRaisesRegex(ValueError, "explicit render-only"):
            cloud_georeferencing_evidence(
                cloud,
                evidence,
                allow_failed_georeferencing_for_render=False,
            )
        verify_cloud_matches_poses(
            cloud,
            self.viewmats,
            require_fingerprint=True,
        )
        stored = cloud_georeferencing_evidence(
            cloud,
            evidence,
            allow_failed_georeferencing_for_render=True,
        )
        self.assertEqual(stored, evidence)

    def test_cloud_cannot_drop_failed_georeferencing_status(self):
        self._write_modern_sidecar()
        selected = cfg("diagnostic", workdir=self.work)
        evidence = pose_georeferencing_evidence(self.seg, selected)
        cloud = cloud_path(self.seg, selected)
        cloud.parent.mkdir(parents=True)
        np.savez(
            cloud,
            xyz=np.zeros((1, 3)),
            rgb=np.zeros((1, 3)),
            pose_fingerprint=np.asarray(pose_fingerprint(self.viewmats)),
        )
        with self.assertRaisesRegex(ValueError, "dropped pose georeferencing"):
            cloud_georeferencing_evidence(
                cloud,
                evidence,
                allow_failed_georeferencing_for_render=True,
            )

    def test_raw_pose_cloud_provenance_uses_pickle_free_null_hashes(self):
        selected = cfg(workdir=self.work)
        evidence = pose_georeferencing_evidence(self.seg, selected)
        cloud = cloud_path(self.seg, selected)
        cloud.parent.mkdir(parents=True)
        np.savez(
            cloud,
            pose_fingerprint=np.asarray(pose_fingerprint(self.viewmats)),
            georeferencing_json=np.asarray(
                canonical_georeferencing_json(evidence)
            ),
            artifact_class=np.asarray(evidence["artifact_class"]),
            georeferencing_status=np.asarray(evidence["georeferencing_status"]),
            metric_georeferencing_claim_eligible=np.asarray(False),
            pose_quality_sha256=np.asarray(""),
            pose_manifest_sha256=np.asarray(""),
            pose_georeferencing_sha256=np.asarray(""),
        )
        stored = cloud_georeferencing_evidence(
            cloud,
            evidence,
            allow_failed_georeferencing_for_render=False,
        )
        self.assertEqual(stored, evidence)
        self.assertEqual(splat_output_name(evidence), "splat.ply")

    def test_training_run_verifier_checks_diagnostic_ply_and_all_sidecars(self):
        sidecar, _ = self._write_modern_sidecar()
        evidence = verify_pose_georeferencing_artifact(sidecar)
        run = self.work / "runs" / "diagnostic"
        run.mkdir(parents=True)
        evidence_text = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        (run / "georeferencing.json").write_text(
            evidence_text, encoding="utf-8"
        )
        (run / "GEOREFERENCING_FAILED.json").write_text(
            evidence_text, encoding="utf-8"
        )
        (run / "run_provenance.json").write_text(
            json.dumps(
                {
                    "pose_artifact": evidence["pose_artifact"],
                    "georeferencing": evidence,
                }
            ),
            encoding="utf-8",
        )
        ply = run / "splat.DIAGNOSTIC_ONLY.ply"
        ply.write_bytes(b"ply\nnot-empty\n")
        splat_evidence = {
            **evidence,
            "splat_file": ply.name,
            "splat_sha256": self._sha256(ply),
        }
        (run / "splat.georeferencing.json").write_text(
            json.dumps(splat_evidence), encoding="utf-8"
        )
        report = verify_training_run_georeferencing(run, evidence)
        self.assertEqual(report["splat_sha256"], self._sha256(ply))
        self.assertEqual(report["georeferencing"], evidence)

        (run / "splat.ply").write_bytes(b"ambiguous")
        with self.assertRaisesRegex(ValueError, "ambiguous alternate"):
            verify_training_run_georeferencing(run, evidence)
        (run / "splat.ply").unlink()
        ply.write_bytes(b"ply\ntampered\n")
        with self.assertRaisesRegex(ValueError, "file/hash evidence"):
            verify_training_run_georeferencing(run, evidence)

    def test_cloud_pose_mismatch_fails(self):
        cloud = self.work / "cloud_artifacts" / "rtk" / "init_cloud.npz"
        cloud.parent.mkdir(parents=True)
        np.savez(cloud, xyz=np.zeros((1, 3)), rgb=np.zeros((1, 3)),
                 pose_fingerprint=np.asarray("wrong"))
        with self.assertRaisesRegex(ValueError, "different poses"):
            verify_cloud_matches_poses(
                cloud, self.viewmats, require_fingerprint=False)

    def test_invalid_rotation_fails(self):
        bad = self.viewmats.copy()
        bad[1, 0, 0] = 2.0
        self._write_initial_poses(bad)
        with self.assertRaisesRegex(ValueError, "non-orthonormal"):
            load_pose_artifact(self.seg, cfg(workdir=self.work))


if __name__ == "__main__":
    unittest.main()
