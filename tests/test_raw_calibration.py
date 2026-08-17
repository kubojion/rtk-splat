import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rtk_splat.diagnostics.raw_calibration import (
    _PLAN_FILES,
    _PLAN_KIND,
    _audit_config_record,
    _seal_files,
    atomic_raw_calibration_artifact,
    audited_raw_calibration_plan,
    baseline_initial_rotation,
)
from rtk_splat.diagnostics.calibration_sidecar import AuditConfig
from rtk_splat.diagnostics.metric_calibration import SolverSettings
from rtk_splat.frontends.artifact import ArtifactError, canonical_hash


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _minimal_plan(root: Path) -> None:
    manifest = {}
    bag_audit = {}
    selection = {
        "frames": [{"frame_id": 4}],
        "frame_ids_sha256": canonical_hash([4]),
    }
    body = {
        "schema_version": 1,
        "kind": _PLAN_KIND,
        "input_manifest_sha256": canonical_hash(manifest),
        "bag_audit_sha256": canonical_hash(bag_audit),
        "selection_sha256": canonical_hash(selection),
        "selected_frame_count": 1,
        "selected_frame_ids_sha256": canonical_hash([4]),
        "audit_config": {},
        "topics": {},
        "sim3_diagnostic_scale_range": [0.98, 1.02],
        "fixed_scale": 1.0,
        "uses_imu": False,
        "uses_production_holdout": False,
        "publishes_poses": False,
    }
    plan = {**body, "plan_sha256": canonical_hash(body)}
    _write_json(root / "input_manifest.json", manifest)
    _write_json(root / "bag_audit.json", bag_audit)
    _write_json(root / "selection.json", selection)
    _write_json(root / "calibration_plan.json", plan)
    _write_json(root / "plan_seal.json", _seal_files(root, _PLAN_FILES))


def test_baseline_initial_rotation_uses_heading_excitation_only():
    count = 80
    angles = np.linspace(-1.3, 1.4, count)
    visual_from_camera = Rotation.from_euler("z", angles).as_matrix()
    baseline_camera = np.asarray([1.2, 0.04, -0.02])
    predicted = np.einsum("nij,j->ni", visual_from_camera, baseline_camera)
    expected = Rotation.from_euler("xyz", [4.0, -7.0, 31.0], degrees=True).as_matrix()
    measured = np.einsum("ij,nj->ni", expected, predicted)
    times = np.arange(count, dtype=float) * 0.2
    data = SimpleNamespace(
        camera_t_s=times,
        baseline_t_s=times,
        baseline_good=np.ones(count, dtype=bool),
        visual_from_camera_rotation=visual_from_camera,
        baseline_enu_m=measured,
        baseline_cov_enu_m2=np.repeat(np.eye(3)[None] * 1.0e-4, count, axis=0),
    )

    rotation, report = baseline_initial_rotation(
        data, baseline_camera, maximum_age_s=0.01
    )

    np.testing.assert_allclose(rotation, expected, atol=1.0e-10)
    assert report["uses_positions"] is False
    assert report["uses_production_holdout"] is False
    assert report["fixed_scale"] == 1.0
    assert report["post_alignment_angle_p95_deg"] < 1.0e-5


def test_atomic_raw_artifact_refuses_overwrite_and_cleans_failure(tmp_path):
    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected"):
        with atomic_raw_calibration_artifact(failed) as staging:
            (staging / "partial").write_text("partial")
            raise RuntimeError("injected")
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed.writing-*"))

    published = tmp_path / "published"
    with atomic_raw_calibration_artifact(published) as staging:
        (staging / "complete").write_text("complete")
    assert (published / "complete").read_text() == "complete"
    with pytest.raises(FileExistsError):
        with atomic_raw_calibration_artifact(published):
            pass


def test_sealed_plan_rejects_tampering(tmp_path):
    _minimal_plan(tmp_path)
    audited = audited_raw_calibration_plan(
        tmp_path, verify_runtime_inputs=False
    )
    assert audited["fixed_scale"] == 1.0

    selection = json.loads((tmp_path / "selection.json").read_text())
    selection["frames"][0]["frame_id"] = 5
    _write_json(tmp_path / "selection.json", selection)
    with pytest.raises(ArtifactError, match="files changed"):
        audited_raw_calibration_plan(tmp_path, verify_runtime_inputs=False)


def test_audit_config_is_normalized_to_json_representation():
    solver = SolverSettings(
        initial_clock_offset_s=0.0,
        clock_offset_bounds_s=(-0.25, 0.25),
        lever_bounds_m=np.full(3, 0.3),
        baseline_angle_bounds_rad=np.full(2, 0.1),
        prior_sigma=np.full(6, 0.1),
    )
    audit = AuditConfig(
        source_pose_artifact="source",
        output_artifact="output",
        moving_base_pvt_topic="/pvt",
        body_frame="body",
        camera_frame="camera",
        primary_antenna_frame="primary",
        secondary_antenna_frame="secondary",
        body_from_camera=np.eye(4),
        primary_antenna_body_m=np.asarray([0.3, 0.0, 0.0]),
        secondary_antenna_body_m=np.asarray([1.6, 0.0, 0.0]),
        solver=solver,
        log_time_margin_s=5.0,
        status_max_age_s=0.15,
        minimum_navsat_status=1,
        minimum_position_carrier_solution=2,
        minimum_baseline_carrier_solution=2,
        baseline_length_tolerance_fraction=0.03,
    )

    record = _audit_config_record(audit)

    assert record["solver"]["clock_offset_bounds_s"] == [-0.25, 0.25]
    assert record == json.loads(json.dumps(record))


def test_reusable_code_has_no_field_specific_path_or_frame_window():
    source_root = Path(__file__).parents[1] / "src" / "rtk_splat"
    sources = [
        source_root / "diagnostics" / "raw_calibration.py",
        source_root / "adapters" / "calibration_bag.py",
    ]
    text = "\n".join(path.read_text().lower() for path in sources)
    assert "field1" not in text
    assert "/data/jkobo" not in text
    assert "1100–1800" not in text
    assert "1100-1800" not in text
