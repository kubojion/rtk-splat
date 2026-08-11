from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from rtk_splat.adapters.publication import publish_segment_v2
from rtk_splat.adapters.records import (
    FrameRecord,
    RELPOS_FLAG_NAMES,
    RtkTrack,
    SecondaryGnssEvidence,
)
from rtk_splat.adapters.ros1_rosario_v2 import (
    _apply_ppk_effective_covariance,
    _attach_dual_position_heading,
    _bounded_gnss_log_window,
    _configured_bag,
    _drop_missing_depth_frames,
    _enforce_depth_left_camera_info_alignment,
    _enforce_depth_left_static_tf_alignment,
    _enforce_recorded_depth_acceptance,
    _enforce_recorded_stereo_acceptance,
    _navsat_position_quality,
    _prepare_config,
    _rollback_new_rgb_on_failure,
    validate_adapter_options,
)
from rtk_splat.workflows.configio import load_config


ROOT = Path(__file__).resolve().parents[1]
PPK_CONFIG = ROOT / "configs/sequences/rosario_v2_sequence5_ppk_140_250.yaml"
ONLINE_CONFIG = ROOT / "configs/sequences/rosario_v2_sequence5_online_140_250.yaml"


def _track(east: np.ndarray, north: np.ndarray, timestamps_ns: np.ndarray) -> RtkTrack:
    import pymap3d

    latitude, longitude, altitude = pymap3d.enu2geodetic(
        east,
        north,
        np.zeros_like(east),
        32.0,
        -60.0,
        20.0,
    )
    count = len(east)
    covariance = np.broadcast_to(np.diag([1.0, 1.0, 4.0]), (count, 3, 3)).copy()
    placeholder = np.array([-1], dtype=np.int64)
    position_valid = np.ones(count, dtype=bool)
    result = RtkTrack(
        fix_t=timestamps_ns.astype(float) * 1e-9,
        fix_lat=np.asarray(latitude),
        fix_lon=np.asarray(longitude),
        fix_alt=np.asarray(altitude),
        fix_status=np.full(count, 2, dtype=np.int16),
        fix_cov_max=np.full(count, 4.0),
        relpos_t=placeholder.astype(float),
        relpos_yaw=np.zeros(1),
        relpos_carr=np.full(1, -1, dtype=np.int8),
        fix_header_ns=timestamps_ns,
        fix_log_ns=timestamps_ns + 1_000_000,
        fix_covariance_enu_m2=covariance,
        fix_covariance_type=np.ones(count, dtype=np.int16),
        fix_carrier_status=np.full(count, -1, dtype=np.int8),
        fix_service=np.ones(count, dtype=np.int16),
        fix_position_valid=position_valid,
        fix_position_quality=np.full(count, "unknown_valid", dtype="<U13"),
        relpos_header_ns=placeholder,
        relpos_log_ns=placeholder,
    )
    return result


def test_layered_configs_keep_oracle_and_imu_inputs_out_of_method() -> None:
    ppk = load_config(PPK_CONFIG)
    online = load_config(ONLINE_CONFIG)
    assert ppk.adapter == "ros1_rosario_v2"
    assert ppk.adapter_options.gnss_source == "offline_ppk"
    assert online.adapter_options.gnss_source == "online_differential"
    assert not hasattr(ppk.topics, "imu")
    assert not hasattr(ppk.paths, "ground_truth_csv")
    assert all("pgt" not in str(path).lower() for path in ppk.paths.bags)
    assert all("conventional" not in str(path).lower() for path in ppk.paths.bags)
    assert len(ppk.paths.gnss_bags) == 1
    assert "ppk_gnss" in str(ppk.paths.gnss_bags[0])
    assert not hasattr(online.paths, "gnss_bags")
    validate_adapter_options(ppk.adapter_options)
    validate_adapter_options(online.adapter_options)


def test_calibration_acceptance_options_are_fail_closed() -> None:
    candidate = deepcopy(
        validate_adapter_options(load_config(PPK_CONFIG).adapter_options)
    )
    candidate["calibration_acceptance"]["recorded_stereo"][
        "minimum_fraction_under_1px_per_sample"
    ] = 1.1
    with pytest.raises(ValueError, match="must be in"):
        validate_adapter_options(candidate)


def test_real_image_stereo_acceptance_passes_and_fails_synthetically() -> None:
    gates = validate_adapter_options(
        load_config(PPK_CONFIG).adapter_options
    )["calibration_acceptance"]["recorded_stereo"]
    passing = [
        {
            "match_count": 500,
            "vertical_residual_p95_px": value,
            "fraction_under_1px": fraction,
        }
        for value, fraction in ((0.8, 0.96), (1.0, 0.93), (0.9, 0.95))
    ]
    report = _enforce_recorded_stereo_acceptance(passing, gates)
    assert report["passed"] is True
    assert report["observed"]["total_geometric_matches"] == 1500

    bad_p95 = deepcopy(passing)
    bad_p95[0]["vertical_residual_p95_px"] = 2.0
    bad_p95[1]["vertical_residual_p95_px"] = 2.0
    with pytest.raises(RuntimeError, match="p95 vertical residual"):
        _enforce_recorded_stereo_acceptance(bad_p95, gates)
    bad_matches = deepcopy(passing)
    bad_matches[0]["match_count"] = 99
    with pytest.raises(RuntimeError, match="insufficient geometric matches"):
        _enforce_recorded_stereo_acceptance(bad_matches, gates)


def test_recorded_depth_acceptance_and_alignment_pass_and_fail() -> None:
    acceptance = validate_adapter_options(
        load_config(PPK_CONFIG).adapter_options
    )["calibration_acceptance"]
    scores = [
        {
            "valid_pixel_count": 50_000,
            "disparity_error_p95_px": value,
            "scale_supported": True,
        }
        for value in (0.5, 0.7, 0.6)
    ]
    assert _enforce_recorded_depth_acceptance(
        scores, acceptance["recorded_depth"]
    )["passed"]
    bad_scores = deepcopy(scores)
    bad_scores[1]["disparity_error_p95_px"] = 2.1
    with pytest.raises(RuntimeError, match="p95 disparity"):
        _enforce_recorded_depth_acceptance(
            bad_scores, acceptance["recorded_depth"]
        )

    camera_info = {
        "width": 8,
        "height": 6,
        "k": [[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]],
        "p": [[100.0, 0.0, 4.0, 0.0], [0.0, 100.0, 3.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        "r": np.eye(3).tolist(),
        "d": [0.0] * 5,
    }
    geometry_gate = acceptance["depth_left_camera_info"][
        "maximum_absolute_parameter_difference"
    ]
    aligned = _enforce_depth_left_camera_info_alignment(
        camera_info, deepcopy(camera_info), geometry_gate
    )
    assert aligned["passed"] is True
    assert set(aligned["maximum_absolute_differences"]) == {"K", "P", "R", "D"}
    misaligned = deepcopy(camera_info)
    misaligned["p"][0][2] += 0.01
    with pytest.raises(RuntimeError, match="not aligned"):
        _enforce_depth_left_camera_info_alignment(
            camera_info, misaligned, geometry_gate
        )

    static_gates = acceptance["depth_left_static_tf"]
    assert _enforce_depth_left_static_tf_alignment(
        np.eye(4),
        static_gates,
        depth_frame_id="depth",
        left_frame_id="left",
    )["passed"]
    shifted = np.eye(4)
    shifted[0, 3] = 1.0e-3
    with pytest.raises(RuntimeError, match="not identity-aligned"):
        _enforce_depth_left_static_tf_alignment(
            shifted,
            static_gates,
            depth_frame_id="depth",
            left_frame_id="left",
        )
    with pytest.raises(RuntimeError, match="no depth-optical"):
        _enforce_depth_left_static_tf_alignment(
            None,
            static_gates,
            depth_frame_id="depth",
            left_frame_id="left",
        )


def test_main_bag_guard_rejects_oracle_like_filename(tmp_path: Path) -> None:
    forbidden = tmp_path / "sequence5_pgt.bag"
    forbidden.touch()
    cfg = SimpleNamespace(paths=SimpleNamespace(bags=[forbidden]))
    with pytest.raises(ValueError, match="evaluation/oracle"):
        _configured_bag(cfg)


def test_dual_heading_sign_is_verified_and_raw_skew_survives() -> None:
    cfg = load_config(PPK_CONFIG)
    options = validate_adapter_options(cfg.adapter_options)
    count = 80
    primary_ns = 1_700_000_000_000_000_000 + np.arange(count, dtype=np.int64) * 200_000_000
    secondary_ns = primary_ns + 10_000_000
    time_primary = (primary_ns - primary_ns[0]) * 1e-9
    time_secondary = (secondary_ns - primary_ns[0]) * 1e-9
    primary = _track(time_primary, np.full(count, 0.4), primary_ns)
    secondary = _track(time_secondary, np.full(count, -0.4), secondary_ns)
    report = _attach_dual_position_heading(primary, secondary, options)
    assert report["valid_fraction"] == pytest.approx(1.0)
    assert report["receiver_header_skew_ns"]["median"] == 10_000_000
    assert report["heading_convention"]["median_course_disagreement_deg"] < 1.0
    assert report["heading_convention"]["opposite_sign_disagreement_deg"] > 170.0
    assert primary.heading_quality_kind == "dual_position"
    assert np.all(primary.relpos_carr == -1)

    wrong = deepcopy(options)
    wrong["body_yaw_from_baseline_deg"] = -90.0
    primary_bad = _track(time_primary, np.full(count, 0.4), primary_ns)
    secondary_bad = _track(time_secondary, np.full(count, -0.4), secondary_ns)
    with pytest.raises(RuntimeError, match="approximately 180 degrees"):
        _attach_dual_position_heading(primary_bad, secondary_bad, wrong)


def test_dual_heading_never_interpolates_invalid_fix_brackets() -> None:
    options = validate_adapter_options(load_config(PPK_CONFIG).adapter_options)
    count = 80
    primary_ns = 1_700_000_000_000_000_000 + np.arange(count, dtype=np.int64) * 200_000_000
    secondary_ns = primary_ns + 10_000_000
    primary_time = (primary_ns - primary_ns[0]) * 1e-9
    secondary_time = (secondary_ns - primary_ns[0]) * 1e-9
    primary = _track(primary_time, np.full(count, 0.4), primary_ns)
    secondary = _track(secondary_time, np.full(count, -0.4), secondary_ns)
    primary.fix_position_valid[10] = False
    primary.fix_status[10] = -1
    secondary.fix_position_valid[30] = False
    secondary.fix_status[30] = -1

    report = _attach_dual_position_heading(primary, secondary, options)
    rejected = ~primary.heading_valid
    assert report["validity_rejections"]["primary_fix"] == 1
    assert report["validity_rejections"]["secondary_bracket"] == 2
    assert np.count_nonzero(rejected) == 3
    assert np.isnan(primary.relpos_ned_m[rejected]).all()
    assert np.isnan(primary.relpos_acc_heading_rad[rejected]).all()


def test_online_navsat_quality_is_status_aware() -> None:
    valid, quality = _navsat_position_quality(
        np.asarray([-1, 0, 1, 2], dtype=np.int16),
        np.ones(4, dtype=bool),
        "online_differential",
    )
    assert valid.tolist() == [False, True, True, True]
    assert quality.tolist() == [
        "invalid",
        "standalone",
        "differential",
        "differential",
    ]
    _, offline_quality = _navsat_position_quality(
        np.asarray([0, 2], dtype=np.int16),
        np.ones(2, dtype=bool),
        "offline_ppk",
    )
    assert offline_quality.tolist() == ["unknown_valid", "unknown_valid"]


def test_ppk_effective_covariance_never_overwrites_raw_message_values() -> None:
    cfg = load_config(PPK_CONFIG)
    options = validate_adapter_options(cfg.adapter_options)
    timestamps = 1_700_000_000_000_000_000 + np.arange(10, dtype=np.int64) * 200_000_000
    track = _track(np.arange(10, dtype=float), np.zeros(10), timestamps)
    decision = _apply_ppk_effective_covariance(track, options)
    assert decision["applied"] is True
    assert np.allclose(
        np.diagonal(track.raw_message_covariance_enu_m2, axis1=1, axis2=2),
        [1.0, 1.0, 4.0],
    )
    assert np.allclose(
        np.diagonal(track.fix_covariance_enu_m2, axis1=1, axis2=2),
        [0.01, 0.01, 0.04],
    )
    assert np.all(track.fix_carrier_status == -1)
    assert np.all(track.fix_position_quality == "unknown_valid")


def test_ppk_covariance_override_retains_nonplaceholder_evidence() -> None:
    options = validate_adapter_options(load_config(PPK_CONFIG).adapter_options)
    timestamps = 1_700_000_000_000_000_000 + np.arange(10, dtype=np.int64) * 200_000_000
    track = _track(np.arange(10, dtype=float), np.zeros(10), timestamps)
    recorded = np.asarray(track.fix_covariance_enu_m2).copy()
    recorded[4, 0, 0] = 0.5
    track.fix_covariance_enu_m2 = recorded.copy()

    decision = _apply_ppk_effective_covariance(track, options)
    assert decision["applied"] is False
    assert decision["decision"] == "retained_recorded_covariance"
    assert track.raw_message_covariance_enu_m2 is None
    np.testing.assert_array_equal(track.fix_covariance_enu_m2, recorded)


def test_effective_covariance_is_used_for_heading_accuracy() -> None:
    options = validate_adapter_options(load_config(PPK_CONFIG).adapter_options)
    count = 80
    primary_ns = 1_700_000_000_000_000_000 + np.arange(count, dtype=np.int64) * 200_000_000
    secondary_ns = primary_ns + 10_000_000
    primary_time = (primary_ns - primary_ns[0]) * 1e-9
    secondary_time = (secondary_ns - primary_ns[0]) * 1e-9
    primary = _track(primary_time, np.full(count, 0.4), primary_ns)
    secondary = _track(secondary_time, np.full(count, -0.4), secondary_ns)
    _apply_ppk_effective_covariance(primary, options)
    _apply_ppk_effective_covariance(secondary, options)

    _attach_dual_position_heading(primary, secondary, options)
    expected = np.sqrt(0.01 + 0.01) / 0.8
    assert np.median(primary.relpos_acc_heading_rad) == pytest.approx(
        expected, rel=1e-5
    )


def test_prepare_config_uses_gnss_report_bounds_source_mode_and_weight_order() -> None:
    cfg = load_config(PPK_CONFIG)
    base = 1_700_000_000_000_000_000
    main_report = SimpleNamespace(
        start_log_ns=base,
        stop_log_ns=base + 800_000_000_000,
        as_json=lambda: {"kind": "main"},
    )
    gnss_report = SimpleNamespace(
        start_log_ns=base + 139_000_000_000,
        stop_log_ns=base + 251_000_000_000,
        as_json=lambda: {"kind": "gnss"},
    )
    count = 80
    primary_ns = np.linspace(
        base + 140_000_000_000,
        base + 249_000_000_000,
        count,
        dtype=np.int64,
    )
    secondary_ns = primary_ns + 10_000_000
    primary_east = (primary_ns - primary_ns[0]) * 1e-9 * 0.8
    secondary_east = (secondary_ns - primary_ns[0]) * 1e-9 * 0.8
    primary = _track(primary_east, np.full(count, 0.4), primary_ns)
    secondary = _track(secondary_east, np.full(count, -0.4), secondary_ns)
    original_attach = _attach_dual_position_heading
    covariance_at_attach: list[np.ndarray] = []

    def attach_after_capture(first, second, options):
        covariance_at_attach.append(
            np.diag(np.asarray(first.fix_covariance_enu_m2)[0])
        )
        return original_attach(first, second, options)

    camera_frame = str(cfg.sensor_geometry.frames.camera)
    with (
        patch(
            "rtk_splat.adapters.ros1_rosario_v2._configured_bag",
            return_value=Path("/main.bag"),
        ),
        patch(
            "rtk_splat.adapters.ros1_rosario_v2._configured_gnss_bag",
            return_value=Path("/ppk.bag"),
        ),
        patch(
            "rtk_splat.adapters.ros1_rosario_v2.validate_bag_chain",
            side_effect=[main_report, gnss_report],
        ),
        patch(
            "rtk_splat.adapters.ros1_rosario_v2.build_typestore",
            return_value=object(),
        ),
        patch(
            "rtk_splat.adapters.ros1_rosario_v2._read_navsat_window",
            side_effect=[primary, secondary],
        ) as read_navsat,
        patch(
            "rtk_splat.adapters.ros1_rosario_v2._attach_dual_position_heading",
            side_effect=attach_after_capture,
        ),
        patch(
            "rtk_splat.adapters.ros1_rosario_v2.sample_header_log_offsets",
            return_value=(
                np.asarray([base + 140_000_000_000], dtype=np.int64),
                np.asarray([-1_000_000], dtype=np.int64),
            ),
        ),
        patch(
            "rtk_splat.adapters.ros1_rosario_v2._calibration_audit",
            return_value=(
                {},
                {"depth": {"camera_info": {"frame_id": "depth"}}},
                {"left": {"frame_id": camera_frame}},
            ),
        ),
        patch(
            "rtk_splat.adapters.ros1_rosario_v2._resolved_camera_extrinsic",
            return_value=(np.eye(4), {}),
        ),
        patch(
            "rtk_splat.adapters.ros1_rosario_v2._tf_audit",
            return_value={
                "static_depth_left_alignment": {"passed": True}
            },
        ),
        patch(
            "rtk_splat.adapters.ros1_rosario_v2.resolve_metric_frame_spacing",
            return_value=0.1,
        ),
    ):
        preflight, *_ = _prepare_config(cfg)

    assert read_navsat.call_count == 2
    for call in read_navsat.call_args_list:
        assert call.kwargs["source_mode"] == "offline_ppk"
        assert call.kwargs["start_log_ns"] == gnss_report.start_log_ns
        assert call.kwargs["stop_log_ns"] == gnss_report.stop_log_ns
    np.testing.assert_allclose(covariance_at_attach[0], [0.01, 0.01, 0.04])
    assert preflight["gnss"]["offline_ppk_covariance_decisions"]["primary"][
        "applied"
    ] is True


def test_gnss_window_clamps_to_the_gnss_report() -> None:
    report = SimpleNamespace(start_log_ns=100, stop_log_ns=900)
    assert _bounded_gnss_log_window(report, 200, 800, 150) == (100, 900)


def test_effective_covariance_override_is_atomic_and_single_use() -> None:
    timestamps = 1_700_000_000_000_000_000 + np.arange(3, dtype=np.int64) * 200_000_000
    track = _track(np.arange(3, dtype=float), np.zeros(3), timestamps)
    original = np.asarray(track.fix_covariance_enu_m2).copy()
    with pytest.raises(ValueError, match="shape"):
        track.apply_effective_covariance(
            np.eye(3)[None], policy={"source": "invalid test"}
        )
    assert track.raw_message_covariance_enu_m2 is None
    np.testing.assert_array_equal(track.fix_covariance_enu_m2, original)

    effective = np.broadcast_to(np.eye(3)[None] * 0.01, original.shape)
    track.apply_effective_covariance(effective, policy={"source": "test"})
    np.testing.assert_array_equal(track.raw_message_covariance_enu_m2, original)
    with pytest.raises(ValueError, match="already been applied"):
        track.apply_effective_covariance(effective, policy={"source": "test"})


def test_secondary_gnss_record_rejects_partial_or_lossy_evidence() -> None:
    count = 3
    timestamps = 1_700_000_000_000_000_000 + np.arange(count, dtype=np.int64)
    common = {
        "header_ns": timestamps,
        "log_ns": timestamps + 1,
        "enu_m": np.zeros((count, 3)),
        "geodetic_deg_m": np.zeros((count, 3)),
        "raw_covariance_enu_m2": np.broadcast_to(np.eye(3), (count, 3, 3)),
        "effective_covariance_enu_m2": np.broadcast_to(
            np.eye(3), (count, 3, 3)
        ),
        "fix_status": np.ones(count, dtype=np.int16),
        "carrier_status": np.full(count, -1, dtype=np.int8),
        "covariance_type": np.ones(count, dtype=np.int16),
        "service": np.ones(count, dtype=np.int16),
    }
    with pytest.raises(ValueError, match="equal sample count"):
        SecondaryGnssEvidence(**(common | {"service": np.ones(2, dtype=np.int16)}))
    with pytest.raises(ValueError, match="must contain integers"):
        SecondaryGnssEvidence(
            **(common | {"header_ns": timestamps.astype(np.float64)})
        )


def test_missing_recorded_depth_drops_only_the_unmatched_stereo_sample() -> None:
    frames = ["frame-0", "frame-1", "frame-2"]
    records = [{"header_ns": 10}, None, {"header_ns": 30}]
    kept_frames, kept_depth, dropped = _drop_missing_depth_frames(frames, records)
    assert kept_frames == ["frame-0", "frame-2"]
    assert kept_depth == [records[0], records[2]]
    assert dropped == [1]


def test_new_rgb_sibling_is_rolled_back_if_segment_publication_fails(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "segment.rgb_observations"
    artifact.mkdir()
    (artifact / "evidence").write_text("created by this invocation")
    with pytest.raises(RuntimeError, match="synthetic segment failure"):
        with _rollback_new_rgb_on_failure(artifact):
            raise RuntimeError("synthetic segment failure")
    assert not artifact.exists()


def test_generic_publisher_roundtrips_rosario_depth_and_dual_position_evidence(
    tmp_path: Path,
) -> None:
    count = 3
    timestamps = 1_700_000_000_000_000_000 + np.arange(count, dtype=np.int64) * 100_000_000
    track = _track(np.arange(count, dtype=float), np.zeros(count), timestamps)
    track.enu_xyz = np.column_stack(
        (np.arange(count, dtype=float), np.zeros(count), np.zeros(count))
    )
    _apply_ppk_effective_covariance(
        track,
        validate_adapter_options(load_config(PPK_CONFIG).adapter_options),
    )
    track.relpos_t = timestamps.astype(float) * 1e-9
    track.relpos_header_ns = timestamps.copy()
    track.relpos_log_ns = timestamps + 1_000_000
    track.relpos_yaw = np.zeros(count)
    track.relpos_ned_m = np.tile([-0.8, 0.0, 0.0], (count, 1))
    track.relpos_acc_heading_rad = np.full(count, 1.0)
    track.relpos_carr = np.full(count, -1, dtype=np.int8)
    track.relpos_flags = np.zeros((count, len(RELPOS_FLAG_NAMES)), dtype=bool)
    track.heading_valid = np.ones(count, dtype=bool)
    track.heading_quality_kind = "dual_position"
    track.secondary_gnss = SecondaryGnssEvidence(
        header_ns=timestamps + 10_000_000,
        log_ns=timestamps + 11_000_000,
        enu_m=track.enu_xyz + [0.0, -0.8, 0.0],
        geodetic_deg_m=np.column_stack(
            (track.fix_lat, track.fix_lon, track.fix_alt)
        ),
        raw_covariance_enu_m2=np.broadcast_to(
            np.diag([1.0, 1.0, 4.0]), (count, 3, 3)
        ).copy(),
        effective_covariance_enu_m2=np.broadcast_to(
            np.diag([0.01, 0.01, 0.04]), (count, 3, 3)
        ).copy(),
        fix_status=np.full(count, 2, dtype=np.int16),
        carrier_status=np.full(count, -1, dtype=np.int8),
        covariance_type=np.ones(count, dtype=np.int16),
        service=np.ones(count, dtype=np.int16),
    )

    image = np.full((6, 8), 80, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    frames = [
        FrameRecord(
            t=int(timestamp) * 1e-9,
            t_right=int(timestamp) * 1e-9,
            left_jpeg=encoded.tobytes(),
            right_jpeg=encoded.tobytes(),
            left_header_ns=int(timestamp),
            right_header_ns=int(timestamp),
        )
        for timestamp in timestamps
    ]
    for frame, timestamp in zip(frames, timestamps):
        frame.left_log_ns = int(timestamp) + 2_000_000
        frame.right_log_ns = int(timestamp) + 2_000_000
    poses = []
    for index in range(count):
        center = np.array([float(index), 0.0, 0.0])
        viewmat = np.eye(4)
        viewmat[:3, 3] = -center
        poses.append(SimpleNamespace(viewmat=viewmat, cam_center=center))

    depth_records = []
    for index, timestamp in enumerate(timestamps):
        path = tmp_path / f"staged-depth-{index}.npz"
        raw = np.full((6, 8), 2_000, dtype=np.uint16)
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                depth=(raw.astype(np.float32) * 0.001),
                valid=np.ones_like(raw, dtype=bool),
                raw_depth_units=raw,
            )
        depth_records.append(
            {
                "payload": path,
                "header_ns": int(timestamp),
                "log_ns": int(timestamp) + 2_000_000,
                "source_encoding": "16UC1",
            }
        )

    def camera_info(right: bool) -> dict:
        return {
            "frame_id": "left" if not right else "left-dataset-defect",
            "width": 8,
            "height": 6,
            "k": [[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]],
            "d": [0.0] * 5,
            "r": np.eye(3).tolist(),
            "p": [
                [100.0, 0.0, 4.0, -5.0 if right else 0.0],
                [0.0, 100.0, 3.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            "distortion_model": "plumb_bob",
        }

    reader = publish_segment_v2(
        tmp_path / "segment",
        frames,
        poses,
        track,
        camera_info(False),
        camera_info(True),
        camera_frame_id="left",
        primary_antenna_frame_id="reach_1",
        secondary_antenna_frame_id="reach_2",
        enu_definition={
            "origin_lat_deg": 32.0,
            "origin_lon_deg": -60.0,
            "origin_alt_ellipsoidal_m": 20.0,
            "ellipsoid": "WGS84",
            "vertical_datum": "WGS84 ellipsoid",
            "world_frame_id": "map",
        },
        T_camera_primary_antenna=np.eye(4),
        extrinsic_translation_sigma_m=[0.05, 0.05, 0.05],
        extrinsic_provenance={"method": "synthetic", "status": "test"},
        clock_offset_ns=0,
        association_tolerance_ns=20_000_000,
        stereo_tolerance_ns=1_000,
        capabilities={
            "single_rtk": False,
            "dual_rtk": True,
            "depth_recorded": True,
        },
        splits={"train": [0, 2], "val": [1], "test": []},
        adapter_name="ros1_rosario_v2",
        recorded_depth=depth_records,
        recorded_depth_semantics={
            "format": "npz_depth_valid",
            "units": "m",
            "quantity": "optical_z",
            "aligned_to": "left",
            "invalid_convention": "raw 0/65535 and depth=0 valid=false",
        },
        provenance={
            "gnss_quality": {
                "covariance": {
                    "raw": "driver_static_nominal",
                    "effective": "externally_declared_test_prior",
                }
            }
        },
    ).validate()
    assert reader.meta["capabilities"]["depth_recorded"] is True
    assert reader.meta["heading_observation"]["quality_source"] == "dual_position"
    assert reader.meta["position_observation"]["covariance_provenance"] is not None
    gnss = reader.observations("gnss")
    heading = reader.observations("heading")
    assert gnss is not None and heading is not None
    assert set(gnss["position_quality"].tolist()) == {"unknown_valid"}
    np.testing.assert_allclose(np.diag(gnss["raw_covariance_enu_m2"][0]), [1, 1, 4])
    np.testing.assert_allclose(
        np.diag(gnss["raw_effective_covariance_enu_m2"][0]), [0.01, 0.01, 0.04]
    )
    assert np.all(heading["raw_carrier_status"] == -1)
    assert np.all(heading["raw_valid"])
    assert "secondary_raw_covariance_enu_m2" in heading
    assert "secondary_raw_effective_covariance_enu_m2" in heading
    depth_path = reader.root / str(reader.frames["depth_path"][0])
    with np.load(depth_path, allow_pickle=False) as archive:
        assert archive["depth"].dtype == np.float32
        assert archive["raw_depth_units"].dtype == np.uint16
