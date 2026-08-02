import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rtk_splat.adapters.ros1_citrusfarm import (
    _DistanceSampler,
    ReceiverStateEvidence,
    associate_receiver_state,
    build_typestore,
    derive_gnss_course,
    estimate_clock_offset,
    ingest_config_v2,
    read_navsat_track,
    resolve_relative_window,
    sample_header_log_offsets,
    stage_metric_stereo_frames,
    validate_bag_chain,
)


BASE_NS = 1_689_552_313_475_520_000


def _messages(typestore):
    from rosbags.typesys import get_types_from_msg

    receiver_type = "test_msgs/msg/ReceiverState"
    if receiver_type not in typestore.types:
        typestore.register(
            get_types_from_msg(
                """std_msgs/Header header
uint8 num_sat
bool rtk_mode_fix
uint8 system_error
uint8 io_error
uint8 swift_nap_error
uint8 external_antenna_present
string fix_mode
""",
                receiver_type,
            )
        )
    types = typestore.types
    return SimpleNamespace(
        Time=types["builtin_interfaces/msg/Time"],
        Header=types["std_msgs/msg/Header"],
        Image=types["sensor_msgs/msg/Image"],
        CameraInfo=types["sensor_msgs/msg/CameraInfo"],
        Roi=types["sensor_msgs/msg/RegionOfInterest"],
        Fix=types["sensor_msgs/msg/NavSatFix"],
        Status=types["sensor_msgs/msg/NavSatStatus"],
        ReceiverState=types[receiver_type],
    )


def _header(classes, timestamp_ns, frame):
    return classes.Header(
        0,
        classes.Time(timestamp_ns // 1_000_000_000, timestamp_ns % 1_000_000_000),
        frame,
    )


def _image(classes, header_ns, value):
    data = np.full((2, 3, 3), value, dtype=np.uint8)
    return classes.Image(
        _header(classes, header_ns, "camera"),
        2,
        3,
        "bgr8",
        0,
        9,
        data.reshape(-1),
    )


def _camera_info(classes, header_ns, right):
    return classes.CameraInfo(
        _header(classes, header_ns, "camera"),
        2,
        3,
        "plumb_bob",
        np.zeros(5),
        np.array([100.0, 0.0, 1.5, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0]),
        np.eye(3).reshape(-1),
        np.array(
            [100.0, 0.0, 1.5, -12.0 if right else 0.0,
             0.0, 100.0, 1.0, 0.0,
             0.0, 0.0, 1.0, 0.0]
        ),
        0,
        0,
        classes.Roi(0, 0, 0, 0, False),
    )


def _fix(classes, header_ns, index):
    return classes.Fix(
        _header(classes, header_ns, "piksi"),
        classes.Status(2, 1),
        33.0,
        -117.0 + index * 1.5e-6,
        270.0,
        np.diag([0.0049, 0.0049, 0.01]).reshape(-1),
        1,
    )


def _receiver_state(classes, header_ns, mode="FIXED_RTK", fixed=True):
    return classes.ReceiverState(
        _header(classes, header_ns, "piksi"),
        40,
        fixed,
        0,
        0,
        0,
        1,
        mode,
    )


def _write_bag(path, typestore, messages):
    from rosbags.rosbag1 import Writer

    with Writer(path) as writer:
        connections = {}
        for topic, message, _ in messages:
            if topic not in connections:
                connections[topic] = writer.add_connection(
                    topic, message.__msgtype__, typestore=typestore
                )
        for topic, message, log_ns in messages:
            writer.write(
                connections[topic],
                log_ns,
                typestore.serialize_ros1(message, message.__msgtype__),
            )


class MetricSamplingTests(unittest.TestCase):
    def test_residual_carry_approximates_requested_spacing(self):
        sampler = _DistanceSampler(0.15)
        selected = [
            index
            for index in range(11)
            if sampler.accept([index * 0.1, 0.0, 0.0])
        ]
        self.assertEqual(selected, [0, 2, 3, 5, 6, 8, 9])
        self.assertAlmostEqual(sampler.path_length_m, 1.0)


class ClockTests(unittest.TestCase):
    def _track(self):
        from rtk_splat.adapters.ros2_zed_ublox import RtkTrack

        log = BASE_NS + np.arange(21, dtype=np.int64) * 100_000_000
        header = log - 1_000_000
        track = RtkTrack(
            fix_t=header * 1e-9,
            fix_lat=np.full(21, 33.0),
            fix_lon=np.linspace(-117.0, -116.99999, 21),
            fix_alt=np.full(21, 270.0),
            fix_status=np.full(21, 2),
            fix_cov_max=np.full(21, 0.01),
            relpos_t=np.array([-1.0]),
            relpos_yaw=np.zeros(1),
            relpos_carr=np.full(1, -1),
            fix_header_ns=header,
            fix_log_ns=log,
            fix_covariance_enu_m2=np.repeat(np.eye(3)[None], 21, axis=0),
        )
        return track

    def test_shared_log_clock_recovers_camera_to_gnss_offset(self):
        track = self._track()
        camera_log = BASE_NS + np.array([100, 900, 1700]) * 1_000_000
        camera_offsets = np.array([-71_000_000, -72_000_000, -73_000_000])
        result = estimate_clock_offset(
            track,
            camera_log,
            camera_offsets,
            window_start_log_ns=BASE_NS,
            window_stop_log_ns=BASE_NS + 2_000_000_000,
            configured_offset_ns=71_000_000,
            maximum_error_ns=2_000_000,
            maximum_drift_ns=5_000_000,
        )
        self.assertEqual(result.camera_to_gnss_header_offset_ns, 71_000_000)
        self.assertEqual(result.configured_error_ns, 0)

    def test_bad_configured_offset_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            estimate_clock_offset(
                self._track(),
                BASE_NS + np.array([100, 900, 1700]) * 1_000_000,
                np.full(3, -72_000_000),
                window_start_log_ns=BASE_NS,
                window_stop_log_ns=BASE_NS + 2_000_000_000,
                configured_offset_ns=0,
                maximum_error_ns=10_000_000,
                maximum_drift_ns=5_000_000,
            )


class ReceiverStateTests(unittest.TestCase):
    def _track(self, count):
        from rtk_splat.adapters.ros2_zed_ublox import RtkTrack

        stamps = BASE_NS + np.arange(count, dtype=np.int64) * 100_000_000
        track = RtkTrack(
            fix_t=stamps.astype(np.float64) * 1e-9,
            fix_lat=np.full(count, 33.0),
            fix_lon=np.linspace(-117.0, -116.99999, count),
            fix_alt=np.full(count, 270.0),
            fix_status=np.full(count, 2, dtype=np.int16),
            fix_cov_max=np.full(count, 0.01),
            relpos_t=np.array([-1.0]),
            relpos_yaw=np.zeros(1),
            relpos_carr=np.full(1, -1),
            fix_header_ns=stamps,
            fix_log_ns=stamps + 1_000_000,
            fix_covariance_enu_m2=np.repeat(
                np.diag([0.0049, 0.0049, 0.01])[None], count, axis=0
            ),
            fix_covariance_type=np.full(count, 1, dtype=np.int16),
        )
        track.enu_xyz = np.column_stack(
            (np.arange(count, dtype=float), np.zeros(count), np.zeros(count))
        )
        return track

    def test_receiver_modes_override_ambiguous_navsat_status(self):
        modes = np.array(
            ["SPP", "DGNSS", "FLOAT_RTK", "FIXED_RTK", "FIXED_RTK"],
            dtype=np.str_,
        )
        stamps = BASE_NS + np.arange(5, dtype=np.int64) * 100_000_000 + 5_000_000
        evidence = ReceiverStateEvidence(
            topic="/receiver_state",
            message_type="test_msgs/msg/ReceiverState",
            message_definition_sha256="0" * 64,
            header_ns=stamps,
            log_ns=stamps + 1_000_000,
            fix_mode=modes,
            rtk_mode_fix=np.array([False, False, False, True, True]),
            num_sat=np.full(5, 40, dtype=np.int16),
            system_error=np.array([0, 0, 0, 0, 1], dtype=np.int16),
            io_error=np.zeros(5, dtype=np.int16),
            swift_nap_error=np.zeros(5, dtype=np.int16),
            external_antenna_present=np.ones(5, dtype=np.int16),
        )
        track = self._track(5)
        report = associate_receiver_state(
            track, evidence, tolerance_ns=10_000_000, required=True
        )
        np.testing.assert_array_equal(
            track.fix_position_quality,
            ["standalone", "differential", "rtk_float", "rtk_fixed", "invalid"],
        )
        np.testing.assert_array_equal(track.fix_carrier_status, [0, 0, 1, 2, -1])
        self.assertEqual(report["association_fraction"], 1.0)

    def test_required_receiver_state_fails_on_unmatched_fix(self):
        track = self._track(3)
        stamps = np.array([BASE_NS], dtype=np.int64)
        evidence = ReceiverStateEvidence(
            topic="/receiver_state",
            message_type="test_msgs/msg/ReceiverState",
            message_definition_sha256="0" * 64,
            header_ns=stamps,
            log_ns=stamps,
            fix_mode=np.array(["FIXED_RTK"]),
            rtk_mode_fix=np.array([True]),
            num_sat=np.array([40], dtype=np.int16),
            system_error=np.array([0], dtype=np.int16),
            io_error=np.array([0], dtype=np.int16),
            swift_nap_error=np.array([0], dtype=np.int16),
            external_antenna_present=np.array([1], dtype=np.int16),
        )
        with self.assertRaisesRegex(RuntimeError, "required receiver-state"):
            associate_receiver_state(
                track, evidence, tolerance_ns=10_000_000, required=True
            )


class SyntheticRos1ChainTests(unittest.TestCase):
    def test_chained_bags_preserve_stamps_and_stage_metric_pngs(self):
        typestore = build_typestore()
        classes = _messages(typestore)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            camera_bags = []
            gnss_bags = []
            for chunk in range(2):
                camera = root / f"camera_{chunk}.bag"
                gnss = root / f"gnss_{chunk}.bag"
                camera_messages = []
                gnss_messages = []
                for local in range(5):
                    index = chunk * 5 + local
                    log_ns = BASE_NS + index * 100_000_000
                    left = _image(classes, log_ns - 71_000_000, index)
                    right = _image(classes, log_ns - 70_998_000, index + 1)
                    camera_messages.extend(
                        [("/left", left, log_ns), ("/right", right, log_ns + 2_000)]
                    )
                    fix = _fix(classes, log_ns - 1_000_000, index)
                    gnss_messages.append(("/fix", fix, log_ns))
                    receiver = _receiver_state(
                        classes, log_ns + 4_000_000, "FIXED_RTK", True
                    )
                    gnss_messages.append(
                        ("/receiver_state", receiver, log_ns + 10_000)
                    )
                info_log = BASE_NS + chunk * 500_000_000 + 10_000
                camera_messages.extend(
                    [
                        ("/left_info", _camera_info(classes, info_log, False), info_log),
                        ("/right_info", _camera_info(classes, info_log, True), info_log + 1),
                    ]
                )
                camera_messages.sort(key=lambda item: item[2])
                _write_bag(camera, typestore, camera_messages)
                _write_bag(gnss, typestore, gnss_messages)
                camera_bags.append(camera)
                gnss_bags.append(gnss)

            camera_report = validate_bag_chain(
                camera_bags,
                required_each={"/left", "/right"},
                expected_count=2,
                maximum_gap_ns=200_000_000,
                maximum_overlap_ns=1_000_000,
            )
            self.assertEqual(len(camera_report.chunks), 2)
            track = read_navsat_track(gnss_bags, "/fix", typestore)
            np.testing.assert_array_equal(
                track.fix_header_ns,
                track.fix_log_ns - 1_000_000,
            )
            self.assertEqual(track.fix_covariance_enu_m2.shape, (10, 3, 3))
            np.testing.assert_array_equal(track.fix_service, [1] * 10)
            pose_cfg = SimpleNamespace(yaw_smooth_window=3, min_course_speed_ms=0.01)
            derive_gnss_course(track, pose_cfg)
            epoch, start, stop = resolve_relative_window(
                track, [0.0, 0.9], "first_gnss_log"
            )
            self.assertEqual(epoch, BASE_NS)
            logs, offsets = sample_header_log_offsets(
                camera_report, "/left", typestore, start, stop, samples_per_probe=1
            )
            self.assertEqual(len(logs), 3)
            np.testing.assert_array_equal(offsets, [-71_000_000] * 3)

            output = root / "staged"
            output.mkdir()
            topics = SimpleNamespace(left_image="/left", right_image="/right")
            frames, sampling = stage_metric_stereo_frames(
                camera_bags,
                topics,
                typestore,
                track,
                start_log_ns=start,
                stop_log_ns=stop,
                camera_to_gnss_offset_ns=70_000_000,
                spacing_m=0.12,
                stereo_tolerance_ns=10_000,
                output_dir=output,
                image_encoding="png_lossless",
            )
            self.assertGreaterEqual(len(frames), 5)
            self.assertEqual(sampling["candidate_pairs"], 10)
            self.assertTrue(all(Path(frame.left_jpeg).is_file() for frame in frames))
            self.assertEqual(Path(frames[0].left_jpeg).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

            cfg = SimpleNamespace(
                adapter="ros1_citrusfarm",
                paths=SimpleNamespace(
                    camera_bags=camera_bags,
                    gnss_bags=gnss_bags,
                    workdir=root / "work",
                ),
                topics=SimpleNamespace(
                    left_image="/left",
                    right_image="/right",
                    left_info="/left_info",
                    right_info="/right_info",
                    fix="/fix",
                    receiver_state="/receiver_state",
                ),
                gnss_quality=SimpleNamespace(
                    receiver_state_required=True,
                    receiver_state_association_tolerance_s=0.01,
                    covariance_provenance="driver_static_nominal",
                    covariance_is_live_per_epoch=False,
                ),
                pose=SimpleNamespace(
                    source="gnss_course",
                    time_offset_s=0.070,
                    yaw_smooth_window=3,
                    min_course_speed_ms=0.01,
                    minimum_navsat_status=2,
                    minimum_position_carrier_status=2,
                    maximum_position_covariance_m2=0.02,
                ),
                segment=SimpleNamespace(
                    window_s=[0.0, 0.9],
                    window_epoch_source="first_gnss_log",
                    frame_spacing_m=0.12,
                    stereo_tolerance_s=0.001,
                    association_tolerance_s=0.01,
                    expected_camera_bag_count=2,
                    expected_gnss_bag_count=2,
                    maximum_bag_gap_s=0.2,
                    maximum_bag_overlap_s=0.001,
                    validation_every=3,
                    image_encoding="png_lossless",
                ),
                timing=SimpleNamespace(
                    clock_model="ros_log_constant",
                    maximum_offset_error_s=0.002,
                    maximum_window_drift_s=0.002,
                ),
                sensor_geometry=SimpleNamespace(
                    extrinsic_camera_geometry="raw_left",
                    frames=SimpleNamespace(
                        world="map",
                        camera="camera",
                        primary_antenna="piksi_phase_center",
                    ),
                    T_camera_primary_antenna=np.eye(4).tolist(),
                    extrinsic_translation_sigma_m=[0.05, 0.05, 0.05],
                    extrinsic_provenance=SimpleNamespace(
                        method="synthetic exact test transform",
                        status="test-only",
                        sources=[
                            SimpleNamespace(
                                url="https://example.invalid/calibration",
                                sha256="0" * 64,
                            )
                        ],
                    ),
                ),
            )
            reader = ingest_config_v2(cfg, root / "segment").validate()
            self.assertEqual(reader.meta["adapter"], "ros1_citrusfarm")
            self.assertTrue(reader.meta["capabilities"]["stereo"])
            self.assertTrue(reader.meta["capabilities"]["single_rtk"])
            self.assertTrue(reader.meta["capabilities"]["images_rectified"])
            self.assertFalse(reader.meta["capabilities"]["images_raw"])
            self.assertFalse(reader.meta["capabilities"]["dual_rtk"])
            self.assertFalse(reader.meta["capabilities"]["depth_recorded"])
            self.assertIsNone(reader.observations("heading", required=False))
            published_frames = reader.frames
            self.assertIn("left_log_timestamp_ns", published_frames)
            np.testing.assert_array_equal(
                published_frames["timestamp_ns"],
                published_frames["left_log_timestamp_ns"] - 71_000_000,
            )
            gnss = reader.observations("gnss")
            self.assertIsNotNone(gnss)
            np.testing.assert_array_equal(
                gnss["raw_header_timestamp_ns"],
                gnss["raw_log_timestamp_ns"] - 1_000_000,
            )
            np.testing.assert_array_equal(gnss["raw_fix_status"], [2] * 10)
            np.testing.assert_array_equal(gnss["raw_service"], [1] * 10)
            np.testing.assert_array_equal(gnss["raw_carrier_status"], [2] * 10)
            np.testing.assert_array_equal(
                gnss["position_quality"],
                ["rtk_fixed"] * int(reader.meta["n_frames"]),
            )
            self.assertEqual(len(gnss["receiver_state_raw_fix_mode"]), 10)
            np.testing.assert_array_equal(
                gnss["receiver_state_raw_fix_mode"], ["FIXED_RTK"] * 10
            )
            self.assertTrue(gnss["raw_receiver_state_matched"].all())
            self.assertEqual(gnss["raw_covariance_enu_m2"].shape, (10, 3, 3))
            self.assertTrue(published_frames["pose_valid"].all())
            self.assertEqual(
                reader.meta["clock_alignment"]["camera_to_rtk_offset_ns"],
                70_000_000,
            )
            self.assertEqual(
                reader.calibration["rough_extrinsics"]["provenance"]["sources"][0][
                    "sha256"
                ],
                "0" * 64,
            )
            quality = reader.meta["provenance"]["gnss_quality"]
            self.assertEqual(
                quality["position_classification_source"], "receiver_state"
            )
            self.assertEqual(
                quality["covariance"]["configured_provenance"],
                "driver_static_nominal",
            )
            self.assertTrue(
                quality["covariance"]["observed_static_across_stream"]
            )
            self.assertEqual(
                quality["covariance"]["observed_covariance_type_counts"],
                {"APPROXIMATED": 10},
            )
            self.assertEqual(
                reader.meta["position_observation"]["quality_source"],
                "receiver_state.fix_mode",
            )
            configuration = reader.meta["provenance"]["configuration"]
            self.assertIn("effective_config", configuration)
            self.assertEqual(
                configuration["runtime_resolution"]["derivations"]
                ["frame_spacing_m"]["chosen_value"],
                0.12,
            )


if __name__ == "__main__":
    unittest.main()
