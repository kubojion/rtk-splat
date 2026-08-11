import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import rtk_splat.adapters.ros2_zed_ublox as ros2_adapter
from rtk_splat.core.runtime_resolution import runtime_resolution_plain
from rtk_splat.adapters.image_decode import (
    ImageDecodeError,
    decode_compressed_image,
    decode_raw_image,
    detect_compressed_format,
)
from rtk_splat.adapters.synchronization import (
    TimestampMatchError,
    associate_timestamps,
    monotonic_matches,
    nearest_matches,
)
from rtk_splat.adapters.records import StagedPayload
from rtk_splat.adapters.ros2_zed_ublox import FrameRecord, pair_stereo_timestamps


class TimestampSynchronizationTests(unittest.TestCase):
    def test_nearest_preserves_signed_residuals_and_chooses_earlier_tie(self):
        matches = nearest_matches(
            np.array([100, 200, 300], dtype=np.int64),
            np.array([90, 110, 290], dtype=np.int64),
            tolerance_ns=15,
        )
        np.testing.assert_array_equal(matches.reference_indices, [0, 2])
        np.testing.assert_array_equal(matches.sample_indices, [0, 2])
        np.testing.assert_array_equal(matches.residual_ns, [-10, -10])
        np.testing.assert_array_equal(matches.unmatched_reference_indices, [1])
        self.assertEqual(matches.residual_summary()["max_abs_ns"], 10)

    def test_nearest_allows_reuse_for_slower_sensor_streams(self):
        matches = associate_timestamps(
            [100, 110, 120],
            [105],
            tolerance_ns=20,
            method="nearest",
        )
        np.testing.assert_array_equal(matches.sample_indices, [0, 0, 0])
        np.testing.assert_array_equal(matches.residual_ns, [5, -5, -15])
        self.assertEqual(matches.match_fraction, 1.0)

    def test_monotonic_is_one_to_one_and_maximizes_ordered_cardinality(self):
        matches = monotonic_matches(
            [5, 6],
            [4, 5],
            tolerance_ns=1,
        )
        np.testing.assert_array_equal(matches.reference_indices, [0, 1])
        np.testing.assert_array_equal(matches.sample_indices, [0, 1])
        np.testing.assert_array_equal(matches.residual_ns, [-1, -1])
        self.assertEqual(np.unique(matches.sample_indices).size, 2)

    def test_stereo_adapter_uses_explicit_tolerance_without_rate_assumption(self):
        matches = pair_stereo_timestamps(
            [1_000_000_000, 1_137_000_000, 1_411_000_000],
            [1_004_000_000, 1_140_000_000, 1_450_000_000],
            tolerance_ns=5_000_000,
        )
        np.testing.assert_array_equal(matches.reference_indices, [0, 1])
        np.testing.assert_array_equal(matches.sample_indices, [0, 1])
        np.testing.assert_array_equal(
            matches.residual_ns, [4_000_000, 3_000_000]
        )
        np.testing.assert_array_equal(matches.unmatched_reference_indices, [2])

    def test_empty_sample_stream_marks_every_reference_unmatched(self):
        matches = nearest_matches([1, 2], [], tolerance_ns=0)
        self.assertEqual(matches.match_count, 0)
        np.testing.assert_array_equal(matches.unmatched_reference_indices, [0, 1])
        self.assertIsNone(matches.residual_summary()["median_abs_ns"])

    def test_rejects_lossy_or_non_monotonic_timestamp_inputs(self):
        with self.assertRaisesRegex(TimestampMatchError, "integer nanoseconds"):
            nearest_matches([1.0, 2.0], [1], tolerance_ns=1)
        with self.assertRaisesRegex(TimestampMatchError, "strictly increasing"):
            nearest_matches([1, 1], [1], tolerance_ns=1)
        with self.assertRaisesRegex(TimestampMatchError, "non-negative integer"):
            nearest_matches([1], [1], tolerance_ns=0.1)


class ImageDecodeTests(unittest.TestCase):
    def test_raw_bgr8_numpy_payload_removes_row_padding(self):
        rows = np.array(
            [
                [1, 2, 3, 4, 5, 6, 99, 99],
                [7, 8, 9, 10, 11, 12, 99, 99],
            ],
            dtype=np.uint8,
        )
        image = decode_raw_image(
            rows.reshape(-1),
            width=2,
            height=2,
            encoding="bgr8",
            step=8,
        )
        self.assertEqual(image.shape, (2, 2, 3))
        np.testing.assert_array_equal(image[0], [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(image[1], [[7, 8, 9], [10, 11, 12]])
        self.assertTrue(image.flags.c_contiguous)

    def test_raw_big_endian_depth_becomes_native_uint16(self):
        expected = np.array([[1, 256], [513, 1024]], dtype=np.uint16)
        payload = expected.astype(">u2").tobytes()
        decoded = decode_raw_image(
            payload,
            width=2,
            height=2,
            encoding="16UC1",
            is_bigendian=True,
        )
        np.testing.assert_array_equal(decoded, expected)
        self.assertEqual(decoded.dtype, np.dtype(np.uint16))

    def test_png_signature_and_rgb_conversion(self):
        bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", bgr)
        self.assertTrue(ok)
        self.assertEqual(detect_compressed_format(encoded), "png")
        rgb = decode_compressed_image(
            encoded, format_hint="rgb8; png compressed", color_order="rgb"
        )
        np.testing.assert_array_equal(rgb, bgr[..., ::-1])

    def test_jpeg_decodes_with_explicit_native_color_order(self):
        bgr = np.full((8, 8, 3), (20, 80, 140), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", bgr)
        self.assertTrue(ok)
        decoded = decode_compressed_image(
            encoded.tobytes(), format_hint="jpeg", color_order="bgr"
        )
        self.assertEqual(decoded.shape, bgr.shape)
        self.assertLess(float(np.abs(decoded.astype(int) - bgr).mean()), 3.0)

    def test_compressed_and_raw_metadata_mismatches_fail_explicitly(self):
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", bgr)
        self.assertTrue(ok)
        with self.assertRaisesRegex(ImageDecodeError, "format_hint says jpeg"):
            decode_compressed_image(encoded, format_hint="jpeg")
        with self.assertRaisesRegex(ImageDecodeError, "metadata requires"):
            decode_raw_image(
                b"\x00" * 5,
                width=2,
                height=1,
                encoding="bgr8",
            )
        with self.assertRaisesRegex(ImageDecodeError, "neither a JPEG nor a PNG"):
            decode_compressed_image(b"not an image")


class OptionalDependencyTests(unittest.TestCase):
    def test_ros_adapter_import_does_not_require_rosbags(self):
        source = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "rosbags" or name.startswith("rosbags."):
        raise ModuleNotFoundError("blocked by import-isolation test")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import rtk_splat.adapters.ros2_zed_ublox
import rtk_splat.adapters.ros1_citrusfarm
"""
        completed = subprocess.run(
            [sys.executable, "-c", source],
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONPATH": str(
                    Path(ros2_adapter.__file__).resolve().parents[2]
                )
                + (f":{os.environ['PYTHONPATH']}" if os.environ.get("PYTHONPATH") else ""),
            },
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class AdapterRuntimeDefaultTests(unittest.TestCase):
    def test_ros2_default_stride_has_auditable_constructed_origin(self):
        cfg = SimpleNamespace(segment=SimpleNamespace())
        configured = ros2_adapter.runtime_control_value(
            cfg,
            "frame_stride",
            getattr(cfg.segment, "frame_stride", 1),
            config_path="segment.frame_stride",
        )
        ros2_adapter.record_override(cfg, "frame_stride", int(configured))

        record = runtime_resolution_plain(cfg)["derivations"]["frame_stride"]
        self.assertEqual(record["source"], "override")
        self.assertEqual(record["chosen_value"], 1)
        self.assertEqual(record["origin"]["layer"], "constructed")


class Ros2EvidenceTests(unittest.TestCase):
    def test_stereo_reader_spools_selected_payloads_instead_of_retaining_bytes(self):
        left = SimpleNamespace(topic="/left", msgtype="CompressedImage")
        right = SimpleNamespace(topic="/right", msgtype="CompressedImage")

        class FakeReader:
            def __init__(self, _path):
                self.connections = [left, right]

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def messages(self, *, connections):
                selected = {connection.topic for connection in connections}
                for index in range(32):
                    stamp = SimpleNamespace(
                        sec=10,
                        nanosec=index * 10_000_000,
                    )
                    payload = bytes([index]) * 65_536
                    if left.topic in selected:
                        yield left, 0, SimpleNamespace(
                            header=SimpleNamespace(stamp=stamp),
                            data=b"left" + payload,
                        )
                    if right.topic in selected:
                        yield right, 0, SimpleNamespace(
                            header=SimpleNamespace(stamp=stamp),
                            data=b"right" + payload,
                        )

        topics = SimpleNamespace(left_image=left.topic, right_image=right.topic)
        typestore = SimpleNamespace(deserialize_cdr=lambda raw, _kind: raw)
        with tempfile.TemporaryDirectory() as temporary:
            spool = Path(temporary)
            with patch.object(ros2_adapter, "_reader_type", return_value=FakeReader):
                frames = list(
                    ros2_adapter.read_stereo_frames(
                        [Path("/fake/bag")],
                        topics,
                        typestore,
                        10.0,
                        10.31,
                        1,
                        spool_directory=spool,
                    )
                )

            self.assertEqual(len(frames), 32)
            self.assertTrue(
                all(isinstance(frame.left_jpeg, StagedPayload) for frame in frames)
            )
            self.assertTrue(
                all(isinstance(frame.right_jpeg, StagedPayload) for frame in frames)
            )
            self.assertEqual(len(list(spool.iterdir())), 64)
            self.assertEqual(frames[7].left_jpeg.path.read_bytes()[:4], b"left")
            self.assertEqual(frames[7].right_jpeg.path.read_bytes()[:5], b"right")

    def test_ingest_failure_removes_its_private_payload_spool(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "segment"
            captured = []

            def fail(_cfg, _destination, *, window, payload_spool):
                captured.append(payload_spool)
                (payload_spool / "partial.payload").write_bytes(b"partial")
                raise RuntimeError("synthetic ingest failure")

            with patch.object(
                ros2_adapter, "_ingest_config_v2_spooled", side_effect=fail
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic ingest failure"):
                    ros2_adapter.ingest_config_v2(
                        SimpleNamespace(), destination, window=None
                    )

            self.assertEqual(len(captured), 1)
            self.assertFalse(captured[0].exists())
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".segment.payloads-*")), [])

    def test_camera_rate_measurement_uses_bounded_header_sample(self):
        connection = SimpleNamespace(topic="/left", msgtype="Image")
        messages = [
            SimpleNamespace(
                header=SimpleNamespace(
                    stamp=SimpleNamespace(sec=10, nanosec=index * 100_000_000)
                )
            )
            for index in range(6)
        ]

        class FakeReader:
            def __init__(self, _path):
                self.connections = [connection]

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def messages(self, *, connections, start=None, stop=None):
                self.assert_connections = connections
                for message in messages:
                    yield connection, 10_000_000_000, message

        typestore = SimpleNamespace(deserialize_cdr=lambda raw, _kind: raw)
        with patch.object(ros2_adapter, "_reader_type", return_value=FakeReader):
            rate = ros2_adapter.measure_camera_header_rate(
                [Path("/fake/bag")],
                "/left",
                typestore,
                10.0,
                10.5,
            )
        self.assertAlmostEqual(rate, 10.0)

    def test_track_speed_measurement_uses_selected_window(self):
        track = SimpleNamespace(
            fix_t=np.array([0.0, 1.0, 2.0, 3.0]),
            enu_xyz=np.array(
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0],
                 [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
            ),
        )
        self.assertEqual(ros2_adapter.measure_track_speed(track, 0.0, 2.0), 0.5)

    def test_frame_record_retains_exact_header_nanoseconds(self):
        record = FrameRecord(
            t=1_700_000_000.1234567,
            t_right=1_700_000_000.1235566,
            left_jpeg=b"left",
            right_jpeg=b"right",
            left_header_ns=1_700_000_000_123_456_789,
            right_header_ns=1_700_000_000_123_556_789,
        )
        self.assertEqual(record.left_header_ns, 1_700_000_000_123_456_789)
        self.assertEqual(record.right_header_ns, 1_700_000_000_123_556_789)

    def test_rtk_reader_preserves_full_evidence_without_float_timestamp_loss(self):
        fix_topic = "/fix"
        relpos_topic = "/relpos"
        fix_connection = SimpleNamespace(topic=fix_topic, msgtype="Fix")
        relpos_connection = SimpleNamespace(topic=relpos_topic, msgtype="RelPos")
        header = SimpleNamespace(
            stamp=SimpleNamespace(sec=1_700_000_000, nanosec=123_456_789)
        )
        fix_message = SimpleNamespace(
            header=header,
            latitude=52.0,
            longitude=21.0,
            altitude=100.0,
            status=SimpleNamespace(status=2),
            position_covariance=[
                0.01, 0.001, 0.002,
                0.001, 0.02, 0.003,
                0.002, 0.003, 0.03,
            ],
            position_covariance_type=3,
        )
        relpos_message = SimpleNamespace(
            header=header,
            rel_pos_n=138,
            rel_pos_e=3,
            rel_pos_d=-2,
            rel_pos_hp_n=5,
            rel_pos_hp_e=7,
            rel_pos_hp_d=-4,
            acc_heading=41_000,
            carr_soln=SimpleNamespace(status=2),
            gnss_fix_ok=True,
            diff_soln=True,
            rel_pos_valid=True,
            is_moving=True,
            ref_pos_miss=False,
            ref_obs_miss=False,
            rel_pos_heading_valid=True,
            rel_pos_normalized=False,
        )

        class FakeReader:
            def __init__(self, _path):
                self.connections = [fix_connection, relpos_connection]

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def messages(self, connections):
                topics = {connection.topic for connection in connections}
                if fix_topic in topics:
                    yield (
                        fix_connection,
                        1_700_000_000_223_456_789,
                        fix_message,
                    )
                if relpos_topic in topics:
                    yield (
                        relpos_connection,
                        1_700_000_000_323_456_789,
                        relpos_message,
                    )

        typestore = SimpleNamespace(deserialize_cdr=lambda raw, _kind: raw)
        topics = SimpleNamespace(fix=fix_topic, relpos=relpos_topic)
        with (
            patch.object(ros2_adapter, "_reader_type", return_value=FakeReader),
            patch.object(
                ros2_adapter, "bag_for_topic", return_value=Path("/fake/bag")
            ),
        ):
            track = ros2_adapter.read_rtk_track(
                [Path("/fake/bag")], topics, typestore
            )

        np.testing.assert_array_equal(
            track.fix_header_ns, [1_700_000_000_123_456_789]
        )
        np.testing.assert_array_equal(
            track.fix_log_ns, [1_700_000_000_223_456_789]
        )
        np.testing.assert_array_equal(
            track.relpos_log_ns, [1_700_000_000_323_456_789]
        )
        np.testing.assert_allclose(
            track.fix_covariance_enu_m2[0],
            [[0.01, 0.001, 0.002],
             [0.001, 0.02, 0.003],
             [0.002, 0.003, 0.03]],
        )
        np.testing.assert_array_equal(track.fix_carrier_status, [-1])
        np.testing.assert_allclose(
            track.relpos_ned_m[0], [1.3805, 0.0307, -0.0204]
        )
        self.assertAlmostEqual(
            track.relpos_acc_heading_rad[0], np.deg2rad(0.41)
        )
        np.testing.assert_array_equal(
            track.relpos_flags[0],
            [True, True, True, True, False, False, True, False],
        )
        np.testing.assert_array_equal(track.relpos_carr, [2])


if __name__ == "__main__":
    unittest.main()
