import hashlib
import sqlite3
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rtk_splat.adapters.calibration_bag import (
    CalibrationTopics,
    _bounded_reader_messages,
    historical_stereo_bucket,
    read_calibration_bag_observations,
)
from rtk_splat.diagnostics.calibration_io import (
    CameraAntennaGeometry,
    atomic_calibration_artifact,
    calibration_artifact_path,
    load_raw_colmap_rig_trajectory,
    validate_calibration_artifact_name,
    write_json,
)


def ns_stamp(value):
    sec, nanosec = divmod(int(value), 1_000_000_000)
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def header(value, frame_id="sensor"):
    return SimpleNamespace(stamp=ns_stamp(value), frame_id=frame_id)


def image_message(value, payload):
    return SimpleNamespace(
        header=header(value, "zed"),
        data=np.frombuffer(payload, dtype=np.uint8),
        format="jpeg",
    )


def fix_message(value):
    return SimpleNamespace(
        header=header(value, "gps"),
        latitude=52.1,
        longitude=16.2,
        altitude=121.3,
        status=SimpleNamespace(status=1, service=3),
        position_covariance=[
            0.01, 0.001, 0.0,
            0.001, 0.02, 0.0,
            0.0, 0.0, 0.04,
        ],
        position_covariance_type=3,
    )


def relpos_message(value):
    return SimpleNamespace(
        header=header(value, "gps_rover"),
        version=1,
        ref_station_id=42,
        itow=123456,
        rel_pos_n=138,
        rel_pos_e=3,
        rel_pos_d=-2,
        rel_pos_length=138,
        rel_pos_heading=153028,
        rel_pos_hp_n=5,
        rel_pos_hp_e=7,
        rel_pos_hp_d=-4,
        rel_pos_hp_length=9,
        acc_n=20,
        acc_e=30,
        acc_d=40,
        acc_length=50,
        acc_heading=41000,
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


def pvt_message(value):
    return SimpleNamespace(
        header=header(value, "gps_moving_base"),
        itow=123456,
        gps_fix=SimpleNamespace(fix_type=3),
        gnss_fix_ok=True,
        diff_soln=True,
        carr_soln=SimpleNamespace(status=2),
        invalid_llh=False,
        num_sv=27,
        h_acc=14,
        v_acc=23,
        p_dop=112,
        valid_date=True,
        valid_time=True,
        fully_resolved=True,
        t_acc=25,
        nano=-17,
    )


@dataclass(frozen=True)
class FakeConnection:
    topic: str
    msgtype: str


class FakeTypestore:
    @staticmethod
    def deserialize_cdr(raw, msgtype):
        return raw


class FakeReader:
    datasets = {}
    data_passes = {}

    def __init__(self, path):
        self.path = str(path)
        self.dataset = self.datasets[self.path]
        self.connections = self.dataset["connections"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def messages(self, connections, start=None, stop=None):
        self.data_passes[self.path] = self.data_passes.get(self.path, 0) + 1
        selected = set(connections)
        for connection, log_ns, message in self.dataset["messages"]:
            if connection not in selected:
                continue
            if start is not None and log_ns < start:
                continue
            if stop is not None and log_ns >= stop:
                continue
            yield connection, log_ns, message


class ArtifactAndGeometryTests(unittest.TestCase):
    def test_bounded_mcap_scan_restores_storage_bounds(self):
        requested = SimpleNamespace(id=1, topic="/fix")
        storage_connection = SimpleNamespace(id=7, topic="/fix")
        chunks = [
            SimpleNamespace(
                chunk_start_offset=100,
                message_start_time=0,
                message_end_time=10,
                channel_count={7: 1}),
            SimpleNamespace(
                chunk_start_offset=200,
                message_start_time=10,
                message_end_time=20,
                channel_count={7: 1}),
            SimpleNamespace(
                chunk_start_offset=300,
                message_start_time=20,
                message_end_time=30,
                channel_count={7: 1}),
            SimpleNamespace(
                chunk_start_offset=400,
                message_start_time=30,
                message_end_time=40,
                channel_count={7: 1}),
        ]

        class Storage:
            data_start = 8
            data_end = 500
            connections = [storage_connection]

            def __init__(self):
                self.chunks = chunks
                self.bounds_during_scan = None

            def messages_scan(self, connections, start, stop):
                self.bounds_during_scan = (self.data_start, self.data_end)
                yield storage_connection, 22, b"raw"

        storage = Storage()
        directory = SimpleNamespace(
            storages=[storage],
            metadata=SimpleNamespace(compression_mode=None))

        class ScanReader:
            def __init__(self):
                self.storage = directory

            @staticmethod
            def messages(**_):
                raise AssertionError("indexed fallback must not be used")

        messages = list(_bounded_reader_messages(
            ScanReader(), [requested], 15, 25))
        self.assertEqual(messages, [(requested, 22, b"raw")])
        self.assertEqual(storage.bounds_during_scan, (200, 400))
        self.assertEqual((storage.data_start, storage.data_end), (8, 500))

    def test_safe_name_and_direct_geometry(self):
        self.assertEqual(
            validate_calibration_artifact_name("rtk_stereo-integrity.v1"),
            "rtk_stereo-integrity.v1")
        for bad in ("", "../escape", "a/b", ".hidden", "x y", ".."):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_calibration_artifact_name(bad)

        body_from_camera = np.eye(4)
        body_from_camera[:3, 3] = [3.0, 0.0, 1.0]
        geometry = CameraAntennaGeometry.from_body_geometry(
            body_from_camera,
            primary_antenna_position_body_m=[0.32, 0.0, 0.0],
            secondary_antenna_position_body_m=[1.705, 0.037, 0.0])
        np.testing.assert_allclose(
            geometry.camera_to_primary_antenna_in_camera_m,
            [-2.68, 0.0, -1.0])
        np.testing.assert_allclose(
            geometry.primary_to_secondary_antenna_in_camera_m,
            [1.385, 0.037, 0.0])

    def test_atomic_artifact_publishes_once_and_cleans_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            segment = Path(temporary) / "segment"
            with atomic_calibration_artifact(segment, "audit_v1") as work:
                write_json(work / "result.json", {"accepted": False})
                self.assertFalse(
                    calibration_artifact_path(segment, "audit_v1").exists())
            final = calibration_artifact_path(segment, "audit_v1")
            self.assertTrue(final.is_dir())
            self.assertIn('"accepted": false',
                          (final / "result.json").read_text())
            with self.assertRaises(FileExistsError):
                with atomic_calibration_artifact(segment, "audit_v1"):
                    pass

            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                with atomic_calibration_artifact(segment, "audit_v2") as work:
                    (work / "partial.txt").write_text("partial")
                    raise RuntimeError("synthetic failure")
            self.assertFalse(
                calibration_artifact_path(segment, "audit_v2").exists())
            hidden = list((segment / "calibration_artifacts").glob(
                ".audit_v2.*"))
            self.assertEqual(hidden, [])


class BagObservationTests(unittest.TestCase):
    def setUp(self):
        FakeReader.datasets = {}
        FakeReader.data_passes = {}
        self.topics = CalibrationTopics(
            left_image="/left",
            right_image="/right",
            fix="/fix",
            relpos="/relpos",
            moving_base_pvt="/pvt",
        )
        self.base_ns = 1_700_000_000_123_456_789
        self.step_ns = 66_666_667

    def prepare_fake_bags(self, directory):
        camera_bag = str(Path(directory) / "camera")
        rtk_bag = str(Path(directory) / "rtk")
        left_connection = FakeConnection("/left", "CompressedImage")
        right_connection = FakeConnection("/right", "CompressedImage")
        fix_connection = FakeConnection("/fix", "NavSatFix")
        relpos_connection = FakeConnection("/relpos", "RelPos")
        pvt_connection = FakeConnection("/pvt", "PVT")

        camera_messages = []
        selected_payloads = []
        for index in range(6):
            left_ns = self.base_ns + index * self.step_ns
            right_ns = left_ns + 1_000_000
            left_payload = f"left-{index}".encode()
            right_payload = f"right-{index}".encode()
            camera_messages.extend([
                (left_connection, left_ns + 5_000_000,
                 image_message(left_ns, left_payload)),
                (right_connection, right_ns + 5_000_000,
                 image_message(right_ns, right_payload)),
            ])
            if index % 2 == 0:
                selected_payloads.append((left_payload, right_payload))

        rtk_ns = self.base_ns + self.step_ns
        rtk_messages = [
            (fix_connection, rtk_ns + 2_000_000, fix_message(rtk_ns)),
            (relpos_connection, rtk_ns + 3_000_000, relpos_message(rtk_ns)),
            (pvt_connection, rtk_ns + 4_000_000, pvt_message(rtk_ns)),
        ]
        FakeReader.datasets = {
            camera_bag: {
                "connections": [left_connection, right_connection],
                "messages": camera_messages,
            },
            rtk_bag: {
                "connections": [fix_connection, relpos_connection,
                                pvt_connection],
                "messages": rtk_messages,
            },
        }
        return camera_bag, rtk_bag, selected_payloads

    def write_extracted(self, directory, payloads):
        image_dir = Path(directory) / "images"
        image_dir.mkdir()
        for index, (left, right) in enumerate(payloads):
            (image_dir / f"left_{index:06d}.jpg").write_bytes(left)
            (image_dir / f"right_{index:06d}.jpg").write_bytes(right)
        return image_dir

    def read(self, bags, image_dir):
        return read_calibration_bag_observations(
            bags=bags,
            topics=self.topics,
            typestore=FakeTypestore(),
            image_start_header_ns=self.base_ns,
            image_stop_header_ns=self.base_ns + 5 * self.step_ns,
            frame_stride=2,
            extracted_images_dir=image_dir,
            expected_frame_count=3,
            rtk_start_header_ns=self.base_ns,
            rtk_stop_header_ns=self.base_ns + 5 * self.step_ns,
            log_start_ns=self.base_ns - 1_000_000_000,
            log_stop_ns=self.base_ns + 2_000_000_000,
        )

    def test_grouped_bounded_pass_preserves_full_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            camera_bag, rtk_bag, payloads = self.prepare_fake_bags(temporary)
            image_dir = self.write_extracted(temporary, payloads)
            with patch(
                "rtk_splat.adapters.calibration_bag.Reader", FakeReader
            ):
                observations = self.read(
                    [Path(camera_bag), Path(rtk_bag)], image_dir)

        self.assertEqual(FakeReader.data_passes[camera_bag], 1)
        self.assertEqual(FakeReader.data_passes[rtk_bag], 1)
        self.assertEqual(len(observations.stereo_frames), 3)
        self.assertEqual(
            observations.stereo_frames[0].left_header_ns, self.base_ns)
        self.assertEqual(
            observations.stereo_frames[0].stereo_delta_ns, 1_000_000)
        self.assertEqual(
            observations.stereo_frames[2].left_sha256,
            hashlib.sha256(b"left-4").hexdigest())

        fix = observations.fixes[0]
        self.assertEqual(fix.status, 1)
        self.assertEqual(fix.service, 3)
        np.testing.assert_allclose(
            fix.covariance_matrix_enu_m2,
            [[0.01, 0.001, 0.0],
             [0.001, 0.02, 0.0],
             [0.0, 0.0, 0.04]])

        relpos = observations.relpos[0]
        self.assertEqual(relpos.rel_pos_ned_cm, (138, 3, -2))
        np.testing.assert_allclose(
            relpos.rel_pos_ned_m, [1.3805, 0.0307, -0.0204])
        np.testing.assert_allclose(
            relpos.rel_pos_enu_m, [0.0307, 1.3805, 0.0204])
        np.testing.assert_allclose(
            relpos.accuracy_ned_m, [0.002, 0.003, 0.004])
        self.assertEqual(relpos.carrier_solution, 2)
        self.assertTrue(relpos.rel_pos_heading_valid)

        pvt = observations.moving_base_pvt[0]
        self.assertEqual(pvt.gps_fix_type, 3)
        self.assertEqual(pvt.carrier_solution, 2)
        self.assertAlmostEqual(pvt.horizontal_accuracy_m, 0.014)
        self.assertAlmostEqual(pvt.position_dop, 1.12)

        payload = observations.to_npz_payload()
        self.assertEqual(payload["stereo_left_header_ns"].dtype, np.int64)
        self.assertEqual(payload["fix_covariance_enu_m2"].shape, (1, 3, 3))
        np.testing.assert_array_equal(
            payload["relpos_ned_cm"], [[138, 3, -2]])
        np.testing.assert_array_equal(
            payload["relpos_hp_ned_0p1mm"], [[5, 7, -4]])
        self.assertEqual(payload["relpos_flags"].shape, (1, 8))
        self.assertEqual(payload["pvt_status_flags"].shape, (1, 6))

    def test_jpeg_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            camera_bag, rtk_bag, payloads = self.prepare_fake_bags(temporary)
            image_dir = self.write_extracted(temporary, payloads)
            (image_dir / "right_000001.jpg").write_bytes(b"not-the-bag-image")
            with patch(
                "rtk_splat.adapters.calibration_bag.Reader", FakeReader
            ):
                with self.assertRaisesRegex(ValueError, "right JPEG hash"):
                    self.read([Path(camera_bag), Path(rtk_bag)], image_dir)

    def test_historical_bucket_reproduces_epoch_float_expression(self):
        stamp_ns = self.base_ns + 987_654_321
        sec, nanosec = divmod(stamp_ns, 1_000_000_000)
        expected = round((sec + nanosec * 1.0e-9) * 30)
        self.assertEqual(historical_stereo_bucket(stamp_ns), expected)


class ColmapRigLoaderTests(unittest.TestCase):
    def test_loads_chronological_left_poses_and_composes_nonreference_sensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            # Right camera 2 is the rig reference. Left camera 1 is 12 cm to
            # its left in camera-from-rig coordinates.
            (model / "rigs.txt").write_text(
                "# rigs\n"
                "3 2 CAMERA 2 CAMERA 1 1 1 0 0 0 -0.12 0 0\n")
            # Deliberately write frame 1 first: output must follow filename
            # frame indices, not COLMAP registration/frame order.
            (model / "frames.txt").write_text(
                "# frames\n"
                "101 3 1 0 0 0 1 0 0 2 CAMERA 1 10 CAMERA 2 11\n"
                "100 3 1 0 0 0 0 0 0 2 CAMERA 1 20 CAMERA 2 21\n")

            database = root / "database.db"
            connection = sqlite3.connect(database)
            connection.execute(
                "CREATE TABLE images "
                "(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)")
            connection.executemany(
                "INSERT INTO images VALUES (?, ?, ?)",
                [
                    (10, "zed/left/000001.jpg", 1),
                    (11, "zed/right/000001.jpg", 2),
                    (20, "zed/left/000000.jpg", 1),
                    (21, "zed/right/000000.jpg", 2),
                ])
            connection.commit()
            connection.close()

            trajectory = load_raw_colmap_rig_trajectory(
                model, database, expected_frame_count=2)

        np.testing.assert_array_equal(trajectory.frame_indices, [0, 1])
        np.testing.assert_array_equal(
            trajectory.colmap_frame_ids, [100, 101])
        self.assertEqual(trajectory.image_names, (
            "zed/left/000000.jpg", "zed/left/000001.jpg"))
        np.testing.assert_allclose(
            trajectory.camera_from_visual_world[:, 0, 3],
            [-0.12, 0.88])
        np.testing.assert_allclose(
            trajectory.camera_centers_visual,
            [[0.12, 0.0, 0.0], [-0.88, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
