import unittest

import numpy as np

from rtk_splat.backends.geodetic_gnss import (
    GeodeticGnssTemporalPolicy,
    raw_gnss_temporal_filter,
)


def _stream(
    count: int = 40,
    *,
    intervals_s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if intervals_s is None:
        intervals_s = np.full(count - 1, 0.2, dtype=np.float64)
    seconds = np.concatenate(([0.0], np.cumsum(intervals_s)))
    timestamps = np.rint(seconds * 1.0e9).astype(np.int64) + 1_000_000_000
    positions = np.column_stack(
        (
            0.12 * seconds,
            0.03 * np.sin(0.4 * seconds),
            0.002 * seconds,
        )
    )
    covariance = np.repeat((np.eye(3) * 0.000004)[None], count, axis=0)
    fix = np.ones(count, dtype=np.int16)
    carrier = np.full(count, 2, dtype=np.int16)
    return timestamps, positions, covariance, fix, carrier


class GeodeticGnssTemporalFilterTests(unittest.TestCase):
    def test_isolated_fixed_rtk_spike_is_rejected(self):
        values = list(_stream())
        values[1][20] += np.array([0.18, -0.01, 0.0])

        retained, reasons, audit = raw_gnss_temporal_filter(*values)

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["paired_excursion_count"], 1)
        self.assertEqual(np.flatnonzero(~retained).tolist(), [20])
        self.assertEqual(
            reasons[20], "raw_gnss_paired_discontinuity_excursion"
        )

    def test_short_multiframe_fixed_rtk_spike_is_rejected(self):
        values = list(_stream(count=100))
        values[1][48:52] += np.array([0.15, 0.02, 0.0])

        retained, _, audit = raw_gnss_temporal_filter(*values)

        self.assertTrue(audit["passed"])
        self.assertEqual(np.flatnonzero(~retained).tolist(), [48, 49, 50, 51])
        self.assertEqual(audit["excursions"][0]["rejected_raw_epoch_count"], 4)

    def test_valid_accelerating_motion_and_turn_are_retained(self):
        timestamps, _, covariance, fix, carrier = _stream()
        seconds = (timestamps - timestamps[0]).astype(np.float64) * 1.0e-9
        positions = np.column_stack(
            (
                0.08 * seconds + 0.20 * seconds**2,
                0.35 * np.sin(0.35 * seconds),
                0.01 * seconds,
            )
        )

        retained, _, audit = raw_gnss_temporal_filter(
            timestamps, positions, covariance, fix, carrier
        )

        self.assertTrue(retained.all())
        self.assertEqual(audit["candidate_boundary_count"], 0)

    def test_irregular_timestamps_do_not_create_a_false_jump(self):
        intervals = np.asarray(
            [0.11, 0.29, 0.18, 0.31, 0.13, 0.24, 0.17, 0.26, 0.19] * 5,
            dtype=np.float64,
        )[:39]
        values = _stream(intervals_s=intervals)

        retained, _, audit = raw_gnss_temporal_filter(*values)

        self.assertTrue(retained.all())
        self.assertTrue(audit["passed"])

    def test_covariance_and_status_are_conservative(self):
        high_covariance = list(_stream())
        high_covariance[1][20] += np.array([0.18, 0.0, 0.0])
        high_covariance[2][19:22] = np.eye(3) * 0.01
        retained, _, audit = raw_gnss_temporal_filter(*high_covariance)
        self.assertTrue(retained.all())
        self.assertEqual(audit["candidate_boundary_count"], 0)

        float_status = list(_stream())
        float_status[1][20] += np.array([0.18, 0.0, 0.0])
        float_status[4][14:27] = 1
        retained, _, audit = raw_gnss_temporal_filter(*float_status)
        self.assertTrue(retained.all())
        self.assertEqual(audit["candidate_boundary_count"], 0)

        invalid_status = list(_stream())
        invalid_status[3][20] = -1
        retained, reasons, _ = raw_gnss_temporal_filter(*invalid_status)
        self.assertFalse(retained[20])
        self.assertEqual(reasons[20], "raw_gnss_stream_invalid")

    def test_unpaired_discontinuity_fails_closed(self):
        values = list(_stream())
        values[1][20:] += np.array([0.18, 0.0, 0.0])

        _, _, audit = raw_gnss_temporal_filter(*values)

        self.assertFalse(audit["passed"])
        self.assertEqual(len(audit["unpaired_boundary_ids"]), 1)


if __name__ == "__main__":
    unittest.main()
