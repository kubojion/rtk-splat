import unittest

import numpy as np

from frontends.keyframes import (
    KEYFRAME_PRESETS,
    KeyframeConfig,
    select_keyframes,
)
from frontends.pair_graph import PairGraphConfig, build_pair_graph


def yaw_quaternions(degrees):
    radians = np.deg2rad(np.asarray(degrees, dtype=float)) * 0.5
    return np.column_stack(
        [np.zeros(len(radians)), np.zeros(len(radians)),
         np.sin(radians), np.cos(radians)]
    )


def image_names(n):
    return (
        [f"zed/left/{index:06d}.jpg" for index in range(n)],
        [f"zed/right/{index:06d}.jpg" for index in range(n)],
    )


class KeyframeSelectionTests(unittest.TestCase):
    def test_candidate_presets_match_declared_physical_thresholds(self):
        self.assertEqual(
            (
                KEYFRAME_PRESETS["dense"].translation_m,
                KEYFRAME_PRESETS["dense"].rotation_deg,
                KEYFRAME_PRESETS["dense"].max_elapsed_s,
            ),
            (0.08, 1.0, 1.5),
        )
        self.assertEqual(
            (
                KEYFRAME_PRESETS["balanced"].translation_m,
                KEYFRAME_PRESETS["balanced"].rotation_deg,
            ),
            (0.10, 2.0),
        )
        self.assertEqual(
            (
                KEYFRAME_PRESETS["sparse"].translation_m,
                KEYFRAME_PRESETS["sparse"].rotation_deg,
            ),
            (0.15, 3.0),
        )

    def test_turn_region_and_gnss_transition_force_auditable_frames(self):
        n = 9
        timestamps = np.arange(n, dtype=float) * 0.2
        positions = np.column_stack([timestamps * 0.05, np.zeros((n, 2))])
        yaw = [0, 0, 0, 10, 45, 80, 90, 90, 90]
        turn = np.array([False, False, False, True, True, True, True, False, False])
        status = np.array([2, 2, 2, 2, 2, 0, 0, 0, 0])
        blur = np.ones(n)
        blur[4:7] = 0.0
        selection = select_keyframes(
            timestamps,
            positions,
            yaw_quaternions(yaw),
            config=KeyframeConfig(
                translation_m=10.0,
                rotation_deg=180.0,
                max_elapsed_s=10.0,
                min_blur_score=0.5,
                revisit_distance_m=0,
            ),
            blur_score=blur,
            rtk_status=status,
            turn_region=turn,
        )
        records = {item.frame_index: item for item in selection.keyframes}
        self.assertEqual(selection.indices[0], 0)
        self.assertEqual(selection.indices[-1], n - 1)
        self.assertIn("turn_entry", records[3].reasons)
        self.assertIn("turn_peak", records[4].reasons)
        self.assertIn("gnss_before_change", records[4].reasons)
        self.assertIn("gnss_state_change", records[5].reasons)
        self.assertIn("turn_exit", records[6].reasons)
        self.assertIn("quality_override", records[4].reasons)

    def test_spatial_revisit_forces_both_sides_of_loop(self):
        timestamps = np.arange(8, dtype=float) * 2.0
        positions = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
             [3, 1, 0], [2, 1, 0], [1, 0.1, 0], [0, 0.1, 0]],
            dtype=float,
        )
        selection = select_keyframes(
            timestamps,
            positions,
            yaw_quaternions(np.zeros(8)),
            config=KeyframeConfig(
                translation_m=100,
                rotation_deg=180,
                max_elapsed_s=100,
                revisit_distance_m=0.2,
                revisit_min_separation_s=10,
                revisit_force_spacing_s=0,
            ),
        )
        records = {item.frame_index: item for item in selection.keyframes}
        self.assertIn("spatial_revisit_source", records[1].reasons)
        self.assertIn("spatial_revisit", records[6].reasons)
        self.assertEqual(records[6].revisit_source_index, 1)

    def test_bad_quality_delays_motion_candidate_but_elapsed_is_hard_bound(self):
        timestamps = np.array([0.0, 0.5, 1.0, 1.5])
        positions = np.column_stack([timestamps, np.zeros((4, 2))])
        selection = select_keyframes(
            timestamps,
            positions,
            yaw_quaternions(np.zeros(4)),
            config=KeyframeConfig(
                translation_m=0.4,
                rotation_deg=180,
                max_elapsed_s=1.0,
                min_blur_score=0.5,
                revisit_distance_m=0,
            ),
            blur_score=[1.0, 0.0, 0.0, 1.0],
        )
        records = {item.frame_index: item for item in selection.keyframes}
        self.assertNotIn(1, records)
        self.assertIn("max_elapsed", records[2].reasons)
        self.assertIn("quality_override", records[2].reasons)

    def test_all_sensor_quality_inputs_are_reported_and_gate_candidates(self):
        selection = select_keyframes(
            [0.0, 0.5, 1.0],
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            yaw_quaternions([0, 0, 0]),
            config=KeyframeConfig(
                translation_m=0.5,
                rotation_deg=180,
                max_elapsed_s=10,
                min_blur_score=10,
                min_exposure_quality=0.8,
                max_stereo_sync_residual_s=0.01,
                max_rtk_covariance_m2=0.02,
                acceptable_rtk_status=(2,),
                revisit_distance_m=0,
            ),
            blur_score=[20, 1, 20],
            exposure_quality=[1, 0.2, 1],
            stereo_sync_residual_s=[0, -0.05, 0],
            rtk_covariance_m2=[0.01, 0.5, 0.01],
            rtk_status=np.array([0, 0, 0], dtype=int),
        )
        self.assertNotIn(1, selection.indices)
        self.assertEqual(
            selection.frame_quality[1].issues,
            ("blur", "exposure", "stereo_sync", "rtk_covariance", "rtk_status"),
        )

    def test_metric_selection_is_stable_to_frame_sampling_density(self):
        config = KeyframeConfig(
            translation_m=0.1,
            rotation_deg=180,
            max_elapsed_s=100,
            revisit_distance_m=0,
        )
        selected_positions = []
        for count in (101, 201):
            x = np.linspace(0.0, 1.0, count)
            selection = select_keyframes(
                x * 10,
                np.column_stack([x, np.zeros((count, 2))]),
                yaw_quaternions(np.zeros(count)),
                config=config,
            )
            selected_positions.append(x[selection.indices])
        self.assertLessEqual(
            abs(len(selected_positions[0]) - len(selected_positions[1])), 1
        )
        for value in selected_positions[0]:
            self.assertLess(np.min(np.abs(selected_positions[1] - value)), 0.02)


class PairGraphTests(unittest.TestCase):
    def test_all_frames_get_stereo_but_only_keyframes_enter_solve_set(self):
        n = 5
        left, right = image_names(n)
        graph = build_pair_graph(
            np.arange(n, dtype=float),
            np.column_stack([np.arange(n) * 0.2, np.zeros((n, 2))]),
            np.tile([1.0, 0.0, 0.0], (n, 1)),
            left,
            right,
            [0, 2, 4],
        )
        stereo = [edge for edge in graph.edges if edge.mandatory]
        self.assertEqual(len(stereo), n)
        self.assertTrue(all(edge.reasons == ("stereo",) for edge in stereo))
        self.assertEqual(
            graph.solve_image_names,
            (left[0], right[0], left[2], right[2], left[4], right[4]),
        )
        self.assertEqual(len(graph.all_image_names), 2 * n)
        np.testing.assert_array_equal(graph.unattached_frame_indices, [])

    def test_temporal_bounds_use_distance_and_time_not_frame_index(self):
        timestamps = np.array([0.0, 0.1, 0.2, 5.0])
        positions = np.array(
            [[0, 0, 0], [0.1, 0, 0], [5, 0, 0], [0.2, 0, 0]], dtype=float
        )
        left, right = image_names(4)
        graph = build_pair_graph(
            timestamps,
            positions,
            np.tile([1.0, 0.0, 0.0], (4, 1)),
            left,
            right,
            [0, 1],
            config=PairGraphConfig(
                temporal_max_distance_m=0.5,
                temporal_max_seconds=1.0,
                revisit_distance_m=0,
            ),
        )
        temporal_frames = {
            (edge.frame_a, edge.frame_b)
            for edge in graph.edges_for_scope("solve")
            if "temporal" in edge.reasons
        }
        self.assertEqual(temporal_frames, {(0, 1)})

    def test_rtk_revisit_is_added_but_opposite_view_is_filtered(self):
        timestamps = np.array([0.0, 1.0, 20.0, 40.0])
        positions = np.array(
            [[0, 0, 0], [2, 0, 0], [0.1, 0, 0], [0.1, 0, 0]], dtype=float
        )
        directions = np.array(
            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=float
        )
        left, right = image_names(4)
        graph = build_pair_graph(
            timestamps,
            positions,
            directions,
            left,
            right,
            [0, 2],
            config=PairGraphConfig(
                temporal_max_distance_m=0.5,
                temporal_max_seconds=2,
                revisit_distance_m=0.2,
                revisit_min_separation_s=10,
                max_view_angle_deg=90,
            ),
        )
        revisits = {
            (edge.frame_a, edge.frame_b)
            for edge in graph.edges_for_scope("solve")
            if "rtk_spatial_revisit" in edge.reasons
        }
        self.assertIn((0, 2), revisits)
        self.assertNotIn((0, 3), revisits)
        self.assertNotIn((2, 3), revisits)

    def test_connectivity_edges_are_reserved_before_optional_pruning(self):
        n = 7
        left, right = image_names(n)
        graph = build_pair_graph(
            np.arange(n, dtype=float),
            np.column_stack([np.arange(n) * 0.4, np.zeros((n, 2))]),
            np.tile([1.0, 0.0, 0.0], (n, 1)),
            left,
            right,
            np.arange(n),
            config=PairGraphConfig(
                temporal_max_distance_m=0.5,
                temporal_max_seconds=1.1,
                max_temporal_neighbors=0,
                revisit_distance_m=0,
                max_cross_frame_degree=2,
            ),
        )
        frame_pairs = {
            (edge.frame_a, edge.frame_b)
            for edge in graph.edges_for_scope("solve")
            if edge.frame_a != edge.frame_b
        }
        self.assertEqual(frame_pairs, {(index, index + 1) for index in range(n - 1)})
        self.assertTrue(
            all(
                "connectivity" in edge.reasons
                for edge in graph.edges_for_scope("solve")
                if edge.frame_a != edge.frame_b
            )
        )
        degree = np.zeros(n, dtype=int)
        for first, second in frame_pairs:
            degree[first] += 1
            degree[second] += 1
        self.assertLessEqual(int(degree.max()), 2)

    def test_disconnected_selected_graph_fails_with_component_diagnostics(self):
        left, right = image_names(4)
        with self.assertRaisesRegex(
            ValueError,
            r"no physically admissible bridge; components=\[\[0, 1\], \[2, 3\]\]",
        ):
            build_pair_graph(
                [0.0, 0.5, 10.0, 10.5],
                [[0, 0, 0], [0.1, 0, 0], [100, 0, 0], [100.1, 0, 0]],
                np.tile([1.0, 0.0, 0.0], (4, 1)),
                left,
                right,
                [0, 1, 2, 3],
                config=PairGraphConfig(
                    temporal_max_distance_m=0.5,
                    temporal_max_seconds=1.0,
                    revisit_distance_m=0,
                ),
            )

    def test_degree_bound_that_cannot_connect_is_rejected(self):
        left, right = image_names(3)
        with self.assertRaisesRegex(
            ValueError, "bounded-degree connectivity requirement"
        ):
            build_pair_graph(
                [0.0, 1.0, 2.0],
                [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]],
                np.tile([1.0, 0.0, 0.0], (3, 1)),
                left,
                right,
                [0, 1, 2],
                config=PairGraphConfig(
                    temporal_max_distance_m=1.0,
                    temporal_max_seconds=3.0,
                    revisit_distance_m=0,
                    max_cross_frame_degree=1,
                ),
            )

    def test_neighbor_counts_and_registration_attachments_are_bounded(self):
        n = 12
        timestamps = np.arange(n, dtype=float) * 0.1
        positions = np.column_stack([np.arange(n) * 0.02, np.zeros((n, 2))])
        left, right = image_names(n)
        config = PairGraphConfig(
            temporal_max_distance_m=2,
            temporal_max_seconds=2,
            max_temporal_neighbors=2,
            revisit_distance_m=0,
            max_cross_frame_degree=3,
            max_registration_neighbors=1,
        )
        graph = build_pair_graph(
            timestamps,
            positions,
            np.tile([1.0, 0.0, 0.0], (n, 1)),
            left,
            right,
            [0, 3, 6, 9, 11],
            config=config,
        )
        frame_pairs = {
            (edge.frame_a, edge.frame_b)
            for edge in graph.edges_for_scope("solve")
            if not edge.mandatory
        }
        degree = np.zeros(n, dtype=int)
        for first, second in frame_pairs:
            degree[first] += 1
            degree[second] += 1
        self.assertLessEqual(int(degree.max()), config.max_cross_frame_degree)
        for index in set(range(n)) - {0, 3, 6, 9, 11}:
            parents = {
                (edge.frame_a, edge.frame_b)
                for edge in graph.edges_for_scope("register")
                if "registration" in edge.reasons and index in (edge.frame_a, edge.frame_b)
            }
            self.assertLessEqual(len(parents), config.max_registration_neighbors)
        np.testing.assert_array_equal(graph.unattached_frame_indices, [])

    def test_revisit_edges_receive_capacity_before_dense_temporal_edges(self):
        timestamps = np.array([0, 1, 2, 3, 20], dtype=float)
        positions = np.array(
            [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0], [0, 0.1, 0]],
            dtype=float,
        )
        left, right = image_names(5)
        graph = build_pair_graph(
            timestamps,
            positions,
            np.tile([1.0, 0.0, 0.0], (5, 1)),
            left,
            right,
            [0, 1, 2, 3, 4],
            config=PairGraphConfig(
                temporal_max_distance_m=1,
                temporal_max_seconds=5,
                max_temporal_neighbors=4,
                revisit_distance_m=0.2,
                revisit_min_separation_s=10,
                max_revisit_neighbors=1,
                max_cross_frame_degree=2,
            ),
        )
        revisits = {
            (edge.frame_a, edge.frame_b)
            for edge in graph.edges_for_scope("solve")
            if "rtk_spatial_revisit" in edge.reasons
        }
        self.assertEqual(revisits, {(0, 4)})


if __name__ == "__main__":
    unittest.main()
