"""Deterministic metric/time/RTK pair-graph planning without COLMAP I/O."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


@dataclass(frozen=True)
class PairGraphConfig:
    temporal_max_distance_m: float = 1.5
    temporal_max_seconds: float = 3.0
    max_temporal_neighbors: int = 4
    revisit_distance_m: float = 0.75
    revisit_min_separation_s: float = 10.0
    max_revisit_neighbors: int = 3
    max_cross_frame_degree: int = 8
    max_view_angle_deg: float = 100.0
    registration_max_distance_m: float = 1.5
    registration_max_seconds: float = 3.0
    max_registration_neighbors: int = 2
    match_right_camera_temporally: bool = True

    def __post_init__(self):
        positive = {
            "temporal_max_distance_m": self.temporal_max_distance_m,
            "temporal_max_seconds": self.temporal_max_seconds,
            "max_view_angle_deg": self.max_view_angle_deg,
            "registration_max_distance_m": self.registration_max_distance_m,
            "registration_max_seconds": self.registration_max_seconds,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.max_view_angle_deg < 180:
            raise ValueError("max_view_angle_deg must be between 0 and 180")
        if (
            not np.isfinite(self.revisit_distance_m)
            or self.revisit_distance_m < 0
            or not np.isfinite(self.revisit_min_separation_s)
            or self.revisit_min_separation_s < 0
        ):
            raise ValueError("revisit thresholds must be finite and non-negative")
        for name in (
            "max_temporal_neighbors",
            "max_revisit_neighbors",
            "max_cross_frame_degree",
            "max_registration_neighbors",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class PairEdge:
    image_a: str
    image_b: str
    frame_a: int
    frame_b: int
    reasons: tuple[str, ...]
    scope: Literal["solve", "register"]
    mandatory: bool = False


@dataclass(frozen=True)
class PairGraph:
    frame_count: int
    keyframe_indices: np.ndarray
    all_image_names: tuple[str, ...]
    solve_image_names: tuple[str, ...]
    edges: tuple[PairEdge, ...]
    unattached_frame_indices: np.ndarray

    def edges_for_scope(self, scope: str) -> tuple[PairEdge, ...]:
        return tuple(edge for edge in self.edges if edge.scope == scope)

    def to_pairs_text(self, scope: str | None = None) -> str:
        edges = self.edges if scope is None else self.edges_for_scope(scope)
        return "".join(f"{edge.image_a} {edge.image_b}\n" for edge in edges)


_REASON_ORDER = (
    "stereo",
    "connectivity",
    "temporal",
    "rtk_spatial_revisit",
    "registration",
)


def _timestamps(values) -> np.ndarray:
    timestamps = np.asarray(values, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError("timestamps_s must be a non-empty vector")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps_s must be finite and strictly increasing")
    return timestamps


def _vectors(values, n: int, name: str) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float64)
    if vectors.shape != (n, 3) or not np.isfinite(vectors).all():
        raise ValueError(f"{name} must have finite shape ({n}, 3)")
    return vectors


def _directions(values, n: int) -> np.ndarray:
    directions = _vectors(values, n, "view_directions")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("view_directions must be non-zero")
    return directions / norms[:, None]


def _names(values: Sequence[str], n: int, label: str) -> tuple[str, ...]:
    names = tuple(values)
    if len(names) != n or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{label} must contain {n} non-empty image names")
    return names


def _keyframes(values, n: int) -> np.ndarray:
    indices = np.asarray(values)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("keyframe_indices must be a one-dimensional integer array")
    indices = indices.astype(np.int64, copy=False)
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("keyframe_indices are empty or out of range")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("keyframe_indices must be sorted and unique")
    return indices


def _compatible(
    directions: np.ndarray, first: int, second: int, minimum_dot: float
) -> bool:
    dot = float(np.dot(directions[first], directions[second]))
    return dot + 1e-12 >= minimum_dot


def _within(value: float, limit: float) -> bool:
    return value <= limit + 1e-12 * max(1.0, abs(limit))


def _at_least(value: float, limit: float) -> bool:
    return value + 1e-12 * max(1.0, abs(limit)) >= limit


def _ranked_candidates(
    candidates: list[tuple[int, float, float]],
    distance_limit: float,
    time_limit: float,
) -> list[int]:
    return [
        index
        for index, _, _ in sorted(
            candidates,
            key=lambda item: (
                item[1] / distance_limit + item[2] / time_limit,
                item[1],
                item[2],
                item[0],
            ),
        )
    ]


def _components(
    vertices: np.ndarray, edges: Sequence[tuple[int, int]]
) -> list[list[int]]:
    adjacency = {int(vertex): set() for vertex in vertices}
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    result: list[list[int]] = []
    unseen = set(adjacency)
    while unseen:
        seed = min(unseen)
        stack = [seed]
        component: list[int] = []
        unseen.remove(seed)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        result.append(sorted(component))
    return sorted(result, key=lambda item: item[0])


def _reserve_connected_edges(
    selected: np.ndarray,
    candidates: dict[tuple[int, int], tuple[set[str], float, float]],
    config: PairGraphConfig,
) -> tuple[dict[tuple[int, int], set[str]], np.ndarray]:
    """Reserve a deterministic bounded-degree spanning tree.

    The tree is selected from physically admissible temporal/revisit pairs
    before optional match-density pruning. Stereo pairs are deliberately not
    considered here because they connect cameras within one frame, not two
    independently solvable frames.
    """
    if len(selected) == 1:
        return {}, np.zeros(int(selected.max()) + 1, dtype=np.int64)
    physical_components = _components(selected, list(candidates))
    if len(physical_components) != 1:
        raise ValueError(
            "selected solve frame graph has no physically admissible bridge; "
            f"components={physical_components}"
        )
    if config.max_cross_frame_degree == 0:
        raise ValueError(
            "selected solve frame graph cannot be connected with "
            "max_cross_frame_degree=0"
        )

    parent = {int(vertex): int(vertex) for vertex in selected}

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    degree = np.zeros(int(selected.max()) + 1, dtype=np.int64)
    reserved: dict[tuple[int, int], set[str]] = {}
    ordered = sorted(
        candidates.items(),
        key=lambda item: (
            "temporal" not in item[1][0],
            item[1][1] / config.temporal_max_distance_m
            + item[1][2] / config.temporal_max_seconds
            if "temporal" in item[1][0]
            else item[1][1] / max(config.revisit_distance_m, 1e-12),
            item[1][2],
            item[0][1] - item[0][0],
            item[0],
        ),
    )
    for (first, second), (reasons, _, _) in ordered:
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            continue
        if (
            degree[first] >= config.max_cross_frame_degree
            or degree[second] >= config.max_cross_frame_degree
        ):
            continue
        reserved[(first, second)] = set(reasons) | {"connectivity"}
        degree[first] += 1
        degree[second] += 1
        parent[root_second] = root_first
        if len(reserved) == len(selected) - 1:
            break

    components = _components(selected, list(reserved))
    if len(components) != 1:
        raise ValueError(
            "selected solve frame graph cannot satisfy the bounded-degree "
            f"connectivity requirement (max_cross_frame_degree="
            f"{config.max_cross_frame_degree}); components={components}"
        )
    return reserved, degree


def build_pair_graph(
    timestamps_s,
    positions_m,
    view_directions,
    left_image_names: Sequence[str],
    right_image_names: Sequence[str],
    keyframe_indices,
    *,
    config: PairGraphConfig = PairGraphConfig(),
) -> PairGraph:
    """Build solve and later-registration pairs from physical relationships.

    Same-frame stereo edges cover every input frame. Cross-frame solve edges
    connect only the keyframe subset. Each non-keyframe also receives bounded
    registration edges to compatible solved keyframes, allowing all images to
    be registered after the solve without changing the solve subset.
    """
    timestamps = _timestamps(timestamps_s)
    n = len(timestamps)
    positions = _vectors(positions_m, n, "positions_m")
    directions = _directions(view_directions, n)
    left = _names(left_image_names, n, "left_image_names")
    right = _names(right_image_names, n, "right_image_names")
    if len(set(left + right)) != 2 * n:
        raise ValueError("all left/right image names must be unique")
    selected = _keyframes(keyframe_indices, n)
    selected_set = set(selected.tolist())
    minimum_dot = float(np.cos(np.deg2rad(config.max_view_angle_deg)))

    proposed_pairs: dict[tuple[int, int], set[str]] = {}
    admissible_pairs: dict[
        tuple[int, int], tuple[set[str], float, float]
    ] = {}
    for offset, current in enumerate(selected):
        current = int(current)
        previous = selected[:offset]
        if previous.size == 0:
            continue
        delta_t = timestamps[current] - timestamps[previous]
        distance = np.linalg.norm(positions[previous] - positions[current], axis=1)

        temporal = []
        temporal_lookup: dict[int, tuple[float, float]] = {}
        for candidate, metres, seconds in zip(previous, distance, delta_t):
            candidate = int(candidate)
            if (
                _within(float(seconds), config.temporal_max_seconds)
                and _within(float(metres), config.temporal_max_distance_m)
                and _compatible(directions, candidate, current, minimum_dot)
            ):
                temporal.append((candidate, float(metres), float(seconds)))
                temporal_lookup[candidate] = (float(metres), float(seconds))
        temporal_order = _ranked_candidates(
            temporal,
            config.temporal_max_distance_m,
            config.temporal_max_seconds,
        )[: config.max_temporal_neighbors]

        revisit = []
        revisit_lookup: dict[int, tuple[float, float]] = {}
        if config.revisit_distance_m > 0:
            for candidate, metres, seconds in zip(previous, distance, delta_t):
                candidate = int(candidate)
                if (
                    _at_least(
                        float(seconds), config.revisit_min_separation_s
                    )
                    and _within(float(metres), config.revisit_distance_m)
                    and _compatible(directions, candidate, current, minimum_dot)
                ):
                    revisit.append((candidate, float(metres), float(seconds)))
                    revisit_lookup[candidate] = (float(metres), float(seconds))
        revisit_order = [
            item[0]
            for item in sorted(
                revisit, key=lambda item: (item[1], -item[2], item[0])
            )
        ][: config.max_revisit_neighbors]

        for candidate, reason in [
            *((item, "temporal") for item in temporal_order),
            *((item, "rtk_spatial_revisit") for item in revisit_order),
        ]:
            key = (candidate, current)
            proposed_pairs.setdefault(key, set()).add(reason)

        for candidate in sorted(set(temporal_lookup) | set(revisit_lookup)):
            reasons: set[str] = set()
            if candidate in temporal_lookup:
                reasons.add("temporal")
                metres, seconds = temporal_lookup[candidate]
            else:
                metres, seconds = revisit_lookup[candidate]
            if candidate in revisit_lookup:
                reasons.add("rtk_spatial_revisit")
            admissible_pairs[(candidate, current)] = (
                reasons,
                metres,
                seconds,
            )

    # First reserve a physically admissible frame-level spanning tree. It is
    # never displaced by optional revisit/temporal density pruning.
    reserved, reserved_degree = _reserve_connected_edges(
        selected, admissible_pairs, config
    )
    frame_pairs: dict[tuple[int, int], set[str]] = dict(reserved)
    degree = np.zeros(n, dtype=np.int64)
    degree[: len(reserved_degree)] = reserved_degree
    revisit_degree = np.zeros(n, dtype=np.int64)
    for (first, second), reasons in reserved.items():
        if "rtk_spatial_revisit" in reasons:
            revisit_degree[first] += 1
            revisit_degree[second] += 1

    # Revisit constraints are first-class and receive remaining bounded
    # capacity before ordinary temporal edges.
    ordered_proposals = sorted(
        proposed_pairs.items(),
        key=lambda item: (
            "rtk_spatial_revisit" not in item[1],
            item[0][1],
            item[0][0],
        ),
    )
    for (first, second), proposed_reasons in ordered_proposals:
        if (first, second) in frame_pairs:
            frame_pairs[(first, second)].update(proposed_reasons)
            continue
        reasons = set(proposed_reasons)
        if "rtk_spatial_revisit" in reasons and (
            revisit_degree[first] >= config.max_revisit_neighbors
            or revisit_degree[second] >= config.max_revisit_neighbors
        ):
            reasons.remove("rtk_spatial_revisit")
        if not reasons or (
            degree[first] >= config.max_cross_frame_degree
            or degree[second] >= config.max_cross_frame_degree
        ):
            continue
        frame_pairs[(first, second)] = reasons
        degree[first] += 1
        degree[second] += 1
        if "rtk_spatial_revisit" in reasons:
            revisit_degree[first] += 1
            revisit_degree[second] += 1

    final_components = _components(selected, list(frame_pairs))
    if len(final_components) != 1:
        raise AssertionError(
            "internal error: solve frame graph lost reserved connectivity; "
            f"components={final_components}"
        )

    edge_map: dict[tuple[str, str], PairEdge] = {}

    def add_edge(
        image_a: str,
        image_b: str,
        frame_a: int,
        frame_b: int,
        reasons: set[str],
        scope: Literal["solve", "register"],
        mandatory: bool = False,
    ) -> None:
        key = tuple(sorted((image_a, image_b)))
        existing = edge_map.get(key)
        combined = set(reasons)
        if existing is not None:
            combined.update(existing.reasons)
            mandatory = mandatory or existing.mandatory
        ordered = tuple(reason for reason in _REASON_ORDER if reason in combined)
        edge_map[key] = PairEdge(
            image_a=image_a,
            image_b=image_b,
            frame_a=frame_a,
            frame_b=frame_b,
            reasons=ordered,
            scope=scope,
            mandatory=mandatory,
        )

    for index in range(n):
        add_edge(
            left[index],
            right[index],
            index,
            index,
            {"stereo"},
            "solve" if index in selected_set else "register",
            mandatory=True,
        )
    for (first, second), reasons in frame_pairs.items():
        add_edge(left[first], left[second], first, second, reasons, "solve")
        if config.match_right_camera_temporally:
            add_edge(right[first], right[second], first, second, reasons, "solve")

    unattached = []
    for index in range(n):
        if index in selected_set:
            continue
        candidates = []
        for candidate in selected:
            candidate = int(candidate)
            seconds = abs(float(timestamps[index] - timestamps[candidate]))
            metres = float(np.linalg.norm(positions[index] - positions[candidate]))
            if (
                _within(seconds, config.registration_max_seconds)
                and _within(metres, config.registration_max_distance_m)
                and _compatible(directions, candidate, index, minimum_dot)
            ):
                candidates.append((candidate, metres, seconds))
        chosen = _ranked_candidates(
            candidates,
            config.registration_max_distance_m,
            config.registration_max_seconds,
        )[: config.max_registration_neighbors]
        if not chosen:
            unattached.append(index)
            continue
        for candidate in chosen:
            first, second = sorted((candidate, index))
            add_edge(
                left[first],
                left[second],
                first,
                second,
                {"registration"},
                "register",
            )
            if config.match_right_camera_temporally:
                add_edge(
                    right[first],
                    right[second],
                    first,
                    second,
                    {"registration"},
                    "register",
                )

    edges = tuple(
        sorted(
            edge_map.values(),
            key=lambda edge: (
                edge.scope != "solve",
                edge.frame_a,
                edge.frame_b,
                edge.image_a,
                edge.image_b,
            ),
        )
    )
    all_names = tuple(
        name for index in range(n) for name in (left[index], right[index])
    )
    solve_names = tuple(
        name for index in selected for name in (left[int(index)], right[int(index)])
    )
    return PairGraph(
        frame_count=n,
        keyframe_indices=selected.copy(),
        all_image_names=all_names,
        solve_image_names=solve_names,
        edges=edges,
        unattached_frame_indices=np.asarray(unattached, dtype=np.int64),
    )
