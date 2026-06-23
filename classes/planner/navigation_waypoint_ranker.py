"""Safety-aware ranking for diffusion waypoint samples.

The ranker is deliberately model-free.  It uses only the currently observed
belief graph, its node features, the goal, and sampled waypoint deltas.  Map
collision checks remain in the rollout safety layer and are passed in as a
boolean validation result; no ground-truth or hidden map is consumed here.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


DEFAULT_RANKING_WEIGHTS = {
    "goal": 2.0,
    "connectivity": 1.0,
    "utility": 0.35,
    "guidepost": 0.5,
    "safety": 2.0,
    "smoothness": 0.25,
    "collision": 3.0,
}


class NavigationWaypointRanker:
    """Rank sampled waypoint deltas against the current observed graph."""

    MODES = {"A", "B", "C", "D"}

    def __init__(self, node_resolution: float, weights: Optional[Dict] = None):
        self.node_resolution = float(node_resolution)
        self.weights = dict(DEFAULT_RANKING_WEIGHTS)
        if weights:
            unknown = set(weights) - set(self.weights)
            if unknown:
                raise ValueError(f"Unknown waypoint ranking weights: {unknown}")
            self.weights.update({key: float(value) for key, value in weights.items()})

    @staticmethod
    def _unit_alignment(vectors, direction):
        vectors = np.asarray(vectors, dtype=np.float32)
        direction = np.asarray(direction, dtype=np.float32)
        vector_norm = np.linalg.norm(vectors, axis=-1)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6:
            return np.zeros(len(vectors), dtype=np.float32)
        cosine = np.sum(vectors * direction, axis=-1) / np.maximum(
            vector_norm * direction_norm, 1e-6
        )
        return np.clip(0.5 * (cosine + 1.0), 0.0, 1.0)

    def evaluate(
        self,
        candidate_deltas,
        current_position,
        goal,
        node_coords,
        node_features,
        edge_mask,
        current_index,
        neighbor_indices,
        snap_radius,
        previous_action=None,
        collision_free=None,
    ):
        """Compute per-candidate score components and validation masks.

        ``node_features`` follows DARE's local layout
        ``[rel_x, rel_y, utility, guidepost, occupancy, goal_dx, goal_dy]``.
        ``collision_free`` is the result of the existing belief-map safety
        validator, never an oracle/ground-truth-map signal.
        """
        deltas = np.asarray(candidate_deltas, dtype=np.float32).reshape(-1, 2)
        current = np.asarray(current_position, dtype=np.float32).reshape(2)
        goal = np.asarray(goal, dtype=np.float32).reshape(2)
        coords = np.asarray(node_coords, dtype=np.float32).reshape(-1, 2)
        features = np.asarray(node_features, dtype=np.float32)
        adjacency = np.asarray(edge_mask)
        neighbors = np.asarray(neighbor_indices, dtype=np.int64).reshape(-1)
        neighbors = neighbors[(neighbors >= 0) & (neighbors < len(coords))]
        neighbors = neighbors[neighbors != int(current_index)]

        proposals = current[None] + deltas
        snapped = np.repeat(current[None], len(deltas), axis=0)
        target_indices = np.full(len(deltas), -1, dtype=np.int64)
        snap_errors = np.full(len(deltas), np.inf, dtype=np.float32)
        if len(neighbors):
            distances = np.linalg.norm(
                proposals[:, None, :] - coords[neighbors][None, :, :], axis=-1
            )
            nearest = np.argmin(distances, axis=1)
            target_indices = neighbors[nearest]
            snapped = coords[target_indices]
            snap_errors = distances[np.arange(len(deltas)), nearest]

        graph_valid = snap_errors <= float(snap_radius)
        if collision_free is None:
            collision_free = np.ones(len(deltas), dtype=bool)
        collision_free = np.asarray(collision_free, dtype=bool).reshape(-1)
        if len(collision_free) != len(deltas):
            raise ValueError("collision_free must contain one value per candidate")
        valid_mask = graph_valid & collision_free

        scale = max(self.node_resolution, 1e-6)
        goal_progress = (
            np.linalg.norm(goal - current)
            - np.linalg.norm(goal[None] - snapped, axis=-1)
        ) / scale

        degree = np.zeros(len(deltas), dtype=np.float32)
        valid_targets = target_indices >= 0
        if adjacency.ndim == 2 and adjacency.shape[0] == len(coords):
            node_degrees = np.sum(adjacency == 0, axis=1).astype(np.float32)
            node_degrees = np.maximum(node_degrees - 1.0, 0.0)
            max_degree = max(float(np.max(node_degrees)), 1.0)
            degree[valid_targets] = node_degrees[target_indices[valid_targets]] / max_degree
        graph_connectivity = 0.5 * graph_valid.astype(np.float32) + 0.5 * degree

        utility = np.zeros(len(deltas), dtype=np.float32)
        guidepost = np.zeros(len(deltas), dtype=np.float32)
        travel_direction = snapped - current
        if features.ndim == 2 and len(features) == len(coords):
            utility_values = np.maximum(features[:, 2], 0.0)
            utility_scale = max(float(np.max(utility_values)), 1e-6)
            target_utility = np.zeros(len(deltas), dtype=np.float32)
            target_utility[valid_targets] = (
                utility_values[target_indices[valid_targets]] / utility_scale
            )
            useful = np.flatnonzero(utility_values > 0)
            utility_alignment = np.zeros(len(deltas), dtype=np.float32)
            if len(useful):
                nearest_useful = useful[
                    np.argmin(np.linalg.norm(coords[useful] - current, axis=1))
                ]
                utility_alignment = self._unit_alignment(
                    travel_direction, coords[nearest_useful] - current
                )
            utility = 0.5 * target_utility + 0.5 * utility_alignment

            guidepost_values = np.clip(features[:, 3], 0.0, 1.0)
            target_guidepost = np.zeros(len(deltas), dtype=np.float32)
            target_guidepost[valid_targets] = guidepost_values[
                target_indices[valid_targets]
            ]
            guide_nodes = np.flatnonzero(guidepost_values > 0)
            guide_alignment = np.zeros(len(deltas), dtype=np.float32)
            if len(guide_nodes):
                nearest_guide = guide_nodes[
                    np.argmin(np.linalg.norm(coords[guide_nodes] - current, axis=1))
                ]
                guide_alignment = self._unit_alignment(
                    travel_direction, coords[nearest_guide] - current
                )
            guidepost = 0.5 * target_guidepost + 0.5 * guide_alignment

        # Graph degree is an observable clearance proxy: nodes with several
        # collision-free outgoing edges are less constrained by obstacles.
        snap_margin = np.clip(
            1.0 - snap_errors / max(float(snap_radius), 1e-6), 0.0, 1.0
        )
        safety = (0.5 * snap_margin + 0.5 * degree) * collision_free

        if previous_action is None:
            smoothness_penalty = np.zeros(len(deltas), dtype=np.float32)
        else:
            previous = np.asarray(previous_action, dtype=np.float32).reshape(2)
            smoothness_penalty = np.linalg.norm(deltas - previous, axis=-1) / scale

        snap_risk = np.clip(
            snap_errors / max(float(snap_radius), 1e-6), 0.0, 2.0
        )
        collision_risk = (
            snap_risk
            + (~graph_valid).astype(np.float32)
            + (~collision_free).astype(np.float32)
        )

        w = self.weights
        total = (
            w["goal"] * goal_progress
            + w["connectivity"] * graph_connectivity
            + w["utility"] * utility
            + w["guidepost"] * guidepost
            + w["safety"] * safety
            - w["smoothness"] * smoothness_penalty
            - w["collision"] * collision_risk
        )
        return {
            "scores": total.astype(np.float32),
            "goal_progress": goal_progress.astype(np.float32),
            "graph_connectivity": graph_connectivity.astype(np.float32),
            "node_utility": utility.astype(np.float32),
            "guidepost": guidepost.astype(np.float32),
            "safety": safety.astype(np.float32),
            "smoothness_penalty": smoothness_penalty.astype(np.float32),
            "collision_risk": collision_risk.astype(np.float32),
            "graph_valid": graph_valid,
            "collision_free": collision_free,
            "valid_mask": valid_mask,
            "snapped_waypoints": snapped.astype(np.float32),
            "snap_errors": snap_errors,
            "target_indices": target_indices,
        }

    @staticmethod
    def select(evaluation, mode="D", rng=None):
        """Select an index for one of the four paper experiment modes."""
        mode = str(mode).upper()
        if mode not in NavigationWaypointRanker.MODES:
            raise ValueError(f"Unknown experiment mode {mode!r}; expected A, B, C, or D")
        valid = np.flatnonzero(evaluation["valid_mask"])
        if len(valid) == 0:
            return 0, True
        if mode == "A":
            return 0, not bool(evaluation["valid_mask"][0])
        if mode == "B":
            rng = np.random if rng is None else rng
            return int(rng.choice(valid)), False
        key = "goal_progress" if mode == "C" else "scores"
        return int(valid[np.argmax(evaluation[key][valid])]), False

    @staticmethod
    def selected_breakdown(evaluation, selected_index):
        keys = (
            "goal_progress",
            "graph_connectivity",
            "node_utility",
            "guidepost",
            "safety",
            "smoothness_penalty",
            "collision_risk",
        )
        return {key: float(evaluation[key][selected_index]) for key in keys}
