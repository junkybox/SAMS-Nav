from __future__ import annotations

from typing import Optional

import numpy as np

from classes.env.env import Env
from classes.utils import get_cell_position_from_coords, get_updating_node_coords
from parameter import FREE


class NavigationEnv(Env):
    """Goal-conditioned environment built on DARE's map and sensor model."""

    def __init__(
        self,
        episode_index: int,
        n_agent: int = 1,
        plot: bool = False,
        test: bool = False,
        goal: Optional[np.ndarray] = None,
        goal_tolerance: float = 1.0,
    ):
        if n_agent != 1:
            raise ValueError("The navigation scaffold currently supports one agent.")

        super().__init__(
            episode_index,
            n_agent=n_agent,
            plot=plot,
            test=test,
            randomize_test_start=True,
        )
        self.goal_tolerance = float(goal_tolerance)
        self.goal = (
            np.asarray(goal, dtype=np.float32)
            if goal is not None
            else self.sample_goal()
        )
        self.done = self.is_goal_reached()

    def sample_goal(self) -> np.ndarray:
        """Sample a connected goal on DARE's waypoint lattice."""
        start = self.robot_locations[0]
        connected_nodes, _ = get_updating_node_coords(
            start, self.ground_truth_info, check_connectivity=True
        )
        distances = np.linalg.norm(connected_nodes - start, axis=1)
        candidates = connected_nodes[distances >= self.sensor_range]
        if len(candidates) == 0:
            candidates = connected_nodes[distances > 0]
        if len(candidates) == 0:
            raise RuntimeError("Cannot sample a navigation goal distinct from start.")
        return candidates[np.random.randint(len(candidates))].astype(np.float32)

    def goal_vector(self, agent_id: int = 0) -> np.ndarray:
        return self.goal - self.robot_locations[agent_id]

    def goal_distance(self, agent_id: int = 0) -> float:
        return float(np.linalg.norm(self.goal_vector(agent_id)))

    def is_goal_reached(self, agent_id: int = 0) -> bool:
        return self.goal_distance(agent_id) <= self.goal_tolerance

    def step(self, next_waypoint, agent_id):
        super().step(next_waypoint, agent_id)
        self.done = self.is_goal_reached(agent_id)
        return self.done
