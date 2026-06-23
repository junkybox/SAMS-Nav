from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

from classes.utils import (
    check_collision,
    get_cell_position_from_coords,
    get_updating_node_coords,
)
from parameter import FREE, NODE_RESOLUTION


GridCell = Tuple[int, int]


class NavigationPlanner:
    """Grid A* expert for start-to-goal navigation demonstrations."""

    _NEIGHBORS = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    def __init__(self, map_info):
        self.map_info = map_info

    def plan(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        start_cell = self._world_to_cell(start)
        goal_cell = self._world_to_cell(goal)
        cell_path = self._a_star(start_cell, goal_cell)
        return [self._cell_to_world(cell) for cell in cell_path]

    def plan_waypoints(
        self, start: np.ndarray, goal: np.ndarray
    ) -> List[np.ndarray]:
        """Plan A* over collision-free NODE_RESOLUTION waypoints."""
        nodes, _ = get_updating_node_coords(
            start, self.map_info, check_connectivity=True
        )
        node_lookup = {
            (float(node[0]), float(node[1])): node for node in nodes
        }
        start_key = self._nearest_node_key(start, nodes)
        goal_key = self._nearest_node_key(goal, nodes)

        frontier = [(0.0, start_key)]
        came_from: Dict[GridCell, Optional[GridCell]] = {start_key: None}
        cost_so_far = {start_key: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_key:
                break
            for neighbor in self._waypoint_neighbors(current, node_lookup):
                new_cost = cost_so_far[current] + self._heuristic(current, neighbor)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._heuristic(neighbor, goal_key)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal_key not in came_from:
            return []

        path = []
        current: Optional[GridCell] = goal_key
        while current is not None:
            path.append(node_lookup[current].astype(np.float32))
            current = came_from[current]
        path.reverse()
        return path

    def _waypoint_neighbors(self, current, node_lookup):
        x, y = current
        for dx, dy in self._NEIGHBORS:
            neighbor = (
                float(np.round(x + dx * NODE_RESOLUTION, 1)),
                float(np.round(y + dy * NODE_RESOLUTION, 1)),
            )
            if neighbor not in node_lookup:
                continue
            if not check_collision(
                np.asarray(current), np.asarray(neighbor), self.map_info
            ):
                yield neighbor

    @staticmethod
    def _nearest_node_key(position, nodes):
        nearest = nodes[np.argmin(np.linalg.norm(nodes - position, axis=1))]
        return float(nearest[0]), float(nearest[1])

    def _a_star(self, start: GridCell, goal: GridCell) -> List[GridCell]:
        if not self._is_free(start) or not self._is_free(goal):
            return []

        frontier = [(0.0, start)]
        came_from: Dict[GridCell, Optional[GridCell]] = {start: None}
        cost_so_far = {start: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break

            for neighbor in self._valid_neighbors(current):
                step_cost = np.sqrt(2.0) if self._is_diagonal(current, neighbor) else 1.0
                new_cost = cost_so_far[current] + step_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._heuristic(neighbor, goal)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal not in came_from:
            return []

        path = []
        current: Optional[GridCell] = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def _valid_neighbors(self, cell: GridCell):
        x, y = cell
        for dx, dy in self._NEIGHBORS:
            neighbor = (x + dx, y + dy)
            if not self._is_free(neighbor):
                continue
            if dx != 0 and dy != 0:
                if not self._is_free((x + dx, y)) or not self._is_free((x, y + dy)):
                    continue
            yield neighbor

    def _is_free(self, cell: GridCell) -> bool:
        x, y = cell
        height, width = self.map_info.map.shape
        return 0 <= x < width and 0 <= y < height and self.map_info.map[y, x] == FREE

    def _world_to_cell(self, position: np.ndarray) -> GridCell:
        cell = get_cell_position_from_coords(position, self.map_info).reshape(2)
        return int(cell[0]), int(cell[1])

    def _cell_to_world(self, cell: GridCell) -> np.ndarray:
        x, y = cell
        return np.array(
            [
                self.map_info.map_origin_x + x * self.map_info.cell_size,
                self.map_info.map_origin_y + y * self.map_info.cell_size,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _heuristic(cell: GridCell, goal: GridCell) -> float:
        return float(np.hypot(cell[0] - goal[0], cell[1] - goal[1]))

    @staticmethod
    def _is_diagonal(first: GridCell, second: GridCell) -> bool:
        return first[0] != second[0] and first[1] != second[1]
