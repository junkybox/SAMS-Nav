import os
import random
from copy import deepcopy

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset_parameter import *
from parameter import MAX_EPISODE_STEP
from classes.agent.agent import Agent
from classes.agent.node_manager import NodeManager
from classes.env.navigation_env import NavigationEnv
from classes.planner.navigation_planner import NavigationPlanner
from classes.navigation_metrics import path_efficiency, path_length, spl
from classes.utils import get_cell_position_from_coords, make_gif


os.makedirs(gifs_path, exist_ok=True)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DatasetWorker:
    """Collect one start-to-goal A* navigation demonstration."""

    def __init__(
        self,
        meta_agent_id,
        global_step,
        device='cpu',
        save_image=False,
        greedy=False,
    ):
        if TEST_N_AGENTS != 1:
            raise ValueError("Navigation dataset collection supports one agent.")
        if DATA_TYPE != 'node':
            raise ValueError("Navigation dataset collection currently supports node data.")

        set_random_seed(global_step + 47)
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.env = NavigationEnv(
            global_step,
            n_agent=1,
            plot=save_image,
            test=USE_TEST_DATASET,
        )
        if EXPERT_PLANNER_MAP not in ('ground_truth_oracle', 'belief'):
            raise ValueError(
                "EXPERT_PLANNER_MAP must be 'ground_truth_oracle' or 'belief'."
            )
        self.expert_planner_map = EXPERT_PLANNER_MAP
        self.planner = NavigationPlanner(self.env.ground_truth_info)
        self.node_manager = NodeManager(plot=save_image)
        self.robot = Agent(0, self.node_manager, device, save_image)
        self.episode_data = {}
        self.perf_metrics = {}

    def run_episode(self):
        start = deepcopy(self.env.robot_locations[0])
        oracle_path = NavigationPlanner(
            self.env.ground_truth_info
        ).plan_waypoints(start, self.env.goal)
        oracle_path_length = path_length(oracle_path)

        planner_map_info = (
            self.env.belief_info
            if self.expert_planner_map == 'belief'
            else self.env.ground_truth_info
        )
        self.planner = NavigationPlanner(planner_map_info)
        expert_path = self.planner.plan_waypoints(start, self.env.goal)
        if len(expert_path) < 2:
            self._set_failed_episode(start, oracle_path_length)
            return

        self.env.goal = expert_path[-1].copy()
        self.robot.update_graph(self.env.belief_info, start)
        self.robot.update_planning_state(self.env.robot_locations)

        records = {
            'node_inputs': [],
            'node_padding_mask': [],
            'edge_mask': [],
            'current_index': [],
            'current_edge': [],
            'edge_padding_mask': [],
            'action': [],
            'goal': [],
            'pose': [],
        }

        max_steps = min(len(expert_path) - 1, MAX_EPISODE_STEP)
        for step, next_waypoint in enumerate(expert_path[1:max_steps + 1]):
            observation = self.robot.get_navigation_observation(self.env.goal)
            keys = (
                'node_inputs',
                'node_padding_mask',
                'edge_mask',
                'current_index',
                'current_edge',
                'edge_padding_mask',
            )
            for key, value in zip(keys, observation):
                records[key].append(value.detach().cpu().numpy())

            current_pose = deepcopy(self.env.robot_locations[0])
            records['pose'].append(current_pose)
            records['goal'].append(self.env.goal.copy())
            records['action'].append(next_waypoint - current_pose)

            if self.save_image:
                self.plot_env(step, expert_path)

            self.env.step(next_waypoint, 0)
            self.robot.update_graph(
                self.env.belief_info, deepcopy(self.env.robot_locations[0])
            )
            self.robot.update_planning_state(self.env.robot_locations)
            if self.env.done:
                break

        success = self.env.is_goal_reached()
        episode_length = len(records['action'])
        self.episode_data = {
            'node_inputs': np.squeeze(
                np.asarray(records['node_inputs'], dtype=np.float32), axis=1
            ),
            'node_padding_mask': np.squeeze(
                np.asarray(records['node_padding_mask'], dtype=np.int16), axis=1
            ),
            'edge_mask': np.squeeze(
                np.asarray(records['edge_mask'], dtype=np.int16), axis=1
            ),
            'current_index': np.squeeze(
                np.asarray(records['current_index'], dtype=np.int32), axis=1
            ),
            'current_edge': np.squeeze(
                np.asarray(records['current_edge'], dtype=np.int32), axis=1
            ),
            'edge_padding_mask': np.squeeze(
                np.asarray(records['edge_padding_mask'], dtype=np.int16), axis=1
            ),
            'action': np.asarray(records['action'], dtype=np.float32),
            'goal': np.asarray(records['goal'], dtype=np.float32),
            'pose': np.asarray(records['pose'], dtype=np.float32),
            'episode_end': np.asarray([episode_length], dtype=np.int32),
        }
        self.perf_metrics = {
            'travel_dist': self.robot.travel_dist,
            'success_rate': success,
            'path_steps': episode_length,
            'start_goal_distance': float(np.linalg.norm(self.env.goal - start)),
            'oracle_path_length': oracle_path_length,
            'expert_path_length': path_length(expert_path),
            'path_efficiency': path_efficiency(
                success, oracle_path_length, self.robot.travel_dist
            ),
            'spl': spl(success, oracle_path_length, self.robot.travel_dist),
            'expert_planner_map': self.expert_planner_map,
        }

        if self.save_image and self.env.frame_files:
            self.plot_env(episode_length, expert_path)
            make_gif(
                gifs_path,
                self.global_step,
                self.env.frame_files,
                float(success),
                delete_images=True,
                metric_name='navigation_success',
            )

    def _set_failed_episode(self, start, oracle_path_length):
        self.episode_data = {}
        self.perf_metrics = {
            'travel_dist': 0.0,
            'success_rate': False,
            'path_steps': 0,
            'start_goal_distance': self.env.goal_distance(),
            'oracle_path_length': oracle_path_length,
            'expert_path_length': 0.0,
            'path_efficiency': 0.0,
            'spl': 0.0,
            'expert_planner_map': self.expert_planner_map,
        }

    def plot_env(self, step, expert_path):
        plt.switch_backend('agg')
        figure, axes = plt.subplots(1, 2, figsize=(10, 5))
        robot_cell = get_cell_position_from_coords(
            self.env.robot_locations[0], self.env.belief_info
        )
        goal_cell = get_cell_position_from_coords(
            self.env.goal, self.env.belief_info
        )
        path_cells = get_cell_position_from_coords(
            np.asarray(expert_path), self.env.belief_info
        )

        axes[0].imshow(self.env.robot_belief, cmap='gray')
        axes[0].plot(path_cells[:, 0], path_cells[:, 1], 'b-', linewidth=1)
        axes[0].plot(robot_cell[0], robot_cell[1], 'ro', label='robot')
        axes[0].plot(goal_cell[0], goal_cell[1], 'g*', markersize=14, label='goal')
        axes[0].set_title('Policy observation: partial belief')
        axes[0].legend()
        axes[0].axis('off')

        axes[1].imshow(self.env.ground_truth, cmap='gray')
        axes[1].plot(path_cells[:, 0], path_cells[:, 1], 'b-', linewidth=1)
        axes[1].plot(robot_cell[0], robot_cell[1], 'ro')
        axes[1].plot(goal_cell[0], goal_cell[1], 'g*', markersize=14)
        axes[1].set_title('Expert-only ground truth')
        axes[1].axis('off')

        figure.suptitle(
            f'Navigation episode {self.global_step}, step {step}, '
            f'goal distance {self.env.goal_distance():.2f} m'
        )
        figure.tight_layout()
        frame = os.path.join(
            gifs_path, f'{self.global_step}_{step:04d}_navigation.png'
        )
        figure.savefig(frame, dpi=120)
        plt.close(figure)
        self.env.frame_files.append(frame)
