import os
import random
import collections
import time

from copy import deepcopy

import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from test_parameter import *
from classes.utils import *
from classes.env.navigation_env import NavigationEnv
from classes.agent.agent import Agent
from classes.agent.node_manager import NodeManager
from classes.navigation_metrics import (
    path_efficiency,
    path_length,
    rate,
    smoothness_cost,
    spl,
)
from classes.planner.navigation_planner import NavigationPlanner
from classes.planner.navigation_waypoint_ranker import NavigationWaypointRanker

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure reproducibility in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def proposed_coordinates(candidate, location):
    """Convert one predicted action sequence into absolute coordinates."""
    if USE_DELTA_POSITION:
        return location + np.cumsum(candidate, axis=0)
    return np.asarray(candidate, dtype=np.float32)


def snap_to_neighbors(proposed_coord, neighbor_coords):
    """Nearest valid graph neighbor and its distance to the raw proposal."""
    if len(neighbor_coords) == 0:
        return None, float('inf')
    distances = np.linalg.norm(neighbor_coords - proposed_coord, axis=1)
    nearest = int(np.argmin(distances))
    return neighbor_coords[nearest], float(distances[nearest])


def goal_aware_fallback(neighbor_coords, location, goal, intended_direction):
    """Valid neighbor maximizing goal progress, tie-broken by alignment
    with the policy's intended direction of travel."""
    if len(neighbor_coords) == 0:
        return None
    best_neighbor = None
    best_score = float('inf')
    intended_norm = np.linalg.norm(intended_direction)
    for neighbor in neighbor_coords:
        neighbor_direction = neighbor - location
        neighbor_norm = np.linalg.norm(neighbor_direction)
        score = np.linalg.norm(goal - neighbor)
        if intended_norm > 1e-6 and neighbor_norm > 1e-6:
            cosine = np.dot(intended_direction, neighbor_direction) / (
                intended_norm * neighbor_norm
            )
            score += 0.5 * NODE_RESOLUTION * (1.0 - cosine)
        if score < best_score:
            best_score = score
            best_neighbor = neighbor
    return best_neighbor


class TestWorker:
    def __init__(self, meta_agent_id, policy, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.policy = policy
        self.global_step = global_step
        self.device = device
        self.save_image = save_image

        set_random_seed(global_step + 100047)
        self.env = NavigationEnv(
            global_step,
            TEST_N_AGENTS,
            plot=save_image,
            test=USE_TEST_DATASET,
        )
        self.node_manager = NodeManager(plot=save_image)
        self.robot_list = [Agent(i, self.node_manager, self.device, save_image) for i in range(TEST_N_AGENTS)]
        self.perf_metrics = dict()

        self.obs_horizon = policy.n_obs_steps
        self.action_horizon = policy.n_action_steps if ACTION_HORIZON == None else ACTION_HORIZON
        # assert self.action_horizon == 1, "Action horizon must be 1 for this driver"

        self.planned_path_x = []
        self.planned_path_y = []
        self.frame_index = 0
        self.collision_count = 0
        self.action_attempt_count = 0
        self.raw_invalid_action_count = 0
        self.fallback_count = 0
        self.fallback_failed_count = 0
        self.executed_actions = []
        self.generated_candidate_count = 0
        self.rejected_candidate_count = 0
        self.selected_candidate_scores = []
        self.candidate_scores = []
        self.inference_times = []
        self.ranking_log = []
        self.ranker = NavigationWaypointRanker(
            NODE_RESOLUTION,
            weights={
                'goal': W_GOAL,
                'connectivity': W_CONNECTIVITY,
                'utility': W_UTILITY,
                'guidepost': W_GUIDEPOST,
                'safety': W_SAFETY,
                'smoothness': W_SMOOTHNESS,
                'collision': W_COLLISION,
            },
        )

    def run_episode(self):
        done = False
        start_location = deepcopy(self.env.robot_locations[0])
        oracle_path = NavigationPlanner(
            self.env.ground_truth_info
        ).plan_waypoints(start_location, self.env.goal)
        oracle_path_length = path_length(oracle_path)

        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
        for robot in self.robot_list:
            robot.update_planning_state(self.env.robot_locations)

        # Get the first observation
        if DATA_TYPE == 'node':
            observation = self.robot_list[0].get_navigation_observation(
                self.env.goal
            )
            node_inputs = observation[0].squeeze(0)
            node_padding_mask = observation[1].squeeze(0)
            edge_mask = observation[2].squeeze(0)
            current_index = observation[3].squeeze(0)
            current_edge = observation[4].squeeze(0)
            edge_padding_mask = observation[5].squeeze(0)

            obs = {'node_inputs': node_inputs,
                'node_padding_mask': node_padding_mask,
                'edge_mask': edge_mask,
                'current_index': current_index,
                'current_edge': current_edge,
                'edge_padding_mask': edge_padding_mask}
        elif DATA_TYPE == 'map':
            image = deepcopy(self.env.robot_belief)
            state = deepcopy(self.env.robot_locations[0])

            agent_pos = state.astype(np.float32)
            image = image.astype(np.float32)/255
            image = np.expand_dims(image, axis=0)

            obs = {'image': image,
                'agent_pos': state}
        else:
            raise ValueError('Invalid data type, check test_parameter.py')

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque([obs] * self.obs_horizon, maxlen=self.obs_horizon)

        step = 0
        for step in range(MAX_EPISODE_STEP):
            # stack the last obs_horizon number of observations
            if DATA_TYPE == 'node':
                node_inputs = torch.stack([x['node_inputs'] for x in obs_deque])
                node_padding_mask = torch.stack([x['node_padding_mask'] for x in obs_deque])
                edge_mask = torch.stack([x['edge_mask'] for x in obs_deque])
                current_index = torch.stack([x['current_index'] for x in obs_deque])
                current_edge = torch.stack([x['current_edge'] for x in obs_deque])
                edge_padding_mask = torch.stack([x['edge_padding_mask'] for x in obs_deque])

                # device transfer
                # TODO check if need to down cast to int16/32 then upcast to int64 like dataset
                node_inputs = node_inputs.to(self.device, dtype=torch.float32) # (obs_horizon, 360, 5)
                node_padding_mask = node_padding_mask.to(self.device, dtype=torch.int16) # (obs_horizon, 1, 360)
                edge_mask = edge_mask.to(self.device, dtype=torch.int64) # (obs_horizon, 360, 360)
                current_index = current_index.to(self.device, dtype=torch.int64) # (obs_horizon, 1, 1)
                current_edge = current_edge.to(self.device, dtype=torch.int64) # (obs_horizon, 25, 1)
                edge_padding_mask = edge_padding_mask.to(self.device, dtype=torch.int16) # (obs_horizon, 1, 25)

                # observation dict
                obs_dict = {'node_inputs': node_inputs.unsqueeze(0),
                            'node_padding_mask': node_padding_mask.unsqueeze(0),
                            'edge_mask': edge_mask.unsqueeze(0),
                            'current_index': current_index.unsqueeze(0),
                            'current_edge': current_edge.unsqueeze(0),
                            'edge_padding_mask': edge_padding_mask.unsqueeze(0)}
            elif DATA_TYPE == 'map':
                # stack the last obs_horizon number of observations
                image = torch.stack([torch.tensor(x['image']) for x in obs_deque])
                agent_pos = torch.stack([torch.tensor(x['agent_pos']) for x in obs_deque])

                # device transfer
                image = image.to(self.device, dtype=torch.float32) # (obs_horizon, 512)
                agent_pos = agent_pos.to(self.device, dtype=torch.float32) # (obs_horizon, 2)

                # observation dict
                obs_dict = {'image': image.unsqueeze(0),
                            'agent_pos': agent_pos.unsqueeze(0)}
            else:
                raise ValueError('Invalid data type, check test_parameter.py')
        
            # Infer K trajectories in one batched denoising pass. Mode A is
            # exactly the original single-sample policy path.
            num_samples = 1 if EXPERIMENT_MODE == 'A' else NUM_ACTION_SAMPLES
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            inference_start = time.perf_counter()
            with torch.no_grad():
                action_dict = self.policy.predict_action(
                    obs_dict, num_samples=num_samples
                )
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)

            # (K, pred_horizon, action_dim) continuous waypoint deltas.
            if 'trajectory_candidates' in action_dict:
                candidates = action_dict['trajectory_candidates'][0].cpu().numpy()
            else:
                # Compatibility with checkpoints instantiated from an older
                # policy class implementation.
                candidates = action_dict['action_pred'].cpu().numpy()
                if candidates.ndim == 2:
                    candidates = candidates[None]

            start = self.obs_horizon - 1
            snap_radius = SNAP_RADIUS_FACTOR * NODE_RESOLUTION
            current_location = deepcopy(self.env.robot_locations[0])
            current_node = self.robot_list[0].node_manager.nodes_dict.find(
                current_location.tolist()
            ).data
            neighbor_coords = np.array(
                [
                    neighbor
                    for neighbor in current_node.neighbor_list
                    if not np.all(neighbor == current_location)
                ],
                dtype=np.float32,
            ).reshape(-1, 2)
            candidate_deltas = candidates[:, start, :]
            if not USE_DELTA_POSITION:
                candidate_deltas = candidate_deltas - current_location

            robot = self.robot_list[0]
            base_evaluation = self.ranker.evaluate(
                candidate_deltas=candidate_deltas,
                current_position=current_location,
                goal=self.env.goal,
                node_coords=robot.node_coords,
                node_features=obs['node_inputs'][:len(robot.node_coords)].cpu().numpy(),
                edge_mask=robot.adjacent_matrix,
                current_index=robot.current_index,
                neighbor_indices=robot.neighbor_indices,
                snap_radius=snap_radius,
                previous_action=(
                    self.executed_actions[-1] if self.executed_actions else None
                ),
            )
            collision_free = np.zeros(len(candidates), dtype=bool)
            for candidate_index, snapped_coord in enumerate(
                base_evaluation['snapped_waypoints']
            ):
                if base_evaluation['graph_valid'][candidate_index]:
                    collision_free[candidate_index] = not check_collision(
                        current_location, snapped_coord, self.env.belief_info
                    )
            evaluation = self.ranker.evaluate(
                candidate_deltas=candidate_deltas,
                current_position=current_location,
                goal=self.env.goal,
                node_coords=robot.node_coords,
                node_features=obs['node_inputs'][:len(robot.node_coords)].cpu().numpy(),
                edge_mask=robot.adjacent_matrix,
                current_index=robot.current_index,
                neighbor_indices=robot.neighbor_indices,
                snap_radius=snap_radius,
                previous_action=(
                    self.executed_actions[-1] if self.executed_actions else None
                ),
                collision_free=collision_free,
            )
            selection_mode = EXPERIMENT_MODE
            if not ENABLE_RANKING and selection_mode in ('C', 'D'):
                selection_mode = 'B' if num_samples > 1 else 'A'
            best_index, no_survivor = self.ranker.select(
                evaluation, mode=selection_mode
            )
            action_pred = candidates[best_index]
            action_dict['candidate_scores'] = torch.as_tensor(
                evaluation['scores'][None], device=self.device
            )
            action_dict['selected_waypoint'] = torch.as_tensor(
                action_pred[start:start + self.action_horizon][None],
                device=self.device,
            )
            action_dict['selected_index'] = torch.tensor(
                [best_index], device=self.device, dtype=torch.long
            )

            rejected = int(np.sum(~evaluation['valid_mask']))
            selected_score = float(evaluation['scores'][best_index])
            breakdown = self.ranker.selected_breakdown(evaluation, best_index)
            self.generated_candidate_count += len(candidates)
            self.rejected_candidate_count += rejected
            self.selected_candidate_scores.append(selected_score)
            self.candidate_scores.extend(evaluation['scores'].tolist())
            self.ranking_log.append({
                'step': int(step),
                'mode': selection_mode,
                'generated_candidates': int(len(candidates)),
                'rejected_candidates': rejected,
                'selected_index': int(best_index),
                'selected_score': selected_score,
                'ranking_breakdown': breakdown,
                'fallback_required': bool(no_survivor),
                'fallback_used': False,
            })
            inference_time = time.perf_counter() - inference_start
            self.inference_times.append(inference_time)
            self.ranking_log[-1]['inference_time_ms'] = 1000.0 * inference_time

            # only take action_horizon number of actions
            end = start + self.action_horizon
            action = action_pred[start:end,:] # (action_horizon, action_dim)
            
            # execute action_horizon number of steps without replanning
            for action_step in range(self.action_horizon):
                if action_step == 0:
                    planned_location = deepcopy(self.env.robot_locations[0])
                    self.planned_path_x.append([planned_location[0]])
                    self.planned_path_y.append([planned_location[1]])
                    # get planned path for visualization
                    if USE_DELTA_POSITION:
                        for i in range(start, len(action_pred)):
                            planned_location = planned_location + action_pred[i]
                            self.planned_path_x[step].append(planned_location[0])
                            self.planned_path_y[step].append(planned_location[1])
                    else:
                        for i in range(start, len(action_pred)):
                            planned_location = action_pred[i]
                            self.planned_path_x[step].append(planned_location[0])
                            self.planned_path_y[step].append(planned_location[1])
                else:
                    self.planned_path_x.append(self.planned_path_x[step - action_step])
                    self.planned_path_y.append(self.planned_path_y[step - action_step])
                    pass
                # print(f"Step: {step}, Action Step: {action_step}")
                if USE_DELTA_POSITION:
                    selected_coord = self.env.robot_locations[0] + action[action_step]
                else:
                    selected_coord = action[action_step]
                previous_location = deepcopy(self.env.robot_locations[0])
                self.action_attempt_count += 1

                current_node = self.robot_list[0].node_manager.nodes_dict.find(self.env.robot_locations[0].tolist()).data
                step_neighbor_coords = np.array(
                    [
                        neighbor
                        for neighbor in current_node.neighbor_list
                        if not np.all(neighbor == self.env.robot_locations[0])
                    ],
                    dtype=np.float32,
                ).reshape(-1, 2)

                ## Safety layer: snap the continuous proposal onto the graph
                snapped_coord, snap_error = snap_to_neighbors(
                    selected_coord, step_neighbor_coords
                )
                collision_unsafe = (
                    snapped_coord is not None
                    and snap_error <= snap_radius
                    and check_collision(
                        self.env.robot_locations[0],
                        snapped_coord,
                        self.env.belief_info,
                    )
                )
                if (
                    snapped_coord is None
                    or snap_error > snap_radius
                    or collision_unsafe
                ):
                    # Preserve the legacy collision-rate definition: any raw
                    # proposal rejected by graph/collision safety counts as an
                    # unsafe action. Invalid action rate is also reported
                    # separately for clarity.
                    self.collision_count += 1
                    self.raw_invalid_action_count += 1
                    self.ranking_log[-1]['fallback_used'] = True
                    intended_direction = np.sum(
                        action_pred[start:start + 3], axis=0
                    ) if USE_DELTA_POSITION else (
                        selected_coord - self.env.robot_locations[0]
                    )
                    fallback_coord = goal_aware_fallback(
                        step_neighbor_coords,
                        self.env.robot_locations[0],
                        self.env.goal,
                        intended_direction,
                    )
                    if fallback_coord is None:
                        self.fallback_failed_count += 1
                        selected_coord = deepcopy(self.env.robot_locations[0])
                    else:
                        self.fallback_count += 1
                        selected_coord = fallback_coord
                else:
                    selected_coord = snapped_coord
                
                # step the environment
                self.env.step(selected_coord, 0)
                self.executed_actions.append(
                    np.asarray(selected_coord, dtype=np.float32) - previous_location
                )
                
                # update robot state
                self.robot_list[0].update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[0]))
                self.robot_list[0].update_planning_state(self.env.robot_locations)

                if DATA_TYPE == 'node':
                    observation = self.robot_list[0].get_navigation_observation(
                        self.env.goal
                    )
                    node_inputs = observation[0].squeeze(0)
                    node_padding_mask = observation[1].squeeze(0)
                    edge_mask = observation[2].squeeze(0)
                    current_index = observation[3].squeeze(0)
                    current_edge = observation[4].squeeze(0)
                    edge_padding_mask = observation[5].squeeze(0)

                    obs = {'node_inputs': node_inputs,
                        'node_padding_mask': node_padding_mask,
                        'edge_mask': edge_mask,
                        'current_index': current_index,
                        'current_edge': current_edge,
                        'edge_padding_mask': edge_padding_mask}
                elif DATA_TYPE == 'map':
                    image = deepcopy(self.env.robot_belief)
                    state = deepcopy(self.env.robot_locations[0])

                    agent_pos = state.astype(np.float32)
                    image = image.astype(np.float32)/255
                    if len(image.shape) == 2: # add channel dimension
                        image = np.expand_dims(image, axis=0)

                    obs = {'image': image,
                        'agent_pos': state}
                else:
                    raise ValueError('Invalid data type, check test_parameter.py')

                obs_deque.append(obs)

                done = self.env.is_goal_reached()

                if self.save_image: # save gif
                    self.plot_env(step, frame_index=self.frame_index)
                    self.frame_index += 1

                if done: # exit action loop if done or collision
                    break

            if done: # exit episode loop if done
                break
                
        travel_dist = self.robot_list[0].travel_dist
        self.perf_metrics['travel_dist'] = travel_dist
        self.perf_metrics['start_goal_distance'] = float(
            np.linalg.norm(self.env.goal - start_location)
        )
        self.perf_metrics['oracle_path_length'] = oracle_path_length
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['timeout_rate'] = float(not done)
        self.perf_metrics['collision_rate'] = rate(
            self.collision_count, self.action_attempt_count
        )
        self.perf_metrics['raw_invalid_action_rate'] = rate(
            self.raw_invalid_action_count, self.action_attempt_count
        )
        self.perf_metrics['invalid_action_rate'] = self.perf_metrics[
            'raw_invalid_action_rate'
        ]
        self.perf_metrics['fallback_rate'] = rate(
            self.fallback_count, self.action_attempt_count
        )
        self.perf_metrics['fallback_failed_rate'] = rate(
            self.fallback_failed_count, self.action_attempt_count
        )
        self.perf_metrics['path_efficiency'] = path_efficiency(
            done, oracle_path_length, travel_dist
        )
        self.perf_metrics['spl'] = spl(done, oracle_path_length, travel_dist)
        self.perf_metrics['final_goal_distance'] = self.env.goal_distance()
        self.perf_metrics['smoothness_cost'] = smoothness_cost(
            self.executed_actions
        )
        self.perf_metrics['generated_candidates'] = self.generated_candidate_count
        self.perf_metrics['rejected_candidates'] = self.rejected_candidate_count
        self.perf_metrics['candidate_rejection_rate'] = rate(
            self.rejected_candidate_count, self.generated_candidate_count
        )
        self.perf_metrics['average_candidate_score'] = (
            float(np.mean(self.candidate_scores)) if self.candidate_scores else 0.0
        )
        self.perf_metrics['average_selected_candidate_score'] = (
            float(np.mean(self.selected_candidate_scores))
            if self.selected_candidate_scores else 0.0
        )
        self.perf_metrics['inference_time_ms'] = (
            1000.0 * float(np.mean(self.inference_times))
            if self.inference_times else 0.0
        )
        self.perf_metrics['ranking_breakdown'] = {
            key: float(np.mean([
                item['ranking_breakdown'][key] for item in self.ranking_log
            ]))
            for key in (
                'goal_progress', 'graph_connectivity', 'node_utility',
                'guidepost', 'safety', 'smoothness_penalty', 'collision_risk'
            )
        } if self.ranking_log else {}
        self.perf_metrics['ranking_log'] = self.ranking_log
        
        if self.save_image: # save gif
            make_gif(
                gifs_path,
                self.global_step,
                self.env.frame_files,
                float(done),
                delete_images=True,
                metric_name='navigation_success',
            )

    def plot_env(self, step, planned_paths=None, frame_index=None):
        self.env.global_frontiers = get_frontier_in_map(self.env.belief_info)
        plt.switch_backend('agg')
        color_list = ['r', 'b', 'g', 'y']
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 2)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.axis('off')
        goal_cell = get_cell_position_from_coords(
            self.env.goal, self.env.belief_info
        )
        plt.plot(goal_cell[0], goal_cell[1], 'g*', markersize=14, zorder=6)
        for robot in self.robot_list:
            c = color_list[robot.id]
            robot_cell = get_cell_position_from_coords(robot.location, robot.map_info)
            plt.plot(robot_cell[0], robot_cell[1], c+'o', markersize=16, zorder=5)
            plt.plot((np.array(robot.trajectory_x) - robot.map_info.map_origin_x) / robot.cell_size,
                    (np.array(robot.trajectory_y) - robot.map_info.map_origin_y) / robot.cell_size, c,
                    linewidth=2, zorder=1)
            plt.plot((np.array(self.planned_path_x[step]) - self.env.belief_info.map_origin_x) / self.env.cell_size,
                    (np.array(self.planned_path_y[step]) - self.env.belief_info.map_origin_y) / self.env.cell_size, 'g',
                    linewidth=1, zorder=2)

        plt.subplot(1, 2, 1)
        plt.imshow(self.env.robot_belief, cmap='gray')
        for robot in self.robot_list:
            c = color_list[robot.id]
            if robot.id == 0:
                nodes = get_cell_position_from_coords(robot.node_coords, robot.map_info)
                plt.imshow(robot.map_info.map, cmap='gray')
                plt.axis('off')
                plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.utility, zorder=2)
                for node, utility in zip(nodes, robot.utility):
                    plt.text(node[0], node[1], str(utility), zorder=3)

            robot_cell = get_cell_position_from_coords(robot.location, robot.map_info)
            plt.plot(robot_cell[0], robot_cell[1], c+'o', markersize=16, zorder=5)

        if len(self.env.global_frontiers) > 0:
            frontiers = get_cell_position_from_coords(np.array(list(self.env.global_frontiers)), self.env.belief_info).reshape(-1, 2)
            plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=2)

        # # Agent Edges
        # for coords in self.robot_list[0].node_coords:
        #     node = self.robot_list[0].node_manager.nodes_dict.find(coords.tolist()).data
        #     for neighbor_coords in node.neighbor_list[1:]:
        #         end = (np.array(neighbor_coords) - coords) / 2 + coords
        #         plt.plot((np.array([coords[0], end[0]]) - self.robot_list[0].map_info.map_origin_x) / self.robot_list[0].cell_size,
        #                         (np.array([coords[1], end[1]]) - self.robot_list[0].map_info.map_origin_y) / self.robot_list[0].cell_size, 'tan', zorder=1)

        plt.axis('off')
        plt.suptitle(
            'Goal distance: {:.4g}  Travel distance: {:.4g}'.format(
                self.env.goal_distance(),
                max(robot.travel_dist for robot in self.robot_list),
            )
        )
        plt.tight_layout()
        # plt.show()
        if frame_index is None:
            frame_index = step
        frame = '{}/{}_{}_samples.png'.format(
            gifs_path, self.global_step, frame_index
        )
        plt.savefig(frame, dpi=150)
        self.env.frame_files.append(frame)
        plt.close()
