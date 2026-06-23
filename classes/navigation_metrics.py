from __future__ import annotations

import numpy as np


def path_length(points) -> float:
    """Euclidean length of a waypoint path or executed trajectory."""
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return 0.0
    deltas = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=-1)))


def path_efficiency(success: bool, oracle_path_length: float, travel_dist: float) -> float:
    """Success-gated shortest-path efficiency, capped at 1.0."""
    if not success or oracle_path_length <= 0.0 or travel_dist <= 0.0:
        return 0.0
    return float(min(1.0, oracle_path_length / travel_dist))


def spl(success: bool, oracle_path_length: float, travel_dist: float) -> float:
    """Success weighted by path length, following the common SPL metric."""
    if not success or oracle_path_length <= 0.0:
        return 0.0
    return float(oracle_path_length / max(travel_dist, oracle_path_length))


def smoothness_cost(actions) -> float:
    """Mean squared change between consecutive 2-D action vectors."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[-2] < 2:
        return 0.0
    action_changes = np.diff(actions, axis=-2)
    return float(np.mean(np.sum(np.square(action_changes), axis=-1)))


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def training_proxy_metrics(
    pred_action,
    expert_action,
    obs_dict,
    n_obs_steps: int,
    node_resolution: float,
    coordinate_scale: float,
):
    """Offline proxies used when training has no closed-loop environment."""
    pred = pred_action.detach().cpu().numpy()
    expert = expert_action.detach().cpu().numpy()
    start = n_obs_steps - 1
    pred = pred[:, start:]
    expert = expert[:, start:]

    pred_end = np.sum(pred, axis=1)
    expert_end = np.sum(expert, axis=1)
    endpoint_error = np.linalg.norm(pred_end - expert_end, axis=-1)
    success_rate = float(np.mean(endpoint_error <= node_resolution))

    node_inputs = obs_dict['node_inputs'][:, start].detach().cpu().numpy()
    current_edge = (
        obs_dict['current_edge'][:, start].detach().cpu().numpy().reshape(len(pred), -1)
    )
    edge_padding_mask = (
        obs_dict['edge_padding_mask'][:, start]
        .detach()
        .cpu()
        .numpy()
        .reshape(len(pred), -1)
    )

    # continuous predictions are snapped to the graph at execution time, so a
    # proposal counts as valid when it lands within the snap radius of a
    # reachable neighbor (exact equality only held for discrete actions)
    snap_radius = 0.75 * node_resolution
    invalid_actions = 0
    for batch_index, action in enumerate(pred[:, 0]):
        valid_indices = current_edge[batch_index][edge_padding_mask[batch_index] == 0]
        valid_actions = node_inputs[batch_index, valid_indices, :2] * coordinate_scale
        if len(valid_actions) == 0:
            invalid_actions += 1
            continue
        distances = np.linalg.norm(valid_actions - action, axis=-1)
        if float(np.min(distances)) > snap_radius:
            invalid_actions += 1

    return {
        'collision_rate': rate(invalid_actions, len(pred)),
        'success_rate': success_rate,
        'smoothness_cost': smoothness_cost(pred),
    }
