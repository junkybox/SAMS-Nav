import numpy as np

from classes.planner.navigation_waypoint_ranker import NavigationWaypointRanker


def _graph_fixture():
    coords = np.array([[0, 0], [4, 0], [0, 4], [-4, 0]], dtype=np.float32)
    features = np.zeros((4, 7), dtype=np.float32)
    features[1, 2] = 1.0
    features[1, 3] = 1.0
    adjacency = np.ones((4, 4), dtype=np.int64)
    for target in range(4):
        adjacency[target, target] = 0
    adjacency[0, 1:] = 0
    adjacency[1:, 0] = 0
    return coords, features, adjacency


def test_goal_mode_selects_progressing_valid_candidate():
    coords, features, adjacency = _graph_fixture()
    ranker = NavigationWaypointRanker(4.0)
    evaluation = ranker.evaluate(
        candidate_deltas=np.array([[4, 0], [0, 4], [20, 20]], dtype=np.float32),
        current_position=np.zeros(2),
        goal=np.array([12, 0], dtype=np.float32),
        node_coords=coords,
        node_features=features,
        edge_mask=adjacency,
        current_index=0,
        neighbor_indices=np.arange(4),
        snap_radius=3.0,
        collision_free=np.ones(3, dtype=bool),
    )

    selected, fallback = ranker.select(evaluation, mode='C')
    assert selected == 0
    assert not fallback
    assert evaluation['valid_mask'].tolist() == [True, True, False]


def test_safety_mode_rejects_collision_and_uses_survivor():
    coords, features, adjacency = _graph_fixture()
    ranker = NavigationWaypointRanker(4.0)
    evaluation = ranker.evaluate(
        candidate_deltas=np.array([[4, 0], [0, 4]], dtype=np.float32),
        current_position=np.zeros(2),
        goal=np.array([12, 0], dtype=np.float32),
        node_coords=coords,
        node_features=features,
        edge_mask=adjacency,
        current_index=0,
        neighbor_indices=np.arange(4),
        snap_radius=3.0,
        collision_free=np.array([False, True]),
    )

    selected, fallback = ranker.select(evaluation, mode='D')
    assert selected == 1
    assert not fallback
    assert evaluation['collision_risk'][0] > evaluation['collision_risk'][1]


def test_all_rejected_signals_fallback():
    coords, features, adjacency = _graph_fixture()
    ranker = NavigationWaypointRanker(4.0)
    evaluation = ranker.evaluate(
        candidate_deltas=np.array([[20, 20]], dtype=np.float32),
        current_position=np.zeros(2),
        goal=np.array([12, 0], dtype=np.float32),
        node_coords=coords,
        node_features=features,
        edge_mask=adjacency,
        current_index=0,
        neighbor_indices=np.arange(4),
        snap_radius=3.0,
    )

    selected, fallback = ranker.select(evaluation, mode='D')
    assert selected == 0
    assert fallback
