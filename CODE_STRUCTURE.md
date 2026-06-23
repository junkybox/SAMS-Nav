# SAMS-Nav Code Structure

## Repository Tree

```text
SAMS-Nav/
|-- README.md
|-- CODE_STRUCTURE.md
|-- LICENSE
|-- pyproject.toml
|-- parameter.py
|
|-- dataset_parameter.py
|-- dataset_driver.py
|-- dataset_worker.py
|
|-- train.py
|-- train_navigation_transformer_node.yaml
|-- train_exploration_transformer_node_discrete.yaml
|-- train_exploration_transformer_map_discrete.yaml
|
|-- test_parameter.py
|-- test_driver.py
|-- test_worker.py
|
|-- classes/
|   |-- navigation_metrics.py
|   |-- utils.py
|   |
|   |-- agent/
|   |   |-- agent.py
|   |   |-- node_manager.py
|   |   `-- quads.py
|   |
|   |-- env/
|   |   |-- env.py
|   |   |-- navigation_env.py
|   |   `-- sensor.py
|   |
|   `-- planner/
|       |-- navigation_planner.py
|       |-- expert_planner.py
|       |-- ground_truth_planner.py
|       `-- ortools_solver.py
|
|-- diffusion_policy/
|   |-- common/
|   |   |-- replay_buffer.py
|   |   |-- sampler.py
|   |   |-- normalize_util.py
|   |   |-- pytorch_util.py
|   |   |-- checkpoint_util.py
|   |   |-- json_logger.py
|   |   `-- other shared utilities
|   |
|   |-- config/
|   |   |-- train_navigation_transformer_node.yaml
|   |   |-- other inherited training configs
|   |   `-- task/
|   |       |-- navigation_node.yaml
|   |       |-- exploration_node.yaml
|   |       `-- exploration_map.yaml
|   |
|   |-- dataset/
|   |   |-- base_dataset.py
|   |   |-- navigation_node_dataset.py
|   |   |-- exploration_node_dataset.py
|   |   `-- exploration_map_dataset.py
|   |
|   |-- model/
|   |   |-- common/
|   |   |   |-- normalizer.py
|   |   |   |-- lr_scheduler.py
|   |   |   `-- tensor/module helpers
|   |   |
|   |   |-- diffusion/
|   |   |   |-- transformer_for_diffusion.py
|   |   |   |-- conditional_unet1d.py
|   |   |   |-- ema_model.py
|   |   |   |-- mask_generator.py
|   |   |   `-- positional/Conv1D components
|   |   |
|   |   `-- encoder/
|   |       |-- navigation_node_encoder.py
|   |       |-- exploration_node_encoder.py
|   |       `-- exploration_map_encoder.py
|   |
|   |-- policy/
|   |   |-- diffusion_transformer_navigation_node_policy.py
|   |   |-- diffusion_transformer_node_discrete_policy.py
|   |   |-- diffusion_transformer_node_policy.py
|   |   |-- diffusion_transformer_map_discrete_policy.py
|   |   |-- diffusion_unet_node_discrete_policy.py
|   |   |-- diffusion_unet_node_policy.py
|   |   |-- base_image_policy.py
|   |   `-- base_lowdim_policy.py
|   |
|   |-- workspace/
|   |   |-- base_workspace.py
|   |   |-- train_diffusion_transformer_navigation_node_workspace.py
|   |   |-- train_diffusion_transformer_node_workspace.py
|   |   |-- train_diffusion_transformer_map_workspace.py
|   |   `-- train_diffusion_unet_node_workspace.py
|   |
|   `-- shared_memory/
|       |-- shared_memory_queue.py
|       |-- shared_memory_ring_buffer.py
|       |-- shared_memory_util.py
|       `-- shared_ndarray.py
|
|-- assets/
|   |-- dare_nav_flowchart.md
|   |-- dare_nav_flowchart.dot
|   |-- dare_nav_flowchart.svg
|   |-- dare_nav_flowchart.png
|   |-- dare_main.gif
|   `-- workflow.png
|
|-- maps_train/
|-- maps_test/
|-- dataset/
`-- runs/
```

## Top-Level Entry Points

### Dataset Collection

- `dataset_parameter.py`: dataset method, episode count, CPU device, GIF and
  output settings.
- `dataset_driver.py`: starts Ray workers and writes successful episodes to a
  Zarr replay buffer.
- `dataset_worker.py`: runs one expert episode and is intended to record local
  graph observations, expert waypoint deltas, metrics, frames, and a GIF.

The worker uses `NavigationEnv`, the A* expert planner, partial-belief graph
observations, seven-dimensional goal-conditioned node features, and discrete
expert waypoint deltas. `EXPERT_PLANNER_MAP` in `dataset_parameter.py` selects
whether demonstrations come from a ground-truth oracle map or the current
partial belief map for ablations.

### Training

- `train.py`: Hydra entry point.
- `train_navigation_transformer_node.yaml`: model, optimizer, data loader,
  checkpoint, W&B, GIF, and `cuda:1` settings.
- `diffusion_policy/workspace/`: constructs the policy and runs optimization,
  validation, EMA, checkpointing, W&B logging, and training GIF generation.

### Testing

- `test_parameter.py`: checkpoint, episode count, `cuda:1`, execution horizon,
  and GIF settings.
- `test_driver.py`: loads the checkpoint, starts Ray test workers, writes CSV
  results, and prints collision rate, success rate, and smoothness cost.
- `test_worker.py`: performs closed-loop waypoint-delta prediction, graph and
  collision safety validation, execution, belief updates, metrics, and GIF
  rendering. Episodes finish when `NavigationEnv.is_goal_reached()` succeeds
  or the maximum step count is reached. Test metrics separate raw invalid
  model actions from safety-layer fallback corrections and report SPL/path
  efficiency against a ground-truth A* oracle reference length.

## Navigation Components

### Environment

`classes/env/navigation_env.py`

- Extends the original map and sensor simulator.
- Samples or accepts a navigation goal.
- Computes goal vectors and distances.
- Defines goal-tolerance completion.

The simulator contains the ground-truth map. The policy should receive only
the robot's partial belief/graph and goal-relative features.

### Expert Planner

`classes/planner/navigation_planner.py`

- Converts world coordinates to grid cells.
- Uses eight-connected A* as an expert demonstration generator.
- Generates a collision-free expert waypoint path.
- Prevents invalid diagonal traversal.
- Converts the expert grid path back into world-coordinate waypoints.

A* is not the final learned policy. It supplies expert waypoint deltas for
diffusion-policy imitation learning.

### Observation

The reserved seven-dimensional local navigation node feature vector is:

```text
[relative_x, relative_y, utility, guidepost, occupancy, goal_dx, goal_dy]
```

`Agent.get_navigation_observation()` appends `goal_dx` and `goal_dy`, relative
to the episode goal, to the original five graph features. The learned policy
receives local or partial graph observations, not a full global map.

### Metrics

`classes/navigation_metrics.py`

- Collision rate: rejected unsafe waypoint proposals divided by attempted
  waypoint proposals.
- Success rate: completed episodes divided by total episodes.
- Timeout rate: episodes that did not reach the goal before `MAX_EPISODE_STEP`.
- Path efficiency: success-gated oracle shortest path length divided by
  executed path length.
- SPL: success weighted by path length, using the oracle A* path only as a
  reference length.
- Raw invalid action rate: model proposals that were not valid current graph
  neighbors before fallback.
- Fallback rate: graph-safety corrections applied to invalid model proposals.
- Smoothness cost: mean squared change between consecutive waypoint-delta
  vectors.
- Training uses offline proxy metrics because it has no environment rollout.
- Testing reports closed-loop metrics.

## Diffusion Policy Pipeline

```text
Zarr navigation episodes
    -> NavigationNodeDataset
    -> sequence sampler
    -> NavigationNodeEncoder
    -> graph attention feature
    -> diffusion waypoint policy
    -> sample future waypoint deltas conditioned on recent observations
    -> validate graph connectivity and collision safety
    -> execute first valid waypoint or resample/fallback
    -> update belief
    -> predict again
```

The diffusion waypoint policy is a goal-conditioned waypoint proposal module,
not a low-level final controller.

Navigation-specific classes currently reuse stable DARE implementations:

- `NavigationNodeDataset` extends `ExplorationNodeDataset`.
- `NavigationNodeEncoder` extends `ExplorationNodeEncoder`.
- `DiffusionTransformerNavigationNodePolicy` extends the discrete node policy.
- `TrainDiffusionTransformerNavigationNodeWorkspace` extends the node training
  workspace.

## Generated Data

```text
dataset/navigation_astar_node_train_10000/
|-- data.zarr/
|-- gifs/
`-- training-results CSV

runs/<date>/<run_name>/
|-- checkpoints/
|-- training_gifs/
|-- logs.json.txt
|-- inference/
|   `-- gifs/
`-- test-results CSV
```

Generated datasets, checkpoints, GIFs, and run output are excluded from source
control by `.gitignore`.

## Configuration Relationship

The root navigation training YAML and
`diffusion_policy/config/train_navigation_transformer_node.yaml` are identical.
The root copy supports:

```bash
python train.py --config-dir=. \
  --config-name=train_navigation_transformer_node.yaml
```

The native Hydra copy supports:

```bash
python train.py --config-name=train_navigation_transformer_node
```

Both select `diffusion_policy/config/task/navigation_node.yaml`.
