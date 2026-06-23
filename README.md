# SAMS-Nav — Safety-Aware Multi-Sample Waypoint Selection for Diffusion Policy Navigation

`SAMS-Nav` is the performance-focused successor of `DARE_nav` (itself a
navigation-oriented fork of the DARE diffusion exploration project). It keeps
the same task definition — goal-conditioned navigation on a partial belief
graph with A* expert supervision — and upgrades the learning and inference
stack. No reinforcement learning is used anywhere: SAMS-Nav is pure diffusion-based
imitation learning plus test-time selection.

## Task

- input: partial belief map/graph state, robot state, and a navigation goal
- output: future two-dimensional waypoint deltas
- supervision: A*-based expert waypoint demonstrations
- termination: goal reached, collision, timeout, or invalid plan

## What SAMS-Nav changes over DARE_nav

1. **Continuous waypoint-delta diffusion** —
   `diffusion_policy/policy/diffusion_transformer_navigation_node_policy.py`
   is now a full continuous policy. DARE_nav diffused one-hot class logits
   (per-axis classes {-4, 0, +4} m) and argmax-decoded them, which collapses
   diffusion's multimodal trajectory distribution into brittle per-axis
   classification. SAMS-Nav diffuses normalized continuous 2-D deltas (Diffusion
   Policy style) and snaps proposals onto the belief graph at execution time.

2. **Classifier-free goal guidance (CFG)** — during training the
   goal-relative node features are zeroed with probability `goal_drop_prob`
   (default 0.1); at inference the model blends goal-conditioned and
   goal-free predictions with `goal_guidance_scale` (default 1.3),
   sharpening goal-directed behavior the same way NoMaD uses goal masking.

3. **Goal-fused graph encoder** —
   `diffusion_policy/model/encoder/navigation_node_encoder.py` decodes the
   belief graph from two query points: the robot's current node and the node
   closest to the goal (by the goal-offset features), then fuses both into
   the conditioning embedding. DARE_nav summarized the graph only from the
   robot's point of view.

4. **DDIM 10-step inference** — the noise scheduler is DDIM
   (`train_navigation_transformer_node.yaml`), trained with 100 diffusion
   steps and sampled with 10, ~10x faster closed-loop replanning than the
   100-step DDPM sampler it replaces.

5. **Test-time best-of-K candidate selection** — `test_worker.py` samples
   `NUM_ACTION_SAMPLES` (default 8) trajectories in one batched denoising
   pass, scores each by goal progress of the first action, goal progress of a
   3-step lookahead endpoint, and distance from the raw proposal to the
   nearest valid graph neighbor, then executes the best. This exploits the
   multimodality of the diffusion sampler instead of committing to a single
   sample.

6. **Goal-aware safety fallback** — when no sampled proposal can be snapped
   onto the graph (within `SNAP_RADIUS_FACTOR * NODE_RESOLUTION`), the
   fallback picks the valid neighbor that maximizes goal progress,
   tie-broken by alignment with the policy's intended direction. DARE_nav's
   fallback minimized heading error only and ignored the goal entirely.

7. **Consistent coordinate scaling** — node relative coordinates, per-node
   goal offsets, and the offline proxy metrics now all use the same scale
   (`UPDATING_MAP_SIZE / 2`). DARE_nav's chained division
   (`/ UPDATING_MAP_SIZE / 2`) silently scaled node coordinates by
   `2 * UPDATING_MAP_SIZE`, a 4x mismatch against both the goal features and
   `training_proxy_metrics`. The proxy validity check also uses a snap
   tolerance now instead of exact discrete equality.

Because of change 7 (and the continuous action head), **datasets collected
with DARE_nav must be regenerated with SAMS-Nav's `dataset_driver.py`** before
training.

## Navigation Observation Contract

Seven features per local graph node, all normalized by
`UPDATING_MAP_SIZE / 2` where applicable:

```text
[relative_x, relative_y, utility, guidepost, occupancy, goal_dx, goal_dy]
```

Expert supervision is a continuous two-dimensional waypoint delta. The policy
receives only the partial belief graph at inference time; the ground-truth
map is used by the simulator and the oracle reference metrics only.

## Pipeline

```text
Zarr navigation episodes (A* expert, partial-belief observations)
    -> NavigationNodeDataset (7-D goal-conditioned node features)
    -> NavigationNodeEncoder (graph attention + goal-node fusion)
    -> continuous waypoint-delta diffusion transformer
       (goal dropout in training, CFG + DDIM-10 at inference)
    -> sample K candidate trajectories per replanning step
    -> score: goal progress + lookahead progress + graph snap error
    -> snap best candidate onto the belief graph
    -> goal-aware fallback if nothing is executable
    -> step env, update belief, repeat
```

## Setup

```bash
cd SAMS-Nav
conda create -n env_dpn python=3.12.9 -y
conda activate env_dpn
pip install -e .
wandb login
```

## Usage

1. Collect demonstrations (CPU, Ray):

```bash
python dataset_driver.py   # settings in dataset_parameter.py
```

2. Train (settings in `train_navigation_transformer_node.yaml`, device
   `cuda:1`, W&B online):

```bash
python train.py --config-dir=. --config-name=train_navigation_transformer_node.yaml
```

3. Closed-loop evaluation on unseen `maps_test` (set `run_path` /
   `checkpoint_name` in `test_parameter.py` first):

```bash
python test_driver.py
```

Test metrics: success rate, timeout rate, SPL, path efficiency, final goal
distance, raw invalid action rate (before snapping/fallback), fallback rate,
fallback failed rate, and executed-action smoothness. SPL and path efficiency
use a ground-truth A* path only as an oracle reference length.

## Tuning knobs

- `goal_guidance_scale` (policy/YAML): 1.0 disables CFG; 1.2–2.0 sharpens
  goal seeking at some cost of diversity.
- `NUM_ACTION_SAMPLES` (`test_parameter.py`): candidates per replanning step;
  1 reduces to single-sample diffusion policy.
- `SNAP_RADIUS_FACTOR` (`test_parameter.py`): how far a raw proposal may be
  from a valid neighbor before it counts as invalid and triggers fallback.
- `num_inference_steps` (YAML): DDIM denoising steps; 10 is the default
  speed/quality point.

## Suggested ablations

1. Discrete one-hot backend (DARE_nav) vs. continuous deltas (SAMS-Nav).
2. CFG scale 1.0 / 1.3 / 2.0.
3. K = 1 / 4 / 8 / 16 candidates.
4. Goal-aware fallback vs. angle-only fallback vs. no safety layer.
5. Oracle A* vs. belief-map A* demonstrations
   (`EXPERT_PLANNER_MAP` in `dataset_parameter.py`).

## License and Credit

The reusable code originates from the MIT-licensed DARE project. Keep the
original `LICENSE` and cite DARE and Diffusion Policy when publishing work
derived from this scaffold.
