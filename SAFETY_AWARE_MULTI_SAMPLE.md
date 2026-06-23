# Safety-Aware Multi-Sample Waypoint Selection

This prototype keeps the trained diffusion model, dataset, expert planner, and
checkpoint format unchanged. At each closed-loop step it samples K trajectories
in one vectorized denoising batch, validates their first waypoint against the
observed belief graph and existing collision checker, ranks survivors, executes
one snapped waypoint, updates the belief graph, and replans.

## Experiment modes

- A: one sample, no ranking (original baseline).
- B: K samples, uniformly random selection among valid survivors.
- C: K samples, maximum goal-progress selection among valid survivors.
- D: K samples, full weighted safety-aware ranking.

All overrides are environment variables, so the checkpoint configuration is
not mutated. Example 1000-episode runs:

```bash
DARE_EXPERIMENT_MODE=A DARE_NUM_TEST=1000 DARE_SAVE_GIFS=0 python test_driver.py
DARE_EXPERIMENT_MODE=B DARE_NUM_SAMPLES=8 DARE_NUM_TEST=1000 DARE_SAVE_GIFS=0 python test_driver.py
DARE_EXPERIMENT_MODE=C DARE_NUM_SAMPLES=8 DARE_NUM_TEST=1000 DARE_SAVE_GIFS=0 python test_driver.py
DARE_EXPERIMENT_MODE=D DARE_NUM_SAMPLES=8 DARE_NUM_TEST=1000 DARE_SAVE_GIFS=0 python test_driver.py
```

Configuration variables are `DARE_NUM_SAMPLES`, `DARE_ENABLE_RANKING`,
`DARE_W_GOAL`, `DARE_W_CONNECTIVITY`, `DARE_W_UTILITY`, `DARE_W_GUIDEPOST`,
`DARE_W_SAFETY`, `DARE_W_SMOOTHNESS`, and `DARE_W_COLLISION`. The snap threshold
can be changed with `DARE_SNAP_RADIUS_FACTOR`.

Each results CSV includes the standard navigation metrics plus invalid action
rate, candidate rejection rate, average candidate and selected scores, total
sampling/validation/ranking time, aggregate score components, and a JSON step
log containing generated/rejected counts, the selected score and breakdown,
and fallback usage.
