# Simulator → core data interface (v1)

Every simulator writes the same three files per dataset: the crowd simulator now, highway-env later, a camera tracker as a stretch goal. Everything downstream reads only these files and never imports simulator code. That boundary is what makes the core work for any robot.

```
data/<dataset_name>/
  tracks.csv      one row per agent per time step (required)
  obstacles.csv   static obstacles per episode (optional; empty file if none)
  meta.json       dataset settings (required)
```

CSV with a header row. Parquet with the same columns is also fine; the loader takes either.

## tracks.csv

| Column | Type | Required | Meaning |
|---|---|---|---|
| `episode` | int | yes | Episode id, unique in the dataset |
| `t` | float, s | yes | Time since episode start, multiples of `dt` |
| `agent_id` | int | yes | Unique within an episode; stable for the agent's whole life |
| `kind` | str | yes | `person`, `robot`, `car`, … |
| `is_ego` | 0/1 | yes | 1 for the robot we control (exactly one per episode, may be absent from the data if no robot) |
| `x`, `y` | float, m | yes | Ground-truth position, world frame |
| `vx`, `vy` | float, m/s | yes | Ground-truth velocity |
| `radius` | float, m | yes | Footprint radius (for cars: half the diagonal, or use `length`/`width`) |
| `length`, `width` | float, m | no | Box footprint, cars |
| `heading` | float, rad | no | Body heading; if absent, taken from velocity |
| `regime` | str | yes | Behaviour regime active for this agent at this step (`calm`, `hurried`, `erratic`, …). **Ground truth for analysis only, never a model input** |
| `goal_x`, `goal_y` | float, m | no | Agent's current goal. **Never a model input** (a real robot can't see it); used to explain behaviour and to derive labels |
| `collided` | 0/1 | no | 1 at a step where this agent touches another agent or obstacle |
| `ext_*` | any | no | Domain extras the adapter may use as features (for example `ext_lane_id`, `ext_lane_offset` on the highway) |

Rules:
- **No noise in the file.** The core adds sensor noise and runs the Kalman filter, so every experiment controls the noise level the same way.
- **Agents may enter and leave.** An agent's rows simply start and stop.
- **Rows sorted by** `episode, t, agent_id`.

## obstacles.csv

`episode, obstacle_id, shape, x, y, r, w, h, angle`. `shape` ∈ {`circle`, `rect`}; `r` for circles, `w`, `h`, `angle` for rectangles. World walls go in `meta.json` as bounds.

## meta.json

```json
{
  "sim": "crowd-v1",
  "dt": 0.1,
  "bounds": [-10, -10, 10, 10],
  "episodes": {
    "0": {"seed": 0, "duration": 60.0, "regime_switch_t": 30.0,
          "regime_before": "calm", "regime_after": "hurried"},
    "1": {"seed": 1, "duration": 60.0, "regime_switch_t": null,
          "regime_before": "normal", "regime_after": null}
  },
  "notes": "free text: robot policy used, parameters, version"
}
```

## What the core derives from this

These don't need to be in the files.

| Derived | How | Used for |
|---|---|---|
| Noisy observations + Kalman-smoothed state | Noise added per run, forward-only Kalman filter | Realistic inputs |
| **Targets:** `ddx`, `ddy` at h = 1 s, 2 s | Agent-frame displacement minus constant velocity | **Core prediction (R1)** |
| **Mode label** | From the next 2 s: `stop` (speed < 0.2 m/s), `turn_left` / `turn_right` (heading change > 30°), `straight` | Behaviour classifier (stretch) |
| **Interaction features** | 3 nearest agents, robot distance and closing speed, nearest obstacle, density | Model inputs |
| **Minimum gap to robot, next 2 s** | From ground truth | Learned risk model (stretch); near-miss metric |
| Time to collision, collisions | Geometry + `collided` | Planner evaluation (R3) |
| Regime and time since switch | `regime`, `meta.json` | Adaptation experiment (R2), analysis only |

## What the simulator should produce

These are the properties the experiments need. The R1 gate fails without them.

1. **dt = 0.1 s,** episodes of 40 to 60 s, 10 to 30 people at a time, 300+ episodes (more is better; they're cheap).
2. **Behaviour constant velocity can't predict:**
   - goal-seeking with turns at the goal;
   - stops and waits;
   - avoidance of each other, of the robot and of obstacles (social-force or similar).
3. **Regimes as crowd-wide parameter sets** (speed, comfort distance, how often people stop or turn), with per-person jitter. In about half the episodes, one switch for the whole crowd at `regime_switch_t`.
4. **A robot in every episode** (`is_ego = 1`), moving towards its own goal with a simple policy: straight line, slowing when someone is close. People must react to it, or the robot features carry nothing. In R3 the core replaces this policy with the planner, so the simulator also needs a step function, see below.
5. **Reproducible:** the same seed gives the same episode.

## Closed-loop hook (needed for R3, not for R1)

So the planner can drive the robot, the simulator should also expose:

```python
sim = CrowdSim(seed, config)          # same parameters that wrote the files
state = sim.reset()                    # list of dicts with the tracks.csv columns for this step
state, info = sim.step(robot_cmd)      # robot_cmd = (vx, vy) for the next dt; info["collided"]: bool
```

`state` uses exactly the `tracks.csv` columns, so the core reads live steps and logged files with the same code.

## Checks the loader runs (`foresight/core/schema.py: validate()`)

- Required columns are present and numeric where they should be.
- `t` is on the `dt` grid.
- No duplicate (`episode`, `t`, `agent_id`).
- At most one ego per episode.
- Speeds below a sanity bound (5 m/s for people, 60 m/s for cars).
