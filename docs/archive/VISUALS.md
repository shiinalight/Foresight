# Visuals: one trace, every viewer

Written Sat 21 Nov, about 22:00. **Principle:** each run on a showcase seed writes one **trace**, a set of tables covering every stage. Every figure, the bird's-eye video, the "pipeline at work" frame and the 3D viewer only *read* that trace. Viewers never recompute anything. This is the same idea as `INTERFACE.md`, applied to the outputs: a renderer can't drift from what the robot actually did.

## 1. The trace: one folder per (scene, planner, seed)

`results/trace/<sim>_<planner>_<seed>/`. Tables are parquet, keyed by `t`, `agent_id` and `h`. Parquet reads the same on Windows, the Mac and a web converter; pickled DataFrames don't.

| File | Stage | One row = | Columns (minimum) |
|---|---|---|---|
| `meta.json` | — | — | sim, planner, seed, regime, `switch_t`, dt, bounds, obstacles, horizons, `box_q`, margin, model config (version, estimators, context size, refit), commit hash, `run_id` |
| `tracks.parquet` | S0 data | agent × step | `t`, `agent_id`, `kind`, `is_ego`, true x, y, vx, vy, radius, regime |
| `observed.parquet` | S1 state estimate | agent × step | `t`, `agent_id`, noisy x, y; filtered x, y, vx, vy |
| `rows.parquet` | S2 table | agent × replan × horizon | `t`, `agent_id`, `h`, every model feature (exactly what the model saw) |
| `context.parquet` | S3 context | context row × refit | `t_refit`, row id, `source` (recent / knn / fill / bank-random), the source episode, `t_known` |
| `pred.parquet` | S4 + S5 prediction | agent × replan × horizon | `t`, `agent_id`, `h`, quantiles (5/10/50/90/95 per axis), conformal offset, box in world coordinates (centre, `theta`, `lo`/`hi`) |
| `occupancy.npz` | S6 occupancy | replan × horizon | the grid (probability per cell), grid origin and resolution. Computed by **the same function** the occupancy metric uses |
| `plan.parquet` | S7 planner | candidate × replan | `t`, candidate velocity, worst gap, `veto_reason` (box / short check / bounds / obstacle / none), progress, `chosen` flag, fallback flag |
| `timing.parquet` | S8 system | replan | `t`, ms for features, context, fit (if refit), predict, plan; `refit` flag |
| `outcome.parquet` | evaluation only | agent × replan × horizon | true position at `t + h`, inside-box flag, error. **Never read by the robot**; kept in its own file so nobody mistakes it for an input |

**Rules:**
- Extend the existing `--log` path in `run_closed.py`; don't add a second logger.
- Capture happens **outside the timed region**, so logging can't change behaviour or latency numbers. Check that a traced seed reproduces its overnight row exactly.
- **Each planner's own belief only.** The reactive and velocity-obstacle traces have no predicted boxes (their `pred` is people where they are now, or straight-line motion). TabPFN boxes are never drawn on another planner's panel.
- `context.select()` must return a `source` column. Today it concatenates and drops duplicates, so the origin is lost.
- `planner.choose()` must return every candidate with its veto reason, not only aggregates.
- **Trace showcase seeds only, never the 100-seed run.**

## 2. What gets drawn (all from the trace)

| Output | Content | Priority |
|---|---|---|
| **Four-panel bird's-eye video** | Same seed, regime switch at the midpoint: stop and go, reactive, velocity obstacles, TabPFN live. In each panel: people, trails, the planner's own belief (boxes where it has them), the robot's path, and live counters (collisions, minimum gap, time to goal, box coverage so far) | Must |
| **"Pipeline at work" frame** (one moment, six panels) | (1) raw tracks and noisy observations; (2) the table rows for one person; (3) the context TabPFN read, coloured by source; (4) predicted boxes and quantiles against the true future position; (5) occupancy heatmap; (6) planner candidates, coloured by veto reason, with the chosen move | Must |
| **Occupancy overlay** under the TabPFN panel, plus its metric (Brier score and ROC AUC against true future occupancy, against band and LightGBM) | | Should |
| **Module results index** in `RESULTS.md`: one table mapping each stage (S0–S8) to its gate metric, its number and its figure file | No new dashboard | Should |
| **3D viewer** | Reads a JSON export written by a converter from the trace (`tools/export_3d.py`), never the trace code directly, so the two can't diverge | Stretch: only if a teammate builds it in parallel. The bird's-eye video is the fallback |

**Showcase seed, chosen by a fixed rule.** Among regime-switch seeds where every planner reached the goal, take the one where live TabPFN's time to goal is closest to its median. Caption it "typical seed (median), not cherry-picked". If a second, harder seed is shown, give its rule too.

## 3. Stale figures

`media/` already mixes pilot, pre-fix and current images. Add `media/MANIFEST.csv` with figure, script, input file, commit and a status column (current / superseded). The deck and README use current figures only. Delete or move superseded ones before submission.

## 4. Order

Tracing touches `run_closed.py` and `planner.py`. Edit them only after the last overnight baseline process (vo) has started.

1. Trace writer and the `source` / veto-reason returns (1 h).
2. Trace the showcase seed for the four planners (minutes once the overnight results pick the seed).
3. Four-panel video and the "pipeline at work" frame (1.5 h).
4. Occupancy function, metric and overlay (1.5 h).
5. Results index and manifest (30 min).
6. 3D export converter (30 min); the viewer itself only if a teammate owns it.
