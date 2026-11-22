# Exploratory: same core, another embodiment — highway cars (written before any TabPFN run, Sun 22 Nov 2026)

**Status:** exploratory, offline only (no closed loop). The pedestrian results stay the main evidence.

**Why.** The project says the core works for any robot or form factor (cars, warehouse robots, drones), but every result so far is pedestrians around a delivery robot. This test adds a second embodiment, cars on a highway, and runs it through the **unchanged** core: `features.build_rows` → TabPFN quantiles → the same metrics. Nothing in `foresight/core/` changed. The only new code is the adapter `foresight/domains/highway.py` (writes INTERFACE.md files) and the test script `tools/highway_tests.py`.

## Data: `data/highway` (adapter `foresight/domains/highway.py`)

    python -m foresight.domains.highway --episodes 72 --seed 0 --out data/highway
    python tools/tune_kalman.py data/highway --noise 0.1 --split --q-grid 0.005 0.01 0.02 0.05 0.1 0.25 1 --r-grid 0.01 0.02 0.03 0.05 0.1 0.2

- **Simulator.** highway-env 1.12 `highway-v0` builds a 4-lane straight road and 20 other cars plus the ego (seeded `env.reset`). The adapter then steps the road itself: 15 Hz physics, recorded every third step, so **dt = 0.2 s** (the crowd's dt). It does not call `env.step`, so an ego crash can't end an episode early. Episodes run **40 s**, **72 episodes**.
- **Ego** (`is_ego = 1`, "the robot"). An `IDMVehicle` with highway-env's default IDM/MOBIL settings and a 25 m/s target. Its regime is `ego` and never switches.
- **Other cars.** `IDMVehicle`, highway-env's default traffic model: IDM car following plus MOBIL lane changes. Each regime is a set of IDM/MOBIL constants set on every car. Each car draws its own target speed from the regime's range, and its time gap gets about 10% jitter.

| Regime | Target speed | Time gap | Acceleration / braking | MOBIL politeness | Min. lane-change gain | Lane-change check |
|---|---|---|---|---|---|---|
| defensive | 20–24 m/s | 2.2 s | 1.5 / −3 m/s² | 0.8 | 1.0 m/s² | every 3 s |
| normal | 24–28 m/s | 1.5 s | 3 / −5 m/s² | 0 | 0.2 m/s² | every 1 s |
| aggressive | 30–34 m/s | 0.6 s | 5 / −7 m/s² | 0 | 0.05 m/s² | every 0.5 s |

- **Regime switch.**
  - Every episode switches all other cars from `regime_before` to `regime_after` at `regime_switch_t`, drawn at 35–65% of the episode on the dt grid. Each car keeps its agent_id.
  - The six ordered pairs are dealt out evenly over the 72 episodes (12 each) in a seeded order.
  - The regimes were set from track statistics only, before any model was run. Lane changes per car-minute: aggressive 4.9, normal 0.86, defensive 0.16. Ground-truth CV error at +2 s, p90: aggressive 4.9 m, normal 1.9 m, defensive 1.7 m.
- **Columns.**
  - Positions: x, y in metres in the road frame (x along the road, lanes 4 m apart).
  - Velocities: vx, vy, from speed and heading.
  - **radius 2.69 m**: half the diagonal of highway-env's 5 m × 2 m car, INTERFACE.md's rule for cars. It is the same for every car, so it carries no information. length 5 and width 2 are also written.
  - Other columns: heading, kind `car`, regime, collided, and `ext_lane` (lane index, used for analysis only; `X_of` never reads it).
- **Speed limit.** Lanes get a 40 m/s limit. highway-env's default of 30 would cap the aggressive target speed.
- **Crashes.** A car that crashes would be logged and then removed, but none crashed in 72 episodes.
- **No obstacles and no walls** (`bounds: null`, as in `eth_ucy.py`). The core's obstacle features therefore take their "none" padding.
- **Noise and filter.** Sensor noise 0.1 m (the crowd's value). The Kalman filter was tuned on the seed-0 training split, which is the split used below. It chose **q 0.05, r 0.05**, an interior optimum, with mean CV error at +2 s of 1.37 m. Provenance is in `data/highway/kalman.json`.
- **Features.** `features.build_rows` runs on car tracks without any change. Every feature is in the agent's own frame, `kind` is never read, and the ego enters through the robot features. Two features become constants on a highway: density within 3 m (≈ 0) and the obstacle features (padding). The model input is the same 39 columns as for people.

## Sanity (no models; `python tools/highway_tests.py --sanity`)

f4 rows, noise 0.1, tuned filter. Constant-velocity error is the norm of the target (ddx, ddy), measured from the *filtered* state, which is what the models see.

| | +1 s | +2 s |
|---|---|---|
| Median CV error | 0.32 m | **0.63 m** |
| Mean | 0.58 m | 1.37 m |
| p90 | 1.39 m | 3.80 m |
| p99 | 3.83 m | 9.22 m |

- **CV error at +2 s by regime** (median / mean): aggressive 0.72 / 1.99 m, defensive 0.60 / 1.08 m, normal 0.60 / 1.06 m.
- **Speed:** median filtered speed 23.9 m/s.
- **Residual sd at +2 s:** along 1.88 m, across 1.39 m.
- **Verdict: sane.** The error is neither 0 nor more than 5 m at 2 s on a road at about 24 m/s. Most driving is steady; the tail comes from lane changes and from braking or accelerating after the switch.
- **Mode labels:** 100% `straight`. The core's turn threshold is 30° and cars never turn that far, so the label carries nothing on a highway. It is never a model input, so this does not affect the tests.
- **Lane changes:** a lane change follows within 2 s for 4.1% of car-steps (aggressive 9.6%, normal 2.0%, defensive 0.9%).
- **Rows:** 272,160 in all; at h = 2 s, 100,440 training rows (54 episodes) and 33,480 held-out rows (18 episodes).
- **Plumbing check:** `--dry` ran with LightGBM in TabPFN's place, before this file was committed. No TabPFN output had been produced or seen.

## Tests (h = 2 s; episode split = `split_by_episode(seed 0)`, 54 training / 18 held-out episodes)

### H1: same core on cars

- **Context:** 1,500 random training rows, 3 seeds (0, 1, 2). Seeds change the context only, never the split.
- **Test rows:** 1,500 random held-out rows (random_state 0), all regimes.
- **Models:**
  - constant velocity (`cv`);
  - constant velocity plus band (`cv_band`, residual quantiles of the context rows);
  - LightGBM m50:n100 on the same 1,500 rows (`lgbm`);
  - LightGBM m50:n100 on every training row (`lgbm_all`, about 67x more rows);
  - TabPFN, hosted API, default model.
- **Metrics:**
  - joint coverage of the 5–95% box (target 81%);
  - **90% interval score** (Gneiting & Raftery, α = 0.1, axes averaged), the primary metric;
  - median error;
  - 5–95% width.
- **Statistics:** paired vehicle bootstrap (2,000 draws; vehicle = episode:agent_id), rows pooled over the 3 seeds.
- **Pass rule (fixed now):** **"core transfers to cars"** if TabPFN's interval score is lower than both `lgbm_all`'s and `cv_band`'s, each with a 95% interval excluding 0. Otherwise **"not supported"**.

### H2: adaptation after a sudden behaviour change on cars (claim B across embodiment)

This mirrors live_variants Setting B and realdata_claimb.

- **Unseen regime:** `aggressive`.
- **Bank:** training episodes with no `aggressive` at all, before or after the switch: 17 episodes, 31,620 rows. That leaves only defensive ↔ normal episodes, so no bank row's target window reaches into aggressive traffic.
- **Test:**
  - held-out episodes (5 of the 18: episodes 5, 8, 45, 53, 57) whose `regime_after` is `aggressive`;
  - moments every 2 s after the switch (the first row time after the switch, then +2 s, …), 30 moments drawn at random (seed 0) from 46 candidates;
  - queries = every car at those moments: 600 query rows, 100 vehicles, 0.2–20.2 s after the switch, scored against true positions.
- **frozen:** TabPFN on `context.frozen(bank, 2000)` (2,000 random bank rows), fitted once.
- **split:**
  - frozen averaged with a scene model, quantiles averaged per axis and then sorted, exactly like `run_closed.AvgQuantiles`;
  - scene model = TabPFN on that episode's latest ≤ 250 completed rows (t_known ≤ query t), labelled from observed (Kalman-filtered) positions;
  - fewer than 20 completed rows: frozen alone (not expected; every moment comes after a switch at ≥ 14 s).
- **Also scored, outside the rule:**
  - scene-only;
  - `lgbm_all` (every bank row);
  - `cv`;
  - `cv_band` (bank context).
- **Metrics:** as H1, with a paired vehicle bootstrap.
- **Pass rule (fixed now):** **"adapts on cars"** if split's interval score is lower than frozen's with the 95% vehicle-bootstrap interval excluding 0. Otherwise **"not supported"**.

### H2b (exploratory, no pass rule): the wrong embodiment, rescued by scene data?

- **ped:** TabPFN on 2,000 pedestrian rows (`context.frozen` over the `data/crowd_cross_t2` training split, f4 rows, h = 2 s) predicting the H2 car queries.
- **ped+scene:**
  - *reading of "pedestrian bank plus the car scene rows":* the pedestrian frozen model averaged with H2's own car scene model, the same split construction as H2, with the bank swapped;
  - it reuses H2's stored scene quantiles, so the extra cost is one fit (2 calls).
- **Reported:** ped+scene vs ped, ped+scene vs split, ped vs frozen. Bootstrap intervals are shown, but no verdict is drawn.

## API budget

- Each TabPFN fit plus predict costs 2 calls (one per axis), budgeted at 30,000 credits per call.
- Calls:
  - H1: 3 × 2 = 6;
  - H2 frozen: 2;
  - H2 scene models: 30 × 2 = 60;
  - H2b: 2;
  - **total 70 calls ≈ 2.1M credits**.
- The script prints this plan, refuses to run above 3M, reads server usage before and after, and appends one line to `results/api_usage.csv`.

## Outputs

- `results/highway_h1.csv`, `results/highway_h2.csv`: levels and paired bootstrap differences.
- `*_rows.csv`: per query row.
- `results/highway_h2_q.npz`: H2 quantiles.
- `results/highway.log`.
- `media/fig_highway.png`: one H2 moment, with cars, the last 2 s of each track, split's 5–95% boxes at +2 s, and where each car really was. It is drawn after the run from the stored quantiles, with no extra calls.
