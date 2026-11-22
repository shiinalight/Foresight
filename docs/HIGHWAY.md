# Exploratory: same core, another embodiment — highway cars (plan and pass rules: commit 34f1954, 03:38 Berlin)

**Order.** The first TabPFN output, `results/highway_h1.csv`, has file time 03:39:23 Berlin, 42 s after this commit. The run's start is not logged. Its 6 hosted calls at about 4 s each (the other 64 calls took about 4 min after it, to 03:43:35) fit inside 42 s, so the plan-before-run order is consistent with the files but not provable from the repo. The rows file (`data/highway`, 03:36) comes from the sanity step, which runs no model.

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

## Results (hosted TabPFN, default model; commit 06d8d93, 03:45 Berlin; `results/highway.log`)

**Verdicts by the rules above:** **H1: core transfers to cars. H2: adapts on cars.**

### H1: same core on cars

Setup: 1,500 held-out rows × 3 context seeds = 4,500 scored rows, 355 vehicles, all regimes.

| Model | Joint 5–95 coverage | Interval score | Mean error of the median | Width |
|---|---|---|---|---|
| cv | 0% | 15.62 | 1.32 m | 0 |
| cv_band | 82.9% | 7.78 | 1.32 m | 3.64 m |
| lgbm (same 1,500 rows) | 76.5% | 5.02 | 1.10 m | 2.59 m |
| lgbm_all (100,440 rows) | 82.1% | 4.06 | 0.92 m | 2.30 m |
| **TabPFN (1,500 rows)** | **85.6%** | **3.57** | **0.75 m** | **1.93 m** |

Paired vehicle bootstrap, TabPFN minus rival, with 95% intervals:

| Rival | Interval score | Error | Coverage |
|---|---|---|---|
| lgbm_all | **−0.49 [−0.66, −0.34]** | −0.16 m [−0.20, −0.13] | +3.5 pts [+1.8, +5.3] |
| cv_band | **−4.21 [−4.80, −3.59]** | −0.57 m | +2.7 pts |
| lgbm (same rows) | −1.45 [−1.67, −1.23] | | |

- TabPFN has the lowest interval score in every regime: aggressive 4.77 vs lgbm_all 4.93, defensive 3.03 vs 3.67, normal 3.10 vs 3.72.
- Its coverage is 84–87% in every regime, slightly above the 81% target, while its boxes are the narrowest of the calibrated models.
- On five pedestrian scenes (`docs/LOSO5.md`) TabPFN only tied lgbm_all. On cars it beats the pre-set lgbm_all (67x more rows, same simulator) by the H1 rule, but against a well-fitted LightGBM the gain shrinks to a tie (robustness check below). lgbm_all uses LightGBM m50:n100 (100 trees), the `loso5.py` setting. `realdata_claimb.py` used the 300-tree default; that was not rerun here.

### H1 robustness check (post-hoc, disclosed; run by the CEO review, `docs/archive/CEO-HANDOVER.md` section 4a)

The pre-set `lgbm_all` used LightGBM's small-data setting (min 50 rows per leaf, 100 trees) on 100,440 rows. That setting was chosen before the run, but a judge may ask whether it is underfit. Same split, same 1,500 test rows, same paired vehicle bootstrap (355 vehicles); the first row reproduces the stored `lgbm_all` rows exactly.

| LightGBM on all rows | Box coverage | Interval score | Error | TabPFN − LightGBM, interval score | Coverage | Error |
|---|---|---|---|---|---|---|
| m50, 100 trees (pre-set) | 82.1% | 4.06 | 0.918 m | −0.49 [−0.66, −0.34] | +3.5 pts | −0.164 m |
| untuned default (m10, 300 trees) | 78.9% | 3.90 | 0.854 m | −0.33 [−0.49, −0.18] | +6.7 pts | −0.100 m |
| m20, 1,000 trees, 63 leaves | 71.6% | 3.74 | 0.741 m | −0.18 [−0.36, −0.01] | +14.0 pts | +0.013 m [−0.016, +0.044] |
| m50, 2,000 trees, 127 leaves | 69.1% | 3.75 | 0.705 m | −0.18 [−0.37, +0.00] | +16.5 pts | +0.049 m [+0.020, +0.078] |
| **TabPFN, 1,500 rows** | **85.6%** | **3.57** | 0.754 m | | | |

**Reading:**
- Bigger LightGBM models get sharper and more accurate at the centre, but more overconfident: coverage falls from 82% to 69% against an 81% target.
- TabPFN keeps the best interval score at every setting. Against the largest model, though, the interval touches zero.
- **So the honest wording for H1 is "TabPFN on 1,500 rows matches LightGBM trained on 67 times more, with more honest ranges", not "beats".** That is the same pattern as the five pedestrian scenes.
- Limits: four settings picked after seeing H1, none tuned on a validation split, one run each.

### H2: adaptation after a switch to the unseen `aggressive` regime

Setup: 600 query rows, 100 vehicles, 30 moments, 5 held-out episodes; no moment was thin.

| Model | Joint 5–95 coverage | Interval score | Mean error of the median | Width |
|---|---|---|---|---|
| frozen (2,000 bank rows) | 62.5% | 13.96 | 2.05 m | 2.57 m |
| **split (frozen + scene)** | **76.7%** | **12.98** | **1.91 m** | 2.93 m |
| scene only (≤ 250 rows) | 73.0% | 14.02 | 2.00 m | 3.29 m |
| lgbm_all (31,620 bank rows) | 55.3% | 16.22 | 2.11 m | 1.78 m |
| cv_band (bank context) | 58.2% | 16.80 | 2.36 m | 2.43 m |
| cv | 0% | 27.00 | 2.36 m | 0 |

Paired vehicle bootstrap, split minus rival, with 95% intervals:

| Comparison | Interval score | Coverage | Error |
|---|---|---|---|
| split − frozen (**the rule**) | **−0.98 [−1.41, −0.56]** | +14.2 pts [+11.5, +16.9] | −0.14 m [−0.18, −0.09] |
| split − scene only | −1.04 [−1.41, −0.68] | | |
| split − lgbm_all | −3.24 [−3.87, −2.59] | | |
| frozen − lgbm_all | −2.26 [−2.92, −1.65] | | |

By time since the switch (interval score, frozen → split; coverage, frozen → split):

| Since switch | Rows | Interval score | Coverage | Read |
|---|---|---|---|---|
| 0–6 s | 220 | 31.7 → 29.6 | 20% → 51% | The cars are accelerating towards a new target speed and every model fails, with errors of about 4.6 m. The scene rows help most here. |
| 6–12 s | 180 | 3.68 → 3.24 | 82% → 90% | |
| 12–20 s | 200 | 3.66 → 3.49 | 92% → 93% | |

Once traffic has settled into the new regime, the frozen model is roughly right again on its own, and split still gains a little.

**Where the H2 gain comes from:** about 80% of split's gain over frozen falls in the first 6 s after the switch. So the result means "adapts through the transition". It does not mean that settled aggressive traffic is out of distribution for the bank: from 6 s on, frozen already covers 82–92%.

### H2b (exploratory): pedestrian bank predicting cars

| Model | Joint 5–95 coverage | Interval score | Mean error of the median | Width |
|---|---|---|---|---|
| ped (2,000 crowd_cross_t2 rows) | 92.8% | 51.68 | 3.02 m | **50.6 m** |
| ped + car scene model | 92.3% | 28.16 | 2.33 m | 26.9 m |
| split (car bank + scene), for reference | 76.7% | 12.98 | 1.91 m | 2.93 m |

- **The wrong embodiment fails in a safe direction.** On car rows the pedestrian-trained TabPFN gives boxes about 50 m wide. The car features (distances, speeds about 24 m/s against people's 1–1.6 m/s) are far outside its context, so it says "I don't know" rather than giving narrow, wrong boxes. Its coverage stays high (93%), but the boxes are useless for planning.
- **Scene data halves the damage but does not rescue it.** Averaging with the car scene model cuts the interval score from 51.7 to 28.2 (−23.5 [−23.9, −23.1]) and the error by 0.69 m. That is still about twice split's interval score (+15.2 [+14.3, +16.1]), because averaging quantiles keeps half of the 50 m pedestrian width. A context that mixes the rows, rather than averaging the two models, might do better; it was not tested.
- **This suggests,** without a verdict (there is no pass rule), that the core transfers but each embodiment needs its own bank.

### API use

- **Calls:** 70, as planned.
- **Credits:** the server's usage went from 7,370,000 to 8,070,000, so **700,000 credits** (about 10,000 per call), against the 2.1M budgeted. The account may be shared with other sessions.
- **Log:** `results/api_usage.csv` has the run row plus a correction row to the measured figure.

### Caveats

- **Simulator only.** IDM/MOBIL traffic in highway-env, with regimes I set by hand. Real highway data (highD, NGSIM) would be the next check.
- **Lateral behaviour on a highway is easier than people.** The median CV error at +2 s is 0.63 m. Most of the tail comes from lane changes, about 4% of steps, and from the speed change after a switch.
- **Small H2 test set.** 5 held-out episodes and 100 vehicles. The time bins are descriptive.
- **Figure.** `media/fig_highway.png` shows episode 8, t = 26.2 s, 8.2 s after a defensive → aggressive switch.
  - **How it was chosen:** after the run, from the stored H2 quantiles, so this is not a pre-set rule. Two frames were rejected first: one where a single runaway box filled the plot, and one 0.2 s after the switch, where every box lagged the accelerating cars. The final rule picks the frame ≥ 6 s after the switch with the most lane changes, among frames with no box wider than 10 m.
  - **What it shows, checked against `highway_h2_rows.csv`:** 17 of the 20 cars are inside their split box. Boxes are tight for cars holding their lane. The one car mid lane-change gets a box spanning −5.8 to +3.0 m across, and it is inside it. The 3 misses are cars that start a lane change within the next 2 s (true across moves of 2.3, 3.9 and 4.5 m).
