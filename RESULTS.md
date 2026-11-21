# Results

Live document: numbers update as experiments finish. The figures in `media/` carry only labels and legends; this file holds the explanations.

## Setup (R1 baselines on real pedestrians)

**Data.** ETH Walking Pedestrians (Pellegrini et al. 2009), two Zurich scenes taken from OpenTraj:
- **ETH:** 360 people.
- **Hotel:** 390 people.

Positions are annotated in metres every 0.4 s. Each recording is cut into 60 s windows.

**Protocol.** Context and training rows come from one scene; testing is on the other (ETH → Hotel and Hotel → ETH). The test place is never seen, which is stricter than a random split.

**Horizons.** 1.2 s, 2.0 s and 4.8 s. 4.8 s is the standard horizon in published work on this data.

**Rows.**
- **Definition:** one row = one person, at one moment, for one horizon.
- **Features:**
  - the person's own speed, acceleration and turn rate;
  - the last 2 s of their path;
  - their 3 nearest neighbours (position and velocity);
  - local density.

  All are in the person's own frame of reference.
- **Target:** where they actually are after h seconds, minus where constant velocity would put them.
- **Leakage:** no feature uses the future.

**Test set.** 900 random rows per direction across the three horizons (about 300 per horizon). TabPFN is slow on the laptop CPU, so the test set is capped. A GPU run with all rows is planned.

## The predictors

| Model | What it is | Example rows it uses |
|---|---|---|
| Constant velocity | Keeps the last (Kalman-filtered) velocity. Gives a point, no range | — |
| Constant velocity + band | Constant velocity plus a fixed range: the 5–95% (etc.) quantiles of training residuals, per horizon. The same range for everyone | All training rows |
| LightGBM (1,500 rows) | Gradient-boosted trees, quantile regression | Same 1,500 random rows TabPFN sees |
| LightGBM (all rows) | Same, trained on every training row: its best case | All (14,223 from ETH; 7,640 from Hotel) |
| **TabPFN (1,500 rows)** | Tabular foundation model (`tabpfn` 9.2, TabPFN-3.5, default settings). Reads the rows as context and returns quantiles. No training | 1,500 random rows |

**How LightGBM is trained** (`foresight/core/models.py: LGBMQuantile`):
- **Models:** one model per quantile per axis (11 quantiles × 2 axes = 22 models), objective `quantile` with `alpha` = the quantile.
- **Settings:** 300 trees, learning rate 0.05, 31 leaves, at least 10 rows per leaf; LightGBM defaults otherwise. **No tuning**, so it is a reasonable baseline, not an optimised one.
- **Inputs:** the same features as TabPFN, as raw numbers.
- **Crossing:** predicted quantiles are sorted per row, so they never cross.

**Why two LightGBM versions:**
- the 1,500-row version is a like-for-like comparison with TabPFN;
- the all-rows version answers "what if the classical model just gets all the data".

## How to read the figures

- **Demo (`media/demo_hotel_tabpfn.gif`):**
  - **Blue dot:** a person now; the grey line is their last 2 s.
  - **Orange dot:** the predicted position 2 s ahead (median).
  - **Orange box:** the 80% range. It spans the 5–95% quantiles along and across the person's direction of travel, so it holds the truth about 80% of the time when the model is honest.
  - **Black x:** where the person actually was 2 s later (shown for checking, never given to the model).
  - **People without a box:** they just entered, or their position 2 s later is unknown.
  - **Live numbers:** how often the x landed in its box so far, and the median error.
- **Four predictors (`media/compare_hotel.png`):** the same moment and the same people, one panel per model, so the boxes can be compared directly. The panel titles give coverage and mean error over the whole episode.
- **Error by horizon (`media/fig_error_by_horizon.png`):** mean distance between predicted and actual position. Lower is better.
- **Calibration (`media/fig_calibration.png`):** for ranges that promise 20% to 90% coverage, how often the truth actually fell inside.
  - **On the dotted line:** honest.
  - **Below it:** overconfident, dangerous for a robot.
  - **Above it:** too cautious, makes the robot slow.
- **Data efficiency (`media/fig_data_efficiency.png`):** error at +2 s against the number of example rows (log scale), ETH → Hotel.

## Numbers (rerun 21 Nov evening, after the leakage fix; 900 test rows per direction, 1,500 context rows)

| Direction | Model | Rows seen | Error +1.2 s | Error +2 s | Error +4.8 s | Per-axis coverage of 10–90% (+2 s) | Joint coverage of the 5–95% box (+2 s, target ~80% if the axes are independent) | 10–90% width (+2 s) | Pinball (+2 s) |
|---|---|---|---|---|---|---|---|---|---|
| ETH → Hotel | Const. velocity | — | 0.219 m | 0.349 m | 0.858 m | — | — | — | 0.1085 |
| ETH → Hotel | Const. velocity + band | 14,223 | 0.219 m | 0.349 m | 0.858 m | 88% | 91% | 1.03 m | 0.0872 |
| ETH → Hotel | LightGBM (1,500 rows) | 1,500 | 0.206 m | 0.284 m | 0.665 m | 77% | 76% | 0.57 m | 0.0659 |
| ETH → Hotel | LightGBM (all rows) | 14,223 | 0.188 m | 0.289 m | 0.662 m | 79% | 80% | 0.56 m | **0.0636** |
| ETH → Hotel | **TabPFN** | 1,500 | **0.168 m** | **0.274 m** | 0.673 m | 89% | 92% | 0.91 m | 0.0661 |
| Hotel → ETH | Const. velocity | — | 0.336 m | 0.534 m | 1.442 m | — | — | — | 0.1696 |
| Hotel → ETH | Const. velocity + band | 7,640 | 0.336 m | 0.534 m | 1.442 m | 69% | 70% | 0.77 m | 0.1253 |
| Hotel → ETH | LightGBM (1,500 rows) | 1,500 | 0.285 m | 0.451 m | 1.240 m | 55% | 46% | 0.54 m | 0.1073 |
| Hotel → ETH | LightGBM (all rows) | 7,640 | 0.274 m | **0.428 m** | 1.216 m | 65% | 62% | 0.60 m | 0.1028 |
| Hotel → ETH | **TabPFN** | 1,500 | **0.264 m** | 0.431 m | **1.189 m** | 81% | **83%** | 0.96 m | **0.1007** |

- **Error:** mean distance between the predicted (median) and actual position.
- **Joint coverage:** how often the actual position falls inside the 2D box the robot uses, both axes at once.
- **Pinball:** the average quantile loss over 11 levels. It is a proper score for the whole predicted distribution; lower is better.
- **Compute:** the laptop CPU (x64 under emulation). Fit plus predict takes about 70 s for TabPFN and under 10 s for the others.

**What this says:**
- **Calibration is the finding that holds.** TabPFN's box holds the truth 92% and 83% of the time where about 80% is promised. LightGBM's holds it 76–80% going to Hotel but only 46–62% going to ETH: overconfident in the new place, the dangerous kind of error for a robot. The fixed band is also overconfident going to ETH (70%).
- **Error:** TabPFN is **level with LightGBM on all rows**, not better. Hotel → ETH at +2 s: 0.431 against 0.428 m. ETH → Hotel: 0.274 against 0.284 and 0.289 m. The gaps are within what a single seed can show, so we claim comparable error, not lower.
- **Pinball** (the whole distribution): TabPFN is best going to ETH (0.1007 against 0.1028). Going to Hotel, LightGBM on all rows is slightly better (0.0636 against 0.0661), because TabPFN's ranges are wider there than needed.
- **Width:** TabPFN's ranges are wider (0.91–0.96 m against 0.54–0.60 m). Going to ETH, that width is what keeps them honest. Going to Hotel, they are more cautious than needed (92%). Recalibrating on the scene's own rows is the next step (`docs/GATES.md`, S5).
- **Leakage fix.** Real data no longer gets a "distance to the edge of the area" feature: that edge was the camera's field of view, computed from all recorded positions, including future ones.
  - Removing it changed errors by up to 0.03 m in either direction: lower going to Hotel, higher going to ETH. A feature that helps one way and hurts the other is the signature of scene-specific leakage.
  - TabPFN's coverage going to ETH stayed at 81% with or without it. LightGBM's moved from 64% to 55%.
  - All numbers here are without the feature.
- **Rows from the same place are worth more than rows from another place** (`media/fig_rows_place.png`; LightGBM, 5 seeds, +2 s):
  - **Rows from another place** (ETH, testing on Hotel): error levels off at about 0.30 m from 1,000 rows on.
  - **Rows from the same place** (Hotel, other episodes): error keeps falling, to 0.27 m at 2,000.
  - **Correction:** an earlier version of this figure showed error *rising* with more rows from another place. That came from the leaking feature above and is gone.
  - **Caveat:** the two test sets differ, so compare the shapes. This is the reason for choosing context rows from the current scene (adaptation experiment, next).
- **Hotel is easy for constant velocity:** about 45% of people there stand still. ETH is harder, and every learned model gains more there.

## Calibration layer: on arrival and in the scene (`tools/conformal_eval.py`, `media/fig_conformal_*.png`)

Every model's 10–90% box (per axis) is widened or narrowed by one offset, so that the 2D box holds the truth 80% of the time, jointly over both axes. This uses conformalised quantile regression with a joint score (`foresight/core/conformal.py`).

- **Raw:** the model's own box.
- **On arrival:** the offset is calibrated on a held-out 30% of the *training* scene's episodes, which is all a robot has when it arrives somewhere new.
- **In the scene:** starting from that offset, it adapts online from the new scene's own completed rows (adaptive conformal, Gibbs and Candès 2021; strictly causal).
- **Engine:** TabPFN here runs through the hosted API (`tabpfn-client`, default model).

**Joint coverage of the box at +2 s (target 80%), and width after 20 s in the scene:**

| Direction | Model | Raw | On arrival | In the scene, after 20 s | Width after 20 s |
|---|---|---|---|---|---|
| Hotel → ETH | Const. velocity + band | 50% | 68% | 79% | 1.40 m |
| Hotel → ETH | LightGBM (1,500 rows) | 32% | 67% | 79% | 1.16 m |
| Hotel → ETH | LightGBM (all rows) | 37% | 67% | 79% | 1.15 m |
| Hotel → ETH | **TabPFN** | **62%** | **73%** | 80% | 1.43 m |
| ETH → Hotel | Const. velocity + band | 75% | 92% | 83% | 1.19 m |
| ETH → Hotel | LightGBM (1,500 rows) | 60% | 93% | 82% | 0.90 m |
| ETH → Hotel | LightGBM (all rows) | 56% | 91% | 82% | 0.87 m |
| ETH → Hotel | **TabPFN** | **80%** | 95% | 81% | 1.16 m |

**What this says, plainly:**
- **Out of the box, TabPFN is the only calibrated model.** With no calibration data at all, its box holds the truth 62% and 80% of the time; LightGBM's holds it 32–60%.
- **On arrival, calibrating on the training scene does not fix a new place.** Going to ETH, every model stays overconfident (67–73%), and TabPFN is closest. Going to Hotel, every model is pushed too cautious (91–95%).
- **After about 20 s of outcomes in the new scene, every model reaches about 80%,** and a conformally calibrated LightGBM does it with **narrower boxes** (0.87–1.16 m against TabPFN's 1.16–1.43 m). TabPFN does not win on sharpness once local calibration data exists, and we say so.
- **So TabPFN's edge is the first seconds in a new place,** before there is anything to calibrate on, plus learning from the scene's own rows (adaptation experiment). For a robot, the first seconds in a new place are exactly when an overconfident box causes a collision.

**Limits:**
- Only two real scenes.
- The test sets are capped (CPU time).
- Single seed for this table; a 5-seed table with intervals that resample whole people is next.
- LightGBM is untuned in the main table; tuned and conformal versions are next.
- Adaptation comparisons against "best LightGBM" pick the best of five LightGBM settings on the test scene itself. That favours LightGBM, the conservative direction for any TabPFN claim.
- No obstacles from the scene maps.
- No robot in the real data.

## Closed loop: pre-registered analysis (written Sat 21 Nov, before the overnight run starts)

This section fixes how the overnight closed-loop results will be read, before any of those numbers exist. That way the operating point can't be chosen after seeing the results.

**Run:**
- **Crowds:** Nooshin's crowd simulator, seeds 1000–1099. Odd seeds switch the crowd's behaviour regime at mid-episode (`--switch`).
- **Same conditions everywhere:** one robot top speed (1.0 m/s) for every planner, and the same seeds for every planner.
- **Our planners:** each predictor is swept over its caution dial: the box quantile `box_q` 0.05 / 0.10 / 0.15 at margin 0.2 m. The predictors are the band, LightGBM (1,500 rows, all rows, live) and TabPFN (frozen, live).
- **Baselines:** each swept over its own dial:
  - constant velocity and the reactive planner (DWA-style local planner, our implementation) at margin 0.2 / 0.4 / 0.6 m;
  - velocity obstacles (sampling form, not ORCA) at collision weight 0.5 / 1 / 2;
  - stop-and-go once (it has no setting).

**Main figure:** full curves, one per planner (one point per dial setting):
- **Axes:** mean time to goal, with failures scored at the 40 s limit (`time_capped`), against mean closest gap and against mean seconds with anyone within 0.5 m (`intrusion_s`).
- **Split:** switch and no-switch seeds shown separately.
- **Intervals:** 95% paired-seed bootstrap.

**Headline table, rule fixed now.** For each planner, take the dial setting with the lowest mean `time_capped` among settings that meet both conditions:
- collision rate ≤ 2% of runs;
- mean `intrusion_s` ≤ 1 s.

Show "none qualifies" if no setting does. Also report success rate, and compute on the seeds that every planner finished (common seeds).

**Two claims, kept separate:**
1. A prediction-aware planner against standard navigation (stop-and-go, reactive, velocity obstacles).
2. TabPFN against LightGBM inside the same planner, on the switch seeds in particular (live against frozen).

A win on (1) is not a TabPFN win.

## Deployment notes (wording rules for the deck)

- **Licence** (checked on github.com/PriorLabs/TabPFN, 21 Nov):
  - The TabPFN code is Apache 2.0.
  - The **TabPFN-3.5 and 3.5-Fast weights, which we use, are under a non-commercial licence.**
  - The TabPFN-2 weights are Apache 2.0 with an attribution requirement.
  - Prior Labs sells a commercial licence.
  - So this is a research prototype. A product would need the commercial licence or the v2 weights (the tier grid in `tools/tier_grid.py` includes v2 for that reason).
- **Model size** (counted from the released checkpoints, every tensor in the safetensors header, float32): TabPFN-3.5 has 218,987,657 parameters (836 MB file); TabPFN-3.5-Fast has 83,534,985 (319 MB). The real-time tier (3.5 with 1 estimator, D-5) costs the larger model; the timing comparison is in `tools/latency.py`.
- **Timings:** our CPU timings come from x64 Python emulated on an ARM laptop, which is a worst case, not a deployment number. We quote no industry compute figures.
- **Latency:** the overnight closed-loop run assumes zero latency: a command applies on the step it is computed. A delayed-command run is planned.
- **Not deployable:** `lgbm_live` retrains six LightGBM models every 2 s, so it is an ablation only ("is the gain the rows or TabPFN?"). It stays out of the headline and the product diagram.
- **Deployable rivals:**
  - frozen `lgbm_all` plus the online conformal offset;
  - `tabpfn_live`, which updates its context but never its weights.
