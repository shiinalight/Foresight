# Results

> **Numbers below are being rerun on feature version f4; see PLAN.md for status.**

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

## Claim A on feature version f4 (Sun 22 Nov): honest boxes in a place never seen

`tools/claim_a.py` → `results/claim_a_f4.md`; `tools/claim_a_boot.py` → `results/claim_a_boot_f4.csv`.

**Setup:**
- The test rows are the same 900 rows per direction as the pre-fix table (identified by episode, agent, t and h), scored on f4 features at +2 s: 324 rows from 195 people going to ETH, 335 rows from 169 people going to Hotel.
- There are 5 context seeds, each a different 1,500-row sample of the training scene.
- Constant velocity uses the tuned filter. LightGBM appears in three forms: on the same 1,500 rows; tuned for small data (m50, 100 trees); and on all training rows.
- Intervals are paired person-bootstrap intervals (5,000 resamples by person).

| Direction | Model | Joint 5–95 box (all / moving) | Box miss | Interval score 10–90 / 5–95 | Error +2 s | Point miss > 0.5 m |
|---|---|---|---|---|---|---|
| Hotel → ETH | Const. velocity | — | — | — | 0.428 m | 28.7% |
| Hotel → ETH | Const. velocity + band | 68.5% / 68.3% | 31.5% | 1.49 / 1.99 | 0.428 m | 28.7% |
| Hotel → ETH | LightGBM (1,500 rows) | 45.1% / 44.4% | 54.9% | 1.56 / 2.29 | 0.409 m | 27.0% |
| Hotel → ETH | LightGBM m50:n100 (1,500 rows) | 58.6% / 58.6% | 41.4% | 1.44 / 2.03 | 0.409 m | 27.0% |
| Hotel → ETH | LightGBM (all rows) | 57.7% / 57.8% | 42.3% | 1.45 / 2.05 | 0.405 m | 27.2% |
| Hotel → ETH | **TabPFN** (1,500 rows) | **74.8% / 75.6%** | **25.3%** | **1.33 / 1.75** | **0.403 m** | 27.2% |
| ETH → Hotel | Const. velocity | — | — | — | 0.254 m | 13.1% |
| ETH → Hotel | Const. velocity + band | 92.2% / 87.9% | 7.8% | 1.02 / 1.34 | 0.254 m | 13.1% |
| ETH → Hotel | LightGBM (1,500 rows) | 67.6% / 51.0% | 32.4% | 0.90 / 1.20 | 0.267 m | 12.5% |
| ETH → Hotel | LightGBM m50:n100 (1,500 rows) | 78.0% / 65.8% | 22.0% | 0.87 / 1.14 | 0.258 m | 12.3% |
| ETH → Hotel | LightGBM (all rows) | 74.9% / 64.5% | 25.1% | **0.78 / 1.04** | 0.253 m | 11.9% |
| ETH → Hotel | **TabPFN** (1,500 rows) | 86.4% / 80.8% | 13.6% | 0.81 / 1.04 | **0.244 m** | 12.4% |

**Nominal coverage:**
- 5–95% box: 80–90% joint (81% if the axes are independent);
- 10–90% per axis: 80%.
- Box miss = 1 − joint coverage.
- Point miss = share of rows whose median prediction is more than 0.5 m off at +2 s.

**Paired person-bootstrap, TabPFN minus LightGBM (negative = TabPFN better):**
- **Hotel → ETH (the hard direction):**
  - vs LightGBM on all rows: interval score 10–90 −0.120 [−0.172, −0.069], 5–95 −0.303 [−0.425, −0.184], box miss −17.0 points [−21.0, −13.0];
  - vs LightGBM m50:n100: −0.111 [−0.157, −0.070], −0.284 [−0.401, −0.181], −16.2 points [−19.0, −13.3].
  - TabPFN is **better** on all three.
- **ETH → Hotel:**
  - vs LightGBM on all rows: interval score 10–90 +0.025 [−0.012, +0.058] and 5–95 +0.007 [−0.057, +0.065], so **level**. Box miss −11.5 points [−15.7, −7.5], so TabPFN misses less.
  - vs LightGBM m50:n100: TabPFN is better on all three (−0.066, −0.101, −8.4 points).

**What this says** (direction-dependent):
- **TabPFN's boxes come closest to their promise in a new place.** In the hard direction they still under-cover: 75% against 80–90% nominal.
- **Hotel → ETH:** TabPFN is clearly best (interval score and box miss, intervals excluding 0).
- **ETH → Hotel:** TabPFN is level with LightGBM on all rows on interval score, and misses the box less often. LightGBM on all rows has the narrower boxes there.
- **Point error:** level with LightGBM on all rows, and slightly better than a tuned constant velocity (0.403 against 0.428 m, and 0.244 against 0.254 m). The share of points more than 0.5 m off is level across the learned models.
- **The fixed band** covers more going to Hotel (92%) but under-covers going to ETH (68.5%), and its interval score is worse than TabPFN's in both directions.

## Numbers before f4 (rerun 21 Nov evening, after the leakage fix; 900 test rows per direction, 1,500 context rows; superseded by the f4 section above)

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

## Claim A on five scenes, leave one scene out (Sun 22 Nov; rule fixed before the run)

Plan, rule and results: `docs/LOSO5.md` (58c2d6b). Script `tools/loso5.py`; files `results/loso5.csv`, `results/loso5_rows.csv`.
- **Scenes:** ETH, Hotel, and the three UCY scenes (Zara1, Zara2, Univ), each held out once.
- **Context:** 1,500 rows from the other four scenes (375 each), 3 context seeds.
- **Measure:** 90% interval score at +2 s, paired person bootstrap.

**Verdict by the rule: "not supported".**
- **TabPFN (1,500 rows) against LightGBM trained on all rows of four scenes** (about 22k rows, ~15× more): pooled −0.008 [−0.023, +0.007], a tie.
  - TabPFN is worse on Hotel (+0.027 [+0.003, +0.052]) and Zara1 (+0.032 [+0.002, +0.067]), and better on Univ (−0.050 [−0.095, −0.010]).
- **Against constant velocity plus band,** TabPFN is better in every scene (pooled −0.365 [−0.405, −0.328]).

**What holds:**
- **Data efficiency:** on the same 1,500 rows, TabPFN beats LightGBM (pooled −0.130 [−0.154, −0.107]), clearly in 4 of 5 scenes.
- **Coverage** is the most even across scenes: 77.5–87%, against 72–86% for LightGBM on all rows.

**So:** TabPFN's edge is strongest when data is scarce and narrow, as in the two-scene claim A where LightGBM had one small scene to learn from. It ties a rival trained on about 15 times more varied data.

**Caveat:** Zara1 and Zara2 share a camera, so those folds are not fully never-seen places.

## Claim B on real data with the final live design (Sun 22 Nov)

`tools/realdata_claimb.py` → `results/realdata_claimb.csv` (summary and bootstrap), `results/realdata_claimb_rows.csv` (per query row).
- Hosted TabPFN, default model, 658 calls. Server usage on the new account went from 0 to 7.07M credits; the account is shared with other sessions (the intent layer ran on it meanwhile).

**Design** (as T5's primary, on real pedestrians):
- Hotel → ETH and ETH → Hotel, every test episode, a moment every 2 s, queries at +2 s (f4, noise 0).
- **frozen:** TabPFN on 2,000 bank rows at natural move frequency, fitted once.
- **split:** frozen plus a recent-only scene model on at most 250 completed rows of this episode, labelled from observed positions; quantiles averaged. Frozen alone below 20 rows.
- **lgbm_all:** LightGBM on every bank row.
- Each also with the batch-rule offset.
- Analysis from 5 s on (the first 5 s are cold start), with a paired person bootstrap.

**From 5 s on** (pooled: 1,540 rows, 544 people):

| | Error +2 s | Joint 5–95 coverage | Width | Interval score |
|---|---|---|---|---|
| frozen | 0.341 m | 77.0% | 0.89 m | 1.370 |
| **split** | 0.342 m | 78.3% | 0.89 m | **1.325** |
| lgbm_all | 0.340 m | 61.2% | 0.66 m | 1.579 |
| split − frozen, pooled | +0.001 [−0.004, +0.007] | +1.3 pts [−0.5, +3.1] | −0.001 [−0.020, +0.019] | **−0.045 [−0.080, −0.009]** |
| split − frozen, Hotel → ETH (319 people) | +0.002 [−0.005, +0.009] | +2.3 pts [−0.2, +4.7] | +0.045 [+0.023, +0.069] | −0.045 [−0.093, +0.007] |
| split − frozen, ETH → Hotel (225 people) | −0.001 [−0.008, +0.007] | −0.6 pts [−2.9, +1.9] | −0.085 [−0.104, −0.066] | **−0.046 [−0.086, −0.005]** |
| split − frozen, both with offset, pooled | +0.001 | +0.7 pts [−1.0, +2.3] | −0.019 [−0.037, −0.000] | −0.018 [−0.049, +0.015] |
| split − lgbm_all, pooled | +0.002 [−0.004, +0.008] | **+17.1 pts [+14.6, +19.6]** | +0.23 m | **−0.254 [−0.324, −0.188]** |

**What this says:**
- **On real pedestrians the scene model helps a little, not much.** The pooled interval score improves (−0.045, interval excluding 0). Error and coverage are not different from the frozen model.
  - Going to Hotel, the gain comes from narrower boxes at the same coverage.
  - Going to ETH, the boxes are slightly wider and cover slightly more; that interval score difference includes 0.
- **With the online offset on both, they are level.**
- **Against LightGBM on all rows, split is clearly better calibrated:** +17 points coverage, interval score −0.25.
- **So claim B on real data is "small and direction-dependent".** The simulator's large after-switch gain (+10 points) comes from a behaviour change the real scenes don't contain. Real scenes differ from the bank in place, not in a sudden switch of behaviour.

## Intent layer: TabPFN at two chained layers (exploratory, Sun 22 Nov; gate fixed in the script before any result)

`tools/intent_layer.py` → `results/intent_layer.log`, `media/fig_intent_layer.png` (hosted TabPFN, 47 calls, ~1.8M credits, commit ed06094).

- **Chain:**
  1. A TabPFN classifier predicts each person's manoeuvre at +2 s (straight, stop, turn left, turn right).
  2. Per-manoeuvre TabPFN range models give the range.
  3. The ranges are mixed by the class probabilities, with quantiles read from the exact mixture distribution.
  - It is compared with the single TabPFN range model on the same context rows (2,000 rows per context seed: 3 seeds on real data, 2 in the simulator), at +2 s, with a paired person bootstrap.
- **Gate:** the chain must have a lower interval score on turn rows (interval excluding 0) and an overall interval score that is not worse.

| Setting | Intent balanced accuracy (chance 0.25) | Turn rows: interval score, chain − single | Overall: interval score | Overall: joint coverage | Error | Gate |
|---|---|---|---|---|---|---|
| Simulated (crowd_cross_t2) | 0.30 | −0.143 [−0.275, −0.005] (475 rows) | **+0.229 [+0.175, +0.287]** (worse) | +7.2 pts | +0.015 m | **not adopted** |
| Real (Hotel → ETH) | 0.49 | **−0.263 [−0.481, −0.075]** (73 rows, 37 people) | −0.022 [−0.057, +0.009] | +5.0 pts (turns +16.9) | +0.007 m | **adopted** |

- **Classifier quality:** TabPFN's log loss beats LightGBM's in both settings: 0.81 against 1.53 simulated, 0.24 against 0.71 real. Balanced accuracy is level.
- **Caveat: the chain does not forecast turns.** On real data, turn recall is 0 for both classifiers. Balanced accuracy 0.49 comes from straight (0.99) and stop (0.97). The gain on turn rows is calibrated widening from the mixture: the classifier puts some probability on turning, which widens the box where people do turn (turn-row coverage 45.7% → 62.6%). Say "better-calibrated ranges for people who turn", never "it predicts turns".
- **What this says:**
  - On real pedestrians, the class probabilities are informative about straight and stop, not turns (see the caveat). Feeding them into per-manoeuvre range models gives better-calibrated boxes on turns, the rows where the single model's box holds the truth only 46% of the time; the chain raises that to 63%. Overall it is level or slightly better.
  - In the simulator, intent is random by construction (balanced accuracy 0.30), and the chain makes the overall score worse.
- **Limits:** a small turn sample (73 rows, 37 people), offline only, one direction, not in the closed loop (D-34).

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

## Closed loop: pre-registration amendment 1 (committed Sat 21 Nov 23:51 Berlin, 0d5d335)

Written before any learned planner, the open-loop oracle included, has run in the crossing crowd.

**What changed since the original pre-registration (Sat ~20:30), disclosed in full:**
- **Pilot and T1:** the pilot (seeds 1000–1012) and T1 (100 seeds, default plaza, D-19) showed that the default plaza does not separate planners.
- **Default-scene T2:** a T2 sweep in the default scene ran partly (`results/closed/*t2-r*`), then stopped when the scene changed.
- **Criterion changed mid-sweep:** collisions are now judged on moving-robot collisions only (D-23).
- **Scene changed:** the confirmatory scene is now the crossing crowd, people crossing the plaza and leaving (D-24, Darshak's decision).
- **Features changed twice:** to f2/f3 (leaks and labels, D-20/D-22) and to f4 (tuned tracking filter, D-26). Everything below uses f4 only.
- **Cross baselines seen before this rule:** some T2 baseline cells in the crossing crowd were visible before this definition was written (an f3 sweep, stopped, and the f4 sweep in progress).

**T2 picker, defined mechanically:**
- **Cells:** robot_repulsion {2, 1, 0.5} × ignore_robot {0, 0.2, 0.4} × people {16, 24, 32}, seeds 4100–4119, in the crossing crowd.
- **Best baseline per cell:** of stop_go, cv m0.4, reactive m0.4 and vo w1, the one with the lowest moving-collision rate (share of seeds with at least one moving-collision step). Ties go to the lower mean intrusion_s.
- **A cell qualifies** when its best baseline has a moving-collision rate ≥ 5%, or a mean intrusion_s ≥ 3 s.
- **Mildest qualifying cell:** the one closest to the default crowd, ordered by highest robot_repulsion, then lowest ignore_robot, then fewest people. The first qualifying cell in that order is chosen.
- **If no cell qualifies:** the closed loop is reported as a demo and a stress test, not as evidence for claim C.

**In the chosen cell, every planner is swept over its own dial:**
- constant velocity at margin {0.2, 0.4, 0.6, 0.8};
- reactive at margin {0.2, 0.4, 0.6, 0.8};
- velocity obstacles at weight {0.5, 1, 2};
- stop_go once;
- open-loop oracle at margin {0.2, 0.4, 0.6};
- the primary TabPFN planner at box_q {0.10, 0.15, 0.20}, so its curve has three points;
- the other learned planners (lgbm, lgbm_all, frozen tabpfn) at box_q {0.10, 0.15}.
- q = 0.05 freezes the robot (D-6).
- Smoothing (D-28) is on for every planner using the shared planner if, and only if, T4's gate passes.

**Open-loop oracle gate (D-21), in the chosen cell, before T5:**
- **Gate:** the oracle beats every baseline curve by the D-14 differences.
- **Why "open-loop":** it rolls a twin forward on the current command, while the real people react to the chosen one, so it is not a true upper bound.
- **If it fails:** T5 still runs, but the closed loop is reported as a stress test.

**Primary comparison, named now.**
- **The primary planner:** the live TabPFN setup decided at 02:00, named in amendment 2 before T5 launches; until then `tabpfn_live`.
- **The primary test:** that planner against each of the four baselines (stop_go, constant velocity, reactive, velocity obstacles, each over its own dial), on both safety axes.
- **Correction for four tests:** Holm–Bonferroni across the four baselines at 5%.
- **Secondary planners:** frozen TabPFN, lgbm and lgbm_all are exploratory. They are reported but carry no claim.

**Headline analysis.** This replaces the absolute ≤ 2% collisions and ≤ 1 s intrusion rule, which disqualifies baselines by construction in a scene chosen by T2.
- **Curves:** for each planner, the mean time_capped over its dial, plotted against:
  - (a) mean min_gap;
  - (b) moving-collision rate.
- **What "beats" means:** planner A beats planner B when, at B's safety level (interpolated along A's curve), A's mean time_capped is lower by more than 2 s (D-14), and the 95% paired-seed bootstrap interval of the difference excludes 0. It has to hold on both (a) and (b).
- **Outside a curve's range, no interpolation.** If B's safety level lies outside A's curve, A beats B there only by dominance: some point of A is at least as safe as B's point on both axes, and more than 2 s faster, with the paired interval excluding 0. Otherwise the pair is "not comparable", which counts as not beating.
- **Also reported:** success rate, intrusion_s, heading_change_per_m, on common seeds only.

**Settings carried in from T4, declared in amendment 2 before T5:**
- whether smoothing is on (D-28 gate);
- the latency budget: the virtual-time delay before a command applies, and the delay before refreshed context can be used. *(Correction, Sun 22 Nov: the refreshed-context delay was never implemented. Refits apply immediately in every run, and only the command delay exists, as `--latency`.)*

Both apply to every planner. Amendment 2 also names the 02:00 TabPFN setup and is committed before T5 launches.

**Run:**
- seeds 3000–3039;
- every seed with a crowd-wide switch at mid-episode;
- crossing crowd at the chosen T2 cell;
- features f4.
- **TabPFN tier in the loop:** 3.5-Fast with 1 estimator, or the setup decided at 02:00 from the context-size test, stated per row (tabpfn_cfg). The deployed-tier documents name 3.5 with 1 estimator.

**Fallback, written now:** if neither TabPFN planner beats every baseline curve under the rule above, the closed loop is reported as a stress test, not as evidence for claim C. Claims A and B then rest on the offline and real-data results.

## Closed loop: pre-registration amendment 2, design and analysis (committed Sun 22 Nov 00:34 Berlin, 889bc36)

Written before any planner ran on the T5 seeds (3000–3039). The settings T4 decides are named here. Their values follow in a short dated addendum before launch.

**Runs in this cell before this text, all disclosed:**
- T2 baselines and the oracle gate (seeds 4100–4119).
- A logging and timing check (seed 5199: lgbm_all, reactive, tabpfn_split).
- The T4 offset test and timing smoke (seeds 5100–5103, running).
- Before that, one split smoke run on the old bank (seed 9001, 16 people).

From the learned-planner runs only ms per plan, the logging check and the offset's behaviour are read.

**What changed since amendment 1, disclosed in full:**
- **The oracle gate failed** (`results/oracle_gate_cross.csv`, `tools/curve_compare.py --a oracle`, 20 seeds, every baseline dial). The open-loop oracle beats stop_go, and it fails cv, reactive and vo. Against cv m0.2 it is 1.59 s faster [0.86, 2.39], short of the 2 s margin. vo w0.5 (19.24 s) is faster than every oracle point, so it is outside the oracle's front and not comparable.
- **Claim C is therefore reported as a stress test.** No co-primary is added.
- **The gate was hardly passable.** The time floor (straight start-to-goal distance minus the 0.5 m arrive radius, at 1.0 m/s) averages 17.74 s over the 20 seeds (range 17.50–19.13). vo w0.5 runs 1.50 s above that floor and cv m0.2 3.70 s above it. D-14's 2 s was a detectability margin (from the pilot's paired spread), not an estimate of the headroom.
- **The cell is mild:** it qualifies by one seed. The best baseline, cv m0.4, collides while moving in 2 of 20 seeds (10%); the next milder cell has 0% (D-32).
- **The live setup was chosen on simulated data** (D-30). It came from `tools/live_variants.py` and the WP-J2 rule (`tools/live_variants_cp.py`, `results/live_variants_cp.csv`). In Setting B, after the switch, with paired person bootstrap:
  - **split2k+offset vs frozen2k+offset:** interval score −1.29 [−1.70, −0.94], coverage +12.6 points [+9.0, +16.3], median error −0.057 m [−0.092, −0.021], pinball −0.034. Before the switch the two are level.
  - **split2k+offset vs lgbm_all+offset:** interval score −1.72, coverage +19.6 points, error not different.
  - **Limits:** no variant reaches 80% after the switch (best 68.4%). The offline offset barely moved (8 update rounds per episode), so whether frozen+offset could match split on sharpness stays open. Cost: split ~3.0 s per moment offline, against 0.27 s for frozen.
  - **On real data** (Setting A, Hotel → ETH: 137 query rows over 44 moments, 83 people), no variant is distinguishable from constant velocity. Split 0.303 m against CV 0.318 m: −0.015 m [−0.041, +0.010], paired person bootstrap. Frozen: −0.011 [−0.039, +0.016].
- **T3** is live_variants Setting B: 6 episodes instead of 12 (D-30).
- **Frozen context:** the first version of `context.frozen` forced up to half rare moves. It was corrected to natural frequency (D-31) before any learned planner ran in the cell.

**Scene and data:**
- **Cell:** robot_repulsion 2, ignore_robot 0.2, people 24, crossing crowd, features f4, top speed 1.0 m/s for every planner, duration 40 s.
- **Bank:** `data/crowd_cross_t2`, 120 episodes generated in that cell (seed 24).
  - The Kalman filter was retuned on its training split: q 0.01, r 0.05. r is at the grid edge (0.05 is the grid's smallest value; kalman.json grid_edge true). The crowd_cross filter of D-26 was interior. CV error at +2 s is 0.827 m.
  - Every episode with erratic behaviour before or after a switch is excluded (`--bank-exclude-regime erratic`), so erratic is new to every model.
- **Gates rerun in the cell:**
  - **S0 passes narrowly.** On 2,000 test rows at +2 s: CV 0.848 m; LightGBM on all rows 0.712 m (16.0% better), 5–95 box 80%; TabPFN fast-1 on 1,000 rows 0.777 m (8.4%), box 82%.
  - **check_shift:** the robot features of the live queries match the bank. robot_dist quantiles: bank 1.91 / 4.29 / 6.62 / 9.20 / 13.24 m, live 2.08 / 4.46 / 6.86 / 9.56 / 13.83 m. Robot within 2 m: bank 5.6%, live 4.5%.
- **Switch:** every seed starts calm or hurried (alternating by seed) and switches the whole crowd into erratic at t = 5 s (`--switch --switch-all --switch-to erratic --switch-t 5`). Why: so the scene model can receive erratic labels while the robot is still crossing. The value was chosen before any learned planner ran in the cell, on timing arguments only.

**Planners (flags common to all learned arms: `--frozen-ctx fill --context 2000`, TabPFN 3.5-Fast with 1 estimator, `tabpfn_cfg = fast1-est1`):**
- **Primary:** `tabpfn_split` at box_q {0.10, 0.15, 0.20}, with the online offset if the in-loop offset passes its reproduction test by 03:00, otherwise without it. The addendum states which.
  - `tabpfn_split` is the frozen bank model (2,000 rows at natural move frequency, fitted once) plus a scene model.
  - The scene model is refit every 2 s on at most 250 of this episode's most recent completed rows, labelled from observed (filtered) positions.
  - Quantiles are averaged per axis, then sorted. The frozen model runs alone while fewer than 20 recent rows exist.
- **Secondary (exploratory, no claim), box_q {0.10, 0.15}:** frozen `tabpfn`, `lgbm_all` and `lgbm` (lgbm on the same 2,000 rows). Each gets the offset as well if it is built.
- **Baselines** (each over its own dial):
  - cv at margin {0.2, 0.4, 0.6, 0.8};
  - reactive at margin {0.2, 0.4, 0.6, 0.8};
  - vo at weight {0.5, 1, 2};
  - stop_go once.
- **Open-loop oracle** at margin {0.2, 0.4, 0.6}.
- **Secondary arms on the loop itself:**
  - reactive m0.4 and vo w1 replanning every 0.1 s (industry brief);
  - the primary with a virtual-time latency (`--latency`, value from T4).
  - The primary arm itself runs at zero latency.
- **Online offset spec** (as offline, so it stays pre-specified):
  - alpha 0.2, eta 0.02, d0 0, scored on the raw 5–95% box: d ← d + eta (miss − alpha), per completed row in order of t_known, updated every 2 s.
  - Labels come from OBSERVED (filtered) positions (deployment-realistic). Offline used truth; this is a deliberate difference.
  - The box the planner reads, [box_q, 1 − box_q], is widened by the same d, so the caution dial keeps its meaning.
  - In the loop it gets far more updates than offline (24 people × 2 horizons every 0.4 s). eta stays 0.02.
- **Deadline takeover:** designed, not built.

**Primary analysis:** amendment 1's rule, unchanged:
- matched-safety curves on min_gap and moving-collision rate;
- D-14 margin of 2 s;
- 95% paired-seed bootstrap;
- dominance only outside a curve's range;
- Holm–Bonferroni across the four baselines.

Implementation: `tools/curve_compare.py --glob "*t5-*" --seeds 3000-3039 --a tabpfn_split --holm`.

**Each planner's curve is read on its Pareto front: dominated dial points are dropped, interpolation runs along the front, outside its range only dominance counts, and a bootstrap resample with no comparison counts against the planner under test.**

A consequence to state: a baseline point faster than the primary's fastest arm can only be beaten by dominance.

**Exploratory analyses, added after the oracle gate was seen, carrying no claim-C verdict:**
- **(a) Safety at matched time.** The same Pareto-front rule with the axes swapped (`--matched safety`): A's min_gap and moving-collision rate at each baseline point's time_capped, read along A's front (faster never safer).
  - **Margins**, set from baseline rows only (the T2 cell's 12 baseline arms, all 66 pairs, median paired SD): min_gap 0.356 m, moving-collision indicator 0.550. The detectable differences at 40 seeds (80% power, 5% two-sided) are **0.158 m** and **0.244** (24 points). A must beat each baseline point by more than these on both axes, with the paired interval excluding 0.
  - **Order:** the oracle is tested against each baseline first, on T5's fresh seeds. The primary is tested only against the baselines the oracle beats.
  - **Hint only:** on the 20 T2 seeds the oracle's 0% vs cv's 15% is 3 crowds, on seeds selected for baseline collisions.
- **(b) In-loop claim B.** `tabpfn_split` vs frozen `tabpfn` vs `lgbm_all`, all at box_q 0.15.
  - **Metrics:** joint coverage of the raw 5–95% box, its width, and its interval score (alpha 0.1), at +2 s against true positions. Also the served box's coverage.
  - **Before** the switch (t < 5 s) and **after** it, as per-episode aggregates (`inloop_*` columns in every result row), paired by seed, with 95% paired-seed bootstrap intervals.
  - Reported without multiplicity correction, as exploratory.

**Run:**
- seeds 3000–3039;
- one process per primary caution level, run in parallel;
- baselines, oracle and secondary loop arms in further processes;
- **order:** seed-outer, planner-inner, each seed checkpointed as it finishes. The analysis uses common seeds only, so a run that is cut short still yields paired comparisons;
- **hard stop at 08:00 Berlin:** the analysis uses whatever has completed by then.

**Named for the T4 addendum** (before launch; values only, no new mechanisms):
- smoothing on or off (D-28 gate, every planner using the shared planner);
- online offset on or off (reproduction test);
- the latency value of the secondary latency arm.

**Fallback:** unchanged. Claim C is already a stress test (gate failed). T5 is now the in-loop claim-B test and the source of demo traces.

**T4 addendum, part 1: the offset update rule** (committed Sun 22 Nov 00:36 Berlin, bec10b7, before any T4 offset result was read; `tools/t4_offset.ps1`):
- **Why:** in the loop the offset gets ~2,500 row updates per episode in 2 s batches of correlated rows. One update per row steps d by eta × batch, which can make the box oscillate and freeze the robot. Adaptive conformal assumes one update per time step.
- **Test:** seeds 5100–5105 (outside T5 and T2), T5 config, box_q 0.15; tabpfn_split, tabpfn and lgbm_all; three settings:
  - offset OFF, scored only (eta 0);
  - **(i) per row:** d += 0.02 (miss − 0.2) for every completed row, as now;
  - **(ii) per batch:** d += 0.05 (batch miss rate − 0.2) once per 2 s refresh.
- **Gate, judged on the primary (tabpfn_split):**
  - **Coverage metric:** mean over seeds of `inloop_cov90d_post`, the joint coverage at +2 s, against true positions, of the 5–95% box widened by the offset in force, after the switch.
  - **A rule is stable** when its mean seconds_stopped is at most OFF's + 0.4 s and its mean freeze_rate at most OFF's + 0.02.
  - **Choice:** among stable rules, the one with coverage closest to 80% (ties within 1 point go to the batch rule). The offset is ON in T5's primary only if that rule is also closer to 80% than OFF.
  - **Otherwise:** the offset is OFF in the primary and is reported as a secondary arm only (the rule closer to 80%).
- **The same rule** applies to every secondary arm that carries the offset.

**T4 addendum, part 2: how the smoothing gate is read** (written before any T4 smoothing result was read; `tools/t4_smooth.ps1`):
- **Test:** seeds 5110–5119, T5 config except the margin, box_q 0.15, no offset; tabpfn_split, tabpfn, lgbm_all and cv m0.4, with smoothing off and on (D-28 defaults). *(Disclosure: `tools/t4_smooth.ps1` ran the learned planners at `--margin 0.4`, not T5's 0.2 (D-7). The verdict, OFF, does not change what T5 runs.)*
- **D-28 gate, made exact:**
  - For each learned planner (tabpfn_split, tabpfn, lgbm_all), mean heading_change_per_m drops by at least 30% with smoothing.
  - Paired over the common seeds, mean time_capped is no worse by more than 2 s, and mean intrusion_s no worse by more than 1 s (the D-14 differences).
  - The moving-collision rate rises by at most 0.1 (one seed in ten).
- **Decision:** if all three learned planners pass, smoothing is ON for every planner using the shared planner in T5. Otherwise it is OFF.
- **Latency arm:** set to the primary's measured ms per plan in the T5 config (T4 offset test), rounded up to the next 0.2 s simulation step. *(Replaced by part 3.)*

**T4 addendum, part 3: latency arm and fast-replan arm** (written before any latency result; no latency run exists):
- **Latency arm fixed at 0.4 s**, one replan period: tabpfn_split at q 0.15 with `--latency 0.4`, testing predictions that are one step stale.
- Part 2's rule (measured ms per plan rounded up) would have given ~2.0 s. It was replaced before any result because the ~1.9 s per plan comes from 8 contended processes on an x64-emulated CPU. That is a worst case no deployment would budget; a real system budgets one replan period.
- **Fast-replan baselines:** reactive m0.4 and vo w1 replan every simulation step, 0.2 s. Amendment 2's "0.1 s" is below the 0.2 s simulation step and cannot be run.

**T4 addendum, part 4: values, and the T5 launch** (`tools/t4_report.py`; written before any T5 result):
- **Offset gate: ON, rule (ii) per batch** (eta_b 0.05 per 2 s refresh), on tabpfn_split over 6 seeds (5100–5105).
  - Per row: coverage after the switch 76.8%, but stopped 4.40 s against OFF's 0.20 s and freeze rate 0.123 against 0.008. UNSTABLE: the oscillation froze the robot.
  - Per batch: 61.4%, stopped 0.40 s, freeze 0.015. Stable.
  - OFF: 59.5%.
  - So the batch rule is chosen: it is the only stable rule, and it is closer to 80% than OFF, though only by 1.9 points.
  - Every learned arm in T5 carries the offset with this rule.
- **Smoothing gate: OFF.** At 8 of 10 seeds, run before the last 2 finished so nothing waited:
  - tabpfn_split: heading change −19% (< 30%), time +5.0 s;
  - tabpfn: −30% short of the line, time +1.2 s;
  - lgbm_all: −25%, time +6.4 s.
  - If the full 10 seeds reverse the verdict, the smoothing-ON sets below enter the analysis instead; this is stated before any T5 result.
- **No time gates** (Darshak, Sun ~01:10: "do not block anything by time"):
  - Every arm using choose/choose_vo (baselines, oracle, fast-replan) runs in BOTH smoothing modes (tags `*-sm` = ON). The smoothing gate picks the set that enters the analysis.
  - The primary also runs at q 0.15 with smoothing ON as an extra process.
- **Secondary arms run on seeds 3000–3019 to reduce compute (exploratory):** frozen tabpfn, lgbm_all, lgbm, the 0.4 s latency arm and the 0.2 s replan arms. The primary (tabpfn_split at q 0.10 / 0.15 / 0.20) and every baseline and oracle arm use all 40 seeds. A secondary process that finishes early may be extended to 40 seeds only while the CPU has spare capacity.
- **Analysis sets:**
  - Primary test: the smoothing-OFF rows at latency 0 and replan 0.4 s.
  - The latency, fast-replan and smoothing-ON rows are reported separately.

**T4/T5 addendum, part 5: compute, reduced before results** (Sun 22 Nov 01:36 Berlin).
- **Why:** split q0.20, seed 3000 took 1,030 s (16.2 s per plan), with the CPU at 100% under ~35 python processes. At that rate one split process needs ~11 h. Darshak wants the solution final tonight.
- **No learned result was read** except that one q0.20 timing row, which is discarded (`results/closed/discarded/`).
- **Learned set kept**, all on seeds 3000–3019 (20 seeds, paired), batch offset, smoothing OFF, latency 0, replan 0.4:
  1. tabpfn_split q0.15 (**primary**);
  2. tabpfn_split q0.10 (second curve point);
  3. frozen tabpfn q0.15;
  4. lgbm_all q0.15.
  - The two split processes already running keep running and are stopped after seed 3019 (`tools/seed_cap.ps1`).
  - tabpfn and lgbm_all q0.15 were restarted on their own (`tools/t5_pruned.ps1`).
- **Not run (compute):**
  - tabpfn_split q0.20;
  - tabpfn_split q0.15 with smoothing ON;
  - tabpfn_split q0.15 at 0.4 s latency;
  - frozen tabpfn q0.10;
  - lgbm q0.10 and q0.15;
  - lgbm_all q0.10.
  - Of the 0.2 s replan arms, reactive m0.4 (smoothing OFF) finished its 20 seeds. The other three were stopped part-way, and their partial rows are discarded.
- **Baselines and the oracle** finish their 40 seeds in both smoothing modes. Comparisons with learned planners use the common seeds 3000–3019.
- **Smoothing gate at the full 10 seeds: OFF**, as decided at 8:
  - tabpfn_split: heading change −28%, time +3.6 s;
  - tabpfn: −30% (just short of the line), time +2.0 s;
  - lgbm_all: −31%, time +5.1 s.
  - So the smoothing-OFF sets enter the analysis.
- **Analysis changes:**
  - **The primary test** (Holm across the 4 baselines) runs on 20 seeds. Its curve has two points (q0.10, q0.15).
  - **Detectable time difference at n = 20:** about 2.6 s (80% power, 5% two-sided, the cell's median paired SD of 4.16 s), against 1.84 s at 40. Claim C is a stress test anyway.
  - **Exploratory (a)** keeps its stated margins (0.158 m, 0.244), so it has less power at 20 seeds.
  - **The in-loop claim B comparison** (split vs tabpfn vs lgbm_all at q0.15, before and after the switch, paired by seed) is now the main closed-loop question.
- **Compute, part 6** (Sun 22 Nov ~01:52):
  - Every smoothing-ON arm was stopped: baselines, oracle and fast-replan, all tagged `-sm`. Their partial rows are in `results/closed/discarded/smoothing_on/`.
  - Baseline and oracle seeds beyond 3019 were stopped (`tools/seed_cap.ps1 -PythonOnly`). Arms that had already run 40 seeds keep those rows.
  - Neither set is used in any analysis: smoothing is OFF by the T4 gate, and the primary test runs on the common seeds 3000–3019.

## Closed loop T5 results (Sun 22 Nov, seeds 3000–3019, analysed by the pre-registered rules)

**Setup:**
- Chosen cell (repulsion 2, ignore 0.2, 24 people, crossing crowd), every seed switching into erratic at 5 s, f4.
- Learned arms carry the batch offset, smoothing OFF, latency 0, replan 0.4 s.
- 20 paired seeds (addendum part 5).
- Files: `results/t5_primary.csv`, `results/inloop_claimb.csv`, `results/t5_oracle_matchedtime.csv`. Scripts: `tools/curve_compare.py`, `tools/inloop_claimb.py`.

**Points** (mean time_capped s / mean min_gap m / moving-collision rate):

| Planner | Dial | Time | Min gap | Collision rate |
|---|---|---|---|---|
| tabpfn_split (primary) | q 0.10 | 28.64 | 0.232 | 15% |
| tabpfn_split (primary) | q 0.15 | 26.23 | 0.158 | 15% |
| constant velocity | m 0.2 / 0.4 / 0.6 / 0.8 | 21.36 / 23.23 / 25.83 / 28.92 | 0.089 / 0.199 / 0.323 / 0.254 | 25 / 15 / 5 / 15% |
| reactive | m 0.2 / 0.4 / 0.6 / 0.8 | 23.27 / 24.99 / 26.60 / 30.89 | −0.120 / 0.086 / 0.127 / 0.125 | 50 / 10 / 15 / 10% |
| velocity obstacles | w 0.5 / 1 / 2 | 20.08 / 21.32 / 22.31 | −0.038 / 0.064 / 0.053 | 40 / 20 / 30% |
| stop_go | — | 31.38 | −0.174 | 10% |
| open-loop oracle | m 0.2 / 0.4 / 0.6 | 20.37 / 22.69 / 25.68 | 0.221 / 0.385 / 0.388 | 0 / 0 / 10% |

**Primary test (claim C, already a stress test after the oracle gate): tabpfn_split beats no baseline.**
- All four fail Holm–Bonferroni (p = 1.0).
- The split planner is slower at comparable safety. At cv m0.4's safety it is 4.3 s slower on the min_gap axis and 3.0 s slower on the collision axis.
- **Against cv m0.2 and vo w0.5, w1 and w2, it is safer but slower.** It has a larger closest gap (0.158 m against −0.038 to 0.089 m) and fewer moving collisions (15% against 20–40%), but it is 3.9–6.2 s slower. The primary rule compares time at matched safety, so safer and slower does not count as beating. ("dominance" in `results/t5_primary.csv` names the comparison method, not a finding.)
  - Against the exploratory matched-time margins (0.158 m and 0.244), its safety edge stays below both against cv m0.2 (+0.069 m, −10 points), vo w1 (+0.094 m, −5 points) and vo w2 (+0.105 m, −15 points). It exceeds both only against vo w0.5 (+0.196 m, −25 points), which is 6.2 s faster.
  - The exploratory matched-time test of the primary was not run: by its pre-set rule it applies only to baselines the oracle beats, and the oracle beat none.
- Claim C is not supported in this crowd.

**Exploratory (a), safety at matched time:** by the per-baseline rule (a baseline is beaten only when every one of its dial points is beaten on both axes), the open-loop oracle beats no baseline, so the primary was not tested there (rule written before the result).
- Per point, it does beat cv m0.2, reactive m0.2 and vo w2 on both axes, with collision-rate gains of 0.25–0.50 (`results/t5_oracle_matchedtime.csv`). Each of those baselines has other dial points it does not beat.
- Per point, perfect prediction buys clear clearance (min_gap) at matched time against cv m0.2/m0.4, every reactive dial, vo w1/w2 and stop_go.
- Collision-rate gains exceed the 0.244 margin at 3 of 11 comparable baseline points (cv m0.2, reactive m0.2, vo w2) and stay below it at the other 8, at 20 seeds.
- vo w0.5 is faster than every oracle point, so not comparable.

**Exploratory (b), in-loop claim B** (q 0.15; +2 s against true positions; split minus other, 95% paired-seed intervals):

| | tabpfn_split | frozen tabpfn | lgbm_all | split − tabpfn | split − lgbm_all |
|---|---|---|---|---|---|
| After switch: 5–95 coverage | 59.3% | 49.3% | 41.3% | +10.0 pts [+8.4, +11.6] | +18.0 pts [+16.7, +19.3] |
| After switch: interval score | 5.03 | 6.34 | 7.03 | −1.31 [−1.55, −1.08] | −2.00 [−2.20, −1.80] |
| After switch: width | 2.11 m | 1.85 m | 1.47 m | +0.26 m | +0.63 m |
| After switch: median error | 1.016 m | 1.073 m | 1.036 m | −0.057 [−0.094, −0.021] | −0.020 [−0.051, +0.010] |
| After switch: coverage with offset | 60.8% | 51.3% | 44.6% | +9.5 pts | +16.2 pts |
| Before switch: interval score | 2.70 | 2.52 | 2.46 | +0.18 [+0.10, +0.27] | +0.24 [+0.17, +0.32] |
| Before switch: median error | 0.514 m | 0.499 m | 0.476 m | +0.015 [+0.006, +0.025] | +0.037 [+0.026, +0.048] |

**What this says:**
- **In the loop, the scene model buys calibration after an unseen regime arrives:** +10 points coverage, a better interval score and lower error than the frozen model. It costs a little before the switch, while the scene model sees only calm or hurried rows the bank already covers.
- **No model reaches the 80% target after the switch**, with or without the offset. The batch offset adds ~1.5 points.
- **This reproduces WP-J2's offline Setting B result in closed loop** (simulated crowd only).
- **The speed gap is not TabPFN's doing.** At q 0.15 every learned planner drove the same: split 26.23 s, frozen TabPFN 26.35 s, LightGBM on all rows 25.90 s. Paired: split − frozen −0.12 s [−1.35, +1.25], split − LightGBM +0.33 s [−1.29, +1.99], even though LightGBM's boxes are 0.63 m narrower after the switch. Better prediction did not change how this box planner drives; its caution setting did.

**Extension to 40 seeds** (started Sun 22 Nov ~02:25 Berlin, before any result from seeds 3020–3039; `tools/t5_extend.py`):
- **Why:** spare CPU, as addendum part 5 allows.
- **Same arms and dials:** tabpfn_split q0.15 and q0.10, frozen tabpfn q0.15, lgbm_all q0.15, vo w 0.5/1/2 and oracle m 0.2/0.4/0.6, on only the seeds not yet run.
- **The 20-seed analysis above stays the confirmatory result.** The 40-seed tables follow as an update, computed by the same scripts and rules.

**Update: 40 seeds (3000–3039), addendum part 5 extension** (same arms, rules and scripts; the 20-seed result above stays the confirmatory one):
- Files: `results/t5_primary_40.csv`, `results/t5_oracle_matchedtime_40.csv`, `results/inloop_claimb_40.csv`.
- **Primary:** unchanged. tabpfn_split beats no baseline; all four fail Holm (p = 1.0).
  - Points (time / gap / collision rate): split q0.10 29.28 s / 0.193 m / 15%; q0.15 27.20 / 0.155 / 20%.
  - cv m0.2–0.8: 22.14–28.83 s. vo w0.5–2: 20.02–22.93 s. reactive m0.2–0.8: 22.86–32.26 s. stop_go 31.72 s.
- **Exploratory (a):** unchanged. By the per-baseline rule the oracle beats no baseline. Oracle points: m0.2 20.69 s / 0.221 m / 0%, m0.4 22.37 / 0.383 / 0%, m0.6 25.71 / 0.441 / 5%.
- **In-loop claim B, after the switch, split − frozen tabpfn:**
  - coverage +10.2 points [+9.3, +11.2] (59.5% against 49.3%);
  - interval score −1.33 [−1.47, −1.20];
  - error −0.061 m [−0.085, −0.039].
  - Against lgbm_all: +18.5 points, −2.00, −0.022 m [−0.041, −0.003].
  - Before the switch split is slightly worse, as at 20 seeds (interval score +0.22 against frozen).
- **Speed of the learned planners at q0.15:** split 27.20 s, frozen TabPFN 27.16 s, lgbm_all 25.96 s.
  - split − frozen: +0.03 s [−1.08, +1.16], the same.
  - split − lgbm_all: **+1.24 s [+0.17, +2.34]**. At 40 seeds LightGBM, with 0.64 m narrower boxes after the switch, drives about 1.2 s faster than both TabPFN planners. At 20 seeds the interval included 0.
  - So the two TabPFN planners (split's boxes 0.26 m wider than frozen's) drive the same, while narrower LightGBM boxes buy about a second. The caution setting still dominates: all learned planners stay 3–5 s slower than cv and vo at comparable safety.

**Showcase seed** (CTO's rule): **3013**, a seed on which every primary-test planner reached the goal and tabpfn_split q0.15's time (25.8 s) is closest to its median (26.40 s).

## Caution-dial sweep and latency arm (exploratory, Sun 22 Nov)

Plan, amendment and full results: `docs/EXPLORATORY_DIALS.md` (plan committed before launch, e6bc556; results c912241). Files: `results/dx_split.csv`, `results/dx_lgbm.csv`, `results/closed/*dx-*`, `results/closed/*gpu-lat*`.

**Dial sweep** (fresh seeds 3100–3119; split, the baselines and the oracle on the rented L4 GPU, lgbm_all on the laptop, disclosed; analysed by the T5 curve rules):
- **tabpfn_split beats no baseline under Holm.** Per-baseline p: cv 1.0, reactive 0.158, stop_go 1.0, vo 1.0. lgbm_all also beats none.
- **Lower caution brings it into the baselines' range.** Split at q 0.35 runs 24.37 s / 0.137 m / 15% moving-collision crowds, against constant velocity m0.4 at 24.69 s / 0.060 m / 25%. That is about as fast with more clearance, but not significant.
- **Against reactive,** split is faster at matched safety at every comparable point (+1.3 to +6.7 s), but the every-point rule still fails. Velocity obstacles stay faster (−0.4 to −2.9 s).
- **The open-loop oracle** has 0% moving-collision crowds at every dial.

**Latency arm** (pre-registered in T4 addendum part 3; tabpfn_split q 0.15, seeds 3000–3039, both arms on the L4 so the pair is on one device):
- A 0.4 s command delay (one replan period of stale prediction) costs +1.35 s [+0.18, +2.50].
- Crowds with a moving collision rise from 7.5% to 30% (+22.5 points [+10, +37.5]). The closest gap falls by 0.098 m [−0.207, +0.003].
- **Reading:** prediction has to arrive within the replan period. The L4's 71 ms does; the laptop CPU's 452 ms would not.
- The no-delay arm on the L4 (27.32 s / 7.5%) differs from the laptop T5 rows at the same seeds (27.20 s / 20%). That is the cross-device sensitivity documented in `docs/DEPLOYMENT.md`.

## Generalisation: two more crowds (exploratory, Sun 22 Nov)

Plan (committed before launch), amendment and results: `docs/GENERALISATION.md` (d869817).
- **Crowds:** G1 open plaza (16 people, goals anywhere, nobody ignores the robot) and G2 dense crossing (32 people, half ignore the robot).
- Seeds 3200–3219, every arm on 20 seeds. Devices are mixed (TabPFN arms on the L4; lgbm_all and the G2 baselines on the laptop), as disclosed there.

**Results:**
- **Claim C fails in both new crowds,** so it fails in all three. tabpfn_split beats no baseline under Holm. lgbm_all beats only stop_go, in G1.
- **In-loop claim B, split against lgbm_all after the switch, holds in all three crowds.** Its size depends on the crowd:

  | Crowd | Coverage | Interval score |
  |---|---|---|
  | G1 open plaza | +6.6 points [+5.7, +7.5] | −0.30 |
  | T5 crossing | +18.0 points | −2.00 |
  | G2 dense crossing | +19.7 points [+18.7, +20.8] | −2.05 |

- **G2, safety at similar time** (descriptive; the matched-time curve test does not pass Holm):
  - split q 0.35: 24.53 s, −0.008 m closest gap, 35% moving-collision crowds;
  - constant velocity m0.2: 23.91 s, −0.158 m, 65%;
  - velocity obstacles w1: 23.61 s, −0.199 m, 75%.
  - That is about half the collision crowds at similar time.

## TabPFN picks the move (exploratory, Sun 22 Nov; plan and pass rule written before any run)

Plan: `docs/TABPFN_POLICY.md`. Pilot and DAgger rule: `docs/runs/policy_pilot_rule.md` (6b9ffd4). Code: `foresight/core/policy.py`, `run_closed.py --planners tabpfn_policy` (232e9f0). Analysis: `tools/policy_parity.py` → `results/policy_parity.csv`. Rows: `results/closed/*pol-*`, `results/policy/*.csv`. All runs on the L4 pod.

**Chain:**
1. TabPFN ranges (tabpfn_split, q 0.35, batch offset).
2. Every candidate move becomes one row: robot state relative to the goal; the candidate's speed and turn; its clearance to the nearest person's box at +1 s and +2 s and at the 0.4 s check; its progress and progress rank; in-area; people nearby.
3. A TabPFN classifier (local, fit_with_cache, fitted once) scores P(teacher picks it), and the policy takes the best candidate that passes the hard checks.

- **Teacher:** the box planner choose() on the same ranges, on seeds 3300–3339: 53,372 candidate rows, 2,426 picks.
- **Context:** every pick plus 3 random non-picks per pick (9,704 rows).
- **Kept classical, stated as such:** the tracking filter, the coverage offset, and the hard checks (the 0.4 s constant-velocity check and staying in the area).

**Pilot** (3400–3404): passed the pre-written rule. The robot reached the goal in 5 of 5 crowds and picked the teacher's move at 66–96% of replans, so no DAgger round was needed. The pilot crowds count as the first 5 of the 20 test crowds (same policy and context).

**Test:** 20 fresh crowds, 3400–3419, policy and teacher on the same device, paired. Each difference is policy − teacher, with a 95% paired-seed interval:

| | Policy | Teacher | Difference | Rule | |
|---|---|---|---|---|---|
| Goal reached | 100% | 100% | 0 | ≥ −10 points | pass |
| Time to goal | 24.65 s | 24.10 s | +0.55 [−0.34, +1.65] | ≤ +2.6 s | pass |
| Closest gap | 0.020 m | 0.012 m | +0.009 [−0.113, +0.114] | ≥ −0.158 m | pass |
| Moving-collision crowds | 45% | 40% | +5 points [−15, +25] | ≤ +24.4 points | pass |

**Verdict by the rule: "TabPFN can drive the loop at the teacher's level."**
- **Honesty line:** the plan's margins were applied to point estimates. Read as a non-inferiority test on the 95% intervals, time passes (upper end +1.65 s against 2.6) and so does the gap (lower end −0.113 m against −0.158). The collision rate's upper end, +0.25, just exceeds the 0.244 margin. So it matches its teacher on point estimates, but the collision interval is too wide at 20 crowds to rule out a rate 25 points worse.
- It picks the teacher's move at 87.4% of replans (66–98% per crowd).
- The hard checks overrode its first choice at only 0.94% of replans, so TabPFN does the driving.
- It costs 333 ms per plan on the L4, against 255 ms for the teacher.

**Limits:**
- **It imitates a planner that does not beat the baselines,** so this shows TabPFN can be the controller, not a better one. Claim C stays failed.
- **The teacher itself collides while moving in 40% of these crowds** at q 0.35. The dial sweep's 15% was on other seeds.
- **The collision-rate interval is wide** at 20 crowds.

## Same core, another form factor: highway cars (exploratory, offline, Sun 22 Nov; rules fixed before any TabPFN run)

Plan, rules and results: `docs/HIGHWAY.md` (plan 34f1954, results 06d8d93). Figure `media/fig_highway.png`.
- **Simulator:** highway-env `highway-v0`, a 4-lane road with 20 IDM/MOBIL cars plus an ego car, and regimes defensive, normal and aggressive.
- **Data:** 72 episodes; 54 training and 18 held out. The core is unchanged: same features, same models, same planner-free scoring.
- **Statistics:** paired vehicle bootstrap.

**H1, the same core on cars: PASSED** (interval score lower than both rivals, intervals excluding 0).

| Model | Coverage | Interval score |
|---|---|---|
| TabPFN (1,500 rows) | 85.6% | 3.57 |
| lgbm_all (100,440 rows) | 82.1% | 4.06 |
| cv_band | 82.9% | 7.78 |

- TabPFN − lgbm_all: −0.49 [−0.66, −0.34], with 67× fewer rows.
- TabPFN is also lowest in every regime.

**H2, adapts after traffic turns aggressive** (a style the bank never saw): **PASSED**.
- Split against frozen: coverage 76.7% against 62.5%, interval score −0.98 [−1.41, −0.56], error −0.14 m.
- About 80% of the gain falls in the first 6 s after the switch (coverage 20% → 51%), so it means "adapts through the transition". Once traffic has settled, the frozen model is roughly right again.

**H2b, wrong embodiment** (no pass rule): a pedestrian bank predicting cars gives boxes about 50 m wide. That fails safe but is useless. Adding car scene rows halves the damage (interval score 51.7 → 28.2) but does not rescue it, so the bank must match the embodiment (D-35).

**Limits:** offline and simulated, with no closed loop on cars.

## Deployment notes (wording rules for the deck)

- **Licence** (checked on github.com/PriorLabs/TabPFN, 21 Nov):
  - The TabPFN code is Apache 2.0.
  - The **TabPFN-3.5 and 3.5-Fast weights, which we use, are under a non-commercial licence.**
  - The TabPFN-2 weights are Apache 2.0 with an attribution requirement.
  - Prior Labs sells a commercial licence.
  - So this is a research prototype. A product would need the commercial licence or the v2 weights (the tier grid in `tools/tier_grid.py` includes v2 for that reason).
- **Model size** (counted from the released checkpoints, every tensor in the safetensors header, float32): TabPFN-3.5 has 218,987,657 parameters (836 MB file); TabPFN-3.5-Fast has 83,534,985 (319 MB). The real-time tier (3.5 with 1 estimator, D-5) costs the larger model; the timing comparison is in `tools/latency.py`.
- **Positioning** (docs/INDUSTRY.md; RSS, Shalev-Shwartz et al. 2017; ISO/IEC TR 5469): calibrated ranges are for efficiency, a worst-case floor is for safety. **No statistical safety guarantee is claimed.** Live context updates change inputs, not weights, and would still have to be verified per deployment.
- **Safety floor status:** the short 0.4 s check exists (per planner, D-8). A deadline takeover (constant velocity plus band when the predictor is late) is **designed, not built**, unless T4 adds it; this line changes if it does.
- **TabPFN tier per result:** the closed loop runs TabPFN-3.5-Fast with 1 estimator (recorded per row as `tabpfn_cfg`); the deployed real-time tier in DEPLOYMENT.md is TabPFN-3.5 with 1 estimator (D-5, D-18). Offline real-data results use the 3.5 default unless a row says otherwise.
- **Timings:** our CPU timings come from x64 Python emulated on an ARM laptop, which is a worst case, not a deployment number. We quote no industry compute figures.
- **Latency:** the overnight closed-loop run assumes zero latency: a command applies on the step it is computed. A delayed-command run is planned.
- **Not deployable:** `lgbm_live` retrains six LightGBM models every 2 s, so it is an ablation only ("is the gain the rows or TabPFN?"). It stays out of the headline and the product diagram.
- **Deployable rivals:**
  - frozen `lgbm_all` plus the online conformal offset;
  - `tabpfn_live`, which updates its context but never its weights.
