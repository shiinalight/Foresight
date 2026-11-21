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

**Limits:**
- Only two real scenes.
- The test sets are capped (CPU time).
- Single seed for this table; a 5-seed table with intervals that resample whole people is next.
- LightGBM is untuned; tuned and conformal versions are next.
- No obstacles from the scene maps.
- No robot in the real data.
