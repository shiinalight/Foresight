# CTO review: direction, architecture, validation

Written Sat 21 Nov, about 18:30 (laptop clock). Code freeze is Sun 12:30. Based on the repo at `3c1afc5` (branch `darshak/core`): `TEAM-PLAN.md`, `RESULTS.md`, `SCENARIOS.md`, `INTERFACE.md`, `foresight/`, `run_closed.py`, `results/`.

Each finding is marked **[verified]** (checked in the code or result files) or **[judgment]** (opinion).

## Verdict

- **Problem:** right.
- **Core method:** sound.
- **Headline:** over-claimed.
- **TabPFN differentiator:** not yet shown.

The plan is built as a ladder of features: prediction, then context, planning, a second domain and stretch items. The judges will score three claims, not features. Only one of the three has evidence today, and that one is fragile. The rest of the weekend should go into evidence for the other two, not into new scenarios.

| Claim we make | Evidence today | Gap |
|---|---|---|
| **1. TabPFN gives honest ranges in a place it has never seen** | Cross-scene ETH/Hotel table: about 81–86% coverage where 80% is promised; LightGBM 64–79% **[verified]** | One seed, about 300 rows per horizon, untuned LightGBM, coverage counted per axis rather than for the box (details below) |
| **2. It adapts to new behaviour in seconds, without retraining** | None. `context.select()` exists and `run_offline.py --retrieval` prints three lines, but no result is saved or plotted **[verified]** | This is the claim that sets TabPFN apart, and it has zero numbers. In the closed loop TabPFN is a frozen model: fitted once on 1,000 random rows (`fit_predictor`, `fit_with_cache`) **[verified]** |
| **3. A robot using these ranges is safer at equal speed** | `results/closed.csv`: 6 seeds, one margin, no TabPFN rows. cv and cv_band rows are duplicated (append mode), which skews the means **[verified]** | No headline plot yet |
| **4. "Any robot"** | One holonomic disc, one crowd simulator. The highway adapter was never written **[verified]** | A property of the interface, not a result. Demote it in the pitch |

## 1. Direction and problem

**Right problem [judgment].** A robot moving among people needs forecasts it can trust: a range with honest coverage, not only a point. The pain in real deployments is **distribution shift**. A robot is placed in a new building, city or crowd, and a predictor trained elsewhere degrades. The repo's own data already shows this:
- `fig_rows_place.png`: rows from the other scene make LightGBM *worse* after about 500 rows.
- Every overconfident range in the results is in the Hotel → ETH direction.

**Sharpen the framing.** Change "one layer for any robot" to:

> *A drop-in motion forecaster that gives a robot honest uncertainty in a place it has never seen, and corrects itself from the last few seconds of that place, with no training.*

This matches what TabPFN uniquely gives (in-context learning plus native quantiles), and it is testable with the code we have. "Any robot" stays as one architecture sentence: the interface makes it possible. It should not be a claim we try to prove this weekend.

**Not the problem we should claim to solve [judgment].** State-of-the-art accuracy on ETH/UCY: deep models (Trajectron++, the Social-GAN family) own that, and our protocol differs, so don't compare against published numbers. Our edge is calibration, adaptation and needing no training. Say so explicitly. It is a stronger position than a near-miss on average and final displacement error.

## 2. Architecture

### Right; keep it
- **Residual target** (deviation from constant velocity) in the agent's frame. This is the trick the past winners used (Prior Bahn, U-Bahn).
- **One data contract** (`INTERFACE.md`), with simulators kept out of the core. Clean, and it makes the real-data adapter almost free.
- **Causal context** (`t_known <= t`), forward-only Kalman filter, split by episode or scene.
- **One `predict_quantiles()`** behind every model, so baselines are like for like.
- **Planner:** vetoes candidates, then picks by progress; the quantile level is the caution dial. Simple and explainable. Right for a hackathon.

### Change
1. **Close the loop on context (most important).** The closed loop must call `context.select()` with recency from the live episode, refit every N seconds. Otherwise the system demonstrates nothing that a frozen LightGBM doesn't. How often to refit follows from **measured** refit latency, which nobody has measured yet. That is the first action (plan below).
   - On laptop CPU (x64 emulation, about 75 ms per row): 16 people × 2 horizons is about 32 query rows per replan, or about 2.4 s per replan.
   - A 1,000-row refit every 2 s fits only on a GPU.
2. **Coverage of the box, not per axis.** `evaluate.scores()` averages coverage over the two axes separately **[verified]**. The "80% range holds truth" column is therefore per-axis coverage of the 10–90% interval. The robot uses a 2D box: `QS_BOX` = 5–95% on each axis, in `viz.py` and the planner. Report **joint coverage of that exact box**, since that is the safety promise. Expect lower numbers. Every model is affected equally, so the ranking may hold.
3. **Check the path between horizons.** `planner.choose()` checks the robot only at h = 1 s and 2 s **[verified]**. A crossing at 1.5 s is missed. Horizon is already a feature, so add h ∈ {0.5, 1.0, 1.5, 2.0} to the query rows. It costs one batch, not new models.
4. **Fix the closed-loop log.** `closed.csv` is written in append mode, so re-runs duplicate rows. Write one file per run with a run id, or drop duplicates on (planner, seed, margin, quantile).
5. **Rule for the fallback.** When no candidate is safe the robot takes the one with the largest worst gap, which can mean moving into a box. Report how often the fallback fires per planner (`fallback_rate` is already logged). The constant band planner sits at 23–54% fallback, and freezes or fails on seed 1001 **[verified]**. This is the textbook *freezing robot problem*: uncertainty that is too wide or uncorrelated leaves no safe move (Trautman & Krause, 2010). It is our best argument for *sharp and* honest ranges over a fixed margin. Use it in the pitch.

### Not now (post-hackathon, see section 5)
Interaction-aware prediction (people react to the robot's candidate move), mixture over modes, and a joint 2D distribution.

## 3. Validation methodology

The numbers decide whether judges trust anything else. Fix these before adding features.

**Offline (claim 1)**
- **Seeds:** 5 context seeds (different 1,500-row samples) × both directions.
- **Intervals:** bootstrap 95% confidence intervals **by person** (all rows of one person resampled together), not by row: rows 0.4 s apart are near-copies.
- **Fair LightGBM:** small tuning on a validation split of the *training* scene: `num_leaves` ∈ {7, 15, 31}, `min_child_samples` ∈ {20, 50, 100}. 31 leaves and 10 rows per leaf on 1,500 rows likely overfits, which on its own makes ranges overconfident. A judge from Prior Labs will ask.
- **Conformal LightGBM** (split-conformal on a calibration slice of the training scene). This is what a careful engineer would ship, so it is the honest bar for "calibrated". If it matches TabPFN's coverage, TabPFN's remaining edge is **width at matched coverage**. Report that either way.
- **Lead metrics:**
  - pinball loss (a proper score, already computed);
  - **joint** box coverage;
  - width at matched coverage.

  The error gaps (0.276 against 0.291 m) are probably within noise. Don't lead with them.
- **Test set:** all test rows, on a GPU if possible. Otherwise keep the cap and say so.

**Adaptation (claim 2): the experiment to add**
1. **Real data, in-scene recency.** Test on ETH (the hard direction). The context starts as the Hotel bank only and fills with ETH rows as their targets complete (causal), at 0, 5, 10, 20, 40 and 60 s of in-scene data.
   - Plot error and joint coverage against seconds in the scene.
   - Compare three models:
     - TabPFN re-conditioned on the new context;
     - LightGBM refit on the same rows;
     - LightGBM frozen on the Hotel rows.
   - All the pieces exist (`eth_ucy.py`, `context.select`, `evaluate.py`). This is the headline result for the TabPFN track.
2. **Simulator, regime switch.** The same plot around `regime_switch_t` in the `crowd.py` episodes that have a switch. It is cheaper to run and has a known ground truth.
3. **Retrieval ablation.** Random, kNN only, recency only, and the 60/30/10 mix, on the same queries. Save it to `results/` and plot it. The prints we have today vanish.

**Closed loop (claim 3)**
- **Seeds:** at least 100 paired seeds (same crowd for every planner). Baselines run in 1–3 s per episode, so this is cheap. TabPFN costs what its replan latency costs.
- **Sweep the caution dial** (quantile level, or margin for the constant velocity planners), at least 3 settings per planner. Then plot **time to goal against safety** (minimum gap, near-miss rate, collision rate) as one curve per planner. A single operating point can't separate "safer" from "slower".
- **Report:**
  - freeze rate and fallback rate;
  - collisions with the robot moving (already logged).

  Collisions are rare, so minimum gap and near-misses carry the statistics.
- **Headline experiment:** a regime switch mid-episode, or a held-out regime. A frozen predictor should lose calibration there, and TabPFN with live recency should not. That one plot ties claims 2 and 3 together.
- **Decide the message now if TabPFN ties LightGBM.** Its ranges are wider, so it may be slower at equal margin. The fallback message: "range-aware planning beats a bigger margin (the freezing robot), and TabPFN keeps its ranges honest after the switch." That holds even at a tie on speed.

## 4. Plan to the 12:30 freeze, ranked

Rule: claims before features. Each item ends in a saved result file and a figure.

| # | Task | Box | Why |
|---|---|---|---|
| 0 | **Measure TabPFN latency:** refit (500 and 1,000 rows) plus one predict of 32 and 64 rows, on the GPU (if rented) and the laptop. Write it into `RESULTS.md` | 30 min, now | Sets the replan rate, the refit rate and the overnight budget. Every later choice depends on it |
| 1 | **Fix the closed-loop log** (dedupe, run id); add joint box coverage to `evaluate.py`; add intermediate horizons to the planner | 45 min | Cheap correctness fixes before any big run |
| 2 | **Adaptation on real data** (claim 2, experiment 1 above) | 2 h | The differentiator. It works on real data, with no simulator debate |
| 3 | **Live context in the closed loop:** `context.select()` with live recency, refit at the measured rate. Regime-switch seeds | 1.5 h | Makes the robot demo actually use the TabPFN property |
| 4 | **Overnight run:** 100+ paired seeds × planners {cv, cv_band, lgbm, tabpfn-frozen, tabpfn-live} × 3 caution levels. Order: TabPFN first | overnight | Claim 3 headline curve |
| 5 | **Offline rigour:** 5 seeds, bootstrap by person, tuned plus conformal LightGBM | 1.5 h (can run alongside 4 on the laptop) | Makes claim 1 survive questions |
| 6 | **Side-by-side clip** of one regime-switch seed: cv_band robot freezes, cv robot clips someone, TabPFN robot passes | 1 h, Sun morning | The visual Eddie asked for |
| 7 | **If, and only if, 0–6 are done:** Hotel video with predicted boxes projected through `H.txt` | 1.5 h | Strongest "works on reality" picture, at a fraction of the mixed street's cost |

**Cut:**
- the mixed-street scenario (2–3 h, and it proves "any robot", the weakest claim);
- the highway-env second domain;
- the live camera;
- the behaviour classifier;
- the learned risk model.

They can all appear on a "next" slide as one line each.

**Rolling submission:** the README is stale against `RESULTS.md` (it still describes the stand-in crowd as the domain). Refresh it after items 2 and 4.

## 5. Better, after the hackathon

Ordered by value for a real robot.

1. **Adaptive conformal on top of TabPFN.** Scale the box from recently completed in-scene rows with adaptive conformal inference (Gibbs & Candès, 2021). The roles split cleanly:
   - TabPFN gives the *shape* of the range (which way someone might swerve);
   - the conformal layer *guarantees* long-run coverage under drift.

   It is a few lines of code: one scalar per horizon, updated online.
2. **Interaction-aware planning.** Recompute each person's robot features for each candidate move and score every (candidate, person) row in **one batch**: 15 candidates × 16 people × 4 horizons is about 1,000 rows, one predict call (the trick from the Zelda-style project). People then react to the robot's plan, which removes the main known limit.
3. **Joint and multimodal ranges.** Two independent per-axis quantile models give a box, and at a fork a box is over-wide. Options:
   - predict the manoeuvre first (go, stop, left, right), then a range per mode, giving a mixture of boxes;
   - or sample from TabPFN's full predictive distribution and use ellipses.
4. **Benchmark properly.** All five ETH/UCY scenes (ETH, Hotel, Univ, Zara1, Zara2), leave-one-scene-out, the standard 3.2 s observation and 4.8 s prediction, with displacement errors *and* calibration. Compare against a constant velocity model (Schöller et al., 2020: a hard baseline on this data) and one published deep model. Only then can accuracy be claimed.
5. **Latency path to a real robot:**
   - TabPFN-3.5-Fast;
   - `fit_with_cache`, refitting the context on a background thread every 1–2 s;
   - predicting at 10 Hz on the cached context;
   - constant velocity plus a band as the safety fallback whenever the forecaster is late.
6. **Then the second domain.** Cars at a crossing, with the same core and a `kind` feature, once claims 1–3 hold on people.

## What not to change

The ladder discipline, the stop rules, the fixed Sunday afternoon for the video, README and slides, and the honesty of `RESULTS.md` (limits stated plainly). Those are already better than most hackathon teams manage.
