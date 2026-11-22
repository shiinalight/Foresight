# Deployment realism: refits, compute, model sizes, Mac

Written Sat 21 Nov, about 21:45. Each point is marked **[checked]** (verified in the code, the installed package or the Prior Labs docs, with the source named) or **[judgment]**.

## 1. What "live" means, and whether it is realistic

| Planner | What happens every refit | Is it training? | Realistic on a robot? |
|---|---|---|---|
| `tabpfn` | Nothing: context fixed at start | — | Yes (a frozen model) |
| `tabpfn_live` | Swaps the context rows for this scene's latest completed rows plus similar bank rows. With `fit_with_cache`, one forward pass caches the context | **No.** The weights never change [checked: `fit` only stores the rows and caches them] | Yes. A sliding-window memory update, the same pattern as sliding-window online Gaussian-process regression [judgment] |
| `lgbm` / `lgbm_all` | Nothing: trained once on bank rows | — | Yes (standard: train offline, deploy frozen) |
| `lgbm_live` | **Retrains** 6 boosted-tree models (3 quantiles × 2 axes, 300 trees each) every 2 s on the robot | **Yes** | **No.** Retraining models on the robot every few seconds is not normal practice [judgment]. It costs 1.8 s per plan on this laptop [checked: overnight log] |

**Decision:**
- `lgbm_live` stays in tonight's run only as an **ablation**. It answers "is the gain the new rows, or TabPFN?". It is labelled "not a deployable design" and is left out of the product diagram and headline.
- The **deployable LightGBM** is `lgbm_all`, frozen, plus an **online conformal offset**: one number per horizon, nudged up or down from completed rows. That is cheap and close to standard practice. It is also the strongest rival offline: boxes about 25% narrower after 20 s [checked: RESULTS.md, calibration layer]. It is missing from the closed loop today (`conformal.py` is offline only), so it is added there for every model.

**Refit cadence:**
- A 2 s-horizon label only exists 2 s after the moment it describes, so refitting faster than every 1–2 s adds almost nothing [checked: `t_known = t + h`].
- Sweep refit at 2, 5 and 10 s and never, on about 20 regime-switch seeds. Keep the **slowest** cadence that holds the gain.
- In a real system the refit runs in a background thread while predictions run on the cached context, so the planner never waits for a refit [judgment].

**Simulator honesty.** The simulator waits for the planner, so every overnight result shows what the *information* is worth, not real-time behaviour. Measured plan times are close to or above the 0.4 s replan period:
- frozen TabPFN: 199–516 ms;
- live TabPFN: 861 ms;
- live LightGBM: 1.8 s.

A **delayed-command run** on a subset of seeds closes this gap: each command takes effect after the measured plan time, and a refreshed context only becomes usable after the measured refit time. Say plainly in RESULTS.md that the overnight run assumes zero latency.

## 2. Compute: what the model is, against what robots run

**Model sizes** [checked: parameters counted from the installed checkpoints' safetensors headers, float32]:

| Checkpoint | Parameters | File |
|---|---|---|
| TabPFN-3.5 | 219 M | 836 MB |
| TabPFN-3.5-Fast | 83.5 M | 319 MB |

**Cost model** [checked: package code and docs]:
- `fit` is one pass over the context rows (cached with `fit_with_cache`).
- Each prediction attends to that cached context.
- Cost grows with the number of context rows and with `n_estimators` (ensemble members).
- Predicting rows one at a time is about as many times slower as there are rows, so all people and horizons go in **one batched call** (we already do this).

**Our timings are a worst case:**
- the laptop runs x64 PyTorch under emulation, CPU only (`torch 2.11.0+cpu`);
- the TabPFN docs call CPU execution slow.

They are not deployment evidence. Target-class hardware is a CUDA GPU or an Apple-silicon GPU (MPS), so the first realistic numbers come from Nooshin's Mac (section 4).

**Against industry [judgment].** Robots and cars run their prediction networks on embedded GPUs at about 10 Hz. A model of 80–220 M parameters with a cached context of about 1,000 rows plausibly fits that envelope on a GPU. We will not quote industry figures until we measure ours on a GPU or MPS.

## 3. Model tiers: the right size for each job

Each tier runs where its latency fits:

| Tier | Model | Job | Where |
|---|---|---|---|
| Real-time predictor | **TabPFN-3.5 with 1 estimator** (D-18; tier_seeds was dropped (D-29); the closed loop runs TabPFN-3.5-Fast with 1 estimator), cached, about 1,000 context rows. Fast with 1 estimator fails when the context comes from another place (ETH → Hotel 0.385 against 0.300 m); it stays an option only in-distribution (crowd: every tier within 0.01 m) | Every person, every replan (2.5–10 Hz) | On the robot |
| Second opinion | TabPFN-3.5 (8 estimators) | Only the people whose range is wide or whose box just failed: the uncertainty-triggered step. Runs off the control path, in the background | On the robot (GPU) |
| Offline | TabPFN-3.5 or the hosted Thinking/Plus models (API) | Building and checking the context bank, analysis, evaluation | Cloud. **Never in the control loop** (network delay and dropouts) [judgment] |
| Safety floor | Constant velocity plus band, and the 0.4 s hard check | Takes over whenever the predictor is late or fails | On the robot, independent of TabPFN |

**Correction (red-team, DEFENSE.md #8):** v2 was *better* than 3.5 on Hotel → ETH (0.449 vs 0.474 m, 3 seeds), not "matching". The closed loop runs 3.5-Fast with 1 estimator; every result states its tier. The context-size test (results/context_size.csv) shows TabPFN still improving at 2,000 rows in the simulator, so the real-time context is set per setting at the 02:00 decision.

**Evidence (tier grid, 21 Nov; results/tier_grid*.csv, single context sample, seed repeats in tier_seeds):** across 3.5 / Fast × 1-8 estimators × 250-2,000 rows, accuracy barely changes on Hotel → ETH and on the crowd; on ETH → Hotel Fast-1 is clearly worse from 1,000 rows on, while 3.5-1 equals 3.5-8 in both directions. v2 was better than 3.5 on Hotel→ETH (0.449 vs 0.474 m) and weaker on ETH→Hotel; see the correction above. The cost of 3.5-1 against Fast-1 is the deck trade-off; `tools/latency.py` times both.

**Original evidence request:** the earlier "Fast with 1 estimator equals the default" check (F3) compared Fast with 1 estimator against the default with 8, which mixes two changes. Run a clean grid:
- models: 3.5 and Fast;
- estimators: 1, 2, 4, 8;
- context rows: 250, 500, 1,000, 2,000.

Report pinball, joint coverage, width and fit/predict time. Accuracy does not depend on the hardware, so it runs on the hosted API: about 128 calls, about 1.3 M credits. Timing is measured locally on the Mac.

**Other checkpoints** (v2, 2.5, 2.6 and 3; all selectable in the installed package [checked: `ModelVersion`]) need weight downloads, which needs your approval.

**Licence [checked: Prior Labs docs; confirm on the licence page before the deck].**
- The 3.5 and 3.5-Fast weights (and 2.5, 2.6 and 3) are **non-commercial**; a product needs a commercial licence or the API.
- The v2 weights are Apache 2.0 with attribution.
- So "deployable" in the pitch means: **the 3.5 family under a commercial licence**, or **v2** (if it passes our gates) for open use.

## 4. Mac (Nooshin): MPS

**[checked: installed package]**
- TabPFN picks the device automatically: CUDA, then MPS, then CPU. On Apple silicon it can route attention through MLX to save memory, if `mlx` is installed.
- It ships a workaround for an MPS numerical bug.
- `TABPFN_MPS_MEMORY_FRACTION` defaults to 0.7.
- The docs recommend a recent PyTorch for Apple silicon.

Fixed in bd1e240 (device auto: cuda > mps > cpu). **Bug in our code:** `foresight/core/models.py:83` picks only `"cuda"` or `"cpu"`, so a Mac silently runs on CPU. It should pass `"auto"`.
- **Hold this edit** until tonight's baseline processes have finished. They import the code from disk when each one starts.

**Nooshin's checklist:**
1. Install a recent PyTorch with MPS, then `pip install tabpfn mlx`.
2. Run `python tools/latency.py` with device `auto`. These become our target-class timings.
3. Parity test: the same context and rows on `mps` and `cpu` must give quantiles that agree within a small tolerance. Don't assume they do.
4. Never mix devices within one paired closed-loop set.

## 5. Encoding the table the way TabPFN expects

**[checked: Prior Labs docs]**
- Missing values are handled natively; no imputation needed.
- Don't scale or one-hot encode. We don't.
- Declare categorical columns (`categorical_features_indices` or pandas category).
- Batch the predictions.
- Text and date columns get special handling (not relevant to us).

**Findings for our table:**
1. **Missing neighbour, robot or obstacle is coded as 99, with velocity 0.** That is a made-up value, and NaN is native.
   - Run it as an **A/B test**: Hotel → ETH and the crowd, 5 seeds, TabPFN and LightGBM.
   - It also touches the kNN bank (sklearn rejects NaN), the `robot_dist < 90` test in `failure_modes.py`, and `check_shift.py`.
   - Switching would invalidate every result so far, so **adopt it only if the gain exceeds the seed-to-seed spread**, and only if there is time to rerun. Otherwise report it as a finding.
2. **`kind`** (person, car) becomes a declared categorical feature once a scene has several agent kinds.
3. **Two axes, two models.** Multi-output regression isn't stated as supported. We keep one model per axis, which is correct.

## 6. Ranked against the 12:30 freeze

| # | Item | Cost | Fits? |
|---|---|---|---|
| 1 | Online conformal in the closed loop for every model (`lgbm_all_cp` is the deployable rival); run on 40 seeds | 1 h code + 1–2 h run | Yes, first thing in the morning |
| 2 | Model-tier grid on the API (accuracy) | 30 min + 1.3 M credits | Yes, can start tonight (API, no laptop CPU) |
| 3 | Mac: `"auto"` device fix, Nooshin's latency and MPS parity | 15 min + Nooshin | Yes, after the baselines finish |
| 4 | Refit cadence sweep (2, 5, 10 s, never) on 20 switch seeds | 1.5 h run | Likely |
| 5 | Delayed-command run on a seed subset | 1 h code + 1 h run | Likely |
| 6 | NaN encoding A/B (offline, API) | 45 min | Yes as a finding; **adopting it does not fit** (full rerun) |
| 7 | Other checkpoints (v2 etc.) | Downloads need approval, 1 h | Only with approval |
| — | Mode mixture, street scene | — | **Do not fit**; listed as next steps |
