# How the TabPFN-3.5 hackathon winners used TabPFN

Source: https://platform.priorlabs.ai/hackathon-3.5 (submissions closed 6 Oct 2026). Repos read on 21 Nov 2026 (code, not only READMEs) for the projects marked **code**; README only for those marked **readme**. Numbers are the authors' own; none were re-run.

## The projects in one table

| Project | One row = | Target | TabPFN call | The trick that made it work |
|---|---|---|---|---|
| **1st sEMG gestures** (code/docs) | one 488 ms window of 24 EMG channels | gesture (8 classes) | hosted 3.5-Plus, 8 estimators, **215,009 context rows** x 144 features | Classic hand-crafted window features (WL, ZC, SSC, IAV, VAR, WAMP per channel) + StandardScaler on train only; **evaluated on 4 unseen people** (84.7% vs LDA 82.1%, SVM 77.8%); same features for LDA and TabPFN so the comparison is fair |
| **2nd duckpfn** (code) | (61 robot observations, **joint id**) | that joint's action | local 3.5-**Fast**, `fit_mode="fit_with_cache"`, n_estimators=1 | **One pooled model for 14 outputs** by repeating each state 14x with a joint-id column (4,096 transitions -> 57,344 rows), targets standardised per joint; 50 Hz closed-loop walking; **swap part of the context to adapt** (weaker motors: 0/3 -> 3/3 walks, no weights changed); choose a posture by the **lower 10% quantile** of predicted stability (56 vs 81 falls in 1,024 pushes); fall-warning classifier; images via PCA or raw pixels; text column native in Plus |
| **TabPFN plays Breakout** (code) | game state (10 numbers) | paddle x target | 3.5, fresh model each frame, n_estimators=1 | **Per-frame retrieval of 256 context rows** from a 23,917-row teacher bank: hard gate on matching velocity signs, then weighted standardised nearest neighbours (ball 1.0, paddle 0.25), plus +/-8 neighbouring frames of each anchor in the same episode; imitates a simple rule (cleared the level once; failed on a new start) |
| **Prior Bahn** (code, via agent) | one stop event of one train run | delay change since last known | 3.5 regressor, quantiles | **Retrieval ladder per query (<=5k rows)**: same train today, same train past 14 days, other trains at these stations in the last 60 min, same clock slot on earlier days, balanced random fill; **predict the change from the last observation**; **replay history at the query's clock** with a known-at-now filter (no leakage); quantiles -> CDF -> P(connection holds), calibration checked per bucket (beat XGBoost trained on 2.4M rows) |
| **Berlin U-Bahn** (code, via agent) | one station-hour | surge yes/no and z-score | 3.5 classifier + regressor, 10k context | **Target = deviation from a robust per-station x hour normal** (MAD z-score), not raw flow; **stratified context**: all rare event/closure rows forced in, positives enriched to 15%; what-if = baseline and counterfactual rows in one batch; LLM agent only explains typed tool outputs; alarm by top-2% budget. TabPFN on 10k rows beat XGBoost on the same 10k, XGBoost needed 30x rows to win |
| **Zelda-style game** (code, via agent) | one past situation | class / minutes | REST API, quantiles | Fit once, predict many; **score every candidate action as a row of one batch** (9 options, 1 call); **risk tolerance -> quantile level**: pick the most aggressive plan whose q_level still meets the deadline; prefetch + cache + rule fallback |
| **agentry** (code, via agent) | one agent step | failure mode, final cost | client, 8 estimators | Running-state features (streaks, repetition vs last 3 steps, cumulative cost); grouped split by session; **irreversible action needs probability AND hard evidence**. Caution: TabPFN lost on ROC-AUC here; "memory" never entered the context |
| **PRRT dose** (readme) | organ / lesion | absorbed dose | 3.5, native quantiles | Native quantiles gave **80/94% coverage with no calibration step** (CatBoost needed split-conformal); decision = "decisive" vs "**measure instead**" from P(dose > threshold) |
| **Superconductivity** (readme) | material | critical temperature | 3.5 | Predictive distribution drives **active learning (expected improvement)**: 79.6 top-1% finds in 200 experiments vs 35.6 greedy XGBoost; swapping in a GP lost most of the gain; 95% intervals covered 95.8%, 38% narrower than conformal XGBoost |
| **EUDR land cover** (readme) | **one polygon (mean embedding)** | land class (23) | 3.5, 1.6 s fit | **Collapse near-duplicate pixels to one row per independent observation** (318 rows); Google satellite embeddings (64 numbers) as features; leave-one-polygon-out; then label 2.46 billion pixels; probabilities as a confidence map |
| **Tavolio** (readme) | spreadsheet row | blank cell | 3.5-Fast exported to INT4 ONNX (~44 MB) | Runs in the browser; approximate SHAP for explanations |
| **TabLint** (readme) | table cell | that column from the others | 3.5, out-of-fold | Out-of-fold so the row is never in its own context; flag a value by where it falls in the predicted distribution (10-90%); explain likely slips |
| **RevAgent** (readme) | subscriber | churn | 3.5 | 200 rows: LLM ranking 0.535 AUC vs TabPFN 0.842 -> "LLM agents are bad with tables; TabPFN is their tabular brain" |
| **Goran Nakerst, "Driving a car with a tabular foundation model"** (blog, cited by Breakout) | 180 LiDAR distances | steer left / straight / right | TabPFN v3 classifier, 500 rows | Imitated a follow-the-gap rule from 500 examples and drove a new cave step by step; works because the teacher is a simple threshold rule |

## First-principle strategies that worked

### 1. Build the context per query (retrieval), don't dump everything
TabPFN only knows what is in its context, and the context is limited. The best projects chose it for each prediction:
- **Breakout:** hard gate first (same motion direction), then weighted nearest neighbours in standardised feature space, then add temporal neighbours of each match. 256 rows per frame.
- **Prior Bahn:** a priority ladder: same entity's live past > same place right now > same place, same time slot, earlier days > balanced random fill.
- **U-Bahn:** force every rare-situation row in, enrich the positives (15%).
- **duckpfn:** the opposite case: when the context fits (57k rows with cache on GPU), use it all, but **edit** it to change behaviour.

*Why:* in-context learning is local; similar rows weigh most. Retrieval turns a huge bank into the relevant 256 to 5,000 rows, cuts latency and makes rare situations visible.

### 2. Predict the change, against a baseline
- Prior Bahn: target = delay change since the last observation (biggest gain on running trains).
- U-Bahn: target = deviation from a robust normal profile (z-score).
- duckpfn: optional `delta` target (action minus previous action).

*Why:* the model spends its capacity on what is hard, not on re-learning the obvious level. For motion: predict displacement relative to constant velocity, not absolute position.

### 3. One row per independent observation; split by the independent unit
EUDR collapsed pixels to polygon means; sEMG split by person; agentry by session; Prior Bahn by week with a known-at-now filter. *Why:* near-duplicate rows inflate scores and waste context.

### 4. Use the distribution, not the mean
- duckpfn: pick the action by the **lower tail** of predicted stability -> fewer falls.
- Game: **risk level -> quantile**; most aggressive plan that still meets the deadline at that quantile.
- Prior Bahn: quantiles -> CDF -> P(success) of each step of a plan; calibration per bucket.
- PRRT: P(above threshold) -> "decisive" or "measure instead".
- Superconductivity: expected improvement for active learning.
*Why:* the calibrated distribution is TabPFN's distinctive output; every top project turned it into a decision.

### 5. Many outputs in one model: an id column
duckpfn stacked 14 joints as rows with a joint-id feature (one model, one cache). *Why:* one fit, shared structure between outputs, simpler serving. Alternative: one cached model per output (they implemented both).

### 6. Make it live: cache, batch, prefetch, fall back
`fit_mode="fit_with_cache"` + n_estimators=1 + GPU for 50 Hz (duckpfn); fresh small context per frame (Breakout: slow, 4 h of compute for a 2-min replay); score all candidate actions in one batch (game, U-Bahn what-if); keep a rule-based fallback.

### 7. Imitate a good teacher first
Breakout and the LiDAR car imitated a simple rule; duckpfn imitated a trained walking policy. TabPFN learns threshold-like rules from few hundred examples. *Why:* behaviour cloning is a clean table (state -> action) with a known ceiling.

### 8. Prove it honestly
Every winner: a strong literature or tuned baseline (LDA/SVM, XGBoost on full data, conformal CatBoost), the same features for both, held-out people/places/weeks, calibration checks, and limits stated plainly. The sEMG winner's polish (provenance, reproducible snapshot, demo UI with video) is part of why it won.

### 9. LLM as the narrator, TabPFN as the numbers
U-Bahn, RevAgent, agentry: TabPFN behind typed tools; code decides; the LLM explains. Never let the LLM compute from the table.

## What did not work (their own notes)
- Breakout: success only from the teacher's start state; planning teacher, brick inputs, big fixed contexts and downsampled frames all failed.
- duckpfn: camera-only walking fell after a short walk; raw-pixel decisions took ~24 s each; forecasts uncalibrated.
- agentry: TabPFN lost on ROC-AUC and cost regression; labels leaked from features.
- Prior Bahn: upper quantiles optimistic (q95 held 91%); steps treated as independent.
- Superconductivity: intervals undercovered on a held-out material family (59.8%): out-of-distribution stays hard.

## What this means for a driving / traffic project
- **Retrieval context per agent per moment:** same vehicle's own past > vehicles in the same lane segment in the last seconds > same manoeuvre type (merge, cut-in) from the bank > balanced fill. Gate on hard compatibility first (direction, lane relation), then nearest neighbours.
- **Target = displacement relative to constant velocity**, in the agent's lane frame.
- **Ego-planning rule from the game and duckpfn:** score every candidate ego manoeuvre as a row in one batch; choose by the lower tail of the predicted gap/TTC (risk level -> quantile).
- **Calibration plot** (coverage per quantile) and **data-efficiency curve** (TabPFN vs XGBoost vs rows) as the proof.
- Self-driving brainstorm with dataset options and a build plan: `av-tabpfn-brainstorm.md` (same folder).
