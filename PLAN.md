# Foresight: the living plan

**Read this first.** This page is the single source of truth for strategy, roadmap and framework. It is updated whenever a decision changes, by either Claude session or by any of us, and mirrored to `main` straight away, so everyone reads the same plan. Detail lives in the linked documents; this page holds only the current state.

Last updated: Sun 22 Nov, 02:20 (Berlin). Code freeze Sun 12:30; submission Sun 16:30.

## 0. Orchestration (live; the CTO session owns this section)

Times are Berlin. Freeze Sun 12:30; submit by 16:00 (hard deadline 16:30).

| Window | Evidence (builder session) | TabPFN usage and ablations (CTO + subagents) | Story, defence, visuals (CTO + subagents) |
|---|---|---|---|
| Sat 23:30 → 01:00 | T2 in the crossing crowd (f4); claim A rerun on f4 (local) | Context size and retrieval, local part (LightGBM, TabPFN fast-1 up to 2k rows) | Step-by-step walkthrough page; red-team review and Q&A; presentation storyline and slide plan; industry-practice brief |
| 01:00 → 02:00 | Calibration both directions (API); oracle gate in the T2 setting; T3 in the crossing crowd | Context size on the API (TabPFN-3.5, up to 20k rows; retrieved 5k) | Fold verified references and wording fixes into explainer and docs |
| **02:00 decision** | | **Real-time TabPFN setup** (context size, retrieval, model tier) and whether the scene-tag / history-length ablations run | |
| 02:00 → 03:30 | T4: calibration in the loop, refresh interval, latency budget, 10 crowds each | Ablations, if approved at 02:00 (offline) | Explainer v4 with f4 numbers |
| **03:30** | Dated amendment to the pre-registration; **launch T5** (40 fresh crowds, all with a behaviour change) | | |
| 03:30 → 08:00 | T5 runs | Real-data in-context and adaptation reruns on f4 (API) | Defence dossier v2 from red-team findings |
| 08:00 → 10:30 | T5 analysis per the pre-registered rule; final figures on f4; traces of the showcase crowd; four-panel video render | Walkthrough rerun on f4 | RESULTS.md final; explainer and walkthrough final; README |
| 10:30 → 12:30 | Promote code and results to `main` | | Deck; 3-minute video script final |
| 12:30 → 16:00 | Freeze | | Record the video (people); slides; submit |

**Changes since the plan was written:**
- **T5 is done and analysed** (20 paired seeds, 3000–3019; `RESULTS.md`, "Closed loop T5 results"). It launched at 01:12, not 03:30, so the T5 rows of the schedule above are history. A 40-seed extension (same arms and rules, `tools/t5_extend.py`) was declared at 02:20 (3226e4f), before any of its results; the 20-seed analysis stays the confirmatory result. Showcase seed 3013; demo render: `media/demo/seed3013.gif`.
- **TabPFN API credits are monthly, not daily.** About 15M remain until 1 Dec. The budget is in DECISIONS: context size 0.3M, calibration ≤1.5M, real-data in-context ≤4M, the tier repeats dropped, ≥5M in reserve.
- **On-camera recording moves to 11:00–11:30, before the freeze** (`docs/PRESENTATION.md`). Numbers go only in the voice-over, so a late result means re-cutting the video, not re-filming.
- **The Mac kit** (`docs/MAC_SETUP.md`, `scripts/mac/`) lets Nooshin rebuild the data, check that it matches ours, and measure TabPFN on her M3's GPU. Her smoke-test numbers decide which runs move to the Mac.

**Standing rules for the orchestration:**
- **Gates:** every gate result (S0, T2 pick, oracle, T3, T4) comes to the CTO session before the next step starts.
- **Decisions:** every decision gets a DECISIONS row with its evidence.
- **Docs and status:** `PLAN.md` and the explainer are updated when a claim's status changes.
- **Subagents:** create only their own new files; bugs in shared code go to the file's owner.

## 1. What we are building

A prediction-and-planning layer for a robot moving among people. Each person's position 1 to 2 s ahead is predicted as a **range**, by TabPFN reading a table of recent situations. The robot picks the move that keeps clear of those ranges.

## 2. The claims, and where each stands

| # | Claim | Status | Evidence / what decides it |
|---|---|---|---|
| A | **Calibrated ranges in a new place:** TabPFN's boxes keep their nominal coverage best in a scene it has never seen. **No point-error gain is claimed** | **Supported, direction-dependent (f4, final)** | Real pedestrians, +2 s, 5 context seeds, paired person bootstrap (results/claim_a_f4.csv, results/claim_a_boot_f4.csv). Hotel → ETH: TabPFN clearly better than LightGBM on all rows (interval score 90% −0.30 [−0.43, −0.18], box miss −17 pts [−21, −13]) and than tuned LightGBM; joint coverage 75% (nominal 80–90%). ETH → Hotel: level with LightGBM on all rows on interval score (+0.01 [−0.06, +0.07]) but fewer box misses (−11.5 pts); better than tuned LightGBM; coverage 86%. Point error level (TabPFN 0.403 / 0.244 m, LightGBM all rows 0.405 / 0.253, tuned constant velocity 0.428 / 0.254) |
| B | **Adapts in seconds without retraining:** reading the new scene's own completed rows improves TabPFN | **Supported offline and in the closed loop, in simulation; small and direction-dependent on real data** | T3 (crossing crowd, erratic held out of the examples, switch at 20 s; 6 episodes, 636 query rows), after the switch, paired person bootstrap: split (cached bank + recent scene rows) minus frozen, both with the online offset. Interval score −1.29 [−1.70, −0.94], coverage +12.6 pts [+9.0, +16.3], median error −0.057 m [−0.092, −0.021]. Limits: no variant reaches the 80–90% target after the switch (best 68%); split costs ~3 s per moment against 0.27 s. On real data (Hotel → ETH, n = 137) nothing beat constant velocity. The earlier "first 5 s" pilot is withdrawn (cold start). **In the loop** (T5 exploratory b, q 0.15, +2 s, after the switch, 20 paired seeds): split vs frozen TabPFN coverage 59.3% vs 49.3% (+10.0 pts [+8.4, +11.6]), interval score 5.03 vs 6.34 (−1.31 [−1.55, −1.08]), median error −0.057 m [−0.094, −0.021]; vs lgbm_all +18.0 pts and −2.00. No model reaches 80% after the switch; before it, split is slightly worse (interval score +0.18). Real-data check (`tools/realdata_claimb.py`): on real pedestrians (from 5 s, split vs frozen, paired person bootstrap, `results/realdata_claimb.csv`): small gain, interval score −0.045 [−0.080, −0.009] pooled (1,540 rows, 544 people), coverage level (+1.3 pts [−0.5, +3.1]), level with the offset on both; direction-dependent (ETH→Hotel −0.046 [−0.086, −0.005], narrower boxes at equal coverage; Hotel→ETH −0.045 [−0.093, +0.007], level). Against LightGBM: +17.1 pts coverage, interval score −0.254. The real scenes have no sudden behaviour switch, which is where the simulator's +10 pts came from |
| C | **Prediction-aware planning beats standard navigation** | **Not supported in the closed loop (T5, run as a stress test after the oracle gate)** | In the chosen crowd (repulsion 2, 20% ignore the robot, 24 people), even the open-loop oracle (true future positions) is under 2 s faster than the best baseline at matched safety, so the gate failed and T5 became a stress test (D-33). The fastest baseline (vo w0.5) runs only 1.50 s above the 17.74 s time floor. **T5** (20 paired seeds, 3000–3019): tabpfn_split beats no baseline; all four fail Holm (p = 1.0). It is slower at comparable safety: 4.3 s slower than cv m0.4 at equal min gap, 3.0 s at equal collision rate; cv m0.2 and vo w0.5/w1/w2 are 3.9–6.2 s faster, so it does not beat them either. Reason: not calibration. Split, frozen TabPFN and LightGBM all take about 26 s at q 0.15 on the 20 confirmatory seeds (split − frozen −0.12 s [−1.35, +1.25]; split − LightGBM +0.33 s [−1.29, +1.99]); the box planner's caution setting sets the speed. 40-seed update (exploratory, `results/t5_primary_40.csv`): split and frozen TabPFN still drive the same (+0.03 s [−1.08, +1.16]), but LightGBM's narrower, less honest boxes (41% vs 59% coverage after the switch) make it 1.24 s faster [+0.17, +2.34]; every learned planner stays 3–5 s slower than cv and vo at comparable safety. Exploratory dial sweep on fresh seeds: `docs/EXPLORATORY_DIALS.md`. Exploratory (a): the open-loop oracle beats no baseline on every point and both axes at matched time; it buys clearance (min gap 0.22 m vs 0.09 m for cv m0.2), but collision-rate gains mostly stay below the margin at 20 seeds |
| D | **Deployable** | Partly settled | The loop runs TabPFN-3.5-Fast with 1 estimator. The proposed deployed tier is 3.5 with 1 estimator (D-5, D-18). Licence: 3.5 weights are non-commercial; v2 is Apache 2.0. GPU timing measured on an NVIDIA L4 (`docs/DEPLOYMENT.md`, "Measured on a GPU"): TabPFN 71 ms per replan, full loop 225 ms per plan. Mac (MPS) timing not measured yet |
| — | **TabPFN learns more per example row** | **Supported in the simulator (f4)** | Crossing crowd, +2 s, 3 seeds (results/context_size.csv): at equal rows TabPFN beats LightGBM by 0.04–0.06 m at every size. TabPFN-3.5 (hosted, 8 estimators) on 20k random rows: 0.593 m, pinball 0.096, coverage 87%; LightGBM on all 73k rows: 0.606 m, 0.101, 83%, at about the same box width. Similarity-based row selection adds nothing. One simulated scene (in-distribution) |
| — | **Intent chain:** a TabPFN classifier predicts each person's next manoeuvre (straight / stop / turn left / turn right); per-manoeuvre TabPFN range models give the 2 s range, mixed by the classifier's probabilities | **Supported offline on real data; not in simulation** | Gate fixed in `tools/intent_layer.py` before any result (`results/intent_layer.log`, `media/fig_intent_layer.png`). Real (Hotel → ETH): balanced accuracy 0.49 (chance 0.25); adopted: on turning people interval score 3.59 → 3.33 (−0.263 [−0.481, −0.075]; 73 rows, 37 people), overall level (−0.022 [−0.057, +0.009]), joint coverage +5.0 pts overall and +16.9 on turns, median error +0.007 m. Simulated crowd: balanced accuracy 0.30 (people pick turns at random); not adopted (overall interval score worse). Caveat: turn recall is 0 on real data; the turn gain is calibrated widening from the mixture, not a turn forecast. TabPFN's class probabilities are better calibrated than LightGBM's (log loss 0.24 vs 0.71 real, 0.81 vs 1.53 sim). Limits: 37 turning people, 3 context seeds on real data, 2 in simulation, offline only |
| — | **When to use which** | Holds so far | Many familiar rows: LightGBM is as good or better. Few rows, or a new place: TabPFN gives better-calibrated ranges. Both under-cover turns |

## 3. Framework

The modular sense–predict–plan pipeline. Each stage has its own pass/fail check [GATES], and the closed loop has the final say.

| Stage | Module | Status |
|---|---|---|
| S0 Data | `foresight/domains/` (crowd simulator, ETH/Hotel adapter), `INTERFACE.md` format | Crossing crowd passes on f4, narrowly (D-27: LightGBM 18.7% better than constant velocity, TabPFN-Fast on 1k rows 7.5%). Default plaza fails as a navigation test (T1) |
| S1 State estimate | Kalman filter in `features.py` | Passes (D-26): filter tuned on training data; constant velocity at 4.8 s within 3–14% of the published value (Schöller et al. 2020), depending on the window selection |
| S2 Table | `features.py`: agent-frame rows, target = deviation from constant velocity | Done. Leakage and parity tests pass |
| S3 Context | `context.py`: recent, similar and fill rows | Final: frozen 2,000-row bank at natural move frequency + recent-only scene model (≤250 rows), quantiles averaged ("split"). Similarity (kNN) selection dropped: no gain (D-30, D-31) |
| S4 Forecaster | `models.py`: TabPFN and baselines behind one interface | Done. Model tiers in progress |
| S5 Calibration | `conformal.py`: online offset | In the loop: per-batch online offset (T4 gate); the per-row rule froze the robot. Offline: split/online conformal results in RESULTS |
| S6 Occupancy | `occupancy.py`: one function for the metric and the drawing | Function in. Metric pending |
| S7 Planner | `planner.py`, plus the baselines: stop and go, reactive, velocity obstacles | Done. Pre-registered comparison |
| S8 System | latency, fallback, virtual-time delay | Delay model pending (T4) |
| Trace and visuals | `--log` trace, `tools/export_3d.py`, three.js viewer (Nooshin) | Trace done and reproducible. Viewer in progress |

## 4. Roadmap

Rule: small tests with gates written in advance come first; the big run comes after them. A test that fails stops the work built on it.

| When | Item | Owner | Status |
|---|---|---|---|
| Sat night | T1: can the plaza separate the planners? | builder session | **done: no.** 100 common seeds: lgbm q0.15 23.3 s / 0.80 s intrusion sits on the cv-margin curve (cv m0.6 23.8 s / 0.87 s); vo w0.5 fastest (18.8 s, 2.05 s). No learned config beats every baseline by 2 s or 1 s (`results/t1_gate.csv`, `media/fig_t1_gate.png`) |
| Sat night | T3: does live context recover when behaviour is new (crowd without erratic in the bank)? | builder session | done: offline Setting B of tools/live_variants.py (D-30) |
| Sat night | Real-data adaptation, Hotel → ETH (local) | builder session | running |
| After 01:00 | Real-data adaptation, ETH → Hotel (API, about 20M credits), then the model-size repeats (about 5M) | builder session | queued |
| If T1 overlaps | T2: difficulty settings, chosen from the baselines alone by a rule fixed in advance; regenerate the bank | knobs: CTO session; sweep: builder session | **moved to the crossing crowd (D-24).** Default-scene sweep stopped. Cross bank built (data/crowd_cross), S0 passes (D-25: LightGBM 39.5% better than CV). Cross sweep running from Sat 23:28 |
| Sat night | Robustness tests: simulator (done, 11 tests), core pipeline, closed loop | CTO session + 2 subagents | running |
| Sat night | Data cards for every dataset (`docs/DATA.md`) | CTO subagent | running |
| Sun early | T4: refit interval, latency budget and calibration in the loop, 10 seeds each | builder session | done (T4 addendum parts 1-4) |
| Sun early | T5: confirmatory run, 40 fresh seeds (3000–3039), dated amendment to the pre-registration | builder session | done (results in RESULTS) |
| Sun morning | Mac: MPS check and timings (`tools/check_mps.py`, `tools/latency.py`) | Nooshin | ready |
| Sun morning | Data cards (`docs/DATA.md`); three.js viewer against the export schema | Nooshin | open |
| Sun morning | Four-panel video and six-panel pipeline frame from traces, on the seed chosen by rule | builder session | showcase seed 3013; demo render: `media/demo/seed3013.gif` |
| 12:30 → 16:00 | Freeze; README, slides, video; submit | everyone | fixed |

## 5. Working rules

- **Who edits what (Sat night):**
  - The CTO session owns `foresight/domains/crowd.py` and `tests/test_robustness_*.py`.
  - The builder session owns the loop, planner, models, context, features, calibration, occupancy, `tools/` and `RESULTS.md`.
  - **Nooshin:** `crowd.py` now has two difficulty settings, `robot_repulsion` and `ignore_robot` (`d123543`). Their defaults leave every episode exactly as before. Pull before editing `crowd.py`, so your next push doesn't collide.
  - A bug found in someone else's file goes to its owner as a failing test with a proposed fix.

- **Branches:** work goes on `darshak/core`. Plan documents (this page and `docs/`) are mirrored to `main` immediately with `tools/sync_docs.ps1`. Code and results reach `main` only once final and reviewed, by fast-forward, never by force. Before promoting, merge `origin/main` into `darshak/core`.
- **Decisions:** no value enters a run without a row in `docs/DECISIONS.md` (backed, assumed or pending).
- **Fairness:** same seeds, same features, same speed for every planner; baselines swept over their own settings; claims only at matched safety; analysis rules written before results.
- **API budget:** limits are monthly per account (D-29); a second account was added on Sun 22 Nov; every call is logged in results/api_usage.csv.

## 6. Document map

| Document | What it holds |
|---|---|
| `PLAN.md` (this page) | Current strategy, roadmap, framework, rules |
| `docs/PRESENTATION.md` | Story, timed video script with variants for each result, slide plan, README structure, list of visuals |
| `docs/MAC_SETUP.md` | Running everything on an Apple-silicon Mac (setup, smoke test, runs) |
| `docs/explainer.html` | Picture-first explainer for a third person: problem, pipeline, where TabPFN works, planner, comparisons, results, testing, deployment. Open it in a browser |
| `docs/DECISIONS.md` | Every chosen value, its evidence, the small tests T1–T5 |
| `docs/GATES.md` | Pass/fail check per stage, fair-comparison protocol |
| `docs/DEPLOYMENT.md` | Refits, compute, model tiers, licence, Mac |
| `docs/VISUALS.md` | Trace format and what gets drawn |
| `RESULTS.md` | Numbers and figures (current only, per `media/MANIFEST.csv`) |
| `docs/CTO-REVIEW.md` | The first review (Sat 18:30). Partly superseded by the pages above |
| `SCENARIOS.md`, `INTERFACE.md` | Scene catalogue; data format |
| `TEAM-PLAN.md` | The original Saturday plan, kept for history |
