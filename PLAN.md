# Foresight: the living plan

**Read this first.** This page is the single source of truth for strategy, roadmap and framework. It is updated whenever a decision changes, by either Claude session or by any of us, and mirrored to `main` straight away, so everyone reads the same plan. Detail lives in the linked documents; this page holds only the current state.

Last updated: Sat 21 Nov, 23:20 (Berlin). Code freeze Sun 12:30; submission Sun 16:30.

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
| A | **Calibrated ranges in a new place:** TabPFN's boxes keep their nominal coverage in a scene it has never seen; untuned LightGBM's under-cover. **No point-error gain is claimed** | **Being re-checked on f4** | f1 numbers (Hotel → ETH joint coverage 83% vs 46–62%) are superseded: f1 leaked world heading (D-20) and used a weak constant velocity (D-26). On f4, tuned constant velocity is at or below every model's point error on the f1 test rows (red-team, DEFENSE.md #1). The rerun includes tuned LightGBM (m50:n100) and the interval score. Nominal joint coverage of the 5–95% box: 80–90% |
| B | **Adapts in seconds without retraining:** reading the new scene's own completed rows improves TabPFN | **Supported offline in simulation; real data not shown** | T3 (crossing crowd, erratic held out of the examples, switch at 20 s; 6 episodes, 636 query rows), after the switch, paired person bootstrap: split (cached bank + recent scene rows) minus frozen, both with the online offset. Interval score −1.29 [−1.70, −0.94], coverage +12.6 pts [+9.0, +16.3], median error −0.057 m [−0.092, −0.021]. Limits: no variant reaches the 80–90% target after the switch (best 68%); split costs ~3 s per moment against 0.27 s. On real data (Hotel → ETH, n = 137) nothing beat constant velocity. The earlier "first 5 s" pilot is withdrawn (cold start) |
| C | **Prediction-aware planning beats standard navigation** | **Reported as a stress test (pre-registered verdict)** | In the chosen crowd (repulsion 2, 20% ignore the robot, 24 people), even the open-loop oracle (true future positions) is under 2 s faster than the best baseline at matched safety, so the gate failed and no learned planner can show claim C there (amendment 1). The robot's minimum travel time is likely within about a second of the fastest baseline (being checked). Exploratory, disclosed as added after the gate: safety at matched time, on fresh seeds. The oracle's 0% vs constant velocity's 15% contacts on the 20 selection seeds is only a hint |
| D | **Deployable** | Partly settled | The loop runs TabPFN-3.5-Fast with 1 estimator. The proposed deployed tier is 3.5 with 1 estimator (D-5, D-18). Licence: 3.5 weights are non-commercial; v2 is Apache 2.0. Mac and GPU timings pending |
| — | **TabPFN learns more per example row** | Supported in the simulator (f4) | Crossing crowd, +2 s: TabPFN-Fast on 2,000 rows (0.646 m) ≈ LightGBM on 20,000 rows (0.634 m); LightGBM on all 73k rows is best (0.606 m). results/context_size.csv; API part pending |
| — | **When to use which** | Holds so far | Many familiar rows: LightGBM is as good or better. Few rows, or a new place: TabPFN gives better-calibrated ranges. Both under-cover turns |

## 3. Framework

The modular sense–predict–plan pipeline. Each stage has its own pass/fail check [GATES], and the closed loop has the final say.

| Stage | Module | Status |
|---|---|---|
| S0 Data | `foresight/domains/` (crowd simulator, ETH/Hotel adapter), `INTERFACE.md` format | Crossing crowd passes on f4, narrowly (D-27: LightGBM 18.7% better than constant velocity, TabPFN-Fast on 1k rows 7.5%). Default plaza fails as a navigation test (T1) |
| S1 State estimate | Kalman filter in `features.py` | Passes (D-26): filter tuned on training data; constant velocity at 4.8 s within 3–14% of the published value (Schöller et al. 2020), depending on the window selection |
| S2 Table | `features.py`: agent-frame rows, target = deviation from constant velocity | Done. Leakage and parity tests pass |
| S3 Context | `context.py`: recent, similar and fill rows | Done. Mix beats single sources (pilot) |
| S4 Forecaster | `models.py`: TabPFN and baselines behind one interface | Done. Model tiers in progress |
| S5 Calibration | `conformal.py`: online offset | Done offline. **Not yet in the closed loop** (T4) |
| S6 Occupancy | `occupancy.py`: one function for the metric and the drawing | Function in. Metric pending |
| S7 Planner | `planner.py`, plus the baselines: stop and go, reactive, velocity obstacles | Done. Pre-registered comparison |
| S8 System | latency, fallback, virtual-time delay | Delay model pending (T4) |
| Trace and visuals | `--log` trace, `tools/export_3d.py`, three.js viewer (Nooshin) | Trace done and reproducible. Viewer in progress |

## 4. Roadmap

Rule: small tests with gates written in advance come first; the big run comes after them. A test that fails stops the work built on it.

| When | Item | Owner | Status |
|---|---|---|---|
| Sat night | T1: can the plaza separate the planners? | builder session | **done: no.** 100 common seeds: lgbm q0.15 23.3 s / 0.80 s intrusion sits on the cv-margin curve (cv m0.6 23.8 s / 0.87 s); vo w0.5 fastest (18.8 s, 2.05 s). No learned config beats every baseline by 2 s or 1 s (`results/t1_gate.csv`, `media/fig_t1_gate.png`) |
| Sat night | T3: does live context recover when behaviour is new (crowd without erratic in the bank)? | builder session | running |
| Sat night | Real-data adaptation, Hotel → ETH (local) | builder session | running |
| After 01:00 | Real-data adaptation, ETH → Hotel (API, about 20M credits), then the model-size repeats (about 5M) | builder session | queued |
| If T1 overlaps | T2: difficulty settings, chosen from the baselines alone by a rule fixed in advance; regenerate the bank | knobs: CTO session; sweep: builder session | **moved to the crossing crowd (D-24).** Default-scene sweep stopped. Cross bank built (data/crowd_cross), S0 passes (D-25: LightGBM 39.5% better than CV). Cross sweep running from Sun 00:20 |
| Sat night | Robustness tests: simulator (done, 11 tests), core pipeline, closed loop | CTO session + 2 subagents | running |
| Sat night | Data cards for every dataset (`docs/DATA.md`) | CTO subagent | running |
| Sun early | T4: refit interval, latency budget and calibration in the loop, 10 seeds each | builder session | waiting |
| Sun early | T5: confirmatory run, 40 fresh seeds (3000–3039), dated amendment to the pre-registration | builder session | waiting |
| Sun morning | Mac: MPS check and timings (`tools/check_mps.py`, `tools/latency.py`) | Nooshin | ready |
| Sun morning | Data cards (`docs/DATA.md`); three.js viewer against the export schema | Nooshin | open |
| Sun morning | Four-panel video and six-panel pipeline frame from traces, on the seed chosen by rule | builder session | after T1–T3 |
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
- **API budget:** stop at 30M credits a day; running total in `results/api_usage.csv`.

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
