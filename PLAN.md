# Foresight: the living plan

**Read this first.** This page is the single source of truth for strategy, roadmap and framework. It is updated whenever a decision changes, by either Claude session or by any of us, and mirrored to `main` straight away, so everyone reads the same plan. Detail lives in the linked documents; this page holds only the current state.

Last updated: Sat 21 Nov, 23:20 (Berlin). Code freeze Sun 12:30; submission Sun 16:30.

## 1. What we are building

A prediction-and-planning layer for a robot moving among people. Each person's position 1 to 2 s ahead is predicted as a **range**, by TabPFN reading a table of recent situations. The robot picks the move that keeps clear of those ranges.

## 2. The claims, and where each stands

| # | Claim | Status | Evidence / what decides it |
|---|---|---|---|
| A | **Safe out of the box in a new place:** TabPFN's ranges keep their promise in a scene it has never seen, while LightGBM's collapse | **Supported (offline, real data)** | Hotel → ETH joint box coverage: TabPFN 83% against LightGBM 46–62%. Error is level with LightGBM on all rows [RESULTS] |
| B | **Adapts in seconds without retraining:** reading the new scene's recent rows improves TabPFN fastest | **Promising, unconfirmed** | Pilot, Hotel → ETH: first 5 s, 0.425 against 0.536 m for frozen LightGBM. Pending the dense real-data run with person-level intervals, and **T3** (held-out crowd behaviour) |
| C | **Prediction-aware planning beats standard navigation** | **No support yet** | In the plaza, the standard methods are fast and collision-free (pilot). Decided by **T1**, then **T2** (a scene that doesn't let robots push through) |
| D | **Deployable** | Partly settled | Real-time tier = TabPFN-3.5 with 1 estimator (D-5, D-18). Licence: 3.5 weights are non-commercial, v2 is Apache 2.0. Latency on target-class hardware pending (Nooshin's Mac) |
| — | **When to use which** (honest framing) | Holds so far | Familiar data with many rows: LightGBM is as good. New place or new behaviour, few rows: TabPFN is safer. Both under-cover turns |

## 3. Framework

The modular sense–predict–plan pipeline. Each stage has its own pass/fail check [GATES], and the closed loop has the final say.

| Stage | Module | Status |
|---|---|---|
| S0 Data | `foresight/domains/` (crowd simulator, ETH/Hotel adapter), `INTERFACE.md` format | Plaza passes the prediction check. **Fails as a navigation test** (T2) |
| S1 State estimate | Kalman filter in `features.py` | Works. Velocity error not yet reported |
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
| If T1 overlaps | T2: difficulty settings, chosen from the baselines alone by a rule fixed in advance; regenerate the bank | knobs: CTO session (done, `d123543`); sweep: builder session | **running** since Sat 23:10: 27 cells x 20 fresh seeds (4000-4019), stop_go, cv m0.4, reactive m0.4, vo w1 (`tools/t2.ps1`) |
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
| `docs/explainer.html` | Picture-first explainer for a third person: problem, pipeline, where TabPFN works, planner, comparisons, results, testing, deployment. Open it in a browser |
| `docs/DECISIONS.md` | Every chosen value, its evidence, the small tests T1–T5 |
| `docs/GATES.md` | Pass/fail check per stage, fair-comparison protocol |
| `docs/DEPLOYMENT.md` | Refits, compute, model tiers, licence, Mac |
| `docs/VISUALS.md` | Trace format and what gets drawn |
| `RESULTS.md` | Numbers and figures (current only, per `media/MANIFEST.csv`) |
| `docs/CTO-REVIEW.md` | The first review (Sat 18:30). Partly superseded by the pages above |
| `SCENARIOS.md`, `INTERFACE.md` | Scene catalogue; data format |
| `TEAM-PLAN.md` | The original Saturday plan, kept for history |
