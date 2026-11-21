# Decisions: every chosen value, its reason, and the test behind it

Started Sat 21 Nov, about 22:45. **Rule:** no value goes into a run unless it has a row here, and a row is only "backed" when a test or a calculation supports it. "Assumed" rows are allowed, but they are stated as assumptions in the README and the deck. Small tests come before big runs. A finding that something doesn't work stops the work built on it, the same evening.

Status key: **backed** (test or calculation, cited) · **assumed** (stated, not tested) · **pending** (the test that settles it is in section 3).

## 1. Pilot finding, Sat 22:40: the overnight design was stopped early

Source: the first seeds of the overnight run, 1000–1012, analysed as a pilot. These seeds are pilot data from now on; confirmatory runs use fresh seeds.

1. **The plaza scene doesn't separate the planners.** Baselines on 100 seeds, all reaching the goal 97–100% of the time:

   | Planner | Time to goal | Collisions | Intrusion |
   |---|---|---|---|
   | Velocity obstacles | 19.6 s | 4% | 2.1 s |
   | Constant velocity, margin 0.4 | 21.9 s | 0% | 1.2 s |
   | Reactive, margin 0.4 | 21.8 s | 0% | 1.6 s |

   Learned planners take 23–33 s and lead only on intrusion and minimum gap. A bigger margin (constant velocity at 0.6: 23.8 s, intrusion 0.9 s) almost matches them. **"The pipeline beats standard navigation" has no support in this scene.** Likely cause (hypothesis, test T2): people dodge the robot at 2× strength, so nothing punishes a pushy robot.
2. **The adaptation test was in-distribution.** The bank (`data/crowd`, 80 episodes) already holds every regime and every switch pair; erratic appears in 33 episodes [checked: `meta.json`]. So a frozen model has seen every behaviour, and live context has nothing new to learn. That explains why `tabpfn_live` doesn't beat `tabpfn` or `lgbm` here. It is consistent with the failure-mode result: on familiar data, LightGBM is as good.
3. **`box_q = 0.05` freezes the robot.** In 10 seeds the band reached the goal 40% of the time and TabPFN 70%. It is a wasted setting.
4. **100 seeds was never derived** (it came from the first review, without a calculation). See D-14 for the calculated number.

## 2. Decision table

| # | Value | Used in | Why / evidence | Status |
|---|---|---|---|---|
| D-1 | Horizons 1 s and 2 s (simulator); 1.2, 2.0 and 4.8 s (ETH/Hotel) | features, planner | 2 s at about 1 m/s is about 2 m of look-ahead, the space the robot plans in. At 4.8 s even constant velocity is off by 0.86–1.44 m [RESULTS] and ranges get too wide to plan with. 4.8 s is reported only because it is the standard horizon in published work on ETH/UCY. ETH/Hotel use 1.2 s instead of 1 s because annotations come every 0.4 s, so the shortest horizon snaps to that grid | backed (partly); revisit only if T4 shows the planner needs other horizons |
| D-2 | Replan every 0.4 s | closed loop | Matches the ETH annotation step. Untested for latency | pending T4 (latency budget) |
| D-3 | Live refit every 2 s | `tabpfn_live` | A 2 s label exists only 2 s after its moment [checked: `t_known = t + h`], so faster refits add few new rows. The cadence itself is untested | pending T4 (2 / 5 / 10 s / never) |
| D-4 | Context 1,000 rows | TabPFN | Tier grid, Hotel → ETH: 1,000 rows is as good as or better than 2,000 for 3.5 (0.43–0.45 against 0.47–0.50 m), and v2 is weak at 250 [results/tier_grid.csv]. Single context sample | backed (1 sample); seed repeats running |
| D-5 | **Real time: TabPFN-3.5 with 1 estimator** (changed Sat 23:00 by D-18). The crowd closed loop may keep 3.5-Fast with 1 estimator | closed loop; product tier | Tier grid: Hotel → ETH and the crowd, every tier within 0.01–0.02 m; ETH → Hotel, Fast-1 0.385 against 3.5-1 0.300 m at 1,000 rows (D-18). So any real-data result or deck claim about the deployed model uses 3.5-1; the crowd loop keeps Fast-1 (in-distribution, equal there) | backed (1 context sample); pending tier_seeds |
| D-6 | Box = per-axis [q, 1 − q] quantiles, q ∈ {0.10, 0.15} | planner | q = 0.05 freezes (pilot, section 1.3). 0.10 and 0.15 reach 100% with lower intrusion | backed (10 seeds); confirm in T5 |
| D-7 | Planner margin 0.2 m (learned planners) | planner | Same for every learned planner, so a fair comparison. Its size is a design choice | assumed |
| D-8 | Short constant-velocity check at 0.4 s | planner | One replan period: too short for behaviour to change. Applied to every planner except `reactive` | assumed (reasoned) |
| D-9 | Live context: last 25 s; mix 60% recent / 30% kNN / 10% fill | context | Offline Hotel → ETH: mix 0.340 m against recent 0.353, global 0.371, kNN 0.385 [pilot adaptation]. Proportions taken from the plan, not tuned | backed for "mix beats single sources" (pilot, intervals pending); the proportions are assumed |
| D-10 | Sensor noise 0.1 m; Kalman q = 2, r = max(noise, 0.05) | features | Typical tracking noise for lidar or camera people trackers. The filter's velocity error was never reported (S1 gate) | assumed; S1 gate open |
| D-11 | One row every 0.5 s per agent | features | Rows 0.1 s apart are near-copies and waste context. Untested | assumed |
| D-12 | Plaza: 16 people, 40 s, 3 regimes | simulator | Passes S0: constant velocity 1.347 m against TabPFN 0.979 m at +2 s (27% lower) | backed for prediction; **fails as a navigation test** (section 1.1); pending T2 |
| D-13 | Online conformal: eta 0.02, target 80% joint | calibration layer | Reached 79–82% within 20 s in both directions for every model [RESULTS, calibration]. Eta not tuned | backed (works); eta assumed |
| D-14 | **40 confirmatory seeds**, fresh ones (3000–3039) | closed loop | Paired standard deviations from the pilot (time 2.3–2.8 s; intrusion 0.4–1.6 s). At 80% power and 5% two-sided, the seeds needed are below. 40 covers 2 s and 1 s everywhere and 0.5 s for most pairs. Extend if time allows | backed (calculation from 10–33 pilot seeds) |
| D-15 | LightGBM: 300 trees, 31 leaves, 10 rows per leaf | baselines | Not tuned. Small-context robustness run with 5 settings is running; the best is reported next to the default (that choice favours LightGBM) | pending |
| D-16 | Deployable rivals: `lgbm_all` plus online conformal, and `tabpfn_live` | product | `lgbm_live` retrains on the robot every 2 s, which is not normal practice; it is an ablation only | backed (judgment, DEPLOYMENT.md) |
| D-17 | API budget: 35M tokens/day (resets 00:00 UTC = 01:00 Berlin); stop at 30M; running total in results/api_usage.csv. Order by how decisive: (1) dense in-context test, Hotel → ETH locally with fast-1 (equal there, D-18), ETH → Hotel on the API with the default model, about 20M; (2) tier_seeds, about 5M; (3) about 5M reserve for T3/T5 follow-ups | API runs | The limit was hit at about 22:05 on Sat (dense in-context run). Hotel → ETH needs no API because fast-1 matches the default there | decided (CTO, Sat 22:15) |
| D-18 | Tier for ETH → Hotel: fast-1 is **not** equal there. At 1,000 context rows err 0.385 against 0.300 (v3.5, 1 estimator) and 0.308 (v3.5, 8); at 2,000 rows 0.368 against 0.289. More estimators close most of the gap (fast-8: 0.320 / 0.329). v3.5 with 1 estimator stays level with v3.5 with 8 in both directions | tier choice per direction | results/tier_grid_eth_to_hotel.csv, single context sample; seed repeats in tier_seeds (49 of 96 cells done before the API limit) | backed (1 sample): use v3.5 (1 estimator) rather than fast-1 where the context comes from a different place |
| D-19 | T1 result: the plaza does not separate planners. Go to T2 | closed loop | 100 common seeds (1000-1099), switch on, default crowd: lgbm q0.15 23.3 s / 0.80 s intrusion, lgbm q0.10 24.3 / 0.72; cv m0.4 21.9 / 1.18, cv m0.6 23.8 / 0.87, cv m0.8 25.4 / 0.60; vo w0.5 18.8 / 2.05; all 0-8% collisions. No learned config beats every baseline by > 2 s or > 1 s (D-14). `results/t1_gate.csv`, `media/fig_t1_gate.png` | decided (gate as pre-written) |

Seeds needed (D-14), at 80% power and 5% two-sided:

| Smallest difference worth claiming | Seeds |
|---|---|
| 2 s in time to goal | 11–16 |
| 1 s of intrusion | 2–21 |
| 0.5 s of intrusion | 7–84 |

## 3. Small tests, ordered by how much they could change the plan

Each test is small (minutes to about 45 min), has a gate written before it runs, and decides what comes next.

| Test | Question | Size | Gate (decided in advance) | If it fails |
|---|---|---|---|---|
| **T1** | Can the scene separate planners at all? | Frozen `lgbm` at q 0.10 and 0.15 on the 100 baseline seeds (about 10 s each), plus constant velocity and reactive at margins 0.8 and 1.0 | A learned planner sits on a better time-vs-intrusion curve than every baseline, by more than the D-14 smallest differences | Go to T2: the scene is too easy |
| **T2** | Make the scene a fair navigation test | Baselines only. Sweep robot repulsion (2.0 / 1.0 / 0.5), share of people who ignore the robot (0 / 0.2 / 0.4) and density (16 / 24 / 32) | **Rule, fixed now:** pick the mildest setting where the best baseline has ≥ 5% collisions **or** ≥ 3 s mean intrusion. Chosen from baselines alone, before any learned planner runs there. Then regenerate the bank in that setting and re-check S0 | If no setting qualifies, the closed loop stays a demo, and the claims rest on the offline and real-data results |
| **T3** | Does adaptation matter when behaviour is genuinely new? | Offline, local Fast-1: bank **without** erratic; test on calm → erratic switch episodes. Frozen TabPFN and frozen LightGBM against mix-context TabPFN and LightGBM on the same rows | Frozen models lose coverage after the switch, **and** live context recovers it, intervals separating | Live TabPFN gets no loop time. Claim 2 rests on Hotel → ETH (real data) only |
| **T4** | Loop details | 10 seeds each: refit cadence 2 / 5 / 10 s / never; latency budget 0 / 0.2 / 0.5 / 1.0 s (virtual time); online conformal in the loop | Keep the slowest refit and the largest latency that hold the gain | Pick the fallback values and write them here |
| **T5** | Confirmatory | Fresh seeds 3000–3039, in the scene from T2. Planners trimmed by T1–T4. Every seed has a switch if the claim lives there. Non-freezing caution levels. Pre-registration amended (dated) before launch | Pre-registered rule | — |

**Async (D-2, D-3), and why not real threads.** The simulator advances in **virtual time**. A plan computed at t takes effect at t + latency, and the robot keeps its previous command meanwhile. A refit started at t becomes usable at t + refit time. This is deterministic and gives the same result on any machine. Real threads would make results depend on the laptop and on chance. Laptop timings are emulated and contended, so instead of plugging them in we **sweep a latency budget** and report the latency at which the gain disappears. That number is what a hardware choice needs.

## 4. Data

- **Real:** ETH and Hotel, used for offline prediction, calibration and adaptation.
- **Simulated:** only `data/crowd`: 80 plaza episodes, 16 people, all regimes. Closed-loop test seeds are generated on the fly. The generator is ready:

  ```
  python -m foresight.domains.crowd --episodes N --out data/<name>
  ```

- **Missing:**
  - the knobs T2 needs (robot repulsion, people who ignore the robot), in `CrowdConfig`;
  - a held-out-regime bank (T3);
  - a stored test set;
  - a data card per dataset.
- **Data card** (`docs/DATA.md`), for every dataset:
  - episodes, people per frame, speed distribution;
  - share of stops and turns;
  - constant-velocity error at +2 s;
  - regime mix.

  `crowd.summary()` already computes most of these. It answers "what does our data look like" before anyone trusts a result on it.
