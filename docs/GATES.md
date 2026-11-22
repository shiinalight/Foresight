# Gates: what "done and correct" means for each part

Written Sat 21 Nov, about 19:00. Code freeze is Sun 12:30. Goal: a complete product within the hackathon, with nothing left for later. Every part has to pass its gate before the next part builds on it.

This file sits alongside `docs/CTO-REVIEW.md`, which explains the reasoning. This one is the checklist the builder works from.

## 0. Blocking: fix before any long run

| # | Problem (checked in the code) | Fix |
|---|---|---|
| B1 | `results/closed.csv` holds rows with 15 fields and rows with 17 fields under a 15-column header, and the new `horizons` and `run_id` columns make it worse. `pd.read_csv`, and so `load_closed()` and the deck, will raise an error | Write one file per run, `results/closed/<run_id>.csv`, and `load_closed()` concatenates them. Move the old file to `closed_v1.csv` |
| B2 | `KEY` in `run_closed.py` lacks `horizons`, `context`, `people` and the bank, so runs with different settings overwrite each other | Add them to `KEY` |
| B3 | `run_adapt.py` skips moments whose context has fewer than 20 rows, so `recent` is scored on later moments than the other strategies. The comparison is unfair, and worst in the early bins, the ones that matter | Score every strategy on the **same** queries. When `recent` has too few rows, its prediction is constant velocity plus no range (and is counted as such) |
| B4 | In the closed loop LightGBM sees only the 1,000-row sample; offline it also gets all rows | Add `lgbm_all` to the closed loop |
| B5 | The closed loop has no TabPFN with **live context**: it is fitted once and frozen, so the adaptation claim never reaches the robot | Add a `tabpfn_live` planner: `context.select()` with this episode's completed rows, refitted every N s (N from the `tools/latency.py` result). Add `lgbm_live`, refitted on the same rows, for fairness |
| B6 | Live recency rows must be labelled from **observed** (noisy, filtered) positions, as a real robot would see them. Ground truth is only for scoring | Build the live rows from the same noisy observations the features use |
| B7 | The bank was generated with the simulator's straight-line robot policy, but in the loop the planner drives the robot, so the robot features shift | Compare the `robot_dist` and `robot_closing` distributions between bank rows and live queries. If they differ, regenerate the bank with a randomised robot policy |
| B8 | Train/serve parity is untested | Test: at zero noise, `build_rows(query_only=True)` on the 13-step window gives the same features as the offline row for the same agent and moment |
| B9 | On ETH/Hotel, `meta.bounds` is the bounding box of **all** recorded positions, so `obst_*` encodes where the camera's field of view ends, computed from future positions | Set `bounds` to none for real data (keep it for drawing only), then rerun the offline table |

## 1. The pipeline, stage by stage

The design (simulator → state estimate → table → context retrieval → TabPFN → calibration → occupancy and boxes → planner) is a **modular sense–predict–plan stack**. That is the standard, correct architecture for robots: each stage can be tested and swapped on its own.

Two consequences:
- **Each stage has its own gate metric, and the system has one end-to-end metric.** A stage that looks good on its own can still hurt the end result, so the closed loop has the final say.
- **Per-person coverage is not safety for the whole scene.** With 16 people and 2 horizons, even perfect 80% boxes leave a high chance that at least one person is outside their box at a given moment. Claim safety only from closed-loop numbers, never from coverage alone.

**On the word "agentic":** use it only if uncertainty-triggered re-retrieval (item 9 in section 3) is built and measured. Without it, call the system a pipeline. Prior Labs judges will notice the difference.

| Stage | What it does | Gate (must pass) | Where |
|---|---|---|---|
| S0 Data | Simulator and real data in the `INTERFACE.md` format | `schema.validate()` passes. In each simulated scene, constant velocity loses at least 15% at 2 s to the best learned model (otherwise the scene teaches nothing) | `foresight/domains/` |
| S1 State estimate | Noise plus a forward-only Kalman filter | Filtered velocity error against truth reported at 0.1 m noise. Parity test B8 | `features.py` |
| S2 Table | One row per (agent, moment, horizon); target = deviation from constant velocity, in the agent's frame | Causality test (no feature or context row uses `t > now`). Split by episode or scene. B9 fixed | `features.py`, `context.py` |
| S3 Context retrieval | Recent rows, similar rows, a stratified fill | Ablation on the **same** queries (B3): `mix` no worse than `global` on error, and joint coverage within 5 points of nominal | `run_adapt.py` |
| S4 Forecaster | TabPFN quantiles per axis | Section 2 protocol, 5 seeds, confidence intervals | `evaluate.py` |
| S5 Calibration | Adaptive conformal scale on the box, from completed rows in the scene (Gibbs & Candès, 2021) | Joint coverage of the box within ±3 points of nominal after 20 s in a new scene, **applied to every model** | new `foresight/core/conformal.py` |
| S6 Occupancy | Per-person predicted distributions summed on a grid → probability that each cell is occupied at +1 s and +2 s | Brier score and area under the ROC curve against true future occupancy, against constant velocity plus band and LightGBM | new `foresight/core/occupancy.py` |
| S7 Planner | Candidates vetoed against boxes (and occupancy), then chosen by progress | Section 2 closed-loop protocol | `planner.py`, `run_closed.py` |
| S8 System | Latency, fallback | Replan time per step reported. A deadline fallback to constant velocity plus band when TabPFN is late | `run_closed.py` |

## 2. Fair comparison protocol

**Baselines.** Every learned model gets the same features, the same rows and the same split.

| Baseline | Why it is there |
|---|---|
| Constant velocity | The physics floor. A strong baseline on pedestrians (Schöller et al., 2020) |
| Constant velocity plus a fixed band (per horizon) | Answers "why not just a bigger margin?" |
| LightGBM quantile, same rows as TabPFN | Like for like |
| LightGBM, **tuned** on a validation split of the training scene (`num_leaves` 7/15/31, `min_child_samples` 20/50/100) | Otherwise "TabPFN beat an overfit baseline" |
| LightGBM, all rows | "What if the classical model just gets all the data" |
| LightGBM plus split-conformal | The careful engineer's answer to calibration. If it matches TabPFN's coverage, compare **width at matched coverage** |
| Refit LightGBM on the same live context | Fair partner for TabPFN's re-conditioning in the adaptation test |

**Metrics, in this order:**
1. pinball loss (a proper score);
2. **joint** box coverage (also report per-axis, labelled as such);
3. width at matched coverage;
4. median-error displacement at each horizon;
5. latency.

**Statistics:**
- 5 seeds for context sampling;
- 95% bootstrap intervals that **resample whole people**, not rows;
- the full test set where compute allows, and say so where capped.

**Closed loop:**
- at least 100 paired seeds, the same crowd for every planner;
  - replaced by the power calculation in D-14: 20 confirmatory seeds (T5, 3000–3019), 40 as an update (3000–3039);
- sweep the caution dial: 3 to 5 levels per planner (`box_q` for learned models, margin for constant velocity);
- plot time to goal against minimum gap and collision rate, one curve per planner;
- report freeze rate, fallback rate, and all collisions as well as moving-robot collisions (a person walking into a stopped robot still counts in the real world);
- the short 0.4 s constant-velocity safety check applies to every planner equally. Say so.

**Not allowed:**
- comparing against published ETH/UCY numbers (different protocol);
- changing settings after seeing the test set;
- skipping queries for one method only.

## 3. Everything, in order (nothing deferred)

Ordered by the claim each item feeds. Box = estimated hours. "Who": B = builder session, N = Nooshin, GPU = overnight machine.

| # | Item | Claim it feeds | Box | Who | Gate | Reach by freeze |
|---|---|---|---|---|---|---|
| 1 | Blocking fixes B1–B9 | all | 1.0 | B | Tests pass; `load_closed()` reads every run | sure |
| 2 | Adaptation: Hotel → ETH in-scene, plus the simulator regime switch | adapts | 1.5 + runs | B | S3; figure + CSV saved | sure |
| 3 | **Failure-mode breakdown** (section 4) | where baselines fail | 0.5 | B | Table plus one figure | sure |
| 4 | Conformal layer (S5) on every model; tuned and conformal LightGBM; 5-seed rerun of the offline table | honest ranges | 1.5 | B | S4, S5 | likely |
| 5 | `tabpfn_live` and `lgbm_live` in the closed loop; bank regenerated if B7 fails | adapts → safer | 2.0 | B | Live context is used in every replan (logged) | likely |
| 6 | **Overnight run:** at least 100 seeds × {cv, cv_band, lgbm, lgbm_all, lgbm_live, tabpfn, tabpfn_live} × caution levels, half the seeds with a regime switch | safer | night | GPU | Section 2 closed-loop protocol | likely |
| 7 | Occupancy (S6) and a heatmap overlay | what TabPFN is used for | 1.5 | B | S6 metrics | likely |
| 8 | Per-candidate prediction: recompute robot features for each candidate move and score them all in one batch, so people react to the plan | safer | 1.5 | B | Closed loop with it against without | possible |
| 9 | Uncertainty-triggered re-retrieval (the "agentic" step) | adapts | 1.0 | B | Ranges narrow without coverage dropping below nominal | possible |
| 10 | Doorway scene (has a regime switch) | adapts, any agent | 2.0 | N, starting now | S0 gate | likely |
| 11 | Mode-first mixture (go, stop, left, right), then a range per mode | honest ranges | 1.5 | B | Width at matched coverage better than one box | unlikely |
| 12 | Street crossing (cars and people) | any agent | 3.0 | N, after 10 | S0 gate | unlikely |
| 13 | All five ETH/UCY scenes, leave-one-scene-out. **Needs your approval** to download the UCY files; check the raw format first | honest ranges | 1.5 | B | Section 2 protocol | possible |
| 14 | Hotel video with predicted boxes projected (`H.txt`). **Needs download approval** (50 MB) | visual on real data | 1.5 | B | Boxes line up with people in the frame | possible |
| 15 | Latency path: TabPFN-3.5-Fast, `fit_with_cache`, background refit, deadline fallback (S8) | real robot | 1.0 | B | Replan within budget | possible |

**Rules:**
- **No new scene work in the builder session** until items 2, 3 and 6 have saved results. Nooshin builds scenes in parallel.
- At 22:00, decide what runs overnight from the `tools/latency.py` numbers.
- Items marked unlikely are your call to drop or keep. They are listed so nothing is silently left out.

## 4. Showing where baselines fail and how TabPFN fixes it

The judges must see three things:
1. where the simple methods break;
2. which TabPFN property fixes each break;
3. what that changes for the robot.

Each row below is one figure or one panel, made from data we already produce.

| Where the simple method breaks | Evidence (slice of existing results) | Why it breaks | TabPFN property that fixes it | Robot-level effect |
|---|---|---|---|---|
| **Constant velocity: stops, turns, avoidance.** It knows no behaviour | Error by `mode` label (straight, stop, turn left, turn right) and by distance to the nearest neighbour and the robot, for every model | Physics only | Learns the deviation from constant velocity from context rows | Fewer near-misses at corners and when people stop |
| **Constant velocity plus band: one width for everyone** | Band width against actual error by mode: too wide for people walking straight, too narrow for people turning | Same range for every situation | A range for each row, conditioned on the situation | No freezing: the robot gets through gaps the band closes (freeze and fallback rates) |
| **LightGBM: overconfident in a new place** | Joint coverage, Hotel → ETH: TabPFN against LightGBM (tuned, all rows, conformal) | Fitted to the training place; untuned trees overfit | Native quantiles from the prior; recalibrated from the scene's own rows (S5) | Boxes that hold, so collisions stay low after a scene change |
| **LightGBM: needs many rows, and the right ones** | Error against number of rows (same place against other place): `fig_rows_place` | More rows from the wrong place hurt | Strong with few rows; context picked per query | Useful within seconds of entering a scene |
| **Every frozen model: behaviour changes** | Error and coverage against seconds since the regime switch | Trained before the change | Re-conditioned on the last seconds, no training | The robot stays safe after the crowd turns hurried |

**Where TabPFN is used in the product (one diagram in the deck):**
- **Trajectory ranges:** per person, at +1 s and +2 s, from quantiles (S4).
- **Occupancy map:** the same distributions summed on a grid (S6).
- **Context retrieval:** decides *which* rows TabPFN reads (S3).
- **The agentic step, if built:** re-asks with a narrower context when uncertain (item 9).

Each block in the diagram is labelled with its gate number from section 1.

**The main video:**
- **Four panels, same seed, regime switch at the midpoint:**
  - constant velocity;
  - constant velocity plus band;
  - LightGBM;
  - TabPFN live.
- **Live counters in each panel:** collisions, minimum gap, time to goal, box coverage so far.
- **Occupancy heatmap** under the TabPFN panel.

**One "pipeline at work" frame for the deck:**
- the raw tracks;
- the table rows for one person;
- the context TabPFN chose;
- the predicted box;
- the planner's vetoed and chosen moves.

## 5. Update `RESULTS.md` after item 4

- Rename "80% range holds truth" to "per-axis coverage" and add the joint column.
- Replace the single-seed table with the 5-seed table with intervals.
- Rewrite the claims from the new numbers. If the error gaps fall inside the intervals, say "comparable error, better calibration".
