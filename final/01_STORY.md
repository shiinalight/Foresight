> **Copy of [`docs/STORY.md`](../docs/STORY.md)**, made by `tools/build_final.py` on 2026-11-22 13:51 (after commit `e788ade`). Edit the source and rerun the script; do not edit this copy.

# Foresight: the story we submit (single source of truth)

Written Sun 22 Nov 2026, about 09:00 Berlin, on `darshak/core` at `2ab7e46`. Freeze 12:30, submit by 16:00.

This page settles what we claim, what we sell, and from what angle. The video, the README top, the deck and the Q&A take their wording from here. It supersedes the narrative parts of `docs/archive/PRESENTATION.md` (sections 1, 2 and 3), `docs/archive/CEO-HANDOVER.md` (sections 3 and 5) and the "Claims we can make" list in `docs/DEFENSE.md`. Those pages stay as history. Every number here names its source; `RESULTS.md` holds the full tables.

---

## 1. The conclusion

**The sentence.** *The robot's knowledge is a table of examples. It adapts to new behaviour, and to a new body, by swapping rows, with no training step.*

**The angle: sell the forecaster, not the navigator.**
- Everything that held is about prediction: adapting when behaviour changes, learning more per example, and carrying over to a second body. TabPFN is the best forecaster we tested per example and under change, and level with models fed far more data (say "level with", never "far better").
- TabPFN worked at every layer we tried (ranges, intent, the robot's move); it pays off most as the forecaster.
- **One real limit: the planner** (claim C). Our planner turns each range into a yes/no box, so at the same safety it arrives 3–4 s after standard planners. It is simple on purpose; the next step weighs how likely each spot is.
- So the product is the layer that tells a robot *where everyone will be in the next one to two seconds, and how sure it is*. It sits under whatever planner the customer already runs.
- Re-targeting it to a new site, a new crowd behaviour or a new body is a data operation (swap the example table), not a training project.

**The product in one paragraph.** Foresight is an uncertainty layer for robots that move among people or vehicles. Each agent at each moment becomes one row of a table. TabPFN reads a frozen bank of example rows plus the last seconds of the current scene and returns a calibrated range for every agent at +1 and +2 s, in one forward pass. Moving to a new site means adding that site's rows; a sudden change in behaviour is absorbed because the scene's own recent rows enter the table as soon as their outcomes are known; a new body (we showed cars) needs that body's example bank and the same core.

**Who it is for.**
- Teams deploying robots into new sites (delivery, service, hospital, warehouse), where the first hours at a site have little data and behaviour shifts with time of day.
- Their stacks often have no pedestrian prediction at all: stock ROS 2 Nav2 plans around current obstacle positions plus a margin (`docs/INDUSTRY.md`, section 1). Deployed sidewalk robots publicly describe slow-down and stop rules (section 2).

**Why TabPFN, and not a trained model.**
- Adapting is swapping rows: no training loop on the robot, weights never change (`foresight/core/models.py`).
- Calibrated quantiles come out of one call, so the planner gets a range, not just a point.
- It learns more per example, which matters exactly when a site is new and rows are few.

**The buyer's guide (say it as our honesty, and as the product boundary).**
- Few examples, or behaviour that just changed: TabPFN.
- Plenty of familiar data: a well-fitted LightGBM is as good, and cheaper.
- The next lever for driving is the planner, not the forecaster: one that uses the whole range instead of a box veto.

**One limit and a scope line.**
- **The limit** is the planner (above). Say "where it stops", and say the next step in the same breath.
- **Not limits:** the five-scene tie is data efficiency (proof 3: the same result with 15x fewer rows); each body needing its own examples is true of every method, and ours needs only a new bank, no retraining (the product point).
- **Scope line, said once:** the robot loop and the cars are simulated (the pedestrian forecasts are also tested on five real scenes); cars are offline, with no closed loop; TabPFN-3.5 weights are non-commercial, so a product needs the commercial licence or the Apache-licensed v2 weights.

**Fallback (held, not triggered).** If a review finds an error in proof 1 or proof 2, the same material is told as a case study ("we put a tabular foundation model into a robot loop, under rules written first; here is where it helps"). Nothing so far has triggered it (`docs/archive/CEO-HANDOVER.md`, section 5a).

---

## 2. Claims ledger

Status: **holds** (passed its pre-written rule), **narrowed** (holds only in a weaker form), **failed**, **exploratory** (rule written before the run, but not part of the pre-registration), **offline** (not in the robot loop).

### 2a. The three proofs and the chain

| # | Claim | Status | Number | Source | Say | Never say |
|---|---|---|---|---|---|---|
| P1 | **Adapts when behaviour changes, no retraining** (in the robot loop) | holds, exploratory (b) of the T5 amendment; simulated | After the crowd turns erratic, split TabPFN 59.3% inside its 2 s box vs LightGBM 41.3%: +18.0 pts [+16.7, +19.3]. Reading the scene alone adds +10.0 [+8.4, +11.6] over the same TabPFN without it. Ahead of LightGBM in all three crowds: +6.6, +18.0, +19.7 pts | `results/inloop_claimb.csv`, `docs/GENERALISATION.md`, `media/fig_t5.png` | "Reading the scene's last seconds brought the ranges closer to their promise: 18 points more coverage than LightGBM, 10 of them from the scene alone. Still short of the target." | "Kept the ranges honest", "calibrated" (59% against an 81% target is not calibrated); crediting all 18 points to scene reading |
| P1-fair | **Is it just because LightGBM never saw the scene?** | exploratory, offline, post-hoc check run today | LightGBM refit on rows that include this scene (`lgbm` variant, the full-mix rows): split TabPFN still +12.3 pts 80% coverage [+7.7, +17.0], pinball −0.015 [−0.022, −0.009], error level (+0.021 m [−0.009, +0.048]), simulated, after the switch, 166 people. On real ETH (Hotel bank): +10.2 pts [+2.5, +18.0], pinball level | `results/live_variants.csv` (Setting B t ≥ 20 s; Setting A), person bootstrap; script in this session's scratchpad, not in the repo | Q&A only: "Offline, a LightGBM refit on rows from the same scene still covered about 12 points less." | Putting it in the video; calling it pre-registered. In the loop, `lgbm_live` was dropped from T5 for cost (D-30) |
| P2 | **Same core, another body: cars** | holds by its rule (H1, H2), narrowed in wording; simulated, offline | H1: TabPFN on 1,500 rows covers 85.6% (interval score 3.57) vs LightGBM on 100,440 rows: pre-set setting −0.49 [−0.66, −0.34]; well-fitted settings −0.18 [−0.36, −0.01] and −0.18 [−0.37, +0.00], covering 69–72%. H2: after traffic turns aggressive, scene rows add +14.2 pts coverage (76.7% vs 62.5%), interval score −0.98 [−1.41, −0.56] | `docs/HIGHWAY.md`, `results/highway_h1.csv`, `results/highway_h2.csv`, `media/fig_highway.png` | "The same core on simulated highway traffic: 1,500 rows match a LightGBM trained on 67 times more, with more honest ranges. When traffic turns aggressive, reading the scene adds 14 points of coverage." | "Beats LightGBM on 100,000 rows"; "same code, unchanged" (an adapter was added: say "same core"); "drives a car"; "any robot" |
| P3 | **Learns more per example** (real pedestrians) | holds against equal rows; tie against 15x rows | Five real scenes (ETH, Hotel, Zara1, Zara2, Univ), each held out once: vs LightGBM on the same 1,500 rows, interval score −0.130 [−0.154, −0.107]; vs LightGBM on all rows of the other four scenes (~15x), −0.008 [−0.023, +0.007], a tie; vs constant velocity plus a fixed band, −0.365 [−0.405, −0.328] pooled, lower in every scene | `docs/LOSO5.md`, `results/loso5_rows.csv`, `media/fig_loso5.png`, clip `media/demo/scenes5.gif` | "On five real pedestrian scenes, TabPFN beats the fixed-margin guess, beats LightGBM on the same rows and ties one trained on fifteen times more." (forecast quality only: the constant-velocity planner still out-drove our robot) | "Better calibrated in a new place" (that was claim A; see 2b) |
| + | **TabPFN at every learned step** (ranges, scene adaptation, the move) | exploratory; passed its pre-written rule on point estimates | Second TabPFN picks the move, 20 fresh crowds vs the planner it learned from: goal 100% vs 100%; time 24.65 vs 24.10 s (+0.55 [−0.34, +1.65]); collision crowds 45% vs 40% (+5 [−15, +25]); safety checks overrode it at 0.94% of replans; 333 ms per plan on the L4 | `docs/TABPFN_POLICY.md`, `RESULTS.md` "TabPFN picks the move", `results/policy_parity.csv`, `media/demo/policy_seed3403.gif` | "On average, a second TabPFN drove like the planner it learned from." | "No classical parts" (tracker, offset and hard checks are classical); "TabPFN at every step" without "learned"; "better planner"; "as safe" (the collision interval's upper end, +0.25, exceeds the 0.244 margin) |

### 2b. Narrowed, failed and smaller findings

| Claim | Status | Number | Source | Say | Never say |
|---|---|---|---|---|---|
| **Claim A:** honest ranges in a scene never seen | **narrowed**: holds on two scenes, not supported on five | Two scenes: TabPFN box 75% vs LightGBM 58% (Hotel → ETH). Five scenes vs LightGBM on ~15x rows: tie (−0.008 [−0.023, +0.007]) | `RESULTS.md` claim A f4; `docs/LOSO5.md` | "Ties with 15 times less data." If the two-scene number is used at all: "on two scenes, not on five" | Leading with it; "better calibrated in a new place" alone |
| **Claim B on real pedestrians** | small | Interval score −0.045 [−0.080, −0.009], coverage level | `results/realdata_claimb.csv` | "Small on real pedestrians, who have no sudden switch" | Any real-data adaptation headline |
| **Claim C:** prediction-aware planning beats standard planners | **failed**, in three crowds | No baseline beaten (Holm, p = 1.0). 4.3 s slower than constant velocity at equal gap, 3.0 s at equal collision rate (20 confirmatory crowds) | `RESULTS.md` T5, `docs/GENERALISATION.md` | "Our robot was slower than standard planners. Better ranges didn't change how this planner drives; its caution setting did." | "Safer" without the matched-time caveat; "beats navigation" |
| **Why C failed** | exploratory | At the same caution setting (q 0.15), split, frozen TabPFN and LightGBM drive the same (split − frozen −0.12 s [−1.35, +1.25]) | `docs/EXPLORATORY_DIALS.md`, `RESULTS.md` | "The box veto throws the range's shape away. The planner is the next lever." | "Calibration made it slower" |
| **Intent chain** (TabPFN classifier, then per-manoeuvre ranges) | offline, adopted on real data only | Real Hotel → ETH, turn rows: interval score −0.263 [−0.481, −0.075] (73 rows, 37 people), turn-row coverage 46% → 63%. Worse in the simulator | `RESULTS.md` "Intent layer", `media/fig_intent_layer.png` | "Better-calibrated ranges for people who turn" (deck only) | "Predicts turns" (turn recall is 0) |
| **Context size** | holds, simulated | TabPFN on 2,000 rows ties LightGBM on 20,000 (pinball 0.106); LightGBM on all 73,000 rows is more accurate than TabPFN on 2,000 (0.606 vs 0.646 m), while hosted TabPFN-3.5 on 20,000 rows is at 0.593 m | `results/context_size.csv`, `media/fig_f4_context_size.png` | Supports P3 in the deck notes | — |
| **Pedestrian bank on cars** (H2b) | exploratory | Boxes about 50 m wide; 250 scene rows halve the interval score (51.7 → 28.2) but a car bank gets 13.0 | `docs/HIGHWAY.md` | "The code carries over; the examples don't. Each body needs its own bank." This is the per-body-bank product point | — |
| **Feature engineering** (time and distance to closest approach, threat, companion, path speed) | not adopted | Raw + engineered ties raw (−0.001 [−0.006, +0.004]); engineered only is worse (+0.029 [+0.017, +0.042]) | `docs/FEATURES_ENG.md` | "TabPFN found what the hand-made features add on its own" (deck note, Q&A) | "Feature engineering improved it" |
| **Speed on a GPU** | measured | NVIDIA L4: 71 ms per replan (split predict), 318 ms scene rebuild, 490 ms frozen fit, 225 ms real loop; MPS timings on an M3 Pro | `docs/DEPLOYMENT.md`, `results/latency_*.csv` | "Fits the 0.4 s replan period on an L4" | "10 Hz" or "real-time" (10 Hz needs the background rebuild, designed, not built); "real-time on a Jetson" |

---

## 3. What exists (feature inventory)

**In the robot loop and tested (T5, three crowds):**
- S1 tracking: a forward-only Kalman filter, tuned per dataset on training data only.
- S2 the table: one row per person, moment and look-ahead, in the person's own frame; a live query and an offline row give identical features (`tests/test_parity.py`).
- S3 example selection: a frozen 2,000-row bank, cached once, plus a scene model on at most 250 recent completed rows, refit every 2 s.
- S4 TabPFN ranges: quantiles for every person and horizon in one batched call; bank and scene quantiles averaged.
- S5 calibration: an online offset (adaptive conformal), updated once per 2 s batch.
- S7 planner: rejects moves that enter a predicted box at +1 or +2 s or fail a 0.4 s constant-velocity check; caution dial `box_q`.
- The TabPFN policy (`--planners tabpfn_policy`): exploratory arm, 20 crowds.
- Baselines in the same loop: stop-and-go, reactive, velocity obstacles, constant velocity with a margin, LightGBM, frozen TabPFN, and an open-loop oracle.

**Built, tested offline only:**
- The cars adapter (`foresight/domains/highway.py`) through the same core.
- The intent classifier chain.
- S6 occupancy grid (used for drawing; its metric is not reported).

**Designed, not built (say so if asked):**
- The background scene rebuild (needed for 10 Hz).
- The constant-velocity takeover when the predictor is late (GATES S8).
- A risk-weighted planner that uses the range's shape.

**How we tested (the method is part of the pitch):** every pass rule was committed before its run; amendments were committed before their results; test crowds are fresh and paired; the statistics are paired bootstraps with Holm correction; failed tests stay in the README. Tests: 126 pass plus 1 expected failure on a fresh clone; the git history has no secrets.

---

## 4. The narrative arc (all surfaces)

1. **Problem.** A robot crossing a station has to guess where everyone will be in two seconds, and how sure it is. An overconfident guess is how robots walk into people. Most stacks assume people walk straight or add one margin; learned models are frozen once trained. (Not "they learn one place": that sets up the new-place claim, which is a tie on five scenes.)
2. **Insight.** Turn the scene into a table. TabPFN reads rows as examples, with no training, so the robot's knowledge is the table, and adapting is swapping rows.
3. **Demo.** The full loop in a simulated crowd that turns erratic five seconds in.
4. **Proof 1.** Adapts when behaviour changes (in the loop).
5. **Proof 2.** Same core, another body (cars, offline).
6. **Proof 3.** Learns more per example (five real scenes).
7. **Where it stops.** The simple yes/no planner: same safety, 3–4 s slower. Next: a planner that weighs how likely each spot is. Scope line: simulated loop and cars, five real pedestrian scenes, non-commercial weights.
8. **Close.** One example bank per body, no retraining; the buyer's guide in one line; every number from a script.

The video's frame-by-frame wording for 7 and 8 is in `docs/VIDEO_BRIEF.md` section 0.

The TabPFN policy rides on proof 1 as one sentence, the first thing cut if the video runs long.

---

## 5. Video

The script to record is `docs/VIDEO_BRIEF.md` section 3 (built around Nooshin's 3D street video, 176 s). The earlier version 3 table that stood here is removed; it is in git history. The wording rules in section 2 apply to every line.

---

## 6. Deck, five minutes (only if we win the track)

The 10-slide plan in `docs/archive/PRESENTATION.md` section 3 was built around claim A and says cars were cut. Replace it with this list. Owner: RoboDetect (`tools/build_deck.py`). About 30 s per slide.

| # | Title | Visual | Number in the notes |
|---|---|---|---|
| 1 | Foresight · where will everyone be in two seconds, and how sure are we? | `docs/walkthrough/s6_all_boxes.png` | Names, track |
| 2 | The robot's knowledge is a table | `media/diagrams/02_pipeline.png` | One row per person, moment, look-ahead; no training; weights never change |
| 3 | The full loop | `media/demo/seed3013.gif` | Crowd chosen by rule; turns erratic at 5 s |
| 4 | Proof 1 · adapts when behaviour changes | `media/fig_t5.png` | +18.0 pts vs LightGBM, +10.0 from the scene; three crowds; P1-fair answer in the notes |
| 5 | Proof 2 · same core, another body | `media/fig_highway.png` | H1 match at 67x fewer rows, coverage 86% vs 69–82%; H2 +14 pts; H2b: each body needs its bank |
| 6 | Proof 3 · learns more per example | `media/fig_loso5.png` | −0.130 vs equal rows; tie vs 15x |
| 7 | TabPFN at every learned step | `media/demo/policy_seed3403.gif` | Policy parity on averages; intent chain on real turning people (−0.263); what stays classical |
| 8 | Where it stops | left panel of `media/fig_t5.png` | Simple yes/no planner: same safety, 3–4 s slower (claim C, three crowds); next: weigh how likely each spot is; scope line. Claim A tie and feature engineering go to backup |
| 9 | When to use which · the product | text | Buyer's guide; 71 ms per replan on an L4; per-body banks; licence |
| 10 | Next | text | Risk-weighted planner; closed loop for cars; real robot logs; background rebuild for 10 Hz |

---

## 7. The hardest questions, with the answer

1. **"LightGBM never saw the scene. Isn't proof 1 unfair?"** The clean adaptation number is TabPFN against itself: the scene rows add 10 points over the same TabPFN without them. Offline, a LightGBM refit on rows from the same scene still covered about 12 points less, with a worse pinball loss (P1-fair). We dropped LightGBM-in-the-loop with scene rows from T5 for cost, and we say so.
2. **"Your robot was slower than the baselines. So what is the product?"** The forecaster. On five real scenes its ranges beat a fixed margin in every scene (−0.365 [−0.405, −0.328]), and still the constant-velocity planner out-drove our robot: say both in one breath. Better ranges did not change how a box-veto planner drives; its caution setting did. The forecast held up on its own tests: adaptation, data efficiency, a second body. The planner that uses the range's shape is the next step.
3. **"With enough data LightGBM catches up. Why TabPFN?"** It does, and we say when to use which. The product case is the first hours at a new site and the moments after behaviour changes, when rows are few. On five real scenes TabPFN on 1,500 rows ties LightGBM on 15 times more.
4. **"Swapping rows means you need that body's rows."** Yes (H2b): pedestrian examples on cars give useless boxes. The code carries over; each body needs its own bank, and the scene's own rows bridge the gap until one exists.
5. **"Is it real-time? Can it run on a robot?"** 71 ms per replan on an NVIDIA L4, inside the 0.4 s replan period; a scene rebuild takes 318 ms. 10 Hz needs the background rebuild, designed and not built. No embedded timing yet.
6. **"Did you tune until it worked?"** Every pass rule and every amendment is a commit made before its result. Claim C failed and claim A narrowed under those rules, and both stay in the README.
7. **"Is the TabPFN policy as safe as the planner?"** On averages it matches; at 20 crowds the collision interval cannot rule out a rate 25 points worse. It shows TabPFN can be the controller, not a better one.

---

## 8. Surface fixes before the freeze

From the wording audit on 22 Nov at about 09:15 Berlin. Section 2 of this page is the reference.

**Done:**
- `README.md`: the intro no longer leads with claim A; the claim A section says "two scenes, not five". Also fixed: the start-here row for this page, the walkthrough description, the intent wording, when to use which, and the latency limit (L4 measured, 10 Hz not claimed). Commits 59d5565 and dde784a.
- `PLAN.md` and `RESULTS.md`: each top now points to this page. PLAN section 0 is marked as history, PLAN's data-efficiency row adds LOSO5, and the MPS timing is filled in.
- `docs/INDUSTRY.md`, `docs/DEPLOYMENT.md`: no "2.5–10 Hz"; the shipped setup is stated; "calibrated ranges make the robot efficient" is narrowed.
- `docs/DEFENSE.md`: "the caution setting, not calibration, set the pace"; "Claims we can make" carries a superseded banner. Section 7 of this page holds the new Q&A.
- `docs/archive/PRESENTATION.md`, `docs/archive/CEO-HANDOVER.md`, `docs/archive/VIDEO_SCRIPT.md`: superseded banners.

- `docs/explainer.html`, the published Overview (version 11): it now leads with the sentence and the three proofs, has sections for cars and the five scenes, and the old limits box is fixed (226bddc). `media/fig_highway.png` is retitled "same core".
- `deck/Foresight.pptx` (removed 22 Nov; superseded by the Canva deck and `deck/Foresight_finalist.pptx`): reordered to section 6 with 12 main slides plus 4 backup; "when to use which" no longer says "safer" (b4c00ed, 402ac6f; RoboDetect).

**Open, owner a person:**
- **No LICENSE file**, while the video close and the README say "open source". Add a licence before the freeze or say "the code is public". [Darshak, shiinalight]
- `README.md`: 9 `[confirm]` items (video link, artifact sharing, roles, AI-assistant line, pre-event code). [Darshak]
- `PLAN.md`: the "second account" line (F9). [Darshak]
- `media/v_limits.png` does not exist. The "where it stops" frame is specified in `docs/VIDEO_BRIEF.md` section 0. [whoever renders the video]

**Added 22 Nov, about 11:50 (CEO check; full list in `final/CTO_CHANGES.md`):** `docs/VIDEO_BRIEF.md` section 3 and `docs/DECK_OUTLINE.md` carry seven wording and edit fixes; two clips were made, `media/demo/highway.gif` and `media/demo/scenes5.gif`; `final/` holds the short reference set.

**Known small inconsistencies, not on the video path:**
- The constant-velocity error on the problem slide: 0.37/0.25 m (`docs/DATA.md`) against 0.428/0.254 m in the f4 claim A rows. These come from different windows; the deck should cite the f4 rows.
- The baseline against the published one: "within 3–14%" (PLAN, PRESENTATION) against "3–10%" (DEFENSE). Say "within 3–14%" only with its source, or leave it out.
- PLAN gives an interval as [−0.43, −0.18] and README as [−0.42, −0.18]; the source is −0.425. Not quoted in the video.
- Older pages say "80% target". The box target is 81% if the axes are independent (honest joint coverage 80–90%). The video and deck say 81%.
