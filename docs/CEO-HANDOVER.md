# CEO review and handover to the CTO session

Written Sun 22 Nov 2026, 03:55 Berlin (checked against network time, not the prose times in other docs). Submission closes 16:30; we submit by 16:00. Based on `darshak/core` at `0269ef9`.

This page sets the goal, the story, the order of work and the stop list for the last 12 hours. `PLAN.md` stays the living plan; fold sections 3 to 6 into it and then treat this page as history.

## 1. Verdict

- **Submit it. It is good enough, and the evidence is stronger than the current video says.** The repo holds five real scenes, three simulated crowds, a second body (highway cars, offline, since 03:45 tonight) and TabPFN at three layers (ranges, scene model, and an intent classifier that is offline only). "One scene, one body, one layer" was true on Saturday evening and is out of date.
- **The weak spot is packaging, not evidence.** The video still leads with the one claim we downgraded ourselves (claim A, 75% against 58%), spends about 40 of 180 seconds on what failed, and has no beat for cars.
- **No new experiment families.** The remaining hours go to the story, the submission surface and sleep.

## 2. Goal and non-goals

**Goal:** by 16:00, a public repo, a 3-minute video and a 5-minute deck that make one point a Prior Labs judge can repeat: *the robot's knowledge is a table of examples, so it adapts to new behaviour and to a new body by swapping rows, with no training step.*

**Non-goals:**
- Beating standard navigation planners. Claim C failed in three crowds; we say so in one sentence.
- "Any robot" as a proven claim. We show two bodies offline and one in the closed loop, all simulated except the pedestrian forecasts.
- More tuning, more dials, more scenes.

## 3. The story we submit

One sentence, three proofs, one honest failure.

| | Proof | Number to say | Source | Limit to say with it |
|---|---|---|---|---|
| 1 | **Adapts when behaviour changes, no retraining** | In the loop, reading the scene's last seconds adds 10 points of coverage over the same model without it; against LightGBM the gain holds in all three crowds (+6.6, +18.0, +19.7 points) | `results/inloop_claimb.csv`, `docs/GENERALISATION.md` | Simulated crowds; still short of 80% after the change; small on real pedestrians, which have no sudden switch |
| 2 | **Same core, another body** | Highway cars through the unchanged core: TabPFN on 1,500 rows matches LightGBM trained on 67 times more, and its ranges hold the truth 86% of the time against 69–82% for LightGBM (section 4a); after traffic turns aggressive, scene rows add 14 points of coverage (76.7% against 62.5%, interval score −0.98 [−1.41, −0.56]) | `results/highway_h1.csv`, `results/highway_h2.csv`, `docs/HIGHWAY.md` | Simulated traffic, offline, no closed loop for cars. About 80% of the H2 gain falls in the first 6 s after the switch, so say "adapts through the transition"; 5 held-out episodes, 100 vehicles. **Do not say "beats LightGBM on 100,000 rows":** that holds only against the pre-set small-data setting (section 4a) |
| 3 | **Learns more per example** | On five real pedestrian scenes, TabPFN beats LightGBM on the same 1,500 rows (interval score −0.130 [−0.154, −0.107]) and ties LightGBM trained on about 15 times more | `docs/LOSO5.md` | Against the bigger model it ties, it does not win; with plenty of varied data LightGBM catches up |
| ✗ | **What failed** | Our robot beat no standard planner under the pre-set test, in any of three crowds. In the confirmatory run it was 3 to 4 s slower at equal safety. Among the learned predictors, better ranges did not change how the box planner drives; its caution setting did | `RESULTS.md` T5, `docs/GENERALISATION.md`, `docs/EXPLORATORY_DIALS.md` | At a lower caution setting (q 0.35) and in the dense crowd it is roughly level with the baselines; that is a description from exploratory runs, not a result |

Two smaller findings worth one line each in the README, not the video:
- **A pedestrian example bank does not transfer to cars** (H2b: boxes about 50 m wide). Reading 250 car rows from the scene halves the interval score (51.7 → 28.2) but stays far behind a car bank (13.0). The code is body-agnostic; the examples are not.
- **Scene rows alone, with no bank at all, match the bank on cars** after the switch (interval score 14.0 against 14.0, coverage 73% against 62.5%). Exploratory; the scene rows contain the new behaviour and the bank does not.

## 4. Review findings the CTO must act on

| # | Finding | Evidence | Action |
|---|---|---|---|
| F1 | **The repo is private.** A public repo is a submission requirement | `gh repo view shiinalight/Robodetect`: `PRIVATE` | Owner flips it public after the secrets check in F5. This is a person's action, not a session's |
| F2 | **`main` is behind.** Judges land on `main` | `origin/main` 13+ commits behind `darshak/core` | Promote by fast-forward after the freeze, per `PLAN.md` section 5 |
| F3 | **Prose times in the exploratory docs run 1 to 2.5 hours ahead of the real clock.** Example: `TABPFN_POLICY.md` says "written ~05:45 Berlin"; its first commit is `de81e0a` at 03:32:38 +0100, and network time was 03:43 when this review ran. Also `EXPLORATORY_DIALS.md` (03:00 / ~03:45 / ~04:10 against commits from 02:37), `GENERALISATION.md` (~04:20 / ~04:45 / ~05:20 against 03:04 to 03:21), `LOSO5.md` (~05:00 / ~05:40 against 03:12 to 03:32) | `git log --diff-filter=A --format='%h %ci' -- <file>` | Replace every prose time with the commit hash and its commit time. The order "rule before result" holds by commit order; only the prose is wrong. Also state once that pod run ids are in UTC: `20261122-013935-dx-…` is 02:39 Berlin, two minutes after the plan commit at 02:37; without that note a reader sees runs that predate their plan. Audit `PLAN.md`, `RESULTS.md`, `README.md` and `DEFENSE.md` the same way. Our pitch rests on pre-registration, so this must be clean |
| F4 | **README is not judge-ready.** 9 `confirm` placeholders (video link, explainer link, roles, AI-assistant line, pre-event code), a "pending" table that mixes done and open items, and the note that `RESULTS.md` real-data tables are still f1 | `grep -n confirm README.md` | Fill or delete each. Put a one-screen top on the README: the sentence, the demo clip, the section 3 table, "what failed" |
| F5 | **Secrets check before going public** | `.env` is git-ignored; not verified in history | `git log --all -p -S TABPFN_TOKEN` and a scan for tokens before F1 |
| F6 | **H1 baseline strength.** `lgbm_all` on about 100,000 car rows uses the small-data setting (100 trees, min 50 per leaf), chosen before the run. A judge will ask whether it is underfit | `tools/highway_tests.py` line 186 | Robustness check run by the CEO session (stronger LightGBM settings on all rows, same test rows, paired vehicle bootstrap). Result in section 4a: the win shrinks to a tie against a well-fitted LightGBM. Report it next to H1 as a disclosed post-hoc check, and wherever H1 is quoted (`docs/HIGHWAY.md`, README, deck, explainer) say "matches with 67 times fewer rows, with more honest ranges", not "beats" |
| F7 | **Fresh-clone run.** `results/` and `data/` are git-ignored and only part of the cited result files are force-added | `.gitignore`, `git ls-files results/` | One fresh clone on a second machine or folder: `pip install`, `pytest`, one figure script. Tests pass here (126 passed, 1 xfailed, project venv) |
| F9 | **`PLAN.md` section 5 says "a second account was added" for the TabPFN API.** Prior Labs judges will read a public repo | `PLAN.md` line 100 | Darshak decides before F1 whether that line stays, and whether the arrangement is within the hackathon's terms. Not a session's call |
| F8 | **Video headline contradicts our own verdict.** The 1:40 beat says 75% against 58% (claim A, two scenes); `LOSO5.md` says claim A is not supported on five | `docs/VIDEO_SCRIPT.md` | Replace with proofs 2 and 3 (section 5) |

### 4a. H1 robustness check (post-hoc, disclosed)

**Result: H1's pre-set verdict stands by its rule, but the margin depends on the LightGBM setting. Against a well-fitted LightGBM on all rows it is a tie on interval score, with better coverage and slightly worse point error.**

Same split, same 1,500 test rows, same paired vehicle bootstrap (355 vehicles) as H1; LightGBM fitted on all 100,440 training rows; TabPFN rows read from `results/highway_h1_rows.csv`. The first row reproduces the stored `lgbm_all` rows exactly.

| LightGBM on all rows | Box coverage | Interval score | Error | TabPFN minus LightGBM, interval score | Coverage | Error |
|---|---|---|---|---|---|---|
| m50, 100 trees (pre-set `lgbm_all`) | 82.1% | 4.06 | 0.918 m | −0.49 [−0.66, −0.34] | +3.5 pts | −0.164 m |
| untuned default (m10, 300 trees) | 78.9% | 3.90 | 0.854 m | −0.33 [−0.49, −0.18] | +6.7 pts | −0.100 m |
| m20, 1,000 trees, 63 leaves | 71.6% | 3.74 | 0.741 m | −0.18 [−0.36, −0.01] | +14.0 pts | +0.013 m [−0.016, +0.044] |
| m50, 2,000 trees, 127 leaves | 69.1% | 3.75 | 0.705 m | −0.18 [−0.37, +0.00] | +16.5 pts | +0.049 m [+0.020, +0.078] |
| **TabPFN, 1,500 rows** | **85.6%** | **3.57** | 0.754 m | | | |

"Error" is the mean error of the median, as in `docs/HIGHWAY.md`.

Reading:
- Bigger LightGBM models get sharper and more accurate at the centre, and more overconfident (coverage falls from 82% to 69% against an 81% target).
- TabPFN keeps the best interval score at every setting, but against the largest model the interval touches zero. This is the pedestrian pattern again: a tie on the proper score with far fewer rows, and more honest ranges.
- Limits of this check: four settings picked after seeing H1, none tuned on a validation split, one run each. Scratch script, not in the repo: the CTO session should port it into `tools/highway_tests.py` as a `--robustness` flag and add this table to `docs/HIGHWAY.md` as a disclosed post-hoc check.

## 5. Video, version 2 (180 s)

Eddie's one recommendation was "something visual you can show of how it works". Pictures first, numbers in voice-over.

| Time | On screen | Point |
|---|---|---|
| 0:00–0:15 | On camera | The problem: where is everyone in two seconds, and how sure are we |
| 0:15–0:30 | Problem diagram | Fixed margins and models that learned one place |
| 0:30–0:55 | Pipeline, one table row | The scene as a table; TabPFN reads examples; no training. The robot's knowledge is the table |
| 0:55–1:25 | `media/demo/seed3013.gif` | The full loop; the crowd turns erratic at 5 s |
| 1:25–1:50 | `media/fig_t5.png` right panel | Proof 1: adapts from the last seconds; holds in three crowds |
| 1:50–2:15 | Cars clip or `media/fig_highway.png` | Proof 2: same code on cars; 1,500 rows match a model trained on 67 times more, with more honest ranges; adapts after traffic turns aggressive |
| 2:15–2:30 | Five-scene chart. **It does not exist yet:** `tools/loso5.py` writes no figure. Draw one bar chart from `results/loso5.csv`, or use a text card | Proof 3: real pedestrians, better at equal rows, level with 15 times more data |
| 2:30–2:48 | "Where it breaks" card | Slower than standard planners; pedestrian examples do not carry over to cars; simulated; non-commercial weights |
| 2:48–3:00 | On camera | Close; every number comes from a script in the repo |

Cut from version 1: the 75% against 58% line and the intent-chain sentence (both stay in the README).

## 5a. Fallback narrative: the case study (Darshak, 03:55; held in reserve, not built towards)

If the load-bearing proofs do not hold up, the same material is told as a careful case study instead of a new method: *we put a tabular foundation model into a robot's predict-and-plan loop, compared it with the usual methods under rules fixed in advance, and here is where it helps, where it does not, and how to deploy it.*

- **Trigger (the only one):** a review before 09:30 finds an error in proof 1 (in-loop adaptation) or proof 2 (cars, H2). A failed policy arm, a missing cars clip or a tie instead of a win are not triggers; narrative A already says them.
- **What changes:** the open, the close and the order of beats. No new runs and no new figures.

| | Narrative A (default) | Narrative B (fallback) |
|---|---|---|
| Open | The robot's knowledge is a table; it adapts by swapping rows | We tested TabPFN against the usual methods in a robot loop, with the rules written first |
| Middle | Demo, then proofs 1 to 3 | Demo, then one scoreboard: against constant velocity, fixed band, LightGBM, velocity obstacles and reactive, on calibration, data need, adaptation and driving |
| Close | What failed, in one beat | The decision guide: few rows or a change in behaviour → TabPFN; plenty of familiar data → LightGBM; the planner, not the forecaster, is the next lever; timings (71 ms on an L4) and licence |

- **Already in A either way:** the decision guide ("when to use which") is the last line of the "where it breaks" card and a README section. That costs nothing, and it makes a later flip a re-cut, not a re-write.

## 6. Roadmap for the remaining hours (Berlin)

| When | Item | Owner | Done when |
|---|---|---|---|
| 04:00–05:00 | F3 timestamp fix; F6 result folded in; section 3 rows into `PLAN.md`, `README.md` top, `RESULTS.md`; video script v2; one cars slide in the deck | CTO session | Committed on `darshak/core` |
| 04:00–05:00 | Policy arm (`docs/TABPFN_POLICY.md`): let the pilot rule decide. Its status is unknown to this review (it runs on the pod) | Builder session | Pilot verdict written by 05:00. If it passes, the full test runs unattended and is read at 09:30. **Cut-off 09:30:** no full-test result by then means one line, "tried, not finished". If the pilot has not passed by 05:00, stop it and release the pod |
| optional, ≤ 60 min, only if started before 05:00 | Cars clip: one held-out episode, frozen against split boxes side by side, scene model refit every 2 s as in the loop. About 20 API calls. Labelled an illustration of H2, not a new result. If not started by 05:00, use `media/fig_highway.png` | Builder session | `media/demo/highway.gif` |
| 05:00–09:30 | **Sleep.** Nobody records a pitch well after a night awake | People | — |
| 09:30–11:00 | Read the video script aloud once; F4 placeholders; F5 secrets check; F7 fresh clone | Darshak, Nooshin, CTO session | README has no `confirm` left |
| 11:00–11:30 | On-camera recording (no numbers on camera) | Darshak, Nooshin | Two takes |
| 11:30–12:30 | Promote to `main` (F2); repo public (F1); freeze | CTO session, repo owner | `main` = `darshak/core`; repo opens logged out |
| 12:30–15:30 | Cut the video; deck final; upload unlisted | People | Link opens logged out |
| 15:30–16:00 | Submit the form; update `project.yaml` | Darshak | Confirmation received |

## 7. Stop list

- No new dials, scenes, seeds or model tiers.
- No closed loop for cars.
- No rewrite of the planner. The risk-weighted planner is the first item after the hackathon.
- No result enters the video unless its rule was committed before its run, or it is labelled exploratory.

## 8. Wording rules for every surface

- Say "adapts by swapping examples, no training". Do not say "any robot" as a result; say "two bodies through one unchanged core".
- Cars: "simulated traffic, offline". Never "drives a car".
- Claim A: "ties with 15 times less data". Never "better calibrated in a new place" without "on two scenes, not on five".
- Claim C: "slower than standard planners". Never "safer" without the matched-time caveat.
- Intent chain: never "predicts turns".

## 9. After the hackathon (the original vision, in order)

1. **A planner that uses the range's shape:** a risk-weighted cost over the predicted distribution instead of a box veto. The veto reduces a calibrated distribution to one threshold: split and frozen TabPFN drove the same at q 0.15 although split covered 10 points more. (The open-loop oracle's 0% collisions in the dial sweep, exploratory, says a much better forecast would also pay through this planner.)
2. **TabPFN as the policy** (the duckpfn pattern), if the pilot shows life.
3. **Closed loop on the second body,** then a third (warehouse or drone data).
4. **Real robot data with a robot in the scene,** and timing on embedded hardware.
5. **Per-body example banks behind one interface,** with scene rows as the bridge when no bank exists (the H2b finding).
