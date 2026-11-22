> **Copy of [`docs/VIDEO_BRIEF.md`](../docs/VIDEO_BRIEF.md)**, made by `tools/build_final.py` on 2026-11-22 12:42 (after commit `6a04b36`). Edit the source and rerun the script; do not edit this copy.

# Foresight: video brief (3 minutes)

Written Sun 22 Nov 2026, about 10:15 Berlin, for recording and cutting the submission video. **Revised 11:50** after the CEO check (`final/CTO_CHANGES.md`): seven wording and edit fixes in section 3, two new clips (`media/demo/highway.gif`, `media/demo/scenes5.gif`), and the "open source" line. It condenses `docs/STORY.md` (claims, numbers, wording) and plans the cut around Nooshin's 3D street video (`media/demo_full.mp4` on `nooshin/dev`, commit 28a6b13, 2:29). Every number below comes from STORY.md, which names its source file.


**Revised 12:45 (Nooshin):** the 3D video was re-rendered as one finished file that already contains the proofs and the limits, so the editor only adds the voice-over and the on-camera shots. Section 0 below is its current state and why each change was made; where section 0 and sections 2–3 differ, section 0 is what the video shows.

---

## 0. The 3D video as it is now (2:46)

**File:** 1920×1080, 30 fps, 2:46 (4,985 frames), silent. Rendered by `python tools/record_clip.py --video --fps 30` from `tools/street3d_template.html` on `nooshin/dev`; the street's ranges are the same stored TabPFN predictions as before (no numbers in the scene changed). It replaces the 2:29 cut (`media/demo_full.mp4`, commit 28a6b13).

| Time | On screen | Voice-over cue |
|---|---|---|
| 0:00–0:11 | Title card "Foresight · Prior Labs track" over the street | Open (or cut to the on-camera shot) |
| 0:11–0:26 | The challenge: three cards | As section 3 |
| 0:26–1:04 | How it works: steps 1–5 only (no title or "every agent" card) | One sentence per step, as section 3 |
| 1:04–1:39 | The street: four events, each with a plain-language card | As section 3 |
| 1:39–1:49 | Proof 1 clip: `calibration_seed3013.gif`, "60% vs 45%, target 81%" | Proof 1 sentence starts here |
| 1:49–2:17 | **"Three proofs"**: one frame, three columns appear in turn (at 1:49, 1:58, 2:08); the column being talked about is outlined | One proof sentence per column |
| 2:17–2:37 | **"Where it breaks"**: one frame, two columns (at 2:17, 2:27) and a footer | One limit sentence per column |
| 2:37–2:46 | End card: Foresight, repo link, "Every number comes from a script in the repo", names, "Thank you, Prior Labs" | Close |

**What changed against sections 2–3, and why:**

1. **The caption fixes (section 2, fixes 1–6) are done in the render:**
   - "every half second" became "every 0.4 seconds": the street pipeline predicts every 2 × 0.2 s.
   - "Shown 5,000 example rows" became "thousands of example rows".
   - The problem card now cites 0.25–0.43 m, the f4 rows.
   - The street tests and the old "Honest limits" panel are gone; that is where the three proofs now sit.
   - The street's live "inside its range so far: X%" counter is removed, since no street number goes on screen.
2. **The street's honesty caption sits inside the info card**, as its subtitle: "Simulated street, never seen by the model · robot route scripted". It is not a bar across the top: as a bar it covered the rush-hour card.
3. **The event cards are in plain words, and "zebra" became "crossing":**

   | Event | Card title | Why it matters |
   |---|---|---|
   | Sudden stop | A person stops suddenly | A "keep walking" guess would put them metres ahead. Their range has to stay with them. |
   | Steps onto the crossing | A person steps onto the crossing | The robot's way across is blocked. It can't cross until it's clear. |
   | Hard braking | A car brakes hard for them | It slows fast, so where it will be in 2 seconds changes fast too. |
   | Jaywalker | Someone crosses away from the crossing | No usual path predicts this. A range covers what a single guess misses. |
   | Rush hour | Rush hour: more people, more cars | Everyone hurries. The ranges have to keep up with the new pace. |

   The first version was hard to read for someone new to the project, and "zebra" is not a word everyone knows.
4. **The proofs are drawn as simple bar charts in the video's own style** instead of `fig_t5.png`, `fig_highway.png` and `fig_loso5.png`. Those figures are dense research plots: unreadable at video size, and the left panel of `fig_t5.png` shows the planner result. Each column has one chart, one verdict line and a short source line. The numbers come from the STORY.md ledger:
   - **Proof 1:** 59% vs 41% inside the range after the crowd changes (crossing crowd), +18 points, ahead in all three crowds.
   - **Proof 2:** cars, interval score 3.57 for TabPFN on 1,500 rows against 3.74 for the best-fitted LightGBM on 100,440; 86% vs 72% inside.
   - **Proof 3:** five real scenes, 1.36 for TabPFN on 1,500 rows, 1.49 for LightGBM on the same rows, 1.37 for LightGBM on about 22,000 (15×).
   - The CEO-check clips (`highway.gif`, `scenes5.gif`) are not in this render. The editor can lay them over the proof 2 and 3 columns, or show them in the deck.
5. **Proofs and limits are one steady frame each, not a card per point.** Nine full-screen cards in about 77 s changed every 7–11 s and were tiring to watch. Each frame now builds up column by column in time with the voice-over.
6. **"Where it breaks" has two limits, not four lines:**
   - **The robot wasn't faster:** +4.3 s at the same closest gap, +3.0 s at the same collision rate. Why: the planner only checks whether a move enters a box, so it ignores the range's shape.
   - **Each body needs its own examples:** predicting cars from pedestrian rows scores 51.7, against 13.0 for car rows. Why: people and cars move differently; the code carries over, the examples don't.
   - **Footer:** the robot loop and the cars are simulated (real data: pedestrians only) · TabPFN-3.5 weights are non-commercial.
   - **The new-place tie is no longer listed as a limit.** On five scenes TabPFN on 1,500 rows ties a LightGBM given 15× more data, which is the data-efficiency result of proof 3, not a failure. Proof 3 says it: "Same quality from 15× fewer rows". The honest wording stays in STORY.md: claim A ("honest ranges in a new place") held on two scenes and tied on five. Say so if asked; don't call it a win in calibration.
   - The planner result is a limit both here and in STORY.md, and the bodies result is the per-body bank point.
7. **The length is 2:46**, under 3:00.

**Voice-over for the two frames** (a sentence each, about 9–10 s):
- Proof 1 (over the clip and the first column): as section 3.
- Proof 2: "It works across modalities: on cars it matches LightGBM with 67 times less data."
- Proof 3: "It works with very little data: on five real scenes, the same result as LightGBM given 15 times more."
- Limit 1: "The robot wasn't faster. Our planner only checks whether a move enters a box, so it ignores the range's shape."
- Limit 2: "Each body needs its own examples. People and cars move differently; the code carries over, the examples don't."

**Prepared, not yet rendered:** shorter proof titles, approved in a still frame. Re-render to include them; it takes about 10 min.
- "Adapts to dynamic environments", "Works across modalities", "Works with very little data".
- One-line verdicts: "+18 points, no retraining", "Same result, 67× less data", "Same result, 15× less data".
- "Mistake score (lower is better)" in place of "range score".

---

## 1. The whole story on one card

| Question | Answer to say |
|---|---|
| **What problem?** | A delivery robot shares the street with people, cars and bikes. To plan, it needs to know where everyone will be in the next two seconds, and how sure it can be. An overconfident guess is how robots walk into people. |
| **What do robots do today?** | Assume everyone keeps going straight (constant velocity), add one fixed margin, or slow down and stop when someone is close. Stock ROS 2 Nav2 plans around where obstacles are now; it does not predict people. Learned forecasters do better, but once trained they are frozen: a new site or a change in behaviour means training again. |
| **What do we do?** | Each agent at each moment becomes one row of a table: its motion, its neighbours, the robot, obstacles. TabPFN reads example rows (a frozen bank plus the last seconds of this scene) and returns a range for every agent at +1 and +2 s, in one pass, with no training. The robot keeps clear of those ranges. **The robot's knowledge is the table, so it adapts by swapping rows.** |
| **How well does it work?** | 1) When the crowd suddenly turns erratic, reading the scene's last seconds gives 18 points more coverage than LightGBM, 10 of them from the scene alone, ahead in all three simulated crowds. 2) The same core on simulated highway traffic: 1,500 rows match LightGBM trained on 67 times more, with more honest ranges. 3) On five real pedestrian scenes: better than the fixed-margin guess in every scene (interval score −0.365 [−0.405, −0.328] pooled), better than LightGBM on the same rows, level with one trained on 15 times more. |
| **Where did it fail?** | Our robot was not faster than standard planners (3 to 4 s slower at equal safety, in three crowds): better ranges did not change how a simple box-avoiding planner drives. The "honest ranges in a new place" claim held on two real scenes, not on five (a tie). Each body needs its own examples: pedestrian rows on cars give useless boxes. |
| **What else did we try?** | A second TabPFN that picks the robot's move: on average it drove like the planner it learned from. A TabPFN intent classifier: better ranges for people who turn (offline, real data). Hand-made features such as time to closest approach: no gain, TabPFN already finds them. A latency test: predictions must arrive within the 0.4 s replan (71 ms on an NVIDIA L4 does). |
| **What is it for?** | An uncertainty layer under any planner, re-targeted to a new site, a new behaviour or a new body by swapping its example table. Few examples or changed behaviour: TabPFN. Plenty of familiar data: a well-fitted LightGBM is as good. |
| **How do we know?** | Every pass rule was written before its run; failures stay in the README; every number comes from a script in the repo. |

---

## 2. What we have to show

**Nooshin's 3D video** (`media/demo_full.mp4`, 2:29, rendered from `media/street3d_demo.html` with `tools/record_clip.py --video`; three.js, held-out street episode):

| Part | Content | Use in the final cut |
|---|---|---|
| Problem (~15 s) | Three cards: "Where will everyone be in two seconds, and how sure are we?"; "The usual guess is not good enough" (0.93 m off after 2 s in a crowd); "A guess alone is not enough" (too narrow: collides; too wide: freezes) | **Keep.** Voice-over in section 3 |
| How it works (five steps, one pedestrian) | 1 Watching (noisy dots, Kalman track) · 2 One row of a table · 3 The question (correction to "keep going straight") · 4 TabPFN answers (5%, 50%, 95% in one step, no training) · 5 Sizing the patch (stretched until 80% land inside, on validation) · "Every agent, every half second" | **Keep.** This is the "measurements become a table" walkthrough. Fix two captions (below) |
| Street scene ("last-metre delivery", 40 × 25 m) | Robot leaves the Café through a narrow door (waits for a customer), crosses at the zebra, goes to the Station. Scripted events: 12 s sudden stop · 22 s pedestrian steps onto the zebra, car brakes hard · 32 s jaywalker mid-block · 40 s rush hour (hurried people, commuters, twice the cars). Each event card has a "why it matters" line | **Keep.** Core of the demo. Add one caption: "Illustration: a held-out simulated street; the robot's route is scripted, the ranges are TabPFN's". Its ranges come from a fixed example set (`data/street`, `tools/street3d.py`), so the voice-over does not claim live adaptation here; proof 1 shows that |
| Evidence (outro) | Three street tests (range size at equal coverage; adaptation at rush hour; error against number of rows), the crossing-crowd context-size bar, "Honest limits", end card | **Replace** with our three proofs and "where it breaks" (section 3). See the fixes below |

**Our assets** (`darshak/core`): `media/demo/highway.gif` (the cars test as a clip: the same road twice, frozen bank against bank plus scene rows, boxes turn green or red after 2 s; replay of the stored H2 predictions, episode picked by rule, `tools/render_highway.py`), `media/demo/scenes5.gif` (the five real scenes side by side with TabPFN's boxes; an illustration of the five-scene setup run locally, `tools/render_scenes5.py`), `media/demo/seed3013.gif` (robot loop, crowd turns erratic at 5 s), `media/demo/calibration_seed3013.gif` (coverage counters, this crowd 60% vs 45%), `media/fig_t5.png`, `media/fig_highway.png` (titled "same core"), `media/fig_loso5.png`, `media/demo/policy_seed3403.gif` and `.mp4`, `media/diagrams/01_problem.png` and `02_pipeline.png`.

**Explainers we planned, and what became of them** (`docs/archive/VISUALS.md`, `docs/archive/PRESENTATION.md` section 5, `docs/archive/TEAM-PLAN.md`):

| Planned | Status | Where |
|---|---|---|
| Four-panel bird's-eye video: stop-and-go, reactive, velocity obstacles, TabPFN on the same crowd | **Made** | `media/demo/seed3013.gif` (crowd picked by rule: median) |
| "Pipeline at work": one moment, every stage | **Made** as a stage-by-stage page | `docs/walkthrough.html`, images `docs/walkthrough/s1`–`s9` |
| Explainer diagrams (problem, pipeline, row, TabPFN step, planner, coverage, tests, deployment) | **Made** | `media/diagrams/01`–`12`, used in `docs/explainer.html` |
| 3D viewer | **Made by Nooshin** as the street and doorway scenes with their own pipeline, not as a replay of the robot loop's trace | `nooshin/dev`: `media/street3d_demo.html`, `media/demo_full.mp4` |
| Calibration and policy clips | **Made** | `media/demo/calibration_seed3013.gif`, `media/demo/policy_seed3403.gif` |
| Occupancy overlay with its metric | **Not made** (the grid is drawn; its metric was never reported) | — |
| Live camera demo (webcam, YOLO, ranges drawn live) | **Not made** | — |
| Side-by-side constant velocity against ours (`docs/archive/TEAM-PLAN.md`) | **Superseded** by the four-panel video | — |

**Fixes needed in the 3D video's text** (Nooshin; template `tools/street3d_template.html` on `nooshin/dev`):
1. **The outro's "Honest limits" panel is out of date.** It says "the final pre-registered test is running" and "honest ranges for real people in a new place: being re-checked". Both have finished: the planner test failed (slower than standard planners) and the new-place claim ties on five scenes. Replace the panel with our "where it breaks" card, or cut the outro.
2. **Street Test 2 says "No faster adaptation here"** (the street's rush hour is mild; all models recover within 1 s). It is honest but, next to our proof 1, it confuses a first-time viewer. Leave the street tests out of the video and keep them for the README. If kept, say why: "on this street the switch is mild".
3. **"Every agent, every half second"**: our robot loop replans every 0.4 s (the ETH annotation step). Check the street pipeline's actual step and caption it with that number.
4. **"Shown 5,000 example rows"** is the street demo's setting. Our robot loop reads a 2,000-row bank plus at most 250 rows of the current scene. Either keep 5,000 and label the demo as its own setup, or say "thousands of example rows".
5. **"Sizing the patch ... on validation"** is the street demo's calibration. Our robot loop also corrects the size online from completed rows. Fine as is, but don't let the voice-over say the loop is calibrated: after the switch it reaches 59% against an 81% target.
6. **The problem card "0.25–0.37 m on real Zurich pedestrians"** comes from `docs/DATA.md`. The f4 rows used in our results give 0.25–0.43 m. Either is defensible with its source; pick one.

---

## 3. The three-minute cut

**Clip lengths in the 3D video** (from `tools/street3d_template.html` on `nooshin/dev`; they add up to its 2:29):
- problem: 15 s;
- five-step explainer: 45.5 s (title 2.7 s, then about 7–8 s per step, then "every agent" 4.5 s);
- street: 43 s (establishing shot 4 s, four events of about 5 s each with faster follow shots between, overview tail 6 s);
- evidence outro: 45 s.

**The plan uses about 93 s of it:** the problem, the five steps without their title and "every agent" cards (38 s), and the street without its establishing shot and with the overview tail cut to 2 s (35 s). The outro is replaced by our proofs. These are cuts in the editor, so no re-render is needed. Target 2:53, which leaves a few seconds for fades.

Pace limit: about 2.3 spoken words per second. No numbers on camera; numbers go in the voice-over and captions. The explainer's own step captions are burned in, so the voice-over says one sentence per step, in step with them.

| Time | On screen | Voice-over (words) |
|---|---|---|
| **0:00–0:11** | **On camera:** Darshak. Lower third "Foresight · Prior Labs track" | "A delivery robot shares the street with people, cars and bikes. Where will they be in two seconds, and how sure is it?" (23) |
| 0:11–0:26 | 3D problem cards (15 s) | "Today most robots assume people keep going straight, or keep one fixed margin. Too narrow and they collide, too wide and they freeze. Learned models do better, but once trained they are frozen." (33) |
| 0:26–0:33 | 3D step 1, "Watching" | "A filter turns noisy sensor dots into a smooth track: position and speed." (13) |
| 0:33–0:41 | 3D step 2, "One row of a table" | "Every moment becomes one row of a table: motion, neighbours, the robot, obstacles." (13) |
| 0:41–0:48 | 3D step 3, "The question" | "The model only learns the correction to keep going straight." (10) |
| 0:48–0:56 | 3D step 4, "TabPFN answers" | "TabPFN reads thousands of example rows with known outcomes, and answers in one step, with no training." (17) |
| 0:56–1:04 | 3D step 5, "Sizing the patch" | "The range is sized so the truth lands inside as often as promised." (13) |
| 1:04–1:39 | 3D street: sudden stop, zebra braking, jaywalker, rush hour, every agent's range. Caption "Illustration: held-out simulated street; scripted robot route" | "The robot's knowledge is the table. Here is a street run it never saw. Someone stops dead: walking straight would run past them. A car brakes hard for the zebra; its range has to change with it. A jaywalker leaves every expected path. Then rush hour: the crowd changes how it moves. Can the table keep up? We tested it." (59) |
| 1:39–1:57 | **Proof 1.** `media/demo/calibration_seed3013.gif`, caption "this crowd: about 60% vs 45%, target 81%" (the clip's own counters end at 61% and 46%; the whole run is 60% and 45%); then the **right panel only** of `media/fig_t5.png` (crop it: the left panel is the planner result, where our robot loses, and belongs to "where it breaks") | "Swapping rows works. In our robot loop the crowd turns erratic; reading its last seconds gave eighteen points more coverage than LightGBM, ten from the scene alone, and it led in all three crowds." (34) |
| 1:57–2:10 | **Proof 2.** `media/demo/highway.gif` (the road before and after the switch, two models), caption "this road: 77 vs 67 of 100 forecasts held · all five roads: 77% vs 63%" (the clip's own counters; never caption it with the pooled number alone) | "Same core, simulated highway traffic. Fifteen hundred rows match LightGBM trained on sixty-seven times more. When traffic turns aggressive, reading the scene adds fourteen points." (25) |
| 2:10–2:22 | **Proof 3.** `media/demo/scenes5.gif` for 5 s (the five scenes, people and boxes), caption "one 16 s window per scene" (its counters, 78–95%, are these windows; the test's coverage is 78–87%), then `media/fig_loso5.png` | "On five real pedestrian scenes it beats the fixed-margin guess, beats LightGBM on the same rows, and ties one trained on fifteen times more." (24) |
| 2:22–2:44 | "Where it breaks" text card (make in the editor; the four lines below) | "Where it breaks. Our robot was slower than standard planners: this planner ignores the range's shape. Better ranges in a new place? On five scenes, a tie. Each body needs its own examples. And everything but the real pedestrians is simulated." (41) |
| **2:44–2:56** | **On camera:** Nooshin and Darshak. End card: repo URL, "every number comes from a script in the repo" | "Foresight lets a robot adapt by swapping examples, not retraining. Open source; every number comes from a script. Thank you, Prior Labs." (22) **The repo has no LICENSE file yet.** Add one before the freeze, or the end card and the description say "the code is public" |

Beats: 11 + 15 + 38 (five steps) + 35 + 18 + 13 + 12 + 22 + 12 = 176 s (2:56). Fastest beats: the problem and step 4 at about 2.1 words per second; the rest are 1.4 to 2.0. If the steps' own timings differ by a second in the edit, move the step sentences with them.

**"Where it breaks" card (four lines):**
- Slower than standard planners (3–4 s at equal safety, three crowds)
- New place: held on two real scenes, a tie on five
- Each body needs its own example bank
- Simulated robot loop and cars · non-commercial TabPFN-3.5 weights

**Optional beats, if there is room** (cut first if the cut runs long):
- After proof 1: `media/demo/policy_seed3403.gif`, "A second TabPFN can even pick the robot's move; on average it drove like the planner it learned from." (21)
- `media/demo/seed3013.gif` under the proof 1 beat, to show the real robot loop with all four planners side by side.

**Never cut:** the open, the five-step table walkthrough, the street, the three proofs, "where it breaks", the close.

**Words we never use:** "safer", "beats LightGBM" for cars, "same code, unchanged", "kept the ranges honest" or "calibrated" for the loop, "predicts turns", "any robot", "real-time", "10 Hz", "drives a car". The full list is in STORY.md section 2.

---

## 4. Why we chose what we chose (literature backing)

Short form for the README, the deck notes and Q&A. Full entries in `docs/REFERENCES.md` (numbers in brackets) and `docs/INDUSTRY.md` (letter codes).

| Choice | Why | Backed by |
|---|---|---|
| **The problem: uncertainty, not just a point guess** | A planner needs to know how wrong a forecast might be; freezing and collisions are both real failure modes | Trautman & Krause 2010 [20], Trautman et al. 2015 [21] (the freezing robot problem); Mavrogiannis et al. 2023 [23] (survey) |
| **Baseline: constant velocity** | The strongest simple baseline on real pedestrians; "keep going straight" is what many robots assume | Schöller et al. 2020 [1]: on average it matched or beat the deterministic neural predictors of 2019–20, most clearly on Hotel (not on ETH; later sampled models report lower errors) |
| **Predict the correction to walking straight, in the person's own frame** | Relative, rotation-free inputs carry over to new scenes | Schöller et al. 2020 [1], Sec. VI-A. They predict relative displacements (Sec. III); the residual-to-constant-velocity target in the person's frame is our own design choice |
| **What robots do today** | Stock Nav2 plans on current obstacle positions; deployed sidewalk robots describe slow-down and stop rules | Nav2 docs and maintainer posts [N9, N10, N12]; Starship and Serve public pages [S1–S5] |
| **Data: ETH, Hotel, Zara1, Zara2, Univ** | The standard real pedestrian benchmark, world coordinates, used by every method we compare with | Pellegrini et al. 2009 [5] (ETH, Hotel); Lerner et al. 2007 [6] (UCY: Zara, Univ); read through OpenTraj, Amirian et al. 2020 [7]. REFERENCES.md row 7 still says UCY is not used; it now is (`docs/LOSO5.md`) |
| **Leave one scene out** | Tests a place the model has never seen; the standard protocol is train on four, test on the fifth, which our five-scene test follows | Gupta et al. 2018 [2], Salzmann et al. 2020 [3], Yuan et al. 2021 [4], Schöller et al. 2020 [1]. We score fixed horizons with ranges, not 12-step ADE/FDE, so published numbers are a reference point, not a like-for-like comparison |
| **Step size 0.4 s** | ETH/UCY are annotated every 0.4 s (2.5 fps), the grid every published result uses (8 observed and 12 predicted frames = 3.2 s and 4.8 s); the robot replans on the same step (D-2) | Gupta et al. 2018 [2]; Alahi et al. 2016 [36]; OpenTraj [7] (2.5 fps) |
| **Horizons 1 and 2 s** | At walking speed 2 s is about 2 m, the space a robot plans in; Nav2's local controllers roll the robot's motion 1–3 s ahead; driving benchmarks use 3–8 s for much faster vehicles (D-1). 4.8 s is reported only for comparison with published work | Nav2 docs [N5, N6, N11]; Waymo Open Motion [A1], Argoverse 2 [A2] |
| **TabPFN: examples in, prediction out, no training** | One forward pass over labelled rows; strongest on small data, which is what a robot has in a new place | Hollmann et al. 2025, *Nature* [8]; TabPFN-3.5 technical report [10] (the model we run) |
| **Adapting by choosing which rows to read** | Same-scene in-context examples adapt a trajectory model to new scenes without weight updates; neighbour rows as context help TabPFN | TrajICL, Fujii et al. 2025 [13]; LoCalPFN, Thomas et al. 2024 [12] (precedent, not the same method) |
| **Online calibration offset** | Keeps long-run coverage near target when the scene shifts | Gibbs & Candès 2021 [17] (adaptive conformal); Romano et al. 2019 [39] (conformalized quantile regression); Lindemann et al. 2023 [15] (why this is not a safety guarantee: people react to the robot) |
| **Scores: coverage, interval score, pinball** | Coverage alone rewards wide boxes; the interval score charges width plus misses and is proper | Gneiting & Raftery 2007 [31]; Koenker & Bassett 1978 [30] (quantile regression) |
| **Learned baseline: LightGBM quantile regression** | The standard strong tabular learner. We ran it untuned, with a small-data setting, and well-fitted on 15× and 67× more rows | Ke et al. 2017 [29] |
| **Planner baselines: reactive, velocity obstacles, stop-and-go** | The families deployed stacks and the literature use | Fox et al. 1997 [26]: ours is "reactive, DWA-style" (same velocity sampler, people frozen where they are), not full DWA; van den Berg et al. 2008 [27]: ours is a sampling velocity-obstacle planner without reciprocity, so not RVO or ORCA; stop-and-go as described by deployed sidewalk robots [S1–S5] |
| **Crowd simulator: social force** | The classic model of pedestrian motion; ours adds a sudden behaviour switch | Helbing & Molnár 1995 [22] |
| **Cars: highway-env with IDM and MOBIL** | A standard open driving simulator with the classic car-following and lane-change models | Leurent 2018, highway-env (GitHub); Treiber, Hennecke & Helbing 2000, *Phys. Rev. E* 62:1805 (IDM); Kesting, Treiber & Helbing 2007, *Transportation Research Record* 1999:86–94 (MOBIL). Cited from memory and not yet in REFERENCES.md: check against highway-env's own citation before quoting |
| **Robot metrics: time to goal, closest gap, collisions, freezing** | The recommended set for social navigation | Francis et al. 2025 [25]; Chen et al. 2019 [24] |
| **Statistics: paired crowds, bootstrap, Holm, rules written first** | Same crowds for every planner; intervals instead of single numbers; correction for testing against four baselines; no tuning after seeing results | Law 2015 [42] (common random numbers); Efron 1979 [38]; Agarwal et al. 2021 [33]; Holm 1979 [37]; Nosek et al. 2018 [40] (pre-registration); Gelman & Loken 2013 [41] (forking paths) |

**What has no literature behind it:** our three proofs are our own results. The closest precedents are TrajICL [13] (in-context adaptation for trajectories) and LoCalPFN [12] (retrieval for TabPFN). We found no paper that tests a tabular foundation model's ranges in a robot loop.

---

## 5. Before recording

- **Who does what:** Darshak opens and closes on camera; Nooshin co-presents the close. [confirm roles]
- **Nooshin: the only must is fix 1** (the stale limits panel), and cutting the outro in the editor does it with no re-render. Fixes 2–6 apply only if the outro or those captions stay. If she re-renders: `python tools/record_clip.py --video --fps 30`.
- **The 3D street lives only on `nooshin/dev`.** That branch forked from the first core commit, so it is not merged before the freeze. The README points to it. No street number goes on screen unless her `features.py` has the leak fixes the core got (D-20, D-26).
- **Editor:** the "where it breaks" text card; captions burned in; the policy beat only if under 3:00; `fig_t5.png` cropped to its right panel in proof 1; the two new clips in proofs 2 and 3.
- **If the voice-over is already recorded:** the picture fixes (crop, clips, captions) need no new audio. Re-record only the street's first and last sentence, proof 2, proof 3 and the new-place sentence; if there is no time, the two street sentences come first ("a street it never saw" is not true: it is a held-out run of the same street, and its ranges use a fixed example set).
- **Licence:** the close says "open source" and the repo has no LICENSE file. [Darshak, shiinalight]
- **Read the voice-over aloud once with a timer.** Every beat is at or under about 2.1 words per second, so there is a little room to slow down.
- **Upload unlisted;** the link must open logged out; put it in the README `[confirm: video link]`.
