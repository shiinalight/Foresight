# Foresight: video brief (3 minutes)

Written Sun 22 Nov 2026, about 10:15 Berlin, for recording and cutting the submission video. It condenses `docs/STORY.md` (claims, numbers, wording) and plans the cut around Nooshin's 3D street video (`media/demo_full.mp4` on `nooshin/dev`, commit 28a6b13, 2:29). Every number below comes from STORY.md, which names its source file.

---

## 1. The whole story on one card

| Question | Answer to say |
|---|---|
| **What problem?** | A delivery robot shares the street with people, cars and bikes. To plan, it needs to know where everyone will be in the next two seconds, and how sure it can be. An overconfident guess is how robots walk into people. |
| **What do robots do today?** | Assume everyone keeps going straight (constant velocity), add one fixed margin, or slow down and stop when someone is close. Stock ROS 2 Nav2 plans around where obstacles are now; it does not predict people. Learned forecasters do better, but they learn one place and must be retrained for another. |
| **What do we do?** | Each agent at each moment becomes one row of a table: its motion, its neighbours, the robot, obstacles. TabPFN reads example rows (a frozen bank plus the last seconds of this scene) and returns a range for every agent at +1 and +2 s, in one pass, with no training. The robot keeps clear of those ranges. **The robot's knowledge is the table, so it adapts by swapping rows.** |
| **How well does it work?** | 1) When the crowd suddenly turns erratic, reading the scene's last seconds gives 18 points more coverage than LightGBM, 10 of them from the scene alone, ahead in all three simulated crowds. 2) The same core on simulated highway traffic: 1,500 rows match LightGBM trained on 67 times more, with more honest ranges. 3) On five real pedestrian scenes: better than LightGBM on the same rows, level with one trained on 15 times more. |
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

**Our assets** (`darshak/core`): `media/demo/seed3013.gif` (robot loop, crowd turns erratic at 5 s), `media/demo/calibration_seed3013.gif` (coverage counters, this crowd 60% vs 45%), `media/fig_t5.png`, `media/fig_highway.png` (titled "same core"), `media/fig_loso5.png`, `media/demo/policy_seed3403.gif` and `.mp4`, `media/diagrams/01_problem.png` and `02_pipeline.png`.

**Fixes needed in the 3D video's text** (Nooshin; template `tools/street3d_template.html` on `nooshin/dev`):
1. **The outro's "Honest limits" panel is out of date.** It says "the final pre-registered test is running" and "honest ranges for real people in a new place: being re-checked". Both have finished: the planner test failed (slower than standard planners) and the new-place claim ties on five scenes. Replace the panel with our "where it breaks" card, or cut the outro.
2. **Street Test 2 says "No faster adaptation here"** (the street's rush hour is mild; all models recover within 1 s). It is honest but, next to our proof 1, it confuses a first-time viewer. Leave the street tests out of the video and keep them for the README. If kept, say why: "on this street the switch is mild".
3. **"Every agent, every half second"**: our robot loop replans every 0.4 s (the ETH annotation step). Check the street pipeline's actual step and caption it with that number.
4. **"Shown 5,000 example rows"** is the street demo's setting. Our robot loop reads a 2,000-row bank plus at most 250 rows of the current scene. Either keep 5,000 and label the demo as its own setup, or say "thousands of example rows".
5. **"Sizing the patch ... on validation"** is the street demo's calibration. Our robot loop also corrects the size online from completed rows. Fine as is, but don't let the voice-over say the loop is calibrated: after the switch it reaches 59% against an 81% target.
6. **The problem card "0.25–0.37 m on real Zurich pedestrians"** comes from `docs/DATA.md`. The f4 rows used in our results give 0.25–0.43 m. Either is defensible with its source; pick one.

---

## 3. The three-minute cut

The pace limit is about 2.3 spoken words per second. No numbers on camera; numbers in voice-over and captions only.

| Time | On screen | Voice-over (words) |
|---|---|---|
| **0:00–0:12** | **On camera:** Darshak. Lower third "Foresight · Prior Labs track" | "A delivery robot shares the street with people, cars and bikes. Where will everyone be in two seconds, and how sure can it be?" (24) |
| 0:12–0:27 | 3D problem cards (Nooshin), then `media/diagrams/01_problem.png` | "Today most robots assume people keep going straight, or keep one fixed margin. Too narrow and they collide, too wide and they freeze. Learned models do better, but they learn one place." (32) |
| 0:27–1:00 | 3D five-step explainer on one pedestrian (Nooshin) | "We turn every moment into a row of a table. A filter smooths the sensor track. The row holds the person's motion, their neighbours, the robot and the obstacles. TabPFN reads thousands of example rows like it, with known outcomes, and answers in one step, with no training: where this person will be, as a range. The robot's knowledge is the table." (62) |
| 1:00–1:35 | 3D street: Café door, sudden stop, zebra braking, jaywalker, rush hour, with every agent's range. Caption "Illustration: held-out simulated street; scripted robot route" | "Here is a street the model never saw. Someone stops dead: walking straight would run past them. A car brakes hard for the zebra, and its range has to change with it. A jaywalker leaves every expected path. Then rush hour: the crowd changes how it moves. That is the moment swapping rows is for." (55) |
| 1:35–1:55 | **Proof 1.** `media/demo/calibration_seed3013.gif`, caption "this crowd: 60% vs 45%, target 81%"; then `media/fig_t5.png` | "Does swapping rows work? In our robot loop the crowd turns erratic. Reading the last seconds gave eighteen points more coverage than LightGBM, ten from the scene alone, in all three crowds we tried." (34) |
| 1:55–2:10 | **Proof 2.** `media/fig_highway.png` | "The same core on simulated highway traffic: fifteen hundred rows match LightGBM trained on sixty-seven times more, with more honest ranges." (21) |
| 2:10–2:22 | **Proof 3.** `media/fig_loso5.png` | "On five real pedestrian scenes it beats LightGBM on the same rows, and ties one trained on fifteen times more." (20) |
| 2:22–2:45 | "Where it breaks" text card (make in the editor; four lines, the bullets below) | "Where it breaks. Our robot wasn't faster than standard planners: this planner ignores the range's shape. In a new place it ties a model with far more data. Each body needs its own examples. And it's all simulated except the real pedestrians." (42) |
| **2:45–3:00** | **On camera:** Darshak and Nooshin. End card: repo URL, "every number comes from a script in the repo" | "Foresight lets a robot adapt its forecasts by swapping examples, not retraining. It's open source, and every number comes from a script. Thank you, Prior Labs." (26) |

Beats: 12 + 15 + 33 + 35 + 20 + 15 + 12 + 23 + 15 = 180 s. Fastest beat: the problem, 2.1 words per second; the rest are 1.4 to 2.0.

**"Where it breaks" card (four lines):**
- Not faster than standard planners (3–4 s slower at equal safety)
- New place: ties LightGBM trained on 15× more data
- Each body needs its own example bank
- Simulated robot loop and cars · non-commercial TabPFN-3.5 weights

**Optional beats, if there is room** (cut first if the cut runs long):
- After proof 1: `media/demo/policy_seed3403.gif`, "A second TabPFN can even pick the robot's move; on average it drove like the planner it learned from." (21)
- `media/demo/seed3013.gif` under the 1:35 beat, to show the real robot loop with all four planners.

**Never cut:** the open, the five-step table walkthrough, the street, the three proofs, "where it breaks", the close.

**Words we never use:** "safer", "beats LightGBM" for cars, "same code, unchanged", "kept the ranges honest" or "calibrated" for the loop, "predicts turns", "any robot", "real-time", "10 Hz", "drives a car". The full list is in STORY.md section 2.

---

## 4. Why we chose what we chose (literature backing)

Short form for the README, the deck notes and Q&A. Full entries in `docs/REFERENCES.md` (numbers in brackets) and `docs/INDUSTRY.md` (letter codes).

| Choice | Why | Backed by |
|---|---|---|
| **The problem: uncertainty, not just a point guess** | A planner needs to know how wrong a forecast might be; freezing and collisions are both real failure modes | Trautman & Krause 2010 [20], Trautman et al. 2015 [21] (the freezing robot problem); Mavrogiannis et al. 2023 [23] (survey) |
| **Baseline: constant velocity** | The strongest simple baseline on real pedestrians; "keep going straight" is what many robots assume | Schöller et al. 2020 [1] (CVM beats deterministic deep models on ETH/UCY) |
| **Predict the correction to walking straight, in the person's own frame** | Relative, rotation-free inputs carry over to new scenes; residual targets lower error | Schöller et al. 2020 [1], Sec. III and VI-A |
| **What robots do today** | Stock Nav2 plans on current obstacle positions; deployed sidewalk robots describe slow-down and stop rules | Nav2 docs and maintainer posts [N9, N10, N12]; Starship and Serve public pages [S1–S5] |
| **Data: ETH, Hotel, Zara1, Zara2, Univ** | The standard real pedestrian benchmark, world coordinates, used by every method we compare with | Pellegrini et al. 2009 [5] (ETH, Hotel); Lerner et al. 2007 [6] (UCY: Zara, Univ); read through OpenTraj, Amirian et al. 2020 [7] |
| **Leave one scene out** | Tests a place the model has never seen; the standard protocol is train on four, test on the fifth | Gupta et al. 2018 [2], Salzmann et al. 2020 [3], Yuan et al. 2021 [4], Schöller et al. 2020 [1] |
| **Step size 0.4 s** | ETH/UCY are annotated every 0.4 s (2.5 fps), the grid every published result uses (8 observed and 12 predicted frames = 3.2 s and 4.8 s); the robot replans on the same step (D-2) | Gupta et al. 2018 [2]; Alahi et al. 2016 [36]; OpenTraj [7] (2.5 fps) |
| **Horizons 1 and 2 s** | At walking speed 2 s is about 2 m, the space a robot plans in; Nav2's local controllers roll the robot's motion 1–3 s ahead; driving benchmarks use 3–8 s for much faster vehicles (D-1). 4.8 s is reported only for comparison with published work | Nav2 docs [N5, N6, N11]; Waymo Open Motion [A1], Argoverse 2 [A2] |
| **TabPFN: examples in, prediction out, no training** | One forward pass over labelled rows; strongest on small data, which is what a robot has in a new place | Hollmann et al. 2025, *Nature* [8]; TabPFN-3.5 technical report [10] (the model we run) |
| **Adapting by choosing which rows to read** | Same-scene in-context examples adapt a trajectory model to new scenes without weight updates; neighbour rows as context help TabPFN | TrajICL, Fujii et al. 2025 [13]; LoCalPFN, Thomas et al. 2024 [12] (precedent, not the same method) |
| **Online calibration offset** | Keeps long-run coverage near target when the scene shifts | Gibbs & Candès 2021 [17] (adaptive conformal); Romano et al. 2019 [39] (conformalized quantile regression); Lindemann et al. 2023 [15] (why this is not a safety guarantee: people react to the robot) |
| **Scores: coverage, interval score, pinball** | Coverage alone rewards wide boxes; the interval score charges width plus misses and is proper | Gneiting & Raftery 2007 [31]; Koenker & Bassett 1978 [30] (quantile regression) |
| **Learned baseline: LightGBM quantile regression** | The standard strong tabular learner; we also gave it 15× and 67× more rows | Ke et al. 2017 [29] |
| **Planner baselines: reactive, velocity obstacles, stop-and-go** | The families deployed stacks and the literature use | Fox et al. 1997 [26] (dynamic window); van den Berg et al. 2008, 2011 [27, 28] (velocity obstacles, ORCA); stop-and-go as described by deployed sidewalk robots [S1–S5] |
| **Crowd simulator: social force** | The classic model of pedestrian motion; ours adds a sudden behaviour switch | Helbing & Molnár 1995 [22] |
| **Cars: highway-env with IDM and MOBIL** | A standard open driving simulator with the classic car-following and lane-change models | Leurent 2018, highway-env (GitHub); Treiber, Hennecke & Helbing 2000, *Phys. Rev. E* 62:1805 (IDM); Kesting, Treiber & Helbing 2007, *Transportation Research Record* 1999:86–94 (MOBIL). Not yet in REFERENCES.md |
| **Robot metrics: time to goal, closest gap, collisions, freezing** | The recommended set for social navigation | Francis et al. 2025 [25]; Chen et al. 2019 [24] |
| **Statistics: paired crowds, bootstrap, Holm, rules written first** | Same crowds for every planner; intervals instead of single numbers; correction for testing against four baselines; no tuning after seeing results | Law 2015 [42] (common random numbers); Efron 1979 [38]; Agarwal et al. 2021 [33]; Holm 1979 [37]; Nosek et al. 2018 [40] (pre-registration); Gelman & Loken 2013 [41] (forking paths) |

**What has no literature behind it:** our three proofs are our own results. The closest precedents are TrajICL [13] (in-context adaptation for trajectories) and LoCalPFN [12] (retrieval for TabPFN). We found no paper that tests a tabular foundation model's ranges in a robot loop.

---

## 5. Before recording

- **Who does what:** Darshak opens and closes on camera; Nooshin co-presents the close. [confirm roles]
- **Nooshin:** fixes 1–4 in section 2 to the 3D template, then re-render with `python tools/record_clip.py --video --fps 30`. Or render without the outro and cut our figures in.
- **Editor:** the "where it breaks" text card; captions burned in; the policy beat only if under 3:00.
- **Read the voice-over aloud once with a timer.** Every beat is at or under 2.1 words per second, so there is room to slow down.
- **Upload unlisted;** the link must open logged out; put it in the README `[confirm: video link]`.
