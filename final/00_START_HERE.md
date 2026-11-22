# Foresight: start here

This folder is the short reference set. Eight pages answer almost any question about the project; everything else in the repo is working material or history. Made Sun 22 Nov 2026, about 11:55 Berlin.

## The pages

| Page | Open it when you need |
|---|---|
| **00_START_HERE.md** (this page) | The one-screen summary, the list of clips and figures, what is still open, and where the detail lives |
| [01_STORY.md](01_STORY.md) | What we claim and what we do not: every claim with its number, source and exact wording; the product angle; the seven hardest questions |
| [02_VIDEO.md](02_VIDEO.md) | The 3-minute video: timed script, what is on screen, which clips, why we chose our data and baselines (with references) |
| [03_DECK.md](03_DECK.md) | The 5-minute finalist talk: ten slides, what goes where, what to say, backup slides for questions |
| [04_RESULTS_CARS.md](04_RESULTS_CARS.md) | Proof 2 in full: the highway test, its rules, numbers and caveats |
| [05_RESULTS_FIVE_SCENES.md](05_RESULTS_FIVE_SCENES.md) | Proof 3 in full: five real pedestrian scenes, each held out once |
| [06_QUESTIONS_AND_ANSWERS.md](06_QUESTIONS_AND_ANSWERS.md) | How the work can be attacked, and our answers |
| [CTO_CHANGES.md](CTO_CHANGES.md) | What changed on 22 Nov at 11:50, where, and what is still to do before the freeze |

Pages 01 to 06 are copies. Their sources are in `docs/`; after editing a source, run `python tools/build_final.py`.

## The project on one screen

**The sentence.** The robot's knowledge is a table of examples. It adapts to new behaviour, and to a new body, by swapping rows, with no training step.

**What it is.** A forecasting layer for robots that move among people or vehicles. Each agent at each moment is one row of a table. TabPFN reads a frozen bank of example rows plus the last seconds of the current scene and returns a range for where every agent will be 1 and 2 s ahead. It sits under whatever planner a team already runs.

| | Proof | Number | Limit |
|---|---|---|---|
| 1 | Adapts when behaviour changes, in the robot loop | After the crowd turns erratic: 59% of people inside their 2 s range against 41% for LightGBM (+18 points); 10 of those points come from reading the scene alone; ahead in all three simulated crowds | Simulated; still short of the 81% target |
| 2 | Same core, another body (cars) | 1,500 rows match LightGBM trained on 67 times more (86% inside against 69–82%); when traffic turns aggressive, reading the scene adds 14 points | Simulated, offline, no closed loop for cars |
| 3 | Learns more per example (five real scenes) | Beats the fixed-margin guess in every scene (−0.365 pooled); beats LightGBM on the same 1,500 rows (−0.130); ties LightGBM on 15 times more | A tie, not a win, against the bigger model |
| ✗ | What failed | Our robot beat no standard planner; it was 3 to 4 s slower at equal safety, in three crowds. "Better ranges in a new place" held on two scenes and is a tie on five | The planner uses a yes/no box and throws the range's shape away |

**When to use which.** Few examples, or behaviour that just changed: TabPFN. Plenty of familiar data: a well-fitted LightGBM is as good. The next thing to build is a planner that uses the whole range.

## Clips and figures for the video and the deck

| File | Shows | Used in |
|---|---|---|
| `media/demo_full.mp4` on branch `nooshin/dev` | 3D explainer (five steps) and the street scene | Video 0:11–1:39, deck slides 4–5 |
| `media/demo/calibration_seed3013.gif` | The robot loop: people inside (green) or outside (red) their box, two models | Proof 1 |
| `media/fig_t5.png`, right panel | Coverage before and after the crowd changes | Proof 1 |
| `media/fig_t5.png`, left panel | Time against closest gap for every planner | "Where it stops" only |
| `media/demo/highway.gif` | The same road twice: frozen bank against bank plus scene rows; boxes turn green or red after 2 s | Proof 2 |
| `media/fig_highway.png` | One moment of the same test, as a still | Proof 2, alternative |
| `media/demo/scenes5.gif` | The five real scenes side by side, people and TabPFN's boxes | Proof 3 |
| `media/fig_loso5.png` | Interval score per scene, four models | Proof 3 |
| `media/demo/seed3013.gif` | Four planners in the same crowd | Optional, deck slide 6 |
| `media/demo/policy_seed3403.gif` | The TabPFN policy driving | Optional, deck slide 9b |
| `deck/Foresight_finalist.pptx` | Skeleton deck to finish by hand | Finalist talk |

The two new clips each have a `_stats.json` next to them that says what is drawn and how the episode or window was picked (by a rule, not by eye). Rebuild them with `python tools/render_highway.py` and `python tools/render_scenes5.py`.

## Still open

The open items before submission, with owners, are in [CTO_CHANGES.md](CTO_CHANGES.md), sections 3 and 4. After the submission that list is history.

## Where the detail lives (open only when a page above sends you there)

| Need | Page |
|---|---|
| Every number with its rule, amendment and limits | `RESULTS.md` |
| The public front page, how to run, team | `README.md` |
| Why each design decision was made (D-1 to D-36) | `docs/DECISIONS.md` |
| The data: scenes, sizes, predictability | `docs/DATA.md` |
| Literature, numbered | `docs/REFERENCES.md` |
| What robots do today (Nav2, sidewalk robots) | `docs/INDUSTRY.md` |
| Timings and what a deployment needs | `docs/DEPLOYMENT.md` |
| The TabPFN policy | `docs/TABPFN_POLICY.md` |
| Three crowds, caution dials, feature engineering | `docs/GENERALISATION.md`, `docs/EXPLORATORY_DIALS.md`, `docs/FEATURES_ENG.md` |
| Picture-first explainer; one decision stage by stage | `docs/explainer.html`, `docs/walkthrough.html` |

**History:** `docs/runs/`. The old plans (`PLAN.md`, `docs/archive/`) were removed on 22 Nov; older documents cited elsewhere are in git history.
