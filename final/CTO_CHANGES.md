# Changes after the CEO check, and what is left for the CTO

Sun 22 Nov 2026, about 11:55 Berlin. The story, the video plan and the deck plan keep their structure. This page lists what was reworded, where, and what still has to be done by the CTO session or by a person. Freeze 12:30, submit by 16:00.

## 1. Already changed (no action unless you disagree)

| # | What | Why | Where |
|---|---|---|---|
| 1 | Problem line: "they learn one place" became "once trained they are frozen" | "One place" sets up the new-place claim, which is a tie on five scenes. The proofs are about adapting without training | `docs/VIDEO_BRIEF.md` section 3 (0:11), `docs/STORY.md` section 4 item 1, `README.md` "The problem" |
| 2 | Street: "a street it never saw" became "a street run it never saw" | It is a held-out episode of the same simulated street (`tools/street3d.py` on `nooshin/dev`: "held-out episode; context from data/street") | `docs/VIDEO_BRIEF.md` section 3 (1:04), `docs/DECK_OUTLINE.md` slide 5 |
| 3 | Street: "That is what swapping rows is for" became "Can the table keep up? We tested it." | The street's ranges use a fixed example set, and the street's own test found no faster adaptation. The line claimed adaptation where none is shown | `docs/VIDEO_BRIEF.md` section 3 (1:04) |
| 4 | Proof 1 picture: only the right panel of `media/fig_t5.png`; caption "about 60% vs 45%" | The left panel shows our robot losing to the standard planners while the voice describes a success. The clip's counters end at 61% and 46%; the whole run is 60% and 45% | `docs/VIDEO_BRIEF.md` section 3 (1:39) |
| 5 | Proof 2: new clip, and the voice now says the +14 points after the switch | The old picture showed the aggressive-switch scene while the voice only gave the 67x match. The clip's caption gives its own counters and the pooled number ("this road: 77 vs 67 of 100; all five roads: 77% vs 63%"); "more honest ranges" (86% vs 69–82%) stays in the README and the deck notes | `docs/VIDEO_BRIEF.md` section 3 (1:57), `docs/DECK_OUTLINE.md` slide 7 |
| 6 | Proof 3: "beats the fixed-margin guess" added, and a new clip | The video names "straight line plus a fixed margin" as the problem and never beat it on screen. `docs/LOSO5.md`: TabPFN is better than constant velocity plus band in every scene, −0.365 [−0.405, −0.328] pooled. Scope: forecast quality only | `docs/VIDEO_BRIEF.md` sections 1 and 3 (2:10), `docs/DECK_OUTLINE.md` slides 8 and 9, `docs/STORY.md` P3 row and question 2, `README.md` proof 3 row |
| 7 | "Our new-place claim didn't survive five scenes" became "Better ranges in a new place? On five scenes, a tie." | The viewer never heard the claim; the disclosure now stands alone | `docs/VIDEO_BRIEF.md` section 3 (2:22) |
| 8 | Deck: slide 4 cut to 15 s; slide 6 "Same result in all three crowds" became "Ahead in all three crowds" | Slides 4 and 5 repeat the video; the three gaps differ (+6.6, +18.0, +19.7) | `docs/DECK_OUTLINE.md` slides 4 and 6 |

The video is now 176 s (was 173 s). Every line stays under 2.3 words per second.

## 2. New files

| File | What it is |
|---|---|
| `media/demo/highway.gif`, `highway_stats.json` | The cars test as a clip. Replay of the stored H2 predictions (`results/highway_h2_q.npz`); no model run. Episode 5, picked by rule: the median coverage gap of the five held-out episodes. In this episode 77 of 100 forecasts held against 67 of 100; over all five episodes 76.7% against 62.5% |
| `media/demo/scenes5.gif`, `scenes5_stats.json` | The five real scenes side by side with TabPFN's boxes. An illustration of the five-scene setup, not a result: local TabPFN, 1,500 context rows from the other four scenes (context seed 0), one 16 s window per scene fixed by rule. In these windows 78–95% inside; the test's own coverage is 78–87%, so caption the clip "one 16 s window per scene" |
| `tools/render_highway.py`, `tools/render_scenes5.py` | The two scripts (`render_scenes5.py --predict` first; about 5 minutes on a laptop CPU) |
| `tools/build_final.py` | Refreshes the copies in `final/` |
| `final/` | The short reference set; start at `final/00_START_HERE.md` |

## 3. For the CTO session

| # | Task | Where |
|---|---|---|
| C1 | `tools/build_outline_deck.py` still has the old wording, and `deck/Foresight_finalist.pptx` was built from it: slide 2 notes ("learn one place"), slide 5 title ("A street the model has never seen"), slide 6 notes ("Same result in all three crowds"). Slides 7 and 8 should take the two new clips, slide 8 and 9 notes the fixed-margin sentence. Change the script and rebuild only if nobody has started editing the pptx by hand; otherwise change the pptx by hand | `tools/build_outline_deck.py` lines 34, 44, 52; `deck/Foresight_finalist.pptx` |
| C2 | Check `docs/explainer.html`, the published Overview and the Digest page against changes 1 and 6 (the problem wording; the fixed-margin result in the five-scene section) | `docs/explainer.html`, `tools/frag/` |
| C3 | Nothing committed yet: the changes above are in the working tree. Commit them with your next commit if Darshak has not done so (`final/`, `docs/STORY.md`, `docs/VIDEO_BRIEF.md`, `docs/DECK_OUTLINE.md`, `README.md`, `media/MANIFEST.csv`, the two clips with their stats files, the three tools) | working tree |
| C4 | `docs/STORY.md` section 5 still holds the superseded version 3 script with the old problem line (0:15). It is marked superseded; delete the table or leave it, but do not record from it | `docs/STORY.md` section 5 |
| C5 | After any further edit to `docs/STORY.md`, `VIDEO_BRIEF.md`, `DECK_OUTLINE.md`, `HIGHWAY.md`, `LOSO5.md` or `DEFENSE.md`: `python tools/build_final.py`, then commit | `final/` |
| C6 | `media/demo/policy_seed3403.mp4` is still untracked. Commit it or delete it | working tree |

## 4. For a person

| # | Task | Owner |
|---|---|---|
| P1 | **No LICENSE file**, and the on-camera close says "open source". Add a licence before 12:30, or the end card and the video description say "the code is public". TabPFN-3.5's weights are non-commercial either way; that is separate from the code's licence | Darshak, shiinalight |
| P2 | Repo is still private; `main` already equals `darshak/core` | shiinalight |
| P3 | Video edit: crop `fig_t5.png` to its right panel in proof 1; use `highway.gif` in proof 2 and `scenes5.gif` in proof 3; captions as in `02_VIDEO.md`; cut the 3D outro (its "Honest limits" panel is out of date) | whoever cuts |
| P4 | If the voice-over is already recorded: re-record the street's first and last sentence, proof 2, proof 3 and the new-place sentence. If there is time for only one, the two street sentences | Darshak |
| P5 | `README.md`: 5 `[confirm]` items; `PLAN.md` line 104 (item F9; PLAN.md removed 22 Nov) | Darshak |

## 5. One question to have ready

"You say the forecast beats a fixed margin, yet the constant-velocity planner beat your robot. Which is it?" Both. On five real scenes the ranges are better than a fixed band in every scene. In the robot loop the planner turns each range into a yes/no box, so its caution setting sets the pace and the better range is wasted. The planner that uses the whole range is the next thing to build.
