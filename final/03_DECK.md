> **Copy of [`docs/DECK_OUTLINE.md`](../docs/DECK_OUTLINE.md)**, made by `tools/build_final.py` on 2026-11-22 14:58 (after commit `385fb2c`). Edit the source and rerun the script; do not edit this copy.

# Foresight: finalist deck outline (5 minutes)

The plan for the finalist deck: for each slide, the job it does, the headline, where things go, which pictures to place, and what we say. Numbers come from `docs/STORY.md`. The 3D stills in `media/stills/` are frames from Nooshin's video (`nooshin/dev`, `media/demo_full.mp4`).

**Arc:** problem → the idea → watch it work → three proofs → where it stops → what it's for.
**Pace:** 10 slides at about 30 s each. One idea per slide. Numbers go on the slide only where a number is the point.
**As built (22 Nov, 14:15).** `python tools/build_outline_deck.py` writes the finished deck, `deck/Foresight_finalist.pptx`: 10 talk slides, a closing slide, a "Backup slides" divider and 11 backup slides (23 in all), in the 3D video's look (warm page, white cards, orange #C24E1C as the only accent, Arial). Time, speaker and the spoken text are in the speaker notes. `--static` writes `deck/Foresight_finalist_static.pptx`, where each clip is replaced by its last frame; `deck/Foresight_finalist.pdf` is exported from that one. Where the built deck differs from the plan below, the "Built" line under the slide says how.

**Decision (22 Nov, 14:45, Darshak): we present the local pptx, `deck/Foresight_finalist.pptx` (static copy and PDF beside it). The Canva deck is not used; the table below is kept only as a record.**

**Two decks exist (22 Nov, 14:35).** The Canva deck "Foresight Finalist Pitch" (10 pages, no backups) is the one the team has been presenting; the pptx above has the same ten slides plus a closing slide and the backups. Whichever is shown, the Canva deck needs these fixes first (read from its saved version at 14:32):

| Canva slide | Problem | Fix |
|---|---|---|
| 2 | The picture's card says 0.25–0.37 m; the video, the README and this outline say 0.25–0.43 m | Replace the picture with `media/deck/3d_problem.jpg` |
| 5 | "Cars brake for zebras" (the video now says "crossing"); a still where a clip would show the events | Change the line to "A car brakes hard at the crossing"; optionally place `media/deck/street_clip.mp4` (33 s, silent) |
| 7 | The picture is a mid-run frame whose counters read 47% and 63%, next to "86% inside the range" | Drag in `media/demo/highway.gif`, or use `media/deck/highway_last.png` (77 against 67 of 100); caption "This road: 77 vs 67 of 100 forecasts held · all five roads: 77% vs 63%" |
| 8 | Only the dense chart; "Tie" reads as a weak result | Add `media/demo/scenes5.gif` (or `media/deck/scenes5_last.png`) with "one 16 s window per scene"; change "Tie" to "Level" ("with LightGBM trained on 15× the rows") |
| 8, 9 | The numbers on 8 and the card on 9 are in a different font from the rest | Set the font by hand in Canva |
| 10 | The close lacks the product line | Add "One example bank per body, no retraining" |
| all but 9 | Presenter notes are empty | Paste the "Say" lines from this outline, or read them from the pptx notes |
| — | No backup slides | Keep `deck/Foresight_finalist.pdf` (pages 13–23) open for questions |

**De-slop pass (22 Nov, 15:00, on Darshak's go).** Nine changes after a self-review: slide 8 leads with "5 of 5 scenes", "4 of 5 scenes", "15× less data" and keeps the scores as small print; slide 3 draws the table; slide 4 has larger pictures and no filler line; "rows" is the one word for data; labels are sentence case with no letter-spacing and no decorative bars; the closing slide is four plain lines with the number first; slide 6's headline is one line ("Reading the scene kept more people inside their ranges"); slide 7 says what the interval score is; slide 10 says "rules out moves too close in the next 0.4 s" (the planner's short check, which assumes people keep walking straight, `foresight/core/planner.py`; a check, not a guarantee) and "New body: new bank".

**Speakers:** Darshak opens and takes the idea and the proofs; Nooshin takes the 3D walkthrough and the street demo. Swap if you prefer; switch speakers at most twice.

---

## Main deck

### 1 · Title (15 s)
- **Headline:** Foresight. Where will everyone be in two seconds, and how sure are we?
- **Layout:** the still fills the slide; a dark band across the bottom third holds the title, the question and the names.
- **Visual:** `media/stills/3d_range_every_agent.jpg` (the street, a range under every agent), full bleed.
- **Say:** "A delivery robot shares the street with people, cars and bikes. Foresight tells it where they'll be next, and how much to trust that."
- **On slide:** names, "Prior Labs track".

### 2 · The problem (30 s)
- **Headline:** Robots guess with a straight line and a fixed margin.
- **Layout:** headline on top; two images side by side at equal size; one line under them with the 0.93 m number.
- **Visuals:** `media/stills/3d_problem_093m.jpg` (the 0.93 m card) on the left; `media/diagrams/01_problem.png` (straight-line guess against where the person goes) on the right.
- **Say:** today's options: assume everyone walks straight, add one margin, or stop when someone is close. Stock ROS 2 Nav2 plans around where people are now. Learned forecasters do better, but once trained they are frozen. Too narrow a margin and the robot hits someone; too wide and it freezes.
- **Number:** walking-straight is off by 0.93 m after 2 s in our crowd, 0.25–0.43 m on real Zurich pedestrians.
- **Built:** a frame of the 2:46 video on the left (`media/deck/3d_problem.jpg`, whose card cites 0.25–0.43 m), "0.93 m" large on the right with the two margin cases; the planner diagram `01_problem.png` is not used.

### 3 · The idea (30 s)
- **Headline:** The robot's knowledge is a table.
- **Layout:** headline on top; the row diagram on the left, the live-context diagram on the right; the line "No training step. Adapting = swapping rows." in bold under them.
- **Visuals:** `media/diagrams/03_row.png` (one person becomes one row) and `media/diagrams/05_live_context.png` (frozen bank plus the scene's newest rows, swapped every 2 s).
- **Say:** each person, at each moment, is one row: their motion, neighbours, the robot, obstacles. TabPFN reads a few thousand example rows and answers in one pass, with no training. To adapt to a new place, a new behaviour or a new body, change the rows.
- **On slide:** "No training step. Adapting = swapping rows."
- **Built:** a drawn table on the left (four agents, the numbers of the video's step-2 card, the last column "where in 2 s?" open); on the right a drawn flow: example bank (2,000 rows, frozen) plus this scene (up to 250 newest rows, swapped every 2 s) → TabPFN → a range for every agent.

### 4 · How it reads one person (15 s) · Nooshin
- **Headline:** From sensor dots to a range, in four steps.
- **Layout:** headline on top; four stills in one row, left to right in step order, each with a two-word label underneath (Watch · Row · TabPFN · Size).
- **Visuals:** a strip of four stills: `3d_step1_watching.jpg` → `3d_step2_row.jpg` → `3d_step4_tabpfn.jpg` → `3d_step5_sizing.jpg`.
- **Say:** a filter smooths the noisy track; the moment becomes a row; TabPFN predicts the correction to walking straight as a 5–95% range; the range is sized so the truth lands inside as often as promised.
- **Alternative:** play this part of the 3D video (about 40 s) and drop slide 5's clip.
- **Built:** four pictures, two by two, from frames of the 2:46 video (`media/deck/3d_step1_watch.jpg` to `3d_step4_size.jpg`), each with a number, a word and one sentence.

### 5 · Watch it on a street (35 s) · Nooshin
- **Headline:** A street run the model has never seen.
- **Why "run":** it is a held-out episode of the same simulated street, and its ranges use a fixed example set. Do not say the model adapts here; proof 1 shows that.
- **Layout:** the clip as large as the slide allows, headline above; the caption in small type under it. With stills: three in a row.
- **Visual:** the 3D street clip from the video (about 30 s, sudden stop → zebra braking → jaywalker → rush hour). Fallback stills: `3d_street_sudden_stop.jpg`, `3d_street_zebra_brake.jpg`, `3d_street_rush_hour.jpg`.
- **Say:** narrate the events: someone stops dead, a car brakes for the zebra, a jaywalker leaves every path, rush hour changes the crowd.
- **Caption on slide:** "Simulated street; scripted robot route; ranges from TabPFN."
- **Built:** the clip is in the slide: `media/deck/street_clip.mp4` (33 s, silent, 1:05–1:38 of the 2:46 video, 6 MB); click it to play. The four events are listed beside it; "zebra" became "crossing". The static deck shows the "A person stops suddenly" frame.

### 6 · Proof 1: it adapts when behaviour changes (35 s)
- **Headline:** When the crowd changed, reading the scene kept more people inside their ranges.
- **Layout:** headline on top (two lines is fine); the GIF on the left, the coverage chart on the right; one bold line of numbers under them.
- **Visuals:** `media/demo/seed3013.gif` (four robots in the same crowd, crowd turns erratic at 5 s) for 5 s, then `media/demo/calibration_seed3013_strip.png` or the right panel of `media/fig_t5.png` (coverage before and after the switch).
- **Say:** in our robot loop the crowd turns erratic five seconds in. The TabPFN that reads the last seconds kept 59% of people inside their 2 s range; LightGBM kept 41%. Ten of those points come from reading the scene alone. Ahead in all three crowds.
- **Numbers:** 59% vs 41% (+18 pts), +10 from the scene, ahead in 3 of 3 crowds (+6.6, +18.0, +19.7).
- **Built:** `media/demo/calibration_seed3013.gif` on the left (people inside or outside their box, two models; caption "one crowd of the 20: about 60% vs 45%"), three drawn bars on the right (59%, 49% for the same TabPFN without the scene, 41%). `seed3013.gif` and the full `fig_t5.png` are not used here: both show the planner comparison, which is slide 9.

### 7 · Proof 2: same core, another body (30 s)
- **Headline:** The same core predicts highway traffic.
- **Layout:** headline on top; the highway figure across the full width (it is wide); one bold line of numbers under it.
- **Visual:** `media/demo/highway.gif` (the same road twice: frozen bank on top, bank plus this road's last seconds below; a box is where the model expects a car in 2 s and turns green or red when the car gets there). Caption it "this road: 77 vs 67 of 100; all five roads: 77% vs 63%". Still alternative: `media/fig_highway.png`.
- **Say:** we pointed the same core, same features and same models, at simulated highway traffic, with a small adapter for the cars. 1,500 example rows match LightGBM trained on 67 times more, and our ranges held the truth 86% of the time against 69–82%. When drivers turn aggressive, reading the scene adds 14 points.
- **Numbers:** 1,500 vs 100,440 rows; 86% vs 69–82%; +14 pts after the switch.
- **Built:** the clip on the left with its caption, two drawn bars on the right (interval score 3.57 on 1,500 examples against 3.74 on 100,440, as in the video's proof frame), "Same result, 67× less data".

### 8 · Proof 3: more from each example (30 s)
- **Headline:** On five real scenes, TabPFN gets more from each example.
- **Layout:** headline on top; the five-scene chart across the full width; one bold line of numbers under it.
- **Visual:** `media/demo/scenes5.gif` (the five scenes side by side, people and TabPFN's boxes; caption "one 16 s window per scene", because its counters are for these windows, not the test) for a few seconds, then `media/fig_loso5.png` (each real scene held out once).
- **Say:** ETH, Hotel, Zara 1 and 2, Univ: train on four, test on the fifth. TabPFN beats the fixed-margin guess from slide 2 in all five scenes. On the same 1,500 rows it beats LightGBM in four scenes and ties on Hotel. LightGBM catches up only with about 15 times the data.
- **Numbers:** against the fixed margin −0.365 [−0.405, −0.328] pooled; interval score −0.130 pooled at equal rows; tie at 15×; coverage 78–87% against LightGBM's 64–79% on the same rows.
- **Built:** the clip across the full width, three number cards under it (5 of 5 scenes against the fixed margin, 1.36 against 1.72; 1.36 vs 1.49 on the same rows; 1.36 vs 1.37 at 15× the rows; mean interval score of the five scenes, `docs/LOSO5.md`). `fig_loso5.png` moved to backup B10.

### 9 · Where it stops (30 s)
- **Headline:** Better forecasts, a simple planner.
- **Layout:** headline on top; the chart on the left half, three short lines on the right half; the scope line small at the bottom. No numbers besides "3–4 s".
- **Visual:** left panel of `media/fig_t5.png` (time against closest gap per planner), plus a three-line text card (to make):
  - Simple yes/no planner: same safety, 3–4 s slower
  - Next: a planner that weighs how likely each spot is
  - Scope: robot loop and cars simulated · five real pedestrian scenes
- **Say:** the forecasts are better, but our planner is simple on purpose: it only asks yes or no, is this move inside a range? At the same safety it arrives three to four seconds after standard planners. Next, it weighs how likely each spot is. The robot loop and the cars are simulated; the five pedestrian scenes are real.
- **If asked:** the five-scene tie is proof 3 (15× fewer rows); "each body needs its own examples" is slide 10's per-body bank. Neither is a limit.
- **Built:** two cards as in the video's "Where it stops" frame. Left, the limit: two drawn bars (+4.3 s at the same closest gap, +3.0 s at the same collision rate) and "Same safety, 3–4 s slower". Right, the next step, marked "a plan, not a result". The scope line adds that TabPFN-3.5 weights are non-commercial. The left panel of `fig_t5.png` moved to backup B7 (`media/deck/fig_t5_left.png`).

### 10 · What it's for (35 s)
- **Headline:** A forecasting layer you re-target by swapping rows.
- **Layout:** headline on top; the stack diagram in the middle; the two-line buyer's guide under it ("Few examples or a sudden change: TabPFN. Plenty of familiar data: LightGBM."); "71 ms on an L4" small in a corner.
- **Visual:** `media/diagrams/12_deployment.png` after an edit (see "To make"), or a simple stack: sensors → Foresight (TabPFN ranges) → your planner → safety check.
- **Say:** Foresight sits under whatever planner a team already runs. New site, new crowd behaviour, new vehicle: add rows, no training run. One example bank per body. Few examples or a sudden change: use TabPFN. Plenty of familiar data: a well-fitted LightGBM does as well. 71 ms per replan on an NVIDIA L4. Next: a planner that uses the full range, a closed loop for cars, real robot logs.
- **Close:** "Every number in this talk comes from a script in the repo."
- **Built:** a drawn stack (sensors → Foresight → your planner → safety check) with the example bank under it (pedestrians, cars, "your robot, your site" in grey because it is the offer, not a result) and the two-line buyer's guide. `12_deployment.png` is not used.

### Closing slide (stays up during questions)
- The sentence, the repo link, the names, and four lines: the three proofs with their verdicts and "Where it stops".

### Optional 9b · TabPFN at every learned step (now backup B11)
- **Moved to the backups (22 Nov, 14:15):** the ten slides already take 285 s of the 300, and the 13:15 framing puts this point in the Q&A.
- **Headline:** A second TabPFN can pick the robot's move.
- **Layout:** headline on top; the policy GIF large on the left, the pipeline diagram smaller on the right with the three TabPFN boxes circled.
- **Visual:** `media/demo/policy_seed3403.gif`; small inset `media/diagrams/02_pipeline.png` with the TabPFN stages marked.
- **Say:** ranges, scene adaptation and the move are all TabPFN calls. On 20 new crowds the TabPFN policy drove like the planner it learned from on average. Tracker, calibration offset and hard safety check stay classical.

---

## Backup slides (for questions)

| # | Likely question | Visual | Answer in one line |
|---|---|---|---|
| B1 | "LightGBM never saw the scene. Is proof 1 fair?" | `media/fig_live_variants_cp.png` | The scene adds 10 points over the same TabPFN without it; a LightGBM refit on rows from the same scene still covers about 12 points less (offline) |
| B2 | "Did you tune until it worked?" | text | Every pass rule was committed before its run; claim C failed and claim A narrowed under those rules |
| B3 | "What data?" | `media/fig_f4_data_cards.png` | Five real scenes (ETH, Hotel, Zara 1–2, Univ), three simulated crowds, highway-env traffic |
| B4 | "How much data does TabPFN need?" | `media/fig_f4_context_size.png` | 2,000 rows tie LightGBM on 20,000 |
| B5 | "Ranges in a new place?" | `media/fig_f4_claim_a.png` | Better than LightGBM on two scenes; a tie on five when LightGBM gets all the rows |
| B6 | "Can it predict turns?" | `media/fig_intent_layer.png` | It widens ranges for people who turn (coverage 46% → 63%, real data); it does not call turns |
| B7 | "Why was the robot slower?" | `media/deck/fig_t5_left.png`, `media/fig_f4_oracle_gate.png` | At the same caution setting, all learned forecasters drive the same; even true future positions gain under 2 s in this planner |
| B8 | "Fast enough? Feature engineering? Why these choices?" | text | L4: 71 ms per replan, 318 ms scene rebuild; 21 hand-made properties: no gain; the references for each choice |
| B9 | "One decision in detail?" | `docs/walkthrough/s5_person_box.png`, `s8_planner.png` | One real moment, every stage, with the actual values (`docs/walkthrough.html`) |
| B10 | "Five real scenes, scene by scene?" | `media/fig_loso5.png` | TabPFN on 1,500 rows: ahead of LightGBM on the same rows in 4 scenes, level on Hotel; level with LightGBM on about 22,000 |
| B11 | "Is TabPFN only the forecaster?" | `media/demo/policy_seed3403.gif`, `media/diagrams/02_pipeline.png` | A second TabPFN can pick the robot's move: on 20 new crowds it drove like the planner it learned from, on average |

---

## To make or fix

- **Done in the built deck (22 Nov, 14:15):** the "Where it stops" cards, the slide 10 stack (so `12_deployment.png` no longer needs an edit), the street clip, the proof charts.
- **Pictures made for the deck are in `media/deck/`** and can be reused in Canva or anywhere else: frames of the 2:46 video (`3d_problem.jpg`, `3d_step1_watch.jpg`, `3d_step2_row.jpg`, `3d_step3_tabpfn.jpg`, `3d_step4_size.jpg`, `3d_street_stop.jpg`), the two panels of `fig_t5.png` (`fig_t5_left.png`, `fig_t5_right.png`), the last frame of each clip (`*_last.png`) and `street_clip.mp4`.
- **Before presenting:** open the deck in slide-show mode on the machine that will show it and check that the three clips move (slides 6, 7, 8) and that the street clip plays on a click (slide 5). If anything does not, present `deck/Foresight_finalist_static.pptx` or the PDF.
- **For a person to decide:** whether the repo link on the closing slide stays (the repo is private until someone makes it public).
- **Removed (22 Nov):** the outdated diagrams `08_coverage`, `10_t1_curve` and `11_tests`; use B5, B2 and B7 instead.
- **Two clips made on 22 Nov, 11:45:** `media/demo/highway.gif` (slide 7) and `media/demo/scenes5.gif` (slide 8). Each has a `_stats.json` next to it that says what is drawn and how the episode or window was picked.
