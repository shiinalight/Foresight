> **Copy of [`docs/DECK_OUTLINE.md`](../docs/DECK_OUTLINE.md)**, made by `tools/build_final.py` on 2026-11-22 14:01 (after commit `92ffc00`). Edit the source and rerun the script; do not edit this copy.

# Foresight: finalist deck outline (5 minutes)

The plan for a normal PowerPoint deck we build ourselves: for each slide, the job it does, the headline, where things go, which pictures to place, and what we say. Numbers come from `docs/STORY.md`. The 3D stills in `media/stills/` are frames from Nooshin's video (`nooshin/dev`, `media/demo_full.mp4`).

**Arc:** problem → the idea → watch it work → three proofs → where it stops → what it's for.
**Pace:** 10 slides at about 30 s each. One idea per slide. Numbers go on the slide only where a number is the point.
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

### 3 · The idea (30 s)
- **Headline:** The robot's knowledge is a table.
- **Layout:** headline on top; the row diagram on the left, the live-context diagram on the right; the line "No training step. Adapting = swapping rows." in bold under them.
- **Visuals:** `media/diagrams/03_row.png` (one person becomes one row) and `media/diagrams/05_live_context.png` (frozen bank plus the scene's newest rows, swapped every 2 s).
- **Say:** each person, at each moment, is one row: their motion, neighbours, the robot, obstacles. TabPFN reads a few thousand example rows and answers in one pass, with no training. To adapt to a new place, a new behaviour or a new body, change the rows.
- **On slide:** "No training step. Adapting = swapping rows."

### 4 · How it reads one person (15 s) · Nooshin
- **Headline:** From sensor dots to a range, in four steps.
- **Layout:** headline on top; four stills in one row, left to right in step order, each with a two-word label underneath (Watch · Row · TabPFN · Size).
- **Visuals:** a strip of four stills: `3d_step1_watching.jpg` → `3d_step2_row.jpg` → `3d_step4_tabpfn.jpg` → `3d_step5_sizing.jpg`.
- **Say:** a filter smooths the noisy track; the moment becomes a row; TabPFN predicts the correction to walking straight as a 5–95% range; the range is sized so the truth lands inside as often as promised.
- **Alternative:** play this part of the 3D video (about 40 s) and drop slide 5's clip.

### 5 · Watch it on a street (35 s) · Nooshin
- **Headline:** A street run the model has never seen.
- **Why "run":** it is a held-out episode of the same simulated street, and its ranges use a fixed example set. Do not say the model adapts here; proof 1 shows that.
- **Layout:** the clip as large as the slide allows, headline above; the caption in small type under it. With stills: three in a row.
- **Visual:** the 3D street clip from the video (about 30 s, sudden stop → zebra braking → jaywalker → rush hour). Fallback stills: `3d_street_sudden_stop.jpg`, `3d_street_zebra_brake.jpg`, `3d_street_rush_hour.jpg`.
- **Say:** narrate the events: someone stops dead, a car brakes for the zebra, a jaywalker leaves every path, rush hour changes the crowd.
- **Caption on slide:** "Simulated street; scripted robot route; ranges from TabPFN."

### 6 · Proof 1: it adapts when behaviour changes (35 s)
- **Headline:** When the crowd changed, reading the scene kept more people inside their ranges.
- **Layout:** headline on top (two lines is fine); the GIF on the left, the coverage chart on the right; one bold line of numbers under them.
- **Visuals:** `media/demo/seed3013.gif` (four robots in the same crowd, crowd turns erratic at 5 s) for 5 s, then `media/demo/calibration_seed3013_strip.png` or the right panel of `media/fig_t5.png` (coverage before and after the switch).
- **Say:** in our robot loop the crowd turns erratic five seconds in. The TabPFN that reads the last seconds kept 59% of people inside their 2 s range; LightGBM kept 41%. Ten of those points come from reading the scene alone. Ahead in all three crowds.
- **Numbers:** 59% vs 41% (+18 pts), +10 from the scene, ahead in 3 of 3 crowds (+6.6, +18.0, +19.7).

### 7 · Proof 2: same core, another body (30 s)
- **Headline:** The same core predicts highway traffic.
- **Layout:** headline on top; the highway figure across the full width (it is wide); one bold line of numbers under it.
- **Visual:** `media/demo/highway.gif` (the same road twice: frozen bank on top, bank plus this road's last seconds below; a box is where the model expects a car in 2 s and turns green or red when the car gets there). Caption it "this road: 77 vs 67 of 100; all five roads: 77% vs 63%". Still alternative: `media/fig_highway.png`.
- **Say:** we pointed the same core, same features and same models, at simulated highway traffic, with a small adapter for the cars. 1,500 example rows match LightGBM trained on 67 times more, and our ranges held the truth 86% of the time against 69–82%. When drivers turn aggressive, reading the scene adds 14 points.
- **Numbers:** 1,500 vs 100,440 rows; 86% vs 69–82%; +14 pts after the switch.

### 8 · Proof 3: more from each example (30 s)
- **Headline:** On five real scenes, TabPFN gets more from each example.
- **Layout:** headline on top; the five-scene chart across the full width; one bold line of numbers under it.
- **Visual:** `media/demo/scenes5.gif` (the five scenes side by side, people and TabPFN's boxes; caption "one 16 s window per scene", because its counters are for these windows, not the test) for a few seconds, then `media/fig_loso5.png` (each real scene held out once).
- **Say:** ETH, Hotel, Zara 1 and 2, Univ: train on four, test on the fifth. TabPFN beats the fixed-margin guess from slide 2 in all five scenes. On the same 1,500 rows it beats LightGBM in four scenes and ties on Hotel. LightGBM catches up only with about 15 times the data.
- **Numbers:** against the fixed margin −0.365 [−0.405, −0.328] pooled; interval score −0.130 pooled at equal rows; tie at 15×; coverage 78–87% against LightGBM's 64–79% on the same rows.

### 9 · Where it stops (30 s)
- **Headline:** Better forecasts, a simple planner.
- **Layout:** headline on top; the chart on the left half, three short lines on the right half; the scope line small at the bottom. No numbers besides "3–4 s".
- **Visual:** left panel of `media/fig_t5.png` (time against closest gap per planner), plus a three-line text card (to make):
  - Simple yes/no planner: same safety, 3–4 s slower
  - Next: a planner that weighs how likely each spot is
  - Scope: robot loop and cars simulated · five real pedestrian scenes
- **Say:** the forecasts are better, but our planner is simple on purpose: it only asks yes or no, is this move inside a range? At the same safety it arrives three to four seconds after standard planners. Next, it weighs how likely each spot is. The robot loop and the cars are simulated; the five pedestrian scenes are real.
- **If asked:** the five-scene tie is proof 3 (15× fewer rows); "each body needs its own examples" is slide 10's per-body bank. Neither is a limit.

### 10 · What it's for (35 s)
- **Headline:** A forecasting layer you re-target by swapping rows.
- **Layout:** headline on top; the stack diagram in the middle; the two-line buyer's guide under it ("Few examples or a sudden change: TabPFN. Plenty of familiar data: LightGBM."); "71 ms on an L4" small in a corner.
- **Visual:** `media/diagrams/12_deployment.png` after an edit (see "To make"), or a simple stack: sensors → Foresight (TabPFN ranges) → your planner → safety check.
- **Say:** Foresight sits under whatever planner a team already runs. New site, new crowd behaviour, new vehicle: add rows, no training run. One example bank per body. Few examples or a sudden change: use TabPFN. Plenty of familiar data: a well-fitted LightGBM does as well. 71 ms per replan on an NVIDIA L4. Next: a planner that uses the full range, a closed loop for cars, real robot logs.
- **Close:** "Every number in this talk comes from a script in the repo."

### Optional 9b · TabPFN at every learned step (20 s; use if the timing allows)
- **Headline:** A second TabPFN can pick the robot's move.
- **Layout:** headline on top; the policy GIF large on the left, the pipeline diagram smaller on the right with the three TabPFN boxes circled.
- **Visual:** `media/demo/policy_seed3403.gif`; small inset `media/diagrams/02_pipeline.png` with the TabPFN stages marked.
- **Say:** ranges, scene adaptation and the move are all TabPFN calls. On 20 new crowds the TabPFN policy drove like the planner it learned from on average. Tracker, calibration offset and hard safety check stay classical.

---

## Backup slides (for questions)

| # | Likely question | Visual | Answer in one line |
|---|---|---|---|
| B1 | "LightGBM never saw the scene. Is proof 1 fair?" | `media/fig_live_variants_cp.png` | The scene adds 10 points over the same TabPFN without it; a LightGBM refit on rows from the same scene still covers about 12 points less (offline) |
| B2 | "Did you tune until it worked?" | text: rule → run → result, with commit hashes | Every pass rule was committed before its run; claim C failed and claim A narrowed under those rules |
| B3 | "What data?" | `media/fig_f4_data_cards.png` or `media/fig_data_scenes.png` | Five real scenes (ETH, Hotel, Zara 1–2, Univ), three simulated crowds, highway-env traffic |
| B4 | "How much data does TabPFN need?" | `media/fig_f4_context_size.png` | 2,000 rows tie LightGBM on 20,000 |
| B5 | "Ranges in a new place?" | `media/fig_f4_claim_a.png` | Better than LightGBM on two scenes; a tie on five when LightGBM gets all the rows |
| B6 | "Can it predict turns?" | `media/fig_intent_layer.png` | It widens ranges for people who turn (coverage 46% → 63%, real data); it does not call turns |
| B7 | "Why was the robot slower?" | `media/fig_f4_oracle_gate.png` | At the same caution setting, all learned forecasters drive the same; even true future positions gain under 2 s in this planner |
| B8 | "Is it fast enough?" | text table | L4: 71 ms per replan, 318 ms scene rebuild; Apple M3 Pro: 287–321 ms; fits the 0.4 s replan |
| B9 | "Did you try feature engineering?" | text | 21 hand-made properties (time to closest approach, threat, companion): no gain; TabPFN already finds them |
| B10 | "Why these choices?" | text list | Constant velocity: Schöller 2020. Data: Pellegrini 2009, Lerner 2007. 0.4 s step: the ETH/UCY annotation grid. TabPFN: Hollmann 2025. Calibration: Gibbs & Candès 2021. Full list in `docs/VIDEO_BRIEF.md` section 4 |
| B11 | "One decision in detail?" | `docs/walkthrough/s5_person_box.png`, `s8_planner.png` | One real moment, every stage, with the actual values (`docs/walkthrough.html`) |

---

## To make or fix

- **"Where it stops" text card** (slide 9). Plain text, three lines.
- **`media/diagrams/12_deployment.png`** is out of date: it says "size decided at 02:00" and "takes over if late" (that takeover is designed, not built). Edit it to read "2,000-row bank + 250 scene rows, every 0.4 s" and "safety check: 0.4 s straight-line veto", or draw the simple stack instead.
- **Removed (22 Nov):** the outdated diagrams `08_coverage`, `10_t1_curve` and `11_tests`; use B5, B2 and B7 instead.
- **Two clips made on 22 Nov, 11:45:** `media/demo/highway.gif` (slide 7) and `media/demo/scenes5.gif` (slide 8). Each has a `_stats.json` next to it that says what is drawn and how the episode or window was picked.
- **The 3D clip for slide 5:** cut from `media/demo_full.mp4` on `nooshin/dev` from about 1:03 to 1:43; it has no voice track.
- **Starting file:** `deck/Foresight_finalist.pptx` (built by `tools/build_outline_deck.py`) has every slide below with its headline, its pictures placed and the spoken text in the speaker notes, on a plain white 16:9 page. Each picture's file name sits under it; delete those captions when done.
- **Build the deck by hand in PowerPoint**, 16:9, one plain template: one heading font, one text font, our orange (#C24E1C, the 3D video's accent) as the only accent colour. The script-built `deck/Foresight.pptx` can be a source for images only.
- **All pictures are in the repo:** `media/stills/` (3D video frames), `media/diagrams/`, `media/demo/` (GIFs play in PowerPoint), `media/fig_*.png`, `docs/walkthrough/`.
