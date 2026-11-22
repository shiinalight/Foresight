# Presentation plan: video, README, live deck

Written Sat 21 Nov, 23:40 (Berlin), while the overnight runs are going. **Status of every open result is as of 23:35.** Freeze Sun 12:30; submit by 16:00 (hard deadline 16:30).

**Update Sun 22 Nov, 02:20 (Berlin): how claims B and C landed.** T5 is done (`RESULTS.md`, "Closed loop T5 results"; 20 paired seeds, 3000–3019). Claim C landed as C3 (the oracle gate had already made T5 a stress test, D-33), and the run then showed the TabPFN planner slower than the baselines, not on their curve. A 40-seed extension (same arms and rules) was declared at 02:20 (3226e4f); the 20-seed analysis stays the confirmatory result, so quote 20 seeds. The lines below that waited for T5 are updated in place; the rest of the page is the Sat 23:40 plan.

**Story arc** (video, README, deck):
1. **What works: adaptive, calibrated prediction, and data efficiency.**
   - Reading the scene's recent rows kept TabPFN's ranges better calibrated after a behaviour change, offline (T3) and in the closed loop (T5 exploratory b): after the switch, split against frozen TabPFN, coverage 59.3% vs 49.3% (+10.0 pts [+8.4, +11.6]) and interval score 5.03 vs 6.34 (−1.31 [−1.55, −1.08]). Still short of the 80% target; simulation only (real-data check: result to be added).
   - TabPFN on 2,000 rows is about level with LightGBM on 20,000 (`results/context_size.csv`).
   - TabPFN at two chained layers (intent, then ranges) helps on real turning people: interval score −0.263 [−0.481, −0.075] (Hotel → ETH, 37 people, offline), and not in our random-intent simulator (`results/intent_layer.log`).
2. **What didn't: the closed-loop speed claim (C).** The TabPFN planner beat no baseline: all four fail Holm (p = 1.0). It was 4.3 s slower than cv m0.4 at equal minimum gap, 3.0 s at equal collision rate.
3. **Why:** not calibration. At the same caution setting (q 0.15) the split, frozen TabPFN and LightGBM planners all take about 26 s (split − frozen −0.12 s [−1.34, +1.26]; split − LightGBM +0.33 s [−1.34, +2.02], 20 paired seeds), although LightGBM's boxes are 0.63 m narrower after the switch. Better prediction did not change how this box planner drives; its caution setting did. An exploratory dial sweep on fresh seeds tests lower settings (`docs/EXPLORATORY_DIALS.md`). Next work, not done: a planner that uses the range's shape, e.g. a chance-constrained cost.

The deliverables:
- **3-minute video.** Pitch, solution and demo, with at least one of us on camera.
- **Public GitHub README.**
- **5-minute live deck.** Used only if we win the Prior Labs track.

The judges are Prior Labs. Eddie asked for "something visual you can show of how it works".

**How this page stays true overnight:**
- **Fixed spine.** The story rests on what cannot flip: the one-moment walkthrough (`docs/walkthrough.html`, built on f4 features), the explainer diagrams, and the way we tested (gates written first, a leak found and fixed, our baseline close to the published one (within 3-14%)).
- **Open results go into marked slots,** `[A]`, `[B]`, `[C]` and `[D]`. Section 8 lists every slot with its source file. Section 2.3 gives the sentence to say for each way a result can land.
- **Numbers on this page are placeholders.** Every number currently in `RESULTS.md` was measured on features before f4 (D-26 says all results are rerun on f4). None of them goes into the video or deck until the f4 file exists.

---

## 1. Title and narrative

**Title:** Foresight

**Tagline:** *Where will everyone be in two seconds, and how sure are we?* TabPFN reads a table of what people just did, and the robot keeps clear of the ranges it returns.

**The narrative in five sentences** (problem → insight → what we built → evidence → what it means):

1. **Problem.** A robot moving among people has to plan around where they will be in the next two seconds, and the plan is only as safe as the uncertainty it is given, which is hardest to trust in a place the robot has never seen.
2. **Insight.** If each person at each moment becomes one row of a table, TabPFN can read a few hundred labelled rows, including the last seconds of this very scene, and return a range for every person in one forward pass with no training.
3. **What we built.** Foresight is a sense, predict, plan pipeline whose stages (tracking, table rows, example selection, TabPFN ranges, calibration, a planner that rejects moves into predicted boxes) each have a pass/fail check written before it ran.
4. **Evidence.** On real Zurich pedestrians in a scene it never saw, TabPFN's box kept its promise (`[A-tab]`% against about 81% promised) where LightGBM's did not (`[A-lgbm]`%), and in closed-loop crowd tests reading the scene kept the ranges better calibrated after a behaviour change, but the robot was slower than standard planners, not faster.
5. **What it means.** For the first seconds in a new place, when an overconfident model is most dangerous, a tabular foundation model gives a robot usable uncertainty without retraining, while with plenty of familiar data classical models are as good, and we show where that line falls.

**Detail behind the sentences** (for the README and the speaker notes, not for reading aloud):
- **The row:** the person's own motion, their 3 nearest neighbours and the robot, all in the person's own frame. **The label:** how far they ended up from the walking-straight point.
- **The stages:** Kalman tracker (S1), person-frame rows (S2), example selection from recent, similar and rare rows (S3), TabPFN quantiles (S4), an online calibration offset (S5), occupancy (S6), planner (S7). Gates are in `docs/GATES.md`.
- **Sentence 4** is the "A1, C2" wording. For other outcomes, take its parts from section 2.3. If the f4 conformal rerun agrees with the pre-f4 one, add: "after about 20 s of local outcomes, a calibrated LightGBM catches up with narrower boxes".

---

## 2. Three-minute video

### 2.1 Recording rules (these make late number changes cheap)

- **Keep the on-camera parts free of numbers and results.** The open (0:00–0:15) and the close (2:45–3:00) are worded so they stay true whichever way the results land. Record them early, at 11:00–11:30 before the freeze, in a quiet room, with two takes each. This departs from `PLAN.md`, which puts all recording after 12:30. Agree it with the CTO session; the people are free then, since the builder session owns the last runs.
- **Numbers go in voice-over only,** over screen captures and figures. A late number change then costs a re-recorded sentence and a re-cut, not a re-shoot. Record the voice-over last, after 12:30, against the final `RESULTS.md`.
- **Burn in captions.** Judges may watch with the sound off.
- **Pace.** At most about 2.3 spoken words per second: 15 s ≈ 35 words. If a beat runs long, cut words, not beats.
- **Box definition.** In the video, "a box that should hold the truth about 8 times in 10" means only the claim A box: the 5–95% range on each axis, about 81% when both axes are counted together. Do not quote coverage over the walkthrough pictures. The planner uses a tighter 10–90% box per axis, whose two-axis target is about 64%, and quoting it as "80%" would look like a failure.

### 2.2 Script

| Time | On screen (file) | Said |
|---|---|---|
| **0:00–0:15** | **On camera:** Darshak, plaza or hall behind. Lower third: name, "Foresight · Prior Labs track" | "A robot crossing a station has to guess where everyone will be two seconds from now, and how sure it is. An overconfident guess is how robots walk into people." |
| 0:15–0:30 | Problem diagram: the explainer's "The problem" figure (`docs/explainer.html`, section 1). Export to `media/v_problem.png` (**to make**) | "Most robots assume people keep walking straight, or add one fixed margin. Learned models do better, but they learn one place. In a new place, their confidence can be badly wrong." |
| 0:30–0:55 | Pipeline diagram, then the table row: explainer "The pipeline" figure → `media/v_pipeline.png` (**to make**), then the table row from `docs/walkthrough/s3_features.csv`, shown as the walkthrough page's Stage 3 (screen capture of `docs/walkthrough.html`, **to make**) | "So we turned the scene into a table. One row is one person, at one moment, for one look-ahead: their own motion, their nearest neighbours, the robot. The answer is how far they end up from walking straight. TabPFN reads rows like these as examples, in one pass. No training." |
| 0:55–1:20 | One real decision, stage by stage: `docs/walkthrough/s4_context.png` → `s5_person_box.png` → `s6_all_boxes.png` → `s8_planner.png` (slow pan, about 6 s each) | "Which examples? What people here did in the last few seconds, similar moments from other crowds, and rare ones like sudden stops. Out comes a range for each person, one and two seconds ahead. The planner rejects any move that enters a range, and takes the one with most progress." |
| 1:20–1:45 | **Demo:** bird's-eye render, same crowd for each planner: stop-and-go, reactive, velocity obstacles, and the TabPFN planner T5 ran (tabpfn_split q0.15). Every seed switches into erratic at 5 s. Showcase seed **3013** (picked by the rule in `RESULTS.md`); render: `media/demo/seed3013.gif`. Caption: "typical crowd (median), not cherry-picked". Fallback: `docs/walkthrough/s9_ego_path.png` plus `media/demo_hotel_tabpfn.gif` re-rendered on f4 | "Here is the full loop in a simulated crossing crowd. Same crowd for every robot: stop-and-go, reactive, velocity obstacles, and ours, with the boxes it planned around. Five seconds in, the crowd changes behaviour. This crowd was picked by a rule, not by us." |
| 1:45–2:10 | Claim A chart: joint box coverage in a new place, both directions, five context samples, target line at 81%. `media/fig_claim_a_f4.png` (**to make** from `results/claim_a_f4.csv`). Then `media/fig_conformal_hotel_to_eth.png` (**regenerate on f4**) | Variant A1 (section 2.3): "On real Zurich pedestrians, moving to a scene the model never saw, TabPFN's box held the truth `[A-tab]` percent of the time, where about 81 was promised. LightGBM's held it `[A-lgbm]` percent. The other way round, TabPFN was more cautious than needed. After 20 seconds of local outcomes, a calibrated LightGBM catches up, with narrower boxes." Say the last sentence only if the f4 conformal rerun agrees (section 2.4) |
| 2:10–2:30 | Closed-loop evidence, as landed (C3 plus T5): the in-loop claim B numbers (`results/inloop_claimb.csv`) and the T5 points table (`RESULTS.md`, time / minimum gap / collision rate per planner). A T5 figure is not made yet | (42 words) "In 20 fresh crowds that turn erratic, reading the scene kept TabPFN's ranges better calibrated than the frozen model: ten points more coverage. But our robot was slower than standard planners at comparable safety: wider, more honest ranges made it cautious." |
| 2:30–2:45 | "Where it breaks" card: five short lines on a plain background, `media/v_limits.png` (**to make**) | "Where it breaks. Once local data exists, a tuned model's boxes are narrower. Turns are under-predicted. Our crowd is simulated. The weights are non-commercial. And speed on robot hardware is `[D-sentence]`." |
| **2:45–3:00** | **On camera:** Darshak and Nooshin. End card: repo URL, "every number comes from a script in the repo" | "Foresight is open source: every number here comes from a script in the repo. Next, real robot data and timing on robot hardware. Thank you, Prior Labs." |

Beat lengths: 15 + 15 + 25 + 25 + 25 + 25 + 20 + 15 + 15 = 180 s.

**If the video runs long, cut in this order:**
1. The second figure in the 1:45 beat.
2. The B sentence in the 2:10 beat.
3. One walkthrough stage (s6).

**Never cut:**
- the on-camera open and close;
- the demo;
- "where it breaks".

### 2.3 Variants: what to say for each way the open results land

**Rule for every row:** keep the two claims separate, as `RESULTS.md` does. A planner win is not a TabPFN win. The win that matters to Prior Labs is TabPFN against LightGBM inside the same planner. On f4 in the crossing crowd, LightGBM on all 73k rows already has lower error than TabPFN Fast-1 on 1,000 rows (0.606 against 0.689 m, D-27). So a planner win may come from prediction in general rather than from TabPFN.

| Result | If it lands like this | Say (voice-over) | Show | Beat change |
|---|---|---|---|---|
| **A** (claim A on f4, `tools/claim_a.py`) | **A1 holds:** TabPFN is at or above target (not overconfident) in both directions, LightGBM is clearly below target in at least Hotel → ETH, and the spreads over 5 context samples don't overlap. On f1 this was the shape: TabPFN 83% and 92% (over-wide going to Hotel), LightGBM 46–62% going to ETH but 76–80% going to Hotel | The 1:45 line as written | `media/fig_claim_a_f4.png` | none |
| | **A2 narrows:** TabPFN is closer to its promise, but the spreads overlap, or the gap holds in one direction only, or for moving people only | "In a new place, TabPFN came closest to its promise in `[n]` of `[m]` comparisons. The gap is smaller than we first measured: our own tests found a feature that leaked walking direction, and we reran everything without it." | same chart, gap annotated | Keep 25 s. The leak story is evidence of rigour, so keep it in |
| | **A3 gone:** LightGBM is as well calibrated as TabPFN on f4 | "On fixed features, both models keep their promise in a new place. TabPFN does it reading 1,500 rows with no training, matching a LightGBM trained on all `[N]` rows." Use only if the f4 error and coverage numbers show that parity. Otherwise say "here TabPFN had no edge", and give the time to C or B | same chart | Shrink the 1:45 beat to 15 s; the conformal figure goes |
| **B** (dense real-data adaptation with person-level intervals; T3 held-out behaviour) | **B1 confirmed:** reading the new scene's rows cuts early error, and the intervals separate | "Reading the new place's last seconds cut error by `[B-pct]` percent in the first five seconds, with no retraining." | `media/fig_incontext_hotel_to_eth.png` (f4) | Joins the 2:10 beat with C2 or C3, never with C1 (too many words); with C1 it goes to the README and deck only |
| | **B2 not confirmed:** the intervals overlap, or T3 shows no recovery | Say nothing about adaptation in the video. In the README: "promising in a trial, not confirmed" | — | — |
| **C** (oracle gate D-21, then T5 in the crossing crowd, 40 fresh seeds, pre-registered rule) | **C1 gain:** a learned planner beats every baseline by ≥ 2 s time or ≥ 1 s intrusion at matched safety | "In 40 fresh crowds, scored by a rule fixed in advance, our robot `[C-gain]` than the best standard planner." Then one of: **C1a** (TabPFN ≥ LightGBM in the same planner) "and TabPFN was the best of the predictors"; **C1b** (LightGBM ≥ TabPFN) "and LightGBM did as well in the same planner, so the gain is prediction, not TabPFN alone" | `media/fig_t5.png` | 2:10 beat stays 20 s |
| | **C2 parity:** learned planners sit on the baselines' curve | "In 40 fresh crowds, our robot landed on the same speed-versus-closeness curve as a well-tuned standard planner. So we don't claim it beats standard navigation." | `media/fig_t5.png` | Keep 20 s: add B1 if confirmed, else hold on the figure |
| | **Landed (Sun 02:20): C3, then T5 ran.** The oracle gate failed (D-33); T5's primary failed against all four baselines, slower at comparable safety; in-loop claim B held | The 2:10 line in section 2.2 | `results/inloop_claimb.csv`; T5 points table | Keep 20 s |
| | **C3 untestable:** the oracle gate fails, no T2 setting qualifies, or T5 is not finished by 10:30 | "Even a planner that knew the true future couldn't beat standard planners here by our margin. So the loop is a demonstration, not evidence." If T5 is unfinished: "Our confirmation run did not finish in time, so the loop is a demonstration, not evidence." | oracle-vs-baselines curve if it exists, else none | Keep 20 s: B1 first if confirmed, then this sentence. The demo beat keeps its caption |
| **D** (latency on target hardware: Nooshin's Mac, `tools/latency.py`) | **D1 measured** | "`[D-ms]` milliseconds per plan on an Apple laptop GPU", for `[M-tier]` with `[M-rows]` cached rows | — | inside "where it breaks" |
| | **D2 not measured** | "not yet measured" | — | — |

**Fifth sentence of the narrative, per variant:**
- **A1 or A2:** as written.
- **A3:** "a tabular foundation model matches a fully trained model with a fraction of the data and no training, and we show where each one fails".
- **C1a:** add "and in our crowd tests that turned into a faster or more comfortable robot".

**When several results land well at once.** The pace limit decides. With A1, C1 and B1 all true:
- the 1:45 beat says A1 (about 58 words in 25 s);
- the 2:10 beat says C1 with C1a or C1b (about 42 words in 20 s);
- B1 goes to the README and slide 9's notes.

With A3, the 1:45 beat drops to 15 s, and the 10 s freed go to the 2:10 beat, which can then carry C1 and B1 together (about 50 words in 30 s).

### 2.4 Lines that depend on the f4 reruns

These sentences rest on pre-f4 results. Each has a replacement if its rerun disagrees.

| Line (where) | Rests on | If the f4 rerun disagrees, say |
|---|---|---|
| "After 20 seconds of local outcomes, a calibrated LightGBM catches up, with narrower boxes" (1:45 beat, sentence 4 detail, slide 7) | `tools/conformal_eval.py` on f4 | LightGBM doesn't catch up: "even after 20 seconds in the scene, TabPFN's calibrated boxes were `[as narrow / narrower]`". TabPFN is overconfident on arrival too: drop the line and slide 7 |
| "TabPFN's edge is the first seconds" (slide 7 title) | same | the title becomes the measured statement, e.g. "Calibrated boxes after 20 s in the scene" |
| "Once local data exists, a tuned model's boxes are narrower" (limits beat) | same | replace with "Our people are simulated and don't react to the robot like real ones" |
| "Turns are under-predicted" / "Both miss turns" (limits beat, slide 9) | `tools/failure_modes.py` on f4 | name whichever case the f4 figure shows worst (e.g. "people who stop"), or drop the clause |
| "New place or new behaviour, few rows: TabPFN is safer" (slide 9) | claim A on f4, and T3 | A3: "New place: both kept their promise; TabPFN needed far fewer rows". T3 fails: drop "or new behaviour" |

---

## 3. Five-minute live deck (only if we win the track)

**Builder.** Build it from `tools/build_deck.py` and keep its pattern: kicker, title, one visual, at most a few words, explanation in the speaker notes, placeholder slides for missing figures. That script builds the collect-everything deck. The 5-minute final is filtered from it, so these 10 slides are the filter.

**Pace.** About 30 s per slide; rehearse twice with a timer.

| # | Kicker · title | The one visual | The one sentence | Number(s), with source | Speaker note |
|---|---|---|---|---|---|
| 1 | HackLab Berlin · Prior Labs track · **Foresight** | `docs/walkthrough/s6_all_boxes.png` (everyone's boxes at one moment) | "Where will everyone be in two seconds, and how sure are we?" | — | Names: Darshak Panseriya, Nooshin Pourkamali. One prediction-and-planning layer for a robot among people |
| 2 | Problem · **A robot needs the next two seconds, and how sure it is** | `media/v_problem.png` (**to make**, explainer figure 1) | "In a new place, an overconfident forecast is the dangerous one." | Constant velocity is off by 0.37 m (ETH) and 0.25 m (Hotel) on average at +2 s on real people, and 0.93 m in our plaza crowd (`docs/DATA.md`) | Walking-straight prediction is a strong baseline on real data (Schöller et al. 2020). The hard part is knowing how wrong it might be |
| 3 | Approach · **Scene → table → TabPFN → ranges → robot** | `media/v_pipeline.png` (**to make**, explainer "The pipeline") | "One row per person, moment and look-ahead; TabPFN reads examples, no training." | — | Target is the deviation from walking straight, in the person's own frame. Real-time setup: `[M-tier]`, `[M-rows]` cached rows, `[M-retrieval]` (decided at 02:00; D-5 currently says TabPFN-3.5 with 1 estimator and about 1,000 rows) |
| 4 | How it works · **One real decision, stage by stage** | `media/v_walkthrough_strip.png` (**to make**: s4 context, s5 box, s8 planner side by side) | "These are the real values at one moment, picked by a fixed rule." | 894 example rows: 80 recent, 588 similar, 113 rare, 113 random; 16 of 16 people predicted (`docs/walkthrough.html`, f4, seed 6001) | Moment rule is on the walkthrough page. Context is refreshed every 2 s from completed rows only |
| 5 | Method · **Tested so it can fail** | `media/v_how_we_test.png` (**to make**: explainer "How we test", with T2–T5 statuses updated) | "Every pass condition was written before the run; a failed test stopped the work built on it." | Our constant-velocity baseline at 4.8 s: ETH 1.12 m, Hotel 0.46 m, against 1.15 and 0.51 m published (D-26, Schöller et al.) | Leak found and fixed (D-20). The plaza failed as a navigation test (T1, D-19), so we built a harder crowd chosen by a rule from the baselines alone |
| 6 | Result · **Honest ranges in a place it never saw** | `media/fig_claim_a_f4.png` (**to make**) | Variant A1/A2/A3 sentence | `[A-tab]`, `[A-lgbm]`, `[A-band]` joint coverage, with the spread over 5 context samples, which is not a confidence interval (`results/claim_a_f4.csv`) | Box = 5–95% per axis, target about 81%. Always show width next to coverage: wide boxes cover trivially (Gneiting and Raftery 2007) |
| 7 | Result · **The edge is the first seconds** | `media/fig_conformal_hotel_to_eth.png` (**regenerate on f4**) | "After about 20 s of local outcomes, a calibrated LightGBM catches up, with narrower boxes." (if the f4 rerun agrees; else section 2.4) | Coverage on arrival and after 20 s, width after 20 s (`results/conformal_hotel_to_eth.csv`, f4) | Online offset (Gibbs and Candès 2021) gives long-run coverage, not a per-step guarantee |
| 8 | Closed loop · **Calibrated, but slower** | still from `media/demo/seed3013.gif` (showcase seed 3013) | The 2:10 line | T5 points table (time, minimum gap, collision rate per planner; `results/t5_primary.csv`, 20 paired seeds): primary fails Holm against all four baselines, 4.3 s / 3.0 s slower than cv m0.4 at equal gap / collision rate. In-loop claim B: +10.0 pts coverage, interval score −1.31 against frozen (`results/inloop_claimb.csv`) | Why slower: not calibration; split, frozen TabPFN and LightGBM all take ~26 s at q 0.15 (paired differences within ±1.3 s); the caution setting sets the speed. Next (not done): a planner that uses the range's shape, e.g. a chance-constrained cost |
| 9 | Honest framing · **When to use which** | `media/fig_failure_hotel_to_eth.png` (**regenerate on f4**) | "Familiar place, many rows: LightGBM is as good. New place or new behaviour, few rows: TabPFN is safer. Both miss turns." (each clause per section 2.4) | Box width TabPFN against LightGBM (`results/claim_a_f4.csv`); turn coverage (`results/failure_hotel_to_eth.csv`, f4) | Limits: two real scenes, simulated crowd, no robot in the real data, no safety guarantee (Lindemann et al. need agents that ignore the robot) |
| 10 | Deployment · **On a robot** | `media/v_on_robot.png` (**to make**: explainer "On a real robot") | "A small cached context in the loop, constant velocity as the safety floor, the cloud never in the loop." | 219 M parameters (TabPFN-3.5), 83.5 M (Fast) (`docs/DEPLOYMENT.md`); `[M-tier]` and `[D-ms]` if measured | 3.5 weights are non-commercial: research prototype; product = commercial licence or Apache v2 weights. Next: real robot logs, GPU timing |

**Changes to `tools/build_deck.py`** (requests to the builder session, its owner; this page doesn't edit it):
- **`_closed_table`** filters `box_q == 0.05`, the setting that freezes the robot (D-6). Replace it with the pre-registered headline rule (`RESULTS.md`, closed loop) on the T5 files.
- **Demo slide note** quotes "94%, median error 0.26 m" from the pre-fix render. Replace it from the f4 re-render.
- **Remove the obsolete slides and images:**
  - "Any agent: people, cars, robots" and "Next": cars and the mixed street were cut (CTO review);
  - "Robot crossing a crowd: two planners": `closed_crowd_1001_cv_vs_tabpfn.png` is superseded.
- **Add slides:** the walkthrough strip, the claim A chart, the T5 curve and "How we test".
- **"Numbers" slide** reads `results/eval.json` (pre-f4). Point it at the f4 output, or drop it from the 5-minute filter.

---

## 4. README for the public repo

**Goal.** A judge who opens the repo cold understands what it does in 30 seconds. In 5 minutes they can see how it works and what the evidence is.

GitHub doesn't render `.html` files, so the README embeds PNGs and links to the HTML pages for a local open, or through GitHub Pages if the team enables it.

| # | Section | Must show |
|---|---|---|
| 1 | **Title, tagline, one picture** | `docs/walkthrough/s6_all_boxes.png` or a four-panel still. Two sentences: what it predicts, what the robot does with it. Link to the video |
| 2 | **Start here** | The path below |
| 3 | **What we claim, and its status** | The four claims A–D as in `PLAN.md` section 2, one line each, with the final status word: supported / narrowed / not shown / partly. The honest framing line: "when to use which" |
| 4 | **How it works** | Pipeline figure (`media/v_pipeline.png`), then the nine walkthrough stages as thumbnails linking to `docs/walkthrough.html`. One sentence each for row, target, examples, TabPFN, calibration, planner |
| 5 | **Results, with intervals** | Three tables, each with its source file and feature version (f4); see below |
| 6 | **Where it breaks** | The limits from `RESULTS.md` plus the video's five lines. Also: T1 (the plaza could not separate planners), the f1 leak and its fix, simulated-vs-real differences (`docs/DATA.md`: simulated people turn in 36% of 2 s windows against 2–3% real) |
| 7 | **Architecture** | The current layout block, and the S0–S8 stage table from `PLAN.md` section 3 with each stage's gate and result |
| 8 | **How to run** | Install, token as an environment variable (never committed), data download from OpenTraj (raw data not redistributed unless its licence allows), then one command per result table and figure (`tools/claim_a.py`, `tools/conformal_eval.py`, `run_closed.py`, `tools/walkthrough.py`), `python -m pytest -q`. Note which runs need the hosted API |
| 9 | **Deployment and licence** | Model tiers, sizes, the non-commercial weights and what a product would need (`docs/DEPLOYMENT.md`). Timings: measured on target hardware, or "not yet measured" |
| 10 | **References** | The 8–10 that carry claims, from `docs/REFERENCES.md`, plus a link to the full list. TabPFN-3.5 is a vendor report, not peer reviewed. LoCalPFN covers classification only. No safety guarantee (Lindemann et al., Assumption 1) |
| 11 | **Prepared before the event** | See below |
| 12 | **Team** | Names, roles, what each built (Nooshin: crowd simulator, 3D viewer, Mac timings) |

**The three results tables:**
1. **Prediction on real pedestrians** (claim A). Model, rows seen, error at +2 s, joint coverage of the 5–95% box (target about 81%), 10–90% width and pinball. Each value is the mean over 5 context samples with their spread (label it "spread over context samples", not a confidence interval), both directions (`results/claim_a_f4.csv`). Add person-bootstrap intervals if the builder session produces them.
2. **Calibration on arrival and after 20 s** (`results/conformal_*.csv`, f4).
3. **Closed-loop headline per the pre-registered rule.** Time to goal, intrusion, collisions and success, with 95% paired-seed bootstrap intervals over the analysed seeds (20 for T5: 3000–3019), or "none qualifies". Add one line stating that a planner win is not a TabPFN win.

Also show the baseline-matches-published table (D-26). Drop single-seed rows, or label them "one seed, indicative".

**Prepared-before-the-event disclosure** (organisers asked for it):
- Keep the current README wording: "Generic utilities only: a Kalman filter, a constant-velocity baseline and timing code (`tools/check_tabpfn.py`)." **Confirm** which files hold that code now. `TEAM-PLAN.md` names `traj_kit.py`, which is no longer in the repo.
- Name the reference note `docs/tabpfn-hackathon-strategies.md` if it was written before the event start (and `av-tabpfn-brainstorm.md`, which `TEAM-PLAN.md` mentions but which is not in the repo). **Check the dates: `TEAM-PLAN.md` says the notes were copied in at setup.**
- Optionally, state how AI coding assistants were used.

**Start here path** (replaces the current one, which points to the partly superseded `docs/CTO-REVIEW.md`):
1. The video (link).
2. `docs/explainer.html`: picture-first explainer, about 5 minutes.
3. `docs/walkthrough.html`: one real decision, stage by stage, with the actual values.
4. `RESULTS.md`: every number, its limits, the pre-registered rule.
5. `docs/DECISIONS.md`: every chosen value and the test behind it.
6. `PLAN.md`: claims and status. Then the code, starting at `foresight/core/`.

**Before making the repo public:**
- **Commit the pages.** `docs/walkthrough.html` and `docs/walkthrough/` are untracked at 23:35; commit them or the README links break.
- **Secrets.** No secrets in the history (`TABPFN_TOKEN`, `.env`, `.claude/settings.local.json`).
- **Large files.** No files over GitHub's limits (`results/trace/*.pkl`, GIFs).
- **Repo name.** The repo is `shiinalight/Robodetect`; set its description to "Foresight: …" so the name doesn't confuse.

---

## 5. Visual inventory

**Status key:**
- **current:** usable as is.
- **regenerate on f4:** made on earlier features; D-26 says all results are rerun on f4, so `MANIFEST.csv`'s "current" is not enough.
- **to make:** does not exist yet.

**Owner key:**
- **builder:** builder session (loop, `tools/`, `RESULTS.md`).
- **CTO:** CTO session or its subagents (explainer, walkthrough page, docs).
- **people:** Darshak and Nooshin.

Deadlines are relative to the 12:30 freeze (F); F−2h is 10:30.

| # | Visual | Used in | Source script / file | Status | Owner | Deadline |
|---|---|---|---|---|---|---|
| 1 | Problem figure | video 0:15, deck 2 | `docs/explainer.html` §"The problem" (inline SVG) → screenshot to `media/v_problem.png` | to make (results-independent) | CTO | F−4h (08:30) |
| 2 | Pipeline figure | video 0:30, deck 3, README | explainer §"The pipeline" → `media/v_pipeline.png` | to make | CTO | F−4h |
| 3 | Scene-to-table and examples figures | README, optional video | explainer §"From scene to table to prediction" → `media/v_table.png` | to make | CTO | F−4h |
| 4 | "What each method assumes" (five panels) | README | explainer §"What each method assumes" → `media/v_assumptions.png` | to make | CTO | F−3h |
| 5 | "How we test" T1–T5 | deck 5, README | explainer §"How we test", statuses updated after T5 → `media/v_how_we_test.png` | to make (T5 status now known: primary failed, in-loop B held) | CTO | F−1h |
| 6 | "On a real robot" | deck 10, README | explainer §"On a real robot" → `media/v_on_robot.png` | to make | CTO | F−3h |
| 7 | Walkthrough stages s1–s9 | video 0:30–1:20, deck 1 and 4, README | `tools/walkthrough.py` → `docs/walkthrough/*.png`, `docs/walkthrough.html` | current (f4, seed 6001), but built from a dirty tree at `d25281a`. Rerun once on a clean commit, and again if T2 changes the crowd settings the walkthrough uses | CTO | F−2h |
| 8 | Walkthrough strip (s4 + s5 + s8) | deck 4 | compose from #7 → `media/v_walkthrough_strip.png` | to make | CTO | F−1h |
| 9 | **Four-panel bird's-eye video** (stop-and-go, reactive, VO, TabPFN live; switch into erratic at 5 s; live counters) | video 1:20 (the demo), deck 8 still, README still | traces of the T5 showcase seed 3013 (`docs/VISUALS.md` §2; `render_closed.py` extended) → `media/demo/seed3013.gif` | rendered (c241ffe) | builder | F−2h |
| 10 | Claim A chart: joint coverage, both directions, 5 seeds, moving vs standing | video 1:45, deck 6, README | plot from `results/claim_a_f4.csv` (`tools/claim_a.py` writes the CSV only) → `media/fig_claim_a_f4.png` | to make | builder (or a CTO subagent, plot only) | F−3h |
| 11 | Calibration on arrival / in the scene | video 1:45, deck 7, README | `tools/conformal_eval.py` (API) → `media/fig_conformal_{hotel_to_eth,eth_to_hotel}.png` | regenerate on f4 | builder | F−3h |
| 12 | Adaptation with person-level intervals (claim B) | video 2:10 (B1/C3), README | `tools/incontext_eval.py` + `tools/adapt_ci.py` → `media/fig_incontext_*.png`, `media/fig_*_ci.png` | regenerate on f4 (pilot figures are superseded) | builder | F−3h |
| 13 | T3: held-out behaviour (calm → erratic) | README, deck 9 notes | `run_adapt.py` on `data/crowd_c2e` | to make; T3 ran on f1, so rerun on f4 before showing | builder | F−2h, or drop |
| 14 | **T5 curve** + headline table (time vs intrusion, 95% paired intervals, switch and no-switch split) | video 2:10, deck 8, README | `results/t5_primary.csv`, `results/inloop_claimb.csv`; script in the style of `tools/t1_gate.py` → `media/fig_t5.png` | to make (T5 analysed; T5 reports minimum gap and collision rate, not intrusion) | builder | F−2h |
| 15 | Oracle-gate curve | video 2:10 (C3 only), README | D-21 runs → `media/fig_oracle.png` | to make if C3 | builder | F−2h |
| 16 | Limits card | video 2:30 | text on a plain background → `media/v_limits.png` | to make | CTO | F−1h |
| 17 | Title and end cards (repo URL, names) | video | → `media/v_title.png`, `media/v_end.png` | to make | people | F |
| 18 | Ranges on real people (clip) | video fallback for 1:20, README | `demo.py` → `media/demo_hotel_tabpfn.gif` / `.png` | regenerate on f4 (superseded: pre-B9) | builder | F−3h |
| 19 | Four predictors, same moment | README | `compare.py` → `media/compare_hotel.png` | regenerate on f4 | builder | F−3h, optional |
| 20 | Error by horizon; calibration curve | README | `evaluate.py` → `media/fig_error_by_horizon.png`, `media/fig_calibration.png` | regenerate on f4 | builder | F−3h |
| 21 | When to use which (failure modes) | deck 9, README | `tools/failure_modes.py` → `media/fig_failure_hotel_to_eth.png`, `media/fig_failure_crowd_to_crowd.png` (crowd → `crowd_cross`) | regenerate on f4 | builder | F−2h |
| 22 | Rows from this place are worth more | README | `tools/check_efficiency.py` → `media/fig_rows_place.png` | regenerate on f4 (cheap, LightGBM only) | builder | F−2h, optional |
| 23 | Data efficiency | — (drop: one seed) | `evaluate.py` → `media/fig_data_efficiency.png` | regenerate on f4 only if used | builder | optional |
| 24 | Model tier grid | README deployment | `tools/tier_grid.py` → `media/fig_tier_grid.png` | regenerate on f4, or caption "pre-fix features (f1)" | builder | F−2h |
| 25 | T1 gate (default plaza could not separate planners) | README "where it breaks" | `tools/t1_gate.py` → `media/fig_t1_gate.png` | current as a historical record (pre-f4, default plaza). **Missing from `MANIFEST.csv`: add it** | builder | F−2h |
| 26 | Data cards | README, deck 2 notes | `tools/data_cards.py` → `media/fig_data_overview.png`, `media/fig_data_scenes.png` | current for tracks; mode labels predate f3 (D-22) and `crowd_cross` is missing. Regenerate | CTO | F−3h |
| 27 | 3D viewer clip | stretch only | `tools/export_3d.py` → `media/sample_3d_crowd_1001.json` + Nooshin's three.js viewer | sample is current but not a showcase seed (`lgbm_all`); export the T5 showcase trace if used | people (Nooshin) | F, stretch |
| 28 | On-camera open and close | video 0:00, 2:45 | phone or laptop camera | to make (results-independent) | people | record 11:00–11:30, before F |
| 28b | Intent chain (two chained TabPFN layers) | README, deck 9 notes | `tools/intent_layer.py` → `media/fig_intent_layer.png` | current (offline; real Hotel → ETH and simulated crowd) | builder | — |
| 29 | Superseded clips | — | `media/closed_crowd_1001_cv_vs_tabpfn.gif` / `.png`, `media/fig_adapt*.png` | superseded: move out of `media/` before the repo goes public | builder | F |

---

## 6. Explaining it to a third person

**Three one-liners** (use these words in the video, deck and Q&A):

1. **TabPFN reading examples.** *"We don't train it, we show it examples."* TabPFN has already learned how to learn from a table. Hand it a few hundred solved cases ("this person, in this situation, ended up here") and it answers new cases straight away. It is like a colleague reading a worked-examples sheet, not going back to school. When the robot arrives somewhere new, we swap in that place's latest examples; the model itself never changes.
2. **Calibrated ranges.** *"A forecast that says 8 in 10 should be right 8 times in 10."* Like a weather forecast, our box makes a promise about how often the person lands inside it, and we check that promise against what people actually did. The dangerous forecast is the overconfident one: it tells the robot a path is clear when it isn't.
3. **The planner.** *"Don't walk where someone will be."* It is what you do crossing a busy station. Around each person the robot draws a "probably here in two seconds" box, then takes the fastest move that touches none of them. A bigger box makes a more cautious robot, and too big a box makes it freeze.

**Five terms to define or avoid:**

| Term | Do | Plain words |
|---|---|---|
| In-context learning | Define once, then say "reads examples" | "It reads labelled example rows and answers in one pass; nothing is trained" |
| Quantile, 10–90% interval | Avoid in the video | "a range that should hold the truth 8 times in 10" |
| Coverage, calibration | Define | "how often the person actually landed inside the box"; "calibrated = keeps its promise" |
| Conformal (adaptive) prediction | Avoid in the video; define in the README | "one correction number, nudged up or down from what actually happened" |
| Pre-registration, gate | Define | "the pass rule, written down before the run" |

**Words to keep out entirely:**
- **"guarantee"** and **"safe"** as absolutes. Say "keeps clear of", "fewer close passes", "honest ranges".
- **"state of the art".**
- **"any robot"**: the second domain was cut.
- **"real-time"**: unless D1 is measured.
- **"beats"**: unless the variant row says so.

---

## 7. Risks on submission day, and the fallback for each

| Risk | Sign | Fallback |
|---|---|---|
| T5 not finished or not analysed by 10:30 | No T5 files, or no analysis, at 10:00 | **Resolved Sun 02:20: T5 is analysed.** Was: Variant C3. The demo uses any traced seed, captioned "demonstration seed", with no closed-loop numbers. The 2:10 beat goes to B1 or A |
| Claim A flips on f4 | `results/claim_a_f4.csv` shows A2 or A3 | Use the variant rows. The leak-and-fix story carries the rigour message. Don't reshoot the on-camera parts: they hold no numbers |
| Numbers change after the voice-over is recorded | A late run lands after 13:30 | Numbers live only in voice-over and captions. Re-record one sentence and re-cut. After 14:30, freeze the numbers in the video and let the README carry any later correction, with a dated line |
| Four-panel render fails or looks wrong | No `media/v_four_panel.mp4` by F−2h | Screen-record `docs/walkthrough.html` scrolling through s1–s9, plus the f4 re-render of `media/demo_hotel_tabpfn.gif`. Use the two-panel `render_closed.py` only if it is re-run on a current trace |
| API credit limit hit (30M per day stop) | `results/api_usage.csv` near the stop | Local Fast-1 where it equals the default (Hotel → ETH, D-17). Otherwise show the last f4 run that finished and say which tier was used |
| Recording problems: noise, light, stumbles | First take unusable | Record the on-camera parts early (11:00), two takes, quiet room, phone close to the mouth, script on a laptop beside the lens. The voice-over can be done separately and replaced |
| Video over 3:00 | Cut runs long | Cut in the order of section 2.2. Never cut the open, demo, "where it breaks" or close |
| Upload too slow | YouTube still processing HD at 15:30 | Upload by 15:15. Unlisted. Open the link in a private window. Google Drive with "anyone with the link" as the backup link |
| Repo not public, or links broken | README links 404 on `main` | Promote to `main` by 12:30 (`PLAN.md` timeline). Commit `docs/walkthrough*`. Check every README link from a logged-out browser. Make the repo public by 14:00 |
| Secret or oversized file in the repo | History scan finds a token or a file over 100 MB | Check before going public. If a token is found, revoke it at Prior Labs first, then clean the history. Don't just delete the file |
| HTML pages don't render on GitHub | Judges see source | The README embeds PNG exports. GitHub Pages on `/docs`, if the team enables it, gives a link to the rendered explainer |
| Licence wording challenged | Q&A | "Research prototype on non-commercial 3.5 weights. A product needs the commercial licence or the Apache v2 weights; v2 is in our tier grid" |
| A judge asks "is the gain TabPFN or just prediction?" | Q&A | The separate TabPFN-vs-LightGBM line (C1a or C1b), plus claim A, where the comparison is direct |
| Live deck fails (projector, no internet) | Live slot | PPTX and PDF on two laptops and a USB stick. Embed the video locally, not streamed. Rehearse 5:00 twice |
| Fatigue after the overnight run | Slow decisions in the morning | Decision times are fixed in `PLAN.md`. Script and visual list are written now (this page). Someone fresh reads the final video once against `RESULTS.md` before upload |
| Form problem | Tally form rejects or needs fields we lack | Open the form at 12:30 to see the fields. Submit at 16:00, not 16:25 |

---

## 8. Number slots (fill at 10:30 from the files, never from memory)

| Slot | What | Source file | Value at 23:35 |
|---|---|---|---|
| `[A-tab]`, `[A-lgbm]`, `[A-band]` | Joint coverage of the 5–95% box at +2 s, Hotel → ETH (and ETH → Hotel), mean and spread over 5 context samples (not a confidence interval) | `results/claim_a_f4.csv` | Not yet run. Pre-f4 (f1): TabPFN 83%, LightGBM 46% / 62%, band 70% (Hotel → ETH); 92%, 76% / 80%, 91% (ETH → Hotel) |
| `[A-width]` | 10–90% width, TabPFN against LightGBM | same | f1: 0.91–0.96 against 0.54–0.60 m |
| conformal | Coverage raw / on arrival / after 20 s; width after 20 s | `results/conformal_*.csv` (f4) | Pre-f4: TabPFN 62% / 73% / 80% Hotel → ETH; calibrated LightGBM narrower after 20 s |
| `[B-pct]` | Early-error reduction from reading the scene, first 5 s, with person-level intervals | f4 rerun of `tools/incontext_eval.py` + `tools/adapt_ci.py` | Pilot only: 0.425 against 0.536 m (frozen LightGBM), no intervals. Still unfilled: the in-loop claim B result is a calibration result (coverage, interval score), not this slot; real-data check (`tools/realdata_claimb.py`): result to be added |
| `[C-gain]` / `[C-sentence]` | T5 headline per the pre-registered rule | `results/t5_primary.csv` | T5 not launched (planned 03:30). **Landed Sun 02:20:** no gain; tabpfn_split beats no baseline (all four fail Holm, p = 1.0); 4.3 s slower than cv m0.4 at equal minimum gap, 3.0 s at equal collision rate. `[C-sentence]` filled in section 1 |
| in-loop B | Split vs frozen TabPFN after the switch, q 0.15, +2 s | `results/inloop_claimb.csv` | Landed Sun 02:20: coverage 59.3% vs 49.3% (+10.0 pts [+8.4, +11.6]); interval score 5.03 vs 6.34 (−1.31 [−1.55, −1.08]); median error −0.057 m [−0.094, −0.021] |
| intent | Intent chain vs single TabPFN range model, turning people, Hotel → ETH | `results/intent_layer.log` | Landed Sun 02:16: interval score 3.59 → 3.33 (−0.263 [−0.481, −0.075]; 73 rows, 37 people); overall −0.022 [−0.057, +0.009]; balanced accuracy 0.49 (sim 0.30, not adopted) |
| S0 cross | Constant velocity against LightGBM and TabPFN, crossing crowd, f4 | D-27 | CV 0.745 m; LightGBM all rows 0.606 m (18.7% better); TabPFN Fast-1, 1,000 rows, 0.689 m (7.5% better) |
| S1 baseline | Constant velocity at 4.8 s against the published value | D-26 | ETH 1.12 m, Hotel 0.46 m against 1.15 and 0.51 m |
| `[M-tier]`, `[M-rows]`, `[M-retrieval]` | Real-time TabPFN setup: model and estimators, context size, how rows are retrieved | 02:00 decision row in `docs/DECISIONS.md` | D-5: TabPFN-3.5 with 1 estimator, about 1,000 rows; D-9: last 25 s, 60% recent / 30% similar / 10% fill |
| `[D-ms]` | ms per plan for the `[M-tier]` setup, Apple GPU | `tools/latency.py` on Nooshin's Mac | Not measured. Don't quote laptop timings: they come from emulated x64 CPU (`RESULTS.md`, deployment notes) |
