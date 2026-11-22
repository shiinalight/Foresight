> **Copy of [`docs/DEFENSE.md`](../docs/DEFENSE.md)**, made by `tools/build_final.py` on 2026-11-22 14:01 (after commit `92ffc00`). Edit the source and rerun the script; do not edit this copy.

# Defence dossier: how Foresight can be attacked, and what we answer

Written Sat 21 Nov (commit d9d5751, 23:49 Berlin; adversarial review, read-only). It sequences events by git commit times and the corrected DECISIONS timestamps (commit `adc7dc1`), not by the older labels that ran about an hour ahead.

**Update Sun 22 Nov (commit 8e94fdb, 02:28 Berlin):** T5 is done (`RESULTS.md`, "Closed loop T5 results"; 20 paired seeds, 3000–3019). The primary test failed against all four baselines, and claim B held in the loop. Q0, Q12, Q21, Q32, the claim C row of "Claims we must not make", and the new Q37 (TabPFN planner slower) and Q38 (intent chain) reflect this. The ten attacks are the Sat 23:50 review, kept as written.

**How to read the numbers.** Every number carries its feature version:
- **f1:** the original features. The main real-data table, the calibration table, the tier grid, the adaptation pilot and T1 are all f1.
- **f2–f3:** the heading-leak fix (D-20) and the "unknown" mode label (D-22).
- **f4:** the Kalman filter tuned per dataset (D-26).

At the time of writing, `results/` holds **no real-data result on f2, f3 or f4**:
- the newest offline real-data files are `eval.json` (19:20), `conformal_*.csv` (20:22) and `tier_seeds.csv` (22:02), all f1;
- the only f4 results are the crowd_cross files `context_size.csv` (still being written at 23:39) and the S0 gate (D-27), plus the T2 cross cells.

"No result on current features" means exactly that. It does not mean the claim failed. There was still no `results/claim_a*` file at 23:55.

**Scope.**
- Read line by line: PLAN, DECISIONS, GATES, RESULTS, REFERENCES, DATA, DEPLOYMENT, VISUALS, CTO-REVIEW, README, the explainer, `foresight/core/*.py`, `run_closed.py`, `run_adapt.py`, `evaluate.py`, and the relevant tools and result files.
- Not reviewed line by line: `docs/walkthrough.html` and its folder, the deck slides themselves (only `tools/build_deck.py`), `docs/archive/SCENARIOS.md`, `INTERFACE.md`, `docs/archive/TEAM-PLAN.md`.
- `results/context_size.csv` was still being written when read; recheck its numbers before quoting them.

---

## Top 10 attacks, ranked by damage

### 1. "Your learned models don't beat a properly filtered constant velocity on real data."

**Exposure.** All real-data errors in `RESULTS.md` lines 70–81 are f1, computed with the untuned filter (q = 2). D-26 tuned the filter and made constant velocity (CV) 24–30% stronger.

CV error on the **same test rows** as the published table, computed for this review:
- **Method.** Take the f1 row cache, sample it with `random_state=0` as `evaluate.py` does, and merge it on episode, agent, t and h into the f4 rows. The CV error is the norm of the residual target.
- **Check.** The sample reproduces the published f1 CV exactly: 0.534 m and 0.349 m at +2 s.

| Test scene | f1 CV, +2 s (published) | **f4 CV, +2 s (same rows)** | f1 TabPFN / LightGBM all rows, +2 s | f1 CV, +4.8 s | **f4 CV, +4.8 s** | f1 TabPFN / LightGBM all rows, +4.8 s |
|---|---|---|---|---|---|---|
| ETH (n 324 / 190) | 0.534 | **0.428** | 0.431 / 0.428 | 1.442 | **1.154** | 1.189 / 1.216 |
| Hotel (n 335 / 142) | 0.349 | **0.254** | 0.274 / 0.289 | 0.858 | **0.606** | 0.673 / 0.662 |

On all cached f4 rows the picture is the same: ETH 0.413 m, Hotel 0.269 m at +2 s.

So the tuned CV is **at or below** every f1 learned error we publish:
- it ties LightGBM on ETH at +2 s;
- it is below everything else.

Caveats:
- the gaps at +2 s are small (0.003–0.035 m) and on about 330 rows;
- the learned models will also improve once refit on f4 features.

What dies is the f1 "beats CV" error story, not necessarily the method.

The S0 gate in the crossing crowd is also narrow on f4. TabPFN on 1,000 rows is only 7.5% better than CV, and LightGBM on all 73k rows is 18.7% better (D-27).

**Answer today.**
- Our claim was never lower point error. RESULTS.md:90 already says "comparable error, not lower".
- The claim is honest ranges in a new place. CV gives no range at all, and CV plus a fixed band was overconfident going to ETH on f1 (70%, RESULTS.md:78).
- We found the weak filter ourselves (D-26) and are rerunning everything against the stronger baseline.

**Fix.** The claim A rerun on f4 is scheduled (PLAN section 0, Sat 23:30–01:00). Fix the wording now, before the result:
- if f4 TabPFN and f4 LightGBM are not below f4 CV by more than the person-bootstrap interval, say "no point-error gain over a well-filtered constant velocity on these two scenes; the gain is calibrated ranges";
- every error table shows the f4 CV row next to the models.

Effort: wording 10 min; the run is already scheduled. **Fits.**

### 2. "LightGBM 'collapses' only because you didn't tune it, and conformal LightGBM beats you anyway."

**Exposure.**
- **Untuned settings.** `models.py:122` uses 31 leaves, 10 rows per leaf and 300 trees on 1,500 rows, with no tuning (RESULTS.md:43; D-15 still pending).
- **Protocol not followed.** GATES.md section 2 requires a LightGBM tuned on a validation split of the training scene.
- **Tool gap.** `tools/claim_a.py` (the f4 rerun) does not include a tuned or regularised LightGBM.
- **New f4 evidence** (`results/context_size.csv`, crowd_cross, 3 context seeds, 2,000 test rows, in-distribution): one regularised setting (`lgbm:m50:n100`) moves LightGBM's joint coverage of the 5–95% box from 0.53 to 0.74 at 1,000 rows.

  | Model, 1,000 rows | Joint coverage, 5–95% box | Pinball |
  |---|---|---|
  | LightGBM default | 0.53 | 0.131 |
  | LightGBM m50 n100 | 0.74 | 0.120 |
  | TabPFN Fast-1 | 0.86 | 0.111 |

- **Conformal LightGBM** reaches the 80% target after 20 s in the scene with boxes about 25% narrower than TabPFN's: 0.87–1.16 m against 1.16–1.43 m (RESULTS.md:129, f1).

**Answer today.**
- At equal rows TabPFN has the best proper score. Pinball on crowd_cross, f4, 3 seeds:

  | Rows | TabPFN | LightGBM (default / m50 n100) |
  |---|---|---|
  | 1,000 | 0.111 | 0.120–0.131 |
  | 2,000 | 0.106 | 0.116–0.121 |

  TabPFN's coverage stays near nominal with no tuning and no calibration split.
- TabPFN on 2,000 rows (pinball 0.106) ties LightGBM on 20,000 rows (0.106).
- LightGBM on all 73k rows is still the best point predictor (0.606 m against 0.646 m).
- We already concede sharpness once local calibration data exist (RESULTS.md:129).

**Fix.**
- Add `lgbm:m50:n100` (and, if time allows, the GATES grid of `num_leaves` 7/15/31 × `min_child_samples` 20/50/100, chosen on a validation split of the training scene) to `tools/claim_a.py` before it runs on f4. That is a one-line model list change plus about 5 min of LightGBM compute.
- Replace "collapse" everywhere with "under-cover unless tuned or conformally calibrated".

**Fits.**

### 3. "You changed the closed-loop scene and the judging rule after seeing results: garden of forking paths."

**Exposure (commit and file times).**

| Time | Event |
|---|---|
| 22:41 | T1: a learned planner (LightGBM) fails to beat the baselines in the default plaza (D-19, `results/t1_gate.csv`) |
| 22:40–23:07 | T2 runs in the default scene, baselines only (`results/closed/*-t2-*.csv`, seeds 4000–4019) |
| 22:59 | D-23 changes the T2 criterion to moving-robot collisions, while those cells are arriving |
| 23:02–23:10 | D-24 moves all confirmatory work to the crossing crowd (`66143a1`, `e087f70`, `9d822ab`) |
| 23:28 | The cross sweep starts on f4 |

Further exposure:
- **Default cells already met the written rule before the switch.** For example, repulsion 1.0 / ignore 0 / 16 people: CV moving-collision rate 10%, stop-and-go intrusion 6.3 s, reactive 3.4 s. Repulsion 2.0 / 24 people: CV 5%, reactive 3.7 s intrusion.
- **Feature version drifts inside the default sweep:** the early cells carry no `features_version` column (untagged, before f2), and the later ones are tagged f2 and f3, because code changed on disk while the processes started.
- **The rule is under-specified in the code:**
  - "best baseline" and "mildest" are defined nowhere (`tools/t2.ps1`; no picker script);
  - each baseline runs at one dial setting, carried over from the plaza pilot (`t2.ps1:4`);
  - with 20 seeds per cell, the 5% threshold flips on a single seed.
- **The confirmatory headline rule contradicts the scene rule.** RESULTS.md:159–161 requires ≤ 2% collisions and ≤ 1 s intrusion. T2 picks a scene where the best baseline has ≥ 5% collisions or ≥ 3 s intrusion. So baselines are disqualified from the headline by construction, and with 40 seeds a single collision disqualifies any planner.

**Answer today.**
- No learned planner has run in the crossing crowd. Checked over every file in `results/closed/`: the only `goal_mode == cross` rows are cv (214), reactive (185), stop_go (240) and vo (180).
- The switch was argued from realism, and that evidence was committed before the switch: `docs/DATA.md`, commit `8d6635a` at 22:53, shows turns in 36% of 2 s windows against 2–3% on ETH/Hotel. D-24 cites prediction statistics, not planner outcomes.
- D-23's reason (a person who ignores the robot can walk into a stopped robot) does not depend on any result.
- T5 uses fresh seeds 3000–3039 under a dated amendment (learned arms 3000–3019 after addendum part 5).
- Everything before T5 is exploratory and labelled so.

**Fix (no compute, about 20 min, before any learned planner runs in cross).** A dated pre-registration amendment that:
- (a) reports the default-scene T2 table and states that it already qualified;
- (b) defines "best baseline" and "mildest" mechanically:
  - best baseline = the baseline setting with the lowest moving-collision rate, ties broken by lower intrusion;
  - mildest = repulsion 2 → 1 → 0.5, then ignore 0 → 0.2 → 0.4, then 16 → 24 → 32 people.

  **Disclose** that these definitions are proposed after some cross baseline cells were visible, including to this reviewer. Show which cell each plausible reading picks:
  - so far in cross, repulsion 2.0 / 16 people does not qualify (best baseline: reactive, 0% collisions, 1.09 s intrusion);
  - repulsion 1.0 / 16 people qualifies only through CV's 5% moving collisions, which is one seed in 20;
  - the 24-person cells and every ignore 0.2 / 0.4 cell are incomplete or not yet run;
- (c) re-sweeps each baseline over its own dial in the chosen cell;
- (d) replaces the absolute thresholds with a matched-safety comparison: paired seed differences in time to goal at equal intrusion and equal moving-collision rate, read off each planner's curve;
- (e) runs every planner on one feature version (f4);
- (f) says in advance: "if the oracle gate (D-21) or T5 fails, the closed loop is reported as a stress test, not as evidence for claim C".

**Fits.**

### 4. "Claim B ('adapts in seconds') rests on 10 rows, and what you measure in the first 5 s isn't adaptation."

**Exposure.** The quoted figure, 0.425 against 0.536 m (PLAN.md:38; explainer.html:341–351, "cut error by about a fifth"), comes from `results/adapt_hotel_to_eth.csv` (20:00, f1, no person ids).
- **Sample size.** In the 0–5 s bin every strategy has **n = 10 rows from 4 episodes**.
- **Wrong comparison.** The same file gives TabPFN on the same random rows 0.485 m, and LightGBM refit on those rows 0.478 m. The 0.536 m comparator is a frozen LightGBM on all rows, a different model.
- **Mostly not this scene's rows.** By the causality rule, the "mix" context can hold almost no rows from the new scene in the first 5 s of a window:
  - ETH rows need 2 s of history (`features.py`: 5 samples × 0.4 s), so the first query is at t = 2 s;
  - a +2 s label becomes known 2 s later (`t_known = t + h`), so this scene's first rows exist only from t = 4 s;
  - the 0.425 against 0.485 gap therefore comes mainly from bank rows (kNN plus fill from the *other* scene), not from "reading the new scene". This is derived from the code; the per-source counts were not logged in this file.
- **The "recent" arm is not recency-only, and the B3 fix never fires.**
  - With mix (1, 0, 0), `context.select()` hands the whole unfilled budget to the random bank fill (`context.py:51–63`).
  - In `adapt_hotel_to_eth.csv` the "recent" context therefore always holds 914–965 rows (n_ctx), with `thin` = 0 everywhere: this scene's rows plus about 900 rows from the other scene.
  - `run_adapt.py`'s docstring ("only rows from this episode") and D-9 ("mix 0.340 against recent 0.353") describe a comparison that did not happen as written.
  - GATES B3's "constant velocity when recent is thin" never triggers.
- **The in-context pilot is thinner still.** `results/incontext_hotel_to_eth.csv` has n = 1 in its first bin.
- **The figures are already retired.** `media/MANIFEST.csv` marks both adaptation figures superseded.

**Answer today.** "Pilot, unconfirmed; the dense run with person-level intervals is pending." That is honest but still oversells.

**Fix.**
- Remove the 0.425 bar chart from the explainer and the number from PLAN.md until the f4 rerun with person-bootstrap intervals (`tools/adapt_ci.py`) lands; it is scheduled 03:30–08:00 on the API.
- Report the first-5 s bin with n and a person count.
- Describe the early gain as "retrieval of similar rows", and the later gain (≥ 10 s) as "this scene's rows".
- Before the f4 rerun, make "recent" mean recency only. When `mix[1] + mix[2] == 0`, `select()` should not fill from the bank. That is a one-line change in `context.py` (owner: builder), so `thin` and the B3 rule work.
- Log the per-source counts (`source` column) for each query.

Wording fix: 10 min; the code fix: 10 min plus the scheduled rerun. **Fits.**

### 5. "Your coverage targets are wrong, and coverage isn't a proper score."

**Exposure.** Two different boxes are compared against "80%":
- **Main table, 5–95% per axis** (RESULTS.md:70–81). The nominal joint coverage is 80–90%: 80% is the Bonferroni floor, 81% holds if the axes are independent, 90% if they are fully dependent. TabPFN's 92% on Hotel (RESULTS.md:76) exceeds even the 90% ceiling, so those boxes are over-wide: 0.91–0.96 m against 0.54–0.60 m.
- **Calibration table "raw", 10–90% per axis** (`tools/conformal_eval.py` QS = 0.1, 0.5, 0.9). The nominal joint coverage is 60–80%, yet RESULTS.md:127 scores it against an 80% target and calls TabPFN "the only calibrated model" at 62% and 80%.
- **No proper score for the box.** Pinball favours TabPFN by 0.0021 going to ETH and LightGBM by 0.0025 going to Hotel, on a single seed (RESULTS.md:91). That is a tie.

**Answer today.**
- The qualitative finding survives the correction. On the raw 10–90% box, LightGBM at 32–37% is far below the 60% floor, while TabPFN's 62% sits inside the nominal range.
- On the 5–95% box, LightGBM's 46% going to ETH is far below the 80% floor (f1).

**Fix.**
- Fix the wording: state the nominal range for each box (minutes).
- Report the interval score of the box (Gneiting & Raftery 2007, Sec. 6.2: width plus a 2/α miss penalty) next to coverage in the f4 rerun. That is a few lines in `claim_a.py`, if the builder takes it.

**Fits** as wording; the score fits only if the builder adds it before the rerun.

### 6. "Nothing in the closed loop supports a safety claim: zero latency, no calibration in the loop, and people react to the robot."

**Exposure.**
- **Latency.** The simulator waits for the planner, so latency is zero (RESULTS.md:181). Measured plan times on emulated CPU are 199–516 ms frozen, 861 ms live and 1.8 s for live LightGBM (DEPLOYMENT.md:23–25), against a 0.4 s replan period.
- **Calibration.** The conformal layer is not in the loop (PLAN.md:54), but the explainer's pipeline draws it there ("Calibration keeps 80% of truths inside", explainer.html:141).
- **Interaction.** Predictions see the robot's current relative state (`robot_x`, `robot_y`, `robot_dist`, `robot_closing`), but not the candidate move (GATES item 8 is not built). The social-force people react to the robot, which breaks the guarantee assumption of Lindemann et al. 2023 (Assumption 1); Huang et al. 2024 argue that calibration must depend on the plan.
- **Scene-level coverage.** 16 people × 2 horizons of 80% boxes do not give 80% safety for the scene (GATES.md:27).
- **The safety check is not uniform:**
  - `run_closed.py:14` and D-8 say the 0.4 s check applies to every planner (D-8 says "except reactive");
  - in the code, `vo` (`choose_vo`) and `stop_go` have no 0.4 s check;
  - `reactive` gets a frozen-people version (its velocities are zeroed);
  - `vo` searches 360° with 16 directions and its own margin, 0.1 m, while `choose()` searches ±90° with 7 directions.
- **The checks are discrete.** The planner looks only at 0.4, 1 and 2 s.
- **Evidence so far.** In T1 no learned configuration beats the baseline frontier (`results/t1_gate.csv`). T1 ran on the untuned filter, since `data/crowd` has no `kalman.json`.

**Answer today.**
- We make no safety claim from coverage, and none from the loop yet.
- The loop measures what the information is worth at zero latency.
- T4 sweeps a latency budget in virtual time and reports the latency at which any gain disappears.

**Fix.**
- Correct D-8 and the `run_closed.py` docstring, and list the check differences per planner in RESULTS (10 min).
- T4 latency sweep: already scheduled.
- Explainer: mark calibration as "offline so far".

**Fits.**

### 7. "Your oracle isn't an upper bound."

**Exposure.** `oracle_rows` (`run_closed.py:122–137`) deep-copies the simulator and rolls it forward holding the *current* command (`twin.step(cmd)`). In the real run the planner then picks a different candidate, and the social-force people react to it. So the oracle knows each person's future under an action it may not take. It also gets zero-width boxes.

D-21 calls it "perfect prediction", and the gate reads "no learned predictor can show claim C" if the oracle fails.

**Answer today.** It is perfect open-loop prediction under the current command: a strong reference for what better prediction could buy, not a strict upper bound.

**Result (Sun 22 Nov):** the open-loop oracle gate failed in the chosen cell; see Q&A 0 (time floor 17.74 s, vo w0.5 1.50 s above it).

**Fix.** Rename it to "open-loop oracle" in D-21, RESULTS and the deck (5 min). A per-candidate oracle (one twin rollout per candidate, about 22 per replan) is about 22× the simulation cost. It is feasible on a few seeds only, so it **does not fit** as a gate; list it as a limitation.

### 8. "You starve TabPFN of data and run the wrong tier."

**Exposure.**
- **Rows.** In the main table TabPFN sees 1,500 of 14,223 / 7,640 rows while LightGBM all rows sees everything (RESULTS.md:38–39). In S0 cross, TabPFN gets 1,000 rows against LightGBM's 73k (D-27).
- **Context size cuts both ways:**
  - in-distribution, TabPFN keeps improving with rows: 0.718 → 0.685 → 0.646 m from 500 → 1,000 → 2,000 rows (`context_size.csv`, f4), so D-4's 1,000 rows leaves accuracy on the table;
  - across scenes, 2,000 rows are worse than 1,000 (`tier_seeds.csv`, f1, 3 seeds, Hotel → ETH, 3.5 with 1 estimator: 0.479 → 0.526 m).
- **Tier claims don't match the code:**
  - v2 (Apache) beats 3.5 on Hotel → ETH in `tier_seeds.csv` (0.449 ± 0.011 against 0.474–0.479 m), yet DEPLOYMENT.md:64 says v2 "matches";
  - the closed loop runs 3.5-Fast with 1 estimator (`run_closed.py:281`, defaults `--tabpfn-fast 1 --tabpfn-est 1`), while the explainer (line 436) and D-5 name 3.5 with 1 estimator as the real-time tier.

**Answer today.**
- The context-size study on the API (up to 20k rows, plus retrieved 5k) is scheduled for 01:00–02:00, with a decision at 02:00.
- "Rows from another place hurt" is our own finding (`fig_rows_place.png`), and it explains the cross-scene drop.

**Fix.**
- Correct DEPLOYMENT.md:64 to "v2 is as good or better on Hotel → ETH (3 context seeds, f1)".
- Say in RESULTS which tier each result used.

Both fit. The 02:00 decision should set the context per setting, not one global number.

### 9. "What's actually new? And why TabPFN rather than LightGBM or a deep trajectory model?"

**Exposure.**
- **Precedents exist for each idea:**
  - retrieval as context: LoCalPFN (Thomas et al. 2024), classification only;
  - same-scene in-context examples for trajectories: TrajICL (Fujii et al. 2025);
  - conformal regions in planning: Lindemann et al. 2023, Dixit et al. 2023;
  - adaptive conformal for crowd navigation: Yao et al. 2025.
- **Our target is our own design.** The CV-residual target in the agent frame is our choice (REFERENCES #3). The finding that neighbour features help is not supported, and Schöller et al. 2020 cuts against it (REFERENCES #2b).
- **No deep model was compared.** We ran no deep predictor; GATES.md:77 forbids comparing against published numbers.

**Answer today.** What is new is the combination, tested end to end:
- a tabular foundation model reading per-moment retrieved rows;
- its interval calibration measured under scene transfer against LightGBM, on identical rows and features;
- a gated pipeline that feeds those ranges into a planner.

REFERENCES #24 found no paper testing tabular-foundation-model calibration under pedestrian scene transfer. That is "not found in our search", not "first".

TabPFN is the right tool here, and the reason is not accuracy:
- it needs no training, so its context can be swapped live;
- its quantiles are close to nominal out of the box with few rows.

Deep predictors need training per domain and are evaluated best-of-20 (REFERENCES section 2).

**Fix.** None. Use the safe wording below.

### 10. "Your filter was tuned on the test scene, and 'our baseline matches the published one' contradicts your own rules."

**Exposure.**
- **The tuning file contradicts D-26.** `data/eth/kalman.json` says `"tuned_on": "data/eth", "episodes": "all"`, and Hotel's says the same of Hotel. D-26 and explainer.html:456 say "tuned on the other scene". Resolved: eth/kalman.json and hotel/kalman.json record cross-scene tuning (tuned_on the other scene).
- **Likely cause.** `tools/tune_kalman.py:84` writes into the tuned dataset by default, so the `--report-fde48` gate run overwrote the cross-scene file.
- **Effect on the numbers: none.** Both scenes pick q 0.1, r 0.1.
- **"Matches" breaks three rules:**
  - explainer.html:454 is headed "Our baseline matches the published one";
  - REFERENCES.md:91 says "do not say that our CV 'matches'";
  - GATES.md:77 forbids comparisons with published numbers, and REFERENCES section 2 (lines 81–91) is now stale.
- **The figure depends on the window rule.** The 1.12 / 0.46 m come from gap-free windows with a 3.2 s warm-up. On our cached evaluation rows f4 CV at 4.8 s is 1.128 m on ETH and 0.582 m on Hotel, over all rows; on the published test sample it is 1.154 and 0.606 m. The Hotel all-rows figure is 14% above the published 0.51 m and 27% above our own 0.46 m gate figure.

**Answer today.** The tuned values are the same whichever scene does the tuning. The filter has 2 parameters, chosen by CV error alone, and a stronger CV makes every learned-model gain harder to show, not easier.

**Fix.**
- Rerun `tune_kalman.py data/hotel --noise 0 --apply-to data/eth` and the reverse (1 min, builder), so each file records cross-scene provenance.
- Reword the explainer: "within 3% (ETH) and 10% (Hotel) of the published CVM on comparable gap-free windows (3.2 s observed, 4.8 s ahead); on our own evaluation rows Hotel is 0.58 m".
- Update REFERENCES section 2.

**Fits.**

---

## Q&A sheet

**Closed loop after the oracle gate (Sun 22 Nov, amendment 2)**

0. **Q: Your own oracle failed the gate. Doesn't that sink claim C?**
   - Yes, as pre-registered: claim C is reported as a stress test, not as evidence (D-33, amendment 2).
   - In the chosen cell (repulsion 2, ignore 0.2, 24 people) the open-loop oracle beats stop_go but is only 1.59 s faster than constant velocity at margin 0.2 [0.86, 2.39], short of the 2 s margin. vo w0.5 is faster than every oracle point.
   - **The gate was hardly passable.** The time floor (straight start-to-goal distance minus the 0.5 m arrive radius, at the 1.0 m/s top speed) averages **17.74 s** over the 20 seeds. vo w0.5 runs at 19.24 s, **only 1.50 s above the floor**, so no planner could be 2 s faster than it.
   - D-14's 2 s was a detectability margin (from the pilot's paired spread), not an estimate of the headroom.
   - What T5 then tested, both added after the gate was seen (and labelled so):
     - in-loop claim B, which held: after the switch, split against frozen TabPFN, coverage +10.0 pts [+8.4, +11.6], interval score −1.31 [−1.55, −1.08] (Q37);
     - safety at matched time, where the open-loop oracle beat no baseline on every point and both axes: clear clearance gains, collision-rate gains mostly below the margin at 20 seeds.
   - The primary test failed against all four baselines, as the gate predicted (Q32).

**Framing and novelty**

1. **Q: Why TabPFN and not just LightGBM?**
   - On equal rows it has the better proper score, with no tuning: pinball 0.111 against 0.120–0.131 at 1,000 rows on crowd_cross (f4, 3 seeds, `results/context_size.csv`).
   - Its coverage is near nominal out of the box, and its context can be swapped live without retraining.
   - LightGBM on all 73k rows has the lower error (0.606 against 0.646 m).
   - Once 20 s of local outcomes exist, conformal LightGBM gives narrower calibrated boxes (RESULTS.md:129).
   - Our pitch is "few rows, new place, no training", not "always better".

2. **Q: Why not a deep trajectory predictor (Trajectron++, AgentFormer)?**
   - They need training per domain and report best-of-20 errors.
   - We compare like for like on one table and do not compare with published numbers (GATES.md:77).
   - We make no accuracy claim against them (REFERENCES section 2; the Trajectron++ numbers may be optimistic, Chen et al. 2024).

3. **Q: Is this the first use of a tabular foundation model for pedestrian prediction?**
   - We found none in our search (REFERENCES section 4, last item), but the search was not exhaustive.
   - The closest prior work is TrajICL (Fujii et al. 2025: a trained transformer with same-scene in-context examples) and TabPFN-TS (Hoo et al.).

4. **Q: What does the "CV residual in the agent frame" buy?**
   - Constant velocity becomes the all-zeros prediction, and scene direction cannot be learned.
   - Schöller et al. 2020 (Sec. VI-A) show that absolute directions cause scene-specific priors.
   - The exact encoding is our own design choice (REFERENCES #3).

5. **Q: Do neighbour and history features help?**
   - Not shown. We ran no ablation.
   - Schöller et al. 2020 and Gupta et al. 2018 found that neighbour information helps little (REFERENCES #2b).

**Fairness**

6. **Q: Same rows and features for every model?**
   - Yes. `evaluate.py:65`: the LightGBM-same-rows and TabPFN models get identical `ctx` rows.
   - Every model gets identical feature columns (`features.X_of`).
   - LightGBM also gets an all-rows version.

7. **Q: Was LightGBM tuned?**
   - In the real-data table, no (D-15 pending).
   - On crowd_cross, f4, a regularised setting (min 50 rows per leaf, 100 trees) lifts its box coverage from 0.53 to 0.74 at 1,000 rows. TabPFN is at 0.86 (`context_size.csv`).
   - The tuned grid goes into the f4 claim A rerun.

8. **Q: Isn't TabPFN under-used at 1,000–1,500 rows?**
   - In-distribution, yes: it improves to 0.646 m at 2,000 rows (`context_size.csv`).
   - Across scenes, more source-scene rows hurt (`tier_seeds.csv`: 0.479 → 0.526 m from 1,000 to 2,000, f1).
   - The API context-size run (up to 20k rows) decides this at 02:00 (PLAN section 0).

9. **Q: Which TabPFN model produced each number?**
   - Real-data table: local `tabpfn`, default settings (RESULTS.md:39).
   - Calibration table: the hosted API default model (RESULTS.md:111).
   - Closed loop and S0: 3.5-Fast, 1 estimator (`run_closed.py:281`, D-5).
   - The deployed tier is 3.5 with 1 estimator, because Fast-1 looked worse going to Hotel: 0.348 ± 0.019 against 0.309 ± 0.029 m (`tier_seeds.csv`, f1, 3 seeds; D-18).
   - That gap is about one standard deviation on 3 seeds: suggestive, not established.

10. **Q: Is your constant-velocity baseline weak?**
    - It was. The f1–f3 filter made CV 24–30% worse (D-26).
    - With the tuned f4 filter, CV on the same test rows is at or below every f1 learned error we published (attack 1).
    - Everything is being rerun against it.

11. **Q: Was anything selected on the test data?**
    - Adaptation runs pick the best of five LightGBM settings on the test scene, which favours LightGBM (RESULTS.md:137).
    - The Kalman files record tuning on the scene itself; the values are identical either way (attack 10).
    - The T2 scene is picked by baselines only, but the scene switch came after T1 (attack 3).

12. **Q: Are the closed-loop comparisons paired?**
    - Yes: same seeds, same crowd, start and goal for every planner (`run_closed.py`, seed outer loop).
    - The T1 intervals (`tools/t1_gate.py`) are marginal bootstraps per configuration, not paired differences. That is conservative.
    - T5 reports paired-seed differences (`results/t5_primary.csv`, `results/inloop_claimb.csv`).

13. **Q: Do all planners get the same safety check and action set?**
    - No:
      - the 0.4 s CV check is in `planner.choose()` only (CV, band, LightGBM, TabPFN, oracle);
      - reactive gets it with people frozen;
      - velocity obstacles and stop-and-go have none.
    - VO searches 360° with 16 directions; `choose()` searches ±90° with 7 (`planner.py:16–17`).
    - We state this and do not hide it (attack 6).

**Leakage and validity**

14. **Q: How are splits done?**
    - Real data: train on one scene, test on the other, so the test place is never seen (RESULTS.md:13).
    - Simulator: split by episode (`features.split_by_episode`).
    - Live context rows are used only once `t_known = t + h <= now` (`context.select`).

15. **Q: What was the f1 heading leak?**
    - With no obstacles (ETH/Hotel, `bounds=None`), the "no obstacle" padding vector was rotated into the agent frame, so `obst_x`/`obst_y` encoded the world heading on 100% of real rows (D-20).
    - f2 leaves the padding unrotated (`features.py:203`).
    - Every f1 real-data number is being rerun. Before that, an earlier leak, the field-of-view bounds (B9), had already been removed.

16. **Q: Standing people: Hotel has many. Do they inflate coverage?**
    - 25% of Hotel's time is spent standing, and 35% of its +2 s rows are labelled `stop` (DATA.md).
    - Standing people are easy for CV and for boxes.
    - f2 gives them a robot- or neighbour-facing frame instead of the world x-axis.
    - `tools/claim_a.py` reports moving and standing people separately. Quote the moving-only coverage.

17. **Q: Do the live labels leak ground truth?**
    - No. Live rows are labelled from noisy, filtered observed positions (`label_from="observed"`, `run_adapt.py`, `run_closed.Predictor.predict`).
    - Ground truth is used only for scoring.
    - The `mode` label looks 2 s ahead even for h = 1 s rows. It is used only for the rare-row fill from the *bank* (other episodes), never for live rows.

18. **Q: Is the simulator realistic?**
    - Only partly, and we say so (DATA.md, "Simulated vs real").
    - The default plaza turns in 36% of 2 s windows against 2–3% on ETH/Hotel. CV error is 0.93 m against 0.25–0.37 m.
    - The crossing crowd brings calm and hurried people to 0.33–0.47 m and turns to 8–13% (D-24).
    - Erratic is a synthetic stress regime.
    - It is a social-force model after Helbing & Molnár 1995, with no validation against real trajectories (REFERENCES #14).

19. **Q: Why did you switch to the crossing crowd mid-way?** See attack 3.
    - Realism evidence was committed first (DATA.md, 22:53).
    - No learned planner has run in cross yet.
    - A dated amendment fixes the T2 picker and the T5 rule before any learned run.

20. **Q: Does the bank's robot behave like the planner-driven robot?**
    - Not verified. GATES B7 asked for a comparison of `robot_dist`/`robot_closing` distributions; the robotfeat files exist (`results/closed/robotfeat_*.csv`), but no result is reported.
    - Say "open".

**Statistics**

21. **Q: How many seeds, and why?**
    - Forty fresh confirmatory seeds (D-14), from a power calculation on pilot paired standard deviations: 80% power, 5% two-sided.
    - That covers 2 s of time and 1 s of intrusion, and 0.5 s of intrusion for most pairs.
    - The learned T5 arms were cut to 20 seeds (3000–3019) before any learned result, for compute (addendum part 5). The detectable time difference at 20 seeds is about 2.6 s, against 1.84 s at 40.
    - The f1 real-data tables were single-seed. The f4 claim A rerun uses 5 context samples of 1,500 rows and a paired person bootstrap (5,000 resamples by person; `results/claim_a_boot_f4.csv`, RESULTS "Claim A on feature version f4").

22. **Q: Are the intervals by person?**
    - `tools/adapt_ci.py` and `failure_modes.py` resample whole people.
    - The f1 headline table has no intervals (RESULTS.md:134–135, "single seed").

23. **Q: Multiple comparisons?**
    - The primary T5 test is corrected: amendment 1 added Holm–Bonferroni across the four baselines at 5% (RESULTS pre-registration; `curve_compare.py --holm`). Everything else is not corrected, and there are many planners × dial settings × metrics.
    - The pre-registered rule reduces each planner to one operating point. Two claims are kept separate: navigation against baselines, and TabPFN against LightGBM (RESULTS.md:165–169).
    - Treat everything outside the pre-registered contrast as exploratory.

24. **Q: Coverage alone is not a proper score.**
    - Agreed (Gneiting & Raftery 2007). We report pinball and width next to coverage, and on f1 pinball is a tie (attack 5).

25. **Q: Per-axis or joint coverage?**
    - The main table reports both (RESULTS.md:70).
    - The 5–95% box has a nominal joint coverage of 80–90% and the 10–90% box 60–80%. We will state those ranges instead of "80%".

26. **Q: Does online conformal "guarantee" 80%?**
    - No. Adaptive conformal guarantees long-run average coverage (Gibbs & Candès 2021), not per step or per person.
    - It also drives *any* model to about 80% (RESULTS.md:129), so reaching 80% says nothing about model quality. Compare widths.

**Closed loop**

27. **Q: What exactly are intrusion, minimum gap and collision?**
    - **Intrusion:** seconds in which any person's surface is within 0.5 m of the robot's surface, i.e. an edge-to-edge gap (`run_closed.py:233`, gap = centre distance minus both radii). The threshold matches the 0.5 m personal-space category of Francis et al. 2025; their exact definition was not checked.
    - **Minimum gap:** the smallest such gap over the episode.
    - **Collision:** discs overlap (`crowd.py:298–301`).
    - **Moving collision:** a collision while the robot moves at ≥ 0.05 m/s (D-23).

28. **Q: Why do learned planners detour or run slower?**
    - Boxes widen with the horizon. The planner vetoes any candidate within 0.2 m of any box at 1 s and 2 s (`planner.py:82`), then takes the most progress among the rest.
    - Wide boxes at 2 s veto direct paths, which is the freezing-robot pattern (Trautman et al. 2015).
    - In T1 the learned planner sits on the CV-margin curve: LightGBM q0.15 at 23.3 s and 0.80 s intrusion, against CV margin 0.6 at 23.8 s and 0.87 s (D-19).

29. **Q: What did the oracle show?**
    - The open-loop oracle ran in the chosen cell and failed its gate (D-33): the fastest baseline is only 1.50 s above the 17.74 s time floor, so claim C is reported as a stress test.
    - It is an open-loop oracle: true futures under the current command, not under the chosen candidate (attack 7).

30. **Q: Is the simulation real-time?**
    - No. Latency is zero (RESULTS.md:181), and measured plan times exceed the 0.4 s replan period on emulated CPU (DEPLOYMENT.md:23–25).
    - The planned latency sweep (0 / 0.2 / 0.5 / 1.0 s in virtual time) was replaced by a single 0.4 s latency arm, predictions one replan period stale (T4 addendum part 3). That arm was dropped from T5 for compute (addendum part 5) and is then run on the L4 pod (40 seeds, same device both arms; `docs/EXPLORATORY_DIALS.md`): 0.4 s staleness costs +1.35 s [+0.18, +2.50] and raises crowds with a moving collision from 7.5% to 30% (+22.5 pts [+10, +37.5]). Prediction must arrive within the replan period: the L4's 71 ms does, the laptop CPU's 452 ms would not.

31. **Q: People react to the robot. Do your boxes?**
    - Only through the robot's current relative position and closing speed in the features.
    - Boxes do not change with the candidate move (GATES item 8 is not built).
    - This violates the assumption behind conformal safety guarantees (Lindemann et al. 2023, Assumption 1; Huang et al. 2024), so we claim no guarantee.

32. **Q: Is the robot safer with TabPFN?**
    - Not shown. T1: no learned planner beats the baseline frontier in the plaza (D-19).
    - The crossing-crowd test (T5) was the only confirmatory test, and it is done: the TabPFN planner (tabpfn_split) beats no baseline. All four fail Holm–Bonferroni (p = 1.0), on 20 paired seeds.
    - It is slower at comparable safety: 4.3 s slower than cv m0.4 at equal minimum gap, 3.0 s at equal collision rate. cv m0.2 and vo w0.5/w1/w2 are 3.9–6.2 s faster, so it does not beat them either.
    - It was already a stress test after the oracle gate (D-33, Q0). Report it as a failed primary test, not as a near miss.
    - A 40-seed extension (same arms and rules) was declared before any of its results; the 20-seed analysis stays the confirmatory one (`RESULTS.md`).

**Deployment**

33. **Q: Can this run on a robot?**
    - Not demonstrated on a robot. Laptop timings are x64 PyTorch emulated on an ARM laptop CPU (DEPLOYMENT.md section 2). On a rented NVIDIA L4 GPU (DEPLOYMENT.md, "Measured on a GPU") TabPFN takes 71 ms per replan and the full loop 225 ms per plan, inside the 0.4 s replan period. No MPS (Mac) timing yet, and no embedded (Jetson-class) GPU measured.
    - The architecture batches all people and horizons into one call over a cached context (`fit_with_cache`).
    - Parameter counts: 219 M (3.5) and 83.5 M (Fast) (RESULTS.md:179).

34. **Q: Licence?**
    - The 3.5 and 3.5-Fast weights are non-commercial. A product needs Prior Labs' commercial licence, the API, or v2 (Apache 2.0) (RESULTS.md:173–178).
    - On our f1 tier runs v2 was as good as or better than 3.5 on Hotel → ETH (0.449 against 0.474 m, 3 seeds) and weaker on ETH → Hotel (0.330 against 0.309 m).

35. **Q: What happens if TabPFN is late?**
    - Design: constant velocity plus band and the 0.4 s check take over (DEPLOYMENT.md section 3).
    - The deadline fallback is not implemented in `run_closed.py`. The 0.4 s check runs every replan for the `choose()` planners.
    - Say "designed, not built".

36. **Q: Is the API in the loop?**
    - No. The API is used for offline evaluation only, and the cloud is never in the control loop (DEPLOYMENT.md section 3).

**After T5 (Sun 22 Nov)**

37. **Q: Your TabPFN planner is slower than the baselines, so what does TabPFN buy?**
    - Calibrated ranges that adapt in the loop after a regime shift. After the crowd turns erratic, the split model (frozen bank plus this scene's recent rows) against frozen TabPFN: coverage 59.3% vs 49.3% (+10.0 pts [+8.4, +11.6]), interval score 5.03 vs 6.34 (−1.31 [−1.55, −1.08]), median error −0.057 m [−0.094, −0.021]. Against LightGBM on all rows: +18.0 pts, interval score −2.00 (`results/inloop_claimb.csv`, 20 paired seeds).
    - Limits: no model reaches 80% after the switch, and before it split is slightly worse (interval score +0.18). On real pedestrians (from 5 s, split vs frozen, paired person bootstrap, `results/realdata_claimb.csv`): small gain, interval score −0.045 [−0.080, −0.009] pooled (1,540 rows, 544 people), coverage level (+1.3 pts [−0.5, +3.1]), level with the offset on both; direction-dependent (ETH→Hotel −0.046 [−0.086, −0.005], narrower boxes at equal coverage; Hotel→ETH −0.045 [−0.093, +0.007], level). Against LightGBM: +17.1 pts coverage, interval score −0.254. The real scenes have no sudden behaviour switch, which is where the simulator's +10 pts came from.
    - Why it is slower: not calibration. At the same caution setting (q 0.15) the split, frozen TabPFN and LightGBM planners all take about 26 s (split − frozen −0.12 s [−1.35, +1.25]; split − LightGBM +0.33 s [−1.29, +1.99], 20 paired seeds), although LightGBM's boxes are 0.63 m narrower after the switch. 40-seed update (exploratory, `results/t5_primary_40.csv`): split and frozen TabPFN still drive the same (+0.03 s [−1.08, +1.16]), but LightGBM's narrower, less honest boxes (41% vs 59% coverage after the switch) make it 1.24 s faster [+0.17, +2.34]; every learned planner stays 3–5 s slower than cv and vo at comparable safety. Better prediction did not change how this box planner drives; its caution setting did. An exploratory dial sweep on fresh seeds tests lower settings (`docs/EXPLORATORY_DIALS.md`).
    - Next work, not done: a planner that uses the shape of the range, not only its width, for example a chance-constrained cost.

38. **Q: Do you use TabPFN for anything beyond one range model?**
    - Yes, offline: an intent chain. A TabPFN classifier predicts each person's next manoeuvre (straight / stop / turn left / turn right), and per-manoeuvre TabPFN range models give the 2 s range, mixed by the classifier's probabilities. The gate was fixed in `tools/intent_layer.py` before any result (`results/intent_layer.log`).
    - Real pedestrians (Hotel → ETH): balanced accuracy 0.49 (chance 0.25). Adopted: on turning people interval score 3.59 → 3.33 (−0.263 [−0.481, −0.075]; 73 rows, 37 people); overall level (−0.022 [−0.057, +0.009]); joint coverage +5.0 pts overall, +16.9 on turns; median error +0.007 m (slightly worse).
    - Simulated crowd: balanced accuracy 0.30, because simulated people pick turns at random. Not adopted: overall interval score worse.
    - TabPFN's class probabilities are much better calibrated than LightGBM's (log loss 0.24 vs 0.71 real, 0.81 vs 1.53 sim).
    - Caveats: 37 turning people, 3 context seeds on real data and 2 in simulation, offline only. On real data the classifier names no turn outright (TabPFN's turn recall is 0, LightGBM's at most 0.01).

---

## Claims we must not make (and what to say instead)

| Do not say | Why | Say instead |
|---|---|---|
| "Safe out of the box in a new place" (PLAN.md:37) or "Safe in a new place" (explainer.html:385) | Safety cannot come from coverage (GATES.md:27); there is no closed-loop evidence; f1 only | "Honest ranges in a new place: on held-out-scene data, TabPFN's boxes held close to their nominal coverage where untuned LightGBM's did not (f1; f4 rerun pending)" |
| "LightGBM's ranges collapse" | Untuned; a regularised LightGBM gets 0.74 against 0.53 at 1,000 rows (f4, crowd_cross) | "Untuned LightGBM under-covers in a new place; tuning or conformal calibration fixes most of it, and conformal LightGBM is sharper once local data exist" |
| "TabPFN is more accurate" / "beats constant velocity" (any f1 error) | f4 CV on the same rows is at or below f1 learned errors on both scenes (attack 1) | "Point error comparable to LightGBM; [after the f4 rerun: X% / no gain] against a well-filtered constant velocity" |
| "Only calibrated model" (RESULTS.md:127) | The 10–90% box has a 60–80% nominal range, not 80%; one seed | "Before any calibration data, TabPFN's raw box was within its nominal range in both directions; LightGBM's was far below it (32–37% against a 60% floor)" |
| "Adapts in seconds" / "cut error by about a fifth" (explainer.html:351) | n = 10 rows in 4 episodes; f1; the early bin is retrieval, not this scene's rows | "Context can be swapped in seconds without retraining. Whether that improves accuracy is being measured (pilot too small to quote)" |
| "Prediction-aware planning beats standard navigation" | T1 failed; the oracle gate failed; T5's primary test failed against all four baselines | "In a crossing crowd our TabPFN planner beat no standard planner: it was slower at comparable safety. At the same caution setting, split, frozen TabPFN and LightGBM drove the same, so the caution setting, not calibration, set the pace" |
| "Guaranteed coverage" / "keeps 80% of truths inside" (explainer.html:141) | Adaptive conformal gives long-run average coverage only, and it is not in the loop | "An online offset keeps long-run box coverage near 80% in offline replays" |
| "Our baseline matches the published one" (explainer.html:454) | REFERENCES.md:91, GATES.md:77; depends on the window rule | "Our constant velocity is within 3–10% of the published CVM on comparable windows" |
| "Oracle = perfect prediction / upper bound" (D-21) | Rolled out under the current command, while people react to the actual one | "Open-loop oracle: true futures under the current command" |
| "Real-time" / "runs every 0.4 s on the robot" (explainer.html:436) | Emulated CPU plan times of 199–861 ms; zero-latency simulation; no GPU timing | "Designed for a 0.4 s replan; latency tolerance is measured in virtual time (T4); hardware timing pending" |
| "Agentic" | Uncertainty-triggered re-retrieval was not built (GATES.md:29) | "Pipeline" |
| "Works for any robot / any agent" as a result | One disc robot, one simulator | "The data interface is agent-agnostic; tested on pedestrians only" |
| "First" | Search was not exhaustive | "We found no prior work on …" |
| "Standard ETH/UCY benchmark" | Two scenes, cross-scene, fixed horizons | "Cross-scene (held-out scene) on ETH and Hotel" |

## Claims we can make (exact wording)

> **Superseded by docs/STORY.md section 2 (22 Nov, 09:00 Berlin).** The list below predates the f4 reruns, LOSO5, the highway tests and the policy arm. Present from the STORY.md ledger and its section 7 answers.

1. **Only after the f4 claim A rerun confirms it, and with the f4 numbers:** "On two real pedestrian scenes, trained on one and tested on the other, TabPFN read 1,500 example rows and needed no training. Its 2D box held the truth close to its nominal rate, while an untuned LightGBM on the same rows covered 46% going to ETH." (These numbers are f1 and single seed. Do not quote them as they stand.)
2. "With 1,000–2,000 example rows TabPFN had a lower pinball loss than LightGBM on the same rows (0.111 against 0.120–0.131 at 1,000; 0.106 against 0.116–0.121 at 2,000). On 2,000 rows it tied LightGBM trained on 20,000 (0.106); LightGBM on all 73,000 rows was still more accurate (0.606 against 0.646 m)." (crowd_cross simulator, f4, 3 context seeds, one test set of 2,000 rows; `results/context_size.csv`; recheck at the final version of the file.)
3. "Once about 20 s of outcomes exist in the new scene, an online conformal offset brings every model to about 80% box coverage, and conformal LightGBM does so with narrower boxes than TabPFN." (f1, RESULTS.md:129. This is a concession, safe to say now. Restate the numbers after the rerun.)
4. "We tuned our constant-velocity filter after finding it 24–30% weaker than it should be, and we rerun every result against the stronger baseline." (D-26.)
5. "Every chosen value has a decision row; every gate is written before its test; test seeds are fresh and paired." (DECISIONS.md, GATES.md; with the forking-path history disclosed, attack 3.)
6. "In the default plaza, prediction-aware planning did not beat standard navigation, and we stopped and tested why rather than tuning until it did." (D-19.)
7. "TabPFN's context is a set of rows swapped without retraining; its weights never change on the robot." (`models.py`, DEPLOYMENT.md section 1.) Only add "shuffling changes predictions by under 0.0001 m" once a committed test shows it; none was found in `tests/` or `tools/`.

---

## Stale or overclaiming text found

| File:line | Current text | Proposed replacement |
|---|---|---|
| RESULTS.md:21 | "the last 2 s of their path" | "the last 2 s of their path on ETH/Hotel (5 samples at 0.4 s); 1 s in the simulator (5 at 0.2 s)" (`features.py:120`: `k = max(1, round(0.2/dt))` = 1 step at both dt) |
| RESULTS.md:56 | "the 80% range… holds the truth about 80% of the time when the model is honest" | "the 5–95% box on each axis; if the model is honest it holds the truth 80–90% of the time (81% if the axes are independent)" |
| RESULTS.md:68 | "Numbers (rerun 21 Nov evening, after the leakage fix…)" | Add the banner: "**Features f1, superseded.** Computed before the heading-leak fix (D-20) and the tuned filter (D-26). On f4 rows constant velocity alone is 0.413 m (ETH) and 0.269 m (Hotel) at +2 s. Rerun pending." |
| RESULTS.md:89 | "Calibration is the finding that holds. TabPFN's box holds the truth 92% and 83%… where about 80% is promised" | "On f1, TabPFN's box covered 92% and 83% where 80–90% is nominal (92% is wider than needed); LightGBM's covered 46–62% going to ETH. Rerun on f4 pending." |
| RESULTS.md:102 | "about 45% of people there stand still" | "people in Hotel are standing about a quarter of the time (25.1%), and 35% of +2 s rows are labelled stop (DATA.md)" |
| RESULTS.md:127 | "Out of the box, TabPFN is the only calibrated model… 62% and 80%" | "Raw 10–90% boxes (nominal joint coverage 60–80%): TabPFN 62% and 80%, inside the nominal range; LightGBM 32–60%, below it going to ETH. f1, one seed." |
| RESULTS.md:130 | "the first seconds in a new place are exactly when an overconfident box causes a collision" | "…when an overconfident box would matter most for a robot (not tested in the closed loop)" |
| RESULTS.md:146–148 | "seeds 1000–1099… box_q 0.05 / 0.10 / 0.15" (pre-registration) | Add: "Superseded for confirmation by T1–T5 (DECISIONS section 3): crossing crowd (D-24), fresh seeds 3000–3039, box_q 0.10 / 0.15 (D-6); amendment dated [time]" |
| RESULTS.md:159–161 | "collision rate ≤ 2%… intrusion ≤ 1 s" | Amend before T5 (attack 3d): matched-safety paired comparison, since T2 picks a scene where baselines exceed these thresholds by design |
| PLAN.md:37 | "**Safe out of the box in a new place**… Supported on f1 features; rerun on f2 in progress (Sat 23:40…)" | "**Honest ranges in a new place**… Supported on f1 only; f1 leaked world heading (D-20, decided 22:54) and used a weak CV filter (D-26); rerun on f4 in progress" |
| PLAN.md:38 | "Pilot, Hotel → ETH: first 5 s, 0.425 against 0.536 m for frozen LightGBM" | "Pilot too small to quote (10 rows, 4 episodes, f1). Dense f4 run with person intervals pending" |
| PLAN.md:41 | "New place or new behaviour, few rows: TabPFN is safer" | "…TabPFN's ranges are better calibrated" ("safer" is a closed-loop word) |
| PLAN.md:49 | "Plaza passes the prediction check" | "Plaza passed S0 on the untuned filter (D-12); cross scene passes narrowly on f4 (D-27)" |
| PLAN.md:50 | "Works. Velocity error not yet reported" | "Tuned per dataset (D-26); S1 gate passed" |
| PLAN.md:70 | "Cross sweep running from Sun 00:20" | "Cross sweep running from Sat 23:28 (f4)"; add "default-scene T2 cells had qualified before the switch (see DEFENSE attack 3)" |
| README.md:12 | "`docs/archive/CTO-REVIEW.md`: direction, architecture…" | "`PLAN.md`: current strategy and claims (CTO-REVIEW is the first, partly superseded review)" |
| README.md:65–66 | "observed positions with noise…; the agent's own last 1 s" | "simulator: positions with 0.1 m noise; real data: annotated positions (no added noise); the agent's last 1 s (simulator) or 2 s (ETH/Hotel)" |
| docs/REFERENCES.md:51 | "f2: 5 samples at 0.2 s, about 1 s" | "5 samples at 0.4 s on ETH/Hotel, 2 s (0.2 s steps in the simulator)" |
| docs/REFERENCES.md:81–91 | "Our CV against the published CVM (open issue)… gap is unexplained" | "Resolved by D-26: with the tuned filter, CV FDE at 4.8 s on gap-free windows is 1.12 m (ETH) and 0.46 m (Hotel) against 1.15 and 0.51 published; on our evaluation rows 1.13 and 0.58 m" |
| docs/DECISIONS.md:36 (D-9) | "mix 0.340 m against recent 0.353, global 0.371, kNN 0.385" | Add: "'recent' was padded to about 900–960 rows with random bank rows (`context.select` fills the unused budget), so this compares mix against recent-plus-random, not recency only" |
| run_adapt.py:9 (docstring) | "recent: only rows from this episode whose outcome is already known" | Fix `context.select` so that is true (attack 4), or change the docstring to "this episode's known rows, padded with random bank rows" |
| docs/DECISIONS.md:35 (D-8) | "Applied to every planner except `reactive`" | "Applied in `planner.choose()`: CV, band, LightGBM, TabPFN, oracle; reactive with people frozen; not applied to velocity obstacles or stop-and-go" |
| docs/DECISIONS.md:53 (D-26) | "tuned cross-scene (Hotel -> for ETH, ETH -> for Hotel)" | Keep, but rewrite `data/*/kalman.json` by cross-scene runs; they currently record `tuned_on` = the scene itself (`tune_kalman.py:84` default) |
| docs/DECISIONS.md D-21 | "deep-copies the simulator… plans against every person's TRUE position" | Add: "under the current command; people react to the robot, so this is an open-loop oracle, not a strict upper bound" |
| docs/DEPLOYMENT.md:64 | "v2 (Apache) matches 3.5 from 1,000 rows on Hotel → ETH" | "v2 was as good or better on Hotel → ETH (0.449 against 0.474–0.479 m, 3 seeds, f1)" |
| run_closed.py:14 (docstring) | "The 0.4 s constant-velocity safety check… applies to every planner." | "…applies to every planner that uses planner.choose(); vo and stop_go have none." |
| docs/explainer.html:126 | "where each person will be in 2 s (80% likely)" | "(80–90% likely, if the model is honest)" |
| docs/explainer.html:141 | "Calibration keeps 80% of truths inside" | "Calibration (offline so far): long-run ~80% inside" |
| docs/explainer.html:171 | "last 1 s" | Fine for the simulator; add "(2 s on ETH/Hotel)" where real data is discussed |
| docs/explainer.html:210 | "shuffling them changes predictions by under 0.0001 m (tested)" | Drop "(tested)" unless a committed test or script shows it; none found in `tests/` or `tools/` |
| docs/explainer.html:341–351 | Bar chart 0.425 / 0.485 / 0.536; "cut error by about a fifth (not yet confirmed)" | Remove until the f4 dense run with person intervals; or caption "pilot, 10 rows, too small to read" |
| docs/explainer.html:385 | "Safe in a new place" | "Honest ranges in a new place" |
| docs/explainer.html:405–406 | "T2 · Harder crowd by a fixed rule; baselines only pick it, before any learned run" | "T2 · Harder crowd, picked by baselines only. Scene moved to the crossing crowd for realism (D-24) after the default-scene sweep; rule amended [time]" |
| docs/explainer.html:436 | "Real time, every 0.4 s · TabPFN-3.5, 1 member" | "Planned real-time tier · TabPFN-3.5, 1 member (the simulator runs 3.5-Fast, 1 member; timing pending)"; mark "Second opinion" as not built |
| docs/explainer.html:451 | "1 ensemble member matched 8 in accuracy" | "…matched 8 in accuracy (3 context samples, pre-fix features)" |
| docs/explainer.html:454–456 | "Our baseline matches the published one… (ours, filter tuned on the other scene)… on the standard 8 observed / 12 predicted steps" | "Our baseline is close to the published one: 1.12 m and 0.46 m against 1.15 and 0.51 m (Schöller et al. 2020), on gap-free windows with 3.2 s observed and 4.8 s ahead; on our own evaluation rows 1.13 and 0.58 m" |
| docs/explainer.html:473 | "36 references, each verified" | "36 references, each looked up; the depth of each check is stated" |
| docs/explainer.html:480 | "holds the truth about 80% of the time if the model is honest" | "80–90% of the time (81% if the axes are independent)" |
| deck/Foresight.pptx (built 19:29 by `tools/build_deck.py`; both removed 22 Nov) | Superseded demo figure (MANIFEST: made before B9); hard-coded "box held the truth 94%" (`build_deck.py:138`); "Box holds truth (~80%)" header uses per-axis coverage | Rebuild from current figures only after the f4 rerun; drop the 94% line |
| media/MANIFEST.csv | `fig_error_by_horizon`, `fig_calibration`, `fig_data_efficiency`, `fig_rows_place`, `fig_conformal_*`, `fig_failure_*` and `fig_tier_grid` marked "current" | Mark "superseded (f1)" until rerun on f4 |
