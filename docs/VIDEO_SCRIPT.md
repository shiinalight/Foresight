# Foresight: three-minute video script (final, on the f4 verdicts)

Every number comes from `RESULTS.md` (Sun 22 Nov). The structure and recording rules are in `docs/PRESENTATION.md` section 2:
- no numbers on camera;
- numbers in voice-over only;
- captions burned in;
- at most ~2.3 spoken words per second.

**Box wording.** "Holds the truth about 8 times in 10" refers only to the 5–95% box (about 81% for both axes together).

| Time | On screen | Said (words) |
|---|---|---|
| **0:00–0:15** | **On camera:** Darshak. Lower third: "Foresight · Prior Labs track" | "A robot crossing a station has to guess where everyone will be two seconds from now, and how sure it is. An overconfident guess is how robots walk into people." (30) |
| 0:15–0:30 | `media/diagrams/01_problem.png` | "Most robots assume people keep walking straight, or add one fixed margin. Learned models do better, but they learn one place. In a new place, their confidence can be badly wrong." (31) |
| 0:30–0:55 | `media/diagrams/02_pipeline.png`, then the walkthrough's table row (`docs/walkthrough.html`, stage 3) | "So we turned the scene into a table. One row is one person, at one moment, for one look-ahead: their motion, their neighbours, the robot. The answer is how far they end up from walking straight. TabPFN reads rows like these as examples. No training." (45) |
| 0:55–1:15 | `docs/walkthrough/s4_context.png` → `s5_person_box.png` → `s8_planner.png` | "It reads what people here just did, plus examples from other crowds, and returns a range for each person, one and two seconds ahead. The planner rejects any move into a range." (32) |
| 1:15–1:40 | `media/demo/seed3013.gif`. Caption: "crowd chosen by a fixed rule (median), not cherry-picked" | "Here is the full loop in a simulated crowd. Same crowd for every robot: stop-and-go, reactive, velocity obstacles, and ours, with the ranges it planned around. Five seconds in, the crowd turns erratic, a behaviour none of the models has seen." (42) |
| 1:40–2:05 | `media/fig_f4_claim_a.png`, then `media/fig_intent_layer.png` | "On real Zurich pedestrians, in a scene it never saw, TabPFN's box held the truth seventy-five percent of the time where LightGBM's held fifty-eight. The other way round, they're level. A second TabPFN layer gave better-calibrated ranges for people who turn." (42) |
| 2:05–2:25 | `media/fig_t5.png`: right panel, then left | "When the crowd changed, reading the scene's last seconds kept the ranges honest: ten points more coverage than the frozen model. But our robot was slower than standard planners, and not because of TabPFN: every learned planner drove the same at that caution setting." (45) |
| 2:25–2:45 | "Where it breaks" card, `media/v_limits.png` (five lines) | "Where it breaks. Familiar place, many rows: LightGBM is as good. Still short of eighty percent after a change. Our crowd is simulated. The weights are non-commercial. Next: a planner that uses the range's shape." (36) |
| **2:45–3:00** | **On camera:** Darshak and Nooshin. End card: repo URL, "every number comes from a script in the repo" | "Foresight is open source: every number here comes from a script in the repo. Next, real robot data and timing on robot hardware. Thank you, Prior Labs." (28) |

Beats: 15 + 15 + 25 + 20 + 25 + 25 + 20 + 20 + 15 = 180 s, at most 2.3 words per second.

**Sources for each spoken number:**
- **"seventy-five … fifty-eight":** claim A f4, Hotel → ETH, joint 5–95 box. TabPFN 74.8%, LightGBM on all rows 57.7%; box miss −17.0 points [−21.0, −13.0], paired person bootstrap (`results/claim_a_f4.md`, `results/claim_a_boot_f4.csv`).
- **"The other way round, they're level":** ETH → Hotel interval score, TabPFN against LightGBM on all rows: +0.007 [−0.057, +0.065].
- **"better-calibrated ranges for people who turn":** intent layer, real Hotel → ETH, turn rows. Interval score −0.263 [−0.481, −0.075], 37 people (`results/intent_layer.log`, D-34).
  - It does not forecast turns: turn recall is 0. The gain is calibrated widening, so never say "it predicts turns".
- **"ten points more coverage":** T5 exploratory (b), after the switch, split against frozen TabPFN: +10.0 points [+8.4, +11.6], 20 paired crowds (`results/inloop_claimb.csv`).
- **"slower than standard planners":** T5 primary: no baseline beaten (Holm, p = 1.0); 4.3 s slower than constant velocity m0.4 at equal gap (`results/t5_primary.csv`).
- **"not because of TabPFN: every learned planner drove the same":** T5 at q 0.15, split 26.23 s, frozen 26.35 s, LightGBM 25.90 s. Paired: split − frozen −0.12 s [−1.35, +1.25], split − LightGBM +0.33 s [−1.29, +1.99] (`RESULTS.md`, T5 "What this says").
- **"Still short of eighty percent":** after the switch the best in-loop coverage is 59.3% (split).

**If the real-data claim B result (`results/realdata_claimb.csv`) is positive,** add to the 2:05 beat: "and on real pedestrians too". Say it only if the split − frozen interval excludes 0 from 5 s on.

**Cut order if long:** the intent sentence (1:40 beat), then one walkthrough stage (s5). **Never cut** the open, the close, the demo or "where it breaks".
