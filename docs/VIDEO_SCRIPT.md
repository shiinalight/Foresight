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
| 0:55–1:10 | `docs/walkthrough/s4_context.png` → `s5_person_box.png` → `s8_planner.png` | "It reads what people here just did, plus examples from other crowds, and returns a range for each person, one and two seconds ahead. The planner rejects any move into a range." (32) |
| 1:10–1:35 | `media/demo/seed3013.gif`. Caption: "crowd chosen by a fixed rule (median), not cherry-picked" | "Here is the full loop in a simulated crowd. Same crowd for every robot: stop-and-go, reactive, velocity obstacles, and ours, with the ranges it planned around. Five seconds in, the crowd turns erratic, a behaviour none of the models has seen." (42) |
| 1:35–2:05 | `media/fig_f4_claim_a.png`, then `media/demo/policy_seed3403.gif`, then `media/fig_highway.png` | "On real Zurich pedestrians, in a scene it never saw, TabPFN's box held the truth seventy-five percent of the time where LightGBM's held fifty-eight. The other way round, they're level. And a second TabPFN can pick the robot's move: it matched the planner it learned from on twenty new crowds. The same core, unchanged, predicts highway cars too, and adapts when traffic turns aggressive." (67) |
| 2:05–2:25 | `media/fig_t5.png`: right panel, then left | "When the crowd changed, reading the scene's last seconds kept the ranges honest: ten points more coverage. But our robot was slower than standard planners, not because of calibration: with or without the scene model it drove the same. Caution set the pace." (43) |
| 2:25–2:45 | "Where it breaks" card, `media/v_limits.png` (five lines) | "Where it breaks. Familiar place, many rows: LightGBM is as good. Still short of eighty percent after a change. Our crowd is simulated. The weights are non-commercial. Next: a planner that uses the range's shape." (36) |
| **2:45–3:00** | **On camera:** Darshak and Nooshin. End card: repo URL, "every number comes from a script in the repo" | "Foresight is open source: every number here comes from a script in the repo. Next, real robot data and timing on robot hardware. Thank you, Prior Labs." (28) |

Beats: 15 + 15 + 25 + 15 + 25 + 30 + 20 + 20 + 15 = 180 s, at most 2.3 words per second (the 0:55 beat is 32 words in 15 s, the 1:35 beat 67 in 30 s).

**Sources for each spoken number:**
- **"seventy-five … fifty-eight":** claim A f4, Hotel → ETH, joint 5–95 box. TabPFN 74.8%, LightGBM on all rows 57.7%; box miss −17.0 points [−21.0, −13.0], paired person bootstrap (`results/claim_a_f4.md`, `results/claim_a_boot_f4.csv`).
- **"The other way round, they're level":** ETH → Hotel interval score, TabPFN against LightGBM on all rows: +0.007 [−0.057, +0.065].
- **"a second TabPFN can pick the robot's move: it matched the planner it learned from on twenty new crowds":** TabPFN policy, 20 fresh crowds (3400–3419) against its teacher: goal 100% vs 100%, time +0.55 s [−0.34, +1.65], gap +0.009 m, collision crowds +5 points [−15, +25]. It matches on point estimates; the collision interval is too wide to rule out a rate 25 points worse (`RESULTS.md`, "TabPFN picks the move"; `results/policy_parity.csv`). On screen: crowd 3403, chosen by rule.
- **Intent layer** (not spoken; the deck's "TabPFN at every step" slide): real Hotel → ETH, turn rows, interval score −0.263 [−0.481, −0.075]. It does not forecast turns: turn recall is 0, and the gain is calibrated widening.
- **"ten points more coverage":** T5 exploratory (b), after the switch, split against frozen TabPFN: +10.0 points [+8.4, +11.6], 20 paired crowds (`results/inloop_claimb.csv`).
- **"slower than standard planners":** T5 primary: no baseline beaten (Holm, p = 1.0); 4.3 s slower than constant velocity m0.4 at equal gap (`results/t5_primary.csv`).
- **"not because of calibration: with or without the scene model it drove the same":** T5 at q 0.15, split − frozen TabPFN −0.12 s [−1.35, +1.25] (20 crowds), +0.03 s [−1.08, +1.16] (40 crowds).
  - LightGBM is about 1.2 s faster over 40 crowds (+1.24 [+0.17, +2.34]), but after the switch its boxes hold 41% of people against split's 59%: faster because overconfident. The deck notes carry this; it is not spoken (`RESULTS.md`, T5 "What this says" and the 40-seed update).
- **"predicts highway cars too, and adapts when traffic turns aggressive":** `docs/HIGHWAY.md`, offline, simulated. H1: TabPFN interval score 3.57 vs LightGBM (100k rows) 4.06, −0.49 [−0.66, −0.34]. H2: split vs frozen after the switch, −0.98 [−1.41, −0.56] (D-35).
- **"Still short of eighty percent":** after the switch the best in-loop coverage is 59.3% (split).

**Real-data claim B** (`results/realdata_claimb.csv`) is small and direction-dependent (interval score −0.045 [−0.080, −0.009], coverage level, level with the offset), so it stays out of the video.

**Cut order if long:** the highway sentence, then the policy sentence (1:35 beat), then one walkthrough stage (s5). **Never cut** the open, the close, the demo or "where it breaks".
