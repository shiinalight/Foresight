# Foresight: three-minute video script, version 2 (CEO handover section 5)

Every number comes from `RESULTS.md` and the docs it points to. Recording rules (`docs/PRESENTATION.md` section 2):
- no numbers on camera;
- numbers in voice-over only;
- captions burned in;
- at most ~2.3 spoken words per second.

**The one point:** the robot's knowledge is a table of examples, so it adapts to new behaviour and to a new body by swapping rows, with no training step. Three proofs, then one beat on what failed.

**Changed from version 1:** the claim A line (75% against 58%, two scenes) is cut, because on five scenes claim A is a tie (`docs/LOSO5.md`). The intent sentence and the walkthrough stages are cut too; both stay in the README and the deck.

| Time | On screen | Said (words) |
|---|---|---|
| **0:00–0:15** | **On camera:** Darshak. Lower third: "Foresight · Prior Labs track" | "A robot crossing a station has to guess where everyone will be two seconds from now, and how sure it is. An overconfident guess is how robots walk into people." (30) |
| 0:15–0:30 | `media/diagrams/01_problem.png` | "Most robots assume people keep walking straight, or add one fixed margin. Learned models do better, but they learn one place. In a new place, their confidence can be badly wrong." (31) |
| 0:30–0:55 | `media/diagrams/02_pipeline.png`, then one table row (`docs/walkthrough.html`, stage 3) | "So we turned the scene into a table. One row is one person, at one moment, for one look-ahead: their motion, their neighbours, the robot. TabPFN reads rows like these as examples, with no training. The robot's knowledge is the table." (41) |
| 0:55–1:20 | `media/demo/seed3013.gif`. Caption: "crowd chosen by a fixed rule (median), not cherry-picked" | "Here is the full loop in a simulated crowd. Same crowd for every robot: stop-and-go, reactive, velocity obstacles, and ours, with the ranges it planned around. Five seconds in, the crowd turns erratic, a behaviour none of the models has seen." (42) |
| 1:20–1:50 | **Proof 1.** `media/demo/calibration_seed3013.gif`, caption "this crowd: 60% vs 45%" (its own counters; never caption it 59/41). Then `media/demo/policy_seed3403.gif` for the last sentence | "When the crowd changed, reading the scene's last seconds kept the ranges honest: eighteen points more coverage than LightGBM, and ahead of it in all three crowds we tried. And a second TabPFN can pick the robot's move: it matched the planner it learned from on twenty new crowds." (48) |
| 1:50–2:15 | **Proof 2.** `media/fig_highway.png` | "Second proof: the same code, unchanged, on simulated highway traffic. With fifteen hundred rows it matches LightGBM trained on sixty-seven times more, with more honest ranges. When traffic turns aggressive, reading the scene adds fourteen points of coverage." (37) |
| 2:15–2:30 | **Proof 3.** `media/fig_loso5.png` | "Third: on five real pedestrian scenes, TabPFN beats LightGBM on the same rows, and ties one trained on fifteen times more." (21) |
| 2:30–2:48 | "Where it breaks" card, `media/v_limits.png` | "Where it breaks. Our robot was slower than standard planners: better ranges didn't change how this planner drives. Pedestrian examples don't carry over to cars. The robot loop and the cars are simulated, and the weights are non-commercial." (37) |
| **2:48–3:00** | **On camera:** Darshak and Nooshin. End card: repo URL, "every number comes from a script in the repo" | "Foresight is open source: every number here comes from a script in the repo. Thank you, Prior Labs." (18) |

Beats: 15 + 15 + 25 + 25 + 30 + 25 + 15 + 18 + 12 = 180 s. The fastest beat is "where it breaks" at 2.1 words per second.

**Sources for each spoken claim:**
- **"eighteen points more coverage than LightGBM":** T5 exploratory (b), after the switch, split against lgbm_all: +18.0 points [+16.7, +19.3] (59.3% against 41.3%, 20 confirmatory crowds) (`results/inloop_claimb.csv`). This matches the GIF's comparison.
  - Of that, reading the scene alone adds ten points over the same TabPFN without it: split against frozen TabPFN, +10.0 [+8.4, +11.6]. That goes in the deck notes, not the voice-over.
- **"ahead of it in all three crowds we tried":** split against lgbm_all after the switch, in points:
  - open plaza +6.6 [+5.7, +7.5];
  - crossing +18.0 [+16.7, +19.3];
  - dense crossing +19.7 [+18.7, +20.8].
  - Sources: `docs/GENERALISATION.md`, `RESULTS.md`. Simulated crowds, still short of 80% after the change.
- **"this crowd: 60% vs 45%":** `media/demo/calibration_seed3013.gif`, split against LightGBM on crowd 3013; the tool asserts this equals the T5 row.
- **"a second TabPFN can pick the robot's move: it matched the planner it learned from on twenty new crowds":** TabPFN policy against its teacher, 20 fresh crowds (3400–3419).
  - Goal 100% vs 100%; time +0.55 s [−0.34, +1.65]; gap +0.009 m; collision crowds +5 points [−15, +25].
  - It matches on point estimates, but the collision interval is too wide to rule out a rate 25 points worse (`RESULTS.md`, "TabPFN picks the move"; `results/policy_parity.csv`). On screen: crowd 3403, chosen by rule.
- **"matches LightGBM trained on sixty-seven times more, with more honest ranges":** highway H1 (`docs/HIGHWAY.md`, offline, simulated). TabPFN on 1,500 rows: 85.6% coverage, interval score 3.57.
  - Against the pre-set LightGBM on 100,440 rows: −0.49 [−0.66, −0.34].
  - Against well-fitted LightGBM settings (post-hoc, `RESULTS.md`): −0.18 [−0.36, −0.01] and −0.18 [−0.37, +0.00], with those models covering 69–72%. So "matches", never "beats".
- **"reading the scene adds fourteen points of coverage":** highway H2, split against frozen after traffic turns aggressive. Coverage 76.7% against 62.5% (+14.2 points [+11.5, +16.9]); interval score −0.98 [−1.41, −0.56]. About 80% of the gain falls in the first 6 s after the switch.
- **"beats LightGBM on the same rows, and ties one trained on fifteen times more":** `docs/LOSO5.md`, leave one of five scenes out.
  - Same 1,500 rows: −0.130 [−0.154, −0.107].
  - LightGBM on all rows of four scenes: −0.008 [−0.023, +0.007], a tie.
  - Claim A's two-scene result is not said: it is "on two scenes, not on five".
- **"slower than standard planners: better ranges didn't change how this planner drives":** T5 primary, no baseline beaten (Holm, p = 1.0), in all three crowds. At q 0.15, split − frozen TabPFN −0.12 s [−1.35, +1.25] (`RESULTS.md`).
- **"Pedestrian examples don't carry over to cars":** highway H2b. A pedestrian bank on cars gives boxes about 50 m wide.

**Not in the video:**
- the intent chain (deck "TabPFN at every step" slide; it does not forecast turns);
- real-data claim B (small and direction-dependent: interval score −0.045 [−0.080, −0.009]);
- the LightGBM speed trade-off (deck notes).

**Cut order if long:** the policy sentence (1:20 beat) first, then the table-row shot in the 0:30 beat. **Never cut** the open, the close, the demo, the three proofs or "where it breaks".
