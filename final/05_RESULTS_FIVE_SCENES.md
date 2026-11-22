> **Copy of [`docs/LOSO5.md`](../docs/LOSO5.md)**, made by `tools/build_final.py` on 2026-11-22 14:44 (after commit `15b3ee5`). Edit the source and rerun the script; do not edit this copy.

# Exploratory: claim A on five real scenes, leave one scene out (written before any run; commit fa0b5de, 03:12 Berlin)

**Why.** Claim A ("TabPFN's ranges stay honest in a place it never saw") rests on two real scenes, Hotel → ETH and ETH → Hotel. This test adds the three UCY scenes, so the claim is checked on five scenes, each held out once. That is the standard ETH/UCY leave-one-scene-out design (REFERENCES row 4).

**Data.**
- OpenTraj `obsmat.txt` for ETH, Hotel, UCY Zara01 (`data/zara1`), Zara02 (`data/zara2`) and Students03 (`data/univ`). UCY was downloaded 22 Nov with Darshak's approval; `data/raw/*/info.txt` records the source.
- Converted by `foresight/domains/eth_ucy.py` (0.4 s steps, 60 s windows).
- Features f4, noise 0.
- Kalman q 0.1, r 0.1 for every scene: the ETH/Hotel values, never tuned on a test scene.

**Design.**
- **Held-out scene:** each of the five in turn. Test rows: up to 1,500 random rows at h = 2.0 s from that scene (random_state 0), standing people included.
- **Context:** 1,500 rows from the other four scenes, 375 from each, so the large Univ scene doesn't dominate. 3 context seeds (0, 1, 2).
- **Models**, all on the same rows:
  - constant velocity;
  - constant velocity plus band (residual quantiles of the context rows);
  - LightGBM m50:n100 on the same 1,500 rows;
  - LightGBM m50:n100 on all rows of the four training scenes (`lgbm_all`, the deployable rival);
  - TabPFN on the hosted API (default model).

**Metrics** at +2 s:
- joint coverage of the 5–95% box (target 81%);
- 90% interval score (Gneiting & Raftery), the primary metric;
- median error;
- 5–95% width.

Paired person bootstrap (2,000 draws), TabPFN minus each rival, per held-out scene and pooled over scenes (persons resampled within each scene).

**Verdict rule (fixed now):**
- **"Claim A holds on five scenes"** if two things are true:
  - TabPFN's pooled interval score is lower than `lgbm_all`'s and constant velocity plus band's, each with a 95% interval excluding 0;
  - in no held-out scene is TabPFN significantly worse than either.
- **"Direction-dependent"** if the pooled result holds but at least one scene is significantly worse.
- **"Not supported"** otherwise.

**Script:** `tools/loso5.py`. **Outputs:** `results/loso5.csv` (per scene and pooled), `results/loso5_rows.csv`.

## Results (commit 58c2d6b, 03:32 Berlin; output files dated 03:29–03:30, after the plan commit; `results/loso5.csv`, `results/loso5_rows.csv`, `results/loso5.log`; 30 hosted calls)

**90% interval score at +2 s** (lower is better; mean over 3 context seeds):

| Model | ETH | Hotel | Univ | Zara1 | Zara2 |
|---|---|---|---|---|---|
| TabPFN (1,500 rows) | **1.500** | 1.062 | **1.720** | 1.283 | **1.226** |
| LightGBM all rows (4 scenes, ~22k rows at h = 2) | 1.521 | **1.035** | 1.770 | **1.252** | 1.251 |
| LightGBM same 1,500 rows | 1.650 | 1.080 | 1.999 | 1.346 | 1.368 |
| CV + band | 1.686 | 1.337 | 2.328 | 1.543 | 1.722 |

**Joint 5–95% coverage (target 81%):**
- TabPFN: 82.0 / 87.0 / 77.5 / 85.6 / 82.1%;
- LightGBM all rows: 76.3 / 85.1 / 72.0 / 85.9 / 84.2%;
- LightGBM same rows: 67.6 / 78.7 / 64.4 / 76.7 / 74.1%.

**Verdict by the rule: "not supported".**
- TabPFN against LightGBM on all rows, pooled: −0.008 [−0.023, +0.007], a tie.
- TabPFN is significantly worse on Hotel (+0.027 [+0.003, +0.052]) and Zara1 (+0.032 [+0.002, +0.067]), and better on Univ (−0.050 [−0.095, −0.010]).
- Against CV + band it is better in every scene (pooled −0.365 [−0.405, −0.328]).

**What does hold on five scenes:**
- **Data efficiency.** On the same 1,500 rows, TabPFN beats LightGBM pooled (−0.130 [−0.154, −0.107]), clearly in 4 of 5 scenes (Hotel −0.018 [−0.054, +0.018], level).
- **TabPFN with 1,500 rows ties LightGBM trained on about 15 times more data from four places.**
- **It has the most even coverage:** 77.5–87% across scenes, against 72–86% for LightGBM on all rows and 64–79% on the same rows.

**Why this differs from the two-scene claim A.**
- With Hotel → ETH, LightGBM on "all rows" had one small scene to learn from.
- With four varied scenes it learns enough to match TabPFN.
- So TabPFN's edge is strongest when data is scarce and narrow, which is the "new site, little data" case, and it fades once a rival has plenty of varied data.

**Caveat:** Zara1 and Zara2 are the same camera (identical `H.txt`), so their folds share a site with the training data and are not fully "never-seen places".
