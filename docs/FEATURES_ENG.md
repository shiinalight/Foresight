# Exploratory: engineered features on the five-scene leave-one-out test (written before any TabPFN run, Sun 22 Nov)

**Why.** Darshak asked whether "not just throwing raw data at it, but extracting meaningful properties, numbers, calculations" helps. The 39 f4 features (`foresight/core/features.py`) are mostly raw: 5 history points, the 3 nearest neighbours' relative x/y/vx/vy/dist, robot and obstacle vectors, accelerations, density. On the five-scene leave-one-out test (`docs/LOSO5.md`) TabPFN with 1,500 rows only **tied** LightGBM trained on all four other scenes (pooled interval score −0.008 [−0.023, +0.007]). That is the result most worth improving. This test asks whether properties computed from the same 39 numbers change it.

**What the 39 columns are** (read from `features.py`, f4):
- Every x/y is in the predicted agent's own frame at time t: x = its heading, y = its left.
- `n{k}_x, n{k}_y`: neighbour k's position minus the agent's, k = 0..2 by distance, robot excluded.
- `n{k}_vx, n{k}_vy`: neighbour velocity **minus** the agent's (relative), rotated into the agent frame.
- A missing neighbour is padded x = 99, y = 0, vx = vy = 0, dist = 99.
- `hist_x{j}, hist_y{j}`: position at t − j·step minus the position at t; j = 1 is the most recent, j = 5 the oldest.
  - step = max(1, round(0.2 s / dt))·dt: **0.4 s on ETH/UCY** (window 2.0 s), 0.2 s on the simulator.
- `robot_*`: relative position, distance and closing speed, padded (99, 0, 99, 0) without a robot. `obst_*`: padded (99, 0, 99) without obstacles.
- **On all five scenes, robot, obstacle and radius are constant** (no robot, no obstacles, radius 0.25). This test cannot evaluate robot or obstacle features.

## The engineered features (`tools/feat_eng.py`, `engineer(df, hist_dt=0.4)`)

They are added columns computed only from the 39 columns of the same row: no other row, no future, no core change. 21 columns:

| Column | Definition | Why |
|---|---|---|
| `n{k}_tcpa` (k = 0..2) | time to closest approach under constant relative velocity, −(p·w)/\|w\|², clipped to [0, 5] s; \|w\|² < 1e-6 → 0 | people react to who will pass close soon, not to who is near now |
| `n{k}_dcpa` | distance at the clipped TCPA, \|p + w·tcpa\| (receding → current distance) | how close that pass will be |
| `threat_dcpa` | min DCPA over neighbours that are approaching (raw TCPA > 0) and reach closest approach within 2 s | the one encounter that should drive a dodge |
| `threat_tcpa` | that neighbour's TCPA | how urgent it is |
| `threat_side` | that neighbour's lateral (y) offset at closest approach: + passes on the left, − on the right | which way to dodge (the lateral target) |
| `risk` | Σ over approaching neighbours of exp(−DCPA²/2·1 m²)·exp(−TCPA/2 s) | smooth version of threat that adds up several encounters |
| `robot_ttc` | robot distance / closing speed when closing > 0.05 m/s, clipped to [0, 5] s | time to contact with the robot (constant 5 here, no robot) |
| `ahead_n` | neighbours in the ±45° forward cone within 3 m (of the 3 known) | blocked path |
| `ahead_dist` | distance to the nearest of them | how soon the agent must slow down or swerve |
| `ahead_vx` | that neighbour's relative longitudinal velocity (< 0: it is slower, closing) | following a slower walker |
| `comp_relspeed` | min \|relative velocity\| over neighbours within 1.5 m | walking together (a companion moves with you) |
| `comp_n` | neighbours within 1.5 m with \|relative velocity\| < 0.5 m/s | group size |
| `speed_trend` | speed − speed over the oldest two history segments (t − 2.0 … t − 1.2 s) | accelerating or slowing down |
| `path_speed` | history path length / 2.0 s | mean speed over the window, less noisy than one Kalman sample |
| `heading_change` | heading of the newest two segments minus the oldest two (rad, + = left); 0 if either chord < 0.1 m | already turning |
| `straightness` | net displacement / path length over the window; 1 if the path < 0.2 m | wandering versus walking straight |
| `curvature` | heading_change / path length (rad/m), clipped to ±5; 0 under the same gates | how sharp the turn is |

**Missing values (no NaN or inf by choice):**
- There are no NaN or inf values; the sentinels are listed below.
- Neighbour-related sentinels:
  - a padded neighbour has TCPA = 5 s and DCPA = 99;
  - no threat: `threat_dcpa` 99, `threat_tcpa` 5, `threat_side` 0;
  - no one ahead: `ahead_dist` 99, `ahead_vx` 0;
  - no one within 1.5 m: `comp_relspeed` 99.
- 99 matches the raw padding convention, and 5 s is the TCPA cap.
- Standing or near-standing history: the gates above give 0 (heading change, curvature) or 1 (straightness).
- TabPFN handles NaN natively, but sentinels keep LightGBM and TabPFN on identical inputs.

**Known limits:**
- Only the 3 nearest neighbours are visible, so the cone and companion counts are at most 3.
- Standing agents (33–40% of Hotel, Zara2 and Univ rows) face their nearest neighbour (f2 rule). For them n0 sits at angle 0, so the forward-cone features partly act as a "standing" flag.
- The history positions are Kalman-filtered, so `speed_trend` and `heading_change` are smooth.

**Sanity check** (`python tools/feat_eng.py --check`, before any model): 0 NaN/inf cells in all five scenes.
- Medians, ETH / Hotel / Zara1 / Zara2 / Univ:
  - `n0_dcpa` 0.98 / 0.97 / 0.88 / 0.70 / 0.61 m;
  - `risk` 0.08 / 0.11 / 0.09 / 0.08 / 0.29;
  - `path_speed` 1.46 / 1.13 / 1.16 / 0.90 / 0.58 m/s.
- Share of rows with no threat: 54 / 47 / 57 / 58 / 43%.
- Univ is the crowded scene: smallest DCPAs, highest risk, least straight paths (5th percentile straightness 0.43).

## Variants (column lists frozen here)

- **f4** (39): `h_f` + `feature_names()[1:]`. This is the baseline.
  - TabPFN-f4 rows are reused from `results/loso5_rows.csv`. The script asserts that the test rows and persons match, and reruns `lgbm_all`-f4 to compare with the stored scores.
- **eng** (60): f4 + the 21 engineered columns.
- **compact** (34): the 21 engineered columns plus 13 kept from f4: `h_f, speed, a_lon, a_lat, turn_rate, radius, n0_dist, n1_dist, n2_dist, robot_dist, robot_closing, obst_dist, density_3m`.
  - It drops the raw history and the raw neighbour x/y/vx/vy. That tests "fewer, more meaningful features".

**Design: identical to LOSO5.**
- Five held-out scenes, each with the same up to 1,500 test rows at h = 2.0 s (random_state 0).
- Context: 1,500 rows, 375 per training scene, seeds 0, 1, 2.
- Models, each on each variant's columns:
  - TabPFN (hosted, default model);
  - LightGBM m50:n100 on all rows of the four training scenes (`lgbm_all`).

**Metrics** at +2 s:
- primary: pooled 90% interval score;
- also median error, joint 5–95% coverage and width.
- Paired person bootstrap (2,000 draws, persons resampled within scene), with `boot()` and `scores()` imported from `tools/loso5.py`.

**Rule (fixed now):**
- A variant is **adopted for TabPFN** if two things are true:
  - its pooled interval score beats TabPFN-f4 with a 95% interval excluding 0;
  - it is significantly worse (interval above 0) in no scene.
- Also reported, without a verdict:
  - TabPFN-variant against `lgbm_all`-variant (same features);
  - `lgbm_all`-variant against `lgbm_all`-f4;
  - median error and joint coverage.
- Two variants are tested against one baseline. A pooled interval that only just excludes 0 should be read with that in mind.

**Budget.**
- 2 variants × 5 scenes × 3 seeds × 2 axes = **60 hosted regressor calls**.
- The measured price is about 10k credits per call at ≤ 2,000 × 39 (highway correction), so roughly 0.6–1.8M credits.
- The script reads the server usage after the first fit and **stops if the projected total exceeds 3,000,000 credits**.
- Every TabPFN quantile array is cached per (variant, scene, seed) in `results/feat_eng_cache/`, so a restart does not re-spend.
- Server usage before and after goes to `results/api_usage.csv`.

**Disclosure.**
- Before this doc was committed, a `--dry` plumbing pass ran with LightGBM standing in for TabPFN. It printed the `lgbm_all` scores for all three variants:
  - `lgbm_all`-eng against `lgbm_all`-f4 pooled −0.001 [−0.009, +0.006];
  - compact +0.033 [+0.021, +0.046].
- The feature definitions and column lists were written before that pass and not changed after it. No TabPFN score on any variant has been seen.

**Script:** `tools/feat_eng.py` (`--check`, `--count-only`, `--dry`, default = hosted run).
**Outputs:**
- `results/feat_eng.csv`: comparisons per scene and pooled;
- `results/feat_eng_rows.csv`;
- `results/feat_eng_gain.csv`: LightGBM gain share per feature;
- `results/feat_eng.log`.
