# Exploratory: claim A on five real scenes, leave one scene out (written before any run, Sun 22 Nov ~05:00 Berlin)

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
