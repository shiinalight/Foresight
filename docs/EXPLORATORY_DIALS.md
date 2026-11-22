# Exploratory: caution-dial sweep on fresh seeds (written before any run, Sun 22 Nov 03:00 Berlin)

**Status: exploratory.** T5 (seeds 3000–3019, RESULTS.md "Closed loop T5 results") stays the confirmatory result. Nothing here changes its verdict.

**Why.**
- In T5 the TabPFN planner ran only at q 0.10 and 0.15, which gave 26–29 s. The baselines sat at 20–26 s, so most primary rows could only be judged by dominance or were "not comparable".
- A check on existing rows also showed that the split, frozen TabPFN and LightGBM planners all take about 26 s at q 0.15:
  - split − frozen: −0.12 s [−1.35, +1.25];
  - split − lgbm_all: +0.33 s [−1.29, +1.99].
- So the slowness comes from the box planner's caution setting, not from TabPFN's calibration.
- This sweep asks: across the dial, does the planner's time-safety curve reach the baselines' curves, or beat them?

**Where.** A rented NVIDIA L4 pod (`scripts/gpu/pod_run.sh` setup). Every arm, baselines included, runs on that pod, so the paired set is on one device (DEPLOYMENT.md section 4, rule 4).

**Configuration.**
- T5 config: cell r2 i0.2 p24, crossing crowd, `--bank data/crowd_cross_t2 --bank-exclude-regime erratic`, every seed switches into erratic at 5 s, frozen context 2,000 rows (fill), TabPFN-3.5-Fast with 1 estimator.
- Learned arms carry the batch offset (eta_b 0.05).
- Smoothing OFF, latency 0, replan 0.4 s, top speed 1.0 m/s.
- **Seeds:** 3100–3119. These are fresh: no planner has run on them.

**Arms (all dials are reported, none dropped):**
- tabpfn_split: q 0.10, 0.15, 0.25, 0.35, 0.50.
- tabpfn (frozen): q 0.15, 0.50.
- lgbm_all (frozen, deployable rival): q 0.10, 0.15, 0.25, 0.35, 0.50.
- Baselines, same as T5:
  - stop_go;
  - cv m 0.2 / 0.4 / 0.6 / 0.8;
  - reactive m 0.2 / 0.4 / 0.6 / 0.8;
  - vo w 0.5 / 1 / 2;
  - open-loop oracle m 0.2 / 0.4 / 0.6.

**Analysis (fixed now).**
- `tools/curve_compare.py`, same rules as the T5 primary (Pareto curve, interpolation inside the range, dominance outside, the same margins, Holm–Bonferroni over the four baselines), run with:
  - tabpfn_split's full five-dial curve against each baseline curve;
  - lgbm_all's five-dial curve against each baseline curve, for the same question about the deployable rival.
- Also reported: the paired split − lgbm_all and split − frozen differences at each shared dial (time, min gap, collision rate; paired-seed bootstrap 95%).
- Wording rule: a "beats" result here is labelled exploratory, from 20 fresh seeds and a dial range chosen after T5.

## Amendment (Sun 22 Nov ~03:45 Berlin, before any LightGBM, split q 0.25 or split q 0.50 result existed)

**What went wrong on the pod.**
- The pod reports 48 cores, but its cgroup CPU quota is 5.1 cores (`cpu.cfs_quota_us` 510000).
- With 31 processes on 5 cores:
  - the LightGBM arms finished no episode in 35 minutes (LightGBM's threads thrash);
  - three split arms (q 0.10, 0.25, 0.50) crashed with CUDA out of memory at start-up, with 11 TabPFN processes on one 24 GB GPU.

**Changes:**
- **lgbm_all, all five dials, move to the laptop** (Windows x64, OMP threads capped at 2 per process). LightGBM is not TabPFN. The one-device rule (DEPLOYMENT.md section 4) exists for TabPFN's numerics, so these arms are compared with baselines run on the pod. Disclosed in every table that uses them.
- **Pod learned arms:**
  - split q 0.15 and 0.35 keep running;
  - split q 0.25 and 0.50 restart once the latency arm frees the GPU;
  - split q 0.10 is dropped (T5 already covers it on seeds 3000–3039).
- **Frozen tabpfn q 0.15 / 0.50 are dropped** from this sweep to free the GPU and CPU. T5 already compares split and frozen at q 0.15. Two to six of their episodes had printed when they were stopped; none are used.
- **Analysis unchanged:** tabpfn_split's curve is now four dials (0.15, 0.25, 0.35, 0.50).

## Results (Sun 22 Nov ~04:10 Berlin; `results/dx_split.csv`, `results/dx_lgbm.csv`, `tools/curve_compare.py --glob "*dx-*" --seeds 3100-3119 --holm`)

**Points** (20 paired fresh seeds; mean time_capped s / min gap m / moving-collision rate):

| Planner | Dials | Points |
|---|---|---|
| tabpfn_split | q 0.15 / 0.25 / 0.35 / 0.50 | 27.23 / 0.127 / 30% · 26.09 / 0.181 / 15% · **24.37 / 0.137 / 15%** · 22.63 / 0.045 / 40% |
| lgbm_all (laptop) | q 0.10 / 0.15 / 0.25 / 0.35 / 0.50 | 27.68 / 0.251 / 15% · 26.23 / 0.163 / 25% · 26.86 / 0.051 / 35% · 23.52 / −0.038 / 35% · 22.69 / 0.100 / 35% |
| constant velocity | m 0.2 / 0.4 / 0.6 / 0.8 | 21.88 / 0.052 / 25% · 24.69 / 0.060 / 25% · 27.51 / 0.234 / 20% · 29.61 / 0.247 / 25% |
| reactive | m 0.2 / 0.4 / 0.6 / 0.8 | 23.91 / −0.106 / 50% · 26.23 / −0.003 / 25% · 28.03 / 0.102 / 25% · 30.06 / 0.120 / 30% |
| velocity obstacles | w 0.5 / 1 / 2 | 19.71 / −0.068 / 50% · 21.65 / 0.051 / 40% · 22.93 / 0.093 / 30% |
| stop_go | — | 32.56 / −0.048 / 5% |
| open-loop oracle | m 0.2 / 0.4 / 0.6 | 20.45 / 0.208 / 0% · 23.13 / 0.353 / 0% · 25.09 / 0.481 / 0% |

**Verdicts (exploratory):**
- **tabpfn_split beats no baseline curve under Holm.** Per baseline p: cv 1.0, reactive 0.158, stop_go 1.0, vo 1.0.
- **Lower caution brings it into the baselines' range.** At q 0.35 it is 24.37 s with 0.137 m and 15%, against cv m0.4 at 24.69 s / 0.060 m / 25%: about as fast and safer on both axes, but not significant.
- **Against the reactive planner** it is faster at matched safety at every comparable point (+1.3 to +6.7 s; point p 0.024–0.16). The per-baseline rule (every point, Holm) still fails.
- **Velocity obstacles stay faster** at matched safety (−0.4 to −2.9 s).
- **lgbm_all beats no baseline curve either** (reactive p 0.378, others 1.0). Its curve is no better than split's: at about 23 s it holds −0.038 to 0.100 m with 35% collisions, against split q 0.50 at 0.045 m / 40% and q 0.35 at 0.137 m / 15%.
- **The oracle again keeps 0% collisions at every dial.** Perfect prediction buys safety at matched time, but no learned predictor gets close.

## Latency arm (pre-registered in T4 addendum part 3, run on the L4 pod; seeds 3000–3039, tabpfn_split q 0.15, batch offset)

Run both ways on the pod (same device): commands applied with 0.4 s delay (one replan period, stale predictions) against no delay.
- **Time:** 27.32 → 28.67 s, +1.35 s [+0.18, +2.50].
- **Min gap:** 0.231 → 0.133 m, −0.098 [−0.207, +0.003].
- **Crowds with a moving collision:** 7.5% → 30%, +22.5 pts [+10, +37.5].
- **Reached the goal:** 100% → 97.5%.

**Reading:** one step of staleness costs both time and safety. Prediction must arrive within the replan period. The L4's 71 ms does that; the laptop CPU's 452 ms would not. The safety-floor deadline takeover (DEPLOYMENT tier table) is therefore needed, not optional.

Note: the no-delay arm on the pod (27.32 s / 7.5%) differs from the laptop T5 rows at the same seeds (27.20 s / 20%, 40 seeds). This is the cross-device sensitivity documented in DEPLOYMENT.md; the paired comparison above is within one device.
