# Exploratory: caution-dial sweep on fresh seeds (written before any run, Sun 22 Nov 03:00 Berlin)

**Status: exploratory.** T5 (seeds 3000–3019, RESULTS.md "Closed loop T5 results") stays the confirmatory result. Nothing here changes its verdict.

**Why.**
- In T5 the TabPFN planner ran only at q 0.10 and 0.15, which gave 26–29 s. The baselines sat at 20–26 s, so most primary rows could only be judged by dominance or were "not comparable".
- A check on existing rows also showed that the split, frozen TabPFN and LightGBM planners all take about 26 s at q 0.15:
  - split − frozen: −0.12 s [−1.34, +1.26];
  - split − lgbm_all: +0.33 s [−1.34, +2.02].
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
