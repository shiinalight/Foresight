# Exploratory: do the closed-loop findings hold in other crowds? (written before any run, Sun 22 Nov ~04:20 Berlin)

**Status:** exploratory. T5 (seeds 3000–3019, chosen cell) stays the confirmatory result.

**Why.** Every closed-loop result so far comes from one simulator cell: the crossing crowd with repulsion 2, 20% of people ignoring the robot, 24 people. That cell was chosen by the T2 rule. This run checks two more crowds. They differ in goals, density and how people treat the robot, which tells us whether the verdicts (claim C, in-loop claim B, the oracle's clearance) depend on that one cell.

**Scenarios** (both keep T5's protocol: every seed switches into erratic at 5 s, and the bank excludes erratic so the change is unseen):

| Id | Crowd | Settings | Bank |
|---|---|---|---|
| G1 open plaza | people walk to goals anywhere (`--goal-mode anywhere`), 16 people, default repulsion, nobody ignores the robot | `--people 16` | `data/crowd` (crowd-v1, 120 episodes), erratic excluded |
| G2 dense crossing | crossing crowd, 32 people, half ignore the robot | `--goal-mode cross --people 32 --robot-repulsion 2.0 --ignore-robot 0.5` | `data/crowd_cross_t2`, erratic excluded |

**Seeds:** 3200–3219, fresh.

**Arms:**
- tabpfn_split q 0.15 and 0.35;
- lgbm_all q 0.15 and 0.35 (laptop, like the dial sweep);
- cv m 0.2 / 0.4 / 0.6;
- reactive m 0.4 / 0.6;
- vo w 0.5 / 1 / 2;
- stop_go;
- open-loop oracle m 0.2 / 0.4.

TabPFN arms and baselines run on the L4 pod. Batch offset on the learned arms, smoothing OFF, latency 0, replan 0.4 s, top speed 1.0 m/s, frozen context 2,000 rows (fill).

**Analysis (fixed now), per scenario:**
- `tools/curve_compare.py --holm` for tabpfn_split and for lgbm_all against the baseline curves (T5 rules and margins).
- In-loop claim B for split against lgbm_all after the switch: coverage, interval score and error, with paired-seed bootstrap.
- The oracle's points.

**Reporting rule:** report every scenario and arm. A finding counts as "holds across crowds" only if it has the same direction in T5, G1 and G2.

## Amendment (~04:45 Berlin, before any result was read; only episode counts seen)
The pod's 5-core CPU quota made the TabPFN arms crawl (0–1 of 20 episodes after 12 min). The G2 baselines (stop_go excepted, done) and G2 oracle move to the laptop (numpy only, no TabPFN), restarted from scratch there. G1 baselines stay on the pod. Tables disclose the device per arm.

## Results (Sun 22 Nov ~05:20 Berlin; `results/g{1,2}_{tabpfn_split,lgbm_all}.csv`, `results/inloop_claimb_g{1,2}.csv`)

Devices: TabPFN arms and G1 baselines ran on the L4 pod; lgbm_all (both crowds) and the G2 baselines and oracle ran on the laptop (amendment). Every arm has 20 seeds (3200–3219).

**Points** (mean time_capped s / min gap m / moving-collision rate):

| | G1 open plaza (16 people) | G2 dense crossing (32 people, half ignore the robot) |
|---|---|---|
| tabpfn_split q 0.15 / 0.35 | 26.41 / 0.497 / 0% · 21.88 / 0.331 / 5% | 30.09 / −0.034 / 40% · 24.53 / −0.008 / 35% |
| cv m 0.2 / 0.4 / 0.6 | 21.08 / 0.219 / 5% · 22.63 / 0.285 / 0% · 24.81 / 0.432 / 0% | 23.91 / −0.158 / 65% · 26.30 / −0.138 / 65% · 29.52 / −0.041 / 45% |
| reactive m 0.4 / 0.6 | 22.30 / 0.401 / 0% · 24.07 / 0.414 / 0% | 27.95 / −0.180 / 70% · 30.64 / −0.211 / 40% |
| vo w 0.5 / 1 / 2 | 18.95 / 0.191 / 10% · 20.04 / 0.189 / 0% · 20.85 / 0.202 / 5% | 22.11 / −0.187 / 70% · 23.61 / −0.199 / 75% · 25.64 / −0.194 / 70% |
| stop_go | 28.15 / 0.236 / 0% | 37.39 / −0.503 / 15% |
| oracle m 0.2 / 0.4 | 19.37 / 0.271 / 0% · 20.31 / 0.444 / 0% | 22.79 / 0.155 / 5% · 25.99 / 0.256 / 5% |

**Pre-specified verdicts:**
- **Claim C: not supported in either crowd.**
  - tabpfn_split beats no baseline curve under Holm: G1 p = 0.98 cv, 0.92 reactive, 0.037 stop_go (fails Holm), 1.0 vo; G2 p = 0.90 / 0.20 / 1.0 / 1.0.
  - lgbm_all beats only stop_go in G1 (p 0.0085) and nothing in G2.
  - So **"claim C fails" holds across all three crowds.**
- **In-loop claim B against LightGBM, after the switch: holds in all three crowds.**

  | | Coverage, split vs lgbm_all | Interval score difference | Error difference |
  |---|---|---|---|
  | G1 | 70.5% vs 63.8%, +6.6 pts [+5.7, +7.5] | −0.30 [−0.37, −0.22] | +0.028 m [+0.009, +0.048] |
  | G2 | 61.1% vs 41.3%, +19.7 pts [+18.7, +20.8] | −2.05 [−2.21, −1.89] | −0.035 m [−0.056, −0.016] |
  | T5 | | +18.0 pts, −2.00 | |

  The size depends on the crowd: small in the calm open plaza, large in the dense crossing. Frozen TabPFN was not run here, so split − frozen is T5-only.

**Not pre-specified (labelled exploratory):**
- **G2 safety at similar time.** At q 0.35 the split planner takes 24.53 s with −0.008 m clearance and 35% collision crowds. cv m0.2 takes 23.91 s at −0.158 m and 65%; vo w1 takes 23.61 s at −0.199 m and 75%. That is roughly half the collision crowds at about the same time.
- The matched-time curve test (`--matched safety`) does not pass Holm (reactive p 0.32, others ≥ 0.94), so this stays a description, not a result.
- In the hardest crowd, the calibrated planner is the only learned or classical planner that keeps clearance near zero rather than negative.
