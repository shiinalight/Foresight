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
