"""Showcase seed by the CTO's rule: a T5 switch seed (every T5 seed switches) on which every primary-test planner
reached the goal, and tabpfn_split q0.15's time_capped is closest to its median over seeds 3000-3019.
Primary-test planners here: tabpfn_split q0.15 and q0.10, and the baselines at their T2 settings (stop_go, cv m0.4,
reactive m0.4, vo w1). Ties go to the lower seed.
"""
import glob

import pandas as pd

ARMS = {"split-q15": None, "split-q10": None, "stopgo": None, "cvm04": None, "rea04": None, "vo10": None}
rows = {}
for tag in ARMS:
    f = [x for x in glob.glob(f"results/closed/*-t5-{tag}.csv") if "robotfeat" not in x][0]
    d = pd.read_csv(f); d = d[d.seed.between(3000, 3019)]
    rows[tag] = d.set_index("seed")
seeds = sorted(set.intersection(*[set(v.index) for v in rows.values()]))
ok = [s for s in seeds if all(bool(rows[t].loc[s, "reached"]) for t in rows)]
sp = rows["split-q15"]
med = sp.loc[seeds, "time_capped"].median()
best = min(ok, key=lambda s: (abs(sp.loc[s, "time_capped"] - med), s))
print(f"{len(seeds)} common seeds; {len(ok)} where every primary-test planner reached the goal: {ok}")
print(f"tabpfn_split q0.15 median time_capped {med:.2f} s; showcase seed {best} (time {sp.loc[best, 'time_capped']:.1f} s, "
      f"min gap {sp.loc[best, 'min_gap']:.2f} m)")
print(pd.DataFrame({t: rows[t].loc[best, ["time_capped", "min_gap", "moving_collision_steps"]] for t in rows}).T.to_string())
