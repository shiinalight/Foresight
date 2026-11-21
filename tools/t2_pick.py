"""T2 picker, pre-registration amendment 1 (RESULTS.md), applied mechanically to the crossing-crowd sweep.

    python tools/t2_pick.py            # reads results/closed/*-t2c-*.csv (f4, cross), writes results/t2_cross.csv

Per cell (robot_repulsion x ignore_robot x people, seeds 4100-4119):
  best baseline = of stop_go, cv m0.4, reactive m0.4, vo w1: lowest moving-collision rate (share of seeds with
                  at least one moving-collision step); ties go to the lower mean intrusion_s.
  qualifies     = best baseline's moving-collision rate >= 5%, or its mean intrusion_s >= 3 s.
  chosen        = first qualifying cell ordered by highest robot_repulsion, lowest ignore_robot, fewest people.
Only seeds every baseline finished in that cell are used (paired); cells missing a baseline are listed as incomplete.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = {"stop_go": "stop_go", "cv": "cv m0.4", "reactive": "reactive m0.4", "vo": "vo w1"}


def load():
    files = [f for f in sorted((ROOT / "results/closed").glob("*-t2c-*.csv")) if not f.name.startswith("robotfeat_")]
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[(d.features_version == "f4") & (d.goal_mode == "cross") & d.seed.between(4100, 4119)]
    d = d[((d.planner == "cv") & (d.margin == 0.4)) | ((d.planner == "reactive") & (d.margin == 0.4))
          | ((d.planner == "vo") & (d.vo_w == 1.0)) | (d.planner == "stop_go")]
    return d.drop_duplicates(["robot_repulsion", "ignore_robot", "people", "planner", "seed"], keep="last")


def main():
    d = load()
    rows = []
    for (rep, ign, ppl), g in d.groupby(["robot_repulsion", "ignore_robot", "people"]):
        have = set(g.planner)
        seeds = set.intersection(*[set(g[g.planner == p].seed) for p in have]) if have else set()
        g = g[g.seed.isin(seeds)]
        per = g.groupby("planner").agg(mc_rate=("moving_collision_steps", lambda s: (s > 0).mean()),
                                       intrusion_s=("intrusion_s", "mean"), time_capped=("time_capped", "mean"),
                                       min_gap=("min_gap", "mean"), success=("reached", "mean"))
        best = per.sort_values(["mc_rate", "intrusion_s"]).iloc[0]
        complete = have == set(BASE) and len(seeds) == 20
        rows.append(dict(robot_repulsion=rep, ignore_robot=ign, people=ppl, n_seeds=len(seeds), complete=complete,
                         best=BASE[best.name], best_mc_rate=best.mc_rate, best_intrusion_s=best.intrusion_s,
                         qualifies=bool(best.mc_rate >= 0.05 or best.intrusion_s >= 3.0),
                         **{f"{p}_mc": per.mc_rate.get(p) for p in BASE},
                         **{f"{p}_intr": per.intrusion_s.get(p) for p in BASE},
                         **{f"{p}_time": per.time_capped.get(p) for p in BASE}))
    T = pd.DataFrame(rows).sort_values(["robot_repulsion", "ignore_robot", "people"], ascending=[False, True, True])
    T.to_csv(ROOT / "results/t2_cross.csv", index=False)
    pd.set_option("display.width", 250)
    print(T[["robot_repulsion", "ignore_robot", "people", "n_seeds", "complete", "best", "best_mc_rate",
             "best_intrusion_s", "qualifies"]].round(3).to_string(index=False))
    q = T[T.qualifies & T.complete]
    inc = T[~T.complete]
    if len(q):
        c = q.iloc[0]
        earlier_incomplete = inc[(inc.robot_repulsion > c.robot_repulsion)
                                 | ((inc.robot_repulsion == c.robot_repulsion) & (inc.ignore_robot < c.ignore_robot))
                                 | ((inc.robot_repulsion == c.robot_repulsion) & (inc.ignore_robot == c.ignore_robot)
                                    & (inc.people < c.people))]
        print(f"\nchosen cell: robot_repulsion {c.robot_repulsion:g}, ignore_robot {c.ignore_robot:g}, people {c.people:g} "
              f"(best baseline {c.best}: moving-collision rate {100 * c.best_mc_rate:.0f}%, intrusion {c.best_intrusion_s:.2f} s)")
        if len(earlier_incomplete):
            print(f"NOT FINAL: {len(earlier_incomplete)} milder cell(s) still incomplete")
    else:
        print("\nno complete cell qualifies" + (f" ({len(inc)} incomplete)" if len(inc) else " -> stress test fallback"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
