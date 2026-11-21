"""Exploratory (b) of amendment 2: in-loop claim B. tabpfn_split vs frozen tabpfn vs lgbm_all at box_q 0.15.

    python tools/inloop_claimb.py --glob "*t5-*" --seeds 3000-3039
    python tools/inloop_claimb.py --glob "*t4off-off" --seeds 5100-5105      # works on any runs with inloop_* columns

Per episode (result row): joint coverage of the raw 5-95% box, its width and interval score (alpha 0.1) at +2 s
against true positions, the 5-95% box widened by the offset in force (cov90d), the served box's coverage and the
median error; before the switch (t < switch_t) and after it. Paired by seed (common seeds only); differences
split - other with 95% paired-seed bootstrap intervals (5,000 resamples). Exploratory: no multiplicity correction.
Writes results/inloop_claimb.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("cov90", "w90", "is90", "cov90d", "served_cov", "err")
ARMS = ("tabpfn_split", "tabpfn", "lgbm_all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*t5-*")
    ap.add_argument("--seeds", default="3000-3039")
    ap.add_argument("--box-q", type=float, default=0.15)
    ap.add_argument("--query", default=None)
    ap.add_argument("--out", default="results/inloop_claimb.csv")
    a = ap.parse_args()
    s0, s1 = map(int, a.seeds.split("-"))
    fs = [f for f in sorted((ROOT / "results/closed").glob(a.glob + ".csv")) if not f.name.startswith("robotfeat_")]
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[d.seed.between(s0, s1) & d.planner.isin(ARMS) & np.isclose(d.box_q, a.box_q) & (d.get("latency", 0) == 0)]
    if a.query:
        d = d.query(a.query)
    d = d.drop_duplicates(["planner", "seed"], keep="last")
    common = sorted(set.intersection(*[set(g.seed) for _, g in d.groupby("planner")]))
    d = d[d.seed.isin(common)].set_index(["planner", "seed"]).sort_index()
    print(f"{len(common)} common seeds; arms {sorted(set(d.index.get_level_values(0)))}")
    rows, rng = [], np.random.default_rng(0)
    idx = [rng.integers(0, len(common), len(common)) for _ in range(5000)]
    for w in ("pre", "post"):
        for m in METRICS:
            col = f"inloop_{m}_{w}"
            if col not in d:
                continue
            for p in ARMS:
                if p in d.index.get_level_values(0):
                    rows.append(dict(window=w, metric=m, arm=p, mean=float(d.loc[p, col].mean()), diff=np.nan,
                                     lo=np.nan, hi=np.nan))
            for other in ARMS[1:]:
                if other not in d.index.get_level_values(0):
                    continue
                D = d.loc["tabpfn_split", col].to_numpy(float) - d.loc[other, col].to_numpy(float)
                bs = np.array([np.nanmean(D[i]) for i in idx])
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                rows.append(dict(window=w, metric=m, arm=f"split - {other}", mean=np.nan, diff=float(np.nanmean(D)),
                                 lo=lo, hi=hi))
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(R.round(4).to_string(index=False))
    R.to_csv(ROOT / a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
