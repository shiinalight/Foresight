"""Matched-safety curve comparison, the headline rule of pre-registration amendment 1 (RESULTS.md).

    python tools/curve_compare.py --glob "*t2c-r20-i02-p24-*" --seeds 4100-4119 --a oracle            # oracle gate
    python tools/curve_compare.py --glob "*t5-*" --seeds 3000-3039 --a tabpfn_split --holm              # T5 primary

Curves: each planner's arms (its dial: margin, vo weight or box_q) give points of mean time_capped against
  (a) mean min_gap (higher = safer) and (b) moving-collision rate (share of seeds with >= 1 moving-collision step,
  lower = safer). A curve is read on its Pareto front (safer never faster), so interpolation is well defined.
A beats one point b of B, on one axis, when:
  - b's safety level lies inside A's front: A's time interpolated at that level is lower than b's by more than
    2 s (D-14) and the 95% paired-seed bootstrap interval of the difference excludes 0;
  - outside: dominance only (no extrapolation): some point of A is at least as safe as b on BOTH axes and more
    than 2 s faster, with the paired interval excluding 0. Otherwise "not comparable" (= not beaten).
A beats B when it beats every point of B on both axes. Only seeds present in every arm are used.
--holm: Holm-Bonferroni across the baselines at 5% on the one-sided bootstrap p-values (worst point, worst axis).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIAL = {"cv": "margin", "reactive": "margin", "oracle": "margin", "vo": "vo_w"}
BASELINES = ("stop_go", "cv", "reactive", "vo")
D14 = 2.0


def load(pattern, seeds, extra=None):
    files = [f for f in sorted((ROOT / "results/closed").glob(pattern + ".csv")) if not f.name.startswith("robotfeat_")]
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[d.seed.isin(seeds) & (d.features_version == "f4")]
    if extra:
        d = d.query(extra)
    d = d.copy()
    d["dial"] = [getattr(r, DIAL[r.planner]) if r.planner in DIAL else (np.nan if r.planner == "stop_go" else r.box_q)
                 for r in d.itertuples()]
    d["arm"] = d.planner + ":" + d.dial.map(lambda v: "-" if pd.isna(v) else f"{v:g}")
    d = d.drop_duplicates(["arm", "seed"], keep="last")
    common = set.intersection(*[set(g.seed) for _, g in d.groupby("arm")])
    return d[d.seed.isin(common)], sorted(common)


def points(d, arms, idx=None):
    """Per arm: mean time, mean min_gap, moving-collision rate over seeds (idx = bootstrap resample of seeds)."""
    out = {}
    for arm in arms:
        g = d[arm]                                           # arrays aligned by seed
        sl = slice(None) if idx is None else idx
        out[arm] = (g["time"][sl].mean(), g["gap"][sl].mean(), (g["mc"][sl] > 0).mean())
    return out


def front(pts, axis):
    """Pareto front of (safety, time) for one axis: safety ascending in 'safer' direction, time non-decreasing."""
    s = [(p[1] if axis == "gap" else -p[2], p[0]) for p in pts]   # higher s = safer
    s = sorted(s, key=lambda x: (x[0], x[1]))
    fr = []
    for si, ti in sorted(s, key=lambda x: (-x[0], x[1])):         # from safest down; keep points faster than all safer
        if not fr or ti < fr[-1][1]:
            fr.append((si, ti))
    return sorted(fr)


def interp(fr, s):
    xs, ys = np.array([f[0] for f in fr]), np.array([f[1] for f in fr])
    if s < xs.min() - 1e-12 or s > xs.max() + 1e-12:
        return None
    return float(np.interp(s, xs, ys))


def diff_point(P, a_arms, b_arm, axis):
    """b_time - A_time at b's safety on this axis (interpolated), or dominance difference, or None."""
    pa = [P[a] for a in a_arms]; pb = P[b_arm]
    sb = pb[1] if axis == "gap" else -pb[2]
    t = interp(front(pa, axis), sb)
    if t is not None:
        return pb[0] - t, "interp"
    dom = [pb[0] - p[0] for p in pa if p[1] >= pb[1] - 1e-12 and p[2] <= pb[2] + 1e-12]
    if dom:
        return max(dom), "dominance"
    return None, "not comparable"


def compare(d, a_planner, b_planner, n_boot=2000, seed=0):
    arms_a = sorted(d[d.planner == a_planner].arm.unique()); arms_b = sorted(d[d.planner == b_planner].arm.unique())
    seeds = sorted(d.seed.unique())
    D = {arm: {k: d[d.arm == arm].set_index("seed").loc[seeds, c].to_numpy(float)
               for k, c in (("time", "time_capped"), ("gap", "min_gap"), ("mc", "moving_collision_steps"))}
         for arm in arms_a + arms_b}
    P = points(D, arms_a + arms_b)
    rng = np.random.default_rng(seed)
    boots = [points(D, arms_a + arms_b, rng.integers(0, len(seeds), len(seeds))) for _ in range(n_boot)]
    rows = []
    for b in arms_b:
        for axis in ("gap", "mc"):
            est, how = diff_point(P, arms_a, b, axis)
            if est is None:
                rows.append(dict(a=a_planner, b=b, axis=axis, how=how, diff=np.nan, lo=np.nan, hi=np.nan, p=1.0, beats=False))
                continue
            bs = []
            for Pb in boots:
                v, h2 = diff_point(Pb, arms_a, b, axis)
                bs.append(np.nan if v is None else v)
            bs = np.nan_to_num(np.array(bs, float), nan=-np.inf)  # a resample with no comparison counts against A
            lo, hi = np.percentile(bs, [2.5, 97.5])
            p = float(np.mean(~(bs > 0)))                       # one-sided bootstrap p-value
            rows.append(dict(a=a_planner, b=b, axis=axis, how=how, diff=est, lo=lo, hi=hi, p=p,
                             beats=bool(est > D14 and lo > 0)))
    return pd.DataFrame(rows), P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="results/closed file pattern without .csv")
    ap.add_argument("--seeds", default="4100-4119")
    ap.add_argument("--a", default="oracle")
    ap.add_argument("--b", nargs="*", default=list(BASELINES))
    ap.add_argument("--query", default=None, help="extra pandas query on the rows")
    ap.add_argument("--holm", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    s0, s1 = map(int, a.seeds.split("-"))
    d, seeds = load(a.glob, set(range(s0, s1 + 1)), a.query)
    print(f"{len(seeds)} common seeds; arms: {', '.join(sorted(d.arm.unique()))}")
    res, Ps = [], {}
    for b in a.b:
        if b not in set(d.planner):
            print(f"  missing baseline {b}"); continue
        r, P = compare(d, a.a, b); res.append(r); Ps.update(P)
    R = pd.concat(res, ignore_index=True)
    pts = pd.DataFrame([(k, *v) for k, v in Ps.items()], columns=["arm", "time_capped", "min_gap", "mc_rate"]).sort_values("arm")
    pd.set_option("display.width", 220)
    print("\npoints (paired seeds):\n" + pts.round(3).to_string(index=False))
    print("\npoint comparisons (diff = baseline time - A time at matched safety, s):\n" + R.round(3).to_string(index=False))
    B = R.assign(base=R.b.str.split(":").str[0]).groupby("base").agg(beats=("beats", "all"), p=("p", "max"))
    if a.holm:
        order = B.sort_values("p").index.tolist(); m = len(order); ok = True
        B["holm_pass"] = False
        for i, k in enumerate(order):
            ok = ok and B.loc[k, "p"] <= 0.05 / (m - i)
            B.loc[k, "holm_pass"] = ok
    print("\nper baseline:\n" + B.to_string())
    allb = bool(B.beats.all()) and set(B.index) == set(a.b)
    print(f"\n{a.a} beats every baseline curve: {allb}")
    if a.out:
        R.to_csv(a.out, index=False); pts.to_csv(a.out.replace(".csv", "_points.csv"), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
