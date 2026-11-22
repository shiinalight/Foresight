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
--matched safety (EXPLORATORY, added after the oracle gate was seen, no claim-C verdict): the axes swapped. For
  each point b of B, A's safety at b's time_capped, read along A's front (faster never safer), minus b's safety:
  min_gap higher by more than --m-gap (default 0.158 m) and moving-collision rate lower by more than --m-mc (default
  0.244); the margins are the detectable paired differences at 40 seeds (80% power, 5% two-sided) from the T2
  baseline rows in the chosen cell. Outside A's time range: dominance only (a point of A at least as fast).
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
    files = [f for f in sorted((ROOT / "results/closed").glob(pattern + ".csv"))
             if not f.name.startswith("robotfeat_") and "-sm" not in f.stem and "-r02" not in f.stem
             and "-lat" not in f.stem]
    d = pd.concat([pd.read_csv(f).assign(_file=f.name) for f in files], ignore_index=True)
    for col, keep in (("smooth", False), ("latency", 0.0), ("replan", 0.4)):     # the primary analysis set only
        if col in d:
            d = d[d[col].fillna(keep) == keep]
    d = d[d.seed.isin(seeds) & (d.features_version == "f4")]
    if extra:
        d = d.query(extra, engine="python")
    d = d.copy()
    d["dial"] = [getattr(r, DIAL[r.planner]) if r.planner in DIAL else (np.nan if r.planner == "stop_go" else r.box_q)
                 for r in d.itertuples()]
    d["arm"] = d.planner + ":" + d.dial.map(lambda v: "-" if pd.isna(v) else f"{v:g}")
    dup = d[d.duplicated(["arm", "seed"], keep=False)]
    assert dup.empty, f"{len(dup)} duplicate (arm, seed) rows, e.g. {dup[['arm', 'seed', '_file']].head(6).to_dict('records')}"
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


def diff_safety(P, a_arms, b_arm, axis):
    """A's safety minus b's at b's time (along A's front: faster never safer), or dominance, or None."""
    v = (lambda p: p[1]) if axis == "gap" else (lambda p: -p[2])
    pa = [P[a] for a in a_arms]; pb = P[b_arm]
    fr = []
    for p in sorted(pa, key=lambda p: (p[0], -v(p))):         # fastest first; keep points safer than every faster one
        if not fr or v(p) > fr[-1][1]:
            fr.append((p[0], v(p)))
    xs, ys = np.array([f[0] for f in fr]), np.array([f[1] for f in fr])
    if xs.min() - 1e-12 <= pb[0] <= xs.max() + 1e-12:
        return float(np.interp(pb[0], xs, ys)) - v(pb), "interp"
    dom = [v(p) - v(pb) for p in pa if p[0] <= pb[0] + 1e-12]
    if dom:
        return max(dom), "dominance"
    return None, "not comparable"


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


def compare(d, a_planner, b_planner, n_boot=2000, seed=0, matched="time", margins=None):
    fn = diff_point if matched == "time" else diff_safety
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
            est, how = fn(P, arms_a, b, axis)
            if est is None:
                rows.append(dict(a=a_planner, b=b, axis=axis, how=how, diff=np.nan, lo=np.nan, hi=np.nan, p=1.0, beats=False))
                continue
            bs = []
            for Pb in boots:
                v, h2 = fn(Pb, arms_a, b, axis)
                bs.append(np.nan if v is None else v)
            bs = np.nan_to_num(np.array(bs, float), nan=-np.inf)  # a resample with no comparison counts against A
            lo, hi = np.percentile(bs, [2.5, 97.5])
            p = float(np.mean(~(bs > 0)))                       # one-sided bootstrap p-value
            rows.append(dict(a=a_planner, b=b, axis=axis, how=how, diff=est, lo=lo, hi=hi, p=p,
                             beats=bool(est > (D14 if matched == "time" else margins[axis]) and lo > 0)))
    return pd.DataFrame(rows), P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="results/closed file pattern without .csv")
    ap.add_argument("--seeds", default="4100-4119")
    ap.add_argument("--a", default="oracle")
    ap.add_argument("--b", nargs="*", default=list(BASELINES))
    ap.add_argument("--query", default=None, help="extra pandas query on the rows")
    ap.add_argument("--holm", action="store_true")
    ap.add_argument("--matched", default="time", choices=["time", "safety"])
    ap.add_argument("--m-gap", type=float, default=0.158)
    ap.add_argument("--m-mc", type=float, default=0.244)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    s0, s1 = map(int, a.seeds.split("-"))
    d, seeds = load(a.glob, set(range(s0, s1 + 1)), a.query)
    print(f"{len(seeds)} common seeds; arms: {', '.join(sorted(d.arm.unique()))}")
    res, Ps = [], {}
    for b in a.b:
        if b not in set(d.planner):
            print(f"  missing baseline {b}"); continue
        r, P = compare(d, a.a, b, matched=a.matched, margins={"gap": a.m_gap, "mc": a.m_mc})
        res.append(r); Ps.update(P)
    R = pd.concat(res, ignore_index=True)
    pts = pd.DataFrame([(k, *v) for k, v in Ps.items()], columns=["arm", "time_capped", "min_gap", "mc_rate"]).sort_values("arm")
    pd.set_option("display.width", 220)
    print("\npoints (paired seeds):\n" + pts.round(3).to_string(index=False))
    what = ("baseline time - A time at matched safety, s" if a.matched == "time"
            else "A safety - baseline safety at matched time (min_gap m; collision rate, lower = better)")
    print(f"\npoint comparisons (diff = {what}):\n" + R.round(3).to_string(index=False))
    B = R.assign(base=R.b.str.split(":").str[0]).groupby("base").agg(beats=("beats", "all"), p=("p", "max"))
    if a.holm:
        order = B.sort_values("p").index.tolist(); m = len(order); ok = True
        B["holm_pass"] = False
        for i, k in enumerate(order):
            ok = ok and B.loc[k, "p"] <= 0.05 / (m - i)
            B.loc[k, "holm_pass"] = ok
    print("\nper baseline:\n" + B.to_string())
    allb = bool(B.beats.all()) and set(B.index) == set(a.b)
    print(f"\n{a.a} beats every baseline curve ({'matched safety' if a.matched == 'time' else 'matched time, exploratory'}): {allb}")
    if a.out:
        R.to_csv(a.out, index=False); pts.to_csv(a.out.replace(".csv", "_points.csv"), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
