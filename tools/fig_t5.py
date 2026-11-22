"""T5 figure: closed-loop speed vs safety per planner (left) and in-loop calibration before/after the switch (right).

    python tools/fig_t5.py [--seeds 3000-3019]     -> media/fig_t5.png

Left: mean time_capped against mean min_gap, one line per planner over its dial (paired seeds, smoothing OFF,
latency 0, replan 0.4), the open-loop oracle dashed. Right: joint coverage of the raw 5-95% box at +2 s against true
positions, before and after the switch into erratic, for tabpfn_split, frozen tabpfn and lgbm_all at q0.15, mean over
seeds with 95% seed-bootstrap intervals; the 81% target dotted. Labels and legends only; numbers live in RESULTS.md.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import curve_compare as CC  # noqa: E402

# fixed categorical order (validated default palette: blue, orange, teal, purple, grey ramp)
COL = {"tabpfn_split": "#2a6fdb", "cv": "#e8590c", "vo": "#0f9d8a", "reactive": "#7a3fb5", "stop_go": "#6e7781",
       "oracle": "#1f2328", "tabpfn": "#8fb3f0", "lgbm_all": "#c7860a"}
NAME = {"tabpfn_split": "TabPFN, frozen + this scene (ours)", "cv": "Constant velocity", "vo": "Velocity obstacles",
        "reactive": "Reactive", "stop_go": "Stop-and-go", "oracle": "Open-loop oracle", "tabpfn": "TabPFN, frozen",
        "lgbm_all": "LightGBM, all rows"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="3000-3019"); a = ap.parse_args()
    s0, s1 = map(int, a.seeds.split("-"))
    d, seeds = CC.load("*-t5-*", set(range(s0, s1 + 1)),
                       "planner in ['tabpfn_split','cv','reactive','vo','stop_go','oracle']")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 6.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for p in ("cv", "vo", "reactive", "oracle", "tabpfn_split", "stop_go"):
        g = d[d.planner == p].groupby("dial", dropna=False).agg(t=("time_capped", "mean"), gap=("min_gap", "mean"))
        g = g.sort_index()                                        # connect in dial order (caution rising)
        ls = "--" if p == "oracle" else "-"
        ax.plot(g.gap, g.t, ls, marker="o", ms=8, lw=2, color=COL[p], label=NAME[p],
                markeredgecolor="white", markeredgewidth=2, zorder=3 if p == "tabpfn_split" else 2)
        unit = {"cv": "m ", "reactive": "m ", "oracle": "m ", "vo": "w ", "tabpfn_split": "q "}.get(p)
        if unit:
            for dial, row in g.iterrows():
                ax.annotate(f"{unit}{dial:g}", (row.gap, row.t), xytext=(6, -3), textcoords="offset points",
                            fontsize=7.5, color="#57606a")
    ax.set_xlabel("Closest gap to a person, mean (m)  →  safer")
    ax.set_ylabel("Time to goal, mean (s)  ↓  faster")
    ax.set_title(f"Closed loop: speed against safety ({len(seeds)} paired crowds)", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    ax.grid(color="#e6e8eb", lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fs = [f for f in sorted((ROOT / "results/closed").glob("*-t5-*.csv"))
          if not f.name.startswith("robotfeat_") and not any(x in f.stem for x in ("-sm", "-r02", "-lat"))]
    r = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    r = r[r.seed.between(s0, s1) & r.planner.isin(["tabpfn_split", "tabpfn", "lgbm_all"]) & np.isclose(r.box_q, 0.15)]
    common = set.intersection(*[set(g.seed) for _, g in r.groupby("planner")])
    r = r[r.seed.isin(common)]
    rng = np.random.default_rng(0)
    arms = ["tabpfn_split", "tabpfn", "lgbm_all"]
    x = np.arange(2); w = 0.26
    for i, p in enumerate(arms):
        vals, los, his = [], [], []
        for win in ("pre", "post"):
            v = r[r.planner == p][f"inloop_cov90_{win}"].to_numpy(float) * 100
            bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
            vals.append(v.mean()); los.append(v.mean() - np.percentile(bs, 2.5)); his.append(np.percentile(bs, 97.5) - v.mean())
        bx.bar(x + (i - 1) * w, vals, w - 0.02, color=COL[p], label=NAME[p], yerr=[los, his], capsize=3,
               error_kw=dict(lw=1, ecolor="#57606a"))
    bx.axhline(81, ls=":", color="#57606a", lw=1.2)
    bx.text(1.45, 82, "target ≈ 81%", ha="right", va="bottom", fontsize=9, color="#57606a")
    bx.set_xticks(x, ["Before the switch", "After the switch\n(erratic, never in the bank)"])
    bx.set_ylabel("Box holds the true position at +2 s (%)")
    bx.set_ylim(0, 100)
    bx.set_title(f"In the loop: calibration ({len(common)} paired crowds)", loc="left", fontsize=12)
    bx.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    bx.grid(axis="y", color="#e6e8eb", lw=0.8); bx.set_axisbelow(True)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    fig.tight_layout()
    out = ROOT / "media/fig_t5.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out} ({len(seeds)} seeds left, {len(common)} right)")


if __name__ == "__main__":
    main()
