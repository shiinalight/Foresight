"""T1 gate (docs/DECISIONS.md section 3): can the plaza separate planners at all?

    python tools/t1_gate.py

Reads every v4 closed-loop row (crowd, switch on, top speed 1.0, default crowd knobs), keeps the seeds that
every configuration finished (common seeds), and prints per configuration: success, mean time to goal
(failures at the 40 s limit), mean seconds with someone within 0.5 m (intrusion), collision rate, with 95%
paired-seed bootstrap intervals. Draws time against intrusion, one point per configuration.
Gate: a learned planner sits on a better time-vs-intrusion curve than every baseline, by more than the D-14
smallest differences (time 2 s, intrusion 1 s).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight import viz as V  # noqa: E402
from run_closed import load_closed  # noqa: E402

BASE = {"cv", "reactive", "vo", "stop_go"}


def label(r):
    if r.planner == "vo":
        return f"vo w{r.vo_w:g}"
    if r.planner in ("cv", "reactive"):
        return f"{r.planner} m{r.margin:g}"
    if r.planner == "stop_go":
        return "stop_go"
    return f"{r.planner} q{r.box_q:g}"


def boot(x, n=2000, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return np.quantile(m, 0.025), np.quantile(m, 0.975)


def main():
    d = load_closed(version="v4")
    d = d[(d.sim == "crowd") & (d.switch.astype(str) == "True") & (d.max_speed == 1.0) & (d.robot_repulsion == 2.0)
          & (d.ignore_robot == 0.0) & (d.people == 16) & d.seed.between(1000, 1099)]
    d["cfg"] = d.apply(label, axis=1)
    counts = d.groupby("cfg").seed.nunique()
    print("seeds per configuration:", counts.to_dict())
    full = counts[counts >= 60].index
    d = d[d.cfg.isin(full)]
    common = set.intersection(*[set(g.seed) for _, g in d.groupby("cfg")])
    d = d[d.seed.isin(common)]
    print(f"common seeds: {len(common)}")
    rows = []
    for cfg, g in d.groupby("cfg"):
        g = g.sort_values("seed")
        t, i = g.time_capped.to_numpy(), g.intrusion_s.to_numpy()
        rows.append(dict(cfg=cfg, learned=g.planner.iloc[0] not in BASE, n=len(g), success=g.reached.mean(),
                         time=t.mean(), time_lo=boot(t)[0], time_hi=boot(t)[1], intrusion=i.mean(),
                         intr_lo=boot(i)[0], intr_hi=boot(i)[1], collided=g.collided.mean(), min_gap=g.min_gap.mean(),
                         stopped=g.seconds_stopped.mean()))
    R = pd.DataFrame(rows).sort_values("time")
    with pd.option_context("display.width", 200):
        print(R.round(3).to_string(index=False))
    # dominance check: is any learned config better than EVERY baseline config by > 2 s time or > 1 s intrusion
    # without being worse on the other axis?
    base = R[~R.learned]; verdicts = []
    for _, r in R[R.learned].iterrows():
        beats_all = all(((b.time - r.time) > 2.0 and r.intrusion <= b.intrusion) or
                        ((b.intrusion - r.intrusion) > 1.0 and r.time <= b.time) for _, b in base.iterrows())
        verdicts.append(f"{r.cfg}: {'BEATS every baseline' if beats_all else 'does not beat every baseline'}")
    print("\n".join(verdicts))
    fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor="white"); V.style_axes(ax, equal=False)
    for _, r in R.iterrows():
        col = "#e8590c" if "tabpfn" in r.cfg else "#2a6fdb" if r.learned else "#8b949e"
        ax.errorbar(r.intrusion, r.time, xerr=[[r.intrusion - r.intr_lo], [r.intr_hi - r.intrusion]],
                    yerr=[[r.time - r.time_lo], [r.time_hi - r.time]], fmt="o", color=col, capsize=2)
        ax.annotate(r.cfg, (r.intrusion, r.time), xytext=(4, 3), textcoords="offset points", fontsize=7, color=V.C_INK)
    ax.set_xlabel("Mean seconds with someone within 0.5 m (lower is better)", color=V.C_MUTED)
    ax.set_ylabel("Mean time to goal, failures at 40 s (lower is better)", color=V.C_MUTED)
    ax.set_title(f"T1: plaza, {len(common)} common seeds · grey baselines, blue learned", fontsize=11, weight="bold",
                 color=V.C_INK, loc="left")
    fig.tight_layout(); fig.savefig("media/fig_t1_gate.png", dpi=150)
    R.to_csv("results/t1_gate.csv", index=False)
    print("wrote results/t1_gate.csv and media/fig_t1_gate.png")


if __name__ == "__main__":
    main()
