"""Side-by-side video of two planners on the same crowd (logs from run_closed.py --log).

    python render_closed.py --sim crowd --seed 1001 --left cv --right tabpfn --out media/closed_1001.gif
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.transforms import Affine2D

from foresight import viz as V

NAME = {"cv": "Constant velocity", "cv_band": "Const. velocity + band", "lgbm": "LightGBM", "tabpfn": "TabPFN"}


def load(sim, kind, seed):
    with open(f"results/closed/{sim}_{kind}_{seed}.pkl", "rb") as f:
        return pickle.load(f)


def draw(ax, log, i, title):
    fr = log["frames"][min(i, len(log["frames"]) - 1)]
    t = fr["t"]
    V.style_axes(ax, log["bounds"]); ax.set_xticks([]); ax.set_yticks([])
    for o in log["obst"]:
        ax.add_patch(Circle((o["x"], o["y"]), o["r"], color=V.C_OBST, zorder=1))
    # latest plan at or before t
    plan = None
    for p in log["plans"]:
        if p["t"] <= t + 1e-9:
            plan = p
    if plan is not None and len(plan["pred"]):
        for _, r in plan["pred"][np.isclose(plan["pred"].h, 2.0)].iterrows():
            cvx, cvy = r.px + r.vx * r.h, r.py + r.vy * r.h
            box = Rectangle((r.lo_a, r.lo_c), max(r.hi_a - r.lo_a, 0.05), max(r.hi_c - r.lo_c, 0.05),
                            fill=False, color=V.C_PRED, lw=1.0, alpha=0.8, zorder=3)
            box.set_transform(Affine2D().rotate(r.theta).translate(cvx, cvy) + ax.transData)
            ax.add_patch(box)
    # robot trail
    xs = [f["state"][0]["x"] for f in log["frames"][: i + 1]]; ys = [f["state"][0]["y"] for f in log["frames"][: i + 1]]
    ax.plot(xs, ys, color=V.C_ROBOT, lw=1.5, zorder=4)
    rob = fr["state"][0]
    ax.plot(rob["goal_x"], rob["goal_y"], marker="*", color=V.C_ROBOT, ms=14, zorder=4)
    for r in fr["state"][1:]:
        ax.add_patch(Circle((r["x"], r["y"]), r["radius"], color=V.C_PERSON, zorder=5))
    hit = fr["info"]["collided"]
    ax.add_patch(Circle((rob["x"], rob["y"]), rob["radius"], color=V.C_ROBOT, zorder=6,
                        ec="black" if hit else "none", lw=2.5))
    n_hits = sum(f["info"]["collided"] for f in log["frames"][: i + 1])
    done = i >= len(log["frames"]) - 1
    status = f"t = {t:4.1f} s · collisions {n_hits}" + ("  · goal reached" if done else "")
    ax.set_title(f"{title}\n{status}", fontsize=11, color=V.C_INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="crowd")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--left", default="cv")
    ap.add_argument("--right", default="tabpfn")
    ap.add_argument("--every", type=int, default=1, help="draw every n-th sim step")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    L, R = load(a.sim, a.left, a.seed), load(a.sim, a.right, a.seed)
    n = max(len(L["frames"]), len(R["frames"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), facecolor="white")
    legend = [Line2D([], [], marker="o", ls="", color=V.C_ROBOT, ms=10, label="Robot (black ring = collision)"),
              Line2D([], [], marker="*", ls="", color=V.C_ROBOT, ms=12, label="Robot goal"),
              Line2D([], [], marker="o", ls="", color=V.C_PERSON, ms=8, label="Person"),
              Patch(facecolor="none", edgecolor=V.C_PRED, label="Predicted 80% range, +2 s")]
    out = Path(a.out or f"media/closed_{a.sim}_{a.seed}_{a.left}_vs_{a.right}.gif"); out.parent.mkdir(parents=True, exist_ok=True)
    w = PillowWriter(fps=int(round(1 / (0.2 * a.every))) if a.sim == "crowd" else 5)
    with w.saving(fig, str(out), dpi=90):
        for i in range(0, n, a.every):
            draw(axes[0], L, i, NAME[a.left]); draw(axes[1], R, i, NAME[a.right])
            fig.legends.clear(); fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=9)
            fig.suptitle("Robot crossing a crowd: same people, two planners", fontsize=13, weight="bold",
                         color=V.C_INK, x=0.02, ha="left")
            w.grab_frame()
    draw(axes[0], L, n // 2, NAME[a.left]); draw(axes[1], R, n // 2, NAME[a.right])
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"wrote {out} ({n} steps)")


if __name__ == "__main__":
    main()
