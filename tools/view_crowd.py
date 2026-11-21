"""Watch the crowd simulator: one episode as an animated GIF, in real time (5 frames/s at dt 0.2 s).

    python tools/view_crowd.py                          # episode 4 of seed 0: calm -> erratic
    python tools/view_crowd.py --episode 2 --out results/crowd_ep2.gif

People are coloured by regime, with a 1 s trail. A hollow circle is someone standing still;
a black ring marks a collision. The robot is red, its straight-line route dashed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight.domains.crowd import CrowdConfig, generate  # noqa: E402

COLOURS = {"calm": "tab:green", "hurried": "tab:blue", "erratic": "tab:purple"}
TRAIL_S = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5, help="how many to generate (same as the summary run)")
    ap.add_argument("--episode", type=int, default=4)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--out", default="results/crowd.gif")
    a = ap.parse_args()

    tracks, obstacles, meta = generate(a.episodes, a.seed, CrowdConfig(duration=a.duration))
    tr = tracks[tracks["episode"] == a.episode]
    ob = obstacles[obstacles["episode"] == a.episode]
    m = meta["episodes"][str(a.episode)]
    dt, (x0, y0, x1, y1) = meta["dt"], meta["bounds"]
    times = np.sort(tr["t"].unique())
    robot = tr[tr["is_ego"] == 1].set_index("t")
    trail = round(TRAIL_S / dt)

    fig, ax = plt.subplots(figsize=(6.4, 6.8))
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=r) for r, c in COLOURS.items()]
    handles += [plt.Line2D([], [], marker="o", ls="", color="tab:red", label="robot"),
                plt.Line2D([], [], marker="o", ls="", mfc="none", color="0.3", label="stopped")]

    def draw(k):
        t = times[k]
        ax.clear()
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        regime = tr.loc[(tr["t"] == t) & (tr["kind"] == "person"), "regime"].iloc[0]
        switch = (f"   (switch {m['regime_before']} -> {m['regime_after']} at {m['regime_switch_t']:.0f} s)"
                  if m["regime_after"] else "")
        ax.set_title(f"episode {a.episode}   t = {t:4.1f} s   regime: {regime}{switch}", fontsize=10)
        for _, o in ob.iterrows():
            ax.add_patch(Circle((o.x, o.y), o.r, color="0.75"))
        ax.plot(*robot[["x", "y"]].to_numpy().T, color="tab:red", lw=0.8, ls="--", alpha=0.4)
        recent = tr[(tr["t"] <= t + 1e-9) & (tr["t"] >= times[max(0, k - trail)] - 1e-9)]
        for aid, g in recent.groupby("agent_id"):
            ax.plot(g["x"], g["y"], color="0.5", lw=0.8, alpha=0.6)
        for _, r in tr[tr["t"] == t].iterrows():
            c = "tab:red" if r.is_ego else COLOURS[r.regime]
            stopped = not r.is_ego and np.hypot(r.vx, r.vy) < 0.2
            ax.add_patch(Circle((r.x, r.y), r.radius, color=c, fill=not stopped, lw=1.5))
            if r.collided:
                ax.add_patch(Circle((r.x, r.y), r.radius + 0.15, fill=False, color="k", lw=1.2))
        ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.8)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=round(1 / dt))
    with writer.saving(fig, a.out, dpi=80):
        for k in range(len(times)):
            draw(k); writer.grab_frame()
    print(f"wrote {a.out} ({len(times)} frames, {times[-1]:.0f} s)")


if __name__ == "__main__":
    main()
