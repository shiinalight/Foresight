"""Watch the doorway scene: one GIF per regime from one episode that switches.

    python tools/view_doorway.py                                  # seed 1001 of data/doorway_test
    python tools/view_doorway.py --data data/doorway --episode 3 --seconds 20

Writes results/doorway_<episode>_trickle.gif (the seconds before the switch) and
results/doorway_<episode>_rush.gif (the seconds after it). People are blue in trickle, orange in
rush; a hollow ring means standing still; the robot is red with its path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight.domains import doorway as D  # noqa: E402

C = dict(floor="#f3efe7", wall="#4a4a4a", trickle="#2a78d6", rush="#eb6834", robot="#d62728", text="#222222")
TITLE = dict(trickle="Trickle: a few people wander between the rooms",
             rush="Rush: a meeting ends, 24 people head for one 1.5 m door")


def draw_room(ax):
    L = D.L
    ax.add_patch(Rectangle((-L, -L), 2 * L, 2 * L, color=C["floor"], zorder=0))
    for y0, y1 in D.MAP["wall"]["y"]:
        ax.add_patch(Rectangle((-D.WALL_R, y0), 2 * D.WALL_R, y1 - y0, color=C["wall"], zorder=1))
    ax.add_patch(Rectangle((L - 0.12, -4), 0.12, 8, color=C["rush"], alpha=0.5, zorder=1))
    ax.add_patch(Rectangle((-L, -5), 0.12, 10, color="0.55", alpha=0.6, zorder=1))
    kw = dict(ha="center", fontsize=12, color="0.45", weight="bold", zorder=1)
    ax.text(-L / 2, L - 1.0, "Room A", **kw); ax.text(L / 2, L - 1.0, "Room B", **kw)
    ax.text(0.0, -L + 0.4, "door ↑", ha="center", fontsize=9, color="0.45")
    ax.text(L - 0.35, 4.4, "from the\nmeeting room", ha="right", fontsize=8.5, color=C["rush"])
    ax.text(-L + 0.35, 5.4, "exits", ha="left", fontsize=8.5, color="0.45")


def render(tr, robot, t0, t1, regime, out, dt):
    fig = plt.figure(figsize=(9.6, 6.4), facecolor="white")
    ax = fig.add_axes([0.01, 0.02, 0.62, 0.86])
    title = fig.text(0.02, 0.955, TITLE[regime], fontsize=14, weight="bold", color=C["text"])
    sub = fig.text(0.02, 0.915, "", fontsize=11, color="0.3")
    handles = [Line2D([], [], marker="o", ls="", color=C[regime], ms=10, label=f"person ({regime})"),
               Line2D([], [], marker="o", ls="", mfc="none", mec=C[regime], mew=2, ms=10, label="standing still"),
               Line2D([], [], marker="o", ls="", color=C["robot"], ms=12, label="delivery robot"),
               Line2D([], [], color=C["robot"], lw=1.6, alpha=0.6, label="robot's path"),
               Line2D([], [], color="0.4", lw=1, alpha=0.5, label="last 2 s of each person")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.645, 0.86), fontsize=10.5, frameon=False)
    stats = fig.text(0.655, 0.47, "", fontsize=11, va="top", color=C["text"], linespacing=1.7)
    fig.text(0.655, 0.13, "One way at a time: people wait beside\nthe door while others come through,\n"
             "and walk no faster than the person\nin front of them.", fontsize=9, color="0.4", va="bottom")
    steps = np.round(tr.t.to_numpy() / dt).astype(int)
    frames = list(range(round(t0 / dt), round(t1 / dt) + 1))

    def draw(k):
        ax.clear(); ax.set_xlim(-D.L, D.L); ax.set_ylim(-D.L, D.L); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        draw_room(ax)
        now = tr[steps == k]
        recent = tr[(steps <= k) & (steps >= k - round(2.0 / dt))]
        for _, g in recent.groupby("agent_id"):
            ax.plot(g.x, g.y, color="0.4", lw=0.8, alpha=0.35, zorder=2)
        rp = robot[robot.t <= k * dt + 1e-9]
        ax.plot(rp.x, rp.y, color=C["robot"], lw=1.6, alpha=0.55, zorder=2)
        speed = np.hypot(now.vx, now.vy)
        for (_, r), sp in zip(now.iterrows(), speed):
            col = C[r.regime] if r.regime in C else C["trickle"]
            still = sp < 0.2
            ax.add_patch(Circle((r.x, r.y), r.radius, facecolor="none" if still else col, edgecolor=col,
                                lw=2.0, zorder=4))
        rr = robot[np.isclose(robot.t, k * dt)]
        if len(rr):
            ax.add_patch(Circle((rr.x.iloc[0], rr.y.iloc[0]), rr.radius.iloc[0], color=C["robot"], zorder=5))
        near = (np.abs(now.x) < 3) & (np.abs(now.y) < 3)
        sub.set_text(f"held-out episode, t = {k * dt:4.1f} s")
        stats.set_text(f"people in view: {len(now)}\nwithin 3 m of the door: {near.sum()}\n"
                       f"  of them slowed below 0.5 m/s: {(near & (speed < 0.5)).sum()}\nstanding still: {(speed < 0.2).sum()}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    w = PillowWriter(fps=round(1 / dt))
    with w.saving(fig, out, dpi=80):
        for k in frames:
            draw(k); w.grab_frame()
    plt.close(fig)
    print(f"wrote {out} ({len(frames)} frames, {t0:.1f}-{t1:.1f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/doorway_test")
    ap.add_argument("--episode", type=int, default=1001, help="an episode with a switch (odd seed)")
    ap.add_argument("--seconds", type=float, default=15.0)
    a = ap.parse_args()
    tracks = pd.read_csv(Path(a.data) / "tracks.csv", low_memory=False)
    meta = json.loads((Path(a.data) / "meta.json").read_text())
    em = meta["episodes"][str(a.episode)]
    sw, dt = em["regime_switch_t"], float(meta["dt"])
    if sw is None:
        raise SystemExit(f"episode {a.episode} has no switch; pick an odd seed")
    tr = tracks[tracks.episode == a.episode]
    ppl, robot = tr[tr.kind == "person"], tr[tr.is_ego == 1].sort_values("t")
    render(ppl, robot, max(0.0, sw - a.seconds), sw - dt, "trickle", f"results/doorway_{a.episode}_trickle.gif", dt)
    render(ppl, robot, sw, min(em["duration"], sw + a.seconds), "rush", f"results/doorway_{a.episode}_rush.gif", dt)


if __name__ == "__main__":
    main()
