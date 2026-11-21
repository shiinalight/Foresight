"""Watch the street scene: one episode top-down, real time (5 frames/s at dt 0.2 s).

    python tools/view_street.py                       # episode 0 of seed 0 -> results/street_ep0.mp4 and .gif
    python tools/view_street.py --episode 2 --data data/street

Cars are boxes, cyclists small bars, pedestrians dots (orange from rush hour on), the robot red with
its path. Scripted events are announced in a banner for 3 s and the agent involved is ringed.
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
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight.domains import street as S  # noqa: E402

C = dict(road="#5b5f66", bike="#9c6b5b", walk="#e9e6df", building="#c9c3b6", lane="#f2f2f2",
         car="#2f4b7c", cyclist="#1b9e5a", ped="#3b75af", rush="#eb6834", robot="#d62728", event="#111111")
BANNER_S = 3.0
LABEL = dict(sudden_stop="Pedestrian stops dead", car_brake_crosswalk="Car brakes hard for the zebra",
             steps_onto_crosswalk="", jaywalk="Jaywalker crosses mid-block", rush_hour="Rush hour: hurried crowd, more cars")


def draw_map(ax):
    W, H = S.HALF_W, S.HALF_H
    ax.add_patch(Rectangle((-W, -H), 2 * W, 2 * H, color=C["walk"], zorder=0))
    for b in S.MAP["buildings"]:
        (x0, x1), (y0, y1) = b["x"], b["y"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=C["building"], zorder=0))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, b["name"], ha="center", va="center", fontsize=13, weight="bold", color="0.3")
        ax.plot(*b["door"], marker="s", ms=7, color="0.3", zorder=1)
    for side in (-1, 1):                                  # other building fronts
        ax.add_patch(Rectangle((-W, side * S.WALL if side > 0 else -H), 2 * W, H - S.WALL, color=C["building"], alpha=0.45, zorder=0))
    ax.add_patch(Rectangle((-W, -S.CURB), 2 * W, 2 * S.CURB, color=C["bike"], zorder=0))
    ax.add_patch(Rectangle((-W, -S.ROAD), 2 * W, 2 * S.ROAD, color=C["road"], zorder=0))
    ax.plot([-W, W], [0, 0], color=C["lane"], lw=1.2, ls=(0, (6, 6)), zorder=0)
    for xs in np.arange(-S.CROSS_X + 0.25, S.CROSS_X, 0.5):    # zebra stripes
        ax.add_patch(Rectangle((xs - 0.12, -S.CURB), 0.24, 2 * S.CURB, color="white", alpha=0.85, zorder=0))
    for x, y, r in S.OBSTACLES:
        ax.add_patch(Circle((x, y), r, color="0.45", zorder=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="a dataset folder; default: generate with --seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", nargs="+", default=None)
    a = ap.parse_args()
    if a.data:
        tracks = pd.read_csv(Path(a.data) / "tracks.csv", low_memory=False)
        meta = json.loads((Path(a.data) / "meta.json").read_text())
    else:
        tracks, _, meta = S.generate(a.episodes, a.seed)
    tr = tracks[tracks.episode == a.episode].copy()
    tr["events"] = tr["events"].fillna("")
    em = meta["episodes"][str(a.episode)]
    dt = float(meta["dt"])
    times = np.sort(tr.t.unique())
    steps = np.round(tr.t.to_numpy() / dt).astype(int)
    robot = tr[tr.is_ego == 1].sort_values("t")
    events = [(e["t"], e["name"], e["agent_id"]) for e in em["events"] if LABEL.get(e["name"])]

    fig = plt.figure(figsize=(12.8, 7.2), facecolor="white")
    ax = fig.add_axes([0.02, 0.04, 0.72, 0.84])
    title = fig.text(0.38, 0.95, "", ha="center", fontsize=17, weight="bold")
    banner = fig.text(0.38, 0.905, "", ha="center", fontsize=13, color=C["event"])
    handles = [Patch(color=C["car"], label="car"), Patch(color=C["cyclist"], label="cyclist"),
               Line2D([], [], marker="o", ls="", color=C["ped"], ms=9, label="pedestrian"),
               Line2D([], [], marker="o", ls="", color=C["rush"], ms=9, label="pedestrian (rush hour)"),
               Line2D([], [], marker="o", ls="", color=C["robot"], ms=10, label="delivery robot"),
               Line2D([], [], marker="o", ls="", mfc="none", color=C["event"], ms=14, mew=2, label="agent in a scripted event"),
               Patch(color=C["road"], label="road"), Patch(color=C["bike"], label="bike lane"),
               Patch(facecolor="white", edgecolor="0.5", label="zebra crossing")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.755, 0.86), fontsize=11, frameon=False)
    timeline = fig.text(0.76, 0.33, "Scripted events\n" + "\n".join(f"{t:4.0f} s  {LABEL[n]}" for t, n, _ in events),
                        va="top", fontsize=9.5, color="0.25", linespacing=1.5)

    def draw(k):
        t = times[k]; step = round(t / dt)
        ax.clear(); ax.set_xlim(-S.HALF_W, S.HALF_W); ax.set_ylim(-S.HALF_H, S.HALF_H)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        draw_map(ax)
        now = tr[steps == step]
        recent = tr[(steps <= step) & (steps >= step - round(2.0 / dt)) & (tr.agent_type.isin(["pedestrian", "cyclist"]))]
        for _, g in recent.groupby("agent_id"):
            ax.plot(g.x, g.y, color="0.35", lw=0.8, alpha=0.35, zorder=2)
        rm = robot[robot.t <= t + 1e-9]
        ax.plot(rm.x, rm.y, color=C["robot"], lw=1.6, alpha=0.55, zorder=2)
        active = [(n, aid) for te, n, aid in events if te <= t < te + BANNER_S]
        ringed = {aid for _, aid in active}
        ringed |= {aid for te, n, aid in [(e["t"], e["name"], e["agent_id"]) for e in em["events"]] if te <= t < te + BANNER_S}
        for _, r in now.iterrows():
            if r.agent_type == "car":
                ax.add_patch(Rectangle((r.x - r.length / 2, r.y - r.width / 2), r.length, r.width,
                                       color=C["car"], zorder=4))
                ax.plot([r.x + np.sign(r.vx or np.cos(r.heading)) * r.length / 2] * 2, [r.y - 0.6, r.y + 0.6],
                        color="#ffd84d", lw=2, zorder=5)          # headlights show direction
            elif r.agent_type == "cyclist":
                ax.add_patch(Rectangle((r.x - r.length / 2, r.y - r.width / 2), r.length, r.width,
                                       color=C["cyclist"], zorder=4))
            else:
                col = C["robot"] if r.is_ego else (C["rush"] if r.regime == "rush_hour" else C["ped"])
                stopped = (not r.is_ego) and np.hypot(r.vx, r.vy) < 0.2
                ax.add_patch(Circle((r.x, r.y), r.radius, color=col, fill=not stopped, lw=1.8, zorder=5))
            if r.agent_id in ringed:
                ax.add_patch(Circle((r.x, r.y), max(r.radius, r.length / 2) + 0.6, fill=False, color=C["event"], lw=2, zorder=6))
        title.set_text(f'"Last-metre delivery"   ·   t = {t:4.1f} s   ·   '
                       f'{"rush hour" if em["regime_switch_t"] is not None and t >= em["regime_switch_t"] else "normal"}')
        banner.set_text("   ·   ".join(LABEL[n] for n, _ in active))

    outs = a.out or [f"results/street_ep{a.episode}.mp4", f"results/street_ep{a.episode}.gif"]
    for out in outs:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        if out.endswith(".mp4"):
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            writer, dpi = FFMpegWriter(fps=round(1 / dt), codec="libx264", bitrate=6000, extra_args=["-pix_fmt", "yuv420p"]), 100
        else:
            writer, dpi = PillowWriter(fps=round(1 / dt)), 50
        with writer.saving(fig, out, dpi=dpi):
            for k in range(len(times)):
                draw(k); writer.grab_frame()
        print(f"wrote {out} ({len(times)} frames, {times[-1]:.0f} s)")


if __name__ == "__main__":
    main()
