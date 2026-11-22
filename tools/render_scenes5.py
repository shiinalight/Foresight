"""GIF of the five real pedestrian scenes (ETH, Hotel, Zara1, Zara2, Univ) with TabPFN's 2 s ranges.

    python tools/render_scenes5.py --predict     # local TabPFN, writes results/scenes5_gif_rows.csv
    python tools/render_scenes5.py               # draws media/demo/scenes5.gif from that file

An illustration for the video and the deck, not a new result. The setup is the five-scene test's
(tools/loso5.py, docs/LOSO5.md): for each scene TabPFN reads 1,500 rows drawn from the OTHER four scenes
(context seed 0) and predicts the held-out scene. Differences: TabPFN runs locally here (the test used the
hosted model), and it predicts every row of one 16 s window instead of a 1,500-row sample. The window is
fixed by a rule that does not look at the outcome: it starts at the scene's median (episode, time) row.

Each frame shows where people are now, and the 5-95% box TabPFN drew for them 2 s earlier:
green dot = inside their box, red = outside, grey = no forecast 2 s ago (just arrived).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

SCENES = ["eth", "hotel", "zara1", "zara2", "univ"]
TITLE = {"eth": "ETH", "hotel": "Hotel", "zara1": "Zara 1", "zara2": "Zara 2", "univ": "Univ"}
QS = (0.05, 0.5, 0.95)
H, N_CTX, WIN = 2.0, 1500, 16.0
ROWS = ROOT / "results/scenes5_gif_rows.csv"
OUT = ROOT / "media/demo/scenes5.gif"


def window(d):
    """Episode and start time of the window: the median (episode, t) row, moved back if the episode ends early."""
    et = d[["episode", "t"]].drop_duplicates().sort_values(["episode", "t"]).reset_index(drop=True)
    ep, t0 = et.iloc[len(et) // 2]
    t_end = float(d[d.episode == ep].t.max())
    t_first = float(d[d.episode == ep].t.min())
    return int(ep), max(t_first, min(float(t0), t_end - WIN))


def predict():
    import loso5 as L
    from foresight.core import features as F
    from foresight.core import models as M
    data = {s: L.load(s) for s in SCENES}
    if ROWS.exists():
        ROWS.unlink()
    for test in SCENES:
        d = data[test]
        ep, t0 = window(d)
        q = d[(d.episode == ep) & (d.t >= t0 - 1e-6) & (d.t <= t0 + WIN + 1e-6)].reset_index(drop=True)
        train = [s for s in SCENES if s != test]
        ctx = pd.concat([data[s].sample(N_CTX // len(train), random_state=0) for s in train], ignore_index=True)
        t_start = time.time()
        m = M.make("tabpfn").fit(F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy(float))
        Q = np.asarray(m.predict_quantiles(F.X_of(q), QS))          # (rows, 2 axes, 3 quantiles)
        out = q[["episode", "agent_id", "t", "px", "py", "vx", "vy", "theta", "ddx", "ddy", "true_x", "true_y"]].copy()
        out.insert(0, "scene", test)
        for ai, ax in enumerate("ac"):                               # a = along heading, c = across
            for qi, qq in enumerate(("05", "50", "95")):
                out[f"{ax}{qq}"] = Q[:, ai, qi]
        out.to_csv(ROWS, mode="a", header=not ROWS.exists(), index=False)
        inside = ((out.ddx >= out.a05) & (out.ddx <= out.a95) & (out.ddy >= out.c05) & (out.ddy <= out.c95)).mean()
        print(f"{test}: episode {ep}, t {t0:.1f}-{t0 + WIN:.1f} s, {len(q)} rows, {q.agent_id.nunique()} people, "
              f"{inside:.0%} inside, {time.time() - t_start:.0f} s on {m.device}", flush=True)


def corners(r):
    cx, cy = r.px + r.vx * H, r.py + r.vy * H
    cs, sn = np.cos(r.theta), np.sin(r.theta)
    return [(cx + cs * a - sn * b, cy + sn * a + cs * b)
            for a, b in ((r.a05, r.c05), (r.a95, r.c05), (r.a95, r.c95), (r.a05, r.c95))]


def render(fps=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon
    R = pd.read_csv(ROWS)
    R["inside"] = (R.ddx >= R.a05) & (R.ddx <= R.a95) & (R.ddy >= R.c05) & (R.ddy <= R.c95)
    R["due"] = (R.t + H).round(1)
    scenes = [s for s in SCENES if s in set(R.scene)]
    tracks, t0s, lims = {}, {}, {}
    for s in scenes:
        g = R[R.scene == s]
        tr = pd.read_csv(ROOT / f"data/{s}/tracks.csv")
        tr = tr[tr.episode == int(g.episode.iloc[0])].copy()
        tr["t"] = tr.t.round(1)
        t0s[s] = round(float(g.t.min()) + H, 1)                       # first frame with a box that has come due
        tracks[s] = tr[(tr.t >= t0s[s] - 2.0) & (tr.t <= round(float(g.t.max()) + H, 1))]
        v = tracks[s][tracks[s].t >= t0s[s]]
        cx, cy = (v.x.min() + v.x.max()) / 2, (v.y.min() + v.y.max()) / 2
        half = max(v.x.max() - v.x.min(), v.y.max() - v.y.min()) / 2 + 1.0
        lims[s] = (cx - half, cx + half, cy - half, cy + half)
    dt = 0.4
    n_frames = int(round((WIN - 0.0) / dt)) + 1
    fig, axes = plt.subplots(1, len(scenes), figsize=(3.3 * len(scenes), 4.75), facecolor="white")
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.74, bottom=0.10, wspace=0.04)
    fig.suptitle("Five real pedestrian scenes. For each, TabPFN reads 1,500 example rows from the other four, "
                 "and has never seen this one.", fontsize=12.5, fontweight="bold", x=0.01, ha="left", y=0.98)
    fig.text(0.01, 0.885, "Boxes: where TabPFN said each person would be (5–95% range), drawn 2 s earlier.  "
             "Dots: where they actually are now.", fontsize=10.5, color="#444")
    handles = [plt.Line2D([], [], marker="o", ls="", color="#2b8a3e", label="inside their box"),
               plt.Line2D([], [], marker="o", ls="", color="#c92a2a", label="outside"),
               plt.Line2D([], [], marker="o", ls="", color="#adb5bd", label="not scored yet (first 4 s in view)"),
               plt.Rectangle((0, 0), 1, 1, fc="#fde3d3", ec="#e8590c", label="5–95% box, drawn 2 s earlier")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    count = {s: [0, 0] for s in scenes}
    counted = set()                                              # frame indices already added to the counters

    def draw(k):
        if k == 0:
            counted.clear()
            for s in scenes:
                count[s] = [0, 0]
        first = k not in counted                                 # the last frame is held; count it once
        counted.add(k)
        for ax, s in zip(axes, scenes):
            ax.clear()
            t = round(t0s[s] + k * dt, 1)
            tr, g = tracks[s], R[R.scene == s]
            due = g[np.isclose(g.due, t)].set_index("agent_id")
            for _, r in due.iterrows():
                ax.add_patch(Polygon(corners(r), closed=True, fc="#fde3d3", ec="#e8590c", lw=0.9, alpha=0.85, zorder=2))
            now = tr[np.isclose(tr.t, t)]
            trail = tr[(tr.t <= t + 1e-6) & (tr.t >= t - 2.0 - 1e-6)]
            for aid, p in trail.groupby("agent_id"):
                if aid in set(now.agent_id):
                    ax.plot(p.x, p.y, color="#868e96", lw=0.8, alpha=0.7, zorder=3)
            for _, p in now.iterrows():
                if p.agent_id in due.index:
                    ok = bool(due.loc[p.agent_id, "inside"])
                    count[s][0] += ok and first
                    count[s][1] += first
                    c = "#2b8a3e" if ok else "#c92a2a"
                else:
                    c = "#adb5bd"
                ax.scatter(p.x, p.y, s=26, color=c, zorder=5, edgecolors="white", linewidths=0.4)
            x0, x1, y0, y1 = lims[s]
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#dee2e6")
            n_in, n_all = count[s]
            pct = f"{n_in / n_all:.0%}" if n_all else "–"
            ax.set_title(f"{TITLE[s]}\ninside so far: {n_in} of {n_all} ({pct})", fontsize=11, loc="left")
            ax.plot([x0 + 0.6, x0 + 2.6], [y0 + 0.6, y0 + 0.6], color="#495057", lw=2, zorder=6)
            ax.text(x0 + 1.6, y0 + 0.9, "2 m", ha="center", fontsize=8, color="#495057", zorder=6)

    hold = [n_frames - 1] * (2 * fps)
    anim = FuncAnimation(fig, draw, frames=list(range(n_frames)) + hold, interval=1000 / fps)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=fps), dpi=80)
    stats = {"what": "illustration of docs/LOSO5.md's setup; not a result", "model": "TabPFN-3.5 default, local",
             "context": "1,500 rows from the other four scenes, context seed 0, as tools/loso5.py",
             "window_rule": "16 s from the scene's median (episode, t) row; fixed before looking at outcomes",
             "box": "5-95% on each axis (joint target about 81%)", "fps": fps, "speed": "2x real time",
             "grey": "a person gets a first forecast after 2 s of track, and it comes due 2 s later", "sim_step_s": dt,
             "scenes": {s: {"episode": int(R[R.scene == s].episode.iloc[0]), "t_from": float(R[R.scene == s].t.min()),
                            "t_to": float(R[R.scene == s].t.max()), "rows": int((R.scene == s).sum()),
                            "people": int(R[R.scene == s].agent_id.nunique()),
                            "inside_all_rows": round(float(R[R.scene == s].inside.mean()), 3),
                            "inside_drawn": round(count[s][0] / max(count[s][1], 1), 3)} for s in scenes}}
    OUT.with_name("scenes5_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)", json.dumps(stats["scenes"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true")
    a = ap.parse_args()
    predict() if a.predict else render()
