"""GIF of the highway test (docs/HIGHWAY.md, H2): what happens on the road when every driver turns aggressive.

    python tools/render_highway.py            # writes media/demo/highway.gif and media/demo/highway_stats.json

No model is run: the boxes are the stored predictions of the H2 test (results/highway_h2_q.npz, written by
tools/highway_tests.py), the cars are data/highway/tracks.csv. Two panels show the same road and the same cars:
top, TabPFN reading only the frozen example bank (which has no aggressive driving); bottom, the same TabPFN that
also reads this road's most recent rows. A box is the 5-95% range for where a car will be 2 s later; it is drawn
when the forecast is made, the car drives into it, and after 2 s it turns green (car inside) or red (outside).

The episode is picked by a rule, not by eye: of the five held-out episodes, the one with the median
coverage gap between the two models.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

H = 2.0
OUT = ROOT / "media/demo/highway.gif"
PANELS = [("frozen", "TabPFN, frozen example bank only (it has never seen aggressive driving)"),
          ("split", "Same TabPFN, also reading this road's last seconds (no training)")]


def inside(Q, Y):
    return (Y[:, 0] >= Q[:, 0, 0]) & (Y[:, 0] <= Q[:, 0, 4]) & (Y[:, 1] >= Q[:, 1, 0]) & (Y[:, 1] <= Q[:, 1, 4])


def main(fps=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon

    z = np.load(ROOT / "results/highway_h2_q.npz")
    meta = json.loads((ROOT / "data/highway/meta.json").read_text())
    ok = {k: inside(z[k], z["Y"]) for k, _ in PANELS}
    eps = sorted(set(z["episode"].tolist()))
    gap = {e: float(ok["split"][z["episode"] == e].mean() - ok["frozen"][z["episode"] == e].mean()) for e in eps}
    ep = sorted(eps, key=lambda e: gap[e])[len(eps) // 2]                  # the median-gap episode
    em = meta["episodes"][str(ep)]
    sw = float(em["regime_switch_t"])
    m = z["episode"] == ep
    idx = np.flatnonzero(m)
    t_pred = np.round(z["t"][idx], 1)

    tr = pd.read_csv(ROOT / "data/highway/tracks.csv")
    tr = tr[tr.episode == ep].copy()
    tr["t"] = tr.t.round(1)
    t_from, t_to = round(sw - 4.0, 1), round(min(float(tr.t.max()), float(t_pred.max()) + H + 1.0), 1)
    times = np.round(np.arange(t_from, t_to + 1e-6, 0.2), 1)
    lanes = sorted(tr.ext_lane.unique())

    polys = {}
    for k, _ in PANELS:
        for i in idx:
            th = z["theta"][i]
            cx, cy = z["px"][i] + z["vx"][i] * H, z["py"][i] + z["vy"][i] * H
            cs, sn = np.cos(th), np.sin(th)
            lo_a, hi_a, lo_c, hi_c = z[k][i, 0, 0], z[k][i, 0, 4], z[k][i, 1, 0], z[k][i, 1, 4]
            polys[k, i] = [(cx + cs * a - sn * b, cy + sn * a + cs * b)
                           for a, b in ((lo_a, lo_c), (hi_a, lo_c), (hi_a, hi_c), (lo_a, hi_c))]

    fig = plt.figure(figsize=(13, 7.2), facecolor="white")
    axes = [fig.add_axes([0.02, 0.50, 0.96, 0.30]), fig.add_axes([0.02, 0.09, 0.96, 0.30])]
    head = fig.text(0.02, 0.955, "", fontsize=13.5, fontweight="bold")
    sub = fig.text(0.02, 0.915, "Box = where the model says a car will be 2 s later (5–95% range). After 2 s it turns green "
                   "(car inside) or red (outside). Counts: all 20 cars.", fontsize=10.5, color="#444")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#2a6fdb", alpha=0.3, label="forecast, 2 s ahead"),
               plt.Rectangle((0, 0), 1, 1, fc="#2b8a3e", alpha=0.6, label="car ended inside"),
               plt.Rectangle((0, 0), 1, 1, fc="#c92a2a", alpha=0.6, label="car ended outside"),
               plt.Rectangle((0, 0), 1, 1, fc="#1f2328", label="car"),
               plt.Rectangle((0, 0), 1, 1, fc="#e8590c", label="the robot's own car")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10)
    final = {k: (int(ok[k][m].sum()), int(m.sum())) for k, _ in PANELS}

    def draw(fi):
        t = float(times[min(fi, len(times) - 1)])
        ego = tr[(tr.is_ego == 1) & np.isclose(tr.t, t)].iloc[0]
        now = tr[np.isclose(tr.t, t)]
        if t < sw:
            head.set_text(f"Simulated highway, t = {t:4.1f} s.  Drivers are {em['regime_before']}.")
            head.set_color("#1f2328")
        else:
            head.set_text(f"Simulated highway, t = {t:4.1f} s.  At {sw:.0f} s every driver turned AGGRESSIVE: "
                          "closer gaps, sudden lane changes.")
            head.set_color("#c92a2a")
        for ax, (k, label) in zip(axes, PANELS):
            ax.clear()
            for ln in range(len(lanes) + 1):
                ax.axhline(-2 + 4 * ln, color="#bbb", lw=1, ls="-" if ln in (0, len(lanes)) else "--", zorder=0)
            done = t_pred + H <= t + 1e-6
            n_in, n_all = int(ok[k][idx][done].sum()), int(done.sum())
            for j, i in enumerate(idx):
                age = t - t_pred[j]
                if age < -1e-6 or age > H + 1.2:
                    continue
                if age < H - 1e-6:
                    fc, al = "#2a6fdb", 0.28
                else:
                    fc, al = ("#2b8a3e", 0.55) if ok[k][i] else ("#c92a2a", 0.55)
                ax.add_patch(Polygon(polys[k, i], closed=True, fc=fc, ec=fc, alpha=al, lw=1.8, zorder=3))
                bx, by = np.mean(polys[k, i], axis=0)
                if age < H - 1e-6:                                    # dotted link: which car this box belongs to
                    car = now[now.agent_id == z["agent_id"][i]]
                    if len(car):
                        ax.plot([float(car.x.iloc[0]), bx], [float(car.y.iloc[0]), by], color="#2a6fdb", lw=0.9,
                                ls=":", zorder=3)
                else:                                                 # where the car really was after 2 s
                    ax.scatter(z["true_x"][i], z["true_y"][i], marker="x", s=38, color=fc, zorder=5)
            for _, r in now.iterrows():
                past = tr[(tr.agent_id == r.agent_id) & (tr.t <= t + 1e-6) & (tr.t >= t - 1.5 - 1e-6)]
                ax.plot(past.x, past.y, color="#8b949e", lw=1.2, zorder=2)
                ax.add_patch(plt.Rectangle((r.x - 2.5, r.y - 1.0), 5.0, 2.0, color="#e8590c" if r.is_ego else "#1f2328",
                                           zorder=4))
            ax.set_xlim(float(ego.x) - 40, float(ego.x) + 140)
            ax.set_ylim(-3, 4 * len(lanes) - 1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            pct = f"{n_in} of {n_all} forecasts held ({n_in / n_all:.0%})" if n_all else "no forecast has come due yet"
            ax.set_title(label, fontsize=11.5, loc="left", color="#1f2328")
            ax.text(1.0, 1.035, pct, transform=ax.transAxes, ha="right", fontsize=12.5, fontweight="bold",
                    color="#1f2328")
            ax.annotate("driving direction →", xy=(0.995, -0.09), xycoords="axes fraction", ha="right", fontsize=8.5,
                        color="#666")

    frames = list(range(len(times))) + [len(times) - 1] * (2 * fps)
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=fps), dpi=80)
    stats = {"what": "replay of the H2 test's stored predictions (results/highway_h2_q.npz); no model run here",
             "episode": int(ep), "episode_rule": "median coverage gap (split minus frozen) of the five held-out episodes",
             "gap_by_episode": {str(e): round(gap[e], 3) for e in eps},
             "switch_t": sw, "regime": f"{em['regime_before']} -> {em['regime_after']}",
             "forecast_times_s": sorted(set(t_pred.tolist())), "box": "5-95% on each axis, +2 s",
             "this_episode_inside": {k: f"{a} of {b}" for k, (a, b) in final.items()},
             "all_five_episodes_inside": {k: round(float(ok[k].mean()), 3) for k, _ in PANELS},
             "fps": fps, "sim_step_s": 0.2, "y_axis": "stretched about 1.7x",
             "view": "180 m around the robot's car; the counters cover all 20 cars, also those outside the picture"}
    OUT.with_name("highway_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)", json.dumps(stats))


if __name__ == "__main__":
    main()
