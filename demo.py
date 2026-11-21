"""Demo animation: one test episode with each person's predicted 2 s range drawn.

    python demo.py --data data/toy --model tabpfn --out results/demo_toy_tabpfn.gif
    python demo.py --data data/hotel --train-data data/eth --model tabpfn --out results/demo_hotel_tabpfn.gif

The model only sees context rows from other episodes (or another dataset with --train-data).
Each frame shows, for every person: their path over the last 2 s, the predicted position
in 2 s (median) and the 10-90% range around it, and - for checking only - where they
actually were 2 s later. A running score says how often the range held the truth.
Writes a GIF (or MP4 if the name ends in .mp4 and ffmpeg is installed) plus a PNG still.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.transforms import Affine2D

from foresight.core import features as F
from foresight.core import models as M
from foresight.core import schema
from run_offline import rows_for

# validated categorical palette (dataviz validator, light mode): identity + shape, never colour alone
C_PERSON, C_PRED, C_ROBOT = "#2a6fdb", "#e8590c", "#7a3fb5"
C_INK, C_MUTED, C_OBST, C_GRID = "#1f2328", "#8b949e", "#c9ced6", "#eceef1"
MODEL_LABEL = {"tabpfn": "TabPFN", "lgbm": "LightGBM (quantile)", "cv_band": "Constant velocity + fixed band",
               "cv": "Constant velocity"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--train-data", default=None)
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--model", default="tabpfn")
    ap.add_argument("--context", type=int, default=1500)
    ap.add_argument("--h", type=float, default=2.0)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--out", default="results/demo.gif")
    a = ap.parse_args()

    tracks, obst, meta = schema.load(a.data)
    hz = (a.h,)
    df = rows_for(a.data, a.noise, horizons=hz)
    if a.train_data:
        pool, test = rows_for(a.train_data, a.noise, horizons=hz), df
        src = f"context: {a.context:,} rows from {Path(a.train_data).name} (a different place)"
    else:
        tr, _, te = F.split_by_episode(df)
        pool, test = df[tr], df[te]
        src = f"context: {a.context:,} rows from other episodes"
    ep = a.episode if a.episode is not None else int(test.groupby("episode").size().idxmax())
    ctx = pool.sample(min(a.context, len(pool)), random_state=0)
    q = test[test.episode == ep].copy()
    times = sorted(q.t.unique())[: a.max_frames]
    q = q[q.t.isin(times)]
    m = M.make(a.model).fit(F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy())
    QS = (0.05, 0.5, 0.95)   # per-axis 5-95% -> box holding the truth ~80% of the time (0.9 x 0.9)
    Q = m.predict_quantiles(F.X_of(q), QS)
    q["mx"], q["my"] = F.to_world(q, Q[:, 0, 1], Q[:, 1, 1])
    q["lo_a"], q["hi_a"], q["lo_c"], q["hi_c"] = Q[:, 0, 0], Q[:, 0, 2], Q[:, 1, 0], Q[:, 1, 2]
    Y = q[["ddx", "ddy"]].to_numpy()
    q["inside"] = ((Y >= Q[:, :, 0]) & (Y <= Q[:, :, 2])).all(axis=1)
    q["err"] = np.hypot(q.mx - q.true_x, q.my - q.true_y)

    tr_ep = tracks[tracks.episode == ep]
    has_robot = bool((tr_ep.is_ego == 1).any())
    obst_ep = obst[obst.episode == ep] if len(obst) else obst
    b = meta["bounds"]
    dt = float(meta["dt"])

    fig = plt.figure(figsize=(9.6, 7.2), facecolor="white")
    ax = fig.add_axes([0.07, 0.13, 0.60, 0.74])
    legend_items = [
        Line2D([], [], marker="o", ls="", color=C_PERSON, ms=8, label="Person, now"),
        Line2D([], [], color=C_MUTED, lw=1.2, label="Their path, last 2 s"),
        Line2D([], [], marker="o", ls="-", color=C_PRED, ms=4, lw=1, label=f"Predicted position in {a.h:g} s"),
        Patch(facecolor="none", edgecolor=C_PRED, lw=1.5, label="Predicted range (~80% sure)"),
        Line2D([], [], marker="x", ls="", color=C_INK, ms=7, mew=1.6, label=f"Actual position {a.h:g} s later\n(shown for checking only)"),
    ]
    if has_robot:
        legend_items.append(Line2D([], [], marker="s", ls="", color=C_ROBOT, ms=9, label="Robot"))
    if len(obst_ep):
        legend_items.append(Patch(facecolor=C_OBST, label="Obstacle"))

    def draw(t):
        ax.clear()
        ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3]); ax.set_aspect("equal")
        ax.set_facecolor("white"); ax.grid(color=C_GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        ax.set_xlabel("x (metres)", color=C_MUTED, fontsize=9); ax.set_ylabel("y (metres)", color=C_MUTED, fontsize=9)
        for _, o in obst_ep.iterrows():
            ax.add_patch(Circle((o.x, o.y), o.r, color=C_OBST, zorder=1))
        past = tr_ep[(tr_ep.t <= t + 1e-6) & (tr_ep.t >= t - 2.0 - 1e-6)]
        for aid, g in past.groupby("agent_id"):
            ax.plot(g.x, g.y, color=C_ROBOT if g.is_ego.iloc[0] else C_MUTED, lw=1.0, alpha=0.8, zorder=2)
        now = tr_ep[np.isclose(tr_ep.t, t)]
        for _, r in now.iterrows():
            if r.is_ego:
                ax.plot(r.x, r.y, marker="s", color=C_ROBOT, ms=10, zorder=6)
                ax.annotate("robot", (r.x, r.y), xytext=(7, 7), textcoords="offset points", fontsize=8, color=C_INK)
            else:
                ax.add_patch(Circle((r.x, r.y), r.radius, color=C_PERSON, zorder=5))
        cur = q[np.isclose(q.t, t)]
        for _, r in cur.iterrows():
            # box in the person's frame: constant-velocity point + [lo, hi] along and across travel
            cvx, cvy = r.px + r.vx * r.h, r.py + r.vy * r.h
            box = Rectangle((r.lo_a, r.lo_c), max(r.hi_a - r.lo_a, 0.05), max(r.hi_c - r.lo_c, 0.05),
                            fill=False, color=C_PRED, lw=1.4, zorder=4)
            box.set_transform(Affine2D().rotate(r.theta).translate(cvx, cvy) + ax.transData)
            ax.add_patch(box)
            ax.plot([r.px, r.mx], [r.py, r.my], color=C_PRED, lw=1, zorder=4)
            ax.plot(r.mx, r.my, "o", color=C_PRED, ms=3, zorder=4)
            ax.plot(r.true_x, r.true_y, "x", color=C_INK, ms=6, mew=1.4, zorder=7)
        seen = q[q.t <= t + 1e-6]
        fig.texts.clear()
        fig.text(0.07, 0.95, f"Where will each person be in {a.h:g} seconds?", fontsize=15, weight="bold", color=C_INK)
        fig.text(0.07, 0.91, f"{MODEL_LABEL.get(a.model, a.model)} · {src} · data: {meta.get('sim', '')} · "
                 f"episode {ep}, t = {t:.1f} s", fontsize=9, color=C_MUTED)
        fig.text(0.70, 0.55, "How to read it", fontsize=10, weight="bold", color=C_INK)
        fig.text(0.70, 0.525,
                 "Each person gets a prediction of where\n"
                 f"they will be {a.h:g} s from now. The box is\n"
                 "where the model expects them with\n"
                 "about 80% probability (5th to 95th\n"
                 "percentile along and across their\n"
                 "direction of travel).\n\n"
                 "A good model puts the black x inside\n"
                 "the box about 8 times in 10, with\n"
                 "boxes as small as possible.",
                 fontsize=8.5, color=C_INK, va="top", linespacing=1.4)
        fig.text(0.70, 0.235, "So far in this episode", fontsize=10, weight="bold", color=C_INK)
        fig.text(0.70, 0.11,
                 f"x inside its range:  {100 * seen.inside.mean():.0f}%  (target 80%)\n"
                 f"median error:          {seen.err.median():.2f} m\n"
                 f"predictions:             {len(seen):,}",
                 fontsize=9, color=C_INK, va="bottom", family="monospace", linespacing=1.6)
        fig.text(0.07, 0.025, "Boxes: model quantiles of the deviation from constant velocity, in each person's own frame.\n"
                 "Positions observed with sensor noise and smoothed by a Kalman filter; the model never sees the future.",
                 fontsize=7.5, color=C_MUTED, linespacing=1.4)
        ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.03, 0.97), frameon=False,
                  fontsize=8.5, labelcolor=C_INK, borderaxespad=0)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=4) if a.out.endswith(".mp4") else PillowWriter(fps=4)
    with writer.saving(fig, str(out), dpi=100):
        for t in times:
            draw(t); writer.grab_frame()
    draw(times[len(times) // 2]); fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"wrote {out} and {out.with_suffix('.png')} ({len(times)} frames, episode {ep}); "
          f"inside {100 * q.inside.mean():.0f}%, median error {q.err.median():.2f} m")


if __name__ == "__main__":
    main()
