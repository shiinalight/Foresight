"""Demo animation: one test episode with each person's predicted range drawn.

    python demo.py --data data/hotel --train-data data/eth --noise 0 --model tabpfn --out results/demo_hotel_tabpfn.gif
    python demo.py --data data/toy --model tabpfn --out results/demo_toy_tabpfn.gif

The model only sees context rows from other episodes (or another dataset with --train-data).
Writes a GIF (MP4 if the name ends in .mp4 and ffmpeg is installed) and a PNG still.
How to read the figure: RESULTS.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter

from foresight import viz as V
from foresight.core import features as F
from foresight.core import models as M
from foresight.core import schema
from run_offline import rows_for


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
    df = rows_for(a.data, a.noise, horizons=(a.h,))
    if a.train_data:
        pool, test = rows_for(a.train_data, a.noise, horizons=(a.h,)), df
        src = f"context from {Path(a.train_data).name}"
    else:
        tr, _, te = F.split_by_episode(df)
        pool, test = df[tr], df[te]
        src = "context from other episodes"
    ep = a.episode if a.episode is not None else int(test.groupby("episode").size().idxmax())
    ctx = pool.sample(min(a.context, len(pool)), random_state=0)
    q = test[test.episode == ep]
    times = sorted(q.t.unique())[: a.max_frames]
    m = M.make(a.model).fit(F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy())
    q = V.predict_rows(m, q[q.t.isin(times)])

    tr_ep = tracks[tracks.episode == ep]
    obst_ep = obst[obst.episode == ep] if len(obst) else obst
    legend = V.scene_legend(a.h, bool((tr_ep.is_ego == 1).any()), len(obst_ep) > 0)
    fig = plt.figure(figsize=(8.4, 6.6), facecolor="white")
    ax = fig.add_axes([0.08, 0.08, 0.68, 0.80])

    def draw(t):
        V.draw_scene(ax, tr_ep, obst_ep, q[np.isclose(q.t, t)], t, (meta.get("draw_bounds") or meta["bounds"]))
        seen = q[q.t <= t + 1e-6]
        fig.texts.clear()
        fig.text(0.08, 0.94, f"{V.MODEL_LABEL[a.model]}: where will each person be in {a.h:g} s?",
                 fontsize=13, weight="bold", color=V.C_INK)
        fig.text(0.08, 0.905, f"{meta.get('sim', '')} · {src} · t = {t:.1f} s", fontsize=9, color=V.C_MUTED)
        fig.text(0.78, 0.30, f"inside range\n{100 * seen.inside.mean():.0f}%", fontsize=11, color=V.C_INK, linespacing=1.3)
        fig.text(0.78, 0.18, f"median error\n{seen.err.median():.2f} m", fontsize=11, color=V.C_INK, linespacing=1.3)
        ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, fontsize=9,
                  labelcolor=V.C_INK, borderaxespad=0)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=4) if a.out.endswith(".mp4") else PillowWriter(fps=4)
    with writer.saving(fig, str(out), dpi=100):
        for t in times:
            draw(t); writer.grab_frame()
    draw(times[len(times) // 2]); fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"wrote {out} ({len(times)} frames, episode {ep}); inside {100 * q.inside.mean():.0f}%, "
          f"median error {q.err.median():.2f} m")


if __name__ == "__main__":
    main()
