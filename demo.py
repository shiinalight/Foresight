"""R1 demo: animate one test episode with each person's predicted 2 s range drawn.

    python demo.py --data data/toy --episode 3 --model lgbm --out results/demo_ep3.mp4
    python demo.py --data data/toy --episode 3 --model tabpfn --context 1500

Predictions use a context of training episodes only. Each person gets an ellipse at the
predicted 2 s position (median) spanning the 10-90% range in its own frame; the dot it
later reaches is drawn too, so viewers see whether the truth lands inside.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.patches import Circle, Ellipse

from foresight.core import features as F
from foresight.core import models as M
from foresight.core import schema
from run_offline import rows_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--context", type=int, default=1500)
    ap.add_argument("--h", type=float, default=2.0)
    ap.add_argument("--out", default="results/demo.gif")
    a = ap.parse_args()

    tracks, obst, meta = schema.load(a.data)
    df = rows_for(a.data, 0.1)
    tr, _, te = F.split_by_episode(df)
    ep = a.episode if a.episode is not None else int(df[te].episode.iloc[0])
    ctx = df[tr].sample(min(a.context, int(tr.sum())), random_state=0)
    q = df[(df.episode == ep) & np.isclose(df.h, a.h)].copy()
    m = M.make(a.model).fit(F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy())
    Q = m.predict_quantiles(F.X_of(q))
    mx, my = F.to_world(q, Q[:, 0, 1], Q[:, 1, 1])
    q["mx"], q["my"] = mx, my
    q["wl"], q["wd"] = Q[:, 0, 2] - Q[:, 0, 0], Q[:, 1, 2] - Q[:, 1, 0]

    tr_ep = tracks[tracks.episode == ep]
    times = sorted(q.t.unique())
    b = meta["bounds"]
    fig, ax = plt.subplots(figsize=(6, 6))

    def draw(t):
        ax.clear(); ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3]); ax.set_aspect("equal")
        ax.set_title(f"episode {ep}  t={t:.1f}s  model={a.model}  range at +{a.h:.0f}s")
        for _, o in obst[obst.episode == ep].iterrows():
            ax.add_patch(Circle((o.x, o.y), o.r, color="0.6"))
        now = tr_ep[np.isclose(tr_ep.t, t)]
        for _, r in now.iterrows():
            ax.add_patch(Circle((r.x, r.y), r.radius, color="tab:red" if r.is_ego else "tab:blue"))
        for _, r in q[np.isclose(q.t, t)].iterrows():
            ax.add_patch(Ellipse((r.mx, r.my), max(r.wl, 0.05), max(r.wd, 0.05), angle=np.degrees(r.theta),
                                 fill=False, color="tab:orange", lw=1.2))
            ax.plot([r.px, r.mx], [r.py, r.my], color="tab:orange", lw=0.6)
            ax.plot(r.true_x, r.true_y, "k.", ms=3)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=4) if a.out.endswith(".mp4") else PillowWriter(fps=4)
    with writer.saving(fig, a.out, dpi=100):
        for t in times:
            draw(t); writer.grab_frame()
    print(f"wrote {a.out} ({len(times)} frames)")


if __name__ == "__main__":
    main()
