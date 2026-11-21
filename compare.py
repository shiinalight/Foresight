"""Same moment, four models side by side (constant velocity + band, LightGBM x2, TabPFN).

    python compare.py --data data/hotel --train-data data/eth --noise 0 --out media/compare_hotel.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from foresight import viz as V
from foresight.core import features as F
from foresight.core import models as M
from foresight.core import schema
from run_offline import rows_for

PANELS = [("cv_band", "cv_band", "all"), ("lgbm", "lgbm", "ctx"), ("lgbm", "lgbm", "all"), ("tabpfn", "tabpfn", "ctx")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--t", type=float, default=None)
    ap.add_argument("--context", type=int, default=1500)
    ap.add_argument("--h", type=float, default=2.0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--out", default="media/compare.png")
    a = ap.parse_args()

    tracks, obst, meta = schema.load(a.data)
    pool = rows_for(a.train_data, a.noise, horizons=(a.h,))
    test = rows_for(a.data, a.noise, horizons=(a.h,))
    ep = a.episode if a.episode is not None else int(test.groupby("episode").size().idxmax())
    q_ep = test[test.episode == ep]
    t = a.t if a.t is not None else float(q_ep.groupby("t").size().idxmax())   # busiest moment
    q_t = q_ep[np.isclose(q_ep.t, t)]
    ctx = pool.sample(min(a.context, len(pool)), random_state=0)
    tr_ep = tracks[tracks.episode == ep]
    obst_ep = obst[obst.episode == ep] if len(obst) else obst

    fig, axes = plt.subplots(1, 4, figsize=(16, 5.2), facecolor="white")
    for ax, (key, kind, rows) in zip(axes, PANELS):
        fit = pool if rows == "all" else ctx
        m = M.make(kind).fit(F.X_of(fit), fit[["ddx", "ddy"]].to_numpy())
        # score over the whole episode, draw the chosen moment
        scored = V.predict_rows(m, q_ep)
        V.draw_scene(ax, tr_ep, obst_ep, scored[np.isclose(scored.t, t)], t, (meta.get("draw_bounds") or meta["bounds"]), axis_labels=False)
        name = {"cv_band": "Const. velocity + band", "tabpfn": "TabPFN"}.get(key, "LightGBM")
        sub = f"{len(fit):,} rows" if key != "cv_band" else "fixed band"
        ax.set_title(f"{name} ({sub})\nin range {100 * scored.inside.mean():.0f}% · error {scored.err.mean():.2f} m",
                     fontsize=10, color=V.C_INK)
    axes[-1].legend(handles=V.scene_legend(a.h), loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
                    fontsize=9, labelcolor=V.C_INK)
    nice = {"eth": "ETH", "hotel": "Hotel", "toy": "simulated crowd"}
    fig.suptitle(f"Same people, four predictors (+{a.h:g} s) · context {nice.get(Path(a.train_data).name, Path(a.train_data).name)}, "
                 f"test {nice.get(Path(a.data).name, Path(a.data).name)}", fontsize=13, weight="bold", color=V.C_INK, x=0.01, ha="left")
    fig.tight_layout(); Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"wrote {a.out} (episode {ep}, t={t:.1f})")


if __name__ == "__main__":
    main()
