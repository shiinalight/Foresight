"""R2: choosing context rows per moment, and adapting after the crowd changes behaviour.

    python run_adapt.py --data data/crowd --episodes 8 --every 2.0

For test episodes with a regime switch, at moments every `--every` seconds, the predictor gets a
fresh context chosen by one of four strategies, then predicts the people at that moment:
  global   1,000 random rows from other episodes (fixed)
  knn      1,000 rows from other episodes most similar to the current situations
  recent   only rows from this episode whose outcome is already known (t_known <= now)
  mix      60% recent + 30% similar + 10% random, rare behaviours kept in (cold start: bank only)
Each strategy runs with TabPFN and with LightGBM refitted on exactly the same rows.
Writes results/adapt.csv and media/fig_adapt.png (error against seconds since the switch).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from foresight import viz as V
from foresight.core import context as C
from foresight.core import features as F
from foresight.core import models as M
from run_offline import rows_for

STRATS = {"global": None, "knn": (0, 1, 0), "recent": (1, 0, 0), "mix": (0.6, 0.3, 0.1)}
BINS = [(-20, 0, "before"), (0, 5, "0-5 s"), (5, 10, "5-10 s"), (10, 30, "10+ s")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/crowd")
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--h", type=float, default=2.0)
    ap.add_argument("--models", nargs="*", default=["tabpfn", "lgbm"])
    ap.add_argument("--train-data", default=None,
                    help="bank from another dataset; x axis becomes seconds in the scene (e.g. --train-data data/hotel --data data/eth)")
    ap.add_argument("--horizons", type=float, nargs="*", default=[1.0, 2.0])
    a = ap.parse_args()

    df = rows_for(a.data, a.noise, horizons=tuple(a.horizons))
    df = df[np.isclose(df.h, a.h)]
    if a.train_data:
        src = rows_for(a.train_data, a.noise, horizons=tuple(a.horizons))
        src = src[np.isclose(src.h, a.h)]
        bank, glob, test = C.Bank(src), src.sample(min(a.budget, len(src)), random_state=0), df.copy()
        test["since_switch"] = test["t"]                       # seconds since the robot entered this scene
        frozen = M.make("lgbm").fit(F.X_of(src), src[["ddx", "ddy"]].to_numpy())
        global BINS
        BINS = [(0, 5, "0-5 s"), (5, 10, "5-10 s"), (10, 20, "10-20 s"), (20, 40, "20-40 s"), (40, 61, "40-60 s")]
        xlabel = "Seconds in the new scene"
    else:
        tr, _, te = F.split_by_episode(df)
        bank = C.Bank(df[tr])
        glob = df[tr].sample(a.budget, random_state=0)
        test = df[te & df.since_switch.notna()]
        frozen = None
        xlabel = "Time since the crowd changed behaviour"
    eps = sorted(test.episode.unique())[: a.episodes]
    rows = []
    t0 = time.perf_counter()
    for ep in eps:
        live = df[df.episode == ep]
        q_ep = test[test.episode == ep]
        all_t = sorted(q_ep.t.unique())
        step = max(1, int(round(a.every / np.median(np.diff(all_t)))))
        times = all_t[::step]
        for t in times:
            q = q_ep[np.isclose(q_ep.t, t)]
            for sname, mix in STRATS.items():
                ctx = glob if mix is None else C.select(q, bank, live=live, budget=a.budget, mix=mix)
                if len(ctx) < 20:
                    continue                                   # recent-only before enough outcomes exist
                for mname in a.models + (["lgbm_frozen"] if frozen is not None and sname == "global" else []):
                    m = frozen if mname == "lgbm_frozen" else M.make(mname).fit(F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy())
                    Q = m.predict_quantiles(F.X_of(q), (0.1, 0.5, 0.9))
                    Y = q[["ddx", "ddy"]].to_numpy()
                    err = np.linalg.norm(Q[:, :, 1] - Y, axis=1)
                    cov = ((Y >= Q[:, :, 0]) & (Y <= Q[:, :, 2])).mean(axis=1)
                    for e, c, s in zip(err, cov, q.since_switch):
                        rows.append(dict(episode=ep, t=t, since=s, strategy=sname, model=mname, err=e, cov=c,
                                         n_ctx=len(ctx)))
        print(f"episode {ep} done ({time.perf_counter() - t0:.0f} s)")
    res = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True); res.to_csv("results/adapt.csv", index=False)
    res["bin"] = pd.cut(res.since, [b[0] for b in BINS] + [BINS[-1][1]], labels=[b[2] for b in BINS], right=False)
    tab = res.groupby(["model", "strategy", "bin"], observed=True).err.mean().unstack("bin").round(3)
    print(tab)
    print(res.groupby(["model", "strategy"]).agg(err=("err", "mean"), cov=("cov", "mean")).round(3))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor="white")
    colors = {"global": "#8b949e", "knn": "#2a6fdb", "recent": "#7a3fb5", "mix": "#e8590c"}
    labels = {"global": "Random rows, other episodes", "knn": "Similar situations", "recent": "This crowd only",
              "mix": "Mix: this crowd + similar"}
    for ax, mname in zip(axes, a.models):
        V.style_axes(ax, equal=False)
        for s in STRATS:
            r = tab.loc[(mname, s)] if (mname, s) in tab.index else None
            if r is not None:
                ax.plot(list(r.index.astype(str)), r.values, marker="o", lw=2, color=colors[s], label=labels[s])
        ax.set_title(V.MODEL_LABEL.get(mname, mname), fontsize=11, color=V.C_INK)
        ax.set_xlabel(xlabel, color=V.C_MUTED)
        if mname == "lgbm" and ("lgbm_frozen", "global") in tab.index:
            r = tab.loc[("lgbm_frozen", "global")]
            ax.plot(list(r.index.astype(str)), r.values, marker="s", lw=1.5, ls="--", color="#1f2328",
                    label="Frozen, all rows of the other scene")
    axes[0].set_ylabel(f"Mean error at +{a.h:g} s (m)", color=V.C_MUTED)
    axes[-1].legend(frameon=False, fontsize=9, labelcolor=V.C_INK)
    fig.suptitle("Which context rows, after the crowd changes", fontsize=13, weight="bold", color=V.C_INK, x=0.06, ha="left")
    tag = f"_{Path(a.train_data).name}_to_{Path(a.data).name}" if a.train_data else ""
    fig.tight_layout(); fig.savefig(f"media/fig_adapt{tag}.png", dpi=150)
    res.to_csv(f"results/adapt{tag}.csv", index=False)
    print(f"wrote results/adapt{tag}.csv and media/fig_adapt{tag}.png")


if __name__ == "__main__":
    main()
