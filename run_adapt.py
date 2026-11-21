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
    ap.add_argument("--bank-exclude-regime", default=None,
                    help="drop bank episodes that ever show this regime (T3: a behaviour the bank never saw)")
    ap.add_argument("--axis", default=None, choices=[None, "time", "switch"],
                    help="x axis with --train-data: seconds in the scene (default) or since the regime switch")
    ap.add_argument("--tabpfn-fast", type=int, default=1, help="TabPFN-3.5-Fast, 1 estimator: same accuracy, ~13x faster")
    a = ap.parse_args()

    df = rows_for(a.data, a.noise, horizons=tuple(a.horizons))
    df = df[np.isclose(df.h, a.h)]
    if a.train_data:
        src = rows_for(a.train_data, a.noise, horizons=tuple(a.horizons))
        src = src[np.isclose(src.h, a.h)]
        if a.bank_exclude_regime:
            bad = src[src.regime == a.bank_exclude_regime].episode.unique()
            src = src[~src.episode.isin(bad)]
            print(f"bank: {src.episode.nunique()} episodes without '{a.bank_exclude_regime}'", flush=True)
        bank, glob, test = C.Bank(src), src.sample(min(a.budget, len(src)), random_state=0), df.copy()
        if a.axis != "switch":
            test["since_switch"] = test["t"]                   # seconds since the robot entered this scene
        frozen = M.make("lgbm").fit(F.X_of(src), src[["ddx", "ddy"]].to_numpy())
        global BINS
        if a.axis == "switch":
            BINS = [(-20, 0, "before"), (0, 5, "0-5 s"), (5, 10, "5-10 s"), (10, 21, "10-20 s")]
            xlabel = "Seconds since the crowd switched to a behaviour the bank never saw"
        else:
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
    # rows from the current scene are labelled from observed (noisy, filtered) positions, as the robot
    # would label them itself; queries are still scored against true positions
    obs = rows_for(a.data, a.noise, horizons=tuple(a.horizons), label_from="observed")
    obs = obs[np.isclose(obs.h, a.h)]
    rows = []
    t0 = time.perf_counter()
    for ep in eps:
        live = obs[obs.episode == ep]
        q_ep = test[test.episode == ep]
        all_t = sorted(q_ep.t.unique())
        step = max(1, int(round(a.every / np.median(np.diff(all_t)))))
        times = all_t[::step]
        for t in times:
            q = q_ep[np.isclose(q_ep.t, t)]
            for sname, mix in STRATS.items():
                ctx = glob if mix is None else C.select(q, bank, live=live, budget=a.budget, mix=mix)
                thin = len(ctx) < 20       # e.g. "recent" before enough outcomes exist: scored as constant
                                           # velocity with no range, on the same queries as every strategy
                for mname in a.models + (["lgbm_frozen"] if frozen is not None and sname == "global" else []):
                    if thin and mname != "lgbm_frozen":
                        Q = np.zeros((len(q), 2, 5))
                    else:
                        m = frozen if mname == "lgbm_frozen" else M.make(mname, **({"fast": True, "n_estimators": 1} if (mname == "tabpfn" and a.tabpfn_fast) else {})).fit(F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy())
                        Q = m.predict_quantiles(F.X_of(q), (0.05, 0.1, 0.5, 0.9, 0.95))
                    Y = q[["ddx", "ddy"]].to_numpy()
                    err = np.linalg.norm(Q[:, :, 2] - Y, axis=1)
                    inside = (Y >= Q[:, :, 1]) & (Y <= Q[:, :, 3])
                    cov, joint = inside.mean(axis=1), inside.all(axis=1)      # 10-90: per-axis; 2D box
                    joint90 = ((Y >= Q[:, :, 0]) & (Y <= Q[:, :, 4])).all(axis=1)   # the robot's 5-95% box (~81%)
                    for e, c, j, j9, s, aid in zip(err, cov, joint, joint90, q.since_switch, q.agent_id):
                        rows.append(dict(episode=ep, agent_id=aid, t=t, since=s, strategy=sname, model=mname, err=e,
                                         cov=c, joint=float(j), joint90=float(j9), n_ctx=len(ctx), thin=thin))
        print(f"episode {ep} done ({time.perf_counter() - t0:.0f} s)")
    res = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True); res.to_csv("results/adapt.csv", index=False)
    res["bin"] = pd.cut(res.since, [b[0] for b in BINS] + [BINS[-1][1]], labels=[b[2] for b in BINS], right=False)
    tab = res.groupby(["model", "strategy", "bin"], observed=True).err.mean().unstack("bin").round(3)
    jtab = res.groupby(["model", "strategy", "bin"], observed=True).joint90.mean().unstack("bin").round(3)
    print("mean error (m)")
    print(tab)
    print("joint coverage of the 5-95% box (the robot's box; ~81% if the axes are independent)")
    print(jtab)
    print(res.groupby(["model", "strategy"]).agg(err=("err", "mean"), cov=("cov", "mean"), joint=("joint", "mean"), joint90=("joint90", "mean")).round(3))

    fig, axes2 = plt.subplots(2, len(a.models), figsize=(5.5 * len(a.models), 7.6), sharey="row", facecolor="white",
                              squeeze=False)
    axes = axes2[0]
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
    for ax, mname in zip(axes2[1], a.models):
        V.style_axes(ax, equal=False)
        for s in STRATS:
            if (mname, s) in jtab.index:
                r = jtab.loc[(mname, s)]
                ax.plot(list(r.index.astype(str)), 100 * r.values, marker="o", lw=2, color=colors[s])
        if mname == "lgbm" and ("lgbm_frozen", "global") in jtab.index:
            r = jtab.loc[("lgbm_frozen", "global")]
            ax.plot(list(r.index.astype(str)), 100 * r.values, marker="s", lw=1.5, ls="--", color="#1f2328")
        ax.set_xlabel(xlabel, color=V.C_MUTED)
    axes2[1][0].set_ylabel("5-95% box holds the truth (%)", color=V.C_MUTED)
    for ax in axes2[1]:
        ax.axhline(81, color=V.C_MUTED, ls=":", lw=1); ax.text(0, 82, "~81% target (axes independent)", fontsize=7, color=V.C_MUTED)
    axes[0].set_ylabel(f"Mean error at +{a.h:g} s (m)", color=V.C_MUTED)
    axes[-1].legend(frameon=False, fontsize=9, labelcolor=V.C_INK)
    fig.suptitle("Which context rows, after the crowd changes", fontsize=13, weight="bold", color=V.C_INK, x=0.06, ha="left")
    tag = f"_{Path(a.train_data).name}_to_{Path(a.data).name}" if a.train_data else ""
    fig.tight_layout(); fig.savefig(f"media/fig_adapt{tag}.png", dpi=150)
    res.to_csv(f"results/adapt{tag}.csv", index=False)
    print(f"wrote results/adapt{tag}.csv and media/fig_adapt{tag}.png")


if __name__ == "__main__":
    main()
