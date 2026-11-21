"""95% person-bootstrap intervals for the adaptation results (results/adapt*.csv from run_adapt.py).

    python tools/adapt_ci.py results/adapt_hotel_to_eth.csv

Per (model, strategy, time bin): mean error and joint coverage of the 5-95% box, each with a 95%
interval that resamples whole people (episode:agent_id), and n. A gap is called real only when
the intervals do not overlap. Writes <csv stem>_ci.csv and media/fig_<stem>_ci.png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight import viz as V  # noqa: E402


def boot(df, col, n=400, seed=0):
    g = df.groupby("person")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    if len(s) < 2:
        return np.nan, np.nan
    idx = np.random.default_rng(seed).integers(0, len(s), (n, len(s)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return np.quantile(m, 0.025), np.quantile(m, 0.975)


def main(path):
    d = pd.read_csv(path)
    d["person"] = d.episode.astype(str) + ":" + d.agent_id.astype(str)
    edges = [-20, 0, 5, 10, 30] if (d.since < 0).any() else [0, 5, 10, 20, 40, 61]
    labels = [f"{a}-{b} s" if a >= 0 else "before" for a, b in zip(edges[:-1], edges[1:])]
    d["bin"] = pd.cut(d.since, edges, labels=labels, right=False)
    rows = []
    for (m, s, b), g in d.groupby(["model", "strategy", "bin"], observed=True):
        e_lo, e_hi = boot(g, "err"); j_lo, j_hi = boot(g, "joint90")
        rows.append(dict(model=m, strategy=s, bin=str(b), n=len(g), people=g.person.nunique(),
                         err=g.err.mean(), err_lo=e_lo, err_hi=e_hi, joint90=g.joint90.mean(), j_lo=j_lo, j_hi=j_hi))
    R = pd.DataFrame(rows)
    out = Path(path).with_name(Path(path).stem + "_ci.csv"); R.to_csv(out, index=False)
    with pd.option_context("display.width", 200):
        print(R.round(3).to_string(index=False))
    show = [k for k in ("tabpfn", "lgbm", "lgbm_frozen") if k in set(R.model)]
    colors = {"global": "#8b949e", "knn": "#2a6fdb", "recent": "#7a3fb5", "mix": "#e8590c"}
    fig, axes = plt.subplots(2, len(show), figsize=(5.2 * len(show), 7.4), sharey="row", facecolor="white", squeeze=False)
    for j, m in enumerate(show):
        for s, col in colors.items():
            g = R[(R.model == m) & (R.strategy == s)]
            if not len(g):
                continue
            x = np.arange(len(g)) + (list(colors).index(s) - 1.5) * 0.08
            axes[0][j].errorbar(x, g.err, yerr=[g.err - g.err_lo, g.err_hi - g.err], fmt="o-", color=col, lw=1.5,
                                capsize=2, label=s)
            axes[1][j].errorbar(x, 100 * g.joint90, yerr=[100 * (g.joint90 - g.j_lo), 100 * (g.j_hi - g.joint90)],
                                fmt="o-", color=col, lw=1.5, capsize=2)
            for ax in axes[:, j]:
                ax.set_xticks(np.arange(len(g)), list(g.bin), fontsize=8)
        axes[0][j].set_title({"lgbm_frozen": "LightGBM, frozen (other scene)", "lgbm": "LightGBM, refit"}.get(m, "TabPFN"),
                             fontsize=11, color=V.C_INK)
        axes[1][j].axhline(81, color=V.C_MUTED, ls=":", lw=1)
        for ax in axes[:, j]:
            V.style_axes(ax, equal=False)
    axes[0][0].set_ylabel("Mean error at +2 s (m)", color=V.C_MUTED)
    axes[1][0].set_ylabel("5-95% box holds the truth (%)\n(dotted: ~81% if axes independent)", color=V.C_MUTED)
    axes[0][-1].legend(frameon=False, fontsize=8, title="context rows", title_fontsize=8)
    fig.suptitle(f"{Path(path).stem}: 95% intervals resampling whole people", fontsize=12, weight="bold",
                 color=V.C_INK, x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(f"media/fig_{Path(path).stem}_ci.png", dpi=150)
    print(f"wrote {out} and media/fig_{Path(path).stem}_ci.png")


if __name__ == "__main__":
    main(sys.argv[1])
