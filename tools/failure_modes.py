"""Item 3: where each predictor fails. Error and joint box coverage at +2 s, split by
  - what the person does next (mode label: straight, stop, turn left, turn right)
  - distance to the nearest other person
  - distance to the robot (simulated crowd only)
for constant velocity, constant velocity + band, LightGBM (same rows), LightGBM (all rows), TabPFN.

    python tools/failure_modes.py --train data/hotel --test data/eth --noise 0      # real, hard direction
    python tools/failure_modes.py --train data/crowd --test data/crowd               # simulated, split by episode

Writes results/failure_<name>.csv and media/fig_failure_<name>.png.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight import viz as V  # noqa: E402
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from run_offline import rows_for  # noqa: E402

QS = (0.05, 0.1, 0.5, 0.9, 0.95)
MODELS = ["cv", "cv_band", "lgbm", "lgbm_all", "tabpfn"]
LABEL = dict(V.MODEL_LABEL, lgbm="LightGBM (same rows)", lgbm_all="LightGBM (all rows)")
COLOR = dict(V.MODEL_COLOR, lgbm_all="#5b8def")


def boot_ci(df, col, n_boot=300, seed=0):
    """95% interval of the row-mean of `col`, resampling whole people (rows of one person together)."""
    rng = np.random.default_rng(seed)
    g = df.groupby("person")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    if len(s) < 2:
        return np.nan, np.nan
    idx = rng.integers(0, len(s), (n_boot, len(s)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return np.quantile(means, 0.025), np.quantile(means, 0.975)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--h", type=float, default=2.0)
    ap.add_argument("--context", type=int, default=1500)
    ap.add_argument("--test-rows", type=int, default=1200)
    a = ap.parse_args()
    hz = (1.2, 2.0, 4.8) if "eth" in a.train or "hotel" in a.train else (1.0, 2.0)
    tr = rows_for(a.train, a.noise, horizons=hz); te = rows_for(a.test, a.noise, horizons=hz)
    tr, te = tr[np.isclose(tr.h, a.h)], te[np.isclose(te.h, a.h)]
    if a.train == a.test:
        m_tr, _, m_te = F.split_by_episode(tr)
        tr, te = tr[m_tr], te[m_te]
    te = te.sample(min(a.test_rows, len(te)), random_state=0)
    ctx = tr.sample(min(a.context, len(tr)), random_state=0)
    Y = te[["ddx", "ddy"]].to_numpy()
    out = te[["episode", "agent_id", "mode", "n0_dist", "robot_dist"]].copy()
    out["person"] = out.episode.astype(str) + ":" + out.agent_id.astype(str)
    for key in MODELS:
        rows = tr if key in ("cv_band", "lgbm_all") else ctx
        kind = "lgbm" if key == "lgbm_all" else key
        Q = M.make(kind).fit(F.X_of(rows), rows[["ddx", "ddy"]].to_numpy()).predict_quantiles(F.X_of(te), QS)
        out[f"err_{key}"] = np.linalg.norm(Q[:, :, 2] - Y, axis=1)
        out[f"joint_{key}"] = ((Y >= Q[:, :, 0]) & (Y <= Q[:, :, 4])).all(axis=1).astype(float)
        out[f"width_{key}"] = (Q[:, :, 3] - Q[:, :, 1]).mean(axis=1)
        print(f"{key:<9} done", flush=True)
    out["neigh"] = pd.cut(out.n0_dist, [0, 1, 2, 4, 1e9], labels=["< 1 m", "1-2 m", "2-4 m", "> 4 m"])
    has_robot = (out.robot_dist < 90).any()
    if has_robot:
        out["robot"] = pd.cut(out.robot_dist, [0, 2, 4, 1e9], labels=["< 2 m", "2-4 m", "> 4 m"])
    name = f"{Path(a.train).name}_to_{Path(a.test).name}"
    Path("results").mkdir(exist_ok=True); out.to_csv(f"results/failure_{name}.csv", index=False)

    splits = [("mode", "What the person actually did next\n(from the outcome, never a model input)")] + [("neigh", "Nearest other person")] + \
             ([("robot", "Distance to the robot")] if has_robot else [])
    fig, axes = plt.subplots(2, len(splits), figsize=(5.2 * len(splits), 7.4), facecolor="white", squeeze=False)
    for j, (col, title) in enumerate(splits):
        groups = out.groupby(col, observed=True)
        cats = [str(c) for c in groups.groups.keys()]
        x = np.arange(len(cats)); w = 0.8 / len(MODELS)
        n = groups.size().to_numpy()
        small = n < 50                                         # indicative only
        for i, key in enumerate(MODELS):
            for row, metric, scale in ((0, "err", 1.0), (1, "joint", 100.0)):
                if row == 1 and key == "cv":
                    continue
                m = scale * groups[f"{metric}_{key}"].mean().to_numpy()
                ci = np.array([boot_ci(g, f"{metric}_{key}") for _, g in groups]) * scale
                yerr = np.clip(np.vstack([m - ci[:, 0], ci[:, 1] - m]), 0, None)
                bars = axes[row][j].bar(x + (i - 2) * w, m, w * 0.92, color=COLOR[key], yerr=yerr,
                                        error_kw=dict(ecolor=V.C_INK, lw=0.8, capsize=1.5),
                                        label=LABEL[key] if row == 0 else None)
                for b, s in zip(bars, small):
                    if s:
                        b.set_alpha(0.35)
        for ax in axes[:, j]:
            V.style_axes(ax, equal=False)
            ax.set_xticks(x, [f"{c}\n(n={k}{', indicative' if s else ''})" for c, k, s in zip(cats, n, small)], fontsize=8)
        axes[0][j].set_title(title, fontsize=11, color=V.C_INK)
        axes[1][j].axhline(80, color=V.C_MUTED, ls=":", lw=1)
    axes[0][0].set_ylabel(f"Mean error at +{a.h:g} s (m)", color=V.C_MUTED)
    axes[1][0].set_ylabel("5-95% box holds the truth (%)\ntarget ~80% if the axes are independent", color=V.C_MUTED)
    axes[0][-1].legend(frameon=False, fontsize=8, labelcolor=V.C_INK)
    fig.suptitle(f"Where each predictor fails ({Path(a.train).name} → {Path(a.test).name}) · bars: 95% interval, "
                 f"resampling whole people; faded: fewer than 50 rows", fontsize=12,
                 weight="bold", color=V.C_INK, x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(f"media/fig_failure_{name}.png", dpi=150)
    tab = out.groupby("mode", observed=True)[[f"err_{k}" for k in MODELS] + [f"joint_{k}" for k in MODELS[1:]]].mean().round(3)
    print(tab.to_string())
    print(f"wrote results/failure_{name}.csv and media/fig_failure_{name}.png")


if __name__ == "__main__":
    main()
