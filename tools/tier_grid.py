"""Model-tier grid on the hosted API (docs/DEPLOYMENT.md item 2). Replaces F3, which compared
3.5-Fast with 1 estimator against the default with 8.

    python tools/tier_grid.py            # prints the call count first

Grid: {v3.5, v3.5-fast} x n_estimators {1, 2, 4, 8} x context {250, 500, 1000, 2000}, plus v2 (the
Apache-2.0 checkpoint) at its default ensemble, on two setups at +2 s:
  hotel->eth   context from Hotel, test on ETH (new place)
  crowd        context from crowd training episodes, test on held-out episodes (same place)
Same context sample and same 600 test rows for every cell. Reports error, pinball (5 quantiles),
joint coverage of the 5-95% box (target ~81% if axes independent) and 10-90% width.
Writes results/tier_grid.csv and media/fig_tier_grid.png.
"""
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight import viz as V  # noqa: E402
from foresight.core import features as F  # noqa: E402
from run_offline import rows_for  # noqa: E402

QS = [0.05, 0.1, 0.5, 0.9, 0.95]
MODELS = [("v3.5_default", e) for e in (1, 2, 4, 8)] + [("v3.5-fast_default", e) for e in (1, 2, 4, 8)] + [("v2_default", None)]
CTX = [250, 500, 1000, 2000]


def setups():
    hz = (1.2, 2.0, 4.8)
    src = rows_for("data/hotel", 0.0, horizons=hz); dst = rows_for("data/eth", 0.0, horizons=hz)
    yield "hotel->eth", src[np.isclose(src.h, 2.0)], dst[np.isclose(dst.h, 2.0)].sample(600, random_state=0)
    cr = rows_for("data/crowd", 0.1, horizons=(1.0, 2.0)); cr = cr[np.isclose(cr.h, 2.0)]
    tr, _, te = F.split_by_episode(cr)
    yield "crowd", cr[tr], cr[te].sample(600, random_state=0)


def score(Q, Y):
    err = np.linalg.norm(Q[:, :, 2] - Y, axis=1).mean()
    pin = np.mean([np.mean(np.maximum(q * (Y - Q[:, :, j]), (q - 1) * (Y - Q[:, :, j]))) for j, q in enumerate(QS)])
    joint = ((Y >= Q[:, :, 0]) & (Y <= Q[:, :, 4])).all(axis=1).mean()
    width = (Q[:, :, 3] - Q[:, :, 1]).mean()
    return err, pin, joint, width


def main():
    n_calls = 2 * len(MODELS) * len(CTX) * 2          # setups x models x contexts x 2 axes
    print(f"API predict calls: {n_calls} (= {10_000 * n_calls:,} credits)", flush=True)
    import tabpfn_client
    from tabpfn_client import TabPFNRegressor
    tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
    rows = []
    for name, src, te in setups():
        Y = te[["ddx", "ddy"]].to_numpy(); Xte = F.X_of(te)
        for n in CTX:
            ctx = src.sample(min(n, len(src)), random_state=0)
            X, Yc = F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy()
            for path, est in MODELS:
                t0 = time.perf_counter()
                Q = np.zeros((len(te), 2, len(QS)))
                for ax in range(2):
                    m = TabPFNRegressor(model_path=path, **({"n_estimators": est} if est else {}))
                    m.fit(X, Yc[:, ax])
                    Q[:, ax, :] = np.stack(m.predict(Xte, output_type="quantiles", quantiles=QS), axis=1)
                Q.sort(axis=2)
                err, pin, joint, width = score(Q, Y)
                rows.append(dict(setup=name, model=path.replace("_default", ""), n_estimators=est or "default", context=n,
                                 err=err, pinball=pin, joint90=joint, width80=width, seconds=time.perf_counter() - t0))
                print(f"{name:<11} {path:<18} est {str(est or 'def'):<4} ctx {n:<5} err {err:.3f}  pinball {pin:.4f}  "
                      f"joint90 {100 * joint:.0f}%  width {width:.2f}  ({rows[-1]['seconds']:.0f} s)", flush=True)
                pd.DataFrame(rows).to_csv("results/tier_grid.csv", index=False)     # checkpoint
    R = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), facecolor="white")
    for i, name in enumerate(("hotel->eth", "crowd")):
        g = R[R.setup == name]
        for (model, est), h in g.groupby(["model", "n_estimators"]):
            col = {"v3.5": "#e8590c", "v3.5-fast": "#2a6fdb", "v2": "#7a3fb5"}[model]
            lw = {1: 1.0, 2: 1.5, 4: 2.0, 8: 2.5}.get(est, 2.0)
            lab = f"{model}, {est} est" if est != "default" else f"{model} (default est)"
            axes[i][0].plot(h.context, h.pinball, marker="o", lw=lw, color=col, label=lab)
            axes[i][1].plot(h.context, 100 * h.joint90, marker="o", lw=lw, color=col)
        axes[i][0].set_title(f"{name}: pinball (lower is better)", fontsize=10, color=V.C_INK)
        axes[i][1].set_title(f"{name}: 5-95% box holds the truth (%)", fontsize=10, color=V.C_INK)
        axes[i][1].axhline(81, color=V.C_MUTED, ls=":", lw=1)
        for ax in axes[i]:
            V.style_axes(ax, equal=False); ax.set_xscale("log"); ax.set_xlabel("Context rows", color=V.C_MUTED)
    axes[0][0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Which TabPFN tier: accuracy and calibration against model, ensemble size and context", fontsize=12,
                 weight="bold", color=V.C_INK, x=0.02, ha="left")
    fig.tight_layout(); fig.savefig("media/fig_tier_grid.png", dpi=150)
    print("wrote results/tier_grid.csv and media/fig_tier_grid.png")


if __name__ == "__main__":
    main()
