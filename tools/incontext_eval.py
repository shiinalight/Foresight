"""The decisive in-scene test: does re-conditioning on the new scene's rows make TabPFN sharper
at matched coverage, compared with LightGBM?

    python tools/incontext_eval.py --train data/hotel --test data/eth --episodes 4 --every 4 --api

At moments every --every seconds in --episodes test episodes, the people at that moment get
predictions (+2 s) from:
  tabpfn_mix   TabPFN, context = 60% this scene's completed rows (labelled from observations)
               + 30% similar bank rows + 10% stratified fill (context.select), refreshed per moment
  lgbm_mix     LightGBM refit on exactly the same rows per moment
  lgbm_all     LightGBM on every training-scene row, frozen
  tabpfn       TabPFN on 1,500 training-scene rows, frozen
Each gets the same calibration: a split offset from a held-out 30% of the training scene, then the
online adaptive offset from this scene's outcomes (causal). Reported per seconds in the scene:
joint coverage of the 10-90% box and its width.
Prints the number of API calls (10,000 credits each) before running.
"""
import argparse
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
from foresight.core import conformal as CF  # noqa: E402
from foresight.core import context as C  # noqa: E402
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from run_offline import rows_for  # noqa: E402

QS = (0.1, 0.5, 0.9)
BINS = [0, 5, 10, 20, 40, 61]
LABEL = {"tabpfn_mix": "TabPFN, this scene + similar rows", "lgbm_mix": "LightGBM, same rows refit",
         "lgbm_all": "LightGBM, all training rows (frozen)", "tabpfn": "TabPFN, training rows (frozen)"}
COLOR = {"tabpfn_mix": "#e8590c", "lgbm_mix": "#2a6fdb", "lgbm_all": "#5b8def", "tabpfn": "#f4a261"}
STYLE = {"tabpfn_mix": "-", "lgbm_mix": "-", "lgbm_all": "--", "tabpfn": "--"}


def tab(api):
    return M.make("tabpfn", **({"api": True} if api else {"fast": True, "n_estimators": 1}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--h", type=float, default=2.0)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--every", type=float, default=4.0)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the call-count confirmation pause")
    a = ap.parse_args()
    hz = (1.2, 2.0, 4.8)
    src = rows_for(a.train, a.noise, horizons=hz); src = src[np.isclose(src.h, a.h)]
    dst = rows_for(a.test, a.noise, horizons=hz); dst = dst[np.isclose(dst.h, a.h)]
    obs = rows_for(a.test, a.noise, horizons=hz, label_from="observed"); obs = obs[np.isclose(obs.h, a.h)]
    fit_m, _, cal_m = F.split_by_episode(src, test_frac=0.3, seed=1)
    fit, cal = src[fit_m], src[cal_m]
    bank = C.Bank(fit)
    eps = sorted(dst.episode.unique())[: a.episodes]
    moments = []
    for ep in eps:
        d = dst[dst.episode == ep]
        ts = sorted(d.t.unique())
        step = max(1, int(round(a.every / np.median(np.diff(ts)))))
        moments += [(ep, t) for t in ts[::step]]
    test = pd.concat([dst[(dst.episode == ep) & np.isclose(dst.t, t)] for ep, t in moments]).sort_values(["episode", "t"])
    n_calls = (2 * len(moments) + 2 * 2) if a.api else 0          # mix per moment; frozen: calib + test
    print(f"{len(moments)} moments, {len(test)} query rows. API predict calls: {n_calls} "
          f"(= {10_000 * n_calls:,} credits)", flush=True)
    if a.api and not a.yes:
        time.sleep(5)

    Y = test[["ddx", "ddy"]].to_numpy()
    Yc = cal[["ddx", "ddy"]].to_numpy()
    preds, d0 = {}, {}
    # frozen models: one batch each
    for key in ("lgbm_all", "tabpfn"):
        rows = fit if key == "lgbm_all" else fit.sample(min(a.budget, len(fit)), random_state=0)
        m = (M.make("lgbm") if key == "lgbm_all" else tab(a.api)).fit(F.X_of(rows), rows[["ddx", "ddy"]].to_numpy())
        Qc = m.predict_quantiles(F.X_of(cal), QS)
        d0[key] = CF.split_offset(Qc[:, :, 0], Qc[:, :, 2], Yc, a.alpha)
        preds[key] = m.predict_quantiles(F.X_of(test), QS)
    # arrival offsets for the per-moment models: the same model family on bank rows only
    d0["tabpfn_mix"], d0["lgbm_mix"] = d0["tabpfn"], CF.split_offset(
        *(lambda Q: (Q[:, :, 0], Q[:, :, 2]))(M.make("lgbm").fit(
            F.X_of(fit.sample(min(a.budget, len(fit)), random_state=0)),
            fit.sample(min(a.budget, len(fit)), random_state=0)[["ddx", "ddy"]].to_numpy()).predict_quantiles(F.X_of(cal), QS)),
        Yc, a.alpha)
    # per-moment models
    Qm = {"tabpfn_mix": [], "lgbm_mix": []}
    n_live = []
    t0 = time.perf_counter()
    for ep, t in moments:
        q = test[(test.episode == ep) & np.isclose(test.t, t)]
        live = obs[(obs.episode == ep)]
        ctx = C.select(q, bank, live=live, budget=a.budget, mix=(0.6, 0.3, 0.1))
        n_live.append(int(((live.t_known <= t + 1e-9)).sum()))
        Xc, Yctx = F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy()
        Qm["tabpfn_mix"].append(tab(a.api).fit(Xc, Yctx).predict_quantiles(F.X_of(q), QS))
        Qm["lgbm_mix"].append(M.make("lgbm").fit(Xc, Yctx).predict_quantiles(F.X_of(q), QS))
    preds["tabpfn_mix"] = np.concatenate(Qm["tabpfn_mix"]); preds["lgbm_mix"] = np.concatenate(Qm["lgbm_mix"])
    print(f"per-moment fits done ({time.perf_counter() - t0:.0f} s)", flush=True)

    sec = test.t.to_numpy(); rows = []
    for key, Q in preds.items():
        lo, hi = Q[:, :, 0], Q[:, :, 2]
        d_on, _ = CF.online_offsets(test.t.to_numpy(), test.t_known.to_numpy(), test.episode.to_numpy(),
                                    lo, hi, Y, d0[key], a.alpha, a.eta)
        for variant, d in (("raw", np.zeros(len(test))), ("online", d_on)):
            L, H = CF.widen(lo, hi, d)
            inside = ((Y >= L) & (Y <= H)).all(axis=1); width = (H - L).mean(axis=1)
            err = np.linalg.norm(Q[:, :, 1] - Y, axis=1)
            for b0, b1 in zip(BINS[:-1], BINS[1:]):
                k = (sec >= b0) & (sec < b1)
                if k.any():
                    rows.append(dict(model=key, variant=variant, bin=f"{b0}-{b1 - 1 if b1 == 61 else b1} s", b0=b0,
                                     joint=inside[k].mean(), width=width[k].mean(), err=err[k].mean(), n=int(k.sum())))
    R = pd.DataFrame(rows)
    name = f"{Path(a.train).name}_to_{Path(a.test).name}"
    R.to_csv(f"results/incontext_{name}.csv", index=False)
    piv = R[R.variant == "online"].pivot_table(index="model", columns="b0", values=["joint", "width"]).round(3)
    print(piv.to_string())
    print("raw (no calibration):")
    print(R[R.variant == "raw"].pivot_table(index="model", columns="b0", values=["joint", "err"]).round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor="white")
    for key in preds:
        g = R[(R.model == key) & (R.variant == "online")].sort_values("b0")
        axes[0].plot(g.bin, 100 * g.joint, marker="o", lw=2, ls=STYLE[key], color=COLOR[key], label=LABEL[key])
        axes[1].plot(g.bin, g.width, marker="o", lw=2, ls=STYLE[key], color=COLOR[key])
    axes[0].axhline(100 * (1 - a.alpha), color=V.C_MUTED, lw=1, ls=":")
    for ax in axes:
        V.style_axes(ax, equal=False); ax.set_xlabel("Seconds in the new scene", color=V.C_MUTED)
    axes[0].set_ylabel("Box holds the truth (%), target 80", color=V.C_MUTED)
    axes[1].set_ylabel("Box width (m, per axis); lower = sharper", color=V.C_MUTED)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=V.C_INK)
    fig.suptitle(f"Learning the new scene in context, all calibrated online: {Path(a.train).name} → {Path(a.test).name}, "
                 f"+{a.h:g} s", fontsize=12, weight="bold", color=V.C_INK, x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(f"media/fig_incontext_{name}.png", dpi=150)
    print(f"wrote results/incontext_{name}.csv and media/fig_incontext_{name}.png; median in-scene rows per moment "
          f"{int(np.median(n_live))}")


if __name__ == "__main__":
    main()
