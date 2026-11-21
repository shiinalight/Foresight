"""Item 4 / S5: conformal calibration of the box, for every model.

    python tools/conformal_eval.py --train data/hotel --test data/eth --noise 0
    python tools/conformal_eval.py --train data/eth --test data/hotel --noise 0

Per model (constant velocity + band, LightGBM on 1,500 rows, LightGBM on all rows, TabPFN):
  raw     the model's own 10-90% box (per axis)
  split   widened by an offset from a held-out 30% of the TRAINING scene's episodes (what a robot
          has on arrival; no test-scene data)
  online  starts from the split offset, then adapts from rows of the new scene whose outcome is
          known (adaptive conformal, causal)
Target: the 2D box holds the truth 80% of the time (alpha = 0.2), jointly over both axes.
Reports joint coverage and box width overall and by seconds in the scene.
Writes results/conformal_<train>_to_<test>.csv and media/fig_conformal_<train>_to_<test>.png.
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
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from run_offline import rows_for  # noqa: E402

QS = (0.1, 0.5, 0.9)
BINS = [0, 5, 10, 20, 30, 45, 61]
LABEL = dict(V.MODEL_LABEL, lgbm="LightGBM (1,500 rows)", lgbm_all="LightGBM (all rows)")
COLOR = dict(V.MODEL_COLOR, lgbm_all="#5b8def")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--h", type=float, default=2.0)
    ap.add_argument("--context", type=int, default=1500)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--models", nargs="*", default=["cv_band", "lgbm", "lgbm_all", "tabpfn"])
    ap.add_argument("--api", action="store_true", help="TabPFN through the hosted API (tabpfn-client)")
    a = ap.parse_args()
    hz = (1.2, 2.0, 4.8) if any(s in a.train for s in ("eth", "hotel")) else (1.0, 2.0)
    src = rows_for(a.train, a.noise, horizons=hz); dst = rows_for(a.test, a.noise, horizons=hz)
    src, dst = src[np.isclose(src.h, a.h)], dst[np.isclose(dst.h, a.h)].sort_values(["episode", "t"])
    fit_m, _, cal_m = F.split_by_episode(src, test_frac=0.3, seed=1)
    fit, cal = src[fit_m], src[cal_m]
    Yc, Yt = cal[["ddx", "ddy"]].to_numpy(), dst[["ddx", "ddy"]].to_numpy()
    sec = dst.t.to_numpy()
    summary, binned = [], []
    for key in a.models:
        t0 = time.perf_counter()
        rows = fit if key in ("cv_band", "lgbm_all") else fit.sample(min(a.context, len(fit)), random_state=0)
        kind = "lgbm" if key == "lgbm_all" else key
        m = M.make(kind, **({"api": True} if (kind == "tabpfn" and a.api) else {})).fit(F.X_of(rows), rows[["ddx", "ddy"]].to_numpy())
        Qc, Qt = m.predict_quantiles(F.X_of(cal), QS), m.predict_quantiles(F.X_of(dst), QS)
        lo_c, hi_c, lo_t, hi_t = Qc[:, :, 0], Qc[:, :, 2], Qt[:, :, 0], Qt[:, :, 2]
        d_split = CF.split_offset(lo_c, hi_c, Yc, a.alpha)
        d_on, miss_on = CF.online_offsets(dst.t.to_numpy(), dst.t_known.to_numpy(), dst.episode.to_numpy(),
                                          lo_t, hi_t, Yt, d_split, a.alpha, a.eta)
        variants = {"raw": np.zeros(len(dst)), "split": np.full(len(dst), d_split), "online": d_on}
        for vname, d in variants.items():
            lo, hi = CF.widen(lo_t, hi_t, d)
            inside = ((Yt >= lo) & (Yt <= hi)).all(axis=1)
            width = (hi - lo).mean(axis=1)
            summary.append(dict(model=key, variant=vname, joint=inside.mean(), width=width.mean(),
                                d_split=d_split, late_joint=inside[sec >= 20].mean(), late_width=width[sec >= 20].mean()))
            b = pd.cut(sec, BINS, right=False)
            for (bin_, g_in), (_, g_w) in zip(pd.Series(inside).groupby(b, observed=True),
                                              pd.Series(width).groupby(b, observed=True)):
                binned.append(dict(model=key, variant=vname, bin=str(bin_), joint=g_in.mean(), width=g_w.mean(), n=len(g_in)))
        print(f"{key:<9} split offset {d_split:+.3f} m  ({time.perf_counter() - t0:.0f} s)", flush=True)
    S = pd.DataFrame(summary); B = pd.DataFrame(binned)
    name = f"{Path(a.train).name}_to_{Path(a.test).name}"
    S.to_csv(f"results/conformal_{name}.csv", index=False); B.to_csv(f"results/conformal_{name}_bins.csv", index=False)
    print(S.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor="white")
    for key in a.models:
        for vname, ls in (("split", ":"), ("online", "-")):
            g = B[(B.model == key) & (B.variant == vname)]
            axes[0].plot(g.bin, 100 * g.joint, ls=ls, marker="o" if vname == "online" else None, lw=2,
                         color=COLOR[key], label=f"{LABEL[key]}" if vname == "online" else None)
            axes[1].plot(g.bin, g.width, ls=ls, marker="o" if vname == "online" else None, lw=2, color=COLOR[key])
    axes[0].axhline(100 * (1 - a.alpha), color=V.C_MUTED, lw=1, ls="--")
    for ax in axes:
        V.style_axes(ax, equal=False); ax.set_xlabel("Seconds in the new scene", color=V.C_MUTED)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("Box holds the truth (%), target 80", color=V.C_MUTED)
    axes[1].set_ylabel("Box width (m, per axis)", color=V.C_MUTED)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=V.C_INK)
    fig.suptitle(f"Calibrated on arrival (dotted) and adapting in the scene (solid): {Path(a.train).name} → "
                 f"{Path(a.test).name}, +{a.h:g} s", fontsize=12, weight="bold", color=V.C_INK, x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(f"media/fig_conformal_{name}.png", dpi=150)
    print(f"wrote results/conformal_{name}.csv and media/fig_conformal_{name}.png")


if __name__ == "__main__":
    main()
