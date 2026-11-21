"""Baseline evaluation on real pedestrians: context from one scene, test on the other.

    python evaluate.py                       # ETH->Hotel and Hotel->ETH, all models, all figures
    python evaluate.py --skip-efficiency     # faster

Models (same features, same rows for the 1,500-row ones):
  cv          constant velocity (zero residual)
  cv_band     constant velocity + fixed band (training residual quantiles per horizon)
  lgbm        LightGBM quantile regression on the same 1,500 rows TabPFN sees
  lgbm_all    LightGBM on every training row of the source scene (its best case)
  tabpfn      TabPFN on 1,500 random rows of the source scene
Writes results/eval.json, results/eval_table.md and figures in media/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from foresight import viz as V
from foresight.core import features as F
from foresight.core import models as M
from run_offline import rows_for

QS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
CENTRAL = [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.05, 0.95)]     # 20..90% intervals
HZ = (1.2, 2.0, 4.8)
PAIRS = [("eth", "hotel"), ("hotel", "eth")]
NAME = {"eth": "ETH", "hotel": "Hotel"}
COLOR = dict(V.MODEL_COLOR, lgbm_all="#2a6fdb")
LABEL = dict(V.MODEL_LABEL, lgbm="LightGBM (1,500 rows)", lgbm_all="LightGBM (all rows)")
STYLE = {"lgbm_all": "--"}


def scores(Q, Y):
    j = QS.index(0.5)
    err = np.linalg.norm(Q[:, :, j] - Y, axis=1)
    out = {"err_mean": float(err.mean()), "err_median": float(np.median(err))}
    for lo, hi in CENTRAL:
        a, b = QS.index(lo), QS.index(hi)
        out[f"cov{round(100 * (hi - lo))}"] = float(((Y >= Q[:, :, a]) & (Y <= Q[:, :, b])).mean())
        out[f"width{round(100 * (hi - lo))}"] = float((Q[:, :, b] - Q[:, :, a]).mean())
    return out


def run_pair(src, dst, n_ctx, n_test, seed=0):
    tr = rows_for(f"data/{src}", 0.0, horizons=HZ)
    te = rows_for(f"data/{dst}", 0.0, horizons=HZ)
    te = te.sample(min(n_test, len(te)), random_state=seed)
    ctx = tr.sample(min(n_ctx, len(tr)), random_state=seed)
    Xte, Yte = F.X_of(te), te[["ddx", "ddy"]].to_numpy()
    fits = {"cv": (ctx, "cv"), "cv_band": (tr, "cv_band"), "lgbm": (ctx, "lgbm"), "lgbm_all": (tr, "lgbm"),
            "tabpfn": (ctx, "tabpfn")}
    res, timing = {}, {}
    for key, (rows, kind) in fits.items():
        t0 = time.perf_counter()
        m = M.make(kind).fit(F.X_of(rows), rows[["ddx", "ddy"]].to_numpy())
        Q = m.predict_quantiles(Xte, QS)
        timing[key] = time.perf_counter() - t0
        res[key] = {}
        for h in HZ:
            msk = np.isclose(te["h"].to_numpy(), h)
            res[key][f"{h:g}"] = scores(Q[msk], Yte[msk]) | {"n": int(msk.sum()), "train_rows": len(rows)}
        print(f"{NAME[src]}->{NAME[dst]} {key:<9} rows {len(rows):>6}  "
              + "  ".join(f"h{h:g} {res[key][f'{h:g}']['err_mean']:.3f}m/{100 * res[key][f'{h:g}']['cov80']:.0f}%" for h in HZ)
              + f"  ({timing[key]:.0f} s)")
    return res, timing


def efficiency(src, dst, sizes, n_test, h=2.0, seed=0):
    tr = rows_for(f"data/{src}", 0.0, horizons=HZ)
    te = rows_for(f"data/{dst}", 0.0, horizons=HZ)
    tr, te = tr[np.isclose(tr.h, h)], te[np.isclose(te.h, h)]
    te = te.sample(min(n_test, len(te)), random_state=seed)
    Xte, Yte = F.X_of(te), te[["ddx", "ddy"]].to_numpy()
    out = {"cv": scores(np.zeros((len(te), 2, len(QS))), Yte)["err_mean"], "tabpfn": {}, "lgbm": {}}
    for n in sizes:
        if n > len(tr):
            continue
        ctx = tr.sample(n, random_state=seed)
        for key in ("tabpfn", "lgbm"):
            m = M.make(key).fit(F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy())
            out[key][n] = scores(m.predict_quantiles(Xte, QS), Yte)["err_mean"]
        print(f"efficiency {NAME[src]}->{NAME[dst]} n={n:>5}  TabPFN {out['tabpfn'][n]:.3f}  LightGBM {out['lgbm'][n]:.3f}  CV {out['cv']:.3f}")
    return out


def fig_error(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor="white")
    for ax, (src, dst) in zip(axes, PAIRS):
        r = res[f"{src}->{dst}"]
        V.style_axes(ax, equal=False)
        for key in ("cv", "cv_band", "lgbm", "lgbm_all", "tabpfn"):
            if key == "cv_band":
                continue                      # same median as cv
            y = [r[key][f"{h:g}"]["err_mean"] for h in HZ]
            ax.plot(HZ, y, marker="o", ms=6, lw=2, color=COLOR[key], ls=STYLE.get(key, "-"), label=LABEL[key])
        ax.set_title(f"Context {NAME[src]} → test {NAME[dst]}", fontsize=11, color=V.C_INK)
        ax.set_xlabel("Horizon (s)", color=V.C_MUTED); ax.set_xticks(HZ)
    axes[0].set_ylabel("Mean position error (m)", color=V.C_MUTED)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=V.C_INK)
    fig.suptitle("Prediction error, lower is better", fontsize=13, weight="bold", color=V.C_INK, x=0.06, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_calibration(res, path, h="2"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, facecolor="white")
    nominal = [round(100 * (hi - lo)) for lo, hi in CENTRAL]
    for ax, (src, dst) in zip(axes, PAIRS):
        r = res[f"{src}->{dst}"]
        V.style_axes(ax, equal=False)
        ax.plot([0, 100], [0, 100], color=V.C_MUTED, lw=1, ls=":", label="Perfect calibration")
        for key in ("cv_band", "lgbm", "lgbm_all", "tabpfn"):
            y = [100 * r[key][h][f"cov{n}"] for n in nominal]
            ax.plot(nominal, y, marker="o", ms=6, lw=2, color=COLOR[key], ls=STYLE.get(key, "-"), label=LABEL[key])
        ax.set_xlim(10, 100); ax.set_ylim(0, 100)
        ax.set_title(f"Context {NAME[src]} → test {NAME[dst]}, +{h} s", fontsize=11, color=V.C_INK)
        ax.set_xlabel("Promised coverage of the range (%)", color=V.C_MUTED)
    axes[0].set_ylabel("Actual coverage (%)", color=V.C_MUTED)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=V.C_INK, loc="lower right")
    fig.suptitle("Are the ranges honest? On the dotted line = yes", fontsize=13, weight="bold", color=V.C_INK, x=0.06, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_efficiency(eff, path, src="eth", dst="hotel"):
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor="white")
    V.style_axes(ax, equal=False)
    for key in ("lgbm", "tabpfn"):
        xs = sorted(int(n) for n in eff[key]); ys = [eff[key][n] if n in eff[key] else eff[key][str(n)] for n in xs]
        ax.plot(xs, ys, marker="o", ms=6, lw=2, color=COLOR[key], label=V.MODEL_LABEL[key])
    ax.axhline(eff["cv"], color=COLOR["cv"], lw=1.5, ls="--", label="Const. velocity")
    ax.set_xscale("log"); ax.set_xlabel("Example rows the model sees (log scale)", color=V.C_MUTED)
    ax.set_ylabel("Mean position error at +2 s (m)", color=V.C_MUTED)
    ax.set_title(f"How many examples does it need? ({NAME[src]} → {NAME[dst]})", fontsize=12, weight="bold",
                 color=V.C_INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=V.C_INK)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def table(res, timing):
    lines = ["| Direction | Model | Rows seen | " + " | ".join(f"Error +{h:g} s" for h in HZ)
             + " | 80% range holds truth (+2 s) | 80% range width (+2 s) | Time (s) |",
             "|---|---|---|" + "---|" * len(HZ) + "---|---|---|"]
    for pair, r in res.items():
        a, b = pair.split("->")
        for key in ("cv", "cv_band", "lgbm", "lgbm_all", "tabpfn"):
            row = r[key]
            cov = "—" if key == "cv" else f"{100 * row['2']['cov80']:.0f}%"
            wid = "—" if key == "cv" else f"{row['2']['width80']:.2f} m"
            lines.append(f"| {NAME[a]} → {NAME[b]} | {LABEL[key]} | {row['2']['train_rows']:,} | "
                         + " | ".join(f"{row[f'{h:g}']['err_mean']:.3f} m" for h in HZ)
                         + f" | {cov} | {wid} | {timing[pair][key]:.0f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=1500)
    ap.add_argument("--test-rows", type=int, default=900)
    ap.add_argument("--skip-efficiency", action="store_true")
    a = ap.parse_args()
    Path("results").mkdir(exist_ok=True); Path("media").mkdir(exist_ok=True)
    res, timing = {}, {}
    for src, dst in PAIRS:
        res[f"{src}->{dst}"], timing[f"{src}->{dst}"] = run_pair(src, dst, a.context, a.test_rows)
    fig_error(res, "media/fig_error_by_horizon.png")
    fig_calibration(res, "media/fig_calibration.png")
    tbl = table(res, timing)
    Path("results/eval_table.md").write_text(tbl, encoding="utf-8")
    out = {"results": res, "timing_s": timing, "context": a.context, "test_rows": a.test_rows, "horizons": HZ}
    if not a.skip_efficiency:
        eff = efficiency("eth", "hotel", [50, 100, 250, 500, 1000, 2000, 4000], 400)
        fig_efficiency(eff, "media/fig_data_efficiency.png")
        out["efficiency_eth_hotel"] = {k: (v if not isinstance(v, dict) else {str(n): e for n, e in v.items()})
                                       for k, v in eff.items()}
    Path("results/eval.json").write_text(json.dumps(out, indent=1))
    print(tbl)


if __name__ == "__main__":
    main()
