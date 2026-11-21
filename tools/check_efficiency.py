"""Robustness check for the data-efficiency curve (LightGBM only: seconds, not minutes).

Other place: context ETH, test Hotel. Same place: Hotel episodes split in two.
5 seeds per size (different context sample and test sample each seed), error at +2 s.

    python tools/check_efficiency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from run_offline import rows_for  # noqa: E402

HZ = (1.2, 2.0, 4.8)
SIZES = [50, 100, 250, 500, 1000, 2000, 4000]


def err(model, te):
    Q = model.predict_quantiles(F.X_of(te), (0.1, 0.5, 0.9))
    return float(np.linalg.norm(Q[:, :, 1] - te[["ddx", "ddy"]].to_numpy(), axis=1).mean())


def curve(tr, te, seeds=range(5), n_test=600):
    out = {}
    for n in SIZES:
        if n > len(tr):
            continue
        e = []
        for s in seeds:
            ctx = tr.sample(n, random_state=s); t = te.sample(min(n_test, len(te)), random_state=100 + s)
            e.append(err(M.make("lgbm").fit(F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy()), t))
        out[n] = (np.mean(e), np.std(e))
    return out


def main():
    eth = rows_for("data/eth", 0.0, horizons=HZ); hotel = rows_for("data/hotel", 0.0, horizons=HZ)
    eth, hotel = eth[np.isclose(eth.h, 2.0)], hotel[np.isclose(hotel.h, 2.0)]
    tr_h, _, te_h = F.split_by_episode(hotel, test_frac=0.4)
    curves = {}
    for name, (tr, te) in {"Rows from another place (ETH, test Hotel)": (eth, hotel),
                           "Rows from the same place (Hotel, other episodes)": (hotel[tr_h], hotel[te_h])}.items():
        print(f"{name}: {len(tr)} rows available")
        curves[name] = curve(tr, te)
        for n, (m, s) in curves[name].items():
            print(f"  n={n:>5}  LightGBM error {m:.3f} m  (sd {s:.3f} over 5 seeds)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from foresight import viz as V
    fig, ax = plt.subplots(figsize=(6.6, 4.4), facecolor="white"); V.style_axes(ax, equal=False)
    for (name, c), col in zip(curves.items(), ("#8b949e", "#2a6fdb")):
        xs = list(c); m = np.array([c[n][0] for n in xs]); s = np.array([c[n][1] for n in xs])
        ax.plot(xs, m, marker="o", lw=2, color=col, label=name); ax.fill_between(xs, m - s, m + s, color=col, alpha=0.15)
    ax.set_xscale("log"); ax.set_xlabel("Example rows (log scale)", color=V.C_MUTED)
    ax.set_ylabel("Mean error at +2 s (m)", color=V.C_MUTED)
    ax.set_title("Rows from this place are worth more", fontsize=12, weight="bold", color=V.C_INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=V.C_INK)
    fig.tight_layout(); fig.savefig("media/fig_rows_place.png", dpi=150); print("wrote media/fig_rows_place.png")


if __name__ == "__main__":
    main()
