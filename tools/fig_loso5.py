"""Figure for the five-scene leave-one-out test (docs/LOSO5.md): interval score per held-out scene.

    python tools/fig_loso5.py        # reads results/loso5_rows.csv, writes media/fig_loso5.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCENES = ["eth", "hotel", "zara1", "zara2", "univ"]
LABEL = {"eth": "ETH", "hotel": "Hotel", "zara1": "Zara1*", "zara2": "Zara2*", "univ": "Univ"}
MODELS = [("tabpfn", "TabPFN, 1,500 rows", "#d9480f"),
          ("lgbm", "LightGBM, same 1,500 rows", "#868e96"),
          ("lgbm_all", "LightGBM, all rows of 4 scenes (~15x more)", "#343a40"),
          ("cv_band", "Constant velocity + band", "#adb5bd")]


def main():
    d = pd.read_csv(ROOT / "results/loso5_rows.csv")
    lvl = d.groupby(["scene", "model"]).is90.mean().unstack()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(SCENES)); w = 0.2
    for j, (m, lab, c) in enumerate(MODELS):
        v = [lvl.loc[s, m] for s in SCENES]
        ax.bar(x + (j - 1.5) * w, v, w, label=lab, color=c)
        if m == "tabpfn":
            for xi, vi in zip(x + (j - 1.5) * w, v):
                ax.text(xi, vi + 0.03, f"{vi:.2f}", ha="center", fontsize=9, color=c)
    ax.set_xticks(x, [LABEL[s] for s in SCENES], fontsize=12)
    ax.set_ylabel("90% interval score at +2 s (lower is better)")
    ax.set_title("Five real pedestrian scenes, each held out once: TabPFN beats LightGBM on the same rows, "
                 "ties LightGBM trained on ~15x more", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    ax.set_ylim(0, max(lvl.loc[SCENES, "cv_band"]) * 1.25)
    fig.text(0.01, 0.01, "Mean over 3 context samples; interval score = width + 20 x miss distance. "
             "*Zara1 and Zara2 share one camera. Source: results/loso5_rows.csv (tools/loso5.py), docs/LOSO5.md.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = ROOT / "media/fig_loso5.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
