"""Publication figures from results that already exist (feature version f4). Plotting only: reads CSV files,
fits nothing, runs no simulation.

    python tools/plot_results.py            # all figures whose inputs exist
    python tools/plot_results.py 1 3        # only figures 1 and 3

Writes media/fig_f4_*.png:
    1 fig_f4_claim_a.png        results/claim_a_f4.csv: joint 5-95% coverage and interval score per model
    2 fig_f4_context_size.png   results/context_size.csv: error and coverage vs number of context rows
    3 fig_f4_t3_split.png       results/live_variants_cp.csv: coverage vs width before/after the switch
    4 fig_f4_oracle_gate.png    results/closed/*-t2c-r20-i02-p24-*.csv: time vs minimum gap per planner dial
    5 fig_f4_data_cards.png     data/{crowd_cross_t2,eth,hotel}/tracks.csv: speed, CV error, share stopped
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import glob  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from foresight.viz import C_GRID, C_INK, C_MUTED  # noqa: E402

RES, MEDIA, DATA = ROOT / "results", ROOT / "media", ROOT / "data"

# task palette
C_LGBM, C_TABPFN, C_ROBOT, C_CV, C_BAND = "#2a6fdb", "#e8590c", "#7a3fb5", "#8b949e", "#5a6575"
C_LGBM2, C_LGBM3 = "#6f9be6", "#aac5f1"          # lighter LightGBM variants (same hue)
C_TABPFN2 = "#a63c06"                             # second TabPFN series (darker orange)
C_NOMINAL = "#cfe8d4"                             # nominal coverage band
NOMINAL = (80, 90)                                # joint 5-95% box: nominal 80-90% (81% if axes independent)

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 11, "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.edgecolor": C_MUTED, "axes.labelcolor": C_INK, "xtick.color": C_INK,
    "ytick.color": C_INK, "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "errorbar.capsize": 3,
})


def finish(fig, name, source, bottom=0.07):
    """Reserve a bottom strip, write the source line there, save at slide resolution."""
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.text(0.01, 0.012, source, fontsize=9.5, color=C_MUTED, ha="left", va="bottom")
    out = MEDIA / name
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def nominal_band(ax, label=True, vertical=False):
    kw = dict(color=C_NOMINAL, alpha=0.7, lw=0, zorder=0, label="nominal 80-90%" if label else None)
    (ax.axvspan if vertical else ax.axhspan)(*NOMINAL, **kw)


def missing(fig_no, paths):
    print(f"figure {fig_no} skipped: missing {', '.join(str(Path(p).relative_to(ROOT)) for p in paths)}")


# ------------------------------------------------------------------ 1. claim A
CLAIM_MODELS = [  # key, label, colour, hatch
    ("tabpfn", "TabPFN", C_TABPFN, None),
    ("lgbm_all", "LightGBM\nall rows", C_LGBM, None),
    ("lgbm_m50n100", "LightGBM\ntuned\nm50n100", C_LGBM2, None),
    ("lgbm", "LightGBM\nsame rows", C_LGBM3, None),
    ("cv_band", "Const. vel.\n+ band", C_BAND, None),
    ("cv", "Const.\nvelocity", C_CV, None),
]


def mean_sd(d, col):
    g = d.groupby("model")[col]
    return g.mean(), g.std(ddof=1), g.count()


def fig_claim_a():
    src = RES / "claim_a_f4.csv"
    if not src.exists():
        return missing(1, [src])
    boots = sorted(glob.glob(str(RES / "claim_a_boot*.csv")))
    d = pd.read_csv(src)
    dirs = [("hotel->eth", "Hotel → ETH"), ("eth->hotel", "ETH → Hotel")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharey="row")
    x = np.arange(len(CLAIM_MODELS))
    cap = 2.8
    for j, (dkey, dlabel) in enumerate(dirs):
        dd = d[d.direction == dkey]
        for i, (col, scale, ylab) in enumerate((("joint90", 100, "joint 5-95% coverage (%)"),
                                                ("is90", 1, "interval score, 90% (m, lower is better)"))):
            ax = axes[i, j]
            ma, sa, na = mean_sd(dd[dd.group == "all"], col)
            mm, sm, nm = mean_sd(dd[dd.group == "moving"], col)
            if col == "joint90":
                nominal_band(ax, label=(j == 0))
            for k, (key, lab, c, h) in enumerate(CLAIM_MODELS):
                v, s = ma[key] * scale, (sa[key] * scale if na[key] > 1 else 0)
                vb = min(v, cap) if col == "is90" else v
                ax.bar(k, vb, 0.68, color=c, edgecolor="white", zorder=2,
                       label="all people" if (k == 0 and i == 0 and j == 0) else None)
                if na[key] > 1:
                    ax.errorbar(k, vb, yerr=s, fmt="none", ecolor=C_INK, lw=1.4, zorder=3)
                vm, smv = mm[key] * scale, (sm[key] * scale if nm[key] > 1 else 0)
                vmb = min(vm, cap) if col == "is90" else vm
                ax.errorbar(k + 0.22, vmb, yerr=smv if nm[key] > 1 else None, fmt="D", ms=7, mfc="white",
                            mec=C_INK, mew=1.5, ecolor=C_INK, lw=1.2, zorder=4,
                            label="moving people (> 0.2 m/s)" if (k == 0 and i == 0 and j == 0) else None)
                if col == "is90" and v > cap:
                    ax.annotate(f"{v:.2f}\n(mov.\n{vm:.2f})", (k - 0.08, cap), xytext=(0, -24),
                                textcoords="offset points", ha="center", va="top", fontsize=11, color="white",
                                fontweight="bold", zorder=5)
            ax.set_xticks(x, [m[1] for m in CLAIM_MODELS] if i == 1 else [""] * len(x))
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0:
                ax.set_title(dlabel)
                ax.set_ylim(0, 100)
            else:
                ax.set_ylim(0, cap)
            ax.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend([Patch(color=C_NOMINAL, alpha=0.7), Patch(facecolor="#c9ced6", edgecolor=C_INK),
                Line2D([], [], ls="none", marker="D", ms=7, mfc="white", mec=C_INK, mew=1.5)],
               ["nominal 80-90%", "all people (bar)", "moving people, > 0.2 m/s"],
               loc="upper right", ncols=3, bbox_to_anchor=(0.995, 0.995), fontsize=11.5)
    fig.suptitle("Claim A: does the box keep its promise in an unseen place? (+2 s)", x=0.01, ha="left",
                 fontsize=15)
    note = ("Source: results/claim_a_f4.csv (features f4). +2 s; bars = all people, diamonds = moving people; mean "
            "± sd over 5 context seeds of 1,500 rows for TabPFN and LightGBM tuned / same rows.\nLightGBM all rows, "
            "const. velocity and const. velocity + band are single runs (no error bar). Const. velocity interval "
            "score is off the axis (value written in the bar).")
    if not boots:
        note += " No claim_a_boot*.csv found."
    finish(fig, "fig_f4_claim_a.png", note, bottom=0.08)


# ------------------------------------------------------------------ 2. context size
CTX_SERIES = [  # model, label, colour, linestyle, marker, x-jitter factor
    ("tabpfn_fast1_local", "TabPFN-3.5-Fast, 1 estimator (local)", C_TABPFN, "-", "o", 0.95),
    ("tabpfn_v35_api", "TabPFN-3.5 default (hosted API)", C_TABPFN2, "--", "s", 1.05),
    ("lgbm_default", "LightGBM same rows, default", C_LGBM, "-", "o", 0.98),
    ("lgbm_m50n100", "LightGBM same rows, m50n100", C_LGBM2, "--", "^", 1.02),
]


def fig_context_size():
    src = RES / "context_size.csv"
    if not src.exists():
        return missing(2, [src])
    d = pd.read_csv(src)
    if "features" in d and set(d.features.dropna()) - {"f4"}:
        print("  note: context_size.csv has non-f4 rows:", set(d.features.dropna()))
    g = d.groupby(["model", "config", "n"]).agg(err=("err", "mean"), err_sd=("err", "std"),
                                               cov=("joint90", "mean"), cov_sd=("joint90", "std"),
                                               k=("seed", "count")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
    for ax, col, scale, ylab in ((axes[0], "err", 1, "mean error of the median at +2 s (m)"),
                                 (axes[1], "cov", 100, "joint 5-95% coverage (%)")):
        if col == "cov":
            nominal_band(ax)
        for model, lab, c, ls, mk, jit in CTX_SERIES:
            r = g[(g.model == model) & (g.config == "random")].sort_values("n")
            if len(r):
                ax.errorbar(r.n * jit, r[col] * scale, yerr=r[col + "_sd"] * scale, color=c, ls=ls, marker=mk,
                            ms=7, lw=2, elinewidth=1.3, label=lab, zorder=3)
            q = g[(g.model == model) & (g.config == "retrieved")]
            if len(q):
                ax.errorbar(q.n * jit, q[col] * scale, yerr=q[col + "_sd"] * scale, color=c, ls="none", marker=mk,
                            ms=10, mfc="white", mew=1.8, elinewidth=1.0, zorder=4)
        la = d[d.model == "lgbm_all"]
        if len(la):
            ax.axhline(la[col if col == "err" else "joint90"].mean() * scale, color=C_LGBM, ls=":", lw=2,
                       label=f"LightGBM all rows ({int(la.n.iloc[0]):,})", zorder=2)
        cv = d[d.model == "cv"]
        if len(cv) and col == "err":
            ax.axhline(cv.err.mean(), color=C_CV, ls=":", lw=2.2, label="Constant velocity", zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel("context rows (log scale)")
        ax.set_ylabel(ylab)
        ax.set_xticks([500, 1000, 2000, 5000, 10000, 20000], ["500", "1k", "2k", "5k", "10k", "20k"])
        ax.minorticks_off()
    axes[0].set_title("Error vs context size")
    axes[1].set_title("Box coverage vs context size")
    axes[1].set_ylim(40, 100)
    axes[1].plot([], [], ls="none", marker="o", ms=10, mfc="white", mec=C_INK, mew=1.8,
                 label="hollow = retrieved (nearest-neighbour) context")
    h, lab = axes[0].get_legend_handles_labels()
    h2, lab2 = axes[1].get_legend_handles_labels()
    extra = [(a, b) for a, b in zip(h2, lab2) if b not in lab]
    fig.legend(h + [e[0] for e in extra], lab + [e[1] for e in extra], loc="lower center", ncols=3,
               bbox_to_anchor=(0.5, 0.055), fontsize=11)
    fig.suptitle("How many context rows does TabPFN need? (crossing crowd, +2 s)", x=0.01, ha="left", fontsize=15)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.text(0.01, 0.012, "Source: results/context_size.csv (features f4). 2,000 fixed test rows; mean ± sd over "
             "3 context seeds; constant velocity (0% coverage, no box) and LightGBM all rows are single runs.",
             fontsize=9.5, color=C_MUTED, ha="left", va="bottom")
    out = MEDIA / "fig_f4_context_size.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


# ------------------------------------------------------------------ 3. T3 split
T3_BASES = [  # variant, label, colour, marker
    ("frozen2k", "TabPFN frozen, 2k rows", C_TABPFN, "o"),
    ("split2k", "TabPFN split, 2k + this scene", C_TABPFN2, "^"),
    ("lgbm_all_frozen", "LightGBM, all bank rows", C_LGBM, "s"),
]


def person_boot(df, cols, B=2000, seed=0):
    """Ratio-of-sums means of cols and 95% intervals, resampling persons (episode:agent_id) with replacement;
    the same resample indices for every column (as tools/live_variants.py boot())."""
    g = df.groupby("person")[cols].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (B, len(g)))
    out = {}
    for c in cols:
        s, n = g[(c, "sum")].to_numpy()[idx].sum(1), g[(c, "count")].to_numpy()[idx].sum(1)
        st = s / n
        out[c] = (df[c].mean(), np.percentile(st, 2.5), np.percentile(st, 97.5))
    return out


def fig_t3_split():
    src = RES / "live_variants_cp.csv"
    if not src.exists():
        return missing(3, [src])
    d = pd.read_csv(src)
    if "person" not in d:
        d["person"] = d.episode.astype(str) + ":" + d.agent_id.astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), sharey=True)
    periods = [("before the switch (calm)", d.since < 0), ("after the switch (erratic)", d.since >= 0)]
    for ax, (pname, mask) in zip(axes, periods):
        nominal_band(ax, label=(ax is axes[0]))
        dp = d[mask]
        for base, lab, c, mk in T3_BASES:
            pts = []
            for v, filled in ((base, False), (base + "+offset", True)):
                dv = dp[dp.variant == v]
                b = person_boot(dv, ["width", "joint90"])
                (w, wl, wh), (cv_, cl, ch) = b["width"], b["joint90"]
                pts.append((w, cv_ * 100))
                ax.errorbar(w, cv_ * 100, xerr=[[w - wl], [wh - w]], yerr=[[100 * (cv_ - cl)], [100 * (ch - cv_)]],
                            fmt=mk, ms=11, color=c, mfc=c if filled else "white", mec=c, mew=2, elinewidth=1.4,
                            zorder=4, label=f"{lab}{' + online offset' if filled else ''}")
            ax.annotate("", xy=pts[1], xytext=pts[0], zorder=3,
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.4, shrinkA=8, shrinkB=8))
        ax.set_title(f"{pname}: {dp.person.nunique()} people, {int((dp.variant == 'frozen2k').sum())} rows")
        ax.set_xlabel("mean width of the 5-95% box (m)")
    axes[0].set_ylabel("joint 5-95% coverage (%)")
    axes[0].set_ylim(30, 100)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncols=3, bbox_to_anchor=(0.5, 0.075), fontsize=11)
    fig.suptitle("Is split's gain just wider boxes? Coverage vs width at +2 s (Setting B)", x=0.01, ha="left",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.text(0.01, 0.012, "Source: results/live_variants_cp.csv (features f4; the file does not record the version). "
             "Hollow = raw quantiles, filled = with online conformal offset (arrow).\nBars: 95% person bootstrap "
             "(episode:agent_id, 2,000 resamples).", fontsize=9.5, color=C_MUTED, ha="left", va="bottom")
    out = MEDIA / "fig_f4_t3_split.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


# ------------------------------------------------------------------ 4. oracle gate
GATE_TAG = "t2c-r20-i02-p24"
GATE_SEEDS = set(range(4100, 4120))
TIME_FLOOR = 17.74   # s, mean straight-line time to goal over the 20 seeds at 1.0 m/s (RESULTS.md, D-33)
PLANNERS = [  # planner, label, colour, marker, dial column, dial prefix
    ("cv", "Constant velocity", C_CV, "o", "margin", "m"),
    ("reactive", "Reactive", C_ROBOT, "v", "margin", "m"),
    ("vo", "Velocity obstacles", C_LGBM, "s", "vo_w", "w"),
    ("stop_go", "Stop-and-go", C_BAND, "X", None, ""),
    ("oracle", "Oracle (true future)", C_TABPFN, "D", "margin", "m"),
]


def seed_boot(x, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    st = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.percentile(st, 2.5), np.percentile(st, 97.5)


def pareto(points):
    """Non-dominated points for (minimise time, maximise gap), sorted by time."""
    front, best = [], -np.inf
    for t, gp in sorted(points):
        if gp > best:
            front.append((t, gp))
            best = gp
    return front


def load_gate():
    files = [f for f in sorted(glob.glob(str(RES / "closed" / f"*-{GATE_TAG}-*.csv")))
             if not Path(f).name.startswith("robotfeat_")]
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[(d.robot_repulsion == 2.0) & np.isclose(d.ignore_robot, 0.2) & (d.people == 24) & d.seed.isin(GATE_SEEDS)]
    d["dial"] = np.where(d.planner == "vo", d.vo_w, np.where(d.planner == "stop_go", np.nan, d.margin))
    n = d.groupby(["planner", "dial"], dropna=False).seed.nunique()
    if (n != 20).any():
        print("  WARNING: arms without 20 seeds:", n[n != 20].to_dict())
    return d


def fig_oracle_gate():
    d = load_gate()
    if d is None:
        return missing(4, [RES / "closed" / f"*-{GATE_TAG}-*.csv"])
    pts_file = RES / "oracle_gate_cross_points.csv"
    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.axvline(TIME_FLOOR, color=C_INK, ls="--", lw=1.4, zorder=1)
    ax.text(TIME_FLOOR + 0.12, 0.47, f"time floor {TIME_FLOOR:.2f} s\n(straight line at 1 m/s)", fontsize=11,
            color=C_INK, va="top")
    ax.axhline(0, color=C_MUTED, lw=1, zorder=1)
    summ, check = {}, []
    for planner, lab, c, mk, dial_col, pre in PLANNERS:
        dp = d[d.planner == planner]
        rows = []
        for dial, g in dp.groupby("dial", dropna=False):
            t, tl, th = seed_boot(g.time_capped.to_numpy())
            gp, gl, gh = seed_boot(g.min_gap.to_numpy())
            rows.append((dial, t, tl, th, gp, gl, gh))
            arm = f"{planner}:{'-' if pd.isna(dial) else f'{dial:g}'}"
            check.append((arm, t, gp))
        rows.sort(key=lambda r: (np.nan_to_num(r[0])))
        summ[planner] = rows
        R = np.array([r[1:] for r in rows])
        ax.plot(R[:, 0], R[:, 3], color=c, lw=2.6 if planner == "oracle" else 1.8, zorder=3)
        ax.errorbar(R[:, 0], R[:, 3], xerr=[R[:, 0] - R[:, 1], R[:, 2] - R[:, 0]],
                    yerr=[R[:, 3] - R[:, 4], R[:, 5] - R[:, 3]], fmt=mk, ms=9, color=c, mec="white", mew=1,
                    elinewidth=1.0, alpha=0.75, zorder=4, label=lab)
        for r in rows:
            if not pd.isna(r[0]):
                ax.annotate(f"{pre}{r[0]:g}", (r[1], r[4]), xytext=(7, 5), textcoords="offset points",
                            fontsize=10.5, color=c, zorder=6)
    base = [(r[1], r[4]) for p in ("cv", "reactive", "vo", "stop_go") for r in summ.get(p, [])]
    fb = np.array(pareto(base))
    ax.step(fb[:, 0], fb[:, 1], where="post", color=C_INK, lw=1.2, ls=(0, (4, 3)), zorder=2,
            label="Pareto front, baseline planners")
    ax.scatter(fb[:, 0], fb[:, 1], s=260, facecolors="none", edgecolors=C_INK, lw=1.4, zorder=5)
    fo = np.array(pareto([(r[1], r[4]) for r in summ.get("oracle", [])]))
    ax.scatter(fo[:, 0], fo[:, 1], s=260, facecolors="none", edgecolors=C_TABPFN, lw=1.6, zorder=5,
               label="Pareto front, oracle")
    if pts_file.exists():
        P = pd.read_csv(pts_file).set_index("arm")
        bad = [a for a, t, gp in check if a in P.index and
               (abs(P.loc[a, "time_capped"] - t) > 1e-6 or abs(P.loc[a, "min_gap"] - gp) > 1e-6)]
        print("  per-seed means match oracle_gate_cross_points.csv" if not bad else f"  MISMATCH vs points file: {bad}")
    ax.set_xlabel("time to goal, capped (s, lower is better)")
    ax.set_ylabel("minimum gap to a person (m, higher is safer)")
    ax.set_xlim(17, 33.5)
    ax.legend(loc="lower right", ncols=2)
    ax.set_title("Oracle gate: speed vs safety per planner dial (24 people, crossing, 20 seeds)", loc="left")
    finish(fig, "fig_f4_oracle_gate.png",
           "Source: results/closed/*-t2c-r20-i02-p24-*.csv (features f4; robot repulsion 2, ignore robot 0.2, 24 people, "
           "seeds 4100-4119).\nMean with 95% seed bootstrap (2,000 resamples). Dial: margin m (m) or VO weight w. "
           "Time floor from RESULTS.md (D-33).", bottom=0.08)


# ------------------------------------------------------------------ 5. data cards
DC_SETS = [("crowd_cross_t2", "crowd_cross_t2 (sim)", "#2a78d6"), ("eth", "ETH (real)", "#1baf7a"),
           ("hotel", "Hotel (real)", "#eda100")]   # colours as tools/data_cards.py (crowd, eth, hotel)


def quick_card(name):
    """Speed, share stopped and constant-velocity error at +2 s, with the definitions of tools/data_cards.py
    card() (same load(), constants, sim 'drop step 0' rule and self-join), without its slow per-frame parts."""
    import data_cards as DC
    tracks, _, _, dt = DC.load(name)
    is_sim = tracks["kind"].ne("person").any() or "goal_x" in tracks.columns
    ppl = tracks[tracks["kind"] == "person"].copy()
    ppl["speed"] = np.hypot(ppl["vx"], ppl["vy"])
    sp = ppl if not is_sim else ppl[ppl["step"] > 0]
    s = sp["speed"].to_numpy()
    c = {"name": name, "_speed": s, "speed_mean": float(s.mean()), "stopped_share": float((s < DC.STOP_SPEED).mean())}
    H = int(round(DC.CV_H / dt))
    base = ppl if not is_sim else ppl[ppl["step"] > 0]
    fut = ppl[["episode", "agent_id", "step", "x", "y", "vx", "vy"]].rename(
        columns={"x": "fx", "y": "fy", "vx": "fvx", "vy": "fvy"})
    fut["step"] -= H
    j = base.merge(fut, on=["episode", "agent_id", "step"], how="inner")
    err = np.hypot(j["fx"] - (j["x"] + j["vx"] * DC.CV_H), j["fy"] - (j["y"] + j["vy"] * DC.CV_H))
    c["_cv"], c["cv_mean"], c["cv_p90"] = err.to_numpy(), float(err.mean()), DC.q(err, 90)
    if is_sim:
        c["stopped_by_regime"] = {r: float((sp["speed"][sp["regime"] == r] < DC.STOP_SPEED).mean())
                                  for r in sorted(sp["regime"].unique())}
    return c


def fig_data_cards():
    need = [DATA / n / "tracks.csv" for n, _, _ in DC_SETS]
    if not need[0].exists():
        return missing(5, [need[0]])
    sets = [(n, lab, col) for (n, lab, col), p in zip(DC_SETS, need) if p.exists()]
    cards = {n: quick_card(n) for n, _, _ in sets}
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    import data_cards as DC
    a = axes[0]
    bins = np.linspace(0, 2.6, 53)
    for n, lab, col in sets:
        c = cards[n]
        a.hist(np.clip(c["_speed"], 0, 2.6), bins=bins, density=True, histtype="step", lw=2.2, color=col,
               label=f"{lab}: mean {c['speed_mean']:.2f}")
    a.axvline(DC.STOP_SPEED, color=C_INK, lw=1, ls="--")
    a.text(DC.STOP_SPEED + 0.04, 0.97, "stopped < 0.2", transform=a.get_xaxis_transform(), fontsize=11, va="top")
    a.set(title="Person speed", xlabel="speed (m/s), clipped at 2.6", ylabel="density")
    a.legend(loc="center right", fontsize=10.5)
    a = axes[1]
    bins = np.linspace(0, 4, 41)
    for n, lab, col in sets:
        c = cards[n]
        a.hist(np.clip(c["_cv"], 0, 4), bins=bins, density=True, histtype="step", lw=2.2, color=col,
               label=f"{lab}: mean {c['cv_mean']:.2f}, p90 {c['cv_p90']:.2f}")
    a.set(title="Constant-velocity error at +2 s", xlabel="position error (m), clipped at 4", ylabel="density")
    a.legend(loc="upper right", fontsize=10.5)
    a = axes[2]
    labels, vals, cols = [], [], []
    for n, lab, col in sets:
        c = cards[n]
        labels.append(lab.split(" (")[0])
        vals.append(c["stopped_share"] * 100)
        cols.append(col)
        for reg, v in c.get("stopped_by_regime", {}).items():
            labels.append(f"  {reg}")
            vals.append(v * 100)
            cols.append(col + "88")
    y = np.arange(len(labels))[::-1]
    a.barh(y, vals, color=cols, height=0.7)
    for yy, v in zip(y, vals):
        a.text(v + 0.8, yy, f"{v:.1f}%" if v < 1 else f"{v:.0f}%", va="center", fontsize=11)
    a.set_yticks(y, labels)
    a.set(title="Share of time stopped", xlabel="rows with speed < 0.2 m/s (%)", xlim=(0, max(vals) * 1.25))
    a.grid(axis="y", visible=False)
    fig.suptitle("Data cards: the new crossing crowd vs the real ETH and Hotel scenes", x=0.01, ha="left",
                 fontsize=15)
    finish(fig, "fig_f4_data_cards.png",
           "Source: data/crowd_cross_t2, data/eth, data/hotel tracks.csv (ground truth; definitions of "
           "tools/data_cards.py; feature version f4 data folders). Descriptive, no seeds, so no error bars. "
           "Lighter bars: crowd_cross_t2 by regime.", bottom=0.06)


FIGS = {1: fig_claim_a, 2: fig_context_size, 3: fig_t3_split, 4: fig_oracle_gate, 5: fig_data_cards}

if __name__ == "__main__":
    which = [int(a) for a in sys.argv[1:]] or list(FIGS)
    MEDIA.mkdir(exist_ok=True)
    for k in which:
        FIGS[k]()
