"""How TabPFN's accuracy depends on how many and which context rows it reads, against LightGBM (crossing crowd, +2 s).

    python tools/context_size.py --part local                      # LightGBM (all configs) + TabPFN fast-1 locally
    python tools/context_size.py --part api --not-before 00:05     # TabPFN-3.5 default on the hosted API
    python tools/context_size.py --plot                            # figure only, from the CSV

Data: data/crowd_cross rows (current features), h == 2.0, split by episode (test_frac 0.25, seed 0),
2,000 fixed test rows (random_state 0), as in tools/s0_gate.py.
Contexts: random n rows from train (seeds 0, 1, 2); LightGBM on all train rows; retrieved context = union of each
test row's nearest train rows (context.Bank, which subsamples train to 20k rows with the given seed), smallest k
whose union reaches the target, cut to exactly the target (nearest ranks first).
Models (labelled; different models): tabpfn_v35_api = hosted TabPFN default (v3.5, 8 estimators);
tabpfn_fast1_local = TabPFN-3.5-Fast, 1 estimator, local CPU; lgbm_default; lgbm_m50n100 (min_child 50, 100 trees).
Writes results/context_size.csv (merged per cell, re-runs replace) and media/fig_context_size.png.
"""
import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "2")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from foresight.core import context as C  # noqa: E402
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from run_offline import rows_for  # noqa: E402

QS = (0.05, 0.1, 0.5, 0.9, 0.95)
SEEDS = (0, 1, 2)
CSV = Path("results/context_size.csv")
FIG = Path("media/fig_context_size.png")
USAGE = Path("results/api_usage.csv")
CREDITS_PER_CALL = 10_000      # D-17; the server's estimate_cost gives 10,000 for every size used here
CAP = 1_000_000
KEY = ["model", "config", "n", "seed"]


def load():
    df = rows_for("data/crowd_cross", 0.1, horizons=(1.0, 2.0))
    df = df[np.isclose(df.h, 2.0)]
    tr, _, te = F.split_by_episode(df, test_frac=0.25, seed=0)
    train, test = df[tr], df[te].sample(min(2000, int(te.sum())), random_state=0)
    return train, test


def score(Q, Y):
    Q = np.sort(Q, axis=2)
    j = QS.index(0.5)
    err = float(np.linalg.norm(Q[:, :, j] - Y, axis=1).mean())
    pin = 0.0
    for k, q in enumerate(QS):
        d = Y - Q[:, :, k]
        pin += np.mean(np.maximum(q * d, (q - 1) * d))
    lo, hi = Q[:, :, 0], Q[:, :, -1]
    joint = float(((Y >= lo) & (Y <= hi)).all(axis=1).mean())
    w = (hi - lo).mean(0)
    return dict(err=err, pinball=float(pin / len(QS)), joint90=joint, width_along=float(w[0]), width_across=float(w[1]))


def save(row):
    R = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame()
    if len(R):
        m = np.ones(len(R), bool)
        for k in KEY:
            m &= R[k].astype(str) == str(row[k])
        R = R[~m]
    R = pd.concat([R, pd.DataFrame([row])], ignore_index=True)
    R.to_csv(CSV, index=False)


def run_cell(model, config, n, seed, ctx, test, cv_err, extra=None):
    X, Y = F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy()
    Xte, Yte = F.X_of(test), test[["ddx", "ddy"]].to_numpy()
    t0 = time.perf_counter()
    if model == "tabpfn_v35_api":
        m = M.make("tabpfn", api=True)
    elif model == "tabpfn_fast1_local":
        m = M.make("tabpfn", fast=True, n_estimators=1, fit_mode="fit_with_cache", device="cpu")
    elif model in ("lgbm_default", "lgbm_all"):
        m = M.make("lgbm", n_jobs=2)
    elif model == "lgbm_m50n100":
        m = M.make("lgbm:m50:n100", n_jobs=2)
    else:
        raise ValueError(model)
    Q = m.fit(X, Y).predict_quantiles(Xte, QS)
    secs = time.perf_counter() - t0
    r = dict(model=model, config=config, n=len(ctx), seed=seed, **score(Q, Yte), cv_err=cv_err,
             seconds=round(secs, 1), test_rows=len(test), features=F.FEATURES_VERSION, **(extra or {}))
    save(r)
    print(f"{model:<19} {config:<9} n {len(ctx):>6} seed {seed}  err {r['err']:.3f}  pinball {r['pinball']:.4f}  "
          f"joint {100 * r['joint90']:.1f}%  width {r['width_along']:.2f}/{r['width_across']:.2f}  {secs:.0f} s", flush=True)
    return r


def retrieved(train, test, target, seed):
    bank = C.Bank(train, seed=seed)
    for k in range(1, 50):
        idx = bank.knn(test, k)
        if len(idx) >= target:
            break
    return bank.df.iloc[idx[:target]], dict(k=k, union=len(idx), pool=len(bank.df))


def plan_api():
    random_n = (1000, 5000, 10000, 20000)
    cells = [("random", n, s) for n in random_n for s in SEEDS] + [("retrieved", 5000, s) for s in SEEDS]
    return cells


def log_usage(calls, note):
    day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    U = pd.read_csv(USAGE)
    prev = pd.to_numeric(U.loc[(U.date_utc == day) & (U.run != "day total (server)"), "credits"], errors="coerce").sum()
    total = int(prev + calls * CREDITS_PER_CALL)
    with open(USAGE, "a", newline="", encoding="utf-8") as f:
        note = note.replace(",", ";")
        f.write(f"{day},context_size (tools/context_size.py),{calls},{calls * CREDITS_PER_CALL},"
                f"{note}; running total {day}: {total} credits\n")


def part_api(train, test, cv_err, not_before):
    if not_before:
        hh, mm = map(int, not_before.split(":"))
        now = dt.datetime.now(dt.timezone.utc)
        t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if t < now - dt.timedelta(hours=12):
            t += dt.timedelta(days=1)
        if t > now:
            print(f"waiting until {t:%Y-%m-%d %H:%M} UTC ({(t - now).total_seconds() / 60:.0f} min)", flush=True)
            time.sleep((t - now).total_seconds())
    cells = plan_api()
    n_calls = 2 * len(cells)
    print(f"API plan: {len(cells)} TabPFN fits (2 axes each) = {n_calls} calls = {n_calls * CREDITS_PER_CALL:,} credits "
          f"(cap {CAP:,})", flush=True)
    if n_calls * CREDITS_PER_CALL > CAP:
        sys.exit("above the credit cap; abort")
    if not os.environ.get("TABPFN_TOKEN"):
        import subprocess
        tok = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                              "[Environment]::GetEnvironmentVariable('TABPFN_TOKEN','User')"],
                             capture_output=True, text=True).stdout.strip()
        os.environ["TABPFN_TOKEN"] = tok
    import tabpfn_client
    tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
    before = tabpfn_client.get_api_usage()
    print("server before:", before, flush=True)
    calls, err = 0, None
    try:
        for config, n, s in sorted(cells, key=lambda c: c[1]):
            if config == "random":
                ctx, extra = train.sample(n, random_state=s), {}
            else:
                ctx, extra = retrieved(train, test, n, s)
            calls += 2      # counted before the call: a failed predict may still be charged
            run_cell("tabpfn_v35_api", config, n, s, ctx, test, cv_err, extra)
    except Exception as e:          # quota or network: stop, keep what is done, log calls made
        err = repr(e)[:200]
        print("API error, stopping:", err, flush=True)
    after = tabpfn_client.get_api_usage()
    print("server after:", after, flush=True)
    used = lambda s: s.split("used ")[1].split(" of")[0] if "used " in s else "?"  # noqa: E731
    log_usage(calls, f"TabPFN v3.5 default (8 est) context-size grid on crowd_cross +2 s; server usage {used(before)} -> "
                     f"{used(after)}" + (f"; stopped on error {err}" if err else ""))


def part_local(train, test, cv_err):
    import torch
    torch.set_num_threads(2)
    save(dict(model="cv", config="cv", n=0, seed=0, err=cv_err, pinball=np.nan, joint90=0.0, width_along=0.0,
              width_across=0.0, cv_err=cv_err, seconds=0.0, test_rows=len(test), features=F.FEATURES_VERSION))
    run_cell("lgbm_all", "all", len(train), 0, train, test, cv_err)
    for n in (500, 1000, 2000, 5000, 10000, 20000):
        for s in SEEDS:
            ctx = train.sample(n, random_state=s)
            for model in ("lgbm_default", "lgbm_m50n100") + (("tabpfn_fast1_local",) if n <= 2000 else ()):
                run_cell(model, "random", n, s, ctx, test, cv_err)
    for target in (1000, 5000):
        for s in SEEDS:
            ctx, extra = retrieved(train, test, target, s)
            print(f"retrieved target {target} seed {s}: k {extra['k']}, union {extra['union']}, pool {extra['pool']}", flush=True)
            for model in ("lgbm_default", "lgbm_m50n100") + (("tabpfn_fast1_local",) if target <= 1000 else ()):
                run_cell(model, "retrieved", target, s, ctx, test, cv_err, extra)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = pd.read_csv(CSV)
    cv = R.loc[R.model == "cv", "err"].iloc[0]
    allr = R[R.model == "lgbm_all"].iloc[0]
    styles = {"tabpfn_v35_api": ("#e8590c", "TabPFN-3.5 default (API)"),
              "tabpfn_fast1_local": ("#f2a541", "TabPFN-3.5-Fast, 1 estimator (local)"),
              "lgbm_default": ("#2a6fdb", "LightGBM default, same rows"),
              "lgbm_m50n100": ("#7a3fb5", "LightGBM min-child 50, 100 trees, same rows")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, col, lab in ((axes[0], "err", "mean error of the median at +2 s (m)"),
                         (axes[1], "joint90", "5-95% box holds truth on both axes")):
        for model, (c, name) in styles.items():
            for config, mk, ls in (("random", "o", "-"), ("retrieved", "D", "none")):
                g = R[(R.model == model) & (R.config == config)].groupby("n")[col].agg(["mean", "std"]).reset_index()
                if not len(g):
                    continue
                ax.errorbar(g.n, g["mean"], yerr=g["std"].fillna(0), color=c, marker=mk, ls=ls, capsize=3,
                            mfc="white" if config == "retrieved" else c, ms=7 if config == "retrieved" else 5,
                            label=name + (" (random rows)" if config == "random" else " (retrieved rows)"))
        ax.axhline(allr[col], color="#2a6fdb", ls="--", lw=1, label=f"LightGBM default, all {int(allr.n):,} rows")
        if col == "err":
            ax.axhline(cv, color="grey", ls=":", lw=1.2, label="constant velocity")
        else:
            ax.axhline(0.9, color="grey", ls=":", lw=1.2, label="nominal 90% (per axis; joint target is lower)")
        ax.set_xscale("log"); ax.set_xlabel("context rows"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    extra = [(h, lab) for h, lab in zip(h1, l1) if lab not in l0]
    fig.legend(h0 + [h for h, _ in extra], l0 + [lab for _, lab in extra], fontsize=8, loc="lower center", ncol=3,
               frameon=False)
    fig.suptitle("Crossing crowd, +2 s, 2,000 test rows from held-out episodes; mean ± sd over 3 context samples", fontsize=10)
    fig.tight_layout(rect=(0, 0.2, 1, 1)); fig.savefig(FIG, dpi=130)
    print("wrote", FIG)


def table():
    R = pd.read_csv(CSV)
    g = R.groupby(["model", "config", "n"]).agg(err=("err", "mean"), err_sd=("err", "std"), pinball=("pinball", "mean"),
                                                joint90=("joint90", "mean"), w_along=("width_along", "mean"),
                                                w_across=("width_across", "mean"), secs=("seconds", "mean"),
                                                seeds=("seed", "count")).round(4)
    print(g.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["local", "api"])
    ap.add_argument("--not-before", default=None, help="UTC HH:MM to wait for before API calls")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.part:
        train, test = load()
        cv_err = float(np.linalg.norm(test[["ddx", "ddy"]].to_numpy(), axis=1).mean())
        print(f"{F.FEATURES_VERSION}: {len(train)} train rows ({train.episode.nunique()} ep), {len(test)} test rows; "
              f"constant velocity error {cv_err:.3f} m", flush=True)
        (part_local if a.part == "local" else lambda t, te, c: part_api(t, te, c, a.not_before))(train, test, cv_err)
    table()
    plot()


if __name__ == "__main__":
    main()
