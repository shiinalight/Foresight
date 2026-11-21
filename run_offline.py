"""R1 offline experiment: predict each agent's position 1 s and 2 s ahead as quantiles.

    python run_offline.py --data data/toy                      # baselines + LightGBM
    python run_offline.py --data data/toy --tabpfn --context 500 1500
    python run_offline.py --data data/toy --tabpfn --retrieval  # per-moment context selection

Splits by episode. Prints error at the median (metres), 10-90% coverage and band width per
model, per horizon, and TabPFN timing.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.core import context as C
from foresight.core import eval as E
from foresight.core import features as F
from foresight.core import models as M
from foresight.core import schema


def rows_for(data_dir, noise, seed=0, horizons=(1.0, 2.0), label_from="truth"):
    hz = "_".join(f"{h:g}" for h in horizons)
    lab = "" if label_from == "truth" else f"_{label_from}"
    cache = Path(data_dir) / f"rows_{F.FEATURES_VERSION}_noise{noise}_h{hz}{lab}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    tracks, obst, meta = schema.load(data_dir)
    t0 = time.perf_counter()
    df = F.build_rows(tracks, obst, meta, noise=noise, seed=seed, horizons=horizons, label_from=label_from)
    print(f"built {len(df)} rows from {df.episode.nunique()} episodes in {time.perf_counter() - t0:.0f} s")
    df.to_parquet(cache)
    return df


def report(name, n_ctx, df_te, Q, Y, extra=""):
    lines = []
    for h in sorted(df_te["h"].round(2).unique()):
        m = np.isclose(df_te["h"], h)
        r = E.metrics(Q[m], Y[m])
        lines.append(f"{name:<10} ctx {n_ctx:>6}  h={h:.1f}s  err {r['fde']:.3f} m  p90 {r['fde_p90']:.3f}  "
                     f"cover {100 * r['coverage']:5.1f}%  width {r['width']:.3f}  pinball {r['pinball']:.4f}")
    print("\n".join(lines) + (f"   {extra}" if extra else ""))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--context", type=int, nargs="*", default=[500, 1500])
    ap.add_argument("--tabpfn", action="store_true")
    ap.add_argument("--fast", action="store_true", help="TabPFN-3.5-Fast")
    ap.add_argument("--test-rows", type=int, default=2000, help="cap test rows (TabPFN time)")
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--horizons", type=float, nargs="*", default=[1.0, 2.0])
    ap.add_argument("--train-data", default=None,
                    help="take context rows from this dataset and test on all of --data (e.g. train ETH, test Hotel; or sim to real)")
    a = ap.parse_args()

    hz = tuple(a.horizons)
    df = rows_for(a.data, a.noise, horizons=hz)
    if a.train_data:
        d_tr, d_te = rows_for(a.train_data, a.noise, horizons=hz), df
    else:
        tr, _, te = F.split_by_episode(df)
        d_tr, d_te = df[tr], df[te]
    if len(d_te) > a.test_rows:
        d_te = d_te.sample(a.test_rows, random_state=0)
    X_te, Y_te = F.X_of(d_te), d_te[["ddx", "ddy"]].to_numpy()
    print(f"train rows {len(d_tr)} ({d_tr.episode.nunique()} ep), test rows {len(d_te)} ({d_te.episode.nunique()} ep)")
    print("modes in test:", d_te["mode"].value_counts(normalize=True).round(2).to_dict())

    names = ["cv", "cv_band", "lgbm"] + (["tabpfn"] if a.tabpfn else [])
    results = {}
    for n in a.context:
        ctx = d_tr.sample(min(n, len(d_tr)), random_state=0)
        X, Y = F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy()
        for name in names:
            if name in ("cv",) and n != a.context[0]:
                continue
            m = M.make(name, **({"fast": a.fast} if name == "tabpfn" else {})).fit(X, Y)
            Q = m.predict_quantiles(X_te)
            extra = ""
            if name == "tabpfn":
                extra = f"fit {m.t_fit:.1f} s, predict {m.t_pred:.1f} s for {len(X_te)} rows ({1000 * m.t_pred / len(X_te):.1f} ms/row)"
            results[f"{name}@{n}"] = report(name, len(ctx), d_te, Q, Y_te, extra)

    if a.retrieval and a.tabpfn:
        # per-moment context: queries grouped by (episode, t), context from bank + same episode
        bank = C.Bank(d_tr)
        groups = list(d_te.groupby(["episode", "t"]))[:40]
        for mix_name, mix in [("random", (0, 0, 1)), ("knn", (0, 1, 0)), ("mix", (0.6, 0.3, 0.1))]:
            Qs, Ys = [], []
            for (ep, t), q in groups:
                live = df[(df.episode == ep)]
                ctx = C.select(q, bank, live=live, budget=a.context[-1], mix=mix)
                m = M.make("tabpfn", fast=a.fast).fit(F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy())
                Qs.append(m.predict_quantiles(F.X_of(q))); Ys.append(q[["ddx", "ddy"]].to_numpy())
            r = E.metrics(np.concatenate(Qs), np.concatenate(Ys))
            print(f"retrieval {mix_name:<7} err {r['fde']:.3f}  cover {100 * r['coverage']:.1f}%  width {r['width']:.3f}")

    Path("results").mkdir(exist_ok=True)
    (Path("results") / "offline_latest.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
