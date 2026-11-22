"""TabPFN timing for the shipped closed-loop design, on whatever device torch sees (CUDA pod, Mac MPS, laptop CPU).

The shipped planner (tabpfn_split, T5): TabPFN-3.5-Fast, 1 estimator, fit_with_cache; a frozen context of 2,000 bank
rows fitted once, plus a scene model refitted every 2 s on the latest <= 250 completed rows of this scene; each replan
predicts every person in range at horizons 1 and 2 s (24 people x 2 = up to 48 query rows) with both models and
averages the quantiles (run_closed.AvgQuantiles). 3.5 with 1 estimator is timed alongside as the tier comparison.

What is timed (median and p90 of --reps repeats, after a discarded warm-up fit + predict that pays CUDA start-up and
the weight load):
  frozen_fit     fit on 2,000 rows (one-off per episode)
  predict        predict on the cached 2,000-row context, query rows 16 / 48 / 96 (48 = the T5 crowd)
  scene_refit    fit on 50 / 150 / 250 rows (the cache has no row-append, so every refresh rebuilds it)
  split_predict  frozen predict + scene predict + average, 48 query rows = TabPFN time per replan
Each call covers both axes (one model per axis) and the 5 quantiles the planner asks for. predict returns numpy,
which forces a device sync; torch.cuda.synchronize() is called before every clock read as well.

    python tools/latency_gpu.py                          # -> results/latency_<device>.csv + .json
    python tools/latency_gpu.py --parity results/latency_parity_cpu.npz   # also compare quantiles with that file
Writes results/latency_parity_<device>.npz (quantiles for one fixed context and query) for the parity check.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from run_offline import rows_for  # noqa: E402

QS = (0.05, 0.15, 0.5, 0.85, 0.95)          # what _rows asks for at box-q 0.15
TIERS = [("3.5-Fast, 1 est (shipped)", {"fast": True, "n_estimators": 1}),
         ("3.5, 1 est", {"fast": False, "n_estimators": 1})]


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed(fn, reps):
    out = []
    for _ in range(reps):
        sync(); t0 = time.perf_counter(); fn(); sync(); out.append(time.perf_counter() - t0)
    a = 1000 * np.array(out)
    return float(np.median(a)), float(np.percentile(a, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="data/crowd_cross_t2")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--parity", default=None, help="npz from another device to compare quantiles with")
    ap.add_argument("--tiers", default="all", choices=["all", "fast"])
    a = ap.parse_args()

    import torch
    import importlib.metadata as md
    dev = M.best_device()
    meta = {"device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor(),
            "torch": torch.__version__, "tabpfn": md.version("tabpfn"), "python": platform.python_version(),
            "machine": platform.machine(), "reps": a.reps, "threads": torch.get_num_threads()}
    print(meta, flush=True)

    df = rows_for(a.bank, 0.1, horizons=(1.0, 2.0)).dropna(subset=["ddx", "ddy"])
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(df))
    bank, rec, qry = df.iloc[idx[:2000]], df.iloc[idx[2000:2250]], df.iloc[idx[2250:2346]]
    Xb, Yb = F.X_of(bank), bank[["ddx", "ddy"]].to_numpy(float)
    Xr, Yr = F.X_of(rec), rec[["ddx", "ddy"]].to_numpy(float)
    Xq = F.X_of(qry)

    rows, parity = [], {}
    for name, kw in TIERS[: 1 if a.tiers == "fast" else None]:
        mk = lambda: M.make("tabpfn", fit_mode="fit_with_cache", device=dev, **kw)  # noqa: E731
        w = mk().fit(Xb, Yb); w.predict_quantiles(Xq[:48], QS)                 # warm-up, discarded
        frozen = mk().fit(Xb, Yb)
        rows.append({"tier": name, "step": "frozen_fit", "rows": 2000, "query": 0,
                     **dict(zip(("median_ms", "p90_ms"), timed(lambda: mk().fit(Xb, Yb), max(5, a.reps // 4))))})
        for m in (16, 48, 96):
            rows.append({"tier": name, "step": "predict", "rows": 2000, "query": m,
                         **dict(zip(("median_ms", "p90_ms"), timed(lambda: frozen.predict_quantiles(Xq[:m], QS), a.reps)))})
        for n in (50, 150, 250):
            rows.append({"tier": name, "step": "scene_refit", "rows": n, "query": 0,
                         **dict(zip(("median_ms", "p90_ms"), timed(lambda: mk().fit(Xr[:n], Yr[:n]), a.reps)))})
        scene = mk().fit(Xr, Yr)

        def split_predict():
            return np.sort((frozen.predict_quantiles(Xq[:48], QS) + scene.predict_quantiles(Xq[:48], QS)) / 2, axis=2)
        rows.append({"tier": name, "step": "split_predict", "rows": 2250, "query": 48,
                     **dict(zip(("median_ms", "p90_ms"), timed(split_predict, a.reps)))})
        parity[name] = split_predict()
        for r in rows[-8:]:
            print(f"{dev:<5} {r['tier']:<26} {r['step']:<14} rows {r['rows']:>5} query {r['query']:>3}  "
                  f"median {r['median_ms']:8.1f} ms  p90 {r['p90_ms']:8.1f} ms", flush=True)

    if dev == "cuda":
        meta["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    tag = dev if dev != "cuda" else "cuda_" + meta["gpu"].replace(" ", "_").replace("/", "_")
    out = Path("results")
    pd.DataFrame(rows).assign(**{k: meta[k] for k in ("device", "gpu", "torch", "tabpfn")}).to_csv(
        out / f"latency_{tag}.csv", index=False)
    (out / f"latency_{tag}.json").write_text(json.dumps(meta, indent=2))
    np.savez(out / f"latency_parity_{dev}.npz", **{k.split(",")[0].replace(" ", "_"): v for k, v in parity.items()})
    print(f"wrote results/latency_{tag}.csv", flush=True)

    if a.parity:
        ref = np.load(a.parity)
        for k, v in parity.items():
            key = k.split(",")[0].replace(" ", "_")
            if key in ref:
                d = np.abs(ref[key] - v)
                print(f"parity {k}: max |diff| {d.max():.4f} m, median {np.median(d):.5f} m (vs {a.parity})")


if __name__ == "__main__":
    main()
