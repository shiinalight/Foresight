"""TabPFN latency for the closed loop: one refit (context n rows) and one predict (m query rows).

    python tools/latency.py                 # laptop / whatever device torch sees
Prints cached fit and predict seconds for n in {500, 1000} and m in {32, 64}, for 3.5-1 (real-time tier), Fast-1 and 3.5-8.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from run_offline import rows_for  # noqa: E402


CONFIGS = [("3.5, 1 est (real-time tier, D-5)", {"n_estimators": 1}),
           ("3.5-Fast, 1 est", {"fast": True, "n_estimators": 1}),
           ("3.5, default (8 est)", {})]


def main(data="data/crowd"):
    df = rows_for(data, 0.1, horizons=(1.0, 2.0))
    for name, kw in CONFIGS:
        for n in (500, 1000):
            ctx = df.sample(n, random_state=0)
            mode = "fit_with_cache"
            m = M.make("tabpfn", fit_mode=mode, **kw)
            t0 = time.perf_counter(); m.fit(F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy()); tf = time.perf_counter() - t0
            for k in (32, 64):
                q = df.sample(k, random_state=1)
                m.predict_quantiles(F.X_of(q))                      # warm-up
                t0 = time.perf_counter(); m.predict_quantiles(F.X_of(q)); tp = time.perf_counter() - t0
                print(f"device {m.device}  {name:<34} cached  context {n:>5}  fit {tf:5.2f} s  "
                      f"predict {k} rows {tp:5.2f} s")


if __name__ == "__main__":
    main(*sys.argv[1:])
