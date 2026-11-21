"""TabPFN latency for the closed loop: one refit (context n rows) and one predict (m query rows).

    python tools/latency.py                 # laptop / whatever device torch sees
Prints fit and predict seconds for n in {500, 1000} and m in {32, 64}, default and fit_with_cache.
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


def main(data="data/crowd"):
    df = rows_for(data, 0.1, horizons=(1.0, 2.0))
    for mode in (None, "fit_with_cache"):
        for n in (500, 1000):
            ctx = df.sample(n, random_state=0)
            m = M.make("tabpfn", fit_mode=mode)
            t0 = time.perf_counter(); m.fit(F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy()); tf = time.perf_counter() - t0
            for k in (32, 64):
                q = df.sample(k, random_state=1)
                m.predict_quantiles(F.X_of(q))                      # warm-up
                t0 = time.perf_counter(); m.predict_quantiles(F.X_of(q)); tp = time.perf_counter() - t0
                print(f"device {m.device}  mode {mode or 'default':<15} context {n:>5}  fit {tf:5.2f} s  "
                      f"predict {k} rows {tp:5.2f} s")


if __name__ == "__main__":
    main(*sys.argv[1:])
