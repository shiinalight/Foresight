"""Does the order of TabPFN's example rows matter? (Explainer claim: no; the examples are a set.)

Fits TabPFN-3.5-Fast (1 estimator, cached) on the same 800 example rows in four orders (as given,
shuffled, reversed, sorted by time) and compares the predicted quantiles for 60 query rows.

    python tools/check_row_order.py [data_dir]          # default data/crowd_cross, +2 s rows

Result on 21 Nov (data/crowd f3 rows): max |difference| 1.4e-5 to 2.5e-5 m, i.e. numerical noise.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from run_offline import rows_for  # noqa: E402


def main(data="data/crowd_cross"):
    rows = rows_for(data, 0.1, horizons=(1.0, 2.0))
    rows = rows[np.isclose(rows.h, 2.0)]
    ctx = rows.sample(800, random_state=0)
    q = rows.drop(ctx.index).sample(60, random_state=1)
    X, Y, Xq = F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy(), F.X_of(q)

    def run(order):
        m = M.make("tabpfn", fast=True, n_estimators=1).fit(X[order], Y[order])
        return m.predict_quantiles(Xq, (0.1, 0.5, 0.9))

    base = run(np.arange(len(X)))
    rng = np.random.default_rng(0)
    for name, order in [("shuffled", rng.permutation(len(X))), ("reversed", np.arange(len(X))[::-1]),
                        ("sorted by time", np.argsort(ctx.t.to_numpy()))]:
        d = np.abs(run(order) - base)
        print(f"{name:15s} max |difference| in predicted quantiles {d.max():.2e} m (mean {d.mean():.2e})")


if __name__ == "__main__":
    main(*sys.argv[1:])
