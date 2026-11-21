"""Decision cells of the tier grid with 3 context seeds each (different context samples, same 600 test rows).

    python tools/tier_seeds.py
Cells: v3.5 (1 and 8 estimators), v3.5-fast (1 estimator), v2 default; context 1,000 and 2,000; ETH->Hotel and
Hotel->ETH; +2 s. Reports mean and spread over seeds. Writes results/tier_seeds.csv.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight.core import features as F  # noqa: E402
from run_offline import rows_for  # noqa: E402
from tools.tier_grid import QS, score  # noqa: E402

CELLS = [("v3.5_default", 8), ("v3.5_default", 1), ("v3.5-fast_default", 1), ("v2_default", None)]
CTX, SEEDS = [1000, 2000], [11, 22, 33]


def main():
    n_calls = 2 * len(CELLS) * len(CTX) * len(SEEDS) * 2
    print(f"API predict calls: {n_calls} (= {10_000 * n_calls:,} credits)", flush=True)
    import tabpfn_client
    from tabpfn_client import TabPFNRegressor
    tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
    hz = (1.2, 2.0, 4.8); rows = []
    for a, b in (("hotel", "eth"), ("eth", "hotel")):
        src = rows_for(f"data/{a}", 0.0, horizons=hz); dst = rows_for(f"data/{b}", 0.0, horizons=hz)
        src, te = src[np.isclose(src.h, 2.0)], dst[np.isclose(dst.h, 2.0)].sample(600, random_state=0)
        Y, Xte = te[["ddx", "ddy"]].to_numpy(), F.X_of(te)
        for n in CTX:
            for s in SEEDS:
                ctx = src.sample(min(n, len(src)), random_state=s)
                X, Yc = F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy()
                for path, est in CELLS:
                    Q = np.zeros((len(te), 2, len(QS)))
                    for ax in range(2):
                        m = TabPFNRegressor(model_path=path, **({"n_estimators": est} if est else {}))
                        m.fit(X, Yc[:, ax])
                        Q[:, ax, :] = np.stack(m.predict(Xte, output_type="quantiles", quantiles=QS), axis=1)
                    Q.sort(axis=2)
                    err, pin, joint, width = score(Q, Y)
                    rows.append(dict(setup=f"{a}->{b}", model=path.replace("_default", ""), est=est or "default",
                                     context=n, seed=s, err=err, pinball=pin, joint90=joint, width80=width))
                    pd.DataFrame(rows).to_csv("results/tier_seeds.csv", index=False)
                    print(f"{a}->{b} {path:<18} est {str(est or 'def'):<4} ctx {n} seed {s}  err {err:.3f}  joint {100 * joint:.0f}%", flush=True)
    R = pd.DataFrame(rows)
    print(R.groupby(["setup", "model", "est", "context"]).agg(err=("err", "mean"), err_sd=("err", "std"),
          pinball=("pinball", "mean"), joint90=("joint90", "mean"), width=("width80", "mean")).round(4).to_string())


if __name__ == "__main__":
    main()
