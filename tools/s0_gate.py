"""S0 gate for a simulated scene (docs/GATES.md): does a learned model beat constant velocity by >= 15% at +2 s?

    python tools/s0_gate.py data/crowd_cross

Split by episode (75/25). Constant velocity against LightGBM on all training rows and TabPFN (3.5-Fast,
1 estimator; the crowd loop's model) on 1,000 random training rows, on up to 2,000 test rows at +2 s.
Prints error, joint coverage of the 5-95% box and the gain over constant velocity; mode shares for context.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from run_offline import rows_for  # noqa: E402

QS = (0.05, 0.5, 0.95)


def main(data):
    df = rows_for(data, 0.1, horizons=(1.0, 2.0))
    df = df[np.isclose(df.h, 2.0)]
    tr, _, te = F.split_by_episode(df)
    train, test = df[tr], df[te].sample(min(2000, int(te.sum())), random_state=0)
    Y, X = test[["ddx", "ddy"]].to_numpy(), F.X_of(test)
    print(f"{data} ({F.FEATURES_VERSION}): {len(train)} train rows, {len(test)} test rows; modes in test:",
          test["mode"].value_counts(normalize=True).round(3).to_dict())
    res = {}
    for name, kind, rows, kw in (("constant velocity", "cv", train, {}),
                                 ("LightGBM, all rows", "lgbm", train, {}),
                                 ("TabPFN fast-1, 1,000 rows", "tabpfn", train.sample(1000, random_state=0),
                                  {"fast": True, "n_estimators": 1})):
        Q = M.make(kind, **kw).fit(F.X_of(rows), rows[["ddx", "ddy"]].to_numpy()).predict_quantiles(X, QS)
        err = float(np.linalg.norm(Q[:, :, 1] - Y, axis=1).mean())
        joint = float(((Y >= Q[:, :, 0]) & (Y <= Q[:, :, 2])).all(axis=1).mean())
        res[name] = err
        print(f"  {name:<28} error {err:.3f} m   5-95 box holds truth {100 * joint:.0f}%")
    cv = res["constant velocity"]; best = min(v for k, v in res.items() if k != "constant velocity")
    gain = 1 - best / cv
    print(f"best learned model beats constant velocity by {100 * gain:.1f}%  ->  S0 gate (>= 15%): "
          f"{'PASS' if gain >= 0.15 else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1])
