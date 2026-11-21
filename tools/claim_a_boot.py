"""Claim A: paired person-bootstrap intervals on the interval-score differences (TabPFN minus LightGBM).

    python tools/claim_a_boot.py

Same test rows and context seeds as tools/claim_a.py (+2 s, 5 context seeds of 1,500 rows; lgbm_all on every
training row). Per test row: interval score of the 10-90% (is80) and 5-95% (is90) boxes, averaged over the context
seeds for the seed-dependent models. Differences TabPFN - lgbm_all and TabPFN - lgbm_m50n100 per row, resampled by
PERSON (episode, agent_id) with replacement, 5,000 times; 95% percentile intervals. Negative = TabPFN better.
Also the joint 5-95 box miss difference. Writes results/claim_a_boot_<features_version>.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from claim_a import HZ, QS, interval_score, test_rows  # noqa: E402
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from run_offline import rows_for  # noqa: E402

MODELS = (("tabpfn", "tabpfn", True), ("lgbm_all", "lgbm", False), ("lgbm_m50n100", "lgbm:m50:n100", True))


def main(seeds=5, context=1500, n_boot=5000):
    out = []
    for src, dst in (("hotel", "eth"), ("eth", "hotel")):
        tr = rows_for(f"data/{src}", 0.0, horizons=HZ)
        te = test_rows(dst)
        te = te[np.isclose(te.h, 2.0)].reset_index(drop=True)
        Y, X = te[["ddx", "ddy"]].to_numpy(), F.X_of(te)
        per = {}
        for key, kind, by_seed in MODELS:
            acc = []
            for s in range(seeds if by_seed else 1):
                rws = tr.sample(min(context, len(tr)), random_state=s) if by_seed else tr
                Q = M.make(kind).fit(F.X_of(rws), rws[["ddx", "ddy"]].to_numpy()).predict_quantiles(X, QS)
                miss = ~((Y >= Q[:, :, 0]) & (Y <= Q[:, :, 4])).all(axis=1)
                acc.append(np.column_stack([interval_score(Q[:, :, 1], Q[:, :, 3], Y, 0.2),
                                            interval_score(Q[:, :, 0], Q[:, :, 4], Y, 0.1), miss]))
            per[key] = np.mean(acc, axis=0)
            print(f"{src}->{dst} {key}: is80 {per[key][:, 0].mean():.4f}  is90 {per[key][:, 1].mean():.4f}  "
                  f"box miss {per[key][:, 2].mean():.3f}", flush=True)
        person = te.episode.astype(str) + "_" + te.agent_id.astype(str)
        groups = [np.flatnonzero(person.to_numpy() == p) for p in person.unique()]
        rng = np.random.default_rng(0)
        for other in ("lgbm_all", "lgbm_m50n100"):
            D = per["tabpfn"] - per[other]
            sums = np.array([D[g].sum(axis=0) for g in groups]); cnt = np.array([len(g) for g in groups])
            bs = []
            for _ in range(n_boot):
                idx = rng.integers(0, len(groups), len(groups))
                bs.append(sums[idx].sum(axis=0) / cnt[idx].sum())
            bs = np.array(bs)
            for j, m in enumerate(("is80", "is90", "box_miss")):
                lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
                verdict = "TabPFN better" if hi < 0 else ("TabPFN worse" if lo > 0 else "level")
                out.append(dict(direction=f"{src}->{dst}", comparison=f"tabpfn - {other}", metric=m,
                                diff=float(D[:, j].mean()), lo=lo, hi=hi, persons=len(groups), rows=len(te),
                                verdict=verdict))
    R = pd.DataFrame(out)
    path = f"results/claim_a_boot_{F.FEATURES_VERSION}.csv"
    R.to_csv(path, index=False)
    print(R.round(4).to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
