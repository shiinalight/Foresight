"""Claim A rerun on fixed features: does TabPFN keep its box promise in a place it has never seen?

    python tools/claim_a.py               # local TabPFN default model
    python tools/claim_a.py --api         # hosted TabPFN (counts calls first)

Same test rows as the pre-fix run: the 900-row sample of the f1 table (identified by episode, agent, t, h),
scored on the current features. 5 context seeds (different 1,500-row samples of the training scene).
Models: constant velocity, constant velocity + band, LightGBM (same rows), LightGBM (all rows), TabPFN.
At +2 s, reports per model mean and sd over seeds of: joint coverage of the 5-95% box, per-axis coverage of
10-90%, 10-90% width, pinball, error; for all people, MOVING people (speed > 0.2 m/s) and STANDING people.
Writes results/claim_a_<features_version>.csv and a markdown table.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from run_offline import rows_for  # noqa: E402

QS = (0.05, 0.1, 0.5, 0.9, 0.95)
HZ = (1.2, 2.0, 4.8)
KEYS = ["episode", "agent_id", "t", "h"]


def score(Q, Y):
    j90 = ((Y >= Q[:, :, 0]) & (Y <= Q[:, :, 4])).all(axis=1)
    c80 = ((Y >= Q[:, :, 1]) & (Y <= Q[:, :, 3])).mean(axis=1)
    pin = np.mean([np.maximum(q * (Y - Q[:, :, j]), (q - 1) * (Y - Q[:, :, j])).mean(axis=1) for j, q in enumerate(QS)], axis=0)
    return dict(joint90=j90, cov80=c80, width80=(Q[:, :, 3] - Q[:, :, 1]).mean(axis=1), pinball=pin,
                err=np.linalg.norm(Q[:, :, 2] - Y, axis=1))


def test_rows(dst, n=900):
    """The pre-fix sample: same keys as the f1 evaluation (evaluate.py, random_state 0)."""
    old = Path(f"data/{dst}/rows_noise0.0_h1.2_2_4.8.parquet")
    new = rows_for(f"data/{dst}", 0.0, horizons=HZ)
    if old.exists():
        keys = pd.read_parquet(old).sample(n, random_state=0)[KEYS]
        te = keys.merge(new, on=KEYS, how="inner")
        print(f"{dst}: {len(te)} of {n} pre-fix test rows found in the current features", flush=True)
    else:
        te = new.sample(n, random_state=0)
        print(f"{dst}: pre-fix cache missing; fresh sample of {n}", flush=True)
    return te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--context", type=int, default=1500)
    ap.add_argument("--api", action="store_true")
    a = ap.parse_args()
    if a.api:
        n_calls = 2 * a.seeds * 2
        print(f"API predict calls: {n_calls} (= {10_000 * n_calls:,} credits)", flush=True)
    rows = []
    for src, dst in (("hotel", "eth"), ("eth", "hotel")):
        tr = rows_for(f"data/{src}", 0.0, horizons=HZ)
        te = test_rows(dst)
        te = te[np.isclose(te.h, 2.0)]
        Y, X = te[["ddx", "ddy"]].to_numpy(), F.X_of(te)
        moving = (te.speed > 0.2).to_numpy()
        for s in range(a.seeds):
            ctx = tr.sample(min(a.context, len(tr)), random_state=s)
            for key, rws, kind in (("cv", ctx, "cv"), ("cv_band", tr, "cv_band"), ("lgbm", ctx, "lgbm"),
                                   ("lgbm_all", tr, "lgbm"), ("tabpfn", ctx, "tabpfn")):
                if s > 0 and key in ("cv", "cv_band", "lgbm_all"):
                    continue                                 # do not depend on the context seed
                kw = {"api": True} if (kind == "tabpfn" and a.api) else {}
                m = M.make(kind, **kw).fit(F.X_of(rws), rws[["ddx", "ddy"]].to_numpy())
                sc = score(m.predict_quantiles(X, QS), Y)
                for grp, mask in (("all", np.ones(len(te), bool)), ("moving", moving), ("standing", ~moving)):
                    rows.append(dict(direction=f"{src}->{dst}", model=key, seed=s, group=grp, n=int(mask.sum()),
                                     **{k: float(v[mask].mean()) for k, v in sc.items()}))
                print(f"{src}->{dst} seed {s} {key:<9} joint90 all {100 * sc['joint90'].mean():.0f}%  "
                      f"moving {100 * sc['joint90'][moving].mean():.0f}%  err {sc['err'].mean():.3f}", flush=True)
    R = pd.DataFrame(rows)
    out = f"results/claim_a_{F.FEATURES_VERSION}.csv"; R.to_csv(out, index=False)
    T = R.groupby(["direction", "group", "model"]).agg(n=("n", "first"), seeds=("seed", "nunique"),
                                                        joint90=("joint90", "mean"), joint90_sd=("joint90", "std"),
                                                        cov80=("cov80", "mean"), width80=("width80", "mean"),
                                                        pinball=("pinball", "mean"), err=("err", "mean"),
                                                        err_sd=("err", "std")).round(4)
    print(T.to_string())
    Path(out.replace(".csv", ".md")).write_text(T.to_markdown(), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
