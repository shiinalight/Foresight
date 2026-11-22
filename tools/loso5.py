"""Claim A on five real scenes, leave one scene out (plan: docs/LOSO5.md, fixed before any run).

    python tools/loso5.py --dry      # LightGBM stands in for TabPFN (no API), checks the plumbing
    python tools/loso5.py            # hosted TabPFN (tabpfn-client, TABPFN_TOKEN)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from run_offline import rows_for  # noqa: E402

SCENES = ["eth", "hotel", "zara1", "zara2", "univ"]
QS = (0.05, 0.1, 0.5, 0.9, 0.95)
H = 2.0
N_TEST, N_CTX, SEEDS, BOOT = 1500, 1500, (0, 1, 2), 2000


def load(scene):
    d = rows_for(f"data/{scene}", 0.0, horizons=(1.2, 2.0, 4.8))
    d = d[np.isclose(d.h, H)].dropna(subset=["ddx", "ddy"]).copy()
    d["person"] = scene + ":" + d.episode.astype(str) + ":" + d.agent_id.astype(str)
    return d


def scores(Q, Y):
    lo, hi = Q[:, :, 0], Q[:, :, 4]
    j90 = ((Y >= lo) & (Y <= hi)).all(axis=1).astype(float)
    isc = ((hi - lo) + 20 * (lo - Y) * (Y < lo) + 20 * (Y - hi) * (Y > hi)).mean(axis=1)
    err = np.linalg.norm(Q[:, :, 2] - Y, axis=1)
    return {"joint90": j90, "is90": isc, "err": err, "width90": (hi - lo).mean(axis=1)}


def boot(df, metric, a, b, rng):
    """Paired person bootstrap of mean(a) - mean(b), persons resampled within each scene."""
    d = df[df.model == a].set_index(["scene", "row"])[metric] - df[df.model == b].set_index(["scene", "row"])[metric]
    d = d.groupby(level=[0, 1]).mean().reset_index(name="d").merge(
        df[df.model == a][["scene", "row", "person"]].drop_duplicates(), on=["scene", "row"])
    per = d.groupby(["scene", "person"]).d.agg(["sum", "count"]).reset_index()
    draws = []
    for _ in range(BOOT):
        tot = cnt = 0.0
        for _, g in per.groupby("scene"):
            idx = rng.integers(0, len(g), len(g))
            tot += g["sum"].to_numpy()[idx].sum(); cnt += g["count"].to_numpy()[idx].sum()
        draws.append(tot / cnt)
    return float(d.d.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    data = {s: load(s) for s in SCENES}
    for s, d in data.items():
        print(f"{s}: {len(d)} rows at h={H}, {d.person.nunique()} people", flush=True)
    recs, calls = [], 0
    for test in SCENES:
        te = data[test].sample(min(N_TEST, len(data[test])), random_state=0).reset_index(drop=True)
        Xt, Yt = F.X_of(te), te[["ddx", "ddy"]].to_numpy(float)
        train = [s for s in SCENES if s != test]
        pool = pd.concat([data[s] for s in train], ignore_index=True)
        fixed = {"cv": M.make("cv").fit(F.X_of(pool), pool[["ddx", "ddy"]].to_numpy(float)),
                 "lgbm_all": M.make("lgbm:m50:n100").fit(F.X_of(pool), pool[["ddx", "ddy"]].to_numpy(float))}
        for seed in SEEDS:
            ctx = pd.concat([data[s].sample(N_CTX // len(train), random_state=seed) for s in train], ignore_index=True)
            Xc, Yc = F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy(float)
            models = dict(fixed)
            models["cv_band"] = M.make("cv_band").fit(Xc, Yc)
            models["lgbm"] = M.make("lgbm:m50:n100").fit(Xc, Yc)
            models["tabpfn"] = (M.make("lgbm:m50:n100") if a.dry else M.make("tabpfn", api=True)).fit(Xc, Yc)
            calls += 0 if a.dry else 2
            for name, m in models.items():
                sc = scores(m.predict_quantiles(Xt, QS), Yt)
                recs.append(pd.DataFrame({"scene": test, "seed": seed, "model": name, "row": np.arange(len(te)),
                                          "person": te.person, **sc}))
            print(f"held out {test} seed {seed}: done ({calls} API calls so far)", flush=True)
    df = pd.concat(recs, ignore_index=True)
    out = Path("results") if not a.dry else Path("results/dry"); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "loso5_rows.csv", index=False)
    rng = np.random.default_rng(0)
    summ = []
    lvl = df.groupby(["scene", "model"])[["joint90", "is90", "err", "width90"]].mean().reset_index()
    print(lvl.pivot(index="model", columns="scene", values="is90").round(3).to_string())
    print(lvl.pivot(index="model", columns="scene", values="joint90").round(3).to_string())
    for rival in ("lgbm_all", "cv_band", "lgbm", "cv"):
        for scope in SCENES + ["pooled"]:
            sub = df if scope == "pooled" else df[df.scene == scope]
            for metric in ("is90", "joint90", "err"):
                d, lo, hi = boot(sub, metric, "tabpfn", rival, rng)
                summ.append({"comparison": f"tabpfn - {rival}", "scope": scope, "metric": metric,
                             "diff": d, "lo": lo, "hi": hi})
    S = pd.DataFrame(summ)
    S.to_csv(out / "loso5.csv", index=False)
    print(S[S.metric == "is90"].round(3).to_string(index=False))
    pooled = S[(S.scope == "pooled") & (S.metric == "is90")].set_index("comparison")
    per = S[(S.scope != "pooled") & (S.metric == "is90")]
    pooled_ok = all(pooled.loc[f"tabpfn - {r}", "hi"] < 0 for r in ("lgbm_all", "cv_band"))
    worse = per[per.comparison.isin(["tabpfn - lgbm_all", "tabpfn - cv_band"]) & (per.lo > 0)]
    verdict = ("holds on five scenes" if pooled_ok and worse.empty else
               "direction-dependent" if pooled_ok else "not supported")
    print(f"VERDICT (docs/LOSO5.md rule): {verdict}; significantly worse cells: "
          f"{worse[['comparison', 'scope']].values.tolist()}; API calls {calls}")


if __name__ == "__main__":
    main()
