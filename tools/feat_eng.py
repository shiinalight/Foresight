"""Feature engineering on the five-scene leave-one-out test (plan: docs/FEATURES_ENG.md, fixed before any run).

`engineer(df, hist_dt)` ADDS interaction and kinematic properties computed only from the 39 f4 columns of the
same row (foresight/core/features.py). It reads no other row and no future, and the core is unchanged.

    python tools/feat_eng.py --check        # engineered-feature distributions per scene, NaN/inf check (no models)
    python tools/feat_eng.py --count-only   # count the hosted calls
    python tools/feat_eng.py --dry          # LightGBM stands in for TabPFN (no API), checks the plumbing
    python tools/feat_eng.py                # hosted TabPFN (tabpfn-client, TABPFN_TOKEN)

Frame facts used here (features.py, f4):
- every x/y is in the predicted agent's frame at time t: x = its heading, y = its left;
- n{k}_x, n{k}_y = neighbour k's position minus the agent's; n{k}_vx, n{k}_vy = neighbour velocity MINUS the
  agent's (relative), k = 0..2 by distance, the robot excluded;
- a missing neighbour is padded x = 99, y = 0, vx = vy = 0, dist = 99 (FAR);
- hist_{x,y}{j} = position at t - j * step minus the position at t, j = 1 (most recent) .. 5 (oldest).
  step = max(1, round(0.2 / dt)) * dt: 0.4 s on ETH/UCY (dt 0.4), 0.2 s on the simulator (dt 0.1);
- robot_* padded (99, 0, 99, 0) when there is no robot; obst_* padded (99, 0, 99).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402

FAR = F.FAR          # 99: "no such neighbour / none" for distances and speeds (the raw padding convention)
T_MAX = 5.0          # s: TCPA clip; "no encounter" for times
THREAT_T = 2.0       # s: a neighbour is a threat if it is approaching with closest approach within 2 s
CONE_DEG, CONE_R = 45.0, 3.0
COMP_R, COMP_V = 1.5, 0.5
SIGMA_D, TAU_T = 1.0, 2.0   # risk kernel: 1 m, 2 s
MIN_PATH, MIN_CHORD = 0.2, 0.1   # m: history gates for straightness / curvature / heading change

ENG = ([f"n{k}_{a}" for k in range(F.N_NEIGH) for a in ("tcpa", "dcpa")]
       + ["threat_dcpa", "threat_tcpa", "threat_side", "risk", "robot_ttc",
          "ahead_n", "ahead_dist", "ahead_vx", "comp_relspeed", "comp_n",
          "speed_trend", "path_speed", "heading_change", "straightness", "curvature"])
BASE = ["h_f"] + F.feature_names()[1:]                       # the 39 f4 model columns, F.X_of order
KEEP_COMPACT = ["h_f", "speed", "a_lon", "a_lat", "turn_rate", "radius", "n0_dist", "n1_dist", "n2_dist",
                "robot_dist", "robot_closing", "obst_dist", "density_3m"]
VARIANTS = {"f4": BASE, "eng": BASE + ENG, "compact": KEEP_COMPACT + ENG}


def engineer(df: pd.DataFrame, hist_dt: float = 0.4) -> pd.DataFrame:
    """Return a copy of df with the ENG columns added. hist_dt = seconds between history samples (0.4 on ETH/UCY)."""
    out = df.copy()
    tc, dc, raw_t, cpa_y, real = [], [], [], [], []
    for k in range(F.N_NEIGH):
        px, py = df[f"n{k}_x"].to_numpy(float), df[f"n{k}_y"].to_numpy(float)
        wx, wy = df[f"n{k}_vx"].to_numpy(float), df[f"n{k}_vy"].to_numpy(float)
        ok = df[f"n{k}_dist"].to_numpy(float) < FAR
        w2 = wx * wx + wy * wy
        raw = np.where(w2 > 1e-6, -(px * wx + py * wy) / np.maximum(w2, 1e-6), 0.0)   # no relative motion: TCPA 0
        t = np.clip(raw, 0.0, T_MAX)
        cx, cy = px + wx * t, py + wy * t                     # neighbour's relative position at closest approach
        d = np.hypot(cx, cy)
        out[f"n{k}_tcpa"] = np.where(ok, t, T_MAX)
        out[f"n{k}_dcpa"] = np.where(ok, d, FAR)
        tc.append(np.where(ok, t, T_MAX)); dc.append(np.where(ok, d, FAR))
        raw_t.append(np.where(ok, raw, -1.0)); cpa_y.append(np.where(ok, cy, 0.0)); real.append(ok)
    tc, dc, raw_t, cpa_y, real = map(np.stack, (tc, dc, raw_t, cpa_y, real))       # [3, n]
    # threat: approaching neighbours (raw TCPA > 0) that reach closest approach within THREAT_T
    thr = real & (raw_t > 0) & (raw_t <= THREAT_T)
    dthr = np.where(thr, dc, np.inf)
    j = np.argmin(dthr, axis=0); has = np.isfinite(dthr.min(axis=0)); cols = np.arange(len(df))
    out["threat_dcpa"] = np.where(has, dc[j, cols], FAR)
    out["threat_tcpa"] = np.where(has, tc[j, cols], T_MAX)
    out["threat_side"] = np.where(has, cpa_y[j, cols], 0.0)   # + passes on the left, - on the right
    appr = real & (raw_t > 0)
    out["risk"] = np.where(appr, np.exp(-dc ** 2 / (2 * SIGMA_D ** 2)) * np.exp(-tc / TAU_T), 0.0).sum(axis=0)
    rd, rc = df["robot_dist"].to_numpy(float), df["robot_closing"].to_numpy(float)
    out["robot_ttc"] = np.where((rd < FAR) & (rc > 0.05), np.clip(rd / np.maximum(rc, 0.05), 0, T_MAX), T_MAX)
    # blocked ahead: neighbours in the +-45 deg forward cone within 3 m (only the 3 nearest are known)
    nx = np.stack([df[f"n{k}_x"].to_numpy(float) for k in range(F.N_NEIGH)])
    ny = np.stack([df[f"n{k}_y"].to_numpy(float) for k in range(F.N_NEIGH)])
    nvx = np.stack([df[f"n{k}_vx"].to_numpy(float) for k in range(F.N_NEIGH)])
    nvy = np.stack([df[f"n{k}_vy"].to_numpy(float) for k in range(F.N_NEIGH)])
    nd = np.stack([df[f"n{k}_dist"].to_numpy(float) for k in range(F.N_NEIGH)])
    cone = real & (nx > 0) & (np.abs(ny) <= nx * np.tan(np.radians(CONE_DEG))) & (nd <= CONE_R)
    out["ahead_n"] = cone.sum(axis=0).astype(float)
    dcone = np.where(cone, nd, np.inf); jc = np.argmin(dcone, axis=0); hc = cone.any(axis=0)
    out["ahead_dist"] = np.where(hc, nd[jc, cols], FAR)
    out["ahead_vx"] = np.where(hc, nvx[jc, cols], 0.0)        # < 0: the one ahead is slower (closing)
    # companions: neighbours within 1.5 m, by relative speed
    rs = np.hypot(nvx, nvy); near = real & (nd <= COMP_R)
    out["comp_relspeed"] = np.where(near.any(axis=0), np.where(near, rs, np.inf).min(axis=0), FAR)
    out["comp_n"] = (near & (rs < COMP_V)).sum(axis=0).astype(float)
    # history: P0 = (0, 0) now, Pj = hist_j (older with j); seg_j = P_{j-1} - P_j points forward in time
    P = np.stack([np.zeros((len(df), 2))] + [df[[f"hist_x{q}", f"hist_y{q}"]].to_numpy(float)
                                              for q in range(1, F.N_HIST + 1)])      # [6, n, 2]
    seg = P[:-1] - P[1:]                                       # [5, n, 2], seg[0] newest
    sl = np.hypot(seg[..., 0], seg[..., 1])
    path = sl.sum(axis=0); net = np.hypot(P[-1, :, 0], P[-1, :, 1])
    win = F.N_HIST * hist_dt
    out["path_speed"] = path / win
    out["speed_trend"] = df["speed"].to_numpy(float) - (sl[-2] + sl[-1]) / (2 * hist_dt)
    new, old = seg[0] + seg[1], seg[-2] + seg[-1]              # newest and oldest two segments
    ok_h = (np.hypot(*new.T) >= MIN_CHORD) & (np.hypot(*old.T) >= MIN_CHORD)
    dth = np.arctan2(new[:, 1], new[:, 0]) - np.arctan2(old[:, 1], old[:, 0])
    dth = (dth + np.pi) % (2 * np.pi) - np.pi
    out["heading_change"] = np.where(ok_h, dth, 0.0)          # + turned left over the window
    out["straightness"] = np.where(path >= MIN_PATH, net / np.maximum(path, 1e-9), 1.0)
    out["curvature"] = np.where(ok_h & (path >= MIN_PATH), np.clip(dth / np.maximum(path, 1e-9), -5, 5), 0.0)
    return out


# ---------------------------------------------------------------- experiment (same design as tools/loso5.py)
QS = (0.05, 0.1, 0.5, 0.9, 0.95)
CAP_CREDITS = 3_000_000
USAGE = ROOT / "results/api_usage.csv"
CACHE = ROOT / "results/feat_eng_cache"
NEW = ("eng", "compact")


def _token():
    if not os.environ.get("TABPFN_TOKEN"):
        import subprocess
        os.environ["TABPFN_TOKEN"] = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", "[Environment]::GetEnvironmentVariable('TABPFN_TOKEN','User')"],
            capture_output=True, text=True).stdout.strip()
    import tabpfn_client
    tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
    return tabpfn_client


def _used(tc):
    s = tc.get_api_usage()
    return int(s.split("used ")[1].split(" of")[0])


def check(data):
    print(f"engineered columns ({len(ENG)}): {ENG}")
    for s, d in data.items():
        E = d[ENG]
        bad = (~np.isfinite(E.to_numpy(float))).sum()
        q = E.quantile([0.05, 0.5, 0.95]).T
        q["frac_sentinel"] = [((E[c] >= FAR) | ((E[c] == T_MAX) & c.endswith("tcpa"))).mean() if
                              (c.endswith("dcpa") or c.endswith("tcpa") or c in ("ahead_dist", "comp_relspeed"))
                              else np.nan for c in ENG]
        q["mean_real"] = [E[c][(E[c] < FAR)].mean() for c in ENG]
        print(f"\n== {s}: {len(d)} rows, NaN/inf cells {bad}")
        print(q.round(3).to_string())
        assert bad == 0, f"{s}: non-finite engineered values"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--count-only", action="store_true")
    a = ap.parse_args()
    from tools.loso5 import BOOT, H, N_CTX, N_TEST, SCENES, SEEDS, boot, load, scores  # noqa: F401
    data = {s: engineer(load(s), hist_dt=0.4) for s in SCENES}
    if a.check:
        check(data); return
    n_calls = len(NEW) * len(SCENES) * len(SEEDS) * 2
    print(f"hosted calls planned: {len(NEW)} variants x {len(SCENES)} scenes x {len(SEEDS)} seeds x 2 axes = {n_calls}"
          f" (f4 reused from results/loso5_rows.csv if the rows match)", flush=True)
    if a.count_only:
        return
    # ---- baseline reuse check: same test rows, same persons, cv and lgbm_all reproduce
    old = pd.read_csv(ROOT / "results/loso5_rows.csv")
    tests = {}
    for s in SCENES:
        te = data[s].sample(min(N_TEST, len(data[s])), random_state=0).reset_index(drop=True)
        tests[s] = te
        o = old[(old.scene == s) & (old.model == "tabpfn") & (old.seed == 0)].sort_values("row")
        assert len(o) == len(te) and (o.person.to_numpy() == te.person.to_numpy()).all(), f"{s}: test rows differ"
    print("baseline: test rows and persons match results/loso5_rows.csv", flush=True)
    tc = None
    before = after = None
    if not a.dry:
        tc = _token(); before = _used(tc); print(f"server usage before: {before:,}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    recs, calls, gains, err = [], 0, [], None
    first_used = None
    try:
        for test in SCENES:
            te = tests[test]; Yt = te[["ddx", "ddy"]].to_numpy(float)
            train = [s for s in SCENES if s != test]
            pool = pd.concat([data[s] for s in train], ignore_index=True)
            Yp = pool[["ddx", "ddy"]].to_numpy(float)
            for v, cols in VARIANTS.items():
                m = M.make("lgbm:m50:n100").fit(pool[cols].to_numpy(float), Yp)
                sc = scores(m.predict_quantiles(te[cols].to_numpy(float), QS), Yt)
                for (ax, q), mm in m.models.items():
                    g = mm.booster_.feature_importance("gain"); g = g / max(g.sum(), 1e-12)
                    gains.append(pd.DataFrame({"variant": v, "scene": test, "axis": ax, "q": q, "feature": cols,
                                               "gain_share": g}))
                if v == "f4":
                    o = old[(old.scene == test) & (old.model == "lgbm_all") & (old.seed == 0)].sort_values("row")
                    dmax = float(np.abs(o.is90.to_numpy() - sc["is90"]).max())
                    print(f"  {test}: lgbm_all f4 rerun vs stored is90 max |diff| {dmax:.2e}", flush=True)
                for seed in SEEDS:
                    recs.append(pd.DataFrame({"scene": test, "seed": seed, "model": f"lgbm_all_{v}",
                                              "row": np.arange(len(te)), "person": te.person, **sc}))
            for seed in SEEDS:
                o = old[(old.scene == test) & (old.model == "tabpfn") & (old.seed == seed)].sort_values("row")
                recs.append(pd.DataFrame({"scene": test, "seed": seed, "model": "tabpfn_f4", "row": np.arange(len(te)),
                                          "person": te.person, **{k: o[k].to_numpy() for k in
                                                                  ("joint90", "is90", "err", "width90")}}))
                ctx = pd.concat([data[s].sample(N_CTX // len(train), random_state=seed) for s in train],
                                ignore_index=True)
                Yc = ctx[["ddx", "ddy"]].to_numpy(float)
                for v in NEW:
                    cols = VARIANTS[v]
                    f = CACHE / f"{'dry_' if a.dry else ''}{v}_{test}_{seed}.npy"
                    if f.exists():
                        Q = np.load(f)
                    else:
                        mdl = M.make("lgbm:m50:n100") if a.dry else M.make("tabpfn", api=True)
                        Q = mdl.fit(ctx[cols].to_numpy(float), Yc).predict_quantiles(te[cols].to_numpy(float), QS)
                        np.save(f, Q)
                        if not a.dry:
                            calls += 2
                            if first_used is None:
                                first_used = _used(tc) - before
                                proj = first_used / 2 * n_calls
                                print(f"  first fit: server delta {first_used:,} for 2 calls -> projected {proj:,.0f}"
                                      f" (cap {CAP_CREDITS:,})", flush=True)
                                if proj > CAP_CREDITS:
                                    raise RuntimeError(f"projected spend {proj:,.0f} above cap; stopping")
                    recs.append(pd.DataFrame({"scene": test, "seed": seed, "model": f"tabpfn_{v}",
                                              "row": np.arange(len(te)), "person": te.person, **scores(Q, Yt)}))
                print(f"held out {test} seed {seed}: done ({calls} hosted calls this run)", flush=True)
    except Exception as e:
        err = repr(e)[:200]; print("stopped on error:", err, flush=True)
    if not a.dry:
        after = _used(tc); print(f"server usage after: {after:,}", flush=True)
        day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        U = pd.read_csv(USAGE)
        prev = pd.to_numeric(U.loc[U.date_utc == day, "credits"], errors="coerce").sum()
        cr = after - before
        note = (f"feature engineering (eng + compact vs f4) on the five-scene leave-one-out; TabPFN regressor 1500 x "
                f"{len(VARIANTS['eng'])}/{len(VARIANTS['compact'])} context; 1500 test rows; server usage {before} -> "
                f"{after} (account may be shared with other sessions); credits column = server delta"
                + (f"; stopped on error {err}" if err else "") + f"; running total {day}: {int(prev + cr)} credits")
        with open(USAGE, "a", newline="", encoding="utf-8") as fh:
            fh.write(f"{day},feat_eng (tools/feat_eng.py),{calls},{cr},{note.replace(',', ';')}\n")
    if err:
        return
    df = pd.concat(recs, ignore_index=True)
    out = ROOT / ("results/dry" if a.dry else "results"); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "feat_eng_rows.csv", index=False)
    G = pd.concat(gains, ignore_index=True)
    G.groupby(["variant", "feature"]).gain_share.mean().reset_index().sort_values(
        ["variant", "gain_share"], ascending=[True, False]).to_csv(out / "feat_eng_gain.csv", index=False)
    lvl = df.groupby(["scene", "model"])[["joint90", "is90", "err", "width90"]].mean().reset_index()
    pooled = df.groupby("model")[["joint90", "is90", "err", "width90"]].mean()
    for met in ("is90", "joint90", "err", "width90"):
        t = lvl.pivot(index="model", columns="scene", values=met)[SCENES]; t["pooled"] = pooled[met]
        print(f"\n{met} (mean over 3 seeds)\n{t.round(3).to_string()}")
    rng = np.random.default_rng(0)
    comps = ([(f"tabpfn_{v}", "tabpfn_f4") for v in NEW] + [(f"tabpfn_{v}", f"lgbm_all_{v}") for v in VARIANTS]
             + [(f"lgbm_all_{v}", "lgbm_all_f4") for v in NEW])
    summ = []
    for A_, B_ in comps:
        for scope in SCENES + ["pooled"]:
            sub = df if scope == "pooled" else df[df.scene == scope]
            for met in ("is90", "joint90", "err"):
                d, lo, hi = boot(sub, met, A_, B_, rng)
                summ.append({"comparison": f"{A_} - {B_}", "scope": scope, "metric": met, "diff": d, "lo": lo, "hi": hi})
    S = pd.DataFrame(summ); S.to_csv(out / "feat_eng.csv", index=False)
    print("\n" + S[S.metric == "is90"].round(3).to_string(index=False))
    for v in NEW:
        c = S[(S.comparison == f"tabpfn_{v} - tabpfn_f4") & (S.metric == "is90")]
        p = c[c.scope == "pooled"].iloc[0]; worse = c[(c.scope != "pooled") & (c.lo > 0)].scope.tolist()
        ok = p.hi < 0 and not worse
        print(f"VERDICT {v} (docs/FEATURES_ENG.md rule): {'ADOPTED' if ok else 'not adopted'}; pooled "
              f"{p['diff']:+.3f} [{p.lo:+.3f}, {p.hi:+.3f}]; significantly worse scenes: {worse}")
    top = G[G.variant == "eng"].groupby("feature").gain_share.mean().sort_values(ascending=False)
    print("\nLightGBM (all rows, eng) top features by mean gain share:\n" + top.head(15).round(4).to_string())
    print(f"hosted calls this run {calls}")


if __name__ == "__main__":
    main()
