"""Claim B on REAL data with the final live design: does the scene model help in a place never seen?

    python tools/realdata_claimb.py --count-only      # moments and API calls, no calls
    python tools/realdata_claimb.py                   # both directions, hosted TabPFN (default model)

Directions Hotel -> ETH and ETH -> Hotel. Every test episode; a moment every 2 s; queries at h = 2 s (the f4 row
caches, noise 0). Scene rows are labelled from observed (Kalman-filtered) positions and usable once t_known <= t.
Variants:
  frozen    TabPFN (hosted, default model) on 2,000 bank rows at natural move frequency (context.frozen), fitted once
  split     average of frozen and a recent-only scene model (TabPFN, hosted) on <= 250 most recent completed rows of
            this episode, quantiles averaged per axis then sorted; frozen alone while fewer than 20 recent rows
  lgbm_all  LightGBM quantile (300 trees, n_jobs 2) trained once on every bank row
  <v>+offset  the 5-95% box widened by the batch-rule online offset as in T5: at each moment, every earlier
            prediction whose outcome is now known (t + 2 <= now) is scored; d += 0.05 (batch miss rate - 0.2); d0 = 0
Metrics per query row: error of the median, joint coverage of the 5-95% box, width (axes averaged), interval score
(alpha 0.1). Bins: cold start (t < 5 s in the scene) and from 5 s on. Paired person bootstrap (5,000) of split minus
frozen and split minus lgbm_all, from 5 s on, per direction and pooled.
API: 2 calls per TabPFN fit (one per axis); counted first, logged to results/api_usage.csv with server usage.
Writes results/realdata_claimb_rows.csv and results/realdata_claimb.csv.
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import live_variants as LV  # noqa: E402
from foresight.core import context as C  # noqa: E402
from foresight.core import conformal as CF  # noqa: E402
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from foresight.core import schema  # noqa: E402

QS = (0.05, 0.1, 0.5, 0.9, 0.95)
H, REC_CAP, THIN, EVERY, COLD = 2.0, 250, 20, 2.0, 5.0
ALPHA, ETA_B = 0.2, 0.05
CREDITS_PER_CALL = 10_000
USAGE = ROOT / "results/api_usage.csv"


def direction(src, dst):
    bank = LV.cached_rows(f"data/{src}", 0.0, (1.2, 2.0, 4.8))
    test = LV.cached_rows(f"data/{dst}", 0.0, (1.2, 2.0, 4.8)).copy()
    tracks, obst, meta = schema.load(ROOT / f"data/{dst}")
    F.load_kalman(ROOT / f"data/{dst}", meta)
    obs = F.build_rows(tracks, obst, meta, horizons=(H,), noise=0.0, seed=0, label_from="observed")
    moments = {}
    for ep, g in test.groupby("episode"):
        ts = sorted(g.t.unique())
        step = max(1, int(round(EVERY / np.median(np.diff(ts))))) if len(ts) > 1 else 1
        moments[ep] = ts[::step]
    return bank, test, obs, moments


def plan(obs, moments):
    n_split = 0
    for ep, ts in moments.items():
        live = obs[obs.episode == ep]
        n_split += sum(int((live.t_known <= t).sum() >= THIN) for t in ts)
    return n_split


def tabpfn_api():
    return M.make("tabpfn", api=True)


def batch_offsets(t, episode, lo, hi, Y):
    """Batch rule, causal: at each moment of an episode, score earlier rows with t + H <= now; one step per moment."""
    d_used = np.zeros(len(t)); s = CF.scores(lo, hi, Y)
    for ep in np.unique(episode):
        idx = np.flatnonzero(episode == ep)
        d, scored = 0.0, np.zeros(len(idx), bool)
        for tm in np.unique(t[idx]):
            due = (~scored) & (t[idx] + H <= tm + 1e-9)
            if due.any():
                miss = s[idx][due] > d_used[idx][due]
                d += ETA_B * (float(miss.mean()) - ALPHA)
                scored |= due
            d_used[idx[np.isclose(t[idx], tm)]] = d
    return d_used


def metrics(lo, hi, Y, Q50):
    cov = ((Y >= lo) & (Y <= hi)).all(axis=1).astype(float)
    w = (hi - lo).mean(axis=1)
    isc = ((hi - lo) + 20 * np.maximum(lo - Y, 0) + 20 * np.maximum(Y - hi, 0)).mean(axis=1)
    return dict(err=np.linalg.norm(Q50 - Y, axis=1), joint90=cov, width=w, is90=isc)


def run_direction(src, dst, api):
    bank, test, obs, moments = direction(src, dst)
    q_all = pd.concat([test[(test.episode == ep) & test.t.isin(ts)] for ep, ts in moments.items()], ignore_index=True)
    Xq, Y = F.X_of(q_all), q_all[["ddx", "ddy"]].to_numpy()
    ctx = C.frozen(bank, 2000)
    fz = (tabpfn_api() if api else LV.tabpfn()).fit(F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy())
    Qf = fz.predict_quantiles(Xq, QS)
    calls = 2 if api else 0
    lg = M.make("lgbm", n_jobs=2).fit(F.X_of(bank), bank[["ddx", "ddy"]].to_numpy())
    Ql = lg.predict_quantiles(Xq, QS)
    Qs = Qf.copy(); n_rec = np.zeros(len(q_all), int)
    for ep, ts in moments.items():
        live = obs[obs.episode == ep]
        for t in ts:
            sel = np.flatnonzero((q_all.episode == ep).to_numpy() & np.isclose(q_all.t, t))
            rec = live[live.t_known <= t + 1e-9].sort_values("t_known").tail(REC_CAP)
            n_rec[sel] = len(rec)
            if len(rec) < THIN or not len(sel):
                continue
            calls += 2 if api else 0
            m = (tabpfn_api() if api else LV.tabpfn()).fit(F.X_of(rec), rec[["ddx", "ddy"]].to_numpy())
            Qs[sel] = np.sort((Qf[sel] + m.predict_quantiles(Xq[sel], QS)) / 2, axis=2)
        print(f"{src}->{dst} episode {ep}: {len(ts)} moments done (calls so far {calls})", flush=True)
    frames = []
    for name, Q in (("frozen", Qf), ("split", Qs), ("lgbm_all", Ql)):
        lo, hi = Q[:, :, 0], Q[:, :, 4]
        d = batch_offsets(q_all.t.to_numpy(float), q_all.episode.to_numpy(), lo, hi, Y)
        for v, (l2, h2) in ((name, (lo, hi)), (name + "+offset", CF.widen(lo, hi, d))):
            f = pd.DataFrame(dict(direction=f"{src}->{dst}", variant=v, episode=q_all.episode.to_numpy(),
                                  agent_id=q_all.agent_id.to_numpy(), t=q_all.t.to_numpy(), n_recent=n_rec,
                                  d_used=d if v.endswith("offset") else 0.0, cv_err=np.linalg.norm(Y, axis=1)))
            for k, val in metrics(l2, h2, Y, Q[:, :, 2]).items():
                f[k] = val
            frames.append(f)
    return pd.concat(frames, ignore_index=True), calls


def person_boot(R, a, b, cols, B=5000, seed=0):
    A = R[R.variant == a].reset_index(drop=True); Bv = R[R.variant == b].reset_index(drop=True)
    assert len(A) == len(Bv)
    person = (A.direction + ":" + A.episode.astype(str) + ":" + A.agent_id.astype(str)).to_numpy()
    D = A[cols].to_numpy(float) - Bv[cols].to_numpy(float)
    ups, inv = np.unique(person, return_inverse=True)
    sums = np.zeros((len(ups), len(cols))); np.add.at(sums, inv, D); cnt = np.bincount(inv)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        i = rng.integers(0, len(ups), len(ups)); bs.append(sums[i].sum(0) / cnt[i].sum())
    lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
    return D.mean(0), lo, hi, len(ups), len(A)


def log_usage(calls, note):
    day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    U = pd.read_csv(USAGE)
    prev = pd.to_numeric(U.loc[(U.date_utc == day) & (U.run != "day total (server)"), "credits"], errors="coerce").sum()
    with open(USAGE, "a", newline="", encoding="utf-8") as f:
        f.write(f"{day},realdata_claimb (tools/realdata_claimb.py),{calls},{calls * CREDITS_PER_CALL},"
                f"{note.replace(',', ';')}; running total {day}: {int(prev + calls * CREDITS_PER_CALL)} credits\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--local", action="store_true", help="local TabPFN 3.5-Fast 1 estimator instead of the API")
    a = ap.parse_args()
    dirs = (("hotel", "eth"), ("eth", "hotel"))
    n = 0
    for s, d in dirs:
        _, test, obs, moments = direction(s, d)
        k = plan(obs, moments); n += 2 + 2 * k
        print(f"{s}->{d}: {sum(len(v) for v in moments.values())} moments in {len(moments)} episodes, "
              f"{k} split fits -> {2 + 2 * k} API calls", flush=True)
    print(f"total API calls {n} = {n * CREDITS_PER_CALL:,} credits", flush=True)
    if a.count_only:
        return
    api = not a.local
    before = after = "?"
    if api:
        if not os.environ.get("TABPFN_TOKEN"):
            import subprocess
            os.environ["TABPFN_TOKEN"] = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", "[Environment]::GetEnvironmentVariable('TABPFN_TOKEN','User')"],
                capture_output=True, text=True).stdout.strip()
        import tabpfn_client
        tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
        before = tabpfn_client.get_api_usage(); print("server before:", before, flush=True)
    parts, calls, err = [], 0, None
    try:
        for s, d in dirs:
            R, c = run_direction(s, d, api); parts.append(R); calls += c
    except Exception as e:
        err = repr(e)[:200]; print("stopped on error:", err, flush=True)
    if api:
        import tabpfn_client
        after = tabpfn_client.get_api_usage(); print("server after:", after, flush=True)
        used = lambda x: x.split("used ")[1].split(" of")[0] if isinstance(x, str) and "used " in x else str(x)  # noqa: E731
        log_usage(calls, f"real-data claim B (frozen 2k + recent-only scene model; default model); server usage "
                         f"{used(before)} -> {used(after)}" + (f"; stopped on error {err}" if err else ""))
    if not parts:
        return
    R = pd.concat(parts, ignore_index=True)
    R.to_csv(ROOT / "results/realdata_claimb_rows.csv", index=False)
    cols = ["err", "joint90", "width", "is90"]
    out = []
    for scope, sub in [("pooled", R)] + [(dname, R[R.direction == dname]) for dname in R.direction.unique()]:
        for win, m in (("cold <5 s", sub.t < COLD), ("from 5 s", sub.t >= COLD)):
            S = sub[m]
            for v in S.variant.unique():
                out.append(dict(scope=scope, window=win, comparison=v, n_rows=int((S.variant == v).sum()),
                                **{c: float(S[S.variant == v][c].mean()) for c in cols}))
            if win == "from 5 s":
                for a_, b_ in (("split", "frozen"), ("split", "lgbm_all"), ("split+offset", "frozen+offset"),
                               ("split+offset", "lgbm_all+offset")):
                    dm, lo, hi, npers, nr = person_boot(S, a_, b_, cols)
                    for j, c in enumerate(cols):
                        out.append(dict(scope=scope, window=win, comparison=f"{a_} - {b_}", n_rows=nr, persons=npers,
                                        metric=c, diff=dm[j], lo=lo[j], hi=hi[j]))
    T = pd.DataFrame(out)
    T.to_csv(ROOT / "results/realdata_claimb.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
