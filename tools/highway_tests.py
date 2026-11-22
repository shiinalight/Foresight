"""Same core, another embodiment: highway cars (plan and pass rules: docs/HIGHWAY.md, fixed before any TabPFN run).

    python tools/highway_tests.py --sanity        # constant-velocity error and label shares, no models
    python tools/highway_tests.py --count-only    # API call plan, no calls
    python tools/highway_tests.py --dry           # LightGBM stands in for TabPFN (no API), checks the plumbing
    python tools/highway_tests.py                 # hosted TabPFN (tabpfn-client, TABPFN_TOKEN), H1 + H2 + H2b
    python tools/highway_tests.py --figure        # media/fig_highway.png from the saved H2 quantiles

Data: data/highway (foresight/domains/highway.py, 72 episodes), f4 rows, noise 0.1, Kalman from kalman.json (tuned on
the seed-0 training split), h = 2 s. Episode split = split_by_episode(seed 0), the split the filter was tuned on.
  H1   context 1,500 random training rows x 3 seeds; 1,500 random test rows (random_state 0). cv, cv_band, lgbm
       m50:n100 (same rows), lgbm_all (m50:n100, every training row), TabPFN.
  H2   bank = training episodes with no 'aggressive' regime (before or after). Test = held-out episodes that switch
       to 'aggressive'; moments every 2 s after the switch (first row time after the switch, then +2 s, ...), 30 drawn
       at random (seed 0); queries = every car at those moments. frozen = TabPFN on context.frozen(bank, 2000);
       split = frozen and a scene model (TabPFN on that episode's latest <= 250 completed rows, t_known <= t,
       labelled from observed positions) averaged per quantile then sorted (run_closed.AvgQuantiles). Also scored:
       scene-only, lgbm_all (bank), cv, cv_band (bank).
  H2b  ped = TabPFN on 2,000 crowd_cross_t2 training rows (context.frozen); ped+scene = ped averaged with H2's scene
       model (same stored quantiles). Exploratory.
Metrics per query row: joint coverage of the 5-95% box, 90% interval score (alpha 0.1), median error, 5-95% width.
Paired vehicle bootstrap (2,000; vehicle = episode:agent_id).
Writes results/highway_h1.csv, results/highway_h2.csv (+ _rows.csv), results/highway_h2_q.npz; logs API use.
"""
from __future__ import annotations

import argparse
import datetime as dtm
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from foresight.core import context as C  # noqa: E402
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from foresight.core import schema  # noqa: E402
from run_offline import rows_for  # noqa: E402

DATA, PED = "data/highway", "data/crowd_cross_t2"
QS = (0.05, 0.1, 0.5, 0.9, 0.95)
H, NOISE = 2.0, 0.1
N_TEST, N_CTX, SEEDS, BOOT = 1500, 1500, (0, 1, 2), 2000
BANK, REC_CAP, THIN, EVERY, N_MOM = 2000, 250, 20, 2.0, 30
UNSEEN = "aggressive"
CREDITS_PER_CALL, CAP_CREDITS = 30_000, 3_000_000
USAGE = ROOT / "results/api_usage.csv"


def rows(data):
    d = rows_for(data, NOISE, horizons=(1.0, 2.0))
    return d


def scores(Q, Y):
    lo, hi = Q[:, :, 0], Q[:, :, 4]
    j90 = ((Y >= lo) & (Y <= hi)).all(axis=1).astype(float)
    isc = ((hi - lo) + 20 * (lo - Y) * (Y < lo) + 20 * (Y - hi) * (Y > hi)).mean(axis=1)
    return {"joint90": j90, "is90": isc, "err": np.linalg.norm(Q[:, :, 2] - Y, axis=1),
            "width90": (hi - lo).mean(axis=1)}


def vboot(df, metric, a, b, seed=0):
    """Paired vehicle bootstrap of mean(a) - mean(b): rows matched on `key`, vehicles resampled with replacement."""
    A = df[df.model == a].sort_values("key"); Bm = df[df.model == b].sort_values("key")
    assert (A.key.to_numpy() == Bm.key.to_numpy()).all()
    d = A[metric].to_numpy() - Bm[metric].to_numpy()
    ups, inv = np.unique(A.vehicle.to_numpy(), return_inverse=True)
    s = np.bincount(inv, weights=d); c = np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ups), (BOOT, len(ups)))
    st = s[idx].sum(1) / c[idx].sum(1)
    return float(d.mean()), float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5)), len(ups)


def tabpfn(dry):
    return M.make("lgbm:m50:n100") if dry else M.make("tabpfn", api=True)


# ---------------- data ----------------
def setup():
    df = rows(DATA)
    d = df[np.isclose(df.h, H)].dropna(subset=["ddx", "ddy"]).copy()
    d["vehicle"] = d.episode.astype(str) + ":" + d.agent_id.astype(str)
    tr, _, te = F.split_by_episode(d, seed=0)
    return df, d[tr].reset_index(drop=True), d[te].reset_index(drop=True)


def h2_setup(train, test):
    meta = json.loads((ROOT / DATA / "meta.json").read_text())
    em = meta["episodes"]
    bank_eps = [e for e in train.episode.unique() if UNSEEN not in (em[str(e)]["regime_before"], em[str(e)]["regime_after"])]
    bank = train[train.episode.isin(bank_eps)].reset_index(drop=True)
    test_eps = sorted(e for e in test.episode.unique() if em[str(e)]["regime_after"] == UNSEEN)
    cand = []
    for ep in test_eps:
        g = test[test.episode == ep]
        sw = em[str(ep)]["regime_switch_t"]
        ts = np.array(sorted(g.t.unique()))
        ts = ts[ts > sw + 1e-9]
        if not len(ts):
            continue
        for t in np.arange(ts[0], ts[-1] + 1e-9, EVERY):
            if np.isclose(ts, t).any():
                cand.append((ep, round(float(t), 3)))
    rng = np.random.default_rng(0)
    pick = sorted(cand[i] for i in rng.choice(len(cand), min(N_MOM, len(cand)), replace=False))
    q = pd.concat([test[(test.episode == ep) & np.isclose(test.t, t)] for ep, t in pick], ignore_index=True)
    tracks, obst, m2 = schema.load(ROOT / DATA)
    F.load_kalman(ROOT / DATA, m2)
    obs = F.build_rows(tracks, obst, m2, horizons=(H,), noise=NOISE, seed=0, episodes=test_eps, label_from="observed")
    return bank, bank_eps, test_eps, cand, pick, q, obs


def ped_bank():
    p = rows(PED)
    p = p[np.isclose(p.h, H)].dropna(subset=["ddx", "ddy"])
    tr, _, _ = F.split_by_episode(p, seed=0)
    return p[tr].reset_index(drop=True)


# ---------------- sanity ----------------
def sanity():
    df, train, test = setup()
    print(f"rows: {len(df)} total ({df.episode.nunique()} episodes); h=2: train {len(train)} rows "
          f"({train.episode.nunique()} ep), test {len(test)} rows ({test.episode.nunique()} ep)")
    print("constant velocity (observed, Kalman-filtered state), |(ddx, ddy)| per horizon:")
    for h in (1.0, 2.0):
        e = np.hypot(*df[np.isclose(df.h, h)][["ddx", "ddy"]].to_numpy().T)
        print(f"  +{h:.0f} s: median {np.median(e):.3f} m, mean {e.mean():.3f}, p90 {np.quantile(e, 0.9):.3f}, "
              f"p99 {np.quantile(e, 0.99):.3f}  (n={len(e)})")
    d2 = df[np.isclose(df.h, H)]
    e2 = np.hypot(d2.ddx, d2.ddy)
    print("  +2 s by regime:", d2.assign(e=e2).groupby("regime").e.agg(["median", "mean", "count"]).round(3).to_dict("index"))
    print("  speed (filtered) m/s: median", round(float(np.median(d2.speed)), 2))
    print("mode label shares (h=2 rows):", d2["mode"].value_counts(normalize=True).round(4).to_dict())
    tracks, _, _ = schema.load(ROOT / DATA)
    c = tracks[tracks.is_ego == 0].sort_values(["episode", "agent_id", "t"])
    g = c.groupby(["episode", "agent_id"])
    lc = (g.ext_lane.shift(-10) != c.ext_lane) & g.ext_lane.shift(-10).notna()
    print(f"lane change within the next 2 s: {100 * lc.mean():.1f}% of car-steps; by regime "
          f"{(100 * lc.groupby(c.regime).mean()).round(1).to_dict()}")
    print("along / across residual sd at +2 s:", d2[["ddx", "ddy"]].std().round(3).to_dict())


# ---------------- plan ----------------
def plan(verbose=True):
    _, train, test = setup()
    bank, bank_eps, test_eps, cand, pick, q, obs = h2_setup(train, test)
    n_split = sum(int((obs[obs.episode == ep].t_known <= t + 1e-9).sum() >= THIN) for ep, t in pick)
    calls = {"H1 TabPFN (3 seeds x 2 axes)": 2 * len(SEEDS), "H2 frozen": 2, "H2 scene models": 2 * n_split,
             "H2b pedestrian frozen": 2}
    n = sum(calls.values())
    if verbose:
        print(f"H2: bank {len(bank)} rows from {len(bank_eps)} training episodes without '{UNSEEN}'; "
              f"{len(test_eps)} held-out episodes switch to '{UNSEEN}': {test_eps}")
        print(f"H2: {len(cand)} candidate moments, {len(pick)} drawn, {len(q)} query rows "
              f"({q.vehicle.nunique()} vehicles), since switch {q.since_switch.min():.1f}-{q.since_switch.max():.1f} s")
        for k, v in calls.items():
            print(f"  {k}: {v} calls")
        print(f"total {n} API calls x {CREDITS_PER_CALL:,} = {n * CREDITS_PER_CALL:,} credits (cap {CAP_CREDITS:,})")
    return n


def log_usage(calls, note):
    day = dtm.datetime.now(dtm.timezone.utc).strftime("%Y-%m-%d")
    U = pd.read_csv(USAGE)
    prev = pd.to_numeric(U.loc[(U.date_utc == day) & (U.run != "day total (server)"), "credits"], errors="coerce").sum()
    with open(USAGE, "a", newline="", encoding="utf-8") as f:
        f.write(f"{day},highway (tools/highway_tests.py),{calls},{calls * CREDITS_PER_CALL},"
                f"{note.replace(',', ';')}; running total {day}: {int(prev + calls * CREDITS_PER_CALL)} credits\n")


# ---------------- H1 ----------------
def run_h1(train, test, dry):
    te = test.sample(min(N_TEST, len(test)), random_state=0).reset_index(drop=True)
    Xt, Yt = F.X_of(te), te[["ddx", "ddy"]].to_numpy(float)
    Xa, Ya = F.X_of(train), train[["ddx", "ddy"]].to_numpy(float)
    fixed = {"cv": M.make("cv").fit(Xa, Ya), "lgbm_all": M.make("lgbm:m50:n100").fit(Xa, Ya)}
    recs, calls = [], 0
    for seed in SEEDS:
        ctx = train.sample(N_CTX, random_state=seed)
        Xc, Yc = F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy(float)
        models = dict(fixed)
        models["cv_band"] = M.make("cv_band").fit(Xc, Yc)
        models["lgbm"] = M.make("lgbm:m50:n100").fit(Xc, Yc)
        models["tabpfn"] = tabpfn(dry).fit(Xc, Yc)
        calls += 0 if dry else 2
        for name, m in models.items():
            sc = scores(m.predict_quantiles(Xt, QS), Yt)
            recs.append(pd.DataFrame({"seed": seed, "model": name, "row": np.arange(len(te)),
                                      "key": [f"{seed}:{i:05d}" for i in range(len(te))],
                                      "vehicle": te.vehicle, "regime": te.regime, **sc}))
        print(f"H1 seed {seed} done ({calls} API calls so far)", flush=True)
    return pd.concat(recs, ignore_index=True), calls


def summarise(R, comps, label):
    out = []
    for name, g in R.groupby("model", sort=False):
        out.append(dict(test=label, kind="level", model=name, n_rows=len(g), vehicles=g.vehicle.nunique(),
                        **{m: float(g[m].mean()) for m in ("joint90", "is90", "err", "width90")}))
    for a, b in comps:
        for metric in ("is90", "joint90", "err", "width90"):
            d, lo, hi, nv = vboot(R, metric, a, b)
            out.append(dict(test=label, kind="diff", model=f"{a} - {b}", metric=metric, diff=d, lo=lo, hi=hi,
                            vehicles=nv, n_rows=int((R.model == a).sum())))
    return pd.DataFrame(out)


# ---------------- H2 / H2b ----------------
def run_h2(train, test, dry):
    bank, bank_eps, test_eps, cand, pick, q, obs = h2_setup(train, test)
    Xq, Y = F.X_of(q), q[["ddx", "ddy"]].to_numpy(float)
    ctx = C.frozen(bank, BANK)
    Qf = tabpfn(dry).fit(F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy(float)).predict_quantiles(Xq, QS)
    calls = 0 if dry else 2
    pb = ped_bank(); pctx = C.frozen(pb, BANK)
    Qp = tabpfn(dry).fit(F.X_of(pctx), pctx[["ddx", "ddy"]].to_numpy(float)).predict_quantiles(Xq, QS)
    calls += 0 if dry else 2
    Xb, Yb = F.X_of(bank), bank[["ddx", "ddy"]].to_numpy(float)
    Ql = M.make("lgbm:m50:n100").fit(Xb, Yb).predict_quantiles(Xq, QS)
    Qcb = M.make("cv_band").fit(F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy(float)).predict_quantiles(Xq, QS)
    Qcv = np.zeros_like(Qf)
    Qr = np.full_like(Qf, np.nan); n_rec = np.zeros(len(q), int); n_rec_post = np.zeros(len(q), int)
    meta = json.loads((ROOT / DATA / "meta.json").read_text())["episodes"]
    for k, (ep, t) in enumerate(pick):
        sel = np.flatnonzero((q.episode == ep).to_numpy() & np.isclose(q.t, t))
        live = obs[obs.episode == ep]
        rec = live[live.t_known <= t + 1e-9].sort_values("t_known").tail(REC_CAP)
        n_rec[sel] = len(rec); n_rec_post[sel] = int((rec.t >= meta[str(ep)]["regime_switch_t"] - 1e-9).sum())
        if len(rec) < THIN:
            continue
        Qr[sel] = tabpfn(dry).fit(F.X_of(rec), rec[["ddx", "ddy"]].to_numpy(float)).predict_quantiles(Xq[sel], QS)
        calls += 0 if dry else 2
        print(f"H2 moment {k + 1}/{len(pick)} ep {ep} t {t:.1f}: {len(sel)} cars, {len(rec)} scene rows "
              f"({n_rec_post[sel][0]} after the switch); calls so far {calls}", flush=True)
    thin = np.isnan(Qr[:, 0, 0])
    Qr_f = np.where(thin[:, None, None], Qf, Qr)                        # thin: frozen alone (none expected)
    Qs = np.sort((Qf + Qr_f) / 2, axis=2)
    Qps = np.sort((Qp + Qr_f) / 2, axis=2)
    allQ = {"frozen": Qf, "split": Qs, "scene_only": Qr_f, "lgbm_all": Ql, "cv_band": Qcb, "cv": Qcv,
            "ped": Qp, "ped+scene": Qps}
    recs = []
    for name, Q in allQ.items():
        sc = scores(Q, Y)
        recs.append(pd.DataFrame({"model": name, "key": [f"{i:05d}" for i in range(len(q))], "vehicle": q.vehicle,
                                  "episode": q.episode, "agent_id": q.agent_id, "t": q.t, "since": q.since_switch,
                                  "n_scene": n_rec, "n_scene_post": n_rec_post, "thin": thin, **sc}))
    np.savez(ROOT / ("results/dry" if dry else "results") / "highway_h2_q.npz", **{k.replace("+", "_"): v for k, v in allQ.items()},
             Y=Y, episode=q.episode.to_numpy(), t=q.t.to_numpy(), agent_id=q.agent_id.to_numpy(),
             px=q.px.to_numpy(), py=q.py.to_numpy(), vx=q.vx.to_numpy(), vy=q.vy.to_numpy(),
             theta=q.theta.to_numpy(), true_x=q.true_x.to_numpy(), true_y=q.true_y.to_numpy())
    print(f"H2: bank {len(bank)} rows ({len(bank_eps)} ep); {len(q)} queries, {q.vehicle.nunique()} vehicles, "
          f"{len(pick)} moments in {len(test_eps)} episodes; thin moments {int(thin.any() and thin.sum())}", flush=True)
    return pd.concat(recs, ignore_index=True), calls


# ---------------- figure ----------------
def figure(dry=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    z = np.load(ROOT / ("results/dry" if dry else "results") / "highway_h2_q.npz")
    tracks, _, meta = schema.load(ROOT / DATA)
    ep_t = sorted(set(zip(z["episode"].tolist(), np.round(z["t"], 3).tolist())))
    # a busy but typical frame: most cars with a real lane change in the next 2 s (|true across residual| > 1.5 m),
    # among moments >= 6 s after the switch whose widest across box stays under 10 m (no runaway box filling the plot)
    def busy(et):
        m_ = (z["episode"] == et[0]) & np.isclose(z["t"], et[1])
        wide = float((z["split"][m_][:, 1, 4] - z["split"][m_][:, 1, 0]).max())
        since = et[1] - meta["episodes"][str(et[0])]["regime_switch_t"]
        return (wide < 10.0 and since >= 6.0, int((np.abs(z["Y"][m_][:, 1]) > 1.5).sum()), -abs(since - 8.0))
    best = max(ep_t, key=busy)
    ep, t = best
    m = (z["episode"] == ep) & np.isclose(z["t"], t)
    tr = tracks[tracks.episode == ep]
    ego = tr[(tr.is_ego == 1) & np.isclose(tr.t, t)].iloc[0]
    x0 = float(np.median(z["px"][m]))
    fig, ax = plt.subplots(figsize=(19, 4.6), facecolor="white")
    lanes = sorted(tr.ext_lane.unique())
    for ln in range(len(lanes) + 1):
        ax.axhline(-2 + 4 * ln, color="#bbb", lw=1, ls="-" if ln in (0, len(lanes)) else "--", zorder=0)
    win = (float(ego.x) - 40, float(ego.x) + 300)          # ego in view; boxes sit ~60 m ahead
    now = tr[np.isclose(tr.t, t)]
    for _, r in now.iterrows():
        past = tr[(tr.agent_id == r.agent_id) & (tr.t <= t + 1e-9) & (tr.t >= t - 2.0 - 1e-9)]
        ax.plot(past.x, past.y, color="#1f2328" if r.is_ego else "#8b949e", lw=1.4, zorder=2)
        c = "#e8590c" if r.is_ego else "#1f2328"
        ax.add_patch(plt.Rectangle((r.x - 2.5, r.y - 1.0), 5.0, 2.0, color=c, zorder=4))
    first = True
    for i in np.flatnonzero(m):
        th, h = z["theta"][i], H
        cx, cy = z["px"][i] + z["vx"][i] * h, z["py"][i] + z["vy"][i] * h
        lo_a, hi_a, lo_c, hi_c = z["split"][i, 0, 0], z["split"][i, 0, 4], z["split"][i, 1, 0], z["split"][i, 1, 4]
        cs, sn = np.cos(th), np.sin(th)
        pts = [(cx + cs * a - sn * b, cy + sn * a + cs * b) for a, b in ((lo_a, lo_c), (hi_a, lo_c), (hi_a, hi_c), (lo_a, hi_c))]
        ax.add_patch(Polygon(pts, closed=True, fc="#2a6fdb", ec="#2a6fdb", alpha=0.25, lw=1.2, zorder=3,
                             label="TabPFN 5-95% box at +2 s (split)" if first else None))
        ax.plot([z["px"][i], z["true_x"][i]], [z["py"][i], z["true_y"][i]], color="#2f9e44", lw=0.8, ls=":", zorder=3)
        ax.scatter(z["true_x"][i], z["true_y"][i], marker="x", color="#2f9e44", s=40, zorder=5,
                   label="where it actually was at +2 s" if first else None)
        first = False
    ax.plot([], [], color="#8b949e", lw=1.4, label="last 2 s of track")
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, color="#e8590c", label="ego car (the 'robot')"))
    ax.set_xlim(*win); ax.set_ylim(-4, 4 * len(lanes))
    ax.set_xlabel("x along the road (m)"); ax.set_ylabel("y across (m)\n(stretched 3x)")
    ax.set_aspect(3.0)
    em = meta["episodes"][str(ep)]
    ax.set_title(f"Highway cars on the same core: episode {ep}, t = {t:.1f} s, {t - em['regime_switch_t']:.1f} s after "
                 f"the traffic switched {em['regime_before']} -> {em['regime_after']}. Bank never saw '{UNSEEN}'.",
                 fontsize=10, loc="left")
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=4, bbox_to_anchor=(0, -0.22))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = ROOT / "media/fig_highway.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out} (episode {ep}, t {t})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    if a.sanity:
        return sanity()
    if a.figure:
        return figure(a.dry)
    n = plan()
    if a.count_only:
        return
    if not a.dry and n * CREDITS_PER_CALL > CAP_CREDITS:
        raise SystemExit("planned spend above cap; not running")
    out = ROOT / ("results/dry" if a.dry else "results"); out.mkdir(parents=True, exist_ok=True)
    before = after = "?"
    if not a.dry:
        if not os.environ.get("TABPFN_TOKEN"):
            import subprocess
            os.environ["TABPFN_TOKEN"] = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", "[Environment]::GetEnvironmentVariable('TABPFN_TOKEN','User')"],
                capture_output=True, text=True).stdout.strip()
        import tabpfn_client
        tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
        before = tabpfn_client.get_api_usage(); print("server before:", before, flush=True)
    T0 = time.perf_counter()
    _, train, test = setup()
    calls, err, parts = 0, None, []
    try:
        R1, c = run_h1(train, test, a.dry); calls += c
        R1.to_csv(out / "highway_h1_rows.csv", index=False)
        S1 = summarise(R1, [("tabpfn", r) for r in ("lgbm_all", "cv_band", "lgbm", "cv")], "H1")
        S1.to_csv(out / "highway_h1.csv", index=False); parts.append(S1)
        R2, c = run_h2(train, test, a.dry); calls += c
        R2.to_csv(out / "highway_h2_rows.csv", index=False)
        S2 = summarise(R2, [("split", "frozen"), ("split", "lgbm_all"), ("split", "scene_only"), ("frozen", "lgbm_all"),
                            ("ped+scene", "ped"), ("ped+scene", "split"), ("ped", "frozen")], "H2")
        S2.to_csv(out / "highway_h2.csv", index=False); parts.append(S2)
    except Exception as e:
        err = repr(e)[:300]; print("stopped on error:", err, flush=True)
        import traceback; traceback.print_exc()
    if not a.dry:
        import tabpfn_client
        after = tabpfn_client.get_api_usage(); print("server after:", after, flush=True)
        used = lambda x: x.split("used ")[1].split(" of")[0] if isinstance(x, str) and "used " in x else str(x)  # noqa: E731
        log_usage(calls, f"highway cars H1 + H2 + H2b (default hosted model); server usage {used(before)} -> "
                         f"{used(after)} (account may be shared with other sessions)" + (f"; stopped on error {err}" if err else ""))
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20)
    for S in parts:
        print(S.round(4).to_string(index=False))
    if len(parts) == 2:
        d1 = parts[0][(parts[0].kind == "diff") & (parts[0].metric == "is90")].set_index("model")
        h1 = all(d1.loc[f"tabpfn - {r}", "hi"] < 0 for r in ("lgbm_all", "cv_band"))
        d2 = parts[1][(parts[1].kind == "diff") & (parts[1].metric == "is90")].set_index("model")
        h2 = d2.loc["split - frozen", "hi"] < 0
        print(f"VERDICT H1 (docs/HIGHWAY.md): {'core transfers to cars' if h1 else 'not supported'}")
        print(f"VERDICT H2 (docs/HIGHWAY.md): {'adapts on cars' if h2 else 'not supported'}")
    print(f"API calls {calls}; {time.perf_counter() - T0:.0f} s")


if __name__ == "__main__":
    main()
