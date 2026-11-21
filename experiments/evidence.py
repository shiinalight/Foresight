"""Does TabPFN fit this problem? Three tests on every scene, plus an accuracy check on real data.

    python experiments/evidence.py --scenes plaza            # run one scene (all three tests)
    python experiments/evidence.py --scenes eth hotel        # real data: accuracy and calibrated boxes
    python experiments/evidence.py --report                  # merge, write results/evidence.json, figures, verdict

Scenes run as separate processes in parallel (each writes results/evidence_parts/<scene>.json); TabPFN
API answers are cached in results/cache/evidence/, so a rerun costs no quota.

Protocol (fixed before running; nothing is tuned on held-out episodes)
- Split by episode. Simulated scenes: training folder -> train (80% of episodes) + validation (20%);
  held-out = the separate *_test folder (new seeds). Real data: train / validation / held-out
  episodes (one-minute windows) of the same recording.
- Validation is used for (a) LightGBM's settings (num_leaves, min_child_samples, chosen by pinball loss,
  separately for every training size) and (b) every model's box size. TabPFN has nothing to tune.
- Causality: features use only the past (forward Kalman filter). Context rows come from other
  episodes, or, in Test 2, from the same episode only once their target time has passed; those live
  rows are labelled from OBSERVED (noisy, filtered) positions, as a robot would label them, not truth.
- Boxes: 5-95% quantiles per axis in the agent's frame; "inside" = both axes at once.

Test 1, ranges you can trust: each model's box scaled on validation to 80% coverage, per horizon;
  held-out coverage, mean box area, mean error; box area in crowded vs open situations (local
  density above / below the held-out median).
Test 2, adapts in seconds: episodes with a switch; TabPFN refitted every 1 s on context.select()
  (budget 1500: this episode's completed rows + similar and random bank rows), LightGBM retrained every
  1 s on exactly the same rows, LightGBM frozen (trained once on the bank). Error and raw box coverage
  at +2 s per 1 s bin from -5 to +15 s.
Test 3, learns from little data: error at +2 s against context size, 3 random draws of the context.

Verdict rules (written before the results):
- Test 1: TabPFN's held-out box area is smaller than every LightGBM variant's at both horizons, with
  TabPFN's held-out coverage >= 78%. A LightGBM variant whose held-out coverage falls below 78% is
  not a valid comparison (it got small boxes by missing) and is reported as such.
- Test 2: TabPFN recovers in fewer seconds than RETRAINED LightGBM and its extra error over the first
  5 s after the switch is lower. Recovery = first second from which the error stays within 10% of
  the model's own settled level (mean of +10..+15 s) for two bins in a row.
- Test 3: at 50, 100 and 250 rows, TabPFN's mean error is below LightGBM's by more than both
  standard deviations (over the 3 draws) in at least 2 of the 3 sizes.
- Real data: TabPFN's +2 s error is lower than LightGBM's with the 95% paired bootstrap interval of
  the difference excluding zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo import apply_scale, box_area, calibrate, evaluate  # noqa: E402  (same box definitions as the demo)
from foresight.core import context as C  # noqa: E402
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from foresight.core import schema  # noqa: E402

QS = (0.05, 0.5, 0.95)
SIM = {"plaza": ("data/crowd", "data/crowd_test"), "street": ("data/street", "data/street_test"),
       "doorway": ("data/doorway", "data/doorway_test")}
REAL = {"eth": "data/eth", "hotel": "data/hotel"}
H_SIM, H_REAL = (1.0, 2.0), (1.2, 2.0)          # real data is annotated every 0.4 s: 1.2 s is on its grid
N_VAL, N_TEST = 3000, 4000                       # rows (random, fixed) for box sizing and held-out scoring
TAB_CTX = 5000                                   # TabPFN context in Test 1 (random training rows)
LGBM_CAP = 60000                                 # LightGBM "all rows" (capped for run time)
SIZES = (50, 100, 250, 500, 1000, 2500, 5000)
DRAWS = (0, 1, 2)
N_T3 = 1000                                      # held-out +2 s rows scored in Test 3
BINS = range(-5, 15)
MAX_EP2 = 8                                      # switch episodes per scene in Test 2 (API time)
BUDGET2 = 1500
GRID = [dict(num_leaves=nl, min_child_samples=mc) for nl in (7, 31) for mc in (5, 20, 50)]
OUT = Path("results"); PARTS = OUT / "evidence_parts"; CACHE = OUT / "cache" / "evidence"; FIG = OUT / "figures"
NAMES = {"cv_box": "Constant velocity + fixed box", "lgbm_all": f"LightGBM (up to {LGBM_CAP // 1000}k rows)",
         "lgbm_same": f"LightGBM (same {TAB_CTX} rows)", "tabpfn": f"TabPFN ({TAB_CTX} rows)",
         "tabpfn_refit": "TabPFN, refit every 1 s", "lgbm_retrained": "LightGBM, retrained every 1 s on the same rows",
         "lgbm_frozen": "LightGBM, frozen (bank only)"}


# ---------- models ----------
class TunedLGBM(M.LGBMQuantile):
    """The core's LightGBM quantile model with settings chosen on validation."""

    def __init__(self, num_leaves=31, min_child_samples=10, n_estimators=300):
        super().__init__(n_estimators)
        self.p = dict(num_leaves=num_leaves, min_child_samples=min_child_samples)

    def predict_quantiles(self, X, qs=QS):
        from lightgbm import LGBMRegressor
        out = np.zeros((len(X), 2, len(qs)))
        for ax in range(2):
            for j, q in enumerate(qs):
                if (ax, q) not in self.models:
                    self.models[(ax, q)] = LGBMRegressor(objective="quantile", alpha=q, n_estimators=self.n,
                                                         learning_rate=0.05, verbose=-1, **self.p).fit(self.X, self.Y[:, ax])
                out[:, ax, j] = self.models[(ax, q)].predict(X)
        out.sort(axis=2)
        return out


def pinball(Q, Y, qs=QS):
    d = Y[:, :, None] - Q
    q = np.array(qs)[None, None, :]
    return float(np.maximum(q * d, (q - 1) * d).mean())


def tune_lgbm(train, val, seed=0):
    """Pick LightGBM settings on validation rows by pinball loss (for this training set's size)."""
    Xv, Yv = F.X_of(val), val[["ddx", "ddy"]].to_numpy()
    best = None
    for p in GRID:
        if p["min_child_samples"] * 2 > len(train):
            continue
        m = TunedLGBM(**p).fit(F.X_of(train), train[["ddx", "ddy"]].to_numpy())
        s = pinball(m.predict_quantiles(Xv), Yv)
        if best is None or s < best[0]:
            best = (s, p)
    return best[1] if best else dict(num_leaves=7, min_child_samples=5)


class TabPFN:
    """TabPFN through the hosted API, cached by (context, queries); counts calls and wall time."""
    calls = 0; seconds = []; cached = 0

    def __init__(self):
        self.Xc = self.Yc = None

    def fit(self, X, Y):
        self.Xc, self.Yc = np.asarray(X, float), np.asarray(Y, float)
        return self

    def predict_quantiles(self, X, qs=QS):
        X = np.asarray(X, float)
        key = hashlib.sha1(b"".join(a.tobytes() for a in (self.Xc, self.Yc, X)) + str(qs).encode()).hexdigest()[:20]
        f = CACHE / f"tab_{key}.npy"
        if f.exists():
            TabPFN.cached += 1
            return np.load(f)
        t0 = time.perf_counter()
        m = M.make("tabpfn_client").fit(self.Xc, self.Yc)
        Q = np.sort(m.predict_quantiles(X, qs), axis=2)
        TabPFN.seconds.append(time.perf_counter() - t0); TabPFN.calls += 2           # one fit + predict per axis
        f.parent.mkdir(parents=True, exist_ok=True); np.save(f, Q)
        return Q


def api_used():
    try:
        from tabpfn_client import get_api_usage, set_access_token
        set_access_token(os.environ["TABPFN_TOKEN"])
        u = get_api_usage()
        m = re.search(r"used (\d+)", str(u))
        return int(m.group(1)) if m else None
    except Exception:
        return None


# ---------- data ----------
def rows(data_dir, horizons):
    f = CACHE / f"rows_{Path(data_dir).name}_h{'_'.join(f'{h:g}' for h in horizons)}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    tracks, obst, meta = schema.load(data_dir)
    df = F.build_rows(tracks, obst, meta, horizons=horizons, noise=0.1, seed=0)
    f.parent.mkdir(parents=True, exist_ok=True); df.to_parquet(f)
    return df


def sample(df, n, seed=0):
    return df.sample(min(n, len(df)), random_state=seed) if len(df) > n else df


def splits(scene):
    """train, validation, held-out rows."""
    if scene in SIM:
        tr_dir, te_dir = SIM[scene]
        d = rows(tr_dir, H_SIM)
        tr, _, va = F.split_by_episode(d, test_frac=0.2, seed=0)
        return d[tr], d[va], rows(te_dir, H_SIM)
    d = rows(REAL[scene], H_REAL)
    tr, va, te = F.split_by_episode(d, test_frac=0.25, val_frac=0.25, seed=0)
    return d[tr], d[va], d[te]


def observed_labels(r, dt):
    """Relabel rows from OBSERVED positions: the filtered position of the same agent at t + h, as a
    robot would label its own past rows (rows whose future observation is missing are dropped)."""
    key = r[["episode", "agent_id", "t", "px", "py"]].drop_duplicates(["episode", "agent_id", "t"]).copy()
    key["kf"] = np.round(key.t / dt).astype(int)
    out = r.copy(); out["kf"] = np.round((out.t + out.h) / dt).astype(int)
    out = out.merge(key[["episode", "agent_id", "kf", "px", "py"]].rename(columns={"px": "fx", "py": "fy"}),
                    on=["episode", "agent_id", "kf"], how="inner")
    c, s = np.cos(out.theta), np.sin(out.theta)
    dx = out.fx - out.px - out.vx * out.h; dy = out.fy - out.py - out.vy * out.h
    out["ddx"], out["ddy"] = c * dx + s * dy, -s * dx + c * dy
    return out.drop(columns=["kf", "fx", "fy"])


# ---------- Test 1 (and the real-data check) ----------
def test1(scene, tr, va, te, horizons):
    va, te = sample(va, N_VAL, 0), sample(te, N_TEST, 0)
    Xtr, Ytr = F.X_of(tr), tr[["ddx", "ddy"]].to_numpy()
    same = sample(tr, TAB_CTX, 0)
    big = sample(tr, LGBM_CAP, 0)
    p_all = tune_lgbm(sample(big, 20000, 1), va); p_same = tune_lgbm(same, va)
    models = {"cv_box": M.ConstantBox().fit(Xtr, Ytr),
              "lgbm_all": TunedLGBM(**p_all).fit(F.X_of(big), big[["ddx", "ddy"]].to_numpy()),
              "lgbm_same": TunedLGBM(**p_same).fit(F.X_of(same), same[["ddx", "ddy"]].to_numpy()),
              "tabpfn": TabPFN().fit(F.X_of(same), same[["ddx", "ddy"]].to_numpy())}
    dens_med = {h: float(te.density_3m[np.isclose(te.h, h)].median()) for h in horizons}
    res, raw_cov, Yte = {}, {}, te[["ddx", "ddy"]].to_numpy()
    for name, m in models.items():
        t0 = time.perf_counter()
        Q = m.predict_quantiles(np.concatenate([F.X_of(va), F.X_of(te)]), QS)
        secs = time.perf_counter() - t0
        Qv, Qt = Q[:len(va)], Q[len(va):]
        scales = calibrate(Qv, va)
        Qc = apply_scale(Qt, te, scales)
        ev = evaluate(te, Qc)
        out = {"settings": p_all if name == "lgbm_all" else p_same if name == "lgbm_same" else None,
               "train_rows": {"cv_box": len(tr), "lgbm_all": len(big), "lgbm_same": len(same), "tabpfn": len(same)}[name],
               "predict_seconds": secs}
        for h in horizons:
            k = np.isclose(te.h.to_numpy(), h)
            crowded = k & (te.density_3m.to_numpy() > dens_med[h]); open_ = k & ~crowded
            raw_inside = ((Yte >= Qt[:, :, 0]) & (Yte <= Qt[:, :, 2])).all(axis=1)
            out[f"{h:g}"] = dict(rows=int(k.sum()), scale=scales[h], raw_coverage=float(raw_inside[k].mean()),
                                 coverage=float(ev.inside[k].mean()), area_m2=float(ev.area[k].mean()),
                                 error_m=float(ev.err[k].mean()),
                                 area_crowded_m2=float(ev.area[crowded].mean()), area_open_m2=float(ev.area[open_].mean()),
                                 coverage_crowded=float(ev.inside[crowded].mean()), coverage_open=float(ev.inside[open_].mean()),
                                 crowded_over_open=float(ev.area[crowded].mean() / ev.area[open_].mean()))
        res[name] = out
        res[name]["_err_rows"] = ev.err.to_numpy()      # kept in memory for the paired bootstrap, not saved
    print(f"[{scene}] test 1 done")
    return res, models, dict(density_median=dens_med, val_rows=len(va), test_rows=len(te),
                             test_episodes=int(te.episode.nunique()), horizons=list(horizons))


def paired_ci(a, b, n=2000, seed=0):
    """95% bootstrap interval of mean(a - b) over rows."""
    rng = np.random.default_rng(seed); d = a - b
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ---------- Test 3 ----------
def test3(scene, tr, va, te):
    q = sample(te[np.isclose(te.h, 2.0)], N_T3, 0)
    vq = sample(va[np.isclose(va.h, 2.0)], 2000, 0)
    out = {}
    for n in SIZES:
        p = tune_lgbm(sample(tr, n, 100), vq)                      # settings for this size, on validation
        e = {"tabpfn": [], "lgbm": []}
        for s in DRAWS:
            ctx = sample(tr, n, s)
            for name, m in (("tabpfn", TabPFN()), ("lgbm", TunedLGBM(**p))):
                Q = m.fit(F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy()).predict_quantiles(F.X_of(q), QS)
                mx, my = F.to_world(q, Q[:, 0, 1], Q[:, 1, 1])
                e[name].append(float(np.hypot(mx - q.true_x, my - q.true_y).mean()))
        out[n] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), draws=v) for k, v in e.items()}
        out[n]["lgbm_settings"] = p
        print(f"[{scene}] test 3 n={n}: TabPFN {out[n]['tabpfn']['mean']:.3f}  LightGBM {out[n]['lgbm']['mean']:.3f}", flush=True)
    return out


# ---------- Test 2 ----------
def test2(scene, tr, va, te, frozen, dt):
    sw_eps = sorted(te.loc[te.since_switch.notna(), "episode"].unique())[:MAX_EP2]
    bank = C.Bank(tr, seed=0)
    p_small = tune_lgbm(sample(tr, BUDGET2, 7), sample(va[np.isclose(va.h, 2.0)], 2000, 0))
    live_all = observed_labels(te[te.episode.isin(sw_eps)], dt)
    per = []                                                     # one record per (episode, bin, model)
    for ep in sw_eps:
        e = te[te.episode == ep]; live = live_all[live_all.episode == ep]
        for b in BINS:
            q = e[np.isclose(e.h, 2.0) & (e.since_switch >= b) & (e.since_switch < b + 1)]
            if not len(q):
                continue
            ctx = C.select(q, bank, live=live, budget=BUDGET2, mix=(0.6, 0.3, 0.1), seed=0)
            Xc, Yc = F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy()
            for name, m in (("tabpfn_refit", TabPFN().fit(Xc, Yc)), ("lgbm_retrained", TunedLGBM(**p_small).fit(Xc, Yc)),
                            ("lgbm_frozen", frozen)):
                Q = m.predict_quantiles(F.X_of(q), QS)
                mx, my = F.to_world(q, Q[:, 0, 1], Q[:, 1, 1])
                Y = q[["ddx", "ddy"]].to_numpy()
                per.append(dict(episode=int(ep), bin=b, model=name, rows=len(q), ctx=len(ctx),
                                err=float(np.hypot(mx - q.true_x, my - q.true_y).mean()),
                                coverage=float(((Y >= Q[:, :, 0]) & (Y <= Q[:, :, 2])).all(axis=1).mean())))
        print(f"[{scene}] test 2 episode {ep} done", flush=True)
    P = pd.DataFrame(per)
    curves = {}
    for name, g in P.groupby("model"):
        a = g.groupby("bin").agg(err=("err", "mean"), err_se=("err", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
                                 coverage=("coverage", "mean"), cov_se=("coverage", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
                                 episodes=("episode", "nunique"))
        curves[name] = {int(b): {k: (None if pd.isna(v) else float(v)) for k, v in r.items()} for b, r in a.iterrows()}
    return dict(episodes=[int(x) for x in sw_eps], lgbm_small_settings=p_small, curves=curves,
                recovery={m: recovery(c) for m, c in curves.items()})


def recovery(curve):
    err = {b: v["err"] for b, v in curve.items()}
    settled = np.mean([err[b] for b in range(10, 15) if b in err])
    pre = np.mean([err[b] for b in range(-5, 0) if b in err])
    first = [err[b] for b in range(0, 5) if b in err]
    rec = None
    for b in range(0, 14):
        if all(b + i in err and err[b + i] <= 1.1 * settled for i in (0, 1)):
            rec = b; break
    return dict(settled_err=float(settled), pre_switch_err=float(pre), recovery_s=rec,
                excess_err_0_5s=float(np.mean(first) - settled) if first else None)


# ---------- run one scene ----------
def run_scene(scene):
    t_start, used0 = time.time(), api_used()
    tr, va, te = splits(scene)
    horizons = H_SIM if scene in SIM else H_REAL
    r1, models, info = test1(scene, tr, va, te, horizons)
    part = dict(scene=scene, info=info, train_rows=len(tr), val_rows=len(va), heldout_rows=len(te))
    if scene in REAL:
        part["real_check"] = {k: v for k, v in ((m, {kk: vv for kk, vv in r.items() if kk != "_err_rows"}) for m, r in r1.items())}
        best = min(("lgbm_all", "lgbm_same"), key=lambda m: r1[m]["2"]["error_m"])
        k2 = np.isclose(sample(te, N_TEST, 0).h.to_numpy(), 2.0)
        part["real_check"]["tabpfn_minus_best_lgbm_err_2s"] = dict(
            zip(("mean", "lo", "hi"), paired_ci(r1["tabpfn"]["_err_rows"][k2], r1[best]["_err_rows"][k2])), against=best)
    else:
        part["test1"] = {m: {k: v for k, v in r.items() if k != "_err_rows"} for m, r in r1.items()}
        part["test3"] = test3(scene, tr, va, te)
        part["test2"] = test2(scene, tr, va, te, models["lgbm_all"], dt=0.2)
    used1 = api_used()
    part["cost"] = dict(api_calls=TabPFN.calls, cached_answers=TabPFN.cached,
                        tabpfn_seconds_per_batch=float(np.mean(TabPFN.seconds)) if TabPFN.seconds else None,
                        tabpfn_seconds_p90=float(np.percentile(TabPFN.seconds, 90)) if TabPFN.seconds else None,
                        credits_used=(used1 - used0) if used0 is not None and used1 is not None else None,
                        wall_minutes=(time.time() - t_start) / 60)
    if part["cost"]["credits_used"] is not None and TabPFN.calls:
        part["cost"]["credits_per_call"] = part["cost"]["credits_used"] / TabPFN.calls
    PARTS.mkdir(parents=True, exist_ok=True)
    (PARTS / f"{scene}.json").write_text(json.dumps(part, indent=1, default=float))
    print(f"[{scene}] wrote {PARTS / f'{scene}.json'} ({part['cost']})")


# ---------- report: merge, verdicts, figures ----------
def verdicts(E):
    rows = []
    for s in SIM:
        if s not in E:
            continue
        p = E[s]
        t1 = p["test1"]; ok, notes = True, []
        for h in ("1", "2"):
            tab = t1["tabpfn"][h]
            if tab["coverage"] < 0.78:
                ok = False; notes.append(f"+{h} s TabPFN covers only {tab['coverage']:.0%}")
            for m in ("lgbm_all", "lgbm_same"):
                lg = t1[m][h]
                if lg["coverage"] < 0.78:
                    notes.append(f"+{h} s {NAMES[m]} misses (covers {lg['coverage']:.0%}), not a valid comparison")
                    continue
                if tab["area_m2"] >= lg["area_m2"]:
                    ok = False
                notes.append(f"+{h} s area TabPFN {tab['area_m2']:.2f} vs {NAMES[m]} {lg['area_m2']:.2f} m2")
        rows.append(dict(test="1 trustworthy ranges", scene=s, verdict="TabPFN justified" if ok else "No advantage", detail="; ".join(notes)))
        r = p["test2"]["recovery"]; tab, lr = r["tabpfn_refit"], r["lgbm_retrained"]
        faster = (tab["recovery_s"] is not None and (lr["recovery_s"] is None or tab["recovery_s"] < lr["recovery_s"])
                  and tab["excess_err_0_5s"] < lr["excess_err_0_5s"])
        rows.append(dict(test="2 adapts in seconds", scene=s, verdict="TabPFN justified" if faster else "No advantage",
                         detail=f"recovery TabPFN {tab['recovery_s']} s vs retrained LightGBM {lr['recovery_s']} s "
                                f"(frozen {r['lgbm_frozen']['recovery_s']} s); extra error 0-5 s {tab['excess_err_0_5s']:+.3f} "
                                f"vs {lr['excess_err_0_5s']:+.3f} m; {len(p['test2']['episodes'])} episodes"))
        t3 = p["test3"]; wins = 0; det = []
        for n in ("50", "100", "250"):
            a, b = t3[n]["tabpfn"], t3[n]["lgbm"]
            win = (b["mean"] - a["mean"]) > (a["sd"] + b["sd"])
            wins += win; det.append(f"{n}: {a['mean']:.2f}±{a['sd']:.2f} vs {b['mean']:.2f}±{b['sd']:.2f} m")
        rows.append(dict(test="3 learns from little data", scene=s, verdict="TabPFN justified" if wins >= 2 else "No advantage",
                         detail=f"TabPFN vs LightGBM at +2 s, {'; '.join(det)}"))
    for s in REAL:
        if s not in E:
            continue
        rc = E[s]["real_check"]; d = rc["tabpfn_minus_best_lgbm_err_2s"]
        better = d["hi"] < 0
        rows.append(dict(test="real-data accuracy", scene=s, verdict="TabPFN justified" if better else "No advantage",
                         detail=f"+2 s error TabPFN {rc['tabpfn']['2']['error_m']:.3f} vs {NAMES[d['against']]} "
                                f"{rc[d['against']]['2']['error_m']:.3f} m (difference {d['mean']:+.3f}, 95% CI {d['lo']:+.3f} to {d['hi']:+.3f})"))
        # a LightGBM variant that misses (< 78%) is no valid comparison; none valid -> TabPFN must still cover
        t1ok = all(rc["tabpfn"][h]["area_m2"] < min((rc[m][h]["area_m2"] for m in ("lgbm_all", "lgbm_same")
                                                     if rc[m][h]["coverage"] >= 0.78), default=float("inf"))
                   and rc["tabpfn"][h]["coverage"] >= 0.78 for h in ("1.2", "2"))
        rows.append(dict(test="1 trustworthy ranges (real)", scene=s, verdict="TabPFN justified" if t1ok else "No advantage",
                         detail="; ".join(f"+{h} s area TabPFN {rc['tabpfn'][h]['area_m2']:.2f} (cov {rc['tabpfn'][h]['coverage']:.0%}) vs "
                                          f"LightGBM {rc['lgbm_all'][h]['area_m2']:.2f} (cov {rc['lgbm_all'][h]['coverage']:.0%}) m2" for h in ("1.2", "2"))))
    return rows


COL = {"cv_box": "#8c8c8c", "lgbm_all": "#2a78d6", "lgbm_same": "#1baf7a", "tabpfn": "#eb6834",
       "tabpfn_refit": "#eb6834", "lgbm_retrained": "#2a78d6", "lgbm_frozen": "#1baf7a"}


def figures(E):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8})
    sims = [s for s in SIM if s in E]
    # Test 1: box area at 80% coverage, all and crowded vs open, +2 s
    mods = ["cv_box", "lgbm_all", "lgbm_same", "tabpfn"]
    fig, axes = plt.subplots(1, len(sims), figsize=(4.2 * len(sims), 4.2), squeeze=False)
    for ax, s in zip(axes[0], sims):
        t = E[s]["test1"]; x = np.arange(len(mods)); w = 0.38
        for i, m in enumerate(mods):
            r = t[m]["2"]
            ax.bar(i - w / 2, r["area_open_m2"], w, color=COL[m], alpha=0.45, edgecolor="white", linewidth=2)
            ax.bar(i + w / 2, r["area_crowded_m2"], w, color=COL[m], edgecolor="white", linewidth=2)
            ax.text(i, max(r["area_open_m2"], r["area_crowded_m2"]) * 1.03, f"{r['coverage']:.0%}", ha="center", fontsize=9, color="#333")
        ax.set_xticks(x, ["CV + box", "LightGBM\nall rows", f"LightGBM\n{TAB_CTX} rows", f"TabPFN\n{TAB_CTX} rows"], fontsize=9)
        ax.set_title(s.capitalize(), loc="left", fontweight="bold"); ax.set_ylabel("mean box area at +2 s (m²)")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Test 1 · box size when every model is sized to 80% on validation\n"
                 "light bar = open situations, dark bar = crowded; number = held-out coverage", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "evidence_test1_ranges.png", dpi=150); plt.close(fig)
    # Test 2
    fig, axes = plt.subplots(2, len(sims), figsize=(4.4 * len(sims), 6.4), squeeze=False, sharex=True)
    for j, s in enumerate(sims):
        cv = E[s]["test2"]["curves"]
        for m, mk in (("lgbm_frozen", "^"), ("lgbm_retrained", "s"), ("tabpfn_refit", "o")):
            c = cv[m]; b = np.array(sorted(int(k) for k in c)) + 0.5
            for row, key, se in ((0, "err", "err_se"), (1, "coverage", "cov_se")):
                y = np.array([c[str(int(k - 0.5))][key] if str(int(k - 0.5)) in c else c[int(k - 0.5)][key] for k in b])
                e = np.array([(c[str(int(k - 0.5))] if str(int(k - 0.5)) in c else c[int(k - 0.5)])[se] or 0 for k in b])
                axes[row, j].fill_between(b, y - e, y + e, color=COL[m], alpha=0.12, linewidth=0)
                axes[row, j].plot(b, y, color=COL[m], lw=2, marker=mk, ms=4, label=NAMES[m])
        for row in (0, 1):
            axes[row, j].axvline(0, color="#555", lw=1, ls="--")
        axes[1, j].axhline(0.81, color="#999", lw=1, ls=":")
        axes[0, j].set_title(f"{s.capitalize()} · {len(E[s]['test2']['episodes'])} episodes", loc="left", fontweight="bold")
        axes[0, j].set_ylabel("error at +2 s (m)"); axes[1, j].set_ylabel("coverage of the 5–95% box")
        axes[1, j].set_xlabel("seconds since the behaviour switch")
    axes[0, 0].legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.suptitle("Test 2 · adapting to a behaviour switch (refit every 1 s; dotted line = nominal 81%)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "evidence_test2_adapt.png", dpi=150); plt.close(fig)
    # Test 3
    fig, axes = plt.subplots(1, len(sims), figsize=(4.2 * len(sims), 3.8), squeeze=False)
    for ax, s in zip(axes[0], sims):
        t = E[s]["test3"]; n = np.array(sorted(int(k) for k in t))
        for m, lab, mk in (("lgbm", "LightGBM (tuned per size)", "s"), ("tabpfn", "TabPFN", "o")):
            y = np.array([t[str(k)][m]["mean"] for k in n]); e = np.array([t[str(k)][m]["sd"] for k in n])
            ax.errorbar(n, y, yerr=e, color=COL["lgbm_all" if m == "lgbm" else "tabpfn"], lw=2, marker=mk, ms=5, capsize=3, label=lab)
        ax.set_xscale("log"); ax.set_xticks(n, [str(k) for k in n], fontsize=8)
        ax.set_title(s.capitalize(), loc="left", fontweight="bold"); ax.set_xlabel("context rows"); ax.set_ylabel("error at +2 s (m)")
    axes[0, 0].legend(fontsize=9, frameon=False)
    fig.suptitle("Test 3 · error against amount of data (mean ± sd over 3 draws)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "evidence_test3_little_data.png", dpi=150); plt.close(fig)
    # Real data
    reals = [s for s in REAL if s in E]
    if reals:
        fig, axes = plt.subplots(1, len(reals), figsize=(5 * len(reals), 3.8), squeeze=False)
        for ax, s in zip(axes[0], reals):
            rc = E[s]["real_check"]; hs = ["1.2", "2"]; w = 0.2
            for i, m in enumerate(mods):
                ys = [rc[m][h]["error_m"] for h in hs]
                xs = np.arange(len(hs)) + (i - 1.5) * w
                ax.bar(xs, ys, w, color=COL[m], edgecolor="white", linewidth=2, label=NAMES[m])
                for xx, h, yy in zip(xs, hs, ys):
                    ax.text(xx, yy * 1.02, f"{rc[m][h]['coverage']:.0%}", ha="center", fontsize=7.5, color="#333")
            ax.set_xticks(range(len(hs)), [f"+{h} s" for h in hs]); ax.set_ylabel("mean error (m)")
            ax.set_title(s.upper() if s == "eth" else s.capitalize(), loc="left", fontweight="bold"); ax.grid(axis="x", visible=False)
        axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
        fig.suptitle("Real pedestrians · error (bars) and held-out coverage of the 80% box (numbers)", x=0.01, ha="left", fontsize=11)
        fig.tight_layout(); fig.savefig(FIG / "evidence_real_data.png", dpi=150); plt.close(fig)


def report():
    E = {p.stem: json.loads(p.read_text()) for p in sorted(PARTS.glob("*.json"))}
    V = verdicts(E)
    cost = {s: E[s]["cost"] for s in E}
    out = dict(protocol=__doc__, scenes=E, verdicts=V, cost=cost)
    OUT.mkdir(exist_ok=True); (OUT / "evidence.json").write_text(json.dumps(out, indent=1, default=float))
    figures(E)
    print("\nVERDICT (rules in the docstring, fixed before the run)\n")
    w = max(len(r["test"]) for r in V)
    for r in V:
        print(f"{r['test']:<{w}}  {r['scene']:<8} {r['verdict']:<17} {r['detail']}")
    calls = sum(c.get("api_calls", 0) for c in cost.values()); secs = [c["tabpfn_seconds_per_batch"] for c in cost.values() if c.get("tabpfn_seconds_per_batch")]
    cred = [c["credits_per_call"] for c in cost.values() if c.get("credits_per_call")]
    print(f"\nTabPFN API: {calls} calls; {np.mean(secs):.1f} s per fit+predict batch (both axes)" if secs else "")
    if cred:
        print(f"credits per call: {np.mean(cred):.0f}")
    print(f"wrote {OUT / 'evidence.json'} and {FIG}/evidence_*.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=[])
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    for s in a.scenes:
        run_scene(s)
    if a.report:
        report()


if __name__ == "__main__":
    main()
