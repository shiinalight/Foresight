"""Intent layer test: does a TabPFN intent classifier feeding per-mode TabPFN range models beat the single range model?

    python tools/intent_layer.py --dry      # LightGBM stands in for every TabPFN fit; outputs to the scratchpad only
    python tools/intent_layer.py            # hosted TabPFN API (tabpfn-client); writes the result files
    python tools/intent_layer.py --plot     # figure + tables from results/intent_layer.csv only

GATE (fixed before any result; printed first):
  the intent->range chain is adopted as a validated layer if, on the test rows labelled turn_left/turn_right, its
  interval score (90%, both axes averaged; Gneiting & Raftery) is lower than the single TabPFN range model's with a
  paired person-bootstrap 95% interval excluding 0, AND its overall interval score is not worse (interval includes 0
  or favours the chain). Otherwise it is reported as tested and not adopted.
  Operational form: d = chained - single per test row, averaged over context seeds per row (pooled seed x row records,
  persons resampled), 2,000 person-bootstrap draws (rng 0). Adopt iff upper(d | turns) < 0 and lower(d | all) <= 0.
  Decided per setting. The gate arm 'chained' reads the mixture quantiles from the exact mixture CDF (below); the
  400-draw estimator is a labelled sensitivity arm with its own verdict, because in the dry run it alone shifted the
  single model's interval score by -0.04 to -0.12 (it widens the 5-95 box), larger than the effects being tested.

Data (features f4, horizon h == 2.0 only, mode labels from features.build_rows; 'unknown' rows dropped):
  S (sim): data/crowd_cross_t2, noise 0.1, rows_for(horizons=(1.0, 2.0)) (cache exists); split_by_episode
     (test_frac 0.25, seed 0); 1,500 fixed test rows (random_state 0); context 2,000 random train rows per seed.
  R (real): train data/hotel, test data/eth, noise 0. The f4 caches on disk are for horizons (1.2, 2.0, 4.8);
     build_rows emits each horizon's rows independently, so the h == 2.0 rows are the same as with (1.0, 2.0). They are
     used instead of building new (1.0, 2.0) caches, which would write new files under data/. Context 2,000 random
     Hotel rows per seed; all ETH rows with known mode, capped at 1,500 (random_state 0).
  Person = episode:agent_id.

Models (same context rows for every model within a seed):
  intent: TabPFN classifier (hosted, tabpfn_client.TabPFNClassifier, default settings) vs LGBMClassifier
     (200 trees, n_jobs 2) on X = features.X_of, y = mode in (straight, stop, turn_left, turn_right).
  single: TabPFN regressor quantiles (0.05, 0.1, 0.5, 0.9, 0.95) per axis (ddx, ddy) - models.make('tabpfn', api=True).
  chained: per mode, a TabPFN regressor on the context rows of that mode (modes with < 30 rows are skipped and their
     component is the single model's quantiles), predicting every test row. Per mode, the predictive distribution is
     the piecewise-linear inverse CDF through the 5 sorted quantiles, tails linear from q05 - 1.5*(q95 - q05) at u = 0
     and to q95 + 1.5*(q95 - q05) at u = 1. Mixture weighted by the TabPFN classifier's probabilities p_m.
     Gate arm 'chained': the mixture's 0.05/0.1/0.5/0.9/0.95 quantiles from the exact mixture CDF sum_m p_m F_m
     (piecewise linear between the union of the modes' knots; the limit of the draws; a one-hot input returns the
     mode's own quantiles exactly). Sensitivity arm 'chained_draws400' (the spec as first written): per test row and
     axis, 400 stratified draws u_i = (i + 0.5)/400 per mode, each weighted p_m / 400, quantiles by weighted-CDF
     midpoint interpolation. 'single_via_draws400' = the single model's quantiles through the 400-draw estimator
     (measures that estimator's own effect). oracle-intent ('oracle_intent', 'oracle_draws400'): the same mixtures
     with one-hot weights on the true mode (upper bound, no extra calls).
Metrics per test row: error of the median (m), joint coverage of the 5-95% box (both axes), box width along/across,
  interval score at 90% (alpha 0.1, averaged over the 2 axes), pinball over the 5 quantiles (averaged over quantiles
  and axes). Reported overall and by true mode (straight / stop / turns), with 95% person-bootstrap intervals and the
  paired difference chained - single.
API budget: first attempt assumed 10,000 credits per call with a 600,000 cap; the server measured 130,000 for one
  classifier call and the run stopped there. Re-run on a fresh account with an 8,000,000 cap (--cap): one regressor
  call is made first (--measure) to measure its price on the server; the plan is then costed with measured prices
  (--clf-price, --reg-price; cached calls cost nothing) and aborts if over the cap. Seeds: Setting S 0, 1 (the trim
  made under the first cap, kept as instructed), Setting R 0, 1, 2. After every paid call the server delta is
  compared with the plan and the run stops if the projection exceeds the cap. Raw API outputs cached in the
  scratchpad (--cache) so a crash after the calls costs nothing on re-run. Server-measured usage appended to
  results/api_usage.csv.
Outputs: results/intent_layer.csv (long format), media/fig_intent_layer.png.
"""
import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "2")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from run_offline import rows_for  # noqa: E402

GATE = __doc__.split("GATE (fixed before any result; printed first):\n")[1].split("\n\nData")[0]
QS = (0.05, 0.1, 0.5, 0.9, 0.95)
CLASSES = ("straight", "stop", "turn_left", "turn_right")
SUBSETS = ("all", "straight", "stop", "turns")
ARMS = ("single", "chained", "oracle_intent", "chained_draws400", "oracle_draws400", "single_via_draws400")
N_CTX, N_TEST, MIN_MODE = 2000, 1500, 30
N_DRAW, ALPHA, B = 400, 0.1, 2000
CREDITS_PER_CALL, CAP = 10_000, 600_000     # first-attempt defaults; overridden by --cap / prices
PRICE = {"clf": 130_000, "reg": 10_000}      # credits per call; clf measured on the server 2026-11-22
USAGE = Path("results/api_usage.csv")
DEFAULT_CACHE = Path(os.environ.get("TEMP", "/tmp")) / "claude" / "4d5394ce83e30a2b" / \
    "8b661486-3ed0-45a1-9f00-25c582cb9650" / "scratchpad" / "intent_cache"


# ---------------------------------------------------------------- data
def load_S():
    df = rows_for("data/crowd_cross_t2", 0.1, horizons=(1.0, 2.0))
    df = df[np.isclose(df.h, 2.0) & df["mode"].isin(CLASSES)]
    tr, _, te = F.split_by_episode(df, test_frac=0.25, seed=0)
    train, test = df[tr], df[te]
    return train, test.sample(min(N_TEST, len(test)), random_state=0)


def load_R():
    def one(d):
        df = rows_for(d, 0.0, horizons=(1.2, 2.0, 4.8))
        return df[np.isclose(df.h, 2.0) & df["mode"].isin(CLASSES)]
    train, test = one("data/hotel"), one("data/eth")
    return train, (test.sample(N_TEST, random_state=0) if len(test) > N_TEST else test)


def persons(df):
    return (df["episode"].astype(str) + ":" + df["agent_id"].astype(str)).to_numpy()


# ---------------------------------------------------------------- backends
class Backend:
    """TabPFN (hosted) or, with dry=True, LightGBM stand-ins. Raw outputs are cached per call (classifier: raw
    probabilities + classes_; regressor: per axis), so a failure after a paid call never re-buys it."""

    def __init__(self, dry, cache, on_call=None):
        self.dry, self.cache, self.calls, self.on_call = dry, Path(cache), 0, on_call
        self.cache.mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        return self.cache / f"{'dry' if self.dry else 'api'}_{key}.npz"

    def classifier(self, key, X, y, Xte):
        p = self._path(key)
        if not p.exists():
            if self.dry:
                from lightgbm import LGBMClassifier
                m = LGBMClassifier(n_estimators=200, n_jobs=2, verbose=-1, random_state=1).fit(X, y)
            else:
                from tabpfn_client import TabPFNClassifier
                self.calls += 1              # counted before the call: a failed predict may still be charged
                m = TabPFNClassifier().fit(X, y)
            np.savez(p, proba=m.predict_proba(Xte), classes=np.asarray(m.classes_))
            if self.on_call and not self.dry:
                self.on_call("clf")
        z = np.load(p)
        return align(z["proba"], z["classes"])

    def regressor(self, key, X, Y, Xte):
        out = np.zeros((len(Xte), 2, len(QS)))
        for ax in range(2):
            p = self._path(f"{key}_ax{ax}")
            if not p.exists():
                if self.dry:
                    q = M.make("lgbm:n100", n_jobs=2).fit(X, Y).predict_quantiles(Xte, QS)[:, ax]
                else:   # exactly what models.TabPFNQuantile(api=True) does, one axis per call
                    self.calls += 1
                    reg = M.make("tabpfn", api=True)._new().fit(X, Y[:, ax])
                    q = np.stack(reg.predict(Xte, output_type="quantiles", quantiles=list(QS)), axis=1)
                np.savez(p, a=q)
                if self.on_call and not self.dry:
                    self.on_call("reg")
            out[:, ax] = np.load(p)["a"]
        return np.sort(out, axis=2)


def align(P, classes):
    out = np.zeros((len(P), len(CLASSES)))
    for j, c in enumerate(np.asarray(classes).astype(int)):
        out[:, c] = P[:, j]
    return out


def lgbm_proba(X, y, Xte, seed=0):
    from lightgbm import LGBMClassifier
    m = LGBMClassifier(n_estimators=200, n_jobs=2, verbose=-1, random_state=seed).fit(X, y)
    return align(m.predict_proba(Xte), m.classes_)


# ---------------------------------------------------------------- mixture
PK = np.array([0.0, 0.05, 0.1, 0.5, 0.9, 0.95, 1.0])
U = (np.arange(N_DRAW) + 0.5) / N_DRAW
_SEG = np.clip(np.searchsorted(PK, U, side="right") - 1, 0, len(PK) - 2)
_W = (U - PK[_SEG]) / (PK[_SEG + 1] - PK[_SEG])


def _knots(Qm):
    Qm = np.sort(Qm, axis=-1)
    span = Qm[..., 4] - Qm[..., 0]
    V = np.concatenate([(Qm[..., 0] - 1.5 * span)[..., None], Qm, (Qm[..., 4] + 1.5 * span)[..., None]], axis=-1)
    return V + 1e-9 * np.arange(len(PK))           # strictly increasing for interp (ties in degenerate rows)


def mixture(Qm, W):
    """Qm [n, K, 2, 5] per-mode quantiles, W [n, K] weights -> [n, 2, 5] mixture quantiles (400 stratified draws per
    mode, each weighted p_m / 400; zero-weight draws dropped; quantiles by weighted-CDF midpoint interpolation)."""
    V = _knots(Qm)
    S = V[..., _SEG] * (1 - _W) + V[..., _SEG + 1] * _W            # [n, K, 2, N_DRAW]
    n, K = W.shape
    vals = S.transpose(0, 2, 1, 3).reshape(n, 2, K * N_DRAW)
    w = np.repeat(W / N_DRAW, N_DRAW, axis=1)                     # [n, K*N_DRAW]
    out = np.zeros((n, 2, len(QS)))
    for i in range(n):
        keep = w[i] > 0
        for ax in range(2):
            v, ww = vals[i, ax][keep], w[i][keep]
            o = np.argsort(v, kind="stable")
            v, ww = v[o], ww[o]
            c = (np.cumsum(ww) - ww / 2) / ww.sum()
            out[i, ax] = np.interp(QS, c, v)
    return out


def mixture_exact(Qm, W):
    """Same mixture, quantiles read from the exact mixture CDF (sum_m p_m F_m, piecewise linear between the union of
    all modes' knots): the limit of the draws as their number grows. Used as a check on the 400-draw estimator."""
    V = _knots(Qm)
    n, K = W.shape
    out = np.zeros((n, 2, len(QS)))
    for i in range(n):
        keep = W[i] > 0
        for ax in range(2):
            Vk = V[i, keep, ax]                       # [k, 7]
            grid = np.sort(Vk.ravel())
            Fm = np.stack([np.interp(grid, Vk[j], PK) for j in range(len(Vk))])
            Fg = (W[i, keep][:, None] * Fm).sum(0) / W[i, keep].sum()
            Fg = np.maximum.accumulate(Fg) + 1e-12 * np.arange(len(grid))
            out[i, ax] = np.interp(QS, Fg, grid)
    return out


# ---------------------------------------------------------------- metrics
def range_rows(Q, Y):
    lo, hi, med = Q[:, :, 0], Q[:, :, -1], Q[:, :, 2]
    isc = (hi - lo) + (2 / ALPHA) * np.maximum(lo - Y, 0) + (2 / ALPHA) * np.maximum(Y - hi, 0)
    pin = np.zeros(len(Y))
    for k, q in enumerate(QS):
        d = Y - Q[:, :, k]
        pin += np.maximum(q * d, (q - 1) * d).mean(1)
    return dict(err=np.linalg.norm(med - Y, axis=1), joint90=((Y >= lo) & (Y <= hi)).all(1).astype(float),
                width_along=hi[:, 0] - lo[:, 0], width_across=hi[:, 1] - lo[:, 1],
                interval_score=isc.mean(1), pinball=pin / len(QS))


def intent_rows(P, y):
    Pc = np.clip(P, 1e-12, 1)
    Pc = Pc / Pc.sum(1, keepdims=True)
    oh = np.eye(len(CLASSES))[y]
    return dict(logloss=-np.log(Pc[np.arange(len(y)), y]), brier=((P - oh) ** 2).sum(1),
                correct=(P.argmax(1) == y).astype(float), pmax=P.max(1), pred=P.argmax(1))


def ece(pmax, correct, bins=10):
    b = np.clip((pmax * bins).astype(int), 0, bins - 1)
    return float(sum(abs(correct[b == k].mean() - pmax[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any()))


def boot_weights(pers, seed=0):
    up, inv = np.unique(pers, return_inverse=True)
    rng = np.random.default_rng(seed)
    return rng.multinomial(len(up), np.full(len(up), 1 / len(up)), size=B).astype(float), inv, len(up)


def boot_mean(vals, mask, Wb, inv, P):
    s = np.bincount(inv[mask], weights=vals[mask], minlength=P)
    c = np.bincount(inv[mask], minlength=P).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (Wb @ s) / (Wb @ c)
    bs = bs[np.isfinite(bs)]
    return float(s.sum() / c.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def subset_mask(modes, sub):
    return np.ones(len(modes), bool) if sub == "all" else np.isin(modes, ["turn_left", "turn_right"]) if sub == "turns" \
        else modes == sub


# ---------------------------------------------------------------- usage
def token():
    if not os.environ.get("TABPFN_TOKEN"):
        import subprocess
        tok = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                              "[Environment]::GetEnvironmentVariable('TABPFN_TOKEN','User')"],
                             capture_output=True, text=True).stdout.strip()
        os.environ["TABPFN_TOKEN"] = tok
    import tabpfn_client
    tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
    return tabpfn_client


def used(s):
    s = str(s)
    try:
        return int(s.split("used ")[1].split(" of")[0])
    except (IndexError, ValueError):
        return None


def log_usage(calls, note, credits=None):
    """credits: the server-measured delta when available (the 10,000/call assumption is wrong for the classifier)."""
    day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    credits = calls * CREDITS_PER_CALL if credits is None else int(credits)
    Uu = pd.read_csv(USAGE)
    prev = pd.to_numeric(Uu.loc[(Uu.date_utc == day) & (Uu.run != "day total (server)"), "credits"], errors="coerce").sum()
    total = int(prev + credits)
    with open(USAGE, "a", newline="", encoding="utf-8") as f:
        f.write(f"{day},intent_layer (tools/intent_layer.py),{calls},{credits},"
                f"{note.replace(',', ';')}; running total {day}: {total} credits\n")


# ---------------------------------------------------------------- run
def plan(settings, seeds_of, be=None):
    """Cells with their fitted modes and the number of not-yet-cached calls by kind (clf, reg)."""
    cells, tot = [], {"clf": 0, "reg": 0}
    for name, (train, test) in settings.items():
        for s in seeds_of[name]:
            ctx = train.sample(min(N_CTX, len(train)), random_state=s)
            counts = ctx["mode"].value_counts().reindex(CLASSES, fill_value=0)
            fit_modes = [m for m in CLASSES if counts[m] >= MIN_MODE]
            key = f"{name}_s{s}"
            regkeys = [f"{key}_single_ax{a}" for a in range(2)] + \
                [f"{key}_mode_{m}_ax{a}" for m in fit_modes for a in range(2)]
            n = {"clf": int(not (be and be._path(f"{key}_clf").exists())),
                 "reg": sum(not (be and be._path(k).exists()) for k in regkeys)}
            cells.append(dict(setting=name, seed=s, ctx=ctx, counts=counts.to_dict(), fit_modes=fit_modes, new=n,
                              calls=1 + len(regkeys)))
            for k in tot:
                tot[k] += n[k]
    return cells, tot


def cost(tot):
    return tot["clf"] * PRICE["clf"] + tot["reg"] * PRICE["reg"]


def run(dry, cache, out_csv, out_fig):
    print("GATE (fixed before any result):\n" + GATE + "\n", flush=True)
    settings = {"S": load_S(), "R": load_R()}
    for k, (tr, te) in settings.items():
        print(f"setting {k}: train pool {len(tr)} rows ({tr.episode.nunique()} ep), test {len(te)} rows "
              f"({len(np.unique(persons(te)))} persons); test modes {te['mode'].value_counts().to_dict()}", flush=True)
    seeds_of = {"S": (0, 1), "R": (0, 1, 2)}
    be = Backend(dry, cache)
    cells, tot = plan(settings, seeds_of, None if dry else be)
    n_calls = tot["clf"] + tot["reg"]
    for c in cells:
        print(f"  plan {c['setting']} seed {c['seed']}: context modes {c['counts']}; fitted modes {c['fit_modes']}; "
              f"{c['calls']} calls ({c['new']['clf']} clf + {c['new']['reg']} reg not cached)", flush=True)
    print(f"API plan: {tot['clf']} classifier calls x {PRICE['clf']:,} + {tot['reg']} regressor calls x {PRICE['reg']:,} "
          f"= {cost(tot):,} credits (cap {CAP:,}); "
          f"{'DRY RUN: LightGBM stand-ins, no API calls' if dry else 'hosted TabPFN'}", flush=True)
    if not dry and cost(tot) > CAP:
        sys.exit("above the credit cap; abort")
    planned = cost(tot)

    client, before, err = None, None, None
    state = {"checked": False}
    if not dry:
        client = token()
        before = client.get_api_usage()
        print("server before (raw):", before, flush=True)
        print("server before:", used(before), flush=True)

    def price_check(kind):
        mid, b0 = used(client.get_api_usage()), used(before)
        state["spent_plan"] = state.get("spent_plan", 0) + PRICE[kind]
        if mid is None or b0 is None:
            print("price check: usage string not parseable", flush=True)
            return
        spent = mid - b0
        proj = spent + (planned - state["spent_plan"]) * max(1.0, spent / max(state["spent_plan"], 1))
        if not state["checked"] or spent > 1.05 * state["spent_plan"]:
            print(f"price check after {be.calls} call(s): server {spent:,} vs plan {state['spent_plan']:,}; "
                  f"projected total {proj:,.0f} (cap {CAP:,})", flush=True)
            state["checked"] = True
        if proj > CAP:
            raise RuntimeError(f"projected spend {proj:,.0f} above cap; stopping")

    be.on_call = price_check
    recs, irecs = [], []
    try:
        for ci, c in enumerate(cells):
            name, s, ctx = c["setting"], c["seed"], c["ctx"]
            test = settings[name][1]
            X, Y = F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy()
            Xte, Yte = F.X_of(test), test[["ddx", "ddy"]].to_numpy()
            y = ctx["mode"].map({m: i for i, m in enumerate(CLASSES)}).to_numpy()
            yte = test["mode"].map({m: i for i, m in enumerate(CLASSES)}).to_numpy()
            t0 = time.perf_counter()
            key = f"{name}_s{s}"
            P_tab = be.classifier(f"{key}_clf", X, y, Xte)
            P_lgb = lgbm_proba(X, y, Xte)
            Qs = be.regressor(f"{key}_single", X, Y, Xte)
            Qm = np.repeat(Qs[:, None], len(CLASSES), axis=1)
            for j, m in enumerate(CLASSES):
                if m in c["fit_modes"]:
                    sel = (ctx["mode"] == m).to_numpy()
                    Qm[:, j] = be.regressor(f"{key}_mode_{m}", X[sel], Y[sel], Xte)
            onehot = np.eye(len(CLASSES))[yte]
            Qc, Qo = mixture_exact(Qm, P_tab), mixture_exact(Qm, onehot)              # gate arms
            Qcd, Qod = mixture(Qm, P_tab), mixture(Qm, onehot)                        # 400-draw sensitivity arms
            Qsd = mixture(Qs[:, None], np.ones((len(Qs), 1)))        # single model through the 400-draw estimator
            neutral = np.abs(mixture_exact(Qs[:, None], np.ones((len(Qs), 1))) - Qs).max()
            assert neutral < 1e-6, f"exact mixture not neutral on the single model: {neutral}"
            art = range_rows(Qsd, Yte)["interval_score"].mean() - range_rows(Qs, Yte)["interval_score"].mean()
            print(f"  estimator check: exact mixture of single == single (max dev {neutral:.1e}); single through "
                  f"400 draws - raw single IS {art:+.4f}; max |draws - exact| chained {np.abs(Qc - Qcd).max():.4f}",
                  flush=True)
            meta = dict(setting=name, seed=s, row=np.arange(len(test)), person=persons(test), mode=test["mode"].to_numpy())
            for model, Q in (("single", Qs), ("chained", Qc), ("oracle_intent", Qo), ("chained_draws400", Qcd),
                             ("oracle_draws400", Qod), ("single_via_draws400", Qsd)):
                recs.append(pd.DataFrame({**meta, "model": model, **range_rows(Q, Yte)}))
            for model, P in (("tabpfn_clf", P_tab), ("lgbm_clf", P_lgb)):
                r = intent_rows(P, yte)
                irecs.append(pd.DataFrame({**meta, "model": model, "y": yte, **r}))
            print(f"{name} seed {s}: done in {time.perf_counter() - t0:.0f} s; calls so far {be.calls}", flush=True)
    except Exception as e:      # quota, network or price: keep what is done, log calls made
        err = repr(e)[:300]
        print("ERROR, stopping:", err, flush=True)
    finally:
        if not dry:
            after = client.get_api_usage()
            print("server after:", used(after), flush=True)
            log_usage(be.calls, f"intent layer test (TabPFN classifier + per-mode TabPFN regressors) S crowd_cross_t2 "
                                f"seeds {seeds_of['S']} + R hotel->eth seeds {seeds_of['R']}; h 2 s; server usage "
                                f"{used(before)} -> {used(after)} (account shared with other sessions: the delta may "
                                f"include their use); credits column = calls x measured price (clf {PRICE['clf']}; "
                                f"reg {PRICE['reg']})" + (f"; stopped on error {err}" if err else ""),
                      credits=state.get("spent_plan", 0))
    if err:
        sys.exit(1)
    R_ = pd.concat(recs, ignore_index=True)
    I_ = pd.concat(irecs, ignore_index=True)
    out = summarise(R_, I_, seeds_of)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print("wrote", out_csv, flush=True)
    sanity()
    report(out)
    plot(out, out_fig, I_)
    print(f"API calls made this run: {be.calls}")


def summarise(R_, I_, seeds_of):
    rows = []
    for name in ("S", "R"):
        Rs, Is = R_[R_.setting == name], I_[I_.setting == name]
        base = Rs[Rs.model == "single"].reset_index(drop=True)
        Wb, inv, P = boot_weights(base.person.to_numpy())
        modes = base["mode"].to_numpy()
        nseeds = len(seeds_of[name])
        for model in ARMS:
            g = Rs[Rs.model == model].reset_index(drop=True)
            assert (g.row.to_numpy() == base.row.to_numpy()).all() and (g.seed.to_numpy() == base.seed.to_numpy()).all()
            for sub in SUBSETS:
                mk = subset_mask(modes, sub)
                for met in ("err", "joint90", "width_along", "width_across", "interval_score", "pinball"):
                    v, lo, hi = boot_mean(g[met].to_numpy(), mk, Wb, inv, P)
                    rows.append(dict(section="range", setting=name, model=model, subset=sub, metric=met, value=v, lo=lo,
                                     hi=hi, n_rows=int(mk.sum() // nseeds), n_persons=len(np.unique(base.person[mk])),
                                     seeds=nseeds))
                    if model != "single":
                        d = g[met].to_numpy() - base[met].to_numpy()
                        v, lo, hi = boot_mean(d, mk, Wb, inv, P)
                        rows.append(dict(section="range_diff", setting=name, model=f"{model}-single", subset=sub,
                                         metric=met, value=v, lo=lo, hi=hi, n_rows=int(mk.sum() // nseeds),
                                         n_persons=len(np.unique(base.person[mk])), seeds=nseeds))
                    if met == "interval_score":
                        for s in seeds_of[name]:
                            ms = mk & (g.seed.to_numpy() == s)
                            rows.append(dict(section="per_seed", setting=name, model=model, subset=sub, metric=met,
                                             value=float(g[met].to_numpy()[ms].mean()), seed_id=s,
                                             n_rows=int(ms.sum()), seeds=1))
        # gate (primary: exact-CDF chained; sensitivity: 400-draw chained)
        for arm, label in (("chained", "gate"), ("chained_draws400", "gate_draws400")):
            dsub = {r["subset"]: r for r in rows if r["section"] == "range_diff" and r["setting"] == name
                    and r["model"] == f"{arm}-single" and r["metric"] == "interval_score"}
            adopt = dsub["turns"]["hi"] < 0 and dsub["all"]["lo"] <= 0
            rows.append(dict(section=label, setting=name, model=f"{arm}-single", subset="turns+all",
                             metric="adopted", value=float(adopt), lo=dsub["turns"]["hi"], hi=dsub["all"]["lo"],
                             n_rows=dsub["turns"]["n_rows"], n_persons=dsub["turns"]["n_persons"], seeds=nseeds,
                             note="lo = upper CI of d on turns (must be < 0); hi = lower CI of d on all rows (must be <= 0)"))
        # intent
        bi = Is[Is.model == "tabpfn_clf"].reset_index(drop=True)
        for model in ("tabpfn_clf", "lgbm_clf"):
            g = Is[Is.model == model].reset_index(drop=True)
            y = g.y.to_numpy()
            mk = np.ones(len(g), bool)
            for met in ("logloss", "brier", "correct"):
                v, lo, hi = boot_mean(g[met].to_numpy(), mk, Wb, inv, P)
                rows.append(dict(section="intent", setting=name, model=model, subset="all",
                                 metric="accuracy" if met == "correct" else met, value=v, lo=lo, hi=hi,
                                 n_rows=len(g) // nseeds, seeds=nseeds))
            recs_b, recs_v = [], []
            for k, cl in enumerate(CLASSES):
                mk = y == k
                if mk.any():
                    v, lo, hi = boot_mean(g.correct.to_numpy(), mk, Wb, inv, P)
                    rows.append(dict(section="intent", setting=name, model=model, subset=cl, metric="recall", value=v,
                                     lo=lo, hi=hi, n_rows=int(mk.sum() // nseeds), seeds=nseeds))
                    s_ = np.bincount(inv[mk], weights=g.correct.to_numpy()[mk], minlength=P)
                    c_ = np.bincount(inv[mk], minlength=P).astype(float)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        recs_b.append((Wb @ s_) / (Wb @ c_))
                    recs_v.append(v)
            ba_b = np.nanmean(np.vstack(recs_b), axis=0)
            rows.append(dict(section="intent", setting=name, model=model, subset="all", metric="balanced_accuracy",
                             value=float(np.mean(recs_v)), lo=float(np.nanpercentile(ba_b, 2.5)),
                             hi=float(np.nanpercentile(ba_b, 97.5)), n_rows=len(g) // nseeds, seeds=nseeds))
            per_seed_ece = [ece(g.pmax[g.seed == s].to_numpy(), g.correct[g.seed == s].to_numpy()) for s in seeds_of[name]]
            per_seed_ba = [np.mean([g.correct[(g.seed == s) & (g.y == k)].mean() for k in range(4) if ((g.seed == s) & (g.y == k)).any()])
                           for s in seeds_of[name]]
            rows.append(dict(section="intent", setting=name, model=model, subset="all", metric="ece10",
                             value=float(np.mean(per_seed_ece)), lo=float(np.min(per_seed_ece)),
                             hi=float(np.max(per_seed_ece)), n_rows=len(g) // nseeds, seeds=nseeds,
                             note="per-seed mean; lo/hi = min/max over seeds"))
            rows.append(dict(section="intent", setting=name, model=model, subset="all", metric="balanced_accuracy_seed_sd",
                             value=float(np.std(per_seed_ba, ddof=1)) if nseeds > 1 else np.nan, n_rows=len(g) // nseeds,
                             seeds=nseeds))
            if model == "lgbm_clf":
                for met in ("logloss", "brier"):
                    d = bi[met].to_numpy() - g[met].to_numpy()
                    v, lo, hi = boot_mean(d, np.ones(len(g), bool), Wb, inv, P)
                    rows.append(dict(section="intent_diff", setting=name, model="tabpfn-lgbm", subset="all", metric=met,
                                     value=v, lo=lo, hi=hi, n_rows=len(g) // nseeds, seeds=nseeds))
        # reliability bins (for the figure)
        for model in ("tabpfn_clf", "lgbm_clf"):
            g = Is[Is.model == model]
            b = np.clip((g.pmax.to_numpy() * 10).astype(int), 0, 9)
            for k in range(10):
                m = b == k
                if m.any():
                    rows.append(dict(section="reliability", setting=name, model=model, subset=f"bin{k}", metric="acc_vs_conf",
                                     value=float(g.correct.to_numpy()[m].mean()), lo=float(g.pmax.to_numpy()[m].mean()),
                                     hi=np.nan, n_rows=int(m.sum()), seeds=nseeds))
    return pd.DataFrame(rows)


def sanity():
    """One-hot weights must reproduce the mode's own quantiles (exact CDF: to rounding; 400 draws: up to the kink
    resolution of the stratified grid). Quantiles shaped like a normal (z-scores), so the check is in sigma units."""
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    z = norm.ppf(QS)
    Q = rng.normal(size=(50, 4, 2, 1)) + np.exp(rng.normal(-0.5, 0.3, size=(50, 4, 2, 1))) * z
    W = np.eye(4)[rng.integers(0, 4, 50)]
    want = Q[np.arange(50), W.argmax(1)]
    print(f"mixture sanity (one-hot, sigma ~0.6): draws max abs dev {np.abs(mixture(Q, W) - want).max():.4f}, "
          f"exact max abs dev {np.abs(mixture_exact(Q, W) - want).max():.2e}")


def fmt(r):
    return f"{r.value:.3f} [{r.lo:.3f}, {r.hi:.3f}]"


def report(out):
    for name in ("S", "R"):
        a = out[(out.section == "gate") & (out.setting == name)].value.iloc[0]
        b = out[(out.section == "gate_draws400") & (out.setting == name)].value.iloc[0]
        if a != b:
            print(f"NOTE: Setting {name}: the exact-CDF gate ({'adopted' if a else 'not adopted'}) and the 400-draw "
                  f"sensitivity ({'adopted' if b else 'not adopted'}) DISAGREE")
    for name in ("S", "R"):
        print(f"\n=== Setting {name} ===")
        g = out[(out.section == "intent") & (out.setting == name)]
        print("intent classifier (95% person bootstrap):")
        for met in ("balanced_accuracy", "accuracy", "logloss", "brier", "ece10"):
            line = f"  {met:<18}"
            for model in ("tabpfn_clf", "lgbm_clf"):
                r = g[(g.model == model) & (g.metric == met)].iloc[0]
                line += f"  {model}: {fmt(r)}"
            print(line)
        for cl in CLASSES:
            line = f"  recall {cl:<11}"
            for model in ("tabpfn_clf", "lgbm_clf"):
                r = g[(g.model == model) & (g.metric == "recall") & (g.subset == cl)]
                line += f"  {model}: {fmt(r.iloc[0]) if len(r) else 'n/a'}"
            print(line + (f"  (n {int(r.iloc[0].n_rows)})" if len(r) else ""))
        d = out[(out.section == "intent_diff") & (out.setting == name)]
        for _, r in d.iterrows():
            print(f"  tabpfn - lgbm {r.metric}: {fmt(r)}")
        print("range (95% person bootstrap):")
        rg = out[(out.section == "range") & (out.setting == name)]
        for sub in SUBSETS:
            n = rg[(rg.subset == sub)].iloc[0]
            print(f"  [{sub}] rows {int(n.n_rows)}, persons {int(n.n_persons)}")
            for model in ("single", "chained", "oracle_intent"):
                x = rg[(rg.subset == sub) & (rg.model == model)].set_index("metric")
                print(f"    {model:<14} IS {fmt(x.loc['interval_score'])}  err {x.loc['err'].value:.3f}  "
                      f"joint {100 * x.loc['joint90'].value:.1f}%  width {x.loc['width_along'].value:.2f}/"
                      f"{x.loc['width_across'].value:.2f}  pinball {x.loc['pinball'].value:.4f}")
            dd = out[(out.section == "range_diff") & (out.setting == name) & (out.subset == sub)]
            for model in ("chained-single", "oracle_intent-single", "chained_draws400-single", "single_via_draws400-single"):
                x = dd[dd.model == model].set_index("metric")
                print(f"    d {model:<20} IS {fmt(x.loc['interval_score'])}  err {fmt(x.loc['err'])}  "
                      f"joint {fmt(x.loc['joint90'])}")
        ps = out[(out.section == "per_seed") & (out.setting == name) & (out.subset.isin(["all", "turns"]))]
        print("per seed interval score:")
        print(ps.pivot_table(index=["subset", "seed_id"], columns="model", values="value").round(3).to_string())
        for sec, lab_ in (("gate", "GATE"), ("gate_draws400", "sensitivity (400-draw chained)")):
            gt = out[(out.section == sec) & (out.setting == name)].iloc[0]
            print(f"{lab_} {name}: {'ADOPTED' if gt.value else 'tested, NOT adopted'}  (upper CI of d on turns "
                  f"{gt.lo:.3f} must be < 0; lower CI of d on all {gt.hi:.3f} must be <= 0; turn rows {int(gt.n_rows)}, "
                  f"turn persons {int(gt.n_persons)})")


def plot(out, fig_path, I_=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"tabpfn_clf": "#e8590c", "lgbm_clf": "#2a6fdb", "single": "#6c757d", "chained": "#e8590c",
           "oracle_intent": "#2b8a3e"}
    lab = {"tabpfn_clf": "TabPFN classifier", "lgbm_clf": "LightGBM classifier", "single": "single TabPFN range",
           "chained": "intent -> range (chained)", "oracle_intent": "oracle intent (true mode)"}
    sname = {"S": "S: crowd_cross_t2 (sim)", "R": "R: Hotel -> ETH (real)"}
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    # 1a reliability
    ax = axs[0, 0]
    for name, ls in (("S", "-"), ("R", "--")):
        for model in ("tabpfn_clf", "lgbm_clf"):
            g = out[(out.section == "reliability") & (out.setting == name) & (out.model == model)]
            e = out[(out.section == "intent") & (out.setting == name) & (out.model == model) & (out.metric == "ece10")]
            ax.plot(g.lo, g.value, ls=ls, marker="o", color=col[model], ms=4,
                    label=f"{lab[model]}, {name} (ECE {e.value.iloc[0]:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls=":")
    ax.set_xlabel("max predicted probability (bin mean)"); ax.set_ylabel("accuracy in bin")
    ax.set_title("Intent calibration (10 bins, all seeds pooled)", fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    # 1b recall
    ax = axs[0, 1]
    x0 = np.arange(len(CLASSES))
    k = 0
    for name in ("S", "R"):
        for model in ("tabpfn_clf", "lgbm_clf"):
            g = out[(out.section == "intent") & (out.setting == name) & (out.model == model) & (out.metric == "recall")]
            g = g.set_index("subset").reindex(CLASSES)
            off = (k - 1.5) * 0.2
            ax.bar(x0 + off, g.value, 0.19, color=col[model], alpha=1.0 if name == "S" else 0.45,
                   yerr=[g.value - g.lo, g.hi - g.value], capsize=2,
                   label=f"{lab[model]}, {name}")
            k += 1
    ba = out[(out.section == "intent") & (out.metric == "balanced_accuracy")]
    txt = "; ".join(f"{r.setting} {('TabPFN' if r.model == 'tabpfn_clf' else 'LGBM')} {r.value:.2f}" for r in ba.itertuples())
    ax.set_xticks(x0); ax.set_xticklabels(CLASSES); ax.set_ylabel("recall"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Intent per-class recall (balanced acc: {txt})", fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    # 2 interval score and coverage by true mode
    groups = [(n, s) for n in ("S", "R") for s in SUBSETS]
    xg = np.arange(len(groups))
    for ax, met, ylab in ((axs[1, 0], "interval_score", "interval score, 90% (m, lower is better)"),
                          (axs[1, 1], "joint90", "joint coverage of the 5-95% box")):
        for j, model in enumerate(("single", "chained", "oracle_intent")):
            vals = [out[(out.section == "range") & (out.setting == n) & (out.subset == s) & (out.model == model)
                        & (out.metric == met)].iloc[0] for n, s in groups]
            v = np.array([r.value for r in vals]); lo = np.array([r.lo for r in vals]); hi = np.array([r.hi for r in vals])
            ax.bar(xg + (j - 1) * 0.27, v, 0.26, color=col[model], yerr=[v - lo, hi - v], capsize=2, label=lab[model])
        nrow = [int(out[(out.section == "range") & (out.setting == n) & (out.subset == s)].iloc[0].n_rows) for n, s in groups]
        ax.set_xticks(xg); ax.set_xticklabels([f"{n}\n{s}\n(n {r})" for (n, s), r in zip(groups, nrow)], fontsize=8)
        ax.axvline(3.5, color="black", lw=0.8)
        if met == "joint90":
            ax.axhline(0.81, color="grey", ls=":", lw=1, label="0.9 x 0.9 (independent axes)")
            ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylab); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7)
        gates = out[out.section == "gate"].set_index("setting")
        ax.set_title(f"By true mode, 95% person bootstrap. Gate: S {'adopted' if gates.loc['S'].value else 'not adopted'}, "
                     f"R {'adopted' if gates.loc['R'].value else 'not adopted'}", fontsize=9)
    fig.suptitle("Intent layer (TabPFN classifier -> per-mode TabPFN range) vs single TabPFN range, +2 s, f4 features; "
                 + ", ".join(sname.values()), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=130)
    print("wrote", fig_path)


def measure(cache):
    """One regressor call (Setting S, seed 0, single model, axis 0; cached for the run), priced on the server."""
    train, test = load_S()
    ctx = train.sample(N_CTX, random_state=0)
    be = Backend(False, cache)
    if be._path("S_s0_single_ax0").exists():
        sys.exit("S_s0_single_ax0 already cached: nothing to measure")
    client = token()
    b = client.get_api_usage()
    print("server before (raw):", b, flush=True)
    calls, err = 0, None
    try:
        calls = 1
        reg = M.make("tabpfn", api=True)._new().fit(F.X_of(ctx), ctx["ddx"].to_numpy())
        q = np.stack(reg.predict(F.X_of(test), output_type="quantiles", quantiles=list(QS)), axis=1)
        np.savez(be._path("S_s0_single_ax0"), a=q)
    except Exception as e:
        err = repr(e)[:300]
        print("ERROR:", err, flush=True)
    a = client.get_api_usage()
    d = used(a) - used(b) if None not in (used(a), used(b)) else None
    print(f"server after: {used(a)}; regressor call price {d} credits (2,000 x 39 context, 1,500 test rows, 5 quantiles)",
          flush=True)
    log_usage(calls, f"intent layer price probe: 1 TabPFN regressor call (S seed 0 single ddx; cached for the run); server "
                     f"usage {used(b)} -> {used(a)}" + (f"; error {err}" if err else ""), credits=d)


def main():
    global CAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="LightGBM stand-ins, no API; outputs to the cache dir")
    ap.add_argument("--plot", action="store_true", help="figure and tables from the CSV only")
    ap.add_argument("--measure", action="store_true", help="one regressor call to price it on the server")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--clf-price", type=int, default=PRICE["clf"])
    ap.add_argument("--reg-price", type=int, default=PRICE["reg"])
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    a = ap.parse_args()
    CAP = a.cap
    PRICE.update(clf=a.clf_price, reg=a.reg_price)
    csv, fig = Path("results/intent_layer.csv"), Path("media/fig_intent_layer.png")
    if a.dry:
        csv, fig = Path(a.cache) / "dry_intent_layer.csv", Path(a.cache) / "dry_fig_intent_layer.png"
    if a.plot:
        print("GATE (fixed before any result):\n" + GATE + "\n")
        out = pd.read_csv(csv)
        report(out)
        plot(out, fig)
        return
    if a.measure:
        measure(a.cache)
        return
    run(a.dry, a.cache, csv, fig)


if __name__ == "__main__":
    main()
