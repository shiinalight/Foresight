"""Is split's gain just wider boxes? Frozen vs split vs LightGBM, each with and without an online conformal offset.

    python -B tools/live_variants_cp.py                        # full Setting B, ~5 min on 3 CPU threads
    python -B tools/live_variants_cp.py --smoke --out <dir>    # 1 episode, 2 moments

Setting B exactly as tools/live_variants.py builds it (imported: same bank, same 6 test episodes that switch
calm -> erratic at 20 s, same moments every 4 s, same query rows). Variants, all at h = 2 s:
  frozen1k         TabPFN on 1,000 random bank rows (identical to live_variants 'frozen'; reference)
  frozen2k         TabPFN on 2,000 random bank rows, cache built once
  split2k          average of frozen2k and recent-only (<= 250 completed rows of this scene) quantiles;
                   frozen2k alone while fewer than 20 recent rows
  lgbm_all_frozen  LightGBM quantile (make('lgbm', n_jobs=2), 300 trees) trained ONCE on all bank rows
  <v>+offset       the same quantiles, 5-95% box widened by an online conformal offset
                   (foresight.core.conformal.online_offsets: d0 = 0, alpha = 0.2, eta = 0.02, group = episode,
                   t = query time, t_known = t + 2; strictly causal)

Writes results/live_variants_cp.csv (variant x query row) and media/fig_live_variants_cp.png.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import live_variants as LV  # noqa: E402  (Setting B construction, tabpfn(), boot())
from foresight.core import conformal as CP  # noqa: E402
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402

torch.set_num_threads(3)
QS = LV.QS
REC_CAP, THIN = 250, 20
ALPHA_CP, ETA, D0, LAG = 0.2, 0.02, 0.0, 2.0
IS_ALPHA = 0.1
BASES = ["frozen1k", "frozen2k", "split2k", "lgbm_all_frozen"]
VARIANTS = [v for b in BASES for v in (b, b + "+offset")]
KEY = ["episode", "t", "agent_id"]
PERIODS = [("overall", None), ("before", lambda d: d.since < 0), ("after", lambda d: d.since >= 0)]


def periods(res):
    """The periods that have rows (a smoke run may have no after-switch rows)."""
    return [(n, m) for n, m in PERIODS if (m is None or m(res).any())]
COLORS = {"frozen1k": "#8b949e", "frozen2k": "#1f2328", "split2k": "#2a6fdb", "lgbm_all_frozen": "#e8590c"}
LABELS = {"frozen1k": "Frozen 1k (ref)", "frozen2k": "Frozen 2k", "split2k": "Split 2k + this scene",
          "lgbm_all_frozen": "LightGBM, all bank rows"}


def box_metrics(lo, hi, Y):
    """Joint coverage, mean width (along, across averaged) and 90% interval score (axes averaged) of a box."""
    cov = ((Y >= lo) & (Y <= hi)).all(axis=1).astype(float)
    w = hi - lo
    a = IS_ALPHA
    isc = w + (2 / a) * np.maximum(lo - Y, 0) + (2 / a) * np.maximum(Y - hi, 0)
    return dict(joint90=cov, width=w.mean(1), w_along=w[:, 0], w_across=w[:, 1], is90=isc.mean(1))


def pinball(Q, Y):
    a = np.asarray(QS)[None, None, :]
    d = Y[:, :, None] - Q
    return np.maximum(a * d, (a - 1) * d).mean(axis=(1, 2))


def run(src, test, obs, moments, max_moments=None):
    glob1 = src.sample(min(1000, len(src)), random_state=0)          # = live_variants 'frozen'
    glob2 = src.sample(min(2000, len(src)), random_state=0)
    build = {}
    models = {}
    for name, g in (("frozen1k", glob1), ("frozen2k", glob2)):
        m = LV.tabpfn()
        t0 = time.perf_counter(); m.fit(F.X_of(g), g[["ddx", "ddy"]].to_numpy()); build[name] = time.perf_counter() - t0
        models[name] = m
        print(f"{name}: cache built once on {len(g)} rows in {build[name]:.1f} s", flush=True)
    lg = M.make("lgbm", n_jobs=2)
    Xb, Yb = F.X_of(src), src[["ddx", "ddy"]].to_numpy()
    print(f"lgbm_all_frozen: training once on all {len(src)} bank rows (10 quantile models x 300 trees) ...", flush=True)
    t0 = time.perf_counter(); lg.fit(Xb, Yb); lg.predict_quantiles(Xb[:5], QS); build["lgbm_all_frozen"] = time.perf_counter() - t0
    print(f"lgbm_all_frozen: trained in {build['lgbm_all_frozen']:.1f} s (warm-up predict included)", flush=True)
    build["split2k"] = build["frozen2k"]

    per = {v: [] for v in BASES}           # list of (Q, Y, qrows, t_moment, n_ctx, n_rec, fallback)
    n_mom = sum(len(v[:max_moments]) for v in moments.values()); k = 0; t_start = time.perf_counter()
    for ep, times in moments.items():
        live = obs[obs.episode == ep]
        q_ep = test[test.episode == ep]
        for t in times[:max_moments]:
            k += 1
            q = q_ep[np.isclose(q_ep.t, t)]
            Xq, Y = F.X_of(q), q[["ddx", "ddy"]].to_numpy()
            out = {}
            for name in ("frozen1k", "frozen2k"):
                t1 = time.perf_counter(); Q = models[name].predict_quantiles(Xq, QS)
                out[name] = (Q, time.perf_counter() - t1, len(glob1 if name == "frozen1k" else glob2), False)
            t1 = time.perf_counter(); Q = lg.predict_quantiles(Xq, QS)
            out["lgbm_all_frozen"] = (Q, time.perf_counter() - t1, len(src), False)
            rec = live[live.t_known <= t].sort_values("t_known").tail(REC_CAP)
            Qf, tpf = out["frozen2k"][0], out["frozen2k"][1]
            if len(rec) < THIN:
                out["split2k"] = (Qf, tpf, len(glob2), True)
            else:
                Qr, b, p = LV.timed(LV.tabpfn(), F.X_of(rec), rec[["ddx", "ddy"]].to_numpy(), Xq)
                out["split2k"] = (np.sort((Qf + Qr) / 2, axis=2), tpf + b + p, len(glob2) + len(rec), False)
            for v in BASES:
                Q, sec, nctx, fb = out[v]
                per[v].append((Q, Y, q, float(t), sec, nctx, len(rec), fb))
            el = time.perf_counter() - t_start
            print(f"  ep {ep} t {t:5.1f} (since {q.since.iloc[0]:+5.1f}): {len(q)} queries, recent {len(rec)}"
                  f"{' (thin -> frozen)' if len(rec) < THIN else ''}; s/moment frozen2k {out['frozen2k'][1]:.2f} "
                  f"split2k {out['split2k'][1]:.2f} lgbm {out['lgbm_all_frozen'][1]:.2f}  [{k}/{n_mom}] {el:.0f} s, "
                  f"eta {el / k * (n_mom - k):.0f} s", flush=True)

    frames = []
    for v in BASES:
        Q = np.concatenate([p[0] for p in per[v]]); Y = np.concatenate([p[1] for p in per[v]])
        qr = pd.concat([p[2] for p in per[v]], ignore_index=True)
        rep = lambda j: np.repeat([p[j] for p in per[v]], [len(p[2]) for p in per[v]])  # noqa: E731
        base = pd.DataFrame(dict(episode=qr.episode.to_numpy(), agent_id=qr.agent_id.astype(int).to_numpy(),
                                 t=rep(3), since=qr.since.to_numpy(), t_moment_s=rep(4), n_ctx=rep(5),
                                 n_recent_avail=rep(6), fallback_frozen=rep(7)))
        base["person"] = base.episode.astype(str) + ":" + base.agent_id.astype(str)
        base["t_build_once_s"] = build[v]
        base["err"] = np.linalg.norm(Q[:, :, 2] - Y, axis=1)
        base["cv_err"] = np.linalg.norm(Y, axis=1)
        base["pinball"] = pinball(Q, Y)
        for j, qq in enumerate(QS):
            for ax, an in enumerate(("along", "across")):
                base[f"q{int(round(qq * 100)):02d}_{an}"] = Q[:, ax, j]
        base["y_along"], base["y_across"] = Y[:, 0], Y[:, 1]
        lo, hi = Q[:, :, 0], Q[:, :, 4]
        tt = base.t.to_numpy(float)
        d_used, _ = CP.online_offsets(tt, tt + LAG, base.episode.to_numpy(), lo, hi, Y, D0, alpha=ALPHA_CP, eta=ETA)
        base["score"] = CP.scores(lo, hi, Y)
        for name, (l2, h2, d) in ((v, (lo, hi, 0.0)), (v + "+offset", (*CP.widen(lo, hi, d_used), d_used))):
            dv = base.copy()
            dv.insert(0, "variant", name)
            dv["d_used"] = d
            for m, val in box_metrics(l2, h2, Y).items():
                dv[m] = val
            frames.append(dv)
    res = pd.concat(frames, ignore_index=True)
    assert all(np.isfinite(res.d_used))
    return res


# ---------------------------------------------------------------- statistics
def boot(df, col, B=2000, seed=0):
    return LV.boot(df, col, B=B, seed=seed)


def paired(res, a, b, cols, mask=None, B=2000):
    da = res[res.variant == a].set_index(KEY).sort_index()
    db = res[res.variant == b].set_index(KEY).sort_index()
    assert da.index.equals(db.index)
    dd = da[["person"]].copy()
    for c in cols:
        dd[c] = da[c].to_numpy() - db[c].to_numpy()
    dd["since"] = da.since.to_numpy()
    if mask is not None:
        dd = dd[mask(dd)]
    return {c: boot(dd.reset_index(), c, B) for c in cols}


def matched_width(res, bases, target=0.8, mask=None, B=2000, seed=0):
    """Oracle (post hoc, not causal): constant offset per base variant giving joint coverage = target on the
    slice, then mean width of the widened box. Bootstrap over persons recomputes the offset in each resample."""
    out = {}
    raw = {}
    for v in bases:
        d = res[res.variant == v].set_index(KEY).sort_index()
        if mask is not None:
            d = d[mask(d)]
        raw[v] = d
    persons = raw[bases[0]].person.to_numpy()
    up, inv = np.unique(persons, return_inverse=True)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(inv == i) for i in range(len(up))]

    def width_at(d, idx):
        s = d.score.to_numpy()[idx]
        lo = d[["q05_along", "q05_across"]].to_numpy()[idx]; hi = d[["q95_along", "q95_across"]].to_numpy()[idx]
        off = np.quantile(s, target)
        l2, h2 = CP.widen(lo, hi, off)
        return (h2 - l2).mean(), off

    full = np.arange(len(persons))
    samples = {v: [] for v in bases}
    for _ in range(B):
        pick = rng.integers(0, len(up), len(up))
        idx = np.concatenate([groups[i] for i in pick])
        for v in bases:
            samples[v].append(width_at(raw[v], idx)[0])
    for v in bases:
        w, off = width_at(raw[v], full)
        sv = np.array(samples[v])
        out[v] = (w, off, np.percentile(sv, 2.5), np.percentile(sv, 97.5), sv)
    return out


METRICS = (("err", 1, "{:.3f}"), ("joint90", 100, "{:.1f}"), ("width", 1, "{:.3f}"), ("is90", 1, "{:.3f}"),
           ("pinball", 1, "{:.4f}"))


def summarise(res):
    L = []
    L.append("DECISION RULE (fixed before looking at the numbers), after-switch slice (since >= 0):")
    L.append("  primary = frozen2k+offset, UNLESS split2k+offset beats it on BOTH:")
    L.append("    (a) interval score: paired 95% CI of (split2k+offset - frozen2k+offset) lies entirely below 0, and")
    L.append("    (b) joint coverage: paired 95% CI of the coverage difference does NOT lie entirely below 0 (not worse).")
    L.append("  Stricter reading also reported: (b') coverage point difference >= 0.")
    fz = res[res.variant == "frozen2k"]
    L.append(f"\nSetting B: {fz.person.nunique()} people, {len(fz)} query rows, {fz.groupby(['episode', 't']).ngroups} moments "
             f"({fz.groupby(['episode', 't']).ngroups // max(1, fz.episode.nunique())} per episode, so the offset gets that many "
             f"update rounds per episode); before {int((fz.since < 0).sum())} rows, after {int((fz.since >= 0).sum())}; "
             f"split2k fallback-to-frozen rows {int(res[res.variant == 'split2k'].fallback_frozen.sum())}")
    summ = []
    for pname, mask in periods(res):
        d = res if mask is None else res[mask(res)]
        L.append(f"\n[{pname}] mean [95% person bootstrap]; error and pinball use the raw quantiles; coverage, width, "
                 f"interval score use the (widened) 5-95% box")
        L.append(f"{'variant':<24}{'error m':>22}{'joint 5-95 %':>20}{'width m':>22}{'IS90 m':>22}{'pinball':>26}"
                 f"{'mean d m':>10}{'s/moment':>10}")
        for v in VARIANTS:
            dv = d[d.variant == v]
            r = dict(period=pname, variant=v)
            cells = []
            for col, sc, fmt in METRICS:
                m, lo, hi = boot(dv, col)
                r.update({col: m, col + "_lo": lo, col + "_hi": hi})
                cells.append(f"{fmt.format(m * sc)} [{fmt.format(lo * sc)}, {fmt.format(hi * sc)}]")
            tm = dv.groupby(["episode", "t"]).t_moment_s.first().mean()
            r.update(d_mean=dv.d_used.mean(), t_moment=tm)
            summ.append(r)
            L.append(f"{v:<24}{cells[0]:>22}{cells[1]:>20}{cells[2]:>22}{cells[3]:>22}{cells[4]:>26}"
                     f"{dv.d_used.mean():>10.3f}{tm:>10.3f}")
        L.append(f"  constant-velocity error {d[d.variant == 'frozen2k'].cv_err.mean():.3f} m")
    L.append("\nbuild-once seconds: " + ", ".join(f"{v} {res[res.variant == v].t_build_once_s.iloc[0]:.1f}" for v in BASES)
             + " (split2k reuses frozen2k's cache; its recent model is rebuilt every moment, counted in s/moment)")

    L.append("\nPAIRED DIFFERENCES (A minus B), 95% person bootstrap, 2,000 resamples")
    pairs = [("split2k+offset", "frozen2k+offset"), ("split2k+offset", "lgbm_all_frozen+offset"),
             ("frozen2k+offset", "lgbm_all_frozen+offset"), ("split2k", "frozen2k"), ("frozen2k", "frozen1k")]
    cols = [c for c, _, _ in METRICS]
    pres = {}
    for pname, mask in periods(res):
        L.append(f"[{pname}]")
        for a, b in pairs:
            pr = paired(res, a, b, cols, mask)
            pres[(pname, a, b)] = pr
            parts = []
            for col, sc, fmt in METRICS:
                m, lo, hi = pr[col]
                f2 = fmt.replace("{:", "{:+")
                parts.append(f"{col} {f2.format(m * sc)} [{f2.format(lo * sc)}, {f2.format(hi * sc)}]")
            L.append(f"  {a} - {b}:  " + "  ".join(parts))

    L.append("\nWIDTH AT MATCHED COVERAGE (oracle post hoc constant offset per variant giving 80% joint coverage on the "
             "slice; not causal, a sharpness comparison)")
    mw_all = {}
    for pname, mask in periods(res):
        mw = matched_width(res, ["frozen2k", "split2k", "lgbm_all_frozen", "frozen1k"], 0.8, mask)
        mw_all[pname] = mw
        cells = [f"{v} {mw[v][0]:.3f} [{mw[v][2]:.3f}, {mw[v][3]:.3f}] (d {mw[v][1]:+.3f})" for v in mw]
        L.append(f"[{pname}] " + "; ".join(cells))
        for a, b in (("split2k", "frozen2k"), ("split2k", "lgbm_all_frozen"), ("frozen2k", "lgbm_all_frozen")):
            diff = mw[a][4] - mw[b][4]
            L.append(f"    width {a} - {b} at 80%: {mw[a][0] - mw[b][0]:+.3f} m "
                     f"[{np.percentile(diff, 2.5):+.3f}, {np.percentile(diff, 97.5):+.3f}]")

    if ("after", "split2k+offset", "frozen2k+offset") not in pres:
        L.append("\nDECISION: no after-switch rows in this run")
        return "\n".join(L), pd.DataFrame(summ), mw_all
    pr = pres[("after", "split2k+offset", "frozen2k+offset")]
    is_m, is_lo, is_hi = pr["is90"]; c_m, c_lo, c_hi = pr["joint90"]
    a_ok = is_hi < 0
    b_ok = c_hi >= 0
    b_strict = c_m >= 0
    L.append("\nDECISION (after switch, split2k+offset - frozen2k+offset):")
    L.append(f"  (a) IS90 {is_m:+.3f} [{is_lo:+.3f}, {is_hi:+.3f}] -> {'PASS' if a_ok else 'FAIL'} (CI entirely below 0)")
    L.append(f"  (b) joint coverage {c_m * 100:+.1f} [{c_lo * 100:+.1f}, {c_hi * 100:+.1f}] pts -> "
             f"{'PASS' if b_ok else 'FAIL'} (CI not entirely below 0); strict (b') point >= 0: {'PASS' if b_strict else 'FAIL'}")
    prim = "split2k+offset" if (a_ok and b_ok) else "frozen2k+offset"
    prim_strict = "split2k+offset" if (a_ok and b_strict) else "frozen2k+offset"
    L.append(f"  PRIMARY: {prim}" + ("" if prim == prim_strict else f"  (strict reading would give {prim_strict})"))
    return "\n".join(L), pd.DataFrame(summ), mw_all


def figure(summ, path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), facecolor="white")
    for ax, pname in zip(axes, ("overall", "before", "after")):
        s = summ[summ.period == pname].set_index("variant")
        if s.empty:
            continue
        for b in BASES:
            pts = []
            for v, mk, fill in ((b, "o", "none"), (b + "+offset", "s", COLORS[b])):
                r = s.loc[v]
                x, y = r.width, r.joint90 * 100
                ax.errorbar(x, y, xerr=[[x - r.width_lo], [r.width_hi - x]],
                            yerr=[[y - r.joint90_lo * 100], [r.joint90_hi * 100 - y]],
                            fmt=mk, color=COLORS[b], mfc=fill, ms=8, capsize=3, lw=1.2,
                            label=f"{LABELS[b]}{' + online offset' if v.endswith('offset') else ''}")
                pts.append((x, y))
            ax.annotate("", xy=pts[1], xytext=pts[0], arrowprops=dict(arrowstyle="->", color=COLORS[b], lw=1, alpha=0.6))
            ax.text(pts[1][0], pts[1][1] + 1.2, f"IS {s.loc[b + '+offset'].is90:.2f}", fontsize=7, color=COLORS[b], ha="center")
        ax.axhspan(80, 90, color="#2f9e44", alpha=0.07)
        ax.axhline(80, color="#2f9e44", ls=":", lw=1)
        ax.text(ax.get_xlim()[0], 80.5, " target 80% (offset alpha 0.2); nominal 5-95 joint 80-90%", fontsize=7, color="#2f9e44")
        ttl = {"overall": "All moments", "before": "Before the switch (calm)", "after": "After the switch (erratic, since >= 0)"}
        ax.set_title(ttl[pname], fontsize=10, loc="left")
        ax.set_xlabel("Mean width of the 5-95% box (m, along/across averaged)")
        ax.set_ylabel("Joint coverage of the 5-95% box (%)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Coverage vs width at +2 s: open = raw box, filled = with online conformal offset "
                 "(bars: 95% person bootstrap; label: 90% interval score of the offset box)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="1 episode, 2 moments")
    ap.add_argument("--out", default=str(ROOT), help="root for results/ and media/")
    ap.add_argument("--from-csv", default=None, help="skip running: summarise and plot this CSV")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True); (out / "media").mkdir(parents=True, exist_ok=True)
    csv, png = out / "results/live_variants_cp.csv", out / "media/fig_live_variants_cp.png"
    T0 = time.perf_counter()
    if a.from_csv:
        res = pd.read_csv(a.from_csv)
    else:
        n_eps, max_m = (1, 2) if a.smoke else (a.episodes, None)
        src, test, obs, moments = LV.setting_b(n_eps)
        res = run(src, test, obs, moments, max_m)
        res.to_csv(csv, index=False)
        print(f"wrote {csv} ({len(res)} rows) at {time.perf_counter() - T0:.0f} s", flush=True)
    text, summ, _ = summarise(res)
    print(text)
    figure(summ, png)
    print(f"wrote {png}; total {time.perf_counter() - T0:.0f} s")


if __name__ == "__main__":
    main()
