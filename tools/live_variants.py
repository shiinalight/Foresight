"""Offline test: how should a robot's TabPFN read the current scene?

    python -B tools/live_variants.py                 # both settings, ~30 min on 3 CPU threads
    python -B tools/live_variants.py --smoke --out <dir>   # 1 episode, 2 moments per setting
    python -B tools/live_variants.py --settings A --out <dirA>   # run settings in parallel, then
    python -B tools/live_variants.py --from-csv <dirA>/results/live_variants.csv <dirB>/results/live_variants.csv

Two settings, the same query rows for every variant at each moment:
  A  bank = Hotel rows, test = the first 6 ETH episodes (h = 2.0 s), moments every 4 s
     (as run_adapt.py --train-data data/hotel --data data/eth). Real data, noise 0.
  B  bank = data/crowd_cross training episodes without any erratic episode; test = 6 new cross-mode
     episodes (seeds 7000+) that switch calm -> erratic at 20 s; moments every 4 s from ~10 s.
Rows from the current scene are labelled from observed (filtered) positions and usable once t_known <= t;
queries are scored against true positions.

Variants:
  frozen       TabPFN on 1,000 random bank rows, cache built once per setting
  full_mix     context.select(mix=(0.6, 0.3, 0.1), budget=1000), rebuilt every moment (tabpfn_live)
  recent_only  this scene's completed rows only (250 most recent); < 20 rows: frozen (flagged thin)
  split        average of frozen and recent_only quantiles; frozen alone while recent is thin
  mix_tags     full_mix rows plus columns in_scene (1 recent/queries, 0 bank) and age_s (0 queries, 300 bank)
  lgbm         LightGBM (m50, 100 trees) on the full_mix rows (it trains inside predict: all time is build)

Writes results/live_variants.csv and media/fig_live_variants.png. Reads row caches, never writes them.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from foresight.core import context as C  # noqa: E402
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from foresight.core import schema  # noqa: E402
from foresight.domains.crowd import CrowdConfig, simulate  # noqa: E402

torch.set_num_threads(3)
QS = (0.05, 0.1, 0.5, 0.9, 0.95)
H = 2.0
BUDGET, REC_CAP, THIN, BANK_AGE = 1000, 250, 20, 300.0
VARIANTS = ["frozen", "full_mix", "recent_only", "split", "mix_tags", "lgbm"]
LABELS = {"frozen": "Frozen\n(1,000 bank rows)", "full_mix": "Full mix\n(rebuilt)", "recent_only": "This scene\nonly",
          "split": "Split: frozen\n+ this scene", "mix_tags": "Mix +\nscene tags", "lgbm": "LightGBM\n(mix rows)"}
COLORS = {"frozen": "#8b949e", "full_mix": "#e8590c", "recent_only": "#7a3fb5", "split": "#2a6fdb",
          "mix_tags": "#2f9e44", "lgbm": "#1f2328"}


def tabpfn():
    return M.make("tabpfn", fast=True, n_estimators=1, fit_mode="fit_with_cache")


def cached_rows(data_dir, noise, horizons):
    """Truth-labelled f4 rows from the existing cache (read only; fails rather than building a cache)."""
    meta = F.load_kalman(data_dir, {})
    kq, kr = F.kalman_params(meta, noise)
    hz = "_".join(f"{h:g}" for h in horizons)
    p = ROOT / data_dir / f"rows_{F.FEATURES_VERSION}_noise{noise}_h{hz}_kq{kq:g}_kr{kr:g}.parquet"
    assert p.exists(), f"missing row cache {p}"
    df = pd.read_parquet(p)
    return df[np.isclose(df.h, H)].reset_index(drop=True)


def setting_a(n_eps):
    src = cached_rows("data/hotel", 0.0, (1.2, 2.0, 4.8))
    test = cached_rows("data/eth", 0.0, (1.2, 2.0, 4.8))
    eps = sorted(test.episode.unique())[:n_eps]
    test = test[test.episode.isin(eps)].copy()
    test["since"] = test["t"]                                    # seconds in the scene
    tracks, obst, meta = schema.load(ROOT / "data/eth")
    F.load_kalman(ROOT / "data/eth", meta)
    obs = F.build_rows(tracks, obst, meta, horizons=(H,), noise=0.0, seed=0, episodes=eps, label_from="observed")
    # sanity: the in-memory rows have the same features as the cached rows
    key = ["episode", "t", "agent_id"]
    mm = test.merge(obs, on=key, suffixes=("", "_o"))
    fc = ["h_f"] + F.feature_names()[1:]
    diff = np.nanmax(np.abs(mm[fc].to_numpy(float) - mm[[c + "_o" for c in fc]].to_numpy(float)))
    print(f"A: bank {len(src)} Hotel rows; {len(test)} ETH query rows in {len(eps)} episodes; "
          f"{len(obs)} observed rows; feature mismatch vs cache {diff:.2e} over {len(mm)} rows", flush=True)
    moments = {}
    for ep in eps:
        all_t = sorted(test[test.episode == ep].t.unique())
        step = max(1, int(round(4.0 / np.median(np.diff(all_t)))))
        moments[ep] = all_t[::step]
    return src, test, obs, moments


def setting_b(n_eps):
    cc = cached_rows("data/crowd_cross", 0.1, (1.0, 2.0))
    tr, _, _ = F.split_by_episode(cc)
    src = cc[tr]
    import json
    em = json.loads((ROOT / "data/crowd_cross/meta.json").read_text())["episodes"]
    bad = set(src[src.regime == "erratic"].episode.unique())
    bad |= {int(e) for e, m in em.items() if "erratic" in (m.get("regime_before"), m.get("regime_after"))}
    src = src[~src.episode.isin(bad)]
    cfg = CrowdConfig(n_people=16, duration=40.0, dt=0.2, half_size=10.0, robot_repulsion=2.0, goal_mode="cross",
                      regime="calm", switch_t=20.0, switch_to="erratic")
    tracks, obstl, eps = [], [], {}
    for i in range(n_eps):
        ep = 7000 + i
        t_, o_, m_ = simulate(7000 + i, cfg, episode=ep)
        tracks.append(t_); obstl.append(o_); eps[str(ep)] = m_
    tracks = pd.concat(tracks, ignore_index=True).sort_values(["episode", "t", "agent_id"]).reset_index(drop=True)
    obst = pd.concat(obstl, ignore_index=True)
    meta = F.load_kalman(ROOT / "data/crowd_cross", dict(dt=0.2, bounds=[-10.0, -10.0, 10.0, 10.0], episodes=eps))
    test = F.build_rows(tracks, obst, meta, horizons=(H,), noise=0.1, seed=0)
    obs = F.build_rows(tracks, obst, meta, horizons=(H,), noise=0.1, seed=0, label_from="observed")
    test["since"] = test["since_switch"]
    print(f"B: bank {len(src)} rows from {src.episode.nunique()} training episodes without erratic "
          f"(dropped {len(bad & set(cc[tr].episode.unique()))}); kalman q {meta['kalman_q']} r {meta['kalman_r']}; "
          f"{len(test)} query rows in {n_eps} new episodes", flush=True)
    moments = {}
    for ep in sorted(test.episode.unique()):
        all_t = np.array(sorted(test[test.episode == ep].t.unique()))
        start = int(np.flatnonzero(all_t >= 9.75)[0])
        step = max(1, int(round(4.0 / np.median(np.diff(all_t)))))
        moments[ep] = list(all_t[start::step])
    return src, test, obs, moments


def row_metrics(Q, Y):
    err = np.linalg.norm(Q[:, :, 2] - Y, axis=1)
    j90 = ((Y >= Q[:, :, 0]) & (Y <= Q[:, :, 4])).all(axis=1).astype(float)
    j80 = ((Y >= Q[:, :, 1]) & (Y <= Q[:, :, 3])).all(axis=1).astype(float)
    w80 = Q[:, :, 3] - Q[:, :, 1]
    w90 = Q[:, :, 4] - Q[:, :, 0]
    a = np.asarray(QS)[None, None, :]
    d = Y[:, :, None] - Q
    pin = np.maximum(a * d, (a - 1) * d).mean(axis=(1, 2))
    return dict(err=err, joint90=j90, joint80=j80, w80_along=w80[:, 0], w80_across=w80[:, 1],
                w90_along=w90[:, 0], w90_across=w90[:, 1], width=(w80.mean(1) + w90.mean(1)) / 2, pinball=pin)


def timed(model, Xc, Yc, Xq):
    t0 = time.perf_counter(); model.fit(Xc, Yc)
    t1 = time.perf_counter(); Q = model.predict_quantiles(Xq, QS)
    return Q, t1 - t0, time.perf_counter() - t1


def run_setting(name, src, test, obs, moments, max_moments=None):
    bank = C.Bank(src)
    glob = src.sample(min(BUDGET, len(src)), random_state=0)
    frozen = tabpfn()
    t0 = time.perf_counter(); frozen.fit(F.X_of(glob), glob[["ddx", "ddy"]].to_numpy())
    t_frozen_build = time.perf_counter() - t0
    print(f"{name}: frozen cache built once in {t_frozen_build:.1f} s", flush=True)
    rows, t_start = [], time.perf_counter()
    n_mom = sum(len(v[:max_moments]) for v in moments.values()); k = 0
    for ep, times in moments.items():
        live = obs[obs.episode == ep]
        q_ep = test[test.episode == ep]
        for t in times[:max_moments]:
            k += 1
            q = q_ep[np.isclose(q_ep.t, t)]
            Xq, Y = F.X_of(q), q[["ddx", "ddy"]].to_numpy()
            out = {}
            # 1 frozen
            t1 = time.perf_counter(); Qf = frozen.predict_quantiles(Xq, QS); tpf = time.perf_counter() - t1
            out["frozen"] = (Qf, 0.0, tpf, False)
            # 2 full_mix
            ctx = C.select(q, bank, live=live, budget=BUDGET, mix=(0.6, 0.3, 0.1))
            Xc, Yc = F.X_of(ctx), ctx[["ddx", "ddy"]].to_numpy()
            n_rec_mix = int((ctx.source == "recent").sum())
            Q, b, p = timed(tabpfn(), Xc, Yc, Xq); out["full_mix"] = (Q, b, p, False)
            # 3 recent_only, 4 split
            rec = live[live.t_known <= t].sort_values("t_known").tail(REC_CAP)
            thin = len(rec) < THIN
            if thin:
                out["recent_only"] = (Qf, 0.0, tpf, True)
                out["split"] = (Qf, 0.0, tpf, True)
            else:
                Qr, b, p = timed(tabpfn(), F.X_of(rec), rec[["ddx", "ddy"]].to_numpy(), Xq)
                out["recent_only"] = (Qr, b, p, False)
                Qs = np.sort((Qf + Qr) / 2, axis=2)
                out["split"] = (Qs, b, p + tpf, False)
            # 5 mix_tags
            is_rec = (ctx.source == "recent").to_numpy()
            tag_c = np.column_stack([is_rec.astype(float), np.where(is_rec, t - ctx.t.to_numpy(), BANK_AGE)])
            tag_q = np.column_stack([np.ones(len(q)), np.zeros(len(q))])
            Q, b, p = timed(tabpfn(), np.hstack([Xc, tag_c]), Yc, np.hstack([Xq, tag_q])); out["mix_tags"] = (Q, b, p, False)
            # 6 lgbm on the full_mix rows (trains lazily inside predict_quantiles)
            Q, b, p = timed(M.make("lgbm:m50:n100", n_jobs=2), Xc, Yc, Xq); out["lgbm"] = (Q, b + p, 0.0, False)
            cv_err = np.linalg.norm(Y, axis=1)
            for v in VARIANTS:
                Qv, b, p, fb = out[v]
                met = row_metrics(Qv, Y)
                n_ctx = {"frozen": len(glob), "recent_only": len(rec) if not fb else len(glob),
                         "split": len(glob) + (0 if fb else len(rec))}.get(v, len(ctx))
                for i in range(len(q)):
                    rows.append(dict(setting=name, variant=v, episode=ep, agent_id=int(q.agent_id.iloc[i]),
                                     person=f"{ep}:{int(q.agent_id.iloc[i])}", t=float(t), since=float(q.since.iloc[i]),
                                     n_queries=len(q), n_ctx=n_ctx, n_recent_avail=len(rec), n_recent_in_mix=n_rec_mix,
                                     thin=bool(thin), fallback_frozen=bool(fb),
                                     t_build=b, t_pred=p, t_total=b + p, t_frozen_build_once=t_frozen_build,
                                     cv_err=cv_err[i], **{m: float(val[i]) for m, val in met.items()}))
            el = time.perf_counter() - t_start
            print(f"  {name} ep {ep} t {t:5.1f}: {len(q)} queries, recent {len(rec)}{' (thin)' if thin else ''}, "
                  f"[{k}/{n_mom}] {el:.0f} s, eta {el / k * (n_mom - k):.0f} s", flush=True)
    return pd.DataFrame(rows)


def boot(df, col, B=2000, seed=0):
    """Mean of `col` and 95% interval, resampling persons with replacement."""
    g = df.groupby("person")[col].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (B, len(g)))
    s, c = g["sum"].to_numpy()[idx].sum(1), g["count"].to_numpy()[idx].sum(1)
    st = s / c
    return df[col].mean(), np.percentile(st, 2.5), np.percentile(st, 97.5)


def summarise(res):
    lines, summ = [], []
    for sname, d in res.groupby("setting", sort=False):
        lines.append(f"\nSetting {sname}: {d.person.nunique()} people, {d[d.variant == 'frozen'].shape[0]} query rows, "
                     f"{d.groupby(['episode', 't']).ngroups} moments; thin moments "
                     f"{d[d.variant == 'frozen'].groupby(['episode', 't']).thin.first().sum()}")
        lines.append(f"{'variant':<12} {'error m':>22} {'joint 5-95 %':>22} {'width m':>22} {'pinball':>26} {'s/moment med (mean)':>20}")
        base = d[d.variant == "full_mix"].reset_index(drop=True)
        for v in VARIANTS:
            dv = d[d.variant == v].reset_index(drop=True)
            r = {"setting": sname, "variant": v}
            cells = []
            for col, scale, fmt in (("err", 1, "{:.3f}"), ("joint90", 100, "{:.1f}"), ("width", 1, "{:.3f}"), ("pinball", 1, "{:.4f}")):
                m, lo, hi = boot(dv, col)
                r.update({col: m, col + "_lo": lo, col + "_hi": hi})
                cells.append(f"{fmt.format(m * scale)} [{fmt.format(lo * scale)}, {fmt.format(hi * scale)}]")
            tm = dv.groupby(["episode", "t"]).t_total.first()
            r.update(t_med=tm.median(), t_mean=tm.mean())
            lines.append(f"{v:<12} {cells[0]:>22} {cells[1]:>22} {cells[2]:>22} {cells[3]:>26} {tm.median():>9.2f} ({tm.mean():.2f})")
            summ.append(r)
        lines.append(f"CV (zero residual) error {base.cv_err.mean():.3f} m; frozen cache built once in "
                     f"{d.t_frozen_build_once.iloc[0]:.1f} s")
        lines.append("paired differences vs full_mix (variant minus full_mix), 95% person bootstrap:")
        for v in ("split", "mix_tags", "recent_only", "frozen", "lgbm"):
            dv = d[d.variant == v].reset_index(drop=True)
            dd = dv[["person"]].copy()
            parts = []
            for col, scale, fmt in (("err", 1, "{:+.3f}"), ("joint90", 100, "{:+.1f}"), ("pinball", 1, "{:+.4f}")):
                dd[col] = dv[col].to_numpy() - base[col].to_numpy()
                m, lo, hi = boot(dd, col)
                parts.append(f"{col} {fmt.format(m * scale)} [{fmt.format(lo * scale)}, {fmt.format(hi * scale)}]")
            lines.append(f"  {v:<12} " + "  ".join(parts))
    return "\n".join(lines), pd.DataFrame(summ)


BINS = {"A": [(0, 10, "0-10 s"), (10, 20, "10-20 s"), (20, 1e9, "20+ s")],
        "B": [(-1e9, 0, "before"), (0, 10, "0-10 s"), (10, 20, "10-20 s"), (20, 1e9, "20+ s")]}


def figure(res, summ, path):
    sets = list(res.setting.unique())
    fig, axes = plt.subplots(len(sets), 3, figsize=(16, 4.6 * len(sets)), facecolor="white", squeeze=False)
    xl = {"A": "Seconds since the robot entered the scene (ETH, bank: Hotel)",
          "B": "Seconds since the crowd switched calm -> erratic"}
    for row, s in zip(axes, sets):
        sm = summ[summ.setting == s].set_index("variant").loc[VARIANTS]
        x = np.arange(len(VARIANTS))
        cols = [COLORS[v] for v in VARIANTS]
        ticks = [f"{LABELS[v]}\n{sm.loc[v, 't_med']:.2f} s" for v in VARIANTS]
        for ax, col, scale, ylab in ((row[0], "err", 1, "Mean error at +2 s (m)"),
                                     (row[1], "joint90", 100, "5-95% box holds the truth (%)")):
            m = sm[col].to_numpy() * scale
            ax.bar(x, m, color=cols, alpha=0.85)
            ax.errorbar(x, m, yerr=[m - sm[col + "_lo"] * scale, sm[col + "_hi"] * scale - m], fmt="none", ecolor="#444", capsize=4)
            ax.set_xticks(x); ax.set_xticklabels(ticks, fontsize=7.5)
            ax.set_ylabel(ylab); ax.spines[["top", "right"]].set_visible(False)
            for xi, mi in zip(x, m):
                ax.text(xi, mi * 0.5, f"{mi:.3f}" if scale == 1 else f"{mi:.0f}", ha="center", color="white", fontsize=8)
        cv = res[(res.setting == s) & (res.variant == "frozen")].cv_err.mean()
        row[0].axhline(cv, color="#888", ls="--", lw=1)
        row[0].text(-0.4, cv * 1.01, f"constant velocity {cv:.3f} m", fontsize=7, color="#666", va="bottom")
        row[1].axhline(81, color="#888", ls=":", lw=1)
        row[1].text(-0.4, 82, "~81% if the axes are independent", fontsize=7, color="#666")
        row[0].set_title(f"Setting {s}: error (bars: 95% person-bootstrap; tick: median s per moment)", fontsize=9, loc="left")
        row[1].set_title(f"Setting {s}: joint coverage of the robot's 5-95% box", fontsize=9, loc="left")
        d = res[res.setting == s].copy()
        b = BINS[s]
        d["bin"] = pd.cut(d.since, [e[0] for e in b] + [b[-1][1]], labels=[e[2] for e in b], right=False)
        tab = d.groupby(["variant", "bin"], observed=False).err.mean().unstack("bin")
        cnt = d[d.variant == "frozen"].groupby("bin", observed=False).size()
        for v in VARIANTS:
            row[2].plot(range(len(b)), tab.loc[v].values, marker="o", lw=2, color=COLORS[v], label=LABELS[v].replace("\n", " "))
        cvb = d[d.variant == "frozen"].groupby("bin", observed=False).cv_err.mean()
        row[2].plot(range(len(b)), cvb.values, ls="--", lw=1.2, color="#888", label="Constant velocity")
        row[2].set_xticks(range(len(b)))
        row[2].set_xticklabels([f"{e[2]}\n(n={cnt.get(e[2], 0)})" for e in b], fontsize=8)
        row[2].set_xlabel(xl[s], fontsize=8); row[2].set_ylabel("Mean error at +2 s (m)")
        row[2].spines[["top", "right"]].set_visible(False)
        row[2].set_title(f"Setting {s}: error over time in the scene", fontsize=9, loc="left")
    axes[0][2].legend(frameon=False, fontsize=7)
    fig.suptitle("How the robot's TabPFN reads the current scene: six variants, same queries", fontsize=12,
                 weight="bold", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="1 episode, 2 moments per setting")
    ap.add_argument("--settings", nargs="*", default=["A", "B"])
    ap.add_argument("--out", default=str(ROOT), help="root for results/ and media/")
    ap.add_argument("--from-csv", nargs="*", default=None,
                    help="skip running: merge these per-setting CSVs (e.g. A and B run in parallel), then summarise and plot")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True); (out / "media").mkdir(parents=True, exist_ok=True)
    csv, png = out / "results/live_variants.csv", out / "media/fig_live_variants.png"
    n_eps, max_m = (1, 2) if a.smoke else (a.episodes, None)
    T0 = time.perf_counter()
    parts = [pd.read_csv(p) for p in a.from_csv] if a.from_csv else []
    for s in ([] if a.from_csv else a.settings):
        src, test, obs, moments = (setting_a if s == "A" else setting_b)(n_eps)
        parts.append(run_setting(s, src, test, obs, moments, max_m))
        pd.concat(parts).to_csv(csv, index=False)
        print(f"setting {s} done at {time.perf_counter() - T0:.0f} s; wrote {csv}", flush=True)
    res = pd.concat(parts, ignore_index=True)
    if a.from_csv:
        res.to_csv(csv, index=False)
    text, summ = summarise(res)
    print(text)
    figure(res, summ, png)
    print(f"wrote {csv} and {png} in {time.perf_counter() - T0:.0f} s")


if __name__ == "__main__":
    main()
