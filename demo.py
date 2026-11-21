"""R1 demo video: "Where will everyone be in 2 seconds?"

Panels, left to right: constant velocity with a fixed box | LightGBM | TabPFN (hosted API).
One held-out episode, each person's predicted 80% range at +2 s drawn as they move, the true
position 2 s later as a black dot, and per panel: coverage now and so far, mean error, mean box area.
Also, over ALL held-out rows: a table per model, horizon and segment (no switch / before / after the
behaviour switch), and the "seconds to adapt" plot (coverage and error against time since the switch).

    export TABPFN_TOKEN=<key from ux.priorlabs.ai/account>
    python demo.py --data data/crowd                            # three panels, table, plot
    python demo.py --data data/crowd --models cv_box lgbm       # without TabPFN
    python demo.py --data data/crowd --no-video                 # table and plot only

Writes results/demo_ep<N>.mp4 (1280x720, for the video), results/demo_ep<N>.gif (640x360, README),
results/summary_r1.csv and results/adapt_r1.png.

Ranges are boxes in the person's own frame (x = heading), calibrated on VALIDATION episodes only:
- LightGBM and TabPFN: q05..q50..q95 along and across the heading, then the half-widths around the
  median are scaled (one factor per model and horizon) so 80% of validation positions fall inside.
- Constant velocity: a fixed box centred on its prediction, one size per horizon, sized the same way.
Test episodes are never used for sizing. "Inside" means inside the exact box drawn, so the boxes
and the numbers agree.
Split by episode: train (context for the models) / validation (box size) / test (everything shown).
Predictions are cached in results/cache/, so re-rendering costs no API quota.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon

from foresight.core import features as F
from foresight.core import models as M
from foresight.core import schema
from run_offline import rows_for

QS = (0.05, 0.5, 0.95)
VAL_FRAC = 0.2
NAMES = {"tabpfn_client": "TabPFN", "tabpfn": "TabPFN (local)", "lgbm": "LightGBM",
         "cv_box": "Constant velocity + fixed box", "cv": "Constant velocity", "cv_band": "Constant velocity + band"}
C_PERSON, C_ROBOT, C_OBST, C_RANGE, C_MISS = "#3b75af", "#d62728", "0.65", "#ff7f0e", "#d62728"
TRAIL_S = 2.0


def predict_sets(model, fit_rows, sets, tag):
    """Fit once on fit_rows, predict raw residual quantiles [n, 2, 3] for each row set.
    sets: {cache_name: rows}. Each set is cached on its own; only missing sets are predicted."""
    cache = {k: Path("results/cache") / f"{tag}_{k}_{model}.npy" for k in sets}
    out = {k: np.load(p) for k, p in cache.items() if p.exists()}
    todo = [k for k in sets if k not in out]
    if not todo:
        return out
    X_all = np.concatenate([F.X_of(sets[k]) for k in todo])
    if model == "tabpfn_client":
        estimate(F.X_of(fit_rows), X_all)
    m = M.make(model).fit(F.X_of(fit_rows), fit_rows[["ddx", "ddy"]].to_numpy())
    Q_all = np.sort(m.predict_quantiles(X_all, QS), axis=2)
    i = 0
    for k in todo:
        out[k] = Q_all[i:i + len(sets[k])]; i += len(sets[k])
        cache[k].parent.mkdir(parents=True, exist_ok=True)
        np.save(cache[k], out[k])
    return out


def estimate(X_fit, X_pred):
    """Print the API's own cost estimate before spending quota (two calls: one per axis)."""
    try:
        from tabpfn_client import estimate_cost
        r = estimate_cost(X_fit, X_pred)
        print(f"TabPFN API estimate per axis: {r.estimated_cost} (x2 axes); "
              f"{len(X_fit)} context rows, {len(X_pred)} rows to predict")
    except Exception as e:                       # the estimate is informational only
        print(f"(no TabPFN cost estimate: {type(e).__name__})")


def boxes(q, Q):
    """World-frame corners [n, 4, 2] of each box, median world positions, inside flags, metres outside."""
    c, s = np.cos(q.theta.to_numpy()), np.sin(q.theta.to_numpy())
    base_x = q.px.to_numpy() + q.vx.to_numpy() * q.h.to_numpy()
    base_y = q.py.to_numpy() + q.vy.to_numpy() * q.h.to_numpy()
    lo, hi = Q[:, :, 0], Q[:, :, 2]
    corners = []
    for ax_, ay_ in ((lo[:, 0], lo[:, 1]), (hi[:, 0], lo[:, 1]), (hi[:, 0], hi[:, 1]), (lo[:, 0], hi[:, 1])):
        corners.append(np.stack([base_x + c * ax_ - s * ay_, base_y + s * ax_ + c * ay_], axis=1))
    corners = np.stack(corners, axis=1)
    mx, my = F.to_world(q, Q[:, 0, 1], Q[:, 1, 1])
    Y = q[["ddx", "ddy"]].to_numpy()
    inside = ((Y >= lo) & (Y <= hi)).all(axis=1)
    out_d = np.hypot(np.maximum(0, np.maximum(lo[:, 0] - Y[:, 0], Y[:, 0] - hi[:, 0])),
                     np.maximum(0, np.maximum(lo[:, 1] - Y[:, 1], Y[:, 1] - hi[:, 1])))
    return corners, np.asarray(mx), np.asarray(my), inside, out_d


def box_area(Q):
    return (Q[:, 0, 2] - Q[:, 0, 0]) * (Q[:, 1, 2] - Q[:, 1, 0])


def needed_scale(Q, Y):
    """Per row: the smallest factor on the median-to-quantile half-widths that puts the truth inside
    the box on both axes. A box scaled by s holds every row whose needed scale is <= s."""
    med = Q[:, :, 1]
    up, down = np.maximum(Q[:, :, 2] - med, 1e-9), np.maximum(med - Q[:, :, 0], 1e-9)
    need = np.where(Y >= med, (Y - med) / up, (med - Y) / down)
    return need.max(axis=1)


def calibrate(Q_val, val_rows, target=0.8):
    """Scale per horizon so `target` of the VALIDATION rows fall inside the box: {h: s}."""
    need, h = needed_scale(Q_val, val_rows[["ddx", "ddy"]].to_numpy()), val_rows.h.to_numpy()
    return {float(v): float(np.quantile(need[np.isclose(h, v)], target)) for v in np.unique(h)}


def apply_scale(Q, rows, scales):
    """Stretch (s > 1) or shrink (s < 1) each box about its median, per horizon."""
    Q = Q.copy(); med = Q[:, :, 1:2]
    s = np.ones(len(Q))
    for h, v in scales.items():
        s[np.isclose(rows.h.to_numpy(), h)] = v
    return med + (Q - med) * s[:, None, None]


def evaluate(rows, Q):
    """Per row: inside the box, box area, error of the median, as a DataFrame aligned with rows."""
    _, mx, my, inside, _ = boxes(rows, Q)
    return pd.DataFrame(dict(inside=inside, area=box_area(Q),
                             err=np.hypot(mx - rows.true_x.to_numpy(), my - rows.true_y.to_numpy())))


def segment(rows):
    s = rows.since_switch.to_numpy()
    return np.where(np.isnan(s), "no switch", np.where(s < 0, "before switch", "after switch"))


def summary(T, results, n_eps, n_switch):
    """Table per model, horizon and segment over all held-out rows."""
    out = []
    seg = segment(T)
    for m, R in results.items():
        for h in sorted(T.h.unique()):
            kh = np.isclose(T.h.to_numpy(), h)
            for name in ("all", "no switch", "before switch", "after switch"):
                k = kh if name == "all" else kh & (seg == name)
                if k.any():
                    out.append(dict(model=NAMES.get(m, m), horizon=f"+{h:g} s", segment=name, rows=int(k.sum()),
                                    coverage=R.inside[k].mean(), box_area_m2=R.area[k].mean(), error_m=R.err[k].mean()))
    t = pd.DataFrame(out)
    print(f"\nAll held-out rows: {n_eps} episodes ({n_switch} with a behaviour switch). "
          "Boxes calibrated on validation episodes to 80%.")
    fmt = {"coverage": "{:.0%}".format, "box_area_m2": "{:.2f}".format, "error_m": "{:.2f}".format}
    for h, g in t.groupby("horizon", sort=False):
        print(f"\n{h}")
        print(g.drop(columns="horizon").to_string(index=False, formatters=fmt))
    Path("results").mkdir(exist_ok=True)
    t.to_csv("results/summary_r1.csv", index=False)
    print("\nwrote results/summary_r1.csv")


SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3, fixed order (validated)
MARKERS = ["o", "s", "^"]


def adapt_plot(T, results, h, out="results/adapt_r1.png", bin_s=2.0, lo=-10.0, hi=16.0, min_rows=150):
    """Coverage and error against seconds since the behaviour switch, one line per model.
    Bins with fewer than min_rows rows (the tail, where few episodes reach) are left out."""
    k = np.isclose(T.h.to_numpy(), h) & T.since_switch.notna().to_numpy()
    since = T.since_switch.to_numpy()[k]
    edges = np.arange(lo, hi + bin_s, bin_s)
    b = np.digitize(since, edges) - 1
    ok = (b >= 0) & (b < len(edges) - 1)
    counts = np.bincount(b[ok], minlength=len(edges) - 1)
    ok &= np.isin(b, np.flatnonzero(counts >= min_rows))
    mid = edges[:-1] + bin_s / 2
    n_eps = T[k].episode.nunique()
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.4), facecolor="white")
    ymax_err, ends = 0.0, []
    for i, (m, R) in enumerate(results.items()):
        g = pd.DataFrame(dict(b=b[ok], inside=R.inside.to_numpy()[k][ok], err=R.err.to_numpy()[k][ok])).groupby("b")
        cov, err = g.inside.mean(), g.err.mean()
        x = mid[cov.index]
        ymax_err = max(ymax_err, err.max())
        kw = dict(color=SERIES[i % 3], marker=MARKERS[i % 3], ms=8, lw=2, mec="white", mew=1.5, label=NAMES.get(m, m))
        a1.plot(x, cov.to_numpy() * 100, **kw); a2.plot(x, err.to_numpy(), **kw)
        ends.append((err.to_numpy()[-1], x[-1], NAMES.get(m, m).replace("Constant velocity + fixed box", "Const. velocity")))
    a1.axhline(80, color="0.55", lw=1, ls="--")
    a1.text(hi + 0.3, 80, "target 80%", va="center", fontsize=10.5, color="0.4")
    a1.set_ylim(0, 105); a2.set_ylim(0, ymax_err * 1.18)
    gap, last = ymax_err * 0.07, -np.inf                 # direct labels, nudged apart so none overlap
    for y, x_end, label in sorted(ends):
        y_lab = max(y, last + gap); last = y_lab
        a2.annotate(label, (x_end, y), xytext=(x_end + 0.6, y_lab), va="center", fontsize=10.5, color="0.2")
    for ax, title, yl in ((a1, f"Coverage of the +{h:g} s range", "true positions inside the box (%)"),
                          (a2, f"Error at +{h:g} s (lower is better)", "mean error (m)")):
        ax.axvline(0, color="0.3", lw=1)
        ax.text(0.3, 0.03, "behaviour switch", transform=ax.get_xaxis_transform(), fontsize=10.5, color="0.3")
        ax.set_xlabel("seconds since the crowd switched behaviour"); ax.set_ylabel(yl)
        ax.set_title(title, loc="left", fontsize=14, weight="bold")
        ax.grid(axis="y", color="0.9"); ax.set_xlim(lo, hi + 4.5)
        ax.tick_params(colors="0.35")
    fig.legend(*a1.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("How fast does each model adapt when the crowd changes behaviour?", fontsize=16, weight="bold", y=0.985)
    fig.text(0.5, 0.915, f"{n_eps} held-out episodes with a switch · {bin_s:g} s bins · boxes calibrated on validation "
                         "episodes · context from training episodes only (no recent rows yet)",
             ha="center", fontsize=10.5, color="0.35")
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def pick_episode(test_eps, meta):
    """First test episode with a regime switch (shows adaptation), else the first test episode."""
    for ep in test_eps:
        if meta["episodes"][str(ep)].get("regime_switch_t") is not None:
            return int(ep)
    return int(test_eps[0])


def main():
    ap = argparse.ArgumentParser(description="R1 demo video")
    ap.add_argument("--data", required=True)
    ap.add_argument("--episode", type=int, default=None, help="default: first test episode with a regime switch")
    ap.add_argument("--models", nargs="+", default=["cv_box", "lgbm", "tabpfn_client"], help="panels, left to right")
    ap.add_argument("--context", type=int, default=1500, help="context rows from training episodes")
    ap.add_argument("--h", type=float, default=2.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", nargs="+", default=None, help="default: results/demo_ep<N>.mp4 and .gif")
    a = ap.parse_args()

    tracks, obst, meta = schema.load(a.data)
    dt = float(meta["dt"])
    df = rows_for(a.data, 0.1)
    tr, va, te = F.split_by_episode(df, val_frac=VAL_FRAC)
    test_eps = sorted(df[te].episode.unique())
    print(f"episodes: train {df[tr].episode.nunique()}, validation {df[va].episode.nunique()}, test {len(test_eps)}")
    ep = a.episode if a.episode is not None else pick_episode(test_eps, meta)
    if ep not in test_eps:
        raise SystemExit(f"episode {ep} is not a held-out test episode; choose from {test_eps}")
    ctx = df[tr].sample(min(a.context, int(tr.sum())), random_state=0)
    val = df[va]

    # all held-out rows (summary, adaptation plot), all validation rows (calibration), and one row
    # per person per sim step of the shown episode for a smooth video (the table has one every 0.4 s)
    T = df[te].reset_index(drop=True)
    V = val.reset_index(drop=True)
    q = F.build_rows(tracks, obst, meta, horizons=(a.h,), row_every=dt, episodes=[ep]).reset_index(drop=True)
    tag = f"{Path(a.data).name}_val{VAL_FRAC:g}_ctx{len(ctx)}"
    sets = {"val": V, "test": T, f"ep{ep}_h{a.h:g}": q}
    results, preds_ep = {}, {}
    print("\nbox scale from validation (1.00 = raw q05..q95; constant velocity is already sized there):")
    for m in a.models:
        fit_rows = V if m == "cv_box" else ctx
        P = predict_sets(m, fit_rows, sets, tag)
        scales = calibrate(P["val"], V)
        print(f"  {NAMES.get(m, m):30s} " + "   ".join(f"+{h:g} s x{s:.2f}" for h, s in scales.items()))
        results[m] = evaluate(T, apply_scale(P["test"], T, scales))
        preds_ep[m] = apply_scale(P[f"ep{ep}_h{a.h:g}"], q, scales)
    n_switch = df[te & df.since_switch.notna()].episode.nunique()
    summary(T, results, len(test_eps), n_switch)
    adapt_plot(T, results, a.h)
    if a.no_video:
        return

    panels = []
    for m in a.models:
        corners, mx, my, inside, _ = boxes(q, preds_ep[m])
        err = np.hypot(mx - q.true_x.to_numpy(), my - q.true_y.to_numpy())
        panels.append(dict(model=m, corners=corners, mx=mx, my=my, inside=inside, err=err,
                           area=box_area(preds_ep[m])))

    tr_ep = tracks[tracks.episode == ep]
    ob_ep = obst[obst.episode == ep] if obst is not None and len(obst) else obst
    em = meta["episodes"][str(ep)]
    times = np.sort(q.t.unique())
    qt = np.round(q.t.to_numpy() / dt).astype(int)
    b = meta["bounds"]
    trail = round(TRAIL_S / dt)
    robot = tr_ep[tr_ep.is_ego == 1].sort_values("t")
    robot_k = np.round(robot.t.to_numpy() / dt).astype(int)
    ppl = tr_ep[tr_ep.is_ego == 0]
    ppl_k = np.round(ppl.t.to_numpy() / dt).astype(int)
    tr_k = np.round(tr_ep.t.to_numpy() / dt).astype(int)

    n = len(panels)
    fig = plt.figure(figsize=(12.8, 7.2), facecolor="white")
    aspect = 12.8 / 7.2
    if n == 1:
        axes = [fig.add_axes([0.03, 0.04, 0.55, 0.78])]
    else:
        gap = 0.025
        w = min((0.96 - gap * (n - 1)) / n, 0.665 / aspect)
        hgt = w * aspect
        x0 = (1 - (n * w + (n - 1) * gap)) / 2
        y0 = 0.155 if n == 2 else 0.80 - 0.06 - hgt
        axes = [fig.add_axes([x0 + i * (w + gap), y0, w, hgt]) for i in range(n)]
    fig.text(0.5, 0.96, f"Where will everyone be in {a.h:g} seconds?", ha="center", va="center",
             fontsize=22, weight="bold")
    sub = fig.text(0.5, 0.915, "", ha="center", va="center", fontsize=12, color="0.3")
    handles = [
        Line2D([], [], marker="o", ls="", ms=10, color=C_PERSON, label="person (last 2 s as a trail)"),
        Line2D([], [], marker="o", ls="", ms=11, color=C_ROBOT, label="robot (its path so far)"),
        Patch(color=C_OBST, label="obstacle"),
        Patch(facecolor=C_RANGE, alpha=0.25, edgecolor=C_RANGE, label=f"predicted 80% range at +{a.h:g} s"),
        Line2D([], [], marker="o", ls="", ms=5, color="k", label=f"true position at +{a.h:g} s"),
        Line2D([], [], marker="o", ls="", ms=5, color="k", mec=C_MISS, mew=2, label="true position outside the range"),
    ]
    if n > 1:
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10.5, frameon=False,
                   bbox_to_anchor=(0.5, 0.045))
        fig.text(0.5, 0.012, f"Black dot: where that person really was {a.h:g} s later, shown now so you can check.   "
                              "Dot inside its box = the prediction was right.  Target: about 80% inside.",
                 ha="center", fontsize=10, color="0.25")
    else:
        fig.legend(handles=handles, loc="upper left", fontsize=12, frameon=False, bbox_to_anchor=(0.61, 0.82))
        fig.text(0.615, 0.50, f"Black dot: where that person really was {a.h:g} s later,\nshown now so you can check.\n\n"
                              "Dot inside its box = the prediction was right.\nTarget: about 80% inside.",
                 ha="left", va="top", fontsize=12, color="0.25")
    small = n >= 3

    def draw(k):
        t = times[k]; step = round(t / dt)
        regime = ppl.regime.to_numpy()[ppl_k == step]
        regime = regime[0] if len(regime) else ""
        sw = em.get("regime_switch_t")
        sw_txt = f"   ·   crowd switches {em['regime_before']} → {em['regime_after']} at {sw:.0f} s" if sw is not None else ""
        sub.set_text(f"held-out episode {ep}   ·   t = {t:4.1f} s   ·   crowd behaviour now: {regime}{sw_txt}")
        rows_now = qt == step
        so_far_rows = qt <= step
        now = tr_ep[tr_k == step]
        recent = ppl[(ppl_k <= step) & (ppl_k >= step - trail)]
        for ax, P in zip(axes, panels):
            ax.clear(); ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3]); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("0.75")
            if ob_ep is not None:
                for _, o in ob_ep.iterrows():
                    ax.add_patch(Circle((o.x, o.y), o.r, color=C_OBST, zorder=1))
            rmask = robot_k <= step
            ax.plot(robot.x.to_numpy()[rmask], robot.y.to_numpy()[rmask], color=C_ROBOT, lw=1.5, alpha=0.5, zorder=2)
            for _, g in recent.groupby("agent_id"):
                ax.plot(g.x, g.y, color=C_PERSON, lw=1.2, alpha=0.35, zorder=2)
            for i in np.flatnonzero(rows_now):
                ax.add_patch(Polygon(P["corners"][i], closed=True, facecolor=matplotlib.colors.to_rgba(C_RANGE, 0.15),
                                     edgecolor=C_RANGE, lw=1.2 if small else 1.4, zorder=3))
                ax.plot([q.px[i], P["mx"][i]], [q.py[i], P["my"][i]], color=C_RANGE, lw=0.8, alpha=0.8, zorder=3)
            ins = P["inside"][rows_now]
            tx, ty = q.true_x.to_numpy()[rows_now], q.true_y.to_numpy()[rows_now]
            ms = 3.5 if small else 4
            ax.plot(tx[ins], ty[ins], "o", color="k", ms=ms, zorder=6)
            ax.plot(tx[~ins], ty[~ins], "o", color="k", ms=ms, mec=C_MISS, mew=1.8, zorder=6)
            for _, r in now.iterrows():
                ax.add_patch(Circle((r.x, r.y), r.radius, color=C_ROBOT if r.is_ego else C_PERSON, zorder=5))
            cov = P["inside"][so_far_rows].mean()
            err = P["err"][so_far_rows].mean()
            area = P["area"][so_far_rows].mean()
            ax.set_title(NAMES.get(P["model"], P["model"]), fontsize=14 if small else 16, weight="bold",
                         pad=36 if small else 40)
            ax.text(0.5, 1.012, f"inside range now: {ins.sum()}/{len(ins)} ({ins.mean():.0%})   ·   so far: {cov:.0%}\n"
                                f"mean error so far: {err:.2f} m   ·   mean box area: {area:.1f} m²",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5 if small else 11.5,
                    linespacing=1.4, color="0.15" if abs(cov - 0.8) < 0.1 else C_MISS)

    outs = a.out or [f"results/demo_ep{ep}.mp4", f"results/demo_ep{ep}.gif"]
    fps = round(1 / dt)                                    # real time
    for out in outs:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        if out.endswith(".mp4"):
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            writer, dpi = FFMpegWriter(fps=fps, codec="libx264", bitrate=6000,
                                       extra_args=["-pix_fmt", "yuv420p"]), 100   # 1280x720
        else:
            writer, dpi = PillowWriter(fps=fps), 50                                  # 640x360
        with writer.saving(fig, out, dpi=dpi):
            for k in range(len(times)):
                draw(k); writer.grab_frame()
        print(f"wrote {out} ({len(times)} frames, {times[-1] - times[0]:.0f} s)")


if __name__ == "__main__":
    main()
