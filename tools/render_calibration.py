"""Calibration demo: do the predicted 5-95% boxes contain the people, after the crowd turns erratic? (T5 config.)

    python tools/render_calibration.py --seed 3013 --out media/demo/calibration_seed3013

Two closed-loop runs on the same seed, side by side: "TabPFN (reads the scene)" = tabpfn_split, box_q 0.15, and
"LightGBM (all rows)" = lgbm_all, box_q 0.15, both in tools/walkthrough.final_args (smoothing off, online offset per
batch, everyone switches into erratic at t = 5 s, duration 40 s), through run_closed.run_episode as in
tools/render_compare.py.

What a frame at time T shows, per panel: the box each planner drew at its replan at T - 2 s for where each person would
be at T (the RAW 5-95% box, q05/q95 along and across the person's heading, centred on the constant-velocity point;
not the q15-85 box widened by the online offset that the planner steers with), and where each person actually is at T:
green inside their box, red outside. People the planner had no forecast for are grey. Frames are 0.4 s apart (one per
replan), so every box meets its own outcome.

Scoring is run_closed's in-loop claim-B rule exactly (Predictor.inloop_stats): truth rows from features.build_rows on
the full episode history at h = 2 s, inner-merged on (agent_id, t); a forecast is "after the change" when it was MADE
at t >= 5 s, so the counter starts at 7 s, when the first of those comes due. The tool asserts that its full-episode
post-switch coverage and count equal run_episode's inloop_cov90_post / inloop_n_post, and compares them with the T5
CSVs (results/closed/*t5-split-q15.csv, *t5-lgbmall-q15.csv).

Writes
  <out>.gif          1280x720, frames for T = --t0 .. --t1, held 3 s on the last frame
  <out>_strip.png    3 moments after the switch (--strip), both planners
  <out>_stats.json   window and full-episode coverage per planner, the T5 CSV values, settings
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import walkthrough as W  # noqa: E402  (sets sys.path, env vars, matplotlib Agg)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

from render_compare import style  # noqa: E402

F = W.F
C_PERSON, C_PRED, C_ROBOT, C_RED, C_GREEN = W.C_PERSON, W.C_PRED, W.C_ROBOT, W.C_VETO, W.C_OK
C_INK, C_MUTED, C_GRID = W.C_INK, W.C_MUTED, W.C_GRID
BOX_FC = (0.91, 0.35, 0.05, 0.16)                            # C_PRED at 16% opacity
H = 2.0
PLANNERS = [("tabpfn_split", "TabPFN (reads the scene)", "T5 CSV: *t5-split-q15.csv"),
            ("lgbm_all", "LightGBM (all rows)", "T5 CSV: *t5-lgbmall-q15.csv")]
CSV_GLOB = {"tabpfn_split": "results/closed/2*-t5-split-q15.csv", "lgbm_all": "results/closed/2*-t5-lgbmall-q15.csv"}


# ----------------------------------------------------------------------------------------------
def scored_rows(logd, init, args, seed):
    """Every logged +2 s forecast joined to its truth, by Predictor.inloop_stats' rule. One row per (plan t, person):
    box (cx, cy, theta, q05a, q95a, q05c, q95c), truth (true_x, true_y, ddx, ddy), inside flag."""
    hist = [[dict(r, episode=seed) for r in init]] + [fr["state"] for fr in logd["frames"]]
    hist_df = pd.DataFrame([r for s in hist for r in s])
    obst_df = pd.DataFrame([dict(episode=seed, obstacle_id=i, shape="circle", **o) for i, o in enumerate(logd["obst"])])
    meta = F.load_kalman(args.bank, {"dt": args.dt, "bounds": list(logd["bounds"]), "episodes": {}})
    tr = F.build_rows(hist_df, obst_df, meta, horizons=(H,), noise=args.noise, seed=seed, row_every=meta["dt"])
    lab = tr[["agent_id", "t", "ddx", "ddy", "true_x", "true_y"]].assign(t=tr.t.round(6))
    cols = ["agent_id", "t", "h", "px", "py", "vx", "vy", "theta", "q05a", "q95a", "q05c", "q95c"]
    A = pd.concat([pl["pred"][cols] for pl in logd["plans"]
                   if pl.get("pred") is not None and len(pl["pred"]) and "q05a" in pl["pred"]], ignore_index=True)
    A = A[np.isclose(A.h, H)]
    m = A.assign(t=A.t.round(6)).merge(lab, on=["agent_id", "t"], how="inner").dropna(subset=["ddx", "ddy"])
    y = m[["ddx", "ddy"]].to_numpy(float)
    lo, hi = m[["q05a", "q05c"]].to_numpy(float), m[["q95a", "q95c"]].to_numpy(float)
    m = m.assign(inside=((y >= lo) & (y <= hi)).all(axis=1), cx=m.px + m.vx * H, cy=m.py + m.vy * H)
    # the dot is the true world position; check it against the box frame (dd = R(theta) (p_true - centre))
    c, s = np.cos(m.theta.to_numpy()), np.sin(m.theta.to_numpy())
    wx, wy = m.cx + c * m.ddx - s * m.ddy, m.cy + s * m.ddx + c * m.ddy
    assert np.allclose(wx, m.true_x, atol=1e-6) and np.allclose(wy, m.true_y, atol=1e-6), "truth frame mismatch"
    return m.reset_index(drop=True)


def coverage(m, switch_t, upto=None):
    """Post-switch (forecast made at t >= switch_t) joint 5-95% coverage, over forecasts that came due by `upto`."""
    sel = m.t >= switch_t - 1e-9
    if upto is not None:
        sel &= m.t + H <= upto + 1e-9
    n = int(sel.sum())
    return (float(m.inside[sel].mean()) if n else np.nan), n


def t5_row(kind, seed):
    hits = sorted(glob.glob(str(W.REPO / CSV_GLOB[kind])))
    if not hits:
        return None
    d = pd.read_csv(hits[-1])
    r = d[d.seed == seed]
    if not len(r):
        return None
    r = r.iloc[0]
    return {"csv": str(Path(hits[-1]).relative_to(W.REPO)).replace("\\", "/"),
            "inloop_cov90_post": float(r.inloop_cov90_post), "inloop_n_post": int(r.inloop_n_post),
            "inloop_cov90_pre": float(r.inloop_cov90_pre), "time_to_goal": float(r.time_to_goal),
            "csv_mean_cov90_post_all_seeds": float(d.inloop_cov90_post.mean()), "csv_seeds": int(d.seed.nunique())}


# ----------------------------------------------------------------------------------------------
def rbox(ax, cx, cy, th, la, ha, lc, hc, **kw):
    w, h = max(ha - la, 0.08), max(hc - lc, 0.08)            # display floor only; colour comes from the numeric test
    rect = Rectangle((la + (ha - la) / 2 - w / 2, lc + (hc - lc) / 2 - h / 2), w, h, **kw)
    rect.set_transform(Affine2D().rotate(th).translate(cx, cy) + ax.transData)
    return ax.add_patch(rect)


def state_at(R, T, dt):
    k = int(round(T / dt))
    frames = R["logd"]["frames"]
    return R["init"] if k == 0 else frames[min(k, len(frames)) - 1]["state"]


def draw_panel(ax, R, T, args):
    """Everything time-dependent in one panel at time T; returns (artists, the scored rows shown)."""
    out = []
    st = state_at(R, T, args.dt)
    m = R["m"][np.isclose(R["m"].t, round(T - H, 6))]
    have = set(m.agent_id.astype(int))
    for r in st[1:]:                                         # people with no scored forecast: grey
        if int(r["agent_id"]) not in have:
            out.append(ax.add_patch(Circle((r["x"], r["y"]), 0.2, color=C_MUTED, alpha=0.45, zorder=4)))
    for row in m.itertuples():
        out.append(rbox(ax, row.cx, row.cy, row.theta, row.q05a, row.q95a, row.q05c, row.q95c,
                        fc=BOX_FC, ec=C_PRED, lw=1.0, zorder=3))
        if not row.inside:                                   # tie a miss to its own box
            out += ax.plot([row.cx, row.true_x], [row.cy, row.true_y], color=C_RED, lw=1.0, alpha=0.55, zorder=5)
        out.append(ax.add_patch(Circle((row.true_x, row.true_y), 0.27, fc=C_GREEN if row.inside else C_RED,
                                       ec="white", lw=0.8, zorder=6)))
    k = min(int(round(T / args.dt)), len(R["S_rob"]) - 1)
    tr = R["S_rob"][:k + 1]
    out += ax.plot(tr[:, 0], tr[:, 1], color=C_ROBOT, lw=2.0, alpha=0.6, zorder=5)
    out.append(ax.add_patch(Circle(tuple(R["S_rob"][k]), 0.4, color=C_ROBOT, zorder=7)))
    return out, m


def base_panel(ax, R, goal_ms=16):
    style(ax, R["logd"]["bounds"], R["logd"]["obst"])
    ax.plot(*R["goal"], marker="*", color=C_ROBOT, ms=goal_ms, mec="white", zorder=6)


def legend(fig, y):
    fig.legend(handles=[Patch(facecolor=BOX_FC, edgecolor=C_PRED, label="5–95% box, drawn 2 s earlier"),
                        Line2D([], [], marker="o", ls="", color=C_GREEN, ms=12, label="person inside their box"),
                        Line2D([], [], marker="o", ls="", color=C_RED, ms=12, label="person outside"),
                        Line2D([], [], marker="o", ls="", color=C_MUTED, alpha=0.6, ms=9, label="no forecast"),
                        Line2D([], [], marker="o", ls="", color=C_ROBOT, ms=12, label="robot")],
               loc="center", bbox_to_anchor=(0.5, y), ncol=5, fontsize=12, frameon=False, handletextpad=0.3,
               columnspacing=1.4)


def draw_anim(runs, args, out, t0, t1, fps, hold_s, dpi=100):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(1, 2, left=0.02, right=0.98, bottom=0.115, top=0.745, wspace=0.04)
    axs = [fig.add_subplot(gs[0, i]) for i in range(2)]
    clock = fig.text(0.02, 0.98, "", fontsize=30, fontweight="bold", color=C_INK, va="top", family="monospace")
    banner = fig.text(0.5, 0.975, "", fontsize=22, fontweight="bold", va="top", ha="center")
    fig.text(0.98, 0.975, f"seed {args.seed} · {args.people} people\n{fps * 0.4:g}x speed", fontsize=11,
             color=C_MUTED, va="top", ha="right")
    fig.text(0.5, 0.905, "Boxes: where each model said each person would be, 2 s earlier.   "
                         "Dots: where they actually are.", fontsize=14, color=C_INK, va="top", ha="center")
    labels, pcts = [], []
    for ax, R in zip(axs, runs):
        base_panel(ax, R)
        bb = ax.get_position()
        fig.text(bb.x0, 0.835, R["label"], fontsize=18, color=C_INK, fontweight="bold", va="bottom")
        labels.append(fig.text(bb.x0, 0.758, "", fontsize=13, color=C_MUTED, va="bottom"))
        pcts.append(fig.text(bb.x1, 0.755, "", fontsize=34, color=C_INK, fontweight="bold", va="bottom", ha="right"))
    legend(fig, 0.072)
    # timeline: t0 .. t1 with the switch marked
    tl = fig.add_axes([0.06, 0.012, 0.88, 0.03])
    tl.set_xlim(t0, t1); tl.set_ylim(0, 1); tl.axis("off")
    tl.plot([t0, t1], [0.3, 0.3], color=C_GRID, lw=5, solid_capstyle="round")
    tl.plot([args.switch_t] * 2, [0.0, 0.75], color=C_RED, lw=2.2)
    tl.text(args.switch_t + 0.12, 0.62, "crowd turns erratic", color=C_RED, fontsize=11, va="center")
    tl.text(t0 - 0.15, 0.3, f"{t0:g} s", color=C_MUTED, fontsize=10, va="center", ha="right")
    tl.text(t1 + 0.15, 0.3, f"{t1:g} s", color=C_MUTED, fontsize=10, va="center", ha="left")
    prog, = tl.plot([t0, t0], [0.3, 0.3], color=C_INK, lw=5, solid_capstyle="round")
    times = np.round(np.arange(t0, t1 + 1e-9, 0.4), 6)
    n_frames = len(times) + int(hold_s * fps)
    dyn = [[], []]

    def frame(f):
        T = float(times[min(f, len(times) - 1)])
        clock.set_text(f"t = {T:4.1f} s")
        prog.set_data([t0, T], [0.3, 0.3])
        if T >= args.switch_t - 1e-9:
            banner.set_text(f"crowd turned erratic at {args.switch_t:g} s"); banner.set_color(C_RED)
        else:
            banner.set_text(f"calm crowd · turns erratic at {args.switch_t:g} s"); banner.set_color(C_MUTED)
        for i, (ax, R) in enumerate(zip(axs, runs)):
            for a in dyn[i]:
                a.remove()
            dyn[i], m = draw_panel(ax, R, T, args)
            cov, n = coverage(R["m"], args.switch_t, upto=T)
            now = f"now {int(m.inside.sum())} of {len(m)} inside"
            if n == 0:
                due = "" if T < args.switch_t else f" · counts from {args.switch_t + H:g} s"
                labels[i].set_text(f"inside the box since the change\n{now}{due}")
                pcts[i].set_text("—")
            else:
                extra = (f"whole run, to {R['end_s']:.1f} s: {100 * R['full_cov']:.0f}%" if f >= len(times)
                         else f"{now} · {n} scored")
                labels[i].set_text(f"inside the box since the change\n{extra}")
                pcts[i].set_text(f"{100 * cov:.0f}%")
        return []

    anim = animation.FuncAnimation(fig, frame, frames=n_frames, interval=1000 / fps, blit=False)
    anim.save(out, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return n_frames


def draw_strip(runs, args, path, moments):
    fig = plt.figure(figsize=(15, 11.4), facecolor="white")
    gs = fig.add_gridspec(2, len(moments), left=0.02, right=0.98, bottom=0.07, top=0.85, wspace=0.04, hspace=0.3)
    for i, R in enumerate(runs):
        for j, T in enumerate(moments):
            ax = fig.add_subplot(gs[i, j])
            base_panel(ax, R, goal_ms=14)
            _, m = draw_panel(ax, R, T, args)
            cov, n = coverage(R["m"], args.switch_t, upto=T)
            run_txt = f"since the change: {100 * cov:.0f}%" if n else "since the change: —"
            ax.set_title(f"t = {T:.1f} s · now {int(m.inside.sum())} of {len(m)} inside\n{run_txt}", fontsize=13,
                         color=C_INK, loc="left")
            if j == 0:
                bb = ax.get_position()
                fig.text(bb.x0, bb.y1 + 0.05, R["label"], fontsize=17, fontweight="bold", color=C_INK, va="bottom")
    legend(fig, 0.03)
    fig.text(0.02, 0.975, f"Seed {args.seed}: the crowd turns erratic at {args.switch_t:g} s", fontsize=18,
             fontweight="bold", color=C_INK, va="top")
    fig.text(0.02, 0.945, "Boxes: each model's 5–95% range for every person, drawn 2 s earlier.  "
                          "Dots: where the people actually were.", fontsize=14, color=C_INK, va="top")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=3013)
    ap.add_argument("--out", default="media/demo/calibration_seed3013")
    ap.add_argument("--box-q", type=float, default=0.15)
    ap.add_argument("--t0", type=float, default=3.2, help="first frame (s; a replan time + 2 s, multiple of 0.4)")
    ap.add_argument("--t1", type=float, default=20.0, help="last frame (s)")
    ap.add_argument("--fps", type=float, default=3.0, help="frames per s; one frame = 0.4 s of simulation")
    ap.add_argument("--strip", type=float, nargs=3, default=[8.0, 12.0, 16.0])
    ap.add_argument("--cache", default=None, help="pickle of the two runs (reused if present)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(3)
    T0 = time.perf_counter()
    base = W.final_args(a.seed, box_q=a.box_q)               # T5: smoothing off, offset batch, duration 40
    base.dt = 0.2
    print(f"T5 configuration: seed {a.seed}, {W.settings_line(base)}", flush=True)
    cache = Path(a.cache) if a.cache else None
    if cache and cache.exists():
        raw = pickle.loads(cache.read_bytes())
        print(f"runs read from {cache}", flush=True)
    else:
        bank = W.load_bank(base.noise, base.hz)[0]
        raw = {}
        for kind, _, _ in PLANNERS:
            res, logd, init, wall = W.run_planner(kind, bank, copy.copy(base), a.seed)
            logd = {k: logd[k] for k in ("frames", "plans", "obst", "bounds")}
            logd["plans"] = [{"t": p["t"], "pred": p["pred"]} for p in logd["plans"]]
            raw[kind] = dict(res=res, logd=logd, init=init, wall=wall)
            print(f"{kind}: {len(logd['plans'])} plans, goal {res['time_to_goal']} s, {wall:.0f} s wall", flush=True)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps(raw))
    runs, rows = [], []
    for kind, label, _ in PLANNERS:
        d = raw[kind]
        res, logd, init = d["res"], d["logd"], d["init"]
        m = scored_rows(logd, init, base, a.seed)
        full, n_full = coverage(m, base.switch_t)
        assert n_full == int(res["inloop_n_post"]) and abs(full - res["inloop_cov90_post"]) < 1e-9, \
            (kind, full, n_full, res["inloop_cov90_post"], res["inloop_n_post"])
        win, n_win = coverage(m, base.switch_t, upto=a.t1)
        rob = np.array([[init[0]["x"], init[0]["y"]]] + [[fr["state"][0]["x"], fr["state"][0]["y"]] for fr in logd["frames"]])
        R = dict(kind=kind, label=label, res=res, logd=logd, init=init, m=m, S_rob=rob, full_cov=full,
                 end_s=float(res["episode_s"]), goal=(init[0]["goal_x"], init[0]["goal_y"]))
        runs.append(R)
        t5 = t5_row(kind, a.seed)
        pre = m[m.t < base.switch_t - 1e-9]
        rows.append({"planner": kind, "label": label, "box_q": a.box_q,
                     "window": {"from_s": a.t0, "to_s": a.t1,
                                "cov90_post": win, "n_post": n_win,
                                "note": f"forecasts made at t >= {base.switch_t:g} s that came due by {a.t1:g} s "
                                        f"(what the GIF's last frame shows)"},
                     "full_episode": {"cov90_post": full, "n_post": n_full,
                                      "cov90_pre": float(pre.inside.mean()) if len(pre) else None,
                                      "episode_s": float(res["episode_s"]), "reached": bool(res["reached"]),
                                      "time_to_goal": None if not res["reached"] else float(res["time_to_goal"]),
                                      "equals_run_episode_inloop_cov90_post": True},
                     "t5_csv": t5,
                     "matches_t5": None if t5 is None else bool(abs(full - t5["inloop_cov90_post"]) < 1e-6
                                                                and n_full == t5["inloop_n_post"]),
                     "wall_s": float(d["wall"])})
        t5s = "-" if t5 is None else f"{100 * t5['inloop_cov90_post']:.1f}% (n {t5['inloop_n_post']})"
        print(f"{label:<26} post-switch cov90: window to {a.t1:g} s {100 * win:.1f}% (n {n_win}), whole run "
              f"{100 * full:.1f}% (n {n_full}); T5 CSV {t5s}", flush=True)
    out = W.REPO / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "--no-optional-locks", "rev-parse", "--short", "HEAD"], cwd=W.REPO,
                            capture_output=True, text=True).stdout.strip()
    stats = {"seed": a.seed, "commit": commit, "generated": time.strftime("%Y-%m-%d %H:%M"),
             "settings": W.settings_line(base), "switch_t": base.switch_t, "switch_to": base.switch_to,
             "box_drawn": "raw 5-95% box (q05/q95 along and across heading, logged in plans[*]['pred']), the box "
                          "run_closed's inloop_cov90 scores; not the q15-85 + online-offset box the planner steers with",
             "truth": "features.build_rows(full history, h = 2 s, row_every = dt) true positions, inner-merged on "
                      "(agent_id, t), as Predictor.inloop_stats",
             "post_definition": "forecast made at t >= switch_t (so the first post forecasts come due at switch_t + 2 s)",
             "nominal_joint_coverage": "a 5-95% interval on each of two axes covers ~0.9 x 0.9 = 81% jointly if "
                                       "calibrated and independent",
             "planners": rows}
    Path(f"{out}_stats.json").write_text(json.dumps(stats, indent=1, default=str), encoding="utf-8")
    draw_strip(runs, base, f"{out}_strip.png", a.strip)
    print(f"wrote {out}_stats.json, {out}_strip.png", flush=True)
    t1 = time.perf_counter()
    n = draw_anim(runs, base, f"{out}.gif", a.t0, a.t1, a.fps, hold_s=3.0)
    print(f"wrote {out}.gif ({n} frames, {Path(f'{out}.gif').stat().st_size / 1e6:.1f} MB, "
          f"{time.perf_counter() - t1:.0f} s); total {time.perf_counter() - T0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
