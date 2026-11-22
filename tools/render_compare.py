"""Side-by-side demo: several planners on the same crowd, in the FINAL closed-loop configuration (T5).

    python tools/render_compare.py --seed S --planners stop_go reactive:0.4 vo:1 tabpfn_split:0.15 [oracle:0.4] \
        --out media/demo/seedS

planner:dial sets the margin for cv / cv_band / reactive / oracle, the collision weight for vo, and box_q for the
learned planners (tabpfn*, lgbm*); stop_go takes no dial. Without a dial: margin 0.2, vo weight 1, box_q 0.15.
Every planner runs the same seed (same crowd, start and goal) through run_closed.run_episode with trace writing off,
in a temp working directory (nothing under results/). The configuration is tools/walkthrough.final_args: cell r2 i0.2
p24, crossing crowd, everyone switches into erratic at t = 5 s, bank data/crowd_cross_t2 without erratic episodes,
frozen context 2,000 rows, replan 0.4 s, top speed 1.0 m/s, TabPFN-3.5-Fast 1 estimator, and the T4 gate's
smoothing / online offset (defaults from walkthrough.FINAL_SMOOTH / FINAL_OFFSET; override with --smooth / --offset).

Writes
  <out>_paths.png   one panel per planner: robot path coloured by speed, people's paths faint, red rings where the
                    robot came within 0.5 m of a person; title: time to goal, time near people, contacts, stopped
  <out>.mp4         (ffmpeg found) or <out>.gif (otherwise): synchronized animation, one panel per planner, people,
                    robot and trail, that planner's own boxes at +2 s, a big clock, the switch marked, live counters;
                    one frame per 0.2 s simulation step, played at 2x speed (10 fps), 1280x720
  <out>_stats.json  per-planner episode metrics (run_closed.run_episode's result row) and the settings used
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import walkthrough as W  # noqa: E402  (sets sys.path, env vars, matplotlib Agg)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.collections import LineCollection, PatchCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

C_PERSON, C_PRED, C_ROBOT, C_NEAR = W.C_PERSON, W.C_PRED, W.C_ROBOT, W.C_VETO
C_INK, C_MUTED, C_OBST, C_GRID = W.C_INK, W.C_MUTED, W.C_OBST, W.C_GRID
NAMES = {"stop_go": "Stop-and-go", "reactive": "Reactive", "vo": "Velocity obstacle", "cv": "Constant velocity",
         "cv_band": "CV + fixed band", "oracle": "Oracle (true future)", "tabpfn_split": "Foresight (TabPFN split)",
         "tabpfn": "TabPFN frozen", "tabpfn_live": "TabPFN live", "lgbm": "LightGBM", "lgbm_all": "LightGBM (all rows)",
         "lgbm_live": "LightGBM live"}
NO_BOXES = ("stop_go", "reactive", "vo")
NEAR = 0.5                                                   # run_episode's intrusion threshold (surface gap, m)
FPS = 10                                                     # 0.2 s steps at 2x speed
BOX_FC = (0.91, 0.35, 0.05, 0.22)                            # C_PRED at 22% opacity


def parse_spec(spec):
    kind, _, dial = spec.partition(":")
    dial = float(dial) if dial else None
    if kind.startswith(("tabpfn", "lgbm")):
        field, default, tag = "box_q", 0.15, "q"
    elif kind == "vo":
        field, default, tag = "vo_w", 1.0, "w"
    elif kind == "stop_go":
        field, default, tag = None, None, ""
    else:
        field, default, tag = "margin", 0.2, "m"
    val = default if dial is None else dial
    label = NAMES.get(kind, kind) + (f"  ({tag} {val:g})" if field else "")
    return dict(spec=spec, kind=kind, field=field, dial=val, label=label)


def series(init, frames, dt):
    """Per step k = 0..K: robot xy, speed, people (id, x, y, r), near flag, contact flag; run_episode's definitions."""
    states = [init] + [fr["state"] for fr in frames]
    rob = np.array([[s[0]["x"], s[0]["y"]] for s in states])
    spd = np.array([np.hypot(s[0]["vx"], s[0]["vy"]) for s in states])
    ppl = [np.array([[r["agent_id"], r["x"], r["y"], r["radius"]] for r in s[1:]]) for s in states]
    near = np.zeros(len(states), bool)
    hit = np.zeros(len(states), bool)
    for k in range(1, len(states)):
        r0 = states[k][0]
        g = np.array([np.hypot(r["x"] - r0["x"], r["y"] - r0["y"]) - r["radius"] - r0["radius"] for r in states[k][1:]])
        near[k] = (g < NEAR).any()
        hit[k] = bool(frames[k - 1]["info"]["collided"])
    events = hit & ~np.r_[False, hit[:-1]]
    return dict(t=np.arange(len(states)) * dt, rob=rob, spd=spd, ppl=ppl, near=near, hit=hit,
                near_s=np.cumsum(near) * dt, hit_steps=np.cumsum(hit), events=np.cumsum(events),
                stopped=np.cumsum(np.r_[False, spd[1:] < 0.05]) * dt)


def boxes_by_plan(plans):
    """[(t, array of rotated +2 s boxes: cx, cy, theta, lo_a, hi_a, lo_c, hi_c)] from the logged predictions."""
    out = []
    for pl in plans:
        pr = pl.get("pred")
        if pr is None or not len(pr) or "lo_a" not in pr:
            out.append((pl["t"], np.zeros((0, 7))))
            continue
        r = pr[np.isclose(pr.h, 2.0)]
        out.append((pl["t"], np.c_[r.px + r.vx * r.h, r.py + r.vy * r.h, r.theta, r.lo_a, r.hi_a, r.lo_c, r.hi_c]))
    return out


def box_patches(B, ax):
    ps = []
    for cx, cy, th, la, ha, lc, hc in B:
        w, h = max(ha - la, 0.12), max(hc - lc, 0.12)
        x0, y0 = (la + ha) / 2 - w / 2, (lc + hc) / 2 - h / 2
        rect = Rectangle((x0, y0), w, h)
        rect.set_transform(Affine2D().rotate(th).translate(cx, cy) + ax.transData)
        ps.append(rect)
    return ps


def grid(n):
    return {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2)}.get(n, (2, 3) if n <= 6 else (3, 3))


def style(ax, bounds, obst):
    ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3]); ax.set_aspect("equal")
    ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(C_GRID)
    for o in obst:
        ax.add_patch(Circle((o["x"], o["y"]), o["r"], color=C_OBST, zorder=1))


def goal_txt(res, duration):
    return f"goal in {res['time_to_goal']:.1f} s" if res["reached"] else f"goal not reached in {duration:g} s"


# ----------------------------------------------------------------------------------------------
def draw_paths(runs, args, path):
    nr, nc = grid(len(runs))
    fig, axs = plt.subplots(nr, nc, figsize=(5.0 * nc, 5.6 * nr), squeeze=False)
    norm = plt.Normalize(0, args.max_speed)
    for ax, R in zip(axs.flat, runs):
        S, res, logd = R["S"], R["res"], R["logd"]
        style(ax, logd["bounds"], logd["obst"])
        ids = np.unique(np.concatenate([p[:, 0] for p in S["ppl"] if len(p)]))
        for aid in ids:
            xy = np.array([p[p[:, 0] == aid][0, 1:3] if (p[:, 0] == aid).any() else [np.nan, np.nan] for p in S["ppl"]])
            ax.plot(xy[:, 0], xy[:, 1], color=C_PERSON, lw=0.6, alpha=0.18, zorder=2)
        pts = S["rob"]
        lc = LineCollection(np.stack([pts[:-1], pts[1:]], 1), cmap="viridis", norm=norm, lw=3.0, zorder=4)
        lc.set_array(S["spd"][1:]); ax.add_collection(lc)
        for k in np.flatnonzero(S["near"]):
            ax.add_patch(Circle(tuple(pts[k]), 0.4, fill=False, ec=C_NEAR, lw=1.0, alpha=0.7, zorder=5))
        ks = int(round(args.switch_t / args.dt))
        if ks < len(pts):
            ax.plot(*pts[ks], "D", color=C_INK, ms=7, mec="white", zorder=6)
            ax.annotate(f"crowd turns erratic ({args.switch_t:g} s)", pts[ks], xytext=(6, -14), textcoords="offset points",
                        fontsize=8, color=C_INK)
        ax.plot(*pts[0], "o", color=C_ROBOT, ms=8, zorder=6)
        g = R["goal"]
        ax.plot(*g, marker="*", color=C_ROBOT, ms=16, mec="white", zorder=6)
        ax.set_title(f"{R['label']}\n{goal_txt(res, args.duration)} · near people {res['intrusion_s']:.1f} s\n"
                     f"contacts {R['events']} ({int(res['collision_steps'])} steps) · stopped {res['seconds_stopped']:.1f} s",
                     fontsize=11, color=C_INK, loc="left")
    for ax in list(axs.flat)[len(runs):]:
        ax.axis("off")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axs, fraction=0.02, pad=0.02)
    cb.set_label("robot speed (m/s)", fontsize=10, color=C_MUTED)
    fig.legend(handles=[Line2D([], [], color=C_PERSON, lw=1, alpha=0.4, label="people's paths"),
                        Line2D([], [], marker="o", ls="", mfc="none", mec=C_NEAR, ms=9, label=f"robot within {NEAR} m of a person"),
                        Line2D([], [], marker="D", ls="", color=C_INK, label="where the robot was at the switch"),
                        Line2D([], [], marker="*", ls="", color=C_ROBOT, ms=12, label="goal")],
               loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.suptitle(f"Same crowd, seed {args.seed}: {args.people} people, everyone turns erratic at {args.switch_t:g} s  "
                 f"({W.settings_line(args)})", fontsize=12, color=C_INK, x=0.02, ha="left")
    fig.savefig(path, dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def draw_anim(runs, args, path_noext, dpi=100):
    use_ff = shutil.which("ffmpeg") is not None and animation.writers.is_available("ffmpeg")
    out = Path(f"{path_noext}.mp4" if use_ff else f"{path_noext}.gif")
    nr, nc = grid(len(runs))
    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi, facecolor="white")
    top = 0.86
    gs = fig.add_gridspec(nr, nc, left=0.02, right=0.98, bottom=0.03, top=top, wspace=0.06, hspace=0.22)
    axs = [fig.add_subplot(gs[i // nc, i % nc]) for i in range(len(runs))]
    fs = 15 if len(runs) <= 3 else 12
    clock = fig.text(0.02, 0.975, "", fontsize=30, fontweight="bold", color=C_INK, va="top", family="monospace")
    banner = fig.text(0.5, 0.965, "", fontsize=20, fontweight="bold", va="top", ha="center")
    fig.text(0.98, 0.975, f"seed {args.seed} · {args.people} people · 2x speed", fontsize=11, color=C_MUTED, va="top", ha="right")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=C_ROBOT, ms=10, label="robot"),
                        Line2D([], [], marker="o", ls="", color=C_PERSON, ms=8, label="person"),
                        Patch(facecolor=C_PRED, alpha=0.35, edgecolor=C_PRED, label="predicted range at +2 s"),
                        Line2D([], [], marker="o", ls="", mfc="none", mec=C_NEAR, ms=10, mew=2, label=f"within {NEAR} m")],
               loc="upper right", bbox_to_anchor=(0.98, 0.935), ncol=4, fontsize=11, frameon=False)
    dyn = [[] for _ in runs]
    counters = []
    for ax, R in zip(axs, runs):
        style(ax, R["logd"]["bounds"], R["logd"]["obst"])
        ax.plot(*R["goal"], marker="*", color=C_ROBOT, ms=16, mec="white", zorder=6)
        ax.set_title(R["label"], fontsize=fs, color=C_INK, loc="left", fontweight="bold")
        counters.append(ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=fs - 3, va="top", color=C_INK, zorder=20,
                                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_GRID, alpha=0.9)))
    K = max(len(R["S"]["t"]) for R in runs)
    n_frames = K + 2 * FPS                                   # hold the last frame 2 s

    def frame(f):
        k_glob = min(f, K - 1)
        tt = k_glob * args.dt
        clock.set_text(f"t = {tt:4.1f} s")
        if tt >= args.switch_t - 1e-9:
            banner.set_text(f"crowd turns erratic at {args.switch_t:g} s"); banner.set_color(C_NEAR)
        else:
            banner.set_text(f"calm crowd · turns erratic at {args.switch_t:g} s"); banner.set_color(C_MUTED)
        for i, (ax, R) in enumerate(zip(axs, runs)):
            for a in dyn[i]:
                a.remove()
            dyn[i] = []
            S = R["S"]
            k = min(f, len(S["t"]) - 1)
            done = k < f or (R["res"]["reached"] and k == len(S["t"]) - 1)
            P = S["ppl"][k]
            pc = PatchCollection([Circle((x, y), r) for _, x, y, r in P], fc=C_PERSON, ec="white", lw=0.5, zorder=5)
            dyn[i].append(ax.add_collection(pc))
            if R["boxes"] is not None:
                t_k = S["t"][k]
                B = None
                for tp, b in R["boxes"]:
                    if tp <= t_k + 1e-9:
                        B = b
                    else:
                        break
                if B is not None and len(B):
                    # one patch per box: each carries its own rotation (heading frame)
                    for p in box_patches(B, ax):
                        p.set_facecolor(BOX_FC); p.set_edgecolor(C_PRED); p.set_linewidth(0.9); p.set_zorder(3)
                        dyn[i].append(ax.add_patch(p))
            tr = S["rob"][max(0, k - 400):k + 1]
            dyn[i] += ax.plot(tr[:, 0], tr[:, 1], color=C_ROBOT, lw=2.2, alpha=0.8, zorder=6)
            dyn[i].append(ax.add_patch(Circle(tuple(S["rob"][k]), 0.4, color=C_ROBOT, zorder=7)))
            if S["near"][k]:
                dyn[i].append(ax.add_patch(Circle(tuple(S["rob"][k]), 0.75, fill=False, ec=C_NEAR, lw=2.5, zorder=8)))
            txt = f"near people {S['near_s'][k]:4.1f} s\ncontacts {int(S['events'][k])}"
            if done and R["res"]["reached"]:
                txt += f"\ngoal reached {R['res']['time_to_goal']:.1f} s"
            counters[i].set_text(txt)
        return []

    anim = animation.FuncAnimation(fig, frame, frames=n_frames, interval=1000 / FPS, blit=False)
    writer = animation.FFMpegWriter(fps=FPS, bitrate=4000) if use_ff else animation.PillowWriter(fps=FPS)
    anim.save(out, writer=writer, dpi=dpi)
    plt.close(fig)
    return out, n_frames


# ----------------------------------------------------------------------------------------------
def jsonable(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--planners", nargs="+", required=True, help="planner[:dial], e.g. reactive:0.4 vo:1 tabpfn_split:0.15")
    ap.add_argument("--out", required=True, help="output prefix, e.g. media/demo/seed3007")
    ap.add_argument("--smooth", default=W.FINAL_SMOOTH, choices=["on", "off"])
    ap.add_argument("--offset", default=W.FINAL_OFFSET, choices=["off", "row", "batch"])
    ap.add_argument("--duration", type=float, default=40.0, help="episode cap in s (T5: 40)")
    ap.add_argument("--dpi", type=int, default=100, help="animation dpi (100 = 1280x720)")
    ap.add_argument("--no-anim", action="store_true", help="paths and stats only")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(3)
    T0 = time.perf_counter()
    specs = [parse_spec(s) for s in a.planners]
    base = W.final_args(a.seed, smooth=a.smooth, offset=a.offset, duration=a.duration)
    base.dt = 0.2
    print(f"final configuration: seed {a.seed}, duration {a.duration:g} s, people {base.people}, switch into "
          f"{base.switch_to} at {base.switch_t:g} s, replan {base.replan:g} s, latency {base.latency:g} s, "
          f"{W.settings_line(base)}", flush=True)
    need_bank = any(s["kind"].startswith(("tabpfn", "lgbm", "cv_band")) for s in specs)
    bank = W.load_bank(base.noise, base.hz)[0] if need_bank else None
    runs = []
    for sp in specs:
        args = copy.copy(base)
        if sp["field"]:
            setattr(args, sp["field"], sp["dial"])
        res, logd, init, wall = W.run_planner(sp["kind"], bank, args, a.seed)
        S = series(init, logd["frames"], base.dt)
        assert abs(S["near_s"][-1] - res["intrusion_s"]) < 1e-6, (S["near_s"][-1], res["intrusion_s"])
        assert int(S["hit_steps"][-1]) == int(res["collision_steps"]), (S["hit_steps"][-1], res["collision_steps"])
        boxes = None if sp["kind"] in NO_BOXES else boxes_by_plan(logd["plans"])
        R = dict(sp, res=res, logd=logd, S=S, boxes=boxes, wall=wall, events=int(S["events"][-1]),
                 goal=(init[0]["goal_x"], init[0]["goal_y"]))
        runs.append(R)
        print(f"{sp['label']:<34} {goal_txt(res, a.duration):<26} near {res['intrusion_s']:4.1f} s  contacts {R['events']} "
              f"({int(res['collision_steps'])} steps)  stopped {res['seconds_stopped']:.1f} s  min gap {res['min_gap']:.2f} m  "
              f"({wall:.0f} s wall)", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "--no-optional-locks", "rev-parse", "--short", "HEAD"], cwd=W.REPO,
                            capture_output=True, text=True).stdout.strip()
    cfg = {k: jsonable(v) if not isinstance(v, (list, tuple)) else list(v) for k, v in vars(base).items()
           if k not in ("log", "tag")}
    stats = {"seed": a.seed, "commit": commit, "generated": time.strftime("%Y-%m-%d %H:%M"),
             "settings": W.settings_line(base), "config": cfg,
             "planners": [{"spec": R["spec"], "planner": R["kind"], "label": R["label"], "dial_field": R["field"],
                           "dial": R["dial"], "contact_events": R["events"], "wall_s": R["wall"],
                           **{k: jsonable(v) for k, v in R["res"].items()}} for R in runs]}
    Path(f"{out}_stats.json").write_text(json.dumps(stats, indent=1, default=str), encoding="utf-8")
    draw_paths(runs, base, f"{out}_paths.png")
    print(f"wrote {out}_stats.json, {out}_paths.png", flush=True)
    if not a.no_anim:
        t1 = time.perf_counter()
        anim_path, n = draw_anim(runs, base, str(out), dpi=a.dpi)
        print(f"wrote {anim_path} ({n} frames, {anim_path.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - t1:.0f} s)")
    print(f"done in {time.perf_counter() - T0:.0f} s; {W.settings_line(base)}")


if __name__ == "__main__":
    main()
