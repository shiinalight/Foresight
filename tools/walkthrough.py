"""One real decision moment of the closed loop, stage by stage, with the real values.

    python tools/walkthrough.py                 # bank + episode + figures + tables + docs/walkthrough.html
    python tools/walkthrough.py --seed 6001 --smooth off --offset off

Runs one tabpfn_split episode in the FINAL closed-loop configuration (T5, amendment 2 in RESULTS.md: cell r2 i0.2
p24, crossing crowd, every person switches into erratic at t = 5 s; bank data/crowd_cross_t2 without erratic
episodes; frozen context of 2,000 natural-frequency rows + scene model on <= 250 recent rows refit every 2 s;
box_q 0.15, margin 0.2, replan 0.4 s, top speed 1.0 m/s, TabPFN-3.5-Fast with 1 estimator), picks one replan
moment after the switch by a fixed rule (see choose_moment), and writes
    docs/walkthrough/*.png    one figure per stage
    docs/walkthrough/*.csv / *.json   the tables behind them
    docs/walkthrough.html     the page (CSS copied from docs/explainer.html at run time)
Nothing else is written: no caches, no results/ files (trace.write is disabled, the episode runs in a
temp working directory). Rerun after a feature change; the page records FEATURES_VERSION and the commit.

--smooth on|off and --offset off|row|batch: the two settings the T4 gate decided (defaults FINAL_SMOOTH /
FINAL_OFFSET below = T4 addendum part 4: smoothing off, offset on with the per-batch rule).
--dev-cache PATH: pickle (res, logd) there and reuse it on the next run (figure iteration only).

final_args / load_bank / run_planner are shared with tools/render_compare.py.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

from foresight.core import conformal as CF  # noqa: E402
from foresight.core import context as C  # noqa: E402
from foresight.core import features as F  # noqa: E402
from foresight.core import models as M  # noqa: E402
from foresight.core import occupancy as O  # noqa: E402
from foresight.core import planner as PL  # noqa: E402
from foresight.core import schema  # noqa: E402

OUT = REPO / "docs" / "walkthrough"
PAGE = REPO / "docs" / "walkthrough.html"
BANK_DIR = REPO / "data" / "crowd_cross_t2"
BANK_EXCLUDE = "erratic"
R_ROBOT, R_PERSON = 0.4, 0.3
# T4 gate outcomes, RESULTS.md 'T4 addendum, part 4' (3df14bf): smoothing OFF; online offset ON, rule (ii) per batch
# (d += 0.05 (batch miss rate - 0.2) per 2 s refresh, observed labels). Override with --smooth / --offset.
FINAL_SMOOTH, FINAL_OFFSET = "off", "batch"
SPLIT_MIN = 20                                   # run_closed.SPLIT_THIN: fewer recent rows = frozen model alone


def qs5(box_q):
    """The five quantiles every learned planner predicts: 5-95% plus the planner box [box_q, 1 - box_q]."""
    return (0.05, box_q, 0.5, 1 - box_q, 0.95)


def pct(box_q):
    return f"{100 * box_q:.0f}–{100 - 100 * box_q:.0f}%"

# figure palette (matches docs/explainer.html tokens)
C_PERSON, C_PRED, C_ROBOT, C_VETO, C_OK = "#2a6fdb", "#e8590c", "#7a3fb5", "#c2362f", "#2b7a4b"
C_INK, C_MUTED, C_OBST, C_GRID, C_WARN = "#1f2328", "#8b949e", "#c9ced6", "#eceef1", "#a86a00"
SRC_COLOR = {"recent": C_PERSON, "frozen": C_MUTED}
SRC_LABEL = {"recent": "scene model: recent rows, this run", "frozen": "frozen model: bank rows"}
MODEL_COLOR = {"frozen": C_MUTED, "scene": C_PERSON, "final": C_PRED}
REASON_COLOR = {"none": C_OK, "box": C_VETO, "short check": C_WARN, "bounds": C_MUTED, "obstacle": C_MUTED}
DPI = 110


# ----------------------------------------------------------------------------------------------
# 1. data: bank, episode
# ----------------------------------------------------------------------------------------------
def load_bank(noise=0.1, hz=(1.0, 2.0), exclude=BANK_EXCLUDE):
    """run_closed.main's bank: rows_for(data/crowd_cross_t2) minus every episode with `exclude` before or after a
    switch. Reads the rows_for parquet cache if present; otherwise builds the rows in memory (never writes data/)."""
    meta = F.load_kalman(BANK_DIR, {})
    kq, kr = F.kalman_params(meta, noise)
    hzs = "_".join(f"{h:g}" for h in hz)
    cache = BANK_DIR / f"rows_{F.FEATURES_VERSION}_noise{noise}_h{hzs}_kq{kq:g}_kr{kr:g}.parquet"
    t0 = time.perf_counter()
    if cache.exists():
        bank = pd.read_parquet(cache)
    else:
        tracks, obst, meta = schema.load(BANK_DIR)
        F.load_kalman(BANK_DIR, meta)
        bank = F.build_rows(tracks, obst, meta, noise=noise, seed=0, horizons=hz)
    n_all, e_all = len(bank), bank.episode.nunique()
    if exclude:
        em = json.loads((BANK_DIR / "meta.json").read_text())["episodes"]
        bad = set(bank[bank.regime == exclude].episode.unique())
        bad |= {int(e) for e, m in em.items() if exclude in (m.get("regime_before"), m.get("regime_after"))}
        bank = bank[~bank.episode.isin(bad)].reset_index(drop=True)
    print(f"bank {BANK_DIR.name}: {bank.episode.nunique()} of {e_all} episodes, {len(bank)} of {n_all} rows without "
          f"'{exclude}' ({'parquet cache' if cache.exists() else 'built in memory'}, {time.perf_counter() - t0:.0f} s); "
          f"Kalman q = {kq:g}, r = {kr:g}", flush=True)
    return bank, {"rows_all": n_all, "episodes_all": int(e_all), "kalman_q": kq, "kalman_r": kr}


def final_args(seed=None, smooth=FINAL_SMOOTH, offset=FINAL_OFFSET, duration=40.0, box_q=0.15, margin=0.2,
               vo_w=1.0):
    """Every field run_closed's Predictor / make_sim / run_episode read, at the T5 values (amendment 2, D-30..D-33,
    tools/t5.ps1). smooth: 'on'/'off'; offset: 'off'/'row'/'batch' (t5.ps1's mapping)."""
    if offset not in ("off", "row", "batch"):
        raise SystemExit(f"--offset must be off, row or batch, not {offset!r}")
    on = offset != "off"
    return argparse.Namespace(
        sim="crowd", seed=seed, duration=float(duration), people=24, replan=0.4, refit=2.0, margin=margin, noise=0.1,
        box_q=box_q, hz=(1.0, 2.0), horizons=[1.0, 2.0], max_speed=1.0, vo_w=vo_w, vo_h=5.0,
        robot_repulsion=2.0, ignore_robot=0.2, goal_mode="cross",
        switch=True, switch_all=True, switch_to="erratic", switch_t=5.0,
        bank=str(BANK_DIR), bank_exclude_regime=BANK_EXCLUDE, frozen_ctx="fill", context=2000,
        tabpfn_fast=1, tabpfn_est=1, threads=3, latency=0.0,
        smooth=(smooth == "on"), sm_a_max=1.0, sm_a_brake=2.0, sm_w_max=90.0, sm_l_turn=0.3, sm_l_dv=0.2, sm_hyst=0.1,
        offset=on, offset_alpha=None, offset_eta=0.02, offset_rule=("batch" if offset == "batch" else "row"),
        offset_eta_b=0.05, offset_mode=offset, tag="", log=True)


def settings_line(args):
    return (f"smoothing {'on' if args.smooth else 'off'}, online offset "
            f"{args.offset_mode if args.offset else 'off'}"
            + (f" (alpha 0.2, eta {args.offset_eta:g} per row)" if args.offset and args.offset_rule == "row" else "")
            + (f" (alpha 0.2, eta_b {args.offset_eta_b:g} per 2 s batch)" if args.offset and args.offset_rule == "batch" else ""))


class in_tmp:
    """Run inside a fresh temp working directory (no stray files in the repo), then restore the cwd."""

    def __enter__(self):
        self.cwd, self.tmp = os.getcwd(), tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)
        return self.tmp.name

    def __exit__(self, *exc):
        os.chdir(self.cwd)
        self.tmp.cleanup()


def run_planner(kind, bank, args, seed):
    """One closed-loop episode with logging, trace writing disabled, in a temp cwd. Returns (res, logd, init, wall)."""
    import run_closed as RC
    from foresight import trace
    trace.write = lambda *a, **k: None          # no results/traces files
    with in_tmp():
        t0 = time.perf_counter()
        pred = RC.Predictor(kind, bank, C.Bank(bank) if kind.endswith("_live") else None, args)
        res, logd, _ = RC.run_episode(pred, seed, args, log=True)
        wall = time.perf_counter() - t0
    sim, _, _ = RC.make_sim(args.sim, seed, args)
    init = [dict(r, episode=seed) for r in sim.reset()]
    return res, logd, init, wall


def run(seed, bank, args):
    res, logd, init, wall = run_planner("tabpfn_split", bank, args, seed)
    print(f"episode: {len(logd['plans'])} plans, {wall:.0f} s", flush=True)
    return res, logd, init, wall, len(bank)


def track_df(init, frames):
    rows = [dict(r, t=0.0) for r in init]
    for fr in frames:
        rows += [dict(r, t=round(fr["t"], 6)) for r in fr["state"]]
    d = pd.DataFrame(rows)
    d["t"] = d["t"].round(6)
    return d


# ----------------------------------------------------------------------------------------------
# 2. the moment
# ----------------------------------------------------------------------------------------------
def gov_ctx(plans, i):
    """Context the model read at plan i: ctx of the latest plan <= i that refitted."""
    for j in range(i, -1, -1):
        if plans[j].get("ctx") is not None:
            return plans[j]["ctx"], plans[j]["t"]
    return None, None


def veto_sources(p_robot, pred, cand, margin, hz):
    """For each 'box' candidate: which people's boxes (per horizon) fall under the margin."""
    out = {}
    for i, r in cand.iterrows():
        if r.veto_reason != "box":
            continue
        cmd = np.array([r.vx, r.vy])
        hits = []
        for h in hz:
            rows = pred[np.isclose(pred.h, h)]
            g = PL.gap_to_box(p_robot + cmd * h, rows) - R_ROBOT - R_PERSON
            hits += [(int(a), h, float(gg)) for a, gg in zip(rows.agent_id, g) if gg < margin]
        out[i] = hits
    return out


def choose_moment(plans, tr, args):
    """Earliest replan t that satisfies all of:
      - t >= the switch time (the crowd is already erratic);
      - the scene model is active: the latest refit read >= SPLIT_MIN recent rows of this episode
        (otherwise the frozen model runs alone), and >= SPLIT_MIN of them were made after the switch;
      - >= 3 candidate moves vetoed for 'box', each confirmed by a predicted box (not the 0.4 s check);
      - a person P with a prediction row, filtered speed >= 0.5 m/s, centre within 3 m of the robot,
        whose own box vetoes at least one candidate, and who is still in the scene at t + 1 s and t + 2 s.
    P = the nearest such person."""
    t_end = tr.t.max()
    for i, pl in enumerate(plans):
        t = round(pl["t"], 6)
        if t < args.switch_t - 1e-9 or t + 2.0 > t_end + 1e-9 or not len(pl["pred"]):
            continue
        ctx, t_fit = gov_ctx(plans, i)
        if ctx is None or (ctx["source"] == "recent").sum() < SPLIT_MIN or pl.get("n_live", 0) < SPLIT_MIN:
            continue
        if (ctx.t >= args.switch_t - 1e-9).sum() < SPLIT_MIN:     # the scene model has seen the erratic crowd
            continue
        cand = pd.DataFrame(pl["candidates"])
        if (cand.veto_reason == "box").sum() < 3:
            continue
        rob = tr[(tr.t == t) & (tr.is_ego == 1)].iloc[0]
        p_robot = np.array([rob.x, rob.y])
        vs = veto_sources(p_robot, pl["pred"], cand, args.margin, args.hz)
        if sum(1 for h in vs.values() if h) < 3:
            continue
        blockers = {a for h in vs.values() for a, _, _ in h}
        now = pl["pred"].drop_duplicates("agent_id")
        later = set(tr[np.isclose(tr.t, t + 2.0)].agent_id)
        mid = set(tr[np.isclose(tr.t, t + 1.0)].agent_id)
        best = None
        for _, r in now.iterrows():
            d = float(np.hypot(r.px - p_robot[0], r.py - p_robot[1]))
            sp = float(np.hypot(r.vx, r.vy))
            if d <= 3.0 and sp >= 0.5 and int(r.agent_id) in blockers and r.agent_id in later and r.agent_id in mid:
                if best is None or d < best[1]:
                    best = (int(r.agent_id), d, sp)
        if best:
            return dict(i=i, t=t, P=best[0], dist=best[1], speed=best[2], t_fit=round(t_fit, 6), vs=vs)
    raise SystemExit("no moment satisfies the rule")


# ----------------------------------------------------------------------------------------------
# 3. drawing helpers
# ----------------------------------------------------------------------------------------------
def style(ax, lim=None):
    if lim is not None:
        ax.set_xlim(lim[0], lim[2]); ax.set_ylim(lim[1], lim[3])
    ax.set_aspect("equal"); ax.set_facecolor("white"); ax.grid(color=C_GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=8)
    ax.set_xlabel("x (m)", color=C_MUTED, fontsize=9); ax.set_ylabel("y (m)", color=C_MUTED, fontsize=9)


def draw_base(ax, tr, t, obst, goal, P=None, trails=True, people=True, unpred=()):
    for o in obst:
        ax.add_patch(Circle((o["x"], o["y"]), o["r"], color=C_OBST, zorder=1))
    if trails:
        past = tr[(tr.t <= t + 1e-6) & (tr.t >= t - 2.0 - 1e-6)]
        for aid, g in past.groupby("agent_id"):
            col = C_ROBOT if g.is_ego.iloc[0] else (C_PERSON if aid == P else C_MUTED)
            ax.plot(g.x, g.y, color=col, lw=1.6 if aid == P else 1.0, alpha=0.9, zorder=2)
    now = tr[tr.t == t]
    for _, r in now.iterrows():
        if r.is_ego:
            ax.add_patch(Circle((r.x, r.y), r.radius, color=C_ROBOT, zorder=6))
        elif people:
            if r.agent_id in unpred:
                ax.add_patch(Circle((r.x, r.y), r.radius, fill=False, ec=C_PERSON, lw=1.2, ls="--", zorder=5))
            else:
                ax.add_patch(Circle((r.x, r.y), r.radius, color=C_PERSON, alpha=1.0 if (P is None or r.agent_id == P) else 0.55,
                                    zorder=5))
            if r.agent_id == P:
                ax.add_patch(Circle((r.x, r.y), r.radius + 0.18, fill=False, ec=C_INK, lw=1.4, zorder=6))
    ax.plot(*goal, marker="*", color=C_ROBOT, ms=14, zorder=6, mec="white")


def draw_box(ax, r, lo_a, hi_a, lo_c, hi_c, **kw):
    cvx, cvy = r.px + r.vx * r.h, r.py + r.vy * r.h
    box = Rectangle((lo_a, lo_c), max(hi_a - lo_a, 0.02), max(hi_c - lo_c, 0.02), **kw)
    box.set_transform(Affine2D().rotate(r.theta).translate(cvx, cvy) + ax.transData)
    ax.add_patch(box)


def to_world1(r, a, c):
    cth, sth = np.cos(r.theta), np.sin(r.theta)
    return r.px + r.vx * r.h + cth * a - sth * c, r.py + r.vy * r.h + sth * a + cth * c


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def legend(ax, items, loc="upper left", ncol=1):
    lg = ax.legend(handles=items, loc=loc, fontsize=8, frameon=True, ncol=ncol)
    lg.get_frame().set_edgecolor(C_GRID)


# ----------------------------------------------------------------------------------------------
# 4. feature meanings
# ----------------------------------------------------------------------------------------------
def meaning(c):
    m = re.match(r"hist_([xy])(\d+)$", c)
    if m:
        return f"where P was {0.2 * int(m.group(2)):.1f} s ago, {'along' if m.group(1) == 'x' else 'across'} its heading, from now (m)"
    m = re.match(r"n(\d)_(x|y|vx|vy|dist)$", c)
    if m:
        k = ["nearest", "2nd nearest", "3rd nearest"][int(m.group(1))] if int(m.group(1)) < 3 else f"neighbour {m.group(1)}"
        return {"x": f"{k} person: offset along P's heading (m)", "y": f"{k} person: offset across (m, + = left)",
                "vx": f"{k} person: velocity relative to P, along (m/s)", "vy": f"{k} person: relative velocity, across (m/s)",
                "dist": f"{k} person: distance (m; 99 = nobody)"}[m.group(2)]
    return {
        "h_f": "horizon asked about (s)", "speed": "filtered speed (m/s)",
        "a_lon": "acceleration along heading (m/s²)", "a_lat": "acceleration sideways (m/s², + = left)",
        "turn_rate": "turning rate (rad/s, + = left)", "radius": "body radius (m)",
        "robot_x": "robot offset along P's heading (m)", "robot_y": "robot offset across (m, + = left)",
        "robot_dist": "distance to robot centre (m)", "robot_closing": "speed at which robot and P approach (m/s, + = closing)",
        "obst_x": "nearest pillar or wall surface, along (m)", "obst_y": "nearest pillar or wall surface, across (m)",
        "obst_dist": "distance to that surface (m)", "density_3m": "people within 3 m",
    }.get(c, "(no description yet: new feature)")


def group(c):
    if c in ("h_f", "speed", "a_lon", "a_lat", "turn_rate", "radius"):
        return "Own motion"
    if c.startswith("hist_"):
        return "Recent path"
    if re.match(r"n\d_", c):
        return "Neighbours"
    if c.startswith("robot_"):
        return "Robot"
    if c.startswith("obst_") or c == "density_3m":
        return "Surroundings"
    return "Other"


# ----------------------------------------------------------------------------------------------
# 5. HTML helpers
# ----------------------------------------------------------------------------------------------
def fmt(v, nd=2):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "–"
        if np.isinf(v):
            return "∞"
        return f"{v:.{nd}f}".replace("-", "−")
    return html.escape(str(v))


def table(df, left=(), nd=2, row_class=None, groups=None):
    head = "".join(f'<th class="{"l" if c in left else ""}">{html.escape(str(c))}</th>' for c in df.columns)
    body, last = [], None
    for i, (_, r) in enumerate(df.iterrows()):
        if groups is not None and groups[i] != last:
            body.append(f'<tr class="grp"><td class="l" colspan="{len(df.columns)}">{html.escape(groups[i])}</td></tr>')
            last = groups[i]
        cls = f' class="{row_class(r)}"' if row_class and row_class(r) else ""
        cells = "".join(f'<td class="{"l" if c in left else ""}">{fmt(r[c], nd)}</td>' for c in df.columns)
        body.append(f"<tr{cls}>{cells}</tr>")
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def fig_html(name, alt, caption):
    return (f'<figure><img src="walkthrough/{name}" alt="{html.escape(alt)}" loading="lazy">'
            f"<figcaption>{caption}</figcaption></figure>")


EXTRA_CSS = """
figure img { display: block; width: 100%; height: auto; border-radius: 6px; background: #fff }
.tbl { overflow-x: auto; margin-top: 12px; background: var(--surface); border: 1px solid var(--line); border-radius: 10px }
.tbl table { border-collapse: collapse; width: 100%; font: 400 13px/1.35 var(--body); font-variant-numeric: tabular-nums }
.tbl th, .tbl td { padding: 6px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--soft) }
.tbl th { font: 500 11px/1.2 var(--mono); color: var(--muted); letter-spacing: .03em; background: var(--soft) }
.tbl .l { text-align: left; white-space: normal; min-width: 7rem }
.tbl tr.grp td { font: 600 13px/1.2 var(--display); color: var(--fg); background: var(--bg); padding-top: 10px }
.tbl tr.hi td { background: var(--pred-soft) }
.tbl tr.chosen td { background: var(--pred-soft); font-weight: 600 }
.tbl tr:last-child td { border-bottom: 0 }
.tcap { font-size: 14px; color: var(--muted); margin: 8px 0 0; max-width: 86ch }
.facts { margin: 14px 0 0; padding-left: 1.2em; max-width: 86ch } .facts li { margin: 4px 0 }
code { font: 13px var(--mono); background: var(--soft); padding: 1px 4px; border-radius: 4px }
.stage { font: 500 12px/1 var(--mono); color: var(--pred); letter-spacing: .06em; text-transform: uppercase; margin-bottom: 6px }
"""


def explainer_head():
    src = (REPO / "docs" / "explainer.html").read_text(encoding="utf-8")
    links = "\n".join(re.findall(r'<link [^>]*fonts\.(?:googleapis|gstatic)[^>]*>', src))
    m = re.search(r"<style>.*?</style>", src, re.S)
    if not m:
        raise SystemExit("docs/explainer.html has no <style> block to copy")
    return links, m.group(0)


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=6001)
    ap.add_argument("--smooth", default=FINAL_SMOOTH, choices=["on", "off"], help="D-28 smoothing (T4 gate)")
    ap.add_argument("--offset", default=FINAL_OFFSET, choices=["off", "row", "batch"], help="online offset (T4 gate)")
    ap.add_argument("--threads", type=int, default=3, help="torch threads")
    ap.add_argument("--dev-cache", default=None, help="pickle (res, logd) here and reuse it (figure iteration)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    T0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    args = final_args(a.seed, smooth=a.smooth, offset=a.offset)
    seed = a.seed
    print(f"final configuration: tabpfn_split, box_q {args.box_q}, margin {args.margin}, replan {args.replan} s, "
          f"{settings_line(args)}; seed {seed}", flush=True)

    bank, bank_meta = load_bank(args.noise, args.hz)
    kq, kr = bank_meta["kalman_q"], bank_meta["kalman_r"]
    QS5 = qs5(args.box_q)
    PCT = pct(args.box_q)
    iq_lo, iq_hi = f"q{int(round(100 * args.box_q)):02d}", f"q{int(round(100 - 100 * args.box_q)):02d}"
    cache = Path(a.dev_cache) if a.dev_cache else None
    if cache and cache.exists():
        with open(cache, "rb") as f:
            res, logd, init, wall, n_bank = pickle.load(f)
        print("reused", cache)
    else:
        res, logd, init, wall, n_bank = run(seed, bank, args)
        if cache:
            with open(cache, "wb") as f:
                pickle.dump((res, logd, init, wall, n_bank), f)
    plans, frames, obst, bounds = logd["plans"], logd["frames"], logd["obst"], logd["bounds"]
    tr = track_df(init, frames)
    dt = 0.2

    mo = choose_moment(plans, tr, args)
    i, t, P = mo["i"], mo["t"], mo["P"]
    pl = plans[i]
    pred, q = pl["pred"].reset_index(drop=True), pl["q"].reset_index(drop=True)
    cand = pd.DataFrame(pl["candidates"])
    ctx, t_fit = gov_ctx(plans, i)
    rob = tr[(tr.t == t) & (tr.is_ego == 1)].iloc[0]
    p_robot, goal = np.array([rob.x, rob.y]), np.array([rob.goal_x, rob.goal_y])
    regime = tr[(tr.t == t) & (tr.agent_id == P)].regime.iloc[0]
    print(f"moment: t={t:.1f} s, P={P}, dist {mo['dist']:.2f} m, speed {mo['speed']:.2f} m/s, context fit at {t_fit:.1f} s")
    now_t = tr[tr.t == t]
    pred_ids = set(pred.agent_id)
    unpred = [int(x) for x in now_t[(now_t.is_ego == 0) & ~now_t.agent_id.isin(pred_ids)].agent_id]

    def truth_at(tt, ids):
        d = tr[np.isclose(tr.t, tt) & tr.agent_id.isin(ids)]
        return d.set_index("agent_id")[["x", "y"]]

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=no", "--", "foresight", "run_closed.py", "run_offline.py", "data"],
                                cwd=REPO, capture_output=True, text=True).stdout.strip())

    # ---------------- stage 1: scene ----------------
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    style(ax, bounds)
    draw_base(ax, tr, t, obst, goal, P=P, unpred=unpred)
    ax.annotate(f"P (id {P})", xy=(tr[(tr.t == t) & (tr.agent_id == P)].x.iloc[0], tr[(tr.t == t) & (tr.agent_id == P)].y.iloc[0]),
                xytext=(12, 12), textcoords="offset points", color=C_INK, fontsize=9, fontweight="bold")
    legend(ax, [Line2D([], [], marker="o", ls="", color=C_PERSON, ms=8, label="Person (true position)"),
                Line2D([], [], marker="o", ls="", mfc="none", mec=C_PERSON, ms=8, label="Person, no prediction yet"),
                Line2D([], [], color=C_MUTED, lw=1.2, label="Path, last 2 s"),
                Line2D([], [], marker="o", ls="", color=C_ROBOT, ms=9, label="Robot"),
                Line2D([], [], marker="*", ls="", color=C_ROBOT, ms=11, label="Robot goal"),
                Patch(facecolor=C_OBST, label="Pillar")], loc="lower left")
    ax.set_title(f"t = {t:.1f} s", fontsize=10, color=C_INK, loc="left")
    save(fig, "s1_scene.png")

    # ---------------- stage 2: sensing ----------------
    s_t = round(t / dt)
    g = tr[tr.agent_id == P].copy()
    steps = np.round(g.t.to_numpy() / dt).astype(int)
    S = s_t + 1
    Pt = np.full((S, 2), np.nan)
    keep = steps <= s_t
    Pt[steps[keep]] = g[["x", "y"]].to_numpy()[keep]
    noisy = Pt + np.random.default_rng([seed, int(P)]).normal(0, args.noise, (S, 2))
    run_meta = F.load_kalman(args.bank, {}) if hasattr(F, "load_kalman") else {}
    rkq, rkr = F.kalman_params(run_meta, args.noise) if hasattr(F, "kalman_params") else (2.0, max(args.noise, 0.05))
    Ps, Vs = F._kalman_cv(noisy, dt, q=rkq, r=rkr) if hasattr(F, "kalman_params") else F._kalman_cv(noisy, dt, r=max(args.noise, 0.05))
    if (rkq, rkr) != (kq, kr):
        raise SystemExit(f"Kalman settings changed during the run: bank ({kq}, {kr}) vs episode ({rkq}, {rkr}); rerun")
    qP = q[q.agent_id == P].sort_values("h")
    q1 = qP.iloc[0]
    filt = np.array([Ps[s_t, 0], Ps[s_t, 1], Vs[s_t, 0], Vs[s_t, 1]])
    sense_ok = np.allclose(filt, [q1.px, q1.py, q1.vx, q1.vy], atol=1e-9)
    print("stage 2 reconstruction matches the query row:", sense_ok)
    w0 = max(0, s_t - 10)
    sl = slice(w0, s_t + 1)
    tt = np.arange(S)[sl] * dt
    pd.DataFrame({"t": tt, "true_x": Pt[sl, 0], "true_y": Pt[sl, 1], "noisy_x": noisy[sl, 0], "noisy_y": noisy[sl, 1],
                  "filt_x": Ps[sl, 0], "filt_y": Ps[sl, 1], "filt_vx": Vs[sl, 0], "filt_vy": Vs[sl, 1]}).round(4) \
        .to_csv(OUT / "s2_sensing.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.plot(Pt[sl, 0], Pt[sl, 1], color=C_MUTED, lw=1.2, ls="--", label="True path (simulator)")
    ax.plot(noisy[sl, 0], noisy[sl, 1], "o", color=C_PERSON, ms=4, alpha=0.55, label="Noisy observations (σ = 0.1 m)")
    ax.plot(Ps[sl, 0], Ps[sl, 1], "-", color=C_PERSON, lw=2.0, label="Kalman-filtered track")
    ax.plot(Ps[s_t, 0], Ps[s_t, 1], "o", color=C_PERSON, ms=9, mec="white", mew=1.5, zorder=5)
    ax.annotate("", xy=(Ps[s_t, 0] + Vs[s_t, 0], Ps[s_t, 1] + Vs[s_t, 1]), xytext=(Ps[s_t, 0], Ps[s_t, 1]),
                arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=1.4))
    ax.text(Ps[s_t, 0] + Vs[s_t, 0], Ps[s_t, 1] + Vs[s_t, 1], "  velocity × 1 s", fontsize=8, color=C_INK, va="center")
    xs_all = np.concatenate([Pt[sl, 0], noisy[sl, 0], [Ps[s_t, 0] + Vs[s_t, 0]]])
    ys_all = np.concatenate([Pt[sl, 1], noisy[sl, 1], [Ps[s_t, 1] + Vs[s_t, 1]]])
    cx, cy = np.nanmean([np.nanmin(xs_all), np.nanmax(xs_all)]), np.nanmean([np.nanmin(ys_all), np.nanmax(ys_all)])
    half = max(np.nanmax(xs_all) - np.nanmin(xs_all), np.nanmax(ys_all) - np.nanmin(ys_all)) / 2 + 0.5
    style(ax, (cx - half, cy - half, cx + half, cy + half))
    legend(ax, ax.get_legend_handles_labels()[0], loc="best")
    ax.set_title(f"Person {P}, t = {t - (s_t - w0) * dt:.1f} to {t:.1f} s", fontsize=10, color=C_INK, loc="left")
    save(fig, "s2_sensing.png")
    sense_tab = pd.DataFrame({"quantity": ["x (m)", "y (m)", "vx (m/s)", "vy (m/s)"], "filtered": filt,
                              "true (simulator)": [g[g.t == t].x.iloc[0], g[g.t == t].y.iloc[0], g[g.t == t].vx.iloc[0], g[g.t == t].vy.iloc[0]]})

    # ---------------- stage 3: the table row ----------------
    cols = ["h_f"] + F.feature_names()[1:]
    q2 = qP.iloc[1] if len(qP) > 1 else qP.iloc[0]
    ft = pd.DataFrame({"feature": cols, "group": [group(c) for c in cols],
                       "h = 1 s": [float(q1[c]) for c in cols], "h = 2 s": [float(q2[c]) for c in cols],
                       "meaning": [meaning(c) for c in cols]})
    order = ["Own motion", "Recent path", "Neighbours", "Robot", "Surroundings", "Other"]
    ft["_o"] = ft.group.map({k: j for j, k in enumerate(order)})
    ft = ft.sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)
    ft.to_csv(OUT / "s3_features.csv", index=False)

    # ---------------- stage 4: the examples (frozen bank rows + scene rows) ----------------
    frz = plans[0]["ctx"]
    if frz is None or len(frz) != args.context or (frz["source"] == "recent").any():
        raise SystemExit("plans[0] does not hold the frozen context")
    frz = frz.assign(source="frozen")
    scn = ctx.assign(source="recent")
    both = pd.concat([frz, scn], ignore_index=True)
    counts = {"frozen": int(len(frz)), "recent": int(len(scn))}

    def mode_share(d):
        return {k: round(float(v), 4) for k, v in d["mode"].value_counts(normalize=True).items()}

    s4 = {"refit_t": t_fit, "frozen_rows": counts["frozen"], "scene_rows": counts["recent"], "scene_rows_cap": 250,
          "scene_min_rows": SPLIT_MIN, "n_live_at_t": int(pl.get("n_live", 0)),
          "scene_rows_h1": int(np.isclose(scn.h_f, 1.0).sum()), "scene_rows_h2": int(np.isclose(scn.h_f, 2.0).sum()),
          "scene_rows_after_switch": int((scn.t >= args.switch_t - 1e-9).sum()),
          "scene_t_range": [float(scn.t.min()), float(scn.t.max())],
          "scene_t_known_range": [float(scn.t_known.min()), float(scn.t_known.max())],
          "frozen_episodes": int(frz.episode.nunique()), "bank_rows": n_bank, "bank_all_rows": bank_meta["rows_all"],
          "bank_episodes": int(bank.episode.nunique()), "bank_all_episodes": bank_meta["episodes_all"],
          "mode_share_frozen": mode_share(frz), "mode_share_scene": mode_share(scn), "mode_share_bank": mode_share(bank),
          "recent_episode": seed}
    json.dump(s4, open(OUT / "s4_context_counts.json", "w"), indent=1)
    ex = pd.concat([frz.head(6), scn.sort_values("t_known").tail(6)])
    exs = pd.DataFrame({"model": ex.source.map({"frozen": "frozen (bank)", "recent": "scene (recent)"}),
                        "from": [f"this run, t = {r.t:.1f} s" if r.source == "recent" else f"bank episode {int(r.episode)}, t = {r.t:.1f} s"
                                 for _, r in ex.iterrows()],
                        "h (s)": ex.h_f.to_numpy(), "speed": ex.speed.to_numpy(), "n0_dist": ex.n0_dist.to_numpy(),
                        "robot_dist": ex.robot_dist.to_numpy(), "robot_closing": ex.robot_closing.to_numpy(),
                        "mode": ex["mode"].to_numpy(), "ddx (along)": ex.ddx.to_numpy(), "ddy (across)": ex.ddy.to_numpy()})
    exs.to_csv(OUT / "s4_context_excerpt.csv", index=False)

    # refit both models on the same rows: all five quantiles for everyone at t, per model and averaged
    with in_tmp():
        kw = dict(fast=args.tabpfn_fast, n_estimators=args.tabpfn_est, fit_mode="fit_with_cache")
        mf = M.make("tabpfn", **kw).fit(F.X_of(frz), frz[["ddx", "ddy"]].to_numpy())
        ms = M.make("tabpfn", **kw).fit(F.X_of(scn), scn[["ddx", "ddy"]].to_numpy())
        Qf = mf.predict_quantiles(F.X_of(pred), QS5)          # [n, 2, 5]
        Qs = ms.predict_quantiles(F.X_of(pred), QS5)
    Q = np.sort((Qf + Qs) / 2, axis=2)                          # run_closed.AvgQuantiles
    raw_match = float(np.max(np.abs(np.stack([Q[:, 0, 0] - pred.q05a, Q[:, 0, 4] - pred.q95a,
                                              Q[:, 1, 0] - pred.q05c, Q[:, 1, 4] - pred.q95c]))))
    # the served box is [box_q, 1 - box_q] widened by the online offset d in force (CF.widen); recover d
    d_off = float(np.median(((pred.hi_a - pred.lo_a) - (Q[:, 0, 3] - Q[:, 0, 1])) / 2)) if args.offset else 0.0
    lo_w, hi_w = CF.widen(np.stack([Q[:, 0, 1], Q[:, 1, 1]], 1), np.stack([Q[:, 0, 3], Q[:, 1, 3]], 1), d_off)
    match = float(np.max(np.abs(np.stack([lo_w[:, 0] - pred.lo_a, hi_w[:, 0] - pred.hi_a,
                                          lo_w[:, 1] - pred.lo_c, hi_w[:, 1] - pred.hi_c]))))
    print(f"refit vs logged 5-95% box: max |diff| = {raw_match:.2e} m; served box (offset d = {d_off:+.3f} m): {match:.2e} m")
    iP = pred.index[pred.agent_id == P].tolist()
    iP = sorted(iP, key=lambda j: pred.h[j])

    def qbox(ax, Qm, j, k_lo, k_hi, **kw):
        ax.add_patch(Rectangle((Qm[j, 0, k_lo], Qm[j, 1, k_lo]), Qm[j, 0, k_hi] - Qm[j, 0, k_lo],
                               Qm[j, 1, k_hi] - Qm[j, 1, k_lo], **kw))

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.4))
    lim_a = np.nanpercentile(np.abs(both.ddx), 99) * 1.1
    lim_c = np.nanpercentile(np.abs(both.ddy), 99) * 1.1
    for ax, h, j in zip(axs, (1.0, 2.0), iP):
        d = both[np.isclose(both.h_f, h)]
        for src, s_, al in (("frozen", 6, 0.35), ("recent", 14, 0.8)):
            e = d[d.source == src]
            ax.scatter(e.ddx, e.ddy, s=s_, color=SRC_COLOR[src], alpha=al, lw=0, label=f"{SRC_LABEL[src]} ({len(e)})")
        qbox(ax, Qf, j, 1, 3, fill=False, ec=C_INK, lw=1.2, ls="--", zorder=5)
        qbox(ax, Qs, j, 1, 3, fill=False, ec=C_PERSON, lw=1.6, zorder=5)
        qbox(ax, Q, j, 1, 3, fill=False, ec=C_PRED, lw=2.2, zorder=6)
        ax.plot(Qf[j, 0, 2], Qf[j, 1, 2], marker="D", color=C_INK, ms=6, mec="white", zorder=7, ls="",
                label=f"P, frozen model: median, {PCT} box (dashed)")
        ax.plot(Qs[j, 0, 2], Qs[j, 1, 2], marker="D", color=C_PERSON, ms=6, mec="white", zorder=7, ls="",
                label=f"P, scene model: median, {PCT} box")
        ax.plot(Q[j, 0, 2], Q[j, 1, 2], marker="*", color=C_PRED, ms=15, mec="white", zorder=8, ls="",
                label="P, average of the two (planner's box before offset)")
        ax.axhline(0, color=C_MUTED, lw=0.6); ax.axvline(0, color=C_MUTED, lw=0.6)
        ax.set_xlim(-lim_a, lim_a); ax.set_ylim(-lim_c, lim_c)
        ax.set_facecolor("white"); ax.grid(color=C_GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        ax.set_xlabel("ddx: along heading, true − constant velocity (m)", color=C_MUTED, fontsize=9)
        ax.set_ylabel("ddy: across, + = left (m)", color=C_MUTED, fontsize=9)
        n_out = int(((d.ddx.abs() > lim_a) | (d.ddy.abs() > lim_c)).sum())
        ax.set_title(f"+{h:g} s examples ({len(d)} rows, {n_out} outside view)", fontsize=10, color=C_INK, loc="left")
        legend(ax, ax.get_legend_handles_labels()[0], loc="upper left")
    save(fig, "s4_context.png")

    # ---------------- stage 5: P's output: frozen model, scene model, average ----------------
    qrows = []
    MODELS = (("frozen model", Qf), ("scene model", Qs), ("average (final)", Q))
    for j in iP:
        r = pred.loc[j]
        tru = truth_at(t + r.h, [P]).loc[P]
        R = np.array([[np.cos(r.theta), np.sin(r.theta)], [-np.sin(r.theta), np.cos(r.theta)]])
        act = R @ (np.array([tru.x, tru.y]) - np.array([r.px, r.py]) - np.array([r.vx, r.vy]) * r.h)
        for ax_i, name in ((0, "along (ddx)"), (1, "across (ddy)")):
            for mname, Qm in MODELS:
                qrows.append({"horizon": f"+{r.h:g} s", "axis": name, "model": mname,
                              **{f"q{int(round(100 * qq)):02d}": Qm[j, ax_i, k] for k, qq in enumerate(QS5)},
                              "actual": act[ax_i], f"{PCT} width": Qm[j, ax_i, 3] - Qm[j, ax_i, 1]})
            if args.offset:
                lo_s, hi_s = (r.lo_a, r.hi_a) if ax_i == 0 else (r.lo_c, r.hi_c)
                qrows.append({"horizon": f"+{r.h:g} s", "axis": name, "model": f"served (offset {d_off:+.2f} m)",
                              **{f"q{int(round(100 * qq)):02d}": np.nan for qq in QS5}, iq_lo: lo_s, iq_hi: hi_s,
                              "actual": act[ax_i], f"{PCT} width": hi_s - lo_s})
    qtab = pd.DataFrame(qrows)
    qtab.to_csv(OUT / "s5_quantiles.csv", index=False)
    errP = {}
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.8))
    xy = [[pred.loc[iP[0]].px, pred.loc[iP[0]].py], list(p_robot)]
    for j in iP:
        r = pred.loc[j]
        for Qm in (Qf, Qs, Q):
            for a_ in (0, 4):
                for c_ in (0, 4):
                    xy.append(list(to_world1(r, Qm[j, 0, a_], Qm[j, 1, c_])))
        tru = truth_at(t + r.h, [P]).loc[P]; xy.append([tru.x, tru.y])
    xy = np.array(xy)
    c5 = (xy.min(0) + xy.max(0)) / 2
    h5 = max(2.5, (xy.max(0) - xy.min(0)).max() / 2 + 1.0)
    win5 = (c5[0] - h5, c5[1] - h5, c5[0] + h5, c5[1] + h5)
    for ax, j in zip(axs, iP):
        r = pred.loc[j]
        tru = truth_at(t + r.h, [P]).loc[P]
        cvp = (r.px + r.vx * r.h, r.py + r.vy * r.h)
        for key, Qm in (("frozen", Qf), ("scene", Qs), ("final", Q)):
            mm = to_world1(r, Qm[j, 0, 2], Qm[j, 1, 2])
            errP[(key, r.h)] = float(np.hypot(mm[0] - tru.x, mm[1] - tru.y))
        med = to_world1(r, Q[j, 0, 2], Q[j, 1, 2])
        errP[r.h] = errP[("final", r.h)]
        errP[("cv", r.h)] = float(np.hypot(cvp[0] - tru.x, cvp[1] - tru.y))
        style(ax, win5)
        draw_base(ax, tr, t, obst, goal, P=P, unpred=unpred)
        draw_box(ax, r, Qf[j, 0, 1], Qf[j, 0, 3], Qf[j, 1, 1], Qf[j, 1, 3], fill=False, ec=C_INK, lw=1.2, ls="--", zorder=4)
        draw_box(ax, r, Qs[j, 0, 1], Qs[j, 0, 3], Qs[j, 1, 1], Qs[j, 1, 3], fill=False, ec=C_PERSON, lw=1.6, zorder=4)
        draw_box(ax, r, Q[j, 0, 1], Q[j, 0, 3], Q[j, 1, 1], Q[j, 1, 3], fc=C_PRED, alpha=0.18, ec="none", zorder=4)
        draw_box(ax, r, Q[j, 0, 1], Q[j, 0, 3], Q[j, 1, 1], Q[j, 1, 3], fill=False, ec=C_PRED, lw=2.0, zorder=4)
        if args.offset:
            draw_box(ax, r, r.lo_a, r.hi_a, r.lo_c, r.hi_c, fill=False, ec=C_PRED, lw=1.0, ls=":", zorder=4)
        for Qm, col in ((Qf, C_INK), (Qs, C_PERSON)):
            ax.plot(*to_world1(r, Qm[j, 0, 2], Qm[j, 1, 2]), "D", color=col, ms=5, mec="white", zorder=5)
        ax.plot([r.px, cvp[0]], [r.py, cvp[1]], color=C_MUTED, lw=1, ls=":", zorder=4)
        ax.plot(*cvp, "o", mfc="white", mec=C_MUTED, ms=6, zorder=5)
        ax.plot([r.px, med[0]], [r.py, med[1]], color=C_PRED, lw=1.2, zorder=4)
        ax.plot(*med, "*", color=C_PRED, ms=12, mec="white", zorder=6)
        path = tr[(tr.agent_id == P) & (tr.t >= t - 1e-6) & (tr.t <= t + r.h + 1e-6)]
        ax.plot(path.x, path.y, color=C_INK, lw=1, alpha=0.6, zorder=6)
        ax.plot(tru.x, tru.y, "x", color=C_INK, ms=10, mew=2, zorder=7)
        ax.set_title(f"+{r.h:g} s: average median off by {errP[r.h]:.2f} m", fontsize=10, color=C_INK, loc="left")
        items = [Patch(facecolor="none", edgecolor=C_INK, ls="--", label=f"frozen model, {PCT}"),
                 Patch(facecolor="none", edgecolor=C_PERSON, label=f"scene model, {PCT}"),
                 Patch(facecolor=C_PRED, alpha=0.3, edgecolor=C_PRED, label=f"average of the two, {PCT}")]
        if args.offset:
            items.append(Patch(facecolor="none", edgecolor=C_PRED, ls=":", label=f"served: average widened by offset ({d_off:+.2f} m)"))
        items += [Line2D([], [], marker="o", ls="", mfc="white", mec=C_MUTED, label="Constant velocity"),
                  Line2D([], [], marker="*", ls="", color=C_PRED, ms=10, label="Average median (diamonds: each model)"),
                  Line2D([], [], marker="x", ls="-", color=C_INK, mew=2, lw=1, label=f"Actual path / position at t+{r.h:g} s")]
        lg = ax.legend(handles=items, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8, frameon=False)
        lg.get_frame().set_edgecolor(C_GRID)                 # below the plaza, so it never hides P or the truth
    save(fig, "s5_person_box.png")

    # ---------------- stage 6: all boxes ----------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 6.2))
    for ax, h in zip(axs, (1.0, 2.0)):
        style(ax, bounds)
        draw_base(ax, tr, t, obst, goal, P=P, trails=False, unpred=unpred)
        rows = pred[np.isclose(pred.h, h)]
        tru = truth_at(t + h, rows.agent_id)
        for _, r in rows.iterrows():
            draw_box(ax, r, r.lo_a, r.hi_a, r.lo_c, r.hi_c, fc=C_PRED, alpha=0.15, ec="none", zorder=3)
            draw_box(ax, r, r.lo_a, r.hi_a, r.lo_c, r.hi_c, fill=False, ec=C_PRED, lw=2.0 if r.agent_id == P else 1.1, zorder=4)
            ax.plot([r.px, r.mx], [r.py, r.my], color=C_PRED, lw=0.8, zorder=4)
            if r.agent_id in tru.index:
                ax.plot(tru.loc[r.agent_id].x, tru.loc[r.agent_id].y, "x", color=C_INK, ms=6, mew=1.4, zorder=7)
        ax.set_title(f"+{h:g} s: {len(rows)} people predicted", fontsize=10, color=C_INK, loc="left")
    legend(axs[0], [Line2D([], [], marker="o", ls="", color=C_PERSON, ms=8, label="Person at t"),
                    Line2D([], [], marker="o", ls="", mfc="none", mec=C_PERSON, ms=8, label="Person, no prediction (< 1 s seen)"),
                    Patch(facecolor=C_PRED, alpha=0.3, edgecolor=C_PRED, label=f"{PCT} box" + (" + offset" if args.offset else "")),
                    Line2D([], [], marker="x", ls="", color=C_INK, mew=1.6, label="Actual position at t+h"),
                    Line2D([], [], marker="o", ls="", color=C_ROBOT, ms=9, label="Robot")], loc="lower left")
    save(fig, "s6_all_boxes.png")

    # ---------------- stage 7: occupancy ----------------
    occ = {}
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 5.8))
    fig.subplots_adjust(wspace=0.38)
    for ax, h in zip(axs, (1.0, 2.0)):
        G, xs, ys = O.occupancy(pred, h, bounds, box_q=args.box_q)
        rows = pred[np.isclose(pred.h, h)]
        tru = truth_at(t + h, rows.agent_id)
        T = O.truth_grid(tru[["x", "y"]].to_numpy(), xs, ys, xs[1] - xs[0])
        brier, auc = O.scores(G, T)
        occ[f"{h:g}"] = {"brier": brier, "auc": auc, "brier_all_zero": float(T.mean()), "cells": int(T.size),
                         "true_cells": int(T.sum()), "expected_people": float(G.sum()), "people_predicted": int(len(rows)),
                         "p_at_truth_mean": float(G[T == 1].mean()) if T.sum() else np.nan,
                         "p_max": float(G.max())}
        im = ax.imshow(G, origin="lower", extent=bounds and (bounds[0], bounds[2], bounds[1], bounds[3]), cmap="Oranges",
                       vmin=0, vmax=max(0.05, float(G.max())), interpolation="nearest", zorder=1)
        for o in obst:
            ax.add_patch(Circle((o["x"], o["y"]), o["r"], fill=False, ec=C_MUTED, lw=1, zorder=2))
        ax.add_patch(Circle(tuple(p_robot), R_ROBOT, color=C_ROBOT, zorder=5))
        ax.plot(tru.x, tru.y, "x", color=C_INK, ms=6, mew=1.4, zorder=6)
        style(ax, bounds); ax.grid(False)
        ax.set_title(f"+{h:g} s   Brier {brier:.5f}   AUC {auc:.3f}", fontsize=10, color=C_INK, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("P(cell holds a person's centre)", fontsize=8, color=C_MUTED); cb.ax.tick_params(labelsize=7, colors=C_MUTED)
    save(fig, "s7_occupancy.png")
    json.dump(occ, open(OUT / "s7_occupancy_scores.json", "w"), indent=1)

    # ---------------- stage 8: planner ----------------
    base = np.arctan2(*(goal - p_robot)[::-1])
    cand["speed"] = np.hypot(cand.vx, cand.vy)
    cand["direction"] = [np.nan if s < 1e-9 else (np.degrees(np.arctan2(vy, vx) - base) + 180) % 360 - 180
                         for vx, vy, s in zip(cand.vx, cand.vy, cand.speed)]
    cand["blocked by"] = [", ".join(f"{a}@+{h:g}s" for a, h in sorted({(a, h) for a, h, _ in mo["vs"].get(k, [])})) if k in mo["vs"] else ""
                          for k in cand.index]
    ctab = cand[["speed", "direction", "worst_gap", "progress", "veto_reason", "blocked by", "chosen"]].rename(
        columns={"speed": "speed (m/s)", "direction": "direction vs goal (°)", "worst_gap": "worst gap (m)",
                 "progress": "progress in 2 s (m)", "veto_reason": "reason"})
    ctab.to_csv(OUT / "s8_candidates.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    span = 3.6
    style(ax, (p_robot[0] - span, p_robot[1] - span, p_robot[0] + span, p_robot[1] + span))
    draw_base(ax, tr, t, obst, goal, P=P, trails=False, unpred=unpred)
    for _, r in pred[np.isclose(pred.h, 2.0)].iterrows():
        draw_box(ax, r, r.lo_a, r.hi_a, r.lo_c, r.hi_c, fill=False, ec=C_PRED, lw=1.6 if r.agent_id == P else 0.9,
                 alpha=0.9, zorder=3)
    for _, r in pred[np.isclose(pred.h, 1.0)].iterrows():
        draw_box(ax, r, r.lo_a, r.hi_a, r.lo_c, r.hi_c, fill=False, ec=C_PRED, lw=0.8, ls=":", zorder=3)
    for k, r in cand.iterrows():
        e = p_robot + np.array([r.vx, r.vy]) * 2.0
        col = REASON_COLOR.get(r.veto_reason, C_MUTED)
        ax.plot([p_robot[0], e[0]], [p_robot[1], e[1]], color=col, lw=3.4 if r.chosen else 1.2, alpha=1 if r.chosen else 0.85,
                zorder=8 if r.chosen else 7, solid_capstyle="round")
        ax.plot(*e, "o", color=col, ms=7 if r.chosen else 4, zorder=8, mec=C_INK if r.chosen else col, mew=1.5 if r.chosen else 0)
    st = cand.iloc[0]
    ax.text(p_robot[0], p_robot[1] + 0.62, f"stop: {'allowed' if st.veto_reason == 'none' else 'vetoed (' + st.veto_reason + ')'}",
            ha="center", va="bottom", fontsize=8, color=REASON_COLOR.get(st.veto_reason, C_MUTED), fontweight="bold", zorder=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    if cand.chosen.any() and cand[cand.chosen].speed.iloc[0] < 1e-9:
        ax.add_patch(Circle(tuple(p_robot), 0.55, fill=False, ec=C_INK, lw=2.0, zorder=9))
    reasons = list(dict.fromkeys(cand.veto_reason))
    legend(ax, [Line2D([], [], color=REASON_COLOR.get(rs, C_MUTED), lw=2,
                       label=f"{'allowed' if rs == 'none' else 'vetoed: ' + rs} ({int((cand.veto_reason == rs).sum())})") for rs in reasons]
           + [Line2D([], [], color=C_INK, lw=3.4, label="chosen (thick line)"),
              Patch(facecolor="none", edgecolor=C_PRED, label=f"{PCT} box at +2 s (dotted: +1 s)" + (", with offset" if args.offset else ""))], loc="lower left")
    ax.set_title(f"{len(cand)} candidate moves, end points at +2 s", fontsize=10, color=C_INK, loc="left")
    save(fig, "s8_planner.png")

    # ---------------- stage 9: ego trajectory ----------------
    ego = tr[tr.is_ego == 1].sort_values("t")
    sp = np.hypot(ego.vx, ego.vy).to_numpy()
    pts = ego[["x", "y"]].to_numpy()
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    style(ax, bounds)
    for o in obst:
        ax.add_patch(Circle((o["x"], o["y"]), o["r"], color=C_OBST, zorder=1))
    seg = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap="viridis", norm=plt.Normalize(0, args.max_speed), lw=3.2, zorder=3)
    lc.set_array(sp[1:])
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("robot speed (m/s)", fontsize=8, color=C_MUTED); cb.ax.tick_params(labelsize=7, colors=C_MUTED)
    ax.plot(*pts[0], "o", color=C_ROBOT, ms=8, zorder=5); ax.text(pts[0][0], pts[0][1] - 0.7, "start", fontsize=8, ha="center", color=C_ROBOT)
    ax.plot(*goal, marker="*", color=C_ROBOT, ms=15, mec="white", zorder=5)
    ax.add_patch(Circle(tuple(p_robot), 0.55, fill=False, ec=C_PRED, lw=2.2, zorder=6))
    ax.annotate(f"t = {t:.1f} s (this walkthrough)", xy=tuple(p_robot), xytext=(-60, 45), textcoords="offset points", ha="right",
                fontsize=8.5, color=C_PRED, fontweight="bold", arrowprops=dict(arrowstyle="-", color=C_PRED, lw=1))
    ax.set_title(f"Robot path, seed {seed}", fontsize=10, color=C_INK, loc="left")
    save(fig, "s9_ego_path.png")
    resj = {k: (float(v) if isinstance(v, (float, np.floating)) else (bool(v) if isinstance(v, (bool, np.bool_)) else v))
            for k, v in res.items()}
    json.dump(resj, open(OUT / "s9_metrics.json", "w"), indent=1, default=str)

    # ---------------- episode-wide numbers (for the "what stands out" list) ----------------
    reason_tot, cover, err, width, n_ctx_src = {}, {1.0: [], 2.0: []}, {1.0: [], 2.0: []}, {1.0: [], 2.0: []}, []
    cverr = {1.0: [], 2.0: []}
    for k, p_ in enumerate(plans):
        for rs, n in pd.Series(p_["candidates"]["veto_reason"]).value_counts().items():
            reason_tot[rs] = reason_tot.get(rs, 0) + int(n)
        if not len(p_["pred"]):
            continue
        for h in (1.0, 2.0):
            rows = p_["pred"][np.isclose(p_["pred"].h, h)]
            tru = truth_at(round(p_["t"] + h, 6), rows.agent_id)
            rows = rows[rows.agent_id.isin(tru.index)]
            if not len(rows):
                continue
            tx, ty = tru.loc[rows.agent_id].x.to_numpy(), tru.loc[rows.agent_id].y.to_numpy()
            c_, s_ = np.cos(rows.theta.to_numpy()), np.sin(rows.theta.to_numpy())
            dx = tx - rows.px.to_numpy() - rows.vx.to_numpy() * h; dy = ty - rows.py.to_numpy() - rows.vy.to_numpy() * h
            aa, bb = c_ * dx + s_ * dy, -s_ * dx + c_ * dy
            cover[h] += list((aa >= rows.lo_a) & (aa <= rows.hi_a) & (bb >= rows.lo_c) & (bb <= rows.hi_c))
            err[h] += list(np.hypot(rows.mx.to_numpy() - tx, rows.my.to_numpy() - ty))
            cverr[h] += list(np.hypot(dx, dy))
            width[h] += list(np.sqrt((rows.hi_a - rows.lo_a) * (rows.hi_c - rows.lo_c)))
        if p_.get("ctx") is not None and p_["t"] > 0:
            n_ctx_src.append(p_["ctx"]["source"].value_counts().to_dict() | {"t": p_["t"]})
    ep = {"veto_reasons_all_plans": reason_tot, "plans": len(plans),
          "fallback_rate": float(res["fallback_rate"]),
          **{f"cover_box_{h:g}s": float(np.mean(cover[h])) for h in cover if cover[h]},
          **{f"median_err_{h:g}s": float(np.median(err[h])) for h in err if err[h]},
          **{f"median_cv_err_{h:g}s": float(np.median(cverr[h])) for h in cverr if cverr[h]},
          **{f"box_side_geo_{h:g}s": float(np.median(width[h])) for h in width if width[h]},
          "context_by_refit": n_ctx_src}
    json.dump(ep, open(OUT / "episode_stats.json", "w"), indent=1, default=float)

    meta = {"features_version": F.FEATURES_VERSION, "seed": seed, "planner": "tabpfn_split", "t": t, "P": P, "smooth": bool(args.smooth),
            "offset": args.offset_mode if args.offset else "off", "offset_d_at_t": d_off, "refit_vs_logged_5_95_max_diff_m": raw_match,
            "switch_t": args.switch_t, "switch_to": args.switch_to, "robot_repulsion": args.robot_repulsion, "ignore_robot": args.ignore_robot,
            "replan": args.replan, "max_speed": args.max_speed, "frozen_ctx": args.frozen_ctx, "bank_exclude": args.bank_exclude_regime,
            "bank_all_episodes": bank_meta["episodes_all"], "s4": s4, "n_live_at_t": int(pl.get("n_live", 0)),
            "P_dist_m": mo["dist"], "P_speed": mo["speed"], "context_refit_t": t_fit, "commit": commit, "dirty": dirty,
            "kalman_q": rkq, "kalman_r": rkr, "bank_kalman_q": kq, "bank_kalman_r": kr, "box_q": args.box_q, "margin": args.margin, "context": args.context,
            "refit": args.refit, "goal_mode": args.goal_mode, "people": args.people, "duration": args.duration,
            "switch": args.switch, "regime_P_at_t": regime, "bank_rows": n_bank, "bank_all_rows": bank_meta["rows_all"],
            "unpredicted_ids": unpred, "sense_reconstruction_ok": bool(sense_ok), "refit_vs_planner_max_diff_m": match,
            "bank_episodes": int(bank.episode.nunique()),
            "refit_queries": int(len(next(p_["q"] for p_ in plans if abs(p_["t"] - t_fit) < 1e-6))),
            "episode_wall_s": wall, "generated": time.strftime("%Y-%m-%d %H:%M"), "timing_at_t": pl.get("timing")}
    json.dump(meta, open(OUT / "meta.json", "w"), indent=1, default=str)

    # ---------------- page ----------------
    write_page(meta, mo, res, ep, occ, counts, ctx, sense_tab, ft, exs, qtab, ctab, cand, errP, pred, pl, args, unpred)
    print(f"seed {seed}: t = {t:.1f} s, P = {P}; {settings_line(args)}")
    print(f"done in {time.perf_counter() - T0:.0f} s -> {PAGE}")


def write_page(meta, mo, res, ep, occ, counts, ctx, sense_tab, ft, exs, qtab, ctab, cand, errP, pred, pl, args, unpred):
    links, style_block = explainer_head()
    PCT = pct(args.box_q)
    iq_lo, iq_hi = f"q{int(round(100 * args.box_q)):02d}", f"q{int(round(100 - 100 * args.box_q)):02d}"
    t, P = meta["t"], meta["P"]
    nbox = int((cand.veto_reason == "box").sum())
    chosen = cand[cand.chosen].iloc[0]
    ch_txt = "stop" if chosen.speed < 1e-9 else f"{chosen.speed:.2f} m/s at {chosen.direction:+.0f}° from the goal direction"
    rP = pred[(pred.agent_id == P) & np.isclose(pred.h, 2.0)].iloc[0]
    wa2, wc2 = rP.hi_a - rP.lo_a, rP.hi_c - rP.lo_c
    rs = ep["veto_reasons_all_plans"]
    rs_txt = ", ".join(f"{k} {v}" for k, v in sorted(rs.items(), key=lambda kv: -kv[1]))
    dirty = " (plus uncommitted changes in foresight/, run_closed.py, run_offline.py or data/)" if meta["dirty"] else ""
    timing = meta.get("timing_at_t") or {}
    S = []
    S.append(f"""<title>Foresight Walkthrough</title>
<meta name="claude:viewport-fit-sides" content="cover">
{links}
{style_block}
<style>{EXTRA_CSS}</style>

<div class="wrap">
<header>
  <div class="eyebrow">Foresight · one decision, real values</div>
  <h1>One decision, step by step</h1>
  <p class="lede">The robot is crossing a plaza with {args.people} people. At t = {args.switch_t:g} s the whole crowd turns erratic, a behaviour the robot's bank of past crowds never contained. At t = {t:.1f} s, after the switch, it has to pick its next move. This page follows that one decision through the pipeline, with the numbers the code produced.</p>
  <p class="meta">features {meta['features_version']} · seed {meta['seed']} · planner {meta['planner']} · t = {t:.1f} s · person P = id {P} · commit {meta['commit']}{dirty} · generated {meta['generated']} by <code>tools/walkthrough.py</code><br>
  final closed-loop configuration (T5, amendment 2 and T4 addendum part 4): crossing crowd of {args.people}, robot_repulsion {args.robot_repulsion:g}, ignore_robot {args.ignore_robot:g}, every person switches into {html.escape(args.switch_to)} at t = {args.switch_t:g} s (P's regime at t: {html.escape(str(meta['regime_P_at_t']))}) · bank <code>{html.escape(Path(args.bank).name)}</code> without {html.escape(args.bank_exclude_regime)} episodes · box = {PCT} per axis · margin {args.margin} m · replan {args.replan:g} s · top speed {args.max_speed:g} m/s · {html.escape(settings_line(args))} · frozen context {args.context:,} rows + scene model on ≤ 250 recent rows, refit every {args.refit:g} s · TabPFN-3.5-Fast, 1 estimator · Kalman q = {meta['kalman_q']:g}, r = {meta['kalman_r']:g}</p>
</header>

<section>
  <h2>How this moment was chosen</h2>
  <p class="sub">Rule, applied in time order, first match wins: (1) t is at or after the switch (t ≥ {args.switch_t:g} s), so the crowd is already erratic; (2) the scene model is active: its latest refit read at least {SPLIT_MIN} completed rows of this episode (below that the frozen model runs alone), and at least {SPLIT_MIN} of those rows were made after the switch, so the scene model has seen the erratic crowd; (3) at least 3 of the {len(cand)} candidate moves are vetoed for <b>box</b>, each confirmed by recomputing the gap to a predicted box (not the 0.4 s check); (4) a person P has a prediction, moves at ≥ 0.5 m/s, is within 3 m of the robot, vetoes at least one move with their own box, and is still in the scene 1 s and 2 s later. P is the nearest such person.</p>
  <p class="sub">Result: t = {t:.1f} s, P = id {P}, {mo['dist']:.2f} m from the robot, {mo['speed']:.2f} m/s. {nbox} moves vetoed for box. The scene model in use was fitted at t = {meta['context_refit_t']:.1f} s on {counts['recent']} recent rows.</p>
</section>
""")
    S.append(f"""<section><div class="stage">Stage 1</div><h2>The scene</h2>
{fig_html('s1_scene.png', 'Plaza at t with people, trails, robot, pillars and goal', f'<b>True positions at t = {t:.1f} s, {t - args.switch_t:.1f} s after the crowd turned erratic.</b> Grey lines: last 2 s. P is ringed. {len(unpred)} dashed person(s) entered less than 1 s ago and have no prediction yet, so the planner cannot see them.')}
</section>""")
    S.append(f"""<section><div class="stage">Stage 2</div><h2>Sensing: noisy positions, filtered track</h2>
{fig_html('s2_sensing.png', 'Noisy observations and Kalman track of P', '<b>The robot never sees the true path.</b> It sees positions with 0.1 m noise and runs a constant-velocity Kalman filter. The filtered state at t is all that enters the features.')}
{table(sense_tab, left=('quantity',), nd=3)}
<p class="tcap">Reconstructed from the noise seed and checked against the planner's query row: {'match' if meta['sense_reconstruction_ok'] else 'MISMATCH'}.</p>
</section>""")
    groups = ft.group.tolist()
    S.append(f"""<section><div class="stage">Stage 3</div><h2>The table row TabPFN gets</h2>
<p class="sub">One row per person and horizon, {len(ft)} numbers. Everything is in P's own frame: x = along P's heading, y = across (left is +). The two rows differ only in <code>h_f</code>. Both models below get exactly these rows.</p>
{table(ft.drop(columns='group'), left=('feature', 'meaning'), nd=3, groups=groups)}
</section>""")
    s4 = meta["s4"]
    ms_f, ms_s = s4["mode_share_frozen"], s4["mode_share_scene"]

    def turn(m):
        return 100 * (m.get("turn_left", 0) + m.get("turn_right", 0))

    cap4 = (f"<b>What the examples say happens next.</b> Each dot is one example's outcome: grey = frozen bank rows, "
            f"blue = this run's recent rows. Diamonds and boxes: P's median and {PCT} range from each model (dashed: frozen, "
            f"blue: scene); the orange star and box are their average.")
    S.append(f"""<section><div class="stage">Stage 4</div><h2>The examples TabPFN read: two models</h2>
<p class="sub">TabPFN does not train: it reads labelled rows and predicts from them. The planner runs <b>two</b> TabPFN models and averages their answers.</p>
<p class="sub"><b>Frozen model: {counts['frozen']:,} bank rows</b>, read once before the episode starts. Random rows at their natural move frequency (no forced stops or turns), from {s4['bank_rows']:,} rows of {s4['bank_episodes']} crossing-crowd episodes (the {s4['bank_all_episodes'] - s4['bank_episodes']} episodes with any erratic behaviour were removed, so erratic is new to it). These {counts['frozen']:,} rows come from {s4['frozen_episodes']} episodes.</p>
<p class="sub"><b>Scene model: {counts['recent']} rows from this run</b>, refitted every {args.refit:g} s on the most recent completed rows (at most 250; outcome already known, t + h ≤ refit time), labelled from the robot's own filtered observations, not from the simulator. At t = {meta['context_refit_t']:.1f} s: {s4['scene_rows_h1']} rows at +1 s, {s4['scene_rows_h2']} at +2 s, made between t = {s4['scene_t_range'][0]:.1f} and {s4['scene_t_range'][1]:.1f} s; {s4['scene_rows_after_switch']} of them after the switch. Moves labelled as turns: {turn(ms_f):.0f}% of the frozen rows, {turn(ms_s):.0f}% of the scene rows.</p>
{table(exs, left=('model', 'from', 'mode'), nd=2)}
<p class="tcap">12 of the {counts['frozen'] + counts['recent']:,} rows: the first 6 frozen rows and the 6 most recent scene rows. ddx/ddy is the target: where that person really was after h seconds, minus where constant velocity put them, in their own frame.</p>
{fig_html('s4_context.png', 'Scatter of both models targets', cap4)}
</section>""")
    off_txt = (f", widened on every side by the online offset d = {meta['offset_d_at_t']:+.2f} m in force at t (the 'served' rows: what the planner checked)"
               if args.offset else "")
    cap5 = (f"<b>P's ranges drawn in the plaza.</b> Median error at +1 s / +2 s: frozen {errP.get(('frozen', 1.0), float('nan')):.2f} / "
            f"{errP.get(('frozen', 2.0), float('nan')):.2f} m, scene {errP.get(('scene', 1.0), float('nan')):.2f} / {errP.get(('scene', 2.0), float('nan')):.2f} m, "
            f"average {errP.get(1.0, float('nan')):.2f} / {errP.get(2.0, float('nan')):.2f} m; constant velocity alone "
            f"{errP.get(('cv', 1.0), float('nan')):.2f} / {errP.get(('cv', 2.0), float('nan')):.2f} m. The served +2 s box is {wa2:.2f} m along by {wc2:.2f} m across.")
    S.append(f"""<section><div class="stage">Stage 5</div><h2>TabPFN's answer for P: frozen, scene, average</h2>
{table(qtab, left=('horizon', 'axis', 'model'), nd=2, row_class=lambda r: 'hi' if r['model'].startswith('average') else '')}
<p class="tcap">Metres, in P's frame, relative to the constant-velocity point. The final answer averages the two models' quantiles per axis, then sorts them so the bounds never cross. {iq_lo} and {iq_hi} of the average are the box edges{off_txt}. "actual" is where P really went (simulator). The quantiles come from refitting both models on the same rows; largest difference from the planner's logged 5–95% box over all people: {meta['refit_vs_logged_5_95_max_diff_m']:.1e} m, and from its served box: {meta['refit_vs_planner_max_diff_m']:.1e} m.</p>
{fig_html('s5_person_box.png', 'P boxes of the frozen model, scene model and their average at +1 s and +2 s', cap5)}
</section>""")
    S.append(f"""<section><div class="stage">Stage 6</div><h2>Everyone's boxes</h2>
{fig_html('s6_all_boxes.png', 'All predicted boxes at +1 s and +2 s', f'<b>{pred.agent_id.nunique()} people predicted, {len(unpred)} not.</b> Crosses: where each predicted person actually was at t + h. P\'s box is drawn thicker.')}
</section>""")
    o1, o2 = occ.get("1", {}), occ.get("2", {})
    S.append(f"""<section><div class="stage">Stage 7</div><h2>Occupancy grid</h2>
{fig_html('s7_occupancy.png', 'Occupancy heatmaps at +1 s and +2 s', '<b>Each box becomes a probability per 0.25 m cell</b> (a Gaussian per axis fitted to the box, cells combined over people). Crosses: true centres of the predicted people at t + h.')}
{table(pd.DataFrame([{"horizon": f"+{h} s", "Brier": o["brier"], "Brier if all zero": o["brier_all_zero"], "AUC": o["auc"],
                      "true cells": o["true_cells"], "cells": o["cells"], "mean P at true cells": o["p_at_truth_mean"],
                      "max P": o["p_max"]} for h, o in occ.items()]), left=('horizon',), nd=4)}
<p class="tcap">Only {o2.get('true_cells', 0)} of {o2.get('cells', 0)} cells hold a person, so a grid of zeros already scores Brier {o2.get('brier_all_zero', float('nan')):.4f}. Brier alone says little here; the AUC (does a true cell rank above an empty one?) is the informative number.</p>
</section>""")
    S.append(f"""<section><div class="stage">Stage 8</div><h2>The planner's choice</h2>
{fig_html('s8_planner.png', 'Candidate moves coloured by veto reason', f'<b>Every candidate move, drawn to where it puts the robot in 2 s.</b> A move is vetoed if the robot (0.4 m) comes within {args.margin} m of a person (0.3 m) or their box at +1 s or +2 s, leaves the plaza, or hits a pillar. Chosen: {ch_txt}.')}
{table(ctab, left=('reason', 'blocked by'), nd=2, row_class=lambda r: 'chosen' if r['chosen'] else '')}
<p class="tcap">"blocked by" lists person ids (and horizon) whose box gap falls under the margin, recomputed with <code>planner.gap_to_box</code>. Worst gap includes the 0.4 s constant-velocity check. Planner time at this step: features {timing.get('features_ms', float('nan')):.0f} ms, prediction {timing.get('predict_ms', float('nan')):.0f} ms, planning {timing.get('plan_ms', float('nan')):.0f} ms.</p>
</section>""")
    met = pd.DataFrame([{"metric": k, "value": v} for k, v in (
        ("reached goal", "yes" if res["reached"] else "no"), ("time to goal (s)", res["time_to_goal"]),
        ("time near people, gap < 0.5 m (s)", res["intrusion_s"]), ("min gap (m)", res["min_gap"]),
        ("seconds stopped", res["seconds_stopped"]), ("collision steps", int(res["collision_steps"])),
        ("moving collision steps", int(res["moving_collision_steps"])), ("path length (m)", res["path_length"]),
        ("fallback rate (no safe move)", res["fallback_rate"]), ("prediction ms per plan", res["pred_ms_per_plan"]))])
    S.append(f"""<section><div class="stage">Stage 9</div><h2>The whole run</h2>
{fig_html('s9_ego_path.png', 'Robot path coloured by speed', f'<b>The robot\'s full path, coloured by its actual speed.</b> The ring marks t = {t:.1f} s.')}
{table(met, left=('metric', 'value'), nd=2)}
</section>""")
    q2 = qtab[(qtab.horizon == "+2 s") & (qtab.axis == "along (ddx)")].set_index("model")
    q2a, q2f, q2s = q2.loc["average (final)"], q2.loc["frozen model"], q2.loc["scene model"]
    facts = [f"Veto reasons over all {ep['plans']} plans of the episode: {rs_txt}.",
             f"Episode-wide, the served {PCT} box{' (with the online offset)' if args.offset else ''} held the true position {100 * ep.get('cover_box_1s', float('nan')):.0f}% of the time at +1 s and {100 * ep.get('cover_box_2s', float('nan')):.0f}% at +2 s (≈ {100 * (1 - 2 * args.box_q) ** 2:.0f}% expected for a 2-axis box of {100 - 200 * args.box_q:.0f}% intervals, if the axes were independent). Median error of the TabPFN median {ep.get('median_err_1s', float('nan')):.2f} / {ep.get('median_err_2s', float('nan')):.2f} m, constant velocity {ep.get('median_cv_err_1s', float('nan')):.2f} / {ep.get('median_cv_err_2s', float('nan')):.2f} m; median box side (geometric mean) {ep.get('box_side_geo_1s', float('nan')):.2f} / {ep.get('box_side_geo_2s', float('nan')):.2f} m.",
             f"Rows read at this moment: frozen model {counts['frozen']:,} bank rows (fixed for the whole episode), scene model {counts['recent']} recent rows, of which {int(np.isclose(ctx.h_f, 2.0).sum())} at +2 s (a +2 s outcome is known only 2 s later)." + (f" Online offset (per-batch rule): d = {meta['offset_d_at_t']:+.3f} m at t, {res['offset_final']:+.3f} m at the end of the episode, mean {res['offset_mean']:+.3f} m; raw 5–95% box missed {100 * res['offset_miss']:.0f}% of scored rows (target 20%)." if args.offset else ""),
             f"Stopping was {'vetoed (' + cand.iloc[0].veto_reason + ')' if cand.iloc[0].veto_reason != 'none' else 'allowed'} at this moment (worst gap {cand.iloc[0].worst_gap:.2f} m): blocked by {ctab.iloc[0]['blocked by'] or 'nobody (short check)'}: a box reaches the spot where the robot stands." if cand.iloc[0].veto_reason == 'box' else f"Stopping: {cand.iloc[0].veto_reason}.",
             f"P's +2 s range (average of the two models) along the heading: q05 {q2a.q05:.2f} m vs q95 {q2a.q95:.2f} m around the constant-velocity point; frozen model alone {q2f.q05:.2f} to {q2f.q95:.2f} m, scene model alone {q2s.q05:.2f} to {q2s.q95:.2f} m.",
             f"Occupancy at +2 s: Brier {o2.get('brier', float('nan')):.5f} against {o2.get('brier_all_zero', float('nan')):.5f} for an all-zero grid ({100 * (1 - o2.get('brier', 0) / max(o2.get('brier_all_zero', 1), 1e-12)):.0f}% better); AUC {o2.get('auc', float('nan')):.3f}. Mean probability in the cells that really hold someone: {o2.get('p_at_truth_mean', float('nan')):.3f}.",
             f"Prediction cost: {res['pred_ms_per_plan']:.0f} ms per replan on average, refits included (wall time on a shared CPU, 3 threads, other jobs running; it varies between runs), against a 400 ms replan period. The episode clock is simulated, so this does not change the robot's behaviour."]
    S.append(f"""<section><h2>Numbers worth a second look</h2><ul class="facts">{''.join('<li>' + html.escape(f) + '</li>' for f in facts)}</ul>
<p class="tcap">Tables behind every figure: <code>docs/walkthrough/*.csv</code> and <code>*.json</code>. Rerun with <code>python tools/walkthrough.py</code>.</p></section>
</div>""")
    PAGE.write_text("\n".join(S), encoding="utf-8")


if __name__ == "__main__":
    main()
