"""Data cards: what each dataset in data/ looks like, before anyone trusts results on it.

Reads only the INTERFACE.md files (tracks.csv, obstacles.csv, meta.json) and the cached feature
rows (data/<name>/rows_*.parquet, for the mode labels). Imports nothing from foresight/, trains
nothing. Prints the numbers as markdown tables (the source for docs/DATA.md) and writes two figures:

    media/fig_data_overview.png   speed, constant-velocity error at +2 s, mode shares, nearest neighbour
    media/fig_data_scenes.png     one top-down episode per dataset

    python tools/data_cards.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, MEDIA = ROOT / "data", ROOT / "media"
DATASETS = ["crowd", "crowd_c2e", "eth", "hotel", "toy"]
IN_FIGURE = ["crowd", "crowd_c2e", "eth", "hotel"]           # toy is a test stand-in only
COLORS = {"crowd": "#2a78d6", "crowd_c2e": "#eb6834", "eth": "#1baf7a", "hotel": "#eda100"}
MODES = ["straight", "stop", "turn_left", "turn_right"]
STOP_SPEED = 0.2      # m/s, as in INTERFACE.md and features.build_rows
CV_H = 2.0            # s
TURN_DEG = 30.0       # heading change for a turn, as in features.build_rows
TURN_MIN_SPEED = 0.5  # m/s at both ends, for the turn cross-check from tracks
PERSON_SPEED_BOUND = 5.0


def q(a, p):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if len(a) else float("nan")


def load(name):
    d = DATA / name
    tracks = pd.read_csv(d / "tracks.csv")
    meta = json.loads((d / "meta.json").read_text())
    obst_path = d / "obstacles.csv"
    obstacles = pd.read_csv(obst_path) if obst_path.exists() and obst_path.stat().st_size else pd.DataFrame()
    dt = float(meta["dt"])
    tracks["step"] = np.round(tracks["t"] / dt).astype(int)
    return tracks, obstacles, meta, dt


def card(name):
    tracks, obstacles, meta, dt = load(name)
    is_sim = tracks["kind"].ne("person").any() or "goal_x" in tracks.columns
    c = {"name": name, "sim": meta.get("sim"), "dt": dt, "notes": meta.get("notes", "")}
    ppl = tracks[tracks["kind"] == "person"].copy()
    ppl["speed"] = np.hypot(ppl["vx"], ppl["vy"])

    # ---- size ----
    c["episodes"] = len(meta["episodes"])
    c["episodes_in_tracks"] = tracks["episode"].nunique()
    span = tracks.groupby("episode")["step"].agg(["min", "max"])
    c["duration_s_total"] = float(((span["max"] - span["min"]) * dt).sum())
    c["episode_len_s"] = (float(((span["max"] - span["min"]) * dt).min()),
                          float(((span["max"] - span["min"]) * dt).max()))
    c["rows"] = len(tracks)
    # continuous recording span: windows of fixed length laid end to end (matters for real data,
    # whose windows can end, or start, with nobody in view)
    per = int(round(float(next(iter(meta["episodes"].values()))["duration"]) / dt))
    per = max(per, int(tracks["step"].max()) + 1)        # simulated episodes include both t = 0 and t = duration
    kk = tracks["episode"] * per + tracks["step"]
    c["span_s"] = float((kk.max() - kk.min() + 1) * dt)
    c["episode_t_range"] = {int(e): (float(a), float(b)) for e, (a, b) in
                            tracks.groupby("episode")["t"].agg(["min", "max"]).iterrows()}
    raw = DATA / "raw" / name / "obsmat.txt"
    c["raw_rows"] = int(sum(1 for line in raw.open() if line.strip())) if raw.exists() else None

    # people per frame over the continuous span (frames with nobody count as 0)
    cnt = ppl.groupby(["episode", "step"]).size()
    grid_frames = int(kk.max() - kk.min() + 1)                 # every step of the continuous span
    c["ppf_mean_grid"] = float(cnt.sum() / grid_frames)
    c["ppf_mean_present"] = float(cnt.mean())
    c["ppf_max"] = int(cnt.max())
    c["empty_frames_share"] = 1.0 - len(cnt) / grid_frames
    c["tracks"] = int(ppl.groupby(["episode", "agent_id"]).ngroups)
    c["unique_ids"] = int(ppl["agent_id"].nunique())
    tl = ppl.groupby(["episode", "agent_id"])["step"].agg(lambda s: (s.max() - s.min()) * dt)
    c["track_len_s_p50"] = float(tl.median())

    # ---- speed (sim agents start from rest at t = 0: drop that step) ----
    sp = ppl if not is_sim else ppl[ppl["step"] > 0]
    s = sp["speed"].to_numpy()
    c["speed_mean"], c["speed_p10"], c["speed_p50"], c["speed_p90"] = s.mean(), q(s, 10), q(s, 50), q(s, 90)
    c["speed_max"] = float(s.max())
    c["stopped_share"] = float((s < STOP_SPEED).mean())
    c["over_bound"] = int((s > PERSON_SPEED_BOUND).sum())
    w = s[s >= STOP_SPEED]
    c["walk_mean"], c["walk_p50"], c["walk_p90"] = float(w.mean()), q(w, 50), q(w, 90)
    c["over_3"] = int((s > 3.0).sum())
    c["over_3_tracks"] = int(sp[sp["speed"] > 3.0].groupby(["episode", "agent_id"]).ngroups)
    c["zero_speed_share"] = float((s == 0).mean())
    ms = sp.groupby(["episode", "agent_id"])["speed"].mean()
    n_rows = sp.groupby(["episode", "agent_id"]).size()
    c["standing_tracks"] = int((ms < STOP_SPEED).sum())
    c["standing_rows_share"] = float(n_rows[ms < STOP_SPEED].sum() / n_rows.sum())
    c["_speed"] = s

    # ---- integrity: duplicates, gaps, displacement vs stored velocity ----
    c["duplicates"] = int(tracks.duplicated(["episode", "t", "agent_id"]).sum())
    c["t_off_grid"] = int((np.abs(tracks["t"] / dt - tracks["step"]) > 1e-6).sum())
    srt = ppl.sort_values(["episode", "agent_id", "step"])
    g = srt.groupby(["episode", "agent_id"])
    dstep = g["step"].diff()
    c["tracks_with_gaps"] = int((dstep > 1).groupby([srt["episode"], srt["agent_id"]]).any().sum())
    c["max_gap_steps"] = int(np.nan_to_num(dstep.max()))
    one = dstep == 1
    fdx = g["x"].diff()[one] / dt
    fdy = g["y"].diff()[one] / dt
    fd_speed = np.hypot(fdx, fdy)
    vx_avg = (srt["vx"] + g["vx"].shift(1))[one] / 2
    vy_avg = (srt["vy"] + g["vy"].shift(1))[one] / 2
    mism = np.hypot(fdx - vx_avg, fdy - vy_avg)
    c["fd_speed_max"] = float(fd_speed.max())
    c["fd_speed_over_bound"] = int((fd_speed > PERSON_SPEED_BOUND).sum())
    c["vel_mismatch_p50"], c["vel_mismatch_p99"] = q(mism, 50), q(mism, 99)
    c["vel_mismatch_over_1"] = float((mism > 1.0).mean())
    c["vel_mismatch_over_1_n"] = int((mism > 1.0).sum())
    # "stall then catch up": a step that barely moves although the stored speed says walking
    stored = np.hypot(srt["vx"], srt["vy"])[one]
    c["stall_steps"] = int(((fd_speed < 0.3) & (stored > 1.0)).sum())
    c["max_step_m"] = float(fd_speed.max() * dt)

    # ---- constant velocity at +2 s on ground truth (self-join on step + H: correct with gaps) ----
    H = int(round(CV_H / dt))
    base = ppl if not is_sim else ppl[ppl["step"] > 0]
    fut = ppl[["episode", "agent_id", "step", "x", "y", "vx", "vy"]].rename(
        columns={"x": "fx", "y": "fy", "vx": "fvx", "vy": "fvy"})
    fut["step"] -= H
    j = base.merge(fut, on=["episode", "agent_id", "step"], how="inner")
    err = np.hypot(j["fx"] - (j["x"] + j["vx"] * CV_H), j["fy"] - (j["y"] + j["vy"] * CV_H))
    c["cv_n"] = len(err)
    c["cv_mean"], c["cv_p50"], c["cv_p90"] = float(err.mean()), q(err, 50), q(err, 90)
    c["cv_under_0.5"] = float((err < 0.5).mean())
    c["_cv"] = err.to_numpy()
    # stop/go/no-change split: how much of the CV error comes from people who stop or start
    # turn cross-check from tracks: heading change over 2 s, both ends faster than 0.5 m/s
    fast = (np.hypot(j["vx"], j["vy"]) > TURN_MIN_SPEED) & (np.hypot(j["fvx"], j["fvy"]) > TURN_MIN_SPEED)
    dth = np.arctan2(j["fvy"], j["fvx"]) - np.arctan2(j["vy"], j["vx"])
    dth = np.degrees((dth + np.pi) % (2 * np.pi) - np.pi)
    c["turn_share_fast"] = float((np.abs(dth[fast]) > TURN_DEG).mean()) if fast.any() else float("nan")
    c["fast_share"] = float(fast.mean())
    if "goal_x" in ppl.columns:
        # how often people pick a new goal, and how many 2 s turns come with a goal change
        gs = srt.groupby(["episode", "agent_id"])
        chg = ((gs["goal_x"].diff().abs() > 1e-9) | (gs["goal_y"].diff().abs() > 1e-9)).astype(int)
        c["goal_changes_per_min"] = float(chg.sum() / (len(srt) * dt / 60))
        # goal change in (step, step + H]: difference of the cumulative count per track
        cs_ = chg.groupby([srt["episode"], srt["agent_id"]]).cumsum()
        key = srt[["episode", "agent_id", "step"]].assign(cs=cs_.to_numpy())
        later = key.assign(step=key["step"] - H).rename(columns={"cs": "cs_f"})
        jj = (j[["episode", "agent_id", "step"]].merge(key, on=["episode", "agent_id", "step"], how="left")
              .merge(later, on=["episode", "agent_id", "step"], how="left"))
        gc = (jj["cs_f"] - jj["cs"]).to_numpy() > 0
        turn = fast.to_numpy() & (np.abs(dth.to_numpy()) > TURN_DEG)
        c["turns_with_goal_change"] = float(gc[turn].mean())
        c["turn_share_fast_no_goal_change"] = float(turn[fast.to_numpy() & ~gc].mean())
    if is_sim:
        c["cv_by_regime"] = {r: float(err[j["regime"] == r].mean()) for r in sorted(j["regime"].unique())}
        c["turn_fast_by_regime"] = {r: float((np.abs(dth[fast & (j["regime"] == r)]) > TURN_DEG).mean())
                                    for r in sorted(j["regime"].unique())}
        c["stopped_by_regime"] = {r: float((sp["speed"][sp["regime"] == r] < STOP_SPEED).mean())
                                  for r in sorted(sp["regime"].unique())}
        c["speed_by_regime"] = {r: float(sp["speed"][sp["regime"] == r].mean())
                                for r in sorted(sp["regime"].unique())}
        c["walk_by_regime"] = {r: float(sp["speed"][(sp["regime"] == r) & (sp["speed"] >= STOP_SPEED)].mean())
                               for r in sorted(sp["regime"].unique())}

    # ---- regimes ----
    c["regime_rows"] = ppl["regime"].value_counts(normalize=True).round(3).to_dict()
    eps = meta["episodes"].values()
    pairs = {}
    for e in eps:
        k = e["regime_before"] + (f" -> {e['regime_after']}" if e.get("regime_after") else " (no switch)")
        pairs[k] = pairs.get(k, 0) + 1
    c["regime_pairs"] = dict(sorted(pairs.items(), key=lambda kv: -kv[1]))
    sw = [e["regime_switch_t"] for e in eps if e.get("regime_switch_t") is not None]
    c["switch_eps"], c["switch_t_range"] = len(sw), ((min(sw), max(sw)) if sw else None)

    # ---- nearest neighbour (person to person, centre to centre), robot gap, overlaps ----
    radius = float(ppl["radius"].median())
    nn, robot_gap, robot_gap_ep, inside = [], [], [], 0
    robot = tracks[tracks["is_ego"] == 1].set_index(["episode", "step"])[["x", "y", "radius"]]
    obst_by_ep = {e: o for e, o in obstacles.groupby("episode")} if len(obstacles) else {}
    for (ep, st), fr in ppl.groupby(["episode", "step"], sort=False):
        P = fr[["x", "y"]].to_numpy()
        if len(P) >= 2:
            D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
            np.fill_diagonal(D, np.inf)
            nn.append(D.min(axis=1))
        if len(robot) and (ep, st) in robot.index:
            rx, ry, rr = robot.loc[(ep, st)]
            robot_gap.append(np.hypot(P[:, 0] - rx, P[:, 1] - ry) - rr - fr["radius"].to_numpy())
            robot_gap_ep.append((ep, float(robot_gap[-1].min())))
        if ep in obst_by_ep:
            o = obst_by_ep[ep]
            od = np.hypot(P[:, None, 0] - o["x"].to_numpy()[None], P[:, None, 1] - o["y"].to_numpy()[None])
            inside += int((od < o["r"].to_numpy()[None] + fr["radius"].to_numpy()[:, None] - 1e-3).any(axis=1).sum())
    nn = np.concatenate(nn) if nn else np.array([])
    c["nn_p10"], c["nn_p50"], c["nn_mean"] = q(nn, 10), q(nn, 50), float(nn.mean())
    c["radius"] = radius
    c["nn_overlap_share"] = float((nn < 2 * radius).mean())          # discs touching or overlapping
    c["nn_under_0.3"] = float((nn < 0.3).mean())
    c["nn_min"] = float(nn.min())
    c["_nn"] = nn
    c["inside_obstacle"] = inside
    c["collided_share"] = float(ppl["collided"].mean()) if "collided" in ppl else None
    c["robot"] = bool((tracks["is_ego"] == 1).any())
    if c["robot"]:
        rg = np.concatenate(robot_gap)
        c["robot_gap_min"], c["robot_gap_p10"] = float(rg.min()), q(rg, 10)
        c["robot_gap_under_0.3"] = float((rg < 0.3).mean())
        c["robot_overlap_rows"] = int((rg < 0).sum())
        c["robot_overlap_eps"] = len({e for e, g in robot_gap_ep if g < 0})
        rb = tracks[tracks["is_ego"] == 1]
        c["robot_speed_mean"] = float(np.hypot(rb["vx"], rb["vy"]).mean())
    if meta.get("bounds"):
        x0, y0, x1, y1 = meta["bounds"]
        c["outside_bounds"] = int(((ppl["x"] < x0) | (ppl["x"] > x1) | (ppl["y"] < y0) | (ppl["y"] > y1)).sum())
    c["obstacles_total"] = len(obstacles)
    c["obstacles_per_ep"] = (float(obstacles.groupby("episode").size().mean()) if len(obstacles) else 0.0)
    c["obstacle_r"] = ((float(obstacles["r"].min()), float(obstacles["r"].max())) if len(obstacles) else None)

    # ---- mode labels from the cached rows (ground-truth labels; *_observed has the same labels) ----
    files = sorted(p for p in (DATA / name).glob("rows_*.parquet") if not p.stem.endswith("_observed"))
    c["rows_file"] = files[0].name if files else None
    if files:
        r = pd.read_parquet(files[0], columns=["h", "mode", "regime", "speed", "agent_id", "episode", "t"])
        r["h"] = r["h"].round(2)
        c["modes"] = {float(h): {m: float((x["mode"] == m).mean()) for m in MODES} | {"n": len(x)}
                      for h, x in r.groupby("h")}
        r2 = r[r["h"] == 2.0]
        c["rows_slow_at_t_h2"] = float((r2["speed"] < STOP_SPEED).mean())
        if r["regime"].nunique() > 1:
            c["modes_by_regime"] = {reg: {m: float((x["mode"] == m).mean()) for m in MODES}
                                    for reg, x in r2.groupby("regime")}
    c["_tracks"], c["_obstacles"], c["_meta"] = tracks, obstacles, meta
    return c


# ---------------- output ----------------
def f(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{d}f}"


def pct(x, d=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def print_card(c):
    p = print
    p(f"\n## {c['name']}  ({c['sim']}, dt {c['dt']} s)\n")
    p("| Quantity | Value |\n|---|---|")
    rows = [
        ("Episodes (meta / in tracks)", f"{c['episodes']} / {c['episodes_in_tracks']}"),
        ("Time from first to last row, summed over episodes", f"{c['duration_s_total'] / 60:.1f} min (episodes {c['episode_len_s'][0]:.1f}-{c['episode_len_s'][1]:.1f} s)"),
        ("Continuous span (windows end to end)", f"{c['span_s'] / 60:.1f} min"),
        ("Rows in tracks.csv" + (" / raw obsmat.txt" if c["raw_rows"] is not None else ""),
         f"{c['rows']:,}" + (f" / {c['raw_rows']:,}" if c["raw_rows"] is not None else "")),
        ("People per frame, mean (all frames / frames with someone)", f"{c['ppf_mean_grid']:.1f} / {c['ppf_mean_present']:.1f}"),
        ("People per frame, max", c["ppf_max"]),
        ("Frames with nobody", pct(c["empty_frames_share"])),
        ("Person tracks (episode, id) / unique ids", f"{c['tracks']} / {c['unique_ids']}"),
        ("Track length, median", f"{c['track_len_s_p50']:.1f} s"),
        ("Speed mean / p10 / p50 / p90 / max (m/s)",
         f"{f(c['speed_mean'])} / {f(c['speed_p10'])} / {f(c['speed_p50'])} / {f(c['speed_p90'])} / {f(c['speed_max'])}"),
        ("Time stopped (speed < 0.2 m/s)", pct(c["stopped_share"])),
        ("Walking speed (rows >= 0.2 m/s) mean / p50 / p90",
         f"{f(c['walk_mean'])} / {f(c['walk_p50'])} / {f(c['walk_p90'])}"),
        ("Rows with speed exactly 0", pct(c["zero_speed_share"])),
        ("Standing tracks (mean speed < 0.2) / their share of rows",
         f"{c['standing_tracks']} / {pct(c['standing_rows_share'])}"),
        ("CV error at +2 s, mean / p50 / p90 (m)", f"{f(c['cv_mean'])} / {f(c['cv_p50'])} / {f(c['cv_p90'])}  (n = {c['cv_n']:,})"),
        ("CV error at +2 s under 0.5 m", pct(c["cv_under_0.5"])),
        ("Heading change > 30 deg over 2 s, both ends > 0.5 m/s (tracks)", f"{pct(c['turn_share_fast'])} of {pct(c['fast_share'])} of pairs"),
        ("Nearest person, p10 / p50 / mean (m, centre to centre)", f"{f(c['nn_p10'])} / {f(c['nn_p50'])} / {f(c['nn_mean'])}"),
        (f"Nearest person closer than 2 x radius ({2 * c['radius']:.2f} m) / under 0.3 m / min",
         f"{pct(c['nn_overlap_share'], 2)} / {pct(c['nn_under_0.3'], 2)} / {f(c['nn_min'], 3)} m"),
        ("Robot", "yes" if c["robot"] else "no"),
        ("Obstacles", f"{c['obstacles_total']} total, {c['obstacles_per_ep']:.1f} per episode"
         + (f", r {c['obstacle_r'][0]:.2f}-{c['obstacle_r'][1]:.2f} m" if c["obstacle_r"] else "")),
        ("Regime / scene, share of person rows", ", ".join(f"{k} {v:.0%}" for k, v in c["regime_rows"].items())),
        ("Episodes by regime (switch)", "; ".join(f"{k}: {v}" for k, v in c["regime_pairs"].items())),
    ]
    if c["switch_t_range"]:
        rows.append(("Switch time range", f"{c['switch_t_range'][0]:.1f}-{c['switch_t_range'][1]:.1f} s in {c['switch_eps']} episodes"))
    if c["robot"]:
        rows.append(("Robot gap (edge to edge), min / p10; share under 0.3 m",
                     f"{f(c['robot_gap_min'])} / {f(c['robot_gap_p10'])} m; {pct(c['robot_gap_under_0.3'], 2)}"))
        rows.append(("Person rows overlapping the robot (gap < 0) / episodes",
                     f"{c['robot_overlap_rows']} / {c['robot_overlap_eps']}"))
        rows.append(("Robot mean speed", f"{f(c['robot_speed_mean'])} m/s"))
    if "goal_changes_per_min" in c:
        rows.append(("New goals per person-minute", f(c["goal_changes_per_min"], 1)))
        rows.append(("2 s turns (fast pairs) with a goal change inside the 2 s",
                     f"{pct(c['turns_with_goal_change'])}; turn share without one "
                     f"{pct(c['turn_share_fast_no_goal_change'])}"))
    if c["collided_share"] is not None:
        rows.append(("Person rows with collided = 1", pct(c["collided_share"], 2)))
    for k, v in rows:
        p(f"| {k} | {v} |")

    if c.get("modes"):
        p(f"\nMode labels ({c['rows_file']}; label always looks 2 s ahead):\n")
        p("| h (s) | rows | straight | stop | turn_left | turn_right |\n|---|---|---|---|---|---|")
        for h, m in c["modes"].items():
            p(f"| {h:g} | {m['n']:,} | " + " | ".join(pct(m[k]) for k in MODES) + " |")
        p(f"\nRows at h = 2 s whose speed feature at t is under 0.2 m/s: {pct(c['rows_slow_at_t_h2'])}")
        if c.get("modes_by_regime"):
            p("\nMode labels at h = 2 s by regime:\n")
            p("| regime | straight | stop | turn_left | turn_right |\n|---|---|---|---|---|")
            for reg, m in c["modes_by_regime"].items():
                p(f"| {reg} | " + " | ".join(pct(m[k]) for k in MODES) + " |")
    else:
        p("\nMode labels: n/a (no cached rows for this dataset)")
    if c.get("cv_by_regime"):
        p("\nBy regime (person rows, t > 0):\n")
        p("| regime | mean speed | walking speed | stopped | CV err +2 s mean | turn > 30 deg (fast pairs) |\n"
          "|---|---|---|---|---|---|")
        for reg in c["cv_by_regime"]:
            p(f"| {reg} | {f(c['speed_by_regime'].get(reg))} | {f(c['walk_by_regime'].get(reg))} | "
              f"{pct(c['stopped_by_regime'].get(reg))} | "
              f"{f(c['cv_by_regime'][reg])} | {pct(c['turn_fast_by_regime'][reg])} |")
    p("\nIntegrity:\n")
    p("| Check | Value |\n|---|---|")
    checks = [
        ("Duplicate (episode, t, agent_id)", c["duplicates"]),
        ("t off the dt grid", c["t_off_grid"]),
        ("Person tracks with time gaps / longest gap", f"{c['tracks_with_gaps']} / {c['max_gap_steps']} steps"),
        ("Stored speed > 5 m/s", c["over_bound"]),
        ("Stored speed > 3 m/s (rows / tracks)", f"{c['over_3']} / {c['over_3_tracks']}"),
        ("Step displacement / dt: max; rows > 5 m/s", f"{f(c['fd_speed_max'])} m/s; {c['fd_speed_over_bound']}"),
        ("Displacement/dt vs stored velocity, mismatch p50 / p99; share > 1 m/s",
         f"{f(c['vel_mismatch_p50'], 3)} / {f(c['vel_mismatch_p99'], 3)} m/s; {pct(c['vel_mismatch_over_1'], 2)}"
         f" ({c['vel_mismatch_over_1_n']} steps)"),
        ("Stall steps (moves < 0.3 m/s while stored speed > 1 m/s)", c["stall_steps"]),
        ("Largest single-step displacement", f"{f(c['max_step_m'])} m"),
        ("Episode t range (first-last row)", "; ".join(f"{e}: {a:g}-{b:g}" for e, (a, b) in c["episode_t_range"].items())
         if len(c["episode_t_range"]) <= 20 else "all episodes " +
         f"{min(a for a, _ in c['episode_t_range'].values()):g}-{max(b for _, b in c['episode_t_range'].values()):g} s"),
        ("Person rows inside an obstacle", c["inside_obstacle"]),
    ]
    if "outside_bounds" in c:
        checks.append(("Person rows outside bounds", c["outside_bounds"]))
    for k, v in checks:
        p(f"| {k} | {v} |")


def figures(cards):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    cs = [cards[n] for n in IN_FIGURE if n in cards]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))

    a = ax[0, 0]
    bins = np.linspace(0, 2.6, 53)
    for c in cs:
        a.hist(np.clip(c["_speed"], 0, 2.6), bins=bins, density=True, histtype="step", lw=2,
               color=COLORS[c["name"]], label=f"{c['name']} (mean {c['speed_mean']:.2f})")
    a.axvline(STOP_SPEED, color="#52514e", lw=1, ls="--")
    a.text(STOP_SPEED + 0.03, a.get_ylim()[1] * 0.92, "stop threshold 0.2", color="#52514e", fontsize=8)
    a.set(title="Person speed (ground truth)", xlabel="speed (m/s), clipped at 2.6", ylabel="density")
    a.legend(frameon=False, fontsize=8)

    a = ax[0, 1]
    bins = np.linspace(0, 4, 41)
    for c in cs:
        a.hist(np.clip(c["_cv"], 0, 4), bins=bins, density=True, histtype="step", lw=2,
               color=COLORS[c["name"]], label=f"{c['name']} (mean {c['cv_mean']:.2f}, p90 {c['cv_p90']:.2f})")
    a.set(title="Constant-velocity error at +2 s (ground truth)", xlabel="position error (m), clipped at 4",
          ylabel="density")
    a.legend(frameon=False, fontsize=8)

    a = ax[1, 0]
    w = 0.8 / len(cs)
    x = np.arange(len(MODES))
    for i, c in enumerate(cs):
        m = c.get("modes", {}).get(2.0)
        if not m:
            continue
        vals = [m[k] for k in MODES]
        bars = a.bar(x + (i - (len(cs) - 1) / 2) * w, vals, w * 0.92, color=COLORS[c["name"]], label=c["name"])
        for b, v in zip(bars, vals):
            a.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{100 * v:.0f}", ha="center", fontsize=7, color="#52514e")
    a.set_xticks(x, MODES)
    a.set(title="Mode label share, rows at h = 2 s (label looks 2 s ahead)", ylabel="share of rows", ylim=(0, 1))
    a.legend(frameon=False, fontsize=8)

    a = ax[1, 1]
    bins = np.linspace(0, 8, 65)
    for c in cs:
        a.hist(np.clip(c["_nn"], 0, 8), bins=bins, density=True, histtype="step", lw=2,
               color=COLORS[c["name"]], label=f"{c['name']} (p50 {c['nn_p50']:.2f})")
    a.set(title="Distance to nearest other person", xlabel="centre-to-centre distance (m), clipped at 8",
          ylabel="density")
    a.legend(frameon=False, fontsize=8)

    fig.suptitle("Foresight datasets at a glance (tools/data_cards.py)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    MEDIA.mkdir(exist_ok=True)
    fig.savefig(MEDIA / "fig_data_overview.png", dpi=150)
    plt.close(fig)

    # one top-down episode per dataset
    fig, ax = plt.subplots(1, len(cs), figsize=(4 * len(cs), 4.4))
    for a, c in zip(np.atleast_1d(ax), cs):
        tr, ob = c["_tracks"], c["_obstacles"]
        counts = tr[tr["kind"] == "person"].groupby("episode").size()
        ep = counts.idxmax() if c["name"] in ("eth", "hotel") else counts.index[0]
        e = tr[tr["episode"] == ep]
        for (_, kind), g in e.groupby(["agent_id", "kind"]):
            g = g.sort_values("step")
            a.plot(g["x"], g["y"], lw=0.6, color="#0b0b0b" if kind == "robot" else "#b5b4ae",
                   zorder=3 if kind == "robot" else 1)
        mid = e["step"].quantile(0.5, interpolation="nearest")
        now = e[(e["step"] == mid) & (e["kind"] == "person")]
        a.scatter(now["x"], now["y"], s=22, color=COLORS[c["name"]], zorder=4, edgecolor="#fcfcfb", lw=0.8)
        a.quiver(now["x"], now["y"], now["vx"], now["vy"], angles="xy", scale_units="xy", scale=1,
                 width=0.004, color=COLORS[c["name"]], zorder=4)
        rob = e[(e["step"] == mid) & (e["is_ego"] == 1)]
        if len(rob):
            a.scatter(rob["x"], rob["y"], s=60, marker="s", color="#0b0b0b", zorder=5, label="robot")
        if len(ob):
            for _, o in ob[ob["episode"] == ep].iterrows():
                a.add_patch(plt.Circle((o["x"], o["y"]), o["r"], color="#52514e", alpha=0.5, zorder=2))
        b = c["_meta"].get("bounds") or c["_meta"].get("draw_bounds")
        if b:
            a.set_xlim(b[0], b[2]); a.set_ylim(b[1], b[3])
        a.set_aspect("equal")
        a.set_title(f"{c['name']}, episode {ep}: paths (grey), people at t = {mid * c['dt']:.0f} s\n"
                    "(arrow = 1 s of velocity" + ("; square = robot, discs = pillars)" if len(rob) else ")"),
                    fontsize=8)
        a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(MEDIA / "fig_data_scenes.png", dpi=150)
    plt.close(fig)


def comparison(cards):
    keys = [("speed_mean", "mean speed (m/s)", f), ("speed_p50", "median speed (m/s)", f),
            ("walk_mean", "walking speed mean, rows >= 0.2 m/s", f), ("walk_p90", "walking speed p90", f),
            ("stopped_share", "time stopped", pct), ("cv_mean", "CV err +2 s mean (m)", f),
            ("cv_p90", "CV err +2 s p90 (m)", f), ("turn_share_fast", "turn > 30 deg / 2 s (fast pairs)", pct),
            ("nn_p50", "nearest person p50 (m)", f), ("ppf_mean_grid", "people per frame", lambda x: f(x, 1))]
    names = [n for n in DATASETS if n in cards]
    print("\n## Side by side\n")
    print("| | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for k, label, fmt in keys:
        print(f"| {label} | " + " | ".join(fmt(cards[n][k]) for n in names) + " |")
    print("| stop label, h = 2 s | " + " | ".join(pct(cards[n]["modes"][2.0]["stop"]) if cards[n].get("modes") else "n/a"
                                                for n in names) + " |")
    print("| turn labels, h = 2 s | " + " | ".join(
        pct(cards[n]["modes"][2.0]["turn_left"] + cards[n]["modes"][2.0]["turn_right"]) if cards[n].get("modes") else "n/a"
        for n in names) + " |")


def main():
    cards = {}
    for n in DATASETS:
        if (DATA / n / "tracks.csv").exists():
            print(f"computing {n} ...", file=sys.stderr)
            cards[n] = card(n)
    print("# Data card numbers (python tools/data_cards.py)")
    for n in cards:
        print_card(cards[n])
    comparison(cards)
    figures(cards)
    print("\nwrote media/fig_data_overview.png, media/fig_data_scenes.png", file=sys.stderr)


if __name__ == "__main__":
    main()
