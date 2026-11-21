"""Choose the robot's next move from predicted ranges. Embodiment-agnostic: the robot is a
disc with a velocity command; a car adapter would swap the candidate set.

For every candidate velocity (speed x direction relative to the goal):
  1. robot position at each horizon h = now + cmd * h
  2. distance from that position to each person's predicted box at h (box = constant-velocity
     point + model quantile range, in the person's frame), minus both radii
  3. veto the candidate if any gap < margin, or it leaves the area / enters an obstacle
  4. among the rest, take the one that gets closest to the goal; if none is safe, take the
     one with the largest worst-case gap (fallback)
"""
from __future__ import annotations

import numpy as np

SPEEDS = (0.0, 0.4, 0.8, 1.2)
TURNS_DEG = (-90, -60, -30, 0, 30, 60, 90)


def candidates(p_robot, goal, speeds=SPEEDS, turns=TURNS_DEG):
    g = np.asarray(goal) - p_robot
    base = np.arctan2(g[1], g[0])
    out = [np.zeros(2)]
    for s in speeds:
        if s == 0:
            continue
        for d in turns:
            a = base + np.radians(d)
            out.append(s * np.array([np.cos(a), np.sin(a)]))
    return np.array(out)


def gap_to_box(pt, rows):
    """Distance from world point pt to each row's box (rows: DataFrame with px, py, vx, vy, h,
    theta, lo_a, hi_a, lo_c, hi_c). Zero if inside."""
    cvx = rows.px.to_numpy() + rows.vx.to_numpy() * rows.h.to_numpy()
    cvy = rows.py.to_numpy() + rows.vy.to_numpy() * rows.h.to_numpy()
    c, s = np.cos(rows.theta.to_numpy()), np.sin(rows.theta.to_numpy())
    dx, dy = pt[0] - cvx, pt[1] - cvy
    a, b = c * dx + s * dy, -s * dx + c * dy                      # into the person's frame
    ea = np.maximum(0, np.maximum(rows.lo_a.to_numpy() - a, a - rows.hi_a.to_numpy()))
    eb = np.maximum(0, np.maximum(rows.lo_c.to_numpy() - b, b - rows.hi_c.to_numpy()))
    return np.hypot(ea, eb)


def choose(p_robot, goal, pred_rows, radius_robot=0.4, radius_person=0.3, margin=0.2, bounds=None,
           obstacles=(), horizons=(1.0, 2.0), short_h=0.4, stop_bias=0.15):
    """Return (cmd, info). pred_rows: predicted boxes for the people at this moment.

    Besides the predicted boxes at `horizons`, a hard short-range check: at `short_h` seconds
    people are assumed to keep their velocity (too short for behaviour to change), and any
    candidate that would touch someone then is vetoed. If no candidate is safe, the robot
    stops unless some move keeps a clearly larger worst-case gap (by `stop_bias` metres)."""
    cands = candidates(p_robot, goal)
    worst = np.full(len(cands), np.inf)
    ok = np.ones(len(cands), bool)
    now = pred_rows.drop_duplicates("agent_id") if len(pred_rows) else pred_rows
    for i, cmd in enumerate(cands):
        if len(now):                                   # short-range constant-velocity check
            pt = p_robot + cmd * short_h
            d = np.hypot(now.px.to_numpy() + now.vx.to_numpy() * short_h - pt[0],
                         now.py.to_numpy() + now.vy.to_numpy() * short_h - pt[1]) - radius_robot - radius_person
            if d.min() < 0.05:
                ok[i] = False
            worst[i] = min(worst[i], d.min())
        for h in horizons:
            pt = p_robot + cmd * h
            rows = pred_rows[np.isclose(pred_rows.h, h)]
            if len(rows):
                g = gap_to_box(pt, rows) - radius_robot - radius_person
                worst[i] = min(worst[i], g.min())
            if bounds is not None and not (bounds[0] + radius_robot < pt[0] < bounds[2] - radius_robot
                                           and bounds[1] + radius_robot < pt[1] < bounds[3] - radius_robot):
                ok[i] = False
            for o in obstacles:
                if np.hypot(pt[0] - o["x"], pt[1] - o["y"]) < o["r"] + radius_robot + margin:
                    ok[i] = False
    safe = ok & (worst >= margin)
    end = p_robot + cands * max(horizons)
    progress = np.linalg.norm(goal - p_robot) - np.linalg.norm(goal - end, axis=1)
    if safe.any():
        i = int(np.argmax(np.where(safe, progress, -np.inf)))
        fallback = False
    else:
        fallback = True
        best = int(np.argmax(np.where(ok, worst, -np.inf))) if ok.any() else 0
        i = 0 if (not ok[best] or worst[0] >= worst[best] - stop_bias) else best     # 0 = stop
    return cands[i], {"n_safe": int(safe.sum()), "fallback": fallback, "worst_gap": float(worst[i])}
