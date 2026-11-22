"""TabPFN picks the move (exploratory, docs/TABPFN_POLICY.md).

Every candidate move of planner.candidates() at one replan becomes one row. A TabPFN classifier, fitted once on
rows logged from the teacher (the box planner choose() driving tabpfn_split), scores P(teacher picks this
candidate); the policy takes the highest-scoring candidate that passes the hard checks (the 0.4 s constant-velocity
check, and staying inside the area). How often the hard checks override the policy's first choice is counted.

Row (FEATURES): robot state relative to the goal (distance, speed, heading of its motion relative to the goal
direction), the candidate's speed and turn relative to the goal direction, its clearance (gap minus both radii) to
the nearest person's predicted box at +1 s and +2 s and to the constant-velocity point at the 0.4 s short check,
the smallest of those, its progress toward the goal over 2 s and that progress's rank among the candidates, whether
it stays in the area, and the number of people within 3 m of the robot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from foresight.core import planner as P

FEATURES = ["dist_goal", "robot_speed", "robot_head_rel", "cand_speed", "cand_turn", "gap_short", "gap1", "gap2",
            "gap_min", "progress", "progress_rank", "in_bounds", "n_near"]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def candidate_rows(p_robot, v_robot, goal, cands, pred_rows, bounds, horizons=(1.0, 2.0), short_h=0.4,
                   radius_robot=0.4, radius_person=0.3):
    """One feature row per candidate (same order as cands)."""
    g = np.asarray(goal, float) - p_robot
    base = np.arctan2(g[1], g[0])
    rs = float(np.hypot(*v_robot))
    head_rel = float(_wrap(np.arctan2(v_robot[1], v_robot[0]) - base)) if rs > 0.05 else 0.0
    sp = np.hypot(cands[:, 0], cands[:, 1])
    turn = np.where(sp > 1e-6, _wrap(np.arctan2(cands[:, 1], cands[:, 0]) - base), 0.0)
    n = len(cands)
    gs, g1, g2 = np.full(n, 10.0), np.full(n, 10.0), np.full(n, 10.0)
    now = pred_rows.drop_duplicates("agent_id") if len(pred_rows) else pred_rows
    inb = np.ones(n, bool)
    for i, cmd in enumerate(cands):
        if len(now):
            pt = p_robot + cmd * short_h
            d = np.hypot(now.px.to_numpy() + now.vx.to_numpy() * short_h - pt[0],
                         now.py.to_numpy() + now.vy.to_numpy() * short_h - pt[1]) - radius_robot - radius_person
            gs[i] = d.min()
        for h, arr in zip(horizons, (g1, g2)):
            pt = p_robot + cmd * h
            rows = pred_rows[np.isclose(pred_rows.h, h)] if len(pred_rows) else pred_rows
            if len(rows):
                arr[i] = (P.gap_to_box(pt, rows) - radius_robot - radius_person).min()
            if bounds is not None and not (bounds[0] + radius_robot < pt[0] < bounds[2] - radius_robot
                                           and bounds[1] + radius_robot < pt[1] < bounds[3] - radius_robot):
                inb[i] = False
    end = p_robot + cands * max(horizons)
    progress = np.linalg.norm(goal - p_robot) - np.linalg.norm(goal - end, axis=1)
    rank = (-progress).argsort().argsort() / max(n - 1, 1)               # 0 = most progress
    n_near = int((np.hypot(now.px.to_numpy() - p_robot[0], now.py.to_numpy() - p_robot[1]) < 3.0).sum()) if len(now) else 0
    return pd.DataFrame(dict(dist_goal=float(np.linalg.norm(g)), robot_speed=rs, robot_head_rel=head_rel,
                             cand_speed=sp, cand_turn=turn, gap_short=np.clip(gs, -2, 10), gap1=np.clip(g1, -2, 10),
                             gap2=np.clip(g2, -2, 10), gap_min=np.clip(np.minimum(np.minimum(gs, g1), g2), -2, 10),
                             progress=progress, progress_rank=rank, in_bounds=inb.astype(float), n_near=float(n_near)))


class Policy:
    """TabPFN classifier over candidate rows, fitted once on logged teacher rows (all positives, negatives
    subsampled at neg_ratio per positive, seed 0)."""

    def __init__(self, ctx_paths, neg_ratio=3.0, seed=0, device=None):
        from tabpfn import TabPFNClassifier
        from foresight.core.models import best_device
        D = pd.concat([pd.read_csv(p) for p in ctx_paths], ignore_index=True)
        pos, neg = D[D.chosen == 1], D[D.chosen == 0]
        k = min(len(neg), int(round(neg_ratio * len(pos))))
        neg = neg.sample(k, random_state=seed)
        self.ctx = pd.concat([pos, neg], ignore_index=True)
        self.n_pos, self.n_neg, self.n_all = len(pos), len(neg), len(D)
        self.clf = TabPFNClassifier(device=device or best_device(), fit_mode="fit_with_cache",
                                    ignore_pretraining_limits=True)
        self.clf.fit(self.ctx[FEATURES].to_numpy(float), self.ctx.chosen.to_numpy(int))

    def scores(self, rows):
        return self.clf.predict_proba(rows[FEATURES].to_numpy(float))[:, 1]

    def choose(self, rows, short_min=0.05):
        """Highest-scoring candidate that passes the hard checks. Returns (index, overridden, policy_top)."""
        s = self.scores(rows)
        order = np.argsort(-s)
        top = int(order[0])
        ok = (rows.gap_short.to_numpy() >= short_min) & (rows.in_bounds.to_numpy() > 0.5)
        for i in order:
            if ok[i]:
                return int(i), int(i) != top, top
        return 0, top != 0, top                                          # nothing passes: stop (candidate 0)
