"""Robustness tests for the planning loop: planner, occupancy, conformal calibration, the closed
loop on the crowd simulator and the crowd's difficulty knobs. No TabPFN, short episodes.

Known bugs are kept as strict xfails (an XPASS means the bug was fixed: drop the marker)."""
import argparse

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from foresight.core import conformal as CF
from foresight.core import occupancy as O
from foresight.core import planner as P
from foresight.domains.crowd import CrowdConfig, CrowdSim

R_ROBOT, R_PERSON = 0.4, 0.3
SPEEDS = (0.0, 0.4, 0.8, 1.2)
BOUNDS = (-10.0, -10.0, 10.0, 10.0)
Z05 = norm.ppf(0.95)                     # occupancy's z for box_q = 0.05


def boxes(*people, horizons=(1.0, 2.0)):
    """Prediction table: people = (px, py, vx, vy, lo_a, hi_a, lo_c, hi_c[, theta]); one row per horizon."""
    rows = []
    for i, p in enumerate(people):
        theta = p[8] if len(p) > 8 else 0.0
        for h in horizons:
            rows.append(dict(agent_id=i + 1, px=p[0], py=p[1], vx=p[2], vy=p[3], h=h, theta=theta,
                             lo_a=p[4], hi_a=p[5], lo_c=p[6], hi_c=p[7]))
    cols = ["agent_id", "px", "py", "vx", "vy", "h", "theta", "lo_a", "hi_a", "lo_c", "hi_c"]
    return pd.DataFrame(rows, columns=cols)


def chosen_gaps(p, cmd, rows, horizons=(1.0, 2.0)):
    """Independent recomputation of the worst box gap (minus radii) along cmd."""
    out = []
    for h in horizons:
        r = rows[np.isclose(rows.h, h)]
        if len(r):
            out.append((P.gap_to_box(p + cmd * h, r) - R_ROBOT - R_PERSON).min())
    return min(out) if out else np.inf


# ======================================================================================
# 1. Planner
# ======================================================================================
@pytest.mark.parametrize("goal", [(8.0, 0.0), (6.0, 8.0), (-6.0, -6.0)])
def test_empty_scene_goes_straight_at_top_speed(goal):
    goal = np.asarray(goal, float)                         # goal farther than 2 s * top speed
    cmd, info = P.choose(np.zeros(2), goal, boxes(), speeds=SPEEDS, bounds=BOUNDS)
    np.testing.assert_allclose(cmd, max(SPEEDS) * goal / np.linalg.norm(goal), atol=1e-12)
    assert not info["fallback"] and info["n_safe"] == len(info["candidates"]["vx"])


def test_box_on_straight_path_vetoes_straight_and_keeps_gap():
    # stationary person 2.5 m ahead, box +-0.3 m: 1.2 m/s and 0.8 m/s straight ahead end in or near it
    rows = boxes((2.5, 0.0, 0.0, 0.0, -0.3, 0.3, -0.3, 0.3))
    p, goal, margin = np.zeros(2), np.array([8.0, 0.0]), 0.2
    cmd, info = P.choose(p, goal, rows, margin=margin, bounds=BOUNDS, speeds=SPEEDS)
    tab = pd.DataFrame(info["candidates"])
    straight = (np.isclose(tab.vy, 0) & np.isclose(tab.vx, 1.2))
    assert straight.sum() == 1 and tab.loc[straight, "veto_reason"].item() == "box"
    assert not info["fallback"]
    assert not np.allclose(cmd, [1.2, 0.0])
    assert info["worst_gap"] >= margin
    assert chosen_gaps(p, cmd, rows) >= margin - 1e-12          # recomputed, not trusted
    assert cmd[0] > 0                                           # still makes progress


def test_all_vetoed_by_short_check_stops():
    # someone standing on the robot: every candidate fails the 0.4 s check -> stop, fallback
    rows = boxes((0.1, 0.0, 0.0, 0.0, 0, 0, 0, 0))
    cmd, info = P.choose(np.zeros(2), np.array([8.0, 0.0]), rows, bounds=BOUNDS, speeds=SPEEDS)
    assert info["n_safe"] == 0 and info["fallback"]
    np.testing.assert_array_equal(cmd, [0.0, 0.0])
    assert set(info["candidates"]["veto_reason"]) == {"short check"}


def test_all_unsafe_equal_gaps_stops():
    # a box covering the whole plaza: every candidate has the same worst gap -> stop wins the tie
    rows = boxes((0.0, 3.0, 0.0, 0.0, -50, 50, -50, 50))
    cmd, info = P.choose(np.zeros(2), np.array([8.0, 0.0]), rows, bounds=BOUNDS, speeds=SPEEDS)
    assert info["fallback"] and info["n_safe"] == 0
    np.testing.assert_array_equal(cmd, [0.0, 0.0])


def test_all_unsafe_but_moving_clearly_better_moves_away():
    # margin too large for any candidate; person 1.5 m in front, goal behind: backing off keeps a
    # worst gap > stop's gap + stop_bias, so the fallback moves (to the largest worst gap)
    rows = boxes((1.5, 0.0, 0.0, 0.0, 0, 0, 0, 0))
    cmd, info = P.choose(np.zeros(2), np.array([-8.0, 0.0]), rows, margin=5.0, bounds=BOUNDS, speeds=SPEEDS)
    tab = pd.DataFrame(info["candidates"])
    assert info["fallback"] and info["n_safe"] == 0
    assert np.linalg.norm(cmd) > 0 and cmd[0] < 0
    ok = tab.veto_reason.isin(["none", "box"])
    assert np.isclose(info["worst_gap"], tab.worst_gap[ok].max())
    assert info["worst_gap"] > tab.worst_gap[0] + 0.15


def test_all_unsafe_small_gain_prefers_stop():
    # same, but stop_bias larger than any gain from moving: the robot stops
    rows = boxes((1.5, 0.0, 0.0, 0.0, 0, 0, 0, 0))
    cmd, info = P.choose(np.zeros(2), np.array([-8.0, 0.0]), rows, margin=5.0, bounds=BOUNDS, speeds=SPEEDS,
                         stop_bias=10.0)
    assert info["fallback"]
    np.testing.assert_array_equal(cmd, [0.0, 0.0])


def _random_scene(rng):
    p = rng.uniform(-9.3, 9.3, 2)
    goal = rng.uniform(-14, 14, 2)                          # may lie outside the plaza
    n = rng.integers(0, 5)
    people = []
    for _ in range(n):
        lo = -rng.uniform(0, 1.0, 2); hi = rng.uniform(0, 1.0, 2)
        people.append((*(p + rng.uniform(-4, 4, 2)), *rng.uniform(-1.2, 1.2, 2), lo[0], hi[0], lo[1], hi[1],
                       rng.uniform(-np.pi, np.pi)))
    k = rng.integers(0, 3)
    obst = [dict(x=float(x), y=float(y), r=float(r)) for (x, y), r in
            zip(p + rng.uniform(-3, 3, (k, 2)), rng.uniform(0.3, 1.0, k))]
    obst = [o for o in obst if np.hypot(o["x"] - p[0], o["y"] - p[1]) > o["r"] + R_ROBOT + 0.25]
    return p, goal, boxes(*people), obst


def test_random_scenes_bounds_obstacles_speed_and_gap():
    rng = np.random.default_rng(0)
    speeds = (0.0, 0.3, 0.6, 0.9)
    margin = 0.2
    for _ in range(150):
        p, goal, rows, obst = _random_scene(rng)
        cmd, info = P.choose(p, goal, rows, margin=margin, bounds=BOUNDS, obstacles=obst, speeds=speeds)
        assert np.linalg.norm(cmd) <= max(speeds) + 1e-12
        cands = P.candidates(p, goal, speeds=speeds)
        assert np.all(np.linalg.norm(cands, axis=1) <= max(speeds) + 1e-12)
        if np.allclose(cmd, 0):
            continue                                        # standing still is always allowed
        assert not info["fallback"] or info["worst_gap"] >= info["candidates"]["worst_gap"][0] + 0.15
        for h in (1.0, 2.0):
            q = p + cmd * h
            assert BOUNDS[0] + R_ROBOT < q[0] < BOUNDS[2] - R_ROBOT and BOUNDS[1] + R_ROBOT < q[1] < BOUNDS[3] - R_ROBOT
            for o in obst:
                assert np.hypot(q[0] - o["x"], q[1] - o["y"]) >= o["r"] + R_ROBOT + margin
        if not info["fallback"]:
            assert chosen_gaps(p, cmd, rows) >= margin - 1e-12


def test_wall_ahead_never_leaves_bounds():
    # robot 0.6 m from the right wall, goal beyond it: every forward move is vetoed by bounds
    p = np.array([BOUNDS[2] - 0.6, 0.0])
    cmd, info = P.choose(p, np.array([30.0, 0.0]), boxes(), bounds=BOUNDS, speeds=SPEEDS)
    for h in (1.0, 2.0):
        assert p[0] + cmd[0] * h < BOUNDS[2] - R_ROBOT
    tab = pd.DataFrame(info["candidates"])
    assert (tab.veto_reason[(tab.vx > 0.1) & (np.abs(tab.vy) < 1e-9)] == "bounds").all()


def test_obstacle_on_straight_path_is_avoided():
    obst = [dict(x=2.0, y=0.0, r=0.6)]
    p, goal = np.zeros(2), np.array([8.0, 0.0])
    cmd, info = P.choose(p, goal, boxes(), bounds=BOUNDS, obstacles=obst, speeds=SPEEDS)
    assert not np.allclose(cmd, [1.2, 0.0])
    for h in (1.0, 2.0):
        q = p + cmd * h
        assert np.hypot(q[0] - 2.0, q[1]) >= 0.6 + R_ROBOT + 0.2


def _brute_box_distance(pt, cx, cy, theta, lo_a, hi_a, lo_c, hi_c, n=801):
    """Distance from pt to a densely sampled rotated rectangle (oracle independent of gap_to_box)."""
    a, b = np.meshgrid(np.linspace(lo_a, hi_a, n), np.linspace(lo_c, hi_c, n))
    c, s = np.cos(theta), np.sin(theta)
    x, y = cx + c * a - s * b, cy + s * a + c * b               # person frame -> world
    return np.hypot(x - pt[0], y - pt[1]).min()


def test_gap_to_box_rotated_analytic_and_oracle():
    # box centre (1, 2) at h = 0 (cv point = px, py); theta = 90 deg: a runs along +y, c along -x
    r = boxes((1.0, 2.0, 0.0, 0.0, -1.0, 1.0, -0.5, 0.5, np.pi / 2), horizons=(0.0,))
    assert P.gap_to_box((1.0, 2.0), r)[0] == 0.0                # centre
    assert P.gap_to_box((1.3, 2.9), r)[0] == 0.0                # inside: |dx| < 0.5, |dy| < 1
    assert np.isclose(P.gap_to_box((3.0, 2.0), r)[0], 1.5)       # 2 m right of centre, half-width 0.5
    assert np.isclose(P.gap_to_box((1.0, 5.0), r)[0], 2.0)       # 3 m up, half-length 1
    assert np.isclose(P.gap_to_box((3.0, 5.0), r)[0], np.hypot(1.5, 2.0))   # corner region
    # cv displacement: px + vx h
    r2 = boxes((0.0, 0.0, 1.0, 0.5, -0.2, 0.2, -0.1, 0.1, 0.0), horizons=(2.0,))
    assert P.gap_to_box((2.0, 1.0), r2)[0] == 0.0
    # random asymmetric rotated boxes against a brute-force oracle
    rng = np.random.default_rng(1)
    for _ in range(25):
        lo_a, lo_c = -rng.uniform(0, 1.5, 2); hi_a, hi_c = rng.uniform(0, 1.5, 2)
        cx, cy, th = rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-np.pi, np.pi)
        pt = rng.uniform(-6, 6, 2)
        rr = boxes((cx, cy, 0.0, 0.0, lo_a, hi_a, lo_c, hi_c, th), horizons=(0.0,))
        got = P.gap_to_box(pt, rr)[0]
        oracle = _brute_box_distance(pt, cx, cy, th, lo_a, hi_a, lo_c, hi_c)
        inside = oracle < 0.01 and got == 0.0
        assert inside or abs(got - oracle) < 5e-3, (got, oracle)


# ======================================================================================
# 2. Occupancy
# ======================================================================================
def occ_row(px, py, width_a, width_c=None, theta=0.0, h=1.0, vx=0.0, vy=0.0):
    wc = width_a if width_c is None else width_c
    return boxes((px, py, vx, vy, -width_a / 2, width_a / 2, -wc / 2, wc / 2, theta), horizons=(h,))


def test_occupancy_range_and_empty():
    rng = np.random.default_rng(2)
    people = [(*rng.uniform(-9, 9, 2), *rng.uniform(-1, 1, 2), -rng.uniform(0, 2), rng.uniform(0, 2),
               -rng.uniform(0, 2), rng.uniform(0, 2), rng.uniform(-3, 3)) for _ in range(40)]
    Pm, xs, ys = O.occupancy(boxes(*people), 1.0, BOUNDS)
    assert Pm.shape == (len(ys), len(xs)) and np.all((Pm >= 0) & (Pm <= 1))
    E, _, _ = O.occupancy(boxes(), 1.0, BOUNDS)
    assert E.shape == Pm.shape and not E.any()
    W, _, _ = O.occupancy(boxes(*people, horizons=(2.0,)), 1.0, BOUNDS)     # no rows at h = 1
    assert not W.any()


def test_occupancy_mass_is_one_on_fine_grid():
    # sd = 0.5 m per axis (box = +-z sd), rotated, centre off the grid, far from the walls
    sd = 0.5
    Pm, _, _ = O.occupancy(occ_row(0.3, -0.2, 2 * Z05 * sd, theta=0.7), 1.0, BOUNDS, res=0.1)
    assert abs(Pm.sum() - 1.0) < 0.05
    # moving person: the mass follows the cv point and stays ~1
    Pm2, xs, ys = O.occupancy(occ_row(-3.0, 1.0, 2 * Z05 * sd, vx=1.0, vy=0.5, h=2.0), 2.0, BOUNDS, res=0.1)
    assert abs(Pm2.sum() - 1.0) < 0.05
    iy, ix = np.unravel_index(Pm2.argmax(), Pm2.shape)
    assert abs(xs[ix] - (-1.0)) <= 0.1 and abs(ys[iy] - 2.0) <= 0.1


def test_wider_box_spreads_probability():
    narrow, _, _ = O.occupancy(occ_row(0.3, 0.3, 2 * Z05 * 0.4), 1.0, BOUNDS, res=0.1)
    wide, _, _ = O.occupancy(occ_row(0.3, 0.3, 2 * Z05 * 0.8), 1.0, BOUNDS, res=0.1)
    assert narrow.max() < 0.99 and wide.max() < 0.99          # neither peak hits the cap
    assert wide.max() < narrow.max()
    assert abs(wide.sum() - narrow.sum()) < 0.05
    assert (wide > 1e-3).sum() > (narrow > 1e-3).sum()


def test_occupancy_mass_does_not_depend_on_subcell_position():
    # fixed in f4e103c: narrow boxes are supersampled within each cell
    # cv model -> zero-width box -> sd = min_sd = 0.05 m; default grid res = 0.25 (cell centres at 0.125 + 0.25 k)
    masses = [O.occupancy(occ_row(0.125 + d, 0.125 + d, 0.0), 1.0, BOUNDS)[0].sum() for d in (0.0, 0.06, 0.125)]
    assert np.allclose(masses, 1.0, atol=0.1), masses


def test_truth_grid_cells():
    xs, ys, res = O.grid(BOUNDS, 0.25)
    T = O.truth_grid([(0.1, -0.1), (9.99, 9.99), (12.0, 0.0)], xs, ys, res)
    assert T.sum() == 2                                          # out-of-bounds point dropped
    iy, ix = np.nonzero(T)
    pts = set(zip(xs[ix].round(4), ys[iy].round(4)))
    assert (0.125, -0.125) in pts and (9.875, 9.875) in pts


def test_scores_perfect_prediction_and_oracle_auc():
    rng = np.random.default_rng(3)
    T = (rng.random((20, 30)) < 0.1).astype(float)
    brier, auc = O.scores(T.copy(), T)
    assert brier == 0.0 and auc == 1.0
    Pm = rng.random(T.shape)                                     # continuous: no ties
    _, auc = O.scores(Pm, T)
    p, t = Pm.ravel(), T.ravel()
    oracle = (p[t == 1][:, None] > p[t == 0][None, :]).mean()
    assert np.isclose(auc, oracle)
    assert np.isnan(O.scores(Pm, np.zeros_like(T))[1])            # no positives -> undefined


def test_scores_partial_ties_match_oracle():
    # realistic grid: most cells exactly 0 (positives among them), a few distinct values
    rng = np.random.default_rng(11)
    Pm = np.where(rng.random((30, 40)) < 0.8, 0.0, rng.choice([0.2, 0.5, 0.9], (30, 40)))
    T = (rng.random(Pm.shape) < 0.05 + 0.3 * (Pm > 0)).astype(float)
    _, auc = O.scores(Pm, T)
    p, t = Pm.ravel(), T.ravel()
    pos, neg = p[t == 1][:, None], p[t == 0][None, :]
    oracle = (pos > neg).mean() + 0.5 * (pos == neg).mean()
    assert np.isclose(auc, oracle)


def test_scores_constant_prediction_auc_half():
    # regression: ranks from argsort(argsort(.)) did not average ties and gave AUC 0.0 here
    T = np.zeros((4, 4)); T[1, 1] = T[2, 3] = 1
    brier, auc = O.scores(np.full((4, 4), 0.3), T)
    assert np.isclose(brier, np.mean((0.3 - T) ** 2))
    assert auc == 0.5


# ======================================================================================
# 3. Conformal
# ======================================================================================
def synth(n, sd, rng, width=1.0):
    lo = np.full((n, 2), -width); hi = np.full((n, 2), width)
    return lo, hi, rng.normal(0, 1, (n, 2)) * np.asarray(sd, float).reshape(-1, 1)


def _expected_offsets(t, t_known, miss, d0, alpha, eta):
    """d_used[i] = d0 + eta * sum(miss_j - alpha) over outcomes with t_known_j <= t_i."""
    order = np.argsort(t_known, kind="stable")
    cum = np.concatenate([[0.0], np.cumsum(miss[order].astype(float) - alpha)])
    k = np.searchsorted(t_known[order], t, side="right")
    return d0 + eta * cum[k]


def test_online_offsets_converge_to_alpha_stationary():
    rng = np.random.default_rng(4)
    n, alpha = 4000, 0.2
    lo, hi, Y = synth(n, 1.5, rng)                               # box +-1 m vs sd 1.5: ~75% misses at d = 0
    t = np.arange(n) * 0.5; t_known = t + 2.0
    d, miss = CF.online_offsets(t, t_known, np.zeros(n), lo, hi, Y, d0=0.0, alpha=alpha, eta=0.02)
    assert miss[:100].mean() > 0.5                              # badly miscalibrated at the start
    assert abs(miss.mean() - alpha) < 0.03
    assert abs(miss[1000:].mean() - alpha) < 0.03
    np.testing.assert_allclose(d, _expected_offsets(t, t_known, miss, 0.0, alpha, 0.02), atol=1e-9)
    np.testing.assert_array_equal(miss, CF.scores(lo, hi, Y) > d)


def test_online_offsets_recover_after_scale_shift():
    rng = np.random.default_rng(5)
    n, alpha = 6000, 0.2
    sd = np.where(np.arange(n) < 3000, 1.0, 2.5)
    lo, hi, Y = synth(n, sd, rng)
    t = np.arange(n) * 0.5; t_known = t + 2.0
    d, miss = CF.online_offsets(t, t_known, np.zeros(n), lo, hi, Y, d0=0.0, alpha=alpha, eta=0.02)
    assert miss[3000:3200].mean() > alpha + 0.1                  # the shift hurts at first
    assert abs(miss[4000:].mean() - alpha) < 0.03                 # ... then recovers
    assert d[-1] > d[2999] + 0.5                                  # by widening


def test_online_offsets_causal():
    rng = np.random.default_rng(6)
    n, d0 = 400, 0.3
    lo, hi, Y = synth(n, 1.0, rng)
    t = np.arange(n) * 0.5
    # every outcome known far in the future: no prediction may move
    d, _ = CF.online_offsets(t, t + 1e6, np.zeros(n), lo, hi, Y, d0=d0)
    np.testing.assert_array_equal(d, d0)
    # perturbing one outcome changes nothing before its t_known, and (miss flips) something after
    t_known = t + 2.0
    j = 50
    base, _ = CF.online_offsets(t, t_known, np.zeros(n), lo, hi, Y, d0=d0)
    Y2 = Y.copy(); Y2[j] = 0.0 if CF.scores(lo, hi, Y)[j] > base[j] else 50.0
    pert, _ = CF.online_offsets(t, t_known, np.zeros(n), lo, hi, Y2, d0=d0)
    before = t < t_known[j]
    np.testing.assert_array_equal(pert[before], base[before])
    assert not np.allclose(pert[~before], base[~before])
    # boundary: an outcome known exactly at a prediction's time is applied first
    tt = np.array([0.0, 1.0]); tk = np.array([1.0, 2.0])
    lo2, hi2 = np.zeros((2, 2)), np.ones((2, 2)); Yb = np.array([[5.0, 5.0], [0.5, 0.5]])   # row 0 misses
    d2, m2 = CF.online_offsets(tt, tk, np.zeros(2), lo2, hi2, Yb, d0=0.0, alpha=0.2, eta=0.1)
    assert m2[0] and np.isclose(d2[1], 0.0 + 0.1 * (1 - 0.2))


def test_online_offsets_restart_per_group():
    rng = np.random.default_rng(7)
    n = 600
    lo, hi, Y = synth(n, 1.5, rng)
    t = np.tile(np.arange(n // 2) * 0.5, 2); group = np.repeat([0, 1], n // 2)
    d, _ = CF.online_offsets(t, t + 1.0, group, lo, hi, Y, d0=0.1)
    assert d[0] == 0.1 and d[n // 2] == 0.1
    assert d[n // 2 - 1] != 0.1


def test_split_offset_coverage_matches_theory():
    rng = np.random.default_rng(8)
    n, alpha, reps, m = 19, 0.2, 2000, 200
    k = int(np.ceil((n + 1) * (1 - alpha)))                      # rank of the chosen score (1-based)
    cov = np.empty(reps)
    for r in range(reps):
        lo, hi, Y = synth(n + m, 1.3, rng)
        dd = CF.split_offset(lo[:n], hi[:n], Y[:n], alpha=alpha)
        cov[r] = (CF.scores(lo[n:], hi[n:], Y[n:]) <= dd).mean()
    assert abs(cov.mean() - k / (n + 1)) < 0.01
    assert cov.mean() >= 1 - alpha - 0.006
    # in-sample: at least k of the n calibration scores fall inside the widened box
    lo, hi, Y = synth(500, 1.3, rng)
    dd = CF.split_offset(lo, hi, Y, alpha=alpha)
    assert (CF.scores(lo, hi, Y) <= dd).sum() >= int(np.ceil(501 * (1 - alpha)))


@pytest.mark.xfail(strict=True, reason="BUG: split_offset clips the conformal rank to n-1; for n < (1-alpha)/alpha "
                                       "(n=3 at alpha=0.2) it returns max(s) and covers 3/4 < 1-alpha "
                                       "(the finite-sample rule asks for an infinite offset)")
def test_split_offset_tiny_calibration_still_covers():
    rng = np.random.default_rng(9)
    n, alpha, reps, m = 3, 0.2, 3000, 100
    cov = np.empty(reps)
    for r in range(reps):
        lo, hi, Y = synth(n + m, 1.3, rng)
        dd = CF.split_offset(lo[:n], hi[:n], Y[:n], alpha=alpha)
        cov[r] = (CF.scores(lo[n:], hi[n:], Y[n:]) <= dd).mean()
    assert cov.mean() >= 1 - alpha - 0.01


def test_widen_keeps_order():
    rng = np.random.default_rng(10)
    lo = rng.normal(0, 1, (300, 2)); hi = lo + rng.uniform(0, 2, (300, 2))
    for d in (0.3, -0.4, -5.0, rng.normal(0, 1.5, 300)):
        a, b = CF.widen(lo, hi, d)
        assert a.shape == lo.shape and np.all(b >= a)
    a, b = CF.widen(lo, hi, 0.25)                                # non-negative offset: exact widening
    np.testing.assert_allclose(a, lo - 0.25); np.testing.assert_allclose(b, hi + 0.25)
    dv = rng.uniform(0, 1, 300)
    a, b = CF.widen(lo, hi, dv)
    np.testing.assert_allclose(a, lo - dv[:, None]); np.testing.assert_allclose(b, hi + dv[:, None])


# ======================================================================================
# 4. Closed loop, end to end (no TabPFN: cv, stop_go, reactive, vo)
# ======================================================================================
PLANNERS = ["cv", "stop_go", "reactive", "vo"]
E2E_SEEDS = [1000, 1001, 1002]                                   # seed % 3 -> calm, hurried, erratic
TIMING_KEYS = {"pred_seconds", "pred_ms_per_plan"}
METRIC_KEYS = {"seed", "reached", "time_to_goal", "time_capped", "collision_steps", "collided",
               "moving_collision_steps", "min_gap", "near_miss", "freeze_rate", "seconds_stopped",
               "intrusion_s", "path_length", "path_efficiency", "people_decel", "episode_s", "fallback_rate",
               "pred_seconds", "pred_ms_per_plan"}


def loop_args(**kw):
    """16 people (the default): with ~6 the robot never comes within 0.5 m, and the frame
    cross-checks below would compare zeros."""
    a = dict(sim="crowd", duration=8.0, people=16, replan=0.4, refit=2.0, margin=0.2, noise=0.1, box_q=0.05,
             hz=(1.0, 2.0), switch=False, max_speed=1.0, vo_w=1.0, vo_h=5.0, context=1000, tabpfn_fast=1,
             tabpfn_est=1, threads=0, bank=None, tag="", log=False, robot_repulsion=2.0, ignore_robot=0.0)
    a.update(kw)
    return argparse.Namespace(**a)


@pytest.fixture(scope="module")
def RC(tmp_path_factory):
    """run_closed with trace.write disabled and the cwd in a temp dir: nothing lands in results/."""
    import run_closed
    from foresight import trace
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(trace, "write", lambda *a, **k: None)
        mp.chdir(tmp_path_factory.mktemp("closed"))
        yield run_closed


def episode(RC, kind, seed, args):
    return RC.run_episode(RC.Predictor(kind, None, None, args), seed, args, log=True)


@pytest.fixture(scope="module")
def runs(RC):
    args = loop_args()
    return {(k, s): episode(RC, k, s, args) for k in PLANNERS for s in E2E_SEEDS}


@pytest.mark.parametrize("kind", PLANNERS)
def test_closed_loop_metrics_valid_and_consistent(RC, runs, kind):
    args = loop_args()
    for seed in E2E_SEEDS:
        res, logd, rf = runs[(kind, seed)]
        assert METRIC_KEYS <= set(res)
        assert res["seed"] == seed
        assert isinstance(res["reached"], bool) and isinstance(res["collided"], bool)
        nan_ok = {"time_to_goal", "path_efficiency"}
        for key in METRIC_KEYS - nan_ok - {"reached", "collided", "near_miss"}:
            assert np.isfinite(res[key]), (key, res[key])
        assert np.isnan(res["time_to_goal"]) == (not res["reached"])
        assert np.isnan(res["path_efficiency"]) == (not res["reached"])
        assert 0 < res["episode_s"] <= args.duration + 1e-9
        assert res["time_capped"] == (res["time_to_goal"] if res["reached"] else args.duration)
        assert 0 <= res["freeze_rate"] <= 1 and 0 <= res["fallback_rate"] <= 1
        assert 0 <= res["seconds_stopped"] <= res["episode_s"] + 1e-9
        assert 0 <= res["intrusion_s"] <= res["episode_s"] + 1e-9
        assert res["collided"] == (res["collision_steps"] > 0)
        assert 0 <= res["moving_collision_steps"] <= res["collision_steps"]
        assert res["near_miss"] == (res["min_gap"] < 0.2)
        assert res["path_length"] >= 0 and res["people_decel"] >= 0
        # recompute from the logged frames
        frames, dt = logd["frames"], 0.2
        assert np.isclose(len(frames) * dt, res["episode_s"])
        sim, _, _ = RC.make_sim("crowd", seed, args)
        states = [sim.reset()] + [f["state"] for f in frames]
        xy = np.array([[s[0]["x"], s[0]["y"]] for s in states])
        assert np.isclose(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum(), res["path_length"])
        gaps = np.array([[np.hypot(r["x"] - s[0]["x"], r["y"] - s[0]["y"]) - r["radius"] - s[0]["radius"]
                          for r in s[1:]] for s in states[1:]])
        assert np.isclose(gaps.min(), res["min_gap"])
        assert np.isclose((gaps < 0.5).any(axis=1).sum() * dt, res["intrusion_s"])
        v = np.array([np.hypot(s[0]["vx"], s[0]["vy"]) for s in states[1:]])
        assert np.isclose((v < 0.05).sum() * dt, res["seconds_stopped"])
        assert sum(f["info"]["collided"] for f in frames) == res["collision_steps"]


def test_closed_loop_crosschecks_not_vacuous(runs):
    # the recomputation asserts above only mean something if close encounters happen
    assert sum(r[0]["intrusion_s"] for r in runs.values()) > 0
    assert min(r[0]["min_gap"] for r in runs.values()) < 0.5


@pytest.mark.parametrize("kind", PLANNERS)
def test_closed_loop_deterministic(RC, runs, kind):
    seed = E2E_SEEDS[1]
    res2, logd2, _ = episode(RC, kind, seed, loop_args())
    res1, logd1, _ = runs[(kind, seed)]
    np.testing.assert_equal({k: v for k, v in res1.items() if k not in TIMING_KEYS},
                            {k: v for k, v in res2.items() if k not in TIMING_KEYS})
    xy = lambda L: np.array([[f["state"][0]["x"], f["state"][0]["y"]] for f in L["frames"]])
    np.testing.assert_array_equal(xy(logd1), xy(logd2))


def _robot_speeds(logd, seed, RC, args, dt=0.2):
    sim, _, _ = RC.make_sim("crowd", seed, args)
    states = [sim.reset()] + [f["state"] for f in logd["frames"]]
    xy = np.array([[s[0]["x"], s[0]["y"]] for s in states])
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    vel = np.array([np.hypot(s[0]["vx"], s[0]["vy"]) for s in states[1:]])
    return step, vel


@pytest.mark.parametrize("kind", ["cv", "reactive", "vo"])
def test_robot_never_faster_than_max_speed_default(RC, runs, kind):
    for seed in E2E_SEEDS:
        step, vel = _robot_speeds(runs[(kind, seed)][1], seed, RC, loop_args())
        assert step.max() <= 1.0 + 1e-9 and vel.max() <= 1.0 + 1e-9


@pytest.mark.parametrize("kind", [
    "cv", "reactive", "vo",
    "stop_go",       # fixed in 9f04ab1: make_sim forwards --max-speed to the simulator's own policy
])
def test_robot_never_faster_than_max_speed_nondefault(RC, kind):
    args = loop_args(max_speed=0.5, duration=4.0)
    res, logd, _ = episode(RC, kind, 1001, args)
    step, vel = _robot_speeds(logd, 1001, RC, args)
    assert step.max() <= 0.5 + 1e-9 and vel.max() <= 0.5 + 1e-9


# ======================================================================================
# 5. Difficulty knobs (CrowdSim directly; see notes on run_closed.make_sim pass-through)
# ======================================================================================
KNOB_SEEDS = [0, 1, 2, 3, 4]


def _standing_robot_min_gap(seed, steps=100, **knobs):
    """Robot stands still in a small plaza (half_size 5, no pillars); mean over steps of the gap to
    the nearest person. Returns (mean min gap, initial positions)."""
    cfg = CrowdConfig(n_people=10, duration=steps * 0.2, half_size=5.0, n_obstacles=0, **knobs)
    sim = CrowdSim(seed, cfg)
    s0 = sim.reset()
    init = np.array([[r["x"], r["y"]] for r in s0])
    gaps = []
    for _ in range(steps):
        sim.step((0.0, 0.0))
        d = np.hypot(*(sim.P[1:] - sim.P[0]).T) - sim.radius[1:] - sim.radius[0]
        gaps.append(d.min())
    return float(np.mean(gaps)), init


def test_ignore_robot_initial_state_unchanged_and_people_come_closer():
    near, far = [], []
    for seed in KNOB_SEEDS:
        g_ign, i_ign = _standing_robot_min_gap(seed, ignore_robot=1.0)
        g_obs, i_obs = _standing_robot_min_gap(seed, ignore_robot=0.0)
        np.testing.assert_array_equal(i_ign, i_obs)              # separate random stream
        near.append(g_ign); far.append(g_obs)
    assert np.mean(near) < np.mean(far), (near, far)


def test_ignore_robot_flags_drawn_per_share():
    sim = CrowdSim(3, CrowdConfig(n_people=200, ignore_robot=0.3)); sim.reset()
    assert not sim.ignores_robot[0]
    assert 0.2 < sim.ignores_robot[1:].mean() < 0.4
    sim = CrowdSim(3, CrowdConfig(n_people=20, ignore_robot=1.0)); sim.reset()
    assert sim.ignores_robot[1:].all()


def test_robot_repulsion_weaker_lets_people_closer():
    weak, strong = [], []
    for seed in KNOB_SEEDS:
        g_w, i_w = _standing_robot_min_gap(seed, robot_repulsion=0.5)
        g_s, i_s = _standing_robot_min_gap(seed, robot_repulsion=2.0)
        np.testing.assert_array_equal(i_w, i_s)
        weak.append(g_w); strong.append(g_s)
    assert np.mean(weak) < np.mean(strong), (weak, strong)


def test_knob_validation():
    with pytest.raises(ValueError):
        CrowdSim(0, CrowdConfig(ignore_robot=1.5))
    with pytest.raises(ValueError):
        CrowdSim(0, CrowdConfig(robot_repulsion=-1.0))


def test_make_sim_forwards_difficulty_knobs(RC):
    args = loop_args(robot_repulsion=0.5, ignore_robot=1.0)
    sim, _, _ = RC.make_sim("crowd", 1000, args)
    assert sim.cfg.robot_repulsion == 0.5 and sim.cfg.ignore_robot == 1.0
    sim.reset()
    assert sim.ignores_robot[1:].all()
    base, _, _ = RC.make_sim("crowd", 1000, loop_args())
    assert base.cfg.robot_repulsion == CrowdConfig().robot_repulsion and base.cfg.ignore_robot == 0.0
