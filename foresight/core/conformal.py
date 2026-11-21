"""Conformal calibration of the 2D box (S5 in docs/GATES.md). Works on any model's quantiles.

Box = [lo, hi] per axis in the agent frame. Conformity score of a row (conformalised quantile
regression, Romano et al. 2019, taken jointly over both axes):

    s = max over axes of max(lo - y, y - hi)          (negative = inside, by that margin)

The box widened by d on every side, [lo - d, hi + d], holds the truth exactly when s <= d.

split_offset():   d from a held-out calibration slice of the TRAINING scene: the (1 - alpha)
                  quantile of s, with the finite-sample correction. That is all a robot has on
                  arrival in a new place (calibrating on the test scene would be leakage).
online_offsets(): adaptive conformal in the new scene (Gibbs & Candes 2021, additive form):
                  d moves after each row whose outcome is now known:
                      d <- d + eta * (miss - alpha)
                  so long-run miss rate -> alpha under drift. Strictly causal: a prediction at
                  time t uses only outcomes with t_known <= t.
"""
from __future__ import annotations

import numpy as np


def scores(lo, hi, Y):
    """lo, hi, Y: [n, 2]. Joint conformity score per row."""
    return np.max(np.maximum(lo - Y, Y - hi), axis=1)


def split_offset(lo, hi, Y, alpha=0.2):
    s = np.sort(scores(lo, hi, Y))
    n = len(s)
    k = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)
    return float(s[k])


def online_offsets(t, t_known, group, lo, hi, Y, d0, alpha=0.2, eta=0.02):
    """Offsets d used for each row's prediction (made at time t), updated causally.

    t, t_known, group: [n] arrays (group = episode: offsets restart at d0 in each episode).
    Returns d_used [n] and miss [n] (truth outside the box widened by d_used)."""
    n = len(t)
    d_used = np.full(n, np.nan)
    miss = np.zeros(n, bool)
    s = scores(lo, hi, Y)
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        # events: (time, kind, row); kind 0 = outcome becomes known (applied first), 1 = predict
        ev = sorted([(t_known[i], 0, i) for i in idx] + [(t[i], 1, i) for i in idx])
        d = d0
        for _, kind, i in ev:
            if kind == 1:
                d_used[i] = d
                miss[i] = s[i] > d
            else:
                d = d + eta * (float(miss[i]) - alpha)
    return d_used, miss


def widen(lo, hi, d):
    """Apply offsets d (scalar or [n]); keeps hi >= lo."""
    d = np.asarray(d, float).reshape(-1, 1) if np.ndim(d) else d
    lo2, hi2 = lo - d, hi + d
    mid = (lo2 + hi2) / 2
    return np.minimum(lo2, mid), np.maximum(hi2, mid)
