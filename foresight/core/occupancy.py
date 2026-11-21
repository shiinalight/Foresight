"""Occupancy at +h from per-person predicted ranges (S6 in docs/GATES.md). One function, used by
both the metric and the renderer.

Each person's predicted position at +h is approximated, in their own frame, by a Gaussian per
axis: centre = constant-velocity point + median deviation, sd = (q90 - q10) / 2.563 (taken from the
box columns lo/hi when only those exist: sd = (hi - lo) / (2 * z), z from box_q). The probability
that a grid cell holds the person's centre is the Gaussian mass over the cell (density x area,
capped at 1); cells combine over people as 1 - prod(1 - p_j).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def grid(bounds, res=0.25):
    x0, y0, x1, y1 = bounds
    xs = np.arange(x0 + res / 2, x1, res); ys = np.arange(y0 + res / 2, y1, res)
    return xs, ys, res


def occupancy(rows, h, bounds, res=0.25, box_q=0.05, min_sd=0.05):
    """rows: predicted rows (px, py, vx, vy, h, theta, lo_a, hi_a, lo_c, hi_c) at one moment.
    Returns (P [ny, nx], xs, ys)."""
    xs, ys, res = grid(bounds, res)
    X, Y = np.meshgrid(xs, ys)
    free = np.ones_like(X)
    z = norm.ppf(1 - box_q)
    r = rows[np.isclose(rows.h, h)] if len(rows) else rows
    for _, p in r.iterrows():
        mu_a, mu_c = (p.lo_a + p.hi_a) / 2, (p.lo_c + p.hi_c) / 2
        sd_a = max((p.hi_a - p.lo_a) / (2 * z), min_sd); sd_c = max((p.hi_c - p.lo_c) / (2 * z), min_sd)
        cx, cy = p.px + p.vx * p.h, p.py + p.vy * p.h
        c, s = np.cos(p.theta), np.sin(p.theta)
        dx, dy = X - cx, Y - cy
        a, b = c * dx + s * dy, -s * dx + c * dy                 # into the person's frame
        dens = np.exp(-0.5 * (((a - mu_a) / sd_a) ** 2 + ((b - mu_c) / sd_c) ** 2)) / (2 * np.pi * sd_a * sd_c)
        free *= 1 - np.clip(dens * res * res, 0, 1)
    return 1 - free, xs, ys


def truth_grid(true_xy, xs, ys, res):
    """1 in every cell holding a true future centre."""
    T = np.zeros((len(ys), len(xs)))
    for x, y in true_xy:
        i, j = int((y - (ys[0] - res / 2)) // res), int((x - (xs[0] - res / 2)) // res)
        if 0 <= i < len(ys) and 0 <= j < len(xs):
            T[i, j] = 1
    return T


def scores(P, T):
    """Brier score and ROC AUC of occupancy P against truth T (both [ny, nx])."""
    p, t = P.ravel(), T.ravel()
    brier = float(np.mean((p - t) ** 2))
    pos, neg = p[t == 1], p[t == 0]
    if len(pos) == 0 or len(neg) == 0:
        return brier, np.nan
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    auc = (ranks[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))
    return brier, float(auc)
