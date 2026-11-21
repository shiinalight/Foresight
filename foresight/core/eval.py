"""Metrics for quantile predictions of residual targets."""
from __future__ import annotations

import numpy as np


def metrics(Q, Y, qs=(0.1, 0.5, 0.9)):
    """Q: [n, 2, len(qs)] predicted residual quantiles; Y: [n, 2] true residuals.
    The agent frame is a rotation of the world frame, so errors are world-frame metres."""
    j50 = list(qs).index(0.5)
    err = np.linalg.norm(Q[:, :, j50] - Y, axis=1)
    lo, hi = Q[:, :, 0], Q[:, :, -1]
    inside = (Y >= lo) & (Y <= hi)
    pinball = 0.0
    for j, q in enumerate(qs):
        d = Y - Q[:, :, j]
        pinball += np.mean(np.maximum(q * d, (q - 1) * d))
    return {
        "fde": float(err.mean()),
        "fde_p90": float(np.quantile(err, 0.9)),
        "coverage": float(inside.mean()),          # target: qs[-1] - qs[0] (0.8)
        "width": float((hi - lo).mean()),
        "pinball": float(pinball / len(qs)),
    }


def bootstrap_ci(values, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    means = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
