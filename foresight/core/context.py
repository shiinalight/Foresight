"""Choose which labelled rows TabPFN sees for a batch of queries.

Causality: a row is usable at query time t only once its target is known (t_known <= t),
and only from other episodes (bank) or from the same episode (recency).
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

# situation features used for nearest-neighbour retrieval (standardised)
KNN_FEATS = ["speed", "a_lon", "turn_rate", "n0_dist", "n0_x", "n0_y", "robot_dist",
             "robot_closing", "obst_dist", "density_3m", "h_f"]


class Bank:
    """Rows from training episodes, subsampled, with a kNN index."""

    def __init__(self, df, max_rows=20000, seed=0):
        if len(df) > max_rows:
            df = df.sample(max_rows, random_state=seed)
        self.df = df.reset_index(drop=True)
        Z = self.df[KNN_FEATS].to_numpy(float)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        self.nn = NearestNeighbors().fit((Z - self.mu) / self.sd)

    def knn(self, queries, k):
        Z = (queries[KNN_FEATS].to_numpy(float) - self.mu) / self.sd
        _, idx = self.nn.kneighbors(Z, n_neighbors=min(k, len(self.df)))
        order = idx.T.ravel()                     # rank-major: every query's 1-NN first, then 2-NN, ...
        _, first = np.unique(order, return_index=True)
        return order[np.sort(first)]              # unique, nearest ranks first, so [:n] cuts the farthest


def select(queries, bank: Bank, live=None, budget=1500, mix=(0.6, 0.3, 0.1), rare_col="mode",
           rare_values=("stop", "turn_left", "turn_right"), seed=0):
    """Build a context DataFrame for this batch of queries.

    queries: rows to predict (all from one moment t of one episode).
    live: rows of the same episode (any time); only those with t_known <= t are used.
    mix: shares of (recency, kNN from bank, stratified random from bank). Shares not filled
    by recency (cold start) go to the bank.
    """
    rng = np.random.default_rng(seed)
    t_now = queries["t"].min()
    parts = []
    n_rec = 0
    if live is not None and mix[0] > 0:
        rec = live[live["t_known"] <= t_now].sort_values("t_known").tail(int(budget * mix[0]))
        parts.append(rec.assign(source="recent")); n_rec = len(rec)
    rest = budget - n_rec if (mix[1] + mix[2]) > 0 else 0     # recency-only means no bank rows at all
    share_knn = mix[1] / max(mix[1] + mix[2], 1e-9)
    n_knn = int(rest * share_knn); n_rand = rest - n_knn
    if n_knn > 0:
        k = max(1, int(np.ceil(n_knn / max(len(queries), 1))))
        idx = bank.knn(queries, k)[:n_knn]
        parts.append(bank.df.iloc[idx].assign(source="knn"))
    if n_rand > 0:
        rare = bank.df[bank.df[rare_col].isin(rare_values)]
        n_r = min(len(rare), n_rand // 2)
        if n_r:
            parts.append(rare.sample(n_r, random_state=seed).assign(source="fill-rare"))
        parts.append(bank.df.sample(min(len(bank.df), n_rand - n_r), random_state=seed + 1).assign(source="fill-random"))
    import pandas as pd
    ctx = pd.concat(parts)
    # same rows as before (first occurrence wins); `source` records where each row came from
    ctx = ctx[~ctx.drop(columns="source").duplicated()]
    return ctx
