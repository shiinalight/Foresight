"""S1: tune the forward Kalman filter (q = process noise, r = measurement noise) per dataset, on training
episodes only, by constant-velocity error at +2 s (the planning horizon). Writes <dataset>/kalman.json,
which build_rows (via run_offline.rows_for and run_closed) reads.

    python tools/tune_kalman.py data/crowd_cross --noise 0.1 --split          # tune on its training episodes
    python tools/tune_kalman.py data/hotel --noise 0 --apply-to data/eth       # cross-scene: tune on Hotel, store for ETH
    python tools/tune_kalman.py data/eth --noise 0 --report-fde48               # also CV FDE at 4.8 s (S1 gate)

The noise draw matches build_rows (per agent, rng([seed, agent_id]) over absolute steps, seed 0).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight.core import features as F  # noqa: E402
from foresight.core import schema  # noqa: E402

Q_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2]
R_GRID = [0.05, 0.1, 0.2, 0.3]


def tracks_by_agent(tracks, dt, episodes):
    out = []
    for (ep, aid), g in tracks[tracks.episode.isin(episodes) & (tracks.is_ego == 0)].groupby(["episode", "agent_id"]):
        steps = np.round(g.t.to_numpy() / dt).astype(int)
        S = steps.max() + 1
        P = np.full((S, 2), np.nan); P[steps] = g[["x", "y"]].to_numpy()
        out.append((ep, aid, P))
    return out


def cv_errors(agents, dt, q, r, noise, h=2.0, every=0.4, warm=1.0, seed=0):
    hs, es, ws = round(h / dt), max(1, round(every / dt)), round(warm / dt)
    errs = []
    for ep, aid, P in agents:
        z = P + np.random.default_rng([seed, int(aid)]).normal(0, noise, P.shape)
        Ps, Vs = F._kalman_cv(z, dt, q=q, r=r)
        for s in range(ws, len(P) - hs, es):
            if np.isnan(P[s - ws:s + hs + 1, 0]).any():
                continue                                   # gap-free window only
            errs.append(np.hypot(*(Ps[s] + Vs[s] * h - P[s + hs])))
    return float(np.mean(errs)), len(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--split", action="store_true", help="tune on the training episodes of an episode split")
    ap.add_argument("--apply-to", nargs="*", default=None, help="write kalman.json into these datasets (default: data)")
    ap.add_argument("--report-fde48", action="store_true")
    ap.add_argument("--q-grid", type=float, nargs="*", default=None)
    ap.add_argument("--r-grid", type=float, nargs="*", default=None)
    a = ap.parse_args()
    global Q_GRID, R_GRID
    Q_GRID, R_GRID = a.q_grid or Q_GRID, a.r_grid or R_GRID
    tracks, _, meta = schema.load(a.data)
    dt = float(meta["dt"])
    eps = np.array(sorted(tracks.episode.unique()))
    if a.split:
        rng = np.random.default_rng(0); perm = eps.copy(); rng.shuffle(perm)
        eps = np.array(sorted(perm[max(1, int(len(perm) * 0.25)):]))       # same rule as split_by_episode
    agents = tracks_by_agent(tracks, dt, eps)
    grid = []
    for q in Q_GRID:
        for r in R_GRID:
            e, n = cv_errors(agents, dt, q, r, a.noise)
            grid.append((e, q, r, n))
            print(f"q {q:<6} r {r:<5} CV error at +2 s {e:.4f} m  ({n} windows)", flush=True)
    e, q, r, n = min(grid)
    old, _ = cv_errors(agents, dt, 2.0, max(a.noise, 0.05), a.noise)
    print(f"best: q {q}, r {r}: {e:.4f} m (old q 2, r {max(a.noise, 0.05)}: {old:.4f} m, {100 * (1 - e / old):.1f}% lower)")
    if a.report_fde48:
        for qq, rr, lab in ((2.0, max(a.noise, 0.05), "old"), (q, r, "tuned")):
            e48, n48 = cv_errors(agents, dt, qq, rr, a.noise, h=4.8, every=0.4, warm=3.2)
            print(f"CV FDE at 4.8 s ({lab}, 3.2 s warm-up, gap-free): {e48:.3f} m over {n48} windows")
    edge = q in (Q_GRID[0], Q_GRID[-1]) or r in (R_GRID[0], R_GRID[-1])
    if edge:
        print("note: best value on the grid edge")
    for d in (a.apply_to or [a.data]):
        Path(d, "kalman.json").write_text(json.dumps({"kalman_q": q, "kalman_r": r, "tuned_on": a.data,
                                                       "episodes": "train split" if a.split else "all",
                                                       "cv_err_2s": e, "noise": a.noise, "grid_edge": edge}, indent=1))
        print(f"wrote {d}/kalman.json")


if __name__ == "__main__":
    main()
