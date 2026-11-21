"""R3 closed loop: a robot crosses the simulated crowd, choosing its moves from predicted ranges.

    python run_closed.py --planners cv tabpfn --seeds 1000 1001 --log      # logs for render_closed.py
    python run_closed.py --planners cv cv_band lgbm tabpfn --seeds $(seq 1000 1019)

Each planner uses the same seeds (same crowd, same start and goal). Predictors are fitted once
on rows from the stand-in crowd dataset (data/toy, other seeds), then queried every replan.
Writes results/closed.csv (one row per planner x seed) and, with --log, results/closed/*.pkl.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from foresight import viz as V
from foresight.core import features as F
from foresight.core import models as M
from foresight.core import planner as P
from foresight.domains.toy_crowd import BOUND, CrowdSim
from run_offline import rows_for

HZ = (1.0, 2.0)
WINDOW = 12            # sim steps of history passed to the feature builder (>= 1.0 s + 1)


def fit_predictor(kind, bank, n_ctx, seed=0):
    if kind == "cv":
        return M.make("cv")
    rows = bank if kind == "cv_band" else bank.sample(min(n_ctx, len(bank)), random_state=seed)
    kw = {"fit_mode": "fit_with_cache"} if kind == "tabpfn" else {}
    return M.make(kind, **kw).fit(F.X_of(rows), rows[["ddx", "ddy"]].to_numpy())


def make_sim(name, seed, args):
    if name == "toy":
        sim = CrowdSim(seed, n_people=args.people, duration=args.duration,
                       robot_start=(-BOUND + 1.5, 0.0), robot_goal=(BOUND - 1.5, 0.0))
        return sim, sim.dt, BOUND
    from foresight.domains.crowd import REGIMES, CrowdConfig, CrowdSim as NCrowdSim
    regime = list(REGIMES)[seed % len(REGIMES)]                  # vary the crowd across seeds
    cfg = CrowdConfig(n_people=args.people, duration=args.duration, regime=regime)
    return NCrowdSim(seed, cfg), cfg.dt, cfg.half_size


def run_episode(model, seed, args, log=False):
    sim, dt, L = make_sim(args.sim, seed, args)
    state = [dict(r, episode=seed) for r in sim.reset()]
    hist = [state]
    meta = {"dt": dt, "bounds": [-L, -L, L, L], "episodes": {}}
    obstacles = getattr(sim, "obst", None) or getattr(sim, "obstacles", [])
    obst_df = pd.DataFrame([dict(episode=seed, obstacle_id=i, shape="circle", **o) for i, o in enumerate(obstacles)])
    every = max(1, round(args.replan / dt))
    window = max(WINDOW, int(round(1.2 / dt)) + 1)
    frames, plans = [], []
    collided_steps, moving_collisions, min_gap, t_goal, t_pred = 0, 0, np.inf, None, 0.0
    n_plans, n_freeze, n_fallback, k = 0, 0, 0, 0
    cmd = np.zeros(2)
    while k * dt < args.duration:
        p_robot = np.array([state[0]["x"], state[0]["y"]])
        goal = np.array([state[0]["goal_x"], state[0]["goal_y"]])
        if k % every == 0:
            win = pd.DataFrame([r for s in hist[-window:] for r in s])
            q = F.build_rows(win, obst_df, meta, horizons=HZ, noise=args.noise, seed=k, query_only=True)
            t0 = time.perf_counter()
            pred = V.predict_rows(model, q, (args.box_q, 0.5, 1 - args.box_q)) if len(q) else q
            t_pred += time.perf_counter() - t0
            cmd, info = P.choose(p_robot, goal, pred, margin=args.margin, bounds=meta["bounds"],
                                 obstacles=obstacles, horizons=HZ)
            n_plans += 1; n_freeze += int(np.hypot(*cmd) < 0.05); n_fallback += int(info["fallback"])
            if log:
                plans.append({"t": k * dt, "pred": pred, "cmd": cmd, **info})
        state, info = sim.step(cmd)
        state = [dict(r, episode=seed) for r in state]
        k += 1
        hist.append(state)
        rob = state[0]
        gaps = [np.hypot(r["x"] - rob["x"], r["y"] - rob["y"]) - r["radius"] - rob["radius"] for r in state[1:]]
        min_gap = min(min_gap, min(gaps))
        if info["collided"]:
            collided_steps += 1
            moving_collisions += int(np.hypot(*cmd) >= 0.05)        # robot was moving (its fault, roughly)
        if log:
            frames.append({"t": k * dt, "state": state, "info": info})
        if np.hypot(goal[0] - rob["x"], goal[1] - rob["y"]) < 0.5:
            t_goal = k * dt
            break
    res = {"seed": seed, "reached": t_goal is not None, "time_to_goal": t_goal if t_goal else np.nan,
           "collision_steps": collided_steps, "collided": collided_steps > 0,
           "moving_collision_steps": moving_collisions, "min_gap": min_gap, "near_miss": min_gap < 0.2,
           "freeze_rate": n_freeze / max(n_plans, 1), "fallback_rate": n_fallback / max(n_plans, 1),
           "pred_seconds": t_pred}
    return res, ({"frames": frames, "plans": plans, "obst": obstacles, "bounds": meta["bounds"]} if log else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planners", nargs="*", default=["cv", "tabpfn"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[1000])
    ap.add_argument("--sim", default="crowd", choices=["crowd", "toy"])
    ap.add_argument("--bank", default=None, help="dataset for context rows (default data/<sim>)")
    ap.add_argument("--context", type=int, default=1000)
    ap.add_argument("--people", type=int, default=16)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--replan", type=float, default=0.4)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--box-q", type=float, default=0.05, help="box = [q, 1-q] per axis (risk dial)")
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()

    bank = rows_for(a.bank or f"data/{a.sim}", a.noise, horizons=HZ)
    out = []
    Path("results/closed").mkdir(parents=True, exist_ok=True)
    for kind in a.planners:
        model = fit_predictor(kind, bank, a.context)
        for seed in a.seeds:
            t0 = time.perf_counter()
            res, logd = run_episode(model, seed, a, log=a.log)
            res |= {"sim": a.sim, "planner": kind, "wall_s": time.perf_counter() - t0, "margin": a.margin,
                    "box_q": a.box_q, "version": "v2"}
            out.append(res)
            print(f"{kind:<8} seed {seed}  reached {res['reached']}  t_goal {res['time_to_goal']}  "
                  f"collisions {res['collision_steps']} (moving {res['moving_collision_steps']})  min gap {res['min_gap']:.2f} m  "
                  f"freeze {100 * res['freeze_rate']:.0f}%  fallback {100 * res['fallback_rate']:.0f}%  ({res['wall_s']:.0f} s)")
            if logd:
                with open(f"results/closed/{a.sim}_{kind}_{seed}.pkl", "wb") as f:
                    pickle.dump(logd, f)
    df = pd.DataFrame(out)
    path = Path("results/closed.csv")
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    print(df.groupby("planner")[["reached", "time_to_goal", "collided", "moving_collision_steps", "near_miss", "min_gap",
                                  "freeze_rate", "fallback_rate"]].mean().round(3))


if __name__ == "__main__":
    main()
