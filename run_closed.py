"""R3 closed loop: a robot crosses the simulated crowd, choosing its moves from predicted ranges.

    python run_closed.py --planners cv tabpfn tabpfn_live --seeds 1000 1001 --log
    python run_closed.py --planners cv cv_band lgbm lgbm_all lgbm_live tabpfn tabpfn_live --seeds 1000 1001 1002

Planners (same seeds = same crowd, start and goal for all):
  cv           constant velocity, no range
  cv_band      constant velocity + fixed band from all bank rows
  lgbm         LightGBM on --context random bank rows (same rows as tabpfn)
  lgbm_all     LightGBM on every bank row
  tabpfn       TabPFN on --context random bank rows, fitted once (frozen)
  lgbm_live    refitted every --refit s on context.select(): this episode's completed rows
  tabpfn_live  (labelled from observed, filtered positions, as a robot would) + similar bank rows
The 0.4 s constant-velocity safety check in planner.choose() applies to every planner.

Writes results/closed/<run_id>.csv (one row per planner x seed), robot features of the live
queries to results/closed/robotfeat_<run_id>.csv (bank-vs-live shift check), and with --log
results/closed/<sim>_<planner>_<seed>.pkl for render_closed.py.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from foresight import viz as V
from foresight.core import context as C
from foresight.core import features as F
from foresight.core import models as M
from foresight.core import planner as P
from foresight.domains.toy_crowd import BOUND, CrowdSim
from run_offline import rows_for

HZ = (1.0, 2.0)
VERSION = "v3"          # v3: train/serve parity (full history, fixed noise seed), live planners
KEY = ["sim", "planner", "seed", "margin", "box_q", "version", "horizons", "context", "people", "bank", "refit",
       "tabpfn_cfg"]
DEFAULTS = {"margin": 0.2, "box_q": 0.05, "version": "v1", "horizons": "1 2", "context": 1000, "people": 16,
            "bank": "data/crowd", "refit": 0.0, "tabpfn_cfg": "default"}
LIVE_WINDOW = 25.0      # seconds of episode history used to label live rows


def load_closed(folder="results/closed", version=VERSION):
    """All closed-loop runs, one row per KEY (the latest run wins). version=None keeps all."""
    files = [f for f in sorted(Path(folder).glob("*.csv")) if not f.name.startswith("robotfeat_")]
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for c, v in DEFAULTS.items():
        if c not in d:
            d[c] = v
        d[c] = d[c].fillna(v)
    d["horizons"] = d["horizons"].astype(str)
    d = d.drop_duplicates(KEY, keep="last")
    return d[d.version == version] if version else d


class Predictor:
    """Frozen or live predictor behind one call: predict(q, t, hist_df)."""

    def __init__(self, kind, bank, bank_obj, args):
        self.kind, self.args, self.bank_obj = kind, args, bank_obj
        self.live = kind.endswith("_live")
        self.base = kind.replace("_live", "").replace("_all", "")
        self.last_fit, self.model, self.n_live = -1e9, None, 0
        if not self.live:
            if kind == "cv":
                self.model = M.make("cv")
            else:
                rows = bank if kind in ("cv_band", "lgbm_all") else bank.sample(min(args.context, len(bank)), random_state=0)
                self.model = self._fit(rows)

    def _fit(self, rows):
        kw = {}
        if self.base == "tabpfn":
            # 3.5-Fast, 1 estimator: same offline accuracy on the crowd as the default (0.99 vs 0.98 m at
            # +2 s, same coverage) and ~13x faster on CPU (tools/latency.py notes in RESULTS.md)
            kw = {"fit_mode": "fit_with_cache", "fast": self.args.tabpfn_fast, "n_estimators": self.args.tabpfn_est}
        return M.make(self.base, **kw).fit(F.X_of(rows), rows[["ddx", "ddy"]].to_numpy())

    def predict(self, q, t, hist_df, obst_df, meta, seed):
        if self.live and t - self.last_fit >= self.args.refit - 1e-9:
            recent = hist_df[hist_df.t >= t - LIVE_WINDOW]
            live = F.build_rows(recent, obst_df, meta, horizons=self.args.hz, noise=self.args.noise, seed=seed,
                                label_from="observed")
            live = live[live.t_known <= t + 1e-9]
            ctx = C.select(q, self.bank_obj, live=live if len(live) else None, budget=self.args.context,
                           mix=(0.6, 0.3, 0.1))
            self.model, self.last_fit, self.n_live = self._fit(ctx), t, int(len(live))
        return V.predict_rows(self.model, q, (self.args.box_q, 0.5, 1 - self.args.box_q))


def make_sim(name, seed, args):
    if name == "toy":
        sim = CrowdSim(seed, n_people=args.people, duration=args.duration,
                       robot_start=(-BOUND + 1.5, 0.0), robot_goal=(BOUND - 1.5, 0.0))
        return sim, sim.dt, BOUND
    from foresight.domains.crowd import REGIMES, CrowdConfig, CrowdSim as NCrowdSim
    names = list(REGIMES)
    regime = names[seed % len(names)]                            # vary the crowd across seeds
    switch = dict(switch_t=round(args.duration / 2, 6), switch_to=names[(seed + 1) % len(names)]) \
        if (args.switch and seed % 2 == 1) else {}
    cfg = CrowdConfig(n_people=args.people, duration=args.duration, regime=regime, **switch)
    return NCrowdSim(seed, cfg), cfg.dt, cfg.half_size


def run_episode(pred, seed, args, log=False):
    sim, dt, L = make_sim(args.sim, seed, args)
    state = [dict(r, episode=seed) for r in sim.reset()]
    hist = [state]
    meta = {"dt": dt, "bounds": [-L, -L, L, L], "episodes": {}}
    obstacles = getattr(sim, "obst", None) or getattr(sim, "obstacles", [])
    obst_df = pd.DataFrame([dict(episode=seed, obstacle_id=i, shape="circle", **o) for i, o in enumerate(obstacles)])
    every = max(1, round(args.replan / dt))
    if pred.live:
        pred.last_fit = -1e9
    frames, plans, robotfeat = [], [], []
    collided_steps, moving_collisions, min_gap, t_goal, t_pred = 0, 0, np.inf, None, 0.0
    n_plans, n_freeze, n_fallback, k = 0, 0, 0, 0
    cmd = np.zeros(2)
    while k * dt < args.duration:
        p_robot = np.array([state[0]["x"], state[0]["y"]])
        goal = np.array([state[0]["goal_x"], state[0]["goal_y"]])
        if k % every == 0:
            # full history + fixed noise seed: same observations and Kalman state as the offline rows
            hist_df = pd.DataFrame([r for s in hist for r in s])
            q = F.build_rows(hist_df, obst_df, meta, horizons=args.hz, noise=args.noise, seed=seed, query_only=True)
            t0 = time.perf_counter()
            pr = pred.predict(q, k * dt, hist_df, obst_df, meta, seed) if len(q) else q
            t_pred += time.perf_counter() - t0
            cmd, info = P.choose(p_robot, goal, pr, margin=args.margin, bounds=meta["bounds"],
                                 obstacles=obstacles, horizons=args.hz)
            n_plans += 1; n_freeze += int(np.hypot(*cmd) < 0.05); n_fallback += int(info["fallback"])
            if len(q):
                robotfeat.append(q[["robot_dist", "robot_closing"]].assign(t=k * dt))
            if log:
                plans.append({"t": k * dt, "pred": pr, "cmd": cmd, "n_live": pred.n_live, **info})
        state, info = sim.step(cmd)
        state = [dict(r, episode=seed) for r in state]
        k += 1
        hist.append(state)
        rob = state[0]
        gaps = [np.hypot(r["x"] - rob["x"], r["y"] - rob["y"]) - r["radius"] - rob["radius"] for r in state[1:]]
        min_gap = min(min_gap, min(gaps))
        if info["collided"]:
            collided_steps += 1
            moving_collisions += int(np.hypot(*cmd) >= 0.05)
        if log:
            frames.append({"t": k * dt, "state": state, "info": info})
        if np.hypot(goal[0] - rob["x"], goal[1] - rob["y"]) < 0.5:
            t_goal = k * dt
            break
    res = {"seed": seed, "reached": t_goal is not None, "time_to_goal": t_goal if t_goal else np.nan,
           "collision_steps": collided_steps, "collided": collided_steps > 0,
           "moving_collision_steps": moving_collisions, "min_gap": min_gap, "near_miss": min_gap < 0.2,
           "freeze_rate": n_freeze / max(n_plans, 1), "fallback_rate": n_fallback / max(n_plans, 1),
           "pred_seconds": t_pred, "pred_ms_per_plan": 1000 * t_pred / max(n_plans, 1)}
    logd = {"frames": frames, "plans": plans, "obst": obstacles, "bounds": meta["bounds"]} if log else None
    rf = pd.concat(robotfeat) if robotfeat else pd.DataFrame()
    return res, logd, rf


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
    ap.add_argument("--refit", type=float, default=2.0, help="live planners refit every N seconds")
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--box-q", type=float, default=0.05, help="box = [q, 1-q] per axis (caution dial)")
    ap.add_argument("--horizons", type=float, nargs="*", default=list(HZ), help="planner checks the boxes at these")
    ap.add_argument("--switch", action="store_true", help="odd seeds: crowd-wide regime switch at mid-episode")
    ap.add_argument("--tabpfn-fast", type=int, default=1, help="1 = TabPFN-3.5-Fast")
    ap.add_argument("--tabpfn-est", type=int, default=1, help="TabPFN ensemble members")
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()

    a.hz = tuple(a.horizons)
    bank_path = a.bank or f"data/{a.sim}"
    bank = rows_for(bank_path, a.noise, horizons=a.hz)
    bank_obj = C.Bank(bank) if any(p.endswith("_live") for p in a.planners) else None
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out, feats = [], []
    Path("results/closed").mkdir(parents=True, exist_ok=True)
    for kind in a.planners:
        pred = Predictor(kind, bank, bank_obj, a)
        for seed in a.seeds:
            t0 = time.perf_counter()
            res, logd, rf = run_episode(pred, seed, a, log=a.log)
            res |= {"sim": a.sim, "planner": kind, "wall_s": time.perf_counter() - t0, "margin": a.margin,
                    "box_q": a.box_q, "version": VERSION, "horizons": " ".join(f"{h:g}" for h in a.hz),
                    "context": a.context, "people": a.people, "bank": bank_path,
                    "refit": a.refit if kind.endswith("_live") else 0.0, "switch": a.switch, "run_id": run_id,
                    "tabpfn_cfg": (f"fast{a.tabpfn_fast}-est{a.tabpfn_est}" if "tabpfn" in kind else "-")}
            out.append(res)
            if len(rf):
                feats.append(rf.assign(planner=kind, seed=seed))
            print(f"{kind:<12} seed {seed}  reached {res['reached']}  t_goal {res['time_to_goal']}  "
                  f"collisions {res['collision_steps']} (moving {res['moving_collision_steps']})  min gap {res['min_gap']:.2f} m  "
                  f"freeze {100 * res['freeze_rate']:.0f}%  fallback {100 * res['fallback_rate']:.0f}%  "
                  f"{res['pred_ms_per_plan']:.0f} ms/plan  ({res['wall_s']:.0f} s)", flush=True)
            if logd:
                with open(f"results/closed/{a.sim}_{kind}_{seed}.pkl", "wb") as f:
                    pickle.dump(logd, f)
    df = pd.DataFrame(out)
    df.to_csv(f"results/closed/{run_id}.csv", index=False)
    if feats:
        pd.concat(feats).to_csv(f"results/closed/robotfeat_{run_id}.csv", index=False)
    print(df.groupby("planner")[["reached", "time_to_goal", "collided", "moving_collision_steps", "near_miss", "min_gap",
                                  "freeze_rate", "fallback_rate", "pred_ms_per_plan"]].mean().round(3))
    print(f"wrote results/closed/{run_id}.csv")


if __name__ == "__main__":
    main()
