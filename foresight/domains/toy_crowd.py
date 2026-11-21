"""Stand-in crowd simulator that writes the INTERFACE.md format.

Placeholder so the core can be built and tested before the team's simulator is ready.
Social-force-style people with goals, stops, avoidance (of each other, the robot,
obstacles and walls) and crowd-wide behaviour regimes; one robot walking to its goal.

    python -m foresight.domains.toy_crowd --episodes 60 --out data/toy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REGIMES = {   # desired speed, comfort range B, stop rate /s, goal-change rate /s
    "calm":    dict(v0=0.9, B=0.6, A=2.0, stop=0.04, regoal=0.01),
    "hurried": dict(v0=1.6, B=0.3, A=1.5, stop=0.01, regoal=0.01),
    "erratic": dict(v0=1.2, B=0.4, A=2.0, stop=0.12, regoal=0.15),
}
BOUND = 10.0


def simulate(seed, n_people=16, duration=40.0, dt=0.1, switch=True):
    rng = np.random.default_rng(seed)
    names = list(REGIMES)
    r0 = names[rng.integers(len(names))]
    r1 = names[(names.index(r0) + 1 + rng.integers(len(names) - 1)) % len(names)] if switch else None
    t_switch = float(np.round(rng.uniform(0.35, 0.65) * duration, 1)) if switch else None
    obst = [dict(x=float(rng.uniform(-5, 5)), y=float(rng.uniform(-5, 5)), r=float(rng.uniform(0.5, 1.2)))
            for _ in range(2)]

    def goal():
        return rng.uniform(-BOUND + 1, BOUND - 1, 2)

    n = n_people + 1                                   # agent 0 = robot
    P = rng.uniform(-BOUND + 1, BOUND - 1, (n, 2)); V = np.zeros((n, 2))
    G = np.array([goal() for _ in range(n)])
    jit = rng.normal(1.0, 0.12, n).clip(0.7, 1.3)      # per-person jitter
    stop_left = np.zeros(n)
    rad = np.full(n, 0.3); rad[0] = 0.4
    rows = []
    for k in range(int(duration / dt) + 1):
        t = k * dt
        regime = r1 if (switch and t >= t_switch) else r0
        p = REGIMES[regime]
        for i in range(n):
            rows.append([seed, round(t, 3), i, "robot" if i == 0 else "person", int(i == 0),
                         P[i, 0], P[i, 1], V[i, 0], V[i, 1], rad[i], "robot" if i == 0 else regime,
                         G[i, 0], G[i, 1]])
        F = np.zeros((n, 2))
        for i in range(n):
            to_g = G[i] - P[i]; dg = np.linalg.norm(to_g)
            if dg < 0.5 or (i > 0 and rng.random() < p["regoal"] * dt):
                G[i] = goal(); to_g = G[i] - P[i]; dg = np.linalg.norm(to_g)
            if i == 0:                                 # robot: straight line, slow near people
                near = np.min(np.linalg.norm(P[1:] - P[0], axis=1))
                v_des = 1.0 * np.clip((near - 0.8) / 1.5, 0.0, 1.0)
                V[0] = to_g / max(dg, 1e-6) * v_des
                continue
            if stop_left[i] > 0:
                stop_left[i] -= dt; v_des = 0.0
            elif rng.random() < p["stop"] * dt:
                stop_left[i] = rng.uniform(1.0, 3.0); v_des = 0.0
            else:
                v_des = p["v0"] * jit[i]
            F[i] += (to_g / max(dg, 1e-6) * v_des - V[i]) / 0.5        # drive to goal
            for j in range(n):                                          # people and robot
                if j == i:
                    continue
                d = P[i] - P[j]; dist = np.linalg.norm(d)
                A = p["A"] * (2.5 if j == 0 else 1.0)                   # keep clear of the robot
                F[i] += A * np.exp((rad[i] + rad[j] - dist) / p["B"]) * d / max(dist, 1e-6)
            for o in obst:
                d = P[i] - np.array([o["x"], o["y"]]); dist = np.linalg.norm(d)
                F[i] += 3.0 * np.exp((o["r"] + rad[i] - dist) / 0.3) * d / max(dist, 1e-6)
            for ax in range(2):                                         # walls
                F[i, ax] += 3.0 * (np.exp(-(P[i, ax] + BOUND - rad[i]) / 0.3)
                                   - np.exp(-(BOUND - P[i, ax] - rad[i]) / 0.3))
        V[1:] += F[1:] * dt
        sp = np.linalg.norm(V, axis=1, keepdims=True)
        V = np.where(sp > 2.5, V / np.maximum(sp, 1e-9) * 2.5, V)
        P = np.clip(P + V * dt, -BOUND + 0.05, BOUND - 0.05)
    cols = ["episode", "t", "agent_id", "kind", "is_ego", "x", "y", "vx", "vy", "radius",
            "regime", "goal_x", "goal_y"]
    meta_ep = {"seed": seed, "duration": duration, "regime_switch_t": t_switch,
               "regime_before": r0, "regime_after": r1}
    return pd.DataFrame(rows, columns=cols), obst, meta_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--people", type=int, default=16)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--out", default="data/toy")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tracks, obstacles, eps = [], [], {}
    for e in range(a.episodes):
        tr, ob, me = simulate(e, a.people, a.duration, switch=(e % 2 == 0))
        tracks.append(tr); eps[str(e)] = me
        obstacles += [dict(episode=e, obstacle_id=i, shape="circle", x=o["x"], y=o["y"], r=o["r"],
                           w=np.nan, h=np.nan, angle=np.nan) for i, o in enumerate(ob)]
    pd.concat(tracks).to_csv(out / "tracks.csv", index=False)
    pd.DataFrame(obstacles).to_csv(out / "obstacles.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"sim": "toy-crowd-v1", "dt": 0.1,
        "bounds": [-BOUND, -BOUND, BOUND, BOUND], "episodes": eps,
        "notes": "stand-in social-force crowd; robot walks straight to goal, slows near people"}, indent=1))
    print(f"wrote {a.episodes} episodes to {out}")


if __name__ == "__main__":
    main()
