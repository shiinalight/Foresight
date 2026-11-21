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


COLS = ["episode", "t", "agent_id", "kind", "is_ego", "x", "y", "vx", "vy", "radius", "regime", "goal_x", "goal_y"]


class CrowdSim:
    """Step-by-step crowd (INTERFACE.md closed-loop hook). Agent 0 is the robot.

        sim = CrowdSim(seed); state = sim.reset()
        state, info = sim.step()                 # built-in robot policy
        state, info = sim.step((vx, vy))         # planner drives the robot
    state: list of dicts with the tracks.csv columns for the current step.
    """

    def __init__(self, seed, n_people=16, duration=40.0, dt=0.1, switch=True, robot_start=None, robot_goal=None):
        self.seed, self.n_people, self.duration, self.dt, self.switch = seed, n_people, duration, dt, switch
        self.robot_start, self.robot_goal = robot_start, robot_goal

    def reset(self):
        rng = self.rng = np.random.default_rng(self.seed)
        names = list(REGIMES)
        self.r0 = names[rng.integers(len(names))]
        self.r1 = names[(names.index(self.r0) + 1 + rng.integers(len(names) - 1)) % len(names)] if self.switch else None
        self.t_switch = float(np.round(rng.uniform(0.35, 0.65) * self.duration, 1)) if self.switch else None
        self.obst = [dict(x=float(rng.uniform(-5, 5)), y=float(rng.uniform(-5, 5)), r=float(rng.uniform(0.5, 1.2)))
                     for _ in range(2)]
        n = self.n = self.n_people + 1
        self.P = rng.uniform(-BOUND + 1, BOUND - 1, (n, 2)); self.V = np.zeros((n, 2))
        self.G = np.array([self._goal() for _ in range(n)])
        self.jit = rng.normal(1.0, 0.12, n).clip(0.7, 1.3)
        self.stop_left = np.zeros(n)
        self.rad = np.full(n, 0.3); self.rad[0] = 0.4
        if self.robot_start is not None:
            self.P[0] = self.robot_start
        if self.robot_goal is not None:
            self.G[0] = self.robot_goal
        self.k = 0
        return self.state()

    def _goal(self):
        return self.rng.uniform(-BOUND + 1, BOUND - 1, 2)

    @property
    def t(self):
        return round(self.k * self.dt, 3)

    def regime(self):
        return self.r1 if (self.switch and self.t >= self.t_switch) else self.r0

    def state(self):
        reg = self.regime()
        return [dict(zip(COLS, [self.seed, self.t, i, "robot" if i == 0 else "person", int(i == 0),
                                self.P[i, 0], self.P[i, 1], self.V[i, 0], self.V[i, 1], self.rad[i],
                                "robot" if i == 0 else reg, self.G[i, 0], self.G[i, 1]])) for i in range(self.n)]

    def step(self, robot_cmd=None):
        P, V, G, n, dt, rng = self.P, self.V, self.G, self.n, self.dt, self.rng
        p = REGIMES[self.regime()]
        F = np.zeros((n, 2))
        for i in range(n):
            to_g = G[i] - P[i]; dg = np.linalg.norm(to_g)
            fixed_robot_goal = i == 0 and self.robot_goal is not None
            if (dg < 0.5 and not fixed_robot_goal) or (i > 0 and rng.random() < p["regoal"] * dt):
                G[i] = self._goal(); to_g = G[i] - P[i]; dg = np.linalg.norm(to_g)
            if i == 0:
                if robot_cmd is None:                  # built-in policy: straight line, slow near people
                    near = np.min(np.linalg.norm(P[1:] - P[0], axis=1))
                    v_des = 1.0 * np.clip((near - 0.8) / 1.5, 0.0, 1.0)
                    V[0] = to_g / max(dg, 1e-6) * v_des
                else:
                    V[0] = np.asarray(robot_cmd, float)
                continue
            if self.stop_left[i] > 0:
                self.stop_left[i] -= dt; v_des = 0.0
            elif rng.random() < p["stop"] * dt:
                self.stop_left[i] = rng.uniform(1.0, 3.0); v_des = 0.0
            else:
                v_des = p["v0"] * self.jit[i]
            F[i] += (to_g / max(dg, 1e-6) * v_des - V[i]) / 0.5        # drive to goal
            for j in range(n):                                          # people and robot
                if j == i:
                    continue
                d = P[i] - P[j]; dist = np.linalg.norm(d)
                A = p["A"] * (2.5 if j == 0 else 1.0)                   # keep clear of the robot
                F[i] += A * np.exp((self.rad[i] + self.rad[j] - dist) / p["B"]) * d / max(dist, 1e-6)
            for o in self.obst:
                d = P[i] - np.array([o["x"], o["y"]]); dist = np.linalg.norm(d)
                F[i] += 3.0 * np.exp((o["r"] + self.rad[i] - dist) / 0.3) * d / max(dist, 1e-6)
            for ax in range(2):                                         # walls
                F[i, ax] += 3.0 * (np.exp(-(P[i, ax] + BOUND - self.rad[i]) / 0.3)
                                   - np.exp(-(BOUND - P[i, ax] - self.rad[i]) / 0.3))
        V[1:] += F[1:] * dt
        sp = np.linalg.norm(V, axis=1, keepdims=True)
        V[:] = np.where(sp > 2.5, V / np.maximum(sp, 1e-9) * 2.5, V)
        self.P = P = np.clip(P + V * dt, -BOUND + 0.05, BOUND - 0.05)
        self.k += 1
        gaps = np.linalg.norm(P[1:] - P[0], axis=1) - self.rad[1:] - self.rad[0]
        obst_gap = min((np.linalg.norm(P[0] - [o["x"], o["y"]]) - o["r"] - self.rad[0]) for o in self.obst)
        info = {"min_gap": float(gaps.min()), "collided": bool(gaps.min() < 0 or obst_gap < 0),
                "goal_dist": float(np.linalg.norm(G[0] - P[0]))}
        return self.state(), info


def simulate(seed, n_people=16, duration=40.0, dt=0.1, switch=True):
    sim = CrowdSim(seed, n_people, duration, dt, switch)
    rows = list(sim.reset())
    for _ in range(int(duration / dt)):
        state, _ = sim.step()
        rows += state
    meta_ep = {"seed": seed, "duration": duration, "regime_switch_t": sim.t_switch,
               "regime_before": sim.r0, "regime_after": sim.r1}
    return pd.DataFrame(rows, columns=COLS), sim.obst, meta_ep


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
