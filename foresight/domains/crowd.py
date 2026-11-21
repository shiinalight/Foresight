"""2D social-force crowd simulator (domain 1), writing the INTERFACE.md format.

People walk to goals and pick a new goal on arrival. They steer around each other, the robot,
static obstacles and the walls (Helbing-style social force with a forward-looking field of view).
Three crowd-wide behaviour regimes, each person drawing jittered settings from the current one:

    calm     slow, wide berth, rare stops
    hurried  fast, cuts close, almost never stops
    erratic  sudden stops and sharp turns

The regime can switch for the whole crowd at `switch_t`. The robot walks a straight line to a goal
on the far side of the plaza, slowing when someone is close (the R3 planner replaces this policy
through `CrowdSim.step(robot_cmd)`).

Two outputs:
- `generate()` / `simulate()`: noise-free ground truth in the INTERFACE.md format, read by the core.
- `observe()`: 0.1 m position noise plus a forward-only Kalman filter, as one wide DataFrame
  (episode, t, agent_id, x, y, vx, vy, regime, robot_x, robot_y) for quick checks and plots.

    python -m foresight.domains.crowd --episodes 5 --seed 0            # print a summary
    python -m foresight.domains.crowd --episodes 300 --out data/crowd  # write a dataset
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.core.features import _kalman_cv

# v0 desired speed (m/s), tau reaction time (s), A/B social-force strength (m/s^2) and range (m),
# stop rate (/s), stop duration (s), sharp-turn rate (/s).
REGIMES = {
    "calm":    dict(v0=0.9, tau=0.6,  A=2.0, B=0.6,  stop=0.02,  stop_s=(1.0, 3.0), turn=0.0),
    "hurried": dict(v0=1.6, tau=0.35, A=2.0, B=0.3,  stop=0.005, stop_s=(0.5, 1.0), turn=0.0),
    "erratic": dict(v0=1.1, tau=0.3,  A=1.8, B=0.4,  stop=0.15,  stop_s=(0.6, 2.5), turn=0.2),
}
PERSON_R, ROBOT_R = 0.3, 0.4
ROBOT_REPULSION = 2.0   # people push this many times harder away from the robot than from each other
ROBOT_B = 0.6           # m, minimum interaction range towards the robot (whatever the regime)
LOOKAHEAD = 0.6         # s, people react to where others will be, not only where they are
LAMBDA = 0.35          # weight of interactions behind a person (1 = all round, 0 = ahead only)
OBST_A, OBST_B = 10.0, 0.3   # obstacle and wall repulsion strength (m/s^2) and range (m)
MAX_FORCE = 10.0       # m/s^2, keeps the explicit Euler step stable at dt = 0.2
ARRIVE = 0.5            # m, goal reached
COLUMNS = ["episode", "t", "agent_id", "kind", "is_ego", "x", "y", "vx", "vy", "radius",
           "regime", "goal_x", "goal_y", "collided"]
OBSERVED = ["episode", "t", "agent_id", "x", "y", "vx", "vy", "regime", "robot_x", "robot_y"]


@dataclass(frozen=True)
class CrowdConfig:
    n_people: int = 16
    duration: float = 40.0
    dt: float = 0.2
    half_size: float = 10.0          # plaza is [-half_size, half_size]^2
    n_obstacles: int = 2
    regime: str = "calm"             # regime at t = 0
    switch_t: float | None = None    # whole crowd switches regime at this time ...
    switch_to: str | None = None     # ... to this one
    jitter: float = 0.1              # sd of per-person multiplicative jitter
    robot_speed: float = 1.0
    # difficulty knobs (T2 in docs/DECISIONS.md). Defaults reproduce the original crowd exactly.
    robot_repulsion: float = ROBOT_REPULSION   # how much harder people push away from the robot
    ignore_robot: float = 0.0        # share of people who do not react to the robot at all
                                     # (looking at a phone); drawn from a separate random stream, so
                                     # the rest of the episode is the same crowd as with 0.0

    def regime_at(self, t: float) -> str:
        if self.switch_t is not None and t >= self.switch_t - 1e-9:
            return self.switch_to
        return self.regime


class CrowdSim:
    """One episode. Agent 0 is the robot, agents 1..n_people are people."""

    def __init__(self, seed: int, config: CrowdConfig = CrowdConfig()):
        for name in (config.regime, config.switch_to):
            if name is not None and name not in REGIMES:
                raise ValueError(f"unknown regime {name!r}; choose from {list(REGIMES)}")
        if config.switch_t is not None and config.switch_to is None:
            raise ValueError("switch_t needs switch_to")
        if not 0.0 <= config.ignore_robot <= 1.0:
            raise ValueError(f"ignore_robot must be in [0, 1], got {config.ignore_robot}")
        if config.robot_repulsion < 0:
            raise ValueError(f"robot_repulsion must be >= 0, got {config.robot_repulsion}")
        self.seed, self.cfg = seed, config

    # ---------- setup ----------
    def reset(self) -> list[dict]:
        c, rng = self.cfg, np.random.default_rng(self.seed)
        self.rng = rng
        L = c.half_size
        side = rng.choice([-1.0, 1.0])                         # robot crosses left<->right
        self.robot_start = np.array([side * (L - 1), rng.uniform(-L / 2, L / 2)])
        self.robot_goal = np.array([-side * (L - 1), rng.uniform(-L / 2, L / 2)])
        seg = self.robot_goal - self.robot_start
        self.obstacles = []                                   # circular pillars, clear of each other
        for _ in range(10_000):                               # and of the robot's straight line
            if len(self.obstacles) == c.n_obstacles:
                break
            o = dict(x=rng.uniform(-L / 2, L / 2), y=rng.uniform(-L / 2, L / 2), r=rng.uniform(0.5, 1.2))
            q = np.array([o["x"], o["y"]])
            s = np.clip((q - self.robot_start) @ seg / (seg @ seg), 0, 1)
            if (np.linalg.norm(q - self.robot_start - s * seg) > o["r"] + ROBOT_R + 1.0
                    and all(np.hypot(o["x"] - p["x"], o["y"] - p["y"]) > o["r"] + p["r"] + 1.5
                            for p in self.obstacles)):
                self.obstacles.append(o)
        self._obs_xy = np.array([[o["x"], o["y"]] for o in self.obstacles]).reshape(-1, 2)
        self._obs_r = np.array([o["r"] for o in self.obstacles])

        n = c.n_people + 1
        self.radius = np.full(n, PERSON_R); self.radius[0] = ROBOT_R
        self.P = np.zeros((n, 2)); self.P[0] = self.robot_start
        for i in range(1, n):
            self.P[i] = self._free_point(self.P[:i], self.radius[i])
        self.V = np.zeros((n, 2))
        self.G = np.array([self.robot_goal] + [self._free_point(np.empty((0, 2)), PERSON_R)
                                               for _ in range(1, n)])
        self.jit = rng.normal(1.0, c.jitter, (n, 4)).clip(0.7, 1.3)   # v0, A, B, rates
        self.stop_left = np.zeros(n)
        # who ignores the robot: a separate stream keyed on the seed, so self.rng (and with it every
        # other draw in the episode) is untouched whatever ignore_robot is
        self.ignores_robot = np.zeros(n, bool)
        if c.ignore_robot > 0:
            u = np.random.default_rng([self.seed, 7919]).random(n)
            self.ignores_robot[1:] = u[1:] < c.ignore_robot
        self.k = 0
        return self._state(np.zeros(n, bool))

    def _free_point(self, others, r):
        L = self.cfg.half_size
        for _ in range(1000):
            p = self.rng.uniform(-L + 1, L - 1, 2)
            if (np.all(np.hypot(*(others - p).T) > 2 * r + 0.3)
                    and np.all(np.hypot(*(self._obs_xy - p).T) > self._obs_r + r + 0.3)):
                return p
        raise RuntimeError("plaza too crowded to place an agent")

    # ---------- dynamics ----------
    def step(self, robot_cmd: tuple[float, float] | None = None) -> tuple[list[dict], dict]:
        """Advance one dt. robot_cmd = (vx, vy) for the robot, or None for the straight-line policy."""
        c, rng, dt = self.cfg, self.rng, self.cfg.dt
        p = REGIMES[c.regime_at(self.k * dt)]
        P, V, n = self.P, self.V, len(self.P)

        # robot
        self.V[0] = self._robot_policy() if robot_cmd is None else np.asarray(robot_cmd, float)

        # people: goals, stops, sharp turns
        v0 = p["v0"] * self.jit[:, 0]; A = p["A"] * self.jit[:, 1]; B = p["B"] * self.jit[:, 2]
        rate = self.jit[:, 3] * dt
        v_des = v0.copy()
        for i in range(1, n):
            if np.linalg.norm(self.G[i] - P[i]) < ARRIVE:
                self.G[i] = self._free_point(np.empty((0, 2)), PERSON_R)
            if p["turn"] and rng.random() < p["turn"] * rate[i]:
                self.G[i] = self._turn_goal(i)
            if self.stop_left[i] > 0:
                self.stop_left[i] -= dt
            elif rng.random() < p["stop"] * rate[i]:
                self.stop_left[i] = rng.uniform(*p["stop_s"])
            if self.stop_left[i] > 0:
                v_des[i] = 0.0
        to_g = self.G - P
        e = to_g / np.maximum(np.linalg.norm(to_g, axis=1, keepdims=True), 1e-9)
        F = (e * v_des[:, None] - V) / p["tau"]

        # agent-agent repulsion from the closest approach in the next LOOKAHEAD s (anticipation),
        # weaker for what is behind you
        d = P[:, None, :] - P[None, :, :]                     # i minus j
        dv = V[:, None, :] - V[None, :, :]
        t_ca = np.clip(-(d * dv).sum(axis=2) / np.maximum((dv * dv).sum(axis=2), 1e-9), 0, LOOKAHEAD)
        d_ca = d + dv * t_ca[..., None]
        d = np.where(np.linalg.norm(d_ca, axis=2, keepdims=True) > 1e-3, d_ca, d)
        dist = np.maximum(np.linalg.norm(d, axis=2), 1e-6)
        nrm = d / dist[..., None]
        cos_phi = -(e[:, None, :] * nrm).sum(axis=2)          # 1 when j is straight ahead of i
        w = LAMBDA + (1 - LAMBDA) * (1 + cos_phi) / 2
        is_robot = np.arange(n)[None, :] == 0
        robot_w = np.where(self.ignores_robot, 0.0, c.robot_repulsion)[:, None]   # per person i
        strength = A[:, None] * np.where(is_robot, robot_w, 1.0)
        rng_ = np.where(is_robot, np.maximum(B[:, None], ROBOT_B), B[:, None])
        mag = strength * np.exp((self.radius[:, None] + self.radius[None, :] - dist) / rng_) * w
        np.fill_diagonal(mag, 0.0)
        F += (mag[..., None] * nrm).sum(axis=1)

        # obstacles and walls
        if len(self._obs_r):
            od = P[:, None, :] - self._obs_xy[None]
            odist = np.maximum(np.linalg.norm(od, axis=2), 1e-6)
            gap = odist - self._obs_r[None] - self.radius[:, None]
            on = od / odist[..., None]                        # obstacle -> agent
            F += (OBST_A * np.exp(-gap / OBST_B)[..., None] * on).sum(axis=1)
            # walk around, not into: steer sideways (on the goal's side) when heading at an obstacle
            tang = np.stack([-on[..., 1], on[..., 0]], axis=-1)
            side = np.sign((tang * e[:, None, :]).sum(axis=2) + 1e-9)
            facing = np.clip(-(on * e[:, None, :]).sum(axis=2), 0, 1)
            steer = facing * np.clip(1.0 - gap / 1.5, 0, 1) * v0[:, None] / p["tau"]
            F += (steer[..., None] * side[..., None] * tang).sum(axis=1)
        L = c.half_size
        F += 3.0 * (np.exp(-(P + L - self.radius[:, None]) / 0.25)
                    - np.exp(-(L - P - self.radius[:, None]) / 0.25))

        fm = np.linalg.norm(F, axis=1, keepdims=True)
        F = np.where(fm > MAX_FORCE, F / np.maximum(fm, 1e-9) * MAX_FORCE, F)
        V[1:] += F[1:] * dt
        sp = np.linalg.norm(V[1:], axis=1)
        cap = 1.3 * np.maximum(v0[1:], 0.5)
        V[1:] *= np.minimum(1.0, cap / np.maximum(sp, 1e-9))[:, None]
        self.P = np.clip(P + V * dt, -L + 0.05, L - 0.05)
        self._push_out_of_obstacles()
        self.k += 1

        collided = self._collisions()
        return self._state(collided), {"collided": bool(collided[0])}

    def _push_out_of_obstacles(self):
        """Hard constraint: nobody stands inside a pillar. Move them to its edge and drop the
        inward part of their velocity."""
        for c_, r in zip(self._obs_xy, self._obs_r):
            d = self.P - c_
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            inside = dist < r + self.radius
            if not inside.any():
                continue
            n = d[inside] / dist[inside, None]
            self.P[inside] = c_ + n * (r + self.radius[inside, None])
            vn = (self.V[inside] * n).sum(axis=1)
            self.V[inside] -= np.minimum(vn, 0)[:, None] * n

    def _robot_policy(self):
        to_g = self.robot_goal - self.P[0]
        dg = np.linalg.norm(to_g)
        if dg < ARRIVE:                                       # walk back the same line
            self.robot_start, self.robot_goal = self.robot_goal, self.robot_start
            self.G[0] = self.robot_goal
            to_g = self.robot_goal - self.P[0]; dg = np.linalg.norm(to_g)
        near = np.min(np.linalg.norm(self.P[1:] - self.P[0], axis=1)) if len(self.P) > 1 else np.inf
        speed = self.cfg.robot_speed * np.clip((near - 0.8) / 1.5, 0.0, 1.0)
        return to_g / max(dg, 1e-9) * speed

    def _turn_goal(self, i):
        """A goal 60-150 degrees left or right of the current heading."""
        v = self.V[i] if np.linalg.norm(self.V[i]) > 0.1 else self.G[i] - self.P[i]
        ang = np.arctan2(v[1], v[0]) + self.rng.choice([-1, 1]) * np.radians(self.rng.uniform(60, 150))
        L = self.cfg.half_size
        return np.clip(self.P[i] + self.rng.uniform(3, 7) * np.array([np.cos(ang), np.sin(ang)]),
                       -L + 1, L - 1)

    def _collisions(self):
        d = np.linalg.norm(self.P[:, None] - self.P[None], axis=2)
        hit = d < self.radius[:, None] + self.radius[None]
        np.fill_diagonal(hit, False)
        hit = hit.any(axis=1)
        if len(self._obs_r):
            od = np.linalg.norm(self.P[:, None] - self._obs_xy[None], axis=2)
            hit |= (od < self._obs_r[None] + self.radius[:, None] + 1e-6).any(axis=1)  # incl. pushed out
        return hit

    def _state(self, collided) -> list[dict]:
        t = round(self.k * self.cfg.dt, 6)
        regime = self.cfg.regime_at(t)
        return [dict(t=t, agent_id=i, kind="robot" if i == 0 else "person", is_ego=int(i == 0),
                     x=self.P[i, 0], y=self.P[i, 1], vx=self.V[i, 0], vy=self.V[i, 1],
                     radius=self.radius[i], regime="robot" if i == 0 else regime,
                     goal_x=self.G[i, 0], goal_y=self.G[i, 1], collided=int(collided[i]))
                for i in range(len(self.P))]


# ---------- episodes and datasets ----------
def simulate(seed: int, config: CrowdConfig = CrowdConfig(), episode: int = 0):
    """Run one episode. Returns (tracks, obstacles, episode_meta), ground truth, no noise."""
    sim = CrowdSim(seed, config)
    rows = sim.reset()
    for _ in range(int(round(config.duration / config.dt))):
        rows += sim.step()[0]
    tracks = pd.DataFrame(rows)
    tracks.insert(0, "episode", episode)
    obstacles = pd.DataFrame([dict(episode=episode, obstacle_id=i, shape="circle", x=o["x"], y=o["y"],
                                   r=o["r"], w=np.nan, h=np.nan, angle=np.nan)
                              for i, o in enumerate(sim.obstacles)])
    meta = dict(seed=seed, duration=config.duration, regime_switch_t=config.switch_t,
                regime_before=config.regime, regime_after=config.switch_to)
    return tracks[COLUMNS], obstacles, meta


def generate(n_episodes: int, seed: int = 0, config: CrowdConfig = CrowdConfig(),
             switch_prob: float = 0.5):
    """N episodes with random regimes; a crowd-wide switch in about `switch_prob` of them.

    Same seed, same dataset. Returns (tracks, obstacles, meta) in the INTERFACE.md format.
    """
    rng = np.random.default_rng(seed)
    names = list(REGIMES)
    tracks, obstacles, eps = [], [], {}
    for e in range(n_episodes):
        ep_seed = int(rng.integers(2**31))
        before = names[rng.integers(len(names))]
        cfg = replace(config, regime=before, switch_t=None, switch_to=None)
        if rng.random() < switch_prob:
            after = rng.choice([r for r in names if r != before])
            t_sw = round(rng.uniform(0.35, 0.65) * config.duration / config.dt) * config.dt
            cfg = replace(cfg, switch_t=round(t_sw, 6), switch_to=str(after))
        tr, ob, me = simulate(ep_seed, cfg, episode=e)
        tracks.append(tr); obstacles.append(ob); eps[str(e)] = me
    L = config.half_size
    meta = dict(sim="crowd-v1", dt=config.dt, bounds=[-L, -L, L, L], episodes=eps,
                config={k: v for k, v in asdict(config).items()
                        if k not in ("regime", "switch_t", "switch_to")},
                notes="social-force crowd; robot walks a straight line to its goal, slowing near people")
    return (pd.concat(tracks, ignore_index=True), pd.concat(obstacles, ignore_index=True), meta)


def observe(tracks: pd.DataFrame, noise: float = 0.1, seed: int = 0, dt: float | None = None):
    """Noisy positions smoothed by a forward-only Kalman filter, one row per person per step.

    The robot is given as columns (robot_x, robot_y) with its true position: a robot knows its own
    pose far better than it sees other people. Velocities are the filter's estimates.
    """
    dt = dt if dt is not None else float(np.round(np.diff(np.unique(tracks["t"]))[0], 6))
    rng = np.random.default_rng(seed)
    tracks = tracks.sort_values(["episode", "agent_id", "t"])
    people = []
    for _, a in tracks[tracks["is_ego"] == 0].groupby(["episode", "agent_id"], sort=True):
        z = a[["x", "y"]].to_numpy() + rng.normal(0, noise, (len(a), 2))
        p, v = _kalman_cv(z, dt, r=max(noise, 1e-3))
        people.append(a[["episode", "t", "agent_id", "regime"]].assign(
            x=p[:, 0], y=p[:, 1], vx=v[:, 0], vy=v[:, 1]))
    robot = (tracks[tracks["is_ego"] == 1][["episode", "t", "x", "y"]]
             .rename(columns={"x": "robot_x", "y": "robot_y"}))
    obs = pd.concat(people, ignore_index=True).merge(robot, on=["episode", "t"], how="left")
    return obs[OBSERVED].sort_values(["episode", "t", "agent_id"]).reset_index(drop=True)


def save(out_dir, tracks, obstacles, meta):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(out / "tracks.csv", index=False)
    obstacles.to_csv(out / "obstacles.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def summary(tracks: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Per-episode numbers that say whether the crowd is worth predicting."""
    dt = float(meta["dt"]); h = round(2.0 / dt)
    rows = []
    for ep, tr in tracks.groupby("episode"):
        ppl = tr[tr["kind"] == "person"].sort_values(["agent_id", "t"])
        speed = np.hypot(ppl["vx"], ppl["vy"])
        # constant-velocity error at 2 s on ground truth: how much behaviour CV can't explain
        g = ppl.groupby("agent_id")
        fx, fy = g["x"].shift(-h), g["y"].shift(-h)
        cv_err = np.hypot(fx - (ppl["x"] + ppl["vx"] * 2.0), fy - (ppl["y"] + ppl["vy"] * 2.0))
        robot = tr[tr["is_ego"] == 1].set_index("t")[["x", "y"]]
        r = robot.loc[ppl["t"]].to_numpy()
        gap = np.hypot(ppl["x"].to_numpy() - r[:, 0], ppl["y"].to_numpy() - r[:, 1])
        m = meta["episodes"][str(ep)]
        rows.append(dict(
            episode=ep,
            regimes=m["regime_before"] + (f" -> {m['regime_after']} @ {m['regime_switch_t']:.0f}s"
                                          if m["regime_after"] else ""),
            people=ppl["agent_id"].nunique(), steps=tr["t"].nunique(),
            mean_speed=speed.mean(), stopped_frac=(speed < 0.2).mean(),
            cv_err_2s=np.nanmean(cv_err), min_robot_gap=gap.min() - PERSON_R - ROBOT_R,
            collision_frac=ppl["collided"].mean()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--people", type=int, default=16)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--switch-prob", type=float, default=0.5)
    ap.add_argument("--robot-repulsion", type=float, default=ROBOT_REPULSION,
                    help="how much harder people push away from the robot than from each other")
    ap.add_argument("--ignore-robot", type=float, default=0.0, help="share of people who ignore the robot")
    ap.add_argument("--out", help="write tracks.csv, obstacles.csv, meta.json here")
    a = ap.parse_args()
    cfg = CrowdConfig(n_people=a.people, duration=a.duration, robot_repulsion=a.robot_repulsion,
                      ignore_robot=a.ignore_robot)
    tracks, obstacles, meta = generate(a.episodes, a.seed, cfg, a.switch_prob)
    obs = observe(tracks, seed=a.seed)
    pd.set_option("display.width", 160)
    print(summary(tracks, meta).round(3).to_string(index=False))
    true = tracks[tracks["kind"] == "person"].sort_values(["episode", "t", "agent_id"])
    err = np.hypot(obs["x"].to_numpy() - true["x"].to_numpy(), obs["y"].to_numpy() - true["y"].to_numpy())
    print(f"\nobserved table: {len(obs)} rows, columns {list(obs.columns)}")
    print(f"position error after Kalman filter: mean {err.mean():.3f} m (sensor noise sd 0.1 m per axis)")
    if a.out:
        save(a.out, tracks, obstacles, meta)
        print(f"wrote {a.episodes} episodes to {a.out}")


if __name__ == "__main__":
    main()
