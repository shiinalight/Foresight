"""Doorway bottleneck (scene 3): two rooms joined by one 1.5 m door; people and the robot must pass it.

A small change to `crowd.py`: the same social force (`interaction_force`), obstacle push-out and
constants, plus a wall at x = 0 with a door in the middle and goals on the other side.

    room A (x < 0)      wall      room B (x > 0)    meeting room (beyond the right wall)
                         ||
          people  -->   door  <--  people          --> at switch_t a crowd pours in from here
                         ||

Regimes (whole crowd, sharp switch):
    trickle  8 people wander between the rooms, calm, an occasional stop
    rush     at `regime_switch_t` a meeting ends: 24 people pour into room B within 3 s and head
             for the door and the exits on the far left; everyone walks faster and closer

Door rules that make the rush a real queue, not a smooth stream:
    one way at a time   you only step into the door if nobody is coming through the other way;
                        otherwise you wait beside the door, on your right, out of their path
    follow              heading for the door, nobody walks faster than the person right in front
    keep right          each direction aims slightly right of the door's middle

The wall is a chain of small circles in `obstacles.csv`, so the core's obstacle features and the
planner's obstacle check work unchanged. Agents enter (the rush crowd) and leave (at the exits).
`activity` says what a person is doing: walk, stop, queue, wait, door, leave.

    python -m foresight.domains.doorway                                    # print the summary
    python -m foresight.domains.doorway --out data/doorway --test-out data/doorway_test
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.domains.crowd import ARRIVE, MAX_FORCE, PERSON_R, ROBOT_R, CrowdSim, interaction_force, save

L = 8.0                   # the space is [-L, L]^2: room A x < 0, room B x > 0
DOOR_HALF = 0.75          # door is |y| < 0.75 at x = 0 (1.5 m wide)
WALL_R, WALL_STEP = 0.25, 0.4    # wall = overlapping circles (no gap a person fits through)
LANE = 0.3                # keep right: offset of each direction's line through the door
APPROACH = 1.4            # m in front of the door: where you line up
THROUGH = 2.5             # m behind the door: where you aim while passing it
WAIT_SPOT = (1.7, 1.4)    # m back from the wall and to your right: where you wait for the other side
WALL_A, WALL_B = 3.0, 0.15   # wall repulsion, nearest wall circle only (door jambs must not block the door;
                             # the hard push-out from crowd.py keeps everyone out of the wall)
NEAR_DOOR = 1.0           # within this of the line-up point you go through
EXITS_X = -L + 0.6        # rush crowd leaves through the left wall
ENTRY_X = L - 0.6         # ... and arrives through the right wall (the meeting room)
PRESS = 0.03              # m: bodies pressed deeper than this into each other (before being separated) = contact
FOLLOW_GAP = 0.35         # m between bodies: closer than this behind someone, you match their pace
REGIMES = {
    "trickle": dict(v0=1.0, tau=0.5,  A=2.0, B=0.5, stop=0.03, stop_s=(1.0, 3.0)),
    "rush":    dict(v0=1.5, tau=0.35, A=2.0, B=0.3, stop=0.0,  stop_s=(0.5, 1.0)),
}
COLUMNS = ["episode", "t", "agent_id", "kind", "is_ego", "x", "y", "vx", "vy", "radius",
           "regime", "goal_x", "goal_y", "collided", "activity"]
MAP = dict(rooms=[dict(name="Room A", x=[-L, 0.0], y=[-L, L]), dict(name="Room B", x=[0.0, L], y=[-L, L])],
           wall=dict(x=0.0, y=[[-L, -DOOR_HALF], [DOOR_HALF, L]], thickness=2 * WALL_R),
           door=dict(x=0.0, y=[-DOOR_HALF, DOOR_HALF]),
           entry=dict(name="from the meeting room", x=L, y=[-4.0, 4.0]),
           exits=dict(name="exits", x=-L, y=[-5.0, 5.0]))


def wall_obstacles() -> list[dict]:
    ys = np.arange(DOOR_HALF + WALL_R, L + 1e-9, WALL_STEP)
    return [dict(x=0.0, y=round(float(s * y), 6), r=WALL_R) for s in (1, -1) for y in ys]


@dataclass(frozen=True)
class DoorwayConfig:
    n_people: int = 8                # trickle: always there
    n_rush: int = 24                 # arrive at switch_t
    duration: float = 60.0
    dt: float = 0.2
    half_size: float = L
    switch_t: float | None = None
    rush_spawn_s: float = 3.0        # the crowd pours in over this many seconds
    jitter: float = 0.1
    robot_speed: float = 1.0

    def regime_at(self, t: float) -> str:
        return "rush" if self.switch_t is not None and t >= self.switch_t - 1e-9 else "trickle"


def episode_config(seed: int, base: DoorwayConfig = DoorwayConfig()) -> DoorwayConfig:
    """Odd seeds switch to rush at a random time in 35-60% of the episode; even seeds stay trickle."""
    if seed % 2 == 0:
        return replace(base, switch_t=None)
    u = np.random.default_rng([seed, 7]).uniform(0.35, 0.6)
    return replace(base, switch_t=round(round(u * base.duration / base.dt) * base.dt, 6))


class DoorwaySim(CrowdSim):
    """One episode. Agent 0 is the robot, 1..n_people the trickle, then the rush crowd (if any).
    Closed-loop hook as in crowd.py: reset() -> state, step(robot_cmd) -> (state, {"collided"})."""

    def __init__(self, seed: int, config: DoorwayConfig = DoorwayConfig()):
        self.seed, self.cfg = seed, config

    # ---------- setup ----------
    def reset(self) -> list[dict]:
        c = self.cfg
        rng = self.rng = np.random.default_rng(self.seed)
        self.obstacles = wall_obstacles()
        self._obs_xy = np.array([[o["x"], o["y"]] for o in self.obstacles])
        self._obs_r = np.array([o["r"] for o in self.obstacles])
        n_rush = c.n_rush if c.switch_t is not None else 0
        n = 1 + c.n_people + n_rush
        self.radius = np.full(n, PERSON_R); self.radius[0] = ROBOT_R
        self.P = np.full((n, 2), np.nan); self.V = np.zeros((n, 2))
        self.final = np.full((n, 2), np.nan)                  # where the agent really wants to go
        self.G = np.full((n, 2), np.nan)                      # current waypoint (door line-up, door, goal)
        self.phase = np.zeros(n, int)                         # 0 free, 1 line up, 2 through the door
        self.blocked = np.zeros(n, bool)                      # lined up, waiting for the other side
        self.active = np.zeros(n, bool)
        self.rush = np.zeros(n, bool); self.rush[1 + c.n_people:] = True
        self.jit = rng.normal(1.0, c.jitter, (n, 4)).clip(0.7, 1.3)
        self.stop_left = np.zeros(n)
        self.activity = np.array(["walk"] * n, dtype=object)
        self.to_spawn = list(range(1 + c.n_people, n))
        self.external = False                                 # robot driven by a planner (step(robot_cmd))

        side = rng.choice([-1.0, 1.0])                        # the robot shuttles between the rooms
        self.robot_start = np.array([side * (L - 1.5), rng.uniform(-L / 2, L / 2)])
        self.robot_goal = np.array([-side * (L - 1.5), rng.uniform(-L / 2, L / 2)])
        self.P[0], self.final[0], self.active[0] = self.robot_start, self.robot_goal, True
        for i in range(1, 1 + c.n_people):
            self.P[i] = self._room_point(rng.choice([-1.0, 1.0]), self.radius[i])
            self.active[i] = True
            self.final[i] = self._new_goal(i)
        self.k = 0
        self._route()
        return self._state(np.zeros(n, bool))

    def _room_point(self, side, r, avoid=True):
        for _ in range(1000):
            p = np.array([side * self.rng.uniform(0.9, L - 0.8), self.rng.uniform(-L + 0.8, L - 0.8)])
            others = self.P[self.active]
            if not avoid or np.all(np.hypot(*(others - p).T) > 2 * r + 0.3):
                return p
        raise RuntimeError("room too crowded to place an agent")

    def _new_goal(self, i):
        """60% of goals are in the other room, so people keep using the door."""
        here = np.sign(self.P[i, 0]) or 1.0
        side = -here if self.rng.random() < 0.6 else here
        return self._room_point(side, PERSON_R, avoid=False)

    # ---------- routing through the door ----------
    def _door_busy(self, d):
        """Someone near the door is committed to passing it in direction -d."""
        act = np.flatnonzero(self.active & (self.phase == 2))
        return any(np.sign(self.final[j, 0]) == -d and abs(self.P[j, 0]) < APPROACH + NEAR_DOOR + 0.2 for j in act)

    def _route(self):
        """Update phase and waypoint of every active agent (robot included)."""
        for i in np.flatnonzero(self.active):
            p, f = self.P[i], self.final[i]
            d = np.sign(f[0]) or 1.0                           # travel direction along x
            lane = np.array([0.0, -d * LANE])                  # keep right
            here = np.sign(p[0]) or -d
            if self.phase[i] == 0 and here != d:
                self.phase[i] = 1
            self.blocked[i] = False
            if self.phase[i] == 1:
                line_up = np.array([-d * APPROACH, 0.0]) + lane
                # a planner-driven robot does not use the give-way rule: avoiding people is the planner's job
                # (and a waypoint it could stand on would end run_closed's episode as "reached")
                self.blocked[i] = self._door_busy(d) and not (i == 0 and self.external)
                if not self.blocked[i] and (np.linalg.norm(p - line_up) < NEAR_DOOR
                                            or (abs(p[0]) < APPROACH - 0.2 and abs(p[1]) < DOOR_HALF + 0.3)):
                    self.phase[i] = 2
            if self.phase[i] == 2 and np.sign(p[0]) == d and abs(p[0]) > 1.2:
                self.phase[i] = 0
            if self.phase[i] == 1 and self.blocked[i]:          # each person has their own spot near it
                u = self.jit[i, :2] - 0.7                       # in [0, 0.6]
                self.G[i] = np.array([-d * (WAIT_SPOT[0] + u[0]), -d * (WAIT_SPOT[1] + 2 * u[1])])
            else:
                self.G[i] = {0: f, 1: np.array([-d * APPROACH, 0.0]) + lane,
                             2: np.array([d * THROUGH, 0.0]) + lane}[self.phase[i]]

    def _spawn_rush(self):
        c = self.cfg
        if c.switch_t is None or self.k * c.dt < c.switch_t - 1e-9 or not self.to_spawn:
            return
        per_step = int(np.ceil(c.n_rush * c.dt / c.rush_spawn_s))
        for _ in range(per_step):
            if not self.to_spawn:
                break
            i = self.to_spawn[0]
            for _ in range(20):                               # a free spot at the entrance, or wait
                p = np.array([ENTRY_X - self.rng.uniform(0, 0.6), self.rng.uniform(-4.0, 4.0)])
                if np.all(np.hypot(*(self.P[self.active] - p).T) > 2 * PERSON_R + 0.2):
                    break
            else:
                return
            self.to_spawn.pop(0)
            self.P[i], self.V[i] = p, np.array([-1.2, 0.0])
            self.final[i] = np.array([EXITS_X, self.rng.uniform(-5.0, 5.0)])
            self.active[i] = True

    # ---------- dynamics ----------
    def step(self, robot_cmd: tuple[float, float] | None = None) -> tuple[list[dict], dict]:
        """Advance one dt. robot_cmd = (vx, vy), or None for the default door-crossing policy."""
        c, rng, dt = self.cfg, self.rng, self.cfg.dt
        p = REGIMES[c.regime_at(self.k * dt)]
        self.external = robot_cmd is not None
        self._spawn_rush()
        self._route()
        idx = np.flatnonzero(self.active)
        P, V = self.P[idx], self.V[idx].copy()
        people = idx != 0
        v0 = p["v0"] * self.jit[idx, 0]; A = p["A"] * self.jit[idx, 1]; B = p["B"] * self.jit[idx, 2]
        v_des = v0.copy()
        act = np.array(["walk"] * len(idx), dtype=object)

        to_g = self.G[idx] - P
        e = to_g / np.maximum(np.linalg.norm(to_g, axis=1, keepdims=True), 1e-9)
        for j, i in enumerate(idx):
            if i == 0:
                continue
            # random stops while wandering (trickle only)
            if self.stop_left[i] > 0:
                self.stop_left[i] -= dt
            elif self.phase[i] == 0 and rng.random() < p["stop"] * self.jit[i, 3] * dt:
                self.stop_left[i] = rng.uniform(*p["stop_s"])
            if self.stop_left[i] > 0 and self.phase[i] == 0:
                v_des[j], act[j] = 0.0, "stop"
            if self.phase[i]:
                act[j] = "door" if abs(P[j, 0]) < 1.0 else "walk"
        # follow: heading for the door, nobody walks faster than the person right in front
        dvec = P[None, :, :] - P[:, None, :]                       # j minus i
        dist = np.linalg.norm(dvec, axis=2); np.fill_diagonal(dist, np.inf)
        ahead = (dvec * e[:, None, :]).sum(axis=2) > 0.8 * dist
        close = dist < self.radius[idx][:, None] + self.radius[idx][None, :] + FOLLOW_GAP
        pace = np.maximum((V[None, :, :] * e[:, None, :]).sum(axis=2), 0.0)
        for j, i in enumerate(idx):
            if i == 0 or not self.phase[i]:
                continue
            front = ahead[j] & close[j]
            if front.any():
                lim = pace[j, front].min()
                if lim < v_des[j]:
                    v_des[j] = lim
                    if lim < 0.3:
                        act[j] = "queue"
        # one way at a time: blocked people walk to the spot beside the door and stand there
        for j, i in enumerate(idx):
            if i and self.blocked[i]:
                v_des[j] = min(v_des[j], 0.8 * np.linalg.norm(to_g[j]))
                act[j] = "wait" if np.linalg.norm(to_g[j]) < 1.0 else "walk"

        F = (e * v_des[:, None] - V) / p["tau"]
        F += interaction_force(P, V, e, self.radius[idx], A, B, strong=idx == 0)
        od = P[:, None, :] - self._obs_xy[None]
        odist = np.maximum(np.linalg.norm(od, axis=2), 1e-6)
        m = odist.argmin(axis=1); r_ = np.arange(len(idx))
        gap = odist[r_, m] - self._obs_r[m] - self.radius[idx]
        F += (WALL_A * np.exp(-gap / WALL_B))[:, None] * od[r_, m] / odist[r_, m, None]
        F += 3.0 * (np.exp(-(P + L - self.radius[idx][:, None]) / 0.25)
                    - np.exp(-(L - P - self.radius[idx][:, None]) / 0.25))
        fm = np.linalg.norm(F, axis=1, keepdims=True)
        F = np.where(fm > MAX_FORCE, F / np.maximum(fm, 1e-9) * MAX_FORCE, F)
        V[people] += F[people] * dt
        sp = np.linalg.norm(V[people], axis=1)
        cap = 1.3 * np.maximum(v0[people], 0.5)
        V[people] *= np.minimum(1.0, cap / np.maximum(sp, 1e-9))[:, None]

        # robot
        V[~people] = self._robot_policy(P, idx) if robot_cmd is None else np.asarray(robot_cmd, float)
        self.V[idx] = V
        self.P[idx] = np.clip(P + V * dt, -L + 0.05, L - 0.05)
        self.activity[idx] = act
        pressed = self._separate_bodies(idx)
        self._push_out_of_obstacles()
        self.k += 1

        # arrivals: rush crowd leaves at the exits, the trickle picks a new goal, the robot turns back
        for i in np.flatnonzero(self.active):
            if self.phase[i] == 0 and np.linalg.norm(self.final[i] - self.P[i]) < ARRIVE:
                if i == 0:
                    self.robot_start, self.robot_goal = self.robot_goal, self.robot_start
                    self.final[0] = self.robot_goal
                elif self.rush[i]:
                    self.active[i] = False
                    self.P[i] = np.nan
                else:
                    self.final[i] = self._new_goal(i)
        self._route()
        collided = self._collisions(pressed)
        return self._state(collided), {"collided": bool(collided[0])}

    def _separate_bodies(self, idx, passes=3):
        """Bodies don't overlap: push overlapping people apart (the robot is not moved, so planner
        commands stay exact; a person overlapping it moves the whole way). Returns, per agent, whether
        it was pressed deeper than PRESS into someone before separation: that is a contact."""
        pressed = np.zeros(len(self.P), bool)
        r = self.radius[idx]
        for k in range(passes):
            P = self.P[idx]
            d = P[:, None] - P[None]
            dist = np.maximum(np.linalg.norm(d, axis=2), 1e-9)
            over = r[:, None] + r[None] - dist
            np.fill_diagonal(over, -1.0)
            if k == 0:
                pressed[idx] = (over > PRESS).any(axis=1)
            if not (over > 0).any():
                break
            share = np.where(idx[None, :] == 0, 1.0, 0.5)          # j is the robot: i takes the whole push
            move = (np.clip(over, 0, None) * share)[..., None] * d / dist[..., None]
            move = move.sum(axis=1); move[idx == 0] = 0.0
            self.P[idx] = P + move
        return pressed

    def _robot_policy(self, P, idx):
        """Straight at the current waypoint (beside the door while the other side passes), slowing near people."""
        j = int(np.flatnonzero(idx == 0)[0])
        to_g = self.G[0] - P[j]
        dg = np.linalg.norm(to_g)
        others = np.delete(np.arange(len(idx)), j)
        near = np.min(np.linalg.norm(P[others] - P[j], axis=1)) if len(others) else np.inf
        speed = self.cfg.robot_speed * np.clip((near - 0.8) / 1.5, 0.0, 1.0)
        if self.blocked[0]:
            speed = min(speed, 0.8 * dg)
        return to_g / max(dg, 1e-9) * speed

    def _collisions(self, pressed=None):
        """Contact with a person or the robot (pressed in before separation, or still overlapping),
        or with the wall (as in crowd.py: touching it, including being pushed out of it)."""
        hit = np.zeros(len(self.P), bool) if pressed is None else pressed.copy()
        idx = np.flatnonzero(self.active)
        P = self.P[idx]
        d = np.linalg.norm(P[:, None] - P[None], axis=2)
        h = d < self.radius[idx][:, None] + self.radius[idx][None] - PRESS
        np.fill_diagonal(h, False)
        od = np.linalg.norm(P[:, None] - self._obs_xy[None], axis=2)
        hit[idx] |= h.any(axis=1) | (od < self._obs_r[None] + self.radius[idx][:, None] + 1e-6).any(axis=1)
        return hit

    def _state(self, collided) -> list[dict]:
        t = round(self.k * self.cfg.dt, 6)
        regime = self.cfg.regime_at(t)
        return [dict(t=t, agent_id=int(i), kind="robot" if i == 0 else "person", is_ego=int(i == 0),
                     x=self.P[i, 0], y=self.P[i, 1], vx=self.V[i, 0], vy=self.V[i, 1], radius=self.radius[i],
                     regime="robot" if i == 0 else regime, goal_x=self.G[i, 0], goal_y=self.G[i, 1],
                     collided=int(collided[i]), activity="robot" if i == 0 else self.activity[i])
                for i in np.flatnonzero(self.active)]


# ---------- episodes and datasets ----------
def simulate(seed: int, config: DoorwayConfig = DoorwayConfig(), episode: int | None = None):
    """One episode, ground truth. The episode id is the seed unless given."""
    episode = seed if episode is None else episode
    sim = DoorwaySim(seed, config)
    rows = sim.reset()
    for _ in range(int(round(config.duration / config.dt))):
        rows += sim.step()[0]
    tracks = pd.DataFrame(rows)
    tracks.insert(0, "episode", episode)
    obstacles = pd.DataFrame([dict(episode=episode, obstacle_id=i, shape="circle", x=o["x"], y=o["y"], r=o["r"],
                                   w=np.nan, h=np.nan, angle=np.nan) for i, o in enumerate(sim.obstacles)])
    meta = dict(seed=seed, duration=config.duration, regime_switch_t=config.switch_t, regime_before="trickle",
                regime_after="rush" if config.switch_t is not None else None)
    return tracks[COLUMNS], obstacles, meta


def generate(seeds, config: DoorwayConfig = DoorwayConfig()):
    """One episode per seed (episode id = seed); odd seeds switch to rush. Same seeds, same data."""
    tracks, obstacles, eps = [], [], {}
    for s in seeds:
        tr, ob, me = simulate(int(s), episode_config(int(s), config))
        tracks.append(tr); obstacles.append(ob); eps[str(int(s))] = me
    meta = dict(sim="doorway-v1", dt=config.dt, bounds=[-L, -L, L, L], episodes=eps, map=MAP,
                config={k: v for k, v in asdict(config).items() if k != "switch_t"},
                notes="doorway bottleneck: two rooms, one 1.5 m door (wall = circle chain in obstacles.csv); "
                      "trickle -> rush (24 people from a meeting room) at regime_switch_t on odd seeds; "
                      "robot shuttles between the rooms, slowing near people and giving way at the door")
    return pd.concat(tracks, ignore_index=True), pd.concat(obstacles, ignore_index=True), meta


def save_all(out_dir, tracks, obstacles, meta):
    save(out_dir, tracks, obstacles, meta)
    (Path(out_dir) / "map.json").write_text(json.dumps(MAP, indent=1))


def summary(tracks: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Per regime: people, speed, share standing still, queue at the door, collisions per minute,
    door crossings per minute and how wrong constant velocity is at 2 s."""
    dt = float(meta["dt"]); k2 = round(2.0 / dt)
    tr = tracks.sort_values(["episode", "agent_id", "t"]).copy()
    tr["speed"] = np.hypot(tr.vx, tr.vy)
    g = tr.groupby(["episode", "agent_id"])
    same = np.isclose(g.t.shift(-k2) - tr.t, 2.0)
    tr["cv_err_2s"] = np.hypot(g.x.shift(-k2) - (tr.x + 2 * tr.vx), g.y.shift(-k2) - (tr.y + 2 * tr.vy)).where(same)
    tr["crossed"] = (np.sign(g.x.shift(1)) != np.sign(tr.x)) & g.x.shift(1).notna()
    # contact type: another body, or the wall (squeezed against the door frame)
    W = np.array([[o["x"], o["y"]] for o in wall_obstacles()])
    near_wall = (np.hypot(tr.x.to_numpy()[:, None] - W[None, :, 0], tr.y.to_numpy()[:, None] - W[None, :, 1]).min(axis=1)
                 - WALL_R - tr.radius.to_numpy()) < 1e-3
    tr["wall"] = (tr.collided == 1) & near_wall
    tr["body"] = (tr.collided == 1) & ~near_wall
    for c in ("wall", "body"):
        tr[c + "_start"] = tr[c] & ~g[c].shift(1, fill_value=False).astype(bool)
    ppl = tr[tr.kind == "person"]
    rows = []
    for reg, d in ppl.groupby("regime", sort=False):
        minutes = d.groupby("episode").t.nunique().sum() * dt / 60
        near = d[(np.abs(d.x) < 3) & (np.abs(d.y) < 3)]
        per_step = d.groupby(["episode", "t"])
        rows.append(dict(regime=reg, episodes=d.episode.nunique(), minutes=round(minutes, 1),
                         people_present=per_step.size().mean(), people_max=per_step.size().max(),
                         mean_speed=d.speed.mean(), stopped=(d.speed < 0.2).mean(),
                         queue_at_door=(near.speed < 0.3).groupby([near.episode, near.t]).sum().reindex(
                             per_step.size().index, fill_value=0).mean(),
                         queueing_share=d.activity.isin(["queue", "wait"]).mean(),
                         crossings_per_min=d.crossed.sum() / minutes,
                         person_contacts_per_min=d.body_start.sum() / minutes,
                         wall_contacts_per_min=d.wall_start.sum() / minutes, cv_err_2s=d.cv_err_2s.mean()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", type=int, default=80, help="training seeds 0..N-1")
    ap.add_argument("--test", type=int, default=20, help="held-out seeds 1000..1000+N-1")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--out", help="write the training set here (e.g. data/doorway)")
    ap.add_argument("--test-out", help="write the held-out set here (e.g. data/doorway_test)")
    a = ap.parse_args()
    cfg = DoorwayConfig(duration=a.duration)
    pd.set_option("display.width", 200)
    for name, seeds, out in (("train", range(a.train), a.out), ("test", range(1000, 1000 + a.test), a.test_out)):
        if not len(seeds):
            continue
        tracks, obstacles, meta = generate(seeds, cfg)
        n_sw = sum(m["regime_switch_t"] is not None for m in meta["episodes"].values())
        print(f"\n{name}: seeds {seeds.start}-{seeds.stop - 1}, {len(seeds)} episodes of {a.duration:g} s, "
              f"{n_sw} switch to rush, {len(tracks)} rows")
        print(summary(tracks, meta).round(3).to_string(index=False))
        if out:
            save_all(out, tracks, obstacles, meta)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
