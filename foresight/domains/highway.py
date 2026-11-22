"""Highway cars (domain 2) from highway-env, writing the INTERFACE.md format.

Same core, another embodiment: cars on a 4-lane straight highway instead of people around a delivery robot.
Only this adapter knows about highway-env; the core reads tracks.csv / obstacles.csv / meta.json as for the crowd.

Simulator: highway-env 1.12 `highway-v0` builds the road and the initial traffic (seeded `env.reset(seed=...)`).
The road is then stepped directly (`road.act(); road.step(1/15)`, 15 Hz physics, recorded every 3rd step, so
dt = 0.2 s). Stepping the road rather than `env.step` means an ego crash cannot end the episode early, and the
ego is a car like the others: it is replaced by an `IDMVehicle` (IDM car following + MOBIL lane changes, default
parameters, target speed 25 m/s) that never changes regime. It is `is_ego = 1`, so the core uses it as "the robot".

Other cars: `IDMVehicle` (highway-env's default traffic model), with the IDM / MOBIL settings of the current
crowd-wide behaviour regime set on each car (instance attributes, per-car jitter on target speed and time gap):

    defensive   target 20-24 m/s, time gap 2.2 s, gentle acceleration, polite, rare lane changes
    normal      target 24-28 m/s, time gap 1.5 s, highway-env defaults otherwise
    aggressive  target 30-34 m/s, time gap 0.6 s, hard acceleration and braking, impolite, frequent lane changes

Every episode switches: all other cars start in `regime_before` and switch to `regime_after` at `regime_switch_t`
(35-65% of the episode, on the dt grid). The six ordered (before, after) pairs are dealt out evenly over the
episodes in a seeded order. Cars keep their agent_id across the switch (same car, new driver settings).

Recorded per car per step: x, y (m, road frame: x along the road, y across, lanes 4 m apart), vx, vy (speed and
heading), radius 2.69 m (half the diagonal of highway-env's 5 m x 2 m car; INTERFACE.md's rule for cars; constant,
so it carries no information), length 5, width 2, heading, kind "car", regime ("ego" for the ego), collided,
ext_lane (lane index; analysis only, not a model input). A car that crashes is recorded at the crash step and then
removed from the road (it would otherwise stop dead in its lane). Lanes have a 40 m/s speed limit (highway-env
default 30 would cap the aggressive target speed). No obstacles; `bounds` null as in eth_ucy.py (no walls).

    python -m foresight.domains.highway --episodes 72 --seed 0 --out data/highway
    python tools/tune_kalman.py data/highway --noise 0.1 --split
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

DT, SIM_HZ = 0.2, 15
SUB = round(DT * SIM_HZ)                  # physics steps per recorded step
LENGTH, WIDTH = 5.0, 2.0
RADIUS = round(0.5 * float(np.hypot(LENGTH, WIDTH)), 2)   # 2.69 m
SPEED_LIMIT = 40.0
EGO_TARGET = 25.0
# target speed range (m/s), then IDM / MOBIL constants of highway_env.vehicle.behavior.IDMVehicle
REGIMES = {
    "defensive": dict(target=(20.0, 24.0), TIME_WANTED=2.2, DISTANCE_WANTED=12.0, COMFORT_ACC_MAX=1.5,
                      COMFORT_ACC_MIN=-3.0, POLITENESS=0.8, LANE_CHANGE_MIN_ACC_GAIN=1.0,
                      LANE_CHANGE_MAX_BRAKING_IMPOSED=1.0, LANE_CHANGE_DELAY=3.0),
    "normal":    dict(target=(24.0, 28.0), TIME_WANTED=1.5, DISTANCE_WANTED=10.0, COMFORT_ACC_MAX=3.0,
                      COMFORT_ACC_MIN=-5.0, POLITENESS=0.0, LANE_CHANGE_MIN_ACC_GAIN=0.2,
                      LANE_CHANGE_MAX_BRAKING_IMPOSED=2.0, LANE_CHANGE_DELAY=1.0),
    "aggressive": dict(target=(30.0, 34.0), TIME_WANTED=0.6, DISTANCE_WANTED=6.0, COMFORT_ACC_MAX=5.0,
                       COMFORT_ACC_MIN=-7.0, POLITENESS=0.0, LANE_CHANGE_MIN_ACC_GAIN=0.05,
                       LANE_CHANGE_MAX_BRAKING_IMPOSED=4.0, LANE_CHANGE_DELAY=0.5),
}
JITTER = 0.1                              # sd of the per-car multiplicative jitter on the time gap
COLUMNS = ["episode", "t", "agent_id", "kind", "is_ego", "x", "y", "vx", "vy", "radius", "length", "width",
           "heading", "regime", "collided", "ext_lane"]


def _apply(v, regime, rng):
    p = REGIMES[regime]
    v.target_speed = float(rng.uniform(*p["target"]))
    for k, val in p.items():
        if k != "target":
            setattr(v, k, float(val))
    v.TIME_WANTED = float(p["TIME_WANTED"] * np.clip(rng.normal(1.0, JITTER), 0.7, 1.3))


def simulate(seed, before, after, switch_t, duration=40.0, n_vehicles=20, lanes=4, episode=0):
    """One episode. Returns (tracks DataFrame, n_crashes)."""
    import gymnasium as gym
    import highway_env  # noqa: F401  (registers the envs)
    from highway_env.vehicle.behavior import IDMVehicle

    env = gym.make("highway-v0", config={"simulation_frequency": SIM_HZ, "policy_frequency": round(1 / DT),
                                         "vehicles_count": n_vehicles, "lanes_count": lanes,
                                         "duration": duration, "vehicles_density": 1.0})
    env.reset(seed=int(seed))
    road = env.unwrapped.road
    for lanes_ in road.network.graph.values():
        for ls in lanes_.values():
            for lane in ls:
                lane.speed_limit = SPEED_LIMIT
    old_ego = env.unwrapped.vehicle
    ego = IDMVehicle.create_from(old_ego)
    ego.target_speed = EGO_TARGET
    road.vehicles[road.vehicles.index(old_ego)] = ego
    rng = np.random.default_rng([int(seed), 1])
    ids = {}
    for i, v in enumerate(road.vehicles):
        ids[id(v)] = i
        if v is not ego:
            _apply(v, before, rng)
    rows, crashes = [], 0
    n_steps = int(round(duration / DT))
    for k in range(n_steps + 1):
        t = round(k * DT, 6)
        if k > 0 and abs(t - switch_t) < 1e-6:
            for v in road.vehicles:
                if v is not ego:
                    _apply(v, after, rng)
        regime = after if t >= switch_t - 1e-9 else before
        for v in road.vehicles:
            vel = v.speed * np.array([np.cos(v.heading), np.sin(v.heading)])
            rows.append((episode, t, ids[id(v)], "car", int(v is ego), float(v.position[0]), float(v.position[1]),
                         float(vel[0]), float(vel[1]), RADIUS, LENGTH, WIDTH, float(v.heading),
                         "ego" if v is ego else regime, int(bool(v.crashed)), int(v.lane_index[2])))
        crashed = [v for v in road.vehicles if v.crashed and v is not ego]
        crashes += len(crashed)
        for v in crashed:
            road.vehicles.remove(v)
        if ego.crashed:
            crashes += 1
            ego.crashed = False                # keep the ego driving; logged as collided at that step
        if k == n_steps:
            break
        for _ in range(SUB):
            road.act()
            road.step(1.0 / SIM_HZ)
    env.close()
    return pd.DataFrame(rows, columns=COLUMNS), crashes


def generate(n_episodes, seed=0, duration=40.0, n_vehicles=20, lanes=4, verbose=True):
    rng = np.random.default_rng(seed)
    names = list(REGIMES)
    pairs = [(a, b) for a in names for b in names if a != b]
    order = np.concatenate([rng.permutation(len(pairs)) for _ in range(int(np.ceil(n_episodes / len(pairs))))])
    tracks, eps, t0 = [], {}, time.perf_counter()
    for e in range(n_episodes):
        ep_seed = int(rng.integers(2**31))
        before, after = pairs[order[e]]
        t_sw = round(round(rng.uniform(0.35, 0.65) * duration / DT) * DT, 6)
        tr, n_crash = simulate(ep_seed, before, after, t_sw, duration, n_vehicles, lanes, episode=e)
        tracks.append(tr)
        eps[str(e)] = dict(seed=ep_seed, duration=duration, regime_switch_t=t_sw, regime_before=before,
                           regime_after=after, crashes=int(n_crash))
        if verbose:
            print(f"episode {e}: {before} -> {after} at {t_sw:.1f} s, {tr.agent_id.nunique()} cars, "
                  f"{n_crash} crashes ({time.perf_counter() - t0:.0f} s)", flush=True)
    tracks = pd.concat(tracks, ignore_index=True).sort_values(["episode", "t", "agent_id"]).reset_index(drop=True)
    meta = dict(sim="highway-env-1.12:highway-v0", dt=DT, bounds=None, episodes=eps,
                config=dict(n_vehicles=n_vehicles, lanes=lanes, duration=duration, sim_hz=SIM_HZ,
                            speed_limit=SPEED_LIMIT, ego_target=EGO_TARGET, radius=RADIUS, jitter=JITTER,
                            regimes=REGIMES),
                notes="highway-env highway-v0 road and initial traffic; road stepped directly at 15 Hz, recorded "
                      "at 5 Hz. Ego = IDMVehicle (defaults, target 25 m/s, regime 'ego', never switches). Other "
                      "cars IDMVehicle with regime IDM/MOBIL settings; every episode switches all of them at "
                      "regime_switch_t. Crashed cars removed after the crash step. No obstacles, no walls.")
    return tracks, meta


def save(out_dir, tracks, meta):
    o = Path(out_dir); o.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(o / "tracks.csv", index=False)
    pd.DataFrame(columns=["episode", "obstacle_id", "shape", "x", "y", "r", "w", "h", "angle"]).to_csv(
        o / "obstacles.csv", index=False)
    (o / "meta.json").write_text(json.dumps(meta, indent=1))


def summary(tracks, meta, h=2.0):
    """Per regime: speed, ground-truth CV error at +h (true velocity), lane changes per car-minute."""
    dt = float(meta["dt"]); hs = round(h / dt)
    c = tracks[tracks.is_ego == 0].sort_values(["episode", "agent_id", "t"]).copy()
    g = c.groupby(["episode", "agent_id"])
    c["cv_err"] = np.hypot(g.x.shift(-hs) - (c.x + c.vx * h), g.y.shift(-hs) - (c.y + c.vy * h))
    c["lc"] = (g.ext_lane.diff().fillna(0) != 0).astype(int)
    c["speed"] = np.hypot(c.vx, c.vy)
    out = c.groupby("regime").agg(rows=("t", "size"), speed=("speed", "mean"), cv_err_med=("cv_err", "median"),
                                  cv_err_mean=("cv_err", "mean"), cv_err_p90=("cv_err", lambda s: s.quantile(0.9)),
                                  lane_changes=("lc", "sum"))
    out["lc_per_car_min"] = out.lane_changes / (out.rows * dt / 60)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vehicles", type=int, default=20)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--out", help="write tracks.csv, obstacles.csv, meta.json here")
    a = ap.parse_args()
    tracks, meta = generate(a.episodes, a.seed, a.duration, a.vehicles, a.lanes)
    pd.set_option("display.width", 160)
    print(summary(tracks, meta).round(3).to_string())
    if a.out:
        save(a.out, tracks, meta)
        print(f"wrote {a.episodes} episodes, {len(tracks)} rows to {a.out}")


if __name__ == "__main__":
    main()
