import numpy as np
import pandas as pd
import pytest

from foresight.core.schema import validate
from foresight.domains.crowd import OBSERVED, REGIMES, CrowdConfig, CrowdSim, generate, observe, simulate

SHORT = CrowdConfig(n_people=10, duration=20.0)


@pytest.fixture(scope="module")
def data():
    return generate(4, seed=7, config=SHORT)


def people(tracks):
    return tracks[tracks["kind"] == "person"]


def test_dataset_passes_core_validation(data):
    tracks, obstacles, meta = data
    validate(tracks, meta)
    assert meta["dt"] == 0.2
    assert tracks.groupby("episode")["t"].nunique().eq(101).all()      # 20 s at 5 Hz, t=0 included
    assert (tracks.groupby("episode")["is_ego"].sum() == 101).all()     # one robot every step
    assert list(obstacles.columns) == ["episode", "obstacle_id", "shape", "x", "y", "r", "w", "h", "angle"]


def test_same_seed_same_data_other_seed_differs():
    a = generate(2, seed=3, config=SHORT)[0]
    b = generate(2, seed=3, config=SHORT)[0]
    c = generate(2, seed=4, config=SHORT)[0]
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["x"], c["x"])


def test_regime_switch_for_whole_crowd_at_given_time():
    cfg = CrowdConfig(n_people=8, duration=10.0, regime="calm", switch_t=4.0, switch_to="erratic")
    tr = people(simulate(1, cfg)[0])
    assert set(tr.loc[tr["t"] < 4.0, "regime"]) == {"calm"}
    assert set(tr.loc[tr["t"] >= 4.0, "regime"]) == {"erratic"}


def test_generate_switches_in_some_episodes_only():
    meta = generate(12, seed=0, config=CrowdConfig(n_people=3, duration=4.0))[2]
    switched = [e for e in meta["episodes"].values() if e["regime_switch_t"] is not None]
    assert 0 < len(switched) < 12
    for e in switched:
        assert e["regime_after"] != e["regime_before"]


def test_people_reach_goals_and_pick_new_ones():
    tr = people(simulate(2, CrowdConfig(n_people=8, duration=40.0))[0])
    new_goals = tr.groupby("agent_id")[["goal_x", "goal_y"]].nunique().max(axis=1)
    assert (new_goals > 1).mean() > 0.5


def test_people_avoid_each_other_obstacles_and_robot():
    for regime in REGIMES:
        tr = simulate(5, CrowdConfig(regime=regime, duration=30.0))[0]
        assert people(tr)["collided"].mean() < 0.02, regime


def test_avoidance_actually_matters():
    """Turning repulsion off must produce many more collisions, or the test above proves nothing."""
    import foresight.domains.crowd as crowd
    saved = {k: dict(v) for k, v in REGIMES.items()}
    try:
        for v in REGIMES.values():
            v["A"] = 0.0
        tr = simulate(5, CrowdConfig(regime="calm", duration=30.0))[0]
    finally:
        for k in REGIMES:
            crowd.REGIMES[k].update(saved[k])
    assert people(tr)["collided"].mean() > 0.02


def test_regimes_differ_as_described():
    stats = {}
    for regime in REGIMES:
        tr = people(simulate(11, CrowdConfig(regime=regime, duration=30.0))[0])
        speed = np.hypot(tr["vx"], tr["vy"])
        stats[regime] = dict(speed=speed[tr["t"] > 3].mean(), stopped=(speed < 0.2).mean())
    assert stats["hurried"]["speed"] > stats["erratic"]["speed"] > stats["calm"]["speed"]
    assert stats["erratic"]["stopped"] > 2 * stats["calm"]["stopped"]


def test_erratic_people_make_sharp_turns():
    def sharp_turn_rate(regime):
        tr = people(simulate(4, CrowdConfig(regime=regime, duration=30.0))[0]).sort_values(["agent_id", "t"])
        h = np.arctan2(tr["vy"], tr["vx"])
        dh = np.abs((h.groupby(tr["agent_id"]).diff(5) + np.pi) % (2 * np.pi) - np.pi)   # over 1 s
        moving = np.hypot(tr["vx"], tr["vy"]) > 0.5
        return (dh[moving] > np.radians(60)).mean()
    assert sharp_turn_rate("erratic") > 2 * sharp_turn_rate("calm")


def test_per_person_jitter():
    sim = CrowdSim(0, CrowdConfig(n_people=20))
    sim.reset()
    assert sim.jit[1:, 0].std() > 0.03


def test_robot_walks_a_straight_line():
    tr = simulate(9, CrowdConfig(n_people=6, duration=10.0))[0]
    r = tr[tr["is_ego"] == 1][["x", "y"]].to_numpy()
    a, b = r[0], r[-1]
    u = (b - a) / np.linalg.norm(b - a)
    off_line = np.abs((r - a) @ np.array([-u[1], u[0]]))
    assert np.linalg.norm(b - a) > 3.0
    assert off_line.max() < 1e-6


def test_step_takes_robot_command():
    sim = CrowdSim(0, CrowdConfig(n_people=3))
    s0 = sim.reset()[0]
    state, info = sim.step((0.5, -0.5))
    assert state[0]["x"] == pytest.approx(s0["x"] + 0.1)
    assert state[0]["y"] == pytest.approx(s0["y"] - 0.1)
    assert set(state[0]) >= {"t", "agent_id", "kind", "is_ego", "x", "y", "vx", "vy", "radius", "regime"}
    assert isinstance(info["collided"], bool)


def test_observe_table(data):
    tracks = data[0]
    obs = observe(tracks, noise=0.1, seed=0)
    assert list(obs.columns) == OBSERVED
    assert len(obs) == len(people(tracks))
    assert obs[["x", "y", "vx", "vy", "robot_x", "robot_y"]].notna().all().all()
    true = people(tracks).sort_values(["episode", "t", "agent_id"])
    err = np.hypot(obs["x"].to_numpy() - true["x"].to_numpy(), obs["y"].to_numpy() - true["y"].to_numpy())
    assert 0.03 < np.median(err) < 0.2                    # noisy but filtered, not ground truth
    robot = tracks[tracks["is_ego"] == 1].set_index(["episode", "t"])
    joined = obs.join(robot[["x"]].rename(columns={"x": "rx"}), on=["episode", "t"])
    assert np.allclose(joined["robot_x"], joined["rx"])


def test_observe_is_forward_only(data):
    """Changing the future must not change the filtered past (no look-ahead leakage)."""
    tracks = data[0]
    tracks = tracks[tracks["episode"] == 0]
    cut = tracks[tracks["t"] <= 10.0]
    a = observe(tracks, seed=1)
    b = observe(cut, seed=1)
    a = a[a["t"] <= 10.0].reset_index(drop=True)
    # the first agent's noise is drawn first, so it is identical in both calls
    first = a["agent_id"].min()
    ok = a["agent_id"] == first
    assert np.allclose(a.loc[ok, ["x", "y", "vx", "vy"]], b.loc[b["agent_id"] == first, ["x", "y", "vx", "vy"]])
