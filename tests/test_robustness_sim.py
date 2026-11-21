"""Robustness of the crowd simulator's difficulty knobs (T2 in docs/DECISIONS.md).

The knobs must not change anything at their defaults: every earlier result (bank, pilot, overnight
rows) was produced before they existed. The golden checksums below were recorded from the code
before the knobs were added (commit 0860038) and must keep matching.
"""
import hashlib
import json

import numpy as np
import pytest

from foresight.domains import crowd as Cr
from foresight.domains.crowd import CrowdConfig, CrowdSim, generate, simulate


def _digest(tracks):
    a = tracks[["x", "y", "vx", "vy", "goal_x", "goal_y", "collided"]].to_numpy(float)
    return hashlib.sha256(np.round(a, 10).tobytes()).hexdigest()[:16]


GOLDEN = [
    (3, CrowdConfig(duration=10.0), "b1df396af40b802d"),
    (11, CrowdConfig(duration=10.0, regime="erratic", switch_t=4.0, switch_to="hurried", n_people=24),
     "728e7804eff186a1"),
]


@pytest.mark.parametrize("seed,cfg,digest", GOLDEN)
def test_defaults_reproduce_pre_knob_episodes(seed, cfg, digest):
    tracks, _, _ = simulate(seed, cfg)
    assert _digest(tracks) == digest


def test_defaults_reproduce_closed_loop_steps():
    sim = CrowdSim(5, CrowdConfig(duration=6.0))
    sim.reset()
    xs = []
    for _ in range(30):
        state, _ = sim.step((0.5, 0.1))
        xs.append([r["x"] for r in state])
    assert hashlib.sha256(np.round(np.array(xs), 10).tobytes()).hexdigest()[:16] == "ac5ce8e03bcc74e3"


def test_explicit_defaults_equal_implicit():
    a, _, _ = simulate(4, CrowdConfig(duration=6.0))
    b, _, _ = simulate(4, CrowdConfig(duration=6.0, robot_repulsion=Cr.ROBOT_REPULSION, ignore_robot=0.0))
    assert _digest(a) == _digest(b)


@pytest.mark.parametrize("kw", [dict(ignore_robot=-0.1), dict(ignore_robot=1.5), dict(robot_repulsion=-1.0)])
def test_invalid_knobs_rejected(kw):
    with pytest.raises(ValueError):
        CrowdSim(0, CrowdConfig(**kw))


def test_ignore_share_uses_its_own_random_stream():
    # same seed: the crowd's start (positions, goals, jitter) must be identical whatever the share,
    # so difficulty levels are paired comparisons of the same crowd
    base, ign = CrowdSim(9, CrowdConfig()), CrowdSim(9, CrowdConfig(ignore_robot=0.5))
    base.reset(); ign.reset()
    assert np.array_equal(base.P, ign.P) and np.array_equal(base.G, ign.G) and np.array_equal(base.jit, ign.jit)
    assert not base.ignores_robot.any()
    assert not ign.ignores_robot[0]                                   # the robot is never a "person"
    assert 0 < ign.ignores_robot[1:].sum() < len(ign.ignores_robot) - 1


def test_ignore_share_is_nested_across_levels():
    # a person who ignores the robot at share 0.3 also ignores it at 0.6 (one uniform draw per person)
    lo, hi = CrowdSim(21, CrowdConfig(n_people=40, ignore_robot=0.3)), CrowdSim(21, CrowdConfig(n_people=40, ignore_robot=0.6))
    lo.reset(); hi.reset()
    assert np.all(hi.ignores_robot[lo.ignores_robot])
    assert hi.ignores_robot.sum() >= lo.ignores_robot.sum()


def test_share_one_and_zero_extremes():
    s = CrowdSim(2, CrowdConfig(ignore_robot=1.0)); s.reset()
    assert s.ignores_robot[1:].all()


def test_knobs_recorded_in_dataset_meta():
    _, _, meta = generate(2, seed=0, config=CrowdConfig(duration=4.0, robot_repulsion=1.0, ignore_robot=0.2))
    assert meta["config"]["robot_repulsion"] == 1.0 and meta["config"]["ignore_robot"] == 0.2
    json.dumps(meta)                                                  # still serialisable for meta.json


# ---------------------------------------------------------------- goal_mode "cross" (realism, docs/DATA.md)
def test_cross_mode_rejects_unknown_mode():
    with pytest.raises(ValueError):
        CrowdSim(0, CrowdConfig(goal_mode="wander"))


def test_cross_mode_people_leave_and_new_ids_enter():
    tracks, _, _ = simulate(7, CrowdConfig(goal_mode="cross", duration=40.0, regime="hurried"))
    ppl = tracks[tracks["kind"] == "person"]
    assert ppl["agent_id"].nunique() > 16                    # people walked out, new people came in
    assert tracks.groupby("t")["agent_id"].apply(lambda s: s.is_unique).all()
    assert (tracks.groupby("t").size() == 17).all()          # constant head count: 16 people + robot
    L = CrowdConfig().half_size
    assert (ppl["x"].abs() < L).all() and (ppl["y"].abs() < L).all()


def test_cross_mode_tracks_have_no_teleports():
    tracks, _, _ = simulate(8, CrowdConfig(goal_mode="cross", duration=40.0, regime="hurried"))
    ppl = tracks[tracks["kind"] == "person"].sort_values(["agent_id", "t"])
    g = ppl.groupby("agent_id")
    step = np.hypot(g["x"].diff(), g["y"].diff()).dropna()
    assert step.max() < 0.6                                  # 0.2 s steps: < 3 m/s, never a jump
    first = g.head(1)                                        # newcomers enter near an edge
    new = first[first["agent_id"] > 16]
    near_edge = np.maximum(new["x"].abs(), new["y"].abs()) > CrowdConfig().half_size - 1.5
    assert near_edge.all()


def test_cross_mode_turns_less_than_anywhere():
    def turn_share(mode):
        tr, _, _ = simulate(9, CrowdConfig(goal_mode=mode, regime="calm", duration=30.0))
        p = tr[tr["kind"] == "person"].sort_values(["agent_id", "t"])
        g = p.groupby("agent_id")
        fvx, fvy = g["vx"].shift(-10), g["vy"].shift(-10)
        ok = (np.hypot(p["vx"], p["vy"]) > 0.5) & (np.hypot(fvx, fvy) > 0.5)
        d = np.abs((np.arctan2(fvy, fvx) - np.arctan2(p["vy"], p["vx"]) + np.pi) % (2 * np.pi) - np.pi)
        return float((d[ok] > np.radians(30)).mean())
    assert turn_share("cross") < turn_share("anywhere")
