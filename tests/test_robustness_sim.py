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
