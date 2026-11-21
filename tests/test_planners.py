"""Sanity tests for the navigation baselines."""
import numpy as np
import pandas as pd

from foresight.core import planner as P


def people(*rows):
    return pd.DataFrame([dict(px=r[0], py=r[1], vx=r[2], vy=r[3]) for r in rows])


def test_ttc_head_on():
    # walker 5 m ahead closing at 2 m/s, combined radius 1 m: contact after (5 - 1) / 2 = 2 s
    assert abs(P.time_to_collision(np.array([5.0, 0]), np.array([-2.0, 0]), 1.0) - 2.0) < 1e-9
    assert P.time_to_collision(np.array([5.0, 0]), np.array([2.0, 0]), 1.0) == np.inf


def test_vo_empty_scene_goes_straight():
    v, _ = P.choose_vo(np.zeros(2), np.zeros(2), np.array([8.0, 0]), people(), max_speed=1.0)
    assert np.allclose(v, [1.0, 0], atol=1e-6)


def test_vo_head_on_walker_swerves():
    v, info = P.choose_vo(np.zeros(2), np.array([1.0, 0]), np.array([8.0, 0]), people((3.0, 0.0, -1.2, 0.0)),
                          max_speed=1.0)
    assert abs(v[1]) > 0.2 or np.linalg.norm(v) < 0.5        # sidesteps or slows, never ploughs on
    assert info["ttc"] > 1.0


def test_choose_respects_top_speed():
    cmd, _ = P.choose(np.zeros(2), np.array([8.0, 0]), people().assign(h=[], theta=[], lo_a=[], hi_a=[], lo_c=[],
                      hi_c=[], agent_id=[]), speeds=(0.0, 0.5, 1.0))
    assert np.linalg.norm(cmd) <= 1.0 + 1e-9


def test_ttc_overlapping():
    # inside the combined radius: separating -> no collision ahead; approaching -> collision now
    assert P.time_to_collision(np.array([0.5, 0]), np.array([1.0, 0]), 1.0) == np.inf
    assert P.time_to_collision(np.array([0.5, 0]), np.array([-1.0, 0]), 1.0) == 0.0
