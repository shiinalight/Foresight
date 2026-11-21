import numpy as np
import pandas as pd
import pytest

from foresight.core import features as F
from foresight.core.schema import validate
from foresight.domains.doorway import (DOOR_HALF, DoorwayConfig, DoorwaySim, episode_config, generate,
                                       summary, wall_obstacles)

SEEDS = [0, 1, 3]                      # 0 stays trickle, 1 and 3 switch to rush


@pytest.fixture(scope="module")
def data():
    return generate(SEEDS)


def test_dataset_passes_core_validation(data):
    tracks, obstacles, meta = data
    validate(tracks, meta)
    assert set(tracks.kind) == {"person", "robot"}
    assert (tracks.groupby("episode").is_ego.sum() == tracks.groupby("episode").t.nunique()).all()
    assert sorted(meta["episodes"]) == sorted(str(s) for s in SEEDS)
    assert all(m["seed"] == int(e) for e, m in meta["episodes"].items())       # episode id = seed
    assert "map" in meta and len(obstacles) == len(SEEDS) * len(wall_obstacles())


def test_same_seed_same_data_other_seed_differs():
    a = generate([5], DoorwayConfig(duration=20.0))[0]
    b = generate([5], DoorwayConfig(duration=20.0))[0]
    c = generate([7], DoorwayConfig(duration=20.0))[0]
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a.x.to_numpy()[:50], c.x.to_numpy()[:50])


def test_about_half_the_seeds_switch():
    sw = [episode_config(s).switch_t is not None for s in range(80)]
    assert sum(sw) == 40
    assert all(0.35 * 60 <= episode_config(s).switch_t <= 0.6 * 60 for s in range(1, 80, 2))


def test_switch_is_sharp_and_at_the_right_time(data):
    tracks, _, meta = data
    for ep, m in meta["episodes"].items():
        tr = tracks[(tracks.episode == int(ep)) & (tracks.kind == "person")]
        t_sw = m["regime_switch_t"]
        if t_sw is None:
            assert set(tr.regime) == {"trickle"} and tr.agent_id.nunique() == 8
            continue
        assert set(tr.loc[tr.t < t_sw - 1e-9, "regime"]) == {"trickle"}
        assert set(tr.loc[tr.t >= t_sw - 1e-9, "regime"]) == {"rush"}
        first_seen = tr.groupby("agent_id").t.min()
        crowd = first_seen[first_seen > 0]
        assert len(crowd) == 24                                     # the meeting crowd ...
        assert crowd.min() >= t_sw and crowd.max() <= t_sw + 4.0    # ... all arrives within 4 s


def test_nobody_inside_the_wall_or_through_it(data):
    tracks, _, _ = data
    W = np.array([[o["x"], o["y"], o["r"]] for o in wall_obstacles()])
    gap = (np.hypot(tracks.x.to_numpy()[:, None] - W[None, :, 0], tracks.y.to_numpy()[:, None] - W[None, :, 1])
           - W[None, :, 2] - tracks.radius.to_numpy()[:, None])
    assert gap.min() > -0.02
    tr = tracks.sort_values(["episode", "agent_id", "t"])
    g = tr.groupby(["episode", "agent_id"])
    x0, y0 = g.x.shift(1), g.y.shift(1)
    crossed = (np.sign(x0) != np.sign(tr.x)) & x0.notna()
    assert crossed.sum() > 20                                       # the door is used ...
    assert (((y0 + tr.y) / 2)[crossed].abs() < DOOR_HALF).all()     # ... and only the door


def test_speeds_within_bounds(data):
    tracks, _, _ = data
    sp = np.hypot(tracks.vx, tracks.vy)
    assert sp[tracks.kind == "person"].max() < 2.5 and sp[tracks.kind == "robot"].max() <= 1.0 + 1e-9


def test_rush_makes_a_queue(data):
    tracks, _, meta = data
    s = summary(tracks, meta).set_index("regime")
    assert s.loc["rush", "people_present"] > 2.5 * s.loc["trickle", "people_present"]
    assert s.loc["rush", "queue_at_door"] > 1.5 * s.loc["trickle", "queue_at_door"]
    assert s.loc["rush", "queueing_share"] > 0.1
    assert s.loc["rush", "cv_err_2s"] > 1.3 * s.loc["trickle", "cv_err_2s"]


def test_robot_passes_the_door(data):
    tracks, _, _ = data
    for ep, r in tracks[tracks.is_ego == 1].groupby("episode"):
        assert (np.diff(np.sign(r.x.to_numpy())) != 0).sum() >= 1, ep


def test_step_takes_robot_command():
    sim = DoorwaySim(2, episode_config(2))
    s0 = sim.reset()
    s1, info = sim.step((0.5, -0.25))
    assert s1[0]["x"] == pytest.approx(s0[0]["x"] + 0.1) and s1[0]["y"] == pytest.approx(s0[0]["y"] - 0.05)
    assert set(info) == {"collided"}


def test_core_builds_rows(data):
    tracks, obstacles, meta = data
    rows = F.build_rows(tracks, obstacles, meta, episodes=[1])
    assert len(rows) and F.X_of(rows).shape[1] == len(F.feature_names())
    assert rows.obst_dist.min() < 0.5                               # the wall is seen as an obstacle
