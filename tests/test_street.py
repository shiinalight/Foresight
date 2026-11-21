import numpy as np
import pandas as pd
import pytest

from foresight.core import features as F
from foresight.core.schema import validate
from foresight.domains.street import CROSS_X, StreetConfig, StreetSim, generate

SHORT = StreetConfig(duration=45.0)


@pytest.fixture(scope="module")
def data():
    return generate(2, seed=3, config=SHORT)


def test_dataset_passes_core_validation_and_has_all_types(data):
    tracks, obstacles, meta = data
    validate(tracks, meta)
    assert set(tracks.agent_type) == {"car", "pedestrian", "cyclist", "robot"}
    assert (tracks.groupby("episode").is_ego.sum() == tracks.groupby("episode").t.nunique()).all()
    one_hot = tracks[["ext_type_pedestrian", "ext_type_car", "ext_type_cyclist"]].sum(axis=1)
    assert (one_hot == (tracks.agent_type != "robot")).all()
    assert "map" in meta and len(obstacles)


def test_same_seed_same_data():
    a = generate(1, seed=5, config=StreetConfig(duration=10.0))[0]
    b = generate(1, seed=5, config=StreetConfig(duration=10.0))[0]
    pd.testing.assert_frame_equal(a, b)


def test_scripted_events_logged_at_their_times(data):
    tracks, _, meta = data
    for ep, m in meta["episodes"].items():
        names = {e["name"]: e["t"] for e in m["events"]}
        assert names.keys() >= {"sudden_stop", "car_brake_crosswalk", "jaywalk", "rush_hour"}
        assert names["rush_hour"] == pytest.approx(40.0)
        logged = tracks[(tracks.episode == int(ep)) & tracks.events.fillna("").str.contains("sudden_stop")]
        assert len(logged) == 1


def test_rush_hour_switches_pedestrians(data):
    tracks, _, _ = data
    ped = tracks[tracks.agent_type == "pedestrian"]
    assert set(ped.loc[ped.t < 40, "regime"]) == {"normal"}
    assert set(ped.loc[ped.t > 40.5, "regime"]) == {"rush_hour"}
    assert ped.loc[ped.t > 40.5].agent_id.nunique() > ped.loc[ped.t < 40].agent_id.nunique()   # commuters


def test_vehicles_never_hit_anyone_and_never_stop_on_the_zebra(data):
    tracks, _, _ = data
    veh = tracks[tracks.agent_type.isin(["car", "cyclist"])]
    assert veh.collided.sum() == 0
    cars = tracks[tracks.agent_type == "car"]
    on_zebra = cars[np.abs(cars.x) < CROSS_X + cars.length / 2 - 0.3]   # standing on it, not grazing its edge
    assert (np.hypot(on_zebra.vx, on_zebra.vy) > 0.5).mean() > 0.95


def test_cars_brake_for_the_jaywalker_scenario(data):
    tracks, _, meta = data
    cars = tracks[tracks.agent_type == "car"].sort_values(["episode", "agent_id", "t"])
    dv = cars.groupby(["episode", "agent_id"]).vx.diff().abs() / 0.2
    assert dv.max() > 3.0                    # someone brakes hard somewhere


def test_robot_reaches_the_station():
    sim = StreetSim(1, StreetConfig())
    sim.reset()
    ys = [sim.step()[0] for _ in range(450)]     # 90 s: the Café door, then the cautious zebra wait at rush hour
    robot_y = [r["y"] for rows in ys for r in rows if r["agent_id"] == 0]
    assert min(robot_y) < -7.0               # it crossed and got to the Station door


def test_core_turns_agent_type_into_features(data):
    tracks, obstacles, meta = data
    rows = F.build_rows(tracks, obstacles, meta, episodes=[0])
    X = F.X_of(rows)
    assert {"ext_type_car", "ext_type_cyclist", "ext_type_pedestrian"} <= set(rows.columns)
    assert X.shape[1] == len(F.feature_names()) + 3


def test_robot_waits_at_the_cafe_door_then_goes():
    for seed in range(5):
        sim = StreetSim(seed, StreetConfig())
        sim.reset()
        assert sim.agents[0].p[1] > 8.0                      # starts inside the Café
        for _ in range(75):                                  # 15 s
            sim.step()
        ev = {e["name"]: e["t"] for e in sim.event_log}
        assert ev["robot_waits_door"] < ev["robot_door_clear"] < 10.0
        assert sim.agents[0].p[1] < 7.5                      # it got out
        cust = [a for a in sim.agents.values() if a.indoor]
        assert cust and cust[0].mode == "order"              # and the customer got in
