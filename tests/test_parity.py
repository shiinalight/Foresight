"""Train/serve parity: a live query row (query_only=True on the history up to now) must have the
same features as the offline row for the same agent, moment and horizon."""
import numpy as np
import pandas as pd
import pytest

from foresight.core import features as F
from foresight.domains.crowd import CrowdConfig, simulate

FEATS = ["h_f"] + F.feature_names()[1:]


@pytest.fixture(scope="module")
def episode():
    tracks, obst, me = simulate(7, CrowdConfig(duration=12.0, n_people=8))
    meta = {"dt": 0.2, "bounds": [-10, -10, 10, 10], "episodes": {"0": me}}
    return tracks, obst, meta


@pytest.mark.parametrize("noise", [0.0, 0.1])
def test_query_matches_offline(episode, noise):
    tracks, obst, meta = episode
    off = F.build_rows(tracks, obst, meta, horizons=(1.0, 2.0), noise=noise, seed=3)
    for t in sorted(off.t.unique())[2:6]:
        live = F.build_rows(tracks[tracks.t <= t + 1e-9], obst, meta, horizons=(1.0, 2.0), noise=noise, seed=3,
                            query_only=True)
        a = off[np.isclose(off.t, t)].set_index(["agent_id", "h"])[FEATS]
        b = live.set_index(["agent_id", "h"])[FEATS]
        common = a.index.intersection(b.index)
        assert len(common) > 0
        np.testing.assert_allclose(a.loc[common].to_numpy(), b.loc[common].to_numpy(), atol=1e-9)


def test_observed_labels_differ_from_truth_only_by_noise(episode):
    tracks, obst, meta = episode
    tru = F.build_rows(tracks, obst, meta, horizons=(1.0,), noise=0.1, seed=3)
    obs = F.build_rows(tracks, obst, meta, horizons=(1.0,), noise=0.1, seed=3, label_from="observed")
    d = np.hypot(tru.true_x - obs.true_x, tru.true_y - obs.true_y)
    assert 0 < d.mean() < 0.3          # filtered observation error, not a different target


def test_parity_with_agent_entering_mid_episode(episode):
    """Agents that appear later (doorway, street, real data) must not shift anyone's noise."""
    tracks, obst, meta = episode
    tr = tracks[~((tracks.agent_id == 3) & (tracks.t < 4.0))]       # agent 3 enters at t = 4 s
    off = F.build_rows(tr, obst, meta, horizons=(1.0,), noise=0.1, seed=3)
    for t in [2.0, 3.0, 5.0, 6.0]:
        t_row = off.t.unique()[np.argmin(np.abs(off.t.unique() - t))]
        live = F.build_rows(tr[tr.t <= t_row + 1e-9], obst, meta, horizons=(1.0,), noise=0.1, seed=3, query_only=True)
        a = off[np.isclose(off.t, t_row)].set_index(["agent_id", "h"])[FEATS]
        b = live.set_index(["agent_id", "h"])[FEATS]
        common = a.index.intersection(b.index)
        assert len(common) > 0
        np.testing.assert_allclose(a.loc[common].to_numpy(), b.loc[common].to_numpy(), atol=1e-9)
