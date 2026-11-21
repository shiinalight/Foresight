"""Robustness properties of the core: features, models, context selection, schema.

Each test states a property the rest of the pipeline relies on and checks it against
independent ground truth (raw tracks, hand-built scenes), not against a copy of the code.
No TabPFN calls; LightGBM runs with 30 trees on one thread.
"""
import numpy as np
import pandas as pd
import pytest

from foresight.core import context as C
from foresight.core import features as F
from foresight.core import models as M
from foresight.core.schema import validate
from foresight.domains.crowd import CrowdConfig, simulate

FEATS = ["h_f"] + F.feature_names()[1:]
DT = 0.2
KEY = ["agent_id", "t", "h"]


# ---------------------------------------------------------------- helpers
def _sim(seed, episode=0, n_obstacles=2, duration=10.0, n_people=7, regime="calm"):
    cfg = CrowdConfig(duration=duration, n_people=n_people, n_obstacles=n_obstacles, regime=regime)
    return simulate(seed, cfg, episode=episode)


def _meta(eps_meta, bounds=(-10, -10, 10, 10)):
    return {"dt": DT, "bounds": None if bounds is None else list(bounds),
            "episodes": {str(k): v for k, v in eps_meta.items()}}


def _transform(tracks, obst, phi, shift):
    """Rotate the whole scene by phi about the origin, then translate by shift."""
    c, s = np.cos(phi), np.sin(phi)
    tr = tracks.copy()
    x, y, vx, vy = (tracks[k].to_numpy() for k in ("x", "y", "vx", "vy"))
    tr["x"], tr["y"] = c * x - s * y + shift[0], s * x + c * y + shift[1]
    tr["vx"], tr["vy"] = c * vx - s * vy, s * vx + c * vy
    ob = obst.copy() if obst is not None else None
    if ob is not None and len(ob):
        ox, oy = obst["x"].to_numpy(), obst["y"].to_numpy()
        ob["x"], ob["y"] = c * ox - s * oy + shift[0], s * ox + c * oy + shift[1]
    return tr, ob


def _hand_tracks(agents, episode=0, ego=None):
    """agents: {agent_id: (first_step, array [n, 4] of x, y, vx, vy)} -> tracks DataFrame."""
    rows = []
    for aid, (s0, arr) in agents.items():
        for j, (x, y, vx, vy) in enumerate(np.asarray(arr, float)):
            rows.append(dict(episode=episode, t=round((s0 + j) * DT, 6), agent_id=aid,
                             kind="robot" if aid == ego else "person", is_ego=int(aid == ego),
                             x=x, y=y, vx=vx, vy=vy, radius=0.4 if aid == ego else 0.3,
                             regime="robot" if aid == ego else "calm"))
    return pd.DataFrame(rows).sort_values(["episode", "t", "agent_id"]).reset_index(drop=True)


def _walker(p0, v, n):
    """Constant-velocity walker: n steps from p0 with velocity v."""
    p0, v = np.asarray(p0, float), np.asarray(v, float)
    pos = p0 + np.arange(n)[:, None] * DT * v
    return np.hstack([pos, np.tile(v, (n, 1))])


def _stander(p, n):
    return np.hstack([np.tile(np.asarray(p, float), (n, 1)), np.zeros((n, 2))])


def _feat_nan_free(rows):
    return not np.isnan(F.X_of(rows)).any()


@pytest.fixture(scope="module")
def ep0():
    tracks, obst, me = _sim(11)
    return tracks, obst, _meta({0: me})


@pytest.fixture(scope="module")
def multi():
    """Four short episodes: 0 is the query episode, 1..3 the bank."""
    trs, obs, mes = [], [], {}
    for e in range(4):
        tr, ob, me = _sim(100 + e, episode=e)
        trs.append(tr); obs.append(ob); mes[e] = me
    tracks, obst, meta = pd.concat(trs, ignore_index=True), pd.concat(obs, ignore_index=True), _meta(mes)
    rows = F.build_rows(tracks, obst, meta, horizons=(1.0, 2.0), noise=0.1, seed=0)
    return tracks, obst, meta, rows


# ---------------------------------------------------------------- 1. agent-frame invariance
TRANSFORMS = [(0.7, (3.0, -5.0)), (np.pi * 0.93, (-40.0, 12.5))]


@pytest.mark.parametrize("phi,shift", TRANSFORMS)
@pytest.mark.parametrize("obstacles", [
    "rotated_pillars",
    pytest.param("none", marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
        "BUG: with no wall bounds and no obstacles the padding vector [FAR, 0] is rotated by "
        "R(theta) (features.py _obstacle_vec / base), so obst_x, obst_y encode the world heading"))),
])
def test_features_and_targets_invariant_to_rigid_motion(phi, shift, obstacles):
    n_obst = 2 if obstacles == "rotated_pillars" else 0
    tracks, obst, me = _sim(5, n_obstacles=n_obst)
    meta = _meta({0: me}, bounds=None)               # axis-aligned walls are not rotation invariant
    a = F.build_rows(tracks, obst, meta, noise=0.0, seed=1)
    tr2, ob2 = _transform(tracks, obst, phi, shift)
    b = F.build_rows(tr2, ob2, meta, noise=0.0, seed=1)
    assert len(a) > 50
    a, b = a.set_index(KEY).sort_index(), b.set_index(KEY).sort_index()
    assert a.index.equals(b.index)                     # same rows exist
    # the agent frame is defined when the agent has a heading: it moves (speed > 0.2 m/s) or has
    # moved over the last 1 s. Standing agents have no heading; see the standing-agent test below.
    moved = (a.speed > 0.2) | (np.hypot(a.hist_x5, a.hist_y5) > 0.05)
    assert moved.mean() > 0.9
    a, b = a[moved], b[moved]
    cols = FEATS + ["ddx", "ddy"]
    bad = ~np.isclose(a[cols].to_numpy(), b[cols].to_numpy(), atol=1e-7, rtol=0)
    assert not bad.any(), f"columns that change: {sorted({cols[j] for j in np.flatnonzero(bad.any(0))})}"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "BUG: a standing agent (speed <= 0.2, displacement <= 0.05 over 1 s) gets theta = 0.0, the world "
    "x axis (features.py theta fallback), so its neighbour / robot features rotate with the world"))
def test_standing_agent_features_invariant_to_rotation():
    n = 15
    tracks = _hand_tracks({1: (0, _stander((0.0, 0.0), n)),           # stands still the whole time
                           2: (0, _walker((-3.0, 1.5), (1.0, 0.0), n))})  # walks past it
    meta = _meta({0: {}}, bounds=None)
    a = F.build_rows(tracks, None, meta, horizons=(1.0,), noise=0.0)
    b = F.build_rows(_transform(tracks, None, 1.1, (2.0, -1.0))[0], None, meta, horizons=(1.0,), noise=0.0)
    a = a[a.agent_id == 1].set_index(KEY).sort_index(); b = b[b.agent_id == 1].set_index(KEY).sort_index()
    assert len(a) > 0 and a.index.equals(b.index)
    np.testing.assert_allclose(a[FEATS].to_numpy(), b[FEATS].to_numpy(), atol=1e-7)


# ---------------------------------------------------------------- 2. to_world inverts the target
@pytest.mark.parametrize("noise", [0.0, 0.1])
def test_to_world_recovers_true_future_position(ep0, noise):
    tracks, obst, meta = ep0
    rows = F.build_rows(tracks, obst, meta, horizons=(1.0, 2.0), noise=noise, seed=2)
    wx, wy = F.to_world(rows, rows["ddx"].to_numpy(), rows["ddy"].to_numpy())
    # ground truth straight from the tracks file, not from the row's own columns
    truth = tracks.assign(step=np.round(tracks.t / DT).astype(int)).set_index(["agent_id", "step"])
    idx = list(zip(rows.agent_id, np.round(rows.t_known / DT).astype(int)))
    np.testing.assert_allclose(wx, truth.loc[idx, "x"].to_numpy(), atol=1e-9)
    np.testing.assert_allclose(wy, truth.loc[idx, "y"].to_numpy(), atol=1e-9)
    # zero residual is exactly the constant-velocity extrapolation of the observed state
    cx, cy = F.to_world(rows, np.zeros(len(rows)), np.zeros(len(rows)))
    np.testing.assert_allclose(cx, rows.px + rows.vx * rows.h, atol=1e-12)
    np.testing.assert_allclose(cy, rows.py + rows.vy * rows.h, atol=1e-12)


def test_to_world_inverts_observed_labels(ep0):
    tracks, obst, meta = ep0
    rows = F.build_rows(tracks, obst, meta, horizons=(1.0,), noise=0.1, seed=2, label_from="observed")
    wx, wy = F.to_world(rows, rows["ddx"].to_numpy(), rows["ddy"].to_numpy())
    np.testing.assert_allclose(wx, rows.true_x, atol=1e-9)
    np.testing.assert_allclose(wy, rows.true_y, atol=1e-9)


# ---------------------------------------------------------------- 3. causality of offline rows
@pytest.mark.parametrize("noise", [0.0, 0.1])
def test_rows_unchanged_when_episode_is_truncated(ep0, noise):
    tracks, obst, meta = ep0
    full = F.build_rows(tracks, obst, meta, horizons=(1.0, 2.0), noise=noise, seed=4).set_index(KEY)
    T = 7.0
    cut = F.build_rows(tracks[tracks.t <= T + 1e-9], obst, meta, horizons=(1.0, 2.0), noise=noise,
                       seed=4).set_index(KEY)
    assert len(cut) > 0
    assert cut.index.isin(full.index).all()
    # every full row whose target is known by T survives the cut
    expect = full.index[full.t_known <= T + 1e-9]
    assert set(expect) == set(cut.index)
    cols = FEATS + ["ddx", "ddy", "px", "py", "vx", "vy", "theta"]
    np.testing.assert_allclose(cut[cols].to_numpy(), full.loc[cut.index, cols].to_numpy(), atol=1e-12)


@pytest.mark.parametrize("noise", [0.0, 0.1])
@pytest.mark.parametrize("t_cut", [2.0, 5.0])
def test_features_blind_to_scrambled_future(ep0, noise, t_cut):
    tracks, obst, meta = ep0
    rng = np.random.default_rng(0)
    fut = tracks.t > t_cut + 1e-9
    tr2 = tracks.copy()
    for c in ("x", "y", "vx", "vy"):
        tr2.loc[fut, c] = tr2.loc[fut, c] + rng.normal(0, 2.0, fut.sum())
    a = F.build_rows(tracks, obst, meta, horizons=(1.0, 2.0), noise=noise, seed=4).set_index(KEY)
    b = F.build_rows(tr2, obst, meta, horizons=(1.0, 2.0), noise=noise, seed=4).set_index(KEY)
    now = a.index[a.index.get_level_values("t") <= t_cut + 1e-9]
    assert len(now) > 0 and now.isin(b.index).all()
    np.testing.assert_allclose(a.loc[now, FEATS].to_numpy(), b.loc[now, FEATS].to_numpy(), atol=1e-12)
    # sanity: the scramble did reach the targets of those rows that look past t_cut
    later = now[a.loc[now, "t_known"] > t_cut + 1e-9]
    assert len(later) > 0
    assert not np.allclose(a.loc[later, "ddx"], b.loc[later, "ddx"])


# ---------------------------------------------------------------- 4. build_rows edge cases
def test_lone_agent_gets_padded_neighbours():
    tracks = _hand_tracks({1: (0, _walker((0, 0), (1.0, 0.5), 20))})
    rows = F.build_rows(tracks, None, _meta({0: {}}), noise=0.0)
    assert len(rows) > 0 and _feat_nan_free(rows)
    for n in range(F.N_NEIGH):
        assert (rows[f"n{n}_dist"] == F.FAR).all() and (rows[f"n{n}_x"] == F.FAR).all()
        assert (rows[[f"n{n}_y", f"n{n}_vx", f"n{n}_vy"]] == 0).all().all()
    assert (rows.density_3m == 0).all()
    # no robot in this episode: robot features padded too
    assert (rows.robot_dist == F.FAR).all() and (rows.robot_x == F.FAR).all()
    assert (rows[["robot_y", "robot_closing"]] == 0).all().all()


def test_no_robot_episode_pads_robot_features(ep0):
    tracks, obst, meta = ep0
    rows = F.build_rows(tracks[tracks.is_ego == 0], obst, meta, noise=0.1, seed=0)
    assert len(rows) > 0 and _feat_nan_free(rows)
    assert (rows.robot_dist == F.FAR).all() and (rows.robot_x == F.FAR).all()
    assert (rows[["robot_y", "robot_closing"]] == 0).all().all()
    with_robot = F.build_rows(tracks, obst, meta, noise=0.1, seed=0)
    assert (with_robot.robot_dist < F.FAR).all()              # robot present every step there


def test_late_and_short_lived_agents():
    n = 30
    tracks = _hand_tracks({1: (0, _walker((0, 0), (1.0, 0.0), n)),
                           2: (12, _walker((0, 3), (0.0, -0.8), n - 12)),    # enters at step 12
                           3: (4, _walker((5, 5), (0.5, 0.5), 3)),           # 3 steps only
                           4: (0, _walker((-5, -5), (0.3, 0.9), 1))})        # a single sample
    rows = F.build_rows(tracks, None, _meta({0: {}}), horizons=(1.0,), noise=0.1, seed=0)
    assert _feat_nan_free(rows)
    assert set(rows.agent_id) == {1, 2}
    # rows for agent 2 only once it has N_HIST observation steps of its own history
    first = rows[rows.agent_id == 2].t.min()
    assert first >= (12 + F.N_HIST) * DT - 1e-9
    # agent 1 sees a neighbour only while agent 3 (steps 4..6) or agent 2 (steps >= 12) is in the scene
    a1 = rows[rows.agent_id == 1].assign(step=lambda d: np.round(d.t / DT).astype(int))
    alone = a1.step.between(7, 11)
    assert alone.any() and (a1[alone].n0_dist == F.FAR).all() and (a1[alone].density_3m == 0).all()
    assert (a1[~alone].n0_dist < F.FAR).all() and (a1.n1_dist == F.FAR).all()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "BUG: build_rows only checks that the agent is seen at s and s - N_HIST*k (features.py history "
    "check), so an agent occluded in between yields NaN history/acceleration features and a "
    "zero-velocity Kalman restart"))
def test_track_gap_gives_no_nan_features():
    n = 30
    arr = _walker((0, 0), (1.0, 0.2), n)
    keep = np.r_[0:10, 13:n]                                      # occluded for steps 10..12
    tracks = _hand_tracks({1: (0, arr), 2: (0, _walker((2, 2), (0.0, 1.0), n))})
    tracks = tracks[~((tracks.agent_id == 1) & ~np.isin(np.round(tracks.t / DT).astype(int), keep))]
    rows = F.build_rows(tracks, None, _meta({0: {}}), horizons=(1.0,), noise=0.0)
    assert len(rows) > 0
    bad = rows[np.isnan(F.X_of(rows)).any(axis=1)]
    assert bad.empty, f"NaN features for {bad[['agent_id', 't']].values.tolist()}"


def test_horizon_past_episode_end_drops_rows(ep0):
    tracks, obst, meta = ep0
    T = tracks.t.max()
    rows = F.build_rows(tracks, obst, meta, horizons=(1.0, 2.0, T + 5.0), noise=0.1, seed=0)
    assert len(rows) > 0
    assert set(np.round(rows.h, 6)) == {1.0, 2.0}
    assert rows[["ddx", "ddy", "true_x", "true_y"]].notna().all().all()
    assert (rows.t_known <= T + 1e-9).all()
    none = F.build_rows(tracks, obst, meta, horizons=(T + 5.0,), noise=0.1, seed=0)
    assert len(none) == 0 and list(none.columns) == list(rows.columns)


# ---------------------------------------------------------------- 5. models
@pytest.fixture(scope="module")
def train_test(multi):
    *_, rows = multi
    tr, te = rows[rows.episode != 0], rows[rows.episode == 0]
    return F.X_of(tr), tr[["ddx", "ddy"]].to_numpy(), F.X_of(te)


def _model(name):
    return M.make(name, n_jobs=1) if name.startswith("lgbm") else M.make(name)


@pytest.mark.parametrize("name", ["cv", "cv_band", "lgbm:n30"])
@pytest.mark.parametrize("qs", [(0.1, 0.5, 0.9), (0.05, 0.1, 0.5, 0.9, 0.95)])
def test_quantiles_shape_and_order(train_test, name, qs):
    Xtr, Ytr, Xte = train_test
    Q = _model(name).fit(Xtr, Ytr).predict_quantiles(Xte, qs)
    assert Q.shape == (len(Xte), 2, len(qs))
    assert np.isfinite(Q).all()
    assert (np.diff(Q, axis=2) >= -1e-12).all()


def test_cv_is_zero_residual(train_test):
    Xtr, Ytr, Xte = train_test
    assert not _model("cv").fit(Xtr, Ytr).predict_quantiles(Xte).any()


def test_cv_band_is_constant_per_horizon_with_zero_median(train_test):
    Xtr, Ytr, Xte = train_test
    Q = _model("cv_band").fit(Xtr, Ytr).predict_quantiles(Xte, (0.1, 0.5, 0.9))
    assert not Q[:, :, 1].any()
    for h in np.unique(Xte[:, 0]):
        band = Q[np.isclose(Xte[:, 0], h)]
        assert (band == band[0]).all()
        # band equals the training residual quantiles of that horizon
        ref = np.quantile(Ytr[np.isclose(Xtr[:, 0], h)], [0.1, 0.9], axis=0).T
        np.testing.assert_allclose(band[0][:, [0, 2]], ref)
    assert not np.allclose(Q[np.isclose(Xte[:, 0], 1.0)][0], Q[np.isclose(Xte[:, 0], 2.0)][0])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "BUG: ConstantBand forces the 0.5 quantile to 0 without clamping (models.py ConstantBand."
    "predict_quantiles), so a training residual distribution that does not straddle 0 gives "
    "q10 > q50 (crossing quantiles)"))
def test_cv_band_monotone_when_residuals_are_biased():
    rng = np.random.default_rng(0)
    X = np.c_[np.full(200, 1.0), rng.normal(size=(200, 3))]
    Y = rng.normal(0.8, 0.2, (200, 2))                  # e.g. everyone systematically lagging CV
    Q = M.make("cv_band").fit(X, Y).predict_quantiles(X[:5], (0.1, 0.5, 0.9))
    assert (np.diff(Q, axis=2) >= 0).all()


@pytest.mark.xfail(strict=True, raises=ValueError, reason=(
    "BUG (latent, every caller passes 0.5): ConstantBand evaluates list(qs).index(0.5) in the "
    "assignment target even when 0.5 is not in qs, so qs=(0.1, 0.9) raises ValueError"))
def test_cv_band_accepts_quantiles_without_median(train_test):
    Xtr, Ytr, Xte = train_test
    Q = M.make("cv_band").fit(Xtr, Ytr).predict_quantiles(Xte, (0.1, 0.9))
    assert Q.shape == (len(Xte), 2, 2) and (Q[:, :, 0] <= Q[:, :, 1]).all()


# ---------------------------------------------------------------- 6. context.select
@pytest.fixture(scope="module")
def ctx_setup(multi):
    *_, rows = multi
    bank = C.Bank(rows[rows.episode != 0])
    live = rows[rows.episode == 0]
    t_now = sorted(live.t.unique())[len(live.t.unique()) // 2]
    queries = live[np.isclose(live.t, t_now)]
    return bank, live, queries, t_now


def _key(df):
    return df[["episode", "t", "agent_id", "h"]].round(6)


@pytest.mark.parametrize("budget", [40, 150, 600])
def test_select_is_causal_bounded_and_unique(ctx_setup, budget):
    bank, live, queries, t_now = ctx_setup
    ctx = C.select(queries, bank, live=live, budget=budget)
    assert 0 < len(ctx) <= budget
    assert not _key(ctx).duplicated().any()
    assert "source" in ctx.columns
    assert set(ctx.source) <= {"recent", "knn", "fill-rare", "fill-random"}
    same_ep = ctx[ctx.episode == queries.episode.iloc[0]]
    assert (same_ep.t_known <= t_now + 1e-9).all()               # no label from the future
    assert (same_ep.source == "recent").all()
    assert (ctx[ctx.source != "recent"].episode != queries.episode.iloc[0]).all()
    assert (ctx.source == "recent").any()


@pytest.mark.parametrize("live_kind", ["none", "empty"])
def test_select_cold_start_uses_bank(ctx_setup, live_kind):
    bank, live, queries, _ = ctx_setup
    lv = None if live_kind == "none" else live.iloc[:0]
    ctx = C.select(queries, bank, live=lv, budget=200)
    assert 150 <= len(ctx) <= 200
    assert (ctx.source != "recent").all()
    assert not _key(ctx).duplicated().any()


@pytest.mark.parametrize("budget", [
    40,
    pytest.param(41, marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
        "BUG: Bank.knn returns np.unique(idx) (sorted by bank row index, distance order lost) and "
        "select truncates it with [:n_knn] (context.py), so when k*len(queries) > n_knn the rows "
        "dropped are the highest bank indices, not the farthest neighbours: a query's own 1-NN can go"))),
])
def test_select_keeps_each_querys_nearest_bank_row(budget):
    """kNN share is meant to give each query its most similar situations: when the kNN budget
    covers at least one row per query, each query's single nearest bank row must be included."""
    rng = np.random.default_rng(0)
    n_bank, n_q = 400, 4
    bank_df = pd.DataFrame(rng.normal(size=(n_bank, len(C.KNN_FEATS))), columns=C.KNN_FEATS)
    bank_df = bank_df.assign(episode=1 + np.arange(n_bank) // 100, t=np.arange(n_bank) * 0.2,
                             t_known=np.arange(n_bank) * 0.2 + 1.0, agent_id=1, h=1.0, mode="straight")
    bank = C.Bank(bank_df)
    queries = bank_df.iloc[[350, 360, 370, 399]][C.KNN_FEATS].assign(
        episode=0, t=5.0, t_known=6.0, agent_id=np.arange(n_q), h=1.0, mode="straight")
    # all of the budget goes to kNN; 40 = 10 per query exactly, 41 -> k = 11 per query, 44 candidates
    ctx = C.select(queries, bank, live=None, budget=budget, mix=(0.0, 1.0, 0.0))
    assert len(ctx) == budget
    picked = set(map(tuple, ctx[["episode", "t"]].round(6).to_numpy()))
    for i in (350, 360, 370, 399):                                     # the query itself is its 1-NN
        assert (bank_df.episode[i], round(bank_df.t[i], 6)) in picked


# ---------------------------------------------------------------- 7. schema.validate
@pytest.fixture(scope="module")
def clean(ep0):
    tracks, _, meta = ep0
    validate(tracks, meta)                                            # the clean data passes
    return tracks, meta


def test_validate_rejects_duplicate_rows(clean):
    tracks, meta = clean
    with pytest.raises(ValueError, match=r"1 duplicate \(episode, t, agent_id\)"):
        validate(pd.concat([tracks, tracks.iloc[[10]]], ignore_index=True), meta)


def test_validate_rejects_off_grid_time(clean):
    tracks, meta = clean
    bad = tracks.copy(); bad.loc[5, "t"] += 0.07
    with pytest.raises(ValueError, match=r"1 rows have t off the dt=0.2 grid"):
        validate(bad, meta)


def test_validate_rejects_two_egos(clean):
    tracks, meta = clean
    bad = tracks.copy(); bad.loc[bad.agent_id == 3, "is_ego"] = 1
    with pytest.raises(ValueError, match=r"more than one ego: \[0\]"):
        validate(bad, meta)


def test_validate_rejects_missing_column_and_wrong_types(clean):
    tracks, meta = clean
    with pytest.raises(ValueError, match="missing columns: \\['radius'\\]"):
        validate(tracks.drop(columns="radius"), meta)
    with pytest.raises(ValueError, match="column x is not numeric"):
        validate(tracks.assign(x=tracks.x.astype(str)), meta)


def test_validate_rejects_impossible_speed(clean):
    tracks, meta = clean
    bad = tracks.copy(); bad.loc[bad.agent_id == 2, "vx"] = 9.0
    with pytest.raises(ValueError, match="faster than the sanity bound"):
        validate(bad, meta)
