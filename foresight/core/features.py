"""Turn interface tracks into prediction rows (one row = agent, time, horizon).

Domain-agnostic: everything is in the predicted agent's own frame (x axis = its heading),
so the same features work for a person in a plaza and a car on a road. Domain extras
(`ext_*` columns) can be appended by an adapter.

Target = where the agent really is after h seconds, minus where constant velocity (from the
*observed*, filtered state) says it would be, rotated into the agent frame:
    dd = R(theta) @ (p_true(t+h) - p_obs(t) - v_obs(t) * h)
A prediction is turned back into a world position with `to_world()`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OBS_STEP = 0.2      # seconds between history samples
N_HIST = 5          # history samples (1 s)
N_NEIGH = 3         # nearest agents described
FAR = 99.0          # padding distance for "no such neighbour"

META_COLS = ["episode", "t", "t_known", "agent_id", "h", "regime", "since_switch", "mode",
             "px", "py", "vx", "vy", "theta", "ddx", "ddy", "true_x", "true_y"]


def feature_names():
    f = ["h", "speed", "a_lon", "a_lat", "turn_rate", "radius"]
    f += [f"hist_{a}{k}" for k in range(1, N_HIST + 1) for a in ("x", "y")]
    for n in range(N_NEIGH):
        f += [f"n{n}_x", f"n{n}_y", f"n{n}_vx", f"n{n}_vy", f"n{n}_dist"]
    f += ["robot_x", "robot_y", "robot_dist", "robot_closing"]
    f += ["obst_x", "obst_y", "obst_dist", "density_3m"]
    return f


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [-s, c]])        # world -> agent frame


def _kalman_cv(z, dt, q=2.0, r=0.1):
    """Forward-only constant-velocity Kalman filter over one agent. z: [n, 2] with NaN gaps
    allowed at the ends only. Returns smoothed positions and velocities [n, 2] each."""
    n = len(z)
    F = np.eye(4); F[0, 2] = F[1, 3] = dt
    H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1
    Q = q * np.array([[dt**3 / 3, 0, dt**2 / 2, 0], [0, dt**3 / 3, 0, dt**2 / 2],
                      [dt**2 / 2, 0, dt, 0], [0, dt**2 / 2, 0, dt]])
    R = (r ** 2) * np.eye(2)
    P_out = np.full((n, 2), np.nan); V_out = np.full((n, 2), np.nan)
    x = None; P = None
    for k in range(n):
        if np.isnan(z[k, 0]):
            x = None
            continue
        if x is None:
            x = np.array([z[k, 0], z[k, 1], 0.0, 0.0]); P = np.diag([r**2, r**2, 4.0, 4.0])
        else:
            x = F @ x; P = F @ P @ F.T + Q
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (z[k] - H @ x); P = (np.eye(4) - K @ H) @ P
        P_out[k] = x[:2]; V_out[k] = x[2:]
    return P_out, V_out


def _obstacle_vec(p, obst, bounds):
    """Vector from p to the nearest obstacle surface point, and its distance."""
    best_d, best_v = FAR, np.array([FAR, 0.0])
    if bounds is not None:
        x0, y0, x1, y1 = bounds
        for d, v in ((p[0] - x0, (-1, 0)), (x1 - p[0], (1, 0)), (p[1] - y0, (0, -1)), (y1 - p[1], (0, 1))):
            if d < best_d:
                best_d, best_v = d, np.array(v, float) * d
    for o in obst:
        c = np.array([o["x"], o["y"]]); r = o["r"]
        diff = c - p; dist = np.linalg.norm(diff)
        d = dist - r
        if d < best_d and dist > 1e-6:
            best_d, best_v = d, diff / dist * d
    return best_v, best_d


def build_rows(tracks, obstacles, meta, horizons=(1.0, 2.0), row_every=0.5, noise=0.1, seed=0,
               episodes=None, query_only=False, label_from="truth"):
    """Return a DataFrame with META_COLS + feature_names() columns.

    query_only=True: live use. Rows only for the last time step in `tracks`, one per agent and
    horizon, with no target (ddx, ddy, true_x, true_y are NaN).

    label_from="observed": targets from the noisy, Kalman-filtered future positions, as a robot
    would label its own live rows. "truth" (default) uses simulator / annotated positions.

    Noise is drawn per agent from (seed, agent_id) over absolute time steps, so a growing history
    gives identical observations, also when agents enter or leave: live queries match offline rows."""
    dt = float(meta["dt"])
    k = max(1, round(OBS_STEP / dt))
    hsteps = [round(h / dt) for h in horizons]
    every = max(1, round(row_every / dt))
    rng = np.random.default_rng(seed)
    bounds = meta.get("bounds")
    out = []
    eps = sorted(tracks["episode"].unique()) if episodes is None else episodes
    for ep in eps:
        tr = tracks[tracks["episode"] == ep]
        em = meta.get("episodes", {}).get(str(ep), {}) or {}
        t_switch = em.get("regime_switch_t")
        obst = []
        if obstacles is not None and len(obstacles):
            for _, o in obstacles[obstacles["episode"] == ep].iterrows():
                r = o["r"] if o["shape"] == "circle" else 0.5 * np.hypot(o["w"], o["h"])
                obst.append({"x": o["x"], "y": o["y"], "r": r})
        steps = np.round(tr["t"].to_numpy() / dt).astype(int)
        agents = np.sort(tr["agent_id"].unique())
        aidx = {a: i for i, a in enumerate(agents)}
        S, A = steps.max() + 1, len(agents)
        Pt = np.full((S, A, 2), np.nan); Vt = np.full((S, A, 2), np.nan)
        ai = tr["agent_id"].map(aidx).to_numpy()
        Pt[steps, ai] = tr[["x", "y"]].to_numpy(); Vt[steps, ai] = tr[["vx", "vy"]].to_numpy()
        reg = np.full((S, A), "", dtype=object); reg[steps, ai] = tr["regime"].to_numpy()
        info = tr.groupby("agent_id").first()
        radius = info.loc[agents, "radius"].to_numpy()
        is_ego = info.loc[agents, "is_ego"].to_numpy().astype(bool)
        ego = int(np.flatnonzero(is_ego)[0]) if is_ego.any() else None
        # observe: noise + forward Kalman filter per agent
        # noise seeded per agent over absolute time steps: identical observations for a growing history
        # even when agents enter or leave (a shared (S, A, 2) draw would shift with the agent set)
        Po = Pt.copy()
        for a_i, aid in enumerate(agents):
            Po[:, a_i] += np.random.default_rng([seed, int(aid)]).normal(0, noise, (S, 2))
        Ps = np.full_like(Pt, np.nan); Vs = np.full_like(Pt, np.nan)
        for a in range(A):
            Ps[:, a], Vs[:, a] = _kalman_cv(Po[:, a], dt, r=max(noise, 0.05))
        step_iter = [S - 1] if query_only else range(N_HIST * k, S, every)
        for s in step_iter:
            if s < N_HIST * k:
                continue
            alive = ~np.isnan(Ps[s, :, 0])
            for a in np.flatnonzero(alive):
                if a == ego or np.isnan(Ps[s - N_HIST * k, a, 0]):
                    continue
                p, v = Ps[s, a], Vs[s, a]
                speed = float(np.hypot(*v))
                if speed > 0.2:
                    theta = np.arctan2(v[1], v[0])
                else:
                    disp = p - Ps[s - N_HIST * k, a]
                    theta = np.arctan2(disp[1], disp[0]) if np.hypot(*disp) > 0.05 else 0.0
                R = _rot(theta)
                vprev = Vs[s - k, a]
                acc = R @ ((v - vprev) / (k * dt))
                turn = np.arctan2(v[1], v[0]) - np.arctan2(vprev[1], vprev[0])
                turn = ((turn + np.pi) % (2 * np.pi) - np.pi) / (k * dt) if speed > 0.2 else 0.0
                hist = [R @ (Ps[s - j * k, a] - p) for j in range(1, N_HIST + 1)]
                others = [b for b in np.flatnonzero(alive) if b != a and b != ego]
                rel = sorted(((np.linalg.norm(Ps[s, b] - p), b) for b in others))
                neigh = []
                for n in range(N_NEIGH):
                    if n < len(rel):
                        d, b = rel[n]
                        neigh += [*(R @ (Ps[s, b] - p)), *(R @ (Vs[s, b] - v)), d]
                    else:
                        neigh += [FAR, 0.0, 0.0, 0.0, FAR]
                if ego is not None and not np.isnan(Ps[s, ego, 0]):
                    dvec = Ps[s, ego] - p; dist = float(np.linalg.norm(dvec))
                    closing = float(-(dvec @ (Vs[s, ego] - v)) / max(dist, 1e-6))
                    robot = [*(R @ dvec), dist, closing]
                else:
                    robot = [FAR, 0.0, FAR, 0.0]
                ov, od = _obstacle_vec(p, obst, bounds)
                density = sum(1 for d, _ in rel if d < 3.0)
                base = [speed, acc[0], acc[1], turn, radius[a], *np.ravel(hist), *neigh, *robot,
                        *(R @ ov), od, density]
                for hs in hsteps:
                    h = hs * dt                  # horizon snapped to the data's time grid
                    if query_only:
                        true_f, dd, mode = np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), ""
                    elif s + hs >= S or np.isnan(Pt[s + hs, a, 0]):
                        continue
                    else:
                        true_f = Ps[s + hs, a] if label_from == "observed" else Pt[s + hs, a]
                        dd = R @ (true_f - p - v * h)
                    # mode label from ground truth over the next 2 s (or the horizon)
                    s2 = min(S - 1, s + round(2.0 / dt))
                    vt0, vt2 = Vt[s, a], Vt[s2, a]
                    if query_only:
                        pass
                    elif np.isnan(vt2[0]) or np.nanmean(np.hypot(*Vt[s:s2 + 1, a].T)) < 0.2:
                        mode = "stop"
                    else:
                        dth = np.arctan2(vt2[1], vt2[0]) - np.arctan2(vt0[1], vt0[0])
                        dth = (dth + np.pi) % (2 * np.pi) - np.pi
                        mode = "turn_left" if dth > np.radians(30) else "turn_right" if dth < -np.radians(30) else "straight"
                    t = s * dt
                    since = (t - t_switch) if t_switch is not None else np.nan
                    out.append([ep, t, t + h, agents[a], h, reg[s, a], since, mode,
                                p[0], p[1], v[0], v[1], theta, dd[0], dd[1], true_f[0], true_f[1],
                                h, *base])
    # META already holds "h"; the model's copy of the horizon is the feature column "h_f"
    return pd.DataFrame(out, columns=META_COLS + ["h_f"] + feature_names()[1:])


def X_of(df):
    """Model input matrix: the feature columns (horizon included)."""
    return df[["h_f"] + feature_names()[1:]].to_numpy(dtype=float)


def to_world(df, ddx, ddy):
    """Agent-frame residual predictions -> world positions at t+h."""
    c, s = np.cos(df["theta"].to_numpy()), np.sin(df["theta"].to_numpy())
    wx = c * ddx - s * ddy; wy = s * ddx + c * ddy
    h = df["h"].to_numpy()
    return (df["px"].to_numpy() + df["vx"].to_numpy() * h + wx,
            df["py"].to_numpy() + df["vy"].to_numpy() * h + wy)


def split_by_episode(df, test_frac=0.25, val_frac=0.0, seed=0):
    eps = np.array(sorted(df["episode"].unique()))
    rng = np.random.default_rng(seed); rng.shuffle(eps)
    n_te = max(1, int(len(eps) * test_frac)); n_va = int(len(eps) * val_frac)
    te, va = set(eps[:n_te]), set(eps[n_te:n_te + n_va])
    ep = df["episode"]
    return ~ep.isin(te | va), ep.isin(va), ep.isin(te)
