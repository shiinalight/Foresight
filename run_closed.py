"""R3 closed loop: a robot crosses the simulated crowd, choosing its moves from predicted ranges.

    python run_closed.py --planners cv tabpfn tabpfn_live --seeds 1000 1001 --log
    python run_closed.py --planners cv cv_band lgbm lgbm_all lgbm_live tabpfn tabpfn_live --seeds 1000 1001 1002

Planners (same seeds = same crowd, start and goal for all):
  cv           constant velocity, no range
  cv_band      constant velocity + fixed band from all bank rows
  lgbm         LightGBM on --context random bank rows (same rows as tabpfn)
  lgbm_all     LightGBM on every bank row
  tabpfn       TabPFN on --context random bank rows, fitted once (frozen)
  lgbm_live    refitted every --refit s on context.select(): this episode's completed rows
  tabpfn_live  (labelled from observed, filtered positions, as a robot would) + similar bank rows
  tabpfn_split frozen bank model + a scene model refitted every --refit s on this episode's <= 250 most recent
               completed rows (observed labels); quantiles averaged per axis; frozen alone below 20 recent rows
--frozen-ctx fill: lgbm and tabpfn read context.frozen(bank, --context): natural move frequency, exact budget, no kNN.
--bank-exclude-regime erratic: drop every bank episode with that regime before or after a switch.
--switch-to erratic --switch-all: every seed switches into that regime at mid-episode, from a start regime
               other than it (start = the other regimes in turn by seed).
--offset: in-loop online conformal offset on any learned planner's box (Gibbs & Candes additive form,
               d <- d + eta (miss - alpha), d0 = 0, eta 0.02): every --refit s, predictions whose outcome is now
               known (t + h <= now) are scored against OBSERVED (filtered) positions, in order of t_known. As
               offline (conformal_eval.py): the score is on the RAW 5-95% box, target miss alpha = 0.2. The box the
               planner reads ([box_q, 1 - box_q]) is widened by the same d, so the caution dial keeps its meaning.
--latency L:   virtual-time latency: a command planned at t applies at t + L; the robot keeps its previous
               command meanwhile.
Short 0.4 s check: constant velocity from the filtered state for cv, cv_band, lgbm* and tabpfn*; people frozen for
reactive; true positions for the open-loop oracle; none for vo (time-to-collision penalty) and stop_go (sim policy).

Writes results/closed/<run_id>.csv (one row per planner x seed), robot features of the live
queries to results/closed/robotfeat_<run_id>.csv (bank-vs-live shift check), and with --log
results/closed/<sim>_<planner>_<seed>.pkl for render_closed.py.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.core import conformal as CF
from foresight.core import context as C
from foresight.core import features as F
from foresight.core import models as M
from foresight.core import planner as P
from foresight.domains.toy_crowd import BOUND, CrowdSim
from run_offline import rows_for

HZ = (1.0, 2.0)
VERSION = "v4"          # v4: one top speed, metrics from actual robot velocity, navigation baselines
KEY = ["sim", "planner", "seed", "margin", "box_q", "version", "horizons", "context", "people", "bank", "refit", "switch",
       "max_speed", "vo_w", "robot_repulsion", "ignore_robot", "goal_mode", "smooth",
       "tabpfn_cfg", "frozen_ctx", "bank_exclude", "switch_to", "switch_all", "switch_t", "offset", "offset_alpha", "offset_eta", "offset_rule", "offset_eta_b", "latency", "replan"]
DEFAULTS = {"margin": 0.2, "box_q": 0.05, "version": "v1", "horizons": "1 2", "context": 1000, "people": 16,
            "bank": "data/crowd", "refit": 0.0, "tabpfn_cfg": "default", "switch": False, "threads": 0,
            "max_speed": 1.2, "vo_w": -1.0, "robot_repulsion": 2.0, "ignore_robot": 0.0, "goal_mode": "anywhere", "smooth": False,
            "frozen_ctx": "random", "bank_exclude": "-", "switch_to": "-", "switch_all": False, "switch_t": -1.0,
            "offset": False, "offset_alpha": -1.0, "offset_eta": 0.02, "offset_rule": "row", "offset_eta_b": -1.0, "latency": 0.0, "replan": 0.4}
SPLIT_REC, SPLIT_THIN = 250, 20   # tabpfn_split: scene-model rows (most recent); fewer than SPLIT_THIN = frozen alone
LIVE_WINDOW = 25.0      # seconds of episode history used to label live rows


def load_closed(folder="results/closed", version=VERSION):
    """All closed-loop runs, one row per KEY (the latest run wins). version=None keeps all."""
    files = [f for f in sorted(Path(folder).glob("*.csv")) if not f.name.startswith("robotfeat_")]
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for c, v in DEFAULTS.items():
        if c not in d:
            d[c] = v
        d[c] = d[c].fillna(v)
    d["horizons"] = d["horizons"].astype(str)
    d = d.drop_duplicates(KEY, keep="last")
    return d[d.version == version] if version else d


class Predictor:
    """Frozen or live predictor behind one call: predict(q, t, hist_df)."""

    def __init__(self, kind, bank, bank_obj, args):
        self.kind, self.args, self.bank_obj = kind, args, bank_obj
        self.live = kind.endswith("_live")
        self.split = kind.endswith("_split")
        self.base = kind.replace("_live", "").replace("_all", "").replace("_split", "")
        self.frozen_model, self.n_rec = None, 0
        self.agg = []                                            # boxes served this episode (in-loop aggregates)
        self.offset = bool(getattr(args, "offset", False)) and kind.startswith(("lgbm", "tabpfn", "cv_band"))
        oa = getattr(args, "offset_alpha", None)
        self.eta = float(getattr(args, "offset_eta", 0.02))      # 0 = score the box only, never move it
        self.rule = getattr(args, "offset_rule", "row")
        self.eta_b = float(getattr(args, "offset_eta_b", 0.05))
        self.alpha = 0.2 if oa is None else oa
        self.d, self.pending, self.last_off, self.d_hist = 0.0, [], -1e9, []
        self.n_miss, self.n_scored, self.n_miss_late, self.n_late = 0.0, 0, 0.0, 0
        self.last_fit, self.model, self.n_live = -1e9, None, 0
        self.ctx_rows, self.refit_at = None, None                  # for the trace: what the model last read
        if kind in ("stop_go", "reactive", "vo", "oracle"):
            return                                              # baselines and the oracle: no learned predictor
        if not self.live:
            if kind == "cv":
                self.model = M.make("cv")
            else:
                if kind in ("cv_band", "lgbm_all"):
                    rows = bank.assign(source="bank-all")
                elif getattr(args, "frozen_ctx", "random") == "fill":
                    rows = C.frozen(bank, args.context)
                else:
                    rows = bank.sample(min(args.context, len(bank)), random_state=0).assign(source="bank-random")
                self.model = self._fit(rows)
                self.frozen_model = self.model
                self.ctx_rows, self.refit_at = rows, 0.0

    def _fit(self, rows):
        kw = {}
        if self.base == "tabpfn":
            # 3.5-Fast, 1 estimator: same offline accuracy on the crowd as the default (0.99 vs 0.98 m at
            # +2 s, same coverage) and ~13x faster on CPU (tools/latency.py notes in RESULTS.md)
            kw = {"fit_mode": "fit_with_cache", "fast": self.args.tabpfn_fast, "n_estimators": self.args.tabpfn_est}
        return M.make(self.base, **kw).fit(F.X_of(rows), rows[["ddx", "ddy"]].to_numpy())

    def reset(self):
        """Start of an episode: live and split models forget the last episode's scene."""
        self.last_fit, self.n_live, self.n_rec = -1e9, 0, 0
        self.d, self.pending, self.last_off, self.d_hist = 0.0, [], -1e9, []
        self.n_miss, self.n_scored, self.n_miss_late, self.n_late = 0.0, 0, 0.0, 0
        self.agg = []
        if self.split:
            self.model = self.frozen_model

    def _update_offset(self, t, live):
        """Score every stored prediction whose outcome is known by t against observed labels; d moves per row."""
        if not self.pending:
            return
        P_ = pd.concat(self.pending, ignore_index=True)
        P_["t_known"] = P_.t + P_.h
        done = P_[P_.t_known <= t + 1e-9]
        self.pending = [P_[P_.t_known > t + 1e-9].drop(columns="t_known")]
        if not len(done):
            return
        lab = live[["agent_id", "t", "h", "ddx", "ddy"]].assign(t=live.t.round(6), h=live.h.round(6))
        m = done.assign(t=done.t.round(6), h=done.h.round(6)).merge(lab, on=["agent_id", "t", "h"], how="inner")
        m = m.dropna(subset=["ddx", "ddy"]).sort_values(["t_known", "agent_id", "h"], kind="stable")
        s = np.maximum.reduce([(m.q05a - m.ddx).to_numpy(), (m.ddx - m.q95a).to_numpy(),
                               (m.q05c - m.ddy).to_numpy(), (m.ddy - m.q95c).to_numpy()])   # raw 5-95% box
        misses = (s > m.d_used.to_numpy()).astype(float)
        if self.eta == 0:
            pass                                                 # measure only
        elif self.rule == "batch":                               # one step per refresh: d += eta_b (batch miss rate - alpha)
            if len(misses):
                self.d += self.eta_b * (float(misses.mean()) - self.alpha)
        else:                                                    # per row, in order of t_known
            for miss in misses:
                self.d += self.eta * (miss - self.alpha)
        self.n_miss += float(misses.sum())
        self.n_scored += len(m)
        st = getattr(self.args, "switch_t", None)
        late = (m.t >= (self.args.duration / 2 if st is None else st) + 2.0).to_numpy()   # predictions made after the switch has settled
        self.n_miss_late += float((s > m.d_used.to_numpy())[late].sum()); self.n_late += int(late.sum())
        self.d_hist.append((t, self.d, int(len(m))))

    def offset_stats(self):
        if not self.offset:
            return {"offset": False, "offset_alpha": -1.0, "offset_final": np.nan, "offset_updates": 0,
                    "offset_mean": np.nan, "offset_miss": np.nan, "offset_miss_after_switch": np.nan,
                    "offset_eta": np.nan, "offset_rule": "-", "offset_eta_b": np.nan}
        n = sum(h[2] for h in self.d_hist)
        return {"offset": True, "offset_alpha": self.alpha, "offset_eta": self.eta, "offset_rule": self.rule,
                "offset_eta_b": self.eta_b if self.rule == "batch" else np.nan, "offset_final": self.d, "offset_updates": n,
                "offset_mean": float(np.mean([h[1] for h in self.d_hist])) if self.d_hist else 0.0,
                "offset_miss": self.n_miss / max(self.n_scored, 1),
                "offset_miss_after_switch": self.n_miss_late / self.n_late if self.n_late else np.nan}

    def predict(self, q, t, hist_df, obst_df, meta, seed):
        pr = self._predict(q, t, hist_df, obst_df, meta, seed)
        if not self.offset:
            self.log_boxes(pr)
            return pr
        if t - self.last_off >= self.args.refit - 1e-9:
            # rows at every step (row_every = dt), so each stored prediction (made every --replan s) finds its label
            live = F.build_rows(hist_df, obst_df, meta, horizons=self.args.hz, noise=self.args.noise, seed=seed,
                                label_from="observed", row_every=meta["dt"])
            self._update_offset(t, live)
            self.last_off = t
        if len(pr):
            # store the RAW box with the offset in force: a miss is s(raw) > d_used, i.e. outside the served box
            self.pending.append(pr[["agent_id", "t", "h", "q05a", "q95a", "q05c", "q95c"]].assign(d_used=self.d))
            lo, hi = CF.widen(pr[["lo_a", "lo_c"]].to_numpy(float), pr[["hi_a", "hi_c"]].to_numpy(float), self.d)
            pr = pr.assign(lo_a=lo[:, 0], lo_c=lo[:, 1], hi_a=hi[:, 0], hi_c=hi[:, 1])
        self.log_boxes(pr)
        return pr

    def _predict(self, q, t, hist_df, obst_df, meta, seed):
        if self.split and t - self.last_fit >= self.args.refit - 1e-9:
            live = F.build_rows(hist_df, obst_df, meta, horizons=self.args.hz, noise=self.args.noise, seed=seed,
                                label_from="observed")
            rec = live[live.t_known <= t + 1e-9].sort_values("t_known").tail(SPLIT_REC)
            self.last_fit, self.n_rec, self.n_live = t, int(len(rec)), int(len(rec))
            if len(rec) < SPLIT_THIN:
                self.model = self.frozen_model                     # thin scene: frozen alone
            else:
                self.model = AvgQuantiles(self.frozen_model, self._fit(rec))
                self.ctx_rows, self.refit_at = rec.assign(source="recent"), t
        if self.live and t - self.last_fit >= self.args.refit - 1e-9:
            # full history, so the Kalman state matches the queries; then keep the last LIVE_WINDOW s
            live = F.build_rows(hist_df, obst_df, meta, horizons=self.args.hz, noise=self.args.noise, seed=seed,
                                label_from="observed")
            live = live[(live.t_known <= t + 1e-9) & (live.t >= t - LIVE_WINDOW)]
            ctx = C.select(q, self.bank_obj, live=live if len(live) else None, budget=self.args.context,
                           mix=(0.6, 0.3, 0.1))
            self.model, self.last_fit, self.n_live = self._fit(ctx), t, int(len(live))
            self.ctx_rows, self.refit_at = ctx, t
        return self._rows(q)

    def _rows(self, q):
        """Planner box [box_q, 1 - box_q] plus the 5-95% box (q05*/q95*, for the in-loop claim-B aggregates),
        from one predict_quantiles call."""
        bq = self.args.box_q
        Q = self.model.predict_quantiles(F.X_of(q), (0.05, bq, 0.5, 1 - bq, 0.95))
        q = q.copy()
        q["mx"], q["my"] = F.to_world(q, Q[:, 0, 2], Q[:, 1, 2])
        q["lo_a"], q["hi_a"], q["lo_c"], q["hi_c"] = Q[:, 0, 1], Q[:, 0, 3], Q[:, 1, 1], Q[:, 1, 3]
        q["q05a"], q["q95a"], q["q05c"], q["q95c"] = Q[:, 0, 0], Q[:, 0, 4], Q[:, 1, 0], Q[:, 1, 4]
        q["q50a"], q["q50c"] = Q[:, 0, 2], Q[:, 1, 2]
        return q

    def log_boxes(self, pr):
        if len(pr) and "q05a" in pr:
            self.agg.append(pr[["agent_id", "t", "h", "lo_a", "hi_a", "lo_c", "hi_c", "q05a", "q95a", "q05c", "q95c",
                                "q50a", "q50c"]].assign(d_used=self.d if self.offset else 0.0))

    def inloop_stats(self, hist, obst_df, meta, seed, switch_t):
        """Episode aggregates at h = 2 s against TRUE positions: joint coverage, width and interval score of the
        5-95% box, joint coverage of the served (planner) box, median error; before and after the switch."""
        keys = [f"inloop_{m}_{w}" for w in ("pre", "post")
                for m in ("n", "cov90", "w90", "is90", "served_cov", "err", "cov90d")]
        out = dict.fromkeys(keys, np.nan)
        if not self.agg:
            return out
        A = pd.concat(self.agg, ignore_index=True)
        A = A[np.isclose(A.h, 2.0)]
        hist_df = pd.DataFrame([r for s in hist for r in s])
        tr = F.build_rows(hist_df, obst_df, meta, horizons=(2.0,), noise=self.args.noise, seed=seed, row_every=meta["dt"])
        lab = tr[["agent_id", "t", "ddx", "ddy"]].assign(t=tr.t.round(6))
        m = A.assign(t=A.t.round(6)).merge(lab, on=["agent_id", "t"], how="inner").dropna(subset=["ddx", "ddy"])
        y = m[["ddx", "ddy"]].to_numpy(float)
        lo, hi = m[["q05a", "q05c"]].to_numpy(float), m[["q95a", "q95c"]].to_numpy(float)
        slo, shi = m[["lo_a", "lo_c"]].to_numpy(float), m[["hi_a", "hi_c"]].to_numpy(float)
        cov = ((y >= lo) & (y <= hi)).all(axis=1)
        scov = ((y >= slo) & (y <= shi)).all(axis=1)
        dlo, dhi = CF.widen(lo, hi, m.d_used.to_numpy(float))     # 5-95% box widened by the offset in force
        dcov = ((y >= dlo) & (y <= dhi)).all(axis=1)
        w = (hi - lo).mean(axis=1)
        isc = ((hi - lo) + 20 * (lo - y) * (y < lo) + 20 * (y - hi) * (y > hi)).mean(axis=1)
        err = np.linalg.norm(m[["q50a", "q50c"]].to_numpy(float) - y, axis=1)
        post = (m.t >= switch_t - 1e-9).to_numpy() if switch_t is not None else np.zeros(len(m), bool)
        for wname, mask in (("pre", ~post), ("post", post)):
            if mask.any():
                out |= {f"inloop_n_{wname}": int(mask.sum()), f"inloop_cov90_{wname}": float(cov[mask].mean()),
                        f"inloop_w90_{wname}": float(w[mask].mean()), f"inloop_is90_{wname}": float(isc[mask].mean()),
                        f"inloop_served_cov_{wname}": float(scov[mask].mean()),
                        f"inloop_err_{wname}": float(np.median(err[mask])),
                        f"inloop_cov90d_{wname}": float(dcov[mask].mean())}
        return out


class AvgQuantiles:
    """Average two models' quantiles per axis (then sorted, so the bounds never cross)."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def predict_quantiles(self, X, qs):
        return np.sort((self.a.predict_quantiles(X, qs) + self.b.predict_quantiles(X, qs)) / 2, axis=2)


def zigzag(headings, path_len, window=2.0, big=np.radians(45)):
    """heading_change_per_m: total |heading change| of the robot's motion per metre travelled (rad/m).
    reversals_45: turns of > 45 deg followed within `window` s by a turn of > 45 deg the other way
    (turning points of the cumulative heading)."""
    if len(headings) < 3 or path_len <= 0:
        return {"heading_change_per_m": 0.0, "reversals_45": 0}
    t = np.array([h[0] for h in headings]); a = np.unwrap(np.array([h[1] for h in headings]))
    d = np.diff(a)
    total = float(np.abs(d).sum())
    sgn = np.sign(np.where(np.abs(d) < np.radians(1), 0, d))
    nz = np.flatnonzero(sgn)
    turns = [nz[0]] + [j for i, j in zip(nz[:-1], nz[1:]) if sgn[j] != sgn[i]] if len(nz) else []
    pts = [(t[i], a[i]) for i in turns] + [(t[-1], a[-1])]
    rev = 0
    for (t0, a0), (t1, a1), (t2, a2) in zip(pts[:-2], pts[1:-1], pts[2:]):
        if abs(a1 - a0) > big and abs(a2 - a1) > big and (t2 - t0) <= 2 * window and np.sign(a1 - a0) != np.sign(a2 - a1):
            rev += 1
    return {"heading_change_per_m": total / path_len, "reversals_45": rev}


def oracle_rows(sim, cmd, horizons, dt, t_now, short_h=0.4):
    """Perfect prediction: deep-copy the simulator, roll it forward holding the current robot command,
    and return each person's TRUE position at short_h and every horizon as zero-width boxes
    (px, py = true future position, no velocity, no range). The h = short_h rows come first per agent,
    so planner.choose's 0.4 s check reads the true position at +0.4 s."""
    import copy
    twin = copy.deepcopy(sim)
    hs = sorted({short_h, *horizons})
    steps = {round(h / dt): h for h in hs}
    rows = []
    for k in range(1, max(steps) + 1):
        state, _ = twin.step(None if cmd is None else np.asarray(cmd, float))
        if k in steps:
            rows += [dict(t=t_now, agent_id=r["agent_id"], h=steps[k], px=r["x"], py=r["y"], vx=0.0, vy=0.0,
                          theta=0.0, lo_a=0.0, hi_a=0.0, lo_c=0.0, hi_c=0.0) for r in state[1:]]
    return pd.DataFrame(rows).sort_values(["h", "agent_id"], kind="stable") if rows else pd.DataFrame()


def make_sim(name, seed, args):
    if name == "toy":
        sim = CrowdSim(seed, n_people=args.people, duration=args.duration,
                       robot_start=(-BOUND + 1.5, 0.0), robot_goal=(BOUND - 1.5, 0.0))
        return sim, sim.dt, BOUND
    from foresight.domains.crowd import REGIMES, CrowdConfig, CrowdSim as NCrowdSim
    names = list(REGIMES)
    to = getattr(args, "switch_to", None)
    if to:                                                       # every start regime but the target, in turn
        others = [n for n in names if n != to]
        regime = others[seed % len(others)]
    else:
        regime = names[seed % len(names)]                        # vary the crowd across seeds
    on = args.switch and (getattr(args, "switch_all", False) or seed % 2 == 1)
    st = getattr(args, "switch_t", None)
    switch = dict(switch_t=round(args.duration / 2 if st is None else st, 6), switch_to=to or names[(seed + 1) % len(names)]) if on else {}
    cfg = CrowdConfig(n_people=args.people, duration=args.duration, regime=regime,
                      robot_repulsion=args.robot_repulsion, ignore_robot=args.ignore_robot,
                      robot_speed=args.max_speed, goal_mode=getattr(args, "goal_mode", "anywhere"), **switch)
    return NCrowdSim(seed, cfg), cfg.dt, cfg.half_size


def run_episode(pred, seed, args, log=False):
    sim, dt, L = make_sim(args.sim, seed, args)
    state = [dict(r, episode=seed) for r in sim.reset()]
    hist = [state]
    meta = F.load_kalman(args.bank or f"data/{args.sim}", {"dt": dt, "bounds": [-L, -L, L, L], "episodes": {}})
    obstacles = getattr(sim, "obst", None) or getattr(sim, "obstacles", [])
    obst_df = pd.DataFrame([dict(episode=seed, obstacle_id=i, shape="circle", **o) for i, o in enumerate(obstacles)])
    every = max(1, round(args.replan / dt))
    pred.reset()
    lag = int(round(getattr(args, "latency", 0.0) / dt))
    queue, applied = [], np.zeros(2)                         # (apply step, command) for the latency budget
    frames, plans, robotfeat = [], [], []
    collided_steps, moving_collisions, min_gap, t_goal, t_pred = 0, 0, np.inf, None, 0.0
    n_plans, n_fallback, k = 0, 0, 0
    path_len, stopped_steps, intrusion_steps, decels = 0.0, 0, 0, []
    start = np.array([state[0]["x"], state[0]["y"]]); goal0 = np.array([state[0]["goal_x"], state[0]["goal_y"]])
    speeds = tuple(s * args.max_speed for s in (0.0, 1 / 3, 2 / 3, 1.0))   # one top speed for every planner
    smooth = (dict(P.SMOOTH, a_max=args.sm_a_max, a_brake=args.sm_a_brake, w_max=np.radians(args.sm_w_max),
                   dt=args.replan, l_turn=args.sm_l_turn, l_dv=args.sm_l_dv, hyst=args.sm_hyst)
              if getattr(args, "smooth", False) else None)
    headings = []                                            # (t, heading) while moving, for zigzag metrics
    cmd = np.zeros(2)
    while k * dt < args.duration:
        p_robot = np.array([state[0]["x"], state[0]["y"]])
        goal = np.array([state[0]["goal_x"], state[0]["goal_y"]])
        if pred.kind == "stop_go":
            cmd = None                                          # the simulator's own stop-and-go policy
        elif k % every == 0:
            # full history + fixed noise seed: same observations and Kalman state as the offline rows
            tf0 = time.perf_counter()
            hist_df = pd.DataFrame([r for s in hist for r in s])
            q = F.build_rows(hist_df, obst_df, meta, horizons=args.hz, noise=args.noise, seed=seed, query_only=True)
            t0 = time.perf_counter()
            refit_before = pred.refit_at
            if pred.kind == "vo":
                v_robot = np.array([state[0]["vx"], state[0]["vy"]])
                cmd, info = P.choose_vo(p_robot, v_robot, goal, q.drop_duplicates("agent_id") if len(q) else q,
                                        max_speed=args.max_speed, w=args.vo_w, horizon=args.vo_h,
                                        bounds=meta["bounds"], obstacles=obstacles,
                                        prev_cmd=cmd if smooth else None, smooth=smooth)
                pr = q
            else:
                if pred.kind == "reactive":                     # people frozen where they are: no prediction at all
                    pr = q.assign(vx=0.0, vy=0.0, lo_a=0.0, hi_a=0.0, lo_c=0.0, hi_c=0.0) if len(q) else q
                elif pred.kind == "oracle":                     # true future positions (upper bound on prediction)
                    pr = oracle_rows(sim, cmd, args.hz, dt, k * dt)
                else:
                    pr = pred.predict(q, k * dt, hist_df, obst_df, meta, seed) if len(q) else q
                tp1 = time.perf_counter()
                cmd, info = P.choose(p_robot, goal, pr, margin=args.margin, bounds=meta["bounds"],
                                     obstacles=obstacles, horizons=args.hz, speeds=speeds,
                                     prev_cmd=cmd if smooth else None, smooth=smooth)
            t1 = time.perf_counter()
            t_pred += t1 - t0
            n_plans += 1; n_fallback += int(info.get("fallback", False))
            if len(q):
                robotfeat.append(q[["robot_dist", "robot_closing"]].assign(t=k * dt))
            if log:                                              # capture outside the timed region
                refit = pred.refit_at is not None and pred.refit_at != refit_before
                plans.append({"t": k * dt, "pred": pr, "q": q, "cmd": cmd, "n_live": pred.n_live, **info,
                              "ctx": pred.ctx_rows if (refit or not plans) else None,
                              "timing": {"features_ms": 1000 * (t0 - tf0),
                                         "predict_ms": 1000 * ((tp1 if pred.kind not in ("vo",) else t1) - t0),
                                         "plan_ms": 1000 * (t1 - (tp1 if pred.kind not in ("vo",) else t0)),
                                         "refit": bool(refit)}})
        prev = state
        if cmd is None or lag == 0:
            applied = cmd
        else:
            if k % every == 0:
                queue.append((k + lag, np.array(cmd, float)))
            while queue and queue[0][0] <= k:
                applied = queue.pop(0)[1]
        state, info = sim.step(applied)
        state = [dict(r, episode=seed) for r in state]
        k += 1
        hist.append(state)
        rob = state[0]
        v_act = float(np.hypot(rob["vx"], rob["vy"]))           # the robot's actual velocity, every planner
        if v_act >= 0.05:
            headings.append((k * dt, float(np.arctan2(rob["vy"], rob["vx"]))))
        path_len += float(np.hypot(rob["x"] - prev[0]["x"], rob["y"] - prev[0]["y"]))
        stopped_steps += int(v_act < 0.05)
        gaps = np.array([np.hypot(r["x"] - rob["x"], r["y"] - rob["y"]) - r["radius"] - rob["radius"] for r in state[1:]])
        min_gap = min(min_gap, float(gaps.min()))
        intrusion_steps += int((gaps < 0.5).any())
        for r0, r1, g in zip(prev[1:], state[1:], gaps):        # how much the robot makes nearby people brake
            if g < 2.0:
                decels.append(max(0.0, (np.hypot(r0["vx"], r0["vy"]) - np.hypot(r1["vx"], r1["vy"])) / dt))
        if info["collided"]:
            collided_steps += 1
            moving_collisions += int(v_act >= 0.05)
        if log:
            frames.append({"t": k * dt, "state": state, "info": info})
        if np.hypot(goal[0] - rob["x"], goal[1] - rob["y"]) < 0.5:
            t_goal = k * dt
            break
    t_end = k * dt
    res = {"seed": seed, "reached": t_goal is not None, "time_to_goal": t_goal if t_goal else np.nan,
           "time_capped": t_goal if t_goal else args.duration,
           "collision_steps": collided_steps, "collided": collided_steps > 0,
           "moving_collision_steps": moving_collisions, "min_gap": min_gap, "near_miss": min_gap < 0.2,
           "freeze_rate": stopped_steps / max(k, 1), "seconds_stopped": stopped_steps * dt,
           "intrusion_s": intrusion_steps * dt, "path_length": path_len,
           "path_efficiency": float(max(np.linalg.norm(goal0 - start) - 0.5, 0) / path_len) if (t_goal and path_len > 0) else np.nan,
           "people_decel": float(np.mean(decels)) if decels else 0.0, "episode_s": t_end,
           **zigzag(headings, path_len),
           "fallback_rate": n_fallback / max(n_plans, 1),
           "pred_seconds": t_pred, "pred_ms_per_plan": 1000 * t_pred / max(n_plans, 1),
           **pred.offset_stats(),
           **pred.inloop_stats(hist, obst_df, meta, seed,
                               (args.duration / 2 if getattr(args, "switch_t", None) is None else args.switch_t)
                               if (args.switch and (getattr(args, "switch_all", False) or seed % 2 == 1)) else None)}
    logd = {"frames": frames, "plans": plans, "obst": obstacles, "bounds": meta["bounds"]} if log else None
    if log:
        from foresight import trace
        trace.write(args, pred, seed, hist, plans, obstacles, meta, res)
    rf = pd.concat(robotfeat) if robotfeat else pd.DataFrame()
    return res, logd, rf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planners", nargs="*", default=["cv", "tabpfn"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[1000])
    ap.add_argument("--sim", default="crowd", choices=["crowd", "toy"])
    ap.add_argument("--bank", default=None, help="dataset for context rows (default data/<sim>)")
    ap.add_argument("--context", type=int, default=1000)
    ap.add_argument("--people", type=int, default=16)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--replan", type=float, default=0.4)
    ap.add_argument("--refit", type=float, default=2.0, help="live planners refit every N seconds")
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--box-q", type=float, default=0.05, help="box = [q, 1-q] per axis (caution dial)")
    ap.add_argument("--horizons", type=float, nargs="*", default=list(HZ), help="planner checks the boxes at these")
    ap.add_argument("--switch", action="store_true", help="odd seeds: crowd-wide regime switch at mid-episode")
    ap.add_argument("--switch-to", default=None, help="with --switch: always switch into this regime (start = another)")
    ap.add_argument("--offset", action="store_true", help="in-loop online conformal offset (learned planners)")
    ap.add_argument("--offset-alpha", type=float, default=None, help="target miss rate of the raw 5-95%% box (default 0.2)")
    ap.add_argument("--offset-eta", type=float, default=0.02, help="offset step; 0 = measure misses only")
    ap.add_argument("--offset-rule", default="row", choices=["row", "batch"],
                    help="row: d moves per scored row; batch: d += eta_b (miss rate of the refresh batch - alpha)")
    ap.add_argument("--offset-eta-b", type=float, default=0.05, help="batch rule step per refresh")
    ap.add_argument("--latency", type=float, default=0.0, help="virtual-time latency of each command, s")
    ap.add_argument("--switch-t", type=float, default=None, help="with --switch: switch time in s (default mid-episode)")
    ap.add_argument("--switch-all", action="store_true", help="with --switch: every seed switches, not only odd ones")
    ap.add_argument("--frozen-ctx", default="random", choices=["random", "fill"], help="frozen lgbm/tabpfn context")
    ap.add_argument("--bank-exclude-regime", default=None, help="drop bank episodes with this regime (before or after)")
    ap.add_argument("--tabpfn-fast", type=int, default=1, help="1 = TabPFN-3.5-Fast")
    ap.add_argument("--tabpfn-est", type=int, default=1, help="TabPFN ensemble members")
    ap.add_argument("--threads", type=int, default=0, help="cap torch/OpenMP threads (parallel runs)")
    ap.add_argument("--max-speed", type=float, default=1.0, help="robot top speed, every planner (= CrowdConfig.robot_speed)")
    ap.add_argument("--robot-repulsion", type=float, default=2.0, help="T2 knob: how hard people push away from the robot")
    ap.add_argument("--ignore-robot", type=float, default=0.0, help="T2 knob: share of people who ignore the robot")
    ap.add_argument("--goal-mode", default="anywhere", choices=["anywhere", "cross"], help="crowd: goals anywhere, or cross the plaza and leave")
    ap.add_argument("--smooth", action="store_true", help="dynamic window + smoothness cost + commitment (all planners using choose/choose_vo)")
    ap.add_argument("--sm-a-max", type=float, default=1.0, help="max acceleration, m/s^2")
    ap.add_argument("--sm-a-brake", type=float, default=2.0, help="max braking, m/s^2")
    ap.add_argument("--sm-w-max", type=float, default=90.0, help="max turn rate, deg/s")
    ap.add_argument("--sm-l-turn", type=float, default=0.3, help="smoothness cost per rad of heading change (m)")
    ap.add_argument("--sm-l-dv", type=float, default=0.2, help="smoothness cost per m/s of speed change (m)")
    ap.add_argument("--sm-hyst", type=float, default=0.1, help="keep the previous command if within this much score (m)")
    ap.add_argument("--vo-w", type=float, default=1.0, help="velocity-obstacle collision weight")
    ap.add_argument("--vo-h", type=float, default=5.0, help="velocity-obstacle time horizon (s)")
    ap.add_argument("--tag", default="", help="suffix for the run id (parallel runs)")
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    if a.threads:
        import os
        import torch
        os.environ["OMP_NUM_THREADS"] = str(a.threads)
        torch.set_num_threads(a.threads)

    a.hz = tuple(a.horizons)
    bank_path = a.bank or f"data/{a.sim}"
    bank = rows_for(bank_path, a.noise, horizons=a.hz)
    if a.bank_exclude_regime:
        import json
        em = json.loads((Path(bank_path) / "meta.json").read_text())["episodes"]
        bad = set(bank[bank.regime == a.bank_exclude_regime].episode.unique())
        bad |= {int(e) for e, m in em.items() if a.bank_exclude_regime in (m.get("regime_before"), m.get("regime_after"))}
        bank = bank[~bank.episode.isin(bad)].reset_index(drop=True)
        print(f"bank: {bank.episode.nunique()} episodes, {len(bank)} rows without '{a.bank_exclude_regime}'", flush=True)
    bank_obj = C.Bank(bank) if any(p.endswith("_live") for p in a.planners) else None
    run_id = time.strftime("%Y%m%d-%H%M%S") + (f"-{a.tag}" if a.tag else "")
    out, feats = [], []
    Path("results/closed").mkdir(parents=True, exist_ok=True)
    csv_path = Path(f"results/closed/{run_id}.csv")
    preds = {kind: Predictor(kind, bank, bank_obj, a) for kind in a.planners}
    # seed outer, planner inner: a run cut short still holds complete, paired seeds
    for seed in a.seeds:
        for kind, pred in preds.items():
            t0 = time.perf_counter()
            res, logd, rf = run_episode(pred, seed, a, log=a.log)
            res |= {"sim": a.sim, "planner": kind, "wall_s": time.perf_counter() - t0, "margin": a.margin,
                    "box_q": a.box_q, "version": VERSION, "horizons": " ".join(f"{h:g}" for h in a.hz),
                    "context": a.context, "people": a.people, "bank": bank_path,
                    "refit": a.refit if kind.endswith(("_live", "_split")) else 0.0, "switch": a.switch, "run_id": run_id,
                    "frozen_ctx": a.frozen_ctx if kind.startswith(("lgbm", "tabpfn")) and kind != "lgbm_all" else "-",
                    "bank_exclude": a.bank_exclude_regime or "-", "switch_to": a.switch_to or "-",
                    "switch_all": bool(a.switch_all), "latency": a.latency, "replan": a.replan,
                    "switch_t": (a.duration / 2 if a.switch_t is None else a.switch_t) if a.switch else -1.0,
                    "tabpfn_cfg": (f"fast{a.tabpfn_fast}-est{a.tabpfn_est}" if "tabpfn" in kind else "-"),
                    "threads": a.threads, "max_speed": a.max_speed,
                    "vo_w": a.vo_w if kind == "vo" else np.nan, "vo_h": a.vo_h if kind == "vo" else np.nan,
                    "robot_repulsion": a.robot_repulsion, "ignore_robot": a.ignore_robot,
                    "features_version": F.FEATURES_VERSION, "goal_mode": a.goal_mode, "smooth": a.smooth}
            out.append(res)
            pd.DataFrame([res]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)   # checkpoint
            if len(rf):
                feats.append(rf.assign(planner=kind, seed=seed))
            print(f"{kind:<12} seed {seed}  reached {res['reached']}  t_goal {res['time_to_goal']}  "
                  f"collisions {res['collision_steps']} (moving {res['moving_collision_steps']})  min gap {res['min_gap']:.2f} m  "
                  f"freeze {100 * res['freeze_rate']:.0f}%  fallback {100 * res['fallback_rate']:.0f}%  "
                  f"{res['pred_ms_per_plan']:.0f} ms/plan  ({res['wall_s']:.0f} s)", flush=True)
            if logd:
                with open(f"results/closed/{a.sim}_{kind}_{seed}.pkl", "wb") as f:
                    pickle.dump(logd, f)
    df = pd.DataFrame(out)
    if feats:
        pd.concat(feats).to_csv(f"results/closed/robotfeat_{run_id}.csv", index=False)
    print(df.groupby("planner")[["reached", "time_to_goal", "collided", "moving_collision_steps", "near_miss", "min_gap",
                                  "freeze_rate", "fallback_rate", "pred_ms_per_plan"]].mean().round(3))
    print(f"wrote results/closed/{run_id}.csv")


if __name__ == "__main__":
    main()
