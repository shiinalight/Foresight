"""Write one trace per (scene, planner, seed): every stage of the pipeline as its own table
(docs/VISUALS.md). Renderers read only these files.

results/trace/<sim>_<planner>_<seed>/
  meta.json          run settings, model config, commit, episode result
  tracks.parquet     S0 true positions (simulator)
  observed.parquet   S1 noisy observations and Kalman-filtered states
  rows.parquet       S2 every feature row the model saw (per replan)
  context.parquet    S3 the context rows TabPFN / LightGBM read, with their source, per refit
  pred.parquet       S4 predicted ranges per person and horizon, box in world coordinates
  occupancy.npz      S6 occupancy grids per replan and horizon (foresight.core.occupancy)
  plan.parquet       S7 every candidate move with its veto reason, chosen flag
  timing.parquet     S8 ms per replan: features, predict (incl. refit), plan
  outcome.parquet    evaluation only: true position at t+h, inside-box flag, error. Never an input.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.core import features as F
from foresight.core import occupancy as O

CTX_COLS = ["t", "agent_id", "h", "episode", "t_known", "source"]


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def write(args, pred, seed, hist, plans, obstacles, meta, res, root="results/trace"):
    out = Path(root) / f"{args.sim}_{pred.kind}_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    dt = float(meta["dt"])
    tracks = pd.DataFrame([r for s in hist for r in s])
    tracks.to_parquet(out / "tracks.parquet")

    # S1: the same noisy observations and forward Kalman filter the features use (per-agent seeded noise)
    obs = []
    for aid, g in tracks.groupby("agent_id"):
        g = g.sort_values("t")
        steps = np.round(g.t.to_numpy() / dt).astype(int)
        S = steps.max() + 1
        P = np.full((S, 2), np.nan); P[steps] = g[["x", "y"]].to_numpy()
        noisy = P + np.random.default_rng([seed, int(aid)]).normal(0, args.noise, (S, 2))
        Ps, Vs = F._kalman_cv(noisy, dt, r=max(args.noise, 0.05))
        obs.append(pd.DataFrame({"t": steps * dt, "agent_id": aid, "noisy_x": noisy[steps, 0], "noisy_y": noisy[steps, 1],
                                 "filt_x": Ps[steps, 0], "filt_y": Ps[steps, 1], "filt_vx": Vs[steps, 0], "filt_vy": Vs[steps, 1]}))
    pd.concat(obs).to_parquet(out / "observed.parquet")

    rows, ctxs, preds, cands, timing = [], [], [], [], []
    for p in plans:
        t = p["t"]
        if p.get("q") is not None and len(p["q"]):
            rows.append(p["q"].assign(t_plan=t))
        if p.get("ctx") is not None:
            c = p["ctx"]
            ctxs.append(c[[k for k in CTX_COLS if k in c.columns]].assign(t_refit=t))
        pr = p["pred"]
        if pr is not None and len(pr):
            pr = pr.copy()
            for col in ("lo_a", "hi_a", "lo_c", "hi_c"):
                if col not in pr:
                    pr[col] = 0.0                                   # no range: belief = straight-line motion
            mid_a, mid_c = (pr.lo_a + pr.hi_a) / 2, (pr.lo_c + pr.hi_c) / 2
            c_, s_ = np.cos(pr.theta), np.sin(pr.theta)
            pr["box_cx"] = pr.px + pr.vx * pr.h + c_ * mid_a - s_ * mid_c
            pr["box_cy"] = pr.py + pr.vy * pr.h + s_ * mid_a + c_ * mid_c
            keep = ["t", "agent_id", "h", "px", "py", "vx", "vy", "theta", "lo_a", "hi_a", "lo_c", "hi_c", "box_cx", "box_cy"]
            preds.append(pr[keep].assign(t_plan=t))
        if p.get("candidates") is not None:
            cands.append(pd.DataFrame(p["candidates"]).assign(t=t, fallback=p.get("fallback", False)))
        if p.get("timing"):
            timing.append(dict(t=t, **p["timing"]))
    if rows:
        pd.concat(rows).to_parquet(out / "rows.parquet")
    if ctxs:
        pd.concat(ctxs).to_parquet(out / "context.parquet")
    if cands:
        pd.concat(cands).to_parquet(out / "plan.parquet")
    if timing:
        pd.DataFrame(timing).to_parquet(out / "timing.parquet")
    if preds:
        P = pd.concat(preds)
        P.to_parquet(out / "pred.parquet")
        # S6 occupancy, same function as the metric
        hz = sorted(P.h.unique())
        grids, times = [], []
        for t, g in P.groupby("t_plan"):
            grids.append([O.occupancy(g, h, meta["bounds"], box_q=args.box_q)[0].astype(np.float32) for h in hz])
            times.append(t)
        xs, ys, res_ = O.grid(meta["bounds"])
        np.savez_compressed(out / "occupancy.npz", P=np.array(grids), t=np.array(times), h=np.array(hz), xs=xs, ys=ys, res=res_)
        # evaluation only: where each person really was at t + h
        tr = tracks.set_index([np.round(tracks.t / dt).astype(int), "agent_id"])[["x", "y"]]
        k_f = np.round((P.t + P.h) / dt).astype(int)
        idx = list(zip(k_f, P.agent_id))
        true = tr.reindex(idx).to_numpy()
        lo = np.stack([P.lo_a, P.lo_c], 1); hi = np.stack([P.hi_a, P.hi_c], 1)
        c_, s_ = np.cos(P.theta.to_numpy()), np.sin(P.theta.to_numpy())
        dx = true[:, 0] - (P.px + P.vx * P.h).to_numpy(); dy = true[:, 1] - (P.py + P.vy * P.h).to_numpy()
        a, b = c_ * dx + s_ * dy, -s_ * dx + c_ * dy
        inside = (a >= lo[:, 0]) & (a <= hi[:, 0]) & (b >= lo[:, 1]) & (b <= hi[:, 1])
        err = np.hypot(true[:, 0] - P.box_cx.to_numpy(), true[:, 1] - P.box_cy.to_numpy())
        pd.DataFrame({"t": P.t.to_numpy(), "agent_id": P.agent_id.to_numpy(), "h": P.h.to_numpy(), "true_x": true[:, 0],
                      "true_y": true[:, 1], "inside_box": inside, "error": err}).to_parquet(out / "outcome.parquet")

    m = {"sim": args.sim, "planner": pred.kind, "seed": seed, "dt": dt, "bounds": meta["bounds"], "obstacles": obstacles,
         "horizons": list(args.hz), "box_q": args.box_q, "margin": args.margin, "max_speed": args.max_speed,
         "context": args.context, "refit": args.refit, "tabpfn_cfg": f"fast{args.tabpfn_fast}-est{args.tabpfn_est}",
         "switch": args.switch, "noise": args.noise, "commit": _commit(),
         "result": {k: (float(v) if isinstance(v, (np.floating, float, int, np.integer, bool, np.bool_)) else str(v))
                    for k, v in res.items()}}
    (out / "meta.json").write_text(json.dumps(m, indent=1, default=str))
    return out
