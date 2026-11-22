"""Export one trace folder to a single JSON file for the three.js viewer (docs/archive/VISUALS.md).

    python tools/export_3d.py results/trace/crowd_lgbm_all_1001 media/sample_3d.json

Reads only the trace (never recomputes). Coordinates are metres, z = 0 is the floor, time in seconds.
Schema (version 1):
{
  "version": 1, "meta": {...trace meta.json...},
  "bounds": [x0, y0, x1, y1], "obstacles": [{"x", "y", "r"}], "dt": 0.2,
  "frames": [ { "t": 0.0,
                "agents": [ {"id", "kind": "person"|"robot", "x", "y", "vx", "vy", "r", "collided"} ],
                "boxes":  [ {"id", "h", "corners": [[x, y] x4], "centre": [x, y]} ],     // the planner's own belief
                "plan":   {"chosen": [vx, vy], "fallback": bool,
                           "candidates": [ {"v": [vx, vy], "veto": "none"|"box"|"short check"|"bounds"|"obstacle"} ]},
                "occupancy": {"h": 2.0, "x0", "y0", "res", "nx", "ny", "p": [row-major floats, 0..1]} | null
              } ],
  "outcome": {"inside_box_rate": float, "result": {...episode metrics...}}
}
Boxes, plans and occupancy exist only on replan frames (every 0.4 s); the viewer holds the last one.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def corners(r):
    pts = [(r.lo_a, r.lo_c), (r.hi_a, r.lo_c), (r.hi_a, r.hi_c), (r.lo_a, r.hi_c)]
    c, s = np.cos(r.theta), np.sin(r.theta)
    cx, cy = r.px + r.vx * r.h, r.py + r.vy * r.h
    return [[round(cx + c * a - s * b, 3), round(cy + s * a + c * b, 3)] for a, b in pts]


def main(src, dst, occ_stride=2):
    d = Path(src)
    meta = json.loads((d / "meta.json").read_text())
    tracks = pd.read_parquet(d / "tracks.parquet")
    pred = pd.read_parquet(d / "pred.parquet") if (d / "pred.parquet").exists() else None
    plan = pd.read_parquet(d / "plan.parquet") if (d / "plan.parquet").exists() else None
    occ = np.load(d / "occupancy.npz") if (d / "occupancy.npz").exists() else None
    outcome = pd.read_parquet(d / "outcome.parquet") if (d / "outcome.parquet").exists() else None
    frames = []
    for t, g in tracks.groupby("t"):
        f = {"t": round(float(t), 3),
             "agents": [{"id": int(r.agent_id), "kind": "robot" if r.is_ego else "person", "x": round(r.x, 3),
                         "y": round(r.y, 3), "vx": round(r.vx, 3), "vy": round(r.vy, 3), "r": float(r.radius),
                         "collided": bool(getattr(r, "collided", 0))} for r in g.itertuples()],
             "boxes": [], "plan": None, "occupancy": None}
        if pred is not None:
            p = pred[np.isclose(pred.t_plan, t)]
            f["boxes"] = [{"id": int(r.agent_id), "h": float(r.h), "corners": corners(r),
                           "centre": [round(r.box_cx, 3), round(r.box_cy, 3)]} for r in p.itertuples()]
        if plan is not None:
            c = plan[np.isclose(plan.t, t)]
            if len(c):
                ch = c[c.chosen]
                f["plan"] = {"chosen": [round(float(ch.vx.iloc[0]), 3), round(float(ch.vy.iloc[0]), 3)] if len(ch) else None,
                             "fallback": bool(c.fallback.iloc[0]),
                             "candidates": [{"v": [round(r.vx, 3), round(r.vy, 3)], "veto": r.veto_reason} for r in c.itertuples()]}
        if occ is not None:
            k = np.flatnonzero(np.isclose(occ["t"], t))
            if len(k):
                hi = int(np.argmax(occ["h"]))
                P = occ["P"][k[0], hi][::occ_stride, ::occ_stride]
                f["occupancy"] = {"h": float(occ["h"][hi]), "x0": float(occ["xs"][0]), "y0": float(occ["ys"][0]),
                                  "res": float(occ["res"]) * occ_stride, "nx": P.shape[1], "ny": P.shape[0],
                                  "p": [round(float(v), 3) for v in P.ravel()]}
        frames.append(f)
    out = {"version": 1, "meta": meta, "bounds": meta["bounds"], "obstacles": meta["obstacles"], "dt": meta["dt"],
           "frames": frames,
           "outcome": {"inside_box_rate": float(outcome.inside_box.mean()) if outcome is not None else None,
                       "result": meta.get("result", {})}}
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {dst}: {len(frames)} frames, {Path(dst).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:3])
