"""ETH / Hotel pedestrian data (ETH Walking Pedestrians, Pellegrini et al. 2009) -> INTERFACE.md.

Source files: OpenTraj (github.com/crowdbotp/OpenTraj, datasets/ETH/seq_<scene>/obsmat.txt).
obsmat.txt columns: frame, person_id, x, z, y, vx, vz, vy (metres, metres/second; z unused).
Annotated at 2.5 fps -> dt = 0.4 s. Each recording is cut into windows that become episodes.
No robot in these scenes, so robot features stay empty; obstacles are not parsed.

    python -m foresight.domains.eth_ucy --raw data/raw/eth --out data/eth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DT = 0.4


def convert(raw_dir, out_dir, window=60.0, scene=None):
    raw = Path(raw_dir)
    scene = scene or raw.name
    m = np.loadtxt(raw / "obsmat.txt")
    df = pd.DataFrame({"frame": m[:, 0].astype(int), "pid": m[:, 1].astype(int),
                       "x": m[:, 2], "y": m[:, 4], "vx": m[:, 5], "vy": m[:, 7]})
    fstep = int(np.median(np.diff(np.sort(df.frame.unique()))))      # 6 frames = 0.4 s
    df["k"] = (df.frame - df.frame.min()) // fstep
    per = int(round(window / DT))
    df["episode"] = df.k // per
    df["t"] = ((df.k % per) * DT).round(3)
    # a person crossing a window edge gets a new id per window (rows stay, history restarts)
    df["agent_id"] = df.pid
    df = df.drop_duplicates(["episode", "t", "agent_id"])
    out = pd.DataFrame({"episode": df.episode, "t": df.t, "agent_id": df.agent_id, "kind": "person",
                        "is_ego": 0, "x": df.x, "y": df.y, "vx": df.vx, "vy": df.vy, "radius": 0.25,
                        "regime": scene})
    out = out.sort_values(["episode", "t", "agent_id"])
    o = Path(out_dir); o.mkdir(parents=True, exist_ok=True)
    out.to_csv(o / "tracks.csv", index=False)
    pd.DataFrame(columns=["episode", "obstacle_id", "shape", "x", "y", "r", "w", "h", "angle"]).to_csv(
        o / "obstacles.csv", index=False)
    pad = 1.0
    meta = {"sim": f"eth-ucy:{scene}", "dt": DT,
            "bounds": [float(out.x.min() - pad), float(out.y.min() - pad), float(out.x.max() + pad), float(out.y.max() + pad)],
            "episodes": {str(int(e)): {"seed": None, "duration": window, "regime_switch_t": None,
                                       "regime_before": scene, "regime_after": None}
                         for e in sorted(out.episode.unique())},
            "notes": "Real pedestrians, ETH Walking Pedestrians dataset via OpenTraj; no robot, no obstacles."}
    (o / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{scene}: {out.agent_id.nunique()} people, {out.episode.nunique()} episodes, {len(out)} rows -> {o}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=float, default=60.0)
    a = ap.parse_args()
    convert(a.raw, a.out, a.window)


if __name__ == "__main__":
    main()
