"""T3 test set: crowd episodes that switch from one regime to a regime the bank never saw.

    python tools/make_switch_set.py --before calm --after erratic --episodes 12 --out data/crowd_c2e

Uses foresight.domains.crowd (simulate, save) unchanged. Switch at mid-episode. Seeds 5000+ (never used elsewhere).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight.domains.crowd import CrowdConfig, save, simulate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="calm")
    ap.add_argument("--after", default="erratic")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--out", default="data/crowd_c2e")
    a = ap.parse_args()
    cfg = CrowdConfig(duration=a.duration, regime=a.before, switch_t=a.duration / 2, switch_to=a.after)
    tracks, obst, eps = [], [], {}
    for e in range(a.episodes):
        tr, ob, me = simulate(a.seed0 + e, cfg, episode=e)
        tracks.append(tr); obst.append(ob); eps[str(e)] = me
    L = cfg.half_size
    meta = dict(sim="crowd-v1", dt=cfg.dt, bounds=[-L, -L, L, L], episodes=eps,
                notes=f"T3 test set: {a.before} -> {a.after} at {a.duration / 2:g} s, seeds {a.seed0}+")
    save(a.out, pd.concat(tracks, ignore_index=True), pd.concat(obst, ignore_index=True), meta)
    print(f"wrote {a.episodes} episodes to {a.out}")


if __name__ == "__main__":
    main()
