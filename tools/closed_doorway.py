"""Acceptance check 3 for the doorway scene with Darshak's closed loop (run_closed.py, darshak/core).

run_closed.py only knows --sim crowd/toy, so this wrapper swaps in the doorway simulator for
make_sim and then runs run_closed.main() unchanged. Run it from a checkout that has run_closed.py:

    python tools/closed_doorway.py --planners cv --seeds 1000 1001 --bank data/doorway

Same seed = same people as data/doorway_test; odd seeds switch to rush. The robot reaches its goal
in about 18 s, before any recorded switch (21-36 s), so here the rush starts at --rush-at (default
6 s): the crowd reaches the door just as the robot does. --rush-at -1 keeps the recorded times. The robot's goal
is its current waypoint (door line-up, through the door, final goal), so the planner's straight-to-
goal candidates find the door; the episode ends when the robot reaches its final goal.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import run_closed as RC  # noqa: E402

from foresight.domains.doorway import DoorwayConfig, DoorwaySim, episode_config  # noqa: E402


RUSH_AT = 6.0


def make_sim(name, seed, args):
    cfg = episode_config(seed, DoorwayConfig(duration=args.duration))
    if cfg.switch_t is not None and RUSH_AT >= 0:
        cfg = replace(cfg, switch_t=RUSH_AT)
    return DoorwaySim(seed, cfg), cfg.dt, cfg.half_size


if __name__ == "__main__":
    if "--rush-at" in sys.argv:
        i = sys.argv.index("--rush-at")
        RUSH_AT = float(sys.argv[i + 1]); del sys.argv[i:i + 2]
    RC.make_sim = make_sim
    if "--sim" not in sys.argv:
        sys.argv += ["--sim", "crowd"]          # run_closed's argparse only allows crowd/toy; the tag in its csv
    if "--duration" not in sys.argv:
        sys.argv += ["--duration", "60"]
    if "--tag" not in sys.argv:
        sys.argv += ["--tag", "doorway"]
    RC.main()
