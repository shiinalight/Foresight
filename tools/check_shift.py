"""B7: are the robot features of live closed-loop queries distributed like the bank's?

The bank was generated with the simulator's straight-line robot; in the loop the planner drives
the robot. If robot_dist / robot_closing differ a lot, regenerate the bank with a randomised robot.

    python tools/check_shift.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_offline import rows_for  # noqa: E402

bank = rows_for("data/crowd", 0.1, horizons=(1.0, 2.0))
live = pd.concat(pd.read_csv(f) for f in Path("results/closed").glob("robotfeat_*.csv"))
qs = [0.05, 0.25, 0.5, 0.75, 0.95]
for col in ("robot_dist", "robot_closing"):
    b, l = bank[col].quantile(qs).round(2).tolist(), live[col].quantile(qs).round(2).tolist()
    print(f"{col:<14} bank {b}\n{'':<14} live {l}")
near_b, near_l = (bank.robot_dist < 2).mean(), (live.robot_dist < 2).mean()
print(f"share of rows with the robot within 2 m: bank {100 * near_b:.1f}%  live {100 * near_l:.1f}%")
