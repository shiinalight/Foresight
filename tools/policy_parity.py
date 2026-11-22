"""TabPFN policy vs its teacher on the 20 test crowds (docs/TABPFN_POLICY.md pass rule, docs/runs/policy_pilot_rule.md).

    python tools/policy_parity.py        -> results/policy_parity.csv

Policy rows: results/closed/*pol-pilot.csv (3400-3404) and *pol-test-g*.csv (3405-3419); teacher rows:
*pol-teachtest-g*.csv (3400-3419); both on the L4 pod. Paired by seed. Pass rule on the paired mean differences
(policy - teacher): goal-reached rate >= -0.10, time_capped <= +2.6 s, min_gap >= -0.158 m, moving-collision rate
<= +0.244; 95% paired-seed bootstrap intervals are reported next to each. Also: policy_agree and the hard-check
override rate.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(pattern):
    fs = [f for f in glob.glob(str(ROOT / "results/closed" / pattern)) if "robotfeat" not in f]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates("seed").set_index("seed").sort_index()


def main():
    pol = pd.concat([load("*pol-pilot.csv"), load("*pol-test-g*.csv")]).sort_index()
    tea = load("*pol-teachtest-g*.csv")
    seeds = sorted(set(pol.index) & set(tea.index) & set(range(3400, 3420)))
    pol, tea = pol.loc[seeds], tea.loc[seeds]
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(seeds), len(seeds)) for _ in range(5000)]
    rules = [("reached", "goal-reached rate", lambda d: d >= -0.10), ("time_capped", "time (s)", lambda d: d <= 2.6),
             ("min_gap", "min gap (m)", lambda d: d >= -0.158), ("mc", "moving-collision rate", lambda d: d <= 0.244)]
    pol = pol.assign(mc=(pol.moving_collision_steps > 0).astype(float), reached=pol.reached.astype(float))
    tea = tea.assign(mc=(tea.moving_collision_steps > 0).astype(float), reached=tea.reached.astype(float))
    rows, ok_all = [], True
    for col, name, rule in rules:
        D = (pol[col] - tea[col]).to_numpy(float)
        bs = [D[i].mean() for i in idx]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ok = bool(rule(D.mean())); ok_all &= ok
        rows.append(dict(metric=name, policy=pol[col].mean(), teacher=tea[col].mean(), diff=D.mean(), lo=lo, hi=hi, passes=ok))
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"{len(seeds)} paired test crowds ({seeds[0]}-{seeds[-1]})")
    print(R.round(3).to_string(index=False))
    print(f"policy picks the teacher's move at {100 * pol.policy_agree.mean():.1f}% of replans (per crowd "
          f"{100 * pol.policy_agree.min():.0f}-{100 * pol.policy_agree.max():.0f}%); hard checks override its first "
          f"choice at {100 * pol.policy_override_rate.mean():.2f}% of replans")
    print(f"ms per plan: policy {pol.pred_ms_per_plan.mean():.0f}, teacher {tea.pred_ms_per_plan.mean():.0f}")
    print("VERDICT:", "TabPFN can drive the loop at the teacher's level (all four)" if ok_all else
          "tried; fails on " + ", ".join(R[~R.passes].metric))
    R.to_csv(ROOT / "results/policy_parity.csv", index=False)


if __name__ == "__main__":
    main()
