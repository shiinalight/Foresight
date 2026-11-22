"""T5 extension to seeds 3020-3039 (addendum part 5 allowance: spare CPU), same arms and dials as the 20-seed set.

    python tools/t5_extend.py [--dry-run]

Per arm, only the seeds 3000-3039 not yet in results/closed/*-t5-<tag>.csv run (so no (arm, seed) row repeats),
written under tag <tag>-ext. Arms: tabpfn_split q0.15 and q0.10, frozen tabpfn q0.15, lgbm_all q0.15 (batch offset),
vo w 0.5/1/2, open-loop oracle m 0.2/0.4/0.6. Detached processes; logs in results/overnight/<tag>-ext.log.
"""
import argparse
import glob
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = r"C:\Users\samco\tools\lerobot-test\.venv-x64\Scripts\python.exe"
BASE = ("run_closed.py --sim crowd --robot-repulsion 2.0 --ignore-robot 0.2 --people 24 --goal-mode cross "
        "--bank data/crowd_cross_t2 --bank-exclude-regime erratic --switch --switch-all --switch-to erratic --switch-t 5 "
        "--frozen-ctx fill --context 2000")
OFF = "--offset --offset-rule batch --offset-eta-b 0.05"
GROUPS = [  # (process name, threads, [(tag, args)])
    ("split-q15", 4, [("split-q15", f"--planners tabpfn_split --box-q 0.15 {OFF}")]),
    ("split-q10", 4, [("split-q10", f"--planners tabpfn_split --box-q 0.10 {OFF}")]),
    ("tabpfn-q15", 3, [("tabpfn-q15", f"--planners tabpfn --box-q 0.15 {OFF}")]),
    ("lgbmall-q15", 2, [("lgbmall-q15", f"--planners lgbm_all --box-q 0.15 {OFF}")]),
    ("vo", 1, [("vo05", "--planners vo --vo-w 0.5"), ("vo10", "--planners vo --vo-w 1.0"), ("vo20", "--planners vo --vo-w 2.0")]),
    ("oracle", 1, [("orm02", "--planners oracle --margin 0.2"), ("orm04", "--planners oracle --margin 0.4"),
                   ("orm06", "--planners oracle --margin 0.6")]),
]


def done_seeds(tag):
    fs = [f for f in glob.glob(str(ROOT / f"results/closed/*-t5-{tag}.csv")) if "robotfeat" not in f]
    return set(pd.concat([pd.read_csv(f) for f in fs]).seed) if fs else set()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    for name, threads, runs in GROUPS:
        cmds = []
        for tag, args in runs:
            miss = [s for s in range(3000, 3040) if s not in done_seeds(tag)]
            if miss:
                cmds.append(f"& '{PY}' {BASE} --threads {threads} --seeds {' '.join(map(str, miss))} {args} --tag t5-{tag}-ext")
                print(f"{tag}: {len(miss)} seeds to run ({miss[0]}-{miss[-1]})")
        if not cmds:
            continue
        log = f"results\\overnight\\t5-{name}-ext.log"
        if not a.dry_run:
            ps1 = ROOT / "results" / "overnight" / f"t5-{name}-ext.ps1"     # one launcher script per process
            lines = ["$env:POLARS_SKIP_CPU_CHECK='1'; $env:PYTHONIOENCODING='utf-8'; $env:PYTHONUNBUFFERED='1'",
                     f"Set-Location '{ROOT}'"] + [f"{c} *>> '{log}'" for c in cmds]
            ps1.write_text("\n".join(lines) + "\n", encoding="utf-8")
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"Start-Process powershell -ArgumentList '-NoProfile','-File','{ps1}' -WindowStyle Hidden"],
                           check=True)
            print(f"started {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
