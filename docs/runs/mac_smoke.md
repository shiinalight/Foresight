# Mac smoke test

- Machine: Apple M3 Pro, 18 GB, macOS (arm64)
- torch 2.15.0, MPS available: True, tabpfn 10.2.0; Python 3.11.15
- Code: `darshak/core` at fce98a6
- Setup: data ALL MATCH (eth, hotel, crowd_cross and filter settings); tests 126 passed, 1 xfailed

## Two problems found in the kit

1. **`data/crowd` is missing after `setup.sh`.** `tools/check_mps.py` and `tools/latency.py` read the plaza bank `data/crowd`, but `setup.sh` only generates `data/crowd_cross`. Worked around without code changes: `python -m foresight.domains.crowd --episodes 80 --seed 0 --switch-prob 0.5 --out data/crowd`, which matches `docs/DATA.md` (80 episodes, 36 switch, 1,280 person tracks, mean speed 1.05 m/s). Suggest adding this line to `setup.sh` step 5.
2. **`run_closed.py` segfaults when it reaches the `lgbm_all` planner** (Segmentation fault: 11, right after `cv` and `vo` finished on seed 9000). Likely LightGBM and PyTorch loading two OpenMP runtimes on macOS. The TabPFN planners run fine without it (below). LightGBM in-loop timing on the Mac is therefore not measured.

Also: the first local TabPFN use needs the licence accepted for the account (https://ux.priorlabs.ai/accept-license?hf_repo_id=tabpfn_3_5); worth a line in `docs/MAC_SETUP.md` step 2.

## GPU check (tools/check_mps.py)

```
cpu: err 0.6881 m  joint 5-95 84.0%  (fit 29.99 s, predict 3.89 s)
mps: err 0.6882 m  joint 5-95 84.0%  (fit 2.51 s, predict 6.57 s)
largest |cpu - mps| over all quantiles: 0.0127 m
```

## TabPFN timing (tools/latency.py), device mps

```
3.5, 1 est (real-time tier, D-5)   cached  context   500  fit  2.98 s  predict 32 rows  0.17 s
3.5, 1 est (real-time tier, D-5)   cached  context   500  fit  2.98 s  predict 64 rows  0.18 s
3.5, 1 est (real-time tier, D-5)   cached  context  1000  fit  2.21 s  predict 32 rows  0.17 s
3.5, 1 est (real-time tier, D-5)   cached  context  1000  fit  2.21 s  predict 64 rows  0.18 s
3.5-Fast, 1 est                    cached  context   500  fit 20.18 s  predict 32 rows  0.09 s   (first fit includes the weight download)
3.5-Fast, 1 est                    cached  context   500  fit 20.18 s  predict 64 rows  0.09 s
3.5-Fast, 1 est                    cached  context  1000  fit  0.97 s  predict 32 rows  0.09 s
3.5-Fast, 1 est                    cached  context  1000  fit  0.97 s  predict 64 rows  0.09 s
3.5, default (8 est)               cached  context   500  fit  5.75 s  predict 32 rows  1.31 s
3.5, default (8 est)               cached  context   500  fit  5.75 s  predict 64 rows  1.40 s
3.5, default (8 est)               cached  context  1000  fit  7.06 s  predict 32 rows  1.28 s
3.5, default (8 est)               cached  context  1000  fit  7.06 s  predict 64 rows  1.37 s
```

## Robot runs (crossing crowd, 20 s, box-q 0.10, --switch)

| planner | seed | reached | min gap (m) | collisions | ms per plan | wall (s) |
|---|---|---|---|---|---|---|
| cv | 9000 | no | 0.46 | 0 | 13 | 2 |
| vo | 9000 | yes (19.4 s) | 0.49 | 0 | 5 | 1 |
| lgbm_all | 9000 | — | — | — | segfault | — |
| tabpfn | 9000 | no | 0.71 | 0 | 135 | 8 |
| tabpfn | 9001 | no | 0.38 | 0 | 130 | 8 |
| tabpfn_live | 9000 | no | 0.70 | 0 | 321 | 17 |
| tabpfn_live | 9001 | no | 0.44 | 0 | 287 | 16 |
| oracle | 9000 | yes (19.4 s) | 0.25 | 0 | 14 | 2 |
| oracle | 9001 | yes (20.0 s) | 0.32 | 0 | 17 | 2 |

TabPFN rows were run separately (`--planners tabpfn tabpfn_live oracle`, same arguments, tag macsmoke2) after the segfault.

Main laptop for comparison (x64 emulation, CPU): frozen TabPFN ~200–500 ms per plan, live TabPFN ~1,600–1,700 ms, LightGBM ~10–300 ms. Robot replan interval: 400 ms.
