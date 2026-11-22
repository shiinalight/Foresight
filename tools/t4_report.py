"""T4 gates, applied as written in RESULTS.md (amendment 2, T4 addendum parts 1 and 2).

    python tools/t4_report.py

Offset (results/closed/*t4off-{off,row,batch}.csv, seeds 5100-5105): per setting and planner, means of
inloop_cov90d_post, seconds_stopped, freeze_rate, offset_final, pred_ms_per_plan; the gate on tabpfn_split.
Smoothing (results/closed/*t4sm-{off,on}.csv, seeds 5110-5119): paired means and the D-28 gate per learned planner.
Only seeds present in every compared run are used.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEARNED = ("tabpfn_split", "tabpfn", "lgbm_all")


def load(tag):
    fs = [f for f in sorted((ROOT / "results/closed").glob(f"*-{tag}.csv")) if not f.name.startswith("robotfeat_")]
    return pd.concat([pd.read_csv(f).assign(setting=tag.split("-")[-1]) for f in fs], ignore_index=True) if fs else None


def offset():
    parts = [d for d in (load("t4off-off"), load("t4off-row"), load("t4off-batch")) if d is not None]
    if not parts:
        print("offset: no results yet"); return
    d = pd.concat(parts, ignore_index=True)
    common = set.intersection(*[set(g.seed) for _, g in d.groupby(["setting", "planner"])])
    d = d[d.seed.isin(common)]
    cols = ["inloop_cov90d_post", "inloop_cov90_post", "inloop_w90_post", "inloop_is90_post", "inloop_served_cov_post",
            "seconds_stopped", "freeze_rate", "time_capped", "moving_collision_steps", "offset_final", "offset_mean",
            "offset_miss", "pred_ms_per_plan"]
    T = d.groupby(["planner", "setting"])[cols].mean().round(3)
    pd.set_option("display.width", 250)
    print(f"OFFSET TEST ({len(common)} common seeds: {sorted(common)})\n{T.to_string()}\n")
    s = T.loc["tabpfn_split"] if "tabpfn_split" in T.index.get_level_values(0) else None
    if s is None or "off" not in s.index:
        print("offset gate: tabpfn_split OFF missing"); return
    off = s.loc["off"]
    cands = []
    for r in ("row", "batch"):
        if r not in s.index:
            continue
        x = s.loc[r]
        stable = (x.seconds_stopped <= off.seconds_stopped + 0.4) and (x.freeze_rate <= off.freeze_rate + 0.02)
        dist = abs(x.inloop_cov90d_post - 0.80)
        print(f"  rule {r:<5}: coverage after switch {100 * x.inloop_cov90d_post:.1f}% (|-80| {100 * dist:.1f}), "
              f"stopped {x.seconds_stopped:.2f} s vs OFF {off.seconds_stopped:.2f}, freeze {x.freeze_rate:.3f} vs "
              f"{off.freeze_rate:.3f} -> {'stable' if stable else 'UNSTABLE'}")
        cands.append((r, stable, dist))
    d_off = abs(off.inloop_cov90d_post - 0.80)
    print(f"  OFF        : coverage after switch {100 * off.inloop_cov90d_post:.1f}% (|-80| {100 * d_off:.1f})")
    st = [c for c in cands if c[1]]
    if st:
        if len(st) == 2 and abs(st[0][2] - st[1][2]) <= 0.01:
            best = [c for c in st if c[0] == "batch"][0]
        else:
            best = min(st, key=lambda c: c[2])
        on = best[2] < d_off
        print(f"OFFSET GATE: {'ON, rule ' + best[0] if on else 'OFF (no stable rule closer to 80% than OFF)'}")
    else:
        sec = min(cands, key=lambda c: c[2])[0] if cands else "-"
        print(f"OFFSET GATE: OFF in the primary (no stable rule); secondary arm uses rule {sec}")
    ms = d[d.setting == "off"].groupby("planner").pred_ms_per_plan.mean()
    print("\nms per plan, T5 config (offset scored only):\n" + ms.round(0).to_string())


def smoothing():
    parts = [x for x in (load("t4sm-off"), load("t4sm-on")) if x is not None]
    if len(parts) < 2:
        print("\nsmoothing: results incomplete"); return
    d = pd.concat(parts, ignore_index=True)
    common = set.intersection(*[set(g.seed) for _, g in d.groupby(["setting", "planner"])])
    d = d[d.seed.isin(common)]
    cols = ["heading_change_per_m", "reversals_45", "time_capped", "intrusion_s", "moving_collision_steps", "min_gap"]
    T = d.groupby(["planner", "setting"])[cols].mean()
    T["mc_rate"] = d.assign(mc=d.moving_collision_steps > 0).groupby(["planner", "setting"]).mc.mean()
    print(f"\nSMOOTHING TEST ({len(common)} common seeds)\n{T.round(3).to_string()}\n")
    ok_all = True
    for p in LEARNED:
        if p not in T.index.get_level_values(0):
            ok_all = False; print(f"  {p}: missing"); continue
        a, b = T.loc[(p, "off")], T.loc[(p, "on")]
        drop = 1 - b.heading_change_per_m / a.heading_change_per_m
        ok = drop >= 0.30 and b.time_capped - a.time_capped <= 2.0 and b.intrusion_s - a.intrusion_s <= 1.0 \
            and b.mc_rate - a.mc_rate <= 0.1 + 1e-9
        ok_all &= ok
        print(f"  {p:<13} heading change -{100 * drop:.0f}%  time {b.time_capped - a.time_capped:+.2f} s  "
              f"intrusion {b.intrusion_s - a.intrusion_s:+.2f} s  collision rate {b.mc_rate - a.mc_rate:+.2f} -> {'pass' if ok else 'FAIL'}")
    print(f"SMOOTHING GATE: {'ON' if ok_all else 'OFF'}")


if __name__ == "__main__":
    offset()
    smoothing()
