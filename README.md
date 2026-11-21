# Foresight

Behaviour prediction and safe planning for any robot, from a table. HackLab Berlin 2026, Prior Labs track ("TabPFN for spatial awareness").

Given the tracks of agents around a robot (people, cars), Foresight predicts where each agent will be in 1 s and 2 s as a range (10/50/90% quantiles) with [TabPFN](https://github.com/PriorLabs/TabPFN). The robot then picks the move that keeps clear of those ranges. The core is embodiment-agnostic: every simulator writes the same track format (`INTERFACE.md`), and features live in each agent's own frame.

Status: work in progress (hackathon, 21–22 Nov 2026).

## Layout

```
INTERFACE.md            simulator -> core data format (tracks.csv, obstacles.csv, meta.json)
foresight/core/
  schema.py             load + validate a dataset
  features.py           rows (agent, t, horizon): agent-frame features, residual targets, split
  context.py            per-query context selection: recency, kNN bank, stratified fill
  models.py             TabPFN and baselines behind one predict_quantiles()
  eval.py               error, coverage, band width, pinball loss
foresight/domains/
  toy_crowd.py          stand-in social-force crowd writing the interface format
run_offline.py          R1/R2 offline experiments
tools/check_tabpfn.py   TabPFN access and timing check
TEAM-PLAN.md            the plan: ladder, schedule, experiments
docs/                   notes on past TabPFN hackathon projects
```

## Run

```
pip install -r requirements.txt
export TABPFN_TOKEN=...            # from ux.priorlabs.ai after accepting the licence
python -m foresight.domains.toy_crowd --episodes 40 --out data/toy
python run_offline.py --data data/toy                   # baselines
python run_offline.py --data data/toy --tabpfn          # + TabPFN
```

## Method in brief

- **Row** = one agent, at time t, for horizon h ∈ {1 s, 2 s}. The horizon is a feature, and there is one model per axis.
- **Target** = deviation from constant velocity in the agent's frame. The model learns braking, turning and avoidance, not kinematics.
- **Inputs:**
  - observed positions with sensor noise, smoothed by a forward-only Kalman filter;
  - the agent's own motion over the last 1 s;
  - its 3 nearest neighbours;
  - the robot;
  - the nearest obstacle;
  - local density.
- **Split:** by episode. The context only holds rows whose target time has passed.

## Prepared before the event

Generic utilities only: a Kalman filter, constant-velocity baseline and timing code, adapted from a pre-event starter (`tools/check_tabpfn.py`). Everything else was written at the event.
