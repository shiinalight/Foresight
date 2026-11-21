# Foresight

Behaviour prediction and safe planning for robots among people, from a table. HackLab Berlin 2026, Prior Labs track ("TabPFN for spatial awareness").

From the tracks of the people around a robot, Foresight predicts where each person will be in 1–2 s as a 2D box (quantiles from [TabPFN](https://github.com/PriorLabs/TabPFN)). The robot then picks the move that keeps clear of those boxes. The core works for any kind of agent: every simulator or dataset writes the same track format (`INTERFACE.md`), and features are expressed in each agent's own frame.

**The claim we test:** a forecaster that gives a robot honest uncertainty in a place it has never seen, and corrects itself from the last seconds of that place, with no training.

**Positioning:** calibrated ranges are for efficiency; a worst-case floor is for safety. No statistical safety guarantee is claimed (see `RESULTS.md`, deployment notes).

## Start here

1. **This README:** what it is, how to run it.
2. **`PLAN.md`:** the living plan: claims and their status, framework, timeline, document map. Then `docs/CTO-REVIEW.md` for the reasoning.
3. **`docs/GATES.md`:** stage gates, the fair-comparison protocol, the ordered work list.
4. **`RESULTS.md`:** every number so far, with limits, and the pre-registered rule for the closed-loop results.
5. **`TEAM-PLAN.md`:** the original plan and schedule. Also see `SCENARIOS.md` for scenes and the data handover.

Branches: `main` holds settled, reviewed work; `darshak/core` holds work in progress.

## Layout

```
INTERFACE.md              data format every simulator writes (tracks.csv, obstacles.csv, meta.json)
foresight/core/
  schema.py               load + validate a dataset
  features.py             rows (agent, t, horizon): agent-frame features, residual targets, live queries
  context.py              which rows TabPFN sees: this scene's recent rows, similar rows, stratified fill
  models.py               TabPFN (local or hosted API) and baselines behind one predict_quantiles()
  conformal.py            calibration of the 2D box: on arrival and online in the scene
  planner.py              predict-then-plan planner, velocity-obstacle baseline
  eval.py                 error, coverage, width, pinball
foresight/domains/
  crowd.py                social-force crowd with a robot and behaviour regimes (Nooshin)
  eth_ucy.py              real pedestrians (ETH, Hotel) to the interface format
  toy_crowd.py            early stand-in crowd, kept for tests
evaluate.py               offline table on real data (both directions, all models)
run_adapt.py              adaptation: context strategies after arriving or after a behaviour change
run_closed.py             closed loop: robot crossing the crowd, every planner, per-seed checkpoint
demo.py, compare.py,      figures and videos
render_closed.py
tools/                    conformal_eval, incontext_eval, failure_modes, adapt_ci, latency, checks,
                          build_deck (deck/Foresight.pptx), overnight.ps1
tests/                    simulator, train/serve parity, planners
media/                    figures and clips used in RESULTS.md and the deck
```

## Run

```
pip install -r requirements.txt
set TABPFN_TOKEN=...                      # ux.priorlabs.ai, after accepting the licence
python -m foresight.domains.crowd --episodes 80 --out data/crowd
python -m foresight.domains.eth_ucy --raw data/raw/eth --out data/eth       # raw files from OpenTraj
python -m foresight.domains.eth_ucy --raw data/raw/hotel --out data/hotel
python evaluate.py                                                          # offline table
python tools/conformal_eval.py --train data/hotel --test data/eth --api      # calibration (hosted TabPFN)
python run_closed.py --planners cv tabpfn tabpfn_live --seeds 1000 1001 --switch
python -m pytest -q
```

## Method in brief

- **Row** = one agent, at time t, for horizon h. The horizon is a feature, and there is one model per axis.
- **Target** = deviation from constant velocity, in the agent's own frame.
- **Inputs:**
  - observed positions, smoothed by a forward-only Kalman filter tuned per dataset (`<dataset>/kalman.json`). In the simulator, 0.1 m sensor noise is added first; real data uses the annotations as observed;
  - the agent's own last 1 s;
  - its 3 nearest neighbours;
  - the robot;
  - the nearest obstacle;
  - local density.
- **Train and serve use the same features:** a live query and an offline row give identical features (`tests/test_parity.py`).
- **Splits:**
  - offline, by episode or by scene;
  - live context rows are used only once their outcome is known.
- **Planner:** candidate moves are vetoed if they come too close to any person's predicted box at +1 and +2 s. Among the rest it picks the move with the most progress. The box quantile is the caution dial.

## Prepared before the event

Generic utilities only: a Kalman filter, a constant-velocity baseline and timing code (`tools/check_tabpfn.py`). Everything else was written at the event.
