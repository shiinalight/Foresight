# Scenarios and data plan

Every scenario only has to write tracks in the `INTERFACE.md` format: `episode, t, agent_id, kind, is_ego, x, y, vx, vy, radius, …`. The model, planner, evaluation and drawing code stay the same. So any kind of moving agent works. What a new scenario costs is writing each agent kind's behaviour (how it moves and reacts), not the format.

## What exists

| Scenario | Simulator | Status |
|---|---|---|
| Plaza: people + one robot crossing | `foresight/domains/crowd.py` (Nooshin): social force with a forward field of view, calm / hurried / erratic regimes, mid-episode switch, pillars, `reset()`/`step(robot_cmd)`, viewer `tools/view_crowd.py`, 14 tests | **Done.** Main domain 1 |
| Real pedestrians (ETH, Hotel) | `foresight/domains/eth_ucy.py` | **Done.** Offline only (no robot) |
| Stand-in plaza | `foresight/domains/toy_crowd.py` | Superseded by `crowd.py`; kept for tests |

## Recommended: one new scenario, the mixed street

A delivery robot crosses a street:
- **Cars** drive in lanes and brake for whoever is on the crossing.
- **Pedestrians** walk on the pavements and sometimes cross.

**Why this one:**
- **Several agent kinds in one picture:** the strongest visual for "one layer, any agent" (the judge asked for something visual).
- **Same planner, unchanged:** it reuses `run_closed.py` and the planner as they are.
- **It replaces the highway-env second domain.** That adapter was never written, and this scene covers cars too.

**What to build** (an extension of `crowd.py`, about 2 to 3 hours):

| Element | Behaviour |
|---|---|
| Road | 2 lanes (one each way), about 7 m wide, across the middle; pavements either side |
| Cars (`kind=car`, `length`, `width`) | Follow their lane at a desired speed (IDM car following); brake for anything on the crossing or in their lane ahead; 3 to 8 cars, re-spawn at the far end |
| Pedestrians (`kind=person`) | Social force as now, kept on the pavements. Some cross at the zebra, a few jaywalk (regime-dependent) |
| Robot (`is_ego=1`) | Starts on one pavement, goal on the other; the planner drives it |
| Regimes | calm (cars slow, yield early), busy (fast cars, short gaps, jaywalkers) |
| Map layer (new, drawing only) | `map.json`: road polygon, pavements, crossing. Read by the drawing code only, never a model input |

**What the core needs from it:**
- the three interface files plus `map.json`;
- `reset()`/`step(robot_cmd)` as in `crowd.py`;
- `info["collided"]`;
- a car's footprint as `length`/`width`.

The features already describe neighbours by relative position and velocity. A `kind` feature (person, car) is the one small addition to `features.py`.

## Pitch only, no data (one sentence each)

- **Warehouse:** hundreds of floor robots on aisle grids. The same layer predicts each robot's next 2 s, with lanes as adapter features.
- **Drone swarms:** the same rows with z added, or flattened to 2D. The planner checks 3D boxes instead of 2D ones.

## How each agent kind is modelled, and where the data comes from

| Agent | Model | Source |
|---|---|---|
| Pedestrians (simulated) | Social force (Helbing and Molnár 1995): pulled towards a goal, pushed away from other people, the robot, pillars and walls. Field of view, stops, sharp turns, crowd-wide regimes | Our own numpy code (`crowd.py`); no simulator library |
| Pedestrians (real) | No model: annotated positions from overhead video | ETH Walking Pedestrians dataset (ETH Zurich 2009), via the OpenTraj collection |
| Robot | A disc moving at the commanded velocity (holonomic) | Our code; the planner sets the command |
| Cars (planned, mixed street) | Lane following with the Intelligent Driver Model (car following, braking) | Our code, the same model highway-env uses; highway-env itself is installed (open source, Farama) |

**Open tools in the stack:**
- **Data and models:** numpy, pandas, scikit-learn, LightGBM, TabPFN (Prior Labs).
- **Figures:** matplotlib.
- **Deck:** python-pptx.
- **Installed:** highway-env.

**Other open simulators worth knowing.** Each would cost setup time we don't have this weekend.
- **pysocialforce / PedSim:** pedestrian crowds.
- **SUMO:** city traffic with cars and pedestrians at crossings. It could generate the mixed street at scale.
- **CARLA:** 3D driving. It needs a big GPU.

## Later: a 3D scenario viewer

A small web viewer (three.js) could replay `tracks.csv` with simple 3D objects (people, cars, robots, drones), draw the predicted boxes and the robot's chosen path, and show a side panel of live numbers (coverage, error, gaps). It would work like the 3D agent dashboard built for DeckHQ. The data contract already supports it: tracks, a map layer and per-step predictions. Not for this weekend.

## How much data

Rows are cheap. 40 stand-in episodes already gave 97,000 rows, and the model sees at most 1,500 to 20,000 at a time. So we don't need 300+ episodes.

| Dataset | Episodes | Use |
|---|---|---|
| `data/crowd`: plaza, half the episodes with a regime switch | 80 (seeds 0–79) | Context bank, R1/R2 offline |
| `data/crowd_test`: plaza | 20 (seeds 1000–1019) | Held-out offline tests and closed-loop seeds |
| `data/street` | 60 + 20 held out | Same, second domain |
| `data/eth`, `data/hotel` | as is | Real-data check |

## Visuals each scenario must produce (minimal text, explanations in `RESULTS.md`)

1. **Top-down tracking view:** agents, last-2 s trails, predicted boxes, actual positions (`demo.py`).
2. **Side by side:** the same seed, constant-velocity robot against TabPFN robot (`render_closed.py`, next).
3. **Real video beside the top-down view:**
   - **Source:** the Hotel recording has its original video (`video.avi`, 50 MB, OpenTraj) and the homography matrix (`H.txt`, already downloaded).
   - **What it gives:** predicted boxes projected onto the real footage. That's the strongest "it works on reality" picture.
   - **Status:** the download needs approval.
