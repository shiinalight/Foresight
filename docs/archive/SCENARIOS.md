# Scenarios: catalogue and data handover

For Nooshin, who owns the data side. This file says which scenes to build, what makes each one a good test, and exactly what to deliver. The core (features, TabPFN, planner, evaluation, drawing) reads only the `INTERFACE.md` files, so every scene plugs in without code changes on the model side.

## The idea in one line

Any moving agent becomes rows in a table. Foresight predicts where each agent will be in 1 to 2 seconds as a range, and the robot moves so it stays clear of those ranges. A new scene only has to produce tracks; what it costs is writing how each agent kind moves and reacts.

## What makes a scene a good test

A scene earns its place if it has at least two of these:

1. **Behaviour constant velocity cannot predict:** stops, turns, yielding, queues, bursts. Otherwise a ruler beats any model.
2. **A behaviour change mid-episode:** a regime switch. This is where TabPFN's "learn from the rows you just saw" shows.
3. **Several plausible futures:** go or wait, left or right. Ranges matter more than points.
4. **Interaction with the robot:** people react to it, so its choices change what happens.
5. **A clear visual story** a judge understands in 5 seconds.

## Catalogue

Priority: **A** = build this weekend, **B** = if time allows, **C** = pitch only (one slide, no data).

### 1. Plaza crossing (done, A)

A robot crosses a square while 10 to 30 people walk to their own goals, around two pillars.
- **Agents:** people (social force), one robot.
- **Regimes:** calm, hurried, erratic (sudden stops and sharp turns); the whole crowd switches once in about half the episodes.
- **Why it's good:** the base case, with all five properties. Built in `crowd.py`.

### 2. Delivery robot crossing a street (A, next)

A two-lane road with pavements and a zebra crossing. Cars drive both ways; pedestrians walk the pavements and cross; the robot must get from one pavement to the other.
- **Agents:**
  - **cars** (`kind=car`, `length` 4.5 m, `width` 1.8 m): follow their lane, IDM car following, brake for anyone on the zebra or ahead;
  - **pedestrians:** social force, restricted to pavements, crossing at the zebra or jaywalking;
  - **robot.**
- **Regimes:**
  - **calm:** slow cars, early yielding, no jaywalking;
  - **rush hour:** fast cars, short gaps, 20% jaywalkers.
- **Why it's good:** several agent kinds in one picture, which is the "any agent" claim made visible. Go-or-wait decisions give several plausible futures. Cars and people answer to different rules.
- **Map:** `map.json` with road polygon, lane centrelines, pavements, zebra polygon (drawing only).

### 3. Doorway bottleneck (A, cheap)

A wall with one 1.5 m door between two rooms. People stream through in both directions; the robot must pass the door too.
- **Agents:** people, robot. Walls as rectangle obstacles.
- **Regimes:** **trickle** (few people) and **rush** (a crowd arrives at `switch_t`, like a meeting ending).
- **Why it's good:** queues and waiting are behaviour constant velocity gets badly wrong, and the switch to rush tests adaptation. It's a small change to `crowd.py`: walls plus goals on the other side.

### 4. Station platform when a train arrives (B)

A long platform. Before the train, people wait in clusters near the edge. At arrival (`switch_t`), doors open: a burst of people leaves, and waiting people board. The robot (a cleaning or info robot) moves along the platform.
- **Regimes:** waiting, then arrival burst.
- **Why it's good:** the clearest "the crowd suddenly changed" story for the adaptation figure. Rows from before arrival are the wrong rows afterwards.

### 5. Pavement with two-way flow (B)

A 4 m pavement. People walk both ways and form lanes, as real crowds do. Some stop at shop windows, some overtake. The robot drives along the pavement.
- **Regimes:** quiet, busy (dense counterflow).
- **Why it's good:** lane formation is learnable structure that constant velocity misses, and the robot has to overtake and merge.

### 6. Signalised crossing (B)

A crossing with a pedestrian light. People gather while the light is red and burst across on green (a scheduled switch every 20 s). Cars stop on red. The robot crosses with the crowd or waits.
- **Why it's good:** periodic regime switches give many adaptation events per episode. It combines well with scene 2.

### 7. Warehouse aisles (C)

Up to hundreds of floor robots on an aisle grid, with human pickers walking between shelves. Robots follow lanes and yield at intersections.
- **Pitch line:** same layer, different agents: robots predicting robots and people.
- **If built:** robots follow grid paths and stop at crossings; `ext_aisle_id` as an adapter feature.

### 8. Drone swarm (C)

A flock of drones (boids rules: separation, alignment, cohesion) in 2D airspace; our drone crosses the flock.
- **Pitch line:** the rows just gain a z column; the planner checks 3D boxes.

### 9. Real data (done, A)

ETH and Hotel pedestrians, Zurich: real people for offline checks (no robot). More real scenes, by priority:
- **Zara and Students (UCY):** in OpenTraj, as spline files that need conversion.
- **Stanford Drone Dataset:** a large campus with bikes and cars, but a large download.

## Deliverable per scenario

For each scene, put this in `data/<scene>/` (generated, not committed) plus code in `foresight/domains/<scene>.py`:

| Item | Detail |
|---|---|
| `tracks.csv` | Every `INTERFACE.md` column. `kind` set per agent; `length`/`width` for cars; `collided`; true positions, no noise |
| `obstacles.csv` | Pillars, walls (rectangles: `w`, `h`, `angle`) |
| `meta.json` | `dt` (0.2 s is fine), `bounds`, per-episode `seed`, `regime_switch_t`, `regime_before`/`after`. For scheduled switches (scene 6), list all switch times in `notes` |
| `map.json` (new, optional) | Road, lanes, pavements, crossings, doors, as polygons. Drawing only |
| Episodes | Training: 80 episodes (seeds 0–79). Held out: 20 (seeds 1000–1019) as `data/<scene>_test/`. About half with a regime switch |
| Closed-loop hook | `Sim(seed, config).reset()` and `.step(robot_cmd)`, returning `(state, {"collided": bool})`, exactly as in `crowd.py` |
| Viewer | GIF of one episode per regime, like `tools/view_crowd.py` |
| Tests | Determinism (same seed gives the same tracks), no agent inside an obstacle, speeds within bounds, regime switch at the right time |

**Acceptance checks** (run before handing over):

1. `python -c "from foresight.core import schema; schema.load('data/<scene>')"` passes validation.
2. `python run_offline.py --data data/<scene>`: LightGBM beats constant velocity by at least 15% at +2 s. If not, the scene is too predictable, so add stops, turns or interactions.
3. `python run_closed.py --sim <scene> --planners cv --seeds 1000 1001`: the constant-velocity robot occasionally gets close to people (closest gap under 0.3 m). If it never does, the scene is too easy for the planner comparison.
4. A summary printout per regime: people count, mean speed, share standing still, collisions per minute.

## How each agent kind is modelled, and where the data comes from

| Agent | Model | Source |
|---|---|---|
| Pedestrians (simulated) | Social force (Helbing and Molnár 1995): pulled to a goal, pushed away from others, the robot, obstacles and walls. Field of view, stops, sharp turns | Our numpy code (`crowd.py`) |
| Pedestrians (real) | No model: positions annotated from overhead video | ETH Walking Pedestrians (ETH Zurich 2009), via OpenTraj |
| Robot | A disc moving at the commanded velocity | Our code; the planner sets the command |
| Cars (scenes 2 and 6) | Intelligent Driver Model on lane centrelines | To write, about 60 lines; the same model highway-env uses (installed, open source) |
| Floor robots, drones (C) | Grid path following; boids | Not built |

**Open tools:**
- **Data and models:** numpy, pandas, scikit-learn, LightGBM, TabPFN.
- **Figures:** matplotlib.
- **Deck:** python-pptx.
- **Installed:** highway-env.

**Bigger open simulators, too heavy to set up this weekend:**
- **pysocialforce, PedSim:** crowds.
- **SUMO:** city traffic with pedestrians.
- **CARLA:** 3D driving, needs a big GPU.

## Later: a 3D scenario viewer

A small three.js web page could replay `tracks.csv` with simple 3D objects (people, cars, robots, drones), draw the predicted boxes and the robot's path, and show a live-numbers side panel. It would work like the 3D agent dashboard built for DeckHQ. The data contract already supports it. Not for this weekend.
