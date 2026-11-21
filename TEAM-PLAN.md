# Foresight: team plan (HackLab Berlin, Prior Labs track)

Revised Sat 21 Nov, 16:00. **Submission deadline Sun 22 Nov 16:30** (tally.so/r/GexoxQ); we aim for 16:00. About 24 hours left, sleep included.

## Feedback from Eddie (Prior Labs judge, Sat evening)

Direction approved ("If that's what you want to aim for, go for it!"). His one recommendation: **aim for something visual you can show of how it works.** So the demo animation (ranges drawn on moving people, then the robot choosing its path) is the priority deliverable, not an afterthought. Every rung should end with a clip.

## What we must submit (all three, before 16:30)

1. **A 3-minute video** (unlisted YouTube or Google Drive, open to anyone with the link). It must contain the pitch, an explanation of the solution, and the demo. Include at least one of us speaking to camera.
2. **A public GitHub repository** with the code and technical documentation: README with results, how to run, architecture, limits.
3. **Slides for a 5-minute live presentation,** used only if we win the track.

So the last 3.5 hours are reserved for the video, the README and the slides, not for code.

## Two-minute summary (accept or veto this first)

**What we build.** One prediction-and-planning layer that any robot can use: give it the tracks of the agents around the robot (people, cars), and it returns
1. where each agent will be in 1 s and 2 s, as a range (10%, 50%, 90% quantiles) predicted by TabPFN, and
2. the robot's safest useful next move, chosen so its path stays clear of those ranges.

**Why TabPFN, specifically.**
- **A calibrated range** for each prediction, which drives the safety decision directly.
- **Learning from context rows without training.** We feed it rows from this scene in the last seconds plus the most similar past situations. When behaviour changes, it adapts within seconds with no retraining.

**Embodiment-agnostic by design, proven on two domains.** The core never mentions lanes or legs: a row is (agent, time, horizon) with relative motion and neighbour gaps. Domain-specific features plug in through a small adapter.
- **Domain 1:** a mobile robot (think humanoid or delivery robot) crossing a plaza with people.
- **Domain 2:** a car on a highway (highway-env).

The same code on both, with only the adapter swapped, is the evidence for "any robot".

**Built as a ladder.** Every rung ends in a working demo. If time runs out after any rung, we submit what we have, and it is complete.

**Not built.**
- **Camera detection and clustering:** the simulators give us agent states. Noise plus a Kalman filter stands in for perception. A live camera feed is a stretch rung at the end.
- **CARLA.** It runs on a cloud GPU, but costs 2 to 4 hours we don't have:
  - a 20 GB Docker image;
  - Vulkan drivers;
  - off-screen rendering;
  - a matching Python client;
  - debugging all of that remotely.

  It also adds nothing to the TabPFN question, since we would only read agent positions from it, as from highway-env.
- **Dense occupancy maps from TabPFN.**
- **An LLM in the control loop.**

## The ladder

Each rung has a time box, a demo and a stop condition. A rung is not started until the one below it works.

| Rung | Time box | What exists at the end | Demo if we stop here |
|---|---|---|---|
| **R0 Setup** | Sat 16:00–17:00 | Team repo; cloud GPU running TabPFN; latency measured; roles fixed | — |
| **R1 Prediction** | 17:00–19:30 | Crowd simulator; TabPFN range predictor against baselines; calibration check | Animation of people with predicted ranges drawn; accuracy and calibration plots |
| **R2 Context** | 19:30–22:00 | Per-query context retrieval; behaviour switch; adaptation measured | Plus: "seconds to adapt" and retrieval-ablation plots |
| **R3 Planning** | 22:00–01:00 (+ overnight runs on the GPU) | Robot crossing the crowd, choosing moves from the ranges; five planners compared | Side-by-side video: constant-velocity robot against ours; collisions against speed |
| **R4 Second domain** | Sun 08:00–10:30 | highway-env adapter on the same core | Same layer driving a car: "one layer, any robot" |
| **R5 Stretch** | 10:30–12:30 | One or more of the stretch items below | Whatever is finished |
| **Freeze** | **12:30** | Code frozen. Video, README and slides until 16:00 (schedule below); submit by 16:00 | |

**Stop rules.**
- **R1 gate at 18:30:** in the crowd simulator, a learned model must clearly beat constant velocity at 2 s (roughly 15% lower error). If it doesn't, make highway-env domain 1 and move the crowd to R4. highway-env drivers brake and change lanes, which constant velocity cannot predict.
- **Check at 22:00:** if R2 is not working, skip R3's closed loop. Polish R1 and R2 into the submission, with replayed animations.
- **No new rung after 10:30 Sunday,** except stretch items already half done.
- **Rolling submission:** from the end of R1 onwards, every finished rung updates the README, so the repo is submittable at any moment.

## Final schedule, Sunday afternoon (fixed whatever the code state)

| Time | What |
|---|---|
| 12:30–13:15 | Final numbers with confidence intervals; final plots; record clean demo clips (screen capture of the animations and side-by-side runs) |
| 13:15–14:00 | README: problem, approach, architecture diagram, results table, how to run, limits. Make the repo public and check that nothing secret is in it |
| 14:00–14:30 | Slides (5-minute version; the video reuses them) |
| 14:30–15:30 | Video: write the script, record one of us speaking to camera, cut it with the demo clips. Keep it to 3:00 or less |
| 15:30–16:00 | Upload unlisted; open the link in a private window to check access; fill the form |
| 16:00–16:30 | Buffer |

## Compute: cloud GPU

**Decision: rent a cloud GPU now (or take one from the organisers if they find one).** Any provider with a CUDA GPU of 16 to 24 GB works: an L4, A10, RTX 4090 or A100 is plenty for TabPFN. Darshak rents it himself, since it needs a payment card.

**Setup (part of R0):**
1. Linux image with CUDA and Python 3.11 or 3.12.
2. `pip install tabpfn highway-env lightgbm`.
3. Set `TABPFN_TOKEN` on the machine (licence already accepted).
4. Clone the team repo and run `check_tabpfn.py`.
5. For headless rendering, set `SDL_VIDEODRIVER=dummy` and use `render_mode="rgb_array"`.

**Division of work:**
- **Laptops:** code, small runs, plots, the video.
- **GPU:** the timing check, large contexts, and the overnight closed-loop runs.
- **Code moves through git.** Never edit on the server only.

**Fallback if the GPU fails:**
- R1 and R2 run on laptop CPU or the hosted API, with context up to about 1,500 rows.
- R3 becomes a few short episodes, or ranges drawn on recorded runs.

**Camera stretch item:** runs on the laptop either way. Its neural chip already runs the YOLO detector at about 3 ms per frame.

**Public repo hygiene:**
- Never commit the token: keep it in an environment variable, and put `.env` in `.gitignore`.
- Never commit model weights.
- Never commit anything personal.

Verified in the installed package:
- `tabpfn` 9.2 defaults to the TabPFN-3.5 model.
- On CPU it refuses more than about 5,000 context rows unless `TABPFN_ALLOW_CPU_LARGE_DATASET=1` is set.
- `predict(X, output_type="quantiles", quantiles=[...])` works.
- `fit` takes no sample weights.

**Still unmeasured:** TabPFN time per prediction. That is the first thing R0 does.

## Answers from the organisers (Sat afternoon)

1. **Compute:** none provided yet. We use a rented cloud GPU, or theirs if they find one.
2. **Data:** none provided. We choose the problem, simulator or open data ourselves.
3. **Submission:** video, public repo and slides by Sun 16:30 (top of this file). Judging criteria were not stated, so the video and README have to show the problem, the solution, a working demo and honest numbers.
4. **Prepared code:** say so in the README (`traj_kit.py` was written before the event as generic utilities).
5. **Overnight:** Darshak can stay, so overnight runs are fine.

## R0 Setup (16:00–17:00)

1. **Team repo.** Darshak creates a standalone repo and invites Nooshin. Copy in `traj_kit.py`, `check_tabpfn.py`, this plan and the two reference notes.
2. **TabPFN timing.** On the cloud GPU and on the laptop, time one fit plus one batched predict of about 60 rows, at 500 and 1,500 context rows.
3. **Roles.** One person on simulator, planner and visuals; the other on features, models and evaluation. Darshak's tracking and Kalman background fits the second.
4. **Install highway-env now**, so R4 is not blocked by surprises. Check:
   - it runs;
   - per-car settings take effect, including mid-episode;
   - the collision flag in `info` is found.

## R1 Prediction (17:00–19:30)

**Crowd simulator, upgraded from `traj_kit.py`.** The current synthetic walkers give constant velocity almost nothing to lose (0.303 m against 0.295 m for gradient boosting), so they need real behaviour:
- **Destinations:** each person walks towards a goal, turns when they reach it, sometimes stops.
- **Avoidance:** people steer around each other, around the robot and around static obstacles (a simple social-force rule).
- **Behaviour regimes:**
  - calm: slow, wide berth;
  - hurried: fast, cuts close;
  - erratic: sudden stops and turns.

  Each person draws settings from the current regime with a little jitter, and the regime can switch for the whole crowd mid-episode.
- **Sensing:** noise on positions (about 0.1 m), smoothed by the Kalman filter.

**The table (core schema, shared by every domain).**
- **Row** = one agent *i*, at time *t*, for horizon *h* ∈ {1 s, 2 s}. Horizon is a feature; one model per axis.
- **Frame:** agent-centric. Rotate so the agent's heading is the x axis.
- **Targets:** deviation from constant velocity, `ddx` and `ddy` (where the agent ends up, minus where constant velocity says it would be).
- **Core features:**
  - **Own motion:** speed, acceleration, turn rate.
  - **History:** the last 5 steps of relative motion.
  - **Neighbours:** relative position and velocity of the 3 nearest agents.
  - **Robot:** distance to the robot and closing speed.
  - **Surroundings:** distance to the nearest obstacle, local density, *h*.
- **Adapter features (per domain):**
  - **Driving:** lane offset, lane index, gaps and speed differences to cars ahead and behind in each lane, time to collision.
  - **Crowd:** none beyond the core.

**Baselines on the same features.**
- Constant velocity and constant acceleration.
- Kalman filter.
- Gradient boosting: point predictions, and quantile models for the ranges.

**Leakage rules** (these decide whether the numbers mean anything):
- **Split by episode, never by row.** Rows 0.2 s apart are near-copies.
- **Causality:** at query time *t*, the context holds only rows whose target time has passed (*t₀ + h ≤ t*).
- **Forward-only smoothing** in features. A smoother that looks ahead leaks the answer.
- **Tune on validation episodes; report on held-out episodes.**

**Demo at the end of R1:**
- An animation of the crowd with each person's predicted range drawn as it moves.
- Error at 1 s and 2 s against the baselines.
- Calibration: the 10–90% range should contain the truth about 80% of the time; report range width too.

## R2 Context (19:30–22:00)

Per prediction batch, the context (up to about 1,500 rows, refit every 1 s) is built from three parts:
- **60% recency:** completed rows from this episode in the last 10 to 30 s (this crowd's current behaviour).
- **30% nearest neighbours** from a bank of other episodes, on standardised situation features: situations like this one.
- **10% stratified random fill,** always including rare behaviours (stops, sharp turns).

Details:
- **Cold start:** for the first 1 to 2 s no row has a completed target, so the context is bank-only.
- **Bank size:** subsample the bank to about 20,000 rows before indexing.

**Experiments.**
- **Ablation:** recency only, neighbours only, the mix, and a random fixed context.
- **Seconds to adapt:**
  - **Setup:** switch the regime mid-episode.
  - **Measure:** error and coverage against seconds since the switch.
  - **Compare:** TabPFN re-conditioned, gradient boosting refit on the same rows, gradient boosting frozen.
- **Data efficiency:** error against number of context rows (50 to 5,000), TabPFN against gradient boosting.

This rung is where TabPFN's advantage should show most clearly. It is also the strategy the past winners used (Breakout, Prior Bahn).

## R3 Planning (22:00–01:00, runs overnight on the GPU)

**Generic planner.** The robot has a set of motion candidates: about 15 combinations of speed and heading change, held for 2 s. A car uses its own five actions instead (change left, change right, keep, faster, slower). For each candidate:
1. Roll the robot forward and get its footprint at 1 s and 2 s.
2. Each agent's range becomes a box: constant velocity plus the q10 to q90 deviation.
3. **Veto** candidates whose smallest gap to any box is below the safety margin.
4. Among the rest, pick the one with the most progress towards the goal, with a small penalty for sharp turns.

The quantile level is the caution dial: a more cautious robot uses a wider range.

**Five planners on the same seeds.**
1. Constant velocity plus a fixed margin.
2. **Constant velocity plus a constant band,** measured once on held-out data. This answers "why not just a bigger margin?".
3. Gradient-boosting quantile ranges.
4. TabPFN median, no range.
5. **TabPFN range (ours).**

Each planner is swept over 3 margin settings, so the comparison is at equal speed.

**Overnight budget rule.** The run size follows from the measured speed:

episodes in 8 h = 28,800 s ÷ (seconds per step × steps per episode)

- **Example:** a 40 s episode at 5 Hz is 200 steps. At 0.5 s per step an episode takes about 100 s, so about 290 episodes fit.
- **Order:** TabPFN planners first, then the others. The constant-velocity planners are cheap.
- **Seeds:** prefer more seeds at 3 margins over few seeds at many.

**Demo at the end of R3:**
- A side-by-side video of the same crowd, constant-velocity robot against ours, with the ranges drawn.
- **Headline plot:** collisions and near-misses against average speed, five planners.

**Known limit:** people's predicted ranges do not react to the robot's own move. If time allows, recompute their robot-distance features per candidate; otherwise it goes on the limits slide.

## R4 Second domain (Sun 08:00–10:30)

Write the highway-env adapter:
- lane-frame features;
- the five car actions as candidates;
- regimes as calm, normal and aggressive driver settings.

Reuse every other module unchanged. Show the same plots for cars, more briefly. The pitch line: "we swapped the context table, not the model".

## R5 Stretch (10:30–12:30, pick in this order)

1. **Uncertainty-triggered re-retrieval** (this is what makes it agentic). When an agent's range is wider than a threshold, re-query that agent with context retrieved for its specific situation. Deterministic and bounded; measure whether ranges narrow without losing coverage.
2. **Behaviour classifier first.** TabPFN predicts the mode first (keep going, stop, turn left or right; for cars keep, change left, change right), then the range per mode. This gives a mixture of ranges instead of one over-wide range when the outcome is 50/50.
3. **Live camera.** The laptop webcam with YOLO on the neural chip detects people. A tracker turns the detections into the same table, and the predicted ranges are drawn live on video. This is the real-world moment for the pitch. It needs a bridge between two Python environments (YOLO runs natively, TabPFN under emulation), for example a local socket.
4. **Learned risk model.** One row per (scene, robot candidate), with the minimum gap over the next 2 s as the target.
   - **Data:** rollouts of a randomly moving robot, so the labels don't learn our planner's habits.
   - **Inputs:** prediction features from R1, generated out-of-fold.
   - **Decision:** choose by the predicted q10 of the minimum gap.

## Plots for the pitch

1. **Closed loop (headline, R3):** collisions and near-misses against speed, five planners.
2. **Seconds to adapt (R2).**
3. **Data efficiency (R2):** TabPFN leads with few rows; the gap closes with many. Say so.
4. **Calibration (R1):** nominal against observed coverage, with range width.
5. **Two domains (R4):** the same layer's accuracy gain on people and on cars.

## Video script (3 minutes; the 5-minute slides follow the same order)

| Time | On screen | Said |
|---|---|---|
| 0:00–0:20 | One of us, to camera | A robot in public needs to know where everyone will be in the next seconds, and how sure it is |
| 0:20–0:50 | Diagram: tracks, table, TabPFN, ranges, planner | We turned the scene into a table: one row per agent, moment and horizon. TabPFN reads this crowd from the last seconds and similar past situations, with no training, and returns a range |
| 0:50–1:40 | Demo: side-by-side runs with ranges drawn | The robot picks the move that stays clear of the ranges; the constant-velocity robot doesn't |
| 1:40–2:20 | Plots 1, 2, 3 | Numbers against baselines, adaptation after a behaviour change, calibration |
| 2:20–2:40 | Car demo (R4) | The same layer drives a car: we swapped the table, not the model |
| 2:40–3:00 | To camera | Where it breaks (long horizons, people reacting to the robot, simulated behaviour) and what's next |

If a rung didn't finish, its row shrinks and the others take the time.

## Code layout

```
foresight/
  core/
    schema.py     agent-centric rows, targets, causality filter, split
    context.py    retrieval: recency, neighbours, stratified fill, cold start
    models.py     TabPFN and baselines behind one predict_quantiles()
    planner.py    candidates in, ranges to boxes, veto and choice
    eval.py       errors, coverage, collision rates, bootstrap intervals
  domains/
    crowd.py      2D crowd simulator, regimes, robot, adapter
    highway.py    highway-env wrapper, regimes, adapter
  run_offline.py  R1 and R2 experiments
  run_closed.py   R3 and R4 episodes
  plots.py
  demo.py         animations and side-by-side video
```

`traj_kit.py` supplies the Kalman filter, constant-velocity baseline, split and timing code.

## Reference notes (copied into the team repo)

- `av-tabpfn-brainstorm.md`: reasoning, pitfalls, datasets, code snippets for the driving domain.
- `tabpfn-hackathon-strategies.md`: how past TabPFN hackathon winners used the model.
