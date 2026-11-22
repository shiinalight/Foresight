# Exploratory: TabPFN picks the move (written before any run; commit de81e0a, 03:32 Berlin)

**Times.** Section times are the commit that added the section (git, +0100 = Berlin). Run ids from the L4 pod are UTC (Berlin = UTC+1); laptop run ids are Berlin time. All policy runs are on the pod: teacher `20261122-023825-pol-*` = 03:38:25 Berlin (6 min after this commit), pilot `-024302-` = 03:43:02, test `-024701-` = 03:47:01.

**Question.** Can a second TabPFN, chained after the range model, choose the robot's move about as well as the hand-written planner it learned from? This is a chain of TabPFN steps (ranges, then scene adaptation, then the move), in the spirit of duckpfn (TabPFN-3.5 hackathon, 2nd place), where TabPFN was the controller. **No "beats" claim.** Claim C stays failed whichever way this comes out.

**Teacher:** the box planner `choose()` driving tabpfn_split at q 0.35, the best point in the dial sweep.
- Its choice depends on the TabPFN ranges plus the robot state, so it is learnable from the row.
- The oracle depends on future information the row cannot hold, and vo ignores the ranges, so neither is used as teacher.

**Policy:**
- At each replan, every candidate move from `planner.candidates()` becomes one row of a single batched call:
  - robot state relative to the goal;
  - the candidate's speed and turn;
  - the candidate's predicted clearance to every person's TabPFN box at +1 s and +2 s;
  - its progress to the goal.
- A TabPFN classifier (`fit_with_cache`, fitted once on teacher rows, run locally on the L4) scores P(teacher picks this candidate); the policy takes the highest. Heading is never regressed, because left/right passing is two-peaked.
- **What stays classical, stated as such:** the tracking filter, the coverage offset and the 0.4 s constant-velocity hard check. The check's override rate is reported. If it does most of the driving, we say so.
- So the honest wording is "TabPFN predicts, adapts and picks the move", not "no classical parts".

**Data:**
- Teacher runs on seeds 3300–3339, T5 config, with `--log`.
- Test on fresh seeds 3400–3419, T5 config.
- Pilot on 5 test crowds first. If the robot drifts, do one round of relabelling (DAgger: the states the policy visits get the teacher's choice, added to the context). If it still fails after that, stop and report it as tried.

**Pass rule** (parity with the teacher on the 20 test crowds, paired), using the T5 margins:
- goal-reached rate within 10 points;
- time_capped no more than 2.6 s slower;
- min gap no more than 0.158 m smaller;
- moving-collision rate no more than 0.244 higher.

All four → "TabPFN can drive the loop at the teacher's level". Otherwise → "tried; fails on …".

**Builder:** RoboDetect (owns `run_closed.py`). **Compute:** the L4 pod, at most 4 processes (hosted calls would exceed the monthly credit limit).
