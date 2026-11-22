# Industry practice

How deployed robots and self-driving stacks handle people, prediction, uncertainty, safety and compute, and where Foresight sits against that.

Checked on 21 Nov 2026. Every bullet comes from a page we fetched and read. Nothing here is cited from memory. Academic references already verified in [REFERENCES.md](REFERENCES.md) are linked as `REFERENCES [n]` and not repeated. Each source in the list at the end says how far it was read:

- **full text:** the whole document (or the relevant sections) was read;
- **page:** a product page, documentation page or blog post was read in full;
- **abstract:** only the abstract or landing page was read;
- **preview:** only the free preview of a paid standard (title page, foreword, scope, definitions, table of contents).

Company pages are the company's own claims, not independent evidence. Where public information is thin, the section says so.

Foresight facts used below come from the repo:
- replan every 0.4 s (D-2);
- boxes checked at h = 1 s and 2 s;
- robot top speed 1.0 m/s according to RESULTS.md, although `run_closed.py` runs every planner at 1.0 m/s (--max-speed default); 1.2 appears only as a back-fill value for old rows;
- the reactive baseline treats people as frozen at their current position (`run_closed.py`);
- the safety floor has two parts. The 0.4 s constant-velocity hard check is **implemented** in `planner.choose` (D-8). The takeover by constant velocity plus band when TabPFN is late is **designed** (DEPLOYMENT.md tier table) but still an open gate (GATES S8). The current `fallback` flag in `planner.py` means "no safe candidate, so take the largest worst-case gap", which is not a deadline fallback.

---

## 1. Deployed navigation stacks: ROS 2 Nav2

- **Costmap layers.** By default a Nav2 costmap loads a static layer, an obstacle layer and an inflation layer. Voxel, range, denoise and plugin-container layers are optional. The page does not mention people or tracked moving objects. Default costmap update rate is 5 Hz. The example config updates the local costmap at 5 Hz and the global one at 1 Hz [N1].
- **Inflation is the "margin".** The inflation layer adds a cost that decays exponentially around lethal obstacles: default radius 0.55 m, scaling factor 10 [N2]. It is a static distance buffer around whatever the sensors see now, and it carries no velocity.
- **Controllers.** The controller server runs at 20 Hz by default [N3]. Three controllers matter here:
  - **DWB**, the default, is the ROS 2 fork of a Dynamic Window Approach controller. It samples velocities and scores them with plugin critics such as obstacle distance, path alignment and goal distance [N4].
  - **MPPI** samples 1,000 trajectories by default (56 steps × 0.05 s, so about 2.8 s ahead; our arithmetic) and scores them with critics. The docs say it runs at 100+ Hz on a 4th-generation Intel i5 and avoids dynamic obstacles fluidly. They describe no model of how obstacles move: obstacles come in through the costmap critics [N5].
  - **Regulated Pure Pursuit** targets service and industrial robots. It slows near obstacles (scaling starts at 0.6 m by default) and forward-simulates the current command for 1.0 s, stopping if that collides [N6].
- **People and moving obstacles are not predicted in core Nav2** (as of the docs read on 21 Nov 2026). The rolling configuration index lists the following costmap layers: static, inflation, legacy inflation, asymmetric inflation, obstacle, voxel, range, denoise and plugin container. Its pages on moving objects cover a Following Server and FollowObject actions, which follow one tracked object. It has no people or social layer [N12]. The global path is replanned at 1 Hz in the default behaviour tree [N7, N8]. In forum threads from 2021 and 2023, the Nav2 maintainer (handle `smac`) said three things: the bottleneck is a general detection-and-tracking pipeline, not planning; the tracking work had stalled for lack of resources; and MPPI does well without explicit modelling of moving obstacles [N9]. An April 2025 paper states that existing Nav2 trajectory planners mainly handle static obstacles, or react to where dynamic obstacles are at the present moment [N10].
- **Collision Monitor** is a separate Nav2 node that bypasses the costmaps and planners and acts on raw sensor points. Its stop, slowdown and limit zones are polygons around the robot. Its approach model slows the robot so that it stays at least `time_before_collision` (default 2.0 s, simulated in 0.1 s steps) from a collision. The docs say it gives no hard real-time safety certification [N11].

**What this means for Foresight**
- Our `reactive` baseline carries the same information as stock Nav2: obstacles at their current positions, a clearance margin, and sampled velocity candidates scored on clearance and progress. Calling it "standard practice (costmap plus local planner)" is fair in information content.
- Be clear about the gaps. Nav2 runs its controller at 20 Hz on a 5 Hz costmap, with exponential inflation. Our reactive planner replans every 0.4 s (2.5 Hz) with a hard margin of 0.2, 0.4 or 0.6 m. Replanning that much slower may handicap the baseline. A fairer check is to rerun `reactive` at a 0.1 s replan. INTERFACE.md already specifies dt = 0.1 s for the simulator.
- Our 0.4 s constant-velocity hard check is close in spirit to Nav2's Collision Monitor approach model. It is a short-horizon, model-free check that sits outside the learned part. It is analogous only. It is not certified, and neither is the Collision Monitor. The deadline takeover by constant velocity plus band is not implemented yet, so do not claim it.
- Prediction-aware planning is a gap that the Nav2 maintainer has named in public. That is our opening. Say so without claiming that industry robots never predict, because private stacks are not public.

## 2. Sidewalk delivery and service robots (public sources only)

Information here is **thin**. No company we checked publishes how, or whether, it predicts pedestrian motion.

- **Starship** (FAQ, robots and accessibility pages):
  - Speed is described as low, about 4 mph (6.4 km/h), and as walking pace.
  - The robot slows when an object is beside it and comes to a complete stop when an object is in front of it.
  - The accessibility page says it detects whether, and how fast, objects are moving, then manoeuvres around them or stops.
  - It runs at Level 4 with remote assistance, and handles safety-critical functions on the device.
  - Its homepage claims over 10 million deliveries.
  - No prediction method is described [S1, S2, S3, S4].
- **Serve Robotics** (Head of Safety blog, June 2026):
  - It slows near blind corners and in heavy pedestrian traffic.
  - Its deceleration is designed to match the roughly 0.5 g that pedestrians, cyclists and vehicles use.
  - Remote supervisors step in when needed.
  - No numeric speed limit is given, and no prediction model is described [S5].
  - NVIDIA's case study says Serve moved from Jetson Xavier to Jetson Orin (AGX Orin) in its third generation [S6].
- **Kiwibot** (peer-reviewed field study, Pittsburgh pilot, Jul–Dec 2021):
  - Robots first ran with a nearby operator, later with remote monitoring only.
  - In one observed encounter, a robot stopped about 1–2 ft from a mother and children walking towards it. The family then surrounded it to look. In other encounters children stood in front of robots on purpose, which kept them stopped. The authors note that the gathered people blocked the sidewalk for other pedestrians, and that the robot did nothing to communicate.
  - Robots stuck on obstacles such as pavement cracks needed in-person help from staff [S7].
- **Amazon Scout:** field tests ended in October 2022, when Amazon said parts of the programme were not meeting customer needs. The article we read gives no speed or pedestrian behaviour [S8].
- **Hospital robots:**
  - An Aethon (TUG) patent describes speed limits set per path segment, plus a local speed cap when an obstacle is detected. Its example is a 3.0 ft/s segment limit capped to 2.0 ft/s when an obstacle is within 25 ft [S9]. This is a patent, not evidence of current product behaviour.
  - Diligent describes hospital hallways as full of fast-moving people. It says Moxi 2.0's NVIDIA-powered A2000 compute perceives 10–15× faster than the previous generation, which allows less cautious motion. No speed is given [S10].
  - Hotel robots: not researched.
- **Regulation (one example):** Pennsylvania's Act 106 (effective 30 Jan 2021) puts personal delivery devices inside the legal definition of "pedestrian". It covers remotely or autonomously operated devices up to 550 lb without cargo [S11].

**What this means for Foresight**
- The deployed default is **slow down when something is beside you, stop when something is ahead, plus a remote human**. That is our `stop-and-go` baseline plus a teleoperator. Including stop-and-go is therefore representative, not a straw man.
- In the Pittsburgh deployment, a stop-when-close rule left robots standing still among curious people, and the crowd around them blocked the sidewalk [S7]. That is a different failure from the freezing robot problem of REFERENCES [21], where the planner finds no safe path. It still shows that stopping has costs, and it is one more reason to measure time stopped and time to goal, not only collisions.
- Our robot runs at 1.0 m/s (3.6 km/h). That is below Starship's stated walking pace, so our speed regime is realistic for sidewalk robots.
- Do not claim that "delivery robots don't predict". Say that "public material describes slow-down and stop rules, and no published pedestrian-prediction method".

## 3. Self-driving prediction practice

- **Waymo Open Motion Dataset (WOMD):** 1 s of history, 8 s of future, 10 Hz, up to 8 agents to predict per scene. The paper's baselines output K = 6 trajectories. Metrics are minADE, minFDE, miss rate, overlap rate and mAP.
  - The miss rate uses separate lateral and longitudinal thresholds that grow with prediction time and speed.
  - mAP was introduced because minADE does not measure how well the confidence scores are calibrated, and it keeps improving artificially as K grows.
  - The authors chose 8 s because they consider long-horizon forecasting necessary for safe planning, against 3–5 s in other datasets [A1].
- **CV baseline on WOMD pedestrians:** at 8 s on the standard test set, constant velocity (K = 1) scored minADE 1.58 m, miss rate 0.60 and mAP 0.06, against 0.64 m, 0.13 and 0.22 for their LSTM baseline (K = 6). WOMD deliberately biases its selection towards objects that do *not* move at constant velocity, so these CV numbers are pessimistic by design [A1].
- **Argoverse 2:** 250,000 scenarios at exactly 10 Hz, with 5 s observed and 6 s forecast. Object types include pedestrians. The miss rate is the share of cases where none of the forecasts ends within 2.0 m of the truth, reported at K = 1 and K = 6. The leaderboard sorts by Brier-minFDE.
  - The paper's comparison table lists test horizons of 3 s (Argoverse 1, INTERACTION), 5 s (Lyft, Yandex), 6 s (nuScenes, Argoverse 2) and 8 s (WOMD).
  - On this dataset constant velocity does poorly: K = 1 minFDE 17.4 m on vehicle-like types (vehicle, bus, motorcyclist). The authors attribute this to scenarios mined for kinematically interesting motion [A2].
- **nuScenes prediction:** up to 2 s of history, 6 s of future at 2 Hz, and up to 25 trajectories per agent, each with a probability. Metrics are minADE_k, minFDE_k and miss rate at 2 m. The 2020 challenge predicted vehicles [A3].
- **Uncertainty to planning via occupancy:**
  - The Waymo Occupancy Flow challenge asks for 256 × 256 bird's-eye-view grids of occupancy probability, plus flow, at 8 one-second waypoints out to 8 s, for vehicles only. Occluded vehicles are included. It is scored by AUC and soft IoU, plus flow end-point error [A4].
  - The occupancy-flow paper notes that a grid represents many agents jointly, and that per-cell probabilities express where they may be [A5].
  - StopNet (ICRA 2022; its authors include the WOMD and occupancy-flow authors) predicts both trajectories and occupancy, for hundreds of agents within latency limits. The authors argue that occupancy lets the vehicle reason about groups of agents collectively [A6].

**What this means for Foresight**
- **Report a miss rate, but define it for our setting.** AV miss rates are best-of-K with vehicle thresholds (2 m at 6 s, or scaled lateral and longitudinal). We make one prediction with a box. Report two numbers:
  - **box miss rate**: the share of (person, horizon) pairs whose true position falls outside the joint 2-D box, which is 1 minus joint coverage;
  - **point miss rate**: the share of median predictions more than a stated threshold from the truth. Pick a pedestrian-scale threshold and say why.
  Always pair these with box width, as REFERENCES [31] requires.
- **Our 2 s horizon against AV 6–8 s:**
  - AV benchmarks use 3–8 s, and WOMD chose 8 s because its authors consider long-horizon forecasting necessary for safe planning [A1, A2].
  - Mobile-robot local control in Nav2 looks 1–3 s ahead: RPP 1.0 s [N6], the Collision Monitor 2.0 s [N11], MPPI about 2.8 s (derived) [N5]. These are rollouts of the robot's own command against obstacles held still. They are not forecasts of people, but they show how far ahead robot-speed control reasons.
  - **Our reasoning, not a sourced claim:** cars need longer horizons because of their speeds and stopping distances. A 1 m/s robot covers 2 m in our 2 s horizon, so 1–2 s is the planning-relevant range for our embodiment.
  - We also report 4.8 s offline, as the published ETH/UCY reference (REFERENCES §2).
- **Our per-person boxes are the trajectory-set end of the AV spectrum. Occupancy is the other end.** The `occupancy.npz` artefact in the VISUALS.md trace spec rasterises the boxes into a grid. Once it exists, we can say we produce both representations. Do not claim that our grid matches Waymo's occupancy flow: we have no flow and no occlusion.
- **CV is weak at long horizons on AV data but strong at short horizons on ETH/UCY** (REFERENCES [1]). Our horizon sits where CV is still a serious rival, which is why every claim is made against CV.

## 4. Safety standards and formal envelopes

We read free previews of the standards, not the full texts. "Not found" below means not found in the pages we read.

- **ISO 13482:2014 (personal care robots).**
  - Scope covers mobile servant robots, physical assistant robots and person carrier robots. It is limited to robots on the ground and excludes robots faster than 20 km/h, toys, water-borne and flying robots, industrial robots (ISO 10218), medical devices and military robots.
  - It follows the ISO 13849 and IEC 62061 approach to safety-related control.
  - It defines a **safeguarded space**, in which the robot starts a safety-related function (trajectory change, speed reduction, protective stop) when a person is detected inside it, and a **protective stop space**. Both may be static or dynamic.
  - It notes that, when it was published, no exhaustive internationally recognised data on pain or injury limits for impact existed.
  - No mention of learned predictors in the pages read [T1].
- **ISO 3691-4 (driverless industrial trucks, FDIS text of the 2023 edition).**
  - Scope names AGVs and autonomous mobile robots. Clause 4.8.2 is "Detection of persons in the path", and Annex A covers preparing the operating zones.
  - It normatively references ISO 13849-1/-2.
  - It explicitly excludes hazards from trucks intended for **public zones**, pointing to ISO 13482 instead [T2].
- **IEC 61508 framing.** IEC's overview describes a generic, risk-based standard for electrical, electronic and programmable electronic (E/E/PE) safety-related systems. It is meant to seed sector standards (IEC 62061 for machinery is listed) and grades safety functions from SIL 1 (lowest) to SIL 4 (highest) [T3]. The overview we read does not mention AI or ML.
- **ISO/IEC TR 5469:2024 (AI and functional safety)** is an informative technical report, not a requirements standard.
  - Its scope covers three uses: AI inside a safety function, **non-AI safety functions that keep AI-controlled equipment safe**, and AI used to design safety functions.
  - It warns that continual model updating undermines verification and validation, because behaviour drifts away from what was tested.
  - Annex A maps AI elements onto IEC 61508-3.
  - Its table of contents includes 10.2.2 "Detection mechanisms for switching" and 10.2.3 "Use of a supervision function with constraints to control the behaviour of a system to within safe limits". We saw only those titles, not the clause text [T4].
- **Mobileye RSS.**
  - RSS formalises "reasonable" worst-case assumptions about other road users as safe distances, dangerous situations and proper responses. The paper argues that statistical, miles-driven validation cannot reach the required fatality rates [T5].
  - For pedestrians it bounds the change of heading. It assumes the pedestrian continues in a straight line after the response time. A standing pedestrian is assigned every ray from their position. One worked example gives a 50 cm safety ball [T5].
  - Mobileye's page says a Chinese ITS Industry Alliance standard (0116-2019) uses RSS as its basis, and that IEEE 2846 was still forthcoming. It does not say IEEE 2846 adopted RSS [T6].

**What this means for Foresight**
- Our architecture follows the pattern TR 5469 names in its scope: a **learned predictor** (TabPFN) **guarded by a non-AI function**. Today that function is the implemented 0.4 s constant-velocity check; the constant-velocity-plus-band takeover is designed but not built. Say "follows the supervision pattern described in ISO/IEC TR 5469". Do not say "complies with". We read the scope and contents only, and TR 5469 sets no requirements.
- Our hard check corresponds conceptually to ISO 13482's protective stop space and safeguarded space, which may be dynamic. A product would still need ISO 13849 or IEC 62061 safety-rated sensing and control. Our check is plain Python.
- **TR 5469 cuts against `tabpfn_live` as a safety argument.** Changing the context changes behaviour at runtime, the drift TR 5469 flags. Our defence: the weights are frozen and the context is bounded and logged. The safety floor does not depend on the context. Position the live context as a performance feature *above* the floor, never as part of it.
- **RSS also cuts against us.** It is a worst-case envelope and argues against statistical arguments. An 80% box is a statistical object, and it does not give an RSS-style guarantee. Phrase it as "calibrated ranges for efficiency, a worst-case style floor for safety". RSS's straight-line pedestrian assumption is close to our constant-velocity floor.
- ISO 3691-4 excludes public zones, so for robots among the public the reference is ISO 13482, not the AMR standard.

## 5. Compute in practice

- **Robot-class hardware.** NVIDIA's Jetson Orin page lists AGX Orin (64 GB and 32 GB of unified LPDDR5) at up to 275 TOPS within 15–60 W [C1]. The same page gives inconsistent figures for the smaller NX and Nano modules, so we do not quote them. Serve's third-generation robots use Jetson Orin [S6]. Moxi 2.0 uses NVIDIA-powered A2000 compute [S10].
- **Navigation loop rates (CPU):** controller 20 Hz [N3], local costmap 5 Hz [N1], global replan 1 Hz [N7, N8]. MPPI is documented at 100+ Hz on a 4th-generation i5, and the docs suggest batch sizes of 1,000 at 50 Hz or 2,000 at 30 Hz [N5].
- **AV prediction latency (third-party benchmark).** The ProphNet authors (QCraft) report these inference latencies on a single V100 with 64 agents: Wayformer 102 ms, MultiPath++ 240 ms, LaneGCN 63 ms and their own model 28 ms. They stress that onboard deployment needs low latency [C2]. StopNet is designed around latency limits in dense urban scenes [A6].
- **Planning rates:** AV datasets are sampled at 10 Hz (WOMD, Argoverse 2) [A1, A2]. We found **no primary source for an onboard AV prediction or replanning rate**. Apollo's architecture doc says prediction is triggered by each perception message, without stating a rate [C3].
- **Compute for TabPFN itself (vendor figures).** The README recommends a GPU (about 8 GB of VRAM works, 16 GB for large datasets), calls CPU slow, and caps CPU use at 5,000 samples for the 3.x models [M1]. The TabPFN-3 report gives cached predict at 0.1–3 ms per test point (batches of 100) on an **H100** [M3]. We found **no Jetson or other embedded figure** for TabPFN.

**What this means for Foresight**
- Our 0.4 s replan (2.5 Hz) is slower than Nav2's 20 Hz controller. It is roughly the rate of Nav2's 1 Hz global replanner and 5 Hz costmap. Present TabPFN as a **prediction layer** at the 0.4 s replan (10 Hz needs the background rebuild, designed, not built) that feeds the planner, with a fast local check below it, as DEPLOYMENT.md already does.
- Our own measured plan times (199–861 ms on an emulated CPU, DEPLOYMENT.md) are a worst case. They are not evidence about embedded hardware. The only GPU figures are the vendor's H100 numbers, a datacentre GPU and not a robot one. Measured since: NVIDIA L4 71 ms per replan, Apple M3 Pro 287–321 ms per plan (DEPLOYMENT.md). Neither is embedded hardware: say "fits the 0.4 s replan on an L4; embedded timing not measured". Do not turn DEPLOYMENT.md's 10 Hz judgment into a sourced fact.
- One batched call over every person and horizon (we already do this) is what the README recommends [M1].

## 6. Tabular foundation models in industry

- **Licence (decisive for any product claim):** the TabPFN code is Apache 2.0. The weights for **TabPFN-2.5, 2.6, 3, 3.5 and 3.5-Fast are non-commercial**. The TabPFN-2 weights use the Prior Labs License (Apache 2.0 plus attribution). Commercial production goes through an enterprise licence [M1]. The TabPFN-2.5 report adds that, under its licence, the model, its derivatives and its outputs may not be used for commercial or production purposes [M2]. The TabPFN-3 report says the PyPI release is for evaluation and non-commercial use, with a managed API for commercial workloads [M3].
- **Interface:**
  - `predict` returns the mean by default. It can also return the median, mode or chosen quantiles (`output_type="quantiles"`), or the full predictive distribution [M4].
  - `fit_with_cache` reuses the context representation across `predict` calls.
  - Predicting one row at a time is much slower than one batched call [M1].
  - A hosted API client exists for users without a GPU [M1].
- **Low-latency path (vendor):** a proprietary distillation engine turns a fitted TabPFN into a small MLP or tree ensemble. The TabPFN-3 report says the distilled model runs on CPU at sub-millisecond latency [M2, M3].
- **Public case studies:**
  - Hitachi Rail is using TabPFN for predictive maintenance on Spain's rail network, from track sensors. Prior Labs reports roughly 40% lower RMSE than existing baselines in early deployment (press release, Nov 2025) [M5].
  - The TabPFN-3 report says Creditplus Bank *will* use distilled TabPFN models for CPU-based credit decisions [M3].
  - Robotics and control are not discussed in the part of the TabPFN-3 report we read (the first 100,000 of about 250,000 characters) [M3]. We found **no public robotics or real-time control deployment** of TabPFN.

**What this means for Foresight**
- "Deployable" must say which weights. The TabPFN-3.5 we run needs a commercial licence or the API to ship. The v2 weights are the open option (DEPLOYMENT.md already says this; the licence facts above confirm it).
- Our use (robot motion prediction, inside a control loop) has **no public industry precedent** that we could find. Present it as new territory, not as an established deployment pattern.
- Distillation is the vendor's own answer for low latency. It would replace the live context with a frozen student, which would remove our claim B (adapting without retraining). Mention it as a deployment option with that cost.
- Keeping the hosted API out of the control loop is our own rule (DEPLOYMENT.md), not a vendor statement. Present it as ours.

---

## How to phrase our positioning

1. "Our reactive baseline carries the same information as a standard ROS 2 Nav2 setup (current obstacle positions, a clearance margin, sampled velocity candidates). As of its current docs, stock Nav2 does not predict pedestrian motion [N9, N10, N12]."
2. "Deployed sidewalk robots publicly describe slow-down and stop rules with remote supervision [S1–S5, S7]. Our stop-and-go baseline is that rule. We found no published pedestrian-prediction method from those companies."
3. "We predict 1–2 s ahead because that is the robot's planning range at walking speed. Nav2's own local tools roll the robot's motion 1–3 s ahead [N5, N6, N11]. Self-driving benchmarks use 3–8 s [A1, A2], for vehicles that move much faster."
4. "We report calibrated ranges with coverage and width, plus a box miss rate in the spirit of AV miss rate, defined for single box predictions."
5. "The learned predictor sits above a non-learned short-horizon check, the supervision pattern described in ISO/IEC TR 5469 [T4]. This is not a certified safety function." Add "with a constant-velocity fallback when the predictor is late" only once that takeover is built (GATES S8).
6. "Calibrated ranges are meant for efficiency (in our box planner they did not change the pace; its caution setting did). The worst-case floor is for safety. We claim no RSS-style guarantee [T5]."
7. "TabPFN-3.5 weights are non-commercial. A product would use the commercial licence or the Apache-licensed v2 weights [M1]."
8. Avoid: "industry standard", "complies with ISO 13482", "certified", "Nav2-equivalent", "like Waymo's occupancy flow", "real-time on a Jetson".

---

## Not verified (seen only in search summaries, so not cited above)

- A maintained ROS 2 people or social costmap layer. The ROS 1 `social_navigation_layers` package and the Nav2CAN thesis appeared in search results, but their pages were blocked or missing.
- Serve's top speed (reported as 11 mph in press coverage), Kiwibot's own speed statements, and TUG's stop-and-announce behaviour (press coverage only).
- Any Pennsylvania personal-delivery-device speed cap.
- ISO 13482's review status and the draft revision. The braking-within-detection-range wording of ISO 3691-4.
- The Brier-minFDE formula (we confirmed only that Argoverse 2 sorts by it).
- WOMD's conference venue (the arXiv record lists none) and StopNet's affiliation (inferred from its authors, not stated).
- Other TabPFN case studies (Taktile, BostonGene, TD Bank) and the cloud marketplace listings.

---

## Sources (all accessed 21 Nov 2026)

**Nav2**
- [N1] Nav2 docs, Costmap 2D configuration. https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/core_servers/costmap_2d/ (page)
- [N2] Nav2 docs, Inflation Layer. https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/core_servers/costmap_2d/costmap_plugins/inflation/ (page)
- [N3] Nav2 docs, Controller Server. https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/core_servers/controller_server/ (page)
- [N4] Nav2 docs, DWB Controller. https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/controller_plugins/dwb_controller/ (page)
- [N5] Nav2 docs, Model Predictive Path Integral Controller. https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/controller_plugins/mppi_controller/configuring_mppic/ (page)
- [N6] Nav2 docs, Regulated Pure Pursuit. https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/controller_plugins/configuring_regulated_pp/ (page)
- [N7] Nav2 default behaviour tree `navigate_to_pose_w_replanning_and_recovery.xml` (RateController 1.0 Hz around ComputePathToPose). https://raw.githubusercontent.com/ros-navigation/navigation2/main/nav2_bt_navigator/behavior_trees/navigate_to_pose_w_replanning_and_recovery.xml (full text)
- [N8] Nav2 docs, Planner Server (`expected_planner_frequency` default 1.0). https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/core_servers/configuring_planner_server/ (page)
- [N9] Open Robotics Discourse, "Navigation with dynamic obstacles" (posts 2021–2023, maintainer handle `smac`). https://discourse.openrobotics.org/t/navigation-with-dynamic-obstacles/22321 (page)
- [N10] E. Schöneberg, M. Schröder, D. Görges, H. D. Schotten. *Trajectory Planning with Model Predictive Control for Obstacle Avoidance Considering Prediction Uncertainty.* arXiv:2504.19193, Apr 2025. https://arxiv.org/abs/2504.19193 (abstract)
- [N11] Nav2 docs, Collision Monitor node. https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/core_servers/collision_monitor/configuring_collision_monitor_node/ (page)
- [N12] Nav2 docs, Configuration Guide index (rolling). https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/ (page)

**Delivery and service robots**
- [S1] Starship FAQ. https://www.starship.xyz/faq/ (page)
- [S2] Starship, Our robots. https://www.starship.xyz/our-robots/ (page)
- [S3] Starship, Accessibility. https://www.starship.xyz/autonomous-robots-accessibility/ (page)
- [S4] Starship homepage. https://www.starship.xyz/ (page)
- [S5] J. Streeter (Serve Robotics), robot safety post, 29 Jun 2026. https://serverobotics.com/blog/serve-robot-safety-standards (page)
- [S6] NVIDIA customer story, Serve Robotics. https://www.nvidia.com/en-us/customer-stories/serve-robotics/ (page; vendor marketing)
- [S7] D. Weinberg, H. Dwyer, S. E. Fox, N. Martelaro. *Sharing the Sidewalk: Observing Delivery Robot Interactions with Pedestrians during a Pilot in Pittsburgh, PA.* Multimodal Technologies and Interaction 7(5):53, 2023. DOI 10.3390/mti7050053. https://par.nsf.gov/servlets/purl/10488136 (full text, relevant sections)
- [S8] The Robot Report, "Amazon ends testing of Scout delivery robots", 6 Oct 2022. https://www.therobotreport.com/amazon-ends-testing-of-scout-delivery-robots/ (page)
- [S9] Aethon, Inc. *Robotic ordering and delivery apparatuses, systems and methods.* US 7,894,939 B2, granted 22 Feb 2011. https://patents.google.com/patent/US7894939B2/en (first 100,000 characters read)
- [S10] The Robot Report, Diligent Robotics Moxi 2.0 rollout, 17 Aug 2026. https://www.therobotreport.com/five-years-operation-shape-diligent-robotics-rollout-moxi-2-0/ (page)
- [S11] PennDOT Driver and Vehicle Services Bulletin 21-02, Act 106 personal delivery devices, Jan 2021. https://www.pa.gov/content/dam/copapwp-pagov/en/dmv/documents/business-partners/driver-and-vehicle-services-bulletin/bulletin%2021-02%20act%20106%20personal%20delivery%20devices.pdf (full text)

**Self-driving prediction**
- [A1] S. Ettinger et al. (18 authors, Waymo / Google Brain). *Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset.* arXiv:2104.10133, 2021. https://arxiv.org/abs/2104.10133 and https://arxiv.org/pdf/2104.10133 (full text: Secs. 2–5, Table 2)
- [A2] B. Wilson et al. (13 authors). *Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting.* NeurIPS Datasets and Benchmarks track. arXiv:2301.00493. https://arxiv.org/pdf/2301.00493 (full text: Table 2, Sec. 3.3, Tables 5–6)
- [A3] nuScenes prediction challenge README (nuscenes-devkit). https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/README.md (page)
- [A4] Waymo Open Dataset, 2022 Occupancy and Flow Prediction Challenge. https://waymo.com/open/challenges/2022/occupancy-flow-prediction-challenge/ (page)
- [A5] R. Mahjourian, J. Kim, Y. Chai, M. Tan, B. Sapp, D. Anguelov. *Occupancy Flow Fields for Motion Forecasting in Autonomous Driving.* IEEE RA-L (per arXiv journal ref). arXiv:2203.03875. https://arxiv.org/abs/2203.03875 (abstract)
- [A6] J. Kim, R. Mahjourian, S. Ettinger, M. Bansal, B. White, B. Sapp, D. Anguelov. *StopNet: Scalable Trajectory and Occupancy Prediction for Urban Autonomous Driving.* ICRA 2022. arXiv:2206.00991. https://arxiv.org/abs/2206.00991 (abstract)
- Also read: Waymo Open Dataset 2024 Motion Prediction Challenge page (task: 1 s history, up to 8 agents, 8 s). https://waymo.com/open/challenges/2024/motion-prediction/ (page, partly truncated)

**Safety standards and RSS**
- [T1] ISO 13482:2014, *Robots and robotic devices: Safety requirements for personal care robots.* Free preview via iTeh: https://cdn.standards.iteh.ai/samples/53820/5ddca453a6d141e5a558f4f791ea3229/ISO-13482-2014.pdf (preview: introduction, Clause 1 scope, Clause 3 definitions). Catalogue: https://www.iso.org/standard/53820.html (blocked to our fetcher)
- [T2] ISO/FDIS 3691-4 (2023 edition), *Industrial trucks: Safety requirements and verification, Part 4: Driverless industrial trucks and their systems.* Free preview via iTeh: https://cdn.standards.iteh.ai/samples/83545/b70b16397664461482d0324c7414abfc/ISO-FDIS-3691-4.pdf (preview: table of contents, foreword, Clause 1 scope, normative references). The final published text may differ in wording.
- [T3] IEC, *Overview of IEC 61508 & Functional Safety* (2022 slide deck). https://assets.iec.ch/public/acos/IEC%2061508%20&%20Functional%20Safety-2022.pdf (full text)
- [T4] ISO/IEC TR 5469:2024, *Artificial intelligence: Functional safety and AI systems.* Preview via VDE Verlag: https://assets.vde-verlag.de/iec-normen/preview-pdf/info_isoiectr5469%7Bed1.0%7Den.pdf (preview: table of contents, introduction, Clause 1 scope)
- [T5] S. Shalev-Shwartz, S. Shammah, A. Shashua. *On a Formal Model of Safe and Scalable Self-driving Cars.* arXiv:1708.06374v6, 2018. https://arxiv.org/abs/1708.06374 and https://arxiv.org/pdf/1708.06374v6 (full text, relevant sections: intro, Sec. 2, Sec. 3.8)
- [T6] Mobileye, Responsibility-Sensitive Safety. https://www.mobileye.com/technology/responsibility-sensitive-safety/ (page)

**Compute**
- [C1] NVIDIA Jetson Orin. https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/ (page)
- [C2] X. Wang, T. Su, F. Da, X. Yang (QCraft). *ProphNet: Efficient Agent-Centric Motion Forecasting with Anchor-Informed Proposals.* arXiv:2303.12071v3, 2023. https://arxiv.org/pdf/2303.12071 (full text: Sec. 4.5, Table 4)
- [C3] Apollo software architecture (Apollo 5.5 spec, r6.0.0 branch). https://github.com/ApolloAuto/apollo/blob/r6.0.0/docs/specs/Apollo_5.5_Software_Architecture.md (page)

**TabPFN in industry** (the TabPFN papers themselves are REFERENCES [8]–[11])
- [M1] PriorLabs/TabPFN GitHub README. https://github.com/PriorLabs/TabPFN (page)
- [M2] TabPFN-2.5 technical report, HTML version (Secs. 3, 5, 6). https://arxiv.org/html/2511.08667v2 (first 100,000 characters read). See REFERENCES [9].
- [M3] TabPFN-3 technical report, HTML version (Secs. 1, 2.4, 4, 5). https://arxiv.org/html/2605.13986 (first 100,000 characters read). See REFERENCES [11].
- [M4] Prior Labs docs, Regression. https://docs.priorlabs.ai/capabilities/regression.md (page)
- [M5] Prior Labs, Hitachi Rail case study, 25 Nov 2025. https://priorlabs.ai/case-studies/hitachi (page; vendor press release)
