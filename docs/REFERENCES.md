# References

Every claim, baseline, benchmark and protocol choice in Foresight, with the paper behind it.

Checked on 21 Nov 2026. Each entry was looked up online, not cited from memory. Each entry says how far it was checked:

- **full text:** the paper itself was read and the numbers were taken from its tables;
- **abstract:** only the abstract or landing page was read;
- **metadata:** only the bibliographic record (Crossref) was checked. We cite these only for what their title and venue establish, for example "the original social force model" or "the original DWA". We make no claim about their detailed content.

---

## 1. Claims and the papers behind them

| # | Our claim or choice | Paper | What the paper supports | Caveat: how to phrase it |
|---|---|---|---|---|
| 1 | Constant velocity (CV) is a strong baseline on ETH/UCY | Schöller et al. 2020 [1] | On ETH/UCY (8 observed and 12 predicted steps, leave-one-scene-out), their CVM had the lowest average error of all the deterministic models they tested, including RED and SR-LSTM (ADE/FDE 0.39/0.83 against 0.44/0.91 and above). The lead was largest on Hotel (FDE 0.51 m against 0.74–0.94 m). On **ETH** it did not win: their LSTM, linear regression and RED were slightly better (FDE 1.04–1.14 m against 1.15 m). | Say "on average matched or beat the deterministic neural predictors tested in 2019–20, most clearly on Hotel". It does not beat everything. With 20 samples, S-GAN beats the sampled CVM on the three UCY scenes, and later best-of-20 models (Trajectron++, AgentFormer) report lower errors. |
| 2 | Agent-frame features help a learned model carry over to a new scene | Schöller et al. 2020 [1], Sec. VI-A | Networks given absolute positions learn scene-specific direction priors. Hotel walkways run vertically and the other scenes run mostly horizontally, so Hotel suffers. Feeding relative motion, and adding random rotations, improved FF and RED, most of all on Hotel. | This is evidence for relative and rotated inputs. It does not prove our exact agent-frame encoding. It does fit our f1 leak (D-20), which was a world-heading leak. |
| 2b | Motion history and neighbour features add information | Schöller et al. 2020 [1], Secs. VI-B and VI-C | **Cuts against us.** In their tests a long motion history added little over the most recent steps. Feeding the neighbours' histories, or even their true futures, did not improve FF or RED, and the authors conclude that interactions are hard to exploit from motion histories. | Our features include the last stretch of path and 3 neighbours. Do not claim these features help unless an ablation of ours shows it. Gupta et al. [2] also found that pooling over neighbours did not reliably help (20V-20 against 20VP-20). |
| 3 | Predict the deviation from constant velocity rather than the absolute position | Schöller et al. 2020 [1], Sec. III | They predict relative displacements (residuals) rather than absolute positions, and say this lowers the error. | They predict relative displacements, not deviations from CV. Our target is related in spirit, and we found no paper that uses exactly a CV-residual target in the agent's own frame. Present it as our design choice. |
| 4 | Standard ETH/UCY protocol: 8 observed and 12 predicted frames at 0.4 s (3.2 s / 4.8 s), leave-one-scene-out over 5 scenes, ADE/FDE in metres | Gupta et al. 2018 [2]; Salzmann et al. 2020 [3]; Yuan et al. 2021 [4]; Schöller et al. 2020 [1] | All four state this protocol: train on 4 sets, test on the held-out fifth. Generative models report the minimum over K = 20 samples. | **Our protocol differs.** We transfer between two scenes (ETH → Hotel and Hotel → ETH) instead of leave-one-out over five. We report error at fixed horizons (1.2, 2.0, 4.8 s), not averaged over 12 steps. Our test rows are sampled differently. Published numbers are a reference point, not a like-for-like comparison. |
| 5 | Report error at 4.8 s because that is the standard horizon | Same as 4 | 4.8 s (12 × 0.4 s) is the standard final horizon. | Our "error at +4.8 s" is one point at the final horizon, so compare it with **FDE**, not ADE. Compare only with deterministic (single-prediction) numbers, never with best-of-20. |
| 6 | Cross-scene evaluation tests generalisation to a place the model has never seen | Schöller et al. 2020 [1]; Gupta et al. 2018 [2] | Schöller et al. justify leave-one-out as a test of generalisation to scenes not seen in training [1]. | Our two-scene version holds out a scene too, but each direction trains on a single scene, which is harsher than training on four. Call it "cross-scene (held-out scene)". Do not call it "the standard split". |
| 7 | Datasets: ETH and Hotel (ETH Walking Pedestrians), read through OpenTraj | Pellegrini et al. 2009 [5]; Amirian et al. 2020 [7] | ETH and Hotel come from Pellegrini et al. OpenTraj lists ETH as 2 top-view scenes, 750 pedestrians, world coordinates, 2.5 fps. | Cite Pellegrini for the data and OpenTraj for the loader and copy. UCY (Zara, Univ) is Lerner et al. 2007 [6]. We do not use UCY yet, so do not say "ETH/UCY" for our own data. |
| 8 | TabPFN predicts from example rows in one forward pass, with no training on our data | Hollmann et al. 2025 [8] | TabPFN is pretrained only on synthetic tabular datasets and does training and prediction on a new dataset in one forward pass. It supports regression with a piecewise-constant output distribution. It was evaluated on datasets up to 10,000 samples and 500 features. | The Nature paper describes the 2025 TabPFN ("v2"), not the TabPFN-3.5 we run. For the model we use, cite the TabPFN-3.5 technical report [10]: an arXiv preprint written by the vendor, not peer reviewed. Nature does not describe quantile output as such: we read quantiles off the predictive distribution. |
| 9 | Retrieving nearest-neighbour rows as context improves TabPFN | Thomas et al. 2024 (LoCalPFN) [12] | Using the query's k nearest neighbours as TabPFN's context ("TabPFN-kNN") raised IQM AUC from 0.917 to 0.943 on 95 datasets. Fine-tuning together with retrieval (LoCalPFN) raised it further. | **Classification only, on the original TabPFN (v1).** The paper says regression could not be tested. Our use (regression, TabPFN-3.5, recent same-scene rows mixed with kNN rows) carries the idea over by analogy. It is not shown by the paper. |
| 10 | In-context learning, with examples picked per moment from the same scene, has been used for pedestrian trajectory prediction | Fujii, Saito & Hachiuma 2025 (TrajICL) [13] | TrajICL picks in-context examples from the same scene by similar position and velocity (STES), then re-picks them using its own predicted future (PG-ES). It adapts to new scenes without updating weights. | TrajICL is a transformer **trained** on the synthetic MOTSynth dataset, not a tabular foundation model. Its main benchmarks are MOTSynth, JRDB, WildTrack, SDD and JTA. ETH/UCY appears only in an appendix. It reports minADE/minFDE with K = 20. It shows the retrieval idea has precedent, not that TabPFN works. |
| 11 | An online conformal offset keeps coverage near target when the scene changes | Gibbs & Candès 2021 [17]; Dixit et al. 2023 [16] | Adaptive conformal inference updates one miscoverage parameter online and "provably achieves the desired coverage frequency over long-time intervals" whatever the data-generating process [17]. Dixit et al. use adaptive conformal prediction for online uncertainty sets on multistep trajectory predictions inside an MPC (drone and frisbee in simulation) [16]. | The guarantee is **long-run average coverage**. It is not per step, not per person and not conditional. Say "keeps long-run box coverage near 80%". Check that empirically, and do not call it guaranteed at each moment. |
| 12 | Conformal prediction regions around predicted agent positions can make planning safe | Lindemann et al. 2023 [15] | An MPC uses trajectory predictions plus conformal prediction regions. The safety guarantee holds with user-set probability (shown in CARLA, at an intersection full of pedestrians). | Their guarantee needs **Assumption 1**: the robot does not change the agents' behaviour. It also needs offline calibration data from the same distribution (Remark 4). Our social-force crowd reacts to the robot, and our offset is online, so **claim no safety guarantee**. Huang et al. (WAFR 2024) [32] argue that the robot's plan shifts human motion, so calibration has to depend on the plan. That cuts against treating per-person boxes as independent of the robot. Yao et al. (CoRL 2025) [19] use adaptive conformal inference for crowd navigation under distribution shift. |
| 13 | Freezing is a real failure mode of crowd navigation, so we measure it | Trautman & Krause 2010 [20]; Trautman et al. 2015 [21] | The "freezing robot problem": once the scene is dynamic enough, the planner decides that every forward path is unsafe and the robot stops or makes needless manoeuvres [21, abstract]. In their field study, a DWA-based reactive planner proved insufficient above 0.55 people/m² [21]. | Their remedy is **modelling cooperation** (interacting Gaussian processes). Our planner rejects moves into independent per-person boxes, which is the kind of non-cooperative planner they describe as prone to freezing. Say "we measure freezing". Do not say "we avoid it", unless the data show it. We read the abstract of the 2015 journal version, not the full 2010 conference paper. |
| 14 | Social-force crowd simulation | Helbing & Molnár 1995 [22] | It introduced the social force model of pedestrian motion. | Ours is our own implementation (`foresight/domains/crowd.py`: Helbing-style forces with a forward field of view, our own parameters and a regime switch). We found no validation of it against real trajectories. Write "a social-force crowd simulator (after Helbing & Molnár)", and claim no realism beyond that. |
| 15 | Closed-loop metrics: time to goal, collisions, minimum gap, intrusion (time within 0.5 m), freezing | Francis et al. 2025 [25]; Chen et al. 2019 [24] | The Table I metrics of Francis et al. include time to goal, collisions, minimum distance to a human, personal space compliance **at 0.5 m**, and timeout, failure to progress and stalled time. They recommend reporting as many standard metrics as feasible, with distributions and not only averages [25]. CrowdNav reports success, collision rate, time to goal and discomfort frequency, which is the time fraction within **0.2 m** [24]. | Our 0.5 m intrusion lines up with the 0.5 m personal-space threshold of Francis et al., not with CrowdNav's 0.2 m discomfort. Write "metrics follow the categories of Francis et al.". |
| 16 | Comparison with crowd-navigation work | Chen et al. 2019 (CrowdNav / SARL) [24] | CrowdNav is a Python simulator. Its humans are driven by ORCA with randomised parameters, in circle-crossing scenarios, with visible and invisible robot settings and 500 test cases each. | **Our simulator is not CrowdNav** (social force, not ORCA), so our numbers cannot be compared with CrowdNav's. You may write "similar evaluation metrics (success, collisions, time to goal, closeness to people)". Do not write "on the standard CrowdNav testbed". |
| 17 | Reactive DWA-style baseline planner | Fox, Burgard & Thrun 1997 [26] | The original Dynamic Window Approach. | Our `reactive` baseline (`run_closed.py`) uses the same candidate sampler of (speed, turn) moves as our planner and treats people as frozen where they are now, with no prediction. It shares DWA's idea of sampling velocities and scoring clearance, but we did not check for DWA's acceleration-limited window. Call it "reactive (DWA-style)", not DWA. Trautman et al. [21] report that DWA breaks down in dense crowds, which is a reason to include it as a baseline. |
| 18 | Velocity-obstacle (sampling, RVO-style) baseline planner | van den Berg et al. 2008 (RVO) [27]; van den Berg et al. 2011 (ORCA) [28] | RVO and ORCA are the reference reciprocal velocity-obstacle methods. | Our `choose_vo` (`foresight/core/planner.py`) is a sampling velocity-obstacle planner that assumes people keep their velocity. Its own docstring says "Not ORCA (no reciprocity)". Call it "sampling velocity-obstacle planner". "RVO-style" overstates it, since there is no reciprocity. The ORCA paper is dated 2011 but comes from ISRR 2009. |
| 19 | LightGBM quantile regression baseline | Ke et al. 2017 [29]; Koenker & Bassett 1978 [30] | They are the sources for LightGBM and for quantile regression. | Our LightGBM is untuned (RESULTS.md), so call it "a reasonable baseline, not a tuned one", as RESULTS.md already does. |
| 20 | Interval coverage plus pinball loss is a proper way to judge ranges | Gneiting & Raftery 2007 [31] | Sec. 6.1: the family of quantile scores that includes the pinball loss is proper for quantile forecasts. Sec. 6.2: the interval score for a central (1−α) interval is its width plus a penalty of 2/α times the distance by which the truth falls outside, so misses cost more for higher-confidence intervals. | Coverage **alone is not a proper score**: very wide boxes always cover. Always report width or pinball next to coverage, as RESULTS.md does. |
| 21 | Paired seeds (every planner sees the same crowds) and bootstrap intervals | Francis et al. 2025 [25]; Agarwal et al. 2021 [33] | Francis et al. recommend comparing policies on the same scenarios with low variability, at least about 30 trials for real-robot runs, and distributional reporting. Agarwal et al. recommend bootstrap interval estimates in place of point estimates for few-run RL comparisons. | No paper we checked prescribes **paired seeds**. That is standard common-random-numbers practice and our own design choice, so state it as a choice. Agarwal et al. use a stratified bootstrap over runs and tasks, not a paired-seed bootstrap. |
| 22 | Kalman-filtered velocity for CV | Kalman 1960 [34] | It is the source for the Kalman filter. | Schöller's CVM uses the raw last-two-point velocity [1], not a filtered one. That difference matters when we compare with their numbers (Section 2). |
| 23 | Tabular foundation models can do temporal prediction | Hoo et al. (TabPFN-TS) [14]; TabPFN-3 report [11] | TabPFN-TS frames forecasting as tabular regression on TabPFN-v2 with time features, with no time-series pretraining. The TabPFN-3 report describes a separate time-series checkpoint. | This is univariate and covariate forecasting, not multi-agent spatial prediction. It is background only. |
| 24 | PLAN claim A: TabPFN's ranges stay calibrated in a scene it has never seen, where LightGBM's collapse | None found | No paper tests a tabular foundation model's interval calibration under scene transfer for pedestrians. | This is **our own result**, with no literature behind it. Report it with its protocol (cross-scene, single seed so far) and do not present it as known. |
| 25 | PLAN claim B: reading the new scene's recent rows adapts the prediction in seconds, without retraining | Nearest precedent: TrajICL [13]; LoCalPFN [12] by analogy | TrajICL adapts to new scenes by picking same-scene in-context examples, with no weight updates. | **Our own result.** TrajICL is a trained trajectory transformer with a different setting. Use it only as "the idea has precedent". |
| 26 | Crowd navigation is hard, and evaluation practice is uneven | Mavrogiannis et al. 2023 [23] | A survey of core challenges in social robot navigation and of evaluation methodology. | Background only. |

---

## 2. Published numbers to compare against (ETH and Hotel, 4.8 s horizon)

**Protocol of these numbers.** Leave-one-scene-out over the 5 ETH/UCY scenes. 8 observed steps (3.2 s), 12 predicted (4.8 s), at 0.4 s per step. ADE is the mean over all 12 predicted steps and FDE is the error at the 12th. Units are metres. Schöller et al. keep windows with at least 2 future steps (sequences shorter than 10 are dropped), so some of their test windows end before 12 steps [1].

**How ours differs.** We train on one scene and test on the other. We report the error at fixed horizons on randomly sampled rows. A row needs the person observed throughout a short history window (`features.py`, f2: 5 samples (0.2 s steps in simulation, 0.4 s on ETH/Hotel), about 1 s), not the 8 steps (3.2 s) of the standard protocol. Our CV uses a Kalman-filtered velocity, not the last two points. So these numbers are a **reference point, not a like-for-like comparison**. Our "+4.8 s error" corresponds to **FDE** and should be read against the **deterministic** rows only.

### Deterministic (one prediction per person): comparable in kind to our point error

| Model (source) | ETH ADE / FDE | Hotel ADE / FDE |
|---|---|---|
| CVM, last-two-point velocity (Schöller [1], Table I, "OUR") | 0.58 / 1.15 | **0.27 / 0.51** |
| Linear regression on the full history (Schöller [1], "Lin") | 0.58 / 1.11 | 0.39 / 0.81 |
| RED, re-implemented (Schöller [1]) | 0.60 / 1.14 | 0.47 / 0.94 |
| SR-LSTM, original paper's figures (as reported in Schöller [1]) | 0.63 / 1.25 | 0.37 / 0.74 |
| Linear (Gupta, Social GAN [2], Table 1) | 1.33 / 2.94 | 0.39 / 0.72 |
| LSTM (Gupta [2], Table 1) | 1.09 / 2.41 | 0.86 / 1.91 |
| Social LSTM, as reproduced by Gupta [2] | 1.09 / 2.35 | 0.79 / 1.76 |
| Trajectron++, most-likely output (Salzmann [3], Table 2a) * | 0.71 / 1.66 | 0.22 / 0.46 |

### Best of 20 samples: not comparable with a single point prediction (shown for context only)

| Model (source) | ETH ADE / FDE | Hotel ADE / FDE |
|---|---|---|
| CVM + 25° angular noise, best of 20 (Schöller [1], "OUR-S") | 0.43 / 0.80 | 0.19 / 0.35 |
| S-GAN, author-released pretrained model, re-evaluated by Schöller [1] | 0.59 / 1.04 | 0.38 / 0.79 |
| SGAN-20V-20 (Gupta [2]; the figure Trajectron++ and AgentFormer quote) | 0.81 / 1.52 | 0.72 / 1.61 |
| SGAN-20VP-20 (Gupta [2], full model with pooling) | 0.87 / 1.62 | 0.67 / 1.37 |
| Trajectron++, full sampled output (Salzmann [3], Table 2b) * | 0.39 / 0.83 | 0.12 / 0.21 |
| AgentFormer (Yuan [4], Table 1) | 0.45 / 0.75 | 0.14 / 0.22 |

\* **Trajectron++ caveat.** The appendix of Chen et al. 2024 (MGF, NeurIPS 2024) [35] states that early Trajectron++ code misused `np.gradient`, so the model saw future information, and that fixing it usually lowers the scores substantially. We found no corrected Trajectron++ ETH/UCY figures, so treat its numbers (and the AgentFormer comparison against them) as possibly optimistic.

**Three different "S-GAN" figures exist.** Gupta's 20V-20 and 20VP-20, and Schöller's re-evaluation of the released model. Always name which one.

### Our CV against the published CVM (open issue)

Resolved by D-26 (filter tuned on training data; constant velocity now within 3-14% of the published values). RESULTS.md (21 Nov, after the leakage fix) gives our CV error at +4.8 s as **0.858 m on Hotel** (ETH → Hotel) and **1.442 m on ETH** (Hotel → ETH). Schöller's CVM FDE is **0.51 m** and **1.15 m**. Ours is clearly higher. The gap is **unexplained**. CV needs no training, so the cross-scene split cannot be the reason. Candidate causes, none checked:

- Kalman-filtered velocity against the raw last-two-point velocity;
- a shorter required history (about 1 s against 3.2 s), so people who appeared recently are included;
- the rows that enter the test set (random rows against every sliding window);
- Schöller's test windows can end before 12 steps, which lowers their FDE;
- sample size: our +4.8 s figure rests on about 300 test rows and one seed (RESULTS.md).

Until this is reconciled, do not say that our CV "matches" the published CVM.

---

## 3. Full reference list

**Pedestrian prediction and the benchmark**

1. C. Schöller, V. Aravantinos, F. Lay, A. Knoll. *What the Constant Velocity Model Can Teach Us About Pedestrian Motion Prediction.* IEEE Robotics and Automation Letters 5(2):1696–1703, 2020 (also ICRA 2020). DOI 10.1109/LRA.2020.2969925. arXiv:1903.07933.
   Verified via https://arxiv.org/pdf/1903.07933v3 (full text, Table I, Secs. III–VI) and https://api.crossref.org/works/10.1109/LRA.2020.2969925 (metadata).
2. A. Gupta, J. Johnson, L. Fei-Fei, S. Savarese, A. Alahi. *Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks.* CVPR 2018, pp. 2255–2264. DOI 10.1109/CVPR.2018.00240. arXiv:1803.10892.
   Verified via https://arxiv.org/pdf/1803.10892 (full text, Table 1, Sec. 4) and https://api.crossref.org/works/10.1109/CVPR.2018.00240 (metadata).
3. T. Salzmann, B. Ivanovic, P. Chakravarty, M. Pavone. *Trajectron++: Dynamically-Feasible Trajectory Forecasting with Heterogeneous Data.* ECCV 2020, LNCS, pp. 683–700. DOI 10.1007/978-3-030-58523-5_40. arXiv:2001.03093.
   Verified via https://arxiv.org/pdf/2001.03093 (full text, Table 2, Sec. 5) and https://api.crossref.org/works/10.1007/978-3-030-58523-5_40 (metadata).
4. Y. Yuan, X. Weng, Y. Ou, K. Kitani. *AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting.* ICCV 2021, pp. 9793–9803. DOI 10.1109/ICCV48922.2021.00967. arXiv:2103.14023.
   Verified via https://arxiv.org/pdf/2103.14023 (full text, Table 1, evaluation protocol) and https://api.crossref.org/works/10.1109/ICCV48922.2021.00967 (metadata).
5. S. Pellegrini, A. Ess, K. Schindler, L. van Gool. *You'll Never Walk Alone: Modeling Social Behavior for Multi-Target Tracking.* ICCV 2009, pp. 261–268. DOI 10.1109/ICCV.2009.5459260.
   Verified via https://api.crossref.org/works/10.1109/ICCV.2009.5459260 (metadata) and the DBLP record found by search.
6. A. Lerner, Y. Chrysanthou, D. Lischinski. *Crowds by Example.* Computer Graphics Forum 26(3):655–664, 2007 (Eurographics). DOI 10.1111/j.1467-8659.2007.01089.x.
   Verified via https://api.crossref.org/works/10.1111/j.1467-8659.2007.01089.x (metadata).
7. J. Amirian, B. Zhang, F. V. Castro, J. J. Baldelomar, J.-B. Hayet, J. Pettré. *OpenTraj: Assessing Prediction Complexity in Human Trajectories Datasets.* ACCV 2020, LNCS, pp. 566–582. DOI 10.1007/978-3-030-69544-6_34. arXiv:2010.00890.
   Verified via https://arxiv.org/abs/2010.00890 (abstract), https://github.com/crowdbotp/OpenTraj (dataset table: ETH, 2 scenes, 750 pedestrians, 2.5 fps) and https://api.crossref.org/works/10.1007/978-3-030-69544-6_34 (metadata).

**Tabular foundation models and retrieval**

8. N. Hollmann, S. Müller, L. Purucker, A. Krishnakumar, M. Körfer, S. B. Hoo, R. T. Schirrmeister, F. Hutter. *Accurate predictions on small data with a tabular foundation model.* Nature 637:319–326, 2025. DOI 10.1038/s41586-024-08328-6.
   Verified via https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/ (full text: synthetic pretraining, single forward pass, regression output distribution, ≤10,000 samples / 500 features) and https://api.crossref.org/works/10.1038/s41586-024-08328-6 (metadata).
9. L. Grinsztajn et al. (26 authors). *TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models.* arXiv:2511.08667, 2025 (technical report).
   Verified via https://arxiv.org/abs/2511.08667 (abstract: up to 50,000 rows and 2,000 features).
10. B. Jäger, N. Erickson, L. Grinsztajn et al. (47 authors). *TabPFN-3.5: Technical Report.* arXiv:2609.17895, Sept 2026 (vendor technical report, not peer reviewed). **This is the model Foresight runs.**
    Verified via https://arxiv.org/abs/2609.17895 (abstract: TabPFN-3.5, -Fast, -Plus and -Thinking variants; the abstract states no scale limits and does not mention regression).
11. L. Grinsztajn, K. Flöge, O. Key et al. (41 authors). *TabPFN-3: Technical Report.* arXiv:2605.13986, May 2026 (vendor technical report).
    Verified via https://arxiv.org/abs/2605.13986 and https://priorlabs.ai/technical-reports/tabpfn-3 (abstract: up to 1M training rows, TabPFN-TS-3 time-series checkpoint).
12. V. Thomas, J. Ma, R. Hosseinzadeh, K. Golestan, G. Yu, M. Volkovs, A. Caterini. *Retrieval & Fine-Tuning for In-Context Tabular Models.* NeurIPS 2024. arXiv:2406.05207.
    Verified via https://arxiv.org/html/2406.05207v1 (full text: TabPFN-kNN and LoCalPFN results, classification only, limitations) and https://proceedings.neurips.cc/paper_files/paper/2024/hash/c40daf14d7a6469e65116507c21faeb7-Abstract.html (found by search).
13. R. Fujii, H. Saito, R. Hachiuma. *Towards Predicting Any Human Trajectory In Context* (TrajICL). NeurIPS 2025. arXiv:2506.00871. Affiliations: Keio University (Fujii, Saito) and NVIDIA (Hachiuma).
    Verified via https://arxiv.org/abs/2506.00871 (abstract) and https://arxiv.org/html/2506.00871v3 (full text: datasets, STES/PG-ES, K = 20).
14. S. B. Hoo, S. Müller, D. Salinas, F. Hutter. *From Tables to Time: Extending TabPFN-v2 to Time Series Forecasting* (TabPFN-TS). arXiv:2501.02945, v4 2026 (earlier versions had different titles).
    Verified via https://arxiv.org/abs/2501.02945 (abstract).

**Conformal prediction and safe planning**

15. L. Lindemann, M. Cleaveland, G. Shim, G. J. Pappas. *Safe Planning in Dynamic Environments Using Conformal Prediction.* IEEE Robotics and Automation Letters 8(8):5116–5123, 2023. DOI 10.1109/LRA.2023.3292071. arXiv:2210.10254.
    Verified via https://par.nsf.gov/servlets/purl/10441188 (published full text: Assumption 1, Remark 4, conclusion) and https://api.crossref.org/works/10.1109/LRA.2023.3292071 (metadata).
16. A. Dixit, L. Lindemann, S. X. Wei, M. Cleaveland, G. J. Pappas, J. W. Burdick. *Adaptive Conformal Prediction for Motion Planning among Dynamic Agents.* L4DC 2023, PMLR 211:300–314. arXiv:2212.00278.
    Verified via https://proceedings.mlr.press/v211/dixit23a.html (abstract).
17. I. Gibbs, E. Candès. *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS 2021. arXiv:2106.00170.
    Verified via https://arxiv.org/abs/2106.00170 (abstract: long-run coverage guarantee) and the NeurIPS proceedings listing found by search.
18. A. N. Angelopoulos, S. Bates. *Conformal Prediction: A Gentle Introduction.* Foundations and Trends in Machine Learning 16(4):494–591, 2023. DOI 10.1561/2200000101. Preprint title: *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*, arXiv:2107.07511.
    Verified via https://api.crossref.org/works/10.1561/2200000101 (metadata) and the publisher listing found by search.
19. J. Yao, X. Zhang, Y. Xia, Z. Wang, A. K. Roy-Chowdhury, J. Li. *Towards Generalizable Safety in Crowd Navigation via Conformal Uncertainty Handling.* CoRL 2025. arXiv:2508.05634. They feed adaptive-conformal uncertainty about pedestrian predictions into a constrained RL policy, and test under velocity, policy and grouping shifts.
    Verified via https://arxiv.org/abs/2508.05634 (abstract).

**Crowd navigation**

20. P. Trautman, A. Krause. *Unfreezing the Robot: Navigation in Dense, Interacting Crowds.* IROS 2010, pp. 797–803. DOI 10.1109/IROS.2010.5654369.
    Verified via https://api.crossref.org/works/10.1109/IROS.2010.5654369 (metadata). The full text was not read. The content claims rest on [21].
21. P. Trautman, J. Ma, R. M. Murray, A. Krause. *Robot navigation in dense human crowds: Statistical models and experimental studies of human–robot cooperation.* International Journal of Robotics Research 34(3):335–356, 2015. DOI 10.1177/0278364914557874.
    Verified via https://authors.library.caltech.edu/records/64682-sx438 (abstract: freezing robot problem definition, DWA insufficient above 0.55 people/m²) and https://api.crossref.org/works/10.1177/0278364914557874 (metadata).
22. D. Helbing, P. Molnár. *Social force model for pedestrian dynamics.* Physical Review E 51(5):4282–4286, 1995. DOI 10.1103/PhysRevE.51.4282. arXiv:cond-mat/9805244.
    Verified via https://api.crossref.org/works/10.1103/PhysRevE.51.4282 (metadata).
23. C. Mavrogiannis, F. Baldini, A. Wang, D. Zhao, P. Trautman, A. Steinfeld, J. Oh. *Core Challenges of Social Robot Navigation: A Survey.* ACM Transactions on Human-Robot Interaction 12(3), 2023, pp. 1–39. DOI 10.1145/3583741. arXiv:2103.05668.
    Verified via https://api.crossref.org/works/10.1145/3583741 (metadata) and the arXiv record found by search.
24. C. Chen, Y. Liu, S. Kreiss, A. Alahi. *Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning* (CrowdNav / SARL). ICRA 2019, pp. 6015–6022. DOI 10.1109/ICRA.2019.8794134. arXiv:1809.08835.
    Verified via https://ar5iv.labs.arxiv.org/html/1809.08835 (full text: ORCA-driven humans, circle crossing, metrics, Tables I–II) and https://api.crossref.org/works/10.1109/ICRA.2019.8794134 (metadata).
25. A. Francis, C. Pérez-D'Arpino, C. Li, F. Xia, A. Alahi, R. Alami, A. Bera, A. Biswas, et al. *Principles and Guidelines for Evaluating Social Robot Navigation Algorithms.* ACM Transactions on Human-Robot Interaction 14(2), Article 34, 2025. DOI 10.1145/3700599. arXiv:2306.16740 (2023).
    Verified via https://arxiv.org/html/2306.16740v4 (full text: Table I metrics, PSC at 0.5 m, guidelines M1/M8, B4) and https://api.crossref.org/works/10.1145/3700599 (metadata).
26. D. Fox, W. Burgard, S. Thrun. *The Dynamic Window Approach to Collision Avoidance.* IEEE Robotics & Automation Magazine 4(1):23–33, 1997. DOI 10.1109/100.580977.
    Verified via https://api.crossref.org/works/10.1109/100.580977 (metadata).
27. J. van den Berg, M. Lin, D. Manocha. *Reciprocal Velocity Obstacles for Real-Time Multi-Agent Navigation.* ICRA 2008, pp. 1928–1935. DOI 10.1109/ROBOT.2008.4543489.
    Verified via https://api.crossref.org/works/10.1109/ROBOT.2008.4543489 (metadata).
28. J. van den Berg, S. J. Guy, M. Lin, D. Manocha. *Reciprocal n-Body Collision Avoidance* (ORCA). In *Robotics Research* (ISRR 2009), Springer Tracts in Advanced Robotics 70, pp. 3–19, 2011. DOI 10.1007/978-3-642-19457-3_1.
    Verified via https://api.crossref.org/works/10.1007/978-3-642-19457-3_1 (metadata).

**Baselines, metrics and statistics**

29. G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS (NIPS) 2017.
    Verified via https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html (abstract).
30. R. Koenker, G. Bassett. *Regression Quantiles.* Econometrica 46(1):33ff., 1978 (last page not checked). DOI 10.2307/1913643.
    Verified via https://api.crossref.org/works?query.bibliographic=Koenker+Bassett+Regression+Quantiles (metadata; Crossref gives only the first page, 33).
31. T. Gneiting, A. E. Raftery. *Strictly Proper Scoring Rules, Prediction, and Estimation.* Journal of the American Statistical Association 102(477):359–378, 2007. DOI 10.1198/016214506000001437.
    Verified via https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf (full text, Secs. 6.1–6.2) and https://api.crossref.org/works/10.1198/016214506000001437 (metadata).
32. Z. Huang, T. Ji, H. Zhang, F. Cheraghi Pouria, K. Driggs-Campbell, R. Dong. *Interaction-aware Conformal Prediction for Crowd Navigation.* WAFR 2024. arXiv:2502.06221.
    Verified via https://arxiv.org/abs/2502.06221 (abstract).
33. R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, M. G. Bellemare. *Deep Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS 2021 (Outstanding Paper). arXiv:2108.13264.
    Verified via https://arxiv.org/abs/2108.13264 (abstract).
34. R. E. Kalman. *A New Approach to Linear Filtering and Prediction Problems.* Journal of Basic Engineering 82(1):35–45, 1960. DOI 10.1115/1.3662552.
    Verified via https://api.crossref.org/works/10.1115/1.3662552 (metadata).
35. J. Chen, J. Cao, D. Lin, K. Kitani, J. Pang. *MGF: Mixed Gaussian Flow for Diverse Trajectory Prediction.* NeurIPS 2024. arXiv:2402.12238. Cited only for its note on the Trajectron++ `np.gradient` bug (Appendix A).
    Verified via https://arxiv.org/pdf/2402.12238 (full text, appendix).
36. A. Alahi, K. Goel, V. Ramanathan, A. Robicquet, L. Fei-Fei, S. Savarese. *Social LSTM: Human Trajectory Prediction in Crowded Spaces.* CVPR 2016, pp. 961–971. DOI 10.1109/CVPR.2016.110.
    Verified via https://api.crossref.org/works/10.1109/CVPR.2016.110 (metadata). Its ETH/UCY numbers are disputed: Gupta et al. [2] could not reproduce them, and Salzmann et al. [3] use Gupta's reproduction. Cite it for the idea, not for numbers.

---

## 4. Could not verify, or corrected from the draft

- **"TrajICL, NVIDIA, NeurIPS 2025": exists, with corrections.** The title is *Towards Predicting Any Human Trajectory In Context*, NeurIPS 2025, arXiv:2506.00871. It is mainly a Keio University paper: two Keio authors and one NVIDIA co-author (Hachiuma). The NVIDIA lab lists it, but "NVIDIA paper" overstates it. It is a trained transformer, not a tabular foundation model. ETH/UCY is appendix-only.
- **LoCalPFN claim narrowed.** The paper shows kNN-retrieved context improving TabPFN v1 on **classification** only. It states that regression could not be tested. We found no peer-reviewed paper showing kNN context helping TabPFN **regression**.
- **Hollmann et al. (Nature 2025) is not the model we run.** It describes TabPFN (v2), up to 10,000 samples. Foresight runs TabPFN-3.5, whose source is a vendor technical report (arXiv:2609.17895, Sept 2026), not peer reviewed. TabPFN-3 (arXiv:2605.13986, May 2026) and TabPFN-2.5 (arXiv:2511.08667) are also vendor reports.
- **Schöller et al. 2020 claim narrowed.** "CVM matches or beats neural predictors" holds on average against the deterministic models they tested, and most clearly on Hotel. On ETH, their LSTM, linear regression and RED were slightly better. The sampled CVM (OUR-S) loses to the pretrained S-GAN on Zara1, Zara2 and Univ. Later best-of-20 models report lower errors than OUR-S.
- **Lindemann et al. 2023 does not cover our setting.** Its guarantee assumes the robot does not influence the agents (Assumption 1) and uses offline calibration. Its experiments are in CARLA, not in a reactive crowd.
- **ORCA year.** The ORCA paper is in a 2011 Springer volume, but the symposium was ISRR 2009. Cite it as "van den Berg et al. 2011 (ISRR 2009)".
- **Francis et al.** The draft said "2023 / 2025". It was an arXiv preprint in 2023 and appeared in ACM THRI in Feb 2025. Cite the 2025 journal version.
- **Trautman & Krause 2010.** We checked its metadata but did not read the full paper. The freezing definition and the DWA density finding are taken from the 2015 IJRR version, whose abstract we read. A search summary of Trautman's thesis says that individual prediction and its uncertainty are not the cause of freezing. We did not read that in a primary source, so we do not repeat it as a claim.
- **Trajectron++ numbers.** These may be optimistic because of the `np.gradient` future leak noted in the MGF appendix. We found no corrected figures.
- **Paired seeds.** We found no paper that prescribes them. It is our design choice.
- **Social LSTM numbers.** They are contested in the literature (see [36]). Use Gupta's reproduction if a number is needed.
- **Vovk** (algorithmic learning in a random world) was not checked. Use Angelopoulos & Bates instead.
- **Not found quickly:** a paper applying TabPFN or another tabular foundation model to robot or pedestrian spatial prediction. The closest are TabPFN-TS [14] (time series) and the TabPFN-3 time-series checkpoint [11].
