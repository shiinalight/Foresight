# TabPFN policy: how the pilot and DAgger are judged (written before the pilot; commit 6b9ffd4, 03:38 Berlin)

These are the operational details of `docs/TABPFN_POLICY.md`, fixed before any policy episode ran on the pod.

**Policy context:**
- Teacher rows come from tabpfn_split q0.35 on seeds 3300–3339 (T5 config, batch offset, L4 pod; `--log-cands`).
- The context keeps every positive (the teacher's pick) and 3 negatives per positive, sampled at random with seed 0 (`--neg-ratio 3`).

**Pilot:** tabpfn_policy on test crowds 3400–3404.
- **It passes** if the robot reaches the goal in at least 4 of 5 crowds and picks the teacher's move at ≥ 60% of the replans it visits (`policy_agree`).
- **Otherwise it "drifts".**

**DAgger round** (at most one):
- Run the policy on teacher seeds 3300–3309 with `--log-cands`. Each visited state gets the teacher's pick as its label.
- Add those rows to the context and rerun the pilot on 3400–3404.
- If it still fails, stop and report it as tried. The test crowds are never used as context.

**Full test:**
- tabpfn_policy and the teacher on test crowds 3400–3419, on the same device.
- Parity is judged by the rule in `docs/TABPFN_POLICY.md`: goal reached within 10 points, time ≤ +2.6 s, gap ≥ −0.158 m, moving-collision rate ≤ +0.244.
- The hard-check override rate is reported.
