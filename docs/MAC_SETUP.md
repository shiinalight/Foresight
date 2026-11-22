# Running Foresight on an Apple-silicon Mac

For Nooshin, or her Claude session. Three scripts do everything; this page says what they do, what to check, and what to send back. A Windows `.bat` file can't run on macOS, so these are shell scripts.

## Why

The main laptop runs TabPFN on CPU under x64 emulation. Live TabPFN there needs about 1.7 s per decision, against the robot's 0.4 s replan, and a 1,000-row cache rebuild takes about 4.5 s per axis. An M3's GPU (MPS) should be much faster, but that's an estimate. The smoke test measures it, and its numbers decide which experiments move to the Mac.

## Steps

1. **Get the code.**
   ```
   git clone https://github.com/shiinalight/Robodetect.git && cd Robodetect
   ```
   Or `git pull` in an existing clone. Use `main` once it carries the current code; until then, run `git checkout darshak/core`. `PLAN.md` says which.
2. **TabPFN licence token.** Accept the licence at https://ux.priorlabs.ai and copy your token, then:
   ```
   export TABPFN_TOKEN=...
   ```
   Put that line in `~/.zshrc` to keep it. It's personal: never commit it or paste it into a file in the repo.
3. **Set up** (about 10–15 minutes; safe to re-run):
   ```
   bash scripts/mac/setup.sh
   ```
   It:
   - creates `.venv`;
   - installs the requirements plus PyTorch (with MPS) and `mlx`;
   - downloads the two ETH/Hotel files from OpenTraj and checks their fingerprints;
   - converts them;
   - generates the 80-episode crossing-crowd bank;
   - re-tunes the velocity filter on training data only, exactly as on the main laptop;
   - checks that the data matches the main laptop's;
   - runs the tests.

   **What to check:**
   - the line `MPS available: True`;
   - `ALL MATCH`;
   - the tests passing (127 at the time of writing, 1 expected failure).
4. **Smoke test** (about 10–20 minutes):
   ```
   bash scripts/mac/smoke.sh
   ```
   It runs:
   - the GPU check (MPS and CPU must give the same predictions);
   - TabPFN timing for each model and context size;
   - 2 short robot runs for each of 6 planners in the crossing crowd.

   It writes `docs/runs/mac_smoke.md`.
5. **Send the result back.** The script prints the exact commands: commit only `docs/runs/mac_smoke.md` to a branch `nooshin/mac-smoke` and push it.
6. **Larger runs**, only when the main laptop asks:
   ```
   bash scripts/mac/run.sh <arguments>
   ```
   This runs `run_closed.py` inside `.venv` and keeps the Mac awake. The exact command for each run (for example T5) is published in `PLAN.md` / `docs/DECISIONS.md` before launch. Keep the lid open and the charger in.

## If something differs

| Symptom | Meaning | What to do |
|---|---|---|
| `MPS available: False` | PyTorch without the Apple GPU, or not Apple silicon | `pip install --upgrade torch` inside `.venv`; check `uname -m` says `arm64` |
| Data `DIFFERS` | Floating-point differences between Apple silicon and x64 changed a simulated crowd | Report it before running experiments. Paired comparisons must never mix crowds from the two machines |
| GPU check shows large differences | TabPFN on MPS doesn't match CPU | Report it; run on CPU (`TABPFN_EXCLUDE_DEVICES=mps`) |
| TabPFN asks for a login or fails to download | Token missing or licence not accepted | Step 2 |
| A golden-checksum test fails | Same cause as "Data DIFFERS" | Report it with the failing test's name |
| `FileNotFoundError: data/crowd/meta.json` in the smoke test | `setup.sh` builds only `data/crowd_cross`, but `tools/check_mps.py` and `tools/latency.py` read the plaza bank `data/crowd` | Inside `.venv`: `python -m foresight.domains.crowd --episodes 80 --seed 0 --switch-prob 0.5 --out data/crowd` (matches `docs/DATA.md`: 80 episodes, 36 switch). Then re-run the smoke test |
| `TabPFNLicenseError: ... licence for tabpfn_3_5 has not been accepted` | Local weights need the licence accepted once per account; the hosted API does not ask for it | Open https://ux.priorlabs.ai/accept-license?hf_repo_id=tabpfn_3_5 while logged in, accept, re-run |
| `Segmentation fault: 11` when `run_closed.py` reaches `lgbm_all` | Probably LightGBM and PyTorch loading two OpenMP runtimes on macOS | Report it. The TabPFN planners run fine without it: `--planners tabpfn tabpfn_live oracle` |

## Measured on an M3 Pro (Sun 22 Nov)

Full numbers: `docs/runs/mac_smoke.md`. On the Apple GPU (MPS), with predictions identical to CPU (largest difference 1.3 cm):

| | M3 Pro (MPS) | Main laptop (x64 emulation, CPU) |
|---|---|---|
| TabPFN-3.5, 1 estimator: predict 32–64 people | 0.17–0.18 s | — |
| TabPFN-3.5-Fast, 1 estimator: predict 32–64 people | 0.09 s | — |
| Frozen TabPFN in the robot loop | 130–135 ms per plan | ~200–500 ms |
| Live TabPFN in the robot loop | 287–321 ms per plan (fits the 400 ms replan) | ~1,600–1,700 ms |

**Disk:** about 3 GB free is needed: `.venv` 1.6 GB, TabPFN weights 1.1 GB (in `~/Library/Caches/tabpfn`, delete them afterwards to free the space), data 0.2 GB.

## Rules

- **Results that get compared must come from one machine.** The same seeds give the same crowd only on the same platform, so Mac runs are compared with Mac runs.
- **Nothing in `results/` or `data/` is committed.** These folders are ignored; only `docs/runs/*.md` summaries come back.
- **Don't edit code on the Mac.** Report problems so the change happens in the repo, where the tests run.
