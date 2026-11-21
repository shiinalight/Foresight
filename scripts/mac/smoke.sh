#!/usr/bin/env bash
# Measure whether this Mac can run the Foresight experiments, and how fast. ~10-20 minutes.
#
#   bash scripts/mac/smoke.sh
#
# Writes docs/runs/mac_smoke.md (machine, versions, GPU check, TabPFN timings, a 2-crowd robot run per
# planner). Commit that one file to a branch and push it, so the main laptop can decide what runs where.
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p docs/runs results/mac_smoke
OUT=docs/runs/mac_smoke.md
LOG=results/mac_smoke
: "${TABPFN_TOKEN:?Set TABPFN_TOKEN first (see docs/MAC_SETUP.md)}"

echo "1/4 GPU check (MPS vs CPU give the same predictions)"
python tools/check_mps.py 2>&1 | tee "$LOG/check_mps.txt"

echo "2/4 TabPFN timing (cache build and prediction, per model and context size)"
python tools/latency.py 2>&1 | tee "$LOG/latency.txt"

echo "3/4 robot runs: 2 crowds x 6 planners, crossing crowd, 20 s each"
caffeinate -dims python run_closed.py --sim crowd --bank data/crowd_cross --goal-mode cross --switch \
    --planners cv vo lgbm_all tabpfn tabpfn_live oracle --seeds 9000 9001 --duration 20 --box-q 0.10 \
    --tag macsmoke 2>&1 | tee "$LOG/closed.txt"

echo "4/4 summary -> $OUT"
python - "$OUT" "$LOG" <<'EOF'
import glob, platform, subprocess, sys, pandas as pd, torch, tabpfn
out, log = sys.argv[1], sys.argv[2]
chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) // 2**30
rows = pd.concat([pd.read_csv(f) for f in glob.glob("results/closed/*macsmoke*.csv") if "robotfeat" not in f])
cols = [c for c in ["planner", "seed", "time_to_goal", "intrusion_s", "min_gap", "collision_steps",
                    "pred_ms_per_plan", "wall_s"] if c in rows]
lines = ["# Mac smoke test", "", f"- Machine: {chip}, {mem} GB, macOS {platform.mac_ver()[0]}",
         f"- torch {torch.__version__}, MPS available: {torch.backends.mps.is_available()}, tabpfn {tabpfn.__version__}",
         "", "## GPU check (tools/check_mps.py)", "", "```", open(f"{log}/check_mps.txt").read().strip()[-2000:], "```",
         "", "## TabPFN timing (tools/latency.py)", "", "```", open(f"{log}/latency.txt").read().strip()[-3000:], "```",
         "", "## Robot runs (2 crowds, 20 s, crossing crowd)", "",
         rows[cols].round(2).to_markdown(index=False) if hasattr(rows, "to_markdown") else rows[cols].round(2).to_string(index=False),
         "", "Per-plan time to compare with the main laptop (x64 emulation, CPU): frozen TabPFN ~200-500 ms, "
         "live TabPFN ~1,600-1,700 ms, LightGBM ~10-300 ms."]
open(out, "w").write("\n".join(lines) + "\n")
print(open(out).read())
EOF
echo
echo "Done. Please commit and push the summary:"
echo "    git checkout -b nooshin/mac-smoke && git add docs/runs/mac_smoke.md && git commit -m 'Mac smoke test' && git push -u origin nooshin/mac-smoke"
