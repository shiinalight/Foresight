#!/usr/bin/env bash
# Foresight on an Apple-silicon Mac (M1/M2/M3): environment, data, filter settings, tests.
# Safe to re-run: every step skips work that is already done.
#
#   bash scripts/mac/setup.sh
#
# Needs: macOS on Apple silicon, Python 3.11 or 3.12 (python3.12 from python.org, Homebrew or uv),
# internet access (pip packages, two small ETH/Hotel files from GitHub, TabPFN weights on first use),
# and a TabPFN licence token in the environment (TABPFN_TOKEN; see docs/MAC_SETUP.md). Never commit it.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd)
say() { printf '\n== %s\n' "$*"; }

say "1/7 machine"
[[ "$(uname -s)" == "Darwin" ]] || { echo "This script is for macOS."; exit 1; }
[[ "$(uname -m)" == "arm64" ]] || echo "WARNING: not Apple silicon; the GPU (MPS) will not be available."
sysctl -n machdep.cpu.brand_string 2>/dev/null || true
echo "memory: $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"

say "2/7 python environment (.venv)"
PY=${PYTHON:-}
if [[ -z "$PY" ]]; then
  for c in python3.12 python3.11 python3; do command -v "$c" >/dev/null && { PY=$c; break; }; done
fi
"$PY" -c 'import sys; v=sys.version_info; assert (3,11) <= v[:2] <= (3,12), f"need Python 3.11 or 3.12, found {sys.version}"'
[[ -d .venv ]] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt "torch>=2.6" mlx tabulate
python - <<'EOF'
import torch, tabpfn
print("torch", torch.__version__, "| MPS available:", torch.backends.mps.is_available(), "| tabpfn", tabpfn.__version__)
EOF

say "3/7 TabPFN licence token"
if [[ -z "${TABPFN_TOKEN:-}" ]]; then
  echo "TABPFN_TOKEN is not set. Accept the licence at https://ux.priorlabs.ai, copy your token, then run:"
  echo "    export TABPFN_TOKEN=...   (add it to ~/.zshrc to keep it; never commit it)"
  echo "Setup continues; TabPFN steps will fail until the token is set."
fi

say "4/7 real pedestrian data (ETH and Hotel, from OpenTraj)"
base=https://raw.githubusercontent.com/crowdbotp/OpenTraj/master/datasets/ETH
for s in eth hotel; do
  mkdir -p "data/raw/$s"
  [[ -f "data/raw/$s/obsmat.txt" ]] || curl -fsSL "$base/seq_$s/obsmat.txt" -o "data/raw/$s/obsmat.txt"
done
shasum -a 256 -c - <<'EOF'
d452ae2185ecb1164c2fdf31e75f6236f4c2ffc02c751a6b2ae921740cbc60d1  data/raw/eth/obsmat.txt
2b8577595204a6b780b80258fd50da35adb7293e3e06ad6f8e1b60965e7dde6e  data/raw/hotel/obsmat.txt
EOF
for s in eth hotel; do
  [[ -f "data/$s/tracks.csv" ]] || python -m foresight.domains.eth_ucy --raw "data/raw/$s" --out "data/$s"
done

say "5/7 simulated crowd bank (crossing crowd, 80 episodes; takes a few minutes)"
[[ -f data/crowd_cross/tracks.csv ]] || python -m foresight.domains.crowd --episodes 80 --seed 0 --switch-prob 0.5 \
    --goal-mode cross --out data/crowd_cross > /dev/null
# plaza bank read by tools/check_mps.py and tools/latency.py (DATA.md: 80 episodes, 36 switch); found missing in Nooshin's smoke test
[[ -f data/crowd/tracks.csv ]] || python -m foresight.domains.crowd --episodes 80 --seed 0 --switch-prob 0.5 --out data/crowd > /dev/null

say "6/7 velocity-filter settings (tuned on training data only, as on the main laptop; D-26)"
[[ -f data/eth/kalman.json ]] || python tools/tune_kalman.py data/hotel --noise 0 --apply-to data/eth
[[ -f data/hotel/kalman.json ]] || python tools/tune_kalman.py data/eth --noise 0 --apply-to data/hotel
[[ -f data/crowd_cross/kalman.json ]] || python tools/tune_kalman.py data/crowd_cross --noise 0.1 --split

say "7/7 check that this machine reproduces the main laptop's data, then run the tests"
python - <<'EOF'
import json, pandas as pd
expected = {  # rows, episodes, mean x, mean y on the main laptop (Windows, x64)
    "eth": (8908, 13, 5.217764, 5.31641), "hotel": (6544, 13, 1.392984, -2.977364),
    "crowd_cross": (273360, 80, 0.013246, -0.020451)}
kal = {"eth": (0.1, 0.1), "hotel": (0.1, 0.1), "crowd_cross": (0.01, 0.05)}
ok = True
for d, (n, e, mx, my) in expected.items():
    t = pd.read_csv(f"data/{d}/tracks.csv")
    same = len(t) == n and t.episode.nunique() == e and abs(t.x.mean() - mx) < 1e-3 and abs(t.y.mean() - my) < 1e-3
    k = json.load(open(f"data/{d}/kalman.json")); ksame = (k["kalman_q"], k["kalman_r"]) == kal[d]
    print(f"{d:12s} data {'MATCHES' if same else 'DIFFERS'}  ({len(t)} rows)   filter q={k['kalman_q']} r={k['kalman_r']} "
          f"{'MATCHES' if ksame else 'DIFFERS'}")
    ok &= same and ksame
print("ALL MATCH" if ok else "SOMETHING DIFFERS: report this before running experiments (floating-point "
      "differences between Apple silicon and x64 can change a simulated crowd)")
EOF
python -m pytest -q -p no:cacheprovider 2>&1 | tail -3
echo
echo "Setup done. Next: bash scripts/mac/smoke.sh   (about 10-20 minutes)"
