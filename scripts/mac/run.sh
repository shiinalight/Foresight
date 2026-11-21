#!/usr/bin/env bash
# Run any Foresight robot-run command on the Mac without it falling asleep, inside the project's .venv.
#
#   bash scripts/mac/run.sh <arguments for run_closed.py>
#   e.g. bash scripts/mac/run.sh --sim crowd --bank data/crowd_cross --goal-mode cross --switch \
#            --planners tabpfn_live --seeds 3000 3001 --tag mac
#
# The exact T5 command (planners, seeds, settings) is published in PLAN.md / docs/DECISIONS.md before
# launch; copy it from there. Results land in results/closed/<run_id>-<tag>.csv; keep the lid open and
# the charger in.
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source .venv/bin/activate
: "${TABPFN_TOKEN:?Set TABPFN_TOKEN first (see docs/MAC_SETUP.md)}"
export PYTHONUNBUFFERED=1
caffeinate -dims python run_closed.py "$@"
