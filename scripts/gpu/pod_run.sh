#!/usr/bin/env bash
# GPU timing on a rented CUDA pod (RunPod PyTorch template). Run from the unpacked repo root on the pod:
#   bash scripts/gpu/pod_run.sh
# Expects: weights in /workspace/tabpfn-weights (tabpfn-v3.5-*.safetensors, copied from the laptop; no download),
#          data/crowd_cross_t2 unpacked (T5 bank), results/latency_parity_cpu.npz (laptop reference quantiles).
# Writes:  results/latency_cuda_<gpu>.csv/.json, results/closed/*-gpu-split.csv, results/pod_env.txt
set -euo pipefail
export TABPFN_MODEL_CACHE_DIR=/workspace/tabpfn-weights TABPFN_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1 POLARS_SKIP_CPU_CHECK=1

# venv on top of the template's CUDA torch (the system Python refuses pip installs, PEP 668)
[ -d /workspace/venv ] || python -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate
pip install -q "tabpfn==9.2.0" numpy pandas pyarrow scikit-learn lightgbm matplotlib
python -c "import torch; assert torch.cuda.is_available(), 'torch lost CUDA after pip install'"
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; python -c "import torch,tabpfn,importlib.metadata as m;print('torch',torch.__version__,'cuda',torch.version.cuda,'tabpfn',m.version('tabpfn'))"; nproc; } | tee results/pod_env.txt

# 1. TabPFN call timings for the shipped design (and 3.5-1), plus parity against the laptop CPU quantiles
python tools/latency_gpu.py --reps 30 --parity results/latency_parity_cpu.npz

# 2. Real loop plan time: T5 config, primary planner (tabpfn_split q 0.15, batch offset), 3 seeds incl. showcase 3013
python run_closed.py --sim crowd --robot-repulsion 2.0 --ignore-robot 0.2 --people 24 --goal-mode cross \
  --bank data/crowd_cross_t2 --bank-exclude-regime erratic --switch --switch-all --switch-to erratic --switch-t 5 \
  --frozen-ctx fill --context 2000 --seeds 3013 3001 3002 --planners tabpfn_split --box-q 0.15 \
  --offset --offset-rule batch --offset-eta-b 0.05 --tag gpu-split
echo DONE
