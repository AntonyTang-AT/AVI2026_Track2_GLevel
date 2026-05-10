#!/usr/bin/env bash
# Phase A：仅 S_ref_plateau，batch=32，EARLY_STOP_MIN_EPOCHS=20，固定种子 + 可复现随机种子池。
# 用法（工程根）:
#   export HUNT_DIR=/data/emo/glevel_runs/phaseA_run1
#   export HUNT_DIR
#   bash tools/run_glevel_phaseA_baseline_sweep.sh
# 随机额外种子数量（默认 40）: export PHASE_A_EXTRA_RANDOM=40
# meta-seed: export PHASE_A_META_SEED=20260509
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

export GLEVEL_CUDA_PYTHON="${GLEVEL_CUDA_PYTHON:-/home/emo/txcao/anaconda3/envs/avi2026/bin/python}"
export PYTHON="${GLEVEL_CUDA_PYTHON}"
export NANBEIGE_TEXT="${NANBEIGE_TEXT:-1}"

RUN_TAG="${PHASE_A_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export HUNT_DIR="${HUNT_DIR:-/data/emo/glevel_runs/phaseA_${RUN_TAG}}"
case "${HUNT_DIR}" in
  /*) ;;
  *) HUNT_DIR="${_ROOT}/${HUNT_DIR}" ;;
esac
export HUNT_DIR

EXTRA_RANDOM="${PHASE_A_EXTRA_RANDOM:-40}"
META_SEED="${PHASE_A_META_SEED:-20260509}"

export SEEDS="$(
  python3 <<PY
import random
fixed = [37, 10, 28, 5, 42, 73]
random.seed(int("${META_SEED}"))
pool = [x for x in range(1, 100000) if x not in set(fixed)]
k = int("${EXTRA_RANDOM}")
extra = random.sample(pool, min(k, len(pool)))
print(*(fixed + extra))
PY
)"

export COMBOS="S_ref_plateau"
unset OOM_SAFE_BATCH
export BATCH_SIZE="${PHASE_A_BATCH_SIZE:-32}"
export EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-20}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-40}"
export NUM_EPOCHS="${NUM_EPOCHS:-200}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-4}"
export LAUNCH_STAGGER_SEC="${LAUNCH_STAGGER_SEC:-3}"

echo "[phaseA] HUNT_DIR=${HUNT_DIR}" >&2
echo "[phaseA] BATCH_SIZE=${BATCH_SIZE} EXTRA_RANDOM=${EXTRA_RANDOM} META_SEED=${META_SEED}" >&2
echo "[phaseA] n_seeds=$(wc -w <<<"${SEEDS}")" >&2

bash "${_ROOT}/tools/run_glevel_gpu_combo_sweep.sh"
