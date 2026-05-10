#!/usr/bin/env bash
# Phase B：shared_mlp 扩展网格（LN / StepLR / cosine / select_best）。
# 用法: export HUNT_DIR=/data/emo/glevel_runs/phaseB_run1 && export HUNT_DIR && bash tools/run_glevel_phaseB_grid.sh
# 种子: export PHASE_B_SEEDS="37 10 28 42 73 99 71"
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

export GLEVEL_CUDA_PYTHON="${GLEVEL_CUDA_PYTHON:-/home/emo/txcao/anaconda3/envs/avi2026/bin/python}"
export PYTHON="${GLEVEL_CUDA_PYTHON}"
export NANBEIGE_TEXT="${NANBEIGE_TEXT:-1}"

RUN_TAG="${PHASE_B_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export HUNT_DIR="${HUNT_DIR:-/data/emo/glevel_runs/phaseB_${RUN_TAG}}"
case "${HUNT_DIR}" in
  /*) ;;
  *) HUNT_DIR="${_ROOT}/${HUNT_DIR}" ;;
esac
export HUNT_DIR

export COMBOS="${PHASE_B_COMBOS:-S_ref_plateau,S_ref_sel_acc,S_ref_step,S_ref_cosine,S_plateau_ln,S_plateau_ln_sel_acc,S_step_ln}"
export SEEDS="${PHASE_B_SEEDS:-37 10 28 42 73 99 71}"
unset OOM_SAFE_BATCH
export BATCH_SIZE="${PHASE_B_BATCH_SIZE:-32}"
export EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-20}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-4}"
export LAUNCH_STAGGER_SEC="${LAUNCH_STAGGER_SEC:-3}"

echo "[phaseB] HUNT_DIR=${HUNT_DIR}" >&2
echo "[phaseB] COMBOS=${COMBOS}" >&2
echo "[phaseB] SEEDS=${SEEDS}" >&2

bash "${_ROOT}/tools/run_glevel_gpu_combo_sweep.sh"
