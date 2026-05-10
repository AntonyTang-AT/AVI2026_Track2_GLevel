#!/usr/bin/env bash
# 后台启动「多轮随机 train/val + GPU combo sweep」（与 run_mass_gpu_hunt_background.sh 对应，多一层 CV）。
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

_MAGNUS="${GLEVEL_CUDA_PYTHON:-/home/emo/anaconda3/envs/magnus/bin/python}"
_STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_HUNT_DIR="${BASE_HUNT_DIR:-${_ROOT}/experiments/gpu_combo_sweep/cv_mass_${_STAMP}}"
LOG="${_ROOT}/experiments/glevel_improvement_plan/mass_gpu_hunt_cv.log"
PID_FILE="${_ROOT}/experiments/glevel_improvement_plan/mass_gpu_hunt_cv.pid"
mkdir -p "$(dirname "${LOG}")"

COMBOS="${COMBOS:-S_ref_plateau,S_ref_sel_acc,S_ref_cosine,S_plateau_ln,S_step_ln,AT_plateau,AT_plateau_ln,AT_step_ln}"
SEEDS="${SEEDS:-37 10 42 99 73 28 5}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-8}"
GPU_IDS="${GPU_IDS:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LAUNCH_STAGGER_SEC="${LAUNCH_STAGGER_SEC:-2}"
OOM_SAFE_BATCH="${OOM_SAFE_BATCH:-0}"

SPLIT_ROUNDS="${SPLIT_ROUNDS:-5}"
SPLIT_BASE_SEED="${SPLIT_BASE_SEED:-1000}"
VAL_HOLDOUT_RATIO="${VAL_HOLDOUT_RATIO:-}"
if [[ -z "${VAL_HOLDOUT_RATIO}" ]]; then
  VAL_HOLDOUT_N="${VAL_HOLDOUT_N:-80}"
else
  VAL_HOLDOUT_N="${VAL_HOLDOUT_N:-}"
fi

{
  echo "[mass_gpu_hunt_cv] start $(date -Iseconds) BASE_HUNT_DIR=${BASE_HUNT_DIR}"
  echo "[mass_gpu_hunt_cv] SPLIT_ROUNDS=${SPLIT_ROUNDS} VAL_HOLDOUT_N=${VAL_HOLDOUT_N} VAL_HOLDOUT_RATIO=${VAL_HOLDOUT_RATIO}"
  echo "[mass_gpu_hunt_cv] COMBOS=${COMBOS} SEEDS=${SEEDS}"
  echo "[mass_gpu_hunt_cv] MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS} GPU_IDS=${GPU_IDS}"
} >>"${LOG}"

nohup env \
  GLEVEL_CUDA_PYTHON="${_MAGNUS}" \
  BASE_HUNT_DIR="${BASE_HUNT_DIR}" \
  SPLIT_ROUNDS="${SPLIT_ROUNDS}" \
  SPLIT_BASE_SEED="${SPLIT_BASE_SEED}" \
  VAL_HOLDOUT_N="${VAL_HOLDOUT_N}" \
  VAL_HOLDOUT_RATIO="${VAL_HOLDOUT_RATIO}" \
  COMBOS="${COMBOS}" \
  SEEDS="${SEEDS}" \
  MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS}" \
  GPU_IDS="${GPU_IDS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  LAUNCH_STAGGER_SEC="${LAUNCH_STAGGER_SEC}" \
  OOM_SAFE_BATCH="${OOM_SAFE_BATCH}" \
  NANBEIGE_TEXT="${NANBEIGE_TEXT:-1}" \
  bash "${_ROOT}/tools/run_glevel_gpu_combo_sweep_cv.sh" >>"${LOG}" 2>&1 &
echo $! >"${PID_FILE}"
echo "[mass_gpu_hunt_cv] PID=$(cat "${PID_FILE}") log=${LOG} BASE_HUNT_DIR=${BASE_HUNT_DIR}"
