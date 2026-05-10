#!/usr/bin/env bash
# 大规模 GPU 扫参：调用 tools/run_glevel_gpu_combo_sweep.sh，默认 magnus + 多组合 × 多种子。
# 日志：experiments/glevel_improvement_plan/mass_gpu_hunt.log
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

_MAGNUS="${GLEVEL_CUDA_PYTHON:-/home/emo/anaconda3/envs/magnus/bin/python}"
_STAMP="$(date +%Y%m%d_%H%M%S)"
HUNT_DIR="${HUNT_DIR:-${_ROOT}/experiments/gpu_combo_sweep/mass_${_STAMP}}"
LOG="${_ROOT}/experiments/glevel_improvement_plan/mass_gpu_hunt.log"
PID_FILE="${_ROOT}/experiments/glevel_improvement_plan/mass_gpu_hunt.pid"
mkdir -p "$(dirname "${LOG}")"

# 全组合（可 export COMBOS=... 缩小）；种子池可 export SEEDS="37 42 ..."
COMBOS="${COMBOS:-S_ref_plateau,S_ref_sel_acc,S_ref_cosine,S_plateau_ln,S_step_ln,AT_plateau,AT_plateau_ln,AT_step_ln}"
SEEDS="${SEEDS:-37 10 42 99 73 28 5}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-8}"

# 共享集群建议：GPU_IDS=0,5,6 MAX_PARALLEL_JOBS=3 … 仅占用空闲卡
GPU_IDS="${GPU_IDS:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LAUNCH_STAGGER_SEC="${LAUNCH_STAGGER_SEC:-2}"
OOM_SAFE_BATCH="${OOM_SAFE_BATCH:-0}"

{
  echo "[mass_gpu_hunt] start $(date -Iseconds) HUNT_DIR=${HUNT_DIR}"
  echo "[mass_gpu_hunt] COMBOS=${COMBOS}"
  echo "[mass_gpu_hunt] SEEDS=${SEEDS}"
  echo "[mass_gpu_hunt] MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}"
  echo "[mass_gpu_hunt] GPU_IDS=${GPU_IDS:-<auto 0..PAR-1>}"
  echo "[mass_gpu_hunt] NUM_WORKERS=${NUM_WORKERS} LAUNCH_STAGGER_SEC=${LAUNCH_STAGGER_SEC} OOM_SAFE_BATCH=${OOM_SAFE_BATCH}"
} >>"${LOG}"

nohup env \
  GLEVEL_CUDA_PYTHON="${_MAGNUS}" \
  HUNT_DIR="${HUNT_DIR}" \
  COMBOS="${COMBOS}" \
  SEEDS="${SEEDS}" \
  MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS}" \
  GPU_IDS="${GPU_IDS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  LAUNCH_STAGGER_SEC="${LAUNCH_STAGGER_SEC}" \
  OOM_SAFE_BATCH="${OOM_SAFE_BATCH}" \
  NANBEIGE_TEXT="${NANBEIGE_TEXT:-1}" \
  bash "${_ROOT}/tools/run_glevel_gpu_combo_sweep.sh" >>"${LOG}" 2>&1 &
echo $! >"${PID_FILE}"
echo "[mass_gpu_hunt] PID=$(cat "${PID_FILE}") log=${LOG} metrics=${HUNT_DIR}/combo_sweep_metrics.csv"
