#!/usr/bin/env bash
# 后台启动「合并 train+val 池 + 宽 combo/seed + 多 slot 随机划分」搜索（见 run_glevel_gpu_combo_sweep.sh）。
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

_MAGNUS="${GLEVEL_CUDA_PYTHON:-/home/emo/anaconda3/envs/magnus/bin/python}"
_STAMP="$(date +%Y%m%d_%H%M%S)"
HUNT_DIR="${HUNT_DIR:-${_ROOT}/experiments/gpu_combo_sweep/pool_wide_${_STAMP}}"
LOG="${_ROOT}/experiments/glevel_improvement_plan/pool_merged_wide.log"
PID_FILE="${_ROOT}/experiments/glevel_improvement_plan/pool_merged_wide.pid"
mkdir -p "$(dirname "${LOG}")"

POOL_RANDOM_SPLITS="${POOL_RANDOM_SPLITS:-5}"
POOL_TRAIN_N="${POOL_TRAIN_N:-418}"
POOL_SPLIT_BASE_SEED="${POOL_SPLIT_BASE_SEED:-5000}"
SAVE_TOP_K_MODELS="${SAVE_TOP_K_MODELS:-10}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-8}"
GPU_IDS="${GPU_IDS:-}"
OOM_SAFE_BATCH="${OOM_SAFE_BATCH:-0}"

{
  echo "[pool_merged_wide] start $(date -Iseconds) HUNT_DIR=${HUNT_DIR}"
  echo "[pool_merged_wide] POOL_RANDOM_SPLITS=${POOL_RANDOM_SPLITS} POOL_TRAIN_N=${POOL_TRAIN_N}"
} >>"${LOG}"

nohup env \
  GLEVEL_CUDA_PYTHON="${_MAGNUS}" \
  HUNT_DIR="${HUNT_DIR}" \
  POOL_RANDOM_SPLITS="${POOL_RANDOM_SPLITS}" \
  POOL_TRAIN_N="${POOL_TRAIN_N}" \
  POOL_SPLIT_BASE_SEED="${POOL_SPLIT_BASE_SEED}" \
  PARTITION_ROUNDS=0 \
  SAVE_TOP_K_MODELS="${SAVE_TOP_K_MODELS}" \
  MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS}" \
  GPU_IDS="${GPU_IDS}" \
  OOM_SAFE_BATCH="${OOM_SAFE_BATCH}" \
  NANBEIGE_TEXT="${NANBEIGE_TEXT:-1}" \
  bash "${_ROOT}/tools/run_glevel_gpu_combo_sweep.sh" >>"${LOG}" 2>&1 &
echo $! >"${PID_FILE}"
echo "[pool_merged_wide] PID=$(cat "${PID_FILE}") log=${LOG} metrics=${HUNT_DIR}/combo_sweep_metrics.csv top=${HUNT_DIR}/top_models/"
