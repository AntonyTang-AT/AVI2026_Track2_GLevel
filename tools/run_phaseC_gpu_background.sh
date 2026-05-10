#!/usr/bin/env bash
# Phase C：伪标签扩充 train + train_feat_fallback 到 FEAT_TEST；默认 GPU、conda magnus。
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

_MAGNUS_PY="${PHASE_C_PYTHON:-/home/emo/anaconda3/envs/magnus/bin/python}"
LOG="${_ROOT}/experiments/glevel_improvement_plan/phaseC_gpu_run.log"
PID_FILE="${_ROOT}/experiments/glevel_improvement_plan/phaseC_gpu_run.pid"
mkdir -p "$(dirname "${LOG}")"

echo "[run_phaseC_gpu_background] start $(date -Iseconds)" >>"${LOG}"

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" PYTHON="${_MAGNUS_PY}" NANBEIGE_TEXT="${NANBEIGE_TEXT:-1}" TEXT_DIM="${TEXT_DIM:-2560}" \
  FEAT_TRAIN="${FEAT_TRAIN:-/data/Super-Lu/dataset/train_feature}" FEAT_VAL="${FEAT_VAL:-/data/Super-Lu/dataset/val_feature}" FEAT_TEST="${FEAT_TEST:-/data/Super-Lu/dataset/test_feature}" \
  TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${_ROOT}/data/text_nb}" TEXT_VAL_DIR="${TEXT_VAL_DIR:-${_ROOT}/data/text_nb_val}" TEXT_TEST_DIR="${TEXT_TEST_DIR:-${_ROOT}/data/test_nb}" \
  TRAIN_CSV="${TRAIN_CSV:-${_ROOT}/experiments/glevel_improvement_plan/train_plus_pseudo_unanimous.csv}" \
  VAL_CSV="${VAL_CSV:-/data/Super-Lu/dataset/val_data.csv}" TEST_CSV="${TEST_CSV:-/data/Super-Lu/dataset/test_data_basic_information.csv}" \
  RATING_CSV="${RATING_CSV:-/data/Super-Lu/dataset/train_data.csv}" \
  NUM_EPOCHS="${PHASE_C_NUM_EPOCHS:-30}" LEARNING_RATE="${PHASE_C_LR:-3e-5}" EARLY_STOP_PATIENCE="${PHASE_C_ES:-15}" EARLY_STOP_MIN_EPOCHS="${PHASE_C_ES_MIN:-5}" \
  bash "${_ROOT}/tools/vote_train_glevel_pseudo_augment.sh" >>"${LOG}" 2>&1 &
echo $! >"${PID_FILE}"
echo "[run_phaseC_gpu_background] PID=$(cat "${PID_FILE}") CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} log=${LOG}"
