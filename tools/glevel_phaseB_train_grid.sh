#!/usr/bin/env bash
# Phase B：训练侧网格（CORAL / CE、class_weight、select_best）。默认仅打印命令（DRY_RUN=1）。
# GPU 示例：
#   CUDA_VISIBLE_DEVICES=0 PYTHON=/path/to/cuda/python PHASE_B_RUN=1 bash tools/glevel_phaseB_train_grid.sh
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"

DRY_RUN="${PHASE_B_DRY_RUN:-1}"
RUN="${PHASE_B_RUN:-0}"
JOBS="${PHASE_B_JOBS:-coral,ce,weight,select}" # coral|ce|weight|select(select=macro_f1)
NUM_EPOCHS="${PHASE_B_NUM_EPOCHS:-200}"
export NANBEIGE_TEXT="${NANBEIGE_TEXT:-1}"
export TEXT_DIM="${TEXT_DIM:-2560}"
export FEAT_TRAIN="${FEAT_TRAIN:-/data/Super-Lu/dataset/train_feature}"
export FEAT_VAL="${FEAT_VAL:-/data/Super-Lu/dataset/val_feature}"
export FEAT_TEST="${FEAT_TEST:-/data/Super-Lu/dataset/test_feature}"
export TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${_ROOT}/data/text_nb}"
export TEXT_VAL_DIR="${TEXT_VAL_DIR:-${_ROOT}/data/text_nb_val}"
export TEXT_TEST_DIR="${TEXT_TEST_DIR:-${_ROOT}/data/test_nb}"
export TRAIN_CSV="${TRAIN_CSV:-/data/Super-Lu/dataset/train_data.csv}"
export VAL_CSV="${VAL_CSV:-/data/Super-Lu/dataset/val_data.csv}"
export TEST_CSV="${TEST_CSV:-/data/Super-Lu/dataset/test_data_basic_information.csv}"
export RATING_CSV="${RATING_CSV:-/data/Super-Lu/dataset/train_data.csv}"

BASE="--g_level_int_encoding one --glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001 --label_smoothing 0.05 --cross_modal_attn --cross_modal_layers 1 --modality_dropout_p 0.12 --scheduler_min_lr 1e-6 --sampler_medium_boost 1.5"

_run() {
  local name="$1"
  shift
  export OUTPUT_MODEL="${_ROOT}/experiments/glevel_improvement_plan/phaseB_${name}_best.pth"
  export TEST_OUTPUT_CSV="${_ROOT}/external/submissions_peer/submission_phaseB_${name}.csv"
  export LOSS_PLOT_PATH="${_ROOT}/experiments/glevel_improvement_plan/phaseB_${name}_loss.png"
  export NUM_EPOCHS
  export GLEVEL_OPT="$*"
  echo "[phaseB] ${name} GLEVEL_OPT=${GLEVEL_OPT}" >&2
  if [[ "${RUN}" == "1" ]]; then
    mkdir -p "$(dirname "${OUTPUT_MODEL}")"
    bash "${_ROOT}/vote_train_glevel.sh"
  fi
}

IFS=',' read -r -a ARR <<<"${JOBS}"
for j in "${ARR[@]}"; do
  case "${j}" in
    coral)
      _run "coral_bal_seed42" ${BASE} "--glevel_loss coral --select_best balanced_acc --seed 42"
      ;;
    ce)
      _run "ce_bal_seed42" ${BASE} "--glevel_loss ce --select_best balanced_acc --seed 42"
      ;;
    weight)
      _run "ce_bal_classweight_auto_seed42" ${BASE} "--glevel_loss ce --select_best balanced_acc --class_weight auto --seed 42"
      ;;
    select)
      _run "ce_macrof1_seed42" ${BASE} "--glevel_loss ce --select_best macro_f1 --seed 42"
      ;;
    *)
      echo "[phaseB] unknown job ${j}" >&2
      ;;
  esac
done

if [[ "${RUN}" != "1" && "${DRY_RUN}" == "1" ]]; then
  echo "[phaseB] DRY RUN only. Set PHASE_B_RUN=1 to execute vote_train_glevel.sh jobs." >&2
fi
