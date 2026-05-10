#!/usr/bin/env bash
# 多模态 preset + CORAL 序关系损失（--glevel_loss coral）。其余同 vote_train_glevel_multimodal.sh。
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

MM_PRESET="--glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001 --label_smoothing 0.05 --select_best balanced_acc --cross_modal_attn --cross_modal_layers 1 --modality_dropout_p 0.12 --seed 42 --scheduler_min_lr 1e-6 --glevel_loss coral"
if [ "${MM_TEMPORAL:-0}" = "1" ]; then
  MM_PRESET="${MM_PRESET} --temporal_gru --temporal_pool mean --temporal_dropout 0.12"
fi
export GLEVEL_OPT="${GLEVEL_OPT:-$MM_PRESET}"
export OUTPUT_MODEL="${OUTPUT_MODEL:-best_model_glevel_multimodal_coral.pth}"
export LOSS_PLOT_PATH="${LOSS_PLOT_PATH:-./loss_img/loss_glevel_multimodal_coral.png}"
export TEST_OUTPUT_CSV="${TEST_OUTPUT_CSV:-submission_glevel_multimodal_coral.csv}"
export VAL_ERRORS_CSV="${VAL_ERRORS_CSV:-./logs/val_glevel_multimodal_coral_errors.csv}"

echo "[run_glevel_coral_multimodal] GLEVEL_OPT=$GLEVEL_OPT" >&2
bash "${_ROOT}/vote_train_glevel_multimodal.sh"
