#!/usr/bin/env bash
# Medium / macro-F1 友好预设：manual 类权抬高 Medium + 关闭平衡采样，避免与 class_weight auto 双叠。
# 在 vote_train_glevel_multimodal 基线上追加；仍须与官方或 train_fixed 路径等自行 export 一致。
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

export MM_MEDIUM_FOCUS=1
export OUTPUT_MODEL="${OUTPUT_MODEL:-${_ROOT}/best_model_glevel_multimodal_medium_focus.pth}"
export LOSS_PLOT_PATH="${LOSS_PLOT_PATH:-${_ROOT}/loss_img/loss_glevel_multimodal_medium_focus.png}"
export VAL_ERRORS_CSV="${VAL_ERRORS_CSV:-${_ROOT}/logs/val_glevel_multimodal_medium_focus_errors.csv}"
export TEST_OUTPUT_CSV="${TEST_OUTPUT_CSV:-submission_glevel_multimodal_medium_focus.csv}"

echo "[run_multimodal_medium_focus] 使用 MM_MEDIUM_FOCUS=1（macro_f1 + manual CE + no_balanced_sampler）" >&2
bash "${_ROOT}/vote_train_glevel_multimodal.sh"
