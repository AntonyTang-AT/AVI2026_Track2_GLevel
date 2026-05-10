#!/usr/bin/env bash
# 对照 select_best：balanced_acc / macro_f1 / val_ce（小验证集上指标与保存的 best 可能不一致）。
# 通过覆盖 GLEVEL_OPT 中 --select_best，保留 scripts/glevel_train_multimodal.sh 其余默认（含 cross_modal 等）。
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

BASE_PRESET="--glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001 --label_smoothing 0.05 --cross_modal_attn --cross_modal_layers 1 --modality_dropout_p 0.12 --seed 42 --scheduler_min_lr 1e-6"
LOG_DIR="${LOG_DIR:-${_ROOT}/logs/ablation_select_best}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

run_one() {
  local tag="$1"
  local sel="$2"
  local log="$LOG_DIR/${TS}_${tag}.log"
  echo "[ablation_select_best] === $tag select_best=$sel → $log ===" >&2
  {
    echo "=== $tag at $(date -Iseconds) ==="
    export GLEVEL_OPT="$BASE_PRESET --select_best $sel"
    export OUTPUT_MODEL="${OUTPUT_MODEL:-${_ROOT}/best_model_mm_sb_${tag}.pth}"
    export LOSS_PLOT_PATH="${LOSS_PLOT_PATH:-${_ROOT}/loss_img/loss_mm_sb_${tag}.png}"
    export VAL_ERRORS_CSV="${VAL_ERRORS_CSV:-${_ROOT}/logs/val_mm_sb_${tag}.csv}"
    unset MM_TEMPORAL || true
    bash "${_ROOT}/scripts/glevel_train_multimodal.sh"
  } 2>&1 | tee "$log"
}

run_one "balanced_acc" "balanced_acc"
run_one "macro_f1" "macro_f1"
run_one "val_ce" "val_ce"

echo "[ablation_select_best] 完成。汇总:" >&2
echo "  grep -E '\\[metrics_line\\]|Best checkpoint' $LOG_DIR/${TS}_*.log" >&2
