#!/usr/bin/env bash
# 多模态 g_level（SharedMLPwEnsemble：音 + 视 + 文），与常见「~51%」官方 train/val 条件对齐；
# 融入新版训练策略：更长早停/最小 epoch、Plateau min_lr、详细日志、正则与跨模态注意力。
#
# 与 text_gru 单模态区别：本脚本强制 --glevel_arch shared_mlp，请勿再 export 含 text_gru 的 GLEVEL_OPT。
#
# 用法（服务器项目根、已激活带 CUDA 的 venv，例如 avi2026）：
#   unset GLEVEL_OPT  # 若 shell 里还留着旧参数，先清掉
#   export NUM_WORKERS=0   # 可选：与 train_task2_glevel 内采样器 generator/worker_init 叠加，最大化复现性
#   bash scripts/glevel_train_multimodal.sh
#
# 可选：
#   export NANBEIGE_TEXT=1   # 全链路 Nanbeige 文本维（与 vote_train 一致）；默认 0 则用 FEAT/*/text（SigLIP 768）
#   export MM_TEMPORAL=1   # 额外打开 6 题时序 GRU（更强但更吃数据，默认关）
#   export MM_MEDIUM_BOOST=1  # 温和抬高 Medium：保留 balanced_acc + 平衡采样，仅 --sampler_medium_boost 1.5（计划书阶段二「或」分支）
#   export MM_MEDIUM_FOCUS=1  # 强 Medium：select_best=macro_f1 + manual 类权 + 关平衡采样（与 MM_MEDIUM_BOOST 二选一即可）
#   export MM_BIDIRECTIONAL=1   # 时序 GRU 双向 + 注意力池化（须同时 MM_TEMPORAL=1）
#   export RUN_TEST_AFTER=1  # 训练结束后用同一 GLEVEL_OPT 跑 scripts/glevel_test.sh（需测试特征齐全）
#   export ROUTE_A_PREFLIGHT=1  # 与 vote_train 相同预检
#
# 输出默认单独命名，避免覆盖 text_gru 的 checkpoint：
#   best_model_glevel_multimodal_plus.pth
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

if [[ "${TRAIN_CSV:-}" == *train_fixed* ]] || [[ "${VAL_CSV:-}" == *val_fixed* ]]; then
  echo "[glevel_train_multimodal] 提示: 当前 TRAIN_CSV/VAL_CSV 含 train_fixed/val_fixed，" \
    "与「官方 val ~51%」基线不是同一划分；若要可比请改用赛方 train_data.csv / val_data.csv。" >&2
fi

MM_SELECT_BEST="balanced_acc"
if [ "${MM_MEDIUM_FOCUS:-0}" = "1" ]; then
  MM_SELECT_BEST="macro_f1"
fi

MM_PRESET="--g_level_int_encoding one --glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001 --label_smoothing 0.05 --select_best ${MM_SELECT_BEST} --cross_modal_attn --cross_modal_layers 1 --modality_dropout_p 0.12 --seed 42 --scheduler_min_lr 1e-6"

if [ "${MM_TEMPORAL:-0}" = "1" ]; then
  MM_PRESET="${MM_PRESET} --temporal_gru --temporal_pool mean --temporal_dropout 0.12"
  echo "[glevel_train_multimodal] MM_TEMPORAL=1：已启用 temporal_gru" >&2
fi

if [ "${MM_BIDIRECTIONAL:-0}" = "1" ]; then
  MM_PRESET="${MM_PRESET} --temporal_bidirectional --temporal_attn_pool"
  echo "[glevel_train_multimodal] MM_BIDIRECTIONAL=1：双向 GRU + 注意力池化（建议已开 MM_TEMPORAL）" >&2
fi

if [ "${MM_MEDIUM_FOCUS:-0}" = "1" ]; then
  MM_PRESET="${MM_PRESET} --class_weight manual --class_weight_manual 1.0,2.0,1.0 --no_balanced_sampler"
  echo "[glevel_train_multimodal] MM_MEDIUM_FOCUS=1：macro_f1 + manual Medium 权重 + 无平衡采样" >&2
elif [ "${MM_MEDIUM_BOOST:-0}" = "1" ]; then
  MM_PRESET="${MM_PRESET} --sampler_medium_boost ${MM_SAMPLER_MEDIUM_BOOST:-1.5}"
  echo "[glevel_train_multimodal] MM_MEDIUM_BOOST=1：sampler_medium_boost=${MM_SAMPLER_MEDIUM_BOOST:-1.5}（保留 balanced_acc + 平衡采样）" >&2
fi

export GLEVEL_OPT="${GLEVEL_OPT:-$MM_PRESET}"
export OUTPUT_MODEL="${OUTPUT_MODEL:-best_model_glevel_multimodal_plus.pth}"
export LOSS_PLOT_PATH="${LOSS_PLOT_PATH:-./loss_img/loss_glevel_multimodal_plus.png}"
export TEST_OUTPUT_CSV="${TEST_OUTPUT_CSV:-${_ROOT}/reports/submissions/submission_glevel_multimodal_plus.csv}"
export VAL_ERRORS_CSV="${VAL_ERRORS_CSV:-./logs/val_glevel_multimodal_plus_errors.csv}"

echo "[glevel_train_multimodal] GLEVEL_OPT=${GLEVEL_OPT}" >&2
echo "[glevel_train_multimodal] OUTPUT_MODEL=${OUTPUT_MODEL} TRAIN_CSV=${TRAIN_CSV:-/data/Super-Lu/dataset/train_data.csv} VAL_CSV=${VAL_CSV:-/data/Super-Lu/dataset/val_data.csv}" >&2

bash "${_ROOT}/scripts/glevel_train.sh"

if [ "${RUN_TEST_AFTER:-0}" = "1" ]; then
  export TEST_MODEL="${OUTPUT_MODEL}"
  export TEST_OUTPUT_CSV
  echo "[glevel_train_multimodal] RUN_TEST_AFTER=1 → scripts/glevel_test.sh" >&2
  bash "${_ROOT}/scripts/glevel_test.sh"
fi
