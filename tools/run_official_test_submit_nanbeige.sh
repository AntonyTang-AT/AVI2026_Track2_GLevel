#!/usr/bin/env bash
# 官方测试集：FEAT_TEST=test_feature(512) + 工程内 data/test_nb(2560) + seed37 权重 → submission
# 前置：已用 tools/extract_nanbeige_one_click.py 生成 data/test_nb（与 test_text 对齐）
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

PYTHON="${PYTHON:-python}"
export NANBEIGE_TEXT=1 TEXT_DIM=2560
export FEAT_TRAIN="${FEAT_TRAIN:-/data/Super-Lu/dataset/train_feature}"
export FEAT_VAL="${FEAT_VAL:-/data/Super-Lu/dataset/val_feature}"
export FEAT_TEST="${FEAT_TEST:-/data/Super-Lu/dataset/test_feature}"
export TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${_ROOT}/data/text_nb}"
export TEXT_VAL_DIR="${TEXT_VAL_DIR:-${_ROOT}/data/text_nb_val}"
export TEXT_TEST_DIR="${TEXT_TEST_DIR:-${_ROOT}/data/test_nb}"
export TEST_CSV="${TEST_CSV:-/data/Super-Lu/dataset/test_data_basic_information.csv}"
export TEST_MODEL="${TEST_MODEL:-${_ROOT}/experiments/nb_to58_sweep/round1/seed37/best.pth}"
export TEST_OUTPUT_CSV="${TEST_OUTPUT_CSV:-${_ROOT}/reports/submissions/submission_glevel_test_official.csv}"

WINNER_OPT="--g_level_int_encoding one --glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001 --label_smoothing 0.05 --select_best balanced_acc --cross_modal_attn --cross_modal_layers 1 --modality_dropout_p 0.12 --scheduler_min_lr 1e-6 --sampler_medium_boost 1.5 --seed 37"
export GLEVEL_OPT="${GLEVEL_OPT:-$WINNER_OPT}"
# 可选：抬高 Medium 类 logit，缓解测试集 argmax 塌向 High（须在 val 上扫 grid，勿盲用）。
# 本地 coarse sweep：约 0,0.7,0 时 val_bal_acc 由 0.5414→0.5610（val_acc 0.5873→0.6032）。
if [ -n "${INFER_LOGIT_BIAS:-}" ]; then
  export GLEVEL_OPT="${GLEVEL_OPT} --infer_logit_bias ${INFER_LOGIT_BIAS}"
fi

echo "[run_official_test_submit_nanbeige] TEST_OUTPUT_CSV=$TEST_OUTPUT_CSV" >&2
bash "${_ROOT}/scripts/glevel_test.sh"
