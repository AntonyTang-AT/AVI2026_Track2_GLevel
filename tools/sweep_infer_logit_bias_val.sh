#!/usr/bin/env bash
# Phase A：在验证集上扫 --infer_logit_bias 的 Medium 分量（0,b,0），写出 CSV。
# 用法：在仓库根目录
#   bash tools/sweep_infer_logit_bias_val.sh
# 可选环境变量：PYTHON, OUT_CSV, TEST_MODEL, BIAS_LIST（空格分隔）
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

PYTHON="${PYTHON:-python}"
OUT_CSV="${OUT_CSV:-${_ROOT}/experiments/glevel_improvement_plan/phaseA_infer_bias_sweep.csv}"
TEST_MODEL="${TEST_MODEL:-/data/emo/glevel_runs/archives/nb_to58_sweep/round1/seed37/best.pth}"
BIAS_LIST="${BIAS_LIST:-0.55 0.60 0.65 0.70 0.75 0.80 0.85}"

mkdir -p "$(dirname "${OUT_CSV}")"
echo "infer_bias_medium,val_ce,val_acc,val_macro_f1,val_bal_acc,val_pred_classes" >"${OUT_CSV}"

export NANBEIGE_TEXT="${NANBEIGE_TEXT:-1}"
export TEXT_DIM="${TEXT_DIM:-2560}"
export FEAT_TRAIN="${FEAT_TRAIN:-/data/Super-Lu/dataset/train_feature}"
export FEAT_VAL="${FEAT_VAL:-/data/Super-Lu/dataset/val_feature}"
export FEAT_TEST="${FEAT_TEST:-/data/Super-Lu/dataset/test_feature}"
export TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${_ROOT}/data/text_nb}"
export TEXT_VAL_DIR="${TEXT_VAL_DIR:-${_ROOT}/data/text_nb_val}"
export TEXT_TEST_DIR="${TEXT_TEST_DIR:-${_ROOT}/data/test_nb}"
export TEST_CSV="${TEST_CSV:-/data/Super-Lu/dataset/test_data_basic_information.csv}"
export TEST_MODEL
export TEST_OUTPUT_CSV="/tmp/glevel_inferbias_sweep_discard.csv"

BASE_OPT="--g_level_int_encoding one --glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001 --label_smoothing 0.05 --select_best balanced_acc --cross_modal_attn --cross_modal_layers 1 --modality_dropout_p 0.12 --scheduler_min_lr 1e-6 --sampler_medium_boost 1.5 --seed 37"

for b in ${BIAS_LIST}; do
  export GLEVEL_OPT="${BASE_OPT} --infer_logit_bias 0,${b},0"
  line="$(bash "${_ROOT}/vote_test_glevel.sh" 2>&1 | grep '\[only_test:single\]' || true)"
  # shellcheck disable=SC2001
  metrics="$(echo "${line}" | sed -n 's/.*Val CE=\([0-9.]*\) acc=\([0-9.]*\) macro_f1=\([0-9.]*\) bal_acc=\([0-9.]*\) val_pred_classes=\([0-9]*\).*/\1,\2,\3,\4,\5/p')"
  if [[ -z "${metrics}" ]]; then
    echo "[sweep] WARN b=${b} parse_fail raw=${line}" >&2
    echo "${b},,,,," >>"${OUT_CSV}"
  else
    echo "${b},${metrics}" >>"${OUT_CSV}"
  fi
done

echo "[sweep_infer_logit_bias_val] wrote ${OUT_CSV}" >&2
