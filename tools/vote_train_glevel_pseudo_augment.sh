#!/usr/bin/env bash
# Phase C：使用合并后的 train CSV + 训练集特征回退到 FEAT_TEST（测试 id 的 .npy）。
# 前置：python tools/merge_train_with_pseudo_test_rows.py ...
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

export NANBEIGE_TEXT="${NANBEIGE_TEXT:-1}"
export TEXT_DIM="${TEXT_DIM:-2560}"
export FEAT_TRAIN="${FEAT_TRAIN:-/data/Super-Lu/dataset/train_feature}"
export FEAT_VAL="${FEAT_VAL:-/data/Super-Lu/dataset/val_feature}"
export FEAT_TEST="${FEAT_TEST:-/data/Super-Lu/dataset/test_feature}"
export TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${_ROOT}/data/text_nb}"
export TEXT_VAL_DIR="${TEXT_VAL_DIR:-${_ROOT}/data/text_nb_val}"
export TEXT_TEST_DIR="${TEXT_TEST_DIR:-${_ROOT}/data/test_nb}"
export TRAIN_CSV="${TRAIN_CSV:-${_ROOT}/experiments/glevel_improvement_plan/train_plus_pseudo_unanimous.csv}"
export VAL_CSV="${VAL_CSV:-/data/Super-Lu/dataset/val_data.csv}"
export TEST_CSV="${TEST_CSV:-/data/Super-Lu/dataset/test_data_basic_information.csv}"
export RATING_CSV="${RATING_CSV:-/data/Super-Lu/dataset/train_data.csv}"

export OUTPUT_MODEL="${OUTPUT_MODEL:-${_ROOT}/experiments/glevel_improvement_plan/phaseC_pseudo_finetune_best.pth}"
export TEST_OUTPUT_CSV="${TEST_OUTPUT_CSV:-${_ROOT}/external/submissions_peer/submission_phaseC_pseudo_finetune_test.csv}"
export NUM_EPOCHS="${NUM_EPOCHS:-30}"
export LEARNING_RATE="${LEARNING_RATE:-3e-5}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-12}"
export EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-5}"

export GLEVEL_OPT="${GLEVEL_OPT:---g_level_int_encoding one --glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001 --label_smoothing 0.05 --select_best balanced_acc --cross_modal_attn --cross_modal_layers 1 --modality_dropout_p 0.12 --scheduler_min_lr 1e-6 --sampler_medium_boost 1.5 --seed 37 --train_feat_fallback --train_fallback_use_test_features}"

bash "${_ROOT}/vote_train_glevel.sh"
