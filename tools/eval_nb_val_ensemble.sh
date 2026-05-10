#!/usr/bin/env bash
# Nanbeige 官方 val：多 checkpoint logits 平均（train_task2_glevel --only_test --ensemble_checkpoints）
# 用法（项目根）:
#   export PYTHON=/path/to/python
#   export NANBEIGE_TEXT=1 TEXT_TRAIN_DIR=... TEXT_VAL_DIR=... TEXT_TEST_DIR=...
#   export ENSEMBLE_CHECKPOINTS="path1 path2 path3"
#   bash tools/eval_nb_val_ensemble.sh
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"
PYTHON="${PYTHON:-python}"

export NANBEIGE_TEXT="${NANBEIGE_TEXT:-1}"
export TEXT_DIM="${TEXT_DIM:-2560}"
export TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${_ROOT}/data/text_nb}"
export TEXT_VAL_DIR="${TEXT_VAL_DIR:-${_ROOT}/data/text_nb_val}"
export TEXT_TEST_DIR="${TEXT_TEST_DIR:-${_ROOT}/data/text_nb}"

# shellcheck source=tools/glevel_paths.inc.sh
. "${_ROOT}/tools/glevel_paths.inc.sh"

CKPTS="${ENSEMBLE_CHECKPOINTS:-}"
if [ -z "$CKPTS" ]; then
  R1="${_ROOT}/experiments/nb_to58_sweep/round1"
  CKPTS="${R1}/seed37/best.pth ${R1}/seed99/best.pth ${R1}/seed11/best.pth"
fi

export GLEVEL_OPT="--g_level_int_encoding one --glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001 --label_smoothing 0.05 --select_best balanced_acc --cross_modal_attn --cross_modal_layers 1 --modality_dropout_p 0.12 --scheduler_min_lr 1e-6 --sampler_medium_boost 1.5 --glevel_loss ce"

SPLIT_ARG=""
if [ "${SPLIT_LABELS:-1}" = "1" ]; then
  SPLIT_ARG="--labels_in_split_csv"
fi

# shellcheck disable=SC2086
"${PYTHON}" "${_ROOT}/python/train_task2_glevel.py" \
  --only_test \
  --ensemble_checkpoints ${CKPTS} \
  --train_csv "${TRAIN_CSV}" \
  --val_csv "${VAL_CSV}" \
  --test_csv "${TEST_CSV}" \
  --rating_csv "${RATING_CSV}" \
  ${GLEVEL_OPT} \
  ${SPLIT_ARG} \
  --label_col g_level \
  --question q1 q2 q3 q4 q5 q6 \
  --video_dim 512 \
  --video_dir "${FEAT_TRAIN}/video" \
  --audio_dim 512 \
  --audio_dir "${FEAT_TRAIN}/audio" \
  --text_dim "${TEXT_DIM}" \
  --text_dir "${TEXT_TRAIN_DIR}" \
  --val_video_dir "${FEAT_VAL}/video" \
  --val_audio_dir "${FEAT_VAL}/audio" \
  --val_text_dir "${TEXT_VAL_DIR}" \
  --test_video_dir "${FEAT_TEST}/video" \
  --test_audio_dir "${FEAT_TEST}/audio" \
  --test_text_dir "${TEXT_TEST_DIR}" \
  --batch_size 32 \
  --num_workers "${NUM_WORKERS:-4}" \
  --test_output_csv "${TEST_OUTPUT_CSV:-/tmp/submission_glevel_ensemble_nb.csv}"
