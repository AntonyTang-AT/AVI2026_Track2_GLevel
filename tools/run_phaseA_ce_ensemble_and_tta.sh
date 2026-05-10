#!/usr/bin/env bash
# Phase A 补充：CE-only 多 checkpoint 集成；可选 TTA。结果追加到 experiments/glevel_improvement_plan/phaseA_extras.log
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

LOG="${_ROOT}/experiments/glevel_improvement_plan/phaseA_extras.log"
mkdir -p "$(dirname "${LOG}")"

PYTHON="${PYTHON:-python}"
export NANBEIGE_TEXT="${NANBEIGE_TEXT:-1}"
export TEXT_DIM="${TEXT_DIM:-2560}"
export FEAT_TRAIN="${FEAT_TRAIN:-/data/Super-Lu/dataset/train_feature}"
export FEAT_VAL="${FEAT_VAL:-/data/Super-Lu/dataset/val_feature}"
export FEAT_TEST="${FEAT_TEST:-/data/Super-Lu/dataset/test_feature}"
export TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${_ROOT}/data/text_nb}"
export TEXT_VAL_DIR="${TEXT_VAL_DIR:-${_ROOT}/data/text_nb_val}"
export TEXT_TEST_DIR="${TEXT_TEST_DIR:-${_ROOT}/data/test_nb}"
export TEST_CSV="${TEST_CSV:-/data/Super-Lu/dataset/test_data_basic_information.csv}"
PHASEA_ENS_OUT="${_ROOT}/external/submissions_peer/submission_phaseA_ensemble_seed37_10_99.csv"
PHASEA_TTA_OUT="${_ROOT}/external/submissions_peer/submission_phaseA_seed37_tta8.csv"

BASE_OPT="--g_level_int_encoding one --glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001 --label_smoothing 0.05 --select_best balanced_acc --cross_modal_attn --cross_modal_layers 1 --modality_dropout_p 0.12 --scheduler_min_lr 1e-6 --sampler_medium_boost 1.5 --seed 37"

CKPT37="${CKPT37:-/data/emo/glevel_runs/archives/nb_to58_sweep/round1/seed37/best.pth}"
CKPT10="${CKPT10:-/data/emo/glevel_runs/archives/nb_to58_sweep/round1/seed10/best.pth}"
CKPT99="${CKPT99:-/data/emo/glevel_runs/archives/nb_to58_sweep/round1/seed99/best.pth}"

{
  echo "=== $(date -Iseconds) CE ensemble seed37+10+99 ==="
  "${PYTHON}" train_task2_glevel.py \
    --only_test \
    --ensemble_checkpoints "${CKPT37}" "${CKPT10}" "${CKPT99}" \
    --test_model "${CKPT37}" \
    --train_csv "/data/Super-Lu/dataset/train_data.csv" \
    --val_csv "/data/Super-Lu/dataset/val_data.csv" \
    --test_csv "${TEST_CSV}" \
    --rating_csv "${FEAT_TRAIN}/../train_data.csv" \
    ${BASE_OPT} \
    --labels_in_split_csv \
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
    --test_output_csv "${PHASEA_ENS_OUT}"
} >>"${LOG}" 2>&1 || true

{
  echo "=== $(date -Iseconds) TTA tta_times=8 seed37 single ==="
  "${PYTHON}" train_task2_glevel.py \
    --only_test \
    --test_model "${CKPT37}" \
    --train_csv "/data/Super-Lu/dataset/train_data.csv" \
    --val_csv "/data/Super-Lu/dataset/val_data.csv" \
    --test_csv "${TEST_CSV}" \
    --rating_csv "/data/Super-Lu/dataset/train_data.csv" \
    ${BASE_OPT} \
    --tta_times 8 \
    --tta_noise_std 0.01 \
    --labels_in_split_csv \
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
    --test_output_csv "${PHASEA_TTA_OUT}"
} >>"${LOG}" 2>&1 || true

echo "[phaseA_extras] log appended ${LOG}" >&2
