#!/usr/bin/env bash
# 阶段一：prepare_svm_data + train_svm
# 数据路径与 scripts/glevel_train.sh 完全一致：source tools/glevel_paths.inc.sh
#
# 一键（赛方 1–3 + Nanbeige；测试集文本默认优先用工程内 data/test_nb，若该目录存在）：
#   cd ~/antonytang/AVI2026_Track2_GLevel && NANBEIGE_TEXT=1 G_LEVEL_INT_ENCODING=one bash tools/run_phase1_svm.sh
# 指定 Python： export PYTHON=python3
# CRLF 报错： sed -i 's/\r$//' tools/run_phase1_svm.sh tools/glevel_paths.inc.sh
#
set -eu
_TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_TOOLS}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

PYTHON="${PYTHON:-python3}"

# shellcheck source=glevel_paths.inc.sh
. "${_TOOLS}/glevel_paths.inc.sh"

VIDEO_DIM="${VIDEO_DIM:-512}"
AUDIO_DIM="${AUDIO_DIM:-512}"
G_LEVEL_INT_ENCODING="${G_LEVEL_INT_ENCODING:-zero}"

SVM_OUT_DIR="${SVM_OUT_DIR:-./data/svm}"
SVM_LOG_FILE="${SVM_LOG_FILE:-./logs/svm_results.txt}"
SVM_PCA="${SVM_PCA:-0}"

mkdir -p "${SVM_OUT_DIR}" "$(dirname "${SVM_LOG_FILE}")"

echo "[run_phase1_svm] PYTHON=${PYTHON}" >&2
echo "[run_phase1_svm] FEAT_TRAIN=${FEAT_TRAIN} FEAT_VAL=${FEAT_VAL}" >&2
echo "[run_phase1_svm] TEXT_TRAIN_DIR=${TEXT_TRAIN_DIR} TEXT_VAL_DIR=${TEXT_VAL_DIR} TEXT_TEST_DIR=${TEXT_TEST_DIR} TEXT_DIM=${TEXT_DIM}" >&2
echo "[run_phase1_svm] NANBEIGE_TEXT=${NANBEIGE_TEXT:-0} G_LEVEL_INT_ENCODING=${G_LEVEL_INT_ENCODING}" >&2

"${PYTHON}" tools/prepare_svm_data.py \
  --train_csv "${TRAIN_CSV}" \
  --val_csv "${VAL_CSV}" \
  --rating_csv "${RATING_CSV}" \
  --g_level_int_encoding "${G_LEVEL_INT_ENCODING}" \
  --audio_dir "${FEAT_TRAIN}/audio" \
  --video_dir "${FEAT_TRAIN}/video" \
  --text_dir "${TEXT_TRAIN_DIR}" \
  --val_audio_dir "${FEAT_VAL}/audio" \
  --val_video_dir "${FEAT_VAL}/video" \
  --val_text_dir "${TEXT_VAL_DIR}" \
  --video_dim "${VIDEO_DIM}" \
  --audio_dim "${AUDIO_DIM}" \
  --text_dim "${TEXT_DIM}" \
  --out_dir "${SVM_OUT_DIR}"

if [ -n "${SVM_SAVE_JOBLIB:-}" ]; then
  "${PYTHON}" tools/train_svm.py \
    --data_dir "${SVM_OUT_DIR}" \
    --log_file "${SVM_LOG_FILE}" \
    --pca "${SVM_PCA}" \
    --save_sklearn_bundle "${SVM_SAVE_JOBLIB}"
else
  "${PYTHON}" tools/train_svm.py \
    --data_dir "${SVM_OUT_DIR}" \
    --log_file "${SVM_LOG_FILE}" \
    --pca "${SVM_PCA}"
fi

echo "[run_phase1_svm] 完成。数据: ${SVM_OUT_DIR} 日志: ${SVM_LOG_FILE}" >&2
