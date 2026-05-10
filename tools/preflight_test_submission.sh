#!/usr/bin/env bash
# 提交前仅测「测试集」侧：FEAT_TEST 下 audio/video 与 TEXT_TEST_DIR 文本 .npy 是否覆盖 TEST_CSV。
# 不检查 val。通过后再 bash vote_test_glevel.sh。
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

PYTHON="${PYTHON:-python}"
TEST_CSV="${TEST_CSV:-${_ROOT}/data/test_data_basic_information.csv}"
FEAT_TEST="${FEAT_TEST:-/data/AVI2026/test_feature}"
_NB_SUB="${NANBEIGE_TEXT_SUBDIR:-text_nb}"

echo "[preflight_test_submission] TEST_CSV=$TEST_CSV" >&2
echo "[preflight_test_submission] FEAT_TEST=$FEAT_TEST" >&2

SKIP_TEXT_FLAG=()
if [ "${NANBEIGE_TEXT:-0}" = "1" ]; then
  export TEXT_TEST_DIR="${TEXT_TEST_DIR:-${FEAT_TEST}/${_NB_SUB}}"
  SKIP_TEXT_FLAG=(--skip_text_in_feat_root)
  echo "[preflight_test_submission] NANBEIGE_TEXT=1 TEXT_TEST_DIR=$TEXT_TEST_DIR" >&2
fi

echo "=== FEAT_TEST：audio/video ===" >&2
"${PYTHON}" "${_ROOT}/tools/check_test_feature_coverage.py" \
  --test_csv "$TEST_CSV" \
  --feat_root "$FEAT_TEST" \
  "${SKIP_TEXT_FLAG[@]}"

echo "=== TEXT_TEST_DIR：逐 id 文本 .npy ===" >&2
if [ "${NANBEIGE_TEXT:-0}" = "1" ]; then
  TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${FEAT_TRAIN:-/data/Super-Lu/dataset/train_feature}/${_NB_SUB}}"
  "${PYTHON}" "${_ROOT}/tools/check_text_npy_coverage.py" \
    --csv "$TEST_CSV" \
    --text_dir "${TEXT_TEST_DIR}" \
    --fallback_text_dir "${TEXT_TRAIN_DIR}"
else
  TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${FEAT_TRAIN:-/data/Super-Lu/dataset/train_feature}/text}"
  "${PYTHON}" "${_ROOT}/tools/check_text_npy_coverage.py" \
    --csv "$TEST_CSV" \
    --text_dir "${FEAT_TEST}/text" \
    --fallback_text_dir "${TEXT_TRAIN_DIR}"
fi

echo "" >&2
echo "[preflight_test_submission] 通过。测试命令示例:" >&2
echo "  export TEST_MODEL=./best_model_glevel_multimodal_plus.pth" >&2
echo "  # 推理时 GLEVEL_OPT 须与训练一致（architecture）" >&2
echo "  bash vote_test_glevel.sh" >&2
