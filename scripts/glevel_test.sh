#!/usr/bin/env bash
# g_level 模型：验证集评估 + 测试集导出 submission（维度须与训练一致）
# CRLF 换行会导致 Linux 下报错；修复: sed -i 's/\r$//' scripts/glevel_test.sh
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

PYTHON="${PYTHON:-python}"

TEST_MODEL="${TEST_MODEL:-./best_model_glevel.pth}"

# 数据路径与 scripts/glevel_train.sh 共用 tools/glevel_paths.inc.sh
# shellcheck source=tools/glevel_paths.inc.sh
. "${_ROOT}/tools/glevel_paths.inc.sh"

SPLIT_ARG=""
if [ "${SPLIT_LABELS:-1}" = "1" ]; then
  SPLIT_ARG="--labels_in_split_csv"
fi

VAL_ERR_ARG=()
if [ -n "${VAL_ERRORS_CSV:-}" ]; then
  VAL_ERR_ARG=(--val_errors_csv "${VAL_ERRORS_CSV}")
fi

if [ "${SKIP_TORCH_PREFLIGHT:-0}" != "1" ]; then
  if ! "${PYTHON}" -c "import torch"; then
    echo "[glevel_test] PyTorch 导入失败。正在追加 tools/diagnose_torch_env.py 到 debug-f0e227.log …" >&2
    "${PYTHON}" "${_ROOT}/tools/diagnose_torch_env.py" 2>/dev/null || true
    bash "${_ROOT}/tools/print_torch_env_fix_hint.sh" 2>/dev/null || true
    exit 2
  fi
fi

"${PYTHON}" "${_ROOT}/python/train_task2_glevel.py" \
  --only_test \
  --test_model "${TEST_MODEL}" \
  --train_csv "${TRAIN_CSV}" \
  --val_csv "${VAL_CSV}" \
  --test_csv "${TEST_CSV}" \
  --rating_csv "${RATING_CSV}" \
  ${GLEVEL_OPT:-} \
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
  --num_workers 4 \
  --test_output_csv "${TEST_OUTPUT_CSV:-${_ROOT}/reports/submissions/submission_glevel.csv}" \
  "${VAL_ERR_ARG[@]}"
