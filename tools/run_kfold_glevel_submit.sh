#!/usr/bin/env bash
# K 折训练 + 多数投票 submission（与 vote_kfold_glevel.sh 相同），可选跑前检查测试集文本 .npy。
# 需与单折训练一致地 export FEAT_*、CSV、NANBEIGE_TEXT、TEXT_*、GLEVEL_OPT 等。
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

PYTHON="${PYTHON:-python}"

if [ "${ROUTE_A_PREFLIGHT:-0}" = "1" ]; then
  echo "[run_kfold_glevel_submit] ROUTE_A_PREFLIGHT=1 → tools/route_a_complete.sh" >&2
  bash "${_ROOT}/tools/route_a_complete.sh"
fi

if [ "${PREFLIGHT_KFOLD_TEXT:-1}" = "1" ] && [ -n "${TEST_CSV:-}" ] && [ -n "${TEXT_TEST_DIR:-}" ]; then
  echo "[run_kfold_glevel_submit] 检查 TEST_CSV 与 TEXT_TEST_DIR 文本 .npy …" >&2
  _FB=()
  if [ -n "${TEXT_TRAIN_DIR:-}" ]; then
    _FB=(--fallback_text_dir "${TEXT_TRAIN_DIR}")
  fi
  "${PYTHON}" tools/check_text_npy_coverage.py --csv "${TEST_CSV}" --text_dir "${TEXT_TEST_DIR}" "${_FB[@]}"
else
  echo "[run_kfold_glevel_submit] 跳过文本预检（设 PREFLIGHT_KFOLD_TEXT=0 或缺 TEST_CSV/TEXT_TEST_DIR）" >&2
fi

exec bash "${_ROOT}/vote_kfold_glevel.sh"
