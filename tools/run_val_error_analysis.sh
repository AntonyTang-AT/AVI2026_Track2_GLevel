#!/usr/bin/env bash
# 对已保存的 best checkpoint 跑验证并写出错分 CSV + 终端混淆矩阵。
# 依赖与 vote_test_glevel.sh 相同（FEAT_*、TEXT_*、CSV、GLEVEL_OPT 等须与训练一致）。
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT:-$(cd "$_SCRIPT_DIR/.." && pwd)}"

export VAL_ERRORS_CSV="${VAL_ERRORS_CSV:-./logs/val_glevel_errors.csv}"
mkdir -p "$(dirname "$VAL_ERRORS_CSV")"

echo "[run_val_error_analysis] VAL_ERRORS_CSV=$VAL_ERRORS_CSV" >&2
echo "[run_val_error_analysis] TEST_MODEL=${TEST_MODEL:-./best_model_glevel.pth}" >&2

bash "${PWD}/vote_test_glevel.sh"
