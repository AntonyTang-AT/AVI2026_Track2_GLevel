#!/usr/bin/env bash
# g_level 分层 K 折训练 + 多数投票融合提交（转调 one_click_kfold_glevel.py）。
# 提交前检查测试集 Nanbeige 文本是否齐：bash tools/run_kfold_glevel_submit.sh
# 环境变量：KFOLDS、KFOLD_SEED、KFOLD_OUT_DIR、GLEVEL_OPT、KFOLD_EXTRA、
# TRAIN_CSV、VAL_CSV、TEST_CSV、RATING_CSV、FEAT_*、TEXT_*、NANBEIGE_TEXT 等，
# 与 one_click_kfold_glevel.py / vote_train_glevel.sh 一致（NANBEIGE_TEXT=1 时默认 text_nb）。
# CRLF 会导致 Linux 下报错；修复: sed -i 's/\r$//' vote_kfold_glevel.sh
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT:-$_SCRIPT_DIR}"

PYTHON="${PYTHON:-python}"
"${PYTHON}" one_click_kfold_glevel.py
