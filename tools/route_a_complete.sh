#!/usr/bin/env bash
# 路线 A：训练/提交前的数据与特征补全预检（与 vote_train_glevel.sh 路径变量一致）。
# 用法：在工程根目录先 export 与训练相同的环境变量，再执行：
#   bash tools/route_a_complete.sh
# 退出码 0 表示检查通过；2 表示存在缺失，需补提特征或按 tools/extract_nanbeige_splits.example.sh 提取。
# 若 Linux 报 set/cd 含 \r：sed -i 's/\r$//' tools/route_a_complete.sh（仓库 .gitattributes 已设 *.sh eol=lf）
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT:-$(cd "$_SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"

# 与 vote_train_glevel.sh 相同路径变量（含 Nanbeige / Anton 布局）
# shellcheck source=glevel_paths.inc.sh
. "${_SCRIPT_DIR}/glevel_paths.inc.sh"

_NB_SUB="${NANBEIGE_TEXT_SUBDIR:-text_nb}"

SKIP_TEXT_FLAG=()
if [ "${NANBEIGE_TEXT:-0}" = "1" ]; then
  SKIP_TEXT_FLAG=(--skip_text_in_feat_root)
fi

ERR=0

echo "=== [route_a] 1/4 FEAT_TEST：audio/video（+ 非 Nanbeige 时含 text）==="
if "${PYTHON}" tools/check_test_feature_coverage.py \
  --test_csv "${TEST_CSV}" \
  --feat_root "${FEAT_TEST}" \
  "${SKIP_TEXT_FLAG[@]}"; then
  :
else
  ERR=1
fi

echo "=== [route_a] 2/4 VAL：audio/video/text 三模态（主 val + 回退 train）==="
if "${PYTHON}" tools/report_missing_features_for_csv.py \
  --csv "${VAL_CSV}" \
  --audio_dir "${FEAT_VAL}/audio" --video_dir "${FEAT_VAL}/video" --text_dir "${TEXT_VAL_DIR}" \
  --fallback_audio_dir "${FEAT_TRAIN}/audio" --fallback_video_dir "${FEAT_TRAIN}/video" \
  --fallback_text_dir "${TEXT_TRAIN_DIR}"; then
  :
else
  ERR=1
fi

echo "=== [route_a] 3/4 VAL：文本 .npy 逐 id 覆盖（与 TEXT_VAL_DIR 一致）==="
if "${PYTHON}" tools/check_text_npy_coverage.py \
  --csv "${VAL_CSV}" \
  --text_dir "${TEXT_VAL_DIR}" \
  --fallback_text_dir "${TEXT_TRAIN_DIR}"; then
  :
else
  ERR=1
fi

echo "=== [route_a] 4/4 TEST：Nanbeige/SigLIP 文本 .npy（提交必需，TEXT_TEST_DIR）==="
if "${PYTHON}" tools/check_text_npy_coverage.py \
  --csv "${TEST_CSV}" \
  --text_dir "${TEXT_TEST_DIR}" \
  --fallback_text_dir "${TEXT_TRAIN_DIR}"; then
  :
else
  ERR=1
fi

if [ "$ERR" != "0" ]; then
  echo "" >&2
  echo "[route_a] 未全部通过。请：" >&2
  echo "  - 对缺模态补提 audio/video 或检查 FEAT_VAL/FEAT_TEST 路径；" >&2
  echo "  - 文本：按 tools/extract_nanbeige_splits.example.sh 对 val/test 转写运行 extract_nanbeige_one_click.py（OUT 勿写只读挂载）；" >&2
  echo "  - 或调整 export TEXT_VAL_DIR / TEXT_TEST_DIR / TEXT_TRAIN_DIR。" >&2
  exit 2
fi

echo "[route_a] 全部检查通过。可进行 bash vote_train_glevel.sh 或 K 折。"
exit 0
