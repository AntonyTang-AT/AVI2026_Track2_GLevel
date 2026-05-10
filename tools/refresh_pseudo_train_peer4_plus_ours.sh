#!/usr/bin/env bash
# 用「四 peer + 一条自有 submission」重建测试集伪标签，并生成 train_plus_pseudo_unanimous.csv（仅 unanimous）。
# 依赖：external/submissions_peer 下四份 peer CSV；自有提交默认用 batch 推理产物 inferbias070。
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

SP="${_ROOT}/external/submissions_peer"
OURS="${OURS_SUBMISSION_CSV:-${SP}/submission_batch_infer_seed37_archive_inferbias070.csv}"
OUT_PSEUDO="${OUT_PSEUDO_CSV:-${SP}/test_pseudo_peer4_plus_ours_inferbias070.csv}"
OUT_TRAIN="${OUT_TRAIN_MERGED:-${_ROOT}/experiments/glevel_improvement_plan/train_plus_pseudo_unanimous.csv}"

PEERS=(
  "${SP}/submission1_0.53077.csv"
  "${SP}/submission2_0.50769.csv"
  "${SP}/submission_0.53846.csv"
  "${SP}/submission5_0.55385.csv"
)

for p in "${PEERS[@]}" "${OURS}"; do
  [[ -f "${p}" ]] || { echo "[refresh_pseudo] 缺少文件: ${p}" >&2; exit 2; }
done

python3 "${_ROOT}/tools/build_consensus_pseudo_k.py" --out "${OUT_PSEUDO}" \
  "${PEERS[@]}" "${OURS}"

python3 "${_ROOT}/tools/merge_train_with_pseudo_test_rows.py" \
  --train-csv "/data/Super-Lu/dataset/train_data.csv" \
  --test-basic-csv "/data/Super-Lu/dataset/test_data_basic_information.csv" \
  --pseudo-csv "${OUT_PSEUDO}" \
  --out "${OUT_TRAIN}" \
  --require-unanimous

echo "[refresh_pseudo] pseudo=${OUT_PSEUDO} merged_train=${OUT_TRAIN}"
