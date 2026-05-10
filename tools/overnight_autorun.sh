#!/usr/bin/env bash
# 顺序：① 加权伪标签 ②（可选）中性 DeepSeek 3× 后台 ③ GPU combo sweep ④ 三重指标排名
# 用法（仓库根）: nohup bash tools/overnight_autorun.sh >> experiments/overnight_outer.log 2>&1 &
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${ROOT}/experiments"
LOG="${ROOT}/experiments/overnight_${TS}.log"
export HUNT_DIR="${HUNT_DIR:-${ROOT}/experiments/gpu_combo_sweep/overnight_${TS}}"
mkdir -p "${HUNT_DIR}"

exec >>"${LOG}" 2>&1
echo "[overnight] LOG=${LOG}"
echo "[overnight] HUNT_DIR=${HUNT_DIR}"
echo "[overnight] start $(date -Is)"

echo "=== [1] build weighted pseudo ==="
python3 "${ROOT}/tools/build_weighted_pseudo_peers_plateau.py"

_DEEP_DIR="${ROOT}/reports/deepseek"
mkdir -p "${_DEEP_DIR}"

echo "=== [2] DeepSeek neutral 3-pass (background if not running and no output) ==="
if [[ -f "${_DEEP_DIR}/deepseek_ens_neutral_r3_vote_zyn.json" ]]; then
  echo "skip: ${_DEEP_DIR}/deepseek_ens_neutral_r3_vote_zyn.json exists"
elif pgrep -f "deepseek_ens_neutral_r3_vote_zyn.json" >/dev/null 2>&1; then
  echo "skip: annotate_deepseek_interactive already running for neutral output"
else
  nohup python3 "${ROOT}/python/annotate_deepseek_interactive.py" \
    --pipeline-runs 3 --pipeline-seed-stride 1009 \
    --test-text-dir /data/zyn/test_text \
    --test-temperature 0.12 --test-temperature-stride 0.03 \
    --report "${_DEEP_DIR}/deepseek_ens_neutral_r3_report_zyn.json" \
    -o "${_DEEP_DIR}/deepseek_ens_neutral_r3_vote_zyn.json" \
    --sleep 0.2 \
    >>"${ROOT}/reports/deepseek/deepseek_ens_neutral_r3_run_zyn.log" 2>&1 &
  echo "started deepseek pid=$!"
fi

echo "=== [3] GPU combo sweep ==="
export COMBOS="${COMBOS:-S_ref_plateau,S_plateau_ln,S_ref_cosine,S_step_ln,AT_plateau_ln}"
export SEEDS="${SEEDS:-37 42 28 10 99}"
export MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-1}"
export SAMPLER_MEDIUM_BOOST="${SAMPLER_MEDIUM_BOOST:-1.0}"
export NUM_EPOCHS="${NUM_EPOCHS:-120}"
bash "${ROOT}/tools/run_glevel_gpu_combo_sweep.sh"

echo "=== [4] triple metric rank ==="
DEEP="${_DEEP_DIR}/deepseek_ens_r3_vote_zyn.json"
if [[ -f "${_DEEP_DIR}/deepseek_ens_neutral_r3_vote_zyn.json" ]]; then
  DEEP="${_DEEP_DIR}/deepseek_ens_neutral_r3_vote_zyn.json"
fi
python3 "${ROOT}/tools/rank_gpu_sweep_triple_metric.py" \
  --sweep-csv "${HUNT_DIR}/combo_sweep_metrics.csv" \
  --hunt-dir "${HUNT_DIR}" \
  --pseudo-csv "${ROOT}/external/submissions_peer/test_pseudo_weighted_peers4_plateau_ln28.csv" \
  --deepseek-json "${DEEP}" \
  --out-tsv "${HUNT_DIR}/ranking_triple.tsv"

echo "[overnight] done $(date -Is)"
