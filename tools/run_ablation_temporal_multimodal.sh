#!/usr/bin/env bash
# 对照实验：MM_TEMPORAL=0 vs 1，其余与 scripts/glevel_train_multimodal.sh 默认一致。
# 日志写入 LOG_DIR（默认 ./logs/ablation_temporal）。
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

LOG_DIR="${LOG_DIR:-${_ROOT}/logs/ablation_temporal}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

run_one() {
  local tag="$1"
  local mm="$2"
  local log="$LOG_DIR/${TS}_${tag}.log"
  echo "[ablation_temporal] === $tag (MM_TEMPORAL=$mm) → $log ===" >&2
  {
    echo "=== $tag MM_TEMPORAL=$mm at $(date -Iseconds) ==="
    env MM_TEMPORAL="$mm" bash "${_ROOT}/scripts/glevel_train_multimodal.sh"
  } 2>&1 | tee "$log"
  echo "[ablation_temporal] done $tag" >&2
}

run_one "no_temporal_gru" 0
run_one "with_temporal_gru" 1

echo "[ablation_temporal] 全部完成。对比:" >&2
echo "  grep -E 'Best checkpoint|\\[metrics_line\\]|\\[train_summary\\]' $LOG_DIR/${TS}_*.log" >&2
