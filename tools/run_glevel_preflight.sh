#!/usr/bin/env bash
# 规划前置检查：磁盘、GPU、推荐 CUDA Python。结果写入 OUT（默认 /data/emo/glevel_runs/preflight_latest.txt）
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
OUT="${GLEVEL_PREFLIGHT_OUT:-/data/emo/glevel_runs/preflight_latest.txt}"
mkdir -p "$(dirname "$OUT")"
PY_CAND=("${GLEVEL_CUDA_PYTHON:-}" "/home/emo/txcao/anaconda3/envs/avi2026/bin/python")

{
  echo "=== glevel preflight $(date -Is) ==="
  echo "--- df / /data ---"
  df -h / /data 2>/dev/null || df -h
  echo "--- nvidia-smi ---"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || echo "no nvidia-smi"
  echo "--- CUDA python ---"
  for p in "${PY_CAND[@]}"; do
    [[ -z "${p}" ]] || [[ ! -x "${p}" ]] && continue
    if "${p}" -c "import torch; print(torch.__file__); print('cuda', torch.cuda.is_available(), 'n', torch.cuda.device_count())" 2>/dev/null; then
      echo "OK: ${p}"
    else
      echo "FAIL: ${p}"
    fi
  done
  echo "--- hints ---"
  echo "HUNT_DIR 请使用 /data/emo/glevel_runs/... 绝对路径；export HUNT_DIR 后再跑 sweep。"
  echo "若部分 GPU 已被占满，建议: export GPU_IDS=0,5,6 与 MAX_PARALLEL_JOBS 匹配空闲卡数。"
  echo "全模态 batch=32 时避免与高占用进程同卡；否则 OOM_SAFE_BATCH=1。"
} | tee "${OUT}"
echo "[preflight] wrote ${OUT}" >&2
