#!/usr/bin/env bash
# 在本机（Linux/macOS/Git Bash）执行：从服务器拉取 server_environment_scan 生成的 artifacts 与常见报错日志。
#
# 用法:
#   export HOST=183.196.130.56 PORT=24322 USER=emo
#   export REMOTE_ROOT=/home/emo/antonytang/AVI2026_Track2_GLevel
#   bash tools/pull_server_artifacts.sh
#
# 可选:
#   LOCAL_BASE=./server_pull bash tools/pull_server_artifacts.sh
set -eu
HOST="${HOST:-183.196.130.56}"
PORT="${PORT:-24322}"
USER="${USER:-emo}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/emo/antonytang/AVI2026_Track2_GLevel}"
LOCAL_BASE="${LOCAL_BASE:-./server_pull}"
TS="$(date +%Y%m%d_%H%M%S)"
DEST="${LOCAL_BASE%/}/pull_${TS}"
mkdir -p "$DEST"

echo "[pull] ${USER}@${HOST}:${REMOTE_ROOT} -> $DEST"

scp -P "$PORT" -o "StrictHostKeyChecking=accept-new" -r \
  "${USER}@${HOST}:${REMOTE_ROOT}/artifacts" \
  "$DEST/" || true

for f in debug-f0e227.log train_glevel.log nohup.out; do
  scp -P "$PORT" -o "StrictHostKeyChecking=accept-new" \
    "${USER}@${HOST}:${REMOTE_ROOT}/${f}" \
    "$DEST/" 2>/dev/null || true
done

echo "[pull] 完成。查看: $DEST"
ls -la "$DEST" || true
