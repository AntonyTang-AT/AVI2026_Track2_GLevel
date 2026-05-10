#!/usr/bin/env bash
# 转写 → 文本 .npy：逻辑在 extract_nanbeige_one_click.py（避免 CRLF）
set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python tools/extract_nanbeige_one_click.py
