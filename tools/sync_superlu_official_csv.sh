#!/usr/bin/env bash
# 将 SUPERLU_DATASET（默认 /data/Super-Lu/dataset）下的官方 CSV 拷贝到仓库 data/superlu_official/
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
SRC="${SUPERLU_DATASET:-/data/Super-Lu/dataset}"
DST="${_ROOT}/data/superlu_official"
mkdir -p "${DST}"
for f in train_data.csv val_data.csv test_data_basic_information.csv submission.csv; do
  if [[ -f "${SRC}/${f}" ]]; then
    cp -a "${SRC}/${f}" "${DST}/${f}"
    echo "[sync] ${SRC}/${f} -> ${DST}/${f}"
  else
    echo "[sync] skip missing: ${SRC}/${f}" >&2
  fi
done
echo "[sync] done → ${DST}"
