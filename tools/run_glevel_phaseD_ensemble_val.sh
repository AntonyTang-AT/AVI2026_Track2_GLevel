#!/usr/bin/env bash
# Phase D：对 CSV 中 Top-K 同结构 S_ref* checkpoint 做 logits 平均，评官方 val。
# 用法: bash tools/run_glevel_phaseD_ensemble_val.sh /data/.../combo_sweep_metrics.csv
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

CSV="${1:?usage: $0 /path/to/combo_sweep_metrics.csv}"
PY="${GLEVEL_CUDA_PYTHON:-/home/emo/txcao/anaconda3/envs/avi2026/bin/python}"
TOP="${ENSEMBLE_TOP_K:-5}"
REPORT="$(dirname "${CSV}")/phaseD_ensemble_val.txt"

CKPTS="$("${PY}" "${_ROOT}/tools/glevel_pick_ensemble_ckpts.py" "${CSV}" --top "${TOP}")"
if [[ -z "${CKPTS// /}" ]]; then
  echo "[phaseD] no checkpoints picked from ${CSV}" >&2
  exit 1
fi

export NANBEIGE_TEXT=1 TEXT_DIM=2560
export TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${_ROOT}/data/text_nb}"
export TEXT_VAL_DIR="${TEXT_VAL_DIR:-${_ROOT}/data/text_nb_val}"
export TEXT_TEST_DIR="${TEXT_TEST_DIR:-${_ROOT}/data/test_nb}"
# shellcheck source=/dev/null
. "${_ROOT}/tools/glevel_paths.inc.sh"

{
  echo "### Phase D ensemble Top-${TOP} ($(date -Is))"
  echo "checkpoints: ${CKPTS}"
  echo ""
  # shellcheck disable=SC2086
  "${PY}" "${_ROOT}/tools/eval_glevel_checkpoint_on_csv.py" \
    --eval_csv /data/Super-Lu/dataset/val_data.csv \
    --rating_csv /data/Super-Lu/dataset/train_data.csv \
    --labels_in_split_csv \
    --g_level_int_encoding one \
    --train_audio_dir /data/Super-Lu/dataset/train_feature/audio \
    --train_video_dir /data/Super-Lu/dataset/train_feature/video \
    --train_text_dir "${TEXT_TRAIN_DIR}" \
    --eval_audio_dir /data/Super-Lu/dataset/val_feature/audio \
    --eval_video_dir /data/Super-Lu/dataset/val_feature/video \
    --eval_text_dir "${TEXT_VAL_DIR}" \
    --ensemble_checkpoints ${CKPTS} \
    --text_dim 2560 \
    --mlp_dropout 0.25 \
    --modality_dropout_p 0.12 \
    --cross_modal_attn --cross_modal_layers 1 \
    --num_workers 2
} | tee "${REPORT}"
echo "[phaseD] wrote ${REPORT}" >&2
