#!/usr/bin/env bash
# 学长简化基线：仅音频+文本 + 可选 LayerNorm（--fused_layer_norm），无跨模态/32头。
# 默认跑种子 37、42、99；可通过 SEEDS 覆盖。
#
# 用法（工程根，Nanbeige）:
#   export NANBEIGE_TEXT=1 TEXT_DIM=2560
#   export TEXT_TRAIN_DIR=$PWD/data/text_nb TEXT_VAL_DIR=$PWD/data/text_nb_val
#   source tools/glevel_paths.inc.sh
#   bash tools/run_glevel_audio_text_baseline.sh
#
# 仅 seed37 且加 LN: SEEDS=37 FUSED_LN=1 bash tools/run_glevel_audio_text_baseline.sh
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${_ROOT}"

PY="${PYTHON:-python}"
SUP="${SUPERLU_DATASET:-/data/Super-Lu/dataset}"
export NANBEIGE_TEXT="${NANBEIGE_TEXT:-1}"
export TEXT_DIM="${TEXT_DIM:-2560}"

# shellcheck source=/dev/null
. "${_ROOT}/tools/glevel_paths.inc.sh"

OUT_ROOT="${OUT_ROOT:-${_ROOT}/experiments/at_baseline}"
SEEDS="${SEEDS:-37 42 99}"
FUSED_LN="${FUSED_LN:-0}"
LN_FLAG=()
if [[ "${FUSED_LN}" = "1" ]]; then
  LN_FLAG=(--fused_layer_norm)
fi

BASE_OPT=(
  --glevel_arch audio_text_mlp
  --g_level_int_encoding one
  --mlp_dropout 0.25
  --weight_decay 0.001
  --label_smoothing 0.05
  --select_best balanced_acc
  --sampler_medium_boost 1.5
  --scheduler_min_lr 1e-6
  "${LN_FLAG[@]}"
)

for s in ${SEEDS}; do
  ODIR="${OUT_ROOT}/seed${s}"
  mkdir -p "${ODIR}"
  echo "[at_baseline] train seed=${s} -> ${ODIR}/best.pth" >&2
  "${PY}" "${_ROOT}/python/train_task2_glevel.py" \
    --train_csv "${SUP}/train_data.csv" \
    --val_csv "${SUP}/val_data.csv" \
    --test_csv "${TEST_CSV}" \
    --rating_csv "${SUP}/train_data.csv" \
    --labels_in_split_csv \
    --label_col g_level \
    --question q1 q2 q3 q4 q5 q6 \
    --video_dim 512 --video_dir "${FEAT_TRAIN}/video" \
    --audio_dim 512 --audio_dir "${FEAT_TRAIN}/audio" \
    --text_dim "${TEXT_DIM}" --text_dir "${TEXT_TRAIN_DIR}" \
    --val_video_dir "${FEAT_VAL}/video" --val_audio_dir "${FEAT_VAL}/audio" --val_text_dir "${TEXT_VAL_DIR}" \
    --test_video_dir "${FEAT_TEST}/video" --test_audio_dir "${FEAT_TEST}/audio" --test_text_dir "${TEXT_TEST_DIR}" \
    --batch_size "${AT_BATCH:-32}" --num_workers "${AT_WORKERS:-4}" \
    --num_epochs "${AT_EPOCHS:-200}" \
    --learning_rate "${AT_LR:-1e-4}" \
    --output_model "${ODIR}/best.pth" \
    --loss_plot_path "${ODIR}/loss.png" \
    --test_output_csv "${ODIR}/submission.csv" \
    --lr_scheduler_patience 5 \
    --early_stop_patience "${EARLY_STOP_PATIENCE:-40}" \
    --seed "${s}" \
    "${BASE_OPT[@]}"
done

echo "[at_baseline] done under ${OUT_ROOT}" >&2
