#!/usr/bin/env bash
# Exp0–5 × seeds 37/42/99：可学习头/时间权重/文本增强 消融。
# 训练后用 eval_glevel_checkpoint_on_csv 在留出集与官方 val 上各评一次，metrics 写入 summary TSV。
#
# 用法（在工程根；Nanbeige 需先 export 文本目录）:
#   export NANBEIGE_TEXT=1 TEXT_DIM=2560
#   export TEXT_TRAIN_DIR=$PWD/data/text_nb TEXT_VAL_DIR=$PWD/data/text_nb_val
#   source tools/glevel_paths.inc.sh
#   bash tools/run_glevel_weight_enhancer_ablation.sh
#
# 子集: EXP_IDS=3 SEEDS=37 bash tools/run_glevel_weight_enhancer_ablation.sh
# 仅 eval 已有权重: SKIP_TRAIN=1 bash tools/run_glevel_weight_enhancer_ablation.sh
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

AB_ROOT="${_ROOT}/experiments/ablation_learned_agg"
mkdir -p "${AB_ROOT}"
HOLDOUT_TRAIN="${HOLDOUT_TRAIN:-${AB_ROOT}/train_holdout_sl.csv}"
HOLDOUT_VAL="${HOLDOUT_VAL:-${AB_ROOT}/dev_holdout_sl.csv}"
SUMMARY_TSV="${SUMMARY_TSV:-${AB_ROOT}/summary_runs.tsv}"

# 与 run_official_test_submit_nanbeige 一致（勿含 --seed，种子在循环内单独传）
BASE_GLEVEL_OPT="${BASE_GLEVEL_OPT:---g_level_int_encoding one --glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001 --label_smoothing 0.05 --select_best balanced_acc --cross_modal_attn --cross_modal_layers 1 --modality_dropout_p 0.12 --scheduler_min_lr 1e-6 --sampler_medium_boost 1.5}"

SPLIT_SEED="${SPLIT_SEED:-37}"
EXP_IDS="${EXP_IDS:-0 1 2 3 4 5}"
SEEDS="${SEEDS:-37 42 99}"

if [[ ! -f "${HOLDOUT_VAL}" ]]; then
  echo "[ablation] 生成留出集: ${HOLDOUT_TRAIN} ${HOLDOUT_VAL}" >&2
  "${PY}" "${_SCRIPT_DIR}/split_train_val.py" \
    --in_csv "${SUP}/train_data.csv" \
    --out_train "${HOLDOUT_TRAIN}" \
    --out_val "${HOLDOUT_VAL}" \
    --val_ratio 0.2 --seed "${SPLIT_SEED}"
fi

if [[ ! -f "${SUMMARY_TSV}" ]]; then
  echo -e "exp_id\tseed\tckpt\tholdout_metrics_line\tval_metrics_line" > "${SUMMARY_TSV}"
fi

extra_flags_for_exp() {
  case "$1" in
    0) echo "" ;;
    1) echo "--head_weights" ;;
    2) echo "--time_weights" ;;
    3) echo "--text_enhancer transformer" ;;
    4) echo "--head_weights --time_weights" ;;
    5) echo "--head_weights --time_weights --text_enhancer transformer" ;;
    *) echo "unknown exp_id=$1" >&2; exit 2 ;;
  esac
}

for e in ${EXP_IDS}; do
  EXTRA="$(extra_flags_for_exp "${e}")"
  for s in ${SEEDS}; do
    TAG="exp${e}_seed${s}"
    OUT_DIR="${AB_ROOT}/${TAG}"
    mkdir -p "${OUT_DIR}"
    OUT_PTH="${OUT_DIR}/best.pth"

    if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
      echo "[ablation] TRAIN exp=${e} seed=${s}" >&2
      # shellcheck disable=2086
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
        --batch_size "${ABLATION_BATCH:-32}" --num_workers "${ABLATION_WORKERS:-4}" \
        --num_epochs "${ABLATION_EPOCHS:-200}" \
        --learning_rate "${ABLATION_LR:-1e-4}" \
        --output_model "${OUT_PTH}" \
        --loss_plot_path "${OUT_DIR}/loss.png" \
        --test_output_csv "${OUT_DIR}/submission.csv" \
        --lr_scheduler_patience "${LR_SCHEDULER_PATIENCE:-5}" \
        --early_stop_patience "${EARLY_STOP_PATIENCE:-40}" \
        --early_stop_min_epochs "${EARLY_STOP_MIN_EPOCHS:-12}" \
        --seed "${s}" \
        ${BASE_GLEVEL_OPT} \
        ${EXTRA}
    else
      echo "[ablation] SKIP_TRAIN exp=${e} seed=${s}" >&2
    fi

    # shellcheck disable=2086
    H_LINE="$("${PY}" "${_SCRIPT_DIR}/eval_glevel_checkpoint_on_csv.py" \
      --eval_csv "${HOLDOUT_VAL}" \
      --rating_csv "${SUP}/train_data.csv" \
      --labels_in_split_csv --g_level_int_encoding one \
      --cross_modal_attn --cross_modal_layers 1 \
      --train_audio_dir "${FEAT_TRAIN}/audio" --train_video_dir "${FEAT_TRAIN}/video" --train_text_dir "${TEXT_TRAIN_DIR}" \
      --text_dim "${TEXT_DIM}" --audio_dim 512 --video_dim 512 \
      --batch_size "${ABLATION_EVAL_BATCH:-32}" --num_workers 0 \
      --mlp_dropout 0.25 --modality_dropout_p 0.12 \
      --checkpoint "${OUT_PTH}" \
      ${EXTRA} 2>/dev/null | grep '^\[metrics_line_local\]' | tail -n 1 || true)"

    # shellcheck disable=2086
    V_LINE="$("${PY}" "${_SCRIPT_DIR}/eval_glevel_checkpoint_on_csv.py" \
      --eval_csv "${SUP}/val_data.csv" \
      --rating_csv "${SUP}/train_data.csv" \
      --labels_in_split_csv --g_level_int_encoding one \
      --cross_modal_attn --cross_modal_layers 1 \
      --train_audio_dir "${FEAT_TRAIN}/audio" --train_video_dir "${FEAT_TRAIN}/video" --train_text_dir "${TEXT_TRAIN_DIR}" \
      --eval_audio_dir "${FEAT_VAL}/audio" --eval_video_dir "${FEAT_VAL}/video" --eval_text_dir "${TEXT_VAL_DIR}" \
      --text_dim "${TEXT_DIM}" --audio_dim 512 --video_dim 512 \
      --batch_size "${ABLATION_EVAL_BATCH:-32}" --num_workers 0 \
      --mlp_dropout 0.25 --modality_dropout_p 0.12 \
      --checkpoint "${OUT_PTH}" \
      ${EXTRA} 2>/dev/null | grep '^\[metrics_line_local\]' | tail -n 1 || true)"

    printf '%s\t%s\t%s\t%s\t%s\n' "${e}" "${s}" "${OUT_PTH}" "${H_LINE}" "${V_LINE}" >> "${SUMMARY_TSV}"
    echo "[ablation] done ${TAG}" >&2
  done
done

echo "[ablation] 汇总: ${SUMMARY_TSV}" >&2
