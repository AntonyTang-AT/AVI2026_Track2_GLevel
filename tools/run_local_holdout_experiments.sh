#!/usr/bin/env bash
# 在 train 分层留出集上对比：基线、温度缩放、OvR、特征标准化、多 seed 集成。
# 依赖：Super-Lu train/val 特征、工程内 data/text_nb & data/text_nb_val、权重 .pth。
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${_ROOT}"

PY="${PYTHON:-${_ROOT}/.venv_glevel_cpu/bin/python}"
SUP="${SUPERLU_DATASET:-/data/Super-Lu/dataset}"
mkdir -p "${_ROOT}/experiments/logs"

CKPT_MAIN="${CKPT_MAIN:-${_ROOT}/experiments/nb_to58_sweep/round1/seed37/best.pth}"
CKPT_A="${CKPT_A:-${_ROOT}/experiments/nb_to58_sweep/round1/seed42/best.pth}"
CKPT_B="${CKPT_B:-${_ROOT}/experiments/nb_to58_sweep/round1/seed17/best.pth}"

echo "[run_local_holdout_experiments] PY=$PY SUPERLU=$SUP" | tee "${_ROOT}/experiments/logs/holdout_experiments.log"

"${PY}" "${_SCRIPT_DIR}/split_train_val.py" \
  --in_csv "${SUP}/train_data.csv" \
  --out_train "${_ROOT}/experiments/train_holdout_sl.csv" \
  --out_val "${_ROOT}/experiments/dev_holdout_sl.csv" \
  --val_ratio 0.2 --seed 37 \
  | tee -a "${_ROOT}/experiments/logs/holdout_experiments.log"

COMMON=( --eval_csv "${_ROOT}/experiments/dev_holdout_sl.csv"
  --rating_csv "${SUP}/train_data.csv"
  --labels_in_split_csv
  --g_level_int_encoding one
  --cross_modal_attn --cross_modal_layers 1
  --train_audio_dir "${SUP}/train_feature/audio"
  --train_video_dir "${SUP}/train_feature/video"
  --train_text_dir "${_ROOT}/data/text_nb"
  --text_dim 2560
  --batch_size "${HOLDOUT_BATCH:-8}"
  --num_workers "${HOLDOUT_WORKERS:-0}"
)

run_one () {
  local name="$1"
  shift
  echo "" | tee -a "${_ROOT}/experiments/logs/holdout_experiments.log"
  echo "========== ${name} ==========" | tee -a "${_ROOT}/experiments/logs/holdout_experiments.log"
  "${PY}" "${_SCRIPT_DIR}/eval_glevel_checkpoint_on_csv.py" "${COMMON[@]}" "$@" \
    2>&1 | tee -a "${_ROOT}/experiments/logs/holdout_experiments.log"
}

PROBS="${_ROOT}/experiments/logs/dev_holdout_seed37_probs.npz"
NPZ_NORM="${_ROOT}/experiments/feat_norm/train_holdout_sl_mean_std.npz"
TEMP_JSON="${_ROOT}/experiments/logs/temperature_seed37.json"
OVR_JSON="${_ROOT}/experiments/logs/ovr_seed37.json"

run_one "1_baseline_seed37_T1" --checkpoint "${CKPT_MAIN}" --logit_temperature 1.0 --dump_probs "${PROBS}"

"${PY}" "${_SCRIPT_DIR}/compute_feat_mean_std.py" \
  --train_csv "${_ROOT}/experiments/train_holdout_sl.csv" \
  --audio_dir "${SUP}/train_feature/audio" \
  --video_dir "${SUP}/train_feature/video" \
  --text_dir "${_ROOT}/data/text_nb" \
  --out_npz "${NPZ_NORM}" \
  | tee -a "${_ROOT}/experiments/logs/holdout_experiments.log"

run_one "2_feat_norm_seed37" --checkpoint "${CKPT_MAIN}" --logit_temperature 1.0 \
  --feat_norm_npz "${NPZ_NORM}" --feat_norm_apply all

"${PY}" "${_SCRIPT_DIR}/fit_temperature_scaling.py" \
  --probs_npz "${PROBS}" --out_json "${TEMP_JSON}" \
  | tee -a "${_ROOT}/experiments/logs/holdout_experiments.log"

TVAL="$("${PY}" -c "import json; print(json.load(open('${TEMP_JSON}'))['T'])")"
run_one "3_temperature_scaled_seed37" --checkpoint "${CKPT_MAIN}" --logit_temperature "${TVAL}"

"${PY}" "${_SCRIPT_DIR}/tune_ovr_thresholds.py" \
  --probs_npz "${PROBS}" --out_json "${OVR_JSON}" --grid_steps 15 \
  | tee -a "${_ROOT}/experiments/logs/holdout_experiments.log"

run_one "4_ovr_tuned_seed37_T1" --checkpoint "${CKPT_MAIN}" --logit_temperature 1.0 \
  --ovr_thresholds_json "${OVR_JSON}"

run_one "5_ensemble_seed37_42_17_T1" \
  --ensemble_checkpoints "${CKPT_MAIN}" "${CKPT_A}" "${CKPT_B}" \
  --logit_temperature 1.0

echo "" | tee -a "${_ROOT}/experiments/logs/holdout_experiments.log"
echo "========== official_val_seed37 (封板参考，勿反复调参) ==========" | tee -a "${_ROOT}/experiments/logs/holdout_experiments.log"
VAL_COMMON=( --eval_csv "${SUP}/val_data.csv"
  --rating_csv "${SUP}/train_data.csv"
  --labels_in_split_csv
  --g_level_int_encoding one
  --cross_modal_attn --cross_modal_layers 1
  --train_audio_dir "${SUP}/train_feature/audio"
  --train_video_dir "${SUP}/train_feature/video"
  --train_text_dir "${_ROOT}/data/text_nb"
  --eval_audio_dir "${SUP}/val_feature/audio"
  --eval_video_dir "${SUP}/val_feature/video"
  --eval_text_dir "${_ROOT}/data/text_nb_val"
  --text_dim 2560
  --batch_size "${HOLDOUT_BATCH:-8}"
  --num_workers "${HOLDOUT_WORKERS:-0}"
)
"${PY}" "${_SCRIPT_DIR}/eval_glevel_checkpoint_on_csv.py" "${VAL_COMMON[@]}" \
  --checkpoint "${CKPT_MAIN}" --logit_temperature 1.0 \
  2>&1 | tee -a "${_ROOT}/experiments/logs/holdout_experiments.log"

echo "[run_local_holdout_experiments] done. Grep metrics:" | tee -a "${_ROOT}/experiments/logs/holdout_experiments.log"
grep '\[metrics_line_local\]' "${_ROOT}/experiments/logs/holdout_experiments.log" | tee -a "${_ROOT}/experiments/logs/holdout_experiments.log"
