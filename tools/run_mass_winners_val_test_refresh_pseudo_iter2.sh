#!/usr/bin/env bash
# 1) 从 combo_sweep_metrics.csv 读取验证集 Top-N 有效任务，vote_test → 测试集 CSV + 刷新 val 行。
# 2) 对归档 seed37（plain + infer bias）再跑一遍 vote_test。
# 3) 伪标签迭代：四 peer + inferbias 提交 + 上述 Top-N 提交 → build_consensus_pseudo_k → merge unanimous train。
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

_MAGNUS="${GLEVEL_CUDA_PYTHON:-/home/emo/anaconda3/envs/magnus/bin/python}"
SP="${_ROOT}/external/submissions_peer"
PLAN="${_ROOT}/experiments/glevel_improvement_plan"
METRICS="${SWEEP_METRICS_CSV:-${_ROOT}/experiments/gpu_combo_sweep/mass_20260509_145045/combo_sweep_metrics.csv}"
TOP_N="${TOP_N:-6}"
ARCHIVE37="${ARCHIVE37_CKPT:-/data/emo/glevel_runs/archives/nb_to58_sweep/round1/seed37/best.pth}"

OUT_PSEUDO="${OUT_PSEUDO_CSV:-${SP}/test_pseudo_iter2_peers4_inferbias_top${TOP_N}_mass.csv}"
OUT_TRAIN="${OUT_TRAIN_MERGED:-${PLAN}/train_plus_pseudo_unanimous.csv}"
SUMMARY="${PLAN}/mass_winners_refresh_iter2_summary.tsv"
LOG="${PLAN}/mass_winners_refresh_iter2.log"
mkdir -p "${PLAN}" "${SP}"

PEERS=(
  "${SP}/submission1_0.53077.csv"
  "${SP}/submission2_0.50769.csv"
  "${SP}/submission_0.53846.csv"
  "${SP}/submission5_0.55385.csv"
)

BASE_LEGACY=(
  --g_level_int_encoding one --glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001
  --label_smoothing 0.05 --cross_modal_attn --cross_modal_layers 1 --modality_dropout_p 0.12
  --scheduler_min_lr 1e-6 --sampler_medium_boost 1.5 --select_best balanced_acc --seed 37
)

echo "[mass_winners_iter2] start $(date -Iseconds) PYTHON=${_MAGNUS}" | tee -a "${LOG}"
printf '%s\t%s\t%s\t%s\t%s\n' "kind" "name" "checkpoint" "val_metrics" "submission_csv" >"${SUMMARY}"

_gopt_for_combo_seed() {
  local combo="$1" seed="$2"
  local -a o=(
    --g_level_int_encoding one --mlp_dropout 0.25 --weight_decay 0.001 --label_smoothing 0.05
    --select_best balanced_acc --modality_dropout_p 0.12 --scheduler_min_lr 1e-6 --sampler_medium_boost 1.5
    --seed "${seed}"
  )
  case "${combo}" in
    AT_plateau)
      o+=(--glevel_arch audio_text_mlp --at_mlp_hidden 512)
      ;;
    AT_plateau_ln)
      o+=(--glevel_arch audio_text_mlp --at_mlp_hidden 512 --fused_layer_norm)
      ;;
    AT_step_ln)
      o+=(
        --glevel_arch audio_text_mlp --at_mlp_hidden 512 --fused_layer_norm
        --lr_scheduler step --lr_step_size 30 --lr_gamma 0.5
      )
      ;;
    *)
      o+=(--glevel_arch shared_mlp --cross_modal_attn --cross_modal_layers 1)
      case "${combo}" in
        S_plateau_ln | S_plateau_ln_sel_acc | S_step_ln)
          o+=(--fused_layer_norm)
          ;;
      esac
      case "${combo}" in
        S_ref_cosine | S_ref_cosine)
          o+=(--lr_scheduler cosine)
          ;;
      esac
      case "${combo}" in
        S_ref_step | S_step_ln)
          o+=(--lr_scheduler step --lr_step_size 30 --lr_gamma 0.5)
          ;;
      esac
      ;;
  esac
  echo "${o[@]}"
}

_vote_one() {
  local kind="$1" name="$2" ckpt="$3"
  shift 3
  local -a gopt=( "$@" )
  local safe="${name//\//_}"
  local sub_csv="${SP}/submission_${safe}.csv"

  [[ -f "${ckpt}" ]] || {
    echo "[mass_winners_iter2] skip missing ckpt ${ckpt}" | tee -a "${LOG}"
    printf '%s\t%s\t%s\t%s\t%s\n' "${kind}" "${name}" "${ckpt}" "SKIP_NO_CKPT" "${sub_csv}" >>"${SUMMARY}"
    return 1
  }

  echo "" | tee -a "${LOG}"
  echo "[mass_winners_iter2] === ${kind} ${name} ===" | tee -a "${LOG}"
  local val_line
  val_line="$(
    NANBEIGE_TEXT="${NANBEIGE_TEXT:-1}" TEXT_DIM="${TEXT_DIM:-2560}" \
      PYTHON="${_MAGNUS}" TEST_MODEL="${ckpt}" TEST_OUTPUT_CSV="${sub_csv}" \
      GLEVEL_OPT="${gopt[*]}" \
      bash "${_ROOT}/scripts/glevel_test.sh" 2>&1 | tee -a "${LOG}" | tr -d '\r' | grep '\[only_test:single\]' | tail -n 1 | sed 's/.*\[only_test:single\] //' || true
  )"
  printf '%s\t%s\t%s\t%s\t%s\n' "${kind}" "${name}" "${ckpt}" "${val_line:-NA}" "${sub_csv}" >>"${SUMMARY}"
  echo "[mass_winners_iter2] wrote ${sub_csv}" | tee -a "${LOG}"
}

[[ -f "${METRICS}" ]] || {
  echo "[mass_winners_iter2] missing METRICS=${METRICS}" | tee -a "${LOG}"
  exit 2
}

TMP_TOP="${PLAN}/._top_sweep_jobs.tsv"
python3 <<PY >"${TMP_TOP}"
import csv
path = "${METRICS}"
n = int("${TOP_N}")
rows = []
with open(path, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        outp = (r.get("output_model") or "").strip()
        if not outp:
            continue
        if str(r.get("exit_code", "")).strip() != "0":
            continue
        va = r.get("val_acc") or ""
        if va in ("", "NA"):
            continue
        try:
            fv = float(va)
        except ValueError:
            continue
        rows.append((fv, r["combo_id"], str(r["seed"]).strip(), outp))
rows.sort(key=lambda x: -x[0])
for fv, combo, seed, outp in rows[:n]:
    print(f"{combo}\t{seed}\t{outp}\t{fv}")
PY
cat "${TMP_TOP}" >>"${LOG}"

MASS_SUBMISSIONS=()
while IFS=$'\t' read -r combo seed ckpt _va; do
  [[ -z "${combo}" ]] && continue
  safe="masswinner_${combo}_seed${seed}"
  gopt=( $(_gopt_for_combo_seed "${combo}" "${seed}") )
  _vote_one "mass_top" "${safe}" "${ckpt}" "${gopt[@]}" || true
  MASS_SUBMISSIONS+=( "${SP}/submission_${safe}.csv" )
done <"${TMP_TOP}"

_vote_one "legacy" "rerun_seed37_archive_plain" "${ARCHIVE37}" "${BASE_LEGACY[@]}" || true
_vote_one "legacy" "rerun_seed37_archive_inferbias070" "${ARCHIVE37}" "${BASE_LEGACY[@]}" --infer_logit_bias 0,0.7,0 || true

INFER_SUB="${SP}/submission_rerun_seed37_archive_inferbias070.csv"
for p in "${PEERS[@]}" "${INFER_SUB}" "${MASS_SUBMISSIONS[@]}"; do
  [[ -f "${p}" ]] || {
    echo "[mass_winners_iter2] ERROR missing ${p}" | tee -a "${LOG}"
    exit 3
  }
done

if [[ -f "${OUT_TRAIN}" ]]; then
  cp -a "${OUT_TRAIN}" "${PLAN}/train_plus_pseudo_unanimous_backup_pre_iter2_$(date +%Y%m%d_%H%M%S).csv"
  echo "[mass_winners_iter2] backed up prior ${OUT_TRAIN}" | tee -a "${LOG}"
fi

python3 "${_ROOT}/tools/build_consensus_pseudo_k.py" --out "${OUT_PSEUDO}" \
  "${PEERS[@]}" "${INFER_SUB}" "${MASS_SUBMISSIONS[@]}"

python3 "${_ROOT}/tools/merge_train_with_pseudo_test_rows.py" \
  --train-csv "/data/Super-Lu/dataset/train_data.csv" \
  --test-basic-csv "/data/Super-Lu/dataset/test_data_basic_information.csv" \
  --pseudo-csv "${OUT_PSEUDO}" \
  --out "${OUT_TRAIN}" \
  --require-unanimous

echo "[mass_winners_iter2] pseudo=${OUT_PSEUDO}" | tee -a "${LOG}"
echo "[mass_winners_iter2] merged_train=${OUT_TRAIN}" | tee -a "${LOG}"
echo "[mass_winners_iter2] summary=${SUMMARY}" | tee -a "${LOG}"
echo "[mass_winners_iter2] done $(date -Iseconds)" | tee -a "${LOG}"
