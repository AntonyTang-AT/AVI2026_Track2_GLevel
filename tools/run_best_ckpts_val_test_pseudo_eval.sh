#!/usr/bin/env bash
# 对多条「最优路线」checkpoint 依次：官方 val + 测试集导出 CSV + 与伪标签一致率（非官方 test acc）。
# 默认 conda magnus + CUDA；日志与汇总表写入 experiments/glevel_improvement_plan/
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

_MAGNUS="${GLEVEL_CUDA_PYTHON:-/home/emo/anaconda3/envs/magnus/bin/python}"
OUT_DIR="${_ROOT}/external/submissions_peer"
SUMMARY="${_ROOT}/experiments/glevel_improvement_plan/batch_infer_eval_summary.tsv"
LOG="${_ROOT}/experiments/glevel_improvement_plan/batch_infer_eval.log"
mkdir -p "$(dirname "${SUMMARY}")"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "run_name" "checkpoint" "val_metrics" "pseudo_agree_all" "pseudo_agree_unanimous" "submission_csv" \
  >"${SUMMARY}"

# 四 peer（README）共识伪标签；可与 ONLY_UNANIMOUS=1 组合只看高置信子集
PEER_PSEUDO="${PEER_PSEUDO:-${_ROOT}/external/submissions_peer/test_pseudo_four_peer_consensus_123.csv}"
ONLY_UNANIMOUS="${ONLY_UNANIMOUS:-0}"

# S_ref_plateau 与 seed37 归档一致（Nanbeige + shared_mlp + cross_attn）
BASE_S_REF=(
  --g_level_int_encoding one --glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001
  --label_smoothing 0.05 --cross_modal_attn --cross_modal_layers 1 --modality_dropout_p 0.12
  --scheduler_min_lr 1e-6 --sampler_medium_boost 1.5 --select_best balanced_acc --seed 37
)

PHASEB_BASE=(
  --g_level_int_encoding one --glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001
  --label_smoothing 0.05 --cross_modal_attn --cross_modal_layers 1 --modality_dropout_p 0.12
  --scheduler_min_lr 1e-6 --sampler_medium_boost 1.5
)

PHASEC_INF=(
  --g_level_int_encoding one --glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001
  --label_smoothing 0.05 --select_best balanced_acc --cross_modal_attn --cross_modal_layers 1
  --modality_dropout_p 0.12 --scheduler_min_lr 1e-6 --sampler_medium_boost 1.5 --seed 37
)

echo "[batch_infer_eval] start $(date -Iseconds) PYTHON=${_MAGNUS}" | tee -a "${LOG}"

_append_summary() {
  local name="$1" ckpt="$2" val_line="$3" pseudo_agree="$4" pseudo_agree_uni="$5" sub_csv="$6"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${ckpt}" "${val_line}" "${pseudo_agree}" "${pseudo_agree_uni}" "${sub_csv}" \
    >>"${SUMMARY}"
}

_run_job() {
  local name="$1"
  local ckpt="$2"
  shift 2
  local -a gopt=( "$@" )
  local sub_csv="${OUT_DIR}/submission_batch_infer_${name}.csv"

  echo "" | tee -a "${LOG}"
  echo "[batch_infer_eval] === ${name} ===" | tee -a "${LOG}"
  echo "[batch_infer_eval] ckpt=${ckpt}" | tee -a "${LOG}"

  local val_line pseudo_a pseudo_u
  val_line="$(
    NANBEIGE_TEXT="${NANBEIGE_TEXT:-1}" TEXT_DIM="${TEXT_DIM:-2560}" \
    PYTHON="${_MAGNUS}" TEST_MODEL="${ckpt}" TEST_OUTPUT_CSV="${sub_csv}" \
    GLEVEL_OPT="${gopt[*]}" \
    bash "${_ROOT}/scripts/glevel_test.sh" 2>&1 | tee -a "${LOG}" | tr -d '\r' | grep '\[only_test:single\]' | tail -n 1 | sed 's/.*\[only_test:single\] //' || true
  )"

  if [[ ! -f "${sub_csv}" ]]; then
    echo "[batch_infer_eval] ERROR missing ${sub_csv}" | tee -a "${LOG}"
    _append_summary "${name}" "${ckpt}" "${val_line:-NA}" "NA" "NA" "${sub_csv}"
    return 1
  fi

  pseudo_a="$("${_MAGNUS}" "${_ROOT}/tools/score_submission_vs_pseudo.py" "${PEER_PSEUDO}" "${sub_csv}" 2>&1 | tee -a "${LOG}" | grep '^\[pseudo_eval\] n=' | tail -n 1 | sed 's/.*agreement=//' | awk '{print $1}' || true)"
  pseudo_u="$("${_MAGNUS}" "${_ROOT}/tools/score_submission_vs_pseudo.py" --only-unanimous "${PEER_PSEUDO}" "${sub_csv}" 2>&1 | tee -a "${LOG}" | grep '^\[pseudo_eval\] n=' | tail -n 1 | sed 's/.*agreement=//' | awk '{print $1}' || true)"

  _append_summary "${name}" "${ckpt}" "${val_line:-NA}" "${pseudo_a:-NA}" "${pseudo_u:-NA}" "${sub_csv}"
  echo "[batch_infer_eval] wrote ${sub_csv}" | tee -a "${LOG}"
}

# --- jobs：多思路最优 ckpt（路径可按环境 export 覆盖） ---
ARCHIVE37="${ARCHIVE37_CKPT:-/data/emo/glevel_runs/archives/nb_to58_sweep/round1/seed37/best.pth}"

_run_job "seed37_archive_S_ref_plateau" "${ARCHIVE37}" "${BASE_S_REF[@]}" || true

_run_job "seed37_archive_inferbias070" "${ARCHIVE37}" "${BASE_S_REF[@]}" --infer_logit_bias 0,0.7,0 || true

_run_job "phaseB_coral_bal_s42" "${_ROOT}/experiments/glevel_improvement_plan/phaseB_coral_bal_seed42_best.pth" \
  "${PHASEB_BASE[@]}" --glevel_loss coral --select_best balanced_acc --seed 42 || true

_run_job "phaseB_ce_bal_s42" "${_ROOT}/experiments/glevel_improvement_plan/phaseB_ce_bal_seed42_best.pth" \
  "${PHASEB_BASE[@]}" --glevel_loss ce --select_best balanced_acc --seed 42 || true

_run_job "phaseB_ce_classweight_s42" "${_ROOT}/experiments/glevel_improvement_plan/phaseB_ce_bal_classweight_auto_seed42_best.pth" \
  "${PHASEB_BASE[@]}" --glevel_loss ce --select_best balanced_acc --class_weight auto --seed 42 || true

_run_job "phaseB_ce_macrof1_s42" "${_ROOT}/experiments/glevel_improvement_plan/phaseB_ce_macrof1_seed42_best.pth" \
  "${PHASEB_BASE[@]}" --glevel_loss ce --select_best macro_f1 --seed 42 || true

_run_job "phaseC_pseudo_finetune_s37" "${_ROOT}/experiments/glevel_improvement_plan/phaseC_pseudo_finetune_best.pth" \
  "${PHASEC_INF[@]}" || true

echo "[batch_infer_eval] summary → ${SUMMARY}" | tee -a "${LOG}"
echo "[batch_infer_eval] done $(date -Iseconds)" | tee -a "${LOG}"
