#!/usr/bin/env bash
# 计划阶段 A.2：在 Nanbeige 上对 label_smoothing / LEARNING_RATE / modality_dropout_p 做小网格，每格多 seed。
# 已存在 round1 大扫描且 seed37 达标 58.73% 时，本脚本仍可用于局部细调。
# 用法:
#   export HUNT_DIR=/abs/path/experiments/nb_to58_mini_grid/run1
#   export SEEDS_PER_CELL="99 123 37"
#   bash tools/run_nb_mini_hparam_grid.sh
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

HUNT_DIR="${HUNT_DIR:-${_ROOT}/experiments/nb_to58_mini_grid/default}"
case "$HUNT_DIR" in
  /*) ;;
  *) HUNT_DIR="${_ROOT}/${HUNT_DIR}" ;;
esac
mkdir -p "$HUNT_DIR"

# 默认 3 seed/格（计划写「至少 5」；完整基线已由 round1 大扫描覆盖，此处只做偏离基线的小网格）
SEEDS_PER_CELL="${SEEDS_PER_CELL:-99 123 37}"
OUT_CSV="${HUNT_DIR}/metrics_hparam_grid.csv"
echo "cell,label_smoothing,learning_rate,modality_dropout_p,seed,val_acc,val_macro_f1,val_bal_acc,best_epoch,epochs_run,output_model,log" >"$OUT_CSV"

run_cell() {
  local cell="$1"
  local ls="$2"
  local lr="$3"
  local md="$4"
  export NANBEIGE_TEXT=1 TEXT_DIM=2560
  export TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${_ROOT}/data/text_nb}"
  export TEXT_VAL_DIR="${TEXT_VAL_DIR:-${_ROOT}/data/text_nb_val}"
  export TEXT_TEST_DIR="${TEXT_TEST_DIR:-${_ROOT}/data/text_nb}"
  export NUM_WORKERS="${NUM_WORKERS:-4}"
  export EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-12}"
  export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-40}"
  export LR_SCHEDULER_PATIENCE="${LR_SCHEDULER_PATIENCE:-5}"
  for s in $SEEDS_PER_CELL; do
    local subdir="$HUNT_DIR/${cell}_seed${s}"
    mkdir -p "$subdir"
    export LEARNING_RATE="$lr"
    export GLEVEL_OPT="--g_level_int_encoding one --glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001 --label_smoothing ${ls} --select_best balanced_acc --cross_modal_attn --cross_modal_layers 1 --modality_dropout_p ${md} --scheduler_min_lr 1e-6 --sampler_medium_boost 1.5 --seed ${s}"
    export OUTPUT_MODEL="${subdir}/best.pth"
    export LOSS_PLOT_PATH="${subdir}/loss.png"
    export TEST_OUTPUT_CSV="${subdir}/submission.csv"
    local log="$subdir/train.log"
    echo "[mini_grid] ${cell} seed=${s}" >&2
    bash "${_ROOT}/scripts/glevel_train.sh" 2>&1 | tee "$log"
    local ml
    ml="$(grep '^\[metrics_line\]' "$log" | tail -n 1 || true)"
    if [[ -z "$ml" ]]; then
      echo "${cell},${ls},${lr},${md},${s},NA,NA,NA,NA,NA,${OUTPUT_MODEL},${log}" >>"$OUT_CSV"
      continue
    fi
    local va vf vb be er
    va="$(echo "$ml" | sed -n 's/.*val_acc=\([0-9.]*\).*/\1/p')"
    vf="$(echo "$ml" | sed -n 's/.*val_macro_f1=\([0-9.]*\).*/\1/p')"
    vb="$(echo "$ml" | sed -n 's/.*val_bal_acc=\([0-9.]*\).*/\1/p')"
    be="$(echo "$ml" | sed -n 's/.*best_epoch=\([0-9]*\).*/\1/p')"
    er="$(echo "$ml" | sed -n 's/.*epochs_run=\([0-9]*\).*/\1/p')"
    echo "${cell},${ls},${lr},${md},${s},${va},${vf},${vb},${be},${er},${OUTPUT_MODEL},${log}" >>"$OUT_CSV"
  done
}

# 仅跑某一格（例如 ONLY_CELL=c01_ls008）时设置 ONLY_CELL，否则跑全套非基线组合
if [ -n "${ONLY_CELL:-}" ]; then
  case "$ONLY_CELL" in
    c00_baseline) run_cell "c00_baseline" "0.05" "1e-4" "0.12" ;;
    c01_ls008) run_cell "c01_ls008" "0.08" "1e-4" "0.12" ;;
    c02_lr5e5) run_cell "c02_lr5e5" "0.05" "5e-5" "0.12" ;;
    c03_md016) run_cell "c03_md016" "0.05" "1e-4" "0.16" ;;
    c04_combo) run_cell "c04_combo" "0.08" "5e-5" "0.16" ;;
    *) echo "unknown ONLY_CELL=$ONLY_CELL" >&2; exit 2 ;;
  esac
else
  # 基线已与 round1/metrics_seeds.csv 中 GLEVEL_OPT 一致，默认不再重复训 c00（export RUN_BASELINE_CELL=1 可打开）
  if [ "${RUN_BASELINE_CELL:-0}" = "1" ]; then
    run_cell "c00_baseline" "0.05" "1e-4" "0.12"
  fi
  run_cell "c01_ls008" "0.08" "1e-4" "0.12"
  run_cell "c02_lr5e5" "0.05" "5e-5" "0.12"
  run_cell "c03_md016" "0.05" "1e-4" "0.16"
  run_cell "c04_combo" "0.08" "5e-5" "0.16"
fi

echo "[mini_grid] wrote $OUT_CSV" >&2
