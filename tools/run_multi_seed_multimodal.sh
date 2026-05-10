#!/usr/bin/env bash
# 多随机种子完整训练：与 scripts/glevel_train_multimodal.sh（MM_MEDIUM_BOOST）对齐的超参，
# 按验证集 val_acc 选出最优 checkpoint 复制到 hunt 目录（小 val 集下单 seed 方差大）。
#
# 用法（项目根）:
#   export PYTHON=/path/to/avi2026/bin/python   # 建议 CUDA 环境
#   bash tools/run_multi_seed_multimodal.sh
# 可选:
#   export SEEDS="42 7 99 123 2024 2026"
#   export MM_TEMPORAL=1
#   export NUM_WORKERS=0
#   export CUDA_VISIBLE_DEVICES=0
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

_feat="${FEAT_TRAIN:-/data/Super-Lu/dataset/train_feature}"
_val="${FEAT_VAL:-/data/Super-Lu/dataset/val_feature}"
if [ "$_feat" = "..." ] || [ "$_val" = "..." ]; then
  echo "[run_multi_seed_multimodal] FEAT_TRAIN/FEAT_VAL 不能为占位符 \"...\"" >&2
  exit 2
fi
for _d in "$_feat/audio" "$_feat/video" "$_val/audio" "$_val/video"; do
  if [ ! -d "$_d" ]; then
    echo "[run_multi_seed_multimodal] 特征目录不存在: $_d" >&2
    exit 2
  fi
done

SEEDS="${SEEDS:-42 7 99 123 2024}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
HUNT_DIR="${HUNT_DIR:-${_ROOT}/experiments/glevel_hunt_${TS}}"
mkdir -p "$HUNT_DIR"

SMB="${MM_SAMPLER_MEDIUM_BOOST:-1.5}"
# 与 scripts/glevel_train_multimodal.sh + MM_MEDIUM_BOOST=1 一致；select_best=balanced_acc 利于三分类均衡
BASE_PRESET="--g_level_int_encoding one --glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001 --label_smoothing 0.05 --select_best balanced_acc --cross_modal_attn --cross_modal_layers 1 --modality_dropout_p 0.12 --scheduler_min_lr 1e-6 --sampler_medium_boost ${SMB}"

if [ "${MM_TEMPORAL:-0}" = "1" ]; then
  BASE_PRESET="$BASE_PRESET --temporal_gru --temporal_pool mean --temporal_dropout 0.12"
  echo "[run_multi_seed_multimodal] MM_TEMPORAL=1" >&2
fi
if [ "${MM_BIDIRECTIONAL:-0}" = "1" ]; then
  BASE_PRESET="$BASE_PRESET --temporal_bidirectional --temporal_attn_pool"
  echo "[run_multi_seed_multimodal] MM_BIDIRECTIONAL=1" >&2
fi

OUT_CSV="$HUNT_DIR/metrics_seeds.csv"
echo "seed,val_acc,val_macro_f1,val_bal_acc,best_epoch,epochs_run,output_model,log" >"$OUT_CSV"

for s in $SEEDS; do
  tag="seed${s}"
  log="$HUNT_DIR/${tag}.log"
  outp="$HUNT_DIR/best_mm_${tag}.pth"
  echo "[multi_seed] seed=$s → $log" >&2
  {
    echo "=== seed=$s at $(date -Iseconds) ==="
    export GLEVEL_OPT="$BASE_PRESET --seed $s"
    export OUTPUT_MODEL="$outp"
    export LOSS_PLOT_PATH="$HUNT_DIR/loss_${tag}.png"
    export VAL_ERRORS_CSV="$HUNT_DIR/val_err_${tag}.csv"
    export NUM_WORKERS="$NUM_WORKERS"
    export EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-15}"
    export TEST_OUTPUT_CSV="$HUNT_DIR/submission_${tag}.csv"
    # 直接调用 vote_train：GLEVEL_OPT 已含全部训练超参，勿再走 multimodal 的 MM_PRESET 分支
    bash "${_ROOT}/scripts/glevel_train.sh"
  } 2>&1 | tee "$log"

  ml="$(grep '^\[metrics_line\]' "$log" | tail -n 1 || true)"
  if [ -z "$ml" ]; then
    echo "[multi_seed] WARN: 无 [metrics_line] seed=$s" >&2
    echo "$s,NA,NA,NA,NA,NA,$outp,$log" >>"$OUT_CSV"
    continue
  fi
  va="$(echo "$ml" | sed -n 's/.*val_acc=\([0-9.]*\).*/\1/p')"
  vf="$(echo "$ml" | sed -n 's/.*val_macro_f1=\([0-9.]*\).*/\1/p')"
  vb="$(echo "$ml" | sed -n 's/.*val_bal_acc=\([0-9.]*\).*/\1/p')"
  be="$(echo "$ml" | sed -n 's/.*best_epoch=\([0-9]*\).*/\1/p')"
  er="$(echo "$ml" | sed -n 's/.*epochs_run=\([0-9]*\).*/\1/p')"
  echo "$s,$va,$vf,$vb,$be,$er,$outp,$log" >>"$OUT_CSV"
done

echo "[multi_seed] 汇总: $OUT_CSV" >&2
BEST_PTH="$HUNT_DIR/best_by_val_acc.pth"
WINNER_TXT="$HUNT_DIR/WINNER.txt"
python3 - "$OUT_CSV" "$BEST_PTH" "$WINNER_TXT" <<'PY'
import csv, shutil, sys

out_csv, best_pth, winner_txt = sys.argv[1:4]
with open(out_csv, newline="") as f:
    rows = list(csv.DictReader(f))
ok = [r for r in rows if r.get("val_acc") not in (None, "", "NA")]
if not ok:
    print("[multi_seed] 无有效 val_acc，跳过复制最优权重")
    sys.exit(0)


def key(r):
    return (float(r["val_acc"]), float(r.get("val_bal_acc") or 0))


best = max(ok, key=key)
src = best["output_model"]
shutil.copy2(src, best_pth)
lines = [
    f"best_seed={best['seed']}",
    f"val_acc={best['val_acc']}",
    f"val_bal_acc={best.get('val_bal_acc', '')}",
    f"val_macro_f1={best.get('val_macro_f1', '')}",
    f"best_epoch={best.get('best_epoch', '')}",
    f"source_checkpoint={src}",
    f"copied_to={best_pth}",
]
with open(winner_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("[multi_seed] 最优（按 val_acc，平局看 val_bal_acc）:", lines[0], lines[1])
print("[multi_seed] 已复制 →", best_pth)
PY
