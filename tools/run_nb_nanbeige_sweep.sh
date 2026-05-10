#!/usr/bin/env bash
# Nanbeige 2560 多 seed 训练扫描（与 experiments/nb_hunt_20260508/WINNER 超参对齐）。
# 用法（项目根）:
#   export PYTHON=/path/to/avi2026/bin/python CUDA_VISIBLE_DEVICES=0
#   export SEEDS="7 42 99 ..."   # 可选，空格分隔
#   export HUNT_DIR=./experiments/nb_to58_sweep/run1
#   export GLEVEL_EXTRA="--label_smoothing 0.06"   # 可选，追加到 BASE_GLEVEL_OPT
#   bash tools/run_nb_nanbeige_sweep.sh
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

PYTHON="${PYTHON:-python}"
HUNT_DIR="${HUNT_DIR:-${_ROOT}/experiments/nb_to58_sweep/default}"
case "$HUNT_DIR" in
  /*) ;;
  *) HUNT_DIR="${_ROOT}/${HUNT_DIR}" ;;
esac
mkdir -p "$HUNT_DIR"

# 默认 30 个 seed（不含 0，部分环境上 seed=0 曾触发 CUDA 异常）
SEEDS="${SEEDS:-7 42 2024 123 99 11 17 13 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 87 91 97 101 103 107}"
APPEND_CSV="${APPEND_CSV:-0}"

export NANBEIGE_TEXT=1 TEXT_DIM=2560
export TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${_ROOT}/data/text_nb}"
export TEXT_VAL_DIR="${TEXT_VAL_DIR:-${_ROOT}/data/text_nb_val}"
export TEXT_TEST_DIR="${TEXT_TEST_DIR:-${_ROOT}/data/text_nb}"

export NUM_WORKERS="${NUM_WORKERS:-4}"
export EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-12}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-40}"
export LR_SCHEDULER_PATIENCE="${LR_SCHEDULER_PATIENCE:-5}"

BASE_GLEVEL_OPT="--g_level_int_encoding one --glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001 --label_smoothing 0.05 --select_best balanced_acc --cross_modal_attn --cross_modal_layers 1 --modality_dropout_p 0.12 --scheduler_min_lr 1e-6 --sampler_medium_boost 1.5"
GLEVEL_EXTRA="${GLEVEL_EXTRA:-}"

OUT_CSV="${HUNT_DIR}/metrics_seeds.csv"
if [[ "$APPEND_CSV" != "1" ]] || [[ ! -s "$OUT_CSV" ]]; then
  echo "seed,val_acc,val_macro_f1,val_bal_acc,best_epoch,epochs_run,output_model,log,exit_code" >"$OUT_CSV"
fi

for s in $SEEDS; do
  tag="seed${s}"
  subdir="$HUNT_DIR/${tag}"
  mkdir -p "$subdir"
  log="$subdir/train.log"
  export GLEVEL_OPT="${BASE_GLEVEL_OPT} ${GLEVEL_EXTRA} --seed ${s}"
  export OUTPUT_MODEL="${subdir}/best.pth"
  export LOSS_PLOT_PATH="${subdir}/loss.png"
  export TEST_OUTPUT_CSV="${subdir}/submission.csv"
  echo "[nb_sweep] ${tag} → $log" >&2
  ec=0
  bash "${_ROOT}/vote_train_glevel.sh" 2>&1 | tee "$log" || ec=$?
  if [[ "$ec" != "0" ]]; then
    echo "$s,NA,NA,NA,NA,NA,$OUTPUT_MODEL,$log,$ec" >>"$OUT_CSV"
    continue
  fi
  ml="$(grep '^\[metrics_line\]' "$log" | tail -n 1 || true)"
  if [[ -z "$ml" ]]; then
    echo "$s,NA,NA,NA,NA,NA,$OUTPUT_MODEL,$log,no_metrics" >>"$OUT_CSV"
    continue
  fi
  va="$(echo "$ml" | sed -n 's/.*val_acc=\([0-9.]*\).*/\1/p')"
  vf="$(echo "$ml" | sed -n 's/.*val_macro_f1=\([0-9.]*\).*/\1/p')"
  vb="$(echo "$ml" | sed -n 's/.*val_bal_acc=\([0-9.]*\).*/\1/p')"
  be="$(echo "$ml" | sed -n 's/.*best_epoch=\([0-9]*\).*/\1/p')"
  er="$(echo "$ml" | sed -n 's/.*epochs_run=\([0-9]*\).*/\1/p')"
  echo "$s,$va,$vf,$vb,$be,$er,$OUTPUT_MODEL,$log,0" >>"$OUT_CSV"
done

echo "[nb_sweep] wrote $OUT_CSV" >&2
python3 - "$OUT_CSV" <<'PY'
import csv, math, sys
p = sys.argv[1]
rows = list(csv.DictReader(open(p, newline="")))
from collections import defaultdict

by_seed = defaultdict(list)
for r in rows:
    by_seed[r.get("seed", "")].append(r)
out_rows = []
for sk in sorted(by_seed.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
    lst = by_seed[sk]
    valid = []
    for r in lst:
        if r.get("val_acc") in (None, "", "NA"):
            continue
        try:
            float(r["val_acc"])
        except (TypeError, ValueError):
            continue
        valid.append(r)
    if valid:
        out_rows.append(max(valid, key=lambda r: float(r["val_acc"])))
    else:
        out_rows.append(lst[-1])
rows = out_rows
with open(p, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "seed",
            "val_acc",
            "val_macro_f1",
            "val_bal_acc",
            "best_epoch",
            "epochs_run",
            "output_model",
            "log",
            "exit_code",
        ],
    )
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"[nb_sweep] deduped CSV → {len(rows)} seeds", flush=True)
PY
python3 - "$OUT_CSV" <<'PY'
import csv, math, sys
p = sys.argv[1]
rows = list(csv.DictReader(open(p, newline="")))
ok = [r for r in rows if r.get("val_acc") not in (None, "", "NA")]
if not ok:
    print("[nb_sweep] no valid val_acc"); sys.exit(0)
acc = [float(r["val_acc"]) for r in ok]
mx = max(acc)
print(f"[nb_sweep] n={len(acc)} val_acc max={mx:.4f} mean={sum(acc)/len(acc):.4f} ", end="")
if len(acc) > 1:
    m = sum(acc) / len(acc)
    var = sum((x - m) ** 2 for x in acc) / (len(acc) - 1)
    print(f"stdev={math.sqrt(var):.4f}")
else:
    print("stdev=NA")
best = max(ok, key=lambda r: float(r["val_acc"]))
print(f"[nb_sweep] best_seed={best['seed']} val_acc={best['val_acc']} path={best['output_model']}")
PY
