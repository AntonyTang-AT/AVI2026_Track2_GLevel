#!/usr/bin/env bash
# 【旧版 CV】仅从官方 train 池内划分 train_fold/val_fold（验证特征强制与 train 对齐）。
#
# **新版稳定性搜索**（大验证 = train_holdout ∪ 官方 val；部分官方 val 并入扩展 test）请用：
#   PARTITION_ROUNDS=5 TRAIN_HOLDOUT_N=80 VAL_TO_TEST_N=15 bash tools/run_glevel_gpu_combo_sweep.sh
#
# 多轮随机 train/val 划分 + 每轮内层完整 GPU combo×seed 搜索（与 run_glevel_gpu_combo_sweep.sh 相同内层逻辑）。
#
# 动机：官方 val 仅 ~63 条，指标方差大；从 450 条 train 中 **分层随机** 划出较大/多轮验证，
# 用 **不同划分反复训练** 再汇总，减少「选对验证运气」成分。
#
# 重要：验证行 id 均来自 **训练池**，特征在 train_feature + text_nb，**不能**用 val_feature / text_nb_val。
# 本脚本在 source glevel_paths 后强制：FEAT_VAL=FEAT_TRAIN、TEXT_VAL_DIR=TEXT_TRAIN_DIR。
#
# 用法（工程根）:
#   export BASE_HUNT_DIR=./experiments/gpu_combo_sweep/cv_run1
#   SPLIT_BASE_SEED=1000 → 第 1 轮划分 seed=1000，第 2 轮=1001，…
#
# 或用占比代替条数:
#   VAL_HOLDOUT_RATIO=0.18 SPLIT_ROUNDS=8 bash tools/run_glevel_gpu_combo_sweep_cv.sh
#
# 向内层透传（与单轮 sweep 相同）: COMBOS SEEDS MAX_PARALLEL_JOBS GPU_IDS OOM_SAFE_BATCH ...
#
# shellcheck source=/dev/null
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

SUP="${SUPERLU_DATASET:-/data/Super-Lu/dataset}"
SOURCE_TRAIN_CSV="${SOURCE_TRAIN_CSV:-${SUP}/train_data.csv}"
RATING_CSV_FIXED="${RATING_CSV_FIXED:-${SUP}/train_data.csv}"

SPLIT_ROUNDS="${SPLIT_ROUNDS:-5}"
SPLIT_BASE_SEED="${SPLIT_BASE_SEED:-1000}"
VAL_HOLDOUT_N="${VAL_HOLDOUT_N:-}"
VAL_HOLDOUT_RATIO="${VAL_HOLDOUT_RATIO:-}"

_STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_HUNT_DIR="${BASE_HUNT_DIR:-${_ROOT}/experiments/gpu_combo_sweep/cv_${_STAMP}}"
case "$BASE_HUNT_DIR" in
  /*) ;;
  *) BASE_HUNT_DIR="${_ROOT}/${BASE_HUNT_DIR}" ;;
esac

if [[ -n "${VAL_HOLDOUT_N}" && -n "${VAL_HOLDOUT_RATIO}" ]]; then
  echo "[gpu_combo_sweep_cv] 请只设置 VAL_HOLDOUT_N 或 VAL_HOLDOUT_RATIO 之一" >&2
  exit 2
fi
if [[ -z "${VAL_HOLDOUT_N}" && -z "${VAL_HOLDOUT_RATIO}" ]]; then
  VAL_HOLDOUT_N=80
fi

# 划分脚本优先用与内层一致的 Python（可无 torch）
_PY_SPLIT="${PYTHON:-python3}"
if [[ -x "${_ROOT}/.venv_glevel_cpu/bin/python" ]]; then
  _PY_SPLIT="${_ROOT}/.venv_glevel_cpu/bin/python"
fi

export NANBEIGE_TEXT="${NANBEIGE_TEXT:-1}"
export TEXT_DIM="${TEXT_DIM:-2560}"
export TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${_ROOT}/data/text_nb}"
export TEXT_VAL_DIR="${TEXT_VAL_DIR:-${_ROOT}/data/text_nb_val}"
export TEXT_TEST_DIR="${TEXT_TEST_DIR:-${_ROOT}/data/test_nb}"

. "${_ROOT}/tools/glevel_paths.inc.sh"

export FEAT_VAL="${FEAT_TRAIN}"
export TEXT_VAL_DIR="${TEXT_TRAIN_DIR}"
export TRAIN_CSV=""
export VAL_CSV=""
export RATING_CSV="${RATING_CSV_FIXED}"
export SPLIT_ROUND=""

mkdir -p "${BASE_HUNT_DIR}"
_META="${BASE_HUNT_DIR}/cv_meta.txt"
{
  echo "start $(date -Iseconds)"
  echo "SOURCE_TRAIN_CSV=${SOURCE_TRAIN_CSV}"
  echo "RATING_CSV_FIXED=${RATING_CSV_FIXED}"
  echo "SPLIT_ROUNDS=${SPLIT_ROUNDS} SPLIT_BASE_SEED=${SPLIT_BASE_SEED}"
  echo "VAL_HOLDOUT_N=${VAL_HOLDOUT_N:-} VAL_HOLDOUT_RATIO=${VAL_HOLDOUT_RATIO:-}"
  echo "FEAT_TRAIN=${FEAT_TRAIN} FEAT_VAL(override)=${FEAT_VAL}"
  echo "TEXT_TRAIN_DIR=${TEXT_TRAIN_DIR} TEXT_VAL_DIR(override)=${TEXT_VAL_DIR}"
  echo "inner: COMBOS=${COMBOS:-<default>} SEEDS=${SEEDS:-<default>}"
} | tee "${_META}"

for ((round = 1; round <= SPLIT_ROUNDS; round++)); do
  _split_seed=$((SPLIT_BASE_SEED + round - 1))
  _rd="${BASE_HUNT_DIR}/round_${round}"
  mkdir -p "${_rd}"
  echo "[gpu_combo_sweep_cv] ===== split round ${round}/${SPLIT_ROUNDS} seed=${_split_seed} → ${_rd} =====" | tee -a "${_META}"

  _split_args=(
    "${_PY_SPLIT}" "${_SCRIPT_DIR}/split_train_val.py"
    --in_csv "${SOURCE_TRAIN_CSV}"
    --out_train "${_rd}/train_fold.csv"
    --out_val "${_rd}/val_fold.csv"
    --seed "${_split_seed}"
    --label_col g_level
  )
  if [[ -n "${VAL_HOLDOUT_RATIO}" ]]; then
    _split_args+=(--val_ratio "${VAL_HOLDOUT_RATIO}")
  else
    _split_args+=(--val_n "${VAL_HOLDOUT_N}")
  fi
  "${_split_args[@]}" 2>&1 | tee -a "${_META}"

  export TRAIN_CSV="${_rd}/train_fold.csv"
  export VAL_CSV="${_rd}/val_fold.csv"
  export HUNT_DIR="${_rd}/sweep"
  export SPLIT_ROUND="${round}"
  mkdir -p "${HUNT_DIR}"

  bash "${_SCRIPT_DIR}/run_glevel_gpu_combo_sweep.sh" 2>&1 | tee -a "${_META}"
done

echo "[gpu_combo_sweep_cv] 全部完成。BASE_HUNT_DIR=${BASE_HUNT_DIR}" | tee -a "${_META}"
echo "[gpu_combo_sweep_cv] 合并指标: ${_PY_SPLIT} ${_SCRIPT_DIR}/summarize_cv_combo_sweep.py ${BASE_HUNT_DIR}" | tee -a "${_META}"
"${_PY_SPLIT}" "${_SCRIPT_DIR}/summarize_cv_combo_sweep.py" "${BASE_HUNT_DIR}"
