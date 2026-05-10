#!/usr/bin/env bash
# GPU 上系统性搜索 g_level 最优模型：覆盖「两次改进」相关组合 + 多种子。
#
# 改进 A（全模态 shared_mlp + cross-modal）：可选 --fused_layer_norm；可选 StepLR（--lr_scheduler step）。
# 改进 B（audio_text_mlp）：无 cross-modal；可选 --fused_layer_norm；可选 StepLR。
# 早停 / Plateau 等与 scripts/glevel_train.sh 对齐（EARLY_STOP_MIN_EPOCHS 默认 20）。
#
# 用法（工程根）:
#   HUNT_DIR=./experiments/gpu_combo_sweep/run1
#   export HUNT_DIR
#   bash tools/run_glevel_gpu_combo_sweep.sh
# （勿写 export HUNT_DIR ./path — bash 不会给 HUNT_DIR 赋值）
#
# 多卡并行（默认占满可见 GPU，每进程一张卡）:
#   export MAX_PARALLEL_JOBS=8
#
# 指定 CUDA Python（服务器上 anaconda base 常为 CPU 版，需 avi2026 等）:
#   export GLEVEL_CUDA_PYTHON=/path/to/avi2026/bin/python
#
# 仅跑部分组合（逗号分隔，无空格）:
#   export COMBOS="S_plateau_ln,S_step_ln,AT_plateau_ln"
#
# 自定义种子（空格分隔）；随机 8 个不重复种子示例:
#   export SEEDS="$(python3 -c 'import random; random.seed(); print(*random.sample(range(1, 99999), 8))')"
#
# 冒烟（只打印、不训练）:
#   DRY_RUN=1 bash tools/run_glevel_gpu_combo_sweep.sh
#
# 限制总任务数（调试用）:
#   MAX_RUNS=2 bash tools/run_glevel_gpu_combo_sweep.sh
#
# 合并 train+val 池随机划分（推荐 POOL_TRAIN_N≈418，pool≈514 时验证≈96；更大验证可调小 train_n）:
#   export POOL_RANDOM_SPLITS=5 POOL_TRAIN_N=418 POOL_SPLIT_BASE_SEED=5000
#   循环顺序 combo×seed×slot；默认启用更宽 COMBOS/SEEDS；训练集含原 val id 时自动 --train_feat_fallback
#   结束后若 SAVE_TOP_K_MODELS>0（pool 模式默认 10）复制最优 checkpoint → \${HUNT_DIR}/top_models/
#
# 稳定性搜索（数据划分为「相对模型」的最内层）：每一组 (combo, seed) 下重复 PARTITION_ROUNDS 次，
# 每次用不同随机种子重新划分 train_holdout / val_keep / val→test，再训练一整轮。
#   export PARTITION_ROUNDS=5 PARTITION_BASE_SEED=1000 TRAIN_HOLDOUT_N=80 VAL_TO_TEST_N=15
#   （循环顺序：外层 combo×seed，内层 part=1..PARTITION_ROUNDS；指标 CSV 的 split_round 列为 part）
#
# 完整 6×6 网格但外层曾 export SEEDS=37 导致只跑 6 个任务时:
#   unset SEEDS; bash tools/run_glevel_gpu_combo_sweep.sh
#   或 FORCE_DEFAULT_SEEDS=1 bash ...
#
# 多卡并行显存顶满（CUDA OOM）时:
#   OOM_SAFE_BATCH=1 BATCH_SIZE=24 LAUNCH_STAGGER_SEC=2 bash ...
#
# shellcheck source=/dev/null
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

_pick_cuda_python() {
  local c
  for c in "${GLEVEL_CUDA_PYTHON:-}" \
    "/home/emo/anaconda3/envs/magnus/bin/python" \
    "/home/emo/txcao/anaconda3/envs/avi2026/bin/python" \
    "/home/emo/antonytang/miniconda3/envs/avi2026/bin/python" \
    "${HOME}/anaconda3/envs/avi2026/bin/python" \
    "${HOME}/miniconda3/envs/avi2026/bin/python"; do
    [[ -n "${c}" ]] && [[ -x "${c}" ]] || continue
    if "${c}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
      printf '%s\n' "${c}"
      return 0
    fi
  done
  return 1
}

# 解释器：GLEVEL_CUDA_PYTHON 优先；否则若未 export PYTHON 则自动发现 CUDA（避免默认 python3 实为 CPU wheel）
if [[ -n "${GLEVEL_CUDA_PYTHON:-}" ]] && [[ -x "${GLEVEL_CUDA_PYTHON}" ]]; then
  PYTHON="${GLEVEL_CUDA_PYTHON}"
  echo "[gpu_combo_sweep] 使用 GLEVEL_CUDA_PYTHON=${PYTHON}" >&2
elif [[ -z "${PYTHON:-}" ]]; then
  if _cuda_py="$(_pick_cuda_python)"; then
    PYTHON="${_cuda_py}"
    echo "[gpu_combo_sweep] 自动选用 CUDA Python: ${PYTHON}" >&2
  else
    PYTHON="python3"
    echo "[gpu_combo_sweep] 未找到 CUDA Python，回退 python3（可设 GLEVEL_CUDA_PYTHON）" >&2
  fi
fi
# 环境中已 export PYTHON 但实为 CPU 版时，强制回退到可探测到的 CUDA 解释器
if ! "${PYTHON}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  if _cuda_py="$(_pick_cuda_python)"; then
    echo "[gpu_combo_sweep] 当前 PYTHON=${PYTHON} 无 CUDA，已切换为: ${_cuda_py}" >&2
    PYTHON="${_cuda_py}"
  else
    echo "[gpu_combo_sweep] 警告: ${PYTHON} 无 CUDA 且未找到其它 CUDA Python（请设置 GLEVEL_CUDA_PYTHON）" >&2
  fi
fi
export PYTHON

HUNT_DIR="${HUNT_DIR:-${_ROOT}/experiments/gpu_combo_sweep/default}"
case "$HUNT_DIR" in
  /*) ;;
  *) HUNT_DIR="${_ROOT}/${HUNT_DIR}" ;;
esac
mkdir -p "$HUNT_DIR"

# 与 nb_to58 封板路线一致的全模态超参（Nanbeige 2560）
export NANBEIGE_TEXT="${NANBEIGE_TEXT:-1}"
export TEXT_DIM="${TEXT_DIM:-2560}"
export TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${_ROOT}/data/text_nb}"
export TEXT_VAL_DIR="${TEXT_VAL_DIR:-${_ROOT}/data/text_nb_val}"
export TEXT_TEST_DIR="${TEXT_TEST_DIR:-${_ROOT}/data/test_nb}"

. "${_ROOT}/tools/glevel_paths.inc.sh"

SUP="${SUPERLU_DATASET:-/data/Super-Lu/dataset}"
TRAIN_CSV="${TRAIN_CSV:-${SUP}/train_data.csv}"
VAL_CSV="${VAL_CSV:-${SUP}/val_data.csv}"
RATING_CSV="${RATING_CSV:-${SUP}/train_data.csv}"
TEST_CSV="${TEST_CSV:-${SUP}/test_data_basic_information.csv}"

PARTITION_ROUNDS="${PARTITION_ROUNDS:-0}"
PARTITION_BASE_SEED="${PARTITION_BASE_SEED:-1000}"
TRAIN_HOLDOUT_N="${TRAIN_HOLDOUT_N:-80}"
VAL_TO_TEST_N="${VAL_TO_TEST_N:-15}"
SOURCE_TRAIN_CSV="${SOURCE_TRAIN_CSV:-${TRAIN_CSV}}"
OFFICIAL_VAL_CSV="${OFFICIAL_VAL_CSV:-${VAL_CSV}}"
OFFICIAL_TEST_BASIC_CSV="${OFFICIAL_TEST_BASIC_CSV:-${TEST_CSV}}"

# 合并官方 train+val 为池，每次分层随机抽 POOL_TRAIN_N 条训练、剩余验证（内层 slot 与 PARTITION_ROUNDS 互斥）
POOL_RANDOM_SPLITS="${POOL_RANDOM_SPLITS:-0}"
POOL_TRAIN_N="${POOL_TRAIN_N:-418}"
POOL_SPLIT_BASE_SEED="${POOL_SPLIT_BASE_SEED:-5000}"
POOL_MERGED_TRAIN_CSV="${POOL_MERGED_TRAIN_CSV:-${SOURCE_TRAIN_CSV}}"
POOL_MERGED_VAL_CSV="${POOL_MERGED_VAL_CSV:-${OFFICIAL_VAL_CSV}}"

if [[ "${PARTITION_ROUNDS}" != "0" ]] && [[ "${POOL_RANDOM_SPLITS}" != "0" ]]; then
  echo "[gpu_combo_sweep] 错误: PARTITION_ROUNDS 与 POOL_RANDOM_SPLITS 不能同时非 0" >&2
  exit 2
fi

if [[ "${POOL_RANDOM_SPLITS}" != "0" ]]; then
  SAVE_TOP_K_MODELS="${SAVE_TOP_K_MODELS:-10}"
else
  SAVE_TOP_K_MODELS="${SAVE_TOP_K_MODELS:-0}"
fi

COMBOS_DEFAULT="S_ref_plateau,S_plateau_ln,S_step_ln,AT_plateau,AT_plateau_ln,AT_step_ln"
SEEDS_DEFAULT="37 10 28 5 42 73"
if [[ "${POOL_RANDOM_SPLITS}" != "0" ]]; then
  # 模块化「宽搜索」默认：含 sel_acc / cosine / step 等；若已 export COMBOS/SEEDS 则不会被覆盖
  COMBOS_DEFAULT="S_ref_plateau,S_ref_sel_acc,S_ref_cosine,S_ref_step,S_plateau_ln,S_plateau_ln_sel_acc,S_step_ln,AT_plateau,AT_plateau_ln,AT_step_ln"
  SEEDS_DEFAULT="11 17 23 29 37 42 53 61 73 79 88 97 101 113 127 137 149 163 181 199 211 233 256 271 307 333 353 404 419 433 503 521 577 628 719 777 888"
fi
COMBOS="${COMBOS:-$COMBOS_DEFAULT}"
IFS=',' read -r -a COMBO_ARR <<< "${COMBOS}"

SEEDS="${SEEDS:-$SEEDS_DEFAULT}"
# 单词数过小时常为外层误 export 了窄 SEEDS；需要完整网格可: unset SEEDS 或 FORCE_DEFAULT_SEEDS=1
if [[ "${FORCE_DEFAULT_SEEDS:-0}" == "1" ]]; then
  SEEDS="${SEEDS_DEFAULT}"
fi
_NUM_SEEDS="$(wc -w <<<"${SEEDS}")"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
# 多卡并行、全模态 Transformer 显存峰值高；可 export OOM_SAFE_BATCH=1 改用 batch=24
if [[ "${OOM_SAFE_BATCH:-0}" == "1" ]]; then
  BATCH_SIZE=24
fi
LAUNCH_STAGGER_SEC="${LAUNCH_STAGGER_SEC:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-40}"
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-20}"
LR_SCHEDULER_PATIENCE="${LR_SCHEDULER_PATIENCE:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
STEP_SIZE="${STEP_SIZE:-30}"
STEP_GAMMA="${STEP_GAMMA:-0.5}"
MAX_RUNS="${MAX_RUNS:-0}"
DRY_RUN="${DRY_RUN:-0}"
SAMPLER_MEDIUM_BOOST="${SAMPLER_MEDIUM_BOOST:-1.5}"
# 并行进程数；设为 1 则顺序执行。默认可通过 nvidia-smi 推断（上限 8，避免误开过大）
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-}"
GPU_IDS="${GPU_IDS:-}"

OUT_CSV="${HUNT_DIR}/combo_sweep_metrics.csv"
_METRICS_LOCK="${HUNT_DIR}/.combo_sweep_metrics.lock"
touch "${_METRICS_LOCK}"

if [[ ! -s "$OUT_CSV" ]]; then
  echo "combo_id,seed,val_acc,val_macro_f1,val_bal_acc,best_epoch,epochs_run,output_model,log,exit_code,split_round" >"$OUT_CSV"
fi

if ! "${PYTHON}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "[gpu_combo_sweep] 警告: ${PYTHON} 未报告 CUDA 可用，训练将落在 CPU 上（极慢）。请设置 GLEVEL_CUDA_PYTHON 或 export PYTHON=.../avi2026/bin/python" >&2
fi

_append_row() {
  local combo_id="$1" seed="$2" va="$3" vf="$4" vb="$5" be="$6" er="$7" outp="$8" logp="$9" ec="${10}"
  local sr="${SPLIT_ROUND:-}"
  local line="${combo_id},${seed},${va},${vf},${vb},${be},${er},${outp},${logp},${ec},${sr}"
  (
    flock -x 200 || exit 1
    printf '%s\n' "${line}" >>"${OUT_CSV}"
  ) 200>>"${_METRICS_LOCK}"
}

_train_one() {
  local combo_id="$1"
  local seed="$2"
  local part="${3:-}"
  local SPLIT_ROUND=""
  [[ -n "${part}" ]] && SPLIT_ROUND="${part}"
  export SPLIT_ROUND

  local subdir="${HUNT_DIR}/${combo_id}/seed${seed}"
  if [[ -n "${part}" ]]; then
    if [[ "${POOL_RANDOM_SPLITS:-0}" != "0" ]]; then
      subdir="${subdir}/pool${part}"
    else
      subdir="${subdir}/part${part}"
    fi
  fi
  mkdir -p "${subdir}"

  local log="${subdir}/train.log"
  local outp="${subdir}/best.pth"

  local tr_csv="${TRAIN_CSV}"
  local va_csv="${VAL_CSV}"
  local te_csv="${TEST_CSV}"
  local -a test_val_fb_flag=()
  local -a train_fb_flag=()

  if [[ -n "${part}" ]] && [[ "${POOL_RANDOM_SPLITS:-0}" != "0" ]]; then
    local pdata="${subdir}/pool_split_data"
    mkdir -p "${pdata}"
    echo "[gpu_combo_sweep] merged-pool split slot=${part} split_seed=$((POOL_SPLIT_BASE_SEED + part - 1)) POOL_TRAIN_N=${POOL_TRAIN_N} → ${pdata}" >&2
    if [[ "${DRY_RUN}" != "1" ]]; then
      "${PYTHON}" "${_ROOT}/tools/make_merged_pool_train_val_split.py" \
        --official_train_csv "${POOL_MERGED_TRAIN_CSV}" \
        --official_val_csv "${POOL_MERGED_VAL_CSV}" \
        --train_n "${POOL_TRAIN_N}" \
        --split_seed "$((POOL_SPLIT_BASE_SEED + part - 1))" \
        --out_dir "${pdata}"
    fi
    tr_csv="${pdata}/train_fold.csv"
    va_csv="${pdata}/val_fold.csv"
    te_csv="${TEST_CSV}"
    train_fb_flag=(--train_feat_fallback)
  elif [[ -n "${part}" ]]; then
    local pdata="${subdir}/partition_data"
    mkdir -p "${pdata}"
    echo "[gpu_combo_sweep] stability partition part=${part} seed=$((PARTITION_BASE_SEED + part - 1)) → ${pdata}" >&2
    if [[ "${DRY_RUN}" != "1" ]]; then
      "${PYTHON}" "${_ROOT}/tools/make_stability_data_partition.py" \
        --train_pool_csv "${SOURCE_TRAIN_CSV}" \
        --official_val_csv "${OFFICIAL_VAL_CSV}" \
        --official_test_basic_csv "${OFFICIAL_TEST_BASIC_CSV}" \
        --partition_seed "$((PARTITION_BASE_SEED + part - 1))" \
        --train_holdout_n "${TRAIN_HOLDOUT_N}" \
        --val_to_test_n "${VAL_TO_TEST_N}" \
        --out_dir "${pdata}"
    fi
    tr_csv="${pdata}/train.csv"
    va_csv="${pdata}/val_merged.csv"
    te_csv="${pdata}/test_merged.csv"
    test_val_fb_flag=(--test_fallback_val_features)
  fi

  local select_best_metric="balanced_acc"
  local -a extra=()
  case "${combo_id}" in
    S_ref_plateau)
      extra=(
        --glevel_arch shared_mlp
        --cross_modal_attn --cross_modal_layers 1
      )
      ;;
    S_ref_sel_acc)
      select_best_metric="val_acc"
      extra=(
        --glevel_arch shared_mlp
        --cross_modal_attn --cross_modal_layers 1
      )
      ;;
    S_ref_step)
      extra=(
        --glevel_arch shared_mlp
        --cross_modal_attn --cross_modal_layers 1
        --lr_scheduler step
        --lr_step_size "${STEP_SIZE}"
        --lr_gamma "${STEP_GAMMA}"
      )
      ;;
    S_ref_cosine)
      extra=(
        --glevel_arch shared_mlp
        --cross_modal_attn --cross_modal_layers 1
        --lr_scheduler cosine
      )
      ;;
    S_plateau_ln)
      extra=(
        --glevel_arch shared_mlp
        --cross_modal_attn --cross_modal_layers 1
        --fused_layer_norm
      )
      ;;
    S_plateau_ln_sel_acc)
      select_best_metric="val_acc"
      extra=(
        --glevel_arch shared_mlp
        --cross_modal_attn --cross_modal_layers 1
        --fused_layer_norm
      )
      ;;
    S_step_ln)
      extra=(
        --glevel_arch shared_mlp
        --cross_modal_attn --cross_modal_layers 1
        --fused_layer_norm
        --lr_scheduler step
        --lr_step_size "${STEP_SIZE}"
        --lr_gamma "${STEP_GAMMA}"
      )
      ;;
    AT_plateau)
      extra=(--glevel_arch audio_text_mlp --at_mlp_hidden 512)
      ;;
    AT_plateau_ln)
      extra=(--glevel_arch audio_text_mlp --at_mlp_hidden 512 --fused_layer_norm)
      ;;
    AT_step_ln)
      extra=(
        --glevel_arch audio_text_mlp
        --at_mlp_hidden 512
        --fused_layer_norm
        --lr_scheduler step
        --lr_step_size "${STEP_SIZE}"
        --lr_gamma "${STEP_GAMMA}"
      )
      ;;
    *)
      echo "[gpu_combo_sweep] 未知 combo_id=${combo_id}" >&2
      exit 2
      ;;
  esac

  local -a common=(
    --g_level_int_encoding one
    --mlp_dropout 0.25
    --weight_decay 0.001
    --label_smoothing 0.05
    --select_best "${select_best_metric}"
    --modality_dropout_p 0.12
    --scheduler_min_lr 1e-6
    --sampler_medium_boost "${SAMPLER_MEDIUM_BOOST}"
  )

  echo "[gpu_combo_sweep] === ${combo_id} seed=${seed} slot=${part:-—} cuda_visible=${CUDA_VISIBLE_DEVICES:-<unset>} → ${log}" >&2
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${PYTHON}" "${_ROOT}/python/train_task2_glevel.py" ... --seed "${seed}" "${common[@]}" "${extra[@]}" "${train_fb_flag[@]}" "${test_val_fb_flag[@]}" >&2
    return 0
  fi

  local ec=0
  "${PYTHON}" "${_ROOT}/python/train_task2_glevel.py" \
    --train_csv "${tr_csv}" \
    --val_csv "${va_csv}" \
    --test_csv "${te_csv}" \
    --rating_csv "${RATING_CSV}" \
    --labels_in_split_csv \
    --label_col g_level \
    --question q1 q2 q3 q4 q5 q6 \
    --video_dim 512 --video_dir "${FEAT_TRAIN}/video" \
    --audio_dim 512 --audio_dir "${FEAT_TRAIN}/audio" \
    --text_dim "${TEXT_DIM}" --text_dir "${TEXT_TRAIN_DIR}" \
    --val_video_dir "${FEAT_VAL}/video" \
    --val_audio_dir "${FEAT_VAL}/audio" \
    --val_text_dir "${TEXT_VAL_DIR}" \
    --test_video_dir "${FEAT_TEST}/video" \
    --test_audio_dir "${FEAT_TEST}/audio" \
    --test_text_dir "${TEXT_TEST_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_epochs "${NUM_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --lr_scheduler_patience "${LR_SCHEDULER_PATIENCE}" \
    --early_stop_patience "${EARLY_STOP_PATIENCE}" \
    --early_stop_min_epochs "${EARLY_STOP_MIN_EPOCHS}" \
    --output_model "${outp}" \
    --loss_plot_path "${subdir}/loss.png" \
    --test_output_csv "${subdir}/submission.csv" \
    --seed "${seed}" \
    "${common[@]}" \
    "${extra[@]}" \
    "${train_fb_flag[@]}" \
    "${test_val_fb_flag[@]}" \
    2>&1 | tee "${log}" || ec=$?

  local ml va vf vb be er
  ml="$(grep '^\[metrics_line\]' "${log}" | tail -n 1 || true)"
  if [[ -z "${ml}" ]]; then
    _append_row "${combo_id}" "${seed}" NA NA NA NA NA "${outp}" "${log}" "${ec}"
    return 0
  fi
  va="$(echo "${ml}" | sed -n 's/.*val_acc=\([0-9.]*\).*/\1/p')"
  vf="$(echo "${ml}" | sed -n 's/.*val_macro_f1=\([0-9.]*\).*/\1/p')"
  vb="$(echo "${ml}" | sed -n 's/.*val_bal_acc=\([0-9.]*\).*/\1/p')"
  be="$(echo "${ml}" | sed -n 's/.*best_epoch=\([0-9]*\).*/\1/p')"
  er="$(echo "${ml}" | sed -n 's/.*epochs_run=\([0-9]*\).*/\1/p')"
  _append_row "${combo_id}" "${seed}" "${va}" "${vf}" "${vb}" "${be}" "${er}" "${outp}" "${log}" "${ec}"
}

# 构建任务列表：默认 combo_id:seed；PARTITION_ROUNDS>0 时为 combo_id:seed:part
JOBS=()
_run_count=0
for combo_id in "${COMBO_ARR[@]}"; do
  combo_id="${combo_id// /}"
  [[ -z "${combo_id}" ]] && continue
  for s in ${SEEDS}; do
    if [[ "${POOL_RANDOM_SPLITS:-0}" != "0" ]]; then
      for ((ps = 1; ps <= POOL_RANDOM_SPLITS; ps++)); do
        if [[ "${MAX_RUNS}" != "0" ]] && [[ "${_run_count}" -ge "${MAX_RUNS}" ]]; then
          break 3
        fi
        JOBS+=("${combo_id}:${s}:${ps}")
        _run_count=$((_run_count + 1))
      done
    elif [[ "${PARTITION_ROUNDS}" != "0" ]]; then
      for ((p = 1; p <= PARTITION_ROUNDS; p++)); do
        if [[ "${MAX_RUNS}" != "0" ]] && [[ "${_run_count}" -ge "${MAX_RUNS}" ]]; then
          break 3
        fi
        JOBS+=("${combo_id}:${s}:${p}")
        _run_count=$((_run_count + 1))
      done
    else
      if [[ "${MAX_RUNS}" != "0" ]] && [[ "${_run_count}" -ge "${MAX_RUNS}" ]]; then
        break 2
      fi
      JOBS+=("${combo_id}:${s}")
      _run_count=$((_run_count + 1))
    fi
  done
done

_resolve_parallel() {
  if [[ -n "${MAX_PARALLEL_JOBS}" ]]; then
    printf '%s' "${MAX_PARALLEL_JOBS}"
    return
  fi
  local n
  n="$("${PYTHON}" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)"
  if [[ "${n}" -lt 1 ]]; then
    n=1
  fi
  if [[ "${n}" -gt 8 ]]; then
    n=8
  fi
  printf '%s' "${n}"
}

PAR="$(_resolve_parallel)"
if [[ "${DRY_RUN}" == "1" ]]; then
  PAR=1
fi

# 解析 GPU_IDS → 数组 _GPU_ARR；为空则用 0..PAR-1
_GPU_ARR=()
if [[ -n "${GPU_IDS}" ]]; then
  IFS=',' read -r -a _TMP <<< "${GPU_IDS// /}"
  for x in "${_TMP[@]}"; do
    [[ -n "${x}" ]] && _GPU_ARR+=("${x}")
  done
else
  for ((i = 0; i < PAR; i++)); do
    _GPU_ARR+=("${i}")
  done
fi
if [[ "${#_GPU_ARR[@]}" -lt 1 ]]; then
  _GPU_ARR=(0)
fi
_NG="${#_GPU_ARR[@]}"
if [[ "${PAR}" -gt "${_NG}" ]]; then
  PAR="${_NG}"
  echo "[gpu_combo_sweep] 并行度已限制为 GPU 列表长度 ${_NG}" >&2
fi

if [[ "${PAR}" -ge 6 ]] && [[ "${OOM_SAFE_BATCH:-0}" != "1" ]] && [[ "${BATCH_SIZE}" == "32" ]]; then
  echo "[gpu_combo_sweep] 提示: 并行≥6 且 batch=32 时，全模态模型偶发 OOM（尤其 GPU 上尚有其它进程）。可设 OOM_SAFE_BATCH=1 或 BATCH_SIZE=24。" >&2
fi

echo "[gpu_combo_sweep] jobs=${#JOBS[@]} POOL_RANDOM_SPLITS=${POOL_RANDOM_SPLITS} PARTITION_ROUNDS=${PARTITION_ROUNDS} SEEDS(wordcount)=${_NUM_SEEDS} parallel=${PAR} gpu_ids=${_GPU_ARR[*]} BATCH_SIZE=${BATCH_SIZE} NUM_WORKERS=${NUM_WORKERS} LAUNCH_STAGGER_SEC=${LAUNCH_STAGGER_SEC}" >&2

if [[ "${#JOBS[@]}" -eq 0 ]]; then
  echo "[gpu_combo_sweep] 无任务，退出。" >&2
  exit 0
fi

if [[ "${PAR}" -le 1 ]] || [[ "${DRY_RUN}" == "1" ]]; then
  for job in "${JOBS[@]}"; do
    IFS=':' read -r _c _s _p <<<"${job}"
    _train_one "${_c}" "${_s}" "${_p}"
  done
else
  _idx=0
  for job in "${JOBS[@]}"; do
    while true; do
      _running="$(jobs -r -p | wc -l)"
      if [[ "${_running}" -lt "${PAR}" ]]; then
        break
      fi
      # bash 5+ wait -n；旧版回退为 wait 任一
      if ! wait -n 2>/dev/null; then
        wait || true
      fi
    done
    IFS=':' read -r _c _s _p <<<"${job}"
    _gid="${_GPU_ARR[$((_idx % ${#_GPU_ARR[@]}))]}"
    _idx=$((_idx + 1))
    if [[ "${LAUNCH_STAGGER_SEC}" != "0" ]]; then
      sleep "${LAUNCH_STAGGER_SEC}"
    fi
    (
      export CUDA_VISIBLE_DEVICES="${_gid}"
      _train_one "${_c}" "${_s}" "${_p}"
    ) &
  done
  wait || true
fi

echo "[gpu_combo_sweep] 完成。指标: ${OUT_CSV}" >&2
echo "[gpu_combo_sweep] 汇总: ${PYTHON} ${_ROOT}/tools/summarize_glevel_combo_sweep.py ${OUT_CSV}" >&2
"${PYTHON}" "${_ROOT}/tools/summarize_glevel_combo_sweep.py" "${OUT_CSV}"
if [[ "${SAVE_TOP_K_MODELS:-0}" != "0" ]] && [[ -s "${OUT_CSV}" ]]; then
  _TOPDIR="${HUNT_DIR}/top_models"
  echo "[gpu_combo_sweep] Top-${SAVE_TOP_K_MODELS} checkpoint → ${_TOPDIR}" >&2
  "${PYTHON}" "${_ROOT}/tools/copy_top_k_models.py" "${OUT_CSV}" --top_k "${SAVE_TOP_K_MODELS}" --out_dir "${_TOPDIR}"
fi
