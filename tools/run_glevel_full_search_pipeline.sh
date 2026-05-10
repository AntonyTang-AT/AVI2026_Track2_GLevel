#!/usr/bin/env bash
# 顺序执行 Phase A → Phase B → 合并 CSV 摘要 → Phase D 集成 → 生成 SEARCH_REPORT.md
# 可通过环境变量覆盖各阶段 HUNT_DIR；默认均在 /data/emo/glevel_runs/
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

TAG="${GLEVEL_SEARCH_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASE="${GLEVEL_SEARCH_BASE:-/data/emo/glevel_runs/search_${TAG}}"
mkdir -p "${BASE}"

export GLEVEL_CUDA_PYTHON="${GLEVEL_CUDA_PYTHON:-/home/emo/txcao/anaconda3/envs/avi2026/bin/python}"
export PYTHON="${GLEVEL_CUDA_PYTHON}"

bash "${_ROOT}/tools/run_glevel_preflight.sh"
cp -f /data/emo/glevel_runs/preflight_latest.txt "${BASE}/preflight.txt"

export HUNT_DIR="${PHASE_A_HUNT_DIR:-${BASE}/phaseA}"
export HUNT_DIR
bash "${_ROOT}/tools/run_glevel_phaseA_baseline_sweep.sh"

export HUNT_DIR="${PHASE_B_HUNT_DIR:-${BASE}/phaseB}"
export HUNT_DIR
bash "${_ROOT}/tools/run_glevel_phaseB_grid.sh"

MERGED="${BASE}/combo_sweep_merged.csv"
{
  echo "combo_id,seed,val_acc,val_macro_f1,val_bal_acc,best_epoch,epochs_run,output_model,log,exit_code,phase"
  tail -n +2 "${BASE}/phaseA/combo_sweep_metrics.csv" | sed 's/$/,phaseA/'
  tail -n +2 "${BASE}/phaseB/combo_sweep_metrics.csv" | sed 's/$/,phaseB/'
} >"${MERGED}"

"${PYTHON}" "${_ROOT}/tools/summarize_glevel_combo_sweep.py" "${MERGED}" | tee "${BASE}/summarize_merged.txt"
"${PYTHON}" "${_ROOT}/tools/post_sweep_eval_top.py" "${MERGED}" 8

bash "${_ROOT}/tools/run_glevel_phaseD_ensemble_val.sh" "${MERGED}"

ARCHIVE_CKPT="/data/emo/glevel_runs/archives/nb_to58_sweep/round1/seed37/best.pth"
ARCH_NOTE="归档 seed37（历史约 0.5873，batch32 round1）若文件不存在则略过"
{
  echo "# G-Level 搜索流水线报告"
  echo "生成时间: $(date -Is)"
  echo "目录: ${BASE}"
  echo ""
  echo "## Preflight"
  echo '```'
  head -40 "${BASE}/preflight.txt"
  echo '```'
  echo ""
  echo "## 合并汇总 (summarize_merged.txt)"
  cat "${BASE}/summarize_merged.txt"
  echo ""
  echo "## Top 官方 val 复评"
  cat "${BASE}/post_eval_top_val.txt" 2>/dev/null || true
  echo ""
  echo "## Phase D 集成"
  cat "${BASE}/phaseD_ensemble_val.txt" 2>/dev/null || true
  echo ""
  echo "## 与归档 0.5873 对比"
  if [[ -f "${ARCHIVE_CKPT}" ]]; then
    echo "归档权重: ${ARCHIVE_CKPT}"
    "${PYTHON}" "${_ROOT}/tools/eval_glevel_checkpoint_on_csv.py" \
      --eval_csv /data/Super-Lu/dataset/val_data.csv \
      --rating_csv /data/Super-Lu/dataset/train_data.csv \
      --labels_in_split_csv \
      --g_level_int_encoding one \
      --train_audio_dir /data/Super-Lu/dataset/train_feature/audio \
      --train_video_dir /data/Super-Lu/dataset/train_feature/video \
      --train_text_dir "${_ROOT}/data/text_nb" \
      --eval_audio_dir /data/Super-Lu/dataset/val_feature/audio \
      --eval_video_dir /data/Super-Lu/dataset/val_feature/video \
      --eval_text_dir "${_ROOT}/data/text_nb_val" \
      --checkpoint "${ARCHIVE_CKPT}" \
      --text_dim 2560 \
      --mlp_dropout 0.25 \
      --modality_dropout_p 0.12 \
      --cross_modal_attn --cross_modal_layers 1 \
      --num_workers 2 || true
  else
    echo "${ARCH_NOTE}: 缺失 ${ARCHIVE_CKPT}"
  fi
} | tee "${BASE}/SEARCH_REPORT.md"

echo "[pipeline] done ${BASE}/SEARCH_REPORT.md" >&2
