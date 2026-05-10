#!/usr/bin/env bash
# AVI2026 赛道二 g_level 分类
# 服务器数据布局（与 /data 挂载一致；本地下载镜像 server/data 同结构）：
#   /data/Super-Lu/dataset/train_data.csv val_data.csv
#   /data/Super-Lu/dataset/train_feature|val_feature/{audio,video,text}/*.npy
#   /data/AVI2026/test_feature/{audio,video,text}/*.npy  ← 测试 id，勿用 train_feature
# CRLF 换行会导致 Linux 下 set/cd 报错；修复: sed -i 's/\r$//' vote_train_glevel.sh
# 若 python 在 import torch 阶段报 nccl*: undefined symbol：为 PyTorch 与系统 NCCL/CUDA 栈不匹配，
# 请在新环境中按 pytorch.org 重装 CUDA 版，或改用 CPU 版 wheel（见 train_task2_glevel 报错全文）。
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT:-$_SCRIPT_DIR}"

# DataLoader：默认 4 worker；严格复现或小 val 调试用： export NUM_WORKERS=0
# 解释器：默认 python。若 base 的 CUDA torch+NCCL 损坏，可根本规避：先
#   python3 tools/bootstrap_isolated_cpu_env.py && source .venv_glevel_cpu/bin/activate
#   export PYTHON="$PWD/.venv_glevel_cpu/bin/python"
# 或新建 conda 环境后： export PYTHON=/path/to/env/bin/python
PYTHON="${PYTHON:-python}"

# 数据路径：与 vote_test_glevel.sh、tools/run_phase1_svm.sh 共用 tools/glevel_paths.inc.sh
# Nanbeige：export NANBEIGE_TEXT=1；试跑子目录 export NANBEIGE_TEXT_SUBDIR=text_nb_smoke
# 测试集 Nanbeige 文本：若工程内存在 data/test_nb，默认作 TEXT_TEST_DIR；否则见 FEAT_TEST/text_nb。可手动 export TEXT_TEST_DIR。
# shellcheck source=tools/glevel_paths.inc.sh
. "${_SCRIPT_DIR}/tools/glevel_paths.inc.sh"

_NB_SUB="${NANBEIGE_TEXT_SUBDIR:-text_nb}"
if [ "${NANBEIGE_TEXT:-0}" = "1" ]; then
  case "${_NB_SUB}" in
    *smoke*)
      echo "[vote_train_glevel] NANBEIGE smoke：TEXT_VAL_DIR/TEXT_TEST_DIR 与 TRAIN 文本目录相同（${_NB_SUB}）。若 val 仍全被剔除，请对验证集转写单独提取到含 val id 的目录并 export TEXT_VAL_DIR=..." >&2
      ;;
  esac
fi

# 若只有 /data/AVI2026/train_feature、无独立 val_feature，可执行：
#   export FEAT_TRAIN=/data/AVI2026/train_feature FEAT_VAL=/data/AVI2026/train_feature FEAT_TEST=/data/AVI2026/train_feature
#
# 产出路径（可选覆盖）：
#   export OUTPUT_MODEL=best_model_glevel.pth
#   TTA 测试：在 GLEVEL_OPT 中加 --tta_times 8 --tta_noise_std 0.01（仅影响 only_test/predict_test）
#   SWA：--swa_start_epoch 30 --swa_lr 1e-4 --output_swa_model ./best_swa.pth
#   export LOSS_PLOT_PATH=./loss_img/loss_glevel.png
#   export TEST_OUTPUT_CSV=submission_glevel.csv

# 赛方 train/val CSV 中 g_level 为 1/2/3 时与训练一致；亦会被 train_task2_glevel 自动检测（含 3 且无 0 → one）。
# 若未设置 GLEVEL_OPT，默认显式传入，避免依赖隐式 autofix、日志更清晰。
GLEVEL_OPT="${GLEVEL_OPT:---g_level_int_encoding one}"
# 早停 / Plateau：本脚本在命令行末尾传入默认（会覆盖 GLEVEL_OPT 里同名项）：
#   export EARLY_STOP_PATIENCE=40      # 验证指标连续多少 epoch 无提升则停
#   export EARLY_STOP_MIN_EPOCHS=12    # 至少训满多少个 epoch 才允许早停（小 val 抗噪声）
#   export LR_SCHEDULER_PATIENCE=5     # ReduceLROnPlateau 等待 val loss 改善的 epoch 数
# 提升 val 准确率常用预设（与 train_task2_glevel 默认 mixup=0 配合；可整条复制）:
#   export GLEVEL_OPT="--select_best balanced_acc --label_smoothing 0.05"
# 关闭早停、跑满 --num_epochs（仍保存验证集上历史最优 best_model）:
#   export GLEVEL_OPT="--no_early_stop --cross_modal_attn --cross_modal_layers 1"
# 结构增强（与上条可叠加，注意引号内空格）:
#   export GLEVEL_OPT="--mixup_prob 0 --early_stop_patience 40 --temporal_gru --temporal_pool mean --modality_dropout_p 0.12"
# 若需覆盖默认 GLEVEL_OPT： export GLEVEL_OPT="--g_level_int_encoding one --select_best balanced_acc ..."
# 默认：class_weight=none + 平衡采样；少数类与多数类双叠加易整批塌到 Medium，勿轻易再加 --class_weight auto
# 方法八（消融）：当前 Nanbeige 最优路线建议保持默认，勿轻易启用 --temporal_gru、--glevel_loss coral、
# SWA（--swa_*）、过强 mixup；核查清单见 experiments/ablation_checklist.txt。
# 仍按 val CE 选模： export GLEVEL_OPT="--select_best val_ce"
# 跨模态注意力： export GLEVEL_OPT="--cross_modal_attn --cross_modal_layers 1"
# 验证集错分逐样本 CSV + 混淆矩阵： export VAL_ERRORS_CSV=./logs/val_glevel_errors.csv
SPLIT_LABELS="${SPLIT_LABELS:-1}"
VAL_ERR_ARG=()
if [ -n "${VAL_ERRORS_CSV:-}" ]; then
  VAL_ERR_ARG=(--val_errors_csv "${VAL_ERRORS_CSV}")
fi
SPLIT_ARG=""
if [ "${SPLIT_LABELS}" = "1" ]; then
  SPLIT_ARG="--labels_in_split_csv"
fi

# 路线 A：数据/特征补全预检（与当前 export 一致；通过后再训）： export ROUTE_A_PREFLIGHT=1
if [ "${ROUTE_A_PREFLIGHT:-0}" = "1" ]; then
  echo "[vote_train_glevel] ROUTE_A_PREFLIGHT=1 → bash tools/route_a_complete.sh" >&2
  bash "${_SCRIPT_DIR}/tools/route_a_complete.sh"
fi

# 预检 PyTorch，失败时追加 debug-f0e227.log 并退出（跳过：export SKIP_TORCH_PREFLIGHT=1）
if [ "${SKIP_TORCH_PREFLIGHT:-0}" != "1" ]; then
  if ! "${PYTHON}" -c "import torch"; then
    echo "[vote_train_glevel] PyTorch 导入失败。正在追加 tools/diagnose_torch_env.py 到 debug-f0e227.log …" >&2
    "${PYTHON}" tools/diagnose_torch_env.py 2>/dev/null || true
    bash "${_SCRIPT_DIR}/tools/print_torch_env_fix_hint.sh" 2>/dev/null || true
    exit 2
  fi
fi

"${PYTHON}" train_task2_glevel.py \
  --train_csv "${TRAIN_CSV}" \
  --val_csv "${VAL_CSV}" \
  --test_csv "${TEST_CSV}" \
  --rating_csv "${RATING_CSV}" \
  ${GLEVEL_OPT} \
  ${SPLIT_ARG} \
  --label_col g_level \
  --question q1 q2 q3 q4 q5 q6 \
  --video_dim 512 \
  --video_dir "${FEAT_TRAIN}/video" \
  --audio_dim 512 \
  --audio_dir "${FEAT_TRAIN}/audio" \
  --text_dim "${TEXT_DIM}" \
  --text_dir "${TEXT_TRAIN_DIR}" \
  --val_video_dir "${FEAT_VAL}/video" \
  --val_audio_dir "${FEAT_VAL}/audio" \
  --val_text_dir "${TEXT_VAL_DIR}" \
  --test_video_dir "${FEAT_TEST}/video" \
  --test_audio_dir "${FEAT_TEST}/audio" \
  --test_text_dir "${TEXT_TEST_DIR}" \
  --batch_size 32 \
  --num_epochs "${NUM_EPOCHS:-200}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --output_model "${OUTPUT_MODEL:-best_model_glevel.pth}" \
  --loss_plot_path "${LOSS_PLOT_PATH:-./loss_img/loss_glevel.png}" \
  --test_output_csv "${TEST_OUTPUT_CSV:-submission_glevel.csv}" \
  "${VAL_ERR_ARG[@]}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --lr_scheduler_patience "${LR_SCHEDULER_PATIENCE:-5}" \
  --early_stop_patience "${EARLY_STOP_PATIENCE:-40}" \
  --early_stop_min_epochs "${EARLY_STOP_MIN_EPOCHS:-20}"
