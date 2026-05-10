#!/usr/bin/env bash
# 多随机种子训练示例：复制后改 python 参数与路径；与 vote_train 的 export 一致。
set -eu
cd "${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
for s in 42 123 2024; do
  echo "======== seed=$s ========"
  "${PYTHON}" train_task2_glevel.py \
    --seed "$s" \
    --output_model "best_model_glevel_seed${s}.pth" \
    --loss_plot_path "./loss_img/loss_glevel_seed${s}.png" \
    ${GLEVEL_EXTRA:-}
done
