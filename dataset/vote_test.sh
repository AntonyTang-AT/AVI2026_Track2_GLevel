#!/usr/bin/env bash
# 仅推理 / 生成 submission（参数需与训练时特征维度、路径一致）
set -euo pipefail

EXPERIMENT_NAME="videomae_avi2026"
PROJECT_ROOT="${PROJECT_ROOT:-/home/emo/antonytang/AVI2026_Track2_GLevel}"
cd "$PROJECT_ROOT"

TEST_MODEL="${TEST_MODEL:-best_model_videomae.pth}"

TRAIN_CSV="${TRAIN_CSV:-/data/Super-Lu/dataset/train_data.csv}"
VAL_CSV="${VAL_CSV:-${PROJECT_ROOT}/data/val_data_new.csv}"
TEST_CSV="${TEST_CSV:-${PROJECT_ROOT}/data/test_data_basic_information.csv}"
RATING_CSV="${RATING_CSV:-${PROJECT_ROOT}/data/all_data.csv}"

AUDIO_DIR="${AUDIO_DIR:-/data/AVI2026/train_feature/audio}"
VIDEO_DIR="${VIDEO_DIR:-/data/AVI2026/train_feature/video}"
TEXT_DIR="${TEXT_DIR:-/data/AVI2026/train_feature/text}"

VIDEO_DIM="${VIDEO_DIM:-512}"
AUDIO_DIM="${AUDIO_DIM:-512}"
TEXT_DIM="${TEXT_DIM:-768}"

mkdir -p ./train_print_log ./loss_img ./logs

python -u python/train_task2_vote.py \
  --only_test \
  --test_model "$TEST_MODEL" \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --test_csv "$TEST_CSV" \
  --question q1 q2 q3 q4 q5 q6 \
  --label_col Integrity Collegiality Social_versatility Development_orientation Hireability \
  --rating_csv "$RATING_CSV" \
  --video_dim "$VIDEO_DIM" \
  --video_dir "$VIDEO_DIR" \
  --audio_dim "$AUDIO_DIM" \
  --audio_dir "$AUDIO_DIR" \
  --text_dim "$TEXT_DIM" \
  --text_dir "$TEXT_DIR" \
  --batch_size 32 \
  --num_workers 4 \
  --log_dir ./logs \
  --loss_plot_path "./loss_img/loss_curve_${EXPERIMENT_NAME}_test.png" \
  --test_output_csv submission.csv
