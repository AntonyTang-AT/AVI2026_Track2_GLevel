#!/usr/bin/env bash
# AVI2026 / VideoMAE 训练示例（在服务器上于项目根目录执行，或先修改 PROJECT_ROOT）
set -euo pipefail

EXPERIMENT_NAME="videomae_avi2026"
PROJECT_ROOT="${PROJECT_ROOT:-/home/emo/antonytang/AVI2026_Track2_GLevel}"
cd "$PROJECT_ROOT"

TRAIN_CSV="${TRAIN_CSV:-/data/Super-Lu/dataset/train_data.csv}"
VAL_CSV="${VAL_CSV:-${PROJECT_ROOT}/data/val_data_new.csv}"
TEST_CSV="${TEST_CSV:-${PROJECT_ROOT}/data/test_data_basic_information.csv}"
RATING_CSV="${RATING_CSV:-${PROJECT_ROOT}/data/all_data.csv}"

AUDIO_DIR="${AUDIO_DIR:-/data/AVI2026/train_feature/audio}"
VIDEO_DIR="${VIDEO_DIR:-/data/AVI2026/train_feature/video}"
TEXT_DIR="${TEXT_DIR:-/data/AVI2026/train_feature/text}"

# 默认与 AVI2026 train_feature 实测一致（audio/video 512, text 768）；其它特征请改环境变量或运行 check_feature_shapes.py
VIDEO_DIM="${VIDEO_DIM:-512}"
AUDIO_DIM="${AUDIO_DIM:-512}"
TEXT_DIM="${TEXT_DIM:-768}"

mkdir -p ./train_print_log ./loss_img ./logs

# 可选手工特征：在下列命令中追加 --transcript_dir /path/to/txt --wav_dir /path/to/wav（与 .npy 同名前缀）

nohup python -u train_task2_vote.py \
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
  --learning_rate 1e-4 \
  --num_epochs 200 \
  --early_stop_patience 8 \
  --lr_scheduler_patience 3 \
  --mixup_prob 0.5 \
  --mixup_alpha 0.2 \
  --num_workers 4 \
  --output_model best_model_videomae.pth \
  --log_dir ./logs \
  --loss_plot_path "./loss_img/loss_curve_${EXPERIMENT_NAME}.png" \
  --test_output_csv submission.csv \
  --training_time "$(date -Iseconds 2>/dev/null || date)" \
  > "./train_print_log/${EXPERIMENT_NAME}.log" 2>&1 &

echo "Started training PID=$!  log: ./train_print_log/${EXPERIMENT_NAME}.log"
