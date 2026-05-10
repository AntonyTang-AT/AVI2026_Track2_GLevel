#!/usr/bin/env bash
# Nanbeige 按划分提取示例（勿直接执行整文件：按需复制 export 行）。
# OUT_ROOT 建议放在工程目录下，避免写入只读的 /data/Super-Lu。
# 依赖：与 extract_nanbeige_one_click.py 相同（transformers、torch、torchvision）。
#
# 用法（在项目根、已激活 venv）:
#   1) 设置 TEXT_ROOT 为赛方「该划分」转写根目录（内含 **/*.txt，相对路径决定输出子目录）
#   2) 设置 OUT_ROOT 为要写 .npy 的目录
#   3) python tools/extract_nanbeige_one_click.py
#
# 训练时示例（与 vote_train_glevel.sh 一致）:
#   TEXT_TRAIN_DIR=$PWD/data/text_nb_train   或合并目录 $PWD/data/text_nb
#   TEXT_VAL_DIR=$PWD/data/text_nb_val
#   TEXT_TEST_DIR=$PWD/data/text_nb_test
#   export NANBEIGE_TEXT=1 TEXT_DIM=2560
#
# === 训练集转写（示例路径请改成真实 TEXT_ROOT）===
# export TEXT_ROOT=/path/to/train_transcripts
# export OUT_ROOT=/path/to/AVI2026_Track2_GLevel/data/text_nb_train
# export MODEL_ID=Nanbeige/Nanbeige4-3B-Base BATCH=4
# python tools/extract_nanbeige_one_click.py
#
# === 验证集 ===
# export TEXT_ROOT=/path/to/val_transcripts
# export OUT_ROOT=/path/to/AVI2026_Track2_GLevel/data/text_nb_val
# python tools/extract_nanbeige_one_click.py
#
# === 测试集（提交必需，否则 predict_test 跳过）===
# export TEXT_ROOT=/path/to/test_transcripts
# export OUT_ROOT=/path/to/AVI2026_Track2_GLevel/data/text_nb_test
# python tools/extract_nanbeige_one_click.py
#
# 提取后与训练前全量预检（路线 A）:
#   bash tools/route_a_complete.sh
