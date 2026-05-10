#!/usr/bin/env bash
# g_level 消融预设：复制下方 export GLEVEL_OPT 到 shell，勿 source 本文件（仅作文档）。
# 建议每次只改 1～2 个变量，便于对比 val_summary。
#
# --- 路线 B：结构 / 选模 / 正则（与 cross_modal 可叠加，自行拼接）---
#
# B1 时序 GRU
#   export GLEVEL_OPT="--temporal_gru --temporal_pool mean"
#
# B2 模态 dropout
#   export GLEVEL_OPT="--modality_dropout_p 0.12"
#
# B3 选模更贴近 accuracy / 均衡
#   export GLEVEL_OPT="--select_best balanced_acc"
#   export GLEVEL_OPT="--select_best val_ce"
#
# B4 标签平滑（单独试）
#   export GLEVEL_OPT="--label_smoothing 0.05"
#
# B5 学习率（默认 vote_train 为 1e-4，须在 scripts/glevel_train.sh 中改传参或改用 python python/train_task2_glevel.py --learning_rate）
# 推荐：复制 vote_train 最后一行 python 调用到命令行并加 --learning_rate 5e-5
#
# B6 组合示例（关早停 + 跨模态 + GRU，训练久）
#   export GLEVEL_OPT="--no_early_stop --cross_modal_attn --cross_modal_layers 1 --temporal_gru --temporal_pool mean"
#
# --- 路线 D：Medium / 类不平衡（勿与默认平衡采样叠用 class_weight，二选一）---
# 见 data/GLEVEL_README.txt 「十二」节。
#
echo "[glevel_ablation_presets] 本文件为注释文档；请打开编辑复制所需 export 行。" >&2
