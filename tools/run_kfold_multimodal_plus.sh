#!/usr/bin/env bash
# 分层 K 折（合并 train+val）+ 提交投票融合，超参与 vote_train_glevel_multimodal 默认一致。
# 环境变量：KFOLDS KFOLD_SEED KFOLD_OUT_DIR（同 one_click_kfold_glevel.py）
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

# 勿把文档里的占位符原样 export：FEAT_TRAIN="..." 会变成字面路径 .../audio（不存在），导致 411/411 行全被剔除。
_feat="${FEAT_TRAIN:-/data/Super-Lu/dataset/train_feature}"
_val="${FEAT_VAL:-/data/Super-Lu/dataset/val_feature}"
if [ "$_feat" = "..." ] || [ "$_val" = "..." ]; then
  echo "[run_kfold_multimodal_plus] 错误: FEAT_TRAIN/FEAT_VAL 不能为占位符 \"...\"。请设为与 vote_train_glevel 能跑通时相同的真实路径。" >&2
  exit 2
fi
for _d in "$_feat/audio" "$_feat/video" "$_val/audio" "$_val/video"; do
  if [ ! -d "$_d" ]; then
    echo "[run_kfold_multimodal_plus] 错误: 特征目录不存在: $_d（当前 FEAT_TRAIN=$_feat FEAT_VAL=$_val）" >&2
    exit 2
  fi
done

# run_kfold_glevel.py 已为每折传入 --labels_in_split_csv，勿在 GLEVEL_OPT 重复。
MM_PRESET="--glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001 --label_smoothing 0.05 --select_best balanced_acc --cross_modal_attn --cross_modal_layers 1 --modality_dropout_p 0.12 --seed 42 --scheduler_min_lr 1e-6"
if [ "${MM_TEMPORAL:-0}" = "1" ]; then
  MM_PRESET="$MM_PRESET --temporal_gru --temporal_pool mean --temporal_dropout 0.12"
  echo "[run_kfold_multimodal_plus] MM_TEMPORAL=1" >&2
fi

export GLEVEL_OPT="${GLEVEL_OPT:-$MM_PRESET}"
export KFOLD_OUT_DIR="${KFOLD_OUT_DIR:-${_ROOT}/kfold_glevel_multimodal_plus}"
export KFOLDS="${KFOLDS:-5}"
export KFOLD_SEED="${KFOLD_SEED:-42}"

echo "[run_kfold_multimodal_plus] GLEVEL_OPT=$GLEVEL_OPT" >&2
echo "[run_kfold_multimodal_plus] KFOLDS=$KFOLDS out=$KFOLD_OUT_DIR" >&2

"${PYTHON:-python3}" "${_ROOT}/one_click_kfold_glevel.py"

echo "[run_kfold_multimodal_plus] 融合提交: ${KFOLD_OUT_DIR}/submission_glevel_kfold_vote.csv" >&2
