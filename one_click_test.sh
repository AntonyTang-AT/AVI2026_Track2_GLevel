#!/usr/bin/env bash
# =============================================================================
# 服务器一键测试（上传代码到服务器后，在项目根目录执行本脚本）
#
# 本机打包上传示例（在含 AVI2026_Track2_GLevel 的目录执行，按你的用户与路径改）：
#   rsync -avz --progress \
#     --exclude '.git' --exclude '__pycache__' --exclude '*.pth' --exclude '*.zip' \
#     ./AVI2026_Track2_GLevel/  user@server:~/AVI2026_Track2_GLevel/
#
# 或使用 scp（端口示例 -P 24322）：
#   scp -P 24322 -r dataset/baseline_dataset2_vote.py train_task2_glevel.py \\
#       emo@183.196.130.56:~/antonytang/AVI2026_Track2_GLevel/
#
# ⚠️ 须 scp 到「你 ssh 上去跑 bash one_click_test.sh 的那台机器」的同一目录。
#    若登录跳板机 IP 与 计算节点 node243 不同盘，请把文件同步到 node243 可见的路径
#    （共享家目录 / NFS / 或在 node243 上再 scp/rsync 一次）。
#
# 服务器上：
#   cd ~/AVI2026_Track2_GLevel
#   chmod +x one_click_test.sh
#   export CONDA_ENV=365Aspects-main          # 可选，有 conda 环境则设
#   bash one_click_test.sh                    # 默认：检查依赖 + 特征形状 + 短训 1 epoch
#
# 环境变量（可选，与 vote_train_glevel.sh 一致）：
#   QUICK_ONLY=1           只做环境/导入/特征检查，不训练
#   PIP_INSTALL=1          先 pip install -r requirements.txt
#   RUN_SHORT_TRAIN=0      跳过短训（默认 1：在数据就绪时跑 1 个 epoch）
#   TEXT_DIM               文本特征维，默认 768
#   TRAIN_CSV VAL_CSV RATING_CSV FEAT_TRAIN FEAT_VAL FEAT_TEST（默认 /data/AVI2026/test_feature）
#   GLEVEL_OPT             例如 --glevel_csv ./data/xxx.csv
#   SPLIT_LABELS=0         不设则加 --labels_in_split_csv（划分表含 g_level 时）
#
# -----------------------------------------------------------------------------
# 若在 Linux 上出现: invalid option / $'\r': command not found / cd: '...\r': No such file
# 说明脚本被保存成 Windows 换行(CRLF)。在项目根执行一次修复后再 bash：
#   sed -i 's/\r$//' one_click_test.sh vote_train_glevel.sh vote_test_glevel.sh
# 或使用 dos2unix（需安装）。自检：file one_click_test.sh 不应含 “CRLF line terminators”。
# -----------------------------------------------------------------------------
# =============================================================================
set -eu

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TRAIN_CSV="${TRAIN_CSV:-/data/Super-Lu/dataset/train_data.csv}"
VAL_CSV="${VAL_CSV:-/data/Super-Lu/dataset/val_data.csv}"
TEST_CSV="${TEST_CSV:-${ROOT}/data/test_data_basic_information.csv}"
RATING_CSV="${RATING_CSV:-/data/Super-Lu/dataset/train_data.csv}"
FEAT_TRAIN="${FEAT_TRAIN:-/data/Super-Lu/dataset/train_feature}"
FEAT_VAL="${FEAT_VAL:-/data/Super-Lu/dataset/val_feature}"
FEAT_TEST="${FEAT_TEST:-/data/AVI2026/test_feature}"
TEXT_DIM="${TEXT_DIM:-768}"
TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${FEAT_TRAIN}/text}"
TEXT_VAL_DIR="${TEXT_VAL_DIR:-${FEAT_VAL}/text}"
TEXT_TEST_DIR="${TEXT_TEST_DIR:-${FEAT_TEST}/text}"

RUN_SHORT_TRAIN="${RUN_SHORT_TRAIN:-1}"
QUICK_ONLY="${QUICK_ONLY:-0}"
PIP_INSTALL="${PIP_INSTALL:-0}"

SPLIT_ARG=""
if [ "${SPLIT_LABELS:-1}" = "1" ]; then
  SPLIT_ARG="--labels_in_split_csv"
fi

echo "=============================================="
echo "[one_click_test] ROOT=$ROOT"
echo "=============================================="

if [ -n "${CONDA_ENV:-}" ]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  echo "[one_click_test] conda: $CONDA_ENV"
fi

if [ "${PIP_INSTALL}" = "1" ]; then
  echo "== pip install -r requirements.txt"
  pip install -r requirements.txt
fi

echo "== 1) Python / PyTorch / 工程导入"
python - <<'PY'
import sys
print("python", sys.version.split()[0])
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
from dataset.baseline_dataset2_vote import MultimodalDatasetForTrainT2
from model.vote_model.M_model import SharedMLPwEnsemble
print("import dataset + model: ok")
PY

echo "== 2) 训练脚本参数（确认含 glevel / labels_in_split_csv）"
python train_task2_glevel.py --help | grep -E "glevel|labels_in_split|text_dim" | head -n 6 || true

echo "== 3) 特征目录与抽样 shape（FEAT_TRAIN=$FEAT_TRAIN）"
if [ -d "${FEAT_TRAIN}/audio" ] && [ -d "${FEAT_TRAIN}/video" ] && [ -d "${FEAT_TRAIN}/text" ]; then
  python tools/check_feature_shapes.py \
    --audio_dir "${FEAT_TRAIN}/audio" \
    --video_dir "${FEAT_TRAIN}/video" \
    --text_dir "${TEXT_TRAIN_DIR}"
else
  echo "WARN: FEAT_TRAIN 下缺少 audio/video/text 子目录，跳过 shape 检查。"
  echo "      请 export FEAT_TRAIN=你的特征根目录"
fi

if [ "${QUICK_ONLY}" = "1" ]; then
  echo "== QUICK_ONLY=1 ，结束。"
  exit 0
fi

if [ "${RUN_SHORT_TRAIN}" != "1" ]; then
  echo "== RUN_SHORT_TRAIN!=1 ，跳过短训。"
  exit 0
fi

if [ ! -f "$TRAIN_CSV" ] || [ ! -f "$VAL_CSV" ]; then
  echo "WARN: TRAIN_CSV 或 VAL_CSV 不存在，跳过短训。"
  echo "      TRAIN_CSV=$TRAIN_CSV"
  echo "      VAL_CSV=$VAL_CSV"
  exit 0
fi

if [ ! -d "${FEAT_TRAIN}/audio" ]; then
  echo "WARN: 训练特征目录不可用，跳过短训。"
  exit 0
fi

mkdir -p ./smoke_check ./smoke_check/loss_img ./smoke_check/logs

echo "== 4) 短训 1 epoch（num_workers=0，结果写入 smoke_check/）"
set +e
python train_task2_glevel.py \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --test_csv "$TEST_CSV" \
  --rating_csv "$RATING_CSV" \
  ${GLEVEL_OPT:-} \
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
  --batch_size 8 \
  --num_epochs 1 \
  --early_stop_patience 99 \
  --num_workers 0 \
  --output_model ./smoke_check/best_smoke.pth \
  --loss_plot_path ./smoke_check/loss_img/loss_smoke.png \
  --log_dir ./smoke_check/logs \
  --test_output_csv ./smoke_check/submission_smoke.csv
EC=$?
set -e

if [ "$EC" -eq 0 ]; then
  echo "=============================================="
  echo "[one_click_test] 短训成功。模型: ./smoke_check/best_smoke.pth"
  echo "可继续: bash vote_train_glevel.sh  全量训练"
  echo "或仅推理: TEST_MODEL=./smoke_check/best_smoke.pth bash vote_test_glevel.sh"
  echo "=============================================="
else
  echo "=============================================="
  echo "[one_click_test] 短训失败 (exit $EC)。请根据上方 Traceback 检查 CSV / 特征 / TEXT_DIM。"
  echo "=============================================="
  exit "$EC"
fi
