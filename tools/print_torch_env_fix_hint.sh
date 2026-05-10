#!/usr/bin/env bash
# 由 vote_train_glevel.sh / vote_test_glevel.sh 在 PyTorch 预检失败时调用；仅向 stderr 打印可复制命令。
cat <<'HINT' >&2

========== PyTorch 导入失败（多为 NCCL/CUDA 栈与 torch 不匹配）==========
请在**新 conda 环境**重装（cu124 请按节点 nvidia-smi / 集群文档改为 cu121、cu118 等）:

  conda create -n avi_torch python=3.11 -y
  conda activate avi_torch
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  pip install -r requirements.txt
  export PYTHON="$CONDA_PREFIX/bin/python"
  cd /path/to/AVI2026_Track2_GLevel && bash vote_train_glevel.sh

根本规避 NCCL（CPU 训练）推荐:
  python3 tools/bootstrap_isolated_cpu_env.py

仅 CPU 验证管线（训练很慢）: 将上一行 pip 的 index-url 改为
  https://download.pytorch.org/whl/cpu

环境证据已尝试写入项目根: debug-f0e227.log
  （请 cat debug-f0e227.log 发维护者，或本机 scp 取回）
======================================================================

HINT
