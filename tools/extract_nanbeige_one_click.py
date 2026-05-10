#!/usr/bin/env python3
"""
Nanbeige / CausalLM 文本特征一键提取（纯 Python，避免 .sh CRLF 问题）。

在 shell 里只粘贴以 #、export、python 开头的行；不要把「用法…」等中文整行当命令执行。

# cd /path/to/AVI2026_Track2_GLevel
# export TEXT_ROOT=/data/Super-Lu/dataset/train_text
# export OUT_ROOT=/data/Super-Lu/dataset/train_feature/text_nb
# python tools/extract_nanbeige_one_click.py

可选环境变量: MODEL_ID, BATCH, MAX_FILES, EXTRA_EXTRACT
全量提取前请勿继承 shell 中的 MAX_FILES（或执行 env -u MAX_FILES）。
终止: Ctrl+C 或  pkill -f "features.extract_text"
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


def _torchvision_import_help(exc: BaseException) -> str:
    msg = str(exc)
    if "nccl" in msg.lower() or "libtorch_cuda" in msg or "undefined symbol" in msg:
        return (
            "根因更像是当前环境的 CUDA 版 PyTorch/NCCL 动态库不一致（常见：pip/conda 混装或多次升级残留），"
            "不是单纯「少装 torchvision」。\n"
            "处理思路（择一）：\n"
            "  1) 同一渠道成套重装 torch + torchvision + torchaudio（与 requirements 一致），例如先卸载再装：\n"
            "       pip uninstall -y torch torchvision torchaudio\n"
            "       pip install -r requirements.txt\n"
            "     若仍报 nccl 符号错误，检查是否还有其它路径的 libnccl / 旧 conda torch 抢载。\n"
            "  2) 仅做文本特征提取时，可另建 venv 装 CPU 版 PyTorch（避免加载坏掉的 CUDA .so）：\n"
            "       pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
            "     提取会慢一些，但能跑通；训练仍可在别的正常 CUDA 环境里跑。\n"
        )
    return (
        "训练 g_level 只装了 torch 时常见此问题。请安装与当前 torch 匹配的 torchvision，例如：\n"
        "  pip install torchvision\n"
        "CUDA 版请与 torch 同源安装，例如（按 CUDA 版本改 cu121）：\n"
        "  pip install torchvision --index-url https://download.pytorch.org/whl/cu121\n"
        "完整依赖：pip install -r requirements.txt\n"
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        print(
            "当前 Python 未安装 transformers（训练脚本不依赖它，但 extract_text 需要）。\n"
            "请在本环境中执行其一：\n"
            "  pip install 'transformers>=4.36' accelerate\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    # 先触达 torch，便于与 torchvision 报错区分（NCCL 符号错常表现为 ImportError 但根因在 libtorch_cuda）
    try:
        import torch  # noqa: F401
    except Exception as e:
        print(
            "无法导入 torch。请检查 PyTorch 安装是否与当前 Python/CUDA 匹配。\n"
            f"原始错误: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # 新版 transformers 在导入 Llama 等配置时会间接加载 image_utils → 需要 torchvision.io
    try:
        import torchvision  # noqa: F401
        from torchvision.io import decode_image  # noqa: F401
    except Exception as e:
        print(
            "导入 torchvision / torchvision.io 失败（会导致 AutoTokenizer 相关导入失败）。\n"
            + _torchvision_import_help(e)
            + f"原始错误: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    text_root = os.environ.get("TEXT_ROOT")
    out_root = os.environ.get("OUT_ROOT")
    if not text_root or not out_root:
        print("请先 export TEXT_ROOT=转写根目录 与 OUT_ROOT=输出 npy 根目录", file=sys.stderr)
        sys.exit(2)

    model_id = os.environ.get("MODEL_ID", "Nanbeige/Nanbeige4-3B-Base")
    batch = os.environ.get("BATCH", "4")

    cmd = [
        sys.executable,
        "-m",
        "features.extract_text",
        "--text_dir",
        text_root,
        "--out_dir",
        out_root,
        "--model_id",
        model_id,
        "--batch_size",
        batch,
        "--pooling",
        "attn",
        "--layer_fuse",
        "mean_k",
        "--num_last_layers",
        "4",
    ]
    mf = (os.environ.get("MAX_FILES") or "").strip()
    if mf.isdigit() and int(mf) > 0:
        cmd.extend(["--max_files", mf])
    cmd.extend(shlex.split(os.environ.get("EXTRA_EXTRACT", "")))
    print("[extract_nanbeige_one_click]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(root))


if __name__ == "__main__":
    main()
