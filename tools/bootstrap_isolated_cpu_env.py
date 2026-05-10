#!/usr/bin/env python3
"""
在仓库根目录创建独立 venv，安装 CPU 版 torch 三件套 + requirements-core.txt。
不依赖 bash/set -o pipefail，避免 Windows CRLF 同步到 Linux 后脚本首行即失败。

用法（在项目根）:
  python3 tools/bootstrap_isolated_cpu_env.py

自定义 venv 路径:
  VENV_DIR=/path/to/venv python3 tools/bootstrap_isolated_cpu_env.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    venv_dir = Path(os.environ.get("VENV_DIR", str(root / ".venv_glevel_cpu"))).resolve()
    py = Path(sys.executable)

    print(f"[bootstrap] ROOT={root}")
    print(f"[bootstrap] venv -> {venv_dir}")
    print(f"[bootstrap] 用于创建 venv 的解释器: {py}")

    subprocess.run([str(py), "-m", "venv", str(venv_dir)], check=True, cwd=str(root))

    if os.name == "nt":
        pip = venv_dir / "Scripts" / "pip.exe"
        python_v = venv_dir / "Scripts" / "python.exe"
    else:
        pip = venv_dir / "bin" / "pip"
        python_v = venv_dir / "bin" / "python"

    subprocess.run([str(python_v), "-m", "pip", "install", "-U", "pip", "wheel"], check=True)
    subprocess.run(
        [
            str(python_v),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
        check=True,
    )
    subprocess.run(
        [str(python_v), "-m", "pip", "install", "-r", str(root / "requirements-core.txt")],
        check=True,
    )

    print("\n=== 完成。请在终端执行（Linux/macOS）===")
    print(f'  source "{venv_dir}/bin/activate"')
    print(f'  export PYTHON="{venv_dir}/bin/python"')
    print(f'  export PROJECT_ROOT="{root}"')
    print(f'  cd "{root}" && python -c "import torch; print(torch.__version__, torch.cuda.is_available())"')
    print('  bash vote_train_glevel.sh')
    print("\n期望最后一行 cuda 为 False。\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
