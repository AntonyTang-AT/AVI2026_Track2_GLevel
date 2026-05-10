#!/usr/bin/env bash
# 从根本上避开 base 里损坏的 CUDA+NCCL：独立 venv + 仅 CPU 版 PyTorch。
# 若本脚本在 Linux 上报 set: pipefail 或 : invalid option，多为 CRLF；请改用：
#   python3 tools/bootstrap_isolated_cpu_env.py
# 或: sed -i 's/\r$//' tools/bootstrap_isolated_cpu_env.sh
set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV_DIR="${VENV_DIR:-$ROOT/.venv_glevel_cpu}"
PY="${PYTHON_BOOTSTRAP:-}"
if [ -z "$PY" ]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PY="$(command -v python3.11)"
  else
    PY="$(command -v python3)"
  fi
fi

echo "[bootstrap] ROOT=$ROOT"
echo "[bootstrap] venv=$VENV_DIR  使用解释器: $PY"
"$PY" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel
python -m pip install --force-reinstall torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r "$ROOT/requirements-core.txt"

echo ""
echo "=== 已完成。请在本终端执行 ==="
echo "  source \"$VENV_DIR/bin/activate\""
echo "  export PYTHON=\"$VENV_DIR/bin/python\""
echo "  cd \"$ROOT\" && python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\""
echo "  bash vote_train_glevel.sh"
echo ""
echo "（期望输出 cuda False。若 bash 脚本异常，请用: python3 tools/bootstrap_isolated_cpu_env.py）"
