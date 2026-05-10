#!/usr/bin/env python3
"""将当前解释器与 PyTorch 导入结果写入项目根 debug-f0e227.log（NDJSON，一行一条）。"""
# #region agent log
from __future__ import annotations

import json
import os
import sys
import time


def _root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _append(payload: dict) -> None:
    path = os.path.join(_root(), "debug-f0e227.log")
    payload.setdefault("sessionId", "f0e227")
    payload.setdefault("timestamp", int(time.time() * 1000))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    _append(
        {
            "hypothesisId": "H_env",
            "location": "tools/diagnose_torch_env.py:env",
            "message": "pre_torch_env",
            "data": {
                "executable": sys.executable,
                "python": sys.version.split()[0],
                "ld_library_path_len": len(_ld),
                "ld_library_path_head": _ld[:500] if _ld else "",
                "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
            },
        }
    )
    try:
        import torch

        _append(
            {
                "hypothesisId": "H_ok",
                "location": "tools/diagnose_torch_env.py:import",
                "message": "torch_import_ok",
                "data": {
                    "torch_version": torch.__version__,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "cuda": getattr(torch.version, "cuda", None),
                },
            }
        )
        print(
            "[diagnose_torch_env] OK",
            torch.__version__,
            "cuda=" + str(torch.cuda.is_available()),
            flush=True,
        )
        return 0
    except Exception as e:
        _append(
            {
                "hypothesisId": "H1_nccl",
                "location": "tools/diagnose_torch_env.py:import",
                "message": "torch_import_failed",
                "data": {"exc_type": type(e).__name__, "error": repr(e)},
            }
        )
        print(
            "[diagnose_torch_env] torch 导入失败，详情已写入",
            os.path.join(_root(), "debug-f0e227.log"),
            flush=True,
        )
        return 1


# #endregion

if __name__ == "__main__":
    raise SystemExit(main())
