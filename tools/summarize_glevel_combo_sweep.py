#!/usr/bin/env python3
"""读取 run_glevel_gpu_combo_sweep.sh 产出的 CSV，按 val_acc / val_bal_acc 排序并打印最优行。"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


def _f(x: str) -> float | None:
    if x in ("", "NA", None):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: summarize_glevel_combo_sweep.py <combo_sweep_metrics.csv>", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"找不到文件: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ok = [
        r
        for r in rows
        if str(r.get("exit_code", "")).strip() == "0" and _f(r.get("val_acc", "")) is not None
    ]
    if not ok:
        print(f"[summarize] {path} 中无 exit_code=0 且 val_acc 有效的行。")
        return
    by_acc = max(ok, key=lambda r: _f(r["val_acc"]) or -1.0)
    by_bacc = max(ok, key=lambda r: _f(r["val_bal_acc"]) or -1.0)

    def fmt(r: dict) -> str:
        return (
            f"combo={r.get('combo_id')} seed={r.get('seed')} "
            f"val_acc={r.get('val_acc')} val_bal_acc={r.get('val_bal_acc')} "
            f"macro_f1={r.get('val_macro_f1')} best_epoch={r.get('best_epoch')} "
            f"path={r.get('output_model')}"
        )

    accs = [_f(r["val_acc"]) for r in ok if _f(r["val_acc"]) is not None]
    baccs = [_f(r["val_bal_acc"]) for r in ok if _f(r["val_bal_acc"]) is not None]
    print(f"[summarize] n_valid={len(ok)}  file={path}")
    if accs:
        print(
            f"[summarize] val_acc: max={max(accs):.4f} mean={sum(accs)/len(accs):.4f} "
            f"stdev={math.sqrt(sum((x - sum(accs)/len(accs))**2 for x in accs) / (len(accs)-1)) if len(accs) > 1 else 0.0:.4f}"
        )
    print("[summarize] 按 val_acc 最优:\n  " + fmt(by_acc))
    print("[summarize] 按 val_bal_acc 最优:\n  " + fmt(by_bacc))


if __name__ == "__main__":
    main()
