#!/usr/bin/env python3
"""从 combo_sweep_metrics.csv 选取若干同结构 checkpoint（默认可集成：无 fused LN 的 S_ref*）。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

# 与 SharedMLPwEnsemble 结构一致且评估时可混用（均无 --fused_layer_norm）
DEFAULT_COMBOS = frozenset(
    {"S_ref_plateau", "S_ref_sel_acc", "S_ref_step", "S_ref_cosine"}
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument(
        "--combos",
        type=str,
        default=",".join(sorted(DEFAULT_COMBOS)),
        help="逗号分隔 combo_id 白名单",
    )
    args = ap.parse_args()
    allowed = {x.strip() for x in args.combos.split(",") if x.strip()}
    rows: list[tuple[float, str]] = []
    with args.csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("exit_code", "")).strip() != "0":
                continue
            cid = (r.get("combo_id") or "").strip()
            if cid not in allowed:
                continue
            va = r.get("val_acc", "")
            if va in ("", "NA"):
                continue
            try:
                v = float(va)
            except ValueError:
                continue
            p = Path(r.get("output_model", ""))
            if p.is_file():
                rows.append((v, str(p)))
    rows.sort(key=lambda x: -x[0])
    seen: set[str] = set()
    out: list[str] = []
    for _, path in rows:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
        if len(out) >= args.top:
            break
    print(" ".join(out))


if __name__ == "__main__":
    main()
