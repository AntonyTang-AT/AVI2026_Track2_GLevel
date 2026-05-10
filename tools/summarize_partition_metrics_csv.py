#!/usr/bin/env python3
"""汇总 PARTITION_ROUNDS 模式下单一 combo_sweep_metrics.csv（含 split_round 列）的跨 part 统计。"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
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
        print("用法: summarize_partition_metrics_csv.py <combo_sweep_metrics.csv>", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1])
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if str(r.get("exit_code", "")).strip() != "0":
            continue
        sr = (r.get("split_round") or "").strip()
        if not sr:
            continue
        va = _f(r.get("val_acc", ""))
        if va is None:
            continue
        groups[(r.get("combo_id", ""), str(r.get("seed", "")))].append(va)

    out_lines = []
    for (c, s), vals in sorted(groups.items()):
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        out_lines.append((c, s, len(vals), m, sd))

    out_lines.sort(key=lambda x: x[3], reverse=True)
    print(f"[partition_metrics] file={path} groups={len(out_lines)}")
    for c, s, n, m, sd in out_lines[:20]:
        print(f"  combo={c} seed={s} n_parts={n} mean_val_acc={m:.4f} stdev={sd:.4f}")


if __name__ == "__main__":
    main()
