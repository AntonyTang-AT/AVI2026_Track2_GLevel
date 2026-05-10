#!/usr/bin/env python3
"""
汇总「多轮随机划分」产生的多个 combo_sweep_metrics.csv（位于 BASE/round_*/sweep/）。

输出：
  - 打印每个 (combo_id, seed) 在各 round 上的 val_acc 均值 / 标准差（跨 round）
  - 写入 BASE/cv_agg_by_combo_seed.tsv

用法:
  python tools/summarize_cv_combo_sweep.py experiments/gpu_combo_sweep/cv_20260510_120000
"""
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
        print("用法: summarize_cv_combo_sweep.py <BASE_HUNT_DIR>", file=sys.stderr)
        sys.exit(2)
    base = Path(sys.argv[1]).resolve()
    if not base.is_dir():
        print(f"不是目录: {base}", file=sys.stderr)
        sys.exit(1)

    paths = sorted(base.glob("round_*/sweep/combo_sweep_metrics.csv"))
    if not paths:
        print(f"[cv_summarize] 未找到 {base}/round_*/sweep/combo_sweep_metrics.csv")
        return

    # (combo, seed) -> list of (round, val_acc, val_bal_acc)
    accs: dict[tuple[str, str], list[tuple[str, float, float | None]]] = defaultdict(list)
    for p in paths:
        round_tag = p.parents[1].name  # round_k
        with p.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("exit_code", "")).strip() != "0":
                    continue
                va = _f(row.get("val_acc", ""))
                if va is None:
                    continue
                vb = _f(row.get("val_bal_acc", ""))
                key = (row.get("combo_id", ""), str(row.get("seed", "")))
                accs[key].append((round_tag, va, vb))

    rows_out: list[tuple[str, str, int, str, str, str]] = []
    for (combo, seed), lst in sorted(accs.items()):
        vas = [x[1] for x in lst]
        vbs = [x[2] for x in lst if x[2] is not None]
        m = sum(vas) / len(vas)
        sd = (
            math.sqrt(sum((x - m) ** 2 for x in vas) / (len(vas) - 1))
            if len(vas) > 1
            else 0.0
        )
        mb = sum(vbs) / len(vbs) if vbs else float("nan")
        rounds = ",".join(x[0] for x in lst)
        rows_out.append((combo, seed, len(vas), f"{m:.4f}", f"{sd:.4f}", f"{mb:.4f}" if vbs else "NA"))

    rows_out.sort(key=lambda r: float(r[3]), reverse=True)

    out_tsv = base / "cv_agg_by_combo_seed.tsv"
    with out_tsv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["combo_id", "seed", "n_rounds", "mean_val_acc", "stdev_val_acc", "mean_val_bal_acc"])
        w.writerows(rows_out)

    print(f"[cv_summarize] 读取 {len(paths)} 个 metrics 文件；有效 (combo,seed) 键 {len(rows_out)}")
    print(f"[cv_summarize] 已写 {out_tsv}")
    if rows_out:
        print("[cv_summarize] 按 mean_val_acc 前 5:")
        for r in rows_out[:5]:
            print(f"  combo={r[0]} seed={r[1]} rounds={r[2]} mean_acc={r[3]} stdev={r[4]} mean_bal={r[5]}")


if __name__ == "__main__":
    main()
