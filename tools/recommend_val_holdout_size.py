#!/usr/bin/env python3
"""
基于「准确率估计的标准误 + 训练剩余样本量」给出从官方 train 里划出验证集的推荐规模。

无 sklearn 依赖；可选读 CSV 打印各类计数（否则假定 n≈450、近似均衡三类）。

用法:
  python tools/recommend_val_holdout_size.py
  python tools/recommend_val_holdout_size.py --train_csv /data/Super-Lu/dataset/train_data.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path


def _se_binomial(p: float, n: int) -> float:
    """准确率 p 下，n 条样本时 acc 抽样标准误近似 sqrt(p(1-p)/n)。"""
    if n <= 0:
        return float("nan")
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def main() -> None:
    ap = argparse.ArgumentParser(description="推荐从 train 划出的验证集规模（兼顾方差与剩余训练量）")
    ap.add_argument("--train_csv", default="", help="官方 train_data.csv；给定则统计总行数与 g_level 分布")
    ap.add_argument(
        "--p_guess",
        type=float,
        default=0.55,
        help="用于估算 SE 的猜测准确率（默认 0.55，接近中等难度三分类）",
    )
    args = ap.parse_args()

    n_total = 450
    counts = None
    if args.train_csv:
        p = Path(args.train_csv)
        if not p.is_file():
            raise SystemExit(f"找不到 {p}")
        import pandas as pd

        df = pd.read_csv(p)
        n_total = len(df)
        if "g_level" in df.columns:
            counts = df["g_level"].value_counts().sort_index().to_dict()

    # 候选：接近赛方 val(~63)、常用 15%/20%、略大以压制方差
    candidates = sorted(
        set(
            [
                max(30, n_total // 20),
                max(45, int(round(0.1 * n_total))),
                max(50, int(round(0.14 * n_total))),
                int(round(0.15 * n_total)),
                int(round(0.18 * n_total)),
                int(round(0.2 * n_total)),
                int(round(0.25 * n_total)),
            ]
        )
    )
    candidates = [v for v in candidates if 20 <= v < n_total - 50]

    print(f"[recommend_val_holdout] 总行数 n_total={n_total}（用于划分的官方 train）")
    if counts:
        print(f"[recommend_val_holdout] g_level 计数: {counts}")
    print()
    print(
        "下列假设：分层划分后验证集 acc 仍可用二项近似估 SE（真实相关性会使 SE 略偏小，"
        "仅作量级参考）。p_guess=%.2f。\n" % args.p_guess
    )
    print(f"{'val_n':>6} {'train_n':>8} {'SE(acc)≈':>12} {'95%粗间隔≈±':>14}")
    print("-" * 46)
    p = args.p_guess
    for vn in candidates:
        tn = n_total - vn
        se = _se_binomial(p, vn)
        half95 = 1.96 * se
        print(f"{vn:6d} {tn:8d} {se:12.4f} {half95:14.4f}")

    print()
    print(
        "[结论（实践向）]\n"
        "· **过小（如 ~45～55）**：标准误大，单次划分下 val_acc 抖动明显，容易「选对模型」是运气。\n"
        "· **赛方量级（~60～70）**：与当前官方 val 规模同阶，可作对照，但仍属于「中等噪声」。\n"
        "· **推荐主力区间**：**约为总样本的 15%～20%**（450 条时约 **68～90** 条验证），"
        "在「val 指标更稳」与「训练仍剩 ~360～380 条」之间较均衡。\n"
        "· **更大验证（≥100）**：估计更稳，但训练数据减少，若任务数据稀缺可能伤泛化；可用 **多轮随机划分（CV）** 弥补。\n"
        "· 若采用 **多轮随机 train/val（本仓库 `run_glevel_gpu_combo_sweep_cv.sh`）**："
        "单轮 val 可略小（如 **60～80**），靠 **跨轮平均 / 方差** 选模型。\n"
    )


    print()
    print(
        "--- 合并官方 train+val 为池时的 train_n 建议（≈450+64 条）---\n"
        "· 合并池约 **514** 条（无重复 id 时）。\n"
        "· 推荐 **train_n=418**，剩余验证 **≈96**：约为 pool 的 **18.5%** 作验证，acc 标准误约 **0.05**（p≈0.55 量级）。\n"
        "· 若更重视验证稳定性：train_n **400**（验证 ~114）；若更重视训练量：train_n **432**（验证 ~82）。\n"
        "· 实际脚本默认 **POOL_TRAIN_N=418**，可通过环境变量覆盖。\n"
    )


if __name__ == "__main__":
    main()
