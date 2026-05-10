#!/usr/bin/env python3
"""
从含 g_level 的单一划分表中分层抽样出 train_fixed / val_fixed（默认 val 占 15%，与调参稳定性建议一致）。
用于扩大/稳定验证评估（不依赖赛方 val 划分）。

本地评估协议（方法一/九）：将本脚本生成的 val_fixed 作为 dev_holdout，所有调参、温度缩放、
阈值与集成选择优先在该集上完成；官方 val_data.csv 建议仅「封板」时评估 1～2 次，避免过拟合。
详见 experiments/local_eval_protocol.txt。

示例：
  python tools/split_train_val.py \\
    --in_csv /data/Super-Lu/dataset/train_data.csv \\
    --out_train ./data/train_fixed.csv \\
    --out_val ./data/val_fixed.csv \\
    --val_ratio 0.15 --seed 42 --label_col g_level

  # 固定验证条数（与 val_ratio 二选一）
  python tools/split_train_val.py ... --val_n 90 --seed 1001
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="含 id 与标签列的表（常与官方 train 相同）")
    ap.add_argument("--out_train", default="./data/train_fixed.csv")
    ap.add_argument("--out_val", default="./data/val_fixed.csv")
    ap.add_argument(
        "--val_ratio",
        type=float,
        default=None,
        help="验证集占比，分层抽样（与 --val_n 二选一；均未给时默认 0.15）",
    )
    ap.add_argument(
        "--val_n",
        type=int,
        default=None,
        help="验证集绝对条数（与 --val_ratio 二选一；须 < 总行数，且各类尽量≥2 以便 stratify）",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_col", default="g_level", help="分层用标签列名")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if "id" not in df.columns:
        raise SystemExit("输入 CSV 须含 id 列")
    if args.label_col not in df.columns:
        raise SystemExit(f"输入 CSV 须含标签列 {args.label_col!r}，当前为 {list(df.columns)}")

    y = df[args.label_col].astype(str)
    vc = y.value_counts()
    strat = y if vc.min() >= 2 else None
    if strat is None:
        print(
            "[split_train_val] 警告: 某类样本<2，无法 stratify，改为随机划分（可能类别分布略有偏）",
            file=sys.stderr,
        )

    if args.val_n is not None and args.val_ratio is not None:
        raise SystemExit("请只指定 --val_n 或 --val_ratio 之一")
    if args.val_n is not None:
        if args.val_n <= 0 or args.val_n >= len(df):
            raise SystemExit(f"--val_n 须在 1..{len(df)-1} 之间")
        test_size = args.val_n
    else:
        vr = 0.15 if args.val_ratio is None else args.val_ratio
        if not (0.0 < vr < 1.0):
            raise SystemExit("--val_ratio 须在 (0,1)")
        test_size = vr

    tr, va = train_test_split(
        df,
        test_size=test_size,
        random_state=args.seed,
        stratify=strat,
        shuffle=True,
    )
    tr = tr.reset_index(drop=True)
    va = va.reset_index(drop=True)

    tr.to_csv(args.out_train, index=False)
    va.to_csv(args.out_val, index=False)
    print(
        f"[split_train_val] in={args.in_csv} n={len(df)} → "
        f"train={len(tr)} val={len(va)} | out_train={args.out_train} out_val={args.out_val} "
        f"(stratify={args.label_col}, seed={args.seed})"
    )


if __name__ == "__main__":
    main()
