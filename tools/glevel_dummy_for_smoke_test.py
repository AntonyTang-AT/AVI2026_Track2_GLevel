#!/usr/bin/env python3
"""
⚠️ 仅用于「跑通训练管线」冒烟测试：生成假的 g_level（0/1/2），不能用于真实比赛或论文。

真实标签必须从赛方 / 官方标注表获得，再用 build_glevel_train_csv.py 从源表抽取。
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="生成假的 glevel_train.csv（仅测试代码能否训练，指标无意义）"
    )
    ap.add_argument("--train_csv", default="./data/train_data.csv")
    ap.add_argument("--val_csv", default="./data/val_data_new.csv")
    ap.add_argument("--out", default="./data/glevel_train.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--strategy",
        choices=("random", "balanced"),
        default="random",
        help="random=均匀随机三类；balanced=尽量三等分",
    )
    args = ap.parse_args()

    warnings.warn(
        "你正在生成假标签，训练得到的模型与验证指标没有任何实际意义！",
        UserWarning,
        stacklevel=1,
    )
    print("=" * 60)
    print("DUMMY LABELS — 仅冒烟测试，提交或报告前请替换为真实 glevel_train.csv")
    print("=" * 60)

    tr = pd.read_csv(args.train_csv)
    va = pd.read_csv(args.val_csv)
    ids = pd.unique(pd.concat([tr["id"], va["id"]], ignore_index=True).astype(str))
    rng = np.random.default_rng(args.seed)

    if args.strategy == "random":
        y = rng.integers(0, 3, size=len(ids))
    else:
        n = len(ids)
        base = np.array([0] * (n // 3) + [1] * (n // 3) + [2] * (n // 3))
        pad = n - len(base)
        base = np.concatenate([base, rng.integers(0, 3, size=pad)])
        rng.shuffle(base)
        y = base

    out = pd.DataFrame({"id": ids, "g_level": y.astype(int)})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"已写入 {os.path.abspath(args.out)} ，共 {len(out)} 行")


if __name__ == "__main__":
    main()
