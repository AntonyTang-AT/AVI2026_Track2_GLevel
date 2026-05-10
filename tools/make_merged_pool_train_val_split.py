#!/usr/bin/env python3
"""
将官方 train CSV 与 val CSV 纵向合并为「标注池」，再按给定条数分层随机划出：
  - train_fold.csv（固定 train_n 条，用于训练）
  - val_fold.csv（池中剩余，用于验证准确率）

用于同一模型配置下多次换划分评估稳定性；训练集中可能含原 val id，须在 train_task2_glevel 中开启 --train_feat_fallback。

示例（pool≈514、train_n=418 → val≈96）:
  python tools/make_merged_pool_train_val_split.py \\
    --official_train_csv /data/Super-Lu/dataset/train_data.csv \\
    --official_val_csv /data/Super-Lu/dataset/val_data.csv \\
    --train_n 418 --split_seed 71234 \\
    --out_dir ./experiments/tmp/pool_split1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official_train_csv", required=True)
    ap.add_argument("--official_val_csv", required=True)
    ap.add_argument("--train_n", type=int, required=True, help="从合并池中划入训练集的条数（分层）")
    ap.add_argument("--split_seed", type=int, required=True)
    ap.add_argument("--label_col", default="g_level")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(args.official_train_csv)
    va = pd.read_csv(args.official_val_csv)
    if "id" not in tr.columns or args.label_col not in tr.columns:
        raise SystemExit(f"official_train_csv 须含 id 与 {args.label_col}")
    if "id" not in va.columns or args.label_col not in va.columns:
        raise SystemExit(f"official_val_csv 须含 id 与 {args.label_col}")

    # 列对齐：以 train 列为基准，val 缺列补 NA
    for c in tr.columns:
        if c not in va.columns:
            va[c] = pd.NA
    va = va[tr.columns]
    pool = pd.concat([tr, va], ignore_index=True)
    dup = pool["id"].duplicated()
    if dup.any():
        pool = pool.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

    n = len(pool)
    if args.train_n <= 0 or args.train_n >= n:
        raise SystemExit(f"--train_n 须在 1..{n - 1} 之间（当前 pool_n={n}）")

    y = pool[args.label_col].astype(str)
    strat = y if y.value_counts().min() >= 2 else None
    if strat is None:
        print("[make_merged_pool_train_val_split] 警告: 某类过少，无法 stratify，改为随机划分")

    train_fold, val_fold = train_test_split(
        pool,
        train_size=args.train_n,
        random_state=args.split_seed,
        stratify=strat,
        shuffle=True,
    )
    train_fold = train_fold.reset_index(drop=True)
    val_fold = val_fold.reset_index(drop=True)

    tp = out_dir / "train_fold.csv"
    vp = out_dir / "val_fold.csv"
    train_fold.to_csv(tp, index=False)
    val_fold.to_csv(vp, index=False)

    manifest = {
        "split_seed": args.split_seed,
        "pool_n": n,
        "train_n": len(train_fold),
        "val_n": len(val_fold),
        "train_fold_csv": str(tp),
        "val_fold_csv": str(vp),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[make_merged_pool_train_val_split] pool={n} train={len(train_fold)} val={len(val_fold)} "
        f"seed={args.split_seed} → {out_dir}"
    )


if __name__ == "__main__":
    main()
