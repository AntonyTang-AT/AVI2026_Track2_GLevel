#!/usr/bin/env python3
"""
稳定性搜索用：从官方 train 划出一块进入「大验证集」，并与官方 val 的剩余部分合并；
同时将官方 val 的另一块随机划入「扩展测试 CSV」（与赛方 test_basic 纵向拼接）。

目录约定（与 train_task2_glevel 一致）：
  - 训练仅用 train_core；大验证 = train_holdout ∪ official_val_keep
  - 验证集 loader：主 val_feature，回退 train_feature（train_holdout id）
  - 扩展测试：主 FEAT_TEST，回退 train_feature，再三回退 val_feature（须 --test_fallback_val_features）

示例：
  python tools/make_stability_data_partition.py \\
    --train_pool_csv /data/Super-Lu/dataset/train_data.csv \\
    --official_val_csv /data/Super-Lu/dataset/val_data.csv \\
    --official_test_basic_csv /data/Super-Lu/dataset/test_data_basic_information.csv \\
    --partition_seed 1000 --train_holdout_n 80 --val_to_test_n 15 \\
    --out_dir experiments/tmp/part1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def _align_test_basic_columns(df_val_slice: pd.DataFrame, test_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame()
    for c in test_cols:
        if c in df_val_slice.columns:
            out[c] = df_val_slice[c]
        else:
            out[c] = pd.NA
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pool_csv", required=True)
    ap.add_argument("--official_val_csv", required=True)
    ap.add_argument("--official_test_basic_csv", required=True)
    ap.add_argument("--partition_seed", type=int, required=True)
    ap.add_argument("--train_holdout_n", type=int, required=True, help="从 train_pool 划入大验证集的条数（分层）")
    ap.add_argument("--val_to_test_n", type=int, required=True, help="从官方 val 划入扩展测试的条数（分层）")
    ap.add_argument("--label_col", default="g_level")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_pool = pd.read_csv(args.train_pool_csv)
    official_val = pd.read_csv(args.official_val_csv)
    test_basic = pd.read_csv(args.official_test_basic_csv)

    if "id" not in train_pool.columns or args.label_col not in train_pool.columns:
        raise SystemExit(f"train_pool 须含 id 与 {args.label_col}")
    if "id" not in official_val.columns or args.label_col not in official_val.columns:
        raise SystemExit(f"official_val 须含 id 与 {args.label_col}")

    te_cols = list(test_basic.columns)
    if "id" not in te_cols:
        raise SystemExit("official_test_basic_csv 须含 id")

    v_total = len(official_val)
    if args.val_to_test_n <= 0 or args.val_to_test_n >= v_total:
        raise SystemExit(f"--val_to_test_n 须在 1..{v_total - 1} 之间（当前 val 行数={v_total}）")
    if args.train_holdout_n <= 0 or args.train_holdout_n >= len(train_pool):
        raise SystemExit(f"--train_holdout_n 须在 1..{len(train_pool) - 1} 之间")

    yv = official_val[args.label_col].astype(str)
    strat_v = yv if yv.value_counts().min() >= 2 else None
    val_keep, val_to_test = train_test_split(
        official_val,
        test_size=args.val_to_test_n,
        random_state=args.partition_seed,
        stratify=strat_v,
        shuffle=True,
    )
    val_to_test = val_to_test.reset_index(drop=True)
    val_keep = val_keep.reset_index(drop=True)

    yt = train_pool[args.label_col].astype(str)
    strat_t = yt if yt.value_counts().min() >= 2 else None
    train_core, train_holdout = train_test_split(
        train_pool,
        test_size=args.train_holdout_n,
        random_state=args.partition_seed,
        stratify=strat_t,
        shuffle=True,
    )
    train_core = train_core.reset_index(drop=True)
    train_holdout = train_holdout.reset_index(drop=True)

    val_merged = pd.concat([train_holdout, val_keep], ignore_index=True)
    # id 重复则报错（正常不应发生）
    if val_merged["id"].duplicated().any():
        raise SystemExit("val_merged 出现重复 id，请检查输入表")

    val_as_test_rows = _align_test_basic_columns(val_to_test, te_cols)
    test_merged = pd.concat([test_basic, val_as_test_rows], ignore_index=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val_merged.csv"
    test_path = out_dir / "test_merged.csv"
    train_core.to_csv(train_path, index=False)
    val_merged.to_csv(val_path, index=False)
    test_merged.to_csv(test_path, index=False)

    manifest = {
        "partition_seed": args.partition_seed,
        "train_pool_n": len(train_pool),
        "train_core_n": len(train_core),
        "train_holdout_n": len(train_holdout),
        "official_val_n": len(official_val),
        "val_keep_n": len(val_keep),
        "val_to_test_n": len(val_to_test),
        "val_merged_n": len(val_merged),
        "test_basic_n": len(test_basic),
        "test_merged_n": len(test_merged),
        "paths": {
            "train_csv": str(train_path),
            "val_merged_csv": str(val_path),
            "test_merged_csv": str(test_path),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[make_stability_data_partition] seed={args.partition_seed} → "
        f"train_core={len(train_core)} val_merged={len(val_merged)} test_merged={len(test_merged)} → {out_dir}"
    )


if __name__ == "__main__":
    main()
