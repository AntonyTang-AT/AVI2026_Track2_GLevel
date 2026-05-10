#!/usr/bin/env python3
"""
Phase C：将官方 train CSV 与「测试集基本信息 + 共识伪标签」行合并，供半监督式微调。

- 测试行须能通过 train 主目录或 --train_feat_fallback + --train_fallback_use_test_features
  在 FEAT_TEST / test_nb 下找到特征（见 vote_train 与 train_task2_glevel）。
- 默认仅合并 is_unanimous==1（四票一致）；可用 --min-trust 放宽。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import warnings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--test-basic-csv", type=Path, required=True)
    ap.add_argument("--pseudo-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--require-unanimous",
        action="store_true",
        help="仅保留 pseudo 中 is_unanimous=1",
    )
    ap.add_argument("--min-trust", type=float, default=0.0)
    args = ap.parse_args()

    train = pd.read_csv(args.train_csv)
    test_basic = pd.read_csv(args.test_basic_csv)
    pseudo = pd.read_csv(args.pseudo_csv)

    if args.require_unanimous:
        pseudo = pseudo[pseudo["is_unanimous"].astype(int) == 1]
    if args.min_trust > 0:
        pseudo = pseudo[pseudo["trust_weight"].astype(float) >= args.min_trust]

    pseudo_lab = pseudo[["id", "g_level"]].copy()
    merged_test = test_basic.merge(pseudo_lab, on="id", how="inner")

    for c in train.columns:
        if c not in merged_test.columns:
            merged_test[c] = pd.NA

    merged_test = merged_test[train.columns]
    # 与新行 concat 时统一 dtypes，避免空列触发 pandas FutureWarning
    for col in train.columns:
        if col in merged_test.columns:
            merged_test[col] = merged_test[col].astype(train[col].dtype, errors="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        out_df = pd.concat([train, merged_test], ignore_index=True)
    out_df = out_df.drop_duplicates(subset=["id"], keep="first")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(
        f"[merge_pseudo_train] wrote {args.out} | train_rows={len(train)} "
        f"pseudo_test_rows={len(merged_test)} | out_rows={len(out_df)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
