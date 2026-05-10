#!/usr/bin/env python3
"""
从「含被试 id + 标签列」的源表，导出 train+val 划分里出现的样本，生成 glevel_train.csv。

示例：
  python tools/build_glevel_train_csv.py \\
    --source /data/AVI2026/labels/participants.csv \\
    --id_col id \\
    --label_col g_level \\
    --train_csv ./data/train_data.csv \\
    --val_csv ./data/val_data_new.csv \\
    --out ./data/glevel_train.csv

支持 .csv；若为 .xlsx / .xls 需: pip install openpyxl
"""
import argparse
import os

import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(path)
        except ImportError as e:
            raise SystemExit("读取 Excel 需要: pip install openpyxl") from e
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser(description="生成 glevel_train.csv（供 --glevel_csv 使用）")
    ap.add_argument("--source", required=True, help="含 id 与标签列的 CSV 或 Excel")
    ap.add_argument("--id_col", default="id", help="源表中被试 id 列名")
    ap.add_argument("--label_col", default="g_level", help="源表中标签列名（输出统一为 g_level）")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--out", default="./data/glevel_train.csv")
    args = ap.parse_args()

    src = read_table(args.source)
    if args.id_col not in src.columns:
        raise SystemExit(f"源表须含列 {args.id_col!r}，当前为: {list(src.columns)}")
    if args.label_col not in src.columns:
        raise SystemExit(f"源表须含列 {args.label_col!r}，当前为: {list(src.columns)}")

    tr = pd.read_csv(args.train_csv)
    va = pd.read_csv(args.val_csv)
    if "id" not in tr.columns or "id" not in va.columns:
        raise SystemExit("train_csv / val_csv 须含列 id")

    need = set(tr["id"].astype(str)) | set(va["id"].astype(str))
    src = src.copy()
    src["_sid"] = src[args.id_col].astype(str)
    sub = src[src["_sid"].isin(need)].copy()
    sub = sub.drop_duplicates(subset=["_sid"], keep="last")
    out_df = pd.DataFrame(
        {
            "id": sub[args.id_col].values,
            "g_level": sub[args.label_col].values,
        }
    )

    missing = need - set(out_df["id"].astype(str))
    if missing:
        print(f"警告: {len(missing)} 个 train/val 中的 id 在源表中无标签")
        print("示例 id:", list(sorted(missing))[:5])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"已写入 {os.path.abspath(args.out)} ，共 {len(out_df)} 行（列: id, g_level）")


if __name__ == "__main__":
    main()
