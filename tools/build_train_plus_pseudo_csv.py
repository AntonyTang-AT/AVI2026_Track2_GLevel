#!/usr/bin/env python3
"""
将官方 train 表与测试集 id（基本信息表）合并，并为测试 id 附上伪标签 g_level（如启发式恢复 CSV）。
训练时需同时使用：
  --train_feat_fallback --train_fallback_use_test_features
以便 train split 中测试 id 从 test_feature 回退加载特征。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_basic_csv", required=True, help="test_data_basic_information.csv")
    ap.add_argument("--pseudo_csv", required=True, help="含 id 与 g_level（或 g_level_pred）")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    tr = pd.read_csv(args.train_csv)
    te_full = pd.read_csv(args.test_basic_csv)
    pseudo = pd.read_csv(args.pseudo_csv)
    pred_col = "g_level" if "g_level" in pseudo.columns else "g_level_pred"
    if pred_col not in pseudo.columns:
        raise SystemExit(f"pseudo_csv 须含 g_level 或 g_level_pred，当前列: {list(pseudo.columns)}")
    pseudo = pseudo.rename(columns={pred_col: "g_level"})
    pseudo["g_level"] = pseudo["g_level"].astype(int)

    te = te_full.merge(pseudo[["id", "g_level"]], on="id", how="left")
    if te["g_level"].isna().any():
        bad = te.loc[te["g_level"].isna(), "id"].tolist()[:8]
        raise SystemExit(f"部分测试 id 无伪标签，示例: {bad}")
    te["g_level"] = te["g_level"].astype(int)

    for c in tr.columns:
        if c not in te.columns:
            te[c] = pd.NA
    te = te[tr.columns]

    out = pd.concat([tr, te], ignore_index=True)
    outp = Path(args.out_csv).expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False)
    print(
        f"[build_train_plus_pseudo] train={len(tr)} pseudo_rows={len(te)} total={len(out)} → {outp}",
        flush=True,
    )


if __name__ == "__main__":
    main()
