#!/usr/bin/env python3
"""
从「模型训练表」train CSV 中分层抽取 N 条，写入「仅 DeepSeek 标注管线」使用的 CSV：

1) train_for_deepseek_minus_holdout.csv — train 去掉这 N 条 → **标注用训练条数 = |train_csv| − N**
2) val_for_deepseek_plus_holdout.csv — **官方 val 全量 + 这 N 条**（交互式逐条纠错）

不修改官方 /data/Super-Lu/dataset/*.csv，也不影响 vote_train 默认 TRAIN_CSV。

默认 --train-csv 为官方 train_data.csv（当前约 **450** 条，故 N=200 时标注训练约 **250** 条）。
若你实际建模用的是更大的合并表（例如伪标签扩充 train），请传入该 CSV，则标注训练条数为「该表行数 − N」。

示例：
  python tools/build_deepseek_val_augment_from_train.py \\
    --out-dir external/deepseek_calib_valaug200 \\
    --n-from-train 200 --seed 4242

  # 以伪标签合并训练表为基准（示例路径）：
  # python tools/build_deepseek_val_augment_from_train.py \\
  #   --train-csv experiments/glevel_improvement_plan/train_plus_pseudo_unanimous.csv \\
  #   --out-dir external/deepseek_calib_valaug200_pseudo ...
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

_REPO = Path(__file__).resolve().parents[1]
_SUP_DATASET = Path(os.environ.get("SUPERLU_DATASET", "/data/Super-Lu/dataset"))


def _default_csv(fname: str) -> Path:
    loc = _REPO / "data" / fname
    return loc if loc.is_file() else (_SUP_DATASET / fname)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, default=_default_csv("train_data.csv"))
    ap.add_argument("--val-csv", type=Path, default=_default_csv("val_data.csv"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-from-train", type=int, default=200)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--label-col", type=str, default="g_level")
    args = ap.parse_args()

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    if "id" not in train_df.columns or args.label_col not in train_df.columns:
        raise SystemExit("train CSV 需要列 id 与标签列")
    if train_df["id"].duplicated().any():
        raise SystemExit("train id 存在重复")

    n = int(args.n_from_train)
    if n <= 0 or n >= len(train_df):
        raise SystemExit(f"n-from-train 须在 1..{len(train_df)-1} 之间")

    y = train_df[args.label_col].fillna(-1).astype(int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n, random_state=args.seed)
    _, hold_idx = next(sss.split(np.zeros(len(train_df)), y))
    hold_idx = np.sort(hold_idx)
    hold_rows = train_df.iloc[hold_idx].copy()
    train_minus = train_df.drop(train_df.index[hold_idx]).reset_index(drop=True)
    val_plus = pd.concat([val_df, hold_rows], ignore_index=True)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    p_train = out_dir / "train_for_deepseek_minus_holdout.csv"
    p_val = out_dir / "val_for_deepseek_plus_holdout.csv"
    p_manifest = out_dir / "manifest_val_holdout.json"

    train_minus.to_csv(p_train, index=False)
    val_plus.to_csv(p_val, index=False)

    vc_hold = hold_rows[args.label_col].value_counts().sort_index()
    manifest = {
        "train_csv_source": str(args.train_csv.resolve()),
        "val_csv_source": str(args.val_csv.resolve()),
        "base_train_labeled_rows": int(len(train_df)),
        "official_val_labeled_rows": int(len(val_df)),
        "n_from_train": n,
        "seed": args.seed,
        "label_col": args.label_col,
        "annotation_train_rows_expected": int(len(train_minus)),
        "annotation_val_rows_expected": int(len(val_plus)),
        "formula_zh": "标注用 train 行数 = base_train_labeled_rows − n_from_train；标注用 val = 官方 val + n_from_train",
        "holdout_id_count": int(len(hold_rows)),
        "holdout_g_level_counts": {int(k): int(v) for k, v in vc_hold.items()},
        "train_minus_rows": int(len(train_minus)),
        "val_plus_rows": int(len(val_plus)),
        "holdout_ids": [str(x).strip() for x in hold_rows["id"].tolist()],
        "outputs": {"train_minus_csv": str(p_train), "val_plus_csv": str(p_val)},
        "annotate_deepseek_suggested_flags": (
            f"--train-min-examples {len(train_minus)} --train-max-examples {len(train_minus)}"
        ),
    }
    p_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[valaug] wrote train_minus n={len(train_minus)} → {p_train}")
    print(f"[valaug] wrote val_plus n={len(val_plus)} → {p_val}")
    print(f"[valaug] manifest → {p_manifest}")
    print(f"[valaug] holdout label counts: {dict(vc_hold)}")


if __name__ == "__main__":
    main()
