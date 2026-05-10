#!/usr/bin/env python3
"""
按与 dataset/baseline_dataset2_vote.py 相同的规则，检查 CSV 中每个 id 在
primary / fallback 目录下是否具备 6 题 × audio|video|text 的 .npy。

用于定位验证集被剔除行（如缺 1 条导致 val 从 64→63），便于补提特征。

示例（与 vote_train 路径一致）：
  python tools/report_missing_features_for_csv.py \\
    --csv ./data/val_data_new.csv \\
    --audio_dir "$FEAT_VAL/audio" --video_dir "$FEAT_VAL/video" --text_dir "$TEXT_VAL_DIR" \\
    --fallback_audio_dir "$FEAT_TRAIN/audio" --fallback_video_dir "$FEAT_TRAIN/video" \\
    --fallback_text_dir "$TEXT_TRAIN_DIR"
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def _list_npy_filenames(dir_path: str) -> list[str]:
    if not dir_path or not os.path.isdir(dir_path):
        return []
    return [fn for fn in os.listdir(dir_path) if fn.lower().endswith(".npy")]


def _pick_npy_path(base_dir: str, filenames: list[str], sample_id, q: str) -> str | None:
    if not base_dir or not filenames:
        return None
    sid = str(sample_id).strip()
    prefix = f"{sid}_{q}"
    hits = [fn for fn in filenames if fn.startswith(prefix) and fn.lower().endswith(".npy")]
    if not hits:
        return None
    return os.path.join(base_dir, sorted(hits)[0])


def _resolve(
    sample_id,
    q: str,
    primary_dir: str,
    fallback_dir: str | None,
    names_p: list[str],
    names_f: list[str] | None,
) -> str | None:
    p = _pick_npy_path(primary_dir, names_p, sample_id, q)
    if p is not None:
        return p
    if fallback_dir and names_f is not None:
        return _pick_npy_path(fallback_dir, names_f, sample_id, q)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="报告 CSV 中 id 缺失的题×模态 .npy")
    ap.add_argument("--csv", required=True, help="含 id 列的划分表（如 val_data_new.csv）")
    ap.add_argument("--id_col", default="id")
    ap.add_argument(
        "--question",
        nargs="+",
        default=["q1", "q2", "q3", "q4", "q5", "q6"],
    )
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--text_dir", required=True)
    ap.add_argument("--fallback_audio_dir", default="")
    ap.add_argument("--fallback_video_dir", default="")
    ap.add_argument("--fallback_text_dir", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.id_col not in df.columns:
        print(f"CSV 无列 {args.id_col!r}，列为 {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    ad, vd, td = args.audio_dir, args.video_dir, args.text_dir
    fad, fvd, ftd = (
        (args.fallback_audio_dir or "").strip() or None,
        (args.fallback_video_dir or "").strip() or None,
        (args.fallback_text_dir or "").strip() or None,
    )

    la = _list_npy_filenames(ad)
    lv = _list_npy_filenames(vd)
    lt = _list_npy_filenames(td)
    lab = _list_npy_filenames(fad) if fad else None
    lvb = _list_npy_filenames(fvd) if fvd else None
    ltb = _list_npy_filenames(ftd) if ftd else None

    n = len(df)
    incomplete = 0
    for _, row in df.iterrows():
        sid = row[args.id_col]
        missing: list[str] = []
        for q in args.question:
            if _resolve(sid, q, ad, fad, la, lab or []) is None:
                missing.append(f"{q}/audio")
            if _resolve(sid, q, vd, fvd, lv, lvb or []) is None:
                missing.append(f"{q}/video")
            if _resolve(sid, q, td, ftd, lt, ltb or []) is None:
                missing.append(f"{q}/text")
        if missing:
            incomplete += 1
            print(f"id={str(sid).strip()}\tmissing={', '.join(missing)}")

    print(
        f"\n[summary] csv={os.path.abspath(args.csv)} rows={n} "
        f"incomplete={incomplete} complete={n - incomplete}",
        file=sys.stderr,
    )
    if incomplete:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
