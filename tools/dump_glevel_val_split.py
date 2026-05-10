#!/usr/bin/env python3
"""
与训练一致地加载 val CSV + 特征目录，输出过滤后的验证集 id 列表与被剔除 id。
用于自查「两次实验 val 是否同一批样本」。

示例：
  python3 tools/dump_glevel_val_split.py \\
    --val_csv /data/Super-Lu/dataset/val_data.csv \\
    --rating_csv /data/Super-Lu/dataset/train_data.csv \\
    --val_audio_dir /data/Super-Lu/dataset/val_feature/audio \\
    --val_video_dir /data/Super-Lu/dataset/val_feature/video \\
    --val_text_dir /data/Super-Lu/dataset/val_feature/text \\
    --out_txt ./experiments/val_ids_dump.txt
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset.baseline_dataset2_vote import _drop_rows_missing_features, _list_npy_filenames  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--rating_csv", required=True)
    ap.add_argument("--val_audio_dir", required=True)
    ap.add_argument("--val_video_dir", required=True)
    ap.add_argument("--val_text_dir", required=True)
    ap.add_argument("--fallback_audio_dir", default="")
    ap.add_argument("--fallback_video_dir", default="")
    ap.add_argument("--fallback_text_dir", default="")
    ap.add_argument("--question", nargs=6, default=["q1", "q2", "q3", "q4", "q5", "q6"])
    ap.add_argument("--out_txt", default="", help="非空则写入该路径")
    args = ap.parse_args()

    df = pd.read_csv(args.val_csv)
    n0 = len(df)
    la = _list_npy_filenames(args.val_audio_dir)
    lab = _list_npy_filenames(args.fallback_audio_dir) if args.fallback_audio_dir else None
    lv = _list_npy_filenames(args.val_video_dir)
    lvb = _list_npy_filenames(args.fallback_video_dir) if args.fallback_video_dir else None
    lt = _list_npy_filenames(args.val_text_dir)
    ltb = _list_npy_filenames(args.fallback_text_dir) if args.fallback_text_dir else None

    kept = _drop_rows_missing_features(
        df,
        list(args.question),
        args.val_audio_dir,
        args.val_video_dir,
        args.val_text_dir,
        args.fallback_audio_dir or None,
        args.fallback_video_dir or None,
        args.fallback_text_dir or None,
        "dump_val",
        la,
        lab,
        lv,
        lvb,
        lt,
        ltb,
        allow_empty=True,
    )
    kept_ids = [str(x).strip() for x in kept["id"].tolist()]
    all_ids = [str(x).strip() for x in df["id"].tolist()]
    dropped = sorted(set(all_ids) - set(kept_ids))

    lines = [
        f"val_csv={os.path.abspath(args.val_csv)}",
        f"val_csv_sha256={_file_sha256(args.val_csv)}",
        f"rows_csv={n0} rows_after_filter={len(kept_ids)} dropped_count={len(dropped)}",
        f"dropped_ids={dropped}",
        "kept_ids_sorted=" + ",".join(sorted(kept_ids)),
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out_txt:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_txt)) or ".", exist_ok=True)
        with open(args.out_txt, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {args.out_txt}", file=sys.stderr)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
