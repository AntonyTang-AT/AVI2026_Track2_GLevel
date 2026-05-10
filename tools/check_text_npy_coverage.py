#!/usr/bin/env python3
"""
检查某 CSV 中全部 id 是否在 text 目录（及可选 fallback）下具备 6 题 Nanbeige/SigLIP 文本 .npy。
命名规则与 dataset 一致：前缀 ``{id}_{q}`` 的 .npy。

用于 K 折 / 训练前确认 TEXT_VAL_DIR、TEXT_TEST_DIR 已含对应划分 id（避免 test 全被剔除）。

示例：
  python tools/check_text_npy_coverage.py \\
    --csv "$TEST_CSV" \\
    --text_dir "$TEXT_TEST_DIR" \\
    --fallback_text_dir "$TEXT_TRAIN_DIR"
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def _list_npy(dir_path: str) -> list[str]:
    if not dir_path or not os.path.isdir(dir_path):
        return []
    return [fn for fn in os.listdir(dir_path) if fn.lower().endswith(".npy")]


def _pick(base_dir: str, fnames: list[str], sample_id, q: str) -> str | None:
    if not base_dir or not fnames:
        return None
    sid = str(sample_id).strip()
    prefix = f"{sid}_{q}"
    hits = [fn for fn in fnames if fn.startswith(prefix) and fn.lower().endswith(".npy")]
    if not hits:
        return None
    return os.path.join(base_dir, sorted(hits)[0])


def _resolve(sid, q: str, primary: str, fb: str | None, lp: list[str], lf: list[str] | None):
    p = _pick(primary, lp, sid, q)
    if p is not None:
        return p
    if fb and lf is not None:
        return _pick(fb, lf, sid, q)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="CSV id × 6 题 文本 .npy 是否齐全")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--id_col", default="id")
    ap.add_argument(
        "--question",
        nargs="+",
        default=["q1", "q2", "q3", "q4", "q5", "q6"],
    )
    ap.add_argument("--text_dir", required=True, help="主目录（如 TEXT_TEST_DIR）")
    ap.add_argument(
        "--fallback_text_dir",
        default="",
        help="回退目录（如 TEXT_TRAIN_DIR），可为空",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.id_col not in df.columns:
        print(f"CSV 无列 {args.id_col!r}", file=sys.stderr)
        sys.exit(1)

    td = os.path.abspath(args.text_dir)
    fb = (args.fallback_text_dir or "").strip() or None
    if fb:
        fb = os.path.abspath(fb)

    lt = _list_npy(td)
    lfb = _list_npy(fb) if fb else None

    n = len(df)
    bad_ids: list[str] = []
    for _, row in df.iterrows():
        sid = row[args.id_col]
        ok = True
        for q in args.question:
            if _resolve(sid, q, td, fb, lt, lfb or []) is None:
                ok = False
                break
        if not ok:
            bad_ids.append(str(sid).strip())

    print(f"csv={os.path.abspath(args.csv)} rows={n}")
    print(f"text_dir={td} .npy数={len(lt)}")
    if fb:
        print(f"fallback_text_dir={fb} .npy数={len(lfb or [])}")
    complete = n - len(bad_ids)
    print(f"文本特征完整: {complete}/{n}")
    if bad_ids:
        prev = ", ".join(bad_ids[:15])
        more = f" 等共 {len(bad_ids)} 个 id" if len(bad_ids) > 15 else ""
        print(f"缺失示例 id: {prev}{more}", file=sys.stderr)
        sys.exit(2)
    print("OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
