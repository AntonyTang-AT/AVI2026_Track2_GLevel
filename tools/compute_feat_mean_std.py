#!/usr/bin/env python3
"""
用训练划分 CSV 中出现的 (id, q) 在磁盘上的 .npy 计算各模态逐维 mean/std，
供 dataset 的 --feat_norm_npz + --feat_norm_apply all 使用。
勿用测试集计算统计量。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset.baseline_dataset2_vote import _resolve_from_name_lists, _list_npy_filenames  # noqa: E402


def _accum(mod: str, x: np.ndarray, state: dict) -> None:
    x = x.astype(np.float64, copy=False).ravel()
    d = x.shape[0]
    if mod not in state:
        state[mod] = {"sum": np.zeros(d), "sumsq": np.zeros(d), "n": 0}
    st = state[mod]
    if st["sum"].shape[0] != d:
        raise ValueError(f"{mod} dim mismatch: got {d}, expected {st['sum'].shape[0]}")
    st["sum"] += x
    st["sumsq"] += x * x
    st["n"] += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--text_dir", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--question", nargs=6, default=["q1", "q2", "q3", "q4", "q5", "q6"])
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)
    if "id" not in df.columns:
        raise SystemExit("train_csv 须含 id")

    la = _list_npy_filenames(args.audio_dir)
    lv = _list_npy_filenames(args.video_dir)
    lt = _list_npy_filenames(args.text_dir)
    state: dict = {}

    for _, row in df.iterrows():
        sid = str(row["id"])
        for q in args.question:
            apath = _resolve_from_name_lists(
                sid, q, args.audio_dir, None, la, None
            )
            vpath = _resolve_from_name_lists(
                sid, q, args.video_dir, None, lv, None
            )
            tpath = _resolve_from_name_lists(
                sid, q, args.text_dir, None, lt, None
            )
            if apath and os.path.isfile(apath):
                _accum("audio", np.load(apath), state)
            if vpath and os.path.isfile(vpath):
                _accum("video", np.load(vpath), state)
            if tpath and os.path.isfile(tpath):
                _accum("text", np.load(tpath), state)

    out = {}
    for mod in ("audio", "video", "text"):
        if mod not in state or state[mod]["n"] < 1:
            raise SystemExit(f"无有效 {mod} 样本，请检查路径与 CSV")
        st = state[mod]
        n = float(st["n"])
        mean = (st["sum"] / n).astype(np.float32)
        var = np.maximum(st["sumsq"] / n - (st["sum"] / n) ** 2, 0.0)
        std = np.sqrt(var).astype(np.float32)
        out[f"{mod}_mu"] = mean
        out[f"{mod}_std"] = std
        print(f"[compute_feat_mean_std] {mod} n_vecs={int(n)} dim={mean.shape[0]}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)) or ".", exist_ok=True)
    np.savez(args.out_npz, **out)
    print(f"[compute_feat_mean_std] wrote {args.out_npz}", flush=True)
    print(
        f"[metrics_line_local] method=feat_norm_stats train_csv={args.train_csv} out={args.out_npz}",
        flush=True,
    )


if __name__ == "__main__":
    main()
