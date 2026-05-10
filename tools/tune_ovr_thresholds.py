#!/usr/bin/env python3
"""
在验证集 softmax 概率上网格搜索三维 OvR 阈值，最大化 macro-F1。
输出 JSON 数组 [t0,t1,t2]，供 eval_glevel_checkpoint_on_csv.py --ovr_thresholds_json 使用。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import f1_score


def pred_ovr(probs: np.ndarray, thr: np.ndarray) -> np.ndarray:
    n, k = probs.shape
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        p = probs[i]
        above = p >= thr
        if above.any():
            idx = np.where(above)[0]
            out[i] = int(idx[np.argmax(p[idx])])
        else:
            out[i] = int(np.argmax(p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs_npz", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--grid_lo", type=float, default=0.25)
    ap.add_argument("--grid_hi", type=float, default=0.75)
    ap.add_argument("--grid_steps", type=int, default=11)
    args = ap.parse_args()

    z = np.load(args.probs_npz)
    probs = z["probs"].astype(np.float64)
    labels = z["labels"].astype(np.int64)
    grid = np.linspace(args.grid_lo, args.grid_hi, args.grid_steps)

    best_f1 = -1.0
    best_thr = np.array([0.34, 0.34, 0.34])
    for t0 in grid:
        for t1 in grid:
            for t2 in grid:
                thr = np.array([t0, t1, t2])
                pred = pred_ovr(probs, thr)
                f1 = float(f1_score(labels, pred, average="macro", zero_division=0))
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr.copy()

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump([float(x) for x in best_thr], f, indent=2)
    pred = pred_ovr(probs, best_thr)
    acc = float((pred == labels).mean())
    print(
        f"[tune_ovr_thresholds] best_macro_f1={best_f1:.4f} acc={acc:.4f} thr={best_thr.tolist()} → {args.out_json}",
        flush=True,
    )
    print(
        f"[metrics_line_local] method=ovr_thresholds macro_f1={best_f1:.6f} acc={acc:.6f} "
        f"thr={json.dumps([float(x) for x in best_thr])} out={args.out_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
