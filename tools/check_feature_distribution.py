#!/usr/bin/env python3
"""
对比 train / val / test 各模态特征：按样本对 6 题取 mean 得到 (N,D)，再 PCA-2D 散点图 + RBF-MMD。
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset.baseline_dataset2_vote import _list_npy_filenames, _resolve_from_name_lists  # noqa: E402


def _sample_vecs(
    csv_path: str,
    audio_dir: str,
    video_dir: str,
    text_dir: str,
    questions: list[str],
    max_ids: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    ids = df["id"].astype(str).tolist()
    if max_ids > 0 and len(ids) > max_ids:
        rng = np.random.default_rng(42)
        ids = list(rng.choice(ids, size=max_ids, replace=False))

    la, lv, lt = (
        _list_npy_filenames(audio_dir),
        _list_npy_filenames(video_dir),
        _list_npy_filenames(text_dir),
    )
    av, vv, tv = [], [], []
    for sid in ids:
        ar, vr, tr = [], [], []
        ok = True
        for q in questions:
            ap = _resolve_from_name_lists(sid, q, audio_dir, None, la, None)
            vp = _resolve_from_name_lists(sid, q, video_dir, None, lv, None)
            tp = _resolve_from_name_lists(sid, q, text_dir, None, lt, None)
            if not (ap and vp and tp):
                ok = False
                break
            ar.append(np.load(ap).astype(np.float32))
            vr.append(np.load(vp).astype(np.float32))
            tr.append(np.load(tp).astype(np.float32))
        if not ok:
            continue
        av.append(np.stack(ar, axis=0).mean(axis=0))
        vv.append(np.stack(vr, axis=0).mean(axis=0))
        tv.append(np.stack(tr, axis=0).mean(axis=0))
    if not av:
        return (
            np.zeros((0, 1)),
            np.zeros((0, 1)),
            np.zeros((0, 1)),
        )
    return np.stack(av), np.stack(vv), np.stack(tv)


def _mmd_rbf(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")

    def k(a, b):
        aa = np.sum(a * a, axis=1, keepdims=True)
        bb = np.sum(b * b, axis=1, keepdims=True).T
        ab = a @ b.T
        d2 = np.maximum(aa + bb - 2.0 * ab, 0.0)
        return np.exp(-d2 / (2.0 * sigma * sigma))

    kxx = k(x, x)
    kyy = k(y, y)
    kxy = k(x, y)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def _plot_pca_pair(a: np.ndarray, b: np.ndarray, la: str, lb: str, out_png: str, title: str) -> None:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return
    x = np.vstack([a, b])
    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(x)
    za, zb = z[: len(a)], z[len(a) :]
    plt.figure(figsize=(6, 5))
    plt.scatter(za[:, 0], za[:, 1], s=8, alpha=0.5, label=la)
    plt.scatter(zb[:, 0], zb[:, 1], s=8, alpha=0.5, label=lb)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"[check_feature_distribution] saved {out_png}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--train_audio_dir", required=True)
    ap.add_argument("--train_video_dir", required=True)
    ap.add_argument("--train_text_dir", required=True)
    ap.add_argument("--val_audio_dir", default="")
    ap.add_argument("--val_video_dir", default="")
    ap.add_argument("--val_text_dir", default="")
    ap.add_argument("--test_audio_dir", default="")
    ap.add_argument("--test_video_dir", default="")
    ap.add_argument("--test_text_dir", default="")
    ap.add_argument("--out_dir", default="./experiments/feat_dist")
    ap.add_argument("--max_ids_per_split", type=int, default=400)
    ap.add_argument("--question", nargs=6, default=["q1", "q2", "q3", "q4", "q5", "q6"])
    ap.add_argument("--tsne", action="store_true", help="较慢，默认仅 PCA")
    args = ap.parse_args()

    va_a = args.val_audio_dir or args.train_audio_dir
    va_v = args.val_video_dir or args.train_video_dir
    va_t = args.val_text_dir or args.train_text_dir
    te_a = args.test_audio_dir or args.train_audio_dir
    te_v = args.test_video_dir or args.train_video_dir
    te_t = args.test_text_dir or args.train_text_dir

    splits = {
        "train": (args.train_csv, args.train_audio_dir, args.train_video_dir, args.train_text_dir),
        "val": (args.val_csv, va_a, va_v, va_t),
        "test": (args.test_csv, te_a, te_v, te_t),
    }
    feats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, (csv, ad, vd, td) in splits.items():
        feats[name] = _sample_vecs(csv, ad, vd, td, list(args.question), args.max_ids_per_split)

    os.makedirs(args.out_dir, exist_ok=True)
    mmd_parts: list[str] = []
    for mod, idx in ("audio", 0), ("video", 1), ("text", 2):
        tr = feats["train"][idx]
        va = feats["val"][idx]
        te = feats["test"][idx]
        if tr.size == 0:
            print(f"[check_feature_distribution] skip {mod}: train empty", flush=True)
            continue
        print(
            f"[check_feature_distribution] {mod} shapes train={tr.shape} val={va.shape} test={te.shape}",
            flush=True,
        )
        sigma = float(np.median(np.linalg.norm(tr - tr[:1], axis=1)) or 1.0)
        mmd_tv = float("nan")
        mmd_tt = float("nan")
        if va.shape[0] > 1:
            mmd_tv = _mmd_rbf(tr, va, sigma)
            print(f"[check_feature_distribution] MMD_rbf train vs val ({mod})={mmd_tv:.6f}", flush=True)
        if te.shape[0] > 1:
            mmd_tt = _mmd_rbf(tr, te, sigma)
            print(f"[check_feature_distribution] MMD_rbf train vs test ({mod})={mmd_tt:.6f}", flush=True)
        mmd_parts.append(f"{mod}_mmd_tv={mmd_tv:.6g}")
        mmd_parts.append(f"{mod}_mmd_tt={mmd_tt:.6g}")
        _plot_pca_pair(
            tr,
            te,
            "train",
            "test",
            os.path.join(args.out_dir, f"pca_{mod}_train_vs_test.png"),
            f"PCA {mod} train vs test",
        )

    if mmd_parts:
        print(
            "[metrics_line_local] method=feat_dist out_dir="
            + args.out_dir
            + " "
            + " ".join(mmd_parts),
            flush=True,
        )

    if args.tsne:
        print(
            "[check_feature_distribution] --tsne 未实现完整管线（样本多时建议手动子采样）",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
