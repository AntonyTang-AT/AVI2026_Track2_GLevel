#!/usr/bin/env python3
"""
为传统 ML（SVM 等）准备固定长度特征：每样本对 6 题在时间上取 mean，再拼接
  [video_mean, audio_mean, text_mean, hand_mean]
只读 CSV 与 .npy，不写 /data。输出默认 data/svm/{X,y}_{train,val}.npy

用法示例（与 vote_train 路径一致）：
  python tools/prepare_svm_data.py \\
    --train_csv /data/Super-Lu/dataset/train_data.csv \\
    --val_csv /data/Super-Lu/dataset/val_data.csv \\
    --rating_csv /data/Super-Lu/dataset/train_data.csv \\
    --audio_dir .../train_feature/audio --video_dir .../video --text_dir .../text_nb \\
    --val_audio_dir .../val_feature/audio --val_video_dir ... --val_text_dir ... \\
    --video_dim 1152 --audio_dim 768 --text_dim 2560 \\
    --g_level_int_encoding one --out_dir ./data/svm
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# 项目根
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset.baseline_dataset2_vote import (  # noqa: E402
    _list_npy_filenames,
    _resolve_from_name_lists,
    _row_has_all_features_cached,
    _wav_duration_sec,
    compute_hand_feats_row,
    encode_g_level,
)


def _resolve_raw_label(sid: str, row: pd.Series, id_to_row: dict, label_col: str):
    if label_col in row.index and pd.notna(row.get(label_col)):
        return row[label_col]
    if sid in id_to_row:
        r = id_to_row[sid]
        if label_col in r.index and pd.notna(r.get(label_col)):
            return r[label_col]
    raise KeyError(f"无法解析 id={sid!r} 的标签列 {label_col!r}（划分 CSV 与 rating_csv 均无有效值）")


def _build_arrays(
    df: pd.DataFrame,
    id_to_row: dict,
    question: list[str],
    audio_dir: str,
    video_dir: str,
    text_dir: str,
    fa: str | None,
    fv: str | None,
    ft: str | None,
    la,
    lab,
    lv,
    lvb,
    lt,
    ltb,
    video_dim: int,
    audio_dim: int,
    text_dim: int,
    hand_dim: int,
    label_col: str,
    int_encoding: str,
    transcript_dir: str | None,
    wav_dir: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for _, row in df.iterrows():
        sid = str(row["id"]).strip()
        raw_y = _resolve_raw_label(sid, row, id_to_row, label_col)
        y = encode_g_level(raw_y, int_encoding=int_encoding)

        audio_paths = []
        video_paths = []
        text_paths = []
        for q in question:
            ap = _resolve_from_name_lists(sid, q, audio_dir, fa, la, lab)
            vp = _resolve_from_name_lists(sid, q, video_dir, fv, lv, lvb)
            tp = _resolve_from_name_lists(sid, q, text_dir, ft, lt, ltb)
            if ap is None or vp is None or tp is None:
                raise RuntimeError(f"内部错误: {sid} 应在过滤后完整")
            audio_paths.append(ap)
            video_paths.append(vp)
            text_paths.append(tp)

        va = np.stack([np.load(p) for p in video_paths], axis=0).mean(axis=0)
        aa = np.stack([np.load(p) for p in audio_paths], axis=0).mean(axis=0)
        ta = np.stack([np.load(p) for p in text_paths], axis=0).mean(axis=0)

        if va.shape != (video_dim,) or aa.shape != (audio_dim,) or ta.shape != (text_dim,):
            raise ValueError(
                f"id={sid} 聚合后维数不符: video{va.shape} audio{aa.shape} text{ta.shape} "
                f"期望 ({video_dim},) ({audio_dim},) ({text_dim},)"
            )

        hand_rows = []
        for ap, vp, tp in zip(audio_paths, video_paths, text_paths):
            af = os.path.basename(ap)
            vf = os.path.basename(vp)
            tf = os.path.basename(tp)
            transcript = ""
            duration = 1.0
            if transcript_dir:
                base = os.path.splitext(tf)[0]
                tpath = os.path.join(transcript_dir, base + ".txt")
                if os.path.isfile(tpath):
                    with open(tpath, "r", encoding="utf-8", errors="ignore") as fp:
                        transcript = fp.read()
            if wav_dir:
                base = os.path.splitext(af)[0]
                wp = os.path.join(wav_dir, base + ".wav")
                if not os.path.isfile(wp):
                    wp = os.path.join(wav_dir, base + ".WAV")
                duration = _wav_duration_sec(wp) if os.path.isfile(wp) else 1.0
            hand_rows.append(compute_hand_feats_row(transcript, duration))
        hm = np.stack(hand_rows, axis=0).mean(axis=0)
        if hm.shape != (hand_dim,):
            raise ValueError(f"hand dim {hm.shape} != ({hand_dim},)")

        feat = np.concatenate([va, aa, ta, hm], axis=0).astype(np.float32)
        X_list.append(feat)
        y_list.append(y)

    if not X_list:
        raise ValueError(
            "_build_arrays: 无样本（上游应在过滤后检查 train/val 是否为空）"
        )
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def _filter_df(df, question, ad, vd, td, fa, fv, ft, la, lab, lv, lvb, lt, ltb, name):
    keep = []
    for _, row in df.iterrows():
        sid = row["id"]
        ok = _row_has_all_features_cached(
            sid, question, ad, vd, td, fa, fv, ft, la, lab, lv, lvb, lt, ltb
        )
        keep.append(ok)
    out = df.loc[keep].reset_index(drop=True)
    n0, n1 = len(df), len(out)
    if n0 != n1:
        print(f"[prepare_svm_data] {name}: {n1}/{n0} 行（已剔除缺特征）", file=sys.stderr)
    return out


def _diagnose_all_filtered(
    split_label: str,
    raw_df: pd.DataFrame,
    question: list[str],
    audio_dir: str,
    video_dir: str,
    text_dir: str,
    fa,
    fv,
    ft,
    la,
    lab,
    lv,
    lvb,
    lt,
    ltb,
) -> None:
    """过滤后无样本时打印目录与首条 id 各题解析情况，便于排查路径/特征未提取。"""
    print(
        f"\n[prepare_svm_data] 错误: {split_label} 过滤后 0 条样本（需每题 audio/video/text 均有 .npy）。",
        file=sys.stderr,
    )
    print(
        f"  audio:  存在={bool(audio_dir and os.path.isdir(audio_dir))} .npy数={len(la)} → {audio_dir!r}",
        file=sys.stderr,
    )
    if fa:
        print(
            f"  audio_fb: 存在={bool(os.path.isdir(fa))} .npy数={len(lab)} → {fa!r}",
            file=sys.stderr,
        )
    print(
        f"  video:  存在={bool(video_dir and os.path.isdir(video_dir))} .npy数={len(lv)} → {video_dir!r}",
        file=sys.stderr,
    )
    print(
        f"  text:   存在={bool(text_dir and os.path.isdir(text_dir))} .npy数={len(lt)} → {text_dir!r}",
        file=sys.stderr,
    )
    if ft:
        print(
            f"  text_fb: 存在={bool(os.path.isdir(ft))} .npy数={len(ltb)} → {ft!r}",
            file=sys.stderr,
        )
    if len(lt) == 0 and text_dir:
        print(
            "[prepare_svm_data] 提示: text 目录下无任何 .npy。Nanbeige 常见子目录为 train_feature/text_nb；"
            "若验证集尚未提取，可暂时 export TEXT_VAL_DIR=训练集 text_nb 目录（须含 val 的 id）。",
            file=sys.stderr,
        )
    if len(raw_df) == 0:
        return
    sid = str(raw_df.iloc[0]["id"]).strip()
    print(f"  首条 CSV id={sid!r} 各题解析:", file=sys.stderr)
    for q in question:
        ap = _resolve_from_name_lists(sid, q, audio_dir, fa, la, lab)
        vp = _resolve_from_name_lists(sid, q, video_dir, fv, lv, lvb)
        tp = _resolve_from_name_lists(sid, q, text_dir, ft, lt, ltb)
        print(
            f"    {q}: audio={'OK' if ap else 'MISS'} video={'OK' if vp else 'MISS'} text={'OK' if tp else 'MISS'}",
            file=sys.stderr,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--rating_csv", required=True)
    ap.add_argument("--label_col", default="g_level")
    ap.add_argument(
        "--g_level_int_encoding",
        default="zero",
        choices=("zero", "one"),
        help="与 train_task2_glevel 一致",
    )
    ap.add_argument("--question", nargs=6, default=["q1", "q2", "q3", "q4", "q5", "q6"])
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--text_dir", required=True)
    ap.add_argument("--val_audio_dir", required=True)
    ap.add_argument("--val_video_dir", required=True)
    ap.add_argument("--val_text_dir", required=True)
    ap.add_argument("--fallback_audio_dir", default="")
    ap.add_argument("--fallback_video_dir", default="")
    ap.add_argument("--fallback_text_dir", default="")
    ap.add_argument("--video_dim", type=int, required=True)
    ap.add_argument("--audio_dim", type=int, required=True)
    ap.add_argument("--text_dim", type=int, required=True)
    ap.add_argument("--hand_dim", type=int, default=4)
    ap.add_argument("--transcript_dir", default="", help="转写 .txt 目录（可选，与 Dataset 一致）")
    ap.add_argument("--wav_dir", default="", help="原始 wav 目录（可选，用于 hand 时长）")
    ap.add_argument("--out_dir", default=os.path.join(_ROOT, "data", "svm"))
    args = ap.parse_args()

    fa = args.fallback_audio_dir or None
    fv = args.fallback_video_dir or None
    ft = args.fallback_text_dir or None
    transcript_dir = args.transcript_dir or None
    wav_dir = args.wav_dir or None
    if transcript_dir == "":
        transcript_dir = None
    if wav_dir == "":
        wav_dir = None

    question = list(args.question)
    rating = pd.read_csv(args.rating_csv)
    train_df_raw = pd.read_csv(args.train_csv)
    val_df_raw = pd.read_csv(args.val_csv)

    la = _list_npy_filenames(args.audio_dir)
    lab = _list_npy_filenames(fa) if fa else []
    lv = _list_npy_filenames(args.video_dir)
    lvb = _list_npy_filenames(fv) if fv else []
    lt = _list_npy_filenames(args.text_dir)
    ltb = _list_npy_filenames(ft) if ft else []

    val_la = _list_npy_filenames(args.val_audio_dir)
    val_lab = _list_npy_filenames(fa) if fa else []
    val_lv = _list_npy_filenames(args.val_video_dir)
    val_lvb = _list_npy_filenames(fv) if fv else []
    val_lt = _list_npy_filenames(args.val_text_dir)
    val_ltb = _list_npy_filenames(ft) if ft else []

    train_df = _filter_df(
        train_df_raw,
        question,
        args.audio_dir,
        args.video_dir,
        args.text_dir,
        fa,
        fv,
        ft,
        la,
        lab,
        lv,
        lvb,
        lt,
        ltb,
        "train",
    )
    val_df = _filter_df(
        val_df_raw,
        question,
        args.val_audio_dir,
        args.val_video_dir,
        args.val_text_dir,
        fa,
        fv,
        ft,
        val_la,
        val_lab,
        val_lv,
        val_lvb,
        val_lt,
        val_ltb,
        "val",
    )

    if len(train_df) == 0:
        _diagnose_all_filtered(
            "train",
            train_df_raw,
            question,
            args.audio_dir,
            args.video_dir,
            args.text_dir,
            fa,
            fv,
            ft,
            la,
            lab,
            lv,
            lvb,
            lt,
            ltb,
        )
        raise SystemExit(2)
    if len(val_df) == 0:
        _diagnose_all_filtered(
            "val",
            val_df_raw,
            question,
            args.val_audio_dir,
            args.val_video_dir,
            args.val_text_dir,
            fa,
            fv,
            ft,
            val_la,
            val_lab,
            val_lv,
            val_lvb,
            val_lt,
            val_ltb,
        )
        raise SystemExit(2)

    os.makedirs(args.out_dir, exist_ok=True)

    id_to_row = {str(r["id"]).strip(): r for _, r in rating.iterrows()}
    X_train, y_train = _build_arrays(
        train_df,
        id_to_row,
        question,
        args.audio_dir,
        args.video_dir,
        args.text_dir,
        fa,
        fv,
        ft,
        la,
        lab,
        lv,
        lvb,
        lt,
        ltb,
        args.video_dim,
        args.audio_dim,
        args.text_dim,
        args.hand_dim,
        args.label_col,
        args.g_level_int_encoding,
        transcript_dir,
        wav_dir,
    )
    X_val, y_val = _build_arrays(
        val_df,
        id_to_row,
        question,
        args.val_audio_dir,
        args.val_video_dir,
        args.val_text_dir,
        fa,
        fv,
        ft,
        val_la,
        val_lab,
        val_lv,
        val_lvb,
        val_lt,
        val_ltb,
        args.video_dim,
        args.audio_dim,
        args.text_dim,
        args.hand_dim,
        args.label_col,
        args.g_level_int_encoding,
        transcript_dir,
        wav_dir,
    )

    pxt = os.path.join(args.out_dir, "X_train.npy")
    pyt = os.path.join(args.out_dir, "y_train.npy")
    pxv = os.path.join(args.out_dir, "X_val.npy")
    pyv = os.path.join(args.out_dir, "y_val.npy")
    np.save(pxt, X_train)
    np.save(pyt, y_train)
    np.save(pxv, X_val)
    np.save(pyv, y_val)
    meta = os.path.join(args.out_dir, "meta.txt")
    with open(meta, "w", encoding="utf-8") as f:
        f.write(
            f"video_dim={args.video_dim} audio_dim={args.audio_dim} text_dim={args.text_dim} "
            f"hand_dim={args.hand_dim} fused={X_train.shape[1]}\n"
            f"train_n={len(y_train)} val_n={len(y_val)} int_encoding={args.g_level_int_encoding}\n"
        )
    print(
        f"[prepare_svm_data] OK X_train{X_train.shape} y_train{y_train.shape} "
        f"X_val{X_val.shape} y_val{y_val.shape} → {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
