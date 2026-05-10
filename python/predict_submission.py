#!/usr/bin/env python3
"""
统一提交推理入口（项目目录内读写）。

- --backend sklearn：加载 joblib Pipeline（与 tools/train_svm.py 保存的一致）
- --backend torch：加载 .pth，须与训练时结构一致；推荐传 --torch_args_json（训练时 args_glevel_*.json）

特征聚合与 tools/prepare_svm_data.py 一致：6 题各模态时间维 mean 后拼接
  [video_mean, audio_mean, text_mean, hand_mean]
输出 CSV：id,g_level_pred（官方整数 1/2/3；模型内部 argmax 为 0/1/2 时已在此处 +1）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PY_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, _PY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset.baseline_dataset2_vote import (  # noqa: E402
    _list_npy_filenames,
    _resolve_from_name_lists,
    _wav_duration_sec,
    compute_hand_feats_row,
)
from train_task2_glevel import coral_class_probs  # noqa: E402
from model.vote_model.M_model import SharedMLPwEnsemble  # noqa: E402
from model.vote_model.M_model import TextGRUClassifier, TextOnlyMLPClassifier  # noqa: E402


def _one_sample_svm_vector(
    sid: str,
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
    video_dim: int,
    audio_dim: int,
    text_dim: int,
    hand_dim: int,
    transcript_dir: str | None,
    wav_dir: str | None,
) -> np.ndarray:
    audio_paths = []
    video_paths = []
    text_paths = []
    for q in question:
        ap = _resolve_from_name_lists(sid, q, audio_dir, fa, la, lab)
        vp = _resolve_from_name_lists(sid, q, video_dir, fv, lv, lvb)
        tp = _resolve_from_name_lists(sid, q, text_dir, ft, lt, ltb)
        if ap is None or vp is None or tp is None:
            raise FileNotFoundError(f"缺特征 {sid}_{q}（需三模态 .npy 齐全，与 prepare_svm_data 一致）")
        audio_paths.append(ap)
        video_paths.append(vp)
        text_paths.append(tp)

    va = np.stack([np.load(p) for p in video_paths], axis=0).mean(axis=0)
    aa = np.stack([np.load(p) for p in audio_paths], axis=0).mean(axis=0)
    ta = np.stack([np.load(p) for p in text_paths], axis=0).mean(axis=0)
    if va.shape != (video_dim,) or aa.shape != (audio_dim,) or ta.shape != (text_dim,):
        raise ValueError(f"{sid} 维数不符 got v{va.shape} a{aa.shape} t{ta.shape}")

    hand_rows = []
    for ap, vp, tp in zip(audio_paths, video_paths, text_paths):
        af = os.path.basename(ap)
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
        raise ValueError(f"hand {hm.shape}")
    return np.concatenate([va, aa, ta, hm], axis=0).astype(np.float32)


def _build_torch_model(ns: SimpleNamespace, device: torch.device):
    if ns.glevel_arch == "text_gru":
        return TextGRUClassifier(ns).to(device)
    if ns.glevel_arch == "text_mlp":
        return TextOnlyMLPClassifier(ns).to(device)
    return SharedMLPwEnsemble(ns).to(device)


def _torch_forward_batch(model, batch_feats: dict, glevel_loss: str):
    logits = model(
        batch_feats["audio"],
        batch_feats["video"],
        batch_feats["text"],
        batch_feats["hand"],
    )
    if glevel_loss == "coral":
        return coral_class_probs(logits)
    return F.softmax(logits, dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_csv", default=os.path.join(_ROOT, "submission_predict.csv"))
    ap.add_argument("--backend", choices=("sklearn", "torch"), required=True)
    ap.add_argument("--sklearn_joblib", default="", help="backend=sklearn 时必填")
    ap.add_argument("--torch_ckpt", default="", help="backend=torch 时必填")
    ap.add_argument(
        "--torch_args_json",
        default="",
        help="训练保存的 args JSON（含 glevel_arch、维数、simple_clf、num_heads 等）",
    )
    ap.add_argument("--question", nargs="+", default=["q1", "q2", "q3", "q4", "q5", "q6"])
    ap.add_argument("--audio_dir", default="")
    ap.add_argument("--video_dir", default="")
    ap.add_argument("--text_dir", default="")
    ap.add_argument("--fallback_audio_dir", default="")
    ap.add_argument("--fallback_video_dir", default="")
    ap.add_argument("--fallback_text_dir", default="")
    ap.add_argument("--video_dim", type=int, default=512)
    ap.add_argument("--audio_dim", type=int, default=512)
    ap.add_argument("--text_dim", type=int, default=768)
    ap.add_argument("--hand_dim", type=int, default=4)
    ap.add_argument("--transcript_dir", default="")
    ap.add_argument("--wav_dir", default="")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument(
        "--test_audio_dir",
        default="",
        help="torch：覆盖 json 中 test_audio_dir（默认同 train_task2 回退规则）",
    )
    ap.add_argument("--test_video_dir", default="")
    ap.add_argument("--test_text_dir", default="")
    args = ap.parse_args()

    if args.backend == "sklearn":
        if not args.audio_dir or not args.video_dir or not args.text_dir:
            raise SystemExit("backend=sklearn 需要 --audio_dir / --video_dir / --text_dir（与 prepare_svm_data 一致）")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.test_csv)
    ids = df["id"].astype(str).str.strip().tolist()

    if args.backend == "sklearn":
        import joblib

        if not args.sklearn_joblib:
            raise SystemExit("backend=sklearn 需要 --sklearn_joblib")
        pipe = joblib.load(args.sklearn_joblib)
        fa = args.fallback_audio_dir or None
        fv = args.fallback_video_dir or None
        ft = args.fallback_text_dir or None
        transcript_dir = args.transcript_dir or None
        wav_dir = args.wav_dir or None
        if transcript_dir == "":
            transcript_dir = None
        if wav_dir == "":
            wav_dir = None
        la = _list_npy_filenames(args.audio_dir)
        lab = _list_npy_filenames(fa) if fa else []
        lv = _list_npy_filenames(args.video_dir)
        lvb = _list_npy_filenames(fv) if fv else []
        lt = _list_npy_filenames(args.text_dir)
        ltb = _list_npy_filenames(ft) if ft else []
        question = list(args.question)
        X = []
        for sid in ids:
            v = _one_sample_svm_vector(
                sid,
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
                transcript_dir,
                wav_dir,
            )
            X.append(v)
        X = np.stack(X, axis=0)
        pred = pipe.predict(X)
        out = pd.DataFrame({"id": ids, "g_level_pred": pred.astype(int) + 1})
        out.to_csv(args.out_csv, index=False)
        print(f"[predict_submission] sklearn → {args.out_csv} n={len(out)}", flush=True)
        return

    # torch
    if not args.torch_ckpt or not args.torch_args_json:
        raise SystemExit("backend=torch 需要 --torch_ckpt 与 --torch_args_json")
    with open(args.torch_args_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    ns = SimpleNamespace(**raw)
    # 保证列表类型
    if isinstance(ns.question, str):
        ns.question = [ns.question]
    if isinstance(ns.modalities, str):
        ns.modalities = [m.strip() for m in ns.modalities.split(",")]

    model = _build_torch_model(ns, device)
    try:
        state = torch.load(args.torch_ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.torch_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    from torch.utils.data import DataLoader

    from dataset.baseline_dataset2_vote import MultimodalDatasetForTestT2, collate_fn_test  # noqa: E402

    def _pick(dir_cli: str, ns_test: str, ns_train: str) -> str:
        for c in (dir_cli, ns_test, ns_train):
            s = (c or "").strip()
            if s:
                return s
        return ""

    ta = _pick(args.test_audio_dir, getattr(ns, "test_audio_dir", ""), getattr(ns, "audio_dir", ""))
    tv = _pick(args.test_video_dir, getattr(ns, "test_video_dir", ""), getattr(ns, "video_dir", ""))
    tt = _pick(args.test_text_dir, getattr(ns, "test_text_dir", ""), getattr(ns, "text_dir", ""))
    rating_csv = (getattr(ns, "rating_csv", "") or "").strip() or args.test_csv
    no_fb = bool(getattr(ns, "no_feature_fallback", False))
    fb_a = None if no_fb else getattr(ns, "audio_dir", None)
    fb_v = None if no_fb else getattr(ns, "video_dir", None)
    fb_t = None if no_fb else getattr(ns, "text_dir", None)
    test_set = MultimodalDatasetForTestT2(
        args.test_csv,
        ta,
        tv,
        tt,
        ns.question,
        rating_csv,
        ns,
        fallback_audio_dir=fb_a,
        fallback_video_dir=fb_v,
        fallback_text_dir=fb_t,
    )
    if len(test_set) == 0:
        print("[predict_submission] 测试集过滤后为空，跳过写出。", flush=True)
        return
    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_test,
        num_workers=args.num_workers,
    )
    glevel_loss = str(getattr(ns, "glevel_loss", "ce")).lower()
    all_pred = []
    with torch.no_grad():
        for features, _mask, batch_ids in loader:
            features = {k: v.to(device) for k, v in features.items()}
            probs = _torch_forward_batch(model, features, glevel_loss)
            pred = probs.argmax(dim=1).cpu().numpy()
            all_pred.append(pred)
    pred_idx = np.concatenate(all_pred)
    meta = pd.read_csv(args.test_csv).iloc[:, [0]]
    meta.columns = ["id"]
    out = meta.copy()
    out["g_level_pred"] = pred_idx.astype(int) + 1
    out.to_csv(args.out_csv, index=False)
    print(f"[predict_submission] torch → {args.out_csv} n={len(out)}", flush=True)


if __name__ == "__main__":
    main()
