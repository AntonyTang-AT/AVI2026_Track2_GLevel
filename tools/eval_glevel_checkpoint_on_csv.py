#!/usr/bin/env python3
"""
在任意含 g_level 的划分 CSV 上评估 checkpoint（留出集 / 官方 val 等）。
支持单模型、多模型 logits 平均（CE）、可选温度缩放与 OvR 阈值、导出 probs/logits 供校准脚本使用。

示例（Nanbeige，与 vote_train 路径一致）:
  python tools/eval_glevel_checkpoint_on_csv.py \\
    --eval_csv ./data/dev_holdout.csv \\
    --rating_csv /data/Super-Lu/dataset/train_data.csv \\
    --train_audio_dir /data/Super-Lu/dataset/train_feature/audio \\
    --train_video_dir /data/Super-Lu/dataset/train_feature/video \\
    --train_text_dir /path/to/data/text_nb \\
    --eval_audio_dir /data/Super-Lu/dataset/train_feature/audio \\
    --eval_video_dir /data/Super-Lu/dataset/train_feature/video \\
    --eval_text_dir /path/to/data/text_nb \\
    --checkpoint ./best.pth \\
    --labels_in_split_csv --g_level_int_encoding one \\
    --cross_modal_attn --text_dim 2560 \\
    --dump_probs ./logs/holdout_probs.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset.baseline_dataset2_vote import (  # noqa: E402
    MultimodalDatasetForTrainT2,
    collate_fn_train,
)
from model.vote_model.M_model import AudioTextMLPClassifier  # noqa: E402
from model.vote_model.M_model import SharedMLPwEnsemble  # noqa: E402
from train_task2_glevel import coral_class_probs  # noqa: E402


def _build_ns(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        question=list(args.question),
        glevel_arch=getattr(args, "glevel_arch", "shared_mlp") or "shared_mlp",
        glevel_loss=getattr(args, "glevel_loss", "ce"),
        num_classes=3,
        target_dim=3,
        simple_clf=False,
        temporal_gru=False,
        temporal_pool="mean",
        temporal_dropout=0.0,
        temporal_bidirectional=False,
        temporal_attn_pool=False,
        base_dim=768,
        hand_dim=4,
        num_heads=32,
        mlp_bottleneck_dim=32,
        cross_modal_nhead=8,
        cross_modal_ff_mult=4,
        cross_modal_dropout=0.1,
        modalities=["audio", "video", "text"],
        classification=True,
        freeze_text_features=False,
        no_feature_fallback=False,
        train_feat_fallback=False,
        no_drop_incomplete_features=False,
        transcript_dir="",
        wav_dir="",
        glevel_csv=getattr(args, "glevel_csv", "") or "",
        mixup_prob=0.0,
        sampler_medium_boost=1.0,
        labels_in_split_csv=args.labels_in_split_csv,
        g_level_int_encoding=args.g_level_int_encoding,
        audio_dim=args.audio_dim,
        video_dim=args.video_dim,
        text_dim=args.text_dim,
        mlp_dropout=args.mlp_dropout,
        modality_dropout_p=args.modality_dropout_p,
        cross_modal_attn=args.cross_modal_attn,
        cross_modal_layers=args.cross_modal_layers,
        feat_norm_npz=getattr(args, "feat_norm_npz", "") or "",
        feat_norm_apply=getattr(args, "feat_norm_apply", "none") or "none",
        feat_norm_eps=float(getattr(args, "feat_norm_eps", 1e-6)),
        head_weights=getattr(args, "head_weights", False),
        time_weights=getattr(args, "time_weights", False),
        text_enhancer=getattr(args, "text_enhancer", "none") or "none",
        text_enhancer_dim=int(getattr(args, "text_enhancer_dim", 512)),
        fused_layer_norm=bool(getattr(args, "fused_layer_norm", False)),
        at_mlp_hidden=int(getattr(args, "at_mlp_hidden", 512)),
    )


def _make_eval_model(ns: argparse.Namespace, device: torch.device) -> nn.Module:
    arch = getattr(ns, "glevel_arch", "shared_mlp") or "shared_mlp"
    if arch == "audio_text_mlp":
        return AudioTextMLPClassifier(ns).to(device)
    if arch not in ("shared_mlp",):
        raise SystemExit(
            f"eval_glevel_checkpoint_on_csv 当前仅支持 glevel_arch=shared_mlp 或 audio_text_mlp，"
            f"收到 {arch!r}"
        )
    return SharedMLPwEnsemble(ns).to(device)


def _apply_temperature(logits: torch.Tensor, t: float) -> torch.Tensor:
    if t <= 0:
        return logits
    return logits / t


def _pred_ovr(probs: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """probs: (N,3), thr: (3,) — 每类独立阈值，多过阈取 prob 最大，否则 argmax。"""
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


@torch.no_grad()
def _forward_batch(models: list[nn.Module], features: dict, device, glevel_loss: str):
    if len(models) == 1:
        lg = models[0](
            features["audio"],
            features["video"],
            features["text"],
            features["hand"],
        )
        return lg
    acc = None
    for m in models:
        lg = m(
            features["audio"],
            features["video"],
            features["text"],
            features["hand"],
        )
        acc = lg if acc is None else acc + lg
    return acc / float(len(models))


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_csv", required=True, help="含 id 与 g_level 的表")
    p.add_argument("--rating_csv", required=True)
    p.add_argument("--glevel_csv", default="", help="可选，补全标签")
    p.add_argument("--labels_in_split_csv", action="store_true")
    p.add_argument("--label_col", nargs="+", default=["g_level"])
    p.add_argument("--question", nargs=6, default=["q1", "q2", "q3", "q4", "q5", "q6"])
    p.add_argument("--g_level_int_encoding", default="one", choices=("zero", "one"))
    p.add_argument("--glevel_loss", default="ce", choices=("ce", "coral"))

    p.add_argument("--train_audio_dir", required=True)
    p.add_argument("--train_video_dir", required=True)
    p.add_argument("--train_text_dir", required=True)
    p.add_argument("--eval_audio_dir", default="")
    p.add_argument("--eval_video_dir", default="")
    p.add_argument("--eval_text_dir", default="")

    p.add_argument("--checkpoint", default="", help="单模型 .pth")
    p.add_argument("--ensemble_checkpoints", nargs="*", default=None)
    p.add_argument("--audio_dim", type=int, default=512)
    p.add_argument("--video_dim", type=int, default=512)
    p.add_argument("--text_dim", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--mlp_dropout", type=float, default=0.25)
    p.add_argument("--modality_dropout_p", type=float, default=0.12)
    p.add_argument("--cross_modal_attn", action="store_true")
    p.add_argument("--cross_modal_layers", type=int, default=1)
    p.add_argument(
        "--head_weights",
        action="store_true",
        help="须与训练该 .pth 时一致",
    )
    p.add_argument("--time_weights", action="store_true", help="须与训练时一致")
    p.add_argument(
        "--text_enhancer",
        type=str,
        default="none",
        choices=("none", "transformer", "mlp"),
        help="须与训练时一致",
    )
    p.add_argument("--text_enhancer_dim", type=int, default=512)
    p.add_argument(
        "--glevel_arch",
        type=str,
        default="shared_mlp",
        choices=("shared_mlp", "audio_text_mlp"),
        help="须与训练 checkpoint 一致",
    )
    p.add_argument(
        "--fused_layer_norm",
        action="store_true",
        help="须与训练时一致（shared_mlp / audio_text_mlp）",
    )
    p.add_argument("--at_mlp_hidden", type=int, default=512)

    p.add_argument("--logit_temperature", type=float, default=1.0)
    p.add_argument("--ovr_thresholds_json", default="", help="JSON 数组长度 3，每类阈值")

    p.add_argument("--dump_probs", default="", help="写入 .npz: logits, probs, labels, ids")
    p.add_argument("--feat_norm_npz", default="", help="与 train_task2_glevel --feat_norm_npz 一致")
    p.add_argument(
        "--feat_norm_apply",
        default="none",
        choices=("none", "all"),
        help="须 all 且提供 npz 才启用标准化",
    )
    p.add_argument("--feat_norm_eps", type=float, default=1e-6)

    args = p.parse_args()
    ckpts = []
    if args.checkpoint:
        ckpts.append(args.checkpoint)
    if args.ensemble_checkpoints:
        ckpts.extend(args.ensemble_checkpoints)
    if not ckpts:
        raise SystemExit("请指定 --checkpoint 或 --ensemble_checkpoints")

    ns = _build_ns(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ea = args.eval_audio_dir or args.train_audio_dir
    ev = args.eval_video_dir or args.train_video_dir
    et = args.eval_text_dir or args.train_text_dir

    eval_set = MultimodalDatasetForTrainT2(
        args.eval_csv,
        ea,
        ev,
        et,
        list(args.question),
        args.label_col,
        args.rating_csv,
        ns,
        fallback_audio_dir=args.train_audio_dir,
        fallback_video_dir=args.train_video_dir,
        fallback_text_dir=args.train_text_dir,
    )
    loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_train,
        num_workers=args.num_workers,
    )

    models = []
    for path in ckpts:
        m = _make_eval_model(ns, device)
        try:
            st = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            st = torch.load(path, map_location=device)
        m.load_state_dict(st)
        m.eval()
        models.append(m)

    all_y: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_ids: list[str] = []

    for features, _mask, labels, sids in tqdm(loader, desc="eval_csv"):
        features = {k: v.to(device) for k, v in features.items()}
        labels = labels.to(device)
        logits = _forward_batch(models, features, device, args.glevel_loss)
        logits = _apply_temperature(logits, float(args.logit_temperature))

        if args.glevel_loss == "coral":
            probs = coral_class_probs(logits)
            pred = probs.argmax(dim=1)
        else:
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)

        all_logits.append(logits.float().cpu().numpy())
        all_y.append(labels.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_ids.extend(list(sids))

    y_true = np.concatenate(all_y)
    logits_np = np.concatenate(all_logits)
    if args.glevel_loss == "coral":
        probs_np = coral_class_probs(torch.from_numpy(logits_np).float()).numpy()
    else:
        probs_np = torch.softmax(torch.from_numpy(logits_np).float(), dim=1).numpy()

    y_hat = np.concatenate(all_pred)
    if args.ovr_thresholds_json:
        with open(args.ovr_thresholds_json, "r", encoding="utf-8") as f:
            thr = np.array(json.load(f), dtype=np.float64)
        if thr.shape != (3,):
            raise SystemExit("ovr_thresholds_json 须为长度 3 的数组")
        y_hat = _pred_ovr(probs_np, thr)

    acc = float(accuracy_score(y_true, y_hat))
    bacc = float(balanced_accuracy_score(y_true, y_hat))
    mf1 = float(f1_score(y_true, y_hat, average="macro", zero_division=0))
    print(
        f"[metrics_line_local] split={os.path.basename(args.eval_csv)} "
        f"n={len(y_true)} checkpoints={len(models)} "
        f"acc={acc:.4f} bal_acc={bacc:.4f} macro_f1={mf1:.4f} "
        f"T={args.logit_temperature}"
    )

    if args.dump_probs:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_probs)) or ".", exist_ok=True)
        np.savez(
            args.dump_probs,
            logits=logits_np,
            probs=probs_np,
            labels=y_true,
            pred=y_hat,
            ids=np.array(all_ids, dtype=object),
        )
        print(f"[eval_glevel_checkpoint_on_csv] dumped {args.dump_probs}", flush=True)


if __name__ == "__main__":
    main()
