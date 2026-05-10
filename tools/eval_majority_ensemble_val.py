#!/usr/bin/env python3
"""
在官方 val 上对多份 g_level checkpoint 做逐样本多数投票，打印 accuracy / balanced_acc。
结构超参须与训练一致（cross_modal_attn 等）。
用法示例见文末。
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch
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
from model.vote_model.M_model import SharedMLPwEnsemble  # noqa: E402


def _majority_row(votes: list[int]) -> int:
    c = Counter(votes)
    best = max(c.values())
    return int(sorted(k for k, v in c.items() if v == best)[0])


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--rating_csv", required=True)
    p.add_argument("--labels_in_split_csv", action="store_true")
    p.add_argument("--label_col", nargs="+", default=["g_level"])
    p.add_argument("--question", nargs=6, default=["q1", "q2", "q3", "q4", "q5", "q6"])
    p.add_argument("--g_level_int_encoding", default="one", choices=("zero", "one"))
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--video_dir", required=True)
    p.add_argument("--text_dir", required=True)
    p.add_argument("--val_audio_dir", default="")
    p.add_argument("--val_video_dir", default="")
    p.add_argument("--val_text_dir", default="")
    p.add_argument("--audio_dim", type=int, default=512)
    p.add_argument("--video_dim", type=int, default=512)
    p.add_argument("--text_dim", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--mlp_dropout", type=float, default=0.25)
    p.add_argument("--modality_dropout_p", type=float, default=0.12)
    p.add_argument("--cross_modal_attn", action="store_true")
    p.add_argument("--cross_modal_layers", type=int, default=1)
    p.add_argument("--head_weights", action="store_true")
    p.add_argument("--time_weights", action="store_true")
    p.add_argument(
        "--text_enhancer",
        type=str,
        default="none",
        choices=("none", "transformer", "mlp"),
    )
    p.add_argument("--text_enhancer_dim", type=int, default=512)
    p.add_argument("--fused_layer_norm", action="store_true")
    args_ns, _ = p.parse_known_args()
    # 补齐 train_task2_glevel 里 SharedMLPwEnsemble 依赖的字段
    ns = argparse.Namespace(
        **vars(args_ns),
        glevel_arch="shared_mlp",
        glevel_loss="ce",
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
        glevel_csv="",
        mixup_prob=0.0,
        sampler_medium_boost=1.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _fb_a = ns.audio_dir
    _fb_v = ns.video_dir
    _fb_t = ns.text_dir
    val_set = MultimodalDatasetForTrainT2(
        ns.val_csv,
        ns.val_audio_dir or ns.audio_dir,
        ns.val_video_dir or ns.video_dir,
        ns.val_text_dir or ns.text_dir,
        list(ns.question),
        ns.label_col,
        ns.rating_csv,
        ns,
        fallback_audio_dir=_fb_a,
        fallback_video_dir=_fb_v,
        fallback_text_dir=_fb_t,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=ns.batch_size,
        shuffle=False,
        collate_fn=collate_fn_train,
        num_workers=ns.num_workers,
    )

    models = []
    for path in ns.checkpoints:
        m = SharedMLPwEnsemble(ns).to(device)
        try:
            st = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            st = torch.load(path, map_location=device)
        m.load_state_dict(st)
        m.eval()
        models.append(m)

    all_y: list[int] = []
    all_mv: list[int] = []
    for features, _mask, labels, _sid in tqdm(val_loader, desc="majority_val"):
        features = {k: v.to(device) for k, v in features.items()}
        B = labels.size(0)
        votes_batch: list[list[int]] = [[] for _ in range(B)]
        for model in models:
            logits = model(
                features["audio"],
                features["video"],
                features["text"],
                features["hand"],
            )
            pred = logits.argmax(dim=1).cpu().numpy()
            for i in range(B):
                votes_batch[i].append(int(pred[i]))
        mv = np.array([_majority_row(v) for v in votes_batch], dtype=np.int64)
        all_mv.append(mv)
        all_y.append(labels.numpy())
    y_hat = np.concatenate(all_mv)
    y_true = np.concatenate(all_y)
    acc = float(accuracy_score(y_true, y_hat))
    bacc = float(balanced_accuracy_score(y_true, y_hat))
    mf1 = float(f1_score(y_true, y_hat, average="macro", zero_division=0))
    print(
        f"[majority_ensemble] n={len(y_true)} checkpoints={len(models)} "
        f"val_acc={acc:.4f} val_bal_acc={bacc:.4f} macro_f1={mf1:.4f}"
    )


if __name__ == "__main__":
    main()
