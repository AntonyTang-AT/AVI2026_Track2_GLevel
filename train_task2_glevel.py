"""
赛道二认知能力 g_level 三分类训练（SharedMLPwEnsemble + CrossEntropyLoss；训练可选 MixUp）。
可选 CORAL 序关系损失（--glevel_loss coral，三分类专用）：输出 K-1 个 logits，压跨档混淆。
标签：划分 csv 含 g_level 时用 --labels_in_split_csv；可叠加 --glevel_csv 补缺。
官方数据示例：train/val=/data/Super-Lu/dataset/train_data.csv & val_data.csv；
特征：train_feature + val_feature（可用 --val_*_dir / --test_*_dir）。

用法见文末示例命令。
"""
__version__ = "1.20.0"  # fused LayerNorm、StepLR、audio_text_mlp、早停默认 min_epochs=20

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
except (ImportError, OSError) as _torch_err:
    # #region agent log
    try:
        _dbg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug-f0e227.log")
        with open(_dbg_path, "a", encoding="utf-8") as _df:
            _df.write(
                json.dumps(
                    {
                        "sessionId": "f0e227",
                        "hypothesisId": "H1_nccl",
                        "location": "train_task2_glevel.py:torch_import",
                        "message": "torch_import_failed",
                        "data": {
                            "exc_type": type(_torch_err).__name__,
                            "error": repr(_torch_err),
                            "executable": sys.executable,
                        },
                        "timestamp": int(time.time() * 1000),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion
    print(
        "\n[train_task2_glevel] PyTorch 导入失败（多为 CUDA 版 torch 与 NCCL/CUDA 运行时栈不一致，"
        "与本仓库训练脚本逻辑无关）。\n"
        "可行处理：\n"
        "  1) 按 https://pytorch.org 选择与集群 CUDA 匹配的命令，在**新 conda 环境**中重装 torch 三件套；\n"
        "  2) 或暂用 CPU 版：pip install --force-reinstall torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cpu\n"
        "  3) 检查是否混用 LD_LIBRARY_PATH 指向旧 NCCL；可在一干净 shell 中 unset LD_LIBRARY_PATH 再试。\n"
        f"  原始错误: {_torch_err}\n",
        flush=True,
    )
    raise

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm, trange

from dataset.baseline_dataset2_vote import FEATURE_LOADER_REVISION
from dataset.baseline_dataset2_vote import MultimodalDatasetForTestT2, MultimodalDatasetForTrainT2
from dataset.baseline_dataset2_vote import collate_fn_test, collate_fn_train
from dataset.baseline_dataset2_vote import encode_g_level
from model.vote_model.M_model import AudioTextMLPClassifier
from model.vote_model.M_model import SharedMLPwEnsemble
from model.vote_model.M_model import TextGRUClassifier
from model.vote_model.M_model import TextOnlyMLPClassifier


def save_args(args, save_dir="./args_log"):
    os.makedirs(save_dir, exist_ok=True)
    log_dir = args.log_dir or "./logs"
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for base in (save_dir, log_dir):
        with open(os.path.join(base, f"args_glevel_{ts}.json"), "w") as f:
            json.dump(vars(args), f, indent=4, default=str)
    print(f"Args saved under {save_dir} and {log_dir}")


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def _print_learned_aggregation_if_any(model, args):
    """训练结束打印 ensemble 头 / 时间步 softmax 权重（若存在）。"""
    want = getattr(args, "print_aggregation_weights", False) or getattr(
        args, "head_weights", False
    ) or getattr(args, "time_weights", False)
    if not want or not isinstance(model, SharedMLPwEnsemble):
        return
    with torch.no_grad():
        if model.head_logits is not None:
            w = F.softmax(model.head_logits, dim=0).cpu().numpy()
            print(
                f"[learned_agg] head_weights_softmax (n={len(w)}): {w.tolist()}",
                flush=True,
            )
        if model.time_logits is not None:
            t = F.softmax(model.time_logits, dim=0).cpu().numpy()
            print(
                f"[learned_agg] time_weights_softmax (n={len(t)}): {t.tolist()}",
                flush=True,
            )


def save_loss_plot(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (CE)")
    plt.title("g_level classification")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_worker_init_fn(base_seed: int):
    """多进程 DataLoader 子进程 RNG，减轻 num_workers>0 时轨迹漂移。"""

    def _fn(worker_id: int):
        w = int(base_seed) + int(worker_id)
        random.seed(w)
        np.random.seed(w % (2**32 - 1))
        torch.manual_seed(w)

    return _fn


def _scalar_int_g_level(raw) -> int | None:
    """若 raw 可解释为纯整数类标签则返回 int，否则 None（如 Low/Medium/High 字符串）。"""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    if isinstance(raw, (bool, np.bool_)):
        return None
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    if isinstance(raw, (float, np.floating)):
        if float(raw).is_integer():
            return int(raw)
        return None
    s = str(raw).strip()
    if s.isdigit() or (s.startswith("-") and len(s) > 1 and s[1:].isdigit()):
        return int(s)
    try:
        f = float(s)
        if f.is_integer():
            return int(f)
    except ValueError:
        pass
    return None


def _maybe_autofix_g_level_int_encoding(args, train_set, val_set) -> None:
    """默认 zero 时：若整数标签含 3 且无 0，视为赛方 1–3 标注并自动改为 one。"""
    if args.g_level_int_encoding != "zero":
        return
    ints: set[int] = set()
    for ds in (train_set, val_set):
        if getattr(ds, "glevel_dict", None) is None:
            continue
        for _, row in ds.data.iterrows():
            sid = str(row["id"])
            raw = ds.glevel_dict[sid]
            v = _scalar_int_g_level(raw)
            if v is not None:
                ints.add(v)
    if not ints:
        return
    lo, hi = min(ints), max(ints)
    if lo < 0 or hi > 3:
        raise ValueError(
            f"g_level 整数标签超出三分类范围: 观察到 {sorted(ints)}；请检查 CSV 或 --g_level_int_encoding"
        )
    if 0 in ints and hi >= 3:
        raise ValueError(
            "g_level 整数同时出现 0 与 ≥3，无法判断是 0–2 还是 1–3 编码；"
            "请检查 CSV 或显式指定 --g_level_int_encoding zero 或 one"
        )
    if 3 in ints:
        args.g_level_int_encoding = "one"
        train_set._glevel_int_enc = "one"
        val_set._glevel_int_enc = "one"
        print(
            "[train_task2_glevel] 整数标签含 3 且无 0 → 自动使用 int_encoding=one（1=Low,2=Medium,3=High）。"
            "若赛方实为 0–2 且 3 为笔误，请显式传 --g_level_int_encoding zero 并修正数据。",
            flush=True,
        )
        return
    if ints <= {1, 2} and 0 not in ints:
        print(
            "[train_task2_glevel] 提示: 整数标签仅见 {1,2}（无 0、无 3），"
            "无法区分 0-based(1=Medium) 与 1-based(1=Low)；当前仍用 int_encoding=zero。"
            "若赛方是 1–3 且本划分缺某类，请加 --g_level_int_encoding one。",
            flush=True,
        )


def _collect_train_labels(train_set) -> tuple[list[int], Counter]:
    enc = getattr(train_set, "_glevel_int_enc", "zero")
    ys: list[int] = []
    for _, row in train_set.data.iterrows():
        sid = str(row["id"])
        raw = train_set.glevel_dict[sid]
        ys.append(int(encode_g_level(raw, int_encoding=enc)))
    return ys, Counter(ys)


def _inv_freq_sample_weights(ys: list[int], cnt: Counter) -> list[float]:
    return [1.0 / max(cnt[y], 1) for y in ys]


def _inv_freq_sample_weights_medium_boost(
    ys: list[int], cnt: Counter, medium_boost: float
) -> list[float]:
    """在反比类频基础上，对 Medium（类 1）再乘 medium_boost。"""
    out: list[float] = []
    for y in ys:
        w = 1.0 / max(cnt[y], 1)
        if y == 1 and medium_boost != 1.0:
            w *= medium_boost
        out.append(w)
    return out


def _ce_weights_from_counts(
    cnt: Counter, num_classes: int, n_total: int, device: torch.device
) -> torch.Tensor:
    """仅出现过的类赋权，未出现类为 0；均值归一化到 1（在出现类上）。"""
    n_per = np.array([cnt.get(c, 0) for c in range(num_classes)], dtype=np.float64)
    present = n_per > 0
    n_pres = int(present.sum())
    w_np = np.zeros(num_classes, dtype=np.float64)
    if n_pres > 0:
        for c in range(num_classes):
            if n_per[c] > 0:
                w_np[c] = n_total / (n_pres * n_per[c])
        w_np[present] /= w_np[present].mean()
    return torch.tensor(w_np, dtype=torch.float32, device=device)


def _log_split_label_counts(split_name: str, dataset: MultimodalDatasetForTrainT2):
    """打印原始标签取值分布与 encode 后的类频，便于核对 0/1/2 与 1/2/3 是否混用。"""
    if dataset.glevel_dict is None:
        return
    enc = getattr(dataset, "_glevel_int_enc", "zero")
    raw_c: Counter = Counter()
    cnt: Counter = Counter()
    for _, row in dataset.data.iterrows():
        sid = str(row["id"])
        raw = dataset.glevel_dict[sid]
        rkey = str(raw).strip() if isinstance(raw, str) else repr(raw)
        raw_c[rkey] += 1
        y = int(encode_g_level(raw, int_encoding=enc))
        cnt[y] += 1
    name = {0: "Low", 1: "Medium", 2: "High"}
    raw_parts = [f"{k}×{raw_c[k]}" for k in sorted(raw_c.keys(), key=str)]
    parts = [f"{name.get(c, c)}={cnt[c]}" for c in sorted(cnt)]
    print(
        f"[train_task2_glevel] {split_name} g_level 原始取值计数: "
        + ", ".join(raw_parts)
        + f" | int_encoding={enc!r}",
        flush=True,
    )
    print(
        f"[train_task2_glevel] {split_name} g_level 编码后: "
        + ", ".join(parts)
        + f" | n={len(dataset.data)}",
        flush=True,
    )


def coral_binary_targets(y: torch.Tensor, k_minus_1: int) -> torch.Tensor:
    """CORAL：样本标签 y∈{0..K-1}，第 j 个辅助任务目标为 1[y>j]，形状 [B, K-1]。"""
    y = y.long()
    j = torch.arange(k_minus_1, device=y.device, dtype=torch.long).view(1, -1)
    return (y.unsqueeze(1) > j).float()


def coral_bce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    t = coral_binary_targets(y, logits.size(1))
    return F.binary_cross_entropy_with_logits(logits, t, reduction="mean")


def coral_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """由 CORAL logits 恢复 P(y=k)，k=0..K-1。"""
    s = torch.sigmoid(logits)
    B, km1 = s.shape
    K = km1 + 1
    p = s.new_zeros(B, K)
    p[:, 0] = 1.0 - s[:, 0]
    for j in range(1, K - 1):
        p[:, j] = s[:, j - 1] - s[:, j]
    p[:, K - 1] = s[:, km1 - 1]
    p = p.clamp(min=1e-8)
    return p / p.sum(dim=1, keepdim=True)


def mixup_temporal_chunk_cls(chunk_flat, labels, B, T, num_classes, alpha, device):
    """与 train_task2_vote 中 mixup_temporal_chunk 同构：在样本维混合 [B,T,D] chunk。
    分类任务将标签转为 one-hot 再线性混合，配合 log_softmax 做 soft CE。"""
    D = chunk_flat.size(-1)
    feats = chunk_flat.view(B, T, D)
    index = torch.randperm(B, device=device)
    lam = float(np.random.beta(alpha, alpha))
    mixed_feats = lam * feats + (1.0 - lam) * feats[index]
    y_a = F.one_hot(labels.long(), num_classes).float()
    y_b = y_a[index]
    mixed_y = lam * y_a + (1.0 - lam) * y_b
    return mixed_feats.view(B * T, D), mixed_y


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    mixup_prob=0.5,
    mixup_alpha=0.2,
    num_classes=3,
    glevel_loss: str = "ce",
    use_sam: bool = False,
):
    model.train()
    total_loss = 0.0
    bar = tqdm(loader, desc="Train", leave=False)
    for features, _mask, labels, _sample_ids in bar:
        features = {k: v.to(device) for k, v in features.items()}
        labels = labels.to(device)
        optimizer.zero_grad()
        use_mixup = (
            glevel_loss == "ce"
            and mixup_prob > 0.0
            and random.random() < mixup_prob
            and hasattr(model, "encode_multimodal_chunk")
        )
        if use_mixup:
            chunk_flat, B, T = model.encode_multimodal_chunk(
                features["audio"],
                features["video"],
                features["text"],
                features["hand"],
            )
            mixed_chunk, mixed_y = mixup_temporal_chunk_cls(
                chunk_flat, labels, B, T, num_classes, mixup_alpha, device
            )
            logits = model.forward_heads(mixed_chunk, B, T)
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(mixed_y * log_probs).sum(dim=1).mean()
        else:
            logits = model(
                features["audio"],
                features["video"],
                features["text"],
                features["hand"],
            )
            if glevel_loss == "coral":
                loss = coral_bce_loss(logits, labels)
            else:
                loss = criterion(logits, labels)
        if use_sam:
            loss.backward()
            optimizer.first_step(zero_grad=True)
            if use_mixup:
                logits2 = model.forward_heads(mixed_chunk, B, T)
                log_probs2 = F.log_softmax(logits2, dim=1)
                loss2 = -(mixed_y * log_probs2).sum(dim=1).mean()
            else:
                logits2 = model(
                    features["audio"],
                    features["video"],
                    features["text"],
                    features["hand"],
                )
                if glevel_loss == "coral":
                    loss2 = coral_bce_loss(logits2, labels)
                else:
                    loss2 = criterion(logits2, labels)
            loss2.backward()
            optimizer.second_step(zero_grad=True)
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        bar.set_postfix(loss=loss.item())
    return total_loss / max(len(loader), 1)


def _scale_logits_temp(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = float(temperature or 1.0)
    if t > 0 and abs(t - 1.0) > 1e-12:
        return logits / t
    return logits


def _parse_infer_logit_bias(s: str, num_classes: int, device: torch.device) -> torch.Tensor | None:
    """推理时在 softmax 前给各类 logit 加常数偏置（CE 头）；空字符串表示关闭。"""
    raw = (s or "").strip()
    if not raw:
        return None
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != num_classes:
        raise SystemExit(
            f"--infer_logit_bias 需要恰好 {num_classes} 个浮点数（逗号分隔），当前 {len(parts)} 个: {raw!r}"
        )
    return torch.tensor(parts, dtype=torch.float32, device=device)


def _apply_infer_logit_bias(
    logits: torch.Tensor, infer_logit_bias: torch.Tensor | None
) -> torch.Tensor:
    if infer_logit_bias is None:
        return logits
    return logits + infer_logit_bias.view(1, -1)


@torch.no_grad()
def evaluate_epoch(
    model,
    loader,
    criterion,
    device,
    num_classes: int,
    glevel_loss: str = "ce",
    logit_temperature: float = 1.0,
    infer_logit_bias: torch.Tensor | None = None,
):
    model.eval()
    total_loss = 0.0
    all_pred, all_y = [], []
    bar = tqdm(loader, desc="Val", leave=False)
    for features, _mask, labels, _sample_ids in bar:
        features = {k: v.to(device) for k, v in features.items()}
        labels = labels.to(device)
        logits = model(
            features["audio"],
            features["video"],
            features["text"],
            features["hand"],
        )
        if glevel_loss == "coral":
            if infer_logit_bias is not None:
                raise ValueError("evaluate_epoch: infer_logit_bias 与 glevel_loss=coral 不兼容")
            loss = coral_bce_loss(logits, labels)
            pred = coral_class_probs(logits).argmax(dim=1)
        else:
            logits = _apply_infer_logit_bias(logits, infer_logit_bias)
            logits = _scale_logits_temp(logits, logit_temperature)
            loss = criterion(logits, labels)
            pred = logits.argmax(dim=1)
        total_loss += loss.item()
        all_pred.append(pred.cpu().numpy())
        all_y.append(labels.cpu().numpy())
    y_hat = np.concatenate(all_pred)
    y_true = np.concatenate(all_y)
    acc = accuracy_score(y_true, y_hat)
    present_val = np.unique(y_true)
    macro_f1 = float(
        f1_score(
            y_true,
            y_hat,
            average="macro",
            labels=present_val,
            zero_division=0,
        )
    )
    bal_acc = float(balanced_accuracy_score(y_true, y_hat))
    nuniq_pred = int(len(np.unique(y_hat)))
    return total_loss / max(len(loader), 1), acc, macro_f1, bal_acc, nuniq_pred


@torch.no_grad()
def evaluate_epoch_ensemble(
    models: list,
    loader,
    criterion,
    device,
    num_classes: int,
    glevel_loss: str = "ce",
    logit_temperature: float = 1.0,
    infer_logit_bias: torch.Tensor | None = None,
):
    """多模型对 logits 算术平均后再 softmax / argmax（仅 glevel_loss=ce 支持）。"""
    if glevel_loss != "ce":
        raise ValueError("evaluate_epoch_ensemble 当前仅支持 glevel_loss=ce")
    for m in models:
        m.eval()
    total_loss = 0.0
    all_pred, all_y = [], []
    bar = tqdm(loader, desc="ValEns", leave=False)
    for features, _mask, labels, _sample_ids in bar:
        features = {k: v.to(device) for k, v in features.items()}
        labels = labels.to(device)
        logits_sum = None
        for model in models:
            lg = model(
                features["audio"],
                features["video"],
                features["text"],
                features["hand"],
            )
            logits_sum = lg if logits_sum is None else logits_sum + lg
        logits = logits_sum / float(len(models))
        logits = _apply_infer_logit_bias(logits, infer_logit_bias)
        logits = _scale_logits_temp(logits, logit_temperature)
        loss = criterion(logits, labels)
        pred = logits.argmax(dim=1)
        total_loss += loss.item()
        all_pred.append(pred.cpu().numpy())
        all_y.append(labels.cpu().numpy())
    y_hat = np.concatenate(all_pred)
    y_true = np.concatenate(all_y)
    acc = accuracy_score(y_true, y_hat)
    present_val = np.unique(y_true)
    macro_f1 = float(
        f1_score(
            y_true,
            y_hat,
            average="macro",
            labels=present_val,
            zero_division=0,
        )
    )
    bal_acc = float(balanced_accuracy_score(y_true, y_hat))
    nuniq_pred = int(len(np.unique(y_hat)))
    return total_loss / max(len(loader), 1), acc, macro_f1, bal_acc, nuniq_pred


IDX_TO_NAME = {0: "1", 1: "2", 2: "3"}


@torch.no_grad()
def summarize_val_split(
    model,
    loader,
    criterion,
    device,
    num_classes: int,
    header: str = "[val_summary]",
    glevel_loss: str = "ce",
):
    """训练结束打印：混淆矩阵、per-class precision/recall/F1、错配真→预测、margin 分布。"""
    if loader is None or len(loader.dataset) == 0:
        print(f"{header} 验证集为空，跳过摘要。", flush=True)
        return
    model.eval()
    total_loss = 0.0
    all_pred, all_y = [], []
    margins: list[float] = []
    for features, _mask, labels, _sample_ids in tqdm(loader, desc="val_summary", leave=False):
        features = {k: v.to(device) for k, v in features.items()}
        labels = labels.to(device)
        logits = model(
            features["audio"],
            features["video"],
            features["text"],
            features["hand"],
        )
        if glevel_loss == "coral":
            probs = coral_class_probs(logits)
            pred = probs.argmax(dim=1)
            nll = -torch.log(probs.gather(1, labels.long().unsqueeze(1)).squeeze(1) + 1e-8)
            total_loss += nll.sum().item()
        else:
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
        all_pred.append(pred.cpu().numpy())
        all_y.append(labels.cpu().numpy())
        pi = probs.detach().cpu().numpy()
        for i in range(pi.shape[0]):
            sp = np.sort(pi[i])[::-1]
            margins.append(float(sp[0] - sp[1]) if len(sp) >= 2 else 0.0)
    y_hat = np.concatenate(all_pred)
    y_true = np.concatenate(all_y)
    n = len(y_true)
    mean_ce = total_loss / max(n, 1)
    names = [IDX_TO_NAME[i] for i in range(num_classes)]
    labels_all = list(range(num_classes))
    cm = confusion_matrix(y_true, y_hat, labels=labels_all)
    print(f"{header} n={n} mean_per_sample_ce={mean_ce:.4f}", flush=True)
    print(f"{header} 混淆矩阵 行=真值 列=预测 {names}", flush=True)
    print(cm, flush=True)
    print(
        classification_report(
            y_true,
            y_hat,
            labels=labels_all,
            target_names=names,
            digits=4,
            zero_division=0,
        ),
        end="",
        flush=True,
    )
    wrong_mask = y_true != y_hat
    if wrong_mask.any() and margins:
        m_arr = np.array(margins)
        print(
            f"{header} margin_top2: 全体 mean={m_arr.mean():.4f} median={np.median(m_arr):.4f} | "
            f"错分子集 mean={m_arr[wrong_mask].mean():.4f}",
            flush=True,
        )
    elif margins:
        m_arr = np.array(margins)
        print(
            f"{header} margin_top2: mean={m_arr.mean():.4f} median={np.median(m_arr):.4f}",
            flush=True,
        )
    wrong = y_hat[wrong_mask]
    wtrue = y_true[wrong_mask]
    if len(wrong):
        ctr = Counter(
            (IDX_TO_NAME[int(a)], IDX_TO_NAME[int(b)]) for a, b in zip(wtrue, wrong)
        )
        print(f"{header} 错分 {wrong_mask.sum()}/{n} 真→预测:", flush=True)
        for (a, b), c in ctr.most_common():
            print(f"  {a} → {b}: {c}", flush=True)
    else:
        print(f"{header} 验证集无错分。", flush=True)


@torch.no_grad()
def write_glevel_val_error_report(
    model,
    loader,
    device,
    out_csv: str,
    val_meta_df: pd.DataFrame,
    num_classes: int = 3,
    glevel_loss: str = "ce",
):
    """导出验证集逐样本预测、softmax、CE，合并 val 表列；打印混淆矩阵与错配统计。

    列 `prob_class*` / `margin_top2` 用于诊断「全预测 Medium」等塌缩：若 class1 概率普遍最高且 margin 极小，
    结合训练日志中的 val_pred_classes 与双重点权 WARNING 排查。
    """
    from collections import Counter

    model.eval()
    idx_to_name = {i: IDX_TO_NAME.get(i, str(i)) for i in range(num_classes)}
    rows: list[dict] = []
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    meta_df = val_meta_df.copy()
    meta_df["id"] = meta_df["id"].astype(str)

    for features, _mask, labels, sample_ids in tqdm(loader, desc="val_errors", leave=False):
        features = {k: v.to(device) for k, v in features.items()}
        labels = labels.to(device)
        logits = model(
            features["audio"],
            features["video"],
            features["text"],
            features["hand"],
        )
        if glevel_loss == "coral":
            probs = coral_class_probs(logits)
            pred = probs.argmax(dim=1)
            ce = -torch.log(probs.gather(1, labels.long().unsqueeze(1)).squeeze(1) + 1e-8)
        else:
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            ce = F.cross_entropy(logits, labels, reduction="none")
        for i in range(labels.size(0)):
            y = int(labels[i].item())
            p = int(pred[i].item())
            pid = str(sample_ids[i])
            cm[y, p] += 1
            pi = probs[i].detach().cpu().numpy()
            sp = np.sort(pi)[::-1]
            margin = float(sp[0] - sp[1]) if len(sp) >= 2 else 0.0
            row = {
                "id": pid,
                "y_idx": y,
                "pred_idx": p,
                "y_name": idx_to_name[y],
                "pred_name": idx_to_name[p],
                "correct": y == p,
                "ce": float(ce[i].item()),
                "margin_top2": margin,
            }
            for c in range(num_classes):
                row[f"prob_class{c}"] = float(pi[c])
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.merge(meta_df, on="id", how="left", suffixes=("", "_valcsv"))

    out_dir = os.path.dirname(os.path.abspath(out_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.sort_values(["correct", "ce"], ascending=[True, False]).to_csv(out_csv, index=False)

    print(f"[val_errors] 逐样本分析已写入: {out_csv}（{len(df)} 行）", flush=True)
    names = [idx_to_name[j] for j in range(num_classes)]
    print(f"[val_errors] 混淆矩阵 行=真值 列=预测 {names}", flush=True)
    print(cm, flush=True)
    for j in range(num_classes):
        tot = int(cm[j].sum())
        hit = int(cm[j, j])
        rec = hit / tot if tot else 0.0
        print(
            f"[val_errors] 类 {names[j]} recall={rec:.3f} support={tot}",
            flush=True,
        )
    wrong = df[~df["correct"]]
    print(
        f"[val_errors] 错分 {len(wrong)}/{len(df)}；真→预测 计数:",
        flush=True,
    )
    ctr: Counter = Counter(zip(wrong["y_name"], wrong["pred_name"]))
    for (a, b), c in ctr.most_common():
        print(f"  {a} → {b}: {c}", flush=True)


@torch.no_grad()
def _batch_class_probs_tta(
    model,
    features: dict,
    glevel_loss: str,
    tta_times: int,
    tta_noise_std: float,
    logit_temperature: float = 1.0,
    infer_logit_bias: torch.Tensor | None = None,
):
    """单 batch 类概率；tta_times>0 时对特征加高斯噪声多次平均。"""
    if tta_times <= 0:
        logits = model(
            features["audio"],
            features["video"],
            features["text"],
            features["hand"],
        )
        if glevel_loss == "coral":
            if infer_logit_bias is not None:
                raise ValueError("_batch_class_probs_tta: infer_logit_bias 与 coral 不兼容")
            return coral_class_probs(logits)
        logits = _apply_infer_logit_bias(logits, infer_logit_bias)
        logits = _scale_logits_temp(logits, logit_temperature)
        return F.softmax(logits, dim=1)
    acc = None
    for _ in range(int(tta_times)):
        ff = {
            k: v + torch.randn_like(v) * float(tta_noise_std)
            for k, v in features.items()
        }
        logits = model(
            ff["audio"],
            ff["video"],
            ff["text"],
            ff["hand"],
        )
        if glevel_loss == "coral":
            if infer_logit_bias is not None:
                raise ValueError("_batch_class_probs_tta: infer_logit_bias 与 coral 不兼容")
            p = coral_class_probs(logits)
        else:
            logits = _apply_infer_logit_bias(logits, infer_logit_bias)
            logits = _scale_logits_temp(logits, logit_temperature)
            p = F.softmax(logits, dim=1)
        acc = p if acc is None else acc + p
    return acc / float(tta_times)


def maybe_report_val_errors(args, model, val_loader, val_set, device):
    path = (getattr(args, "val_errors_csv", None) or "").strip()
    if not path:
        return
    write_glevel_val_error_report(
        model,
        val_loader,
        device,
        path,
        val_set.data,
        num_classes=args.num_classes,
        glevel_loss=args.glevel_loss,
    )


@torch.no_grad()
def predict_test_ensemble(
    models: list,
    loader,
    device,
    test_csv_path,
    out_csv,
    write_names: bool,
    glevel_loss: str = "ce",
    tta_times: int = 0,
    tta_noise_std: float = 0.01,
    logit_temperature: float = 1.0,
    infer_logit_bias: torch.Tensor | None = None,
):
    """多模型对每类概率平均（与 evaluate_epoch_ensemble 一致）；tta 对每个成员分别做再平均。"""
    if glevel_loss != "ce":
        raise ValueError("predict_test_ensemble 当前仅支持 glevel_loss=ce")
    if loader is None or len(loader.dataset) == 0:
        print(
            "[predict_test_ensemble] 测试集为空，跳过导出。",
            flush=True,
        )
        return
    for m in models:
        m.eval()
    ids_list, pred_idx = [], []
    bar = tqdm(loader, desc="TestEns", leave=False)
    for features, _mask, batch_ids in bar:
        features = {k: v.to(device) for k, v in features.items()}
        probs_sum = None
        for model in models:
            p = _batch_class_probs_tta(
                model,
                features,
                glevel_loss,
                tta_times,
                tta_noise_std,
                logit_temperature=logit_temperature,
                infer_logit_bias=infer_logit_bias,
            )
            probs_sum = p if probs_sum is None else probs_sum + p
        probs = probs_sum / float(len(models))
        pred = probs.argmax(dim=1).cpu().numpy()
        pred_idx.append(pred)
        ids_list.extend(list(batch_ids))

    pred_idx = np.concatenate(pred_idx)
    meta = pd.read_csv(test_csv_path).iloc[:, [0]]
    meta.columns = ["id"]
    out = meta.copy()
    out["g_level_pred"] = pred_idx.astype(int) + 1
    if write_names:
        out["g_level_pred_name"] = [IDX_TO_NAME[int(i)] for i in pred_idx]
    out.to_csv(out_csv, index=False)
    print(f"Predictions saved to {out_csv} (ensemble n={len(models)})")


@torch.no_grad()
def predict_test(
    model,
    loader,
    device,
    test_csv_path,
    out_csv,
    write_names: bool,
    glevel_loss: str = "ce",
    tta_times: int = 0,
    tta_noise_std: float = 0.01,
    logit_temperature: float = 1.0,
    infer_logit_bias: torch.Tensor | None = None,
):
    if loader is None or len(loader.dataset) == 0:
        print(
            "[predict_test] 测试集过滤后为空（当前 FEAT_TEST 下无可用 .npy），跳过导出 submission。"
            " 补全测试特征后重跑，或设置 FEAT_TEST 指向含测试 id 的特征根目录。",
            flush=True,
        )
        return
    model.eval()
    ids_list, pred_idx = [], []
    bar = tqdm(loader, desc="Test", leave=False)
    for features, _mask, batch_ids in bar:
        features = {k: v.to(device) for k, v in features.items()}
        probs = _batch_class_probs_tta(
            model,
            features,
            glevel_loss,
            tta_times,
            tta_noise_std,
            logit_temperature=logit_temperature,
            infer_logit_bias=infer_logit_bias,
        )
        pred = probs.argmax(dim=1).cpu().numpy()
        pred_idx.append(pred)
        ids_list.extend(list(batch_ids))

    pred_idx = np.concatenate(pred_idx)
    meta = pd.read_csv(test_csv_path).iloc[:, [0]]
    meta.columns = ["id"]
    out = meta.copy()
    out["g_level_pred"] = pred_idx.astype(int) + 1
    if write_names:
        out["g_level_pred_name"] = [IDX_TO_NAME[int(i)] for i in pred_idx]
    out.to_csv(out_csv, index=False)
    print(f"Predictions saved to {out_csv}")


def main():
    print(
        f"[train_task2_glevel] dataset FEATURE_LOADER_REVISION={FEATURE_LOADER_REVISION} "
        f"(仍出现 _resolve_feature_path 则说明加载了旧版 dataset)",
        flush=True,
    )
    p = argparse.ArgumentParser(description=f"train_task2_glevel.py v{__version__}")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--rating_csv", required=True, help="至少含 id；若含 g_level 可直接作标签表，否则配合 --glevel_csv")
    p.add_argument(
        "--glevel_csv",
        type=str,
        default="",
        help="可选。id+label_col；可单独用，或与 --labels_in_split_csv 联用（只填补划分 csv 里缺的 id，如 val 无 g_level）",
    )
    p.add_argument(
        "--labels_in_split_csv",
        action="store_true",
        help="从 train_csv / val_csv 各自读取 label_col（如 g_level），无需单独 glevel_train.csv",
    )
    p.add_argument("--label_col", nargs="+", default=["g_level"], help="分类模式下列名只能有一个，默认 g_level")
    p.add_argument(
        "--g_level_int_encoding",
        type=str,
        default="one",
        choices=("zero", "one"),
        help=(
            "CSV 中为整数标签时的约定：one=赛方官方 1/2/3 → 内部类下标 0/1/2（默认）；"
            "zero=CSV 已为 0/1/2 类下标。"
            "英文 low/medium/high 字符串不受此项影响。"
            "若 train+val 整数标签与默认值不一致，脚本仍会尝试自动推断。"
        ),
    )
    p.add_argument("--question", nargs="+", required=True)

    p.add_argument("--audio_dir", required=True, help="训练集（及默认测试集）音频 .npy 目录")
    p.add_argument("--video_dir", required=True)
    p.add_argument("--text_dir", required=True)
    p.add_argument(
        "--val_audio_dir",
        type=str,
        default="",
        help="验证集特征目录；默认同训练。官方 baseline 常为 .../val_feature/audio",
    )
    p.add_argument("--val_video_dir", type=str, default="")
    p.add_argument("--val_text_dir", type=str, default="")
    p.add_argument(
        "--test_audio_dir",
        type=str,
        default="",
        help="测试集特征；默认同 --audio_dir（赛方测试特征常与 train 同盘）",
    )
    p.add_argument("--test_video_dir", type=str, default="")
    p.add_argument("--test_text_dir", type=str, default="")
    p.add_argument(
        "--test_fallback_val_features",
        action="store_true",
        help=(
            "测试集 .npy 解析增加第三级回退到 --val_audio_dir / val_video / val_text："
            "当 test_merged 含原官方 val 的 id（特征在 val_feature 与 val 文本目录）而主目录为 FEAT_TEST 时使用。"
        ),
    )
    p.add_argument(
        "--no_feature_fallback",
        action="store_true",
        help="禁止验证/测试在 val/test 目录缺 .npy 时回退到 --audio_dir/--video_dir/--text_dir",
    )
    p.add_argument(
        "--train_feat_fallback",
        action="store_true",
        help=(
            "训练集也启用特征回退：主目录为 --audio_dir 等，缺文件时到 --val_audio_dir 等查找。"
            "合并 train/val 做 K 折且部分 id 仅在 val_feature 下有必要开此项。"
        ),
    )
    p.add_argument(
        "--train_fallback_use_test_features",
        action="store_true",
        help=(
            "须与 --train_feat_fallback 联用：训练集特征回退目录改为 --test_audio_dir / "
            "--test_video_dir / --test_text_dir（用于把官方测试 id 并入 train CSV，"
            "其 .npy 仅在 FEAT_TEST 与测试文本目录下）。"
        ),
    )
    p.add_argument(
        "--no_drop_incomplete_features",
        action="store_true",
        help="默认会剔除缺少任一题、任一模态 .npy 的样本；加此项则保留（缺文件时 DataLoader 报错）",
    )
    p.add_argument("--transcript_dir", type=str, default="")
    p.add_argument("--wav_dir", type=str, default="")
    # 默认与 /data/AVI2026/train_feature 下 .npy 实测一致（可用 tools/check_feature_shapes.py 核对）
    p.add_argument("--audio_dim", type=int, default=512)
    p.add_argument("--video_dim", type=int, default=512)
    p.add_argument(
        "--text_dim",
        type=int,
        default=768,
        help=(
            "与 text .npy 最后一维一致；SigLIP 常为 768；"
            "Nanbeige 单层融合为 hidden_size（如 2560）；"
            "extract_text 使用 --layer_fuse concat_k 时为 hidden_size*K"
        ),
    )

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=40,
        help="验证指标无提升的连续 epoch 数后早停（默认 40；与 --no_early_stop 同时使用时此项忽略）",
    )
    p.add_argument(
        "--early_stop_min_epochs",
        type=int,
        default=20,
        help=(
            "至少训练多少个 epoch（当前 epoch 序号）后才允许早停；"
            "小验证集上默认 20，避免假早停。"
        ),
    )
    p.add_argument(
        "--no_early_stop",
        action="store_true",
        help="关闭早停，始终训练满 --num_epochs（仍按 select_best 在改进时覆盖保存 best checkpoint）",
    )
    p.add_argument("--lr_scheduler_patience", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--optim",
        type=str,
        default="adamw",
        choices=("adamw", "adam", "sgd", "sam"),
        help="sam：Sharpness-Aware Minimization，底层为 AdamW，训练步约×2",
    )
    p.add_argument(
        "--sam_rho",
        type=float,
        default=0.05,
        help="SAM 邻域半径 ρ（仅 optim=sam）",
    )
    p.add_argument(
        "--mixup_prob",
        type=float,
        default=0.0,
        help="MixUp 概率（三分类小数据默认 0 关闭，利于硬标签边界；需要正则时可试 0.05~0.1）",
    )
    p.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.2,
        help="Beta(mixup_alpha, mixup_alpha) 采样 λ，与 train_task2_vote 默认一致",
    )
    p.add_argument(
        "--class_weight",
        type=str,
        default="none",
        choices=("none", "auto", "manual"),
        help=(
            "auto：CE 按类频反比加权；manual：用 --class_weight_manual 指定 Low,Medium,High 权重。"
            "默认 none。与平衡采样叠加请谨慎。"
        ),
    )
    p.add_argument(
        "--class_weight_manual",
        type=str,
        default="1.0,2.0,1.0",
        help="class_weight=manual 时，类 0/1/2（Low/Medium/High）权重，逗号分隔，会再按均值归一",
    )
    p.add_argument(
        "--sampler_medium_boost",
        type=float,
        default=1.0,
        help="WeightedRandomSampler 中对 Medium(类1) 额外乘的采样权重（>1 过采样 Medium）",
    )
    p.add_argument("--seed", type=int, default=42, help="随机种子（DataLoader worker 仍可能有少量非确定性）")
    p.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW 权重衰减；计划书建议可试 1e-3")
    p.add_argument(
        "--scheduler_min_lr",
        type=float,
        default=0.0,
        help="ReduceLROnPlateau 的 min_lr；cosine 时为 eta_min",
    )
    p.add_argument(
        "--lr_scheduler",
        type=str,
        default="plateau",
        choices=("plateau", "cosine", "step"),
        help=(
            "plateau：ReduceLROnPlateau（按 val loss）；cosine：余弦退火；"
            "step：StepLR（每 lr_step_size epoch 乘 lr_gamma）"
        ),
    )
    p.add_argument(
        "--lr_step_size",
        type=int,
        default=50,
        help="lr_scheduler=step 时的周期（epoch）",
    )
    p.add_argument(
        "--lr_gamma",
        type=float,
        default=0.1,
        help="lr_scheduler=step 时的 lr 乘子",
    )
    p.add_argument(
        "--glevel_arch",
        type=str,
        default="shared_mlp",
        choices=("shared_mlp", "text_gru", "text_mlp", "audio_text_mlp"),
        help=(
            "shared_mlp：原多模态；text_gru/text_mlp：仅用文本；"
            "audio_text_mlp：仅音频+文本+简单 MLP（学长简化基线）"
        ),
    )
    p.add_argument(
        "--fused_layer_norm",
        action="store_true",
        help=(
            "shared_mlp：融合向量进分类头前 LayerNorm(fused_dim)；"
            "audio_text_mlp：拼接后 LayerNorm(2*base_dim)。旧 checkpoint 勿开"
        ),
    )
    p.add_argument(
        "--at_mlp_hidden",
        type=int,
        default=512,
        help="audio_text_mlp 分类头隐层维",
    )
    p.add_argument(
        "--mlp_dropout",
        type=float,
        default=0.0,
        help="SharedMLPwEnsemble 各分支 MLP 内 Dropout（0~0.5）",
    )
    p.add_argument(
        "--num_heads",
        type=int,
        default=32,
        help="SharedMLPwEnsemble 集成分支数（--simple_clf 时忽略）",
    )
    p.add_argument(
        "--base_dim",
        type=int,
        default=768,
        help="多模态投影维数；须能被 --cross_modal_nhead 整除（启用 cross_modal_attn 时）",
    )
    p.add_argument(
        "--mlp_bottleneck_dim",
        type=int,
        default=32,
        help="各 ensemble 分支内瓶颈维（原硬编码 32）",
    )
    p.add_argument(
        "--simple_clf",
        action="store_true",
        help="简化头：不做 ensemble，在 fused 向量时间均值后单层 Linear（仅 CE，与 CORAL/temporal_gru 互斥）",
    )
    p.add_argument(
        "--no_hand",
        action="store_true",
        help="手工特征置零；模型仍使用 hand_dim=4 以与结构一致",
    )
    p.add_argument("--gru_hidden_dim", type=int, default=128)
    p.add_argument("--gru_dropout", type=float, default=0.3)
    p.add_argument("--gru_num_layers", type=int, default=1)
    p.add_argument(
        "--gru_inter_dropout",
        type=float,
        default=0.0,
        help="GRU 层数>1 时层间 dropout",
    )
    p.add_argument(
        "--text_gru_pool",
        type=str,
        default="last",
        choices=("last", "mean"),
        help="text_gru：用最后隐状态或时间均值",
    )
    p.add_argument("--text_mlp_hidden", type=int, default=512)
    p.add_argument("--text_mlp_dropout", type=float, default=0.3)
    p.add_argument(
        "--freeze_text_features",
        action="store_true",
        help="text 输入 detach，仅训练 GRU/MLP 头（需 Nanbeige 已离线提取）",
    )
    p.add_argument(
        "--no_balanced_sampler",
        action="store_true",
        help="关闭 WeightedRandomSampler（默认开启：按类反比频次重采样，不含 CE 加权）",
    )
    p.add_argument(
        "--select_best",
        type=str,
        default="macro_f1",
        choices=("val_ce", "macro_f1", "balanced_acc"),
        help=(
            "保存 best checkpoint / 早停所依据的验证指标。"
            "macro_f1（默认）在 val 仅含部分类时只对出现类求宏平均；"
            "平局时优先 val_ce 更低。仍可用 val_ce / balanced_acc"
        ),
    )
    p.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="CrossEntropyLoss 的 label_smoothing，0 关闭；建议试 0.05~0.1",
    )
    p.add_argument(
        "--temporal_gru",
        action="store_true",
        help="在 6 题 fused 向量序列上先过一层 GRU，再送入 ensemble（默认仍用时间维均值）",
    )
    p.add_argument(
        "--temporal_pool",
        type=str,
        default="mean",
        choices=("mean", "last"),
        help="GRU 输出沿时间维池化：mean 或取最后一步",
    )
    p.add_argument(
        "--temporal_dropout",
        type=float,
        default=0.1,
        help="GRU 前后 Dropout 概率（仅 temporal_gru 开启时生效）",
    )
    p.add_argument(
        "--temporal_bidirectional",
        action="store_true",
        help="temporal_gru 时使用双向 GRU，输出经 Linear 投回 fused_dim（仅 shared_mlp）",
    )
    p.add_argument(
        "--temporal_attn_pool",
        action="store_true",
        help="对 GRU 逐题输出做 softmax 注意力加权求和（替代 mean/last；仍可与 bidirectional 同用）",
    )
    p.add_argument(
        "--question_pos_embed_dim",
        type=int,
        default=0,
        help=">0 时为每题加可学习位置嵌入并经 Linear 压回 fused_dim；0 关闭",
    )
    p.add_argument(
        "--temporal_step_dropout_p",
        type=float,
        default=0.0,
        help="训练时按时间步随机整步置零 fused 特征的概率（与 modality_dropout_p 不同）",
    )
    p.add_argument(
        "--modality_dropout_p",
        type=float,
        default=0.0,
        help=(
            "训练时按样本以概率 p 整段抹零某一模态（video/text/audio 三选一，hand 保留），"
            "0 关闭；可试 0.1~0.2 增强鲁棒性"
        ),
    )
    p.add_argument(
        "--cross_modal_attn",
        action="store_true",
        help=(
            "在拼接前对 video/text/audio 三向量做 TransformerEncoder（3 token 自注意力，"
            "即跨模态交互），输出维不变，可与 temporal_gru 叠加"
        ),
    )
    p.add_argument(
        "--cross_modal_layers",
        type=int,
        default=1,
        help="跨模态 TransformerEncoder 层数（仅 cross_modal_attn 时生效）",
    )
    p.add_argument(
        "--cross_modal_nhead",
        type=int,
        default=8,
        help="跨模态注意力头数，须整除 --base_dim（仅 cross_modal_attn 时生效）",
    )
    p.add_argument(
        "--cross_modal_dropout",
        type=float,
        default=0.1,
        help="跨模态子层 dropout（仅 cross_modal_attn 时生效）",
    )
    p.add_argument(
        "--cross_modal_ff_mult",
        type=int,
        default=4,
        help="FFN 隐层为 base_dim×该倍数（仅 cross_modal_attn 时生效）",
    )
    p.add_argument(
        "--head_weights",
        action="store_true",
        help="32 个 ensemble MLP 头用 softmax 可学习加权（默认等权 mean；与 --simple_clf 互斥）",
    )
    p.add_argument(
        "--time_weights",
        action="store_true",
        help="6 题 per-step logits 用 softmax 可学习加权（仅非 temporal_gru 路径；与 GRU 时序聚合重叠时仅告警）",
    )
    p.add_argument(
        "--text_enhancer",
        type=str,
        default="none",
        choices=("none", "transformer", "mlp"),
        help="文本投影到 base_dim 后：none | 1层Transformer(4头) | 残差MLP（再进入模态dropout/跨模态）",
    )
    p.add_argument(
        "--text_enhancer_dim",
        type=int,
        default=512,
        help="text_enhancer 的 FFN/MLP 隐层维（transformer 的 dim_feedforward 或 mlp 隐藏维）",
    )
    p.add_argument(
        "--print_aggregation_weights",
        action="store_true",
        help="训练结束加载 best 后打印 head/time 的 softmax 权重（需对应模块已启用）",
    )

    p.add_argument("--only_test", action="store_true")
    p.add_argument(
        "--ensemble_checkpoints",
        nargs="*",
        default=None,
        help=(
            "仅 only_test：多份 .pth 路径，对 logits/概率取平均后再评估 val 与 predict_test；"
            "不传时仍用 --test_model 单模型。"
        ),
    )
    p.add_argument("--test_model", type=str, default="best_model_glevel.pth")
    p.add_argument("--test_output_csv", type=str, default="submission_glevel.csv")
    p.add_argument(
        "--val_errors_csv",
        type=str,
        default="",
        help=(
            "非空则对当前 val_loader 逐样本写出预测/概率/CE，并合并 val_csv 中的列；"
            "训练结束或 --only_test 加载权重后执行。用于分析错分模式"
        ),
    )
    p.add_argument("--write_pred_names", action="store_true", help="导出 Low/Medium/High 字符串列")

    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument(
        "--glevel_loss",
        type=str,
        default="ce",
        choices=("ce", "coral"),
        help=(
            "ce：常规 CrossEntropy；coral：三分类序关系 CORAL（K-1 个 logits + BCE），"
            "利于压 Low↔High 等跨档错分；推理时仍输出 3 类概率。仅 num_classes=3。"
        ),
    )
    p.add_argument("--output_model", type=str, default="best_model_glevel.pth")
    p.add_argument("--loss_plot_path", type=str, default="./loss_img/loss_glevel.png")
    p.add_argument("--log_dir", type=str, default="./logs")
    p.add_argument("--modalities", type=str, default="audio,video,text")
    p.add_argument(
        "--swa_start_epoch",
        type=int,
        default=0,
        help="从第几个 epoch（1-based）起做 SWA 平均；0 关闭。结束后 update_bn 并写入 output_swa_model",
    )
    p.add_argument(
        "--swa_lr",
        type=float,
        default=1e-4,
        help="SWALR 固定学习率（SWA 阶段）",
    )
    p.add_argument(
        "--output_swa_model",
        type=str,
        default="",
        help="SWA 权重保存路径；默认同 output_model 加后缀 .swa.pth",
    )
    p.add_argument(
        "--tta_times",
        type=int,
        default=0,
        help="测试/only_test 时 TTA 次数；0 关闭。对特征加高斯噪声多次前向平均概率",
    )
    p.add_argument(
        "--tta_noise_std",
        type=float,
        default=0.01,
        help="TTA 高斯噪声标准差",
    )
    p.add_argument(
        "--logit_temperature",
        type=float,
        default=1.0,
        help="仅推理/only_test：对 CE logits 除以 T 再 softmax（温度缩放；默认 1 关闭）",
    )
    p.add_argument(
        "--infer_logit_bias",
        type=str,
        default="",
        help=(
            "推理用（验证评估与 predict_test）：对 CE logits 在温度缩放之前加上各类常数偏置，"
            "格式为三个浮点逗号分隔，顺序对应类下标 0/1/2 即 Low/Medium/High。"
            "用于缓解测试集上 Medium 被 argmax 压制；建议在验证集上扫 grid（如 0,b,0）。"
            "与 --glevel_loss coral 互斥。"
        ),
    )
    p.add_argument(
        "--calib_temperature_json",
        type=str,
        default="",
        help="若存在则读取 {\"T\": float} 覆盖 --logit_temperature（由 tools/fit_temperature_scaling.py 写出）",
    )
    p.add_argument(
        "--feat_norm_npz",
        type=str,
        default="",
        help="含 audio_mu/std, video_mu/std, text_mu/std 的 npz（由 tools/compute_feat_mean_std.py 生成）",
    )
    p.add_argument(
        "--feat_norm_apply",
        type=str,
        default="none",
        choices=("none", "all"),
        help="none：不用；all：各 split 加载特征后按训练集统计量标准化",
    )
    p.add_argument(
        "--feat_norm_eps",
        type=float,
        default=1e-6,
        help="标准化时分母加 eps",
    )

    args = p.parse_args()
    cj = (args.calib_temperature_json or "").strip()
    if cj:
        import json

        with open(cj, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        args.logit_temperature = float(data.get("T", args.logit_temperature))
        print(
            f"[train_task2_glevel] calib_temperature_json → logit_temperature={args.logit_temperature}",
            flush=True,
        )
    init_seed(args.seed)
    args.glevel_csv = (args.glevel_csv or "").strip()
    args.val_audio_dir = (args.val_audio_dir or "").strip() or args.audio_dir
    args.val_video_dir = (args.val_video_dir or "").strip() or args.video_dir
    args.val_text_dir = (args.val_text_dir or "").strip() or args.text_dir
    args.test_audio_dir = (args.test_audio_dir or "").strip() or args.audio_dir
    args.test_video_dir = (args.test_video_dir or "").strip() or args.video_dir
    args.test_text_dir = (args.test_text_dir or "").strip() or args.text_dir
    if getattr(args, "train_fallback_use_test_features", False) and not args.train_feat_fallback:
        raise SystemExit(
            "--train_fallback_use_test_features 须与 --train_feat_fallback 同时使用"
        )
    if len(args.label_col) != 1:
        raise SystemExit("分类任务请只指定一列，例如: --label_col g_level")
    args.g_level_int_encoding = str(args.g_level_int_encoding or "one").lower()
    args.glevel_loss = str(args.glevel_loss or "ce").lower()
    if args.glevel_loss == "coral":
        if args.num_classes != 3:
            raise SystemExit("--glevel_loss coral 仅支持 num_classes=3")
        args.target_dim = args.num_classes - 1
        if float(args.label_smoothing or 0) > 0:
            print(
                "[train_task2_glevel] glevel_loss=coral：label_smoothing 与 CORAL 未联合实现，已置 0",
                flush=True,
            )
            args.label_smoothing = 0.0
        if float(args.mixup_prob or 0) > 0:
            print("[train_task2_glevel] glevel_loss=coral：已关闭 MixUp", flush=True)
            args.mixup_prob = 0.0
    else:
        args.target_dim = args.num_classes
    args.modalities = [m.strip() for m in args.modalities.split(",")]
    args.classification = True
    if args.glevel_arch == "shared_mlp":
        if args.simple_clf:
            if args.glevel_loss == "coral":
                raise SystemExit("--simple_clf 与 --glevel_loss coral 互斥，请使用 --glevel_loss ce")
            if args.temporal_gru:
                raise SystemExit("--simple_clf 与 --temporal_gru 互斥")
        if getattr(args, "cross_modal_attn", False):
            nh = int(getattr(args, "cross_modal_nhead", 8))
            if args.base_dim % nh != 0:
                raise SystemExit(
                    f"--base_dim={args.base_dim} 须能被 --cross_modal_nhead={nh} 整除（跨模态注意力要求）"
                )
        if args.simple_clf and args.head_weights:
            raise SystemExit("--head_weights 与 --simple_clf 互斥（简化头无 ensemble）")
        if args.temporal_gru and args.time_weights:
            print(
                "[train_task2_glevel] 警告: --time_weights 仅在非 temporal_gru 路径生效；"
                "当前 temporal_gru 已做时序聚合，time_logits 未使用。",
                flush=True,
            )
        te = str(args.text_enhancer or "none").lower()
        if te == "transformer" and args.base_dim % 4 != 0:
            raise SystemExit(
                f"--text_enhancer transformer 要求 --base_dim 能被 4 整除，当前 base_dim={args.base_dim}"
            )
    elif args.glevel_arch == "audio_text_mlp":
        if getattr(args, "cross_modal_attn", False):
            raise SystemExit("audio_text_mlp 与 --cross_modal_attn 互斥，请关闭跨模态注意力")
        if args.temporal_gru or args.simple_clf:
            raise SystemExit("audio_text_mlp 不支持 --temporal_gru / --simple_clf")
        if args.head_weights or args.time_weights:
            raise SystemExit("audio_text_mlp 不支持 --head_weights / --time_weights")
        te_at = str(args.text_enhancer or "none").lower()
        if te_at != "none":
            raise SystemExit("audio_text_mlp 不支持 --text_enhancer")
    else:
        teo = str(getattr(args, "text_enhancer", "none") or "none").lower()
        if args.head_weights or args.time_weights or teo != "none":
            raise SystemExit(
                "--head_weights / --time_weights / --text_enhancer 仅适用于 --glevel_arch shared_mlp"
            )
    if (args.temporal_bidirectional or args.temporal_attn_pool) and not args.temporal_gru:
        print(
            "[train_task2_glevel] 警告: --temporal_bidirectional / --temporal_attn_pool 需配合 "
            "--temporal_gru，已忽略双向与注意力池化。",
            flush=True,
        )
        args.temporal_bidirectional = False
        args.temporal_attn_pool = False
    if args.swa_start_epoch > 0:
        osw = (args.output_swa_model or "").strip()
        args.output_swa_model = osw or (args.output_model + ".swa.pth")
    save_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    infer_logit_bias_t = _parse_infer_logit_bias(
        args.infer_logit_bias, args.num_classes, device
    )
    if infer_logit_bias_t is not None and args.glevel_loss == "coral":
        raise SystemExit("--infer_logit_bias 与 --glevel_loss coral 不能同时使用")
    if infer_logit_bias_t is not None:
        print(
            "[train_task2_glevel] infer_logit_bias (顺序 Low/Medium/High 类下标 0/1/2)=",
            infer_logit_bias_t.detach().cpu().tolist(),
            flush=True,
        )

    _fb_a = None if args.no_feature_fallback else args.audio_dir
    _fb_v = None if args.no_feature_fallback else args.video_dir
    _fb_t = None if args.no_feature_fallback else args.text_dir

    _tr_fb_a = _tr_fb_v = _tr_fb_t = None
    if args.train_feat_fallback and not args.no_feature_fallback:
        if getattr(args, "train_fallback_use_test_features", False):
            _tr_fb_a = args.test_audio_dir
            _tr_fb_v = args.test_video_dir
            _tr_fb_t = args.test_text_dir
            print(
                "[train_task2_glevel] train_feat_fallback → 使用测试集特征目录作为训练集回退",
                flush=True,
            )
        else:
            _tr_fb_a = args.val_audio_dir
            _tr_fb_v = args.val_video_dir
            _tr_fb_t = args.val_text_dir

    train_set = MultimodalDatasetForTrainT2(
        args.train_csv,
        args.audio_dir,
        args.video_dir,
        args.text_dir,
        args.question,
        args.label_col,
        args.rating_csv,
        args,
        fallback_audio_dir=_tr_fb_a,
        fallback_video_dir=_tr_fb_v,
        fallback_text_dir=_tr_fb_t,
    )
    val_set = MultimodalDatasetForTrainT2(
        args.val_csv,
        args.val_audio_dir,
        args.val_video_dir,
        args.val_text_dir,
        args.question,
        args.label_col,
        args.rating_csv,
        args,
        fallback_audio_dir=_fb_a,
        fallback_video_dir=_fb_v,
        fallback_text_dir=_fb_t,
    )
    _test_fv_a = _test_fv_v = _test_fv_t = None
    if getattr(args, "test_fallback_val_features", False) and not args.no_feature_fallback:
        _test_fv_a = args.val_audio_dir
        _test_fv_v = args.val_video_dir
        _test_fv_t = args.val_text_dir
        print(
            "[train_task2_glevel] test_fallback_val_features=1 → 测试特征链: test → train 回退 → val 回退",
            flush=True,
        )

    test_set = MultimodalDatasetForTestT2(
        args.test_csv,
        args.test_audio_dir,
        args.test_video_dir,
        args.test_text_dir,
        args.question,
        args.rating_csv,
        args,
        fallback_audio_dir=_fb_a,
        fallback_video_dir=_fb_v,
        fallback_text_dir=_fb_t,
        fallback_val_audio_dir=_test_fv_a,
        fallback_val_video_dir=_test_fv_v,
        fallback_val_text_dir=_test_fv_t,
    )

    _maybe_autofix_g_level_int_encoding(args, train_set, val_set)

    _log_split_label_counts("train", train_set)
    _log_split_label_counts("val", val_set)
    print(
        f"[train_task2_glevel] 标签整数编码: --g_level_int_encoding={args.g_level_int_encoding!r}",
        flush=True,
    )

    ys_train, cnt_train = _collect_train_labels(train_set)
    ce_weight = None
    if args.class_weight == "auto":
        ce_weight = _ce_weights_from_counts(
            cnt_train, args.num_classes, len(train_set.data), device
        )
        w_list = ce_weight.detach().cpu().tolist()
        print(
            f"[train_task2_glevel] CE class_weight (auto, mean=1) 类频 {dict(sorted(cnt_train.items()))} "
            f"→ weights {w_list}",
            flush=True,
        )
    elif args.class_weight == "manual":
        parts = [float(x.strip()) for x in args.class_weight_manual.split(",")]
        if len(parts) != args.num_classes:
            raise SystemExit(
                f"--class_weight_manual 须含 {args.num_classes} 个权重，当前 {len(parts)} 个: {parts!r}"
            )
        w_t = torch.tensor(parts, dtype=torch.float32, device=device)
        w_t = w_t / w_t.mean()
        ce_weight = w_t
        print(
            f"[train_task2_glevel] CE class_weight (manual, mean=1) → {w_t.detach().cpu().tolist()}",
            flush=True,
        )
    else:
        print(
            "[train_task2_glevel] class_weight=none（均匀 CE）；"
            "若需少数类更大梯度可再加 --class_weight auto/manual（与平衡采样叠加请谨慎）",
            flush=True,
        )

    if args.sampler_medium_boost != 1.0:
        inv_per_sample = _inv_freq_sample_weights_medium_boost(
            ys_train, cnt_train, args.sampler_medium_boost
        )
        print(
            f"[train_task2_glevel] sampler_medium_boost={args.sampler_medium_boost}（仅 WeightedRandomSampler）",
            flush=True,
        )
    else:
        inv_per_sample = _inv_freq_sample_weights(ys_train, cnt_train)

    pin_mem = torch.cuda.is_available()
    _worker_kw: dict = {}
    if int(args.num_workers) > 0:
        _worker_kw["worker_init_fn"] = _make_worker_init_fn(int(args.seed))
    _tl_kw = dict(
        batch_size=args.batch_size,
        collate_fn=collate_fn_train,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        **_worker_kw,
    )
    use_balanced_sampler = not args.no_balanced_sampler
    if use_balanced_sampler and args.class_weight in ("auto", "manual"):
        print(
            "[train_task2_glevel] WARNING: WeightedRandomSampler 与 "
            f"--class_weight {args.class_weight} 同时启用，"
            "少数类可能在采样与 CE 中被双重点名，易导致验证塌缩。"
            "建议只保留其一，或仅用 --sampler_medium_boost 过采样 Medium。",
            flush=True,
        )
    if use_balanced_sampler:
        sampler = WeightedRandomSampler(
            torch.DoubleTensor(inv_per_sample),
            num_samples=len(inv_per_sample),
            replacement=True,
        )
        train_loader = DataLoader(train_set, sampler=sampler, shuffle=False, **_tl_kw)
        print(
            "[train_task2_glevel] WeightedRandomSampler 已启用（按类反比频次重采样，batch 内近似类平衡）",
            flush=True,
        )
    else:
        train_loader = DataLoader(train_set, shuffle=True, **_tl_kw)
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_train,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        **_worker_kw,
    )
    test_loader = None
    if len(test_set) > 0:
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_test,
            num_workers=args.num_workers,
            pin_memory=pin_mem,
            **_worker_kw,
        )

    criterion = nn.CrossEntropyLoss(weight=ce_weight, label_smoothing=args.label_smoothing)
    if args.glevel_loss == "coral":
        print(
            "[train_task2_glevel] glevel_loss=coral：验证集 loss 为 CORAL 平均 BCE（非 softmax CE）",
            flush=True,
        )

    def _build_model():
        if args.glevel_arch == "text_gru":
            return TextGRUClassifier(args).to(device)
        if args.glevel_arch == "text_mlp":
            return TextOnlyMLPClassifier(args).to(device)
        if args.glevel_arch == "audio_text_mlp":
            return AudioTextMLPClassifier(args).to(device)
        return SharedMLPwEnsemble(args).to(device)

    if args.only_test:
        ckpts = list(args.ensemble_checkpoints or [])
        if not ckpts:
            ckpts = [args.test_model]
        models: list = []
        for path in ckpts:
            try:
                state = torch.load(path, map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(path, map_location=device)
            m = _build_model()
            m.load_state_dict(state)
            models.append(m)
        print(f"[only_test] checkpoints={len(models)} paths={ckpts}", flush=True)
        if len(models) == 1:
            val_loss, val_acc, val_mf1, val_bacc, val_nuniq = evaluate_epoch(
                models[0],
                val_loader,
                criterion,
                device,
                args.num_classes,
                glevel_loss=args.glevel_loss,
                logit_temperature=args.logit_temperature,
                infer_logit_bias=infer_logit_bias_t,
            )
        else:
            if args.glevel_loss != "ce":
                raise SystemExit("[only_test] --ensemble_checkpoints 仅支持 glevel_loss=ce")
            val_loss, val_acc, val_mf1, val_bacc, val_nuniq = evaluate_epoch_ensemble(
                models,
                val_loader,
                criterion,
                device,
                args.num_classes,
                glevel_loss=args.glevel_loss,
                logit_temperature=args.logit_temperature,
                infer_logit_bias=infer_logit_bias_t,
            )
        tag = "ensemble" if len(models) > 1 else "single"
        print(
            f"[only_test:{tag}] Val CE={val_loss:.4f} acc={val_acc:.4f} "
            f"macro_f1={val_mf1:.4f} bal_acc={val_bacc:.4f} "
            f"val_pred_classes={val_nuniq}"
        )
        maybe_report_val_errors(args, models[0], val_loader, val_set, device)
        if len(models) == 1:
            predict_test(
                models[0],
                test_loader,
                device,
                args.test_csv,
                args.test_output_csv,
                args.write_pred_names,
                glevel_loss=args.glevel_loss,
                tta_times=args.tta_times,
                tta_noise_std=args.tta_noise_std,
                logit_temperature=args.logit_temperature,
                infer_logit_bias=infer_logit_bias_t,
            )
        else:
            predict_test_ensemble(
                models,
                test_loader,
                device,
                args.test_csv,
                args.test_output_csv,
                args.write_pred_names,
                glevel_loss=args.glevel_loss,
                tta_times=args.tta_times,
                tta_noise_std=args.tta_noise_std,
                logit_temperature=args.logit_temperature,
                infer_logit_bias=infer_logit_bias_t,
            )
        return

    model = _build_model()
    print(
        f"[train_task2_glevel] glevel_arch={args.glevel_arch} glevel_loss={args.glevel_loss} "
        f"seed={args.seed}",
        flush=True,
    )

    n_train_s = len(train_set)
    n_val_s = len(val_set)
    steps_pe = max((n_train_s + args.batch_size - 1) // max(args.batch_size, 1), 1)
    print(
        f"[train_task2_glevel] 数据与迭代: train_samples={n_train_s} val_samples={n_val_s} "
        f"batch_size={args.batch_size} steps_per_epoch≈{steps_pe} num_workers={args.num_workers}",
        flush=True,
    )
    es_min = int(args.early_stop_min_epochs)
    es_pat = int(args.early_stop_patience)
    print(
        f"[train_task2_glevel] 早停: enabled={not args.no_early_stop} patience={es_pat} "
        f"min_epochs_before_stop={es_min} num_epochs_cap={args.num_epochs} select_best={args.select_best}",
        flush=True,
    )

    use_sam = False
    if args.optim == "sam":
        from model.vote_model.sam import SAM

        optimizer = SAM(
            model.parameters(),
            torch.optim.AdamW,
            rho=float(args.sam_rho),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        use_sam = True
        print(
            f"[train_task2_glevel] optim=SAM rho={args.sam_rho} base=AdamW",
            flush=True,
        )
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    elif args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    swa_model = None
    swa_scheduler = None
    if int(args.swa_start_epoch) > 0:
        from torch.optim.swa_utils import AveragedModel, SWALR

        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=float(args.swa_lr))
        print(
            f"[train_task2_glevel] SWA: start_epoch={args.swa_start_epoch} "
            f"swa_lr={args.swa_lr} → {args.output_swa_model}",
            flush=True,
        )

    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(args.num_epochs), 1),
            eta_min=max(float(args.scheduler_min_lr), 0.0),
        )
        print(
            f"[train_task2_glevel] lr_scheduler=cosine T_max={args.num_epochs} eta_min={args.scheduler_min_lr}",
            flush=True,
        )
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, int(args.lr_step_size)),
            gamma=float(args.lr_gamma),
        )
        print(
            f"[train_task2_glevel] lr_scheduler=step step_size={args.lr_step_size} "
            f"gamma={args.lr_gamma}",
            flush=True,
        )
    else:
        min_lr = float(args.scheduler_min_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args.lr_scheduler_patience,
            min_lr=min_lr if min_lr > 0 else 0.0,
        )
        print(
            f"[train_task2_glevel] lr_scheduler=plateau(ReduceLROnPlateau) min_lr={min_lr}",
            flush=True,
        )

    select = args.select_best
    # val_ce：标量越小越好；macro_f1 / balanced_acc：(主指标, -val_ce) 元组越大越好，平局看 CE
    best_scalar: float | None = None
    best_tuple: tuple[float, float] | None = None

    def _is_better_ce(cur_ce: float) -> bool:
        nonlocal best_scalar
        if best_scalar is None or cur_ce < best_scalar - 1e-8:
            best_scalar = cur_ce
            return True
        return False

    def _is_better_tuple(primary: float, cur_ce: float) -> bool:
        nonlocal best_tuple
        key = (primary, -cur_ce)
        if best_tuple is None or key > best_tuple:
            best_tuple = key
            return True
        return False

    stall = 0
    collapse_streak = 0
    train_losses, val_losses = [], []
    best_epoch = 0
    early_stopped = False
    best_metrics: dict[str, float | int] = {}

    for epoch in trange(args.num_epochs, desc="Epochs", ncols=100):
        tr_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            mixup_prob=args.mixup_prob,
            mixup_alpha=args.mixup_alpha,
            num_classes=args.num_classes,
            glevel_loss=args.glevel_loss,
            use_sam=use_sam,
        )
        va_loss, va_acc, va_mf1, va_bacc, va_nuniq = evaluate_epoch(
            model,
            val_loader,
            criterion,
            device,
            args.num_classes,
            glevel_loss=args.glevel_loss,
        )
        if va_nuniq <= 1:
            collapse_streak += 1
            if collapse_streak == 3:
                tqdm.write(
                    "[train_task2_glevel] WARNING: 已连续 3 个 epoch 满足 val_pred_classes<=1，"
                    "模型在验证集上几乎只输出单一类别。请检查："
                    "① --g_level_int_encoding 是否与 CSV 整数标签一致；"
                    "② 是否同时开启平衡采样与 --class_weight auto；"
                    "③ 用 --val_errors_csv 查看各类 prob 与 margin_top2；④ 特征维数与覆盖。"
                )
            elif collapse_streak > 3 and collapse_streak % 10 == 0:
                tqdm.write(
                    f"[train_task2_glevel] WARNING: val 单类预测已持续 {collapse_streak} epoch，"
                    "仍建议按上条排查。"
                )
        else:
            collapse_streak = 0
        in_swa = swa_model is not None and (epoch + 1) >= int(args.swa_start_epoch)
        if in_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif args.lr_scheduler in ("cosine", "step"):
            scheduler.step()
        else:
            scheduler.step(va_loss)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if select == "val_ce":
            improved = _is_better_ce(va_loss)
        elif select == "macro_f1":
            improved = _is_better_tuple(va_mf1, va_loss)
        else:
            improved = _is_better_tuple(va_bacc, va_loss)

        if improved:
            stall = 0
            best_epoch = epoch + 1
            save_model(model, args.output_model)
            best_metrics = {
                "val_ce": float(va_loss),
                "val_acc": float(va_acc),
                "val_macro_f1": float(va_mf1),
                "val_bal_acc": float(va_bacc),
                "val_pred_classes": int(va_nuniq),
            }
        else:
            stall += 1

        cur_lr = float(optimizer.param_groups[0]["lr"])
        tqdm.write(
            f"[Epoch {epoch + 1}/{args.num_epochs}] lr={cur_lr:.3e} "
            f"stall={stall}/{args.early_stop_patience} best_epoch={best_epoch} "
            f"train_ce={tr_loss:.4f} val_ce={va_loss:.4f} val_acc={va_acc:.4f} "
            f"val_macro_f1={va_mf1:.4f} val_bal_acc={va_bacc:.4f} "
            f"val_pred_classes={va_nuniq} improved={'yes' if improved else 'no'} "
            f"(select_best={select})"
        )
        min_ep = int(args.early_stop_min_epochs)
        reached_min_epochs = (epoch + 1) >= min_ep
        if (
            not args.no_early_stop
            and stall >= args.early_stop_patience
            and reached_min_epochs
        ):
            tqdm.write(
                f"[early_stop] 触发于 epoch {epoch + 1}/{args.num_epochs}："
                f"已连续 {stall} 个 epoch 未提升 select_best={select} "
                f"（耐心阈值 early_stop_patience={args.early_stop_patience}）。"
            )
            tqdm.write(
                f"[early_stop] 当前学习率 lr={cur_lr:.3e}；"
                f"early_stop_min_epochs={min_ep}（仅当 epoch≥此项时才允许早停，本轮已满足）。"
            )
            if best_metrics:
                tqdm.write(
                    f"[early_stop] 历史最佳 checkpoint 在 epoch {best_epoch}: "
                    f"val_ce={best_metrics['val_ce']:.4f} val_acc={best_metrics['val_acc']:.4f} "
                    f"val_macro_f1={best_metrics['val_macro_f1']:.4f} "
                    f"val_bal_acc={best_metrics['val_bal_acc']:.4f} "
                    f"val_pred_classes={best_metrics['val_pred_classes']}"
                )
            else:
                tqdm.write("[early_stop] 警告: 未记录到 best_metrics（不应发生）。")
            tqdm.write(
                f"[early_stop] 若希望继续训满：可加 --no_early_stop 或增大 --early_stop_patience / "
                f"调高 --early_stop_min_epochs；小 val 可试 --select_best val_ce 或 --lr_scheduler cosine。"
            )
            early_stopped = True
            break
        if (
            not args.no_early_stop
            and stall >= args.early_stop_patience
            and not reached_min_epochs
            and stall == args.early_stop_patience
        ):
            tqdm.write(
                f"[early_stop] stall 已达耐心阈值（{args.early_stop_patience}），"
                f"但当前 epoch {epoch + 1} < early_stop_min_epochs={min_ep}，"
                f"本次不触发早停；达到第 {min_ep} epoch 后若仍无改善将结束。"
            )

    epochs_run = epoch + 1
    swa_ran = swa_model is not None and epochs_run >= int(args.swa_start_epoch)
    if swa_ran:
        from torch.optim.swa_utils import update_bn

        try:
            update_bn(train_loader, swa_model, device=device)
        except Exception as e:
            # 多模态 batch 为 dict 时 PyTorch 自带 update_bn 不适用，跳过 BN 校准仍保留 SWA 权重
            print(f"[swa] update_bn 跳过（多模态/非标准 batch）: {e}", flush=True)
        mod = getattr(swa_model, "module", swa_model)
        torch.save(mod.state_dict(), args.output_swa_model)
        swa_vl, swa_va, swa_mf1, swa_bacc, swa_nu = evaluate_epoch(
            swa_model,
            val_loader,
            criterion,
            device,
            args.num_classes,
            glevel_loss=args.glevel_loss,
        )
        print(
            f"[swa_eval] Val CE={swa_vl:.4f} acc={swa_va:.4f} macro_f1={swa_mf1:.4f} "
            f"bal_acc={swa_bacc:.4f} val_pred_classes={swa_nu} | saved {args.output_swa_model}",
            flush=True,
        )

    if args.loss_plot_path:
        save_loss_plot(train_losses, val_losses, args.loss_plot_path)

    try:
        sd = torch.load(
            args.output_model, map_location=device, weights_only=True
        )
    except TypeError:
        sd = torch.load(args.output_model, map_location=device)
    model.load_state_dict(sd)
    _print_learned_aggregation_if_any(model, args)
    val_loss, val_acc, val_mf1, val_bacc, val_nuniq = evaluate_epoch(
        model,
        val_loader,
        criterion,
        device,
        args.num_classes,
        glevel_loss=args.glevel_loss,
    )
    print(
        f"Best checkpoint (select_best={args.select_best}) | "
        f"Val CE: {val_loss:.4f} | acc: {val_acc:.4f} | "
        f"macro_f1: {val_mf1:.4f} | bal_acc: {val_bacc:.4f} | "
        f"val_pred_classes: {val_nuniq}"
    )
    print(
        f"[train_summary] epochs_run={epochs_run} best_epoch={best_epoch} "
        f"early_stop={'triggered' if early_stopped else 'not_triggered'}"
        f"{' (--no_early_stop)' if args.no_early_stop else ''} | "
        f"early_stop_patience={args.early_stop_patience} "
        f"early_stop_min_epochs={args.early_stop_min_epochs} | "
        f"select_best={args.select_best} | mixup_prob={args.mixup_prob}",
        flush=True,
    )
    if best_metrics:
        print(
            f"[train_summary] best_checkpoint_metrics: val_ce={best_metrics['val_ce']:.4f} "
            f"val_acc={best_metrics['val_acc']:.4f} val_macro_f1={best_metrics['val_macro_f1']:.4f} "
            f"val_bal_acc={best_metrics['val_bal_acc']:.4f} "
            f"val_pred_classes={best_metrics['val_pred_classes']}",
            flush=True,
        )
    print(
        f"[metrics_line] val_acc={val_acc:.4f} val_macro_f1={val_mf1:.4f} val_bal_acc={val_bacc:.4f} "
        f"best_epoch={best_epoch} epochs_run={epochs_run}",
        flush=True,
    )
    summarize_val_split(
        model,
        val_loader,
        criterion,
        device,
        args.num_classes,
        header="[val_summary]",
        glevel_loss=args.glevel_loss,
    )
    maybe_report_val_errors(args, model, val_loader, val_set, device)
    predict_test(
        model,
        test_loader,
        device,
        args.test_csv,
        args.test_output_csv,
        args.write_pred_names,
        glevel_loss=args.glevel_loss,
        tta_times=args.tta_times,
        tta_noise_std=args.tta_noise_std,
        logit_temperature=args.logit_temperature,
        infer_logit_bias=infer_logit_bias_t,
    )


if __name__ == "__main__":
    main()
