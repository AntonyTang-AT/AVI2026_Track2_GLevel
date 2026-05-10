"""基于「多模型一致」与加权多数票的初始标签与冻结掩码。"""

from __future__ import annotations

import numpy as np


def unanimous_mask(preds: np.ndarray) -> np.ndarray:
    """(n, k) → (n,) bool，该行所有模型预测相同则为 True。"""
    return (preds.min(axis=1) == preds.max(axis=1))


def agreement_count_modal(preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    对每一行统计众数的得票数及众数标签。
    返回 (votes, mode_label)，均为 (n,) int。
    平票时取较小类别号（1<2<3）以固定行为。
    """
    n = preds.shape[0]
    votes = np.zeros(n, dtype=np.int32)
    mode = np.zeros(n, dtype=np.int32)
    for i in range(n):
        row = preds[i]
        best_c, best_cnt = 1, -1
        for c in (1, 2, 3):
            cnt = int(np.sum(row == c))
            if cnt > best_cnt or (cnt == best_cnt and c < best_c):
                best_c, best_cnt = c, cnt
        mode[i] = best_c
        votes[i] = best_cnt
    return votes, mode


def row_consensus_features(preds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (mode_label, max_votes, strength)。
    strength[i] = max_votes/k ∈ [1/k, 1]，重合度越高越大，可作「置信权重」。
    """
    votes, mode = agreement_count_modal(preds)
    k = int(preds.shape[1])
    strength = votes.astype(np.float64) / float(max(k, 1))
    return mode, votes, strength


def uniform_class_targets(n: int) -> np.ndarray:
    """三类目标条数（整数、和为 n），尽可能 1:1:1。"""
    base = n // 3
    r = n % 3
    c = np.array([base, base, base], dtype=np.float64)
    for j in range(r):
        c[j] += 1.0
    return c


def weighted_majority_labels(preds: np.ndarray, model_weights: np.ndarray) -> np.ndarray:
    """
    preds: (n, k) int in {1,2,3}
    model_weights: (k,) 非负，按列加权投票。
    """
    n, k = preds.shape
    if model_weights.shape != (k,):
        raise ValueError("model_weights 形状须为 (n_models,)")
    w = np.asarray(model_weights, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("权重须非负")
    s = w.sum()
    if s <= 0:
        w = np.ones(k, dtype=np.float64)
    else:
        w = w / s

    scores = np.zeros((n, 3), dtype=np.float64)
    for j in range(k):
        col = preds[:, j]
        for cls_idx, cls in enumerate((1, 2, 3)):
            scores[:, cls_idx] += w[j] * (col == cls).astype(np.float64)

    # 平票取较小类
    out = np.ones(n, dtype=np.int32)
    for i in range(n):
        best_c, best_s = 1, -1.0
        for cls_idx, cls in enumerate((1, 2, 3)):
            sc = scores[i, cls_idx]
            if sc > best_s or (abs(sc - best_s) < 1e-12 and cls < best_c):
                best_c, best_s = cls, sc
        out[i] = best_c
    return out


def build_initial_y(
    preds: np.ndarray,
    model_weights: np.ndarray,
    prefer_unanimous: bool = True,
) -> np.ndarray:
    """
    先对全体一致样本采用该一致标签；其余行用加权多数票。
    """
    n = preds.shape[0]
    uni = unanimous_mask(preds)
    wm = weighted_majority_labels(preds, model_weights)
    y = wm.copy()
    if prefer_unanimous:
        for i in range(n):
            if uni[i]:
                y[i] = int(preds[i, 0])
    return y
