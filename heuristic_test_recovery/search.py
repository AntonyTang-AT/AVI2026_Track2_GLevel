"""目标函数与模拟退火搜索（增量损失 + 可选多进程多起点 + 均衡/重合度先验）。"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SearchResult:
    y: np.ndarray
    initial_y: np.ndarray
    matches_per_model: np.ndarray
    loss: float
    steps_accepted: int
    steps_total: int
    seed: int = 0


def count_matches(y: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """y: (n,), preds: (n,k) → (k,) 每模型与 y 一致的条数。"""
    return np.sum(y[:, None] == preds, axis=0).astype(np.int32)


def total_agreement_from_y(y: np.ndarray, preds: np.ndarray) -> int:
    return int(np.sum(y[:, None] == preds))


def squared_loss(matches: np.ndarray, targets: np.ndarray) -> float:
    d = matches.astype(np.float64) - targets.astype(np.float64)
    return float(np.dot(d, d))


def _class_counts(y: np.ndarray) -> np.ndarray:
    """标签 1/2/3 → 长度 3 的计数。"""
    return np.array([(y == j).sum() for j in (1, 2, 3)], dtype=np.float64)


def _balance_penalty(cnt: np.ndarray, mu: np.ndarray) -> float:
    d = cnt - mu
    return float(np.dot(d, d))


def _assemble_loss(
    matches: np.ndarray,
    agree: int,
    targets: np.ndarray | None,
    lambda_agreement: float,
    *,
    lambda_balance: float,
    balance_pen: float,
    lambda_plurality: float,
    plur_sum: float,
) -> float:
    if targets is None:
        sq = 0.0
    else:
        sq = squared_loss(matches, targets)
    return (
        sq
        - lambda_agreement * float(agree)
        + lambda_balance * balance_pen
        - lambda_plurality * plur_sum
    )


def simulate_annealing(
    y0: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray | None,
    *,
    freeze_mask: np.ndarray | None = None,
    steps: int = 80_000,
    seed: int = 0,
    t0: float = 2.0,
    t_min: float = 1e-3,
    lambda_agreement: float = 0.02,
    trace_every: int = 0,
    lambda_balance: float = 0.0,
    balance_mu: np.ndarray | None = None,
    lambda_plurality: float = 0.0,
    mode_labels: np.ndarray | None = None,
    consensus_strength: np.ndarray | None = None,
    proposal_bias: float = 0.0,
) -> SearchResult:
    """
    最小化
      Σ_k (matches_k-target_k)²
      - λ_agree · Σ 模型一致
      + λ_balance · Σ_c (n_c - μ_c)²      （默认 μ 为近似 1:1:1）
      - λ_plur · Σ_i strength_i · 1[y_i=mode_i]   （高重合行更信众票）

    proposal_bias>0：优先在低重合位置提议翻转，减少对高置信行的破坏。
    """
    rng = np.random.default_rng(seed)
    y = y0.copy().astype(np.int32)
    preds = np.asarray(preds, dtype=np.int32)
    n, k = preds.shape
    freeze_mask = np.zeros(n, dtype=bool) if freeze_mask is None else np.asarray(freeze_mask, dtype=bool)

    tgt = None
    if targets is not None:
        tgt = np.asarray(targets, dtype=np.int64).reshape(-1)
        if tgt.shape[0] != k:
            raise ValueError("targets 长度须等于模型数")

    if balance_mu is None:
        base = n // 3
        r = n % 3
        mu = np.array([base, base, base], dtype=np.float64)
        for j in range(r):
            mu[j] += 1.0
    else:
        mu = np.asarray(balance_mu, dtype=np.float64).reshape(3)
        if abs(mu.sum() - n) > 1e-6:
            raise ValueError("balance_mu 三项之和须等于 n")

    if (lambda_plurality > 0 or proposal_bias > 0) and (
        mode_labels is None or consensus_strength is None
    ):
        raise ValueError("启用 plurality / proposal_bias 时须传入 mode_labels 与 consensus_strength")

    matches = count_matches(y, preds)
    agree = total_agreement_from_y(y, preds)
    cnt = _class_counts(y)
    bal = _balance_penalty(cnt, mu)

    if lambda_plurality > 0 and mode_labels is not None and consensus_strength is not None:
        ml = np.asarray(mode_labels, dtype=np.int32)
        st = np.asarray(consensus_strength, dtype=np.float64)
        plur_sum = float(np.sum(st * (y == ml)))
    else:
        plur_sum = 0.0

    loss = _assemble_loss(
        matches,
        agree,
        tgt,
        lambda_agreement,
        lambda_balance=lambda_balance,
        balance_pen=bal,
        lambda_plurality=lambda_plurality,
        plur_sum=plur_sum,
    )
    y_best = y.copy()
    loss_best = loss
    accepted = 0

    # 提议分布（侧重低重合）
    prop_w = None
    if proposal_bias > 0 and consensus_strength is not None:
        prop_w = np.maximum(1e-9, (1.0 - np.asarray(consensus_strength, dtype=np.float64)) ** proposal_bias)
        prop_w /= prop_w.sum()

    for step in range(steps):
        frac = 1.0 - step / max(steps - 1, 1)
        T = max(t_min, t0 * frac)

        i = -1
        for _try in range(max(30, min(n, 200))):
            if prop_w is not None:
                i = int(rng.choice(n, p=prop_w))
            else:
                i = int(rng.integers(0, n))
            if not freeze_mask[i]:
                break
        if i < 0 or freeze_mask[i]:
            continue

        old = int(y[i])
        choices = [c for c in (1, 2, 3) if c != old]
        new = int(rng.choice(choices))

        row = preds[i]
        dmatch = (row == new).astype(np.int32) - (row == old).astype(np.int32)
        delta_agree = int(np.sum(row == new) - np.sum(row == old))

        cand_matches = matches + dmatch
        cand_agree = agree + delta_agree

        cand_cnt = cnt.copy()
        cand_cnt[old - 1] -= 1.0
        cand_cnt[new - 1] += 1.0
        cand_bal = _balance_penalty(cand_cnt, mu)

        if lambda_plurality > 0 and mode_labels is not None and consensus_strength is not None:
            ml = np.asarray(mode_labels, dtype=np.int32)
            st = np.asarray(consensus_strength, dtype=np.float64)
            d_plur = float(st[i] * (int(new == ml[i]) - int(old == ml[i])))
            cand_plur = plur_sum + d_plur
        else:
            cand_plur = plur_sum

        cand_loss = _assemble_loss(
            cand_matches,
            cand_agree,
            tgt,
            lambda_agreement,
            lambda_balance=lambda_balance,
            balance_pen=cand_bal,
            lambda_plurality=lambda_plurality,
            plur_sum=cand_plur,
        )

        d_l = cand_loss - loss
        if d_l <= 0 or (T > 0 and rng.random() < math.exp(-d_l / T)):
            y[i] = new
            matches = cand_matches
            agree = cand_agree
            cnt = cand_cnt
            bal = cand_bal
            plur_sum = cand_plur
            loss = cand_loss
            accepted += 1
            if loss < loss_best:
                loss_best = loss
                y_best = y.copy()

        if trace_every and (step + 1) % trace_every == 0:
            print(
                f"[SA] seed={seed} step={step + 1} loss={loss:.4f} best={loss_best:.4f} "
                f"T={T:.5g} accept_rate={accepted / (step + 1):.3f}",
                flush=True,
            )

    fin_matches = count_matches(y_best, preds)
    fin_agree = total_agreement_from_y(y_best, preds)
    fin_cnt = _class_counts(y_best)
    fin_bal = _balance_penalty(fin_cnt, mu)
    if lambda_plurality > 0 and mode_labels is not None and consensus_strength is not None:
        ml = np.asarray(mode_labels, dtype=np.int32)
        st = np.asarray(consensus_strength, dtype=np.float64)
        fin_plur = float(np.sum(st * (y_best == ml)))
    else:
        fin_plur = 0.0
    fin_loss = _assemble_loss(
        fin_matches,
        fin_agree,
        tgt,
        lambda_agreement,
        lambda_balance=lambda_balance,
        balance_pen=fin_bal,
        lambda_plurality=lambda_plurality,
        plur_sum=fin_plur,
    )
    return SearchResult(
        y=y_best,
        initial_y=y0.astype(np.int32),
        matches_per_model=fin_matches,
        loss=fin_loss,
        steps_accepted=accepted,
        steps_total=steps,
        seed=int(seed),
    )


def _worker_sa(payload: dict[str, Any]) -> SearchResult:
    """子进程入口（须可 pickle）。"""
    return simulate_annealing(
        payload["y0"],
        payload["preds"],
        payload["targets"],
        freeze_mask=payload.get("freeze_mask"),
        steps=int(payload["steps"]),
        seed=int(payload["seed"]),
        t0=float(payload["t0"]),
        t_min=float(payload["t_min"]),
        lambda_agreement=float(payload["lambda_agreement"]),
        trace_every=int(payload.get("trace_every", 0)),
        lambda_balance=float(payload.get("lambda_balance", 0.0)),
        balance_mu=payload.get("balance_mu"),
        lambda_plurality=float(payload.get("lambda_plurality", 0.0)),
        mode_labels=payload.get("mode_labels"),
        consensus_strength=payload.get("consensus_strength"),
        proposal_bias=float(payload.get("proposal_bias", 0.0)),
    )


def simulate_annealing_parallel(
    y0: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray | None,
    *,
    freeze_mask: np.ndarray | None,
    steps_per_chain: int,
    seed_base: int,
    workers: int,
    t0: float,
    t_min: float,
    lambda_agreement: float,
    trace_every: int,
    lambda_balance: float = 0.0,
    balance_mu: np.ndarray | None = None,
    lambda_plurality: float = 0.0,
    mode_labels: np.ndarray | None = None,
    consensus_strength: np.ndarray | None = None,
    proposal_bias: float = 0.0,
) -> SearchResult:
    workers = max(1, int(workers))
    if workers == 1:
        return simulate_annealing(
            y0,
            preds,
            targets,
            freeze_mask=freeze_mask,
            steps=steps_per_chain,
            seed=seed_base,
            t0=t0,
            t_min=t_min,
            lambda_agreement=lambda_agreement,
            trace_every=trace_every,
            lambda_balance=lambda_balance,
            balance_mu=balance_mu,
            lambda_plurality=lambda_plurality,
            mode_labels=mode_labels,
            consensus_strength=consensus_strength,
            proposal_bias=proposal_bias,
        )

    seeds = [seed_base + w * 10_007 for w in range(workers)]
    results = simulate_annealing_collect_chains(
        y0,
        preds,
        targets,
        freeze_mask=freeze_mask,
        steps_per_chain=steps_per_chain,
        seeds=seeds,
        pool_workers=workers,
        t0=t0,
        t_min=t_min,
        lambda_agreement=lambda_agreement,
        trace_every=trace_every,
        lambda_balance=lambda_balance,
        balance_mu=balance_mu,
        lambda_plurality=lambda_plurality,
        mode_labels=mode_labels,
        consensus_strength=consensus_strength,
        proposal_bias=proposal_bias,
    )

    best = min(results, key=lambda r: r.loss)
    total_steps = sum(r.steps_total for r in results)
    total_acc = sum(r.steps_accepted for r in results)
    print(
        f"[recovery] 并行 {workers} 链：最优 seed={best.seed} loss={best.loss:.4f} "
        f"总接受步数={total_acc}/{total_steps}",
        flush=True,
    )
    return best


def simulate_annealing_collect_chains(
    y0: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray | None,
    *,
    freeze_mask: np.ndarray | None,
    steps_per_chain: int,
    seeds: list[int],
    pool_workers: int,
    t0: float,
    t_min: float,
    lambda_agreement: float,
    trace_every: int,
    lambda_balance: float = 0.0,
    balance_mu: np.ndarray | None = None,
    lambda_plurality: float = 0.0,
    mode_labels: np.ndarray | None = None,
    consensus_strength: np.ndarray | None = None,
    proposal_bias: float = 0.0,
) -> list[SearchResult]:
    """多条独立 SA（不同 seed），返回与 seeds 等长的结果列表。"""
    pool_workers = max(1, min(int(pool_workers), len(seeds)))
    payloads: list[dict[str, Any]] = []
    for idx, s in enumerate(seeds):
        payloads.append(
            {
                "y0": y0,
                "preds": preds,
                "targets": targets,
                "freeze_mask": freeze_mask,
                "steps": steps_per_chain,
                "seed": int(s),
                "t0": t0,
                "t_min": t_min,
                "lambda_agreement": lambda_agreement,
                "trace_every": trace_every if len(seeds) == 1 else 0,
                "lambda_balance": lambda_balance,
                "balance_mu": balance_mu,
                "lambda_plurality": lambda_plurality,
                "mode_labels": mode_labels,
                "consensus_strength": consensus_strength,
                "proposal_bias": proposal_bias,
            }
        )

    with ProcessPoolExecutor(max_workers=pool_workers) as ex:
        results = list(ex.map(_worker_sa, payloads))
    return results


def dedupe_results_by_y(results: list[SearchResult], top_k: int) -> list[SearchResult]:
    """按 loss 升序，去掉完全相同标签向量，保留至多 top_k 条。"""
    seen: set[bytes] = set()
    out: list[SearchResult] = []
    for r in sorted(results, key=lambda x: x.loss):
        key = r.y.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= top_k:
            break
    return out
