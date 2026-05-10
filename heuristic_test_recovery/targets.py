"""
由「榜单准确率」推导目标命中数（整数条数）。

说明
----
1. JSON 里的浮点数可能被解析成二进制近似；优先在 manifest 里写字符串准确率：
     "accuracy": "0.46923"
   或使用百分比：
     "accuracy_pct": "46.923"
2. 若有「官方公布的正确条数」，可直接写整数 hits，覆盖一切换算：
     "hits": 61
3. hit_policy 决定如何把 Decimal(acc) * n 变成整数（详见 hits_from_accuracy）。
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Literal

HitPolicy = Literal["nearest", "half_up", "floor", "ceil", "trunc"]


def _dec(s: object) -> Decimal:
    if isinstance(s, Decimal):
        return s
    if isinstance(s, str):
        return Decimal(s.strip())
    # JSON number → 先 str 再 Decimal，尽量避免二进制展开写入 Decimal
    return Decimal(str(s))


def hits_from_accuracy(acc: Decimal, n: int, policy: HitPolicy) -> int:
    """
    acc ∈ [0,1] 为小数准确率；n 为样本数。
    - nearest: 选整数 k 使 |k/n - acc| 最小（推荐，与「命中率最接近声明值」一致）
    - half_up: 对 acc*n 做四舍五入（HALF_UP）
    - floor / ceil: 对 acc*n 下取整/上取整
    - trunc: 向 0 截断 acc*n
    """
    if n <= 0:
        raise ValueError("n 须为正整数")
    n_d = Decimal(n)
    prod = acc * n_d

    if policy == "nearest":
        # 在 half_up 整数附近 ±1 内找使 |k/n-acc| 最小的 k，避免边界上一档之差
        k0 = int(prod.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        k0 = max(0, min(n, k0))
        candidates = {k0}
        if k0 + 1 <= n:
            candidates.add(k0 + 1)
        if k0 - 1 >= 0:
            candidates.add(k0 - 1)
        return min(candidates, key=lambda k: abs(Decimal(k) / n_d - acc))

    if policy == "half_up":
        return int(prod.quantize(Decimal("1"), rounding=ROUND_HALF_UP))

    if policy == "floor":
        return int(prod.to_integral_value(rounding=ROUND_FLOOR))

    if policy == "ceil":
        return int(prod.to_integral_value(rounding=ROUND_CEILING))

    if policy == "trunc":
        # 非负准确率下同 floor
        return int(prod.to_integral_value(rounding=ROUND_DOWN))

    raise ValueError(f"未知 hit_policy={policy!r}")


def resolve_row_target(
    *,
    n: int,
    hits_explicit: int | None,
    accuracy_raw: object | None,
    accuracy_pct_raw: object | None,
    policy: HitPolicy,
) -> tuple[int, str]:
    """
    返回 (target_hits, provenance_note)。
    """
    if hits_explicit is not None:
        h = int(hits_explicit)
        if h < 0 or h > n:
            raise ValueError(f"hits={h} 须在 [0,{n}]")
        return h, "explicit_hits"

    if accuracy_pct_raw is not None:
        pct = _dec(accuracy_pct_raw)
        acc = pct / Decimal("100")
        h = hits_from_accuracy(acc, n, policy)
        return h, f"accuracy_pct={pct!s}→acc={acc!s}"

    if accuracy_raw is not None:
        acc = _dec(accuracy_raw)
        h = hits_from_accuracy(acc, n, policy)
        return h, f"accuracy={acc!s}"

    raise ValueError("须提供 hits、accuracy 或 accuracy_pct 之一")


def build_targets_vector(
    rows: list[dict],
    n: int,
    policy: HitPolicy,
) -> tuple[list[int], list[str]]:
    """
    rows: manifest 里每个 file 条目解析成的 dict，键可选：
      hits, accuracy / acc, accuracy_pct
    """
    targets: list[int] = []
    notes: list[str] = []
    for i, item in enumerate(rows):
        h_raw = item.get("hits")
        hits_explicit = int(h_raw) if h_raw is not None else None
        tgt, note = resolve_row_target(
            n=n,
            hits_explicit=hits_explicit,
            accuracy_raw=item.get("accuracy", item.get("acc")),
            accuracy_pct_raw=item.get("accuracy_pct"),
            policy=policy,
        )
        targets.append(tgt)
        notes.append(f"m{i}: {note} policy={policy} → hits={tgt}")
    return targets, notes
