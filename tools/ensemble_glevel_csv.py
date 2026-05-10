"""
将多份 submission（或折预测）按 id 对 g_level_pred 做多数投票，生成融合 CSV。
行顺序与第一份输入表一致（便于对齐赛方 test_csv）。
"""

from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd


def _majority(votes: list[int]) -> int:
    c = Counter(votes)
    best = max(c.values())
    winners = sorted(k for k, v in c.items() if v == best)
    return int(winners[0])


def main():
    p = argparse.ArgumentParser(description="g_level 多模型预测 CSV → 多数投票融合")
    p.add_argument("--inputs", nargs="+", required=True, help="多份含 id 与预测列的 CSV")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--id_col", type=str, default="id")
    p.add_argument("--pred_col", type=str, default="g_level_pred")
    args = p.parse_args()

    template = pd.read_csv(args.inputs[0])
    if args.id_col not in template.columns:
        raise SystemExit(f"首份 CSV 无列 {args.id_col!r}，当前为 {list(template.columns)}")
    for path in args.inputs[1:]:
        df = pd.read_csv(path)
        if args.id_col not in df.columns or args.pred_col not in df.columns:
            raise SystemExit(f"{path} 缺少 {args.id_col!r} 或 {args.pred_col!r}")

    votes_by_id: dict[str, list[int]] = {}
    for path in args.inputs:
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            sid = str(r[args.id_col]).strip()
            votes_by_id.setdefault(sid, []).append(int(r[args.pred_col]))

    out = template[[args.id_col]].copy()
    merged: list[int] = []
    for _, r in out.iterrows():
        sid = str(r[args.id_col]).strip()
        vs = votes_by_id.get(sid)
        if not vs or len(vs) != len(args.inputs):
            raise SystemExit(
                f"id {sid!r} 在部分输入中缺失或重复计数异常 "
                f"（期望每份输入各 1 条，当前收集到 {0 if vs is None else len(vs)} 票）"
            )
        merged.append(_majority(vs))
    out[args.pred_col] = merged
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out)} rows, {len(args.inputs)} inputs → majority vote)")


if __name__ == "__main__":
    main()
