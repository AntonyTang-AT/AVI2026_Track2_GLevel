#!/usr/bin/env python3
"""
K 份 submission（K>=2）→ 测试集伪标签；规则与 tools/build_four_peer_consensus_pseudo.py 相同，
将「四票」推广为「K 票」（prior 仍由全部 K×n 个标签统计）。

用法（仓库根）:
  python tools/build_consensus_pseudo_k.py --out external/submissions_peer/foo.csv \\
    external/submissions_peer/a.csv external/submissions_peer/b.csv ...
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from compare_glevel_submissions import (  # noqa: E402
    _infer_label_encoding,
    _norm_lab,
    _read_labels,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="写出 CSV（含 id,g_level,is_unanimous,trust_weight,...）",
    )
    ap.add_argument(
        "csv_paths",
        nargs="+",
        type=Path,
        metavar="CSV",
        help="若干份 peer / 自有 submission（列 id + g_level_pred 或 g_level）",
    )
    args = ap.parse_args()
    k = len(args.csv_paths)
    if k < 2:
        raise SystemExit("至少需要 2 个 CSV")

    tables = [_read_labels(p) for p in args.csv_paths]
    ids_sets = [set(t.keys()) for t in tables]
    common = set.intersection(*ids_sets)
    sizes = [len(s) for s in ids_sets]
    if not common or min(sizes) != max(sizes) or len(common) != sizes[0]:
        raise SystemExit(f"id 交集异常 common={len(common)} per_file={sizes}")

    # 每个文件单独推断 0/1/2 或 1/2/3，再统一规范到 "1","2","3" 字符串（允许 peer 与自有提交编码混用）
    lab_modes: list[str] = []
    for t in tables:
        raw_samples = [t[i] for i in common if i in t]
        lab_modes.append(_infer_label_encoding(raw_samples))

    flat123: list[str] = []
    for t, mode_t in zip(tables, lab_modes):
        for i in sorted(common):
            flat123.append(_norm_lab(t[i], mode_t))
    G = Counter(flat123)
    total_votes = sum(G.values())
    prior = {str(x): G.get(str(x), 0) / max(total_votes, 1) for x in (1, 2, 3)}
    uni_prior = 1.0 / 3.0

    rows: list[dict[str, object]] = []
    for rid in sorted(common):
        votes = [_norm_lab(t[rid], mode_t) for t, mode_t in zip(tables, lab_modes)]
        vc = Counter(votes)
        unanimous = len(set(votes)) == 1
        if unanimous:
            winner = votes[0]
            trust = 1.0
        else:
            scores = {str(x): vc.get(str(x), 0) * prior[str(x)] for x in (1, 2, 3)}
            winner = max(scores.keys(), key=lambda x: (scores[x], -int(x)))
            max_ratio = vc[winner] / float(k)
            pw = prior[winner]
            trust = max_ratio * math.sqrt(pw / uni_prior)
            trust = float(min(0.99, max(0.35, trust)))

        rows.append(
            {
                "id": rid,
                "g_level": int(winner),
                "is_unanimous": 1 if unanimous else 0,
                "n_agree_on_winner": vc[winner],
                "k_sources": k,
                "prior_freq_1": f"{prior['1']:.6f}",
                "prior_freq_2": f"{prior['2']:.6f}",
                "prior_freq_3": f"{prior['3']:.6f}",
                "vote_count_1": vc.get("1", 0),
                "vote_count_2": vc.get("2", 0),
                "vote_count_3": vc.get("3", 0),
                "trust_weight": f"{trust:.6f}",
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "g_level",
        "is_unanimous",
        "n_agree_on_winner",
        "k_sources",
        "trust_weight",
        "prior_freq_1",
        "prior_freq_2",
        "prior_freq_3",
        "vote_count_1",
        "vote_count_2",
        "vote_count_3",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_uni = sum(1 for r in rows if r["is_unanimous"] == 1)
    print(
        f"[consensus_k] wrote {args.out} k={k} n={len(rows)} unanimous={n_uni}/{len(rows)} "
        f"prior_123=({prior['1']:.4f},{prior['2']:.4f},{prior['3']:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
