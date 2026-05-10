#!/usr/bin/env python3
"""
四份 peer submission → 测试集伪标签（赛方 1/2/3）。

规则：
  - 四个标签完全一致：记为高置信「伪真」is_unanimous=1，trust_weight=1。
  - 不一致：用「该等级在四份提交全部投票中出现的频次」作先验 prior，
    score[k] = vote_count_k * prior_freq[k]，argmax 得 g_level；平局取较小等级。
  - trust_weight：非一致时为 max_vote_ratio * sqrt(prior[winner]/(1/3))，clip 到 [0.35, 0.99]。

用法（在仓库根目录）:
  python tools/build_four_peer_consensus_pseudo.py --out external/submissions_peer/foo.csv \\
    external/submissions_peer/a.csv ...
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
        help="写出 CSV（含 id,g_level,is_unanimous,trust_weight,prior_freq_* 等）",
    )
    ap.add_argument(
        "csv_paths",
        nargs=4,
        type=Path,
        metavar=("S1", "S2", "S3", "S4"),
        help="四份 peer submission",
    )
    args = ap.parse_args()

    names = [p.name for p in args.csv_paths]
    tables = [_read_labels(p) for p in args.csv_paths]
    ids_sets = [set(t.keys()) for t in tables]
    common = set.intersection(*ids_sets)
    sizes = [len(s) for s in ids_sets]
    if not common or min(sizes) != max(sizes) or len(common) != sizes[0]:
        raise SystemExit(f"id 交集异常 common={len(common)} per_file={sizes}")

    lab_modes = []
    for t in tables:
        raw_samples = [t[i] for i in common if i in t]
        lab_modes.append(_infer_label_encoding(raw_samples))
    if len(set(lab_modes)) != 1:
        raise SystemExit(f"四文件编码须一致: {dict(zip(names, lab_modes))}")
    mode = lab_modes[0]

    flat123: list[str] = []
    for t in tables:
        for i in sorted(common):
            flat123.append(_norm_lab(t[i], mode))
    G = Counter(flat123)
    total_votes = sum(G.values())
    prior = {str(k): G.get(str(k), 0) / max(total_votes, 1) for k in (1, 2, 3)}
    uni = 1.0 / 3.0

    rows: list[dict[str, object]] = []
    for rid in sorted(common):
        votes = [_norm_lab(t[rid], mode) for t in tables]
        vc = Counter(votes)
        unanimous = len(set(votes)) == 1
        winner: str
        if unanimous:
            winner = votes[0]
            trust = 1.0
        else:
            scores = {str(k): vc.get(str(k), 0) * prior[str(k)] for k in (1, 2, 3)}
            winner = max(scores.keys(), key=lambda x: (scores[x], -int(x)))
            max_ratio = vc[winner] / 4.0
            pw = prior[winner]
            trust = max_ratio * math.sqrt(pw / uni)
            trust = float(min(0.99, max(0.35, trust)))

        rows.append(
            {
                "id": rid,
                "g_level": int(winner),
                "is_unanimous": 1 if unanimous else 0,
                "n_agree_on_winner": vc[winner],
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
        f"[consensus] wrote {args.out} n={len(rows)} unanimous={n_uni}/{len(rows)} "
        f"prior_123=({prior['1']:.4f},{prior['2']:.4f},{prior['3']:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
