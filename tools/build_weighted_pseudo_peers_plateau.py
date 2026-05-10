#!/usr/bin/env python3
"""
四份学长提交（赛方 1/2/3 列 g_level）+ masswinner plateau_ln seed28（g_level_pred 可为 0/1/2 或 1/2/3）→ 加权伪标签。

规则：
  - 各文件先规范到 g_level 1..3（若列为 0/1/2 则 +1）。
  - 五票全相同：is_unanimous=1，trust_weight=1.0。
  - 否则：多数票标签；trust_weight = 该标签得票数 / 5；平局取较小等级（1<2<3）。

输出：external/submissions_peer/test_pseudo_weighted_peers4_plateau_ln28.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def _read_labels(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        fn = r.fieldnames or []
        lab = "g_level_pred" if "g_level_pred" in fn else "g_level"
        if lab not in fn:
            raise SystemExit(f"{path}: need g_level_pred or g_level")
        out: dict[str, str] = {}
        for row in r:
            rid = (row.get("id") or "").strip()
            if rid:
                out[rid] = (row.get(lab) or "").strip()
        return out


def _infer_mode(samples: list[str]) -> str:
    vals = {x.strip() for x in samples if x and x.strip()}
    if "0" in vals:
        return "zero"
    return "one"


def _to123(s: str, mode: str) -> int:
    s = s.strip()
    if mode == "zero":
        if s not in ("0", "1", "2"):
            raise ValueError(f"bad 012 label {s!r}")
        return int(s) + 1
    if s not in ("1", "2", "3"):
        raise ValueError(f"bad 123 label {s!r}")
    return int(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    peer = root / "external/submissions_peer"
    paths = [
        peer / "submission1_0.53077.csv",
        peer / "submission2_0.50769.csv",
        peer / "submission_0.53846.csv",
        peer / "submission5_0.55385.csv",
        peer / "submission_masswinner_S_plateau_ln_seed28.csv",
    ]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    tables = [_read_labels(p) for p in paths]
    common = set.intersection(*(set(t) for t in tables))
    if len(common) < 100:
        raise SystemExit(f"id intersection too small: {len(common)}")

    modes: list[str] = []
    for t in tables:
        raw = [t[i] for i in sorted(common)]
        modes.append(_infer_mode(raw))

    out_path = args.out or (peer / "test_pseudo_weighted_peers4_plateau_ln28.csv")
    rows_out: list[dict[str, object]] = []
    for rid in sorted(common):
        labs: list[int] = []
        for t, mode in zip(tables, modes):
            labs.append(_to123(t[rid], mode))
        vc = Counter(labs)
        unanimous = len(set(labs)) == 1
        mx = max(vc.values())
        cand = sorted([k for k, v in vc.items() if v == mx])
        winner = cand[0]
        if unanimous:
            tw = 1.0
            iso = 1
        else:
            tw = vc[winner] / 5.0
            iso = 0
        rows_out.append(
            {
                "id": rid,
                "g_level": winner,
                "is_unanimous": iso,
                "trust_weight": f"{tw:.6f}",
                "vote_count_1": vc.get(1, 0),
                "vote_count_2": vc.get(2, 0),
                "vote_count_3": vc.get(3, 0),
                "n_sources": 5,
                "encoding_note": "peers=g_level_123 plateau=g_level_pred_012_to_123",
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "g_level",
        "is_unanimous",
        "trust_weight",
        "vote_count_1",
        "vote_count_2",
        "vote_count_3",
        "n_sources",
        "encoding_note",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"[build_weighted_pseudo] n={len(rows_out)} → {out_path}")


if __name__ == "__main__":
    main()
