#!/usr/bin/env python3
"""
比较多个 g_level 测试集 submission CSV（无真值时无法计算官方准确率）。

用途（学长/自己的多份提交）:
  - 两两一致的样本比例（可作为「模型相关性」粗筛）
  - 多数投票生成融合 submission
  - 各类预测占比（sanity check）

输入 CSV 须含列 id；标签列名为 g_level_pred 或 g_level（0/1/2 或 1/2/3 均可，混用时会尝试对齐）。

用法:
  python tools/compare_glevel_submissions.py \\
    ref.csv peer1.csv peer2.csv \\
    --vote-out ./submission_vote.csv

  # 四份学长提交 → 多数票 / 按 leaderboard 分数加权，生成测试集伪标签（g_level 为 1/2/3）
  python tools/compare_glevel_submissions.py s1.csv s2.csv s3.csv s4.csv \\
    --pseudo-majority-123 ./pseudo_majority_123.csv \\
    --pseudo-weighted-123 ./pseudo_weighted_123.csv \\
    --vote-weights 0.55385,0.53846,0.53077,0.50769

说明:
  - 文件名里的 0.53 等数字来自「在线评测」，本地没有测试集 g_level 真值时无法复现该数。
  - 若有带标签的留出集，请用 eval_glevel_checkpoint_on_csv.py 在 val/holdout 上算 acc。
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


def _read_labels(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise SystemExit(f"空表头: {path}")
        id_col = "id"
        lab_col = None
        for c in ("g_level_pred", "g_level", "pred", "prediction"):
            if c in r.fieldnames:
                lab_col = c
                break
        if lab_col is None:
            raise SystemExit(f"{path} 需要列 g_level_pred 或 g_level，当前: {r.fieldnames}")
        out: dict[str, str] = {}
        for row in r:
            rid = (row.get(id_col) or "").strip()
            if not rid:
                continue
            out[rid] = (row.get(lab_col) or "").strip()
        return out


def _infer_label_encoding(sample_labels: list[str]) -> str:
    """区分提交里是 0/1/2（类下标）还是赛方 1/2/3。"""
    vals = {x.strip() for x in sample_labels if x.strip()}
    if not vals:
        return "one"
    if "0" in vals:
        return "zero"
    if vals <= {"1", "2", "3"}:
        return "one"
    raise SystemExit(f"无法推断标签编码，取值集合={sorted(vals)}（需为子集 0/1/2 或 1/2/3）")


def _norm_lab(s: str, mode: str) -> str:
    s = s.strip()
    if mode == "zero":
        if s in ("0", "1", "2"):
            return str(int(s) + 1)
        raise SystemExit(f"_norm_lab(zero): 非法标签 {s!r}")
    if s in ("1", "2", "3"):
        return s
    raise SystemExit(f"_norm_lab(one): 非法标签 {s!r}")


def _majority_label(votes: list[str]) -> tuple[str, Counter]:
    c = Counter(votes)
    mx = max(c.values())
    # 平局时取较小等级（1<2<3），便于复现
    winners = sorted([k for k, v in c.items() if v == mx], key=lambda x: int(x))
    return winners[0], c


def _weighted_label(votes: list[str], weights: list[float]) -> tuple[str, dict[str, float]]:
    acc: dict[str, float] = defaultdict(float)
    for lab, w in zip(votes, weights):
        acc[lab] += float(w)
    best = max(acc.items(), key=lambda kv: (kv[1], -int(kv[0])))
    return best[0], dict(acc)


def main() -> None:
    ap = argparse.ArgumentParser(description="比较无真值的 submission CSV（一致性 / 投票）")
    ap.add_argument("csv_paths", nargs="+", type=Path, help="submission 路径列表，第一份可作参照")
    ap.add_argument("--vote-out", type=Path, default=None, help="多数投票写出路径")
    ap.add_argument(
        "--pseudo-majority-123",
        type=Path,
        default=None,
        help="写出 id,g_level（取值 1/2/3）及投票诊断列，供无真值时的粗参照",
    )
    ap.add_argument(
        "--pseudo-weighted-123",
        type=Path,
        default=None,
        help="加权投票写出 id,g_level（1/2/3），需配合 --vote-weights",
    )
    ap.add_argument(
        "--vote-weights",
        type=str,
        default="",
        help="与 csv_paths 等长的逗号分隔非负权重（通常用各提交线上分数）；用于 --pseudo-weighted-123",
    )
    args = ap.parse_args()

    names = [p.name for p in args.csv_paths]
    tables = [_read_labels(p) for p in args.csv_paths]
    ids_sets = [set(t.keys()) for t in tables]
    common = set.intersection(*ids_sets) if ids_sets else set()
    print(f"[compare] files={len(tables)} names={names}")
    print(f"[compare] ids per file: {[len(s) for s in ids_sets]}")
    print(f"[compare] common ids: {len(common)}")
    if not common:
        raise SystemExit("无交集 id，检查 CSV 是否同一测试划分")

    lab_modes: list[str] = []
    for t in tables:
        raw_samples = [t[i] for i in common if i in t]
        lab_modes.append(_infer_label_encoding(raw_samples))
    if len(set(lab_modes)) > 1:
        raise SystemExit(f"各文件标签编码不一致（须同为 0/1/2 或同为 1/2/3）：{dict(zip(names, lab_modes))}")
    lab_mode = lab_modes[0]
    print(f"[compare] label_encoding={lab_mode} (统一到赛方 1/2/3 再比较/投票)")
    # 分布
    for n, t in zip(names, tables):
        labs = [_norm_lab(t[i], lab_mode) for i in common if i in t]
        cnt = Counter(labs)
        print(f"[dist] {n}: {dict(sorted(cnt.items(), key=lambda x: int(x[0])))}")

    # 两两一致率
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            agree = sum(
                1
                for k in common
                if _norm_lab(tables[i][k], lab_mode) == _norm_lab(tables[j][k], lab_mode)
            )
            rate = agree / len(common)
            print(f"[agree] {names[i]} vs {names[j]}: {agree}/{len(common)} = {rate:.4f}")

    if args.vote_out:
        rows_out: list[tuple[str, str]] = []
        for k in sorted(common):
            votes = [_norm_lab(t[k], lab_mode) for t in tables]
            winner, _ = _majority_label(votes)
            rows_out.append((k, winner))
        args.vote_out.parent.mkdir(parents=True, exist_ok=True)
        with args.vote_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "g_level_pred"])
            for rid, lab in rows_out:
                w.writerow([rid, lab])
        print(f"[compare] vote-out wrote {args.vote_out} (g_level_pred 官方 1/2/3)")

    wts_raw = (args.vote_weights or "").strip()
    weights: list[float] | None = None
    if wts_raw:
        weights = [float(x.strip()) for x in wts_raw.split(",") if x.strip()]
        if len(weights) != len(tables):
            raise SystemExit(
                f"--vote-weights 长度须与 csv 文件数一致：{len(weights)} vs {len(tables)}"
            )

    def _write_pseudo123(
        path: Path,
        mode: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "id",
                    "g_level",
                    "n_voters",
                    "vote_entropy",
                    "counts_1",
                    "counts_2",
                    "counts_3",
                    "weighted_score_1",
                    "weighted_score_2",
                    "weighted_score_3",
                ]
            )
            for k in sorted(common):
                votes = [_norm_lab(t[k], lab_mode) for t in tables]
                maj, cnt = _majority_label(votes)
                n = len(votes)
                probs = [cnt.get(str(i), 0) / n for i in (1, 2, 3)]
                ent = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
                ws = {"1": 0.0, "2": 0.0, "3": 0.0}
                wl = maj
                if weights is not None:
                    wl, score_map = _weighted_label(votes, weights)
                    for kk in ("1", "2", "3"):
                        ws[kk] = score_map.get(kk, 0.0)
                winner = wl if mode == "weighted" else maj
                w.writerow(
                    [
                        k,
                        winner,
                        n,
                        f"{ent:.6f}",
                        cnt.get("1", 0),
                        cnt.get("2", 0),
                        cnt.get("3", 0),
                        f"{ws['1']:.6f}",
                        f"{ws['2']:.6f}",
                        f"{ws['3']:.6f}",
                    ]
                )
        print(f"[compare] wrote {path} (g_level 1/2/3, mode={mode})")

    if args.pseudo_majority_123:
        _write_pseudo123(args.pseudo_majority_123, "majority")

    if args.pseudo_weighted_123:
        if weights is None:
            raise SystemExit("--pseudo-weighted-123 需要同时提供 --vote-weights")
        _write_pseudo123(args.pseudo_weighted_123, "weighted")


if __name__ == "__main__":
    main()
