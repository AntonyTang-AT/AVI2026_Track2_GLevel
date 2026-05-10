#!/usr/bin/env python3
"""将 submission（g_level_pred 0/1/2 或 g_level 1/2/3）与伪标签 CSV（g_level 1/2/3）对齐后算一致率（非官方准确率）。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sklearn.metrics import confusion_matrix


def _read_col(path: Path, id_col: str, lab_col: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return {
            (row[id_col] or "").strip(): (row[lab_col] or "").strip()
            for row in r
            if (row.get(id_col) or "").strip()
        }


def _infer_label_encoding(sample_labels: list[str]) -> str:
    vals = {x.strip() for x in sample_labels if x.strip()}
    if not vals:
        return "one"
    if "0" in vals:
        return "zero"
    if vals <= {"1", "2", "3"}:
        return "one"
    raise SystemExit(f"无法推断标签编码，取值集合={sorted(vals)}")


def _labs_to123(labels: list[str], mode: str) -> list[int]:
    out: list[int] = []
    for raw in labels:
        s = raw.strip()
        if mode == "zero":
            if s not in ("0", "1", "2"):
                raise SystemExit(f"zero 编码下非法标签 {raw!r}")
            out.append(int(s) + 1)
        else:
            if s not in ("1", "2", "3"):
                raise SystemExit(f"one 编码下非法标签 {raw!r}")
            out.append(int(s))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pseudo_csv", type=Path, help="含列 id 与 g_level（1/2/3）")
    ap.add_argument("submission_csv", type=Path)
    ap.add_argument(
        "--only-unanimous",
        action="store_true",
        help="若 pseudo_csv 含列 is_unanimous，则只评估 is_unanimous=1 的样本",
    )
    args = ap.parse_args()

    pseudo_raw: dict[str, dict[str, str]] = {}
    with args.pseudo_csv.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "id" not in r.fieldnames or "g_level" not in r.fieldnames:
            raise SystemExit(f"pseudo 需要列 id,g_level，当前 {r.fieldnames}")
        for row in r:
            rid = (row.get("id") or "").strip()
            if not rid:
                continue
            pseudo_raw[rid] = {k: (row.get(k) or "").strip() for k in (r.fieldnames or [])}

    pseudo = {k: v["g_level"] for k, v in pseudo_raw.items()}
    with args.submission_csv.open(newline="", encoding="utf-8-sig") as f:
        fields = csv.DictReader(f).fieldnames or []
    lab = "g_level_pred" if "g_level_pred" in fields else "g_level"
    if lab not in fields:
        raise SystemExit(f"{args.submission_csv} 需要 g_level_pred 或 g_level，当前 {fields}")
    sub = _read_col(args.submission_csv, "id", lab)

    common = sorted(set(pseudo) & set(sub))
    if args.only_unanimous:
        filtered = [
            i
            for i in common
            if i in pseudo_raw and (pseudo_raw[i].get("is_unanimous") or "") == "1"
        ]
        common = sorted(filtered)
        print(f"[pseudo_eval] only-unanimous subset n={len(common)}", flush=True)
    if not common:
        raise SystemExit("无交集 id")
    pv = [pseudo[i] for i in common]
    sv = [sub[i] for i in common]
    pm = _infer_label_encoding(pv)
    sm = _infer_label_encoding(sv)
    y_true = _labs_to123(pv, pm)
    y_pred = _labs_to123(sv, sm)
    agree = sum(a == b for a, b in zip(y_true, y_pred)) / len(common)
    print(f"[pseudo_eval] n={len(common)} agreement={agree:.4f}")
    print(f"[pseudo_eval] pseudo_encoding={pm} submission_encoding={sm}")
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    print("[pseudo_eval] 混淆矩阵 行=伪标签 列=submission（均为 1/2/3）")
    print(cm)


if __name__ == "__main__":
    main()
