#!/usr/bin/env python3
"""
将 DeepSeek 预标注 JSON（overall_glevel 整数 1/2/3）与多份自有 submission CSV 对比。

用法（仓库根）:
  python tools/compare_best_models_vs_deepseek.py \\
    --deepseek deepseek_annotations.json \\
    external/submissions_peer/submission_masswinner_S_plateau_ln_seed28.csv ...

默认对比一组「当前最优」提交路径（可额外传入更多 CSV）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from glevel_labels import parse_overall_glevel_value

from sklearn.metrics import confusion_matrix

def _read_submission_raw(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        lab = "g_level_pred" if "g_level_pred" in fields else "g_level"
        if lab not in fields:
            raise SystemExit(f"{path} 需要 g_level_pred 或 g_level")
        out: dict[str, str] = {}
        for row in r:
            rid = (row.get("id") or "").strip()
            if not rid:
                continue
            out[rid] = (row.get(lab) or "").strip()
        return out


def _sub_to_123(sub: dict[str, str]) -> dict[str, int]:
    vals = {v.strip() for v in sub.values() if v and v.strip()}
    mode = "zero" if "0" in vals else "one"
    y: dict[str, int] = {}
    for rid, v in sub.items():
        v = v.strip()
        if mode == "zero":
            if v not in ("0", "1", "2"):
                continue
            y[rid] = int(v) + 1
        else:
            if v not in ("1", "2", "3"):
                continue
            y[rid] = int(v)
    return y


def _load_deepseek_123(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, int] = {}
    for rid, rec in raw.items():
        if not isinstance(rec, dict):
            continue
        gl = parse_overall_glevel_value(rec.get("overall_glevel"))
        if gl is not None:
            out[str(rid).strip()] = gl
    return out


def _compare(ref: dict[str, int], pred: dict[str, int], name: str) -> None:
    common = sorted(set(ref) & set(pred))
    if not common:
        print(f"[compare] {name}: no common ids")
        return
    y_true = [ref[i] for i in common]
    y_pred = [pred[i] for i in common]
    agree = sum(a == b for a, b in zip(y_true, y_pred)) / len(common)
    print(f"\n=== {name} ===")
    print(f"n={len(common)} agreement_with_deepseek={agree:.4f}")
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    print("混淆矩阵 行=DeepSeek 列=模型（均为 1=Low 2=Med 3=High）")
    print(cm)
    print("DeepSeek 分布:", Counter(y_true))
    print("模型预测分布:", Counter(y_pred))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--deepseek",
        type=Path,
        default=Path("deepseek_annotations.json"),
        help="annotate_with_deepseek.py 输出",
    )
    ap.add_argument(
        "submissions",
        nargs="*",
        type=Path,
        help="submission CSV；为空则用内置最优列表",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    default_subs = [
        root / "external/submissions_peer/submission_masswinner_S_plateau_ln_seed28.csv",
        root / "external/submissions_peer/submission_rerun_seed37_archive_inferbias070.csv",
        root / "external/submissions_peer/submission_rerun_seed37_archive_plain.csv",
        root / "external/submissions_peer/submission_masswinner_S_ref_cosine_seed5.csv",
        root / "external/submissions_peer/submission_masswinner_S_ref_plateau_seed37.csv",
        root / "external/submissions_peer/submission_ours_seed37_inferbias_0_0p7_0.csv",
    ]
    subs = list(args.submissions) if args.submissions else default_subs

    dk_path = args.deepseek
    if not dk_path.is_absolute():
        cand = root / dk_path
        dk_path = cand if cand.is_file() else dk_path.expanduser().resolve()
    else:
        dk_path = dk_path.resolve()

    if not dk_path.is_file():
        raise SystemExit(f"找不到 DeepSeek JSON: {dk_path}")

    ref = _load_deepseek_123(dk_path)
    print(f"[compare] DeepSeek 标注: {dk_path} n_ids={len(ref)}")
    if not ref:
        raise SystemExit("DeepSeek 无可比对标签")

    for p in subs:
        p = p if p.is_absolute() else (root / p).resolve()
        if not p.is_file():
            print(f"[compare] skip missing {p}")
            continue
        sub_raw = _read_submission_raw(p)
        pred123 = _sub_to_123(sub_raw)
        _compare(ref, pred123, p.name)


if __name__ == "__main__":
    main()
