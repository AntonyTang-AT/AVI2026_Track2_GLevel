#!/usr/bin/env python3
"""
统计 DeepSeek JSON（overall_glevel，整数 1/2/3）与 external/submissions_peer/submission*.csv 的一致性。

- 每条样本：有多少个模型提交与 DeepSeek 档位完全相同。
- 按 DeepSeek g_level 分层输出分布。
- 列出各提交与 DeepSeek 的一致率。

用法（仓库根目录）:
  python tools/stats_deepseek_vs_submissions_agreement.py \\
    --deepseek reports/deepseek/deepseek_ens_r3_vote_zyn.json \\
    --out-csv reports/deepseek/deepseek_zyn_agree_counts.csv
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

from dataset.glevel_labels import parse_overall_glevel_value


def _read_submission_raw(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        lab = "g_level_pred" if "g_level_pred" in fields else "g_level"
        if lab not in fields:
            raise ValueError(f"{path.name}: need g_level_pred or g_level")
        out: dict[str, str] = {}
        for row in r:
            rid = (row.get("id") or "").strip()
            if rid:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepseek", type=Path, required=True)
    ap.add_argument(
        "--peer-dir",
        type=Path,
        default=Path("external/submissions_peer"),
        help="扫描 submission*.csv（排除 test_pseudo*）",
    )
    ap.add_argument(
        "--min-ids",
        type=int,
        default=100,
        help="提交的预测 id 数低于该值则跳过",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="每条样本输出 id, deepseek 档位, 一致模型数",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    dk_path = args.deepseek if args.deepseek.is_absolute() else (root / args.deepseek)
    peer = args.peer_dir if args.peer_dir.is_absolute() else (root / args.peer_dir)

    ref = _load_deepseek_123(dk_path)
    if not ref:
        raise SystemExit("DeepSeek JSON 无有效标签")

    subs_paths = sorted(
        p for p in peer.glob("submission*.csv") if not p.name.startswith("test_pseudo")
    )
    models: dict[str, dict[str, int]] = {}
    for p in subs_paths:
        try:
            pred = _sub_to_123(_read_submission_raw(p))
        except Exception as e:
            print(f"[skip] {p.name}: {e}")
            continue
        if len(pred) < args.min_ids:
            print(f"[skip] {p.name}: only {len(pred)} ids")
            continue
        models[p.name] = pred

    common: set[str] = set(ref.keys())
    for pred in models.values():
        common &= set(pred.keys())

    print(f"DeepSeek: {dk_path} n={len(ref)}")
    print(f"模型提交: {len(models)} 份；统计用交集 id: {len(common)}")

    model_names = list(models.keys())
    n_m = len(model_names)
    agree_per_id: dict[str, int] = {}
    for sid in common:
        d = ref[sid]
        agree_per_id[sid] = sum(1 for m in model_names if models[m][sid] == d)

    hist = Counter(agree_per_id.values())
    print(f"\n=== 每条样本与 {n_m} 个模型预测相同的个数 ===")
    for k in range(n_m + 1):
        if k in hist:
            print(f"  {k:2d} 个一致: {hist[k]:3d} ({100 * hist[k] / len(common):.1f}%)")

    print("\n=== 按 DeepSeek 标签分层（平均一致模型数）===")
    for g in (1, 2, 3):
        sub_ids = [sid for sid in common if ref[sid] == g]
        if not sub_ids:
            continue
        mean = sum(agree_per_id[sid] for sid in sub_ids) / len(sub_ids)
        ctr = Counter(agree_per_id[sid] for sid in sub_ids)
        print(f"\ng_level={g} (n={len(sub_ids)})  mean_agree={mean:.2f}")
        for k in sorted(ctr):
            print(f"    {k} 模型一致: {ctr[k]} ({100 * ctr[k] / len(sub_ids):.1f}%)")

    print(f"\n=== 各模型与 DeepSeek 一致率（n={len(common)}）===")
    rows = []
    for m in sorted(model_names):
        ag = sum(1 for sid in common if models[m][sid] == ref[sid])
        rows.append((ag / len(common), m, ag))
    for rate, m, ag in sorted(rows, reverse=True):
        print(f"  {rate:.4f}  ({ag:3d}/{len(common)})  {m}")

    if args.out_csv:
        out_p = args.out_csv if args.out_csv.is_absolute() else (root / args.out_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "id",
                    "deepseek_g_level",
                    "n_models_agree",
                    "n_models_total",
                ]
            )
            for sid in sorted(common):
                g = ref[sid]
                w.writerow([sid, g, agree_per_id[sid], n_m])
        print(f"\n[wrote] {out_p}")


if __name__ == "__main__":
    main()
