#!/usr/bin/env python3
"""
读取 gpu_combo_sweep 的 combo_sweep_metrics.csv，对每条成功任务加载 submission.csv，
计算：val_acc（来自日志表）、加权伪标签一致率、与 DeepSeek JSON 一致率；
综合分 composite = 0.5*val_acc + 0.25*weighted_pseudo + 0.25*deepseek_agree（均在 [0,1]）。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from glevel_labels import parse_overall_glevel_value


def _read_sub_123(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        fn = r.fieldnames or []
        lab = "g_level_pred" if "g_level_pred" in fn else "g_level"
        raw: dict[str, str] = {}
        for row in r:
            rid = (row.get("id") or "").strip()
            if rid:
                raw[rid] = (row.get(lab) or "").strip()
    vals = {v.strip() for v in raw.values() if v.strip()}
    mode = "zero" if "0" in vals else "one"
    out: dict[str, int] = {}
    for rid, v in raw.items():
        v = v.strip()
        if mode == "zero":
            if v not in ("0", "1", "2"):
                continue
            out[rid] = int(v) + 1
        else:
            if v not in ("1", "2", "3"):
                continue
            out[rid] = int(v)
    return out


def _load_pseudo(path: Path) -> tuple[dict[str, int], dict[str, float]]:
    gl: dict[str, int] = {}
    tw: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rid = (row.get("id") or "").strip()
            if not rid:
                continue
            gl[rid] = int(float(row["g_level"]))
            tw[rid] = float(row.get("trust_weight") or 1.0)
    return gl, tw


def _load_deepseek(path: Path) -> dict[str, int]:
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
    ap.add_argument("--sweep-csv", type=Path, required=True)
    ap.add_argument("--hunt-dir", type=Path, required=True)
    ap.add_argument("--pseudo-csv", type=Path, required=True)
    ap.add_argument("--deepseek-json", type=Path, required=True)
    ap.add_argument("--out-tsv", type=Path, required=True)
    args = ap.parse_args()

    pseudo_gl, pseudo_tw = _load_pseudo(args.pseudo_csv)
    dk = _load_deepseek(args.deepseek_json)

    ranked: list[tuple[float, dict[str, object]]] = []
    with args.sweep_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("exit_code", "")).strip() != "0":
                continue
            combo = (row.get("combo_id") or "").strip()
            seed = (row.get("seed") or "").strip()
            va_s = (row.get("val_acc") or "").strip()
            if not combo or not seed or va_s in ("", "NA"):
                continue
            try:
                val_acc = float(va_s)
            except ValueError:
                continue
            sub_path = args.hunt_dir / combo / f"seed{seed}" / "submission.csv"
            if not sub_path.is_file():
                continue
            pred = _read_sub_123(sub_path)
            ids_p = sorted(set(pseudo_gl) & set(pred))
            if ids_p:
                num = sum(pseudo_tw[i] for i in ids_p if pred.get(i) == pseudo_gl[i])
                den = sum(pseudo_tw[i] for i in ids_p)
                w_acc = num / den if den > 0 else 0.0
            else:
                w_acc = 0.0
            ids_d = sorted(set(dk) & set(pred))
            d_acc = sum(1 for i in ids_d if pred[i] == dk[i]) / len(ids_d) if ids_d else 0.0
            comp = 0.5 * val_acc + 0.25 * w_acc + 0.25 * d_acc
            ranked.append(
                (
                    comp,
                    {
                        "combo_id": combo,
                        "seed": seed,
                        "val_acc": val_acc,
                        "weighted_pseudo_agree": round(w_acc, 6),
                        "deepseek_agree": round(d_acc, 6),
                        "composite": round(comp, 6),
                        "submission_csv": str(sub_path),
                    },
                )
            )

    ranked.sort(key=lambda x: -x[0])
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "composite",
                "val_acc",
                "weighted_pseudo_agree",
                "deepseek_agree",
                "combo_id",
                "seed",
                "submission_csv",
            ],
            delimiter="\t",
        )
        w.writeheader()
        for _, rec in ranked[:25]:
            w.writerow(rec)
    print(f"[rank] wrote top {min(25, len(ranked))} → {args.out_tsv}")


if __name__ == "__main__":
    main()
