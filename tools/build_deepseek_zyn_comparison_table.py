#!/usr/bin/env python3
"""
生成对比表：各模型/提交的官方验证集准确率、测试集三档分布、与 DeepSeek zyn 多数票的重叠率。

数据来源：
- val_acc：EXPERIMENT_LOG.csv、batch_infer_eval_summary.tsv、gpu mass combo_sweep_metrics.csv（masswinner 命名）
- 部分 rerun / phaseB 文件名通过别名表对齐到已知 val（见脚本内 ALIAS_VAL_ACC）

用法（仓库根）:
  python tools/build_deepseek_zyn_comparison_table.py \\
    --deepseek-json reports/deepseek/deepseek_ens_r3_vote_zyn.json \\
    --out-csv reports/deepseek_zyn_comparison_table.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset.glevel_labels import parse_overall_glevel_value

# 无独立日志时，与已知训练/推理管线对齐的近似 val_acc（官方 val 64 条）
ALIAS_VAL_ACC: dict[str, float] = {
    # 与 batch_infer seed37_archive_inferbias070 同 ckpt + infer_bias
    "submission_rerun_seed37_archive_inferbias070.csv": 0.6032,
    "submission_ours_seed37_inferbias_0_0p7_0.csv": 0.6032,
    # 与 seed37_archive_S_ref_plateau（无 infer_bias）一致
    "submission_rerun_seed37_archive_plain.csv": 0.5873,
    "submission_batch_infer_seed37_archive_inferbias070.csv": 0.6032,
    "submission_batch_infer_seed37_archive_S_ref_plateau.csv": 0.5873,
    # PhaseB：与 batch_infer_phaseB_* 同权重文件
    "submission_phaseB_ce_bal_seed42.csv": 0.5238,
    "submission_phaseB_ce_macrof1_seed42.csv": 0.5238,
    "submission_phaseB_coral_bal_seed42.csv": 0.5079,
    "submission_phaseB_ce_bal_classweight_auto_seed42.csv": 0.4921,
    # PhaseB batch 导出文件名（TSV 已含；此处防重复键）
    "submission_batch_infer_phaseB_ce_bal_s42.csv": 0.5238,
    "submission_batch_infer_phaseB_ce_macrof1_s42.csv": 0.5238,
    "submission_batch_infer_phaseB_coral_bal_s42.csv": 0.5079,
    "submission_batch_infer_phaseB_ce_classweight_s42.csv": 0.4921,
    "submission_batch_infer_phaseC_pseudo_finetune_s37.csv": 0.5238,
    # Nanbeige 归档 ≈ S_ref_plateau seed37
    "submission_ours_seed37_best_nanbeige.csv": 0.5873,
}


def _read_sub_raw(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        lab = "g_level_pred" if "g_level_pred" in fields else "g_level"
        if lab not in fields:
            raise ValueError(f"{path}: need g_level_pred or g_level")
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


def _dist_counts(pred: dict[str, int]) -> tuple[int, int, int]:
    c = {1: 0, 2: 0, 3: 0}
    for v in pred.values():
        if v in c:
            c[v] += 1
    return c[1], c[2], c[3]


def _parse_batch_val(metrics: str) -> float | None:
    m = re.search(r"acc=([\d.]+)", metrics)
    return float(m.group(1)) if m else None


def _build_val_acc_map(root: Path) -> dict[str, float]:
    m: dict[str, float] = dict(ALIAS_VAL_ACC)

    log_csv = root / "experiments/glevel_improvement_plan/EXPERIMENT_LOG.csv"
    if log_csv.is_file():
        with log_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = (row.get("submission_csv") or "").strip()
                va = (row.get("val_acc") or "").strip()
                if not p or not va:
                    continue
                m[Path(p).name] = float(va)

    tsv = root / "experiments/glevel_improvement_plan/batch_infer_eval_summary.tsv"
    if tsv.is_file():
        with tsv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                sub = (row.get("submission_csv") or "").strip()
                vm = row.get("val_metrics") or ""
                acc = _parse_batch_val(vm)
                if sub and acc is not None:
                    m[Path(sub).name] = acc

    mass = (
        root
        / "experiments/gpu_combo_sweep/mass_20260509_145045/combo_sweep_metrics.csv"
    )
    if mass.is_file():
        with mass.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                combo = row.get("combo_id") or ""
                seed = row.get("seed") or ""
                va = row.get("val_acc") or ""
                if not combo or not seed or va in ("", "NA"):
                    continue
                try:
                    acc = float(va)
                except ValueError:
                    continue
                fname = f"submission_masswinner_{combo}_seed{seed}.csv"
                m[fname] = acc

    return m


def _deepseek_mean_val_acc(root: Path, pattern: str = "deepseek_ens_r3_report_zyn_run*.json") -> float | None:
    dk_dir = root / "reports" / "deepseek"
    reports = sorted(dk_dir.glob(pattern))
    if not reports:
        return None
    accs: list[float] = []
    for p in reports:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            v = obj.get("val_accuracy")
            if isinstance(v, (int, float)):
                accs.append(float(v))
        except Exception:
            continue
    return sum(accs) / len(accs) if accs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepseek-json", type=Path, default=Path("reports/deepseek/deepseek_ens_r3_vote_zyn.json"))
    ap.add_argument("--peer-dir", type=Path, default=Path("external/submissions_peer"))
    ap.add_argument("--pseudo-csv", type=Path, default=Path("external/submissions_peer/test_pseudo_iter2_peers4_inferbias_top6_mass.csv"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/deepseek_zyn_comparison_table.csv"))
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    dk_path = args.deepseek_json if args.deepseek_json.is_absolute() else root / args.deepseek_json
    peer = args.peer_dir if args.peer_dir.is_absolute() else root / args.peer_dir
    pseudo_path = args.pseudo_csv if args.pseudo_csv.is_absolute() else root / args.pseudo_csv
    out_csv = args.out_csv if args.out_csv.is_absolute() else root / args.out_csv

    ref = _load_deepseek(dk_path)
    val_map = _build_val_acc_map(root)
    dk_mean_val = _deepseek_mean_val_acc(root)

    rows: list[dict[str, object]] = []

    def add_row(
        label: str,
        file_name: str,
        pred: dict[str, int],
        val_acc: float | None,
        note: str = "",
    ) -> None:
        low, med, high = _dist_counts(pred)
        n = low + med + high
        ids = sorted(set(ref) & set(pred))
        overlap = (
            sum(1 for i in ids if ref[i] == pred[i]) / len(ids) if ids else None
        )
        rows.append(
            {
                "label": label,
                "file": file_name,
                "val_acc_official_64": round(val_acc, 4) if val_acc is not None else "",
                "val_acc_note": note,
                "test_n": n,
                "test_glevel_1": low,
                "test_glevel_2": med,
                "test_glevel_3": high,
                "test_123_counts": f"{low}/{med}/{high}",
                "test_123_pct": (
                    f"{round(100 * low / n, 1)}%/{round(100 * med / n, 1)}%/{round(100 * high / n, 1)}%"
                    if n
                    else ""
                ),
                "test_glevel_1_pct": round(100 * low / n, 1) if n else "",
                "test_glevel_2_pct": round(100 * med / n, 1) if n else "",
                "test_glevel_3_pct": round(100 * high / n, 1) if n else "",
                "overlap_vs_deepseek_zyn": round(overlap, 4) if overlap is not None else "",
                "overlap_n": len(ids),
            }
        )

    low0, med0, high0 = _dist_counts(ref)
    nref = len(ref)
    rows.append(
        {
            "label": "DeepSeek_zyn_majority_vote",
            "file": dk_path.name,
            "val_acc_official_64": round(dk_mean_val, 4) if dk_mean_val is not None else "",
            "val_acc_note": "interactive_pipeline_val（三跑平均）",
            "test_n": nref,
            "test_glevel_1": low0,
            "test_glevel_2": med0,
            "test_glevel_3": high0,
            "test_123_counts": f"{low0}/{med0}/{high0}",
            "test_123_pct": (
                f"{round(100 * low0 / nref, 1)}%/{round(100 * med0 / nref, 1)}%/{round(100 * high0 / nref, 1)}%"
                if nref
                else ""
            ),
            "test_glevel_1_pct": round(100 * low0 / nref, 1) if ref else "",
            "test_glevel_2_pct": round(100 * med0 / nref, 1) if ref else "",
            "test_glevel_3_pct": round(100 * high0 / nref, 1) if ref else "",
            "overlap_vs_deepseek_zyn": 1.0,
            "overlap_n": len(ref),
        }
    )

    if pseudo_path.is_file():
        pp = _sub_to_123(_read_sub_raw(pseudo_path))
        add_row(
            "pseudo_iter2_peers4_top6_mass",
            pseudo_path.name,
            pp,
            None,
            "无单一模型 val",
        )

    subs_paths = sorted(
        p for p in peer.glob("submission*.csv") if not p.name.startswith("test_pseudo")
    )
    for p in subs_paths:
        try:
            pred = _sub_to_123(_read_sub_raw(p))
        except Exception:
            continue
        if len(pred) < 100:
            continue
        va = val_map.get(p.name)
        note = ""
        if va is None:
            note = "仓库内未解析到 val_acc"
        add_row(p.stem, p.name, pred, va, note)

    rows_sorted = [rows[0]] + sorted(rows[1:], key=lambda r: (-float(r["overlap_vs_deepseek_zyn"] or -1), str(r["file"])))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "file",
        "val_acc_official_64",
        "val_acc_note",
        "test_n",
        "test_glevel_1",
        "test_glevel_2",
        "test_glevel_3",
        "test_123_counts",
        "test_123_pct",
        "test_glevel_1_pct",
        "test_glevel_2_pct",
        "test_glevel_3_pct",
        "overlap_vs_deepseek_zyn",
        "overlap_n",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_sorted:
            w.writerow(row)

    print(f"[wrote] {out_csv}")
    print("\nMarkdown（全表，按 vs DeepSeek 重叠率降序；DeepSeek 置顶）:\n")
    print(
        "| 条目 | 官方验证集准确率(val 64) | 测试集 g_level 1/2/3 分布（占比） | vs DeepSeek zyn 重叠率 | 备注 |"
    )
    print("|------|--------------------------|-----------------------------------|------------------------|------|")
    for row in rows_sorted:
        lmh = row.get("test_123_counts") or f"{row['test_glevel_1']}/{row['test_glevel_2']}/{row['test_glevel_3']}"
        pct = row.get("test_123_pct") or ""
        dist = f"{lmh}（{pct}）" if pct else str(lmh)
        va = row["val_acc_official_64"]
        va_s = f"{va}" if va != "" else "—"
        ov = row["overlap_vs_deepseek_zyn"]
        ov_s = f"{float(ov):.4f}" if ov != "" else "—"
        note = str(row.get("val_acc_note") or "").replace("|", "/")
        if not note and va == "":
            note = "仓库未解析到 val"
        lab = str(row["file"])
        print(f"| {lab} | {va_s} | {dist} | {ov_s} | {note[:24]} |")


if __name__ == "__main__":
    main()
