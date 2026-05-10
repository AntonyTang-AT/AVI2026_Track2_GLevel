#!/usr/bin/env python3
"""从 combo_sweep_metrics.csv 选取验证指标最优的 Top-K，复制 checkpoint 与日志路径。"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path


def _f(x: str) -> float | None:
    if x in ("", "NA", None):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _key(row: dict) -> tuple:
    vb = _f(row.get("val_bal_acc", "")) or -1.0
    va = _f(row.get("val_acc", "")) or -1.0
    vf = _f(row.get("val_macro_f1", "")) or -1.0
    return (vb, va, vf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_csv", type=Path)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--symlink", action="store_true", help="用符号链接代替复制（省空间）")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.metrics_csv.open(newline="", encoding="utf-8")))
    ok = [
        r
        for r in rows
        if str(r.get("exit_code", "")).strip() == "0" and _f(r.get("val_acc", "")) is not None
    ]
    ok.sort(key=_key, reverse=True)
    top = ok[: args.top_k]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def safe_name(s: str) -> str:
        return re.sub(r"[^\w.\-]+", "_", s)[:180]

    for i, r in enumerate(top, start=1):
        ckpt = Path(r.get("output_model", "").strip())
        combo = r.get("combo_id", "")
        seed = r.get("seed", "")
        sr = (r.get("split_round") or "").strip()
        tag = f"rank{i:02d}_{combo}_seed{seed}"
        if sr:
            tag += f"_slot{sr}"
        tag = safe_name(tag)
        dst = args.out_dir / f"{tag}.pth"
        if ckpt.is_file():
            if args.symlink:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(ckpt.resolve())
            else:
                shutil.copy2(ckpt, dst)
        manifest.append(
            {
                "rank": i,
                "combo_id": combo,
                "seed": seed,
                "split_round": sr or None,
                "val_bal_acc": r.get("val_bal_acc"),
                "val_acc": r.get("val_acc"),
                "val_macro_f1": r.get("val_macro_f1"),
                "source_checkpoint": str(ckpt),
                "saved_as": str(dst),
                "train_log": r.get("log"),
            }
        )

    mp = args.out_dir / "top_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[copy_top_k_models] wrote {len(manifest)} entries → {args.out_dir} manifest={mp}")


if __name__ == "__main__":
    main()
