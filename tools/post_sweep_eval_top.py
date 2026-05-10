#!/usr/bin/env python3
"""读取 combo_sweep_metrics.csv，对 val_acc 最高的若干行在官方 val 上复评（须与训练 arch 一致）。"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _flags_for_combo(combo_id: str) -> list[str]:
    c = combo_id.strip()
    if c == "S_ref_plateau":
        return [
            "--glevel_arch",
            "shared_mlp",
            "--cross_modal_attn",
            "--cross_modal_layers",
            "1",
        ]
    if c == "S_ref_sel_acc":
        return [
            "--glevel_arch",
            "shared_mlp",
            "--cross_modal_attn",
            "--cross_modal_layers",
            "1",
        ]
    if c == "S_ref_step":
        return [
            "--glevel_arch",
            "shared_mlp",
            "--cross_modal_attn",
            "--cross_modal_layers",
            "1",
        ]
    if c == "S_ref_cosine":
        return [
            "--glevel_arch",
            "shared_mlp",
            "--cross_modal_attn",
            "--cross_modal_layers",
            "1",
        ]
    if c == "S_plateau_ln":
        return [
            "--glevel_arch",
            "shared_mlp",
            "--cross_modal_attn",
            "--cross_modal_layers",
            "1",
            "--fused_layer_norm",
        ]
    if c == "S_plateau_ln_sel_acc":
        return [
            "--glevel_arch",
            "shared_mlp",
            "--cross_modal_attn",
            "--cross_modal_layers",
            "1",
            "--fused_layer_norm",
        ]
    if c == "S_step_ln":
        return [
            "--glevel_arch",
            "shared_mlp",
            "--cross_modal_attn",
            "--cross_modal_layers",
            "1",
            "--fused_layer_norm",
        ]
    if c == "AT_plateau":
        return ["--glevel_arch", "audio_text_mlp", "--at_mlp_hidden", "512"]
    if c == "AT_plateau_ln":
        return [
            "--glevel_arch",
            "audio_text_mlp",
            "--at_mlp_hidden",
            "512",
            "--fused_layer_norm",
        ]
    if c == "AT_step_ln":
        return [
            "--glevel_arch",
            "audio_text_mlp",
            "--at_mlp_hidden",
            "512",
            "--fused_layer_norm",
        ]
    raise ValueError(f"未知 combo_id={combo_id!r}")


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "用法: post_sweep_eval_top.py <combo_sweep_metrics.csv> [top_k]",
            file=sys.stderr,
        )
        sys.exit(2)
    csv_path = Path(sys.argv[1])
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    py = os.environ.get("GLEVEL_CUDA_PYTHON", os.environ.get("PYTHON", "python3"))
    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("exit_code", "")).strip() != "0":
                continue
            va = r.get("val_acc", "")
            if va in ("", "NA"):
                continue
            try:
                float(va)
            except (TypeError, ValueError):
                continue
            p = Path(r.get("output_model", ""))
            if p.is_file():
                rows.append(r)
    rows.sort(key=lambda x: float(x["val_acc"]), reverse=True)
    pick = rows[:top_k]
    out_lines: list[str] = []
    for i, r in enumerate(pick, 1):
        combo = r["combo_id"]
        ckpt = r["output_model"]
        extra = _flags_for_combo(combo)
        cmd = [
            py,
            str(_ROOT / "tools/eval_glevel_checkpoint_on_csv.py"),
            "--eval_csv",
            "/data/Super-Lu/dataset/val_data.csv",
            "--rating_csv",
            "/data/Super-Lu/dataset/train_data.csv",
            "--labels_in_split_csv",
            "--g_level_int_encoding",
            "one",
            "--train_audio_dir",
            "/data/Super-Lu/dataset/train_feature/audio",
            "--train_video_dir",
            "/data/Super-Lu/dataset/train_feature/video",
            "--train_text_dir",
            str(_ROOT / "data/text_nb"),
            "--eval_audio_dir",
            "/data/Super-Lu/dataset/val_feature/audio",
            "--eval_video_dir",
            "/data/Super-Lu/dataset/val_feature/video",
            "--eval_text_dir",
            str(_ROOT / "data/text_nb_val"),
            "--checkpoint",
            ckpt,
            "--text_dim",
            "2560",
            "--mlp_dropout",
            "0.25",
            "--modality_dropout_p",
            "0.12",
            "--num_workers",
            "2",
            *extra,
        ]
        out_lines.append(f"\n### [{i}] {combo} seed={r['seed']} train_val_acc={r['val_acc']}\n")
        out_lines.append(" ".join(cmd) + "\n")
        p = subprocess.run(cmd, capture_output=True, text=True)
        out_lines.append(p.stdout)
        if p.stderr:
            out_lines.append(p.stderr)
        out_lines.append(f"exit_code={p.returncode}\n")
    report = csv_path.parent / "post_eval_top_val.txt"
    report.write_text("".join(out_lines), encoding="utf-8")
    print(f"[post_sweep_eval_top] wrote {report}", flush=True)


if __name__ == "__main__":
    main()
