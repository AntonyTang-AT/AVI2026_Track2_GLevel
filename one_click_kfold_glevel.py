#!/usr/bin/env python3
"""
g_level 分层 K 折训练 + 提交多数投票融合（纯 Python，避免 .sh 在 CRLF 下续行失效）。

用法（在项目根目录）:
  python one_click_kfold_glevel.py

环境变量与 vote_train_glevel / vote_kfold 一致，例如:
  export KFOLDS=5 KFOLD_SEED=42 KFOLD_OUT_DIR=./kfold_glevel_out
  export GLEVEL_OPT="--temporal_gru --label_smoothing 0.05"  # 显式覆盖下方默认多模态预设
  export KFOLD_MINIMAL_DEFAULT=1  # 仅 --g_level_int_encoding one（轻量，与 multimodal 不对齐）
  export NANBEIGE_TEXT=1 TEXT_DIM=2560 TEXT_TRAIN_DIR=... TEXT_VAL_DIR=... TEXT_TEST_DIR=...  # 与单折训练一致
  export PROJECT_ROOT=/path/to/AVI2026_Track2_GLevel
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
    os.chdir(root)

    train_csv = os.environ.get("TRAIN_CSV", "/data/Super-Lu/dataset/train_data.csv")
    val_csv = os.environ.get("VAL_CSV", "/data/Super-Lu/dataset/val_data.csv")
    test_csv = os.environ.get("TEST_CSV", str(root / "data/test_data_basic_information.csv"))
    rating_csv = os.environ.get("RATING_CSV", "/data/Super-Lu/dataset/train_data.csv")

    feat_train = os.environ.get("FEAT_TRAIN", "/data/Super-Lu/dataset/train_feature")
    feat_val = os.environ.get("FEAT_VAL", "/data/Super-Lu/dataset/val_feature")
    feat_test = os.environ.get("FEAT_TEST", "/data/AVI2026/test_feature")

    nb_sub = os.environ.get("NANBEIGE_TEXT_SUBDIR", "text_nb")
    if os.environ.get("NANBEIGE_TEXT", "0") == "1":
        text_dim = os.environ.get("TEXT_DIM", "2560")
        text_train = os.environ.get("TEXT_TRAIN_DIR", f"{feat_train}/{nb_sub}")
        if "smoke" in nb_sub:
            text_val = os.environ.get("TEXT_VAL_DIR", text_train)
            text_test = os.environ.get("TEXT_TEST_DIR", text_train)
        else:
            text_val = os.environ.get("TEXT_VAL_DIR", f"{feat_val}/{nb_sub}")
            text_test = os.environ.get("TEXT_TEST_DIR", f"{feat_test}/{nb_sub}")
    else:
        text_dim = os.environ.get("TEXT_DIM", "768")
        text_train = os.environ.get("TEXT_TRAIN_DIR", f"{feat_train}/text")
        text_val = os.environ.get("TEXT_VAL_DIR", f"{feat_val}/text")
        text_test = os.environ.get("TEXT_TEST_DIR", f"{feat_test}/text")

    kfold_out = os.environ.get("KFOLD_OUT_DIR", "./kfold_glevel_out")
    kfold_seed = os.environ.get("KFOLD_SEED", "42")
    kfolds = os.environ.get("KFOLDS", "5")

    glevel_extra = shlex.split(os.environ.get("GLEVEL_OPT", ""))
    if not glevel_extra:
        if os.environ.get("KFOLD_MINIMAL_DEFAULT", "").strip() in ("1", "true", "True", "yes"):
            glevel_extra = ["--g_level_int_encoding", "one"]
            print(
                "[one_click_kfold_glevel] KFOLD_MINIMAL_DEFAULT：仅用标签编码预设",
                flush=True,
            )
        else:
            glevel_extra = shlex.split(
                "--g_level_int_encoding one "
                "--glevel_arch shared_mlp --mlp_dropout 0.25 --weight_decay 0.001 "
                "--label_smoothing 0.05 --select_best balanced_acc "
                "--cross_modal_attn --cross_modal_layers 1 --modality_dropout_p 0.12 "
                "--seed 42 --scheduler_min_lr 1e-6 --sampler_medium_boost 1.5"
            )
            print(
                "[one_click_kfold_glevel] GLEVEL_OPT 未设置 → 使用与 "
                "vote_train_glevel_multimodal（MM_MEDIUM_BOOST=1）一致的训练超参",
                flush=True,
            )
    kfold_extra = shlex.split(os.environ.get("KFOLD_EXTRA", ""))

    run_kfold = root / "tools" / "run_kfold_glevel.py"
    cmd: list[str] = [
        sys.executable,
        str(run_kfold),
        "--merge",
        "--folds",
        kfolds,
        "--seed",
        kfold_seed,
        "--train_csv",
        train_csv,
        "--val_csv",
        val_csv,
        "--rating_csv",
        rating_csv,
        "--test_csv",
        test_csv,
        "--audio_dim",
        "512",
        "--video_dim",
        "512",
        "--text_dim",
        text_dim,
        "--audio_dir",
        f"{feat_train}/audio",
        "--video_dir",
        f"{feat_train}/video",
        "--text_dir",
        text_train,
        "--val_audio_dir",
        f"{feat_val}/audio",
        "--val_video_dir",
        f"{feat_val}/video",
        "--val_text_dir",
        text_val,
        "--test_audio_dir",
        f"{feat_test}/audio",
        "--test_video_dir",
        f"{feat_test}/video",
        "--test_text_dir",
        text_test,
        "--out_dir",
        kfold_out,
        "--",
    ]
    cmd.extend(glevel_extra)
    cmd.extend(kfold_extra)

    print("[one_click_kfold_glevel] cwd=", root, flush=True)
    print("[one_click_kfold_glevel] run:", " ".join(cmd[:6]), "...", flush=True)
    subprocess.check_call(cmd, cwd=str(root))

    out = Path(kfold_out)
    if not out.is_absolute():
        out = (root / out).resolve()
    n = int(kfolds)
    subs = [str(out / f"fold{k}_submission.csv") for k in range(n)]
    ensemble = root / "tools" / "ensemble_glevel_csv.py"
    out_csv = out / "submission_glevel_kfold_vote.csv"
    cmd2 = [sys.executable, str(ensemble), "--inputs", *subs, "--out", str(out_csv)]
    print("[one_click_kfold_glevel] ensemble:", cmd2, flush=True)
    subprocess.check_call(cmd2, cwd=str(root))
    print(f"[one_click_kfold_glevel] 融合结果: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
