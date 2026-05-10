"""
分层 K 折训练 g_level：合并官方 train/val 表后 StratifiedKFold，每折调用 train_task2_glevel.py。
合并后若部分 id 仅在 val_feature 下有 .npy，须加 --merge 且训练进程会带 --train_feat_fallback。

用法（在项目根目录执行）:
  python tools/run_kfold_glevel.py --merge --folds 5 --seed 42 \\
    --train_csv /data/Super-Lu/dataset/train_data.csv \\
    --val_csv /data/Super-Lu/dataset/val_data.csv \\
    --rating_csv /data/Super-Lu/dataset/train_data.csv \\
    --test_csv ./data/test_data_basic_information.csv \\
    --audio_dir /data/Super-Lu/dataset/train_feature/audio \\
    --video_dir /data/Super-Lu/dataset/train_feature/video \\
    --text_dir /data/Super-Lu/dataset/train_feature/text \\
    --val_audio_dir /data/Super-Lu/dataset/val_feature/audio \\
    --val_video_dir /data/Super-Lu/dataset/val_feature/video \\
    --val_text_dir /data/Super-Lu/dataset/val_feature/text \\
    --test_audio_dir /data/AVI2026/test_feature/audio \\
    --test_video_dir /data/AVI2026/test_feature/video \\
    --test_text_dir /data/AVI2026/test_feature/text \\
    --out_dir ./kfold_glevel_out \\
    -- \\
    --text_dim 768 --batch_size 32

融合预测:
  python tools/ensemble_glevel_csv.py --inputs kfold_glevel_out/fold*_submission.csv \\
    --out submission_glevel_kfold_vote.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _split_argv() -> tuple[list[str], list[str]]:
    if "--" in sys.argv:
        i = sys.argv.index("--")
        return sys.argv[1:i], sys.argv[i + 1 :]
    return sys.argv[1:], []


def main() -> None:
    argv, train_extra = _split_argv()
    root = Path(__file__).resolve().parent.parent
    train_py = root / "python" / "train_task2_glevel.py"

    p = argparse.ArgumentParser(
        description="g_level StratifiedKFold，多次调用 train_task2_glevel.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument(
        "--merge",
        action="store_true",
        help="合并 train_csv 与 val_csv（按 id 去重，保留先出现的行）后再划分折",
    )
    p.add_argument("--out_dir", type=str, default="kfold_glevel_out")
    p.add_argument("--rating_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--label_col", type=str, default="g_level")
    p.add_argument("--audio_dim", type=int, default=512)
    p.add_argument("--video_dim", type=int, default=512)
    p.add_argument("--text_dim", type=int, default=768)
    p.add_argument("--audio_dir", type=str, required=True)
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--text_dir", type=str, required=True)
    p.add_argument("--val_audio_dir", type=str, required=True)
    p.add_argument("--val_video_dir", type=str, required=True)
    p.add_argument("--val_text_dir", type=str, required=True)
    p.add_argument("--test_audio_dir", type=str, required=True)
    p.add_argument("--test_video_dir", type=str, required=True)
    p.add_argument("--test_text_dir", type=str, required=True)
    args = p.parse_args(argv)

    if args.merge:
        tr = pd.read_csv(args.train_csv)
        va = pd.read_csv(args.val_csv)
        full = pd.concat([tr, va], axis=0).drop_duplicates(subset=["id"], keep="first")
        full = full.reset_index(drop=True)
    else:
        full = pd.read_csv(args.train_csv).reset_index(drop=True)

    if args.label_col not in full.columns:
        raise SystemExit(
            f"划分用表缺少标签列 {args.label_col!r}，请使用含 g_level 的 csv 或改 --label_col。"
            f" 当前列: {list(full.columns)}"
        )

    y = full[args.label_col].astype(str).to_numpy()
    n = len(full)
    if n < args.folds:
        raise SystemExit(f"样本数 {n} < 折数 {args.folds}")

    counts = pd.Series(y).value_counts()
    if counts.min() < 2:
        print(
            "[run_kfold_glevel] 警告: 某类样本 <2，StratifiedKFold 可能失败；"
            " 可减小 --folds 或检查标签分布。",
            file=sys.stderr,
        )

    out_dir = (root / args.out_dir).resolve() if not os.path.isabs(args.out_dir) else Path(args.out_dir)
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    X_idx = np.arange(n)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_idx, y)):
        fold_train = full.iloc[tr_idx].reset_index(drop=True)
        fold_val = full.iloc[va_idx].reset_index(drop=True)
        ftp = split_dir / f"fold{fold}_train.csv"
        fvp = split_dir / f"fold{fold}_val.csv"
        fold_train.to_csv(ftp, index=False)
        fold_val.to_csv(fvp, index=False)
        print(f"[fold {fold}] train={len(fold_train)} val={len(fold_val)} → {ftp.name}", flush=True)

        model_path = out_dir / f"fold{fold}_best_model_glevel.pth"
        sub_path = out_dir / f"fold{fold}_submission.csv"
        loss_path = out_dir / f"fold{fold}_loss.png"

        cmd = [
            sys.executable,
            str(train_py),
            "--train_csv",
            str(ftp),
            "--val_csv",
            str(fvp),
            "--rating_csv",
            args.rating_csv,
            "--test_csv",
            args.test_csv,
            "--labels_in_split_csv",
            "--label_col",
            args.label_col,
            "--question",
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "--audio_dim",
            str(args.audio_dim),
            "--video_dim",
            str(args.video_dim),
            "--text_dim",
            str(args.text_dim),
            "--audio_dir",
            args.audio_dir,
            "--video_dir",
            args.video_dir,
            "--text_dir",
            args.text_dir,
            "--val_audio_dir",
            args.val_audio_dir,
            "--val_video_dir",
            args.val_video_dir,
            "--val_text_dir",
            args.val_text_dir,
            "--test_audio_dir",
            args.test_audio_dir,
            "--test_video_dir",
            args.test_video_dir,
            "--test_text_dir",
            args.test_text_dir,
            "--output_model",
            str(model_path),
            "--test_output_csv",
            str(sub_path),
            "--loss_plot_path",
            str(loss_path),
        ]
        if args.merge:
            cmd.append("--train_feat_fallback")
        cmd.extend(train_extra)

        print(f"[fold {fold}] subprocess: {' '.join(cmd[:8])} ...", flush=True)
        r = subprocess.run(cmd, cwd=str(root))
        if r.returncode != 0:
            raise SystemExit(f"fold {fold} 训练失败 exit={r.returncode}")

    subs = sorted(out_dir.glob("fold*_submission.csv"))
    print(
        "\nK 折完成。融合示例:\n"
        f"  python tools/ensemble_glevel_csv.py --inputs {' '.join(str(s) for s in subs)} "
        f"--out {out_dir / 'submission_glevel_kfold_vote.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
