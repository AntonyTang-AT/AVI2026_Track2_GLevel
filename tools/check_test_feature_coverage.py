#!/usr/bin/env python3
"""
检查 test CSV 中的 id 在给定特征根目录下是否「6 题 × 三模态」齐全。
用于确认 FEAT_TEST 是否指对（测试 id 通常不在 train_feature 里）。

示例：
  python tools/check_test_feature_coverage.py \\
    --test_csv ./data/test_data_basic_information.csv \\
    --feat_root /data/AVI2026/test_feature
"""
import argparse
import os
import sys

import pandas as pd


def _list_npy(d):
    if not d or not os.path.isdir(d):
        return []
    return [fn for fn in os.listdir(d) if fn.lower().endswith(".npy")]


def _has_file(fnames, sid, q):
    prefix = f"{str(sid).strip()}_{q}"
    return any(fn.startswith(prefix) and fn.lower().endswith(".npy") for fn in fnames)


def _row_ok(sid, questions, la, lv, lt, require_text: bool):
    for q in questions:
        ok_av = _has_file(la, sid, q) and _has_file(lv, sid, q)
        if require_text:
            ok_av = ok_av and _has_file(lt, sid, q)
        if not ok_av:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument(
        "--feat_root",
        required=True,
        help="特征根目录，其下须有 audio/ video/ text 子目录",
    )
    ap.add_argument(
        "--question",
        nargs="+",
        default=["q1", "q2", "q3", "q4", "q5", "q6"],
    )
    ap.add_argument(
        "--skip_text_in_feat_root",
        action="store_true",
        help=(
            "仅检查 audio+video（6 题）。用于文本仅用 Nanbeige、不在 FEAT_TEST/text 下的布局；"
            "文本请另用 tools/check_text_npy_coverage.py 检查 TEXT_TEST_DIR。"
        ),
    )
    args = ap.parse_args()

    root = os.path.abspath(args.feat_root)
    ad = os.path.join(root, "audio")
    vd = os.path.join(root, "video")
    td = os.path.join(root, "text")
    la, lv, lt = _list_npy(ad), _list_npy(vd), _list_npy(td)

    df = pd.read_csv(args.test_csv)
    if "id" not in df.columns:
        print("CSV 须含 id 列", file=sys.stderr)
        sys.exit(1)

    req_txt = not args.skip_text_in_feat_root
    n = len(df)
    ok = 0
    first_bad = None
    for _, row in df.iterrows():
        sid = row["id"]
        if _row_ok(sid, args.question, la, lv, lt, req_txt):
            ok += 1
        elif first_bad is None:
            first_bad = str(sid).strip()

    print(f"feat_root={root}")
    print(
        f"audio .npy 数量: {len(la)} | video: {len(lv)} | text: {len(lt)}"
        + (" | 未要求 feat_root/text" if args.skip_text_in_feat_root else "")
    )
    print(f"test_csv 行数: {n} | 特征完整可推理: {ok}/{n}")
    if args.skip_text_in_feat_root:
        print(
            "[check_test_feature_coverage] 已跳过 FEAT_TEST/text；"
            "请确保 TEXT_TEST_DIR（Nanbeige）覆盖全部测试 id。",
            file=sys.stderr,
        )
    if ok == 0:
        print(
            "\n无一条可推理：该 feat_root 下没有覆盖测试 id 的 .npy。\n"
            "请向赛方确认测试特征路径（常见为 .../test_feature），或自行对测试集跑提取脚本。\n"
            "训练完成后仅导出预测时：\n"
            "  export FEAT_TEST=/path/to/test_feature\n"
            "  python python/train_task2_glevel.py --only_test --test_model best_model_glevel.pth ...",
            file=sys.stderr,
        )
        sys.exit(1)
    if ok < n:
        print(f"示例缺失特征的 id（首个）: {first_bad}", file=sys.stderr)
        sys.exit(2)
    print("OK: 全部测试 id 在当前 feat_root 下特征完整。")
    sys.exit(0)


if __name__ == "__main__":
    main()
