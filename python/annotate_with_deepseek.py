#!/usr/bin/env python3
"""
DeepSeek Chat API：先用官方训练/验证集「带标签示例」做校准（warmup），再标注测试集 overall_glevel。

默认流程（推荐）：
  1) 从 train/val CSV + 对应 *_text 目录抽样若干条（按 g_level 分层）
  2) 调用 API 一次：让模型阅读示例并输出 JSON rubric_summary（贴合本赛题刻度）
  3) 对测试集 130 条：每条 system prompt = 原始规则 + rubric_summary

路径默认优先「脚本所在仓库的 data/」下同名 CSV 与 train_text/val_text/test_text；
若不存在则回退到 ${SUPERLU_DATASET:-/data/Super-Lu/dataset}/…（与 scripts/glevel_train.sh 一致）。

示例：
  python python/annotate_with_deepseek.py --dry-run
  python python/annotate_with_deepseek.py --warmup-only --context-cache reports/deepseek/deepseek_context_cache.json
  python python/annotate_with_deepseek.py --resume --output reports/deepseek/deepseek_annotations_v2.json

关闭「先理解标签」、退回纯规则：
  python python/annotate_with_deepseek.py --no-labeled-context

优先 DEEPSEEK_API_KEY；否则使用脚本内后备密钥。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dataset.glevel_labels import glevel_csv_to_int, parse_overall_glevel_value
_SUP_DATASET = Path(os.environ.get("SUPERLU_DATASET", "/data/Super-Lu/dataset"))


def _pick_csv(fname: str) -> str:
    loc = _REPO / "data" / fname
    return str(loc) if loc.is_file() else str(_SUP_DATASET / fname)


def _pick_text_dir(dirname: str) -> str:
    loc = _REPO / "data" / dirname
    return str(loc) if loc.is_dir() else str(_SUP_DATASET / dirname)


DEFAULT_TEST_CSV = _pick_csv("test_data_basic_information.csv")
DEFAULT_TEXT_DIR = _pick_text_dir("test_text")
DEFAULT_TRAIN_CSV = _pick_csv("train_data.csv")
DEFAULT_VAL_CSV = _pick_csv("val_data.csv")
DEFAULT_TRAIN_TEXT = _pick_text_dir("train_text")
DEFAULT_VAL_TEXT = _pick_text_dir("val_text")
DEFAULT_OUT = str(_REPO / "reports/deepseek/deepseek_annotations.json")
DEFAULT_CONTEXT_CACHE = str(_REPO / "reports/deepseek/deepseek_context_cache.json")

BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions")
MODEL_NAME = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
_FALLBACK_API_KEY = "sk-a7066ed8a57d4d9f90d74a8561b9272c"

QUESTIONS = [
    "What would you consider among your greatest strengths and weaknesses as an employee?",
    "How would your best friend describe you?",
    (
        "Think of situations when you made professional decisions that could affect your status or "
        "how much money you make. How do you usually behave in such situations? Why do you think that is?"
    ),
    (
        "Think of situations when you joined a new team of people. How do you usually behave when you "
        "enter a new team? Why do you think that is?"
    ),
    (
        "Think of situations when someone annoyed you. How do you usually react in such situations? "
        "Why do you think that is?"
    ),
    (
        "Think of situations when your work or workspace were not very organized. How typical is that "
        "of you? Why do you think that is?"
    ),
]

SYSTEM_PROMPT_BASE = """You are a highly specialized expert in evaluating cognitive ability from job interview responses. Your task is strictly to classify a candidate's overall cognitive level.

Official label scale (must match training data — integers only):
- g_level 1: Lower cognitive depth / organization relative to stronger candidates in this dataset.
- g_level 2: Adequate responses with moderate structure or reasoning.
- g_level 3: Stronger clarity, structure, reasoning, and insight relative to typical answers in this dataset.

General linguistic cues (secondary to dataset calibration below):
- g_level 3: Consistently clear logic, well-structured answers, deeper reasoning, concrete examples where relevant.
- g_level 2: Acceptable but limited depth; uneven structure or reasoning.
- g_level 1: Vague or thin reasoning, weak organization, mostly superficial content.

Critical calibration rules for THIS competition:
- Labels are RELATIVE within the pool of interview transcripts: g_level 3 means "stronger than typical here", not "perfect English".
- The official training set uses all three levels with meaningful frequency. Do NOT collapse almost everything to g_level 1 just because answers are spoken, disfluent, or short—ASR noise and fillers are common for levels 2 and 3 in this dataset.
- Reserve g_level 1 for answers that are thin, off-topic, or lack coherent reasoning across questions; use g_level 3 when there is clear self-reflection, causal reasoning, or structured insight repeated across several answers, even if delivery is imperfect.

When judging test candidates, align your thresholds with the dataset-specific calibration section that follows (derived from official labeled train/validation examples).

Output format for TEST labeling calls: a valid JSON object ONLY:
{"overall_glevel": <integer 1, 2, or 3>, "confidence": float in [0,1]}
No other keys unless specified.
"""


def _api_key() -> str:
    return os.environ.get("DEEPSEEK_API_KEY", "").strip() or _FALLBACK_API_KEY


def _clip(s: str, max_chars: int) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + "\n...[truncated]..."


def load_answers(text_dir: Path, sid: str) -> tuple[list[str], list[str]]:
    answers: list[str] = []
    warnings: list[str] = []
    for q in range(1, 7):
        txt_path = text_dir / f"{sid}_q{q}.txt"
        if txt_path.exists():
            answers.append(txt_path.read_text(encoding="utf-8", errors="replace").strip())
        else:
            answers.append("[MISSING TRANSCRIPT]")
            warnings.append(str(txt_path))
    return answers, warnings


def format_labeled_block(
    sid: str,
    answers: list[str],
    official_g_level: int,
    source: str,
    max_chars: int,
) -> str:
    lines = [
        f"=== OFFICIAL EXAMPLE ({source}) id={sid} official_g_level={official_g_level} ===",
    ]
    for i, q in enumerate(QUESTIONS):
        ans = answers[i] if i < len(answers) else ""
        lines.append(f"Q{i + 1}: {q}")
        lines.append(f"A{i + 1}: {_clip(ans, max_chars)}")
    return "\n".join(lines) + "\n"


def build_few_shot_corpus(
    train_csv: Path,
    val_csv: Path,
    train_text: Path,
    val_text: Path,
    *,
    per_class_train: int,
    per_class_val: int,
    seed: int,
    max_chars: int,
) -> str:
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    if "g_level" not in train_df.columns or "id" not in train_df.columns:
        raise SystemExit("train CSV 需要列 id, g_level")
    if "g_level" not in val_df.columns or "id" not in val_df.columns:
        raise SystemExit("val CSV 需要列 id, g_level")

    rng = __import__("random").Random(seed)
    blocks: list[str] = []

    intro = (
        "Below are REAL samples from the competition training and validation splits.\n"
        "Each block gives six interview answers and the OFFICIAL overall integer g_level "
        "(must be 1, 2, or 3). Use them to infer how organizers calibrated difficulty.\n"
    )

    for split_name, df, text_root in (
        ("train", train_df, train_text),
        ("val", val_df, val_text),
    ):
        per = per_class_train if split_name == "train" else per_class_val
        if per <= 0:
            continue
        by_c: dict[int, list[str]] = defaultdict(list)
        for _, row in df.iterrows():
            gid = str(row["id"]).strip()
            glab = glevel_csv_to_int(row.get("g_level"))
            if not gid or glab is None:
                continue
            by_c[glab].append(gid)
        for cls in (1, 2, 3):
            ids = by_c.get(cls, [])
            rng.shuffle(ids)
            pick = ids[:per]
            for sid in pick:
                ans, warns = load_answers(text_root, sid)
                if len(warns) >= 4:
                    continue
                blocks.append(format_labeled_block(sid, ans, cls, split_name, max_chars))

    return intro + "\n".join(blocks)


def run_warmup_rubric(api_key: str, few_shot_corpus: str, *, timeout: int = 180) -> str:
    """单次 API：阅读带标签示例，输出 rubric_summary。"""
    sys_msg = (
        "You are calibrating annotation thresholds for a tri-class cognitive-level task.\n"
        "Read ONLY the labeled examples provided by the user. They are authoritative.\n"
        "Write a concise rubric (English) explaining how g_level 1 vs 2 vs 3 manifests "
        "on THIS dataset (speech-interview transcripts). Focus on observable differences across examples.\n"
        'Reply with JSON ONLY: {"rubric_summary": "<=1200 chars string>"}'
    )
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": few_shot_corpus},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(BASE_URL, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return str(parsed.get("rubric_summary") or parsed.get("summary") or "").strip()


def build_system_for_test(rubric_summary: str) -> str:
    if not rubric_summary:
        return SYSTEM_PROMPT_BASE
    return (
        SYSTEM_PROMPT_BASE
        + "\n\n# Dataset-specific calibration (from official labeled train/val)\n"
        + rubric_summary
    )


def annotate_candidate(
    api_key: str,
    answers: list[str],
    *,
    system_prompt: str,
    temperature: float = 0.01,
    retries: int = 3,
    timeout: int = 120,
) -> dict:
    user_prompt = (
        "Classify this NEW candidate's overall cognitive level using ONLY the official integer "
        "g_level: 1, 2, or 3 (meanings defined in the system prompt).\n\n"
    )
    for i, q in enumerate(QUESTIONS):
        ans = answers[i] if i < len(answers) else "[NO ANSWER]"
        user_prompt += f"Question {i + 1}: {q}\nAnswer: {ans}\n\n"
    user_prompt += (
        'Output JSON only: {"overall_glevel": <1|2|3>, "confidence": 0.0-1.0}\n'
        "overall_glevel must be the integer 1, 2, or 3 (not strings). Choose exactly one according to the system rubric."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err: str | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(BASE_URL, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            gl = parse_overall_glevel_value(parsed.get("overall_glevel"))
            if gl is None:
                parsed["overall_glevel"] = 2
            else:
                parsed["overall_glevel"] = gl
            if "confidence" not in parsed:
                parsed["confidence"] = 0.5
            return parsed
        except Exception as e:
            last_err = str(e)
            time.sleep(2**attempt)
    return {"overall_glevel": "ERROR", "confidence": 0.0, "error": last_err or "API call failed"}


def _cache_sig(cfg: dict, corpus: str) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(cfg, sort_keys=True).encode())
    h.update(corpus.encode())
    return h.hexdigest()[:16]


def _is_done(rec: object) -> bool:
    if not isinstance(rec, dict):
        return False
    if str(rec.get("overall_glevel", "")).upper() == "ERROR":
        return False
    return parse_overall_glevel_value(rec.get("overall_glevel")) is not None


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek：train/val 校准后标注测试集")
    ap.add_argument("--test-csv", type=Path, default=Path(DEFAULT_TEST_CSV))
    ap.add_argument("--text-dir", type=Path, default=Path(DEFAULT_TEXT_DIR))
    ap.add_argument("--train-csv", type=Path, default=Path(DEFAULT_TRAIN_CSV))
    ap.add_argument("--val-csv", type=Path, default=Path(DEFAULT_VAL_CSV))
    ap.add_argument("--train-text-dir", type=Path, default=Path(DEFAULT_TRAIN_TEXT))
    ap.add_argument("--val-text-dir", type=Path, default=Path(DEFAULT_VAL_TEXT))
    ap.add_argument("--output", "-o", type=Path, default=Path(DEFAULT_OUT))
    ap.add_argument("--context-cache", type=Path, default=Path(DEFAULT_CONTEXT_CACHE))
    ap.add_argument("--per-class-train", type=int, default=2)
    ap.add_argument("--per-class-val", type=int, default=1)
    ap.add_argument("--max-chars", type=int, default=480, help="每条答案在示例里的最大字符")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-labeled-context", action="store_true", help="不做 train/val 校准（旧行为）")
    ap.add_argument("--skip-warmup", action="store_true", help="不调 warmup API；仅用缓存里的 rubric_summary")
    ap.add_argument("--rebuild-context", action="store_true", help="忽略缓存摘要，强制重做 warmup")
    ap.add_argument("--warmup-only", action="store_true", help="只构建示例并调用 warmup，写入 context-cache 后退出")
    args = ap.parse_args()

    test_csv = args.test_csv.expanduser().resolve()
    text_dir = args.text_dir.expanduser().resolve()
    out_path = args.output.expanduser().resolve()
    cache_path = args.context_cache.expanduser().resolve()

    if not test_csv.is_file():
        raise SystemExit(f"找不到 TEST_CSV: {test_csv}")
    if not text_dir.is_dir():
        raise SystemExit(f"找不到 TEXT_DIR: {text_dir}")

    api_key = _api_key()
    if not args.dry_run and not api_key:
        raise SystemExit("缺少 API Key")

    cfg = {
        "per_class_train": args.per_class_train,
        "per_class_val": args.per_class_val,
        "max_chars": args.max_chars,
        "sample_seed": args.sample_seed,
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
    }

    rubric_summary = ""
    few_shot_corpus = ""

    if not args.no_labeled_context:
        train_csv = args.train_csv.expanduser().resolve()
        val_csv = args.val_csv.expanduser().resolve()
        train_txt = args.train_text_dir.expanduser().resolve()
        val_txt = args.val_text_dir.expanduser().resolve()
        for p, name in (
            (train_csv, "train_csv"),
            (val_csv, "val_csv"),
            (train_txt, "train_text_dir"),
            (val_txt, "val_text_dir"),
        ):
            if not p.exists():
                raise SystemExit(f"找不到 {name}: {p}")
        few_shot_corpus = build_few_shot_corpus(
            train_csv,
            val_csv,
            train_txt,
            val_txt,
            per_class_train=args.per_class_train,
            per_class_val=args.per_class_val,
            seed=args.sample_seed,
            max_chars=args.max_chars,
        )
        sig = _cache_sig(cfg, few_shot_corpus)

        loaded = False
        if cache_path.is_file() and not args.rebuild_context:
            try:
                blob = json.loads(cache_path.read_text(encoding="utf-8"))
                if blob.get("sig") == sig and blob.get("rubric_summary"):
                    rubric_summary = str(blob["rubric_summary"])
                    loaded = True
                    print(f"[annotate] loaded rubric from cache {cache_path}")
            except Exception:
                pass

        print(f"[annotate] few_shot chars={len(few_shot_corpus)} cache_sig={sig}")

        if args.dry_run:
            print("[annotate] dry-run: labeled context built; skip API")
            print(f"[annotate] rubric loaded_from_cache={loaded}")
            return

        if args.warmup_only:
            summary = run_warmup_rubric(api_key, few_shot_corpus)
            cache_path.write_text(
                json.dumps(
                    {
                        "sig": sig,
                        "config": cfg,
                        "model": MODEL_NAME,
                        "rubric_summary": summary,
                        "few_shot_chars": len(few_shot_corpus),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"[annotate] warmup-only done → {cache_path}")
            return

        if not loaded and not args.skip_warmup:
            rubric_summary = run_warmup_rubric(api_key, few_shot_corpus)
            cache_path.write_text(
                json.dumps(
                    {
                        "sig": sig,
                        "config": cfg,
                        "model": MODEL_NAME,
                        "rubric_summary": rubric_summary,
                        "few_shot_chars": len(few_shot_corpus),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"[annotate] warmup saved → {cache_path}")
        elif args.skip_warmup and not rubric_summary:
            raise SystemExit("--skip-warmup 需要已有缓存 rubric_summary（先 warmup-only 或跑一次完整流程）")

    system_prompt = (
        SYSTEM_PROMPT_BASE if args.no_labeled_context else build_system_for_test(rubric_summary)
    )

    df = pd.read_csv(test_csv)
    test_ids = [str(x).strip() for x in df["id"].tolist() if str(x).strip()]
    if args.limit > 0:
        test_ids = test_ids[: args.limit]

    results: dict[str, object] = {}
    if args.resume and out_path.is_file():
        results = json.loads(out_path.read_text(encoding="utf-8"))

    print(f"[annotate] TEST_CSV={test_csv} n={len(test_ids)} MODEL={MODEL_NAME}")
    print(f"[annotate] labeled_context={not args.no_labeled_context} rubric_chars={len(rubric_summary)}")

    for sid in tqdm(test_ids, desc="Annotating"):
        if args.resume and _is_done(results.get(sid)):
            continue
        answers, warns = load_answers(text_dir, sid)
        annotation = annotate_candidate(api_key, answers, system_prompt=system_prompt)
        annotation["meta"] = {
            "model": MODEL_NAME,
            "missing_transcript_files": len(warns),
            "labeled_context": not args.no_labeled_context,
            "context_cache": str(cache_path) if not args.no_labeled_context else "",
        }
        results[sid] = annotation
        time.sleep(max(0.0, args.sleep))
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[annotate] Done → {out_path}")
    levels = [
        parse_overall_glevel_value(v.get("overall_glevel"))
        for v in results.values()
        if isinstance(v, dict)
    ]
    print("[annotate] distribution:", Counter(x for x in levels if x is not None))


if __name__ == "__main__":
    main()
