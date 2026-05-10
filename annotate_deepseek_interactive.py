#!/usr/bin/env python3
"""
DeepSeek 交互式管线（按用户指定顺序）：
1) 训练集：按 --train-min-examples / --train-max-examples（默认 300～500）抽取带标签样本，分批上传；
   每批基于当前 rubric 草稿迭代更新（避免单条对话无限膨胀）。
2) 验证集：逐条预测 → 与官方标签比对 → 错误则注入真实标签让模型自省后再继续。
3) 记录验证集准确率；将验证阶段对话压缩为 rubric_summary。
4) 测试集：用「规则 + 最终 rubric_summary」单轮 JSON 标注（温度可调，减轻类别塌缩）。

可选：`--pipeline-runs 3` 将上述全流程独立重复 3 次（每轮不同训练抽样 seed 与测试温度偏移），
对测试集标签做多数投票写入 `-o`，各轮明细为 `*_run0.json` / `*_run1.json` …

学长提交择优（官方测试准确率写在文件名里，如 submission1_0.53077.csv）：
  在 train→val 完成后，不向 DeepSeek 逐条重标测试集，而是把「测试集 id + 四份提交的预测」
  以及「各文件对应的官方平台准确率」一并交给模型，由其 **全局选定保留其中一整份提交**，
  并写出 `--peer-final-out-csv`（默认列 id,g_level_pred）。
  **择优元数据写入独立文件**：默认 ``<report>_peer_sel.json``、``<output>_peer_sel.json``，
  也可用 ``--peer-final-report`` / ``--peer-final-json`` 指定；**默认不再改写主 --report**，
  以免与正在跑的交互管线冲突（需要时可加 ``--peer-write-main-report-pointer``）。
  **并行校准**：``--peer-calibrate-from-report path/to/已完成_report.json`` 只读加载 rubric，
  跳过 train/val，直接择优；默认产物与源报告同目录 ``*_peer_sel*``，不占用正在跑的 run 文件。

示例：
  python annotate_deepseek_interactive.py -o deepseek_interactive_test.json \\
    --report deepseek_interactive_report.json

断点（验证/测试进度）：加 --resume；仅重跑测试（需已有 report 里的 rubric_summary）：--test-only
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# 复用与 annotate_with_deepseek 一致的默认路径与常量
from annotate_with_deepseek import (
    BASE_URL,
    DEFAULT_TEST_CSV,
    DEFAULT_TEXT_DIR,
    DEFAULT_TRAIN_CSV,
    DEFAULT_VAL_CSV,
    DEFAULT_TRAIN_TEXT,
    DEFAULT_VAL_TEXT,
    MODEL_NAME,
    QUESTIONS,
    _api_key,
    _clip,
    annotate_candidate,
    format_labeled_block,
    load_answers,
)
from glevel_labels import glevel_csv_to_int, parse_overall_glevel_value

DEFAULT_REPORT = "deepseek_interactive_report.json"
DEFAULT_TEST_OUT = "deepseek_interactive_test.json"

_NEUTRAL_CALIBRATION_RULES = """General rules (no preference for any single class):
- Labels are official integers g_level 1, 2, or 3 (1 = lowest cognitive level in this task, 3 = highest).
- Ground every rule in the labeled examples provided; do not invent criteria beyond what those examples support."""

SYSTEM_PROMPT_NEUTRAL = """You evaluate overall cognitive level from six interview answers per candidate (speech transcripts).
Assign exactly one official integer g_level: 1, 2, or 3. Follow only the rubric and transcript evidence.
When asked for a prediction, reply with JSON only:
{"overall_glevel": <integer 1, 2, or 3>, "confidence": float in [0,1]}"""

_TRAIN_BATCH_SYSTEM = """You are learning the organizer's labeling policy for interview cognitive levels.
You will receive:
1) Your CURRENT rubric draft (may be empty on the first batch).
2) A NEW batch of OFFICIAL labeled training examples (integer g_level 1, 2, or 3 per candidate).

""" + _NEUTRAL_CALIBRATION_RULES + """

Task: merge what you learn from the NEW batch into an UPDATED rubric draft.
The rubric must explain observable differences between g_level 1 vs 2 vs 3 on THIS dataset.

Reply with JSON ONLY:
{"updated_rubric": "<English string, <=1800 chars, no unescaped double quotes inside—use single quotes for emphasis>", "notes": "<optional short English>"}
"""


def _chat(
    api_key: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.1,
    json_object: bool = True,
    timeout: int = 180,
    retries: int = 4,
) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    if json_object:
        payload["response_format"] = {"type": "json_object"}
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
            return str(data["choices"][0]["message"]["content"] or "")
        except Exception as e:
            last_err = str(e)
            time.sleep(2**attempt)
    raise RuntimeError(last_err or "chat failed")


def _parse_json_loose(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```\s*$", "", text, flags=re.IGNORECASE)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
        raise


def _repair_rubric_json(api_key: str, broken: str) -> dict:
    sys_msg = (
        "The user's message was supposed to be JSON with keys updated_rubric (string) and notes (string), "
        "but it may be malformed.\n"
        "Return ONE valid JSON object only. Copy the rubric text into updated_rubric using standard JSON string escaping. "
        "If unsure, put the full readable rubric in updated_rubric and an empty string in notes."
    )
    raw = _chat(
        api_key,
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": broken[:14000]},
        ],
        temperature=0,
        json_object=True,
        timeout=120,
    )
    return _parse_json_loose(raw)


def _normalize_level(s: object) -> int | None:
    return parse_overall_glevel_value(s)


def _prediction_from_response(content: str) -> tuple[int | None, dict]:
    try:
        obj = _parse_json_loose(content)
    except Exception:
        return None, {}
    lvl = _normalize_level(obj.get("overall_glevel"))
    return lvl, obj


def _collect_train_blocks(
    df: pd.DataFrame,
    text_root: Path,
    *,
    max_chars: int,
    skip_if_missing_ge: int,
) -> tuple[list[tuple[str, int, str]], dict]:
    """收集带标签训练块。(sid, g_class 1..3, block)。"""
    triples: list[tuple[str, int, str]] = []
    skipped_bad_label = 0
    skipped_missing = 0
    for _, row in df.iterrows():
        sid = str(row["id"]).strip()
        cls = glevel_csv_to_int(row.get("g_level"))
        if not sid or cls is None:
            skipped_bad_label += 1
            continue
        ans, warns = load_answers(text_root, sid)
        if len(warns) >= skip_if_missing_ge:
            skipped_missing += 1
            continue
        triples.append((sid, cls, format_labeled_block(sid, ans, cls, "train", max_chars)))
    meta = {
        "candidates_labeled": len(triples),
        "skipped_bad_label": skipped_bad_label,
        "skipped_mostly_missing_text": skipped_missing,
    }
    return triples, meta


def _subsample_train_triples(
    triples: list[tuple[str, int, str]],
    *,
    max_examples: int,
    seed: int,
) -> list[tuple[str, int, str]]:
    """随机打乱后截取至多 max_examples 条（保留类别比例近似由原始数据分布决定）。"""
    rng = random.Random(seed)
    out = triples[:]
    rng.shuffle(out)
    return out[:max_examples]


def _subsample_train_triples_balanced_round_robin(
    triples: list[tuple[str, int, str]],
    *,
    max_examples: int,
    seed: int,
) -> list[tuple[str, int, str]]:
    """按 g_level 轮转抽样，使前几批 API 里 1/2/3 暴露更均匀，减轻 rubric 早期忽略高档样本。"""
    rng = random.Random(seed)
    by_c: dict[int, list[tuple[str, int, str]]] = defaultdict(list)
    for t in triples:
        by_c[t[1]].append(t)
    for cls in by_c:
        rng.shuffle(by_c[cls])
    order = [1, 2, 3]
    rng.shuffle(order)
    ptr = {1: 0, 2: 0, 3: 0}
    out: list[tuple[str, int, str]] = []
    while len(out) < max_examples:
        progressed = False
        for c in order:
            if len(out) >= max_examples:
                break
            lst = by_c.get(c, [])
            i = ptr[c]
            if i < len(lst):
                out.append(lst[i])
                ptr[c] = i + 1
                progressed = True
        if not progressed:
            break
    return out


def _batch_train_blocks(blocks: list[str], rows_per_batch: int) -> list[str]:
    batches: list[str] = []
    chunk: list[str] = []
    for b in blocks:
        chunk.append(b)
        if len(chunk) >= rows_per_batch:
            batches.append("\n".join(chunk))
            chunk = []
    if chunk:
        batches.append("\n".join(chunk))
    return batches


def _update_rubric_batch(
    api_key: str,
    *,
    rubric: str,
    batch_text: str,
    batch_idx: int,
    n_batches: int,
) -> str:
    user = (
        f"Batch {batch_idx + 1}/{n_batches}.\n\n"
        f"=== CURRENT RUBRIC DRAFT ===\n{rubric or '(empty)'}\n\n"
        f"=== NEW LABELED TRAINING BATCH ===\n{batch_text}\n"
    )
    messages = [
        {"role": "system", "content": _TRAIN_BATCH_SYSTEM},
        {"role": "user", "content": user},
    ]
    raw = ""
    for attempt in range(4):
        try:
            raw = _chat(
                api_key,
                messages,
                temperature=0.12 if attempt == 0 else 0.05,
                json_object=True,
                timeout=240,
            )
            obj = _parse_json_loose(raw)
            out = str(obj.get("updated_rubric") or obj.get("rubric") or "").strip()
            if out:
                return out
        except Exception:
            time.sleep(1.6 * (attempt + 1))
    try:
        obj = _repair_rubric_json(api_key, raw)
        out = str(obj.get("updated_rubric") or obj.get("rubric") or "").strip()
        if out:
            print(f"[interactive] rubric batch {batch_idx + 1}: recovered via repair JSON call", flush=True)
            return out
    except Exception:
        pass
    print(
        f"[interactive] WARN: rubric batch {batch_idx + 1}/{n_batches} JSON parse failed; keeping previous draft",
        flush=True,
    )
    return rubric


def _format_val_predict_prompt(sid: str, answers: list[str], max_chars: int) -> str:
    lines = [
        f"VALIDATION candidate id={sid}. Official labels are HIDDEN from you.",
        "Predict the overall cognitive level for THIS candidate only.",
        "Choose exactly one official integer g_level: 1, 2, or 3 according to your rubric.",
        "",
    ]
    for i, q in enumerate(QUESTIONS):
        ans = answers[i] if i < len(answers) else ""
        lines.append(f"Q{i + 1}: {q}")
        lines.append(f"A{i + 1}: {_clip(ans, max_chars)}")
    lines.append("")
    lines.append(
        'Reply JSON ONLY: {"overall_glevel": <1|2|3>, "confidence": 0.0-1.0} (overall_glevel must be an integer).'
    )
    return "\n".join(lines)


def _correction_message(gold_glevel: int, sid: str) -> str:
    return (
        f"Correction for validation id={sid}: your prediction did NOT match the official label.\n"
        f"Official overall g_level is **{gold_glevel}** (integer 1, 2, or 3).\n"
        "Briefly reconcile what signal you missed or over-weighted, and how you will adjust "
        "when judging similar answers next.\n"
        'Reply JSON ONLY: {"reflection":"<=600 chars English"}'
    )


def _recap_rubric(api_key: str, rubric_train: str, transcript_tail: str) -> str:
    """把验证阶段的要点压缩成最终 rubric_summary（单轮）。"""
    sys_msg = (
        "You distilled labeling policy from training and an interactive validation session.\n"
        "Merge the INITIAL training rubric with VALIDATION FEEDBACK transcript excerpts into ONE concise rubric.\n"
        "Summarize decision boundaries implied by the examples; avoid biasing toward any one level.\n"
        'Reply JSON ONLY: {"rubric_summary": "<=1400 chars English>"}'
    )
    user = (
        "=== INITIAL TRAINING RUBRIC (after full train batches) ===\n"
        f"{rubric_train}\n\n"
        "=== VALIDATION SESSION (user/assistant excerpts; predictions vs corrections) ===\n"
        f"{transcript_tail}\n"
    )
    raw = _chat(
        api_key,
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
        temperature=0.1,
        json_object=True,
        timeout=240,
    )
    obj = _parse_json_loose(raw)
    return str(obj.get("rubric_summary") or "").strip()



def _train_prior_hint(train_df: pd.DataFrame) -> str:
    """训练集 g_level 频次提示，减轻模型塌缩到单一类别的倾向。"""
    try:
        s = train_df["g_level"].dropna().astype(int)
    except Exception:
        return ""
    total = len(s)
    if total == 0:
        return ""
    lines = [
        "# Official TRAINING_SET label frequencies (calibration anchor—still judge each candidate individually):",
    ]
    for g in (1, 2, 3):
        n = int((s == g).sum())
        lines.append(f"- g_level {g}: {n} examples ({100.0 * n / total:.1f}%)")
    lines.append(
        "Use g_level 2 and 3 whenever rubric cues match; disfluency or spoken style alone is NOT sufficient for g_level 1."
    )
    return "\n".join(lines)


def _artifact_paths(report: Path, output: Path, run_idx: int, n_runs: int) -> tuple[Path, Path]:
    report = report.expanduser().resolve()
    output = output.expanduser().resolve()
    if n_runs <= 1:
        return report, output
    return (
        report.with_name(f"{report.stem}_run{run_idx}{report.suffix}"),
        output.with_name(f"{output.stem}_run{run_idx}{output.suffix}"),
    )


def _default_peer_sel_paths(report: Path, output: Path) -> tuple[Path, Path]:
    """与主交互 --report / -o 分离，避免择优元数据覆盖 train/val 报告。"""
    br = report.expanduser().resolve()
    bo = output.expanduser().resolve()
    return (
        br.with_name(f"{br.stem}_peer_sel{br.suffix}"),
        bo.with_name(f"{bo.stem}_peer_sel{bo.suffix}"),
    )


def _official_accuracy_from_submission_filename(path: Path) -> float | None:
    """从学长提交文件名解析官方平台准确率（小数），如 submission1_0.53077.csv → 0.53077。"""
    m = re.search(r"(\d+\.\d+)\s*$", path.stem)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _peer_label_cell_to_int(v: object) -> int | None:
    i = parse_overall_glevel_value(v)
    if i is not None:
        return i
    try:
        j = int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None
    if j in (0, 1, 2):
        return j + 1
    return None


def _load_peer_submission_labels(path: Path) -> dict[str, int]:
    df = pd.read_csv(path.expanduser().resolve())
    if "id" not in df.columns:
        raise SystemExit(f"{path} 缺少 id 列")
    lab_col = None
    for c in ("g_level_pred", "g_level", "pred", "prediction"):
        if c in df.columns:
            lab_col = c
            break
    if lab_col is None:
        raise SystemExit(f"{path} 需要 g_level_pred 或 g_level 列")
    out: dict[str, int] = {}
    for _, row in df.iterrows():
        sid = str(row["id"]).strip()
        if not sid:
            continue
        lv = _peer_label_cell_to_int(row.get(lab_col))
        if lv is None:
            raise SystemExit(f"{path} id={sid} 标签非法: {row.get(lab_col)!r}")
        out[sid] = lv
    return out


def _pairwise_agreement_fraction(maps: list[dict[str, int]]) -> list[tuple[int, int, float]]:
    """候选两两在同一 id 上标签一致比例。"""
    n = len(maps)
    rows: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = maps[i], maps[j]
            keys = set(a) & set(b)
            if not keys:
                rows.append((i, j, 0.0))
                continue
            agree = sum(1 for k in keys if a[k] == b[k])
            rows.append((i, j, agree / len(keys)))
    return rows


def _histogram_123(m: dict[str, int]) -> dict[int, int]:
    c = Counter(m.values())
    return {1: int(c.get(1, 0)), 2: int(c.get(2, 0)), 3: int(c.get(3, 0))}


def _format_test_transcript_block(sid: str, answers: list[str], *, max_chars: int) -> str:
    lines = [f"id={sid}"]
    for i, q in enumerate(QUESTIONS):
        ans = answers[i] if i < len(answers) else ""
        lines.append(f"Q{i + 1}: {q}")
        lines.append(f"A{i + 1}: {_clip(ans, max_chars)}")
    return "\n".join(lines)


def _run_peer_final_selection(
    api_key: str,
    *,
    letters: tuple[str, str, str, str],
    peer_paths: tuple[Path, Path, Path, Path],
    official_scores: tuple[float | None, float | None, float | None, float | None],
    label_maps: tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]],
    test_ids: list[str],
    rubric_summary: str,
    transcript_blocks: str | None,
) -> dict[str, object]:
    """单次 API：全局选定 A/B/C/D 之一作为最终提交。"""
    hist_lines = []
    for idx in range(4):
        li = letters[idx]
        p = peer_paths[idx]
        sc = official_scores[idx]
        h = _histogram_123(label_maps[idx])
        sc_s = f"{sc:.5f}" if sc is not None else "unknown"
        hist_lines.append(
            f"- Candidate {li}: file={p.name} official_platform_test_accuracy={sc_s} "
            f"class_counts_123={h}"
        )
    agree = _pairwise_agreement_fraction(list(label_maps))
    agree_s = ", ".join(
        f"{letters[i]}-{letters[j]}:{v:.3f}" for i, j, v in agree
    )
    rows = []
    for sid in test_ids:
        labs = [label_maps[k][sid] for k in range(4)]
        rows.append(f"{sid}\t{labs[0]}\t{labs[1]}\t{labs[2]}\t{labs[3]}")
    table = "\n".join(rows)
    sys_sel = (
        "You finished calibration on the official training set and an interactive validation pass.\n"
        "You now decide which SINGLE candidate submission to adopt for the ENTIRE test set.\n"
        "Each candidate is one model's g_level predictions (1/2/3) per interview id.\n"
        "Each candidate's filename embeds that submission's official ONLINE test accuracy "
        "(decimal fraction). This score is a noisy proxy of generalization—use it together with "
        "prediction diversity and your rubric.\n\n"
        "Return JSON ONLY:\n"
        '{"chosen_letter":"A|B|C|D","reason":"<=900 chars English","confidence":0.0-1.0}\n'
        "Rules: chosen_letter must be exactly one of A,B,C,D matching the candidate labels below."
    )
    user_parts = [
        "### Candidate summary\n" + "\n".join(hist_lines),
        f"### Pairwise label agreement on overlapping ids\n{agree_s}",
        "### Rubric recap (training + validation)\n"
        + _clip(rubric_summary or "(empty)", 3200),
        "### Per-test-id labels (tab-separated: id, label_A, label_B, label_C, label_D)\n" + table,
    ]
    if transcript_blocks:
        user_parts.append(
            "### Optional transcript excerpts (truncated)\n"
            + _clip(transcript_blocks, 28000)
        )
    user_body = "\n\n".join(user_parts)
    raw = _chat(
        api_key,
        [{"role": "system", "content": sys_sel}, {"role": "user", "content": user_body}],
        temperature=0.08,
        json_object=True,
        timeout=300,
        retries=5,
    )
    obj = _parse_json_loose(raw)
    letter = str(obj.get("chosen_letter") or obj.get("chosen") or "").strip().upper()
    if letter not in letters:
        raise ValueError(f"invalid chosen_letter: {letter!r}")
    idx = letters.index(letter)
    return {
        "chosen_letter": letter,
        "chosen_index": idx,
        "chosen_file": str(peer_paths[idx]),
        "official_score_in_filename": official_scores[idx],
        "reason": str(obj.get("reason") or ""),
        "confidence": obj.get("confidence"),
        "raw_response": raw[:8000],
    }


def _majority_vote_test(run_dicts: list[dict[str, object]]) -> dict[str, object]:
    ids: set[str] = set()
    for d in run_dicts:
        ids |= {str(k) for k in d}
    merged: dict[str, object] = {}
    for sid in sorted(ids):
        votes: Counter[int] = Counter()
        per_run: list[dict[str, object]] = []
        conf_sum: dict[int, float] = {}
        for ridx, d in enumerate(run_dicts):
            rec = d.get(sid)
            if not isinstance(rec, dict):
                continue
            lvl = _normalize_level(rec.get("overall_glevel"))
            if lvl is None:
                continue
            votes[lvl] += 1
            try:
                c = float(rec.get("confidence", 0.5))
            except (TypeError, ValueError):
                c = 0.5
            conf_sum[lvl] = conf_sum.get(lvl, 0.0) + c
            per_run.append({"run": ridx, "overall_glevel": lvl, "confidence": c})
        if not votes:
            merged[sid] = {
                "overall_glevel": 2,
                "confidence": 0.0,
                "vote_counts": {},
                "per_run": per_run,
            }
            continue
        max_v = max(votes.values())
        cand = [lab for lab, ct in votes.items() if ct == max_v]
        if len(cand) == 1:
            winner = cand[0]
        else:
            winner = max(cand, key=lambda lab: conf_sum.get(lab, 0.0))
        merged[sid] = {
            "overall_glevel": winner,
            "confidence": round(max_v / sum(votes.values()), 4),
            "vote_counts": dict(votes),
            "vote_avg_conf_for_winner": round(conf_sum.get(winner, 0.0) / max_v, 4),
            "per_run": per_run,
            "meta": {
                "ensemble_runs": len(run_dicts),
                "method": "majority_vote_tiebreak_sum_confidence",
            },
        }
    return merged

def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek：train 分批 → val 逐条纠错 → 测试标注")
    ap.add_argument("--train-csv", type=Path, default=Path(DEFAULT_TRAIN_CSV))
    ap.add_argument("--val-csv", type=Path, default=Path(DEFAULT_VAL_CSV))
    ap.add_argument("--train-text-dir", type=Path, default=Path(DEFAULT_TRAIN_TEXT))
    ap.add_argument("--val-text-dir", type=Path, default=Path(DEFAULT_VAL_TEXT))
    ap.add_argument("--test-csv", type=Path, default=Path(DEFAULT_TEST_CSV))
    ap.add_argument("--test-text-dir", type=Path, default=Path(DEFAULT_TEXT_DIR))
    ap.add_argument("--report", type=Path, default=Path(DEFAULT_REPORT))
    ap.add_argument("--output", "-o", type=Path, default=Path(DEFAULT_TEST_OUT))
    ap.add_argument("--train-rows-per-batch", type=int, default=35, help="每轮 API 上传的训练样本条数")
    ap.add_argument("--train-max-chars", type=int, default=1400, help="训练示例中每题答案最大字符")
    ap.add_argument("--train-min-examples", type=int, default=450, help="至少使用的带标签训练样本数")
    ap.add_argument("--train-max-examples", type=int, default=450, help="至多使用的带标签训练样本数")
    ap.add_argument(
        "--train-balanced-mix",
        action="store_true",
        help="训练抽样改为按 1/2/3 轮转凑满 max_examples（总量不变时仍提高各档在早期 batch 的覆盖率）",
    )
    ap.add_argument("--train-sample-seed", type=int, default=42)
    ap.add_argument(
        "--train-skip-if-missing-ge",
        type=int,
        default=6,
        help="缺失转写文件数≥该值则跳过样本；6 表示仅六题全缺才跳过（尽可能多用数据）",
    )
    ap.add_argument("--val-max-chars", type=int, default=900)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test-only", action="store_true", help="跳过 train/val，直接读 report 里的 rubric_summary 标测试集")
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument(
        "--pipeline-runs",
        type=int,
        default=1,
        help="完整重复 train→val→test 的次数；>1 时每轮单独 report/output，并对测试集多数投票写入 -o",
    )
    ap.add_argument("--pipeline-seed-stride", type=int, default=1009, help="每轮训练抽样 seed 增量")
    ap.add_argument(
        "--test-temperature",
        type=float,
        default=0.12,
        help="测试集 annotate_candidate 温度（略高于 0 可减少单一类别塌缩）",
    )
    ap.add_argument(
        "--test-temperature-stride",
        type=float,
        default=0.03,
        help="每一 pipeline 轮在 test-temperature 上的增量",
    )
    ap.add_argument(
        "--peer-final-csvs",
        nargs=4,
        metavar=("CSV_A", "CSV_B", "CSV_C", "CSV_D"),
        type=Path,
        default=None,
        help="学长/队友四份测试集 submission CSV（文件名中含官方准确率小数）；"
        "指定后跳过 DeepSeek 逐条测批注，仅在 train→val 后做一次全局择优并写出 --peer-final-out-csv",
    )
    ap.add_argument(
        "--peer-final-out-csv",
        type=Path,
        default=None,
        help="择优后的提交 CSV（id,g_level_pred）；与 --peer-final-csvs 联用必填",
    )
    ap.add_argument(
        "--peer-final-transcript-chars",
        type=int,
        default=0,
        help=">0 时在择优 prompt 中附带测试集转写节选（每题最多该字符数）；0 表示仅标签表+统计",
    )
    ap.add_argument(
        "--peer-final-transcript-budget",
        type=int,
        default=28000,
        help="附带的转写节选总字符上限（防止超长）",
    )
    ap.add_argument(
        "--peer-final-report",
        type=Path,
        default=None,
        help="择优专用报告 JSON（默认：<--report> 同名加后缀 _peer_sel）；勿与主交互 report 共用",
    )
    ap.add_argument(
        "--peer-final-json",
        type=Path,
        default=None,
        help="择优 mirror JSON（默认：<--output> 同名加后缀 _peer_sel）；勿与 DeepSeek 逐条测试 -o 共用",
    )
    ap.add_argument(
        "--peer-calibrate-from-report",
        type=Path,
        default=None,
        help="只读已完成交互报告中的 rubric_summary，跳过 train/val，直接择优；可与正在跑的 annotate 并行；"
        "默认 peer 产物写在「该报告」同目录（*_peer_sel / *_peer_mirror）",
    )
    ap.add_argument(
        "--peer-write-main-report-pointer",
        action="store_true",
        help="择优后将指针字段合并写入 --report（默认不写；若文件已存在则读入后合并键，避免覆盖丢 train/val）",
    )
    args = ap.parse_args()
    api_key = _api_key()

    if args.dry_run:
        if args.peer_calibrate_from_report and args.peer_final_csvs:
            cr = args.peer_calibrate_from_report.expanduser().resolve()
            pr = cr.with_name(f"{cr.stem}_peer_sel{cr.suffix}")
            pj = cr.with_name(f"{cr.stem}_peer_mirror.json")
            print("[interactive] dry-run：peer-calibrate 模式（不检查 train/val 抽样）")
            print(f"[interactive] dry-run rubric 来源 → {cr}")
            print("[interactive] dry-run peer-final-csvs:")
            for i, p in enumerate(args.peer_final_csvs):
                acc = _official_accuracy_from_submission_filename(p)
                print(f"  {('ABCD'[i])}: {p} score={acc}")
            print(f"[interactive] dry-run 默认 peer report → {pr}")
            print(f"[interactive] dry-run 默认 peer json  → {pj}")
            return
        train_df = pd.read_csv(args.train_csv)
        train_txt = args.train_text_dir.expanduser().resolve()
        triples, meta = _collect_train_blocks(
            train_df,
            train_txt,
            max_chars=args.train_max_chars,
            skip_if_missing_ge=args.train_skip_if_missing_ge,
        )
        pick_fn = (
            _subsample_train_triples_balanced_round_robin
            if args.train_balanced_mix
            else _subsample_train_triples
        )
        picked = pick_fn(
            triples, max_examples=args.train_max_examples, seed=args.train_sample_seed
        )
        if len(picked) < args.train_min_examples:
            raise SystemExit(
                f"可用训练样本 {len(picked)} 条 < --train-min-examples={args.train_min_examples}；"
                f"收集统计: {meta}"
            )
        blocks = [t[2] for t in picked]
        batches = _batch_train_blocks(blocks, args.train_rows_per_batch)
        print(
            f"[interactive] dry-run train pool={meta['candidates_labeled']} "
            f"scheduled={len(picked)} batches={len(batches)} rows_per_batch={args.train_rows_per_batch}"
        )
        val_df = pd.read_csv(args.val_csv)
        print(f"[interactive] dry-run val rows={len(val_df)}")
        print(f"[interactive] dry-run pipeline-runs={args.pipeline_runs}")
        if args.peer_final_csvs:
            pr, pj = _default_peer_sel_paths(args.report, args.output)
            if args.peer_calibrate_from_report:
                cr = args.peer_calibrate_from_report.expanduser().resolve()
                pr = cr.with_name(f"{cr.stem}_peer_sel{cr.suffix}")
                pj = cr.with_name(f"{cr.stem}_peer_mirror.json")
            print("[interactive] dry-run peer-final-csvs:")
            for i, p in enumerate(args.peer_final_csvs):
                acc = _official_accuracy_from_submission_filename(p)
                print(f"  {('ABCD'[i])}: {p} score={acc}")
            print(f"[interactive] dry-run 默认 peer report → {pr}")
            print(f"[interactive] dry-run 默认 peer json  → {pj}")

        return

    if not api_key:
        raise SystemExit("缺少 DEEPSEEK_API_KEY")

    peer_paths_tuple: tuple[Path, Path, Path, Path] | None = None
    if args.peer_final_csvs:
        peer_paths_tuple = tuple(args.peer_final_csvs)
        if args.peer_final_out_csv is None:
            raise SystemExit("--peer-final-out-csv 与 --peer-final-csvs 同时必填")
        for p in peer_paths_tuple:
            if not p.expanduser().resolve().is_file():
                raise SystemExit(f"找不到 peer CSV: {p}")

    if args.peer_calibrate_from_report and not args.peer_final_csvs:
        raise SystemExit("--peer-calibrate-from-report 必须与 --peer-final-csvs 同时使用")

    pipeline_runs = max(1, int(args.pipeline_runs))
    if args.test_only and pipeline_runs > 1:
        raise SystemExit("--test-only 不能与 --pipeline-runs>1 同时使用")
    if peer_paths_tuple and pipeline_runs > 1:
        raise SystemExit("--peer-final-csvs 仅支持 --pipeline-runs 1（择优只做一次）")

    peer_report_path: Path | None = None
    peer_json_path: Path | None = None
    if peer_paths_tuple:
        if args.peer_calibrate_from_report:
            cr = args.peer_calibrate_from_report.expanduser().resolve()
            if not cr.is_file():
                raise SystemExit(f"--peer-calibrate-from-report 不存在: {cr}")
            d_pr = cr.with_name(f"{cr.stem}_peer_sel{cr.suffix}")
            d_pj = cr.with_name(f"{cr.stem}_peer_mirror.json")
        else:
            d_pr, d_pj = _default_peer_sel_paths(args.report, args.output)
        peer_report_path = (args.peer_final_report or d_pr).expanduser().resolve()
        peer_json_path = (args.peer_final_json or d_pj).expanduser().resolve()

    PEER_LETTERS = ("A", "B", "C", "D")

    peer_calibrate_src_resolved: Path | None = None
    if args.peer_calibrate_from_report:
        peer_calibrate_src_resolved = args.peer_calibrate_from_report.expanduser().resolve()

    run_snapshots: list[dict[str, object]] = []

    for run_idx in range(pipeline_runs):
        train_sample_seed = args.train_sample_seed + run_idx * args.pipeline_seed_stride
        report_path, out_path = _artifact_paths(args.report, args.output, run_idx, pipeline_runs)
        test_temp = min(0.95, max(0.0, args.test_temperature + run_idx * args.test_temperature_stride))
        print(
            f"[interactive] ===== pipeline run {run_idx + 1}/{pipeline_runs} "
            f"report={report_path.name} out={out_path.name} "
            f"train_seed={train_sample_seed} test_T={test_temp:.3f} =====",
            flush=True,
        )

        state: dict = {}
        if peer_calibrate_src_resolved is not None:
            src_doc = json.loads(peer_calibrate_src_resolved.read_text(encoding="utf-8"))
            rubric_final = str(src_doc.get("rubric_summary") or "")
            rubric_train = str(src_doc.get("rubric_after_train") or "")
            if not rubric_final.strip():
                raise SystemExit(
                    f"--peer-calibrate-from-report 缺少 rubric_summary: {peer_calibrate_src_resolved}"
                )
            print(
                f"[interactive] rubric 只读来源={peer_calibrate_src_resolved.name} "
                f"(chars={len(rubric_final)}) — 跳过 train/val",
                flush=True,
            )
        elif args.resume and report_path.is_file():
            state = json.loads(report_path.read_text(encoding="utf-8"))

        if peer_calibrate_src_resolved is None:
            rubric_train = str(state.get("rubric_after_train") or "")
            rubric_final = str(state.get("rubric_summary") or "")
        val_log = list(state.get("val_log") or [])
        transcript_parts = list(state.get("transcript_parts") or [])
        val_done_ids = {str(x.get("id")) for x in val_log if x.get("id")}

        def _save_state() -> None:
            report_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

        # ----- Phase train -----
        if not args.test_only and peer_calibrate_src_resolved is None:
            train_df = pd.read_csv(args.train_csv.expanduser().resolve())
            train_txt = args.train_text_dir.expanduser().resolve()
            if not train_txt.is_dir():
                raise SystemExit(f"找不到 train_text: {train_txt}")
            prior_block = _train_prior_hint(train_df)
            state["train_prior_hint_block"] = prior_block

            train_done = bool(state.get("train_complete"))
            if not train_done:
                triples, meta = _collect_train_blocks(
                    train_df,
                    train_txt,
                    max_chars=args.train_max_chars,
                    skip_if_missing_ge=args.train_skip_if_missing_ge,
                )
                pick_fn = (
                    _subsample_train_triples_balanced_round_robin
                    if args.train_balanced_mix
                    else _subsample_train_triples
                )
                picked = pick_fn(
                    triples, max_examples=args.train_max_examples, seed=train_sample_seed
                )
                if len(picked) < args.train_min_examples:
                    raise SystemExit(
                        f"可用训练样本 {len(picked)} 条 < --train-min-examples={args.train_min_examples}；"
                        f"收集统计: {meta}"
                    )
                blocks = [t[2] for t in picked]
                batches = _batch_train_blocks(blocks, args.train_rows_per_batch)
                state["train_selection"] = {
                    "min_examples": args.train_min_examples,
                    "max_examples": args.train_max_examples,
                    "sample_seed": train_sample_seed,
                    "scheduled_examples": len(picked),
                    "rows_per_batch": args.train_rows_per_batch,
                    "max_chars_per_answer": args.train_max_chars,
                    "skip_if_missing_ge": args.train_skip_if_missing_ge,
                    "collection_meta": meta,
                    "balanced_mix": bool(args.train_balanced_mix),
                }
                print(
                    f"[interactive] train scheduled_examples={len(picked)} "
                    f"(pool={meta['candidates_labeled']}) batches={len(batches)} "
                    f"rows_per_batch={args.train_rows_per_batch} max_chars={args.train_max_chars}"
                )
    
                rubric = str(state.get("rubric_after_train") or "")
                start_i = int(state.get("train_batches_done") or 0) if args.resume else 0
                if not args.resume:
                    start_i = 0
                    rubric = ""
                if start_i > len(batches):
                    start_i = 0
                print(f"[interactive] train batches={len(batches)} start_i={start_i}")
                for i in tqdm(
                    range(start_i, len(batches)),
                    desc="Train→rubric",
                    total=max(0, len(batches) - start_i),
                ):
                    rubric = _update_rubric_batch(
                        api_key,
                        rubric=rubric,
                        batch_text=batches[i],
                        batch_idx=i,
                        n_batches=len(batches),
                    )
                    time.sleep(max(0.0, args.sleep))
                    state.update(
                        {
                            "phase": "train",
                            "rubric_after_train": rubric,
                            "train_batches_done": i + 1,
                            "train_batches_total": len(batches),
                        }
                    )
                    _save_state()
                state["train_complete"] = True
                rubric_train = str(state.get("rubric_after_train") or "")
                _save_state()
            else:
                rubric_train = str(state.get("rubric_after_train") or "")
    
            # ----- Phase val (multi-turn) -----
            val_df = pd.read_csv(args.val_csv.expanduser().resolve())
            val_txt = args.val_text_dir.expanduser().resolve()
            if not val_txt.is_dir():
                raise SystemExit(f"找不到 val_text: {val_txt}")
    
            prior_block = str(state.get("train_prior_hint_block") or "")
            system_val = (
                SYSTEM_PROMPT_NEUTRAL
                + ("\n\n" + prior_block + "\n\n" if prior_block else "\n")
                + "# Rubric learned from the official training set (batched updates)\n"
                + rubric_train
                + "\n\n"
                + _NEUTRAL_CALIBRATION_RULES
                + "\n\nYou will now be evaluated on validation candidates one-by-one. "
                "Predict JSON only when asked; if corrected, reflect briefly as instructed."
            )
    
            if not state.get("val_complete"):
                raw_vm = state.get("val_messages")
                if isinstance(raw_vm, list) and raw_vm:
                    messages = [dict(x) for x in raw_vm if isinstance(x, dict)]
                else:
                    messages = [{"role": "system", "content": system_val}]
                val_log = list(state.get("val_log") or [])
                transcript_parts = list(state.get("transcript_parts") or [])
                val_done_ids = {str(x.get("id")) for x in val_log if x.get("id")}
    
                for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Val interactive"):
                    sid = str(row["id"]).strip()
                    gold_int = glevel_csv_to_int(row.get("g_level"))
                    if not sid or gold_int is None:
                        continue
                    if args.resume and sid in val_done_ids:
                        continue

                    answers, _ = load_answers(val_txt, sid)
                    user_p = _format_val_predict_prompt(sid, answers, args.val_max_chars)
                    messages.append({"role": "user", "content": user_p})
                    try:
                        raw_pred = _chat(api_key, messages, temperature=0.05, json_object=False, timeout=120)
                    except Exception as e:
                        raw_pred = json.dumps({"overall_glevel": "ERROR", "error": str(e)})
                    messages.append({"role": "assistant", "content": raw_pred})

                    pred_int, _obj = _prediction_from_response(raw_pred)
                    match = pred_int == gold_int if pred_int is not None else False
                    transcript_parts.append(f"VAL_PRED id={sid} model_raw={raw_pred[:800]}")
                    if not match:
                        cm = _correction_message(gold_int, sid)
                        messages.append({"role": "user", "content": cm})
                        try:
                            raw_fix = _chat(api_key, messages, temperature=0.1, json_object=False, timeout=120)
                        except Exception as e:
                            raw_fix = json.dumps({"reflection": str(e)})
                        messages.append({"role": "assistant", "content": raw_fix})
                        transcript_parts.append(
                            f"VAL_CORRECT id={sid} gold={gold_int} reflection={raw_fix[:800]}"
                        )

                    entry = {
                        "id": sid,
                        "gold": gold_int,
                        "pred": pred_int,
                        "match": match,
                    }
                    val_log.append(entry)
                    val_done_ids.add(sid)
    
                    acc = sum(1 for x in val_log if x.get("match")) / max(1, len(val_log))
                    state.update(
                        {
                            "phase": "val",
                            "val_log": val_log,
                            "val_messages": messages,
                            "val_accuracy_so_far": acc,
                            "transcript_parts": transcript_parts[-400:],
                        }
                    )
                    _save_state()
                    time.sleep(max(0.0, args.sleep))
    
                state["val_complete"] = True
                _save_state()
    
            val_log = list(state.get("val_log") or [])
            transcript_parts = list(state.get("transcript_parts") or [])
            val_accuracy = sum(1 for x in val_log if x.get("match")) / max(1, len(val_log))
    
            if not state.get("rubric_summary"):
                tail = "\n".join(transcript_parts[-120:])
                print(f"[interactive] recap rubric from validation tail chars={len(tail)}")
                rubric_final = _recap_rubric(api_key, rubric_train, tail)
                state.update(
                    {
                        "phase": "done_pre_test",
                        "val_accuracy": val_accuracy,
                        "rubric_summary": rubric_final,
                        "n_val": len(val_log),
                    }
                )
                _save_state()
            else:
                rubric_final = str(state.get("rubric_summary") or "")
            print(f"[interactive] val_accuracy={val_accuracy:.4f} report→ {report_path}")
        else:
            if peer_calibrate_src_resolved is None and not rubric_final:
                raise SystemExit("--test-only 需要 report 中存在 rubric_summary")
            if peer_calibrate_src_resolved is None:
                rubric_train = str(state.get("rubric_after_train") or "")
                rubric_final = str(state.get("rubric_summary") or "")
    
        # ----- Phase test（DeepSeek 逐条标注）或学长提交择优 -----
        test_csv = args.test_csv.expanduser().resolve()
        test_text = args.test_text_dir.expanduser().resolve()
        df = pd.read_csv(test_csv)
        test_ids = [str(x).strip() for x in df["id"].tolist() if str(x).strip()]
        if args.limit_test > 0:
            test_ids = test_ids[: args.limit_test]

        if peer_paths_tuple:
            maps = tuple(_load_peer_submission_labels(p) for p in peer_paths_tuple)
            official_scores = tuple(_official_accuracy_from_submission_filename(p) for p in peer_paths_tuple)
            ref_set = set(test_ids)
            for p, m in zip(peer_paths_tuple, maps):
                if set(m.keys()) != ref_set:
                    miss = ref_set - set(m.keys())
                    extra = set(m.keys()) - ref_set
                    raise SystemExit(
                        f"peer CSV 与 --test-csv id 集合不一致: {p.name} "
                        f"missing={len(miss)} extra={len(extra)}"
                    )

            transcript_blocks: str | None = None
            if args.peer_final_transcript_chars > 0:
                if not test_text.is_dir():
                    raise SystemExit(f"--peer-final-transcript-chars>0 需要有效 --test-text-dir: {test_text}")
                parts_blk: list[str] = []
                used = 0
                budget = max(4000, int(args.peer_final_transcript_budget))
                for sid in test_ids:
                    ans, _ = load_answers(test_text, sid)
                    blk = _format_test_transcript_block(
                        sid, ans, max_chars=args.peer_final_transcript_chars
                    )
                    if used + len(blk) + 2 > budget:
                        parts_blk.append(f"[... transcripts truncated after ~{budget} chars ...]")
                        break
                    parts_blk.append(blk)
                    used += len(blk) + 2
                transcript_blocks = "\n\n".join(parts_blk)

            peer_meta_base: dict[str, object] = {
                "letters": list(PEER_LETTERS),
                "paths": [str(p.resolve()) for p in peer_paths_tuple],
                "official_scores_from_filename": [float(s) if s is not None else None for s in official_scores],
            }

            assert peer_report_path is not None and peer_json_path is not None
            peer_state: dict = {}
            if args.resume and peer_report_path.is_file():
                peer_state = json.loads(peer_report_path.read_text(encoding="utf-8"))

            def _save_peer_state() -> None:
                peer_report_path.parent.mkdir(parents=True, exist_ok=True)
                peer_report_path.write_text(
                    json.dumps(peer_state, indent=2, ensure_ascii=False), encoding="utf-8"
                )

            if not peer_state.get("peer_final_complete"):
                print(
                    f"[interactive] PEER_FINAL：DeepSeek 在四份学长提交中全局择优 … "
                    f"peer_report={peer_report_path.name}",
                    flush=True,
                )
                try:
                    decision_core = _run_peer_final_selection(
                        api_key,
                        letters=PEER_LETTERS,
                        peer_paths=peer_paths_tuple,
                        official_scores=official_scores,
                        label_maps=maps,
                        test_ids=test_ids,
                        rubric_summary=rubric_final,
                        transcript_blocks=transcript_blocks,
                    )
                    decision_core["fallback"] = False
                except Exception as e:
                    scores_num = [s if s is not None else -1.0 for s in official_scores]
                    best_i = int(max(range(4), key=lambda i: scores_num[i]))
                    decision_core = {
                        "chosen_letter": PEER_LETTERS[best_i],
                        "chosen_index": best_i,
                        "chosen_file": str(peer_paths_tuple[best_i].resolve()),
                        "official_score_in_filename": official_scores[best_i],
                        "reason": f"peer selection API failed ({e}); fallback=highest filename score.",
                        "confidence": None,
                        "fallback": True,
                        "error": str(e),
                    }
                peer_state["peer_final_selection"] = {
                    **decision_core,
                    "candidates_meta": peer_meta_base,
                    "source_interactive_report": str(
                        (peer_calibrate_src_resolved or report_path).resolve()
                    ),
                }
                peer_state["peer_final_complete"] = True
                peer_state["phase"] = "done_peer_final"
                _save_peer_state()
            else:
                raw_dec = dict(peer_state["peer_final_selection"])
                print(
                    f"[interactive] PEER_FINAL resume chosen={raw_dec.get('chosen_letter')} "
                    f"file={raw_dec.get('chosen_file')}",
                    flush=True,
                )

            decision_use = dict(peer_state["peer_final_selection"])
            chosen_idx = int(decision_use["chosen_index"])
            chosen_map = maps[chosen_idx]
            peer_out_csv = args.peer_final_out_csv.expanduser().resolve()
            peer_out_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [{"id": sid, "g_level_pred": chosen_map[sid]} for sid in test_ids]
            ).to_csv(peer_out_csv, index=False)

            results = {}
            for sid in test_ids:
                try:
                    conf_f = float(decision_use.get("confidence"))
                except (TypeError, ValueError):
                    conf_f = 0.0
                results[sid] = {
                    "overall_glevel": chosen_map[sid],
                    "confidence": conf_f,
                    "meta": {
                        "pipeline": "peer_final_selection",
                        "interactive_report": str(
                            (peer_calibrate_src_resolved or report_path).resolve()
                        ),
                        "peer_sel_report": str(peer_report_path.resolve()),
                        "chosen_letter": decision_use.get("chosen_letter"),
                        "chosen_file": decision_use.get("chosen_file"),
                        "official_score_in_filename": decision_use.get("official_score_in_filename"),
                        "reason": decision_use.get("reason"),
                        "fallback": decision_use.get("fallback"),
                    },
                }
            peer_json_path.parent.mkdir(parents=True, exist_ok=True)
            peer_json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            peer_state["phase"] = "complete"
            peer_state["peer_final_out_csv"] = str(peer_out_csv.resolve())
            peer_state["peer_final_json"] = str(peer_json_path.resolve())
            _save_peer_state()
            print(f"[interactive] Peer-final 提交 CSV → {peer_out_csv}", flush=True)
            print(f"[interactive] Peer-final JSON（独立）→ {peer_json_path}", flush=True)
            if args.peer_write_main_report_pointer:
                mp = report_path.expanduser().resolve()
                if mp.is_file():
                    merged = json.loads(mp.read_text(encoding="utf-8"))
                else:
                    merged = dict(state) if state else {}
                merged["phase"] = "complete_peer_sel_only"
                merged["peer_sel_report"] = str(peer_report_path.resolve())
                merged["peer_sel_json"] = str(peer_json_path.resolve())
                merged["peer_sel_csv"] = str(peer_out_csv.resolve())
                mp.parent.mkdir(parents=True, exist_ok=True)
                mp.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[interactive] 已合并写入主 report 指针字段 → {mp}", flush=True)
            run_snapshots.append(results)
        else:
            prior_test = str(state.get("train_prior_hint_block") or "")
            if not prior_test:
                tcsv = args.train_csv.expanduser().resolve()
                if tcsv.is_file():
                    prior_test = _train_prior_hint(pd.read_csv(tcsv))

            system_test = (
                SYSTEM_PROMPT_NEUTRAL
                + ("\n\n" + prior_test + "\n\n" if prior_test else "\n")
                + "# Dataset-specific rubric (training + validation recap)\n"
                + (rubric_final or "(empty)")
                + "\n\n"
                + _NEUTRAL_CALIBRATION_RULES
            )
            results = {}
            if args.resume and out_path.is_file():
                results = json.loads(out_path.read_text(encoding="utf-8"))

            def _done(rec: object) -> bool:
                if not isinstance(rec, dict):
                    return False
                return _normalize_level(rec.get("overall_glevel")) is not None

            print(f"[interactive] TEST n={len(test_ids)} MODEL={MODEL_NAME}")
            for sid in tqdm(test_ids, desc="Test annotate"):
                if args.resume and _done(results.get(sid)):
                    continue
                answers, warns = load_answers(test_text, sid)
                ann = annotate_candidate(
                    api_key, answers, system_prompt=system_test, temperature=test_temp
                )
                ann["meta"] = {
                    "model": MODEL_NAME,
                    "pipeline": "interactive_train_val_then_test",
                    "report": str(report_path),
                    "missing_transcript_files": len(warns),
                    "pipeline_run_index": run_idx,
                    "test_temperature": test_temp,
                }
                results[sid] = ann
                time.sleep(max(0.0, args.sleep))
                out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

            state["phase"] = "complete"
            state["pipeline_run_index"] = run_idx
            state["pipeline_runs_total"] = pipeline_runs
            state["test_output"] = str(out_path)
            report_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[interactive] Done test → {out_path}")
            run_snapshots.append(results)

    if pipeline_runs > 1:
        merged = _majority_vote_test(run_snapshots)
        final_out = args.output.expanduser().resolve()
        final_out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[interactive] majority vote → {final_out} (runs={pipeline_runs})", flush=True)


if __name__ == "__main__":
    main()
