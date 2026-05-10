#!/usr/bin/env python3
"""
将仓库内遗留产物统一为赛方 g_level 整数 1/2/3：

1) CSV：若存在列 g_level_pred 或 g_level，且该列非空取值仅为 {0,1,2}（字符串或整数），则逐格 +1。
   - 不修改已为 1/2/3 的文件。
   - 跳过含「人格自评」等非提交列的宽表（列名含 H_self 等）以防误伤。
2) JSON：对含 overall_glevel / vote_counts / val_log 等结构的 DeepSeek 产物递归改写：
   - overall_glevel、gold、pred：LOW/MEDIUM/HIGH → 1/2/3；保留 \"ERROR\" 等非法类名不动。
   - vote_counts 的键 LOW/MEDIUM/HIGH → \"1\"/\"2\"/\"3\"（JSON 键为字符串）。

默认扫描仓库根目录（排除 .venv、.git、site-packages）。

用法（仓库根）:
  python tools/migrate_legacy_glevel_artifacts_to_123.py
  python tools/migrate_legacy_glevel_artifacts_to_123.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    ".venv_glevel_cpu",
    "__pycache__",
    "node_modules",
}
SKIP_PATH_PARTS = {"site-packages"}

_WORD_TO_INT = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


def _csv_label_column(fieldnames: list[str]) -> str | None:
    fn = set(fieldnames)
    if "g_level_pred" in fn:
        return "g_level_pred"
    if "g_level" in fn:
        return "g_level"
    return None


def _csv_is_personality_wide_table(fieldnames: list[str]) -> bool:
    """train_data 一类宽表含 H_self 等，勿把 g_level +1。"""
    fn = set(fieldnames)
    return bool(fn & {"H_self", "E_self", "A_self", "C_self"})


def migrate_csv(path: Path, *, dry_run: bool) -> bool:
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError:
        return False
    lines = raw.splitlines()
    if not lines:
        return False
    try:
        sample = lines[: min(5, len(lines))]
        dialect = csv.Sniffer().sniff("\n".join(sample), delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, dialect=dialect)
        fn = reader.fieldnames or []
        lab = _csv_label_column(list(fn))
        if not lab or _csv_is_personality_wide_table(list(fn)):
            return False
        rows = list(reader)
    if not rows:
        return False
    vals: set[str] = set()
    for row in rows:
        v = (row.get(lab) or "").strip()
        if v:
            vals.add(v)
    if not vals:
        return False
    ints: set[int] = set()
    for v in vals:
        try:
            ints.add(int(float(v)))
        except ValueError:
            return False
    if not ints <= {0, 1, 2}:
        return False

    changed = False
    for row in rows:
        v = (row.get(lab) or "").strip()
        if not v:
            continue
        i = int(float(v))
        new_s = str(i + 1)
        if row.get(lab) != new_s:
            row[lab] = new_s
            changed = True
    if not changed or dry_run:
        return changed

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn, dialect=dialect, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return True


def _migrate_vote_counts(vc: dict) -> bool:
    changed = False
    out: dict[object, object] = {}
    for kk, vv in vc.items():
        ku = str(kk).strip().upper()
        if ku in _WORD_TO_INT:
            out[str(_WORD_TO_INT[ku])] = vv
            changed = True
        else:
            out[kk] = vv
    if changed:
        vc.clear()
        vc.update(out)
    return changed


def migrate_json_obj(obj: object) -> bool:
    changed = False
    if isinstance(obj, dict):
        v = obj.get("overall_glevel")
        if isinstance(v, str):
            u = v.strip().upper()
            if u in _WORD_TO_INT:
                obj["overall_glevel"] = _WORD_TO_INT[u]
                changed = True
        for key in ("gold", "pred"):
            if key not in obj:
                continue
            gv = obj[key]
            if isinstance(gv, str):
                gu = gv.strip().upper()
                if gu in _WORD_TO_INT:
                    obj[key] = _WORD_TO_INT[gu]
                    changed = True
        vc = obj.get("vote_counts")
        if isinstance(vc, dict):
            if _migrate_vote_counts(vc):
                changed = True
        for _k, child in obj.items():
            if migrate_json_obj(child):
                changed = True
    elif isinstance(obj, list):
        for it in obj:
            if migrate_json_obj(it):
                changed = True
    return changed


def _json_maybe_annotation(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:8000]
    except OSError:
        return False
    if '"overall_glevel"' not in head and "'overall_glevel'" not in head:
        return False
    return True


def migrate_json_file(path: Path, *, dry_run: bool) -> bool:
    if path.name.startswith("args_glevel") or path.name.startswith("package"):
        return False
    if not _json_maybe_annotation(path):
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    before = json.dumps(data, sort_keys=True, ensure_ascii=False)
    if not migrate_json_obj(data):
        return False
    after = json.dumps(data, sort_keys=True, ensure_ascii=False)
    if before == after:
        return False
    if not dry_run:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def iter_repo_paths(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in SKIP_DIR_NAMES for part in rel.parts):
            continue
        if any(part in SKIP_PATH_PARTS for part in rel.parts):
            continue
        yield p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--root",
        type=Path,
        default=_REPO,
        help="仓库根目录（默认为本仓库）",
    )
    args = ap.parse_args()
    root = args.root.expanduser().resolve()

    n_csv = n_json = 0
    for path in iter_repo_paths(root):
        suf = path.suffix.lower()
        try:
            if suf == ".csv":
                if migrate_csv(path, dry_run=args.dry_run):
                    print(f"[csv] {'(dry-run) ' if args.dry_run else ''}{path.relative_to(root)}")
                    n_csv += 1
            elif suf == ".json":
                if migrate_json_file(path, dry_run=args.dry_run):
                    print(f"[json] {'(dry-run) ' if args.dry_run else ''}{path.relative_to(root)}")
                    n_json += 1
        except Exception as e:
            print(f"[warn] skip {path.relative_to(root)}: {e}", file=sys.stderr)

    print(f"Done. migrated csv={n_csv} json={n_json} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
