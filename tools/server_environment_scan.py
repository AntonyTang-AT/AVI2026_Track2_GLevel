#!/usr/bin/env python3
"""
在服务器上扫描 AVI2026_Track2_GLevel 训练所需路径、环境变量与关键文件，生成本地可读的报告，
便于对照 vote_train_glevel.sh 默认值与真实挂载目录。

输出（默认写入项目根下 artifacts/）:
  - server_scan_latest.json   机器可读
  - server_scan_latest.txt    人类可读
  - captured_logs.txt         若干日志文件尾部拼接（便于与报错一并拉回本机）

用法（在服务器、项目根目录）:
  python3 tools/server_environment_scan.py
  python3 tools/server_environment_scan.py --project-root /path/to/AVI2026_Track2_GLevel
  python3 tools/server_environment_scan.py --tail-lines 300 --capture-glob "logs/*.log"

本机拉回见: tools/pull_server_artifacts.sh / tools/pull_server_artifacts.ps1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _root_from_args(project_root: str | None) -> Path:
    if project_root:
        return Path(project_root).resolve()
    env = os.environ.get("PROJECT_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


def _count_npy(d: Path, cap: int = 2_000_000) -> int | str:
    if not d.is_dir():
        return 0
    n = 0
    try:
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() == ".npy":
                n += 1
                if n >= cap:
                    return f">={cap}"
    except OSError as e:
        return f"error:{e}"
    return n


def _npy_shape_sample(f: Path) -> str:
    try:
        import numpy as np

        a = np.load(f, mmap_mode="r")
        return str(tuple(a.shape))
    except Exception as e:
        return f"error:{e}"


def _tail_file(path: Path, lines: int) -> str:
    if not path.is_file():
        return ""
    try:
        data = path.read_bytes()
        if len(data) > 8 * 1024 * 1024:
            return f"(file too large, skip tail: {len(data)} bytes)\n"
        text = data.decode("utf-8", errors="replace").splitlines()
        chunk = text[-lines:] if lines > 0 else text
        return "\n".join(chunk) + ("\n" if chunk else "")
    except OSError as e:
        return f"(read error: {e})\n"


def _run_cmd(cmd: list[str], timeout: int = 15) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "") + (r.stderr or "")
        return out.strip() or f"(exit {r.returncode})"
    except Exception as e:
        return f"(error: {e})"


def _path_report(label: str, p: Path | None) -> dict:
    if p is None or str(p).strip() == "":
        return {"label": label, "path": "", "exists": False}
    path = Path(p)
    return {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "is_file": path.is_file(),
    }


def _feat_branch(name: str, base: Path) -> dict:
    out: dict = {"name": name, "base": str(base)}
    if not base.exists():
        out["exists"] = False
        return out
    out["exists"] = True
    for sub in ("audio", "video", "text", "text_nb", "text_nb_smoke"):
        d = base / sub
        out[f"npy_{sub}"] = _count_npy(d)
    return out


def _torch_info() -> dict:
    info: dict = {"import_ok": False}
    try:
        import torch

        info["import_ok"] = True
        info["version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception as e:
        info["error"] = repr(e)
    return info


def _dataset_revision(root: Path) -> str:
    p = root / "dataset" / "baseline_dataset2_vote.py"
    if not p.is_file():
        return "missing_file"
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
        for line in txt.splitlines():
            if line.strip().startswith("FEATURE_LOADER_REVISION"):
                return line.strip()
    except OSError:
        pass
    return "unreadable"


def main() -> int:
    ap = argparse.ArgumentParser(description="服务器路径与环境扫描（AVI2026 g_level）")
    ap.add_argument("--project-root", type=str, default="", help="项目根；默认 PROJECT_ROOT 或本仓库根")
    ap.add_argument("--out-dir", type=str, default="", help="输出目录；默认 <项目根>/artifacts")
    ap.add_argument("--tail-lines", type=int, default=200, help="拼接日志时每文件取最后 N 行")
    ap.add_argument(
        "--capture-glob",
        action="append",
        default=[],
        help="额外要抓取尾部的 glob（相对项目根），可多次指定",
    )
    args = ap.parse_args()

    root = _root_from_args(args.project_root or None)
    out_dir = Path(args.out_dir).resolve() if args.out_dir.strip() else root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 与 vote_train_glevel.sh 一致的默认路径
    def E(name: str, default: str) -> str:
        v = os.environ.get(name, "").strip()
        return v if v else default

    train_csv = Path(E("TRAIN_CSV", "/data/Super-Lu/dataset/train_data.csv"))
    val_csv = Path(E("VAL_CSV", "/data/Super-Lu/dataset/val_data.csv"))
    test_csv = Path(E("TEST_CSV", str(root / "data/test_data_basic_information.csv")))
    rating_csv = Path(E("RATING_CSV", "/data/Super-Lu/dataset/train_data.csv"))
    feat_train = Path(E("FEAT_TRAIN", "/data/Super-Lu/dataset/train_feature"))
    feat_val = Path(E("FEAT_VAL", "/data/Super-Lu/dataset/val_feature"))
    feat_test = Path(E("FEAT_TEST", "/data/AVI2026/test_feature"))

    nb = E("NANBEIGE_TEXT_SUBDIR", "text_nb")
    text_train = E("TEXT_TRAIN_DIR", "")
    text_val = E("TEXT_VAL_DIR", "")
    text_test = E("TEXT_TEST_DIR", "")
    if E("NANBEIGE_TEXT", "0") == "1":
        if not text_train:
            text_train = str(feat_train / nb)
        if not text_val:
            text_val = str(feat_val / nb)
        if not text_test:
            text_test = str(feat_test / nb)
    else:
        if not text_train:
            text_train = str(feat_train / "text")
        if not text_val:
            text_val = str(feat_val / "text")
        if not text_test:
            text_test = str(feat_test / "text")

    report: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "project_root": str(root),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "env_relevant": {
            "PROJECT_ROOT": os.environ.get("PROJECT_ROOT", ""),
            "PYTHON": os.environ.get("PYTHON", ""),
            "CONDA_PREFIX": os.environ.get("CONDA_PREFIX", ""),
            "FEAT_TRAIN": os.environ.get("FEAT_TRAIN", ""),
            "FEAT_VAL": os.environ.get("FEAT_VAL", ""),
            "FEAT_TEST": os.environ.get("FEAT_TEST", ""),
            "TRAIN_CSV": os.environ.get("TRAIN_CSV", ""),
            "VAL_CSV": os.environ.get("VAL_CSV", ""),
            "TEST_CSV": os.environ.get("TEST_CSV", ""),
            "NANBEIGE_TEXT": os.environ.get("NANBEIGE_TEXT", ""),
            "NANBEIGE_TEXT_SUBDIR": os.environ.get("NANBEIGE_TEXT_SUBDIR", ""),
            "TEXT_DIM": os.environ.get("TEXT_DIM", ""),
            "TEXT_TRAIN_DIR": os.environ.get("TEXT_TRAIN_DIR", ""),
            "TEXT_VAL_DIR": os.environ.get("TEXT_VAL_DIR", ""),
            "TEXT_TEST_DIR": os.environ.get("TEXT_TEST_DIR", ""),
            "GLEVEL_OPT": os.environ.get("GLEVEL_OPT", ""),
        },
        "csv": {
            "train": _path_report("TRAIN_CSV", train_csv),
            "val": _path_report("VAL_CSV", val_csv),
            "test_meta": _path_report("TEST_CSV", test_csv),
            "rating": _path_report("RATING_CSV", rating_csv),
        },
        "features": {
            "train": _feat_branch("FEAT_TRAIN", feat_train),
            "val": _feat_branch("FEAT_VAL", feat_val),
            "test": _feat_branch("FEAT_TEST", feat_test),
        },
        "resolved_text_dirs": {
            "TEXT_TRAIN_DIR": _path_report("", Path(text_train)),
            "TEXT_VAL_DIR": _path_report("", Path(text_val)),
            "TEXT_TEST_DIR": _path_report("", Path(text_test)),
        },
        "torch": _torch_info(),
        "dataset_loader": {"FEATURE_LOADER_REVISION_line": _dataset_revision(root)},
        "nvidia_smi": _run_cmd(["nvidia-smi", "-L"], timeout=10),
        "disk_df": _run_cmd(["df", "-h", str(root)], timeout=10)
        if platform.system() != "Windows"
        else "",
    }

    # 任选一个 text npy 看形状
    sample_txt = Path(text_train)
    if sample_txt.is_dir():
        npys = sorted(sample_txt.glob("*.npy"))
        if npys:
            report["sample_text_npy"] = {
                "file": str(npys[0]),
                "shape": _npy_shape_sample(npys[0]),
            }

    # 拼接日志尾部
    capture_parts: list[str] = []
    default_globs = ["debug-f0e227.log", "train_glevel.log", "nohup.out"]
    all_globs = default_globs + list(args.capture_glob or [])
    seen: set[str] = set()
    for pattern in all_globs:
        paths = []
        if any(ch in pattern for ch in "*?["):
            paths = [Path(p) for p in glob.glob(str(root / pattern), recursive=True)]
        else:
            p = root / pattern
            if p.is_file():
                paths = [p]
        for p in paths:
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            block = _tail_file(p, args.tail_lines)
            capture_parts.append(f"===== TAIL {rp} (last {args.tail_lines} lines) =====\n")
            capture_parts.append(block or "(empty or missing)\n")

    captured_txt = "".join(capture_parts)
    (out_dir / "captured_logs.txt").write_text(captured_txt, encoding="utf-8")

    json_path = out_dir / "server_scan_latest.json"
    txt_path = out_dir / "server_scan_latest.txt"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"生成时间(UTC): {report['generated_at_utc']}",
        f"主机: {report['hostname']}",
        f"项目根: {report['project_root']}",
        f"Python: {report['python']['executable']} ({report['python']['version']})",
        "",
        "=== 关键 CSV ===",
    ]
    for k, v in report["csv"].items():
        lines.append(f"  [{k}] exists={v.get('exists')} path={v.get('path')}")
    lines += ["", "=== 特征根目录与 .npy 数量（各子目录一层）==="]
    for split, body in report["features"].items():
        lines.append(f"  [{split}] {json.dumps(body, ensure_ascii=False)}")
    lines += ["", "=== 解析后的文本特征目录 ==="]
    for k, v in report["resolved_text_dirs"].items():
        lines.append(f"  {k}: exists={v.get('exists')} {v.get('path')}")
    lines += ["", "=== PyTorch ===", f"  {json.dumps(report['torch'], ensure_ascii=False)}"]
    lines += ["", "=== Dataset ===", f"  {report['dataset_loader']['FEATURE_LOADER_REVISION_line']}"]
    if report.get("nvidia_smi"):
        lines += ["", "=== nvidia-smi -L ===", report["nvidia_smi"]]
    if report.get("disk_df"):
        lines += ["", "=== df -h (project) ===", report["disk_df"]]
    if report.get("sample_text_npy"):
        lines += ["", "=== 示例 text .npy ===", json.dumps(report["sample_text_npy"], ensure_ascii=False)]
    lines += [
        "",
        f"详细 JSON: {json_path}",
        f"日志尾部汇总: {out_dir / 'captured_logs.txt'}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[server_environment_scan] 已写入:\n  {json_path}\n  {txt_path}\n  {out_dir / 'captured_logs.txt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
