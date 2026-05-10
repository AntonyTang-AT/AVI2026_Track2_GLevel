"""读取并对齐多份提交 CSV。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

PRED_COL_CANDIDATES = ("g_level_pred", "g_level", "prediction", "pred")


def _detect_pred_column(fieldnames: Sequence[str] | None) -> str:
    if not fieldnames:
        raise ValueError("CSV 无表头")
    lower = {n.strip().lower(): n for n in fieldnames}
    for cand in PRED_COL_CANDIDATES:
        if cand in lower:
            return lower[cand]
    # 第二列兜底
    cols = [n.strip() for n in fieldnames if n.strip()]
    if len(cols) >= 2:
        return cols[1]
    raise ValueError(f"无法识别预测列，表头={fieldnames!r}")


def _read_one_csv(path: Path) -> tuple[list[str], list[int], str]:
    path = Path(path).expanduser().resolve()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path}: 空表头")
        pred_col = _detect_pred_column(reader.fieldnames)
        id_col = reader.fieldnames[0].strip()
        ids: list[str] = []
        preds: list[int] = []
        for row in reader:
            sid = (row.get(id_col) or "").strip()
            if not sid:
                continue
            raw = row.get(pred_col)
            if raw is None or str(raw).strip() == "":
                raise ValueError(f"{path}: id={sid} 缺 {pred_col}")
            v = int(round(float(str(raw).strip())))
            if v not in (1, 2, 3):
                raise ValueError(f"{path}: id={sid} 非法 g_level={v}（须为 1/2/3）")
            ids.append(sid)
            preds.append(v)
    return ids, preds, pred_col


def load_aligned_matrix(
    csv_paths: Iterable[str | Path],
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """
    返回:
      preds: int32 数组，形状 (n_samples, n_models)
      ids:   对齐后的样本 id 列表
      paths: 每个模型的绝对路径字符串
      pred_cols: 每个文件使用的预测列名（便于日志）
    """
    paths = [str(Path(p).expanduser().resolve()) for p in csv_paths]
    if len(paths) < 1:
        raise ValueError("至少提供一份 CSV")

    tables: list[tuple[list[str], list[int], str]] = [_read_one_csv(Path(p)) for p in paths]

    id_sets = [set(t[0]) for t in tables]
    common = set.intersection(*id_sets)
    if not common:
        raise ValueError("各 CSV 的 id 交集为空，请检查是否同一测试集")

    # 固定字典序，保证可复现（与具体 CSV 行序无关）
    ordered_ids = sorted(common)

    pred_cols: list[str] = []
    mats: list[np.ndarray] = []
    id_to_idx = {i: k for k, i in enumerate(ordered_ids)}

    for (ids_l, pr_l, pc), pth in zip(tables, paths):
        pred_cols.append(pc)
        mp = dict(zip(ids_l, pr_l))
        col = np.zeros(len(ordered_ids), dtype=np.int32)
        for j, sid in enumerate(ordered_ids):
            col[j] = mp[sid]
        mats.append(col)

    preds = np.stack(mats, axis=1)
    return preds, ordered_ids, paths, pred_cols
