"""赛方 g_level：整数 1 / 2 / 3（认知档位由低到高）。

训练时 CrossEntropy 仍使用类下标 0 / 1 / 2；导出 submission 的 ``g_level_pred`` 一律写官方 1–3。
DeepSeek 与人工可读 prompt 使用同一套整数标签（必要时可解析旧版 LOW/MEDIUM/HIGH JSON）。"""
from __future__ import annotations


def parse_overall_glevel_value(v: object) -> int | None:
    """将 API / JSON 中的 ``overall_glevel`` 规范为 1、2 或 3；无法识别则返回 None。"""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        i = int(v)
        return i if i in (1, 2, 3) else None
    s = str(v).strip()
    if not s:
        return None
    if s.isdigit():
        i = int(s)
        return i if i in (1, 2, 3) else None
    u = s.upper()
    if u == "LOW":
        return 1
    if u in ("MEDIUM", "MID", "MED"):
        return 2
    if u == "HIGH":
        return 3
    return None


def glevel_csv_to_int(g: object) -> int | None:
    """读取划分 CSV 中 ``g_level`` 列（官方为 1–3）。"""
    return parse_overall_glevel_value(g)


def class_index_to_submission_label(idx: int) -> int:
    """CE 类下标 0..2 → 官方 g_level 1..3。"""
    i = int(idx)
    if i not in (0, 1, 2):
        raise ValueError(f"class index must be 0..2, got {idx!r}")
    return i + 1
