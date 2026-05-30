"""
SIS∞ 赛题三类划分（与题号对应）。

第一类（齐次，t≡0）：1, 3, 6, 9  → SVP / BKZ + 核游走，拒绝平凡零解
第二类（非齐次，t≢0）：2, 4, 7, 10 → CVP 提升 / 模拉回种子
第三类（特殊，欧氏下界）：5, 8   → 在 L∞ 可行基础上要求 ||u||_2^2+||v||_2^2 >= q^2
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

import numpy as np

CLASS_1_IDS: Set[int] = {1, 3, 6, 9}
CLASS_2_IDS: Set[int] = {2, 4, 7, 10}
CLASS_3_IDS: Set[int] = {5, 8}
ALL_IDS: Set[int] = CLASS_1_IDS | CLASS_2_IDS | CLASS_3_IDS


def problem_class_from_id(problem_id: int) -> int:
    if problem_id in CLASS_1_IDS:
        return 1
    if problem_id in CLASS_2_IDS:
        return 2
    if problem_id in CLASS_3_IDS:
        return 3
    raise ValueError(f"unknown problem id {problem_id}; expected 1..10")


def problem_class_from_instance(inst: Dict[str, Any]) -> int:
    pid = int(inst.get("id", 0))
    if pid in ALL_IDS:
        return problem_class_from_id(pid)
    t = np.asarray(inst["t"], dtype=np.int64)
    q = int(inst["q"])
    if is_homogeneous_target(t, q):
        if bool(inst.get("require_norm_ge_q2", False)):
            return 3
        return 1
    return 2


def is_homogeneous_target(t: np.ndarray, q: int) -> bool:
    return bool(np.all(np.mod(t, q) == 0))


def effective_require_norm_ge_q2(inst: Dict[str, Any], sis_class: Optional[int] = None) -> bool:
    cls = sis_class if sis_class is not None else problem_class_from_instance(inst)
    if cls == 3:
        return True
    return bool(inst.get("require_norm_ge_q2", False))


def class_label(cls: int) -> str:
    return {1: "homogeneous/SVP", 2: "inhomogeneous/CVP", 3: "special/euclidean"}.get(cls, "unknown")
