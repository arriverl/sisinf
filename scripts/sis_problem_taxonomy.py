"""
赛题分类与校验规则（SIS∞ 2026 赛题一）。

本模块不执行求解，只负责把「官方小问编号」映射到三类数学模型，
并决定第三类是否启用欧氏上界 ``||u||_2^2 + ||v||_2^2 < q^2``（官方计分：≥q² 不得分）。

三类划分（与题面一致）
-----------------------
第一类 — 齐次 SIS（无穷范数）
    小问：1, 3, 6, 9
    条件：``t ≡ 0 (mod q)``
    几何：格上短向量（SVP）近似；须拒绝平凡解 ``u=v=0``
    求解侧重：BKZ 2.0 + 筛法、模 q 核游走

第二类 — 非齐次 SIS（无穷范数）
    小问：2, 4, 7, 10
    条件：``t ≢ 0 (mod q)``
    几何：格上近似 CVP（Kannan 嵌入）
    求解侧重：CVP 提升、模拉回

第三类 — 特殊条件 SIS（无穷范数 + 欧氏上界）
    小问：5, 8
    条件：``||u||_2^2 + ||v||_2^2 < q^2``（官方解析）
    求解侧重：受限 SVP（Wang 2025）、禁止 L₂ 短向量后过滤
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
        if bool(inst.get("require_norm_lt_q2", False)) or bool(inst.get("require_norm_ge_q2", False)):
            return 3
        return 1
    return 2


def is_homogeneous_target(t: np.ndarray, q: int) -> bool:
    return bool(np.all(np.mod(t, q) == 0))


def effective_require_norm_lt_q2(inst: Dict[str, Any], sis_class: Optional[int] = None) -> bool:
    """
    是否启用第三类官方欧氏上界：``norm_sq < q^2``。

    第三类（5、8）强制为 True；其余题读 JSON ``require_norm_lt_q2``（默认 false）。
    """
    cls = sis_class if sis_class is not None else problem_class_from_instance(inst)
    if cls == 3:
        return True
    return bool(inst.get("require_norm_lt_q2", False))


def effective_require_norm_ge_q2(inst: Dict[str, Any], sis_class: Optional[int] = None) -> bool:
    """向后兼容别名 → ``effective_require_norm_lt_q2``。"""
    return effective_require_norm_lt_q2(inst, sis_class)


def class_label(cls: int) -> str:
    return {
        1: "homogeneous/SVP",
        2: "inhomogeneous/CVP",
        3: "special/restricted-SVP",
    }.get(cls, "unknown")
