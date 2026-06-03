"""
赛题分类与校验规则（SIS∞ 2026 赛题一）。

本模块不执行求解，只负责把「官方小问编号」映射到三类数学模型，
并决定是否需要欧氏范数下界 ``||u||_2^2 + ||v||_2^2 >= q^2``。

三类划分（与题面一致）
-----------------------
第一类 — 齐次 SIS（无穷范数）
    小问：1, 3, 6, 9
    条件：``t ≡ 0 (mod q)``
    几何：格上短向量（SVP）近似；须拒绝平凡解 ``u=v=0``
    求解侧重：BKZ/LLL 种子、模 q 核游走（kernel walk）

第二类 — 非齐次 SIS（无穷范数）
    小问：2, 4, 7, 10
    条件：``t ≢ 0 (mod q)``
    几何：嵌入后近似最近向量（CVP）
    求解侧重：CVP 提升种子、模拉回（modular pull），弱化 BKZ

第三类 — 特殊条件 SIS（无穷范数 + 欧氏下界）
    小问：5, 8
    条件：在 L∞ 可行之外，还要求 ``||u||_2^2 + ||v||_2^2 >= q^2``
    求解侧重：在 ``solve_sisinf`` 中提高 ``euclid_weight`` / 熵权重

与 ``solve_sisinf.apply_sis_class_defaults`` 的关系
----------------------------------------------------
读取 JSON 实例后，``solve_instances`` / ``run_class_batch`` 会：
  1. 用本模块判定 ``sis_class``；
  2. 用 ``effective_require_norm_ge_q2`` 决定是否校验欧氏下界；
  3. 再叠加类别相关的 ``SearchConfig`` 默认值。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

import numpy as np

# 官方题号集合（勿随意改动，与赛题说明一致）
CLASS_1_IDS: Set[int] = {1, 3, 6, 9}
CLASS_2_IDS: Set[int] = {2, 4, 7, 10}
CLASS_3_IDS: Set[int] = {5, 8}
ALL_IDS: Set[int] = CLASS_1_IDS | CLASS_2_IDS | CLASS_3_IDS


def problem_class_from_id(problem_id: int) -> int:
    """
    根据小问编号返回类别 1/2/3。

    Parameters
    ----------
    problem_id : int
        赛题 JSON 中的 ``id`` 字段（1..10）。

    Returns
    -------
    int
        1=齐次，2=非齐次，3=欧氏下界特殊题。

    Raises
    ------
    ValueError
        题号不在 1..10 的官方集合中。
    """
    if problem_id in CLASS_1_IDS:
        return 1
    if problem_id in CLASS_2_IDS:
        return 2
    if problem_id in CLASS_3_IDS:
        return 3
    raise ValueError(f"unknown problem id {problem_id}; expected 1..10")


def problem_class_from_instance(inst: Dict[str, Any]) -> int:
    """
    从实例字典推断类别：优先 ``id``，否则用 ``t mod q`` 与 ``require_norm_ge_q2`` 启发式判断。

    用于自定义实例或未带官方 id 的测试数据。
    """
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
    """判断目标向量在模 q 意义下是否为零（齐次 SIS）。"""
    return bool(np.all(np.mod(t, q) == 0))


def effective_require_norm_ge_q2(inst: Dict[str, Any], sis_class: Optional[int] = None) -> bool:
    """
    是否启用 ``verify_solution`` 中的欧氏下界检查。

    第三类（5、8）**强制**为 True；其余题读 JSON 字段 ``require_norm_ge_q2``
    （当前官方 JSON 均为 false，但 5/8 仍按第三类规则覆盖）。
    """
    cls = sis_class if sis_class is not None else problem_class_from_instance(inst)
    if cls == 3:
        return True
    return bool(inst.get("require_norm_ge_q2", False))


def class_label(cls: int) -> str:
    """人类可读的类别标签，用于日志与 batch_report.json。"""
    return {1: "homogeneous/SVP", 2: "inhomogeneous/CVP", 3: "special/euclidean"}.get(cls, "unknown")
