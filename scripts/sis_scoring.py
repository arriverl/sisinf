"""
赛题阶梯计分（与官方解析一致）。

前提：同余成立，第三类另需 ``||u||_2^2 + ||v||_2^2 < q^2``。
记 ``E_inf = max(||u||_inf, ||v||_inf)``。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from sis_problem_taxonomy import problem_class_from_instance


def competition_score(
    gamma: int,
    inf_u: int,
    inf_v: int,
    *,
    congruence_ok: bool = True,
    norm_sq: Optional[int] = None,
    q: Optional[int] = None,
    sis_class: int = 1,
) -> int:
    """返回 0–10 分；不满足前提条件返回 0。"""
    if not congruence_ok:
        return 0
    if sis_class == 3 and q is not None and norm_sq is not None:
        if norm_sq >= q * q:
            return 0
    e_inf = max(int(inf_u), int(inf_v))
    if e_inf <= gamma:
        return 10
    if e_inf == gamma + 1:
        return 8
    if e_inf == gamma + 2:
        return 6
    if e_inf == gamma + 3:
        return 4
    if e_inf == gamma + 4:
        return 2
    return 0


def score_from_verify(
    inst: Dict[str, Any],
    verify: Dict[str, int],
    *,
    sis_class: Optional[int] = None,
) -> Dict[str, Any]:
    """由 verify_solution 指标计算得分与档位列。"""
    cls = sis_class if sis_class is not None else problem_class_from_instance(inst)
    gamma = int(inst["gamma"])
    q = int(inst["q"])
    congr = bool(verify.get("congruence_ok", 0))
    inf_u = int(verify.get("inf_u", 999))
    inf_v = int(verify.get("inf_v", 999))
    norm_sq = int(verify.get("norm_sq", 0))
    pts = competition_score(
        gamma,
        inf_u,
        inf_v,
        congruence_ok=congr,
        norm_sq=norm_sq,
        q=q,
        sis_class=cls,
    )
    e_inf = max(inf_u, inf_v)
    return {
        "score": pts,
        "e_inf": e_inf,
        "gamma": gamma,
        "feasible_linf": int(e_inf <= gamma),
        "feasible_all": int(bool(verify.get("ok", 0))),
        "norm_sq": norm_sq,
        "q2": q * q,
    }
