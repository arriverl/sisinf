"""
Incumbent 收尾算子：小子格真 BKZ 种子 + 全维 v 的 L∞ ILP（ortools CP-SAT）。

用于 ``inf_u`` 卡在平台（如 44）时，从 ``problem*_best.json`` 继续攻击。
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lattice_bkz import (
    _build_ajtai_basis,
    _fpylll_reduce_multi_tour,
    _seeds_from_reduced_basis,
    fpylll_available,
)
from solve_sisinf import center_mod, verify_solution

_CP_STATUS_NAMES: Dict[int, str] = {}


def _cp_status_name(status: int) -> str:
    if not _CP_STATUS_NAMES:
        try:
            from ortools.sat.python import cp_model  # type: ignore

            _CP_STATUS_NAMES.update(
                {
                    cp_model.OPTIMAL: "OPTIMAL",
                    cp_model.FEASIBLE: "FEASIBLE",
                    cp_model.INFEASIBLE: "INFEASIBLE",
                    cp_model.MODEL_INVALID: "MODEL_INVALID",
                    cp_model.UNKNOWN: "UNKNOWN",
                }
            )
        except Exception:
            pass
    return _CP_STATUS_NAMES.get(status, str(status))


def pick_worst_u_row_col_subset(
    A: np.ndarray,
    residual: np.ndarray,
    gamma: int,
    n_rows: int,
    n_cols: int,
    q: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """按 |u| 溢出选取 worst 行，再按列与这些行的耦合强度选列子集。"""
    A = np.mod(np.asarray(A, dtype=np.int64), q)
    abs_r = np.abs(np.asarray(residual, dtype=np.int64).ravel())
    overflow = np.maximum(abs_r - gamma, 0)
    bad = np.flatnonzero(overflow > 0)
    if bad.size == 0:
        bad = np.argsort(-abs_r)[: min(n_rows, abs_r.size)]
    else:
        order = np.argsort(-overflow[bad])
        bad = bad[order[: min(n_rows, bad.size)]]
    if bad.size == 0:
        bad = np.arange(min(n_rows, A.shape[0]), dtype=np.int64)

    col_score = np.sum(np.abs(A[bad, :]), axis=0)
    ncol = min(n_cols, A.shape[1])
    cols = np.argsort(-col_score)[:ncol]
    return bad.astype(np.int64), cols.astype(np.int64)


def collect_sub_bkz_v_seeds(
    A: np.ndarray,
    q: int,
    gamma: int,
    residual: np.ndarray,
    beta: int,
    n_rows: int = 40,
    n_cols: int = 40,
    max_vectors: int = 16,
    combo_depth: int = 4,
    combo_coeff_max: int = 2,
    v_base: Optional[np.ndarray] = None,
    *,
    embed_mode: str = "zero",
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    在 worst-u 对应的 (n_rows × n_cols) 子块上建 Ajtai 格（维数 ≤80），跑**真 BKZ**。

    返回嵌入全维 m 的 v 种子列表及诊断 meta。
    """
    meta: Dict[str, Any] = {
        "sub_bkz_available": fpylll_available(),
        "sub_bkz_rows": 0,
        "sub_bkz_cols": 0,
        "sub_bkz_dim": 0,
        "sub_bkz_seed_count": 0,
    }
    if not fpylll_available() or beta <= 0 or max_vectors <= 0:
        return [], meta

    A = np.mod(np.asarray(A, dtype=np.int64), q)
    n, m = A.shape
    row_idx, col_idx = pick_worst_u_row_col_subset(A, residual, gamma, n_rows, n_cols, q)
    meta["sub_bkz_rows"] = int(row_idx.size)
    meta["sub_bkz_cols"] = int(col_idx.size)

    A_sub = A[row_idx][:, col_idx]
    B, n_s, m_s = _build_ajtai_basis(A_sub, q)
    d = n_s + m_s
    meta["sub_bkz_dim"] = d
    if d < 4:
        return [], meta

    v0 = np.zeros(m, dtype=np.int64) if v_base is None else np.asarray(v_base, dtype=np.int64).copy()

    try:
        R = _fpylll_reduce_multi_tour(B, beta, tours=2, force_bkz=True)
    except Exception as exc:
        meta["sub_bkz_error"] = str(exc)
        return [], meta

    partials: List[np.ndarray] = []
    partial_seen: set = set()
    _seeds_from_reduced_basis(
        R, n_s, m_s, gamma, max_vectors, combo_depth, combo_coeff_max, partial_seen, partials
    )

    out: List[np.ndarray] = []
    full_seen: set = set()
    for pv in partials:
        v_full = v0.copy()
        v_full[col_idx] = np.clip(pv, -gamma, gamma)
        if np.all(v_full == 0):
            continue
        key = v_full.tobytes()
        if key in full_seen:
            continue
        full_seen.add(key)
        out.append(v_full)
        if len(out) >= max_vectors:
            break

    meta["sub_bkz_seed_count"] = len(out)
    return out[:max_vectors], meta


def cp_sat_full_v_linf_finish(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    v0: np.ndarray,
    time_limit_sec: float = 3600.0,
    *,
    use_hint: bool = True,
    num_workers: int = 4,
    accept_suboptimal: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    对**全部** v 维做 CP-SAT，最小化 ``‖center(t−Av)‖∞``（即 u 的 L∞ 溢出上界）。

    始终返回 ``(u, v, meta)``；失败时 u/v 可能为 None，meta 含 ``ilp_error`` / ``ilp_status_name``。
    """
    meta: Dict[str, Any] = {
        "ilp_ok": False,
        "ilp_error": None,
        "ilp_status": None,
        "ilp_status_name": None,
        "ilp_time_sec": 0.0,
    }
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception as exc:
        meta["ilp_error"] = f"ortools import failed: {exc}"
        return None, None, meta

    t0 = time.perf_counter()
    try:
        A = np.mod(np.asarray(A, dtype=np.int64), q)
        t = np.mod(np.asarray(t, dtype=np.int64), q)
        v0 = np.clip(np.asarray(v0, dtype=np.int64).ravel(), -gamma, gamma)
        n, m = A.shape
        if v0.size != m:
            meta["ilp_error"] = f"v0 size {v0.size} != m {m}"
            return None, None, meta

        model = cp_model.CpModel()
        v_vars: Dict[int, Any] = {}
        for j in range(m):
            vj = model.NewIntVar(-gamma, gamma, f"v_{j}")
            v_vars[j] = vj
            if use_hint:
                model.AddHint(vj, int(v0[j]))

        abs_v_move = []
        for j in range(m):
            mv = model.NewIntVar(0, 2 * gamma, f"mv_{j}")
            model.AddAbsEquality(mv, v_vars[j] - int(v0[j]))
            abs_v_move.append(mv)

        over_vars = []
        over_ub = gamma + 8
        for i in range(n):
            s_bound = int(np.sum(np.abs(A[i, :])) * gamma)
            lo_raw = int(t[i]) - s_bound
            hi_raw = int(t[i]) + s_bound
            raw_i = model.NewIntVar(lo_raw, hi_raw, f"raw_{i}")
            model.Add(
                raw_i
                == int(t[i])
                - sum(int(A[i, j]) * v_vars[j] for j in range(m) if int(A[i, j]) != 0)
            )

            k_min = int(np.floor((lo_raw + gamma) / q)) - 1
            k_max = int(np.ceil((hi_raw - gamma) / q)) + 1
            if k_min > k_max:
                k_min, k_max = k_max, k_min
            k_i = model.NewIntVar(k_min, k_max, f"k_{i}")
            centered = raw_i - q * k_i

            abs_hi = max(abs(lo_raw), abs(hi_raw)) + abs(q) * max(abs(k_min), abs(k_max)) + gamma + 8
            abs_c = model.NewIntVar(0, abs_hi, f"abs_{i}")
            model.AddAbsEquality(abs_c, centered)

            over_i = model.NewIntVar(0, abs_hi, f"over_{i}")
            model.Add(over_i + gamma >= abs_c)
            over_vars.append(over_i)
            over_ub = max(over_ub, abs_hi)

        max_over = model.NewIntVar(0, over_ub, "max_over")
        model.AddMaxEquality(max_over, over_vars)
        model.Minimize(1_000_000 * max_over + sum(over_vars) + sum(abs_v_move))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(1.0, float(time_limit_sec))
        solver.parameters.num_search_workers = max(1, int(num_workers))
        status = solver.Solve(model)

        meta["ilp_status"] = int(status)
        meta["ilp_status_name"] = _cp_status_name(int(status))
        meta["ilp_time_sec"] = time.perf_counter() - t0
        meta["ilp_optimal"] = status == cp_model.OPTIMAL
        meta["ilp_feasible"] = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            meta["ilp_error"] = f"solver status {_cp_status_name(int(status))}"
            if status == cp_model.MODEL_INVALID:
                meta["ilp_error"] += " (check constraint linearity)"
            return None, None, meta

        v_new = np.array([int(solver.Value(v_vars[j])) for j in range(m)], dtype=np.int64)
        u_new = center_mod(t - (A @ v_new), q)
        ok, verify = verify_solution(A, t, q, gamma, u_new, v_new, False)
        meta["ilp_max_over"] = int(solver.Value(max_over))
        meta["verify"] = verify
        meta["success"] = bool(ok)
        meta["ilp_ok"] = True

        if ok or accept_suboptimal:
            return u_new, v_new, meta
        meta["ilp_error"] = "solution not accepted"
        return None, None, meta

    except Exception as exc:
        meta["ilp_error"] = f"{type(exc).__name__}: {exc}"
        meta["ilp_traceback"] = traceback.format_exc()
        meta["ilp_time_sec"] = time.perf_counter() - t0
        return None, None, meta
