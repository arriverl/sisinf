"""
ILP/CP-SAT 收尾：建模、求解、execute_finish 管线。
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from lattice_seeds import (
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


def _ilp_meta_base() -> Dict[str, Any]:
    return {
        "ilp_ok": False,
        "ilp_error": None,
        "ilp_status": None,
        "ilp_status_name": None,
        "ilp_time_sec": 0.0,
        "ilp_mode": "full",
    }


def _import_cp_model() -> Tuple[Any, Any, Optional[str]]:
    try:
        from ortools.sat.python import cp_model  # type: ignore

        return cp_model, cp_model, None
    except Exception as exc:
        return None, None, f"ortools import failed: {exc}"


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


def _column_order_by_u_coupling(
    A: np.ndarray,
    residual: np.ndarray,
    gamma: int,
    q: int,
    top_rows: int = 40,
) -> np.ndarray:
    """与 worst-u 行耦合最强的列优先（用于分块 ILP 轮换）。"""
    n, m = A.shape
    rows, _ = pick_worst_u_row_col_subset(A, residual, gamma, top_rows, m, q)
    if rows.size == 0:
        return np.arange(m, dtype=np.int64)
    score = np.sum(np.abs(A[rows, :]), axis=0)
    return np.argsort(-score).astype(np.int64)


def _raw_row_bounds(
    t_i: int,
    A_i: np.ndarray,
    v_fixed: np.ndarray,
    gamma: int,
    free_cols: Set[int],
) -> Tuple[int, int, int]:
    """仅 ``free_cols`` 可动时，``raw_i = t_i - (Av)_i`` 的整数上下界与当前常数项。"""
    base = int(t_i)
    sway = 0
    for j, aij in enumerate(A_i):
        aij = int(aij)
        if aij == 0:
            continue
        if j in free_cols:
            sway += abs(aij) * gamma
        else:
            base -= aij * int(v_fixed[j])
    return base - sway, base + sway, base


def _build_linf_u_cp_model(
    cp_model: Any,
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    v_cur: np.ndarray,
    free_cols: Set[int],
    *,
    use_hint: bool = True,
    name_prefix: str = "",
    max_over_cap: Optional[int] = None,
) -> Tuple[Any, Dict[int, Any], List[Any], Any, List[Any], int]:
    """
    构造 L∞-u 子模型：``free_cols`` 内 v 为变量，其余 v 固定为 ``v_cur``。

    Returns
    -------
    model, v_vars, over_vars, max_over, abs_v_move, over_ub
    """
    n, m = A.shape
    model = cp_model.CpModel()
    v_vars: Dict[int, Any] = {}
    for j in free_cols:
        vj = model.NewIntVar(-gamma, gamma, f"{name_prefix}v_{j}")
        v_vars[j] = vj
        if use_hint:
            model.AddHint(vj, int(v_cur[j]))

    abs_v_move = []
    for j in free_cols:
        mv = model.NewIntVar(0, 2 * gamma, f"{name_prefix}mv_{j}")
        model.AddAbsEquality(mv, v_vars[j] - int(v_cur[j]))
        abs_v_move.append(mv)

    over_vars: List[Any] = []
    over_ub = gamma + 8
    for i in range(n):
        lo_raw, hi_raw, _ = _raw_row_bounds(int(t[i]), A[i], v_cur, gamma, free_cols)
        raw_i = model.NewIntVar(lo_raw, hi_raw, f"{name_prefix}raw_{i}")
        expr_terms = [int(t[i])]
        for j in range(m):
            aij = int(A[i, j])
            if aij == 0:
                continue
            if j in free_cols:
                expr_terms.append(-aij * v_vars[j])
            else:
                expr_terms.append(-aij * int(v_cur[j]))
        model.Add(raw_i == sum(expr_terms))

        k_min = int(np.floor((lo_raw + gamma) / q)) - 1
        k_max = int(np.ceil((hi_raw - gamma) / q)) + 1
        if k_min > k_max:
            k_min, k_max = k_max, k_min
        k_i = model.NewIntVar(k_min, k_max, f"{name_prefix}k_{i}")
        centered = raw_i - q * k_i

        abs_hi = max(abs(lo_raw), abs(hi_raw)) + abs(q) * max(abs(k_min), abs(k_max)) + gamma + 8
        abs_c = model.NewIntVar(0, abs_hi, f"{name_prefix}abs_{i}")
        model.AddAbsEquality(abs_c, centered)

        over_i = model.NewIntVar(0, abs_hi, f"{name_prefix}over_{i}")
        model.Add(over_i + gamma >= abs_c)
        over_vars.append(over_i)
        over_ub = max(over_ub, abs_hi)

    cap = over_ub if max_over_cap is None else min(over_ub, int(max_over_cap))
    max_over = model.NewIntVar(0, cap, f"{name_prefix}max_over")
    model.AddMaxEquality(max_over, over_vars)
    if max_over_cap is not None:
        model.Add(max_over <= int(max_over_cap))
    return model, v_vars, over_vars, max_over, abs_v_move, over_ub


def _solve_cp_model(
    cp_model: Any,
    model: Any,
    time_limit_sec: float,
    num_workers: int,
) -> Tuple[int, Any]:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.5, float(time_limit_sec))
    solver.parameters.num_search_workers = max(1, int(num_workers))
    status = solver.Solve(model)
    return int(status), solver


def _extract_v_from_solver(
    v_cur: np.ndarray,
    v_vars: Dict[int, Any],
    solver: Any,
) -> np.ndarray:
    v_new = v_cur.copy()
    for j, var in v_vars.items():
        v_new[int(j)] = int(solver.Value(var))
    return v_new


def _finalize_ilp_result(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    v_new: np.ndarray,
    meta: Dict[str, Any],
    t0: float,
    status: int,
    solver: Any,
    max_over_var: Any,
    *,
    accept_suboptimal: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    cp_model = meta.get("_cp_model")
    meta.pop("_cp_model", None)
    meta["ilp_status"] = int(status)
    meta["ilp_status_name"] = _cp_status_name(int(status))
    meta["ilp_time_sec"] = time.perf_counter() - t0
    meta["ilp_optimal"] = status == cp_model.OPTIMAL if cp_model else False
    meta["ilp_feasible"] = status in (cp_model.OPTIMAL, cp_model.FEASIBLE) if cp_model else False

    if cp_model and status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        meta["ilp_error"] = f"solver status {_cp_status_name(int(status))}"
        return None, None, meta

    u_new = center_mod(t - (A @ v_new), q)
    ok, verify = verify_solution(A, t, q, gamma, u_new, v_new, False)
    meta["ilp_max_over"] = int(solver.Value(max_over_var))
    meta["verify"] = verify
    meta["success"] = bool(ok)
    meta["ilp_ok"] = True
    if ok or accept_suboptimal:
        return u_new, v_new, meta
    meta["ilp_error"] = "solution not accepted"
    return None, None, meta


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
    """在 worst-u 子块上建格（≤80 维）并跑真 BKZ，返回嵌入全维的 v 种子。"""
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
    """全维 v CP-SAT，最小化 u 的 L∞ 溢出上界。"""
    meta = _ilp_meta_base()
    meta["ilp_mode"] = "full"
    cp_model, _, err = _import_cp_model()
    if err:
        meta["ilp_error"] = err
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

        free_cols = set(range(m))
        model, v_vars, over_vars, max_over, abs_move, _ = _build_linf_u_cp_model(
            cp_model, A, t, q, gamma, v0, free_cols, use_hint=use_hint, name_prefix="f_"
        )
        model.Minimize(1_000_000 * max_over + sum(over_vars) + sum(abs_move))
        status, solver = _solve_cp_model(cp_model, model, time_limit_sec, num_workers)
        v_new = _extract_v_from_solver(v0, v_vars, solver)
        meta["_cp_model"] = cp_model
        return _finalize_ilp_result(
            A, t, q, gamma, v_new, meta, t0, status, solver, max_over, accept_suboptimal=accept_suboptimal
        )
    except Exception as exc:
        meta["ilp_error"] = f"{type(exc).__name__}: {exc}"
        meta["ilp_traceback"] = traceback.format_exc()
        meta["ilp_time_sec"] = time.perf_counter() - t0
        return None, None, meta


def cp_sat_chunked_v_linf_finish(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    v0: np.ndarray,
    time_limit_sec: float = 3600.0,
    *,
    chunk_cols: int = 40,
    chunk_rounds: int = 10,
    chunk_stride: Optional[int] = None,
    use_hint: bool = True,
    num_workers: int = 4,
    accept_suboptimal: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    分块 ILP：每轮只优化与 worst-u 强耦合的一列子集，多轮滑动覆盖全部 v。

    ``chunk_stride`` 默认为 ``chunk_cols // 2``（50% 重叠）。
    """
    meta = _ilp_meta_base()
    meta["ilp_mode"] = "chunk"
    meta["chunk_cols"] = int(chunk_cols)
    meta["chunk_rounds"] = int(chunk_rounds)
    cp_model, _, err = _import_cp_model()
    if err:
        meta["ilp_error"] = err
        return None, None, meta

    t0 = time.perf_counter()
    try:
        A = np.mod(np.asarray(A, dtype=np.int64), q)
        t = np.mod(np.asarray(t, dtype=np.int64), q)
        v_cur = np.clip(np.asarray(v0, dtype=np.int64).ravel(), -gamma, gamma)
        n, m = A.shape
        if v_cur.size != m:
            meta["ilp_error"] = f"v0 size {v_cur.size} != m {m}"
            return None, None, meta

        stride = chunk_stride if chunk_stride is not None else max(1, int(chunk_cols) // 2)
        per_round = max(30.0, float(time_limit_sec) / max(1, int(chunk_rounds)))
        round_logs: List[Dict[str, Any]] = []
        best_inf = int(np.max(np.abs(center_mod(t - (A @ v_cur), q))))
        best_inf = max(best_inf, int(np.max(np.abs(v_cur))))

        for rnd in range(max(1, int(chunk_rounds))):
            residual = center_mod(t - (A @ v_cur), q)
            col_order = _column_order_by_u_coupling(A, residual, gamma, q)
            start = (rnd * stride) % m
            picked: List[int] = []
            for k in range(min(int(chunk_cols), m)):
                picked.append(int(col_order[(start + k) % m]))
            free_cols = set(picked)
            u_before = int(np.max(np.abs(residual)))
            v_before = int(np.max(np.abs(v_cur)))

            model, v_vars, over_vars, max_over, abs_move, _ = _build_linf_u_cp_model(
                cp_model,
                A,
                t,
                q,
                gamma,
                v_cur,
                free_cols,
                use_hint=use_hint,
                name_prefix=f"c{rnd}_",
            )
            model.Minimize(1_000_000 * max_over + sum(over_vars) + sum(abs_move))
            status, solver = _solve_cp_model(cp_model, model, per_round, num_workers)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                round_logs.append(
                    {
                        "round": rnd,
                        "status": _cp_status_name(status),
                        "free_cols": len(free_cols),
                        "skipped": True,
                    }
                )
                continue

            v_try = _extract_v_from_solver(v_cur, v_vars, solver)
            u_try = center_mod(t - (A @ v_try), q)
            inf_u = int(np.max(np.abs(u_try)))
            inf_v = int(np.max(np.abs(v_try)))
            improved = inf_u < u_before or (inf_u == u_before and inf_v < v_before)
            if improved:
                v_cur = v_try
                best_inf = min(best_inf, inf_u)
            round_logs.append(
                {
                    "round": rnd,
                    "status": _cp_status_name(status),
                    "free_cols": len(free_cols),
                    "inf_u": inf_u,
                    "inf_v": inf_v,
                    "max_over": int(solver.Value(max_over)),
                    "improved": bool(improved),
                }
            )
            if int(np.max(np.abs(u_try))) <= gamma and int(np.max(np.abs(v_try))) <= gamma:
                break

        meta["chunk_round_logs"] = round_logs
        meta["ilp_time_sec"] = time.perf_counter() - t0
        u_new = center_mod(t - (A @ v_cur), q)
        ok, verify = verify_solution(A, t, q, gamma, u_new, v_cur, False)
        meta["verify"] = verify
        meta["success"] = bool(ok)
        meta["ilp_max_over"] = max(0, int(np.max(np.abs(u_new))) - gamma)
        meta["ilp_ok"] = True
        meta["ilp_status_name"] = "CHUNK_DONE"
        if ok or accept_suboptimal:
            return u_new, v_cur, meta
        meta["ilp_error"] = "chunk rounds did not improve"
        return None, None, meta
    except Exception as exc:
        meta["ilp_error"] = f"{type(exc).__name__}: {exc}"
        meta["ilp_traceback"] = traceback.format_exc()
        meta["ilp_time_sec"] = time.perf_counter() - t0
        return None, None, meta


def cp_sat_lex_v_linf_finish(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    v0: np.ndarray,
    time_limit_sec: float = 3600.0,
    *,
    phase_fracs: Sequence[float] = (0.45, 0.30, 0.25),
    use_hint: bool = True,
    num_workers: int = 4,
    accept_suboptimal: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    分层 ILP（lexicographic）三阶段：

    1. 最小化 ``max_over``（L∞ 上界）；
    2. 固定 ``max_over ≤ M*``，最小化 ``sum(over_i)``（摊平违规）；
    3. 再固定，最小化顶在 ``M*`` 上的行数 + 次级溢出。
    """
    meta = _ilp_meta_base()
    meta["ilp_mode"] = "lex"
    cp_model, _, err = _import_cp_model()
    if err:
        meta["ilp_error"] = err
        return None, None, meta

    t0 = time.perf_counter()
    phase_logs: List[Dict[str, Any]] = []
    try:
        A = np.mod(np.asarray(A, dtype=np.int64), q)
        t = np.mod(np.asarray(t, dtype=np.int64), q)
        v_cur = np.clip(np.asarray(v0, dtype=np.int64).ravel(), -gamma, gamma)
        n, m = A.shape
        if v_cur.size != m:
            meta["ilp_error"] = f"v0 size {v_cur.size} != m {m}"
            return None, None, meta

        fracs = list(phase_fracs)
        if len(fracs) < 3:
            fracs = (0.45, 0.30, 0.25)
        total = float(time_limit_sec)
        t_budgets = [total * fracs[0], total * fracs[1], total * fracs[2]]
        free_cols = set(range(m))

        # Phase 1: min max_over
        model1, v_vars1, over1, max_over1, move1, _ = _build_linf_u_cp_model(
            cp_model, A, t, q, gamma, v_cur, free_cols, use_hint=use_hint, name_prefix="l1_"
        )
        model1.Minimize(1_000_000 * max_over1 + sum(over1) + sum(move1))
        st1, sol1 = _solve_cp_model(cp_model, model1, t_budgets[0], num_workers)
        if st1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            meta["ilp_error"] = f"lex phase1 {_cp_status_name(st1)}"
            meta["phase_logs"] = phase_logs
            meta["ilp_time_sec"] = time.perf_counter() - t0
            return None, None, meta
        v_cur = _extract_v_from_solver(v_cur, v_vars1, sol1)
        M_star = int(sol1.Value(max_over1))
        phase_logs.append({"phase": 1, "max_over": M_star, "status": _cp_status_name(st1)})

        # Phase 2: max_over <= M_star, min sum(over)
        model2, v_vars2, over2, max_over2, move2, _ = _build_linf_u_cp_model(
            cp_model,
            A,
            t,
            q,
            gamma,
            v_cur,
            free_cols,
            use_hint=use_hint,
            name_prefix="l2_",
            max_over_cap=M_star,
        )
        model2.Minimize(1_000 * sum(over2) + sum(move2) + max_over2)
        st2, sol2 = _solve_cp_model(cp_model, model2, t_budgets[1], num_workers)
        if st2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            v_cur = _extract_v_from_solver(v_cur, v_vars2, sol2)
            phase_logs.append(
                {
                    "phase": 2,
                    "sum_over": int(sum(int(sol2.Value(o)) for o in over2)),
                    "status": _cp_status_name(st2),
                }
            )
        else:
            phase_logs.append({"phase": 2, "status": _cp_status_name(st2), "skipped": True})

        # Phase 3: minimize rows at peak M_star
        model3, v_vars3, over3, max_over3, move3, _ = _build_linf_u_cp_model(
            cp_model,
            A,
            t,
            q,
            gamma,
            v_cur,
            free_cols,
            use_hint=use_hint,
            name_prefix="l3_",
            max_over_cap=M_star,
        )
        peak_flags = []
        threshold = max(0, M_star)
        for idx, ov in enumerate(over3):
            if threshold > 0:
                peak = model3.NewBoolVar(f"l3_peak_{idx}")
                model3.Add(ov >= threshold).OnlyEnforceIf(peak)
                model3.Add(ov <= threshold - 1).OnlyEnforceIf(peak.Not())
                peak_flags.append(peak)
            else:
                z = model3.NewIntVar(0, 1, f"l3_z_{idx}")
                model3.Add(z == ov)
                peak_flags.append(z)
        model3.Minimize(10_000 * sum(peak_flags) + sum(over3) + sum(move3))
        st3, sol3 = _solve_cp_model(cp_model, model3, t_budgets[2], num_workers)
        if st3 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            v_cur = _extract_v_from_solver(v_cur, v_vars3, sol3)
            phase_logs.append(
                {
                    "phase": 3,
                    "peak_rows": int(sum(int(sol3.Value(p)) for p in peak_flags)),
                    "status": _cp_status_name(st3),
                }
            )
        else:
            phase_logs.append({"phase": 3, "status": _cp_status_name(st3), "skipped": True})

        meta["phase_logs"] = phase_logs
        meta["ilp_M_star"] = M_star
        meta["ilp_time_sec"] = time.perf_counter() - t0
        u_new = center_mod(t - (A @ v_cur), q)
        ok, verify = verify_solution(A, t, q, gamma, u_new, v_cur, False)
        meta["verify"] = verify
        meta["success"] = bool(ok)
        meta["ilp_max_over"] = max(0, int(np.max(np.abs(u_new))) - gamma)
        meta["ilp_ok"] = True
        meta["ilp_status_name"] = "LEX_DONE"
        if ok or accept_suboptimal:
            return u_new, v_cur, meta
        meta["ilp_error"] = "lex phases did not yield solution"
        return None, None, meta
    except Exception as exc:
        meta["ilp_error"] = f"{type(exc).__name__}: {exc}"
        meta["ilp_traceback"] = traceback.format_exc()
        meta["ilp_time_sec"] = time.perf_counter() - t0
        meta["phase_logs"] = phase_logs
        return None, None, meta


def run_ilp_finish(
    mode: str,
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    v0: np.ndarray,
    time_limit_sec: float,
    *,
    num_workers: int = 4,
    chunk_cols: int = 40,
    chunk_rounds: int = 10,
    chunk_stride: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """统一入口：``mode`` 为 ``full`` / ``chunk`` / ``lex``。"""
    mode = (mode or "full").strip().lower()
    if mode == "chunk":
        return cp_sat_chunked_v_linf_finish(
            A,
            t,
            q,
            gamma,
            v0,
            time_limit_sec=time_limit_sec,
            chunk_cols=chunk_cols,
            chunk_rounds=chunk_rounds,
            chunk_stride=chunk_stride,
            num_workers=num_workers,
        )
    if mode == "lex":
        return cp_sat_lex_v_linf_finish(
            A,
            t,
            q,
            gamma,
            v0,
            time_limit_sec=time_limit_sec,
            num_workers=num_workers,
        )
    return cp_sat_full_v_linf_finish(
        A,
        t,
        q,
        gamma,
        v0,
        time_limit_sec=time_limit_sec,
        num_workers=num_workers,
    )

# --- finish pipeline ---

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from sis_common import (
    class_label,
    effective_require_norm_lt_q2,
    problem_class_from_id,
    problem_class_from_instance,
)
from solve_sisinf import SearchConfig, apply_sis_class_defaults, center_mod, local_search_one, verify_solution


def load_instance(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data[0]
    return data


def load_incumbent(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def better_verify(
    a: Dict[str, int],
    b: Dict[str, int],
    *,
    require_norm_lt_q2: bool = False,
) -> bool:
    """未可行时比较进展；第三类在 L∞ 相同时优先 norm_req_ok（< q²）与更小 norm_sq。"""
    if not b:
        return True

    def key(v: Dict[str, int]) -> tuple:
        inf_max = max(v.get("inf_u", 999), v.get("inf_v", 999))
        base = (v.get("congruence_ok", 0), -inf_max)
        if require_norm_lt_q2:
            return base + (v.get("norm_req_ok", 0), -v.get("norm_sq", 0))
        return base + (-v.get("norm_sq", 0),)

    return key(a) > key(b)


def default_ilp_mode_for_class(sis_class: int) -> str:
    """按赛题类推荐 ILP 模式（基于题 1/3 实验：full/lex 有效，chunk 对平台题无效）。"""
    if sis_class == 3:
        return "lex"
    return "full"


def execute_finish(
    instance_path: str,
    incumbent_path: str,
    output_path: str,
    *,
    ilp_mode: Optional[str] = None,
    ilp_time_limit: float = 3600.0,
    ilp_workers: int = 4,
    ilp_chunk_cols: int = 40,
    ilp_chunk_rounds: int = 12,
    ilp_chunk_stride: int = 0,
    skip_ilp: bool = False,
    skip_sub_bkz: bool = True,
    sub_bkz_rows: int = 40,
    sub_bkz_cols: int = 40,
    sub_bkz_beta: int = 28,
    sub_bkz_seeds: int = 12,
    euclid_polish: Optional[bool] = None,
    ls_restarts: int = 8,
    ls_iters: int = 4000,
    seed: int = 424242,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    对单题 incumbent 跑完整收尾管线，写入 ``output_path`` 并返回报告 dict。
    """
    inst = load_instance(instance_path)
    inc = load_incumbent(incumbent_path)
    A = np.array(inst["A"], dtype=np.int64)
    t = np.array(inst["t"], dtype=np.int64)
    q, gamma = int(inst["q"]), int(inst["gamma"])
    pid = int(inst.get("id", inc.get("id", 0)))
    sis_class = problem_class_from_instance(inst)
    require_norm = effective_require_norm_lt_q2(inst, sis_class)
    mode = ilp_mode or default_ilp_mode_for_class(sis_class)
    do_polish = euclid_polish if euclid_polish is not None else (sis_class == 3)

    u0 = np.array(inc["u"], dtype=np.int64)
    v0 = np.array(inc["v"], dtype=np.int64)
    ok0, verify0 = verify_solution(A, t, q, gamma, u0, v0, require_norm)

    report: Dict[str, Any] = {
        "id": pid,
        "class": sis_class,
        "class_label": class_label(sis_class),
        "ilp_mode": mode,
        "incumbent_verify": verify0,
        "phases": [],
        "success": bool(ok0),
    }
    best_u, best_v = u0.copy(), v0.copy()
    best_verify = dict(verify0)
    t_all = time.time()

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    if not skip_ilp:
        stride = ilp_chunk_stride if ilp_chunk_stride > 0 else None
        log(
            f"[finish p{pid}] ILP mode={mode} limit={ilp_time_limit}s "
            f"class={sis_class} ..."
        )
        u_ilp, v_ilp, meta = run_ilp_finish(
            mode,
            A,
            t,
            q,
            gamma,
            best_v,
            ilp_time_limit,
            num_workers=ilp_workers,
            chunk_cols=ilp_chunk_cols,
            chunk_rounds=ilp_chunk_rounds,
            chunk_stride=stride,
        )
        phase_name = f"ilp_{mode}"
        phase: Dict[str, Any] = {"name": phase_name, "ok": False, "meta": meta}
        if u_ilp is not None and v_ilp is not None:
            ok_ilp, ver_ilp = verify_solution(A, t, q, gamma, u_ilp, v_ilp, require_norm)
            phase = {"name": phase_name, "ok": bool(ok_ilp), "verify": ver_ilp, "meta": meta}
            log(
                f"[finish p{pid}] ILP: ok={ok_ilp} inf_u={ver_ilp.get('inf_u')} inf_v={ver_ilp.get('inf_v')} "
                f"norm_ok={ver_ilp.get('norm_req_ok')} status={meta.get('ilp_status_name')} "
                f"time={meta.get('ilp_time_sec', 0):.1f}s"
            )
            if better_verify(ver_ilp, best_verify, require_norm_lt_q2=require_norm):
                best_u, best_v = u_ilp.copy(), v_ilp.copy()
                best_verify = ver_ilp
                report["success"] = bool(ok_ilp)
            if ok_ilp:
                report["phases"].append(phase)
                report["verify"] = best_verify
                report["u"] = best_u.tolist()
                report["v"] = best_v.tolist()
                report["elapsed_sec"] = time.time() - t_all
                os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                log(f"[finish p{pid}] feasible -> {output_path}")
                return report
        else:
            phase["error"] = meta.get("ilp_error", "unknown")
            log(f"[finish p{pid}] ILP failed: {phase['error']}")
        report["phases"].append(phase)

    # 第三类：L∞ 近可行但欧氏下界不足时，用高 euclid_weight 局部搜索抛光
    if do_polish and not report["success"] and require_norm:
        log(f"[finish p{pid}] euclid polish (class 3) ...")
        cfg = apply_sis_class_defaults(
            SearchConfig(
                restarts=max(4, ls_restarts),
                iters=ls_iters,
                seed=seed + pid,
                parallel_workers=1,
                timeout_sec=1200.0,
                euclid_weight=5.0,
                entropy_weight=0.5,
            ),
            sis_class,
        )
        u_p, v_p, meta_p = local_search_one(
            A, t, q, gamma, cfg, require_norm, prepend_v_seeds=[best_v]
        )
        ok_p, ver_p = verify_solution(A, t, q, gamma, u_p, v_p, require_norm)
        phase = {"name": "euclid_polish", "ok": bool(ok_p), "verify": ver_p, "meta": meta_p}
        log(
            f"[finish p{pid}] polish: ok={ok_p} inf_u={ver_p.get('inf_u')} norm_sq={ver_p.get('norm_sq')} "
            f"norm_ok={ver_p.get('norm_req_ok')}"
        )
        if better_verify(ver_p, best_verify, require_norm_lt_q2=require_norm):
            best_u, best_v = u_p.copy(), v_p.copy()
            best_verify = ver_p
            report["success"] = bool(ok_p)
        report["phases"].append(phase)
        if ok_p:
            report["verify"] = best_verify
            report["u"] = best_u.tolist()
            report["v"] = best_v.tolist()
            report["elapsed_sec"] = time.time() - t_all
            os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            return report

    # 仅第一类默认尝试 sub-BKZ（题 1/3 实验表明常变差；默认 skip_sub_bkz=True）
    if not skip_sub_bkz and sis_class == 1 and not report["success"]:
        log(f"[finish p{pid}] sub-BKZ + LS (class 1 only) ...")
        res_for_bkz = center_mod(t - (A @ best_v), q)
        seeds, sub_meta = collect_sub_bkz_v_seeds(
            A,
            q,
            gamma,
            res_for_bkz,
            sub_bkz_beta,
            n_rows=sub_bkz_rows,
            n_cols=sub_bkz_cols,
            max_vectors=sub_bkz_seeds,
            v_base=best_v,
            embed_mode="replace",
        )
        prepend = list(seeds) if seeds else [best_v]
        cfg = apply_sis_class_defaults(
            SearchConfig(
                restarts=max(1, ls_restarts),
                iters=ls_iters,
                seed=seed,
                parallel_workers=1,
                timeout_sec=900.0,
                use_bkz_seeds=False,
            ),
            sis_class,
        )
        u_ls, v_ls, meta_ls = local_search_one(
            A, t, q, gamma, cfg, require_norm, prepend_v_seeds=prepend
        )
        ok_ls, ver_ls = verify_solution(A, t, q, gamma, u_ls, v_ls, require_norm)
        phase = {
            "name": "sub_bkz_ls",
            "ok": bool(ok_ls),
            "verify": ver_ls,
            "meta": sub_meta,
            "ls_meta": meta_ls,
            "seed_count": len(prepend),
        }
        log(f"[finish p{pid}] sub-BKZ LS: inf_u={ver_ls.get('inf_u')} inf_v={ver_ls.get('inf_v')}")
        if better_verify(ver_ls, best_verify, require_norm_lt_q2=require_norm):
            best_u, best_v = u_ls.copy(), v_ls.copy()
            best_verify = ver_ls
            report["success"] = bool(ok_ls)
        report["phases"].append(phase)

    report["verify"] = best_verify
    report["u"] = best_u.tolist()
    report["v"] = best_v.tolist()
    report["elapsed_sec"] = time.time() - t_all
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log(
        f"[finish p{pid}] done success={report['success']} "
        f"inf_u={best_verify.get('inf_u')} inf_v={best_verify.get('inf_v')} "
        f"norm_ok={best_verify.get('norm_req_ok')}"
    )
    return report


def save_incumbent_from_record(rec: Dict[str, Any], path: str) -> None:
    """从 batch / finish 记录写出 ``{u,v,verify,id,round}`` incumbent 文件。"""
    payload = {
        "id": rec.get("id"),
        "round": rec.get("round"),
        "verify": rec.get("verify"),
        "u": rec.get("u"),
        "v": rec.get("v"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)