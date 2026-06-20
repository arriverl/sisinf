"""
SIS∞ 公共模块：题号分类、阶梯计分、--full-max 预设、模 q 核基。
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import numpy as np

if TYPE_CHECKING:
    from solve_sisinf import SearchConfig

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

# --- scoring ---

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

# --- full-max stack ---


def apply_full_max_stack(cfg: "SearchConfig", sis_class: int) -> "SearchConfig":
    """在 ``apply_sis_class_defaults`` 结果上叠加全量论文参数。"""
    common = {
        "restarts": max(cfg.restarts, 160),
        "iters": max(cfg.iters, 12000),
        "max_delta": max(cfg.max_delta, 15),
        "delta": max(cfg.delta, 4),
        "parallel_workers": max(cfg.parallel_workers, 8),
        "timeout_sec": max(cfg.timeout_sec or 0, 7200.0) if cfg.timeout_sec else 7200.0,
        "cheby_weight": max(cfg.cheby_weight, 64.0),
        "cp_repair_time_limit": max(cfg.cp_repair_time_limit, 8.0),
        "block_cp_time_limit": max(cfg.block_cp_time_limit, 12.0),
        "use_g6k_sieve": True,
        "g6k_sieve_alg": "bdgl2",
        "g6k_saturation_ratio": 0.95,
        "g6k_threads": max(cfg.g6k_threads, 16),
        "g6k_max_lift_vectors": max(cfg.g6k_max_lift_vectors, 2048),
        "bkz_max_dim": 260,
        "bkz_beta": max(cfg.bkz_beta, 56),
        "bkz_max_vectors": max(cfg.bkz_max_vectors, 128),
        "bkz_combo_depth": max(cfg.bkz_combo_depth, 8),
        "bkz_combo_coeff_max": 3,
    }

    if sis_class == 1:
        return replace(
            cfg,
            **common,
            use_bkz_seeds=True,
            use_sieve_seeds=True,
            use_restricted_svp_seeds=True,
            use_kannan_seeds=False,
            use_wagner_seeds=True,
            wagner_list_cap=max(cfg.wagner_list_cap, 2400),
            wang_enum_tail_rank=max(cfg.wang_enum_tail_rank, 48),
            wang_enum_pool_size=max(cfg.wang_enum_pool_size, 4096),
            wang_enum_coeff_max=4,
            wang_enum_max_trials=50000,
            kernel_max_basis=max(cfg.kernel_max_basis, 64),
            g6k_bkz_block=56,
        )

    if sis_class == 2:
        return replace(
            cfg,
            **common,
            use_bkz_seeds=True,
            use_sieve_seeds=False,
            use_kannan_seeds=True,
            use_restricted_svp_seeds=False,
            bkz_beta=max(cfg.bkz_beta, 52),
            cvp_lift_variants=max(cfg.cvp_lift_variants, 24),
            modular_pull_variants=max(cfg.modular_pull_variants, 20),
            kannan_embedding_factor=0,
            g6k_bkz_block=52,
        )

    # class 3
    return replace(
        cfg,
        **common,
        use_bkz_seeds=True,
        use_sieve_seeds=True,
        use_restricted_svp_seeds=True,
        use_kannan_seeds=False,
        restricted_svp_samples=max(cfg.restricted_svp_samples, 2000),
        wang_enum_tail_rank=max(cfg.wang_enum_tail_rank, 56),
        wang_enum_pool_size=max(cfg.wang_enum_pool_size, 8192),
        wang_enum_coeff_max=4,
        wang_enum_max_trials=80000,
        euclid_weight=max(cfg.euclid_weight, 8.0),
        entropy_weight=max(cfg.entropy_weight, 0.65),
        g6k_bkz_block=56,
    )


def full_max_finish_kwargs(sis_class: int) -> dict:
    """``execute_finish`` 全量 ILP 参数。"""
    return {
        "ilp_time_limit": 14400.0,
        "ilp_mode": "lex" if sis_class == 3 else "full",
        "euclid_polish": sis_class == 3,
    }


PAPER_STACK_TABLE = """
| 类 | 文献路线 | 本仓库模块 | full-max 关键参数 |
|----|----------|------------|-------------------|
| 一 | Chen BKZ2.0 + Becker BDGL + Wang L∞ | lattice_bkz, lattice_g6k, lattice_sieve, lattice_restricted_svp, Wagner | β≥56, g6k bdgl2 sat=0.95, wang pool=4096 |
| 二 | Kannan CVP + BKZ2.0 | lattice_kannan, lattice_bkz, CVP lift | Kannan+β52, CVP≥24 |
| 三 | Wang restricted SVP + BDGL list | lattice_restricted_svp, lattice_g6k, lex CP-SAT | wang pool=8192, g6k+slice, ILP 4h |
"""

# --- mod-q kernel ---

def _is_prime_trial(n: int) -> bool:
    """试除法判断素数（q 通常不大，足够用于分支选择）。"""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _sym_mod_coords(x: np.ndarray, q: int) -> np.ndarray:
    """对称取模到约 ``(-q/2, q/2]``，与 ``center_mod`` 一致。"""
    xi = np.asarray(x, dtype=np.int64).ravel()
    y = np.mod(xi, q).astype(np.int64)
    half = q // 2
    y = np.where(y > half, y - q, y)
    return y


def _kernel_via_augmented_nullspace(A: np.ndarray, q: int, max_basis: int) -> np.ndarray:
    """
    通过 ``[A | -q I]`` 的有理零空间提取核基（需安装 sympy）。

    Returns
    -------
    ndarray, shape (m, k)
        每列为一个 d；k <= max_basis。
    """
    try:
        import sympy as sp
    except ImportError:
        return np.zeros((A.shape[1], 0), dtype=np.int64)

    n, m = A.shape
    B = sp.Matrix(np.hstack([A.astype(np.int64), (-q * np.eye(n, dtype=np.int64))]))
    nsp: List = B.nullspace()
    if not nsp:
        return np.zeros((m, 0), dtype=np.int64)

    basis_cols: list[np.ndarray] = []
    seen: set[bytes] = set()

    for vec in nsp:
        if len(basis_cols) >= max_basis:
            break
        if vec.cols != 1:
            vec = vec.reshape(vec.rows, 1)
        # 有理向量通分为整数，再对 d 的前 m 分量 mod q
        L = 1
        for i in range(vec.rows):
            e = vec[i, 0]
            if e != 0:
                L = sp.ilcm(L, sp.denom(e))
        d_list: list[int] = []
        for j in range(m):
            z = sp.Integer(L * vec[j, 0])
            zq = int(z % q)
            half = q // 2
            if zq > half:
                zq -= q
            d_list.append(zq)
        d = np.asarray(d_list, dtype=np.int64)
        if np.all(d == 0):
            continue
        if not bool(np.all((A.astype(np.int64) @ d.astype(np.int64)) % q == 0)):
            continue
        key = d.tobytes()
        if key in seen:
            continue
        seen.add(key)
        basis_cols.append(d.copy())

    if not basis_cols:
        return np.zeros((m, 0), dtype=np.int64)
    return np.column_stack(basis_cols)


def _kernel_gauss_prime_field(A: np.ndarray, q: int, max_basis: int) -> np.ndarray:
    """
    素数模 q 上的行阶梯形 + 自由变量回代，得到标准核基。

    仅当 Z/qZ 为域时正确。
    """
    from math import gcd

    n, m = A.shape
    M = A.copy()
    pivot_col = np.full(n, -1, dtype=np.int64)
    rank = 0
    for col in range(m):
        if rank >= n:
            break
        piv = -1
        for r in range(rank, n):
            a = int(M[r, col]) % q
            if a != 0 and gcd(a, q) == 1:
                piv = r
                break
        if piv < 0:
            continue
        if piv != rank:
            M[[rank, piv]] = M[[piv, rank]]
        inv = pow(int(M[rank, col]) % q, -1, q)
        M[rank] = (M[rank] * inv) % q
        for r in range(n):
            if r != rank and int(M[r, col]) % q != 0:
                fac = int(M[r, col]) % q
                M[r] = (M[r] - fac * M[rank]) % q
        pivot_col[rank] = col
        rank += 1

    pivot_set = {int(pivot_col[r]) for r in range(rank) if pivot_col[r] >= 0}
    free_cols = [c for c in range(m) if c not in pivot_set]
    if not free_cols or rank == m:
        return np.zeros((m, 0), dtype=np.int64)

    for r in range(rank - 1, -1, -1):
        pc = int(pivot_col[r])
        if pc < 0:
            continue
        for up in range(r):
            if int(M[up, pc]) % q != 0:
                fac = int(M[up, pc]) % q
                M[up] = (M[up] - fac * M[r]) % q

    basis_cols: list[np.ndarray] = []
    for fc in free_cols:
        if len(basis_cols) >= max_basis:
            break
        d = np.zeros(m, dtype=np.int64)
        d[fc] = 1
        for r in range(rank - 1, -1, -1):
            pc = int(pivot_col[r])
            if pc < 0:
                continue
            s = 0
            for j in range(m):
                if j == pc:
                    continue
                s = (s + int(M[r, j]) * int(d[j])) % q
            d[pc] = (-s) % q
        if np.all((A @ d) % q == 0):
            basis_cols.append(d.copy())
    if not basis_cols:
        return np.zeros((m, 0), dtype=np.int64)
    return np.column_stack(basis_cols)


def right_kernel_basis_mod_q(A: np.ndarray, q: int, max_basis: int = 32) -> np.ndarray:
    """
    返回核基矩阵 K，形状 ``(m, k)``，``k <= max_basis``。

    每列 d 满足 ``(A @ d) % q == 0``。
    合数 q 用 SymPy 增广矩阵；素数 q 用域上高斯消元。
    """
    A = (np.asarray(A, dtype=np.int64) % q + q) % q
    n, m = A.shape
    if max_basis <= 0 or m == 0:
        return np.zeros((m, 0), dtype=np.int64)

    if _is_prime_trial(q):
        return _kernel_gauss_prime_field(A, q, max_basis)

    K = _kernel_via_augmented_nullspace(A, q, max_basis)
    if K.shape[1] > 0:
        return K
    return np.zeros((m, 0), dtype=np.int64)


def in_kernel_mod_q(A: np.ndarray, d: np.ndarray, q: int) -> bool:
    """快速校验 d 是否在 A 的模 q 右核中。"""
    r = (A @ d.astype(np.int64)) % q
    return bool(np.all(r == 0))