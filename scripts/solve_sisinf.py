"""
SIS∞ 2026 赛题一 — 主求解器（多 restart 局部搜索 + 格种子 + CP-SAT）。

问题
----
求 ``u ∈ Z^n``, ``v ∈ Z^m`` 使得 ``A v + u ≡ t (mod q)``，
且 ``|u|_∞, |v|_∞ ≤ γ``。第三类题另需 ``||u||_2^2 + ||v||_2^2 < q^2``（官方计分）；
齐次 ``t≡0`` 时拒绝平凡解 ``u=v=0``。

核心变量
--------
- ``residual = center_mod(t - A v, q)`` 即同余意义下的 ``u``；
- 内层主要优化 ``v``，``u`` 由 residual 导出。

主要模块（均在 ``solve_sisinf.py`` 内，按 `# =====` 分段）
--------
- 题号分类 / 阶梯计分 / full-max 预设 / mod-q 核基
- ``SearchConfig`` / ``apply_sis_class_defaults``：搜索与三类题默认参数
- 格种子：BKZ / G6K / sieve / Kannan / Wang
- u 优先启发式：Wagner、ViolationLS、分层投影、高斯种子
- ``local_search_one`` / ``verify_solution``：多 restart 局部搜索与校验
- CP-SAT 收尾：``execute_finish`` / ``run_ilp_finish``

命令行入口见 ``sis_cli.py``。
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

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


def apply_full_max_stack(cfg: SearchConfig, sis_class: int) -> "SearchConfig":
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
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

@dataclass
class SearchConfig:
    """
    局部搜索超参数。字段默认值偏通用；三类题由 ``apply_sis_class_defaults`` 覆盖。

    调参提示：第一类加大 ``bkz_*`` / ``kernel_*``；第二类加大 ``cvp_lift_*`` /
    ``modular_pull_*`` 并关 BKZ；第三类加大 ``euclid_weight`` / ``entropy_weight``。
    """

    # --- 基础循环 ---
    restarts: int = 40  # 独立初值 restart 次数
    iters: int = 2500  # 每个 restart 内最大迭代步数
    delta: int = 2  # 单坐标扰动半径（±delta）
    kick_size: int = 6  # 停滞 kick 时随机翻转坐标个数
    kick_every: int = 120  # 每多少步尝试一次 kick；0 关闭
    seed: int = 2026  # 全局 RNG 种子
    max_delta: int = 6  # 自适应步长上界
    candidate_count: int = 24  # dual-space 保留的 v 种子数
    parallel_workers: int = 1  # restart 并行进程数（Windows 建议 1）
    timeout_sec: Optional[float] = None  # 单 restart 墙钟超时（秒），None 不限
    verbose: bool = False
    log_every: int = 500

    # --- 目标与能量（Chebyshev 优先 + 欧氏/熵次要）---
    entropy_weight: float = 0.25  # 分布熵奖励权重（越大越鼓励“铺开”坐标）
    euclid_weight: float = 1.5  # 欧氏下界缺口惩罚（第三类关键）
    overflow_weight: float = 1.0  # L∞ 溢出量和权重
    entropy_bins: int = 8  # 熵直方图分箱
    dynamic_schedule: bool = True  # 是否随 progress 缩放 euclid/entropy 权重
    entropy_update_interval: int = 50  # 熵计算间隔（步），降低直方图开销
    entropy_disable_after_progress: float = 0.78  # progress 超过此值停算熵；>=1 永不关
    cheby_weight: float = 20.0  # 最大坐标溢出（Chebyshev）在能量中的权重
    cheby_boost_threshold: int = 20
    cheby_boost_factor: float = 2.0  # max_overflow 超阈值时放大 cheby_weight
    energy_topk: int = 5  # 仅对 u 的前 k 大溢出求和惩罚；0 关
    energy_topk_weight: float = 0.12

    # --- 种子构造 ---
    use_dual_space: bool = True  # 是否走完整 dual 候选管线
    modular_pull_variants: int = 4  # 模拉回启发式种子数；0 关
    cvp_lift_variants: int = 6  # 非齐次 CVP 提升种子；齐次题常置 0
    use_bkz_seeds: bool = True
    bkz_beta: int = 0  # BKZ 块大小；0 禁用
    bkz_max_vectors: int = 24
    bkz_max_dim: int = 96  # n+m 超过则跳过 BKZ
    bkz_combo_depth: int = 0  # 约化基前 k 列小整数组合（类 1）
    bkz_combo_coeff_max: int = 2
    use_sieve_seeds: bool = False  # BKZ 后缀 list sieve（类 1/3）
    use_kannan_seeds: bool = False  # Kannan 嵌入 CVP 种子（类 2）
    use_restricted_svp_seeds: bool = False  # Wang 受限 SVP enumerate-then-slice（类 1/3）
    restricted_svp_samples: int = 400
    wang_enum_tail_rank: int = 28  # dimension-for-free 尾块秩
    wang_enum_pool_size: int = 512
    wang_enum_coeff_max: int = 3
    wang_enum_max_trials: int = 8000
    kannan_embedding_factor: int = 0  # 0 = 自动
    # G6K 真筛法（Becker BDGL via fplll/g6k）
    use_g6k_sieve: bool = False
    g6k_sieve_alg: str = "bdgl2"
    g6k_saturation_ratio: float = 0.92
    g6k_threads: int = 8
    g6k_bkz_block: int = 0  # 0 = 跟随 bkz_beta
    g6k_max_lift_vectors: int = 512

    # --- 邻域算子 ---
    pair_relief_every: int = 32  # 双坐标联合移动周期；0 关
    pair_relief_attempts: int = 12
    pair_relief_radius: int = 2
    use_pull_kick: bool = True  # 沿 A^T sign(residual) 的梯度踢
    pull_kick_gain: float = 1.25
    kernel_walk_every: int = 25  # 模 q 核游走周期；0 关
    kernel_coeff_max: int = 2  # 核基线性组合系数界
    kernel_max_basis: int = 24  # 预计算核列数上界
    ls_project_every: int = 35  # 最差行/列最小二乘投影；0 关
    ls_top_rows: int = 14
    ls_top_cols: int = 28

    # --- CP-SAT（需 ortools，缺失则静默跳过）---
    cp_repair_threshold: int = 8  # 违规数 ≤ 此值时触发「近可行」精确 CP 修复
    cp_repair_window: int = 3
    cp_repair_time_limit: float = 0.5
    cp_aggressive_every: int = 0  # >0：远可行时也周期 CP（针对 u 大量溢出）；0 关闭
    cp_aggressive_row_k: int = 20
    u_row_snap_every: int = 14  # 对最差 u 行做定向坐标枚举；0 关闭
    u_row_snap_top_rows: int = 10
    u_row_snap_cols: int = 16
    # 文献驱动 u 优先扩展（见 sis_advanced_u_ops.py）
    use_wagner_seeds: bool = False
    wagner_rows: int = 8
    wagner_cols: int = 16
    wagner_box_radius: int = 2
    wagner_list_cap: int = 600
    use_violation_ls: bool = False
    violation_ls_every: int = 12
    violation_ls_top_rows: int = 6
    violation_ls_top_cols: int = 8
    use_layered_ls: bool = False
    layered_ls_every: int = 40
    gaussian_seed_sigma: float = 3.0
    gaussian_on_stagnation: bool = False
    cheap_lll_trials: int = 0
    cp_periodic_every: int = 0  # 随机列子集周期 CP；0 关
    cp_periodic_cols: int = 16
    block_cp_every: int = 0  # 块 CP 优化周期；0 关
    block_cp_rows: int = 20
    block_cp_cols: int = 28
    block_cp_window: int = 6
    block_cp_time_limit: float = 2.0

    # --- 三阶段进度调度（progress ∈ [0,1]）---
    residual_phase_end: float = 0.45  # 此前侧重同余/L∞ 可行
    kernel_phase_start: float = 0.60  # 此后加强 kernel / 大邻域
    allow_uphill_sa: bool = False  # 是否允许劣化 score_key 的模拟退火步


def center_mod(x: np.ndarray, q: int) -> np.ndarray:
    """对称取模到约 ``(-q/2, q/2]``，得到代表元（即算法中的 u / residual）。"""
    y = np.mod(x, q)
    half = q // 2
    y = np.where(y > half, y - q, y)
    return y.astype(np.int64, copy=False)


def objective_uv(residual: np.ndarray, v: np.ndarray, gamma: int) -> Tuple[int, int, int]:
    """
    L∞ 可行性的离散代理（与 ``verify_solution`` 的 inf 检查一致）。

    Returns
    -------
    violations, overflow_sum, max_overflow
        分别：超界坐标个数、溢出量总和、单坐标最大溢出。
    """
    abs_r = np.abs(residual)
    abs_v = np.abs(v)
    ou = np.maximum(abs_r - gamma, 0)
    ov = np.maximum(abs_v - gamma, 0)
    violations = int(np.count_nonzero(ou) + np.count_nonzero(ov))
    overflow_sum = int(np.sum(ou) + np.sum(ov))
    max_u = int(np.max(ou)) if ou.size else 0
    max_v = int(np.max(ov)) if ov.size else 0
    max_overflow = max(max_u, max_v)
    return violations, overflow_sum, max_overflow


def objective_uv_and_rr_sq(residual: np.ndarray, v: np.ndarray, gamma: int) -> Tuple[int, int, int, float]:
    """Single pass: combined L_inf violations + ||u||_2^2 for energy."""
    viol, overflow_sum, max_overflow = objective_uv(residual, v, gamma)
    rr_sq = float(np.dot(residual.astype(np.float64), residual.astype(np.float64)))
    return viol, overflow_sum, max_overflow, rr_sq


def apply_sis_class_defaults(
    cfg: SearchConfig, sis_class: int, *, aggressive: bool = False, full_max: bool = False
) -> SearchConfig:
    """
    按赛题类别 1/2/3 叠加默认搜索参数（见 ``sis_problem_taxonomy.py``）。

    aggressive=True 时进一步增大 BKZ β、CVP 变体、块 CP 时间等（长跑轮次用）。
    full_max=True 时在 aggressive 基础上套用 ``sis_full_stack.apply_full_max_stack``（论文全量拉满）。
    """
    if sis_class == 1:
        beta = 32 if aggressive else 28
        out = SearchConfig(
            **{
                **asdict(cfg),
                "use_bkz_seeds": True,
                "bkz_beta": beta,
                "bkz_max_vectors": 32 if aggressive else 24,
                "bkz_max_dim": 220 if aggressive else 200,
                "bkz_combo_depth": 6 if aggressive else 5,
                "bkz_combo_coeff_max": 2,
                "cvp_lift_variants": 0,
                "modular_pull_variants": max(2, cfg.modular_pull_variants),
                "kernel_walk_every": cfg.kernel_walk_every if cfg.kernel_walk_every > 0 else 20,
                "kernel_max_basis": max(cfg.kernel_max_basis, 32),
                "euclid_weight": 0.5,
                "entropy_weight": 0.0,
                "cheby_weight": max(cfg.cheby_weight, 48.0),
                "energy_topk": max(cfg.energy_topk, 10),
                "energy_topk_weight": max(cfg.energy_topk_weight, 0.22),
                "max_delta": max(cfg.max_delta, 12),
                "delta": max(cfg.delta, 3),
                "modular_pull_variants": max(cfg.modular_pull_variants, 8),
                "pull_kick_gain": max(cfg.pull_kick_gain, 2.2),
                "cp_aggressive_every": cfg.cp_aggressive_every if cfg.cp_aggressive_every > 0 else (28 if aggressive else 36),
                "cp_aggressive_row_k": max(cfg.cp_aggressive_row_k, 40 if aggressive else 32),
                "cp_repair_window": max(cfg.cp_repair_window, 8 if aggressive else 6),
                "cp_repair_time_limit": max(cfg.cp_repair_time_limit, 3.5 if aggressive else 2.5),
                "block_cp_every": cfg.block_cp_every if cfg.block_cp_every > 0 else 40,
                "block_cp_rows": max(cfg.block_cp_rows, 32),
                "block_cp_cols": max(cfg.block_cp_cols, 40),
                "block_cp_window": max(cfg.block_cp_window, 8),
                "block_cp_time_limit": max(cfg.block_cp_time_limit, 3.5),
                "u_row_snap_every": cfg.u_row_snap_every if cfg.u_row_snap_every > 0 else (6 if aggressive else 8),
                "u_row_snap_top_rows": max(cfg.u_row_snap_top_rows, 20 if aggressive else 16),
                "u_row_snap_cols": max(cfg.u_row_snap_cols, 32 if aggressive else 24),
                "use_wagner_seeds": True,
                "wagner_rows": max(cfg.wagner_rows, 10 if aggressive else 8),
                "wagner_cols": max(cfg.wagner_cols, 20 if aggressive else 16),
                "wagner_box_radius": max(cfg.wagner_box_radius, 3 if aggressive else 2),
                "wagner_list_cap": max(cfg.wagner_list_cap, 800 if aggressive else 600),
                "use_violation_ls": True,
                "violation_ls_every": cfg.violation_ls_every if cfg.violation_ls_every > 0 else (8 if aggressive else 10),
                "use_layered_ls": True,
                "layered_ls_every": cfg.layered_ls_every if cfg.layered_ls_every > 0 else (32 if aggressive else 40),
                "gaussian_on_stagnation": True,
                "gaussian_seed_sigma": max(cfg.gaussian_seed_sigma, 4.0 if aggressive else 3.0),
                "cheap_lll_trials": max(cfg.cheap_lll_trials, 50 if aggressive else 30),
                "residual_phase_end": 0.58,
                "kernel_phase_start": 0.78,
                "use_sieve_seeds": True,
                "use_kannan_seeds": False,
                "use_restricted_svp_seeds": True,
                "wang_enum_tail_rank": 32 if aggressive else 28,
                "wang_enum_pool_size": 640 if aggressive else 512,
                "wang_enum_max_trials": 12000 if aggressive else 8000,
            }
        )
    elif sis_class == 2:
        cvp = 14 if aggressive else 10
        pull = 10 if aggressive else 8
        kb = 28 if aggressive else 24
        out = SearchConfig(
            **{
                **asdict(cfg),
                "use_bkz_seeds": False,
                "bkz_beta": kb,
                "bkz_combo_depth": 0,
                "cvp_lift_variants": max(cvp, cfg.cvp_lift_variants),
                "modular_pull_variants": max(pull, cfg.modular_pull_variants),
                "use_pull_kick": True,
                "pull_kick_gain": max(1.5, cfg.pull_kick_gain),
                "euclid_weight": 0.6,
                "entropy_weight": min(cfg.entropy_weight, 0.2) if cfg.entropy_weight > 0 else 0.0,
                "residual_phase_end": 0.55,
                "kernel_phase_start": 0.70,
                "cp_periodic_every": cfg.cp_periodic_every if cfg.cp_periodic_every > 0 else 80,
                "block_cp_every": cfg.block_cp_every if cfg.block_cp_every > 0 else 55,
                "use_sieve_seeds": False,
                "use_kannan_seeds": True,
                "use_restricted_svp_seeds": False,
                "bkz_max_dim": 200,
                "bkz_max_vectors": 24,
            }
        )
    else:
        euclid = 4.0 if aggressive else 3.0
        out = SearchConfig(
        **{
            **asdict(cfg),
            "use_bkz_seeds": True,
            "bkz_beta": 28 if aggressive else 24,
            "bkz_max_vectors": 32 if aggressive else 24,
            "bkz_max_dim": 160,
            "bkz_combo_depth": 5 if aggressive else 4,
            "bkz_combo_coeff_max": 2,
            "cvp_lift_variants": 0,
            "euclid_weight": max(euclid, cfg.euclid_weight),
            "entropy_weight": max(0.45, cfg.entropy_weight) if cfg.entropy_weight > 0 else 0.45,
            "entropy_disable_after_progress": 0.55,
            "kernel_walk_every": cfg.kernel_walk_every if cfg.kernel_walk_every > 0 else 22,
            "kernel_max_basis": max(cfg.kernel_max_basis, 36),
            "residual_phase_end": 0.30,
            "kernel_phase_start": 0.45,
            "block_cp_every": cfg.block_cp_every if cfg.block_cp_every > 0 else 50,
            "block_cp_time_limit": max(3.0, cfg.block_cp_time_limit),
            "use_sieve_seeds": True,
            "use_kannan_seeds": False,
            "use_restricted_svp_seeds": True,
            "restricted_svp_samples": 600 if aggressive else 400,
            "wang_enum_tail_rank": 36 if aggressive else 28,
            "wang_enum_pool_size": 768 if aggressive else 512,
            "wang_enum_coeff_max": 3,
            "wang_enum_max_trials": 16000 if aggressive else 10000,
        }
    )

    if full_max:
        out = apply_full_max_stack(
            apply_sis_class_defaults(cfg, sis_class, aggressive=True, full_max=False),
            sis_class,
        )
    return out


def score_key(score: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Chebyshev-first ordering: max_overflow -> violations -> overflow_sum."""
    viol, overflow_sum, max_overflow = score
    return max_overflow, viol, overflow_sum


def better_score(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> bool:
    return score_key(a) < score_key(b)


def same_score_key(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> bool:
    """Tie under Chebyshev-first ordering (for secondary energy tie-break)."""
    return score_key(a) == score_key(b)


def topk_u_overflow_penalty(residual: np.ndarray, gamma: int, k: int) -> float:
    """Sum of k largest per-coordinate u-overflows (fast surrogate for order-statistics)."""
    if k <= 0:
        return 0.0
    ou = np.maximum(np.abs(residual.astype(np.float64)) - float(gamma), 0.0)
    if ou.size <= k:
        return float(np.sum(ou))
    idx = np.argpartition(-ou, k - 1)[:k]
    return float(np.sum(ou[idx]))


def objective_uv_rr_sq_temp_vj(
    residual_cand: np.ndarray,
    v: np.ndarray,
    j: int,
    new_vj: int,
    gamma: int,
) -> Tuple[int, int, int, float]:
    """Objective on candidate (residual_cand, v with v[j]=new_vj); restores v[j] (no v.copy)."""
    old = int(v[j])
    v[j] = new_vj
    try:
        return objective_uv_and_rr_sq(residual_cand, v, gamma)
    finally:
        v[j] = old


def entropy_of_abs(x: np.ndarray, gamma: int, bins: int) -> float:
    abs_x = np.abs(x).astype(np.float64)
    clipped = np.clip(abs_x, 0, gamma)
    hist, _ = np.histogram(clipped, bins=bins, range=(0, gamma + 1e-9))
    total = np.sum(hist)
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def entropy_residual_plus_v(
    residual: np.ndarray,
    v: np.ndarray,
    gamma: int,
    bins: int,
) -> float:
    return entropy_of_abs(residual, gamma, bins) + entropy_of_abs(v, gamma, bins)


def entropy_residual_plus_v_temp_vj(
    residual_cand: np.ndarray,
    v: np.ndarray,
    j: int,
    new_vj: int,
    gamma: int,
    bins: int,
) -> float:
    """Histogram entropy over full v with v[j] overridden; restores v[j] (no vector copy)."""
    old = int(v[j])
    v[j] = new_vj
    try:
        return entropy_residual_plus_v(residual_cand, v, gamma, bins)
    finally:
        v[j] = old


def _schedule_weights(cfg: SearchConfig, progress: float) -> Tuple[float, float]:
    euclid_w = cfg.euclid_weight
    entropy_w = cfg.entropy_weight
    if cfg.dynamic_schedule:
        euclid_w = cfg.euclid_weight * (0.25 + 1.75 * progress)
        entropy_w = cfg.entropy_weight * (1.2 - 0.7 * progress)
    return euclid_w, entropy_w


def energy_from_parts(
    viol: int,
    overflow_sum: int,
    max_overflow: int,
    norm_sq: float,
    entropy: float,
    q: int,
    require_norm_lt_q2: bool,
    cfg: SearchConfig,
    progress: float,
    topk_u_pen: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """Energy without recomputing objective (viol/overflow already known)."""
    # 第三类官方：须 norm_sq < q²；超出则惩罚
    euclid_excess = max(0.0, norm_sq - float(q * q) + 1.0) if require_norm_lt_q2 else 0.0
    euclid_w, entropy_w = _schedule_weights(cfg, progress)
    cheby_w = cfg.cheby_weight
    if max_overflow >= cfg.cheby_boost_threshold:
        cheby_w *= cfg.cheby_boost_factor
    energy = (
        1_000_000.0 * viol
        + cheby_w * max_overflow
        + cfg.overflow_weight * overflow_sum
        + cfg.energy_topk_weight * topk_u_pen
        + euclid_w * euclid_excess
        - entropy_w * entropy
    )
    return energy, {
        "violations": float(viol),
        "overflow_sum": float(overflow_sum),
        "max_overflow": float(max_overflow),
        "norm_sq": norm_sq,
        "euclid_excess": euclid_excess,
        "cheby_weight": cheby_w,
        "topk_u_pen": topk_u_pen,
        "entropy": entropy,
        "energy": energy,
    }


def energy_score(
    residual: np.ndarray,
    v: np.ndarray,
    gamma: int,
    q: int,
    require_norm_lt_q2: bool,
    cfg: SearchConfig,
    progress: float,
    *,
    compute_entropy: bool = True,
) -> Tuple[float, Dict[str, float]]:
    viol, overflow_sum, max_overflow, rr_sq = objective_uv_and_rr_sq(residual, v, gamma)
    vv_sq = float(np.dot(v.astype(np.float64), v.astype(np.float64)))
    norm_sq = rr_sq + vv_sq
    if compute_entropy and cfg.entropy_weight != 0.0:
        entropy = entropy_of_abs(residual, gamma, cfg.entropy_bins) + entropy_of_abs(v, gamma, cfg.entropy_bins)
    else:
        entropy = 0.0
    tk = topk_u_overflow_penalty(residual, gamma, cfg.energy_topk) if cfg.energy_topk > 0 else 0.0
    return energy_from_parts(
        viol,
        overflow_sum,
        max_overflow,
        norm_sq,
        entropy,
        q,
        require_norm_lt_q2,
        cfg,
        progress,
        topk_u_pen=tk,
    )


def should_compute_entropy(
    cfg: SearchConfig,
    step: int,
    viol: int,
    require_norm_lt_q2: bool,
    progress: float,
) -> bool:
    if cfg.entropy_disable_after_progress < 1.0 and progress >= cfg.entropy_disable_after_progress:
        return False
    if cfg.entropy_weight == 0.0:
        return False
    if cfg.entropy_update_interval <= 0:
        return False
    # Respect interval even when viol==0 (otherwise inner loops compute histograms every delta).
    if require_norm_lt_q2 and viol == 0:
        # Euclidean phase: slightly denser entropy than coarse phase (still capped).
        denser = max(1, cfg.entropy_update_interval // 4)
        return step % denser == 0
    return step % cfg.entropy_update_interval == 0


def verify_solution(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    u: np.ndarray,
    v: np.ndarray,
    require_norm_lt_q2: bool = False,
) -> Tuple[bool, Dict[str, int]]:
    """
    官方一致性校验：同余、L∞ 盒、第三类欧氏上界 ``norm_sq < q^2``、齐次非平凡。
    """
    lhs = (A @ v + u - t) % q
    congr_ok = bool(np.all(lhs == 0))
    inf_u = int(np.max(np.abs(u)))
    inf_v = int(np.max(np.abs(v)))
    inf_ok = inf_u <= gamma and inf_v <= gamma
    norm_sq = int(np.dot(u, u) + np.dot(v, v))
    if require_norm_lt_q2:
        norm_ok = norm_sq < q * q
    else:
        norm_ok = True
    is_homogeneous = bool(np.all(t % q == 0))
    nontrivial_ok = True if not is_homogeneous else bool(np.any(u != 0) or np.any(v != 0))
    ok = congr_ok and inf_ok and norm_ok and nontrivial_ok
    return ok, {
        "ok": int(ok),
        "congruence_ok": int(congr_ok),
        "inf_u": inf_u,
        "inf_v": inf_v,
        "norm_sq": norm_sq,
        "norm_req_ok": int(norm_ok),
        "nontrivial_ok": int(nontrivial_ok),
    }


def build_dual_space_candidates(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    cfg: SearchConfig,
    rng: np.random.Generator,
    prepend: Optional[List[np.ndarray]] = None,
) -> Tuple[List[np.ndarray], Dict[str, int]]:
    """
    构造并排序 v 初值列表：prepend(BKZ) → modular_pull →（非齐次）cvp_lift → 投影/稀疏/随机。

    齐次题（t≡0）不生成 CVP lift。按 score_key 保留前 ``candidate_count`` 个。
    """
    _, m = A.shape
    candidates: List[np.ndarray] = []
    if prepend:
        for pv in prepend:
            pv = np.asarray(pv, dtype=np.int64).ravel()
            if pv.size != m:
                continue
            candidates.append(pv.copy())
    for pv in modular_pull_seed_vectors(A, t, q, gamma, rng, cfg.modular_pull_variants):
        candidates.append(pv.copy())
    homogeneous = bool(np.all(np.mod(t, q) == 0))
    if not homogeneous and cfg.cvp_lift_variants > 0:
        for pv in cvp_lift_seed_vectors(A, t, q, gamma, rng, cfg.cvp_lift_variants):
            candidates.append(pv.copy())
    if not homogeneous:
        candidates.append(np.zeros(m, dtype=np.int64))

    # Heuristic "lattice-space-like" projection seeds.
    probe_count = max(4, cfg.candidate_count // 3)
    for _ in range(probe_count):
        w = rng.integers(-2, 3, size=A.shape[0], dtype=np.int64)
        proj = A.T @ w
        if np.max(np.abs(proj)) == 0:
            continue
        v = np.rint(-gamma * proj / max(1, np.max(np.abs(proj)))).astype(np.int64)
        v = np.clip(v, -gamma, gamma)
        candidates.append(v)

    # Sparse extreme candidates.
    for _ in range(max(4, cfg.candidate_count // 3)):
        v = np.zeros(m, dtype=np.int64)
        idx = rng.choice(m, size=max(1, m // 12), replace=False)
        signs = rng.choice(np.array([-gamma, gamma], dtype=np.int64), size=idx.shape[0], replace=True)
        v[idx] = signs
        candidates.append(v)

    if homogeneous and cfg.gaussian_seed_sigma > 0:
        try:
            for gv in discrete_gaussian_seeds(
                np.zeros(m, dtype=np.int64),
                gamma,
                rng,
                n_seeds=max(4, cfg.candidate_count // 4),
                sigma=cfg.gaussian_seed_sigma,
            ):
                candidates.append(gv)
        except Exception:
            pass

    # Random box candidates.
    while len(candidates) < cfg.candidate_count:
        v = rng.integers(low=-gamma, high=gamma + 1, size=m, dtype=np.int64)
        candidates.append(v)

    # De-duplicate by bytes signature.
    dedup: Dict[bytes, np.ndarray] = {}
    for v in candidates:
        dedup[v.tobytes()] = v
    candidates = list(dedup.values())

    # Score by residual feasibility proxy.
    scored = []
    for v in candidates:
        if homogeneous and np.all(v == 0):
            continue
        residual = center_mod(t - (A @ v), q)
        viol, overflow_sum, max_overflow = objective_uv(residual, v, gamma)
        scored.append((viol, overflow_sum, max_overflow, v))
    scored.sort(key=lambda x: score_key((x[0], x[1], x[2])))
    candidates = [x[3] for x in scored[: cfg.candidate_count]]

    return candidates, {"num_candidates": len(candidates)}


def modular_pull_seed_vectors(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    rng: np.random.Generator,
    variants: int,
) -> List[np.ndarray]:
    """
    模拉回启发式：``v ≈ -γ · normalize(A^T φ(center(t)))``，φ 为多种残差形状。

    不改变同余类，仅提供 L∞ 盒内的 v 起点（第一、二类均可用）。
    """
    if variants <= 0:
        return []
    _, m = A.shape
    c = center_mod(t, q).astype(np.float64)
    seen: set = set()
    out: List[np.ndarray] = []
    modes = [
        c,
        np.sign(c),
        np.where(np.abs(c) < 1e-9, 0.0, c / np.maximum(1.0, np.abs(c))),
    ]
    for idx in range(variants):
        if idx < len(modes):
            phi = modes[idx]
        else:
            phi = rng.standard_normal(A.shape[0])
        g = (A.T @ phi).astype(np.float64)
        mx = float(np.max(np.abs(g)))
        if mx < 1e-15:
            continue
        seed_vec = np.rint(-gamma * g / mx).astype(np.int64)
        seed_vec = np.clip(seed_vec, -gamma, gamma)
        key = seed_vec.tobytes()
        if key not in seen:
            seen.add(key)
            out.append(seed_vec.copy())
    return out


def cvp_lift_seed_vectors(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    rng: np.random.Generator,
    variants: int,
) -> List[np.ndarray]:
    """
    非齐次 SIS 的 CVP 视角：在若干提升 ``t + q·k`` 上对 ``A v ≈ target`` 做最小二乘，再裁剪到盒内。

    仅用于 ``t ≢ 0``；齐次题应在 ``build_dual_space_candidates`` 中跳过。
    """
    if variants <= 0:
        return []
    n, m = A.shape
    tc = center_mod(t, q).astype(np.int64)
    seen: set[bytes] = set()
    out: List[np.ndarray] = []

    # Deterministic structured lift directions for k.
    k_base = [
        np.zeros(n, dtype=np.int64),
        np.sign(tc).astype(np.int64),
        -np.sign(tc).astype(np.int64),
        np.where(tc > gamma, 1, np.where(tc < -gamma, -1, 0)).astype(np.int64),
    ]
    # A few randomized sparse lift vectors.
    extra = max(0, variants - len(k_base))
    for _ in range(extra):
        kk = np.zeros(n, dtype=np.int64)
        s = max(1, n // 10)
        idx = rng.choice(n, size=s, replace=False)
        kk[idx] = rng.choice(np.array([-1, 1], dtype=np.int64), size=s, replace=True)
        k_base.append(kk)

    # Solve A v ≈ t + q*k in R, then clip to box.
    A_f = A.astype(np.float64)
    for kk in k_base[:variants]:
        y = (tc + q * kk).astype(np.float64)
        try:
            v_real, *_ = np.linalg.lstsq(A_f, y, rcond=1e-8)
        except Exception:
            continue
        for scale in (1.0, 0.75, 0.5):
            v = np.rint(v_real * scale).astype(np.int64)
            v = np.clip(v, -gamma, gamma)
            key = v.tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append(v.copy())

    return out


def pick_best_pair_move(
    residual: np.ndarray,
    v: np.ndarray,
    cols: List[np.ndarray],
    q: int,
    gamma: int,
    j: int,
    k: int,
    vv_sq: int,
    score: Tuple[int, int, int],
    radius: int,
) -> Optional[Tuple[np.ndarray, int, int, Tuple[int, int, int], int, float]]:
    """Joint (-radius..radius)^2 move on (j,k); returns (new_res,nj,nk,new_score,new_vv_sq,rr_sq)."""
    cj, ck = int(v[j]), int(v[k])
    col_j, col_k = cols[j], cols[k]
    best_score = score
    best_pack: Optional[Tuple[np.ndarray, int, int, Tuple[int, int, int], int, float]] = None
    for dj in range(-radius, radius + 1):
        for dk in range(-radius, radius + 1):
            if dj == 0 and dk == 0:
                continue
            nj, nk = cj + dj, ck + dk
            if abs(nj) > gamma or abs(nk) > gamma:
                continue
            cand_res = center_mod(residual - col_j * dj - col_k * dk, q)
            v[j], v[k] = nj, nk
            try:
                cv, cosum, cmaxov, rr_c = objective_uv_and_rr_sq(cand_res, v, gamma)
            finally:
                v[j], v[k] = cj, ck
            cand_score = (cv, cosum, cmaxov)
            if better_score(cand_score, best_score):
                best_score = cand_score
                new_vv = vv_sq - cj * cj - ck * ck + nj * nj + nk * nk
                best_pack = (cand_res.copy(), nj, nk, cand_score, new_vv, rr_c)
    return best_pack


def u_priority_coord_order(
    A: np.ndarray,
    residual: np.ndarray,
    gamma: int,
    bad_v: np.ndarray,
    worst_rows: np.ndarray,
    m: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    u 优先坐标顺序：先处理与最差 u 行强相关的列，再处理 v 超界列，最后其余列。
    """
    if worst_rows.size > 0:
        u_strength = np.sum(np.abs(A[worst_rows, :]), axis=0)
        u_order = np.argsort(-u_strength)
    else:
        u_order = np.arange(m, dtype=np.int64)
    if bad_v.size > 0:
        bv_set = {int(x) for x in bad_v}
        bv = rng.permutation(bad_v)
        rest = np.array([int(j) for j in u_order if int(j) not in bv_set], dtype=np.int64)
        return np.concatenate([bv, rest])
    return u_order


def pick_key_rows(abs_residual: np.ndarray, gamma: int, top_k: int) -> np.ndarray:
    overflow = np.maximum(abs_residual - gamma, 0)
    bad = np.flatnonzero(overflow > 0)
    if bad.size == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(-overflow[bad])
    return bad[order[: min(top_k, bad.size)]]


def try_u_row_snap_move(
    A: np.ndarray,
    residual: np.ndarray,
    v: np.ndarray,
    cols: List[np.ndarray],
    q: int,
    gamma: int,
    score: Tuple[int, int, int],
    cfg: SearchConfig,
    worst_rows: np.ndarray,
    vv_sq: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, int, Tuple[int, int, int], float]]:
    """针对 |u_i|>γ 的最差行，在强相关列上枚举 Δv_j，改善 u 的 L∞ 溢出。"""
    if worst_rows.size == 0:
        return None
    _, m = A.shape
    radius = max(cfg.delta, min(cfg.max_delta, gamma))
    best_score = score
    best_pack: Optional[Tuple[np.ndarray, np.ndarray, int, Tuple[int, int, int], float]] = None
    for i in worst_rows:
        row = A[int(i)]
        col_order = np.argsort(-np.abs(row))[: min(cfg.u_row_snap_cols, m)]
        for j in col_order:
            if int(row[int(j)]) == 0:
                continue
            vj = int(v[int(j)])
            col = cols[int(j)]
            for delta in range(-radius, radius + 1):
                if delta == 0:
                    continue
                nv = vj + delta
                if nv < -gamma or nv > gamma:
                    continue
                cand_res = center_mod(residual - col * delta, q)
                cv, cosum, cmaxov, rr_c = objective_uv_rr_sq_temp_vj(cand_res, v, int(j), nv, gamma)
                cand_score = (cv, cosum, cmaxov)
                if better_score(cand_score, best_score):
                    best_score = cand_score
                    new_vv = vv_sq - vj * vj + nv * nv
                    best_pack = (cand_res.copy(), int(j), nv, cand_score, new_vv, rr_c)
    if best_pack is None:
        return None
    cand_res, j, nv, new_score, new_vv, rr_c = best_pack
    v[j] = nv
    return cand_res, v, new_vv, new_score, rr_c


def cp_sat_repair(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    v: np.ndarray,
    residual: np.ndarray,
    row_top_k: int,
    delta_window: int,
    time_limit_sec: float,
    forced_cols: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    对溢出最严重的若干行、相关列做小规模 CP-SAT 精确修复。

    变量为 ``Δv_j``（有界窗口），目标最小化关键行上的 ``|u_i|`` 代理。
    成功返回 ``(u_new, v_new)``；无 ortools 或无可行解则 None。
    """
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception:
        return None

    n, m = A.shape
    key_rows = pick_key_rows(np.abs(residual), gamma, row_top_k)
    if key_rows.size == 0:
        u = residual.copy()
        return u, v.copy()

    if forced_cols is not None and len(forced_cols) > 0:
        cols = np.unique(np.asarray(forced_cols, dtype=np.int64).ravel())
        cols = cols[(cols >= 0) & (cols < m)]
        max_cols = min(24, m)
        cols = cols[:max_cols]
    else:
        score = np.sum(np.abs(A[key_rows, :]), axis=0)
        col_order = np.argsort(-score)
        max_cols = min(18, m)
        cols = col_order[:max_cols]
    if cols.size == 0:
        return None

    model = cp_model.CpModel()
    dvars: Dict[int, Any] = {}
    for j in cols:
        low = max(-delta_window, -gamma - int(v[j]))
        high = min(delta_window, gamma - int(v[j]))
        dvars[int(j)] = model.NewIntVar(int(low), int(high), f"d_{int(j)}")

    k_vars: Dict[int, Any] = {}
    centered_abs: Dict[int, Any] = {}
    for i in range(n):
        expr = int(residual[i])
        for j in cols:
            expr -= int(A[i, int(j)]) * dvars[int(j)]
        coeff_sum = sum(abs(int(A[i, int(j)])) * delta_window for j in cols)
        lo_expr = int(residual[i]) - coeff_sum
        hi_expr = int(residual[i]) + coeff_sum
        k_min = int(np.floor((lo_expr + gamma) / q)) - 1
        k_max = int(np.ceil((hi_expr - gamma) / q)) + 1
        k_i = model.NewIntVar(k_min, k_max, f"k_{i}")
        k_vars[i] = k_i
        centered = expr - q * k_i
        model.Add(centered <= gamma)
        model.Add(centered >= -gamma)
        abs_c = model.NewIntVar(
            0,
            max(abs(lo_expr), abs(hi_expr)) + abs(q) * max(abs(k_min), abs(k_max)) + 8,
            f"abs_{i}",
        )
        model.AddAbsEquality(abs_c, centered)
        centered_abs[i] = abs_c

    over_vars = []
    for i in key_rows:
        ii = int(i)
        over_i = model.NewIntVar(0, 2_000_000_000, f"over_{ii}")
        model.Add(over_i >= centered_abs[ii] - gamma)
        model.Add(over_i >= 0)
        over_vars.append(over_i)
    max_over = model.NewIntVar(0, 2_000_000_000, "cp_max_over")
    if over_vars:
        model.AddMaxEquality(max_over, over_vars)
    abs_terms = []
    for j in cols:
        a = model.NewIntVar(0, delta_window, f"a_{int(j)}")
        model.AddAbsEquality(a, dvars[int(j)])
        abs_terms.append(a)
    model.Minimize(max_over * 1_000_000 + sum(abs_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.05, float(time_limit_sec))
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    v_new = v.copy()
    for j in cols:
        v_new[int(j)] += int(solver.Value(dvars[int(j)]))
    v_new = np.clip(v_new, -gamma, gamma)
    u_new = center_mod(t - (A @ v_new), q)
    if int(np.max(np.abs(u_new))) <= gamma and int(np.max(np.abs(v_new))) <= gamma:
        return u_new, v_new
    return None


def cp_sat_block_optimize(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    v: np.ndarray,
    residual: np.ndarray,
    score: Tuple[int, int, int],
    cfg: SearchConfig,
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], float]]:
    """Block CP-SAT: optimize overflow surrogate on top violated rows/strong cols."""
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception:
        return None

    n, m = A.shape
    abs_r = np.abs(residual)
    bad = np.where(abs_r > gamma)[0]
    if bad.size == 0:
        return None

    top_rows = min(cfg.block_cp_rows, bad.size)
    row_idx = bad[np.argsort(-abs_r[bad])[:top_rows]]
    col_score = np.sum(np.abs(A[row_idx, :]), axis=0)
    top_cols = min(cfg.block_cp_cols, m)
    cols = np.argsort(-col_score)[:top_cols]
    if cols.size == 0:
        return None

    model = cp_model.CpModel()
    dvars: Dict[int, Any] = {}
    for j in cols:
        low = max(-cfg.block_cp_window, -gamma - int(v[j]))
        high = min(cfg.block_cp_window, gamma - int(v[j]))
        dvars[int(j)] = model.NewIntVar(int(low), int(high), f"bd_{int(j)}")

    max_over = model.NewIntVar(0, 2_000_000_000, "max_over")
    over_vars = []
    abs_move = []
    for j in cols:
        a = model.NewIntVar(0, cfg.block_cp_window, f"ba_{int(j)}")
        model.AddAbsEquality(a, dvars[int(j)])
        abs_move.append(a)

    for i in row_idx:
        ii = int(i)
        expr = int(residual[ii])
        for j in cols:
            expr -= int(A[ii, int(j)]) * dvars[int(j)]
        coeff_sum = sum(abs(int(A[ii, int(j)])) * cfg.block_cp_window for j in cols)
        lo_expr = int(residual[ii]) - coeff_sum
        hi_expr = int(residual[ii]) + coeff_sum
        k_min = int(np.floor((lo_expr + gamma) / q)) - 1
        k_max = int(np.ceil((hi_expr - gamma) / q)) + 1
        k_i = model.NewIntVar(k_min, k_max, f"bk_{ii}")
        centered = expr - q * k_i
        abs_c = model.NewIntVar(
            0,
            max(abs(lo_expr), abs(hi_expr)) + abs(q) * max(abs(k_min), abs(k_max)) + 8,
            f"br_{ii}",
        )
        model.AddAbsEquality(abs_c, centered)
        over_i = model.NewIntVar(0, 2_000_000_000, f"bo_{ii}")
        model.Add(over_i >= abs_c - gamma)
        model.Add(over_i >= 0)
        model.Add(max_over >= over_i)
        over_vars.append(over_i)

    model.Minimize(1_000_000 * max_over + 5_000 * sum(over_vars) + sum(abs_move))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.05, float(cfg.block_cp_time_limit))
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    v_new = v.copy()
    for j in cols:
        v_new[int(j)] += int(solver.Value(dvars[int(j)]))
    v_new = np.clip(v_new, -gamma, gamma)
    u_new = center_mod(t - (A @ v_new), q)
    new_score = objective_uv(u_new, v_new, gamma)
    if not better_score(new_score, score):
        return None
    rr_new = float(np.dot(u_new.astype(np.float64), u_new.astype(np.float64)))
    return u_new, v_new, new_score, rr_new


def try_ls_projection_move(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    residual: np.ndarray,
    v: np.ndarray,
    score: Tuple[int, int, int],
    cfg: SearchConfig,
) -> Optional[Tuple[np.ndarray, np.ndarray, int, Tuple[int, int, int], float]]:
    """Continuous least-squares on worst u-rows + strong columns, then round; must improve score_key."""
    n, m = A.shape
    abs_r = np.abs(residual)
    bad_idx = np.where(abs_r > gamma)[0]
    if bad_idx.size == 0:
        return None
    top = min(cfg.ls_top_rows, bad_idx.size)
    row_idx = bad_idx[np.argsort(-abs_r[bad_idx])[:top]]
    col_score = np.sum(np.abs(A[row_idx, :]), axis=0)
    take = min(cfg.ls_top_cols, m)
    cols_idx = np.argsort(-col_score)[:take]
    AR = A[np.ix_(row_idx, cols_idx)].astype(np.float64)
    b = residual[row_idx].astype(np.float64)
    delta, *_ = np.linalg.lstsq(AR, b, rcond=1e-8)
    for scale in (1.0, 0.62, 0.35, 0.18):
        di = np.rint(delta * scale).astype(np.int64)
        cand_v = v.copy()
        ok_bounds = True
        for jj, cj in enumerate(cols_idx):
            nv = int(cand_v[int(cj)]) + int(di[jj])
            if abs(nv) > gamma:
                ok_bounds = False
                break
            cand_v[int(cj)] = nv
        if not ok_bounds:
            continue
        cand_res = center_mod(t - (A @ cand_v), q)
        viol, osum, maxov, rr_c = objective_uv_and_rr_sq(cand_res, cand_v, gamma)
        cand_score = (viol, osum, maxov)
        if better_score(cand_score, score):
            new_vv = int(np.dot(cand_v.astype(np.int64), cand_v.astype(np.int64)))
            return cand_v, cand_res, new_vv, cand_score, rr_c
    return None


def adaptive_step_radius(
    cfg: SearchConfig,
    bad_count: int,
    max_overflow: int,
    gamma: int,
) -> int:
    if bad_count == 0:
        return cfg.delta
    if max_overflow >= max(gamma - 1, (2 * gamma) // 3):
        return cfg.max_delta
    if max_overflow >= max(2, gamma // 3):
        return max(cfg.delta, (cfg.max_delta + cfg.delta) // 2)
    return cfg.delta


def _single_restart_inner(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    cols: List[np.ndarray],
    cfg: SearchConfig,
    require_norm_lt_q2: bool,
    rng: np.random.Generator,
    restart_idx: int,
    v_init: np.ndarray,
    total_restarts: int,
    K_basis: Optional[np.ndarray] = None,
) -> Tuple[bool, np.ndarray, np.ndarray, Dict[str, Any], Tuple[int, int, int]]:
    """
    单次 restart 的局部搜索主循环（约 ``cfg.iters`` 步）。

    算子包括：单/双坐标移动、kernel walk、LS 投影、CP 修复与块 CP、pull kick 等；
    按 ``residual_phase_end`` / ``kernel_phase_start`` 调节邻域强度。
    返回 ``(success, u, v, meta, score)``；success 时 meta 含详细指标。
    """
    restart_t0 = time.perf_counter()
    v = np.asarray(v_init, dtype=np.int64).copy()
    residual = center_mod(t - (A @ v), q)
    viol, osum, maxov, rr_sq_sync = objective_uv_and_rr_sq(residual, v, gamma)
    score = (viol, osum, maxov)
    vv_sq = int(np.dot(v.astype(np.int64), v.astype(np.int64)))
    best_local_score = score
    best_v_snap = v.copy()
    stagnation = 0
    temperature = 1.0
    _, m = A.shape
    if cfg.kernel_walk_every <= 0:
        K = np.zeros((m, 0), dtype=np.int64)
    elif K_basis is not None:
        K = np.asarray(K_basis, dtype=np.int64)
        if K.shape[0] != m:
            K = np.zeros((m, 0), dtype=np.int64)
    else:
        K = right_kernel_basis_mod_q(A, q, cfg.kernel_max_basis)

    for step in range(cfg.iters):
        if cfg.timeout_sec is not None and (time.perf_counter() - restart_t0) >= cfg.timeout_sec:
            break
        progress = (step + 1) / max(cfg.iters, 1)
        improved = False
        use_entropy = should_compute_entropy(cfg, step, score[0], require_norm_lt_q2, progress)
        in_residual_phase = progress < cfg.residual_phase_end
        in_kernel_phase = progress >= cfg.kernel_phase_start

        if cfg.verbose and cfg.log_every > 0 and step % cfg.log_every == 0:
            elapsed = time.perf_counter() - restart_t0
            print(
                f"[sisinf] restart={restart_idx}/{total_restarts} step={step}/{cfg.iters} "
                f"viol={score[0]} overflow_sum={score[1]} max_ov={score[2]} "
                f"time={elapsed:.2f}s",
                flush=True,
            )

        abs_r = np.abs(residual)
        tk_curr = (
            topk_u_overflow_penalty(residual, gamma, cfg.energy_topk) if cfg.energy_topk > 0 else 0.0
        )
        bad_idx = np.where(abs_r > gamma)[0]
        bad_v = np.flatnonzero(np.abs(v) > gamma)
        bad_count = int(bad_idx.size + bad_v.size)
        if bad_idx.size > 0:
            worst = bad_idx[
                np.argsort(-abs_r[bad_idx])[: min(max(cfg.u_row_snap_top_rows, 8), bad_idx.size)]
            ]
        else:
            worst = np.array([], dtype=np.int64)
        step_radius_main = adaptive_step_radius(cfg, bad_count, score[2], gamma)
        if bad_idx.size > 0 and bad_v.size == 0:
            coord_order = u_priority_coord_order(A, residual, gamma, bad_v, worst, m, rng)
        elif bad_v.size > 0:
            rest = np.setdiff1d(np.arange(m, dtype=np.int64), bad_v)
            coord_order = np.concatenate([rng.permutation(bad_v), rng.permutation(rest)])
        else:
            coord_order = rng.permutation(m)

        for j in coord_order:
            current_vj = int(v[j])
            col = cols[j]

            best_move_score = score
            best_move_delta = 0
            if use_entropy:
                entropy_curr = entropy_residual_plus_v(residual, v, gamma, cfg.entropy_bins)
            else:
                entropy_curr = 0.0
            norm_sq_curr = rr_sq_sync + float(vv_sq)
            current_energy, _ = energy_from_parts(
                score[0],
                score[1],
                score[2],
                norm_sq_curr,
                entropy_curr,
                q,
                require_norm_lt_q2,
                cfg,
                progress,
                tk_curr,
            )
            for delta in range(-step_radius_main, step_radius_main + 1):
                if delta == 0:
                    continue
                new_vj = current_vj + delta
                if new_vj < -gamma or new_vj > gamma:
                    continue
                cand_res = center_mod(residual - col * delta, q)
                cv, cosum, cmaxov, rr_c = objective_uv_rr_sq_temp_vj(cand_res, v, j, new_vj, gamma)
                cand_score = (cv, cosum, cmaxov)
                norm_sq_cand = rr_c + float(vv_sq - current_vj * current_vj + new_vj * new_vj)
                if use_entropy:
                    entropy_cand = entropy_residual_plus_v_temp_vj(
                        cand_res, v, j, new_vj, gamma, cfg.entropy_bins
                    )
                else:
                    entropy_cand = 0.0
                tkc = (
                    topk_u_overflow_penalty(cand_res, gamma, cfg.energy_topk)
                    if cfg.energy_topk > 0
                    else 0.0
                )
                cand_energy, _ = energy_from_parts(
                    cv,
                    cosum,
                    cmaxov,
                    norm_sq_cand,
                    entropy_cand,
                    q,
                    require_norm_lt_q2,
                    cfg,
                    progress,
                    tkc,
                )
                if (better_score(cand_score, best_move_score)) or (
                    same_score_key(cand_score, best_move_score) and cand_energy < current_energy
                ):
                    best_move_score = cand_score
                    best_move_delta = delta
                    current_energy = cand_energy

            if best_move_delta != 0:
                nv = current_vj + best_move_delta
                vv_sq += nv * nv - current_vj * current_vj
                v[j] = nv
                residual = center_mod(residual - col * best_move_delta, q)
                viol, osum, maxov, rr_sq_sync = objective_uv_and_rr_sq(residual, v, gamma)
                score = (viol, osum, maxov)
                improved = True
            else:
                delta = int(rng.integers(-cfg.delta, cfg.delta + 1))
                if delta != 0:
                    new_vj = current_vj + delta
                    if -gamma <= new_vj <= gamma:
                        cand_res = center_mod(residual - col * delta, q)
                        cv, cosum, cmaxov, rr_c = objective_uv_rr_sq_temp_vj(
                            cand_res, v, j, new_vj, gamma
                        )
                        cand_score = (cv, cosum, cmaxov)
                        norm_sq_new = rr_c + float(vv_sq - current_vj * current_vj + new_vj * new_vj)
                        old_energy = current_energy
                        if use_entropy:
                            entropy_new = entropy_residual_plus_v_temp_vj(
                                cand_res, v, j, new_vj, gamma, cfg.entropy_bins
                            )
                        else:
                            entropy_new = 0.0
                        tkn = (
                            topk_u_overflow_penalty(cand_res, gamma, cfg.energy_topk)
                            if cfg.energy_topk > 0
                            else 0.0
                        )
                        new_energy, _ = energy_from_parts(
                            cv,
                            cosum,
                            cmaxov,
                            norm_sq_new,
                            entropy_new,
                            q,
                            require_norm_lt_q2,
                            cfg,
                            progress,
                            tkn,
                        )
                        accept = False
                        if better_score(cand_score, score):
                            accept = True
                        elif same_score_key(cand_score, score) and new_energy < old_energy:
                            accept = True
                        elif cfg.allow_uphill_sa and new_energy > old_energy:
                            # 仅对能量变差的上坡步做 SA；裁剪指数避免 energy≈1e8 时 exp 溢出
                            uphill = (new_energy - old_energy) / max(temperature, 1e-6)
                            prob = np.exp(-min(uphill, 700.0))
                            if rng.random() < prob:
                                accept = True
                        if accept:
                            vv_sq += new_vj * new_vj - current_vj * current_vj
                            v[j] = new_vj
                            residual = cand_res
                            viol, osum, maxov, rr_sq_sync = objective_uv_and_rr_sq(residual, v, gamma)
                            score = (viol, osum, maxov)
                            improved = True

        if worst.size > 0:
            weights = np.sign(residual[worst]).astype(np.int64)
            corr = np.abs((A[worst, :].T @ weights))
            top_cols = np.argsort(-corr)[: min(12, m)]
            repair_radius = adaptive_step_radius(cfg, bad_count, score[2], gamma)
            repair_radius = min(repair_radius, cfg.max_delta)
            for j in top_cols:
                current_vj = int(v[j])
                col = cols[j]
                best_move_score = score
                best_move_delta = 0
                if use_entropy:
                    entropy_curr = entropy_residual_plus_v(residual, v, gamma, cfg.entropy_bins)
                else:
                    entropy_curr = 0.0
                norm_sq_curr = rr_sq_sync + float(vv_sq)
                current_energy, _ = energy_from_parts(
                    score[0],
                    score[1],
                    score[2],
                    norm_sq_curr,
                    entropy_curr,
                    q,
                    require_norm_lt_q2,
                    cfg,
                    progress,
                    tk_curr,
                )
                for delta in range(-repair_radius, repair_radius + 1):
                    if delta == 0:
                        continue
                    new_vj = current_vj + delta
                    if new_vj < -gamma or new_vj > gamma:
                        continue
                    cand_res = center_mod(residual - col * delta, q)
                    cv, cosum, cmaxov, rr_c = objective_uv_rr_sq_temp_vj(
                        cand_res, v, j, new_vj, gamma
                    )
                    cand_score = (cv, cosum, cmaxov)
                    norm_sq_cand = rr_c + float(vv_sq - current_vj * current_vj + new_vj * new_vj)
                    if use_entropy:
                        entropy_cand = entropy_residual_plus_v_temp_vj(
                            cand_res, v, j, new_vj, gamma, cfg.entropy_bins
                        )
                    else:
                        entropy_cand = 0.0
                    tkc2 = (
                        topk_u_overflow_penalty(cand_res, gamma, cfg.energy_topk)
                        if cfg.energy_topk > 0
                        else 0.0
                    )
                    cand_energy, _ = energy_from_parts(
                        cv,
                        cosum,
                        cmaxov,
                        norm_sq_cand,
                        entropy_cand,
                        q,
                        require_norm_lt_q2,
                        cfg,
                        progress,
                        tkc2,
                    )
                    if (better_score(cand_score, best_move_score)) or (
                        same_score_key(cand_score, best_move_score) and cand_energy < current_energy
                    ):
                        best_move_score = cand_score
                        best_move_delta = delta
                        current_energy = cand_energy
                if best_move_delta != 0:
                    nv = current_vj + best_move_delta
                    vv_sq += nv * nv - current_vj * current_vj
                    v[j] = nv
                    residual = center_mod(residual - col * best_move_delta, q)
                    viol, osum, maxov, rr_sq_sync = objective_uv_and_rr_sq(residual, v, gamma)
                    score = (viol, osum, maxov)
                    improved = True

        # Row-associated block search: choose columns most related to largest residual overflows.
        key_rows = pick_key_rows(abs_r, gamma, top_k=5)
        if key_rows.size > 0:
            col_strength = np.sum(np.abs(A[key_rows, :]), axis=0)
            strong_cols = np.argsort(-col_strength)[: min(18, m)]
            if strong_cols.size >= 2:
                for _ in range(min(cfg.pair_relief_attempts, 10)):
                    j, k = rng.choice(strong_cols, size=2, replace=False)
                    pm = pick_best_pair_move(
                        residual,
                        v,
                        cols,
                        q,
                        gamma,
                        int(j),
                        int(k),
                        vv_sq,
                        score,
                        cfg.pair_relief_radius,
                    )
                    if pm is None:
                        continue
                    cr, nj, nk, new_score, new_vv, rr_c = pm
                    oj, ok = int(v[j]), int(v[k])
                    vv_sq += nj * nj + nk * nk - oj * oj - ok * ok
                    v[j], v[k] = nj, nk
                    residual = cr
                    rr_sq_sync = rr_c
                    score = new_score
                    improved = True
                    break

        if (
            cfg.u_row_snap_every > 0
            and bad_idx.size > 0
            and step % cfg.u_row_snap_every == 0
        ):
            snap_rows = (
                worst
                if worst.size > 0
                else bad_idx[np.argsort(-abs_r[bad_idx])[: cfg.u_row_snap_top_rows]]
            )
            snap_out = try_u_row_snap_move(
                A, residual, v, cols, q, gamma, score, cfg, snap_rows, vv_sq
            )
            if snap_out is not None:
                residual, v, vv_sq, score, rr_sq_sync = snap_out
                improved = True

        if cfg.pair_relief_every > 0 and bad_idx.size > 0 and step % cfg.pair_relief_every == 0:
            for _ in range(cfg.pair_relief_attempts):
                j, k = rng.choice(m, size=2, replace=False)
                pm = pick_best_pair_move(
                    residual,
                    v,
                    cols,
                    q,
                    gamma,
                    int(j),
                    int(k),
                    vv_sq,
                    score,
                    cfg.pair_relief_radius,
                )
                if pm is None:
                    continue
                cr, nj, nk, new_score, new_vv, rr_c = pm
                oj, ok = int(v[j]), int(v[k])
                vv_sq += nj * nj + nk * nk - oj * oj - ok * ok
                v[j], v[k] = nj, nk
                residual = cr
                rr_sq_sync = rr_c
                score = new_score
                improved = True
                break

        if score[0] == 0:
            u = residual.copy()
            ok, metrics = verify_solution(A, t, q, gamma, u, v, require_norm_lt_q2)
            if ok:
                return (
                    True,
                    u,
                    v.copy(),
                    {
                        "restart": restart_idx,
                        "steps": step,
                        "violations": score[0],
                        "overflow_sum": score[1],
                        "max_overflow": score[2],
                        **metrics,
                    },
                    score,
                )

        if better_score(score, best_local_score):
            best_local_score = score
            best_v_snap = v.copy()
            stagnation = 0
        else:
            stagnation += 1

        if (
            cfg.use_violation_ls
            and bad_idx.size > 0
            and step % max(1, cfg.violation_ls_every) == 0
        ):
            try:
                vls = violation_ls_step(
                    A,
                    residual,
                    v,
                    cols,
                    q,
                    gamma,
                    score,
                    rng,
                    top_rows=cfg.violation_ls_top_rows,
                    top_cols=cfg.violation_ls_top_cols,
                    better_score=better_score,
                    objective_uv=objective_uv,
                )
                if vls is not None:
                    residual, v, vv_sq, score, rr_sq_sync = vls
                    improved = True
            except Exception:
                pass

        if (
            cfg.use_layered_ls
            and bad_idx.size > 0
            and step % max(1, cfg.layered_ls_every) == 0
        ):
            try:
                lls = layered_row_projection(
                    A,
                    t,
                    q,
                    gamma,
                    residual,
                    v,
                    score,
                    n_layers=3,
                    better_score=better_score,
                    objective_uv_and_rr_sq=objective_uv_and_rr_sq,
                )
                if lls is not None:
                    residual, v, vv_sq, score, rr_sq_sync = lls
                    improved = True
            except Exception:
                pass

        if cfg.cheap_lll_trials > 0 and step > 0 and step % 60 == 0:
            try:
                v, sc2 = cheap_pair_reduction(
                    v,
                    A,
                    t,
                    q,
                    gamma,
                    score,
                    rng,
                    n_trials=cfg.cheap_lll_trials,
                    better_score=better_score,
                    objective_uv=objective_uv,
                )
                if better_score(sc2, score):
                    residual = center_mod(t - (A @ v), q)
                    viol, osum, maxov, rr_sq_sync = objective_uv_and_rr_sq(residual, v, gamma)
                    score = (viol, osum, maxov)
                    improved = True
            except Exception:
                pass

        if not improved and (stagnation % cfg.kick_every == 0):
            score_snap = score
            pull_helped = False
            pull_moved = False
            if cfg.use_pull_kick and bad_idx.size > 0:
                sgn = np.sign(residual).astype(np.float64)
                g = (A.T @ sgn).astype(np.float64)
                mx = float(np.max(np.abs(g)))
                if mx > 1e-12:
                    delta_vec = np.rint(-cfg.pull_kick_gain * (g / mx) * float(cfg.delta)).astype(np.int64)
                    delta_vec = np.clip(delta_vec, -cfg.max_delta, cfg.max_delta)
                    for j in rng.permutation(m):
                        dlv = int(delta_vec[j])
                        if dlv == 0:
                            continue
                        nv = int(np.clip(int(v[j]) + dlv, -gamma, gamma))
                        d = nv - int(v[j])
                        if d == 0:
                            continue
                        v[j] = nv
                        residual = center_mod(residual - cols[j] * d, q)
                        pull_moved = True
                    if pull_moved:
                        viol, osum, maxov, rr_sq_sync = objective_uv_and_rr_sq(residual, v, gamma)
                        score = (viol, osum, maxov)
                        vv_sq = int(np.dot(v.astype(np.int64), v.astype(np.int64)))
                        if better_score(score, score_snap):
                            improved = True
                            pull_helped = True
            if not pull_helped and cfg.gaussian_on_stagnation and bad_idx.size > 0:
                try:
                    sig = max(1.0, cfg.gaussian_seed_sigma * (1.0 + 0.05 * score[2]))
                    for gv in discrete_gaussian_seeds(
                        best_v_snap, gamma, rng, n_seeds=4, sigma=sig
                    ):
                        gres = center_mod(t - (A @ gv), q)
                        gsc = objective_uv(gres, gv, gamma)
                        if better_score(gsc, score):
                            v = gv.copy()
                            residual = gres
                            score = gsc
                            viol, osum, maxov, rr_sq_sync = objective_uv_and_rr_sq(
                                residual, v, gamma
                            )
                            vv_sq = int(np.dot(v.astype(np.int64), v.astype(np.int64)))
                            improved = True
                            pull_helped = True
                            break
                except Exception:
                    pass
            if not pull_helped:
                kick_idx = rng.choice(m, size=min(cfg.kick_size, m), replace=False)
                for j in kick_idx:
                    target = int(v[j] + rng.integers(-cfg.delta * 2, cfg.delta * 2 + 1))
                    target = max(-gamma, min(gamma, target))
                    d = target - int(v[j])
                    if d != 0:
                        v[j] = target
                        residual = center_mod(residual - cols[j] * d, q)
                viol, osum, maxov, rr_sq_sync = objective_uv_and_rr_sq(residual, v, gamma)
                score = (viol, osum, maxov)
                vv_sq = int(np.dot(v.astype(np.int64), v.astype(np.int64)))
        temperature = max(0.02, temperature * 0.996)

        if (
            cfg.ls_project_every > 0
            and bad_idx.size > 0
            and step % cfg.ls_project_every == 0
            and (in_residual_phase or score[0] > 0)
        ):
            ls_out = try_ls_projection_move(A, t, q, gamma, residual, v, score, cfg)
            if ls_out is not None:
                v, residual, vv_sq, score, rr_sq_sync = ls_out
                improved = True

        u_only_overflow = bad_idx.size > 0 and bad_v.size == 0
        if (
            K.shape[1] > 0
            and cfg.kernel_walk_every > 0
            and step % cfg.kernel_walk_every == 0
            and in_kernel_phase
            and not u_only_overflow
        ):
            for _ in range(12):
                coeffs = rng.integers(
                    -cfg.kernel_coeff_max, cfg.kernel_coeff_max + 1, size=K.shape[1]
                )
                if np.all(coeffs == 0):
                    continue
                dcomb = (K @ coeffs.astype(np.int64).reshape(-1, 1)).ravel()
                dcomb = (dcomb % q + q) % q
                dcomb = np.where(dcomb > q // 2, dcomb - q, dcomb).astype(np.int64)
                v_try = np.clip(v + dcomb, -gamma, gamma).astype(np.int64)
                d_eff = (v_try - v).astype(np.int64)
                if not bool(np.all((A @ d_eff) % q == 0)):
                    continue
                v_prev = v.copy()
                v = v_try
                viol, osum, maxov, rr_sq_sync = objective_uv_and_rr_sq(residual, v, gamma)
                new_sc = (viol, osum, maxov)
                if better_score(new_sc, score):
                    vv_sq = int(np.dot(v.astype(np.int64), v.astype(np.int64)))
                    score = new_sc
                    improved = True
                    break
                v = v_prev

        if (
            cfg.cp_periodic_every > 0
            and step > 0
            and step % cfg.cp_periodic_every == 0
            and in_kernel_phase
        ):
            ncol = min(cfg.cp_periodic_cols, m)
            forced = rng.choice(m, size=ncol, replace=False)
            rep2 = cp_sat_repair(
                A=A,
                t=t,
                q=q,
                gamma=gamma,
                v=v,
                residual=residual,
                row_top_k=min(28, max(cfg.cp_repair_threshold, 12)),
                delta_window=cfg.cp_repair_window,
                time_limit_sec=min(0.4, cfg.cp_repair_time_limit),
                forced_cols=forced,
            )
            if rep2 is not None:
                u2, v2 = rep2
                ok2, met2 = verify_solution(A, t, q, gamma, u2, v2, require_norm_lt_q2)
                if ok2:
                    return (
                        True,
                        u2.copy(),
                        v2.copy(),
                        {
                            "restart": restart_idx,
                            "steps": step,
                            "violations": 0,
                            "overflow_sum": 0,
                            "max_overflow": 0,
                            "cp_repaired": 1,
                            "cp_periodic": 1,
                            **met2,
                        },
                        (0, 0, 0),
                    )

        if (
            cfg.block_cp_every > 0
            and step > 0
            and step % cfg.block_cp_every == 0
            and bad_idx.size > 0
        ):
            block_out = cp_sat_block_optimize(A, t, q, gamma, v, residual, score, cfg)
            if block_out is not None:
                ub, vb, scb, rrb = block_out
                v = vb.copy()
                residual = ub.copy()
                score = scb
                rr_sq_sync = rrb
                vv_sq = int(np.dot(v.astype(np.int64), v.astype(np.int64)))
                improved = True

        if (
            cfg.cp_aggressive_every > 0
            and step > 0
            and step % cfg.cp_aggressive_every == 0
            and bad_idx.size > 0
        ):
            agg_rep = cp_sat_repair(
                A=A,
                t=t,
                q=q,
                gamma=gamma,
                v=v,
                residual=residual,
                row_top_k=min(cfg.cp_aggressive_row_k, int(bad_idx.size)),
                delta_window=max(cfg.cp_repair_window, cfg.max_delta),
                time_limit_sec=max(cfg.cp_repair_time_limit, 1.5),
                forced_cols=None,
            )
            if agg_rep is not None:
                u_agg, v_agg = agg_rep
                sc_agg = objective_uv(u_agg, v_agg, gamma)
                if better_score(sc_agg, score):
                    v = v_agg.copy()
                    residual = u_agg.copy()
                    viol, osum, maxov, rr_sq_sync = objective_uv_and_rr_sq(residual, v, gamma)
                    score = (viol, osum, maxov)
                    vv_sq = int(np.dot(v.astype(np.int64), v.astype(np.int64)))
                    improved = True
                    ok_agg, met_agg = verify_solution(A, t, q, gamma, residual, v, require_norm_lt_q2)
                    if ok_agg:
                        return (
                            True,
                            residual.copy(),
                            v.copy(),
                            {"restart": restart_idx, "steps": step, "cp_aggressive": 1, **met_agg},
                            score,
                        )

        # Optional exact repair near feasibility (graceful no-op if ortools unavailable).
        if score[0] <= cfg.cp_repair_threshold:
            repaired = cp_sat_repair(
                A=A,
                t=t,
                q=q,
                gamma=gamma,
                v=v,
                residual=residual,
                row_top_k=cfg.cp_repair_threshold,
                delta_window=cfg.cp_repair_window,
                time_limit_sec=cfg.cp_repair_time_limit,
                forced_cols=None,
            )
            if repaired is not None:
                u_rep, v_rep = repaired
                ok, metrics = verify_solution(A, t, q, gamma, u_rep, v_rep, require_norm_lt_q2)
                if ok:
                    return (
                        True,
                        u_rep.copy(),
                        v_rep.copy(),
                        {
                            "restart": restart_idx,
                            "steps": step,
                            "violations": 0,
                            "overflow_sum": 0,
                            "max_overflow": 0,
                            "cp_repaired": 1,
                            **metrics,
                        },
                        (0, 0, 0),
                    )

    return False, residual.copy(), v.copy(), {}, score


def _kernel_columns_from_payload(payload: Dict[str, Any], m: int) -> Optional[np.ndarray]:
    """从并行 worker payload 的 ``kernel_K`` 列列表重建 (m,k) 核矩阵；缺失则 None。"""
    cols_list = payload.get("kernel_K")
    if cols_list is None:
        return None
    if len(cols_list) == 0:
        return np.zeros((m, 0), dtype=np.int64)
    return np.column_stack([np.asarray(c, dtype=np.int64).ravel() for c in cols_list])


def _parallel_restart_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """ProcessPool 子进程入口：反序列化 payload，跑 ``_single_restart_inner``，返回 u/v/meta/score。"""
    A = np.asarray(payload["A"], dtype=np.int64)
    t = np.asarray(payload["t"], dtype=np.int64)
    q = int(payload["q"])
    gamma = int(payload["gamma"])
    cfg = SearchConfig(**payload["cfg"])
    require_norm_lt_q2 = bool(payload["require_norm_lt_q2"])
    restart_idx = int(payload["restart_idx"])
    total_restarts = int(payload["total_restarts"])
    v_init = np.asarray(payload["v_init"], dtype=np.int64)
    _, m = A.shape
    cols = [A[:, j].copy() for j in range(m)]
    rng = np.random.default_rng(int(cfg.seed) + restart_idx * 10007 + 31)
    K_from_parent = _kernel_columns_from_payload(payload, m)
    ok, u, v, meta, score = _single_restart_inner(
        A,
        t,
        q,
        gamma,
        cols,
        cfg,
        require_norm_lt_q2,
        rng,
        restart_idx,
        v_init,
        total_restarts,
        K_basis=K_from_parent,
    )
    return {
        "restart_idx": restart_idx,
        "success": ok,
        "u": u.tolist(),
        "v": v.tolist(),
        "meta": meta,
        "score": list(score),
    }


def local_search_one(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    cfg: SearchConfig,
    require_norm_lt_q2: bool = False,
    prepend_v_seeds: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    对单实例运行完整搜索：BKZ/核预计算 → dual 种子 → 多 restart → 返回最优 (u,v)。

    若某 restart 内层已可行则立即返回；否则返回 score_key 最优的一轮结果及 meta。
    """
    rng = np.random.default_rng(cfg.seed)
    n, m = A.shape
    A = np.mod(A, q).astype(np.int64, copy=False)
    t = np.mod(t, q).astype(np.int64, copy=False)
    cols = [A[:, j].copy() for j in range(m)]

    if cfg.kernel_walk_every > 0:
        K_shared = right_kernel_basis_mod_q(A, q, cfg.kernel_max_basis)
        kernel_K_payload = [K_shared[:, j].tolist() for j in range(K_shared.shape[1])]
    else:
        K_shared = np.zeros((m, 0), dtype=np.int64)
        kernel_K_payload: List[List[int]] = []

    lattice_prepend: List[np.ndarray] = []
    if prepend_v_seeds:
        for pv in prepend_v_seeds:
            pv = np.asarray(pv, dtype=np.int64).ravel()
            if pv.size == m:
                lattice_prepend.append(pv.copy())
    lattice_backend = "none"
    seed_sources: List[str] = []
    if cfg.use_restricted_svp_seeds:
        try:
            rs = collect_restricted_svp_v_seeds(
                A,
                t,
                q,
                gamma,
                rng,
                max(16, cfg.bkz_max_vectors),
                beta=cfg.bkz_beta,
                max_dim=cfg.bkz_max_dim,
                tail_rank=cfg.wang_enum_tail_rank,
                coeff_max=cfg.wang_enum_coeff_max,
                enum_pool_size=cfg.wang_enum_pool_size,
                enum_max_trials=cfg.wang_enum_max_trials,
                n_random=cfg.restricted_svp_samples,
                require_norm_lt_q2=require_norm_lt_q2,
                use_g6k_enumerate=cfg.use_g6k_sieve,
                g6k_sieve_alg=cfg.g6k_sieve_alg,
                g6k_saturation_ratio=cfg.g6k_saturation_ratio,
                g6k_threads=cfg.g6k_threads,
                g6k_bkz_block=cfg.g6k_bkz_block or cfg.bkz_beta,
                g6k_max_lift_vectors=cfg.g6k_max_lift_vectors,
            )
            lattice_prepend.extend(rs)
            seed_sources.append(f"restricted_svp:{len(rs)}")
        except Exception:
            pass
    if cfg.use_wagner_seeds:
        try:
            lattice_prepend.extend(
                wagner_subsystem_seeds(
                    A,
                    t,
                    q,
                    gamma,
                    rng,
                    n_rows=cfg.wagner_rows,
                    n_cols=cfg.wagner_cols,
                    box_radius=cfg.wagner_box_radius,
                    list_cap=cfg.wagner_list_cap,
                    max_seeds=16,
                )
            )
        except Exception:
            pass
    if cfg.use_kannan_seeds and cfg.bkz_beta > 0:
        try:
            kn = collect_kannan_v_seeds(
                A,
                t,
                q,
                gamma,
                cfg.bkz_beta,
                max(16, cfg.bkz_max_vectors // 2),
                cfg.bkz_max_dim,
                rng,
                embedding_factor=cfg.kannan_embedding_factor or None,
            )
            lattice_prepend.extend(kn)
            seed_sources.append(f"kannan:{len(kn)}")
        except Exception:
            pass
    if cfg.use_bkz_seeds and cfg.bkz_beta > 0:
        try:
            if cfg.use_sieve_seeds:
                bkz_seeds = collect_sieve_v_seeds(
                    A,
                    q,
                    gamma,
                    cfg.bkz_beta,
                    cfg.bkz_max_vectors,
                    cfg.bkz_max_dim,
                    rng,
                    combo_depth=cfg.bkz_combo_depth,
                    combo_coeff_max=cfg.bkz_combo_coeff_max,
                    use_g6k=cfg.use_g6k_sieve,
                    g6k_sieve_alg=cfg.g6k_sieve_alg,
                    g6k_saturation_ratio=cfg.g6k_saturation_ratio,
                    g6k_threads=cfg.g6k_threads,
                    g6k_bkz_block=cfg.g6k_bkz_block or cfg.bkz_beta,
                    g6k_max_lift_vectors=cfg.g6k_max_lift_vectors,
                )
                seed_sources.append(
                    "g6k+bkz" if cfg.use_g6k_sieve else "bkz+sieve"
                )
            else:
                bkz_seeds = collect_bkz_v_seeds(
                    A,
                    q,
                    gamma,
                    cfg.bkz_beta,
                    cfg.bkz_max_vectors,
                    cfg.bkz_max_dim,
                    combo_depth=cfg.bkz_combo_depth,
                    combo_coeff_max=cfg.bkz_combo_coeff_max,
                    rng=rng,
                    bkz_tours=2,
                )
                seed_sources.append("bkz")
            lattice_prepend.extend(bkz_seeds)
            lattice_backend = lattice_backend_label()
        except Exception:
            lattice_backend = "error"

    dual_candidates: List[np.ndarray] = []
    dual_meta: Dict[str, Any] = {
        "num_candidates": 0,
        "lattice_backend": lattice_backend,
        "lattice_seed_count": len(lattice_prepend),
        "seed_sources": seed_sources,
    }

    if cfg.use_dual_space:
        dual_candidates, built_meta = build_dual_space_candidates(
            A, t, q, gamma, cfg, rng, prepend=lattice_prepend if lattice_prepend else None
        )
        dual_meta = {**dual_meta, **built_meta}
    elif lattice_prepend:
        scored = []
        extras: List[np.ndarray] = []
        extras.extend(modular_pull_seed_vectors(A, t, q, gamma, rng, cfg.modular_pull_variants))
        extras.extend(lattice_prepend)
        for vv in extras:
            vv = np.asarray(vv, dtype=np.int64).ravel()
            if vv.size != m:
                continue
            residual = center_mod(t - (A @ vv), q)
            viol, overflow_sum, max_overflow = objective_uv(residual, vv, gamma)
            scored.append((viol, overflow_sum, max_overflow, vv.copy()))
        scored.sort(key=lambda x: score_key((x[0], x[1], x[2])))
        cap = max(cfg.restarts, cfg.candidate_count, len(scored))
        dual_candidates = [x[3] for x in scored[:cap]]
        dual_meta = {"num_candidates": len(dual_candidates)}

    workers = max(1, int(cfg.parallel_workers))
    if workers > 1 and cfg.restarts > 1:
        payloads: List[Dict[str, Any]] = []
        for restart in range(cfg.restarts):
            if dual_candidates and restart < len(dual_candidates):
                v0 = dual_candidates[restart].copy()
            elif restart == 0:
                v0 = np.zeros(m, dtype=np.int64)
            else:
                v0 = rng.integers(low=-gamma, high=gamma + 1, size=m, dtype=np.int64)
                if np.all(v0 == 0):
                    v0[rng.integers(0, m)] = 1
            payloads.append(
                {
                    "A": A,
                    "t": t,
                    "q": q,
                    "gamma": gamma,
                    "cfg": asdict(cfg),
                    "require_norm_lt_q2": require_norm_lt_q2,
                    "restart_idx": restart,
                    "total_restarts": cfg.restarts,
                    "v_init": v0.tolist(),
                    "kernel_K": kernel_K_payload,
                }
            )
        raw: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_parallel_restart_worker, p) for p in payloads]
            for fut in as_completed(futures):
                raw.append(fut.result())

        successes = [r for r in raw if r["success"]]
        if successes:
            best_s = min(successes, key=lambda x: x["restart_idx"])
            meta_succ: Dict[str, Any] = best_s["meta"]
            return (
                np.asarray(best_s["u"], dtype=np.int64),
                np.asarray(best_s["v"], dtype=np.int64),
                meta_succ,
            )

        best_global = None
        best_u: Optional[np.ndarray] = None
        best_v: Optional[np.ndarray] = None
        best_meta: Optional[Dict[str, Any]] = None
        for r in sorted(raw, key=lambda x: x["restart_idx"]):
            score_t = tuple(int(x) for x in r["score"])
            if best_global is None or better_score(score_t, best_global):
                best_global = score_t
                v_arr = np.asarray(r["v"], dtype=np.int64)
                u_arr = np.asarray(r["u"], dtype=np.int64)
                ok, metrics = verify_solution(A, t, q, gamma, u_arr, v_arr, require_norm_lt_q2)
                _, energy_meta = energy_score(
                    u_arr,
                    v_arr,
                    gamma,
                    q,
                    require_norm_lt_q2,
                    cfg,
                    1.0,
                )
                best_u = u_arr.copy()
                best_v = v_arr.copy()
                best_meta = {
                    "restart": r["restart_idx"],
                    "steps": cfg.iters,
                    "violations": score_t[0],
                    "overflow_sum": score_t[1],
                    "max_overflow": score_t[2],
                    "feasible": int(ok),
                    **metrics,
                    "energy": energy_meta["energy"],
                    "entropy": energy_meta["entropy"],
                    "dual_candidates": dual_meta["num_candidates"],
                }
        if best_u is not None and best_v is not None:
            return best_u, best_v, best_meta if best_meta is not None else {}
        last = raw[-1]
        return (
            np.asarray(last["u"], dtype=np.int64),
            np.asarray(last["v"], dtype=np.int64),
            {},
        )

    best_global = None
    best_meta: Optional[Dict[str, Any]] = None
    best_u: Optional[np.ndarray] = None
    best_v: Optional[np.ndarray] = None

    for restart in range(cfg.restarts):
        if dual_candidates and restart < len(dual_candidates):
            v_init = dual_candidates[restart].copy()
        elif restart == 0:
            v_init = np.zeros(m, dtype=np.int64)
        else:
            v_init = rng.integers(low=-gamma, high=gamma + 1, size=m, dtype=np.int64)
            if np.all(v_init == 0):
                v_init[rng.integers(0, m)] = 1

        ok, u_out, v_out, meta_out, score = _single_restart_inner(
            A,
            t,
            q,
            gamma,
            cols,
            cfg,
            require_norm_lt_q2,
            rng,
            restart,
            v_init,
            cfg.restarts,
            K_basis=K_shared,
        )
        if ok:
            return u_out, v_out, meta_out

        if best_global is None or better_score(score, best_global):
            best_global = score
            ok, metrics = verify_solution(A, t, q, gamma, u_out, v_out, require_norm_lt_q2)
            _, energy_meta = energy_score(
                u_out,
                v_out,
                gamma,
                q,
                require_norm_lt_q2,
                cfg,
                1.0,
            )
            best_u = u_out.copy()
            best_v = v_out.copy()
            best_meta = {
                "restart": restart,
                "steps": cfg.iters,
                "violations": score[0],
                "overflow_sum": score[1],
                "max_overflow": score[2],
                "feasible": int(ok),
                **metrics,
                "energy": energy_meta["energy"],
                "entropy": energy_meta["entropy"],
                "dual_candidates": dual_meta.get("num_candidates", 0),
                "lattice_backend": dual_meta.get("lattice_backend", lattice_backend),
                "lattice_seed_count": dual_meta.get("lattice_seed_count", len(lattice_prepend)),
            }

    if best_u is not None and best_v is not None:
        return best_u, best_v, best_meta if best_meta is not None else {}
    return np.zeros(n, dtype=np.int64), np.zeros(m, dtype=np.int64), {}


def solve_instances(instances: List[Dict], cfg: SearchConfig) -> List[Dict]:
    """
    CLI 批量入口：逐实例推断 sis_class，应用 ``apply_sis_class_defaults`` 后调用 ``local_search_one``。
    """
    out = []
    seed_base = cfg.seed
    for idx, inst in enumerate(instances):
        q = int(inst["q"])
        gamma = int(inst["gamma"])
        A = np.array(inst["A"], dtype=np.int64)
        t = np.array(inst["t"], dtype=np.int64)
        try:
            sis_class = problem_class_from_instance(inst)
            require_norm_lt_q2 = effective_require_norm_lt_q2(inst, sis_class)
            local_cfg = apply_sis_class_defaults(cfg, sis_class)
        except Exception:
            sis_class = 0
            require_norm_lt_q2 = bool(inst.get("require_norm_lt_q2", False))
            local_cfg = cfg
        local_cfg = replace(local_cfg, seed=seed_base + idx)
        t0 = time.time()
        u, v, meta = local_search_one(
            A=A,
            t=t,
            q=q,
            gamma=gamma,
            cfg=local_cfg,
            require_norm_lt_q2=require_norm_lt_q2,
        )
        elapsed = time.time() - t0
        ok, verify = verify_solution(A, t, q, gamma, u, v, require_norm_lt_q2)

        out.append(
            {
                "id": inst.get("id", idx + 1),
                "q": q,
                "gamma": gamma,
                "success": bool(ok),
                "elapsed_sec": elapsed,
                "u": u.tolist(),
                "v": v.tolist(),
                "meta": meta,
                "verify": verify,
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solve SIS_infinity instances.")
    p.add_argument("--input", required=True, help="Path to input JSON.")
    p.add_argument("--output", required=True, help="Path to output JSON.")
    p.add_argument("--restarts", type=int, default=40)
    p.add_argument("--iters", type=int, default=2500)
    p.add_argument("--delta", type=int, default=2)
    p.add_argument("--kick-size", type=int, default=6)
    p.add_argument("--kick-every", type=int, default=120)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-delta", type=int, default=6)
    p.add_argument("--candidate-count", type=int, default=24)
    p.add_argument("--entropy-weight", type=float, default=0.25)
    p.add_argument("--euclid-weight", type=float, default=1.5)
    p.add_argument("--overflow-weight", type=float, default=1.0)
    p.add_argument("--entropy-bins", type=int, default=8)
    p.add_argument("--no-dynamic-schedule", action="store_true")
    p.add_argument("--no-dual-space", action="store_true")
    p.add_argument(
        "--entropy-interval",
        type=int,
        default=50,
        help="Histogram entropy every k steps (<=0 disables). Denser schedule when require_norm_lt_q2 and viol==0.",
    )
    p.add_argument("--verbose", action="store_true", help="Print periodic progress to stderr.")
    p.add_argument("--log-every", type=int, default=500, help="Steps between progress lines when --verbose.")
    p.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Per-restart time limit in seconds (0 = no limit).",
    )
    p.add_argument("--no-bkz-seeds", action="store_true", help="Disable optional BKZ lattice seeds (needs fpylll).")
    p.add_argument("--bkz-beta", type=int, default=0, help="BKZ block size; 0 disables lattice reduction.")
    p.add_argument("--bkz-max-dim", type=int, default=96, help="Skip BKZ when n+m exceeds this.")
    p.add_argument("--bkz-max-vectors", type=int, default=24, help="Max clipped v seeds from reduced basis.")
    p.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Parallel processes for restarts (>1 changes RNG vs single-threaded).",
    )
    p.add_argument(
        "--modular-pull-variants",
        type=int,
        default=4,
        help="Number of modular-pull candidate seeds (0 disables).",
    )
    p.add_argument(
        "--cvp-lift-variants",
        type=int,
        default=6,
        help="Number of CVP lifting seed variants from t+qk (0 disables).",
    )
    p.add_argument("--pair-relief-every", type=int, default=32, help="0 disables 2-coordinate joint search.")
    p.add_argument("--pair-relief-attempts", type=int, default=12)
    p.add_argument("--pair-relief-radius", type=int, default=2)
    p.add_argument("--no-pull-kick", action="store_true", help="Disable A^T·sign(u) stagnation kick.")
    p.add_argument("--pull-kick-gain", type=float, default=1.25)
    p.add_argument("--cheby-weight", type=float, default=20.0, help="Weight for max_overflow (Chebyshev).")
    p.add_argument("--cheby-boost-threshold", type=int, default=20)
    p.add_argument("--cheby-boost-factor", type=float, default=2.0)
    p.add_argument("--cp-repair-threshold", type=int, default=8, help="Trigger CP-SAT repair when violations <= this.")
    p.add_argument("--cp-repair-window", type=int, default=3, help="Delta window for CP-SAT repair variables.")
    p.add_argument("--cp-repair-time-limit", type=float, default=0.5, help="Per-call CP-SAT time limit (seconds).")
    p.add_argument(
        "--cp-aggressive-every",
        type=int,
        default=0,
        help="Periodic CP-SAT on worst u rows when far from feasible (0=use class default).",
    )
    p.add_argument(
        "--cp-aggressive-row-k",
        type=int,
        default=0,
        help="Number of worst u rows per aggressive CP call (0=use SearchConfig/class default).",
    )
    p.add_argument(
        "--u-row-snap-every",
        type=int,
        default=0,
        help="Directed coordinate snap on worst u rows every N steps (0=use class default).",
    )
    p.add_argument(
        "--u-row-snap-top-rows",
        type=int,
        default=0,
        help="How many worst u rows to target for row snap (0=use SearchConfig/class default).",
    )
    p.add_argument(
        "--u-row-snap-cols",
        type=int,
        default=0,
        help="How many v columns to enumerate per u row snap (0=use SearchConfig/class default).",
    )
    p.add_argument("--kernel-walk-every", type=int, default=25, help="0 disables mod-q kernel walk on v (u unchanged).")
    p.add_argument("--kernel-coeff-max", type=int, default=2, help="Max absolute coeff on each kernel basis vector.")
    p.add_argument("--kernel-max-basis", type=int, default=24, help="Max number of kernel basis columns to use.")
    p.add_argument("--ls-project-every", type=int, default=35, help="0 disables least-squares projection moves.")
    p.add_argument("--ls-top-rows", type=int, default=14)
    p.add_argument("--ls-top-cols", type=int, default=28)
    p.add_argument("--energy-topk", type=int, default=5, help="0 disables top-k u-overflow penalty in energy.")
    p.add_argument("--energy-topk-weight", type=float, default=0.12)
    p.add_argument(
        "--entropy-disable-after-progress",
        type=float,
        default=0.78,
        help="Disable entropy term when schedule progress reaches this (>=1 never).",
    )
    p.add_argument("--cp-periodic-every", type=int, default=0, help="Periodic CP-SAT on random columns (0=off).")
    p.add_argument("--cp-periodic-cols", type=int, default=16)
    p.add_argument("--block-cp-every", type=int, default=0, help="Periodic block CP-SAT overflow optimization (0=off).")
    p.add_argument("--block-cp-rows", type=int, default=20)
    p.add_argument("--block-cp-cols", type=int, default=28)
    p.add_argument("--block-cp-window", type=int, default=6)
    p.add_argument("--block-cp-time-limit", type=float, default=2.0)
    p.add_argument(
        "--residual-phase-end",
        type=float,
        default=0.45,
        help="Progress cutoff for residual-focused phase (0~1).",
    )
    p.add_argument(
        "--kernel-phase-start",
        type=float,
        default=0.60,
        help="Progress to start stronger kernel/LNS phase (0~1).",
    )
    p.add_argument(
        "--allow-uphill-sa",
        action="store_true",
        help="Allow uphill simulated-annealing acceptance (off by default for monotone convergence).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        instances = json.load(f)
    cfg = SearchConfig(
        restarts=args.restarts,
        iters=args.iters,
        delta=args.delta,
        kick_size=args.kick_size,
        kick_every=args.kick_every,
        seed=args.seed,
        max_delta=args.max_delta,
        candidate_count=args.candidate_count,
        entropy_weight=args.entropy_weight,
        euclid_weight=args.euclid_weight,
        overflow_weight=args.overflow_weight,
        entropy_bins=args.entropy_bins,
        dynamic_schedule=not args.no_dynamic_schedule,
        use_dual_space=not args.no_dual_space,
        entropy_update_interval=args.entropy_interval,
        verbose=args.verbose,
        log_every=args.log_every,
        timeout_sec=None if args.timeout <= 0 else float(args.timeout),
        use_bkz_seeds=not args.no_bkz_seeds,
        bkz_beta=max(0, int(args.bkz_beta)),
        bkz_max_vectors=max(1, int(args.bkz_max_vectors)),
        bkz_max_dim=max(8, int(args.bkz_max_dim)),
        parallel_workers=max(1, int(args.parallel_workers)),
        modular_pull_variants=max(0, int(args.modular_pull_variants)),
        cvp_lift_variants=max(0, int(args.cvp_lift_variants)),
        pair_relief_every=max(0, int(args.pair_relief_every)),
        pair_relief_attempts=max(1, int(args.pair_relief_attempts)),
        pair_relief_radius=max(1, int(args.pair_relief_radius)),
        use_pull_kick=not args.no_pull_kick,
        pull_kick_gain=float(args.pull_kick_gain),
        cheby_weight=float(args.cheby_weight),
        cheby_boost_threshold=max(0, int(args.cheby_boost_threshold)),
        cheby_boost_factor=max(1.0, float(args.cheby_boost_factor)),
        cp_repair_threshold=max(0, int(args.cp_repair_threshold)),
        cp_repair_window=max(1, int(args.cp_repair_window)),
        cp_repair_time_limit=max(0.05, float(args.cp_repair_time_limit)),
        cp_aggressive_every=max(0, int(args.cp_aggressive_every)),
        cp_aggressive_row_k=max(0, int(args.cp_aggressive_row_k)),
        u_row_snap_every=max(0, int(args.u_row_snap_every)),
        u_row_snap_top_rows=max(0, int(args.u_row_snap_top_rows)),
        u_row_snap_cols=max(0, int(args.u_row_snap_cols)),
        kernel_walk_every=max(0, int(args.kernel_walk_every)),
        kernel_coeff_max=max(1, int(args.kernel_coeff_max)),
        kernel_max_basis=max(1, int(args.kernel_max_basis)),
        ls_project_every=max(0, int(args.ls_project_every)),
        ls_top_rows=max(1, int(args.ls_top_rows)),
        ls_top_cols=max(1, int(args.ls_top_cols)),
        energy_topk=max(0, int(args.energy_topk)),
        energy_topk_weight=max(0.0, float(args.energy_topk_weight)),
        entropy_disable_after_progress=float(args.entropy_disable_after_progress),
        cp_periodic_every=max(0, int(args.cp_periodic_every)),
        cp_periodic_cols=max(1, int(args.cp_periodic_cols)),
        block_cp_every=max(0, int(args.block_cp_every)),
        block_cp_rows=max(4, int(args.block_cp_rows)),
        block_cp_cols=max(6, int(args.block_cp_cols)),
        block_cp_window=max(1, int(args.block_cp_window)),
        block_cp_time_limit=max(0.05, float(args.block_cp_time_limit)),
        residual_phase_end=min(0.95, max(0.05, float(args.residual_phase_end))),
        kernel_phase_start=min(0.99, max(0.10, float(args.kernel_phase_start))),
        allow_uphill_sa=bool(args.allow_uphill_sa),
    )
    if cfg.kernel_phase_start <= cfg.residual_phase_end:
        cfg.kernel_phase_start = min(0.99, cfg.residual_phase_end + 0.10)
    results = solve_instances(instances, cfg)

    summary = {
        "num_instances": len(results),
        "num_success": int(sum(1 for r in results if r["success"])),
        "total_time_sec": float(sum(r["elapsed_sec"] for r in results)),
    }
    payload = {"summary": summary, "results": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))


# ===== lattice_seeds =====

# ===== lattice_bkz.py =====

_FPYLLL_OK: Optional[bool] = None


def fpylll_available() -> bool:
    """是否能在本进程 import fpylll。"""
    global _FPYLLL_OK
    if _FPYLLL_OK is not None:
        return _FPYLLL_OK
    try:
        from fpylll import BKZ, IntegerMatrix, LLL  # noqa: F401

        _FPYLLL_OK = True
    except ImportError:
        _FPYLLL_OK = False
    return _FPYLLL_OK


def lattice_backend_label() -> str:
    if fpylll_available():
        return "fpylll"
    if os.environ.get("SIS_USE_WSL_BKZ", "").strip() in ("1", "true", "yes"):
        return "wsl_fpylll_or_heuristic"
    return "heuristic"


def _build_ajtai_basis(A: np.ndarray, q: int) -> Tuple[np.ndarray, int, int]:
    A = np.mod(np.asarray(A, dtype=np.int64), q)
    n, m = A.shape
    d = n + m
    B = np.zeros((d, d), dtype=np.int64)
    for j in range(n):
        B[j, j] = q
    for jj in range(m):
        col = n + jj
        B[:n, col] = -A[:, jj]
        B[n + jj, col] = 1
    return B, n, m


def _append_clipped_v(
    out: List[np.ndarray],
    seen: set,
    v_part: np.ndarray,
    gamma: int,
    max_vectors: int,
) -> bool:
    for cand in (v_part, -v_part):
        clip = np.clip(cand, -gamma, gamma).astype(np.int64, copy=False)
        key = clip.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(clip.copy())
        if len(out) >= max_vectors:
            return True
    return False


def _integer_matrix_from_basis(B: np.ndarray):
    from fpylll import IntegerMatrix

    d = B.shape[0]
    M = IntegerMatrix(d, d)
    for i in range(d):
        for j in range(d):
            M[i, j] = int(B[i, j])
    return M


def _fpylll_reduce_multi_tour(
    B: np.ndarray,
    beta: int,
    *,
    block_sizes: Optional[List[int]] = None,
    tours: int = 2,
    force_bkz: bool = False,
) -> np.ndarray:
    """
    LLL + 多 block_size 的 BKZ（近似 oracle 多种子策略）。

    返回约化后的整数基矩阵 (d,d) 的 numpy 副本（列向量为格基）。
    """
    from fpylll import BKZ, LLL

    d = B.shape[0]
    M = _integer_matrix_from_basis(B)
    LLL.reduction(M)

    if block_sizes is None:
        bs0 = max(2, min(int(beta), d))
        block_sizes = sorted(
            set(
                [
                    max(2, bs0 - 4),
                    bs0,
                    min(d, bs0 + 4),
                    min(d, bs0 + 8),
                ]
            )
        )

    # n+m≈200 时全量 BKZ 极慢；>160 维仅 LLL（仍比纯启发式强）
    # 小子格（如 40×40→80 维）可 force_bkz=True 跑真 BKZ
    run_bkz = force_bkz or d <= 160
    if run_bkz:
        for _ in range(max(1, tours)):
            for bs in block_sizes:
                if bs < 2 or bs > d:
                    continue
                try:
                    BKZ.reduction(M, BKZ.Param(block_size=int(bs), auto_abort=True))
                except TypeError:
                    BKZ.reduction(M, BKZ.Param(block_size=int(bs)))

    out = np.zeros((d, d), dtype=np.int64)
    for i in range(d):
        for j in range(d):
            out[i, j] = int(M[i, j])
    return out


def _seeds_from_reduced_basis(
    R: np.ndarray,
    n: int,
    m: int,
    gamma: int,
    max_vectors: int,
    combo_depth: int,
    combo_coeff_max: int,
    seen: set,
    out: List[np.ndarray],
) -> None:
    basis_vs: List[np.ndarray] = []
    take = R.shape[1] if combo_depth <= 0 else min(R.shape[1], max(1, combo_depth))
    for j in range(take):
        v_part = np.empty(m, dtype=np.int64)
        for k in range(m):
            v_part[k] = int(R[n + k, j])
        basis_vs.append(v_part.copy())
        if _append_clipped_v(out, seen, v_part, gamma, max_vectors):
            return

    if combo_depth > 0 and len(basis_vs) >= 2:
        coeffs_range = range(-int(combo_coeff_max), int(combo_coeff_max) + 1)
        k = len(basis_vs)
        for coeffs in itertools.product(coeffs_range, repeat=k):
            if all(c == 0 for c in coeffs):
                continue
            combo = np.zeros(m, dtype=np.int64)
            for c, bv in zip(coeffs, basis_vs):
                if c:
                    combo += int(c) * bv
            if _append_clipped_v(out, seen, combo, gamma, max_vectors):
                return


def collect_bkz_v_seeds_fpylll(
    A: np.ndarray,
    q: int,
    gamma: int,
    beta: int,
    max_vectors: int,
    max_dim: int,
    combo_depth: int,
    combo_coeff_max: int,
    rng: np.random.Generator,
    *,
    bkz_tours: int = 2,
) -> List[np.ndarray]:
    """仅 fpylll 路径；失败返回 []。"""
    if not fpylll_available() or beta <= 0 or max_vectors <= 0:
        return []

    B, n, m = _build_ajtai_basis(A, q)
    d = n + m
    if d > max_dim:
        return []

    out: List[np.ndarray] = []
    seen: set = set()

    # 多 tour + 列置换：不同局部极小（Li–Nguyen：多种子优于单次精 BKZ）
    perm_count = min(3, max(1, max_vectors // 8))
    for t in range(perm_count):
        Bt = B.copy()
        if t > 0:
            perm = rng.permutation(d)
            Bt = Bt[:, perm]
        try:
            R = _fpylll_reduce_multi_tour(Bt, beta, tours=bkz_tours)
            if t > 0:
                inv = np.empty(d, dtype=np.int64)
                for j, p in enumerate(perm):
                    inv[int(p)] = j
                R = R[:, inv]
        except Exception:
            continue
        _seeds_from_reduced_basis(
            R, n, m, gamma, max_vectors, combo_depth, combo_coeff_max, seen, out
        )
        if len(out) >= max_vectors:
            return out[:max_vectors]

    return out[:max_vectors]


def collect_heuristic_lattice_seeds(
    A: np.ndarray,
    q: int,
    gamma: int,
    max_vectors: int,
    rng: np.random.Generator,
    combo_trials: int = 64,
) -> List[np.ndarray]:
    if max_vectors <= 0:
        return []
    A = np.mod(np.asarray(A, dtype=np.int64), q)
    _, m = A.shape
    out: List[np.ndarray] = []
    seen: set = set()

    for j in range(m):
        v_part = np.zeros(m, dtype=np.int64)
        v_part[j] = 1
        if _append_clipped_v(out, seen, v_part, gamma, max_vectors):
            return out

    for _ in range(combo_trials):
        k = int(rng.integers(2, min(10, m + 1)))
        idx = rng.choice(m, size=k, replace=False)
        coeffs = rng.integers(-3, 4, size=k, dtype=np.int64)
        if np.all(coeffs == 0):
            continue
        combo = np.zeros(m, dtype=np.int64)
        for c, j in zip(coeffs, idx):
            combo[j] = int(c)
        if _append_clipped_v(out, seen, combo, gamma, max_vectors):
            return out

    for _ in range(combo_trials // 2):
        v_part = rng.integers(-gamma, gamma + 1, size=m, dtype=np.int64)
        if np.all(v_part == 0):
            v_part[int(rng.integers(0, m))] = int(rng.choice([-1, 1]))
        if _append_clipped_v(out, seen, v_part, gamma, max_vectors):
            return out
    return out


def _collect_bkz_via_wsl(
    A: np.ndarray,
    q: int,
    gamma: int,
    beta: int,
    max_vectors: int,
    max_dim: int,
    combo_depth: int,
    combo_coeff_max: int,
    seed: int,
) -> List[np.ndarray]:
    """通过 WSL 调用 Linux 侧 fpylll（Windows 无本地 fpylll 时）。"""
    if os.environ.get("SIS_USE_WSL_BKZ", "").strip() not in ("1", "true", "yes"):
        return []
    try:
        subprocess.run(["wsl", "echo", "ok"], capture_output=True, check=True, timeout=5)
    except Exception:
        return []

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wsl_root = root.replace("\\", "/")
    if len(wsl_root) > 1 and wsl_root[1] == ":":
        wsl_root = f"/mnt/{wsl_root[0].lower()}{wsl_root[2:]}"

    payload = {
        "A": np.mod(A, q).tolist(),
        "q": int(q),
        "gamma": int(gamma),
        "beta": int(beta),
        "max_vectors": int(max_vectors),
        "max_dim": int(max_dim),
        "combo_depth": int(combo_depth),
        "combo_coeff_max": int(combo_coeff_max),
        "seed": int(seed),
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(payload, f)
        tmp = f.name
    wsl_tmp = tmp.replace("\\", "/")
    if len(wsl_tmp) > 1 and wsl_tmp[1] == ":":
        wsl_tmp = f"/mnt/{wsl_tmp[0].lower()}{wsl_tmp[2:]}"

    cmd = (
        f"cd {wsl_root}/scripts && python3 -c \""
        "import json, numpy as np; from lattice_seeds import collect_bkz_v_seeds_fpylll; "
        f"d=json.load(open('{wsl_tmp}')); A=np.array(d['A'],dtype=np.int64); "
        "rng=np.random.default_rng(d['seed']); "
        "vs=collect_bkz_v_seeds_fpylll(A,d['q'],d['gamma'],d['beta'],d['max_vectors'],"
        "d['max_dim'],d['combo_depth'],d['combo_coeff_max'],rng); "
        "print(json.dumps([v.tolist() for v in vs]))\""
    )
    try:
        r = subprocess.run(["wsl", "bash", "-lc", cmd], capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return []
        lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
        if not lines:
            return []
        data = json.loads(lines[-1])
        return [np.array(v, dtype=np.int64) for v in data]
    except Exception:
        return []
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def collect_bkz_v_seeds(
    A: np.ndarray,
    q: int,
    gamma: int,
    beta: int,
    max_vectors: int,
    max_dim: int,
    combo_depth: int = 0,
    combo_coeff_max: int = 2,
    rng: Optional[np.random.Generator] = None,
    *,
    bkz_tours: int = 2,
) -> List[np.ndarray]:
    """
    统一入口：fpylll 真 BKZ →（可选）WSL fpylll → 启发式回退。
    """
    if beta <= 0 or max_vectors <= 0:
        return []
    if rng is None:
        rng = np.random.default_rng(0)

    out = collect_bkz_v_seeds_fpylll(
        A, q, gamma, beta, max_vectors, max_dim, combo_depth, combo_coeff_max, rng, bkz_tours=bkz_tours
    )
    if out:
        return out

    out = _collect_bkz_via_wsl(
        A, q, gamma, beta, max_vectors, max_dim, combo_depth, combo_coeff_max, int(rng.integers(0, 2**31))
    )
    if out:
        return out

    return collect_heuristic_lattice_seeds(A, q, gamma, max_vectors, rng)

# ===== lattice_g6k.py =====

_G6K_OK: Optional[bool] = None


def g6k_available() -> bool:
    global _G6K_OK
    if _G6K_OK is not None:
        return _G6K_OK
    if not fpylll_available():
        _G6K_OK = False
        return False
    try:
        from g6k import Siever  # noqa: F401

        _G6K_OK = True
    except ImportError:
        _G6K_OK = False
    return _G6K_OK


def lattice_sieve_backend_label() -> str:
    if g6k_available():
        return "g6k_bdgl2"
    if fpylll_available():
        return "fpylll_bkz+list_sieve"
    return "heuristic"


def _default_threads() -> int:
    try:
        return max(1, min(8, (os.cpu_count() or 4)))
    except Exception:
        return 2


def _sieve_window(d: int) -> Tuple[int, int]:
    """返回 (r0, r1) 供 initialize_local：高维只在尾块筛。"""
    if d <= 96:
        return 0, d
    tail = min(80, max(48, d // 3))
    return d - tail, d


def _coeffs_to_list(coeffs) -> List[int]:
    if coeffs is None:
        return []
    if hasattr(coeffs, "coeffs"):
        coeffs = coeffs.coeffs
    if hasattr(coeffs, "to_list"):
        return [int(x) for x in coeffs.to_list()]
    if hasattr(coeffs, "__iter__") and not isinstance(coeffs, (str, bytes)):
        return [int(c) for c in coeffs]
    return [int(coeffs)]


def _basis_for_g6k(g6k):
    if hasattr(g6k, "M") and g6k.M is not None and hasattr(g6k.M, "B"):
        return g6k.M.B
    if hasattr(g6k, "B"):
        return g6k.B
    raise AttributeError("cannot find basis on g6k object")


def _basis_to_numpy(B, d: int) -> np.ndarray:
    out = np.zeros((d, d), dtype=np.int64)
    for i in range(d):
        for j in range(d):
            out[i, j] = int(B[i, j])
    return out


def _lattice_vector_from_coeffs_np(B_np: np.ndarray, cl: List[int]) -> np.ndarray:
    d = B_np.shape[0]
    if len(cl) < d:
        cl = cl + [0] * (d - len(cl))
    elif len(cl) > d:
        cl = cl[:d]
    return (B_np @ np.asarray(cl, dtype=np.int64)).astype(np.int64, copy=False)


def _lattice_vector_from_g6k(g6k, coeffs, d: int, B_np: np.ndarray) -> Optional[np.ndarray]:
    cl = _coeffs_to_list(coeffs)
    if not cl:
        return None
    try:
        w = _basis_for_g6k(g6k).multiply_left(cl)
        return np.array([int(w[i]) for i in range(d)], dtype=np.int64)
    except Exception:
        return _lattice_vector_from_coeffs_np(B_np, cl)


def _extract_vectors_from_g6k(
    g6k,
    d: int,
    B_np: np.ndarray,
    *,
    max_take: int,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    seen: set = set()

    def _push_w(w: np.ndarray) -> None:
        if len(out) >= max_take or not np.any(w):
            return
        key = w.tobytes()
        if key in seen:
            return
        seen.add(key)
        out.append(w.copy())

    def _push_coeffs(coeffs) -> None:
        w = _lattice_vector_from_g6k(g6k, coeffs, d, B_np)
        if w is not None:
            _push_w(w)

    # best_lifts
    try:
        for lift in g6k.best_lifts():
            if len(out) >= max_take:
                break
            if len(lift) >= 3:
                _push_coeffs(lift[2])
    except Exception:
        pass

    # 数据库
    try:
        for i in range(min(len(g6k), max_take * 8)):
            if len(out) >= max_take:
                break
            try:
                _push_coeffs(g6k.db[i])
            except Exception:
                continue
    except Exception:
        pass

    # 约化基列（保底，筛法未饱和时必有）
    for j in range(min(d, max_take * 2)):
        if len(out) >= max_take:
            break
        col = B_np[:, j]
        if np.any(col):
            _push_w(col)

    return out


def _make_siever(
    B_int,
    *,
    threads: int,
    saturation_ratio: float,
    sieve_alg: str,
    bkz_block: int,
) -> Tuple[object, int, np.ndarray]:
    from fpylll import BKZ, LLL

    d = B_int.nrows
    LLL.reduction(B_int)
    bs = min(d, max(20, int(bkz_block)))
    try:
        BKZ.reduction(B_int, BKZ.Param(block_size=bs, auto_abort=False))
    except TypeError:
        BKZ.reduction(B_int, BKZ.Param(block_size=bs))

    try:
        from fpylll import GSO

        G = GSO.Mat(B_int, float_type="double")
        from g6k import Siever

        g6k = Siever(G)
    except Exception:
        from g6k import Siever

        g6k = Siever(B_int)

    B_np = _basis_to_numpy(_basis_for_g6k(g6k), d)

    try:
        from g6k.siever_params import SieverParams

        sp = SieverParams()
        sp["threads"] = int(threads)
        sp["saturation_ratio"] = float(saturation_ratio)
        sp["saturation_radius"] = 4.0 / 3.0
        sp["otf_lift"] = True
        sp["sample_by_sums"] = True
        g6k.params = sp
    except Exception:
        pass

    r0, r1 = _sieve_window(d)
    g6k.initialize_local(r0, r0, r1)
    alg = sieve_alg if sieve_alg in ("bdgl2", "bdgl", "gauss") else "bdgl2"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            g6k(alg=alg)
        except Exception as exc:
            if "Saturation" not in type(exc).__name__:
                try:
                    g6k(alg="gauss")
                except Exception:
                    pass
    return g6k, d, B_np


def collect_g6k_lattice_vectors(
    A: np.ndarray,
    q: int,
    beta: int,
    max_vectors: int,
    max_dim: int,
    rng: np.random.Generator,
    *,
    sieve_alg: str = "bdgl2",
    saturation_ratio: float = 0.92,
    threads: Optional[int] = None,
    bkz_block: Optional[int] = None,
) -> List[np.ndarray]:
    if not g6k_available() or max_vectors <= 0:
        return []

    B, n, m = _build_ajtai_basis(A, q)
    d = n + m
    if d > max_dim:
        return []

    th = threads if threads is not None else _default_threads()
    try:
        B_int = _integer_matrix_from_basis(B)
        g6k, d_work, B_np = _make_siever(
            B_int,
            threads=th,
            saturation_ratio=saturation_ratio,
            sieve_alg=sieve_alg,
            bkz_block=bkz_block or beta,
        )
        return _extract_vectors_from_g6k(
            g6k, d_work, B_np, max_take=max_vectors
        )[:max_vectors]
    except Exception:
        return []


def collect_g6k_v_seeds(
    A: np.ndarray,
    q: int,
    gamma: int,
    beta: int,
    max_vectors: int,
    max_dim: int,
    rng: np.random.Generator,
    *,
    sieve_alg: str = "bdgl2",
    saturation_ratio: float = 0.92,
    threads: Optional[int] = None,
    bkz_block: Optional[int] = None,
) -> List[np.ndarray]:
    if max_vectors <= 0:
        return []
    _, n, m = _build_ajtai_basis(A, q)
    raw = collect_g6k_lattice_vectors(
        A,
        q,
        beta,
        max(max_vectors * 2, 32),
        max_dim,
        rng,
        sieve_alg=sieve_alg,
        saturation_ratio=saturation_ratio,
        threads=threads,
        bkz_block=bkz_block or beta,
    )
    out: List[np.ndarray] = []
    seen: set = set()
    for w in raw:
        v_part = w[n : n + m]
        if _append_clipped_v(out, seen, v_part, gamma, max_vectors):
            break
    if out:
        return out
    return collect_bkz_v_seeds(
        A, q, gamma, beta, max_vectors, max_dim, 4, 2, rng
    )

# ===== lattice_kannan.py =====

def _build_kannan_embed(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    embedding_factor: int,
) -> tuple[np.ndarray, int, int]:
    B, n, m = _build_ajtai_basis(A, q)
    d = n + m
    M = int(embedding_factor)
    E = np.zeros((d + 1, d + 1), dtype=np.int64)
    E[:d, :d] = B
    tc = np.zeros(d, dtype=np.int64)
    half = q // 2
    for i in range(n):
        x = int(t[i]) % q
        tc[i] = x - q if x > half else x
    E[d, :d] = tc
    E[d, d] = M
    return E, n, m


def collect_kannan_v_seeds(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    beta: int,
    max_vectors: int,
    max_dim: int,
    rng: np.random.Generator,
    *,
    embedding_factor: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Kannan 嵌入 + BKZ → v 种子（仅非齐次题使用）。
    """
    if max_vectors <= 0 or beta <= 0:
        return []
    if bool(np.all(np.mod(t, q) == 0)):
        return []
    if not fpylll_available():
        return []

    B, n, m = _build_ajtai_basis(A, q)
    d = n + m
    if d + 1 > max_dim + 1:
        return []

    if embedding_factor is None:
        embedding_factor = max(gamma * max(4, int(np.sqrt(d))), gamma * 8)

    out: List[np.ndarray] = []
    seen: set = set()

    for trial in range(min(3, max(1, max_vectors // 8))):
        try:
            E, n, m = _build_kannan_embed(A, t, q, embedding_factor + trial * gamma)
            if trial > 0:
                perm = rng.permutation(E.shape[0])
                E = E[:, perm]
            R = _fpylll_reduce_multi_tour(E, min(beta, E.shape[0]), tours=1, force_bkz=d <= 120)
            for j in range(min(R.shape[1], max_vectors * 2)):
                col = R[:, j]
                last = int(col[-1])
                if last == 0:
                    continue
                scale = 1 if last > 0 else -1
                body = scale * col[:-1]
                v_part = body[n : n + m].astype(np.int64, copy=False)
                if _append_clipped_v(out, seen, v_part, gamma, max_vectors):
                    return out[:max_vectors]
        except Exception:
            continue
    return out[:max_vectors]

# ===== lattice_sieve.py =====


def _list_sieve_short_vectors(
    R: np.ndarray,
    n: int,
    m: int,
    *,
    pool_cap: int = 64,
    max_combos: int = 2000,
    coeff_max: int = 2,
) -> List[np.ndarray]:
    d = R.shape[1]
    take = min(d, max(8, n + m // 4))
    cols = [R[:, j].copy() for j in range(take)]
    pool: List[np.ndarray] = []
    seen: Set[bytes] = set()

    def add_col(col: np.ndarray) -> None:
        v_part = col[n : n + m].astype(np.int64, copy=False)
        key = v_part.tobytes()
        if key in seen:
            return
        seen.add(key)
        pool.append(v_part.copy())
        if len(pool) > pool_cap:
            pool.sort(key=lambda v: int(np.max(np.abs(v))))
            del pool[pool_cap:]

    for c in cols:
        add_col(c)

    trials = 0
    k = min(len(cols), 12)
    cr = range(-coeff_max, coeff_max + 1)
    for coeffs in itertools.product(cr, repeat=k):
        if all(c == 0 for c in coeffs):
            continue
        combo = np.zeros_like(cols[0], dtype=np.int64)
        for c, col in zip(coeffs, cols[:k]):
            if c:
                combo += int(c) * col
        add_col(combo)
        trials += 1
        if trials >= max_combos:
            break
    return pool


def collect_sieve_v_seeds(
    A: np.ndarray,
    q: int,
    gamma: int,
    beta: int,
    max_vectors: int,
    max_dim: int,
    rng: np.random.Generator,
    *,
    combo_depth: int = 6,
    combo_coeff_max: int = 2,
    sieve_pool: int = 48,
    use_g6k: bool = False,
    g6k_sieve_alg: str = "bdgl2",
    g6k_saturation_ratio: float = 0.92,
    g6k_threads: Optional[int] = None,
    g6k_bkz_block: Optional[int] = None,
    g6k_max_lift_vectors: int = 512,
) -> List[np.ndarray]:
    """
    BKZ 2.0 + 筛法 → v 种子。``use_g6k=True`` 时走 G6K BDGL2 真筛法。
    """
    if max_vectors <= 0 or beta <= 0:
        return []

    out: List[np.ndarray] = []
    seen: set = set()

    if use_g6k:
        try:

            if g6k_available():
                g6k_vs = collect_g6k_v_seeds(
                    A,
                    q,
                    gamma,
                    beta,
                    min(max_vectors, g6k_max_lift_vectors),
                    max_dim,
                    rng,
                    sieve_alg=g6k_sieve_alg,
                    saturation_ratio=g6k_saturation_ratio,
                    threads=g6k_threads,
                    bkz_block=g6k_bkz_block or beta,
                )
                for v in g6k_vs:
                    if _append_clipped_v(out, seen, v, gamma, max_vectors):
                        return out[:max_vectors]
        except Exception:
            pass

    B, n, m = _build_ajtai_basis(A, q)
    d = n + m
    if d > max_dim:
        return collect_bkz_v_seeds(
            A, q, gamma, beta, max_vectors, max_dim, combo_depth, combo_coeff_max, rng
        )

    if fpylll_available():
        perm_count = min(2, max(1, max_vectors // 12))
        for t in range(perm_count):
            Bt = B.copy()
            if t > 0:
                perm = rng.permutation(d)
                Bt = Bt[:, perm]
            try:
                R = _fpylll_reduce_multi_tour(Bt, beta, tours=2)
                if t > 0:
                    inv = np.empty(d, dtype=np.int64)
                    for j, p in enumerate(perm):
                        inv[int(p)] = j
                    R = R[:, inv]
            except Exception:
                continue
            _seeds_from_reduced_basis(
                R, n, m, gamma, max_vectors, combo_depth, combo_coeff_max, seen, out
            )
            pool_cap = max(sieve_pool, 128 if use_g6k else sieve_pool)
            max_combos = 8000 if use_g6k else 1500
            for v_part in _list_sieve_short_vectors(
                R, n, m, pool_cap=pool_cap, max_combos=max_combos, coeff_max=combo_coeff_max
            ):
                if _append_clipped_v(out, seen, v_part, gamma, max_vectors):
                    return out[:max_vectors]
            if len(out) >= max_vectors:
                return out[:max_vectors]

    if out:
        return out[:max_vectors]
    return collect_bkz_v_seeds(
        A, q, gamma, beta, max_vectors, max_dim, combo_depth, combo_coeff_max, rng
    )

# ===== lattice_restricted_svp.py =====


def _center_mod(x: np.ndarray, q: int) -> np.ndarray:
    y = np.mod(x, q)
    half = q // 2
    y = np.where(y > half, y - q, y)
    return y.astype(np.int64, copy=False)


def _objective_uv(u: np.ndarray, v: np.ndarray, gamma: int) -> Tuple[int, int, int]:
    abs_r = np.abs(u)
    abs_v = np.abs(v)
    ou = np.maximum(abs_r - gamma, 0)
    ov = np.maximum(abs_v - gamma, 0)
    viol = int(np.count_nonzero(ou) + np.count_nonzero(ov))
    overflow_sum = int(np.sum(ou) + np.sum(ov))
    max_u = int(np.max(ou)) if ou.size else 0
    max_v = int(np.max(ov)) if ov.size else 0
    return viol, overflow_sum, max(max_u, max_v)


def _score_key(score: Tuple[int, int, int]) -> Tuple[int, int, int]:
    viol, overflow_sum, max_overflow = score
    return max_overflow, viol, overflow_sum

def _v_from_lattice_column(R: np.ndarray, n: int, m: int, col: int) -> np.ndarray:
    return R[n : n + m, col].astype(np.int64, copy=False)


def _combo_v_from_tail(
    R: np.ndarray,
    n: int,
    m: int,
    tail_indices: Sequence[int],
    coeffs: Sequence[int],
) -> np.ndarray:
    v = np.zeros(m, dtype=np.int64)
    for c, j in zip(coeffs, tail_indices):
        if c:
            col = R[:, int(j)].astype(np.int64, copy=False)
            v += int(c) * col[n : n + m]
    return v


def _enumerate_approx_svp_list(
    R: np.ndarray,
    n: int,
    m: int,
    *,
    tail_rank: int,
    coeff_max: int,
    pool_size: int,
    max_trials: int,
) -> List[Tuple[int, np.ndarray]]:
    """
    Wang「enumerate」阶段：在 BKZ 约化基尾块上枚举小系数组合，按 L₂ 排序保留池。
    返回 [(l2_sq, v), ...]。
    """
    d = R.shape[1]
    if d == 0 or pool_size <= 0:
        return []

    tail_rank = max(2, min(int(tail_rank), d))
    tail_indices = list(range(d - tail_rank, d))

    # 列 L₂ 范数升序，优先短基向量（剪枝启发）
    col_norms = [int(np.dot(R[:, j], R[:, j])) for j in tail_indices]
    order = sorted(range(len(tail_indices)), key=lambda i: col_norms[i])
    tail_indices = [tail_indices[i] for i in order]

    pool: List[Tuple[int, np.ndarray]] = []
    seen: set = set()
    trials = 0
    cr = range(-int(coeff_max), int(coeff_max) + 1)

    def _push(v: np.ndarray) -> None:
        nonlocal pool
        key = v.tobytes()
        if key in seen:
            return
        seen.add(key)
        l2 = int(np.dot(v, v))
        pool.append((l2, v.copy()))
        if len(pool) > pool_size:
            pool.sort(key=lambda x: x[0])
            del pool[pool_size:]

    # 单基向量
    for j in tail_indices:
        _push(_v_from_lattice_column(R, n, m, j))
        trials += 1

    k_enum = min(len(tail_indices), max(4, tail_rank // 2))
    active = tail_indices[:k_enum]
    for coeffs in itertools.product(cr, repeat=len(active)):
        if all(c == 0 for c in coeffs):
            continue
        v = _combo_v_from_tail(R, n, m, active, coeffs)
        _push(v)
        trials += 1
        if trials >= max_trials:
            break

    pool.sort(key=lambda x: x[0])
    return pool


def _slice_restricted(
    v_list: Sequence[np.ndarray],
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    *,
    require_norm_lt_q2: bool,
    homogeneous: bool,
    allow_norm_fallback: bool = True,
) -> List[Tuple[Tuple[int, int, int], int, np.ndarray]]:
    """
    Wang「slice」阶段：按 L∞（经残差）与可选 L₂ 上界过滤，返回 (score_key, norm_sq, v)。
  若 require_norm_lt_q2 且无达标候选，可回退到 L₂ 最小者作种子（局部搜索再抛光）。
    """
    q2 = q * q
    ok: List[Tuple[Tuple[int, int, int], int, np.ndarray]] = []
    all_scored: List[Tuple[Tuple[int, int, int], int, np.ndarray]] = []
    for v in v_list:
        if homogeneous and np.all(v == 0):
            continue
        u = _center_mod(t - A @ v, q)
        ns = int(np.dot(u, u) + np.dot(v, v))
        sk = _score_key(_objective_uv(u, v, gamma))
        all_scored.append((sk, ns, v.copy()))
        if not require_norm_lt_q2 or ns < q2:
            ok.append((sk, ns, v.copy()))
    pool = ok if ok else (all_scored if allow_norm_fallback else [])
    pool.sort(key=lambda x: (x[0], x[1]))
    return pool


def _dimension_for_free_enumerate(
    A: np.ndarray,
    q: int,
    gamma: int,
    beta: int,
    rng: np.random.Generator,
    *,
    max_dim: int,
    tail_rank: int,
    coeff_max: int,
    pool_size: int,
    max_trials: int,
    tours: int = 2,
) -> List[np.ndarray]:
    """
    BKZ 预处理 + 尾块 enumerate（dimension-for-free 实用版）。
    """
    if not fpylll_available() or beta <= 0:
        return []

    B, n, m = _build_ajtai_basis(A, q)
    d = n + m
    if d > max_dim:
        return []

    raw_vs: List[np.ndarray] = []
    perm_count = min(3, max(1, pool_size // 64))
    for t in range(perm_count):
        Bt = B.copy()
        if t > 0:
            perm = rng.permutation(d)
            Bt = Bt[:, perm]
        try:
            R = _fpylll_reduce_multi_tour(Bt, beta, tours=tours)
            if t > 0:
                inv = np.empty(d, dtype=np.int64)
                for j, p in enumerate(perm):
                    inv[int(p)] = j
                R = R[:, inv]
        except Exception:
            continue
        for _l2, v in _enumerate_approx_svp_list(
            R,
            n,
            m,
            tail_rank=tail_rank,
            coeff_max=coeff_max,
            pool_size=pool_size,
            max_trials=max_trials,
        ):
            raw_vs.append(v)

    return raw_vs


def wang_restricted_svp_v_seeds(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    rng: np.random.Generator,
    max_vectors: int,
    *,
    beta: int = 24,
    max_dim: int = 160,
    tail_rank: int = 28,
    coeff_max: int = 3,
    enum_pool_size: int = 512,
    enum_max_trials: int = 8000,
    require_norm_lt_q2: bool = False,
    use_g6k_enumerate: bool = False,
    g6k_sieve_alg: str = "bdgl2",
    g6k_saturation_ratio: float = 0.92,
    g6k_threads: Optional[int] = None,
    g6k_bkz_block: Optional[int] = None,
    g6k_max_lift_vectors: int = 512,
) -> List[np.ndarray]:
    """
    Wang 受限 SVP 主入口：enumerate（d4f + BKZ）→ slice → 取优 v 种子。
    """
    if max_vectors <= 0:
        return []

    A = np.mod(A, q).astype(np.int64, copy=False)
    t = np.mod(t, q).astype(np.int64, copy=False)
    homogeneous = bool(np.all(t == 0))
    n, m = A.shape[0], A.shape[1]

    raw_vs: List[np.ndarray] = []

    # Wang enumerate 阶段：G6K 近似 SVP 列表（论文推荐，优于单 L₂ 最短后过滤）
    if use_g6k_enumerate:
        try:

            if g6k_available():
                lifts = collect_g6k_lattice_vectors(
                    A,
                    q,
                    beta,
                    min(g6k_max_lift_vectors, enum_pool_size),
                    max_dim,
                    rng,
                    sieve_alg=g6k_sieve_alg,
                    saturation_ratio=g6k_saturation_ratio,
                    threads=g6k_threads,
                    bkz_block=g6k_bkz_block or beta,
                )
                for w in lifts:
                    raw_vs.append(w[n : n + m].astype(np.int64, copy=False))
        except Exception:
            pass

    raw_vs.extend(
        _dimension_for_free_enumerate(
            A,
            q,
            gamma,
            beta,
            rng,
            max_dim=max_dim,
            tail_rank=tail_rank,
            coeff_max=coeff_max,
            pool_size=enum_pool_size,
            max_trials=enum_max_trials,
        )
    )

    if not raw_vs:
        # 无 fpylll：启发式基向量枚举 + 盒采样仍走 slice
        _, m = A.shape
        raw_vs = collect_heuristic_lattice_seeds(A, q, gamma, enum_pool_size // 4, rng)

    sliced = _slice_restricted(
        raw_vs,
        A,
        t,
        q,
        gamma,
        require_norm_lt_q2=require_norm_lt_q2,
        homogeneous=homogeneous,
    )

    out: List[np.ndarray] = []
    seen: set = set()
    for _, _, v in sliced:
        if _append_clipped_v(out, seen, v, gamma, max_vectors):
            break
    return out


def _sparse_v_samples(
    m: int,
    gamma: int,
    rng: np.random.Generator,
    n_samples: int,
    *,
    max_nnz: int,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for _ in range(n_samples):
        k = int(rng.integers(1, min(max_nnz, m) + 1))
        idx = rng.choice(m, size=k, replace=False)
        v = np.zeros(m, dtype=np.int64)
        vals = rng.integers(-min(3, gamma), min(3, gamma) + 1, size=k, dtype=np.int64)
        vals[vals == 0] = 1
        v[idx] = vals
        out.append(v)
    return out


def _box_random_samples(
    m: int,
    gamma: int,
    rng: np.random.Generator,
    n_samples: int,
) -> List[np.ndarray]:
    return [rng.integers(-gamma, gamma + 1, size=m, dtype=np.int64) for _ in range(n_samples)]


def collect_restricted_svp_v_seeds(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    rng: np.random.Generator,
    max_vectors: int,
    *,
    beta: int = 24,
    max_dim: int = 160,
    tail_rank: int = 28,
    coeff_max: int = 3,
    enum_pool_size: int = 512,
    enum_max_trials: int = 8000,
    n_random: int = 400,
    n_sparse: int = 200,
    max_nnz: int = 12,
    require_norm_lt_q2: bool = False,
    use_wang_pipeline: bool = True,
    use_g6k_enumerate: bool = False,
    g6k_sieve_alg: str = "bdgl2",
    g6k_saturation_ratio: float = 0.92,
    g6k_threads: Optional[int] = None,
    g6k_bkz_block: Optional[int] = None,
    g6k_max_lift_vectors: int = 512,
) -> List[np.ndarray]:
    """
    受限 SVP 种子收集：默认 Wang enumerate-then-slice；辅以盒内稀疏/随机 slice 增广。
    """
    if max_vectors <= 0:
        return []

    out: List[np.ndarray] = []
    seen: set = set()

    if use_wang_pipeline:
        wang_vs = wang_restricted_svp_v_seeds(
            A,
            t,
            q,
            gamma,
            rng,
            max_vectors,
            beta=beta,
            max_dim=max_dim,
            tail_rank=tail_rank,
            coeff_max=coeff_max,
            enum_pool_size=enum_pool_size,
            enum_max_trials=enum_max_trials,
            require_norm_lt_q2=require_norm_lt_q2,
            use_g6k_enumerate=use_g6k_enumerate,
            g6k_sieve_alg=g6k_sieve_alg,
            g6k_saturation_ratio=g6k_saturation_ratio,
            g6k_threads=g6k_threads,
            g6k_bkz_block=g6k_bkz_block,
            g6k_max_lift_vectors=g6k_max_lift_vectors,
        )
        for v in wang_vs:
            if _append_clipped_v(out, seen, v, gamma, max_vectors):
                return out

    # 增广：稀疏采样优先（第三类 L₂ 友好），再盒内随机，统一 slice
    _, m = A.shape
    extra = _sparse_v_samples(m, gamma, rng, max(n_sparse, 80), max_nnz=min(max_nnz, 8))
    extra.extend(_box_random_samples(m, gamma, rng, min(n_random, 120)))
    homogeneous = bool(np.all(np.mod(t, q) == 0))
    sliced_extra = _slice_restricted(
        extra,
        A,
        t,
        q,
        gamma,
        require_norm_lt_q2=require_norm_lt_q2,
        homogeneous=homogeneous,
    )
    for _, _, v in sliced_extra:
        if _append_clipped_v(out, seen, v, gamma, max_vectors):
            break

    return out

# ===== sis_heuristics =====

Score3 = Tuple[int, int, int]
ObjectiveFn = Callable[[np.ndarray, np.ndarray, int], Score3]
BetterFn = Callable[[Score3, Score3], bool]


def _score_uv(residual: np.ndarray, v: np.ndarray, gamma: int) -> Score3:
    abs_r = np.abs(residual)
    abs_v = np.abs(v)
    ou = np.maximum(abs_r - gamma, 0)
    ov = np.maximum(abs_v - gamma, 0)
    viol = int(np.count_nonzero(ou) + np.count_nonzero(ov))
    overflow_sum = int(np.sum(ou) + np.sum(ov))
    max_overflow = int(max(np.max(ou) if ou.size else 0, np.max(ov) if ov.size else 0))
    return viol, overflow_sum, max_overflow


def estimate_delta(A: np.ndarray, n: int, m: int, q: int, n_samples: int = 40) -> int:
    """随机满秩子矩阵行列式绝对值的最大估计（L∞ SVP 门槛文献中的 Δ 代理）。"""
    if n <= 0 or m < n:
        return 1
    A = np.mod(np.asarray(A, dtype=np.int64), q)
    rng = np.random.default_rng(0)
    deltas: List[int] = []
    for _ in range(n_samples):
        cols = rng.choice(m, size=n, replace=False)
        sub = A[:, cols].astype(np.float64)
        try:
            det_f = abs(float(np.linalg.det(sub)))
            if not np.isfinite(det_f) or det_f <= 0:
                continue
            det = int(min(det_f, 10**15))
        except Exception:
            continue
        deltas.append(det)
    return max(deltas) if deltas else 1


def _small_vectors(dim: int, radius: int, cap: int, rng: np.random.Generator) -> List[np.ndarray]:
    """[-radius,radius]^dim 枚举或随机抽样。"""
    if dim <= 0:
        return [np.zeros(0, dtype=np.int64)]
    total = (2 * radius + 1) ** dim
    if total <= cap:
        out = []
        for coeffs in itertools.product(range(-radius, radius + 1), repeat=dim):
            out.append(np.asarray(coeffs, dtype=np.int64))
        return out
    out = []
    seen: set = set()
    while len(out) < cap:
        v = rng.integers(-radius, radius + 1, size=dim, dtype=np.int64)
        key = v.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _reference_u_rows(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    rng: np.random.Generator,
    n_rows: int,
) -> np.ndarray:
    """用非平凡 probe 得到参考 u，以便选出真正“最差”的行（齐次 t=0 时 v=0 无效）。"""
    n, m = A.shape
    probe = np.zeros(m, dtype=np.int64)
    for _ in range(max(4, m // 20)):
        probe[int(rng.integers(0, m))] = int(rng.integers(-gamma, gamma + 1))
    if np.all(probe == 0):
        probe[0] = 1
    u0 = center_mod(t - (A @ probe), q)
    abs_u = np.abs(u0)
    bad = np.flatnonzero(abs_u > gamma)
    if bad.size > 0:
        order = bad[np.argsort(-abs_u[bad])]
    else:
        order = np.argsort(-abs_u)
    return order[: min(n_rows, n)]


def _wagner_mitm_seeds(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    row_idx: np.ndarray,
    cols: np.ndarray,
    box_radius: int,
    max_seeds: int,
    seen_v: set,
) -> List[np.ndarray]:
    """4+4 列全枚举 meet-in-the-middle（mod q 精确碰撞）。"""
    m = A.shape[1]
    if cols.size < 4:
        return []
    half = min(4, cols.size // 2)
    J1, J2 = cols[:half], cols[half : 2 * half]
    if J1.size == 0 or J2.size == 0:
        return []
    A1 = A[row_idx][:, J1] % q
    A2 = A[row_idx][:, J2] % q
    target = (t[row_idx] % q).astype(np.int64)
    dim = len(J1)
    if (2 * box_radius + 1) ** (2 * dim) > 120_000:
        return []

    seeds: List[np.ndarray] = []
    table: Dict[Tuple[int, ...], np.ndarray] = {}
    for coeffs in itertools.product(range(-box_radius, box_radius + 1), repeat=dim):
        v1 = np.asarray(coeffs, dtype=np.int64)
        r = tuple(int(x) for x in ((A1 @ v1) % q))
        table[r] = v1.copy()

    for coeffs in itertools.product(range(-box_radius, box_radius + 1), repeat=len(J2)):
        v2 = np.asarray(coeffs, dtype=np.int64)
        r2 = (A2 @ v2) % q
        need = tuple(int((target[i] - r2[i]) % q) for i in range(len(target)))
        v1 = table.get(need)
        if v1 is None:
            continue
        v_full = np.zeros(m, dtype=np.int64)
        for jj, cj in enumerate(J1):
            v_full[int(cj)] = int(v1[jj])
        for jj, cj in enumerate(J2):
            v_full[int(cj)] = int(v2[jj])
        v_full = np.clip(v_full, -gamma, gamma)
        key = v_full.tobytes()
        if key in seen_v:
            continue
        seen_v.add(key)
        seeds.append(v_full.copy())
        if len(seeds) >= max_seeds:
            break
    return seeds


def _subsystem_bruteforce_seeds(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    row_idx: np.ndarray,
    cols: np.ndarray,
    box_radius: int,
    max_seeds: int,
    seen_v: set,
    objective_uv: ObjectiveFn,
) -> List[np.ndarray]:
    """在 top 列小子空间暴力枚举，按全局 L∞ 分数选种（Wagner 无碰撞时的后备）。"""
    m = A.shape[1]
    ncol = min(5, cols.size)
    if ncol <= 0:
        return []
    use_cols = cols[:ncol]
    r = min(box_radius, gamma)
    scored: List[Tuple[Tuple[int, int, int], np.ndarray]] = []
    for coeffs in itertools.product(range(-r, r + 1), repeat=ncol):
        v_full = np.zeros(m, dtype=np.int64)
        for jj, cj in enumerate(use_cols):
            v_full[int(cj)] = int(coeffs[jj])
        if np.all(v_full == 0):
            continue
        res = center_mod(t - (A @ v_full), q)
        sc = objective_uv(res, v_full, gamma)
        scored.append((sc, v_full.copy()))
    scored.sort(key=lambda x: (x[0][2], x[0][0], x[0][1]))  # max_overflow, violations, sum
    out: List[np.ndarray] = []
    for sc, v_full in scored[: max_seeds * 4]:
        key = v_full.tobytes()
        if key in seen_v:
            continue
        seen_v.add(key)
        out.append(v_full)
        if len(out) >= max_seeds:
            break
    return out


def wagner_subsystem_seeds(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    rng: np.random.Generator,
    *,
    n_rows: int = 8,
    n_cols: int = 16,
    box_radius: int = 2,
    list_cap: int = 600,
    max_seeds: int = 12,
    objective_uv: Optional[ObjectiveFn] = None,
) -> List[np.ndarray]:
    """
    最差 u 行子系统种子：先 4+4 Wagner 全枚举，不足则 5 列小子空间暴力枚举。
    """
    obj = objective_uv or _score_uv
    n, m = A.shape
    row_idx = _reference_u_rows(A, t, q, gamma, rng, n_rows)
    if row_idx.size == 0:
        return []

    col_score = np.sum(np.abs(A[row_idx, :]), axis=0)
    cols = np.argsort(-col_score)[: min(n_cols, m)]
    if cols.size < 2:
        return []

    seen_v: set = set()
    seeds: List[np.ndarray] = []

    for br in (box_radius, min(box_radius + 1, gamma)):
        seeds.extend(
            _wagner_mitm_seeds(A, t, q, gamma, row_idx, cols, br, max_seeds - len(seeds), seen_v)
        )
        if len(seeds) >= max_seeds:
            return seeds[:max_seeds]

    seeds.extend(
        _subsystem_bruteforce_seeds(
            A, t, q, gamma, row_idx, cols, box_radius, max_seeds - len(seeds), seen_v, obj
        )
    )
    return seeds[:max_seeds]


def discrete_gaussian_seeds(
    v_center: np.ndarray,
    gamma: int,
    rng: np.random.Generator,
    *,
    n_seeds: int = 12,
    sigma: float = 3.0,
) -> List[np.ndarray]:
    """以当前最优 v 为中心的离散 Gaussian 抖动种子。"""
    v_center = np.asarray(v_center, dtype=np.int64).ravel()
    m = v_center.size
    seeds: List[np.ndarray] = []
    seen: set = set()
    sig = max(1.0, float(sigma))
    for _ in range(n_seeds * 3):
        if len(seeds) >= n_seeds:
            break
        noise = np.rint(rng.normal(0, sig, size=m)).astype(np.int64)
        v_new = np.clip(v_center + noise, -gamma, gamma)
        if np.all(v_new == 0):
            continue
        key = v_new.tobytes()
        if key in seen:
            continue
        seen.add(key)
        seeds.append(v_new.copy())
    return seeds


def violation_ls_step(
    A: np.ndarray,
    residual: np.ndarray,
    v: np.ndarray,
    cols: List[np.ndarray],
    q: int,
    gamma: int,
    score: Score3,
    rng: np.random.Generator,
    *,
    top_rows: int = 6,
    top_cols: int = 8,
    joint_radius: int = 3,
    better_score: BetterFn,
    objective_uv: ObjectiveFn,
) -> Optional[Tuple[np.ndarray, np.ndarray, int, Score3, float]]:
    """
    ViolationLS：对 u 超界最严重的若干行，在梯度最大的多列上联合网格搜索 Δv。
    """
    abs_r = np.abs(residual)
    bad = np.flatnonzero(abs_r > gamma)
    if bad.size == 0:
        return None
    viol_rows = bad[np.argsort(-abs_r[bad])[: min(top_rows, bad.size)]]
    grad = (A[viol_rows, :].T @ np.sign(residual[viol_rows]).astype(np.float64)).astype(np.float64)
    col_order = np.argsort(-np.abs(grad))[: min(top_cols, len(grad))]
    if col_order.size == 0:
        return None

    best_score = score
    best_pack: Optional[Tuple[np.ndarray, np.ndarray, int, Score3, float]] = None
    vv_sq = int(np.dot(v.astype(np.int64), v.astype(np.int64)))
    r = max(1, min(joint_radius, gamma))

    # 限制组合：随机抽若干对/三元组列做联合移动
    ncol = int(col_order.size)
    trials = min(48, ncol * (ncol - 1) // 2)
    pairs: List[Tuple[int, int]] = []
    if ncol >= 2:
        all_pairs = [(int(col_order[i]), int(col_order[j])) for i in range(ncol) for j in range(i + 1, ncol)]
        if len(all_pairs) <= trials:
            pairs = all_pairs
        else:
            idx = rng.choice(len(all_pairs), size=trials, replace=False)
            pairs = [all_pairs[int(i)] for i in idx]

    for j, k in pairs:
        cj, ck = int(v[j]), int(v[k])
        col_j, col_k = cols[j], cols[k]
        for dj in range(-r, r + 1):
            for dk in range(-r, r + 1):
                if dj == 0 and dk == 0:
                    continue
                nj, nk = cj + dj, ck + dk
                if abs(nj) > gamma or abs(nk) > gamma:
                    continue
                cand_res = center_mod(residual - col_j * dj - col_k * dk, q)
                v[j], v[k] = nj, nk
                try:
                    cv, cosum, cmaxov = objective_uv(cand_res, v, gamma)
                finally:
                    v[j], v[k] = cj, ck
                cand_score = (cv, cosum, cmaxov)
                if better_score(cand_score, best_score):
                    best_score = cand_score
                    new_vv = vv_sq - cj * cj - ck * ck + nj * nj + nk * nk
                    rr_c = float(np.dot(cand_res.astype(np.float64), cand_res.astype(np.float64)))
                    best_pack = (cand_res.copy(), j, k, nj, nk, cand_score, new_vv, rr_c)

    if best_pack is None:
        return None
    cand_res, j, k, nj, nk, new_score, new_vv, rr_c = best_pack
    v[j], v[k] = nj, nk
    return cand_res, v, new_vv, new_score, rr_c


def layered_row_projection(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    residual: np.ndarray,
    v: np.ndarray,
    score: Score3,
    *,
    n_layers: int = 3,
    better_score: BetterFn,
    objective_uv_and_rr_sq: Callable[..., Tuple[int, int, int, float]],
) -> Optional[Tuple[np.ndarray, np.ndarray, int, Score3, float]]:
    """由最差 u 行开始分层 LS 投影，逐层缩小关注行集。"""
    abs_r = np.abs(residual)
    bad = np.flatnonzero(abs_r > gamma)
    if bad.size == 0:
        return None
    remaining = list(bad[np.argsort(-abs_r[bad])])
    v_work = v.copy()
    res_work = residual.copy()
    sc = score
    vv_sq = int(np.dot(v_work.astype(np.int64), v_work.astype(np.int64)))
    rr_sync = float(np.dot(res_work.astype(np.float64), res_work.astype(np.float64)))

    for layer in range(n_layers):
        if not remaining:
            break
        k = max(1, len(remaining) // (2**layer))
        focus = np.asarray(remaining[:k], dtype=np.int64)
        col_score = np.sum(np.abs(A[focus, :]), axis=0)
        take = min(28, A.shape[1])
        cols_idx = np.argsort(-col_score)[:take]
        AR = A[np.ix_(focus, cols_idx)].astype(np.float64)
        b = res_work[focus].astype(np.float64)
        delta, *_ = np.linalg.lstsq(AR, b, rcond=1e-8)
        improved_layer = False
        for scale in (1.0, 0.55, 0.28):
            di = np.rint(delta * scale).astype(np.int64)
            cand_v = v_work.copy()
            ok = True
            for jj, cj in enumerate(cols_idx):
                nv = int(cand_v[int(cj)]) + int(di[jj])
                if abs(nv) > gamma:
                    ok = False
                    break
                cand_v[int(cj)] = nv
            if not ok:
                continue
            cand_res = center_mod(t - (A @ cand_v), q)
            cv, cosum, cmaxov, rr_c = objective_uv_and_rr_sq(cand_res, cand_v, gamma)
            cand_sc = (cv, cosum, cmaxov)
            if better_score(cand_sc, sc):
                v_work = cand_v
                res_work = cand_res
                sc = cand_sc
                vv_sq = int(np.dot(v_work.astype(np.int64), v_work.astype(np.int64)))
                rr_sync = rr_c
                improved_layer = True
                break
        if improved_layer:
            abs_new = np.abs(res_work)
            remaining = [int(r) for r in remaining if abs_new[int(r)] > gamma]
        else:
            break

    if sc == score:
        return None
    return res_work, v_work, vv_sq, sc, rr_sync


def cheap_pair_reduction(
    v: np.ndarray,
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    score: Score3,
    rng: np.random.Generator,
    *,
    n_trials: int = 40,
    better_score: BetterFn,
    objective_uv: ObjectiveFn,
) -> Tuple[np.ndarray, Score3]:
    """轻量列组合归约（递归格思想的零阶近似）。"""
    v = v.copy()
    res = center_mod(t - (A @ v), q)
    sc = score
    m = v.size
    for _ in range(n_trials):
        j, k = int(rng.integers(0, m)), int(rng.integers(0, m))
        if j == k:
            continue
        for coeff in (-1, 1):
            cand = v.copy()
            cand[j] = int(np.clip(cand[j] + coeff * cand[k], -gamma, gamma))
            cand_res = center_mod(t - (A @ cand), q)
            cand_sc = objective_uv(cand_res, cand, gamma)
            if better_score(cand_sc, sc):
                v, res, sc = cand, cand_res, cand_sc
    return v, sc

# ===== sis_finish =====

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


if __name__ == "__main__":
    main()
