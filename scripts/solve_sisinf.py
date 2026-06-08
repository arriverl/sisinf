"""
SIS∞ 2026 赛题一 — 主求解器（多 restart 局部搜索 + 格种子 + CP-SAT）。

问题
----
求 ``u ∈ Z^n``, ``v ∈ Z^m`` 使得 ``A v + u ≡ t (mod q)``，
且 ``|u|_∞, |v|_∞ ≤ γ``。第三类题另需 ``||u||_2^2 + ||v||_2^2 ≥ q^2``；
齐次 ``t≡0`` 时拒绝平凡解 ``u=v=0``。

核心变量
--------
- ``residual = center_mod(t - A v, q)`` 即同余意义下的 ``u``；
- 内层主要优化 ``v``，``u`` 由 residual 导出。

主要模块
--------
- ``SearchConfig`` / ``apply_sis_class_defaults``：搜索与三类题默认参数
- ``build_dual_space_candidates``：dual / pull / CVP / 随机盒种子
- ``lattice_bkz.collect_bkz_v_seeds``：第一类 BKZ 短向量种子（可选）
- ``modq_kernel``：kernel walk 的模 q 核基
- ``_single_restart_inner``：单 restart 主循环（坐标下降、CP、kernel、LS）
- ``local_search_one``：多 restart 编排（可 ProcessPool 并行）
- ``verify_solution``：与赛题一致的可行性校验

参见 ``sis_problem_taxonomy.py``、``run_class_batch.py``。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from modq_kernel import in_kernel_mod_q, right_kernel_basis_mod_q


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


def apply_sis_class_defaults(cfg: SearchConfig, sis_class: int, *, aggressive: bool = False) -> SearchConfig:
    """
    按赛题类别 1/2/3 叠加默认搜索参数（见 ``sis_problem_taxonomy.py``）。

    aggressive=True 时进一步增大 BKZ β、CVP 变体、块 CP 时间等（长跑轮次用）。
    """
    if sis_class == 1:
        beta = 32 if aggressive else 28
        return SearchConfig(
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
            }
        )
    if sis_class == 2:
        cvp = 14 if aggressive else 10
        pull = 10 if aggressive else 8
        return SearchConfig(
            **{
                **asdict(cfg),
                "use_bkz_seeds": False,
                "bkz_beta": 0,
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
            }
        )
    euclid = 4.0 if aggressive else 3.0
    return SearchConfig(
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
        }
    )


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
    require_norm_ge_q2: bool,
    cfg: SearchConfig,
    progress: float,
    topk_u_pen: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """Energy without recomputing objective (viol/overflow already known)."""
    euclid_gap = max(0.0, float(q * q) - norm_sq) if require_norm_ge_q2 else 0.0
    euclid_w, entropy_w = _schedule_weights(cfg, progress)
    cheby_w = cfg.cheby_weight
    if max_overflow >= cfg.cheby_boost_threshold:
        cheby_w *= cfg.cheby_boost_factor
    energy = (
        1_000_000.0 * viol
        + cheby_w * max_overflow
        + cfg.overflow_weight * overflow_sum
        + cfg.energy_topk_weight * topk_u_pen
        + euclid_w * euclid_gap
        - entropy_w * entropy
    )
    return energy, {
        "violations": float(viol),
        "overflow_sum": float(overflow_sum),
        "max_overflow": float(max_overflow),
        "norm_sq": norm_sq,
        "euclid_gap": euclid_gap,
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
    require_norm_ge_q2: bool,
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
        require_norm_ge_q2,
        cfg,
        progress,
        topk_u_pen=tk,
    )


def should_compute_entropy(
    cfg: SearchConfig,
    step: int,
    viol: int,
    require_norm_ge_q2: bool,
    progress: float,
) -> bool:
    if cfg.entropy_disable_after_progress < 1.0 and progress >= cfg.entropy_disable_after_progress:
        return False
    if cfg.entropy_weight == 0.0:
        return False
    if cfg.entropy_update_interval <= 0:
        return False
    # Respect interval even when viol==0 (otherwise inner loops compute histograms every delta).
    if require_norm_ge_q2 and viol == 0:
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
    require_norm_ge_q2: bool = False,
) -> Tuple[bool, Dict[str, int]]:
    """
    官方一致性校验：同余、L∞ 盒、可选欧氏下界、齐次非平凡。

    Returns
    -------
    ok : bool
    metrics : dict
        congruence_ok, inf_u, inf_v, norm_sq, norm_req_ok, nontrivial_ok（0/1）
    """
    lhs = (A @ v + u - t) % q
    congr_ok = bool(np.all(lhs == 0))
    inf_u = int(np.max(np.abs(u)))
    inf_v = int(np.max(np.abs(v)))
    inf_ok = inf_u <= gamma and inf_v <= gamma
    norm_sq = int(np.dot(u, u) + np.dot(v, v))
    norm_ok = True if not require_norm_ge_q2 else norm_sq >= q * q
    # For homogeneous SIS (t == 0), reject the trivial all-zero solution.
    is_homogeneous = bool(np.all(t % q == 0))
    nontrivial_ok = True if not is_homogeneous else bool(np.any(u != 0) or np.any(v != 0))
    ok = congr_ok and inf_ok and norm_ok and nontrivial_ok
    return ok, {
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
            from sis_advanced_u_ops import discrete_gaussian_seeds

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
    require_norm_ge_q2: bool,
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
        use_entropy = should_compute_entropy(cfg, step, score[0], require_norm_ge_q2, progress)
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
                require_norm_ge_q2,
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
                    require_norm_ge_q2,
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
                            require_norm_ge_q2,
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
                    require_norm_ge_q2,
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
                        require_norm_ge_q2,
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
            ok, metrics = verify_solution(A, t, q, gamma, u, v, require_norm_ge_q2)
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
                from sis_advanced_u_ops import violation_ls_step

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
                from sis_advanced_u_ops import layered_row_projection

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
                from sis_advanced_u_ops import cheap_pair_reduction

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
                    from sis_advanced_u_ops import discrete_gaussian_seeds

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
                ok2, met2 = verify_solution(A, t, q, gamma, u2, v2, require_norm_ge_q2)
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
                    ok_agg, met_agg = verify_solution(A, t, q, gamma, residual, v, require_norm_ge_q2)
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
                ok, metrics = verify_solution(A, t, q, gamma, u_rep, v_rep, require_norm_ge_q2)
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
    require_norm_ge_q2 = bool(payload["require_norm_ge_q2"])
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
        require_norm_ge_q2,
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
    require_norm_ge_q2: bool = False,
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
    if cfg.use_wagner_seeds:
        try:
            from sis_advanced_u_ops import wagner_subsystem_seeds

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
    if cfg.use_bkz_seeds and cfg.bkz_beta > 0:
        try:
            from lattice_bkz import collect_bkz_v_seeds, lattice_backend_label

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
            lattice_prepend.extend(bkz_seeds)
            lattice_backend = lattice_backend_label()
        except Exception:
            lattice_backend = "error"

    dual_candidates: List[np.ndarray] = []
    dual_meta: Dict[str, Any] = {
        "num_candidates": 0,
        "lattice_backend": lattice_backend,
        "lattice_seed_count": len(lattice_prepend),
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
                    "require_norm_ge_q2": require_norm_ge_q2,
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
                ok, metrics = verify_solution(A, t, q, gamma, u_arr, v_arr, require_norm_ge_q2)
                _, energy_meta = energy_score(
                    u_arr,
                    v_arr,
                    gamma,
                    q,
                    require_norm_ge_q2,
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
            require_norm_ge_q2,
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
            ok, metrics = verify_solution(A, t, q, gamma, u_out, v_out, require_norm_ge_q2)
            _, energy_meta = energy_score(
                u_out,
                v_out,
                gamma,
                q,
                require_norm_ge_q2,
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
            from sis_problem_taxonomy import (
                effective_require_norm_ge_q2,
                problem_class_from_instance,
            )

            sis_class = problem_class_from_instance(inst)
            require_norm_ge_q2 = effective_require_norm_ge_q2(inst, sis_class)
            local_cfg = apply_sis_class_defaults(cfg, sis_class)
        except Exception:
            sis_class = 0
            require_norm_ge_q2 = bool(inst.get("require_norm_ge_q2", False))
            local_cfg = cfg
        local_cfg = replace(local_cfg, seed=seed_base + idx)
        t0 = time.time()
        u, v, meta = local_search_one(
            A=A,
            t=t,
            q=q,
            gamma=gamma,
            cfg=local_cfg,
            require_norm_ge_q2=require_norm_ge_q2,
        )
        elapsed = time.time() - t0
        ok, verify = verify_solution(A, t, q, gamma, u, v, require_norm_ge_q2)

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
        help="Histogram entropy every k steps (<=0 disables). Denser schedule when require_norm_ge_q2 and viol==0.",
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


if __name__ == "__main__":
    main()
