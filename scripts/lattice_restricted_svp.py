"""
Wang et al. (PQCrypto 2025) 受限 SVP 启发式 — 全量实现骨架。

论文核心（restricted SVP / ePrint 2025/586）：
  1. 近似 SVP **列表**生成（enumerate-then-slice 的 enumerate 阶段），
     而非仅取 L₂ 最短向量再二次过滤；
  2. **slice** 阶段按额外限制（L∞ 盒、欧氏上界等）筛选；
  3. **dimension for free**：在 BKZ 约化基的尾块子格上枚举，降低有效维数。

本模块将 Ajtai 格短向量后 m 维视为 v 候选，经残差 u=center(t-Av) 做 slice。
"""

from __future__ import annotations

import itertools
from typing import List, Sequence, Tuple

import numpy as np

from lattice_bkz import (
    _append_clipped_v,
    _build_ajtai_basis,
    _fpylll_reduce_multi_tour,
    collect_heuristic_lattice_seeds,
    fpylll_available,
)


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
) -> List[np.ndarray]:
    """
    Wang 受限 SVP 主入口：enumerate（d4f + BKZ）→ slice → 取优 v 种子。
    """
    if max_vectors <= 0:
        return []

    A = np.mod(A, q).astype(np.int64, copy=False)
    t = np.mod(t, q).astype(np.int64, copy=False)
    homogeneous = bool(np.all(t == 0))

    raw_vs = _dimension_for_free_enumerate(
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
