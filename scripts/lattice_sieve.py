"""
BKZ 后缀的 Gauss / list 筛法启发式（BDGL 思路的轻量实现）。

在 BKZ 约化基上维护短向量池，对小整数系数组合再过滤，输出 v 种子。
不依赖 fpylll 内置 sieve（多数发行版无 BDGL API）。
"""

from __future__ import annotations

import itertools
from typing import List, Optional, Set

import numpy as np

from lattice_bkz import (
    _append_clipped_v,
    _build_ajtai_basis,
    _fpylll_reduce_multi_tour,
    _seeds_from_reduced_basis,
    collect_bkz_v_seeds,
    fpylll_available,
)


def _list_sieve_short_vectors(
    R: np.ndarray,
    n: int,
    m: int,
    *,
    pool_cap: int = 64,
    max_combos: int = 2000,
    coeff_max: int = 2,
) -> List[np.ndarray]:
    """
    对约化基列做小系数 Gauss 式合并，返回后 m 维（v 部分）候选。
    """
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
) -> List[np.ndarray]:
    """
    BKZ 2.0 约化 + list sieve 补充 → 裁剪到 [-γ,γ]^m 的 v 种子。
    """
    if max_vectors <= 0 or beta <= 0:
        return []
    B, n, m = _build_ajtai_basis(A, q)
    d = n + m
    if d > max_dim:
        return collect_bkz_v_seeds(
            A, q, gamma, beta, max_vectors, max_dim, combo_depth, combo_coeff_max, rng
        )

    out: List[np.ndarray] = []
    seen: set = set()

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
            for v_part in _list_sieve_short_vectors(
                R, n, m, pool_cap=sieve_pool, max_combos=1500, coeff_max=2
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
