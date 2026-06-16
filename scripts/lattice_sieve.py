"""
BKZ 2.0 + 筛法种子：优先 G6K BDGL2（真筛法），回退 list sieve 启发式。

Chen–Nguyen BKZ 2.0 预处理 + Becker et al. BDGL 筛法（经 G6K ``alg='bdgl2'``）。
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
            from lattice_g6k import collect_g6k_v_seeds, g6k_available

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
