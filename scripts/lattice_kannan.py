"""
Kannan 嵌入：将模 q 非齐次 CVP 归约为更高维 SVP，再 BKZ 约化提取 v 种子。

嵌入结构（维数 d+1，d=n+m）::

    [  Ajtai 基 B (d×d)     0  ]
    [  center(t) 扩展行      M  ]

短向量最后一维为 ±1 时，前 d 维给出接近目标的格点；取后 m 维为 v 候选。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from lattice_bkz import _append_clipped_v, _build_ajtai_basis, _fpylll_reduce_multi_tour, fpylll_available


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
