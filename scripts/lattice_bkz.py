"""
Optional BKZ-style lattice seeds for homogeneous SIS.

Lattice for pairs (u, v) with u in Z^n, v in Z^m and u + A v ≡ 0 (mod q):

    Columns j = 0..n-1 : (q e_j, 0)
    Columns j = n..n+m-1 : (-A[:, j'], e_j')   where j' = j - n

Any integer combination yields u + A v ∈ q Z^n (embedding vector). The trailing v-part of
short reduced columns gives heuristic seeds for local search (always clipped to ±gamma).

Requires package ``fpylll`` (Linux/macOS/WSL typical); if missing or on reduction failure,
returns an empty list (solver falls back to heuristic dual-space seeds only).
"""

from __future__ import annotations

from typing import List

import numpy as np


def collect_bkz_v_seeds(
    A: np.ndarray,
    q: int,
    gamma: int,
    beta: int,
    max_vectors: int,
    max_dim: int,
) -> List[np.ndarray]:
    if beta <= 0 or max_vectors <= 0:
        return []
    A = np.mod(np.asarray(A, dtype=np.int64), q)
    n, m = A.shape
    d = n + m
    if d > max_dim:
        return []

    try:
        from fpylll import BKZ, IntegerMatrix, LLL
    except ImportError:
        return []

    B = np.zeros((d, d), dtype=np.int64)
    for j in range(n):
        B[j, j] = q
    for jj in range(m):
        col = n + jj
        B[:n, col] = -A[:, jj]
        B[n + jj, col] = 1

    M = IntegerMatrix(d, d)
    for i in range(d):
        for j in range(d):
            M[i, j] = int(B[i, j])

    bs = max(2, min(int(beta), d))
    try:
        LLL.reduction(M)
        BKZ.reduction(M, BKZ.Param(block_size=bs))
    except Exception:
        try:
            LLL.reduction(M)
        except Exception:
            return []

    out: List[np.ndarray] = []
    seen: set = set()
    for j in range(d):
        v_part = np.empty(m, dtype=np.int64)
        for k in range(m):
            v_part[k] = int(M[n + k, j])
        for cand in (v_part, -v_part):
            clip = np.clip(cand, -gamma, gamma).astype(np.int64, copy=False)
            key = clip.tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append(clip.copy())
            if len(out) >= max_vectors:
                return out
    return out
