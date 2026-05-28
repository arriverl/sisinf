"""
Right kernel of A over Z/qZ: integer columns d with (A @ d) % q == 0.

Used for 'kernel walk': v <- clip(v + d) leaves Av mod q unchanged, hence
u = center(t - A v) unchanged — only v's box / l_inf can be adjusted.

Gaussian elimination with modular inverses requires **unit pivots** (gcd(pivot,q)=1),
which fails for typical composite moduli (e.g. q=100). We therefore prefer the
rational nullspace of B = [A | -q I]: vectors [d; w] with A d = q w, i.e. Ad ≡ 0 (mod q).
For **prime** q, a fast unit-pivot Gaussian elimination is used (field case).
For **composite** q (typical challenge moduli), sympy on `[A | -q I]` is used when installed;
otherwise an empty basis is returned (kernel walk disabled).
"""

from __future__ import annotations

from typing import List

import numpy as np


def _is_prime_trial(n: int) -> bool:
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
    """Symmetric mod q into about (-q/2, q/2] for each coordinate."""
    xi = np.asarray(x, dtype=np.int64).ravel()
    y = np.mod(xi, q).astype(np.int64)
    half = q // 2
    y = np.where(y > half, y - q, y)
    return y


def _kernel_via_augmented_nullspace(A: np.ndarray, q: int, max_basis: int) -> np.ndarray:
    """Kernel from nullspace of [A | -q I] over Q; requires sympy."""
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
    """Unit-pivot elimination; correct when Z/qZ is a field (q prime)."""
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
    Return K with shape (m, k), k <= max_basis, each column d satisfies (A @ d) % q == 0.
    Uses sympy on [A | -q I] for composite (and general) q; Gauss fallback only for prime q.
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
    r = (A @ d.astype(np.int64)) % q
    return bool(np.all(r == 0))
