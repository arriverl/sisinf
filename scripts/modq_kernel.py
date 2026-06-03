"""
模 q 右核（kernel）基：满足 ``(A @ d) % q == 0`` 的整数列向量 d。

在 ``solve_sisinf`` 中的用途 — kernel walk
-----------------------------------------
固定 ``v`` 时 ``u = center(t - A v)`` 由同余类唯一决定。
若 ``d`` 在核中，则 ``v' = clip(v + d)`` 仍满足 ``A v' ≡ A v (mod q)``，
故 ``u`` 不变，仅改变 ``v`` 的 L∞ 盒内位置，用于在可行邻域内微调 ``v``。

为何不能只用高斯消元
--------------------
在 Z/qZ 上求逆要求主元与 q 互素（``gcd(pivot, q)=1``）。
赛题合数模（如 q=100）上大量主元不可逆，``pow(a,-1,q)`` 会失败。

策略
----
- **q 为素数**：在 Z/qZ 域上做单位主元高斯消元（快速）。
- **q 为合数**：优先用 SymPy 对增广矩阵 ``[A | -q I]`` 求有理零空间；
  零空间向量 ``[d; w]`` 满足 ``A d = q w``，即 ``A d ≡ 0 (mod q)``。
  将 d 分量先 ``% q`` 再转 int64，避免 SymPy 大整数溢出。
- 无 SymPy 且非素数时返回空基（kernel walk 自动关闭）。
"""

from __future__ import annotations

from typing import List

import numpy as np


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
