"""
受限 SVP 启发式（Wang et al. PQCrypto 2025 思路的实用实现）。

在 L∞ 盒 [-γ,γ]^m 上直接采样/枚举稀疏 v，经残差 u=center(t-Av) 评分；
第三类同时过滤 ``||u||_2^2+||v||_2^2 >= q^2``，避免「L₂ 短向量后过滤」路线。
"""

from __future__ import annotations

from typing import List

import numpy as np


def _center_mod(x: np.ndarray, q: int) -> np.ndarray:
    y = np.mod(x, q)
    half = q // 2
    y = np.where(y > half, y - q, y)
    return y.astype(np.int64, copy=False)


def _objective_uv(u: np.ndarray, v: np.ndarray, gamma: int) -> tuple[int, int, int]:
    abs_r = np.abs(u)
    abs_v = np.abs(v)
    ou = np.maximum(abs_r - gamma, 0)
    ov = np.maximum(abs_v - gamma, 0)
    viol = int(np.count_nonzero(ou) + np.count_nonzero(ov))
    overflow_sum = int(np.sum(ou) + np.sum(ov))
    max_u = int(np.max(ou)) if ou.size else 0
    max_v = int(np.max(ov)) if ov.size else 0
    return viol, overflow_sum, max(max_u, max_v)


def _score_key(score: tuple[int, int, int]) -> tuple[int, int, int]:
    viol, overflow_sum, max_overflow = score
    return max_overflow, viol, overflow_sum


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


def collect_restricted_svp_v_seeds(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    rng: np.random.Generator,
    max_vectors: int,
    *,
    n_random: int = 400,
    n_sparse: int = 200,
    max_nnz: int = 12,
    require_norm_lt_q2: bool = False,
) -> List[np.ndarray]:
    """
    受限搜索：L∞ 盒内随机 + 稀疏枚举，按 (V,S,M) 排序取优。
    """
    if max_vectors <= 0:
        return []
    _, m = A.shape
    A = np.mod(A, q).astype(np.int64, copy=False)
    t = np.mod(t, q).astype(np.int64, copy=False)
    q2 = q * q
    homogeneous = bool(np.all(t == 0))

    candidates: List[np.ndarray] = []
    candidates.extend(_sparse_v_samples(m, gamma, rng, n_sparse, max_nnz=max_nnz))
    for _ in range(n_random):
        v = rng.integers(-gamma, gamma + 1, size=m, dtype=np.int64)
        candidates.append(v)

    scored_all = []
    scored_ok = []
    for v in candidates:
        if homogeneous and np.all(v == 0):
            continue
        u = _center_mod(t - A @ v, q)
        ns = int(np.dot(u, u) + np.dot(v, v))
        viol, osum, mov = _objective_uv(u, v, gamma)
        sk = _score_key((viol, osum, mov))
        scored_all.append((sk, ns, v))
        if not require_norm_lt_q2 or ns < q2:
            scored_ok.append((sk, ns, v))

    pool = scored_ok if scored_ok else scored_all
    pool.sort(key=lambda x: (x[0], x[1]))
    out: List[np.ndarray] = []
    seen: set = set()
    for _, _, v in pool:
        key = v.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(v.copy())
        if len(out) >= max_vectors:
            break
    return out
