"""
G6K 真筛法（BDGL2 / Gauss）+ 投影筛法（高维尾块 d4f）。

d > 90 时仅在最后 ``sieve_dim`` 维子格上跑 bdgl2，再从约化基列 / best_lifts 提取 v。
"""

from __future__ import annotations

import os
import warnings
from typing import List, Optional, Tuple

import numpy as np

from lattice_bkz import (
    _append_clipped_v,
    _build_ajtai_basis,
    _integer_matrix_from_basis,
    fpylll_available,
)

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
    # 筛法库空时回退 BKZ 基列
    try:
        from lattice_bkz import collect_bkz_v_seeds

        return collect_bkz_v_seeds(
            A, q, gamma, beta, max_vectors, max_dim, 4, 2, rng
        )
    except Exception:
        return []
