"""
G6K 真筛法（BDGL2 / Gauss）接入 — Becker et al. SODA 2016 实用实现。

依赖（Linux 服务器，源码安装）::
    git clone https://github.com/fplll/g6k.git
    cd g6k && pip install -r requirements.txt
    python setup.py build_ext --inplace && pip install -e .

与 fpylll BKZ 2.0 组合：先 LLL+BKZ 预处理 Ajtai 基，再 ``g6k(alg='bdgl2')``，
从筛法库 / best_lifts 提取短向量后 m 维为 v 种子。
"""

from __future__ import annotations

import os
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
        return max(1, min(32, (os.cpu_count() or 4)))
    except Exception:
        return 4


def _coeffs_to_list(coeffs) -> List[int]:
    if hasattr(coeffs, "__iter__") and not isinstance(coeffs, (str, bytes)):
        return [int(c) for c in coeffs]
    return [int(coeffs)]


def _lattice_vector_from_coeffs(B, coeffs, d: int) -> np.ndarray:
    """从系数向量得到格向量 w ∈ Z^d。"""
    cl = _coeffs_to_list(coeffs)
    try:
        w = B.multiply_left(cl)
        return np.array([int(w[i]) for i in range(d)], dtype=np.int64)
    except Exception:
        pass
    w = np.zeros(d, dtype=np.int64)
    for j, c in enumerate(cl):
        if c:
            for i in range(d):
                w[i] += int(c) * int(B[i, j])
    return w


def _make_siever(B_int, *, threads: int, saturation_ratio: float, sieve_alg: str):
    from fpylll import BKZ, LLL

    d = B_int.nrows
    LLL.reduction(B_int)
    bs = min(d, max(40, d // 2))
    try:
        BKZ.reduction(B_int, BKZ.Param(block_size=bs, auto_abort=False))
    except TypeError:
        BKZ.reduction(B_int, BKZ.Param(block_size=bs))

    g6k = None
    try:
        from fpylll import GSO

        G = GSO.Mat(
            B_int,
            float_type="double",
        )
        from g6k import Siever

        g6k = Siever(G)
    except Exception:
        from g6k import Siever

        g6k = Siever(B_int)

    try:
        from g6k.siever_params import SieverParams

        sp = SieverParams()
        sp["threads"] = int(threads)
        sp["saturation_ratio"] = float(saturation_ratio)
        sp["saturation_radius"] = 4.0 / 3.0
        sp["otf_lift"] = True
        sp["sample_by_sums"] = True
        if sieve_alg in ("bdgl2", "bdgl"):
            sp["sieve"] = "bdgl2"
        g6k.params = sp
    except Exception:
        pass

    g6k.initialize_local(0, 0, d)
    try:
        g6k(alg=sieve_alg if sieve_alg else "bdgl2")
    except Exception as exc:
        # SaturationError 表示筛法达到饱和，属正常结束
        if "Saturation" not in type(exc).__name__:
            try:
                g6k(alg="gauss")
            except Exception:
                pass
    return g6k, B_int, d


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
    """
    对 Ajtai 格运行 G6K 筛法，返回完整格向量 w（长度 d=n+m）。
    """
    if not g6k_available() or max_vectors <= 0:
        return []

    B, n, m = _build_ajtai_basis(A, q)
    d = n + m
    if d > max_dim:
        return []

    th = threads if threads is not None else _default_threads()
    out: List[np.ndarray] = []
    seen: set = set()

    trials = min(3, max(1, max_vectors // 64))
    for t in range(trials):
        Bt = B.copy()
        if t > 0:
            perm = rng.permutation(d)
            Bt = Bt[:, perm]
        try:
            B_int = _integer_matrix_from_basis(Bt)
            if bkz_block and bkz_block > 0:
                from fpylll import BKZ, LLL

                LLL.reduction(B_int)
                bs = min(d, int(bkz_block))
                try:
                    BKZ.reduction(B_int, BKZ.Param(block_size=bs, auto_abort=False))
                except TypeError:
                    BKZ.reduction(B_int, BKZ.Param(block_size=bs))
            g6k, B_work, d_work = _make_siever(
                B_int,
                threads=th,
                saturation_ratio=saturation_ratio,
                sieve_alg=sieve_alg,
            )
        except Exception:
            continue

        def _push_w(w: np.ndarray) -> bool:
            key = w.tobytes()
            if key in seen:
                return False
            seen.add(key)
            out.append(w.copy())
            return len(out) >= max_vectors

        # best_lifts
        try:
            for lift in g6k.best_lifts():
                if len(out) >= max_vectors:
                    break
                try:
                    coeffs = lift[2]
                    w = _lattice_vector_from_coeffs(B_work, coeffs, d_work)
                    if t > 0:
                        inv = np.empty(d, dtype=np.int64)
                        for j, p in enumerate(perm):
                            inv[int(p)] = j
                        w_perm = np.zeros(d, dtype=np.int64)
                        for j in range(d):
                            w_perm[inv[j]] = w[j]
                        w = w_perm
                    _push_w(w)
                except Exception:
                    continue
        except Exception:
            pass

        # 筛法数据库
        try:
            for item in g6k.itervalues():
                if len(out) >= max_vectors:
                    break
                try:
                    if isinstance(item, tuple) and len(item) >= 2:
                        idx = int(item[0])
                        coeffs = g6k.db[idx]
                    else:
                        coeffs = item
                    w = _lattice_vector_from_coeffs(B_work, coeffs, d_work)
                    _push_w(w)
                except Exception:
                    continue
        except Exception:
            pass

        if len(out) >= max_vectors:
            break

    return out[:max_vectors]


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
    """G6K 筛法 → 裁剪到 [-γ,γ]^m 的 v 种子。"""
    if max_vectors <= 0:
        return []
    B, n, m = _build_ajtai_basis(A, q)
    raw = collect_g6k_lattice_vectors(
        A,
        q,
        beta,
        max(max_vectors * 4, 128),
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
    return out
