"""
格种子统一模块：BKZ / G6K / sieve / Kannan / Wang restricted SVP。
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# ===== lattice_bkz.py =====

_FPYLLL_OK: Optional[bool] = None


def fpylll_available() -> bool:
    """是否能在本进程 import fpylll。"""
    global _FPYLLL_OK
    if _FPYLLL_OK is not None:
        return _FPYLLL_OK
    try:
        from fpylll import BKZ, IntegerMatrix, LLL  # noqa: F401

        _FPYLLL_OK = True
    except ImportError:
        _FPYLLL_OK = False
    return _FPYLLL_OK


def lattice_backend_label() -> str:
    if fpylll_available():
        return "fpylll"
    if os.environ.get("SIS_USE_WSL_BKZ", "").strip() in ("1", "true", "yes"):
        return "wsl_fpylll_or_heuristic"
    return "heuristic"


def _build_ajtai_basis(A: np.ndarray, q: int) -> Tuple[np.ndarray, int, int]:
    A = np.mod(np.asarray(A, dtype=np.int64), q)
    n, m = A.shape
    d = n + m
    B = np.zeros((d, d), dtype=np.int64)
    for j in range(n):
        B[j, j] = q
    for jj in range(m):
        col = n + jj
        B[:n, col] = -A[:, jj]
        B[n + jj, col] = 1
    return B, n, m


def _append_clipped_v(
    out: List[np.ndarray],
    seen: set,
    v_part: np.ndarray,
    gamma: int,
    max_vectors: int,
) -> bool:
    for cand in (v_part, -v_part):
        clip = np.clip(cand, -gamma, gamma).astype(np.int64, copy=False)
        key = clip.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(clip.copy())
        if len(out) >= max_vectors:
            return True
    return False


def _integer_matrix_from_basis(B: np.ndarray):
    from fpylll import IntegerMatrix

    d = B.shape[0]
    M = IntegerMatrix(d, d)
    for i in range(d):
        for j in range(d):
            M[i, j] = int(B[i, j])
    return M


def _fpylll_reduce_multi_tour(
    B: np.ndarray,
    beta: int,
    *,
    block_sizes: Optional[List[int]] = None,
    tours: int = 2,
    force_bkz: bool = False,
) -> np.ndarray:
    """
    LLL + 多 block_size 的 BKZ（近似 oracle 多种子策略）。

    返回约化后的整数基矩阵 (d,d) 的 numpy 副本（列向量为格基）。
    """
    from fpylll import BKZ, LLL

    d = B.shape[0]
    M = _integer_matrix_from_basis(B)
    LLL.reduction(M)

    if block_sizes is None:
        bs0 = max(2, min(int(beta), d))
        block_sizes = sorted(
            set(
                [
                    max(2, bs0 - 4),
                    bs0,
                    min(d, bs0 + 4),
                    min(d, bs0 + 8),
                ]
            )
        )

    # n+m≈200 时全量 BKZ 极慢；>160 维仅 LLL（仍比纯启发式强）
    # 小子格（如 40×40→80 维）可 force_bkz=True 跑真 BKZ
    run_bkz = force_bkz or d <= 160
    if run_bkz:
        for _ in range(max(1, tours)):
            for bs in block_sizes:
                if bs < 2 or bs > d:
                    continue
                try:
                    BKZ.reduction(M, BKZ.Param(block_size=int(bs), auto_abort=True))
                except TypeError:
                    BKZ.reduction(M, BKZ.Param(block_size=int(bs)))

    out = np.zeros((d, d), dtype=np.int64)
    for i in range(d):
        for j in range(d):
            out[i, j] = int(M[i, j])
    return out


def _seeds_from_reduced_basis(
    R: np.ndarray,
    n: int,
    m: int,
    gamma: int,
    max_vectors: int,
    combo_depth: int,
    combo_coeff_max: int,
    seen: set,
    out: List[np.ndarray],
) -> None:
    basis_vs: List[np.ndarray] = []
    take = R.shape[1] if combo_depth <= 0 else min(R.shape[1], max(1, combo_depth))
    for j in range(take):
        v_part = np.empty(m, dtype=np.int64)
        for k in range(m):
            v_part[k] = int(R[n + k, j])
        basis_vs.append(v_part.copy())
        if _append_clipped_v(out, seen, v_part, gamma, max_vectors):
            return

    if combo_depth > 0 and len(basis_vs) >= 2:
        coeffs_range = range(-int(combo_coeff_max), int(combo_coeff_max) + 1)
        k = len(basis_vs)
        for coeffs in itertools.product(coeffs_range, repeat=k):
            if all(c == 0 for c in coeffs):
                continue
            combo = np.zeros(m, dtype=np.int64)
            for c, bv in zip(coeffs, basis_vs):
                if c:
                    combo += int(c) * bv
            if _append_clipped_v(out, seen, combo, gamma, max_vectors):
                return


def collect_bkz_v_seeds_fpylll(
    A: np.ndarray,
    q: int,
    gamma: int,
    beta: int,
    max_vectors: int,
    max_dim: int,
    combo_depth: int,
    combo_coeff_max: int,
    rng: np.random.Generator,
    *,
    bkz_tours: int = 2,
) -> List[np.ndarray]:
    """仅 fpylll 路径；失败返回 []。"""
    if not fpylll_available() or beta <= 0 or max_vectors <= 0:
        return []

    B, n, m = _build_ajtai_basis(A, q)
    d = n + m
    if d > max_dim:
        return []

    out: List[np.ndarray] = []
    seen: set = set()

    # 多 tour + 列置换：不同局部极小（Li–Nguyen：多种子优于单次精 BKZ）
    perm_count = min(3, max(1, max_vectors // 8))
    for t in range(perm_count):
        Bt = B.copy()
        if t > 0:
            perm = rng.permutation(d)
            Bt = Bt[:, perm]
        try:
            R = _fpylll_reduce_multi_tour(Bt, beta, tours=bkz_tours)
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
        if len(out) >= max_vectors:
            return out[:max_vectors]

    return out[:max_vectors]


def collect_heuristic_lattice_seeds(
    A: np.ndarray,
    q: int,
    gamma: int,
    max_vectors: int,
    rng: np.random.Generator,
    combo_trials: int = 64,
) -> List[np.ndarray]:
    if max_vectors <= 0:
        return []
    A = np.mod(np.asarray(A, dtype=np.int64), q)
    _, m = A.shape
    out: List[np.ndarray] = []
    seen: set = set()

    for j in range(m):
        v_part = np.zeros(m, dtype=np.int64)
        v_part[j] = 1
        if _append_clipped_v(out, seen, v_part, gamma, max_vectors):
            return out

    for _ in range(combo_trials):
        k = int(rng.integers(2, min(10, m + 1)))
        idx = rng.choice(m, size=k, replace=False)
        coeffs = rng.integers(-3, 4, size=k, dtype=np.int64)
        if np.all(coeffs == 0):
            continue
        combo = np.zeros(m, dtype=np.int64)
        for c, j in zip(coeffs, idx):
            combo[j] = int(c)
        if _append_clipped_v(out, seen, combo, gamma, max_vectors):
            return out

    for _ in range(combo_trials // 2):
        v_part = rng.integers(-gamma, gamma + 1, size=m, dtype=np.int64)
        if np.all(v_part == 0):
            v_part[int(rng.integers(0, m))] = int(rng.choice([-1, 1]))
        if _append_clipped_v(out, seen, v_part, gamma, max_vectors):
            return out
    return out


def _collect_bkz_via_wsl(
    A: np.ndarray,
    q: int,
    gamma: int,
    beta: int,
    max_vectors: int,
    max_dim: int,
    combo_depth: int,
    combo_coeff_max: int,
    seed: int,
) -> List[np.ndarray]:
    """通过 WSL 调用 Linux 侧 fpylll（Windows 无本地 fpylll 时）。"""
    if os.environ.get("SIS_USE_WSL_BKZ", "").strip() not in ("1", "true", "yes"):
        return []
    try:
        subprocess.run(["wsl", "echo", "ok"], capture_output=True, check=True, timeout=5)
    except Exception:
        return []

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wsl_root = root.replace("\\", "/")
    if len(wsl_root) > 1 and wsl_root[1] == ":":
        wsl_root = f"/mnt/{wsl_root[0].lower()}{wsl_root[2:]}"

    payload = {
        "A": np.mod(A, q).tolist(),
        "q": int(q),
        "gamma": int(gamma),
        "beta": int(beta),
        "max_vectors": int(max_vectors),
        "max_dim": int(max_dim),
        "combo_depth": int(combo_depth),
        "combo_coeff_max": int(combo_coeff_max),
        "seed": int(seed),
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(payload, f)
        tmp = f.name
    wsl_tmp = tmp.replace("\\", "/")
    if len(wsl_tmp) > 1 and wsl_tmp[1] == ":":
        wsl_tmp = f"/mnt/{wsl_tmp[0].lower()}{wsl_tmp[2:]}"

    cmd = (
        f"cd {wsl_root}/scripts && python3 -c \""
        "import json, numpy as np; from lattice_seeds import collect_bkz_v_seeds_fpylll; "
        f"d=json.load(open('{wsl_tmp}')); A=np.array(d['A'],dtype=np.int64); "
        "rng=np.random.default_rng(d['seed']); "
        "vs=collect_bkz_v_seeds_fpylll(A,d['q'],d['gamma'],d['beta'],d['max_vectors'],"
        "d['max_dim'],d['combo_depth'],d['combo_coeff_max'],rng); "
        "print(json.dumps([v.tolist() for v in vs]))\""
    )
    try:
        r = subprocess.run(["wsl", "bash", "-lc", cmd], capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return []
        lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
        if not lines:
            return []
        data = json.loads(lines[-1])
        return [np.array(v, dtype=np.int64) for v in data]
    except Exception:
        return []
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def collect_bkz_v_seeds(
    A: np.ndarray,
    q: int,
    gamma: int,
    beta: int,
    max_vectors: int,
    max_dim: int,
    combo_depth: int = 0,
    combo_coeff_max: int = 2,
    rng: Optional[np.random.Generator] = None,
    *,
    bkz_tours: int = 2,
) -> List[np.ndarray]:
    """
    统一入口：fpylll 真 BKZ →（可选）WSL fpylll → 启发式回退。
    """
    if beta <= 0 or max_vectors <= 0:
        return []
    if rng is None:
        rng = np.random.default_rng(0)

    out = collect_bkz_v_seeds_fpylll(
        A, q, gamma, beta, max_vectors, max_dim, combo_depth, combo_coeff_max, rng, bkz_tours=bkz_tours
    )
    if out:
        return out

    out = _collect_bkz_via_wsl(
        A, q, gamma, beta, max_vectors, max_dim, combo_depth, combo_coeff_max, int(rng.integers(0, 2**31))
    )
    if out:
        return out

    return collect_heuristic_lattice_seeds(A, q, gamma, max_vectors, rng)

# ===== lattice_g6k.py =====

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
    return collect_bkz_v_seeds(
        A, q, gamma, beta, max_vectors, max_dim, 4, 2, rng
    )

# ===== lattice_kannan.py =====

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

# ===== lattice_sieve.py =====


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

# ===== lattice_restricted_svp.py =====


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
    use_g6k_enumerate: bool = False,
    g6k_sieve_alg: str = "bdgl2",
    g6k_saturation_ratio: float = 0.92,
    g6k_threads: Optional[int] = None,
    g6k_bkz_block: Optional[int] = None,
    g6k_max_lift_vectors: int = 512,
) -> List[np.ndarray]:
    """
    Wang 受限 SVP 主入口：enumerate（d4f + BKZ）→ slice → 取优 v 种子。
    """
    if max_vectors <= 0:
        return []

    A = np.mod(A, q).astype(np.int64, copy=False)
    t = np.mod(t, q).astype(np.int64, copy=False)
    homogeneous = bool(np.all(t == 0))
    n, m = A.shape[0], A.shape[1]

    raw_vs: List[np.ndarray] = []

    # Wang enumerate 阶段：G6K 近似 SVP 列表（论文推荐，优于单 L₂ 最短后过滤）
    if use_g6k_enumerate:
        try:

            if g6k_available():
                lifts = collect_g6k_lattice_vectors(
                    A,
                    q,
                    beta,
                    min(g6k_max_lift_vectors, enum_pool_size),
                    max_dim,
                    rng,
                    sieve_alg=g6k_sieve_alg,
                    saturation_ratio=g6k_saturation_ratio,
                    threads=g6k_threads,
                    bkz_block=g6k_bkz_block or beta,
                )
                for w in lifts:
                    raw_vs.append(w[n : n + m].astype(np.int64, copy=False))
        except Exception:
            pass

    raw_vs.extend(
        _dimension_for_free_enumerate(
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
    use_g6k_enumerate: bool = False,
    g6k_sieve_alg: str = "bdgl2",
    g6k_saturation_ratio: float = 0.92,
    g6k_threads: Optional[int] = None,
    g6k_bkz_block: Optional[int] = None,
    g6k_max_lift_vectors: int = 512,
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
            use_g6k_enumerate=use_g6k_enumerate,
            g6k_sieve_alg=g6k_sieve_alg,
            g6k_saturation_ratio=g6k_saturation_ratio,
            g6k_threads=g6k_threads,
            g6k_bkz_block=g6k_bkz_block,
            g6k_max_lift_vectors=g6k_max_lift_vectors,
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