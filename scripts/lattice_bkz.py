"""
齐次 SIS 的 Ajtai 格 + 真 BKZ/LLL（fpylll）。

论文对应（Li–Nguyen 2025 等）：
- 用 LLL 预处理 + 多轮 BKZ（不同 block_size），「弱 oracle、多种子」比单次高 β 更划算；
- 约化基每列的后 m 坐标 → v 种子，再小系数组合扩充。

无 fpylll 时：
- 可选 WSL 子进程（环境变量 SIS_USE_WSL_BKZ=1）；
- 否则回退 collect_heuristic_lattice_seeds。
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
        "import json, numpy as np; from lattice_bkz import collect_bkz_v_seeds_fpylll; "
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
