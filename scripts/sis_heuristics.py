"""
第一类 u 优先启发式算子：Wagner、ViolationLS、分层投影、Gaussian 种子等。
"""

"""
文献驱动的 u 优先算子（第一类齐次 SIS∞）。

- Wagner 式子系统列表合并（最差 u 行）
- ViolationLS 多列协同步
- 分层行投影 LS
- 离散 Gaussian 抖动种子
- Δ 估计（剪枝/诊断）
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

Score3 = Tuple[int, int, int]
ObjectiveFn = Callable[[np.ndarray, np.ndarray, int], Score3]
BetterFn = Callable[[Score3, Score3], bool]


def _score_uv(residual: np.ndarray, v: np.ndarray, gamma: int) -> Score3:
    abs_r = np.abs(residual)
    abs_v = np.abs(v)
    ou = np.maximum(abs_r - gamma, 0)
    ov = np.maximum(abs_v - gamma, 0)
    viol = int(np.count_nonzero(ou) + np.count_nonzero(ov))
    overflow_sum = int(np.sum(ou) + np.sum(ov))
    max_overflow = int(max(np.max(ou) if ou.size else 0, np.max(ov) if ov.size else 0))
    return viol, overflow_sum, max_overflow


def center_mod(x: np.ndarray, q: int) -> np.ndarray:
    y = np.mod(x, q)
    half = q // 2
    y = np.where(y > half, y - q, y)
    return y.astype(np.int64, copy=False)


def estimate_delta(A: np.ndarray, n: int, m: int, q: int, n_samples: int = 40) -> int:
    """随机满秩子矩阵行列式绝对值的最大估计（L∞ SVP 门槛文献中的 Δ 代理）。"""
    if n <= 0 or m < n:
        return 1
    A = np.mod(np.asarray(A, dtype=np.int64), q)
    rng = np.random.default_rng(0)
    deltas: List[int] = []
    for _ in range(n_samples):
        cols = rng.choice(m, size=n, replace=False)
        sub = A[:, cols].astype(np.float64)
        try:
            det_f = abs(float(np.linalg.det(sub)))
            if not np.isfinite(det_f) or det_f <= 0:
                continue
            det = int(min(det_f, 10**15))
        except Exception:
            continue
        deltas.append(det)
    return max(deltas) if deltas else 1


def _small_vectors(dim: int, radius: int, cap: int, rng: np.random.Generator) -> List[np.ndarray]:
    """[-radius,radius]^dim 枚举或随机抽样。"""
    if dim <= 0:
        return [np.zeros(0, dtype=np.int64)]
    total = (2 * radius + 1) ** dim
    if total <= cap:
        out = []
        for coeffs in itertools.product(range(-radius, radius + 1), repeat=dim):
            out.append(np.asarray(coeffs, dtype=np.int64))
        return out
    out = []
    seen: set = set()
    while len(out) < cap:
        v = rng.integers(-radius, radius + 1, size=dim, dtype=np.int64)
        key = v.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _reference_u_rows(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    rng: np.random.Generator,
    n_rows: int,
) -> np.ndarray:
    """用非平凡 probe 得到参考 u，以便选出真正“最差”的行（齐次 t=0 时 v=0 无效）。"""
    n, m = A.shape
    probe = np.zeros(m, dtype=np.int64)
    for _ in range(max(4, m // 20)):
        probe[int(rng.integers(0, m))] = int(rng.integers(-gamma, gamma + 1))
    if np.all(probe == 0):
        probe[0] = 1
    u0 = center_mod(t - (A @ probe), q)
    abs_u = np.abs(u0)
    bad = np.flatnonzero(abs_u > gamma)
    if bad.size > 0:
        order = bad[np.argsort(-abs_u[bad])]
    else:
        order = np.argsort(-abs_u)
    return order[: min(n_rows, n)]


def _wagner_mitm_seeds(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    row_idx: np.ndarray,
    cols: np.ndarray,
    box_radius: int,
    max_seeds: int,
    seen_v: set,
) -> List[np.ndarray]:
    """4+4 列全枚举 meet-in-the-middle（mod q 精确碰撞）。"""
    m = A.shape[1]
    if cols.size < 4:
        return []
    half = min(4, cols.size // 2)
    J1, J2 = cols[:half], cols[half : 2 * half]
    if J1.size == 0 or J2.size == 0:
        return []
    A1 = A[row_idx][:, J1] % q
    A2 = A[row_idx][:, J2] % q
    target = (t[row_idx] % q).astype(np.int64)
    dim = len(J1)
    if (2 * box_radius + 1) ** (2 * dim) > 120_000:
        return []

    seeds: List[np.ndarray] = []
    table: Dict[Tuple[int, ...], np.ndarray] = {}
    for coeffs in itertools.product(range(-box_radius, box_radius + 1), repeat=dim):
        v1 = np.asarray(coeffs, dtype=np.int64)
        r = tuple(int(x) for x in ((A1 @ v1) % q))
        table[r] = v1.copy()

    for coeffs in itertools.product(range(-box_radius, box_radius + 1), repeat=len(J2)):
        v2 = np.asarray(coeffs, dtype=np.int64)
        r2 = (A2 @ v2) % q
        need = tuple(int((target[i] - r2[i]) % q) for i in range(len(target)))
        v1 = table.get(need)
        if v1 is None:
            continue
        v_full = np.zeros(m, dtype=np.int64)
        for jj, cj in enumerate(J1):
            v_full[int(cj)] = int(v1[jj])
        for jj, cj in enumerate(J2):
            v_full[int(cj)] = int(v2[jj])
        v_full = np.clip(v_full, -gamma, gamma)
        key = v_full.tobytes()
        if key in seen_v:
            continue
        seen_v.add(key)
        seeds.append(v_full.copy())
        if len(seeds) >= max_seeds:
            break
    return seeds


def _subsystem_bruteforce_seeds(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    row_idx: np.ndarray,
    cols: np.ndarray,
    box_radius: int,
    max_seeds: int,
    seen_v: set,
    objective_uv: ObjectiveFn,
) -> List[np.ndarray]:
    """在 top 列小子空间暴力枚举，按全局 L∞ 分数选种（Wagner 无碰撞时的后备）。"""
    m = A.shape[1]
    ncol = min(5, cols.size)
    if ncol <= 0:
        return []
    use_cols = cols[:ncol]
    r = min(box_radius, gamma)
    scored: List[Tuple[Tuple[int, int, int], np.ndarray]] = []
    for coeffs in itertools.product(range(-r, r + 1), repeat=ncol):
        v_full = np.zeros(m, dtype=np.int64)
        for jj, cj in enumerate(use_cols):
            v_full[int(cj)] = int(coeffs[jj])
        if np.all(v_full == 0):
            continue
        res = center_mod(t - (A @ v_full), q)
        sc = objective_uv(res, v_full, gamma)
        scored.append((sc, v_full.copy()))
    scored.sort(key=lambda x: (x[0][2], x[0][0], x[0][1]))  # max_overflow, violations, sum
    out: List[np.ndarray] = []
    for sc, v_full in scored[: max_seeds * 4]:
        key = v_full.tobytes()
        if key in seen_v:
            continue
        seen_v.add(key)
        out.append(v_full)
        if len(out) >= max_seeds:
            break
    return out


def wagner_subsystem_seeds(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    rng: np.random.Generator,
    *,
    n_rows: int = 8,
    n_cols: int = 16,
    box_radius: int = 2,
    list_cap: int = 600,
    max_seeds: int = 12,
    objective_uv: Optional[ObjectiveFn] = None,
) -> List[np.ndarray]:
    """
    最差 u 行子系统种子：先 4+4 Wagner 全枚举，不足则 5 列小子空间暴力枚举。
    """
    obj = objective_uv or _score_uv
    n, m = A.shape
    row_idx = _reference_u_rows(A, t, q, gamma, rng, n_rows)
    if row_idx.size == 0:
        return []

    col_score = np.sum(np.abs(A[row_idx, :]), axis=0)
    cols = np.argsort(-col_score)[: min(n_cols, m)]
    if cols.size < 2:
        return []

    seen_v: set = set()
    seeds: List[np.ndarray] = []

    for br in (box_radius, min(box_radius + 1, gamma)):
        seeds.extend(
            _wagner_mitm_seeds(A, t, q, gamma, row_idx, cols, br, max_seeds - len(seeds), seen_v)
        )
        if len(seeds) >= max_seeds:
            return seeds[:max_seeds]

    seeds.extend(
        _subsystem_bruteforce_seeds(
            A, t, q, gamma, row_idx, cols, box_radius, max_seeds - len(seeds), seen_v, obj
        )
    )
    return seeds[:max_seeds]


def discrete_gaussian_seeds(
    v_center: np.ndarray,
    gamma: int,
    rng: np.random.Generator,
    *,
    n_seeds: int = 12,
    sigma: float = 3.0,
) -> List[np.ndarray]:
    """以当前最优 v 为中心的离散 Gaussian 抖动种子。"""
    v_center = np.asarray(v_center, dtype=np.int64).ravel()
    m = v_center.size
    seeds: List[np.ndarray] = []
    seen: set = set()
    sig = max(1.0, float(sigma))
    for _ in range(n_seeds * 3):
        if len(seeds) >= n_seeds:
            break
        noise = np.rint(rng.normal(0, sig, size=m)).astype(np.int64)
        v_new = np.clip(v_center + noise, -gamma, gamma)
        if np.all(v_new == 0):
            continue
        key = v_new.tobytes()
        if key in seen:
            continue
        seen.add(key)
        seeds.append(v_new.copy())
    return seeds


def violation_ls_step(
    A: np.ndarray,
    residual: np.ndarray,
    v: np.ndarray,
    cols: List[np.ndarray],
    q: int,
    gamma: int,
    score: Score3,
    rng: np.random.Generator,
    *,
    top_rows: int = 6,
    top_cols: int = 8,
    joint_radius: int = 3,
    better_score: BetterFn,
    objective_uv: ObjectiveFn,
) -> Optional[Tuple[np.ndarray, np.ndarray, int, Score3, float]]:
    """
    ViolationLS：对 u 超界最严重的若干行，在梯度最大的多列上联合网格搜索 Δv。
    """
    abs_r = np.abs(residual)
    bad = np.flatnonzero(abs_r > gamma)
    if bad.size == 0:
        return None
    viol_rows = bad[np.argsort(-abs_r[bad])[: min(top_rows, bad.size)]]
    grad = (A[viol_rows, :].T @ np.sign(residual[viol_rows]).astype(np.float64)).astype(np.float64)
    col_order = np.argsort(-np.abs(grad))[: min(top_cols, len(grad))]
    if col_order.size == 0:
        return None

    best_score = score
    best_pack: Optional[Tuple[np.ndarray, np.ndarray, int, Score3, float]] = None
    vv_sq = int(np.dot(v.astype(np.int64), v.astype(np.int64)))
    r = max(1, min(joint_radius, gamma))

    # 限制组合：随机抽若干对/三元组列做联合移动
    ncol = int(col_order.size)
    trials = min(48, ncol * (ncol - 1) // 2)
    pairs: List[Tuple[int, int]] = []
    if ncol >= 2:
        all_pairs = [(int(col_order[i]), int(col_order[j])) for i in range(ncol) for j in range(i + 1, ncol)]
        if len(all_pairs) <= trials:
            pairs = all_pairs
        else:
            idx = rng.choice(len(all_pairs), size=trials, replace=False)
            pairs = [all_pairs[int(i)] for i in idx]

    for j, k in pairs:
        cj, ck = int(v[j]), int(v[k])
        col_j, col_k = cols[j], cols[k]
        for dj in range(-r, r + 1):
            for dk in range(-r, r + 1):
                if dj == 0 and dk == 0:
                    continue
                nj, nk = cj + dj, ck + dk
                if abs(nj) > gamma or abs(nk) > gamma:
                    continue
                cand_res = center_mod(residual - col_j * dj - col_k * dk, q)
                v[j], v[k] = nj, nk
                try:
                    cv, cosum, cmaxov = objective_uv(cand_res, v, gamma)
                finally:
                    v[j], v[k] = cj, ck
                cand_score = (cv, cosum, cmaxov)
                if better_score(cand_score, best_score):
                    best_score = cand_score
                    new_vv = vv_sq - cj * cj - ck * ck + nj * nj + nk * nk
                    rr_c = float(np.dot(cand_res.astype(np.float64), cand_res.astype(np.float64)))
                    best_pack = (cand_res.copy(), j, k, nj, nk, cand_score, new_vv, rr_c)

    if best_pack is None:
        return None
    cand_res, j, k, nj, nk, new_score, new_vv, rr_c = best_pack
    v[j], v[k] = nj, nk
    return cand_res, v, new_vv, new_score, rr_c


def layered_row_projection(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    residual: np.ndarray,
    v: np.ndarray,
    score: Score3,
    *,
    n_layers: int = 3,
    better_score: BetterFn,
    objective_uv_and_rr_sq: Callable[..., Tuple[int, int, int, float]],
) -> Optional[Tuple[np.ndarray, np.ndarray, int, Score3, float]]:
    """由最差 u 行开始分层 LS 投影，逐层缩小关注行集。"""
    abs_r = np.abs(residual)
    bad = np.flatnonzero(abs_r > gamma)
    if bad.size == 0:
        return None
    remaining = list(bad[np.argsort(-abs_r[bad])])
    v_work = v.copy()
    res_work = residual.copy()
    sc = score
    vv_sq = int(np.dot(v_work.astype(np.int64), v_work.astype(np.int64)))
    rr_sync = float(np.dot(res_work.astype(np.float64), res_work.astype(np.float64)))

    for layer in range(n_layers):
        if not remaining:
            break
        k = max(1, len(remaining) // (2**layer))
        focus = np.asarray(remaining[:k], dtype=np.int64)
        col_score = np.sum(np.abs(A[focus, :]), axis=0)
        take = min(28, A.shape[1])
        cols_idx = np.argsort(-col_score)[:take]
        AR = A[np.ix_(focus, cols_idx)].astype(np.float64)
        b = res_work[focus].astype(np.float64)
        delta, *_ = np.linalg.lstsq(AR, b, rcond=1e-8)
        improved_layer = False
        for scale in (1.0, 0.55, 0.28):
            di = np.rint(delta * scale).astype(np.int64)
            cand_v = v_work.copy()
            ok = True
            for jj, cj in enumerate(cols_idx):
                nv = int(cand_v[int(cj)]) + int(di[jj])
                if abs(nv) > gamma:
                    ok = False
                    break
                cand_v[int(cj)] = nv
            if not ok:
                continue
            cand_res = center_mod(t - (A @ cand_v), q)
            cv, cosum, cmaxov, rr_c = objective_uv_and_rr_sq(cand_res, cand_v, gamma)
            cand_sc = (cv, cosum, cmaxov)
            if better_score(cand_sc, sc):
                v_work = cand_v
                res_work = cand_res
                sc = cand_sc
                vv_sq = int(np.dot(v_work.astype(np.int64), v_work.astype(np.int64)))
                rr_sync = rr_c
                improved_layer = True
                break
        if improved_layer:
            abs_new = np.abs(res_work)
            remaining = [int(r) for r in remaining if abs_new[int(r)] > gamma]
        else:
            break

    if sc == score:
        return None
    return res_work, v_work, vv_sq, sc, rr_sync


def cheap_pair_reduction(
    v: np.ndarray,
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    gamma: int,
    score: Score3,
    rng: np.random.Generator,
    *,
    n_trials: int = 40,
    better_score: BetterFn,
    objective_uv: ObjectiveFn,
) -> Tuple[np.ndarray, Score3]:
    """轻量列组合归约（递归格思想的零阶近似）。"""
    v = v.copy()
    res = center_mod(t - (A @ v), q)
    sc = score
    m = v.size
    for _ in range(n_trials):
        j, k = int(rng.integers(0, m)), int(rng.integers(0, m))
        if j == k:
            continue
        for coeff in (-1, 1):
            cand = v.copy()
            cand[j] = int(np.clip(cand[j] + coeff * cand[k], -gamma, gamma))
            cand_res = center_mod(t - (A @ cand), q)
            cand_sc = objective_uv(cand_res, cand, gamma)
            if better_score(cand_sc, sc):
                v, res, sc = cand, cand_res, cand_sc
    return v, sc