"""
第一类题（1、3、6、9）分阶段追踪运行，输出详细解题流程报告。

与 README「Dual-Space + Entropy IIR-CLS」及第一类 SVP+kernel 配置对齐。
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in os.path.dirname(os.path.abspath(__file__)) and _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from run_problem1_until_success import _cfg_for_round
from sis_problem_taxonomy import CLASS_1_IDS, class_label, effective_require_norm_ge_q2
from solve_sisinf import (
    SearchConfig,
    apply_sis_class_defaults,
    build_dual_space_candidates,
    center_mod,
    local_search_one,
    objective_uv,
    score_key,
    verify_solution,
)

try:
    from modq_kernel import right_kernel_basis_mod_q
except ImportError:
    right_kernel_basis_mod_q = None  # type: ignore

try:
    from lattice_bkz import collect_bkz_v_seeds
except ImportError:
    collect_bkz_v_seeds = None  # type: ignore


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[0] if isinstance(data, list) else data


def _phase_bkz(A: np.ndarray, q: int, gamma: int, cfg: SearchConfig, seed: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out: Dict[str, Any] = {"enabled": cfg.use_bkz_seeds and cfg.bkz_beta > 0}
    if not out["enabled"] or collect_bkz_v_seeds is None:
        out["note"] = "BKZ 未启用或 fpylll/lattice_bkz 不可用"
        out["seeds"] = 0
        out["elapsed_sec"] = 0.0
        return out
    rng = np.random.default_rng(seed)
    seeds = collect_bkz_v_seeds(
        A, q, gamma, cfg.bkz_beta, cfg.bkz_max_vectors, cfg.bkz_max_dim,
        combo_depth=cfg.bkz_combo_depth, combo_coeff_max=cfg.bkz_combo_coeff_max,
        rng=rng,
    )
    out["seeds"] = len(seeds)
    out["beta"] = cfg.bkz_beta
    out["combo_depth"] = cfg.bkz_combo_depth
    if seeds:
        sample = seeds[0]
        r = center_mod(-(A @ sample), q)
        viol, osum, mov = objective_uv(r, sample, gamma)
        out["best_seed_preview"] = {
            "violations": viol, "overflow_sum": osum, "max_overflow": mov,
            "inf_v": int(np.max(np.abs(sample))),
        }
    out["elapsed_sec"] = round(time.perf_counter() - t0, 3)
    return out


def _phase_kernel(A: np.ndarray, q: int, cfg: SearchConfig) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if right_kernel_basis_mod_q is None or cfg.kernel_walk_every <= 0:
        return {"dim_m": A.shape[1], "basis_cols": 0, "elapsed_sec": 0.0}
    K = right_kernel_basis_mod_q(A, q, cfg.kernel_max_basis)
    return {
        "dim_m": int(A.shape[1]),
        "basis_cols": int(K.shape[1]),
        "kernel_walk_every": cfg.kernel_walk_every,
        "elapsed_sec": round(time.perf_counter() - t0, 3),
    }


def _phase_dual(
    A: np.ndarray, t: np.ndarray, q: int, gamma: int, cfg: SearchConfig,
    lattice_prepend: List[np.ndarray], seed: int,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    cands, meta = build_dual_space_candidates(
        A, t, q, gamma, cfg, rng, prepend=lattice_prepend or None,
    )
    scored = []
    for v in cands[: min(8, len(cands))]:
        r = center_mod(t - (A @ v), q)
        viol, osum, mov = objective_uv(r, v, gamma)
        scored.append({
            "violations": viol, "overflow_sum": osum, "max_overflow": mov,
            "inf_v": int(np.max(np.abs(v))),
        })
    return {
        "num_candidates": meta.get("num_candidates", len(cands)),
        "top_scored_preview": scored,
        "elapsed_sec": round(time.perf_counter() - t0, 3),
    }


def trace_one(pid: int, json_dir: str, round_idx: int, seed_base: int, quick: bool = False) -> Dict[str, Any]:
    path = os.path.join(json_dir, f"problem{pid}.json")
    inst = _load(path)
    A = np.array(inst["A"], dtype=np.int64)
    t = np.array(inst["t"], dtype=np.int64)
    q, gamma = int(inst["q"]), int(inst["gamma"])
    n, m = A.shape
    homogeneous = bool(np.all(np.mod(t, q) == 0))

    seed = seed_base + pid * 1009 + round_idx * 100003
    base_cfg = _cfg_for_round(round_idx, seed)
    cfg = apply_sis_class_defaults(base_cfg, sis_class=1, aggressive=round_idx >= 2)
    if quick:
        cfg = replace(
            cfg,
            restarts=6,
            iters=1800,
            timeout_sec=180.0,
            use_bkz_seeds=False,
            bkz_beta=0,
            block_cp_every=0 if cfg.block_cp_every > 100 else cfg.block_cp_every,
        )
    require_norm = effective_require_norm_ge_q2(inst, 1)

    report: Dict[str, Any] = {
        "problem_id": pid,
        "class": 1,
        "class_label": class_label(1),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "instance": {
            "n": n, "m": m, "q": q, "gamma": gamma,
            "homogeneous": homogeneous,
            "require_norm_ge_q2": require_norm,
        },
        "framework": "Dual-Space + Entropy IIR-CLS（第一类叠加 SVP/BKZ + kernel walk）",
        "phases": [],
    }

    def add_phase(name: str, data: Dict[str, Any]) -> None:
        report["phases"].append({"name": name, **data})

    add_phase("0_建模", {
        "description": "残差 r(v)=Center(t-Av)；u=r(v)；在 v∈[-γ,γ]^m 上使 |r|_∞≤γ；齐次须 u,v 非全零",
        "residual_at_v0": _residual_stats(A, t, q, gamma, np.zeros(m, dtype=np.int64)),
    })

    bkz_info = _phase_bkz(A, q, gamma, cfg, seed_base + pid)
    add_phase("1_BKZ格种子", bkz_info)

    lattice_prepend: List[np.ndarray] = []
    if bkz_info.get("seeds", 0) and collect_bkz_v_seeds and cfg.bkz_beta > 0:
        lattice_prepend = collect_bkz_v_seeds(
            A, q, gamma, cfg.bkz_beta, cfg.bkz_max_vectors, cfg.bkz_max_dim,
            combo_depth=cfg.bkz_combo_depth, combo_coeff_max=cfg.bkz_combo_coeff_max,
            rng=np.random.default_rng(seed),
        )

    add_phase("2_模q核", _phase_kernel(A, q, cfg))
    add_phase("3_Dual空间候选", _phase_dual(A, t, q, gamma, cfg, lattice_prepend, seed))

    add_phase("4_配置摘要", {
        "restarts": cfg.restarts,
        "iters": cfg.iters,
        "timeout_sec": cfg.timeout_sec,
        "bkz_beta": cfg.bkz_beta,
        "kernel_walk_every": cfg.kernel_walk_every,
        "pair_relief_every": cfg.pair_relief_every,
        "block_cp_every": cfg.block_cp_every,
        "euclid_weight": cfg.euclid_weight,
        "entropy_weight": cfg.entropy_weight,
    })

    t_search = time.perf_counter()
    u, v, meta = local_search_one(A, t, q, gamma, cfg, require_norm)
    ok, verify = verify_solution(A, t, q, gamma, u, v, require_norm)
    search_sec = round(time.perf_counter() - t_search, 3)

    add_phase("5_局部搜索", {
        "elapsed_sec": search_sec,
        "success": bool(ok),
        "verify": verify,
        "meta": meta,
        "final_residual_stats": _residual_stats(A, t, q, gamma, v),
    })

    report["summary"] = {
        "success": bool(ok),
        "verify": verify,
        "total_elapsed_sec": sum(
            p.get("elapsed_sec", 0) for p in report["phases"] if isinstance(p.get("elapsed_sec"), (int, float))
        ) + search_sec,
    }
    return report


def _residual_stats(A, t, q, gamma, v) -> Dict[str, int]:
    r = center_mod(t - (A @ v), q)
    viol, osum, mov = objective_uv(r, v, gamma)
    return {
        "violations": viol,
        "overflow_sum": osum,
        "max_overflow": mov,
        "inf_u": int(np.max(np.abs(r))),
        "inf_v": int(np.max(np.abs(v))),
    }


def render_markdown(reports: List[Dict[str, Any]], out_path: str) -> None:
    lines = [
        "# 第一类赛题（1、3、6、9）详细解题流程报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 总体框架（README：Dual-Space + Entropy IIR-CLS）",
        "",
        "| 阶段 | 内容 | 第一类扩展 |",
        "|------|------|------------|",
        "| 建模 | r(v)=Center(t−Av)，u=r(v) | t≡0 齐次；拒绝 u=v=0 |",
        "| 格空间 | Ajtai 嵌入 + BKZ/LLL | β≈28–32，组合短向量种子 |",
        "| Dual 空间 | pull/投影/稀疏/随机候选 | 无 CVP lift |",
        "| 模 q 核 | v←clip(v+Kd) 保持 u | SymPy 或素数域高斯 |",
        "| 残差精修 | 坐标下降+Pair+CP+Kick | Chebyshev 分层目标 |",
        "",
    ]
    ok_n = sum(1 for r in reports if r["summary"]["success"])
    lines.append(f"**可行解数量：{ok_n}/{len(reports)}**")
    lines.append("")

    for rep in reports:
        pid = rep["problem_id"]
        inst = rep["instance"]
        lines.append(f"## 小问 {pid}（n=m={inst['n']}, q={inst['q']}, γ={inst['gamma']}）")
        lines.append("")
        for ph in rep["phases"]:
            lines.append(f"### {ph['name']}")
            if ph.get("description"):
                lines.append(ph["description"])
            body = {k: v for k, v in ph.items() if k not in ("name", "description")}
            lines.append("```json")
            lines.append(json.dumps(body, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
        s = rep["summary"]
        lines.append(f"**结果**：success={s['success']}，inf_u={s['verify'].get('inf_u')}，inf_v={s['verify'].get('inf_v')}，"
                     f"congruence={s['verify'].get('congruence_ok')}")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", default=os.path.join("saiti1", "sis_inf_problems_json"))
    p.add_argument("--output-dir", default="results/class1")
    p.add_argument("--problems", default="1,3,6,9")
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--round", type=int, default=0, help="cfg round index for _cfg_for_round")
    p.add_argument("--quick", action="store_true", help="缩短 iters/restarts（筛查用）")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ids = [int(x.strip()) for x in args.problems.split(",") if x.strip()]
    for pid in ids:
        if pid not in CLASS_1_IDS:
            raise SystemExit(f"{pid} not in class 1")

    reports: List[Dict[str, Any]] = []
    for pid in ids:
        print(f"\n========== 追踪 problem {pid} ==========", flush=True)
        rep = trace_one(pid, args.json_dir, args.round, args.seed, quick=args.quick)
        reports.append(rep)
        jpath = os.path.join(args.output_dir, f"problem{pid}_trace.json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(json.dumps(rep["summary"], ensure_ascii=False), flush=True)

    render_markdown(reports, os.path.join(args.output_dir, "class1_detailed_flow_report.md"))
    with open(os.path.join(args.output_dir, "class1_trace_all.json"), "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入 {args.output_dir}/class1_detailed_flow_report.md", flush=True)


if __name__ == "__main__":
    main()
