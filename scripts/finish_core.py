"""
从 incumbent 执行 ILP 收尾（full/chunk/lex）+ 可选 sub-BKZ / 第三类欧氏抛光。

供 ``finish_from_best.py`` CLI 与 ``run_all_pipeline.py`` 共用。
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from sis_finish_ops import collect_sub_bkz_v_seeds, run_ilp_finish
from sis_problem_taxonomy import (
    class_label,
    effective_require_norm_ge_q2,
    problem_class_from_id,
    problem_class_from_instance,
)
from solve_sisinf import SearchConfig, apply_sis_class_defaults, center_mod, local_search_one, verify_solution


def load_instance(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data[0]
    return data


def load_incumbent(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def better_verify(
    a: Dict[str, int],
    b: Dict[str, int],
    *,
    require_norm_ge_q2: bool = False,
) -> bool:
    """未可行时比较进展；第三类在 L∞ 相同时优先 norm_req_ok 与更大 norm_sq。"""
    if not b:
        return True

    def key(v: Dict[str, int]) -> tuple:
        inf_max = max(v.get("inf_u", 999), v.get("inf_v", 999))
        base = (v.get("congruence_ok", 0), -inf_max)
        if require_norm_ge_q2:
            return base + (v.get("norm_req_ok", 0), v.get("norm_sq", 0))
        return base + (-v.get("norm_sq", 0),)

    return key(a) > key(b)


def default_ilp_mode_for_class(sis_class: int) -> str:
    """按赛题类推荐 ILP 模式（基于题 1/3 实验：full/lex 有效，chunk 对平台题无效）。"""
    if sis_class == 3:
        return "lex"
    return "full"


def execute_finish(
    instance_path: str,
    incumbent_path: str,
    output_path: str,
    *,
    ilp_mode: Optional[str] = None,
    ilp_time_limit: float = 3600.0,
    ilp_workers: int = 4,
    ilp_chunk_cols: int = 40,
    ilp_chunk_rounds: int = 12,
    ilp_chunk_stride: int = 0,
    skip_ilp: bool = False,
    skip_sub_bkz: bool = True,
    sub_bkz_rows: int = 40,
    sub_bkz_cols: int = 40,
    sub_bkz_beta: int = 28,
    sub_bkz_seeds: int = 12,
    euclid_polish: Optional[bool] = None,
    ls_restarts: int = 8,
    ls_iters: int = 4000,
    seed: int = 424242,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    对单题 incumbent 跑完整收尾管线，写入 ``output_path`` 并返回报告 dict。
    """
    inst = load_instance(instance_path)
    inc = load_incumbent(incumbent_path)
    A = np.array(inst["A"], dtype=np.int64)
    t = np.array(inst["t"], dtype=np.int64)
    q, gamma = int(inst["q"]), int(inst["gamma"])
    pid = int(inst.get("id", inc.get("id", 0)))
    sis_class = problem_class_from_instance(inst)
    require_norm = effective_require_norm_ge_q2(inst, sis_class)
    mode = ilp_mode or default_ilp_mode_for_class(sis_class)
    do_polish = euclid_polish if euclid_polish is not None else (sis_class == 3)

    u0 = np.array(inc["u"], dtype=np.int64)
    v0 = np.array(inc["v"], dtype=np.int64)
    ok0, verify0 = verify_solution(A, t, q, gamma, u0, v0, require_norm)

    report: Dict[str, Any] = {
        "id": pid,
        "class": sis_class,
        "class_label": class_label(sis_class),
        "ilp_mode": mode,
        "incumbent_verify": verify0,
        "phases": [],
        "success": bool(ok0),
    }
    best_u, best_v = u0.copy(), v0.copy()
    best_verify = dict(verify0)
    t_all = time.time()

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    if not skip_ilp:
        stride = ilp_chunk_stride if ilp_chunk_stride > 0 else None
        log(
            f"[finish p{pid}] ILP mode={mode} limit={ilp_time_limit}s "
            f"class={sis_class} ..."
        )
        u_ilp, v_ilp, meta = run_ilp_finish(
            mode,
            A,
            t,
            q,
            gamma,
            best_v,
            ilp_time_limit,
            num_workers=ilp_workers,
            chunk_cols=ilp_chunk_cols,
            chunk_rounds=ilp_chunk_rounds,
            chunk_stride=stride,
        )
        phase_name = f"ilp_{mode}"
        phase: Dict[str, Any] = {"name": phase_name, "ok": False, "meta": meta}
        if u_ilp is not None and v_ilp is not None:
            ok_ilp, ver_ilp = verify_solution(A, t, q, gamma, u_ilp, v_ilp, require_norm)
            phase = {"name": phase_name, "ok": bool(ok_ilp), "verify": ver_ilp, "meta": meta}
            log(
                f"[finish p{pid}] ILP: ok={ok_ilp} inf_u={ver_ilp.get('inf_u')} inf_v={ver_ilp.get('inf_v')} "
                f"norm_ok={ver_ilp.get('norm_req_ok')} status={meta.get('ilp_status_name')} "
                f"time={meta.get('ilp_time_sec', 0):.1f}s"
            )
            if better_verify(ver_ilp, best_verify, require_norm_ge_q2=require_norm):
                best_u, best_v = u_ilp.copy(), v_ilp.copy()
                best_verify = ver_ilp
                report["success"] = bool(ok_ilp)
            if ok_ilp:
                report["phases"].append(phase)
                report["verify"] = best_verify
                report["u"] = best_u.tolist()
                report["v"] = best_v.tolist()
                report["elapsed_sec"] = time.time() - t_all
                os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                log(f"[finish p{pid}] feasible -> {output_path}")
                return report
        else:
            phase["error"] = meta.get("ilp_error", "unknown")
            log(f"[finish p{pid}] ILP failed: {phase['error']}")
        report["phases"].append(phase)

    # 第三类：L∞ 近可行但欧氏下界不足时，用高 euclid_weight 局部搜索抛光
    if do_polish and not report["success"] and require_norm:
        log(f"[finish p{pid}] euclid polish (class 3) ...")
        cfg = apply_sis_class_defaults(
            SearchConfig(
                restarts=max(4, ls_restarts),
                iters=ls_iters,
                seed=seed + pid,
                parallel_workers=1,
                timeout_sec=1200.0,
                euclid_weight=5.0,
                entropy_weight=0.5,
            ),
            sis_class,
        )
        u_p, v_p, meta_p = local_search_one(
            A, t, q, gamma, cfg, require_norm, prepend_v_seeds=[best_v]
        )
        ok_p, ver_p = verify_solution(A, t, q, gamma, u_p, v_p, require_norm)
        phase = {"name": "euclid_polish", "ok": bool(ok_p), "verify": ver_p, "meta": meta_p}
        log(
            f"[finish p{pid}] polish: ok={ok_p} inf_u={ver_p.get('inf_u')} norm_sq={ver_p.get('norm_sq')} "
            f"norm_ok={ver_p.get('norm_req_ok')}"
        )
        if better_verify(ver_p, best_verify, require_norm_ge_q2=require_norm):
            best_u, best_v = u_p.copy(), v_p.copy()
            best_verify = ver_p
            report["success"] = bool(ok_p)
        report["phases"].append(phase)
        if ok_p:
            report["verify"] = best_verify
            report["u"] = best_u.tolist()
            report["v"] = best_v.tolist()
            report["elapsed_sec"] = time.time() - t_all
            os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            return report

    # 仅第一类默认尝试 sub-BKZ（题 1/3 实验表明常变差；默认 skip_sub_bkz=True）
    if not skip_sub_bkz and sis_class == 1 and not report["success"]:
        log(f"[finish p{pid}] sub-BKZ + LS (class 1 only) ...")
        res_for_bkz = center_mod(t - (A @ best_v), q)
        seeds, sub_meta = collect_sub_bkz_v_seeds(
            A,
            q,
            gamma,
            res_for_bkz,
            sub_bkz_beta,
            n_rows=sub_bkz_rows,
            n_cols=sub_bkz_cols,
            max_vectors=sub_bkz_seeds,
            v_base=best_v,
            embed_mode="replace",
        )
        prepend = list(seeds) if seeds else [best_v]
        cfg = apply_sis_class_defaults(
            SearchConfig(
                restarts=max(1, ls_restarts),
                iters=ls_iters,
                seed=seed,
                parallel_workers=1,
                timeout_sec=900.0,
                use_bkz_seeds=False,
            ),
            sis_class,
        )
        u_ls, v_ls, meta_ls = local_search_one(
            A, t, q, gamma, cfg, require_norm, prepend_v_seeds=prepend
        )
        ok_ls, ver_ls = verify_solution(A, t, q, gamma, u_ls, v_ls, require_norm)
        phase = {
            "name": "sub_bkz_ls",
            "ok": bool(ok_ls),
            "verify": ver_ls,
            "meta": sub_meta,
            "ls_meta": meta_ls,
            "seed_count": len(prepend),
        }
        log(f"[finish p{pid}] sub-BKZ LS: inf_u={ver_ls.get('inf_u')} inf_v={ver_ls.get('inf_v')}")
        if better_verify(ver_ls, best_verify, require_norm_ge_q2=require_norm):
            best_u, best_v = u_ls.copy(), v_ls.copy()
            best_verify = ver_ls
            report["success"] = bool(ok_ls)
        report["phases"].append(phase)

    report["verify"] = best_verify
    report["u"] = best_u.tolist()
    report["v"] = best_v.tolist()
    report["elapsed_sec"] = time.time() - t_all
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log(
        f"[finish p{pid}] done success={report['success']} "
        f"inf_u={best_verify.get('inf_u')} inf_v={best_verify.get('inf_v')} "
        f"norm_ok={best_verify.get('norm_req_ok')}"
    )
    return report


def save_incumbent_from_record(rec: Dict[str, Any], path: str) -> None:
    """从 batch / finish 记录写出 ``{u,v,verify,id,round}`` incumbent 文件。"""
    payload = {
        "id": rec.get("id"),
        "round": rec.get("round"),
        "verify": rec.get("verify"),
        "u": rec.get("u"),
        "v": rec.get("v"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
