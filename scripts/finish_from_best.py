"""
从 incumbent（如 problem1_best.json）做收尾攻击：

1. 全维 v 的 L∞ ILP（ortools，默认 3600s）；
2. 小子格 40×40 真 BKZ 种子 + 短程局部搜索。

用法（项目根目录）::

  python3 scripts/finish_from_best.py \\
    --instance saiti1/sis_inf_problems_json/problem1.json \\
    --incumbent results/class1/problem1_best.json \\
    --output results/p1_finish.json \\
    --ilp-time-limit 3600 \\
    --sub-bkz-beta 28
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from sis_finish_ops import collect_sub_bkz_v_seeds, cp_sat_full_v_linf_finish
from sis_problem_taxonomy import effective_require_norm_ge_q2, problem_class_from_instance
from solve_sisinf import SearchConfig, apply_sis_class_defaults, center_mod, local_search_one, verify_solution


def _load_instance(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data[0]
    return data


def _load_incumbent(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _better_verify(a: Dict[str, int], b: Dict[str, int]) -> bool:
    if not b:
        return True
    ka = (
        a.get("congruence_ok", 0),
        -max(a.get("inf_u", 999), a.get("inf_v", 999)),
        -a.get("norm_sq", 0),
    )
    kb = (
        b.get("congruence_ok", 0),
        -max(b.get("inf_u", 999), b.get("inf_v", 999)),
        -b.get("norm_sq", 0),
    )
    return ka > kb


def main() -> None:
    p = argparse.ArgumentParser(description="ILP finish + sub-lattice BKZ from incumbent.")
    p.add_argument("--instance", required=True)
    p.add_argument("--incumbent", required=True, help="JSON with u, v lists (e.g. problem1_best.json)")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--ilp-time-limit", type=float, default=3600.0, help="Full-v ILP seconds (30–60 min typical)")
    p.add_argument("--ilp-workers", type=int, default=4)
    p.add_argument("--skip-ilp", action="store_true")
    p.add_argument("--skip-sub-bkz", action="store_true")
    p.add_argument("--sub-bkz-rows", type=int, default=40)
    p.add_argument("--sub-bkz-cols", type=int, default=40)
    p.add_argument("--sub-bkz-beta", type=int, default=28)
    p.add_argument("--sub-bkz-seeds", type=int, default=12)
    p.add_argument("--ls-restarts", type=int, default=8, help="Local search restarts after sub-BKZ seeds")
    p.add_argument("--ls-iters", type=int, default=4000)
    args = p.parse_args()

    inst = _load_instance(args.instance)
    inc = _load_incumbent(args.incumbent)
    A = np.array(inst["A"], dtype=np.int64)
    t = np.array(inst["t"], dtype=np.int64)
    q, gamma = int(inst["q"]), int(inst["gamma"])
    pid = int(inst.get("id", inc.get("id", 0)))
    sis_class = problem_class_from_instance(inst)
    require_norm = effective_require_norm_ge_q2(inst, sis_class)

    u0 = np.array(inc["u"], dtype=np.int64)
    v0 = np.array(inc["v"], dtype=np.int64)
    ok0, verify0 = verify_solution(A, t, q, gamma, u0, v0, require_norm)

    report: Dict[str, Any] = {
        "id": pid,
        "incumbent_verify": verify0,
        "phases": [],
        "success": bool(ok0),
    }
    best_u, best_v = u0.copy(), v0.copy()
    best_verify = dict(verify0)
    t_all = time.time()

    if not args.skip_ilp:
        print(f"[finish] full-v ILP, limit={args.ilp_time_limit}s ...", flush=True)
        u_ilp, v_ilp, meta = cp_sat_full_v_linf_finish(
            A,
            t,
            q,
            gamma,
            best_v,
            time_limit_sec=args.ilp_time_limit,
            num_workers=args.ilp_workers,
        )
        phase: Dict[str, Any] = {"name": "full_v_ilp", "ok": False, "meta": meta}
        if u_ilp is not None and v_ilp is not None:
            ok_ilp, ver_ilp = verify_solution(A, t, q, gamma, u_ilp, v_ilp, require_norm)
            phase = {"name": "full_v_ilp", "ok": bool(ok_ilp), "verify": ver_ilp, "meta": meta}
            print(
                f"[finish] ILP done: success={ok_ilp} inf_u={ver_ilp.get('inf_u')} inf_v={ver_ilp.get('inf_v')} "
                f"status={meta.get('ilp_status_name')} time={meta.get('ilp_time_sec', 0):.1f}s",
                flush=True,
            )
            if _better_verify(ver_ilp, best_verify):
                best_u, best_v = u_ilp.copy(), v_ilp.copy()
                best_verify = ver_ilp
                report["success"] = bool(ok_ilp)
            if ok_ilp:
                report["phases"].append(phase)
                report["verify"] = best_verify
                report["u"] = best_u.tolist()
                report["v"] = best_v.tolist()
                report["elapsed_sec"] = time.time() - t_all
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(f"[finish] feasible via ILP -> {args.output}", flush=True)
                return
        else:
            err = meta.get("ilp_error", "unknown")
            phase["error"] = err
            print(f"[finish] ILP failed: {err}", flush=True)
            if meta.get("ilp_traceback"):
                print(meta["ilp_traceback"], flush=True)
        report["phases"].append(phase)

    if not args.skip_sub_bkz and not report["success"]:
        print("[finish] sub-lattice BKZ seeds + short LS ...", flush=True)
        res_for_bkz = center_mod(t - (A @ best_v), q)
        seeds, sub_meta = collect_sub_bkz_v_seeds(
            A,
            q,
            gamma,
            res_for_bkz,
            args.sub_bkz_beta,
            n_rows=args.sub_bkz_rows,
            n_cols=args.sub_bkz_cols,
            max_vectors=args.sub_bkz_seeds,
            v_base=best_v,
            embed_mode="replace",
        )
        prepend = list(seeds) if seeds else [best_v]
        phase = {"name": "sub_bkz_ls", "meta": sub_meta, "seed_count": len(prepend)}
        print(f"[finish] sub-BKZ seeds={len(seeds)} dim={sub_meta.get('sub_bkz_dim')}", flush=True)

        cfg = apply_sis_class_defaults(
            SearchConfig(
                restarts=max(1, args.ls_restarts),
                iters=args.ls_iters,
                seed=args.seed,
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
        phase["ok"] = bool(ok_ls)
        phase["verify"] = ver_ls
        phase["ls_meta"] = meta_ls
        print(
            f"[finish] LS after sub-BKZ: inf_u={ver_ls.get('inf_u')} inf_v={ver_ls.get('inf_v')}",
            flush=True,
        )
        if _better_verify(ver_ls, best_verify):
            best_u, best_v = u_ls.copy(), v_ls.copy()
            best_verify = ver_ls
            report["success"] = bool(ok_ls)
        report["phases"].append(phase)

    report["verify"] = best_verify
    report["u"] = best_u.tolist()
    report["v"] = best_v.tolist()
    report["elapsed_sec"] = time.time() - t_all
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(
        f"[finish] done success={report['success']} inf_u={best_verify.get('inf_u')} "
        f"inf_v={best_verify.get('inf_v')} -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
