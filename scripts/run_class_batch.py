"""
按赛题类别（1/2/3）批量求解 SIS∞ 小题。

流程
----
1. 根据 ``--class`` 选择题号集合（可用 ``--problems`` 覆盖）；
2. 从 ``saiti1/sis_inf_problems_json/problemN.json`` 加载实例；
3. 每题多轮调用 ``local_search_one``，轮次参数来自 ``_cfg_for_round`` +
   ``apply_sis_class_defaults``；
4. 每轮写入 ``results/classN/problem{id}_latest.json``；
5. 首次 ``verify_solution`` 通过则写入 ``problem{id}_solution.json``；
6. 汇总 ``batch_report.json``（含 solved/total/all_success）。

与 ``run_class1_until_success.py`` 的区别
---------------------------------------
本脚本统一处理三类；class1 专用脚本仅 1/3/6/9 且摘要格式略简。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Set

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from run_problem1_until_success import _cfg_for_round
from sis_problem_taxonomy import (
    CLASS_1_IDS,
    CLASS_2_IDS,
    CLASS_3_IDS,
    class_label,
    effective_require_norm_lt_q2,
    problem_class_from_id,
)
from solve_sisinf import apply_sis_class_defaults, local_search_one, verify_solution

# 类别 → 官方题号
CLASS_TO_IDS: Dict[int, Set[int]] = {
    1: CLASS_1_IDS,
    2: CLASS_2_IDS,
    3: CLASS_3_IDS,
}


def _load(path: str) -> Dict[str, Any]:
    """读取 JSON；官方格式为单元素列表 ``[{...}]``。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data[0]
    return data


def _quick_cfg_patch(cfg, quick: bool):
    """
    ``--quick`` 模式：减半 restarts、缩短 iters/timeout，用于快速筛查而非冲解。
    """
    if not quick:
        return cfg
    from dataclasses import replace

    return replace(
        cfg,
        restarts=max(4, cfg.restarts // 3),
        iters=min(1200, cfg.iters),
        timeout_sec=min(120.0, cfg.timeout_sec) if cfg.timeout_sec else 120.0,
        block_cp_every=0 if cfg.block_cp_every > 120 else cfg.block_cp_every,
        use_sieve_seeds=False,
        bkz_beta=min(20, cfg.bkz_beta) if cfg.bkz_beta > 0 else 0,
        bkz_max_vectors=min(12, cfg.bkz_max_vectors),
        restricted_svp_samples=min(120, cfg.restricted_svp_samples),
        parallel_workers=1,
    )


def _better(verify: Dict[str, int], prev: Dict[str, int]) -> bool:
    """
    比较两轮 verify 字典，判断当前是否更优（未可行时保留最佳进展）。

    优先级：同余成立 > L∞ 更小 > norm_req_ok（第三类 <q²）> 更小 norm_sq。
    """
    if not prev:
        return True
    key = (
        verify.get("congruence_ok", 0),
        -max(verify.get("inf_u", 999), verify.get("inf_v", 999)),
        verify.get("norm_req_ok", 1),
        verify.get("norm_sq", 0),
    )
    pkey = (
        prev.get("congruence_ok", 0),
        -max(prev.get("inf_u", 999), prev.get("inf_v", 999)),
        prev.get("norm_req_ok", 1),
        prev.get("norm_sq", 0),
    )
    return key > pkey


def solve_one(
    pid: int,
    inst: Dict[str, Any],
    sis_class: int,
    json_dir: str,
    out_dir: str,
    seed_base: int,
    max_rounds: int,
    quick: bool,
    full_max: bool = False,
) -> Dict[str, Any]:
    """
    单题循环求解直至可行或耗尽 ``max_rounds``（0 表示无限）。

    Returns
    -------
    dict
        含 success、verify、rounds_tried、elapsed_sec 等；可行时无 u/v 在 best 中省略。
    """
    A = np.array(inst["A"], dtype=np.int64)
    t = np.array(inst["t"], dtype=np.int64)
    q, gamma = int(inst["q"]), int(inst["gamma"])
    require_norm = effective_require_norm_lt_q2(inst, sis_class)

    best: Dict[str, Any] = {
        "id": pid,
        "class": sis_class,
        "class_label": class_label(sis_class),
        "success": False,
        "rounds_tried": 0,
    }
    t0 = time.time()
    round_idx = 0
    while max_rounds == 0 or round_idx < max_rounds:
        seed = seed_base + pid * 1009 + round_idx * 100003
        cfg = apply_sis_class_defaults(
            _cfg_for_round(round_idx, seed),
            sis_class,
            aggressive=round_idx >= 6 or full_max,
            full_max=full_max,
        )
        cfg = _quick_cfg_patch(cfg, quick)
        u, v, meta = local_search_one(A, t, q, gamma, cfg, require_norm)
        ok, verify = verify_solution(A, t, q, gamma, u, v, require_norm)
        round_idx += 1
        rec = {
            "id": pid,
            "class": sis_class,
            "round": round_idx - 1,
            "success": bool(ok),
            "verify": verify,
            "meta": meta,
            "require_norm_lt_q2": require_norm,
            "elapsed_sec": time.time() - t0,
        }
        latest = os.path.join(out_dir, f"problem{pid}_latest.json")
        with open(latest, "w", encoding="utf-8") as f:
            json.dump({**rec, "u": u.tolist(), "v": v.tolist()}, f, indent=2, ensure_ascii=False)

        print(
            f"[class{sis_class}] problem{pid} round {round_idx - 1}: "
            f"ok={ok} inf_u={verify.get('inf_u')} inf_v={verify.get('inf_v')} "
            f"congr={verify.get('congruence_ok')} norm_sq={verify.get('norm_sq')}",
            flush=True,
        )
        if ok:
            best = {**rec, "rounds_tried": round_idx, "u": u.tolist(), "v": v.tolist()}
            sol = os.path.join(out_dir, f"problem{pid}_solution.json")
            with open(sol, "w", encoding="utf-8") as f:
                json.dump(
                    {"id": pid, "u": u.tolist(), "v": v.tolist(), "verify": verify, "meta": meta},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            return best
        if _better(verify, best.get("verify", {})):
            best = {**rec, "rounds_tried": round_idx, "u": u.tolist(), "v": v.tolist()}
            best_path = os.path.join(out_dir, f"problem{pid}_best.json")
            with open(best_path, "w", encoding="utf-8") as f:
                json.dump(
                    {**best, "u": best["u"], "v": best["v"]},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
    best["rounds_tried"] = round_idx
    best["elapsed_sec"] = time.time() - t0
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="Batch solve SIS∞ by taxonomy class (1/2/3).")
    p.add_argument("--class", dest="sis_class", type=int, required=True, choices=[1, 2, 3])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p.add_argument(
        "--json-dir",
        default=os.path.join(root, "saiti1", "sis_inf_problems_json"),
        help="Directory with problemN.json (default: <project>/saiti1/sis_inf_problems_json)",
    )
    p.add_argument("--output-dir", default="")
    p.add_argument("--problems", default="", help="Override ids, e.g. 1,3,6,9")
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--max-rounds", type=int, default=12, help="Per problem; 0=unlimited")
    p.add_argument("--quick", action="store_true", help="Shorter timeout/iters for screening")
    p.add_argument(
        "--full-max",
        action="store_true",
        help="Paper full stack: G6K BDGL2 + Wang max enumerate + 4h ILP budget (server only)",
    )
    args = p.parse_args()

    cls = int(args.sis_class)
    ids = (
        [int(x.strip()) for x in args.problems.split(",") if x.strip()]
        if args.problems
        else sorted(CLASS_TO_IDS[cls])
    )
    for pid in ids:
        if pid not in CLASS_TO_IDS[cls]:
            raise SystemExit(f"problem {pid} not in class {cls}")

    out_dir = args.output_dir or os.path.join(root, "results", f"class{cls}")
    os.makedirs(out_dir, exist_ok=True)

    summary: List[Dict[str, Any]] = []
    for pid in ids:
        path = os.path.join(args.json_dir, f"problem{pid}.json")
        inst = _load(path)
        summary.append(
            solve_one(
                pid,
                inst,
                cls,
                args.json_dir,
                out_dir,
                args.seed,
                args.max_rounds,
                args.quick,
                full_max=bool(args.full_max),
            )
        )

    ok_count = sum(1 for r in summary if r.get("success"))
    report = {
        "class": cls,
        "class_label": class_label(cls),
        "problems": ids,
        "solved": ok_count,
        "total": len(ids),
        "all_success": ok_count == len(ids),
        "results": summary,
    }
    rep_path = os.path.join(out_dir, "batch_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n=== Class {cls} done: {ok_count}/{len(ids)} feasible ===", flush=True)
    print(f"Report: {rep_path}", flush=True)


if __name__ == "__main__":
    main()
