"""
运行入口：按类 batch 启发式 + 十题全量验证（batch + ILP + 计分）。
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

from sis_common import (
    CLASS_1_IDS,
    CLASS_2_IDS,
    CLASS_3_IDS,
    class_label,
    effective_require_norm_lt_q2,
    problem_class_from_id,
)
from solve_sisinf import SearchConfig, apply_sis_class_defaults, local_search_one, verify_solution

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


def _cfg_for_round(round_idx: int, seed: int) -> SearchConfig:
    """渐进式搜索配置：轮次越高，restarts/iters/CP 窗口越大。"""
    if round_idx < 8:
        return SearchConfig(
            restarts=16,
            iters=6000,
            seed=seed,
            parallel_workers=1,
            timeout_sec=360.0,
            kernel_walk_every=20,
            kernel_max_basis=32,
            ls_project_every=20,
            cp_periodic_every=90,
            cp_periodic_cols=24,
            cp_repair_threshold=80,
            cp_repair_window=4,
            cp_repair_time_limit=2.0,
            block_cp_every=80,
            block_cp_rows=24,
            block_cp_cols=32,
            block_cp_window=6,
            block_cp_time_limit=2.5,
            delta=3,
            max_delta=8,
            pair_relief_every=24,
            pair_relief_attempts=16,
        )
    if round_idx < 20:
        return SearchConfig(
            restarts=20,
            iters=8000,
            seed=seed,
            parallel_workers=1,
            timeout_sec=520.0,
            kernel_walk_every=18,
            kernel_max_basis=36,
            ls_project_every=18,
            cp_periodic_every=70,
            cp_periodic_cols=28,
            cp_repair_threshold=120,
            cp_repair_window=5,
            cp_repair_time_limit=3.0,
            block_cp_every=60,
            block_cp_rows=28,
            block_cp_cols=36,
            block_cp_window=7,
            block_cp_time_limit=3.2,
            delta=3,
            max_delta=10,
            pair_relief_every=20,
            pair_relief_attempts=20,
            allow_uphill_sa=True,
        )
    return SearchConfig(
        restarts=24,
        iters=10000,
        seed=seed,
        parallel_workers=1,
        timeout_sec=680.0,
        kernel_walk_every=16,
        kernel_max_basis=40,
        ls_project_every=16,
        cp_periodic_every=60,
        cp_periodic_cols=32,
        cp_repair_threshold=140,
        cp_repair_window=6,
        cp_repair_time_limit=4.0,
        block_cp_every=50,
        block_cp_rows=32,
        block_cp_cols=40,
        block_cp_window=8,
        block_cp_time_limit=4.2,
        delta=4,
        max_delta=10,
        pair_relief_every=16,
        pair_relief_attempts=24,
        allow_uphill_sa=True,
    )


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


def main_batch() -> None:
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



# --- full validation CLI ---

from sis_finish import default_ilp_mode_for_class, execute_finish, load_instance, save_incumbent_from_record
from sis_common import ALL_IDS, problem_class_from_id
from sis_common import score_from_verify


def main_full() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description="Full algorithm stack validation (problems 1-10).")
    p.add_argument("--problems", default="", help="comma ids, default all 1-10")
    p.add_argument("--json-dir", default=os.path.join(root, "saiti1", "sis_inf_problems_json"))
    p.add_argument("--output-dir", default=os.path.join(root, "results", "full_validation"))
    p.add_argument("--seed", type=int, default=20260603)
    p.add_argument("--batch-rounds", type=int, default=3)
    p.add_argument("--ilp-time-limit", type=float, default=300.0)
    p.add_argument("--quick", action="store_true", help="2 batch rounds, 120s ILP")
    p.add_argument("--skip-finish", action="store_true")
    p.add_argument(
        "--full-max",
        action="store_true",
        help="Paper full stack (G6K BDGL2 + Wang max + 4h ILP if not --quick)",
    )
    args = p.parse_args()

    if args.problems:
        pids = sorted(int(x.strip()) for x in args.problems.split(",") if x.strip())
        for pid in pids:
            if pid not in ALL_IDS:
                raise SystemExit(f"unknown problem id {pid}")
    else:
        pids = sorted(ALL_IDS)

    batch_rounds = 1 if args.quick else args.batch_rounds
    if args.full_max and not args.quick:
        batch_rounds = max(batch_rounds, 24)
    ilp_limit = 60.0 if args.quick else args.ilp_time_limit
    if args.full_max and not args.quick:
        ilp_limit = max(ilp_limit, 14400.0)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    total_score = 0

    print(
        f"=== Full validation: problems={pids} batch={batch_rounds} ilp={ilp_limit}s ===",
        flush=True,
    )

    for pid in pids:
        sis_class = problem_class_from_id(pid)
        cls_dir = os.path.join(out_dir, f"class{sis_class}")
        os.makedirs(cls_dir, exist_ok=True)
        inst_path = os.path.join(args.json_dir, f"problem{pid}.json")
        inst = load_instance(inst_path)
        best_path = os.path.join(cls_dir, f"problem{pid}_best.json")
        finish_path = os.path.join(cls_dir, f"problem{pid}_finish.json")

        rec: Dict[str, Any] = {
            "id": pid,
            "class": sis_class,
            "gamma": int(inst["gamma"]),
            "n": int(inst["n"]),
            "q": int(inst["q"]),
        }

        print(f"[p{pid}] class{sis_class} batch ...", flush=True)
        batch_rec = solve_one(
            pid,
            inst,
            sis_class,
            args.json_dir,
            cls_dir,
            args.seed,
            batch_rounds,
            args.quick,
            full_max=bool(args.full_max),
        )
        rec["batch"] = {
            "success": batch_rec.get("success"),
            "verify": batch_rec.get("verify"),
            "meta": {
                k: batch_rec.get("meta", {}).get(k)
                for k in ("lattice_backend", "lattice_seed_count", "seed_sources")
                if batch_rec.get("meta")
            },
            "elapsed_sec": batch_rec.get("elapsed_sec"),
        }
        rec["batch_score"] = score_from_verify(inst, batch_rec.get("verify") or {}, sis_class=sis_class)

        if batch_rec.get("success"):
            save_incumbent_from_record(batch_rec, best_path)
            rec["final"] = batch_rec.get("verify")
            rec["final_score"] = rec["batch_score"]
            rec["success"] = True
            results.append(rec)
            total_score += rec["final_score"]["score"]
            print(f"[p{pid}] SOLVED in batch score={rec['final_score']['score']}", flush=True)
            continue

        if "u" in batch_rec:
            save_incumbent_from_record(batch_rec, best_path)

        if args.skip_finish:
            rec["success"] = False
            rec["final"] = batch_rec.get("verify")
            rec["final_score"] = rec["batch_score"]
            results.append(rec)
            continue

        ilp_mode = default_ilp_mode_for_class(sis_class)
        if args.full_max and not args.quick:
            from sis_common import full_max_finish_kwargs

            fm = full_max_finish_kwargs(sis_class)
            ilp_limit = max(ilp_limit, fm["ilp_time_limit"])
            ilp_mode = fm["ilp_mode"]
        print(f"[p{pid}] finish mode={ilp_mode} ilp={ilp_limit}s ...", flush=True)
        finish_rep = execute_finish(
            inst_path,
            best_path,
            finish_path,
            ilp_mode=ilp_mode,
            ilp_time_limit=ilp_limit,
            ilp_workers=8 if args.full_max else 4,
            skip_sub_bkz=not args.full_max,
            seed=args.seed + pid * 1009,
            euclid_polish=(sis_class == 3),
        )
        rec["finish"] = {
            "ilp_mode": finish_rep.get("ilp_mode"),
            "success": finish_rep.get("success"),
            "verify": finish_rep.get("verify"),
            "phases": [ph.get("name") for ph in finish_rep.get("phases", [])],
            "elapsed_sec": finish_rep.get("elapsed_sec"),
        }
        rec["final"] = finish_rep.get("verify")
        rec["final_score"] = score_from_verify(inst, finish_rep.get("verify") or {}, sis_class=sis_class)
        rec["success"] = bool(finish_rep.get("success"))
        total_score += rec["final_score"]["score"]
        v = finish_rep.get("verify") or {}
        print(
            f"[p{pid}] done success={rec['success']} E_inf={rec['final_score']['e_inf']} "
            f"score={rec['final_score']['score']} norm_ok={v.get('norm_req_ok')}",
            flush=True,
        )
        results.append(rec)

    report = {
        "problems": pids,
        "batch_rounds": batch_rounds,
        "ilp_time_limit": ilp_limit,
        "total_competition_score": total_score,
        "max_possible_score": 10 * len(pids),
        "elapsed_sec": time.time() - t0,
        "results": results,
    }
    rep_path = os.path.join(out_dir, "full_validation_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n=== Validation done: score {total_score}/{10 * len(pids)} in {report['elapsed_sec']:.0f}s ===")
    print(f"Report: {rep_path}")
    for r in results:
        fs = r.get("final_score") or {}
        print(
            f"  p{r['id']} class{r['class']}: E_inf={fs.get('e_inf')} "
            f"score={fs.get('score')} success={r.get('success')}"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        main_batch()
    else:
        main_full()
