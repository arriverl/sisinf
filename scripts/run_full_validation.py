"""
全算法接入验证：十题 ×（batch 启发式 + ILP 收尾）× 阶梯计分。

集成的算法模块
--------------
- 第一类：BKZ 2.0 + list sieve (``lattice_sieve``) + Wagner + 核游走 + full CP-SAT
- 第二类：Kannan 嵌入 (``lattice_kannan``) + CVP 提升 + full CP-SAT
- 第三类：受限 SVP (``lattice_restricted_svp``) + sieve + lex CP-SAT + 欧氏抛光

用法::

  python scripts/run_full_validation.py --quick
  python scripts/run_full_validation.py --batch-rounds 4 --ilp-time-limit 600
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from finish_core import default_ilp_mode_for_class, execute_finish, load_instance, save_incumbent_from_record
from run_class_batch import solve_one
from sis_problem_taxonomy import ALL_IDS, problem_class_from_id
from sis_scoring import score_from_verify


def main() -> None:
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
    args = p.parse_args()

    if args.problems:
        pids = sorted(int(x.strip()) for x in args.problems.split(",") if x.strip())
        for pid in pids:
            if pid not in ALL_IDS:
                raise SystemExit(f"unknown problem id {pid}")
    else:
        pids = sorted(ALL_IDS)

    batch_rounds = 1 if args.quick else args.batch_rounds
    ilp_limit = 60.0 if args.quick else args.ilp_time_limit
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
        print(f"[p{pid}] finish mode={ilp_mode} ...", flush=True)
        finish_rep = execute_finish(
            inst_path,
            best_path,
            finish_path,
            ilp_mode=ilp_mode,
            ilp_time_limit=ilp_limit,
            ilp_workers=4,
            skip_sub_bkz=True,
            seed=args.seed + pid * 1009,
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
    main()
