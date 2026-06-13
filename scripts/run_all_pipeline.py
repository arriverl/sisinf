"""
全赛题（1–10，三类）批量流水线：启发式 batch → ILP 收尾 → 汇总报告。

默认策略（基于题 1/3 实验）
----------------------------
- 第一类 1/3/6/9：batch + full ILP，跳过 sub-BKZ
- 第二类 2/4/7/10：batch + full ILP（CVP 种子由 ``apply_sis_class_defaults`` 启用）
- 第三类 5/8：batch + lex ILP + 欧氏抛光

用法（项目根目录）::

  # 快速筛查（短 batch、短 ILP，约数小时量级）
  python3 scripts/run_all_pipeline.py --quick

  # 正式长跑（每题 batch 6 轮 + ILP 3600s，总计可达数日）
  python3 scripts/run_all_pipeline.py --batch-rounds 6 --ilp-time-limit 3600

  # 只跑第二类
  python3 scripts/run_all_pipeline.py --classes 2 --problems 2,4,7,10

  # 跳过已可行题、跳过 batch 仅 ILP
  python3 scripts/run_all_pipeline.py --skip-batch --skip-solved
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from finish_core import (
    default_ilp_mode_for_class,
    execute_finish,
    load_instance,
    save_incumbent_from_record,
)
from run_class_batch import solve_one
from sis_problem_taxonomy import (
    ALL_IDS,
    CLASS_1_IDS,
    CLASS_2_IDS,
    CLASS_3_IDS,
    class_label,
    problem_class_from_id,
)


def _problem_ids(classes: List[int], problems: Optional[List[int]]) -> List[int]:
    if problems:
        for pid in problems:
            if pid not in ALL_IDS:
                raise SystemExit(f"unknown problem id {pid}")
        return sorted(problems)
    out: Set[int] = set()
    for c in classes:
        if c == 1:
            out |= CLASS_1_IDS
        elif c == 2:
            out |= CLASS_2_IDS
        elif c == 3:
            out |= CLASS_3_IDS
        else:
            raise SystemExit(f"invalid class {c}")
    return sorted(out)


def _paths(root: str, pid: int, sis_class: int) -> Dict[str, str]:
    base = os.path.join(root, "results", "pipeline")
    cls_dir = os.path.join(base, f"class{sis_class}")
    os.makedirs(cls_dir, exist_ok=True)
    inst = os.path.join(root, "saiti1", "sis_inf_problems_json", f"problem{pid}.json")
    return {
        "instance": inst,
        "best": os.path.join(cls_dir, f"problem{pid}_best.json"),
        "finish": os.path.join(cls_dir, f"problem{pid}_finish.json"),
        "solution": os.path.join(cls_dir, f"problem{pid}_solution.json"),
    }


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description="Run all SIS∞ problems: batch + ILP finish.")
    p.add_argument("--classes", default="1,2,3", help="e.g. 1,2,3")
    p.add_argument("--problems", default="", help="override ids, e.g. 1,3,6")
    p.add_argument("--json-dir", default=os.path.join(root, "saiti1", "sis_inf_problems_json"))
    p.add_argument("--output-dir", default=os.path.join(root, "results", "pipeline"))
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--batch-rounds", type=int, default=6)
    p.add_argument("--quick", action="store_true", help="短 batch + 短 ILP（筛查用）")
    p.add_argument("--skip-batch", action="store_true", help="仅用已有 problem*_best.json 做 ILP")
    p.add_argument("--skip-finish", action="store_true", help="只跑 batch")
    p.add_argument("--skip-solved", action="store_true", help="跳过已有 solution 的题")
    p.add_argument("--ilp-time-limit", type=float, default=3600.0)
    p.add_argument("--ilp-workers", type=int, default=4)
    p.add_argument("--ilp-mode", default="auto", choices=["auto", "full", "chunk", "lex"])
    args = p.parse_args()

    classes = [int(x.strip()) for x in args.classes.split(",") if x.strip()]
    problems_arg = (
        [int(x.strip()) for x in args.problems.split(",") if x.strip()] if args.problems else None
    )
    pids = _problem_ids(classes, problems_arg)

    batch_rounds = 2 if args.quick else args.batch_rounds
    ilp_limit = 120.0 if args.quick else args.ilp_time_limit

    summary: List[Dict[str, Any]] = []
    t0 = time.time()
    solved_count = 0

    print(f"=== Pipeline: problems={pids} batch_rounds={batch_rounds} ilp_limit={ilp_limit}s ===", flush=True)

    for pid in pids:
        sis_class = problem_class_from_id(pid)
        paths = _paths(root, pid, sis_class)
        cls_dir = os.path.dirname(paths["best"])
        rec: Dict[str, Any] = {
            "id": pid,
            "class": sis_class,
            "class_label": class_label(sis_class),
            "batch": None,
            "finish": None,
            "success": False,
        }

        if args.skip_solved and os.path.isfile(paths["solution"]):
            sol = _load_json(paths["solution"])
            rec["success"] = True
            rec["skipped"] = "already_solved"
            rec["verify"] = sol.get("verify") if sol else {}
            summary.append(rec)
            solved_count += 1
            print(f"[p{pid}] skip (already solved)", flush=True)
            continue

        # --- Phase 1: batch ---
        if not args.skip_batch:
            print(f"[p{pid}] class{sis_class} batch x{batch_rounds} ...", flush=True)
            inst = load_instance(paths["instance"])
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
                "rounds_tried": batch_rec.get("rounds_tried"),
                "verify": batch_rec.get("verify"),
                "success": batch_rec.get("success"),
                "elapsed_sec": batch_rec.get("elapsed_sec"),
            }
            if batch_rec.get("success"):
                save_incumbent_from_record(batch_rec, paths["solution"])
                with open(paths["solution"], "w", encoding="utf-8") as f:
                    json.dump(batch_rec, f, indent=2, ensure_ascii=False)
                rec["success"] = True
                rec["verify"] = batch_rec.get("verify")
                summary.append(rec)
                solved_count += 1
                print(f"[p{pid}] SOLVED in batch", flush=True)
                continue

            if "u" in batch_rec and "v" in batch_rec:
                save_incumbent_from_record(batch_rec, paths["best"])
            elif os.path.isfile(os.path.join(cls_dir, f"problem{pid}_latest.json")):
                latest = _load_json(os.path.join(cls_dir, f"problem{pid}_latest.json"))
                if latest and "u" in latest:
                    save_incumbent_from_record(latest, paths["best"])
        elif not os.path.isfile(paths["best"]):
            print(f"[p{pid}] ERROR: no incumbent {paths['best']} (run batch first)", flush=True)
            rec["error"] = "missing_incumbent"
            summary.append(rec)
            continue

        # --- Phase 2: ILP finish ---
        if args.skip_finish:
            best = _load_json(paths["best"])
            rec["verify"] = best.get("verify") if best else {}
            summary.append(rec)
            continue

        if not os.path.isfile(paths["best"]):
            rec["error"] = "no_best_after_batch"
            summary.append(rec)
            continue

        ilp_mode = None if args.ilp_mode == "auto" else args.ilp_mode
        if ilp_mode is None:
            ilp_mode = default_ilp_mode_for_class(sis_class)

        print(f"[p{pid}] finish ILP mode={ilp_mode} limit={ilp_limit}s ...", flush=True)
        finish_rep = execute_finish(
            paths["instance"],
            paths["best"],
            paths["finish"],
            ilp_mode=ilp_mode,
            ilp_time_limit=ilp_limit,
            ilp_workers=args.ilp_workers,
            skip_sub_bkz=True,
            seed=args.seed + pid * 1009,
        )
        rec["finish"] = {
            "ilp_mode": finish_rep.get("ilp_mode"),
            "verify": finish_rep.get("verify"),
            "success": finish_rep.get("success"),
            "elapsed_sec": finish_rep.get("elapsed_sec"),
            "phases": [ph.get("name") for ph in finish_rep.get("phases", [])],
        }
        rec["success"] = bool(finish_rep.get("success"))
        rec["verify"] = finish_rep.get("verify")
        if rec["success"]:
            save_incumbent_from_record(finish_rep, paths["solution"])
            solved_count += 1
            print(f"[p{pid}] SOLVED after finish", flush=True)
        else:
            save_incumbent_from_record(finish_rep, paths["best"])
            print(
                f"[p{pid}] best inf_u={rec['verify'].get('inf_u')} inf_v={rec['verify'].get('inf_v')} "
                f"norm_ok={rec['verify'].get('norm_req_ok')}",
                flush=True,
            )
        summary.append(rec)

    report = {
        "problems": pids,
        "classes": classes,
        "batch_rounds": batch_rounds,
        "ilp_time_limit": ilp_limit,
        "quick": args.quick,
        "solved": solved_count,
        "total": len(pids),
        "all_success": solved_count == len(pids),
        "elapsed_sec": time.time() - t0,
        "results": summary,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    rep_path = os.path.join(args.output_dir, "all_pipeline_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(
        f"\n=== Pipeline done: {solved_count}/{len(pids)} feasible in {report['elapsed_sec']:.0f}s ===",
        flush=True,
    )
    print(f"Report: {rep_path}", flush=True)
    for r in summary:
        v = r.get("verify") or {}
        print(
            f"  p{r['id']} class{r['class']}: success={r.get('success')} "
            f"inf_u={v.get('inf_u')} inf_v={v.get('inf_v')} norm_ok={v.get('norm_req_ok')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
