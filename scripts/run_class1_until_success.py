"""
循环求解第一类题（齐次 SIS，官方题号 1、3、6、9）直至可行或达到轮次上限。

与 ``run_class_batch.py --class 1`` 类似，但：
- 默认题号 ``1,3,6,9``；
- 默认输出目录 ``results/class1``；
- 摘要写入 ``summary.json``（非 batch_report.json）；
- 未可行时以 inf_u+inf_v 保留较优进展。

用法
----
  cd sisinf_challenge2026
  python scripts/run_class1_until_success.py --max-rounds 12
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from run_problem1_until_success import _cfg_for_round
from sis_problem_taxonomy import (
    CLASS_1_IDS,
    class_label,
    effective_require_norm_ge_q2,
    problem_class_from_id,
)
from solve_sisinf import apply_sis_class_defaults, local_search_one, verify_solution


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data[0]
    return data


def main() -> None:
    p = argparse.ArgumentParser(description="Solve class-1 SIS∞ problems until success.")
    p.add_argument(
        "--problems",
        default="1,3,6,9",
        help="Comma-separated problem ids (default: all class 1)",
    )
    p.add_argument(
        "--json-dir",
        default=os.path.join("saiti1", "sis_inf_problems_json"),
        help="Directory with problemN.json files",
    )
    p.add_argument("--output-dir", default="results/class1")
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--max-rounds", type=int, default=0, help="0 = unlimited per problem")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    problem_ids = [int(x.strip()) for x in args.problems.split(",") if x.strip()]
    for pid in problem_ids:
        if pid not in CLASS_1_IDS:
            raise SystemExit(f"problem {pid} is not class 1 (expected subset of {sorted(CLASS_1_IDS)})")

    summary: List[Dict[str, Any]] = []
    for pid in problem_ids:
        path = os.path.join(args.json_dir, f"problem{pid}.json")
        inst = _load(path)
        cls = problem_class_from_id(pid)
        A = np.array(inst["A"], dtype=np.int64)
        t = np.array(inst["t"], dtype=np.int64)
        q, gamma = int(inst["q"]), int(inst["gamma"])
        require_norm = effective_require_norm_ge_q2(inst, cls)

        round_idx = 0
        best: Dict[str, Any] = {}
        t_start = time.time()
        while args.max_rounds == 0 or round_idx < args.max_rounds:
            seed = args.seed + pid * 1000 + round_idx * 17
            cfg = apply_sis_class_defaults(_cfg_for_round(round_idx, seed), cls, aggressive=round_idx >= 8)
            u, v, meta = local_search_one(A, t, q, gamma, cfg, require_norm)
            ok, verify = verify_solution(A, t, q, gamma, u, v, require_norm)
            rec = {
                "id": pid,
                "class": cls,
                "class_label": class_label(cls),
                "round": round_idx,
                "success": ok,
                "verify": verify,
                "meta": meta,
                "elapsed_total_sec": time.time() - t_start,
            }
            out_path = os.path.join(args.output_dir, f"problem{pid}_latest.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"u": u.tolist(), "v": v.tolist(), **rec}, f, indent=2)
            if not best or verify.get("inf_u", 999) + verify.get("inf_v", 999) < best.get("verify", {}).get(
                "inf_u", 999
            ) + best.get("verify", {}).get("inf_v", 999):
                best = rec
                best["u"], best["v"] = u.tolist(), v.tolist()
            print(f"problem{pid} round {round_idx}: success={ok} verify={verify}")
            if ok:
                with open(os.path.join(args.output_dir, f"problem{pid}_solution.json"), "w", encoding="utf-8") as f:
                    json.dump({"id": pid, "u": u.tolist(), "v": v.tolist(), "verify": verify, "meta": meta}, f, indent=2)
                break
            round_idx += 1
        summary.append(best)

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
