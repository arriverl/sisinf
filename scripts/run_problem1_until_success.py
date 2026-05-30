import argparse
import json
import os
import time
from typing import Any, Dict

import numpy as np

from sis_problem_taxonomy import effective_require_norm_ge_q2, problem_class_from_id
from solve_sisinf import SearchConfig, apply_sis_class_defaults, local_search_one, verify_solution


def _load_one(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            raise ValueError("empty input list")
        return data[0]
    raise ValueError("input must be a list with one instance")


def _cfg_for_round(round_idx: int, seed: int) -> SearchConfig:
    # Progressive schedule: strengthen exact block repair over time.
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


def main() -> None:
    p = argparse.ArgumentParser(description="Repeat problem1 solving until feasible.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", default="results/problem1_progress.json")
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--max-rounds", type=int, default=0, help="0 means infinite")
    args = p.parse_args()

    inst = _load_one(args.input)
    A = np.array(inst["A"], dtype=np.int64)
    t = np.array(inst["t"], dtype=np.int64)
    q = int(inst["q"])
    gamma = int(inst["gamma"])
    pid = int(inst.get("id", 1))
    sis_class = problem_class_from_id(pid)
    require_norm_ge_q2 = effective_require_norm_ge_q2(inst, sis_class)

    best_rec: Dict[str, Any] = {}
    round_idx = 0
    t0 = time.time()
    while True:
        if args.max_rounds > 0 and round_idx >= args.max_rounds:
            break
        seed = int(args.seed + round_idx * 100003)
        cfg = apply_sis_class_defaults(
            _cfg_for_round(round_idx, seed), sis_class, aggressive=round_idx >= 8
        )
        u, v, meta = local_search_one(A, t, q, gamma, cfg, require_norm_ge_q2)
        ok, verify = verify_solution(A, t, q, gamma, u, v, require_norm_ge_q2)
        rec = {
            "round": round_idx,
            "seed": seed,
            "success": bool(ok),
            "meta": meta,
            "verify": verify,
            "u": u.tolist(),
            "v": v.tolist(),
            "elapsed_total_sec": time.time() - t0,
        }
        with open(args.checkpoint, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        better = False
        if not best_rec:
            better = True
        else:
            cur_key = (
                int(meta.get("max_overflow", 10**9)),
                int(meta.get("violations", 10**9)),
                int(meta.get("overflow_sum", 10**9)),
            )
            best_key = (
                int(best_rec["meta"].get("max_overflow", 10**9)),
                int(best_rec["meta"].get("violations", 10**9)),
                int(best_rec["meta"].get("overflow_sum", 10**9)),
            )
            better = cur_key < best_key
        if better:
            best_rec = rec
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(
                    {"summary": {"success": bool(ok), "round": round_idx}, "result": rec},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        print(
            json.dumps(
                {
                    "round": round_idx,
                    "success": bool(ok),
                    "violations": int(meta.get("violations", -1)),
                    "max_overflow": int(meta.get("max_overflow", -1)),
                    "inf_u": int(verify.get("inf_u", -1)),
                    "inf_v": int(verify.get("inf_v", -1)),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if ok:
            return
        round_idx += 1


if __name__ == "__main__":
    main()

