"""检测 ortools 与全维 v ILP 收尾是否可用。"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)


def main() -> None:
    try:
        from ortools.sat.python import cp_model  # type: ignore

        print("OK: ortools import")
    except Exception as e:
        print("FAIL: ortools import:", e)
        sys.exit(1)

    try:
        from sis_finish_ops import cp_sat_full_v_linf_finish, run_ilp_finish

        m = cp_model.CpModel()
        x = m.NewIntVar(0, 3, "x")
        m.Minimize(x)
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = 1.0
        st = s.Solve(m)
        print("OK: tiny CP-SAT solve, status=", st)
    except Exception as e:
        print("FAIL: tiny solve:", e)
        sys.exit(1)

    root = os.path.dirname(_script_dir)
    prob = os.path.join(root, "saiti1", "sis_inf_problems_json", "problem1.json")
    best = os.path.join(root, "results", "class1", "problem1_best.json")
    if os.path.isfile(prob) and os.path.isfile(best):
        with open(prob, encoding="utf-8") as f:
            inst = json.load(f)[0]
        with open(best, encoding="utf-8") as f:
            inc = json.load(f)
        A = np.array(inst["A"], dtype=np.int64)
        t = np.array(inst["t"], dtype=np.int64)
        v0 = np.array(inc["v"], dtype=np.int64)
        q, gamma = int(inst["q"]), int(inst["gamma"])
        print("Smoke: problem1 incumbent, ilp limit=30s ...")
        u, v, meta = cp_sat_full_v_linf_finish(A, t, q, gamma, v0, time_limit_sec=30.0, num_workers=1)
        if u is None or not meta.get("ilp_ok"):
            print("FAIL:", meta)
            sys.exit(1)
        print("OK: smoke inf_u=", meta.get("verify", {}).get("inf_u"), "meta=", meta)
    else:
        print("Skip problem1 smoke (files missing locally)")


if __name__ == "__main__":
    main()
