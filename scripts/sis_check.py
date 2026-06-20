"""
环境自检与冒烟：fpylll / G6K / ILP / 全模块 / 种子生成。

用法::

  python scripts/sis_check.py              # 全部
  python scripts/sis_check.py algorithms   # 全模块（原 check_algorithms）
  python scripts/sis_check.py fpylll
  python scripts/sis_check.py g6k
  python scripts/sis_check.py ilp
  python scripts/sis_check.py smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_script_dir = os.path.join(_root, "scripts")
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

OK = 0
WARN = 0


def _line(name: str, status: str, detail: str = "") -> None:
    global OK, WARN
    tag = "OK" if status == "ok" else ("WARN" if status == "warn" else "FAIL")
    if tag == "OK":
        OK += 1
    elif tag == "WARN":
        WARN += 1
    else:
        print(f"[FAIL] {name}: {detail}")
        sys.exit(1)
    extra = f" — {detail}" if detail else ""
    print(f"[{tag}] {name}{extra}")


def _load(pid: int):
    p = os.path.join(_root, "saiti1", "sis_inf_problems_json", f"problem{pid}.json")
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d[0] if isinstance(d, list) else d


def check_fpylll() -> None:
    try:
        from fpylll import BKZ, IntegerMatrix, LLL

        print("OK: fpylll 已安装，可使用真 BKZ/LLL。")
        M = IntegerMatrix(4, 4)
        for i in range(4):
            M[i, i] = 1
        LLL.reduction(M)
        BKZ.reduction(M, BKZ.Param(block_size=2))
        print("OK: LLL + BKZ 试跑成功。")
    except ImportError as e:
        print("未安装 fpylll:", e)
        print("\n安装建议：conda install -c conda-forge fpylll")
        sys.exit(1)
    except Exception as e:
        print("fpylll 已导入但运行失败:", e)
        sys.exit(1)


def check_g6k() -> int:
    from lattice_seeds import collect_g6k_v_seeds, g6k_available, lattice_sieve_backend_label

    print(f"sieve backend label: {lattice_sieve_backend_label()}")
    if not g6k_available():
        print("G6K: NOT installed")
        print("Install (Linux): bash scripts/install_g6k.sh")
        return 1
    print("G6K: OK")

    rng = np.random.default_rng(0)
    prob_path = os.path.join(_root, "saiti1", "sis_inf_problems_json", "problem1.json")
    if os.path.isfile(prob_path):
        with open(prob_path, encoding="utf-8") as f:
            inst = json.load(f)
        if isinstance(inst, list):
            inst = inst[0]
        A = np.array(inst["A"], dtype=np.int64)
        q, gamma = int(inst["q"]), int(inst["gamma"])
        label = "problem1"
    else:
        A = np.eye(20, dtype=np.int64) * 97
        q, gamma = 97, 15
        label = "identity toy"

    try:
        vs = collect_g6k_v_seeds(
            A, q, gamma, beta=28, max_vectors=16, max_dim=260, rng=rng,
            saturation_ratio=0.55, threads=2, bkz_block=28,
        )
        print(f"smoke g6k seeds on {label} (d={A.shape[0] + A.shape[1]}): {len(vs)}")
        return 0
    except Exception as e:
        print("G6K import OK but smoke failed:", e)
        return 2


def check_ilp_finish() -> None:
    try:
        from ortools.sat.python import cp_model  # type: ignore

        print("OK: ortools import")
    except Exception as e:
        print("FAIL: ortools import:", e)
        sys.exit(1)

    try:
        from sis_finish import cp_sat_full_v_linf_finish, run_ilp_finish

        m = cp_model.CpModel()
        x = m.NewIntVar(0, 3, "x")
        m.Minimize(x)
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = 1.0
        st = s.Solve(m)
        print("OK: tiny CP-SAT solve, status=", st)
        _ = run_ilp_finish
        _ = cp_sat_full_v_linf_finish
    except Exception as e:
        print("FAIL: tiny solve:", e)
        sys.exit(1)

    prob = os.path.join(_root, "saiti1", "sis_inf_problems_json", "problem1.json")
    best = os.path.join(_root, "results", "class1", "problem1_best.json")
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
        print("OK: smoke inf_u=", meta.get("verify", {}).get("inf_u"))
    else:
        print("Skip problem1 smoke (files missing locally)")


def smoke_algorithms() -> None:
    from lattice_seeds import collect_kannan_v_seeds, collect_restricted_svp_v_seeds, collect_sieve_v_seeds
    from sis_common import score_from_verify
    from solve_sisinf import verify_solution

    rng = np.random.default_rng(42)
    for pid in [1, 2, 5]:
        inst = _load(pid)
        A = np.array(inst["A"], dtype=np.int64)
        t = np.array(inst["t"], dtype=np.int64)
        q, gamma = int(inst["q"]), int(inst["gamma"])
        u = np.zeros(A.shape[0], dtype=np.int64)
        v = np.zeros(A.shape[1], dtype=np.int64)
        ok, ver = verify_solution(A, t, q, gamma, u, v, require_norm_lt_q2=(pid in (5, 8)))
        sc = score_from_verify(inst, ver, sis_class={1: 1, 2: 2, 5: 3}[pid])
        print(f"p{pid} verify congr={ver['congruence_ok']} norm_ok={ver['norm_req_ok']} score={sc['score']}")
        if pid == 1:
            print(f"  sieve seeds: {len(collect_sieve_v_seeds(A, q, gamma, 20, 8, 220, rng, combo_depth=3))}")
        if pid == 2:
            print(f"  kannan seeds: {len(collect_kannan_v_seeds(A, t, q, gamma, 20, 8, 200, rng))}")
        if pid == 5:
            print(f"  restricted seeds: {len(collect_restricted_svp_v_seeds(A, t, q, gamma, rng, 12, require_norm_lt_q2=True))}")
    print("smoke OK")


def check_algorithms() -> None:
    global OK, WARN
    OK, WARN = 0, 0
    _line("numpy", "ok")

    try:
        from lattice_seeds import fpylll_available, lattice_backend_label

        if fpylll_available():
            _line("fpylll", "ok", lattice_backend_label())
        else:
            _line("fpylll", "warn", "not installed; BKZ/Kannan/sieve use heuristic fallback")
    except Exception as e:
        _line("fpylll", "warn", str(e))

    try:
        from sis_finish import run_ilp_finish

        _line("ortools CP-SAT", "ok", f"run_ilp_finish={run_ilp_finish.__name__}")
    except Exception as e:
        _line("ortools", "warn", str(e))

    from lattice_seeds import (
        collect_g6k_v_seeds,
        collect_kannan_v_seeds,
        collect_restricted_svp_v_seeds,
        collect_sieve_v_seeds,
        g6k_available,
        wang_restricted_svp_v_seeds,
    )
    from sis_common import problem_class_from_id, score_from_verify
    from solve_sisinf import SearchConfig, apply_sis_class_defaults, verify_solution

    _line("imports", "ok", "lattice_seeds/sis_finish/solver")

    rng = np.random.default_rng(0)
    inst1 = _load(1)
    A1 = np.array(inst1["A"], dtype=np.int64)
    q, g = int(inst1["q"]), int(inst1["gamma"])
    n_sieve = len(collect_sieve_v_seeds(A1, q, g, 20, 8, 220, rng, combo_depth=3))
    _line("lattice_sieve", "ok" if n_sieve > 0 else "warn", f"seeds={n_sieve}")

    if g6k_available():
        n_g6k = len(collect_g6k_v_seeds(A1, q, g, 28, 12, 260, rng, threads=2, saturation_ratio=0.5))
        _line("lattice_g6k", "ok" if n_g6k > 0 else "warn", f"seeds={n_g6k}")
    else:
        _line("lattice_g6k", "warn", "not installed")

    inst2 = _load(2)
    A2 = np.array(inst2["A"], dtype=np.int64)
    t2 = np.array(inst2["t"], dtype=np.int64)
    n_kannan = len(collect_kannan_v_seeds(A2, t2, q, g, 20, 8, 200, rng))
    _line("lattice_kannan", "ok" if n_kannan > 0 else "warn", f"seeds={n_kannan}")

    inst5 = _load(5)
    A5 = np.array(inst5["A"], dtype=np.int64)
    t5 = np.array(inst5["t"], dtype=np.int64)
    n_wang = len(wang_restricted_svp_v_seeds(A5, t5, q, g, rng, 12, require_norm_lt_q2=True, enum_max_trials=2000))
    _line("wang_restricted_svp", "ok" if n_wang > 0 else "warn", f"seeds={n_wang}")

    n_rs = len(collect_restricted_svp_v_seeds(A5, t5, q, g, rng, 12, require_norm_lt_q2=True, enum_max_trials=2000))
    _line("lattice_restricted_svp", "ok" if n_rs > 0 else "warn", f"seeds={n_rs}")

    u_big = np.full(int(inst5["n"]), g, dtype=np.int64)
    v_big = np.full(int(inst5["m"]), g, dtype=np.int64)
    _, ver_bad = verify_solution(A5, t5, q, g, u_big, v_big, require_norm_lt_q2=True)
    if ver_bad.get("norm_req_ok") == 0:
        _line("norm_lt_q2 reject", "ok", f"norm_sq={ver_bad.get('norm_sq')}")
    else:
        _line("norm_lt_q2 reject", "fail", "expected norm_req_ok=0")

    for pid in [1, 2, 5]:
        cls = problem_class_from_id(pid)
        cfg = apply_sis_class_defaults(SearchConfig(), cls)
        flags = f"sieve={cfg.use_sieve_seeds} kannan={cfg.use_kannan_seeds} restricted={cfg.use_restricted_svp_seeds}"
        _line(f"class{cls} defaults p{pid}", "ok", flags)

    print(f"\n=== check_algorithms: {OK} ok, {WARN} warn ===")


def main() -> None:
    p = argparse.ArgumentParser(description="SIS∞ environment checks")
    p.add_argument(
        "cmd",
        nargs="?",
        default="all",
        choices=["all", "fpylll", "g6k", "ilp", "smoke", "algorithms"],
    )
    args = p.parse_args()
    if args.cmd in ("all", "fpylll"):
        check_fpylll()
    if args.cmd in ("all", "g6k"):
        rc = check_g6k()
        if args.cmd == "g6k":
            raise SystemExit(rc)
    if args.cmd in ("all", "ilp"):
        check_ilp_finish()
    if args.cmd in ("all", "smoke"):
        smoke_algorithms()
    if args.cmd in ("all", "algorithms"):
        check_algorithms()


if __name__ == "__main__":
    main()
