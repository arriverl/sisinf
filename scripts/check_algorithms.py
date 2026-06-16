"""
环境与新算法模块自检（服务器部署后先跑本脚本）。

检查项：numpy、fpylll、ortools、四类格种子模块、第三类 norm 校验、阶梯计分。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "scripts"))

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


def main() -> None:
    import numpy as _np  # noqa: F401

    _line("numpy", "ok")

    try:
        from lattice_bkz import fpylll_available, lattice_backend_label

        if fpylll_available():
            _line("fpylll", "ok", lattice_backend_label())
        else:
            _line("fpylll", "warn", "not installed; BKZ/Kannan/sieve use heuristic fallback")
    except Exception as e:
        _line("fpylll", "warn", str(e))

    try:
        from sis_finish_ops import run_ilp_finish

        _line("ortools CP-SAT", "ok", f"run_ilp_finish={run_ilp_finish.__name__}")
    except Exception as e:
        _line("ortools", "warn", str(e))

    from lattice_sieve import collect_sieve_v_seeds
    from lattice_kannan import collect_kannan_v_seeds
    from lattice_restricted_svp import collect_restricted_svp_v_seeds, wang_restricted_svp_v_seeds
    from sis_scoring import score_from_verify
    from sis_problem_taxonomy import effective_require_norm_lt_q2, problem_class_from_id
    from solve_sisinf import apply_sis_class_defaults, local_search_one, verify_solution

    _line("imports", "ok", "sieve/kannan/restricted/scoring/solver")

    rng = np.random.default_rng(0)
    inst1 = _load(1)
    A1 = np.array(inst1["A"], dtype=np.int64)
    t1 = np.array(inst1["t"], dtype=np.int64)
    q, g = int(inst1["q"]), int(inst1["gamma"])
    n_sieve = len(collect_sieve_v_seeds(A1, q, g, 20, 8, 220, rng, combo_depth=3))
    _line("lattice_sieve", "ok" if n_sieve > 0 else "warn", f"seeds={n_sieve}")

    inst2 = _load(2)
    A2 = np.array(inst2["A"], dtype=np.int64)
    t2 = np.array(inst2["t"], dtype=np.int64)
    n_kannan = len(
        collect_kannan_v_seeds(A2, t2, q, g, 20, 8, 200, rng)
    )
    _line(
        "lattice_kannan",
        "ok" if n_kannan > 0 else "warn",
        f"seeds={n_kannan} (0 without fpylll is expected on some hosts)",
    )

    inst5 = _load(5)
    A5 = np.array(inst5["A"], dtype=np.int64)
    t5 = np.array(inst5["t"], dtype=np.int64)
    n_wang = len(
        wang_restricted_svp_v_seeds(
            A5, t5, q, g, rng, 12, require_norm_lt_q2=True, enum_max_trials=2000
        )
    )
    _line("wang_restricted_svp", "ok" if n_wang > 0 else "warn", f"seeds={n_wang}")

    n_rs = len(
        collect_restricted_svp_v_seeds(
            A5, t5, q, g, rng, 12, require_norm_lt_q2=True, enum_max_trials=2000
        )
    )
    _line("lattice_restricted_svp", "ok" if n_rs > 0 else "warn", f"seeds={n_rs}")

    # 第三类 norm：大 L2 应失败
    u_big = np.full(int(inst5["n"]), g, dtype=np.int64)
    v_big = np.full(int(inst5["m"]), g, dtype=np.int64)
    _, ver_bad = verify_solution(A5, t5, q, g, u_big, v_big, require_norm_lt_q2=True)
    if ver_bad.get("norm_req_ok") == 0:
        _line("norm_lt_q2 reject", "ok", f"norm_sq={ver_bad.get('norm_sq')}")
    else:
        _line("norm_lt_q2 reject", "fail", "expected norm_req_ok=0 for dense solution")

    # 类默认启用新算法
    for pid in [1, 2, 5]:
        inst = _load(pid)
        cls = problem_class_from_id(pid)
        cfg = apply_sis_class_defaults(
            __import__("solve_sisinf", fromlist=["SearchConfig"]).SearchConfig(), cls
        )
        flags = (
            f"sieve={cfg.use_sieve_seeds} kannan={cfg.use_kannan_seeds} "
            f"restricted={cfg.use_restricted_svp_seeds}"
        )
        _line(f"class{cls} defaults p{pid}", "ok", flags)

    print(f"\n=== check_algorithms: {OK} ok, {WARN} warn ===")
    print("Next: bash scripts/run_server_validation.sh")


if __name__ == "__main__":
    main()
