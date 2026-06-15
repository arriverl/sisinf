"""快速冒烟：验证新算法模块可导入且种子生成不报错。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "scripts"))

from lattice_kannan import collect_kannan_v_seeds
from lattice_restricted_svp import collect_restricted_svp_v_seeds
from lattice_sieve import collect_sieve_v_seeds
from sis_scoring import score_from_verify
from solve_sisinf import verify_solution


def _load(pid: int):
    p = os.path.join(_root, "saiti1", "sis_inf_problems_json", f"problem{pid}.json")
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d[0] if isinstance(d, list) else d


def main():
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
            seeds = collect_sieve_v_seeds(A, q, gamma, 20, 8, 220, rng, combo_depth=3)
            print(f"  sieve seeds: {len(seeds)}")
        if pid == 2:
            seeds = collect_kannan_v_seeds(A, t, q, gamma, 20, 8, 200, rng)
            print(f"  kannan seeds: {len(seeds)}")
        if pid == 5:
            seeds = collect_restricted_svp_v_seeds(
                A, t, q, gamma, rng, 12, require_norm_lt_q2=True
            )
            print(f"  restricted seeds: {len(seeds)}")
    print("smoke OK")


if __name__ == "__main__":
    main()
