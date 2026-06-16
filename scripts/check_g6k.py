"""检测 G6K（BDGL2 真筛法）是否可用。"""

from __future__ import annotations

import json
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "scripts"))


def main() -> int:
    from lattice_g6k import collect_g6k_v_seeds, g6k_available, lattice_sieve_backend_label

    print(f"sieve backend label: {lattice_sieve_backend_label()}")
    if not g6k_available():
        print("G6K: NOT installed")
        print("Install (Linux):")
        print("  bash scripts/install_g6k.sh")
        return 1
    print("G6K: OK")

    import numpy as np

    rng = np.random.default_rng(0)

    # 官方题 1（Ajtai 40 维玩具不可靠，用真实实例）
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
            A,
            q,
            gamma,
            beta=28,
            max_vectors=16,
            max_dim=260,
            rng=rng,
            saturation_ratio=0.55,
            threads=2,
            bkz_block=28,
        )
        print(f"smoke g6k seeds on {label} (d={A.shape[0] + A.shape[1]}): {len(vs)}")
        if len(vs) == 0:
            print("WARN: 0 v seeds after clip — lattice vectors may still exist (see raw in debug)")
        return 0
    except Exception as e:
        print("G6K import OK but smoke failed:", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
