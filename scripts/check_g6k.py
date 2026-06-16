"""检测 G6K（BDGL2 真筛法）是否可用。"""

from __future__ import annotations

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "scripts"))


def main() -> int:
    from lattice_g6k import g6k_available, lattice_sieve_backend_label

    print(f"sieve backend label: {lattice_sieve_backend_label()}")
    if not g6k_available():
        print("G6K: NOT installed")
        print("Install (Linux):")
        print("  bash scripts/install_g6k.sh")
        return 1
    print("G6K: OK")
    try:
        import numpy as np
        from lattice_g6k import collect_g6k_v_seeds

        rng = np.random.default_rng(0)
        A = rng.integers(0, 97, size=(20, 20), dtype=np.int64)
        vs = collect_g6k_v_seeds(A, 97, 15, 28, 8, 80, rng, saturation_ratio=0.5)
        print(f"smoke g6k seeds on 40-dim toy: {len(vs)}")
    except Exception as e:
        print("G6K import OK but smoke failed:", e)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
