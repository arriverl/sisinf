"""
第一类（齐次 SIS）u 优先策略小型消融：
- A: 当前基线
- B: 提升 u 覆盖面（cp_aggressive_row_k/u_row_snap_cols）
- C: 更激进覆盖面
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from run_problem1_until_success import _cfg_for_round
from solve_sisinf import apply_sis_class_defaults, local_search_one, verify_solution


def run_one(label: str, cfg, A, t, q, gamma, seed: int):
    cfg = replace(cfg, seed=seed)
    t0 = time.time()
    u, v, meta = local_search_one(A, t, q, gamma, cfg, False)
    ok, verify = verify_solution(A, t, q, gamma, u, v, False)
    return {
        "label": label,
        "success": bool(ok),
        "elapsed_sec": round(time.time() - t0, 2),
        "verify": verify,
        "meta": {
            "violations": int(meta.get("violations", -1)),
            "max_overflow": int(meta.get("max_overflow", -1)),
            "overflow_sum": int(meta.get("overflow_sum", -1)),
            "dual_candidates": int(meta.get("dual_candidates", -1)),
        },
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="restarts=2, iters=500, timeout=60")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    p1 = root / "saiti1" / "sis_inf_problems_json" / "problem1.json"
    inst = json.loads(p1.read_text(encoding="utf-8"))[0]
    A = np.array(inst["A"], dtype=np.int64)
    t = np.array(inst["t"], dtype=np.int64)
    q, gamma = int(inst["q"]), int(inst["gamma"])

    tuned = apply_sis_class_defaults(_cfg_for_round(0, 424242), 1)
    if args.quick:
        tuned = replace(tuned, restarts=2, iters=500, timeout_sec=60.0, verbose=False)
    else:
        tuned = replace(tuned, restarts=3, iters=900, timeout_sec=120.0, verbose=False)

    # A=旧覆盖（对照）；B=当前第一类默认（u 优先 B 档）；C=更激进
    cfg_a = replace(
        tuned,
        cp_aggressive_row_k=20,
        u_row_snap_cols=12,
        u_row_snap_top_rows=8,
        u_row_snap_every=14,
        cp_aggressive_every=0,
    )
    cfg_b = tuned
    cfg_c = replace(
        tuned,
        cp_aggressive_row_k=40,
        u_row_snap_cols=32,
        u_row_snap_top_rows=20,
        u_row_snap_every=6,
        cp_aggressive_every=24,
        cp_repair_window=max(tuned.cp_repair_window, 8),
        cp_repair_time_limit=max(tuned.cp_repair_time_limit, 3.5),
        max_delta=max(tuned.max_delta, 14),
    )

    results = [
        run_one("A_baseline", cfg_a, A, t, q, gamma, 424242),
        run_one("B_cover32x24", cfg_b, A, t, q, gamma, 525252),
        run_one("C_cover40x32", cfg_c, A, t, q, gamma, 626262),
    ]

    out = root / "results" / "class1" / "u_priority_ablation_problem1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# problem1 u 优先消融（A/B/C）",
        "",
        "| 组 | inf_u | inf_v | violations | max_overflow | 耗时(s) | success |",
        "|----|-------|-------|------------|--------------|---------|---------|",
    ]
    for r in results:
        v = r["verify"]
        m = r["meta"]
        lines.append(
            f"| {r['label']} | {v.get('inf_u')} | {v.get('inf_v')} | "
            f"{m.get('violations')} | {m.get('max_overflow')} | {r['elapsed_sec']} | {r['success']} |"
        )
    best = min(results, key=lambda r: (r["verify"].get("inf_u", 999), r["meta"].get("max_overflow", 999)))
    lines.append("")
    lines.append(f"**当前最优组**：`{best['label']}`（inf_u={best['verify'].get('inf_u')}）")

    md_path = out.with_suffix(".md")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out}")
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
