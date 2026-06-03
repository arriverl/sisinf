"""
消融实验：对比 baseline / dual_space / BKZ / 完整配置等在相同实例上的成功率与耗时。

输出 CSV/JSON 报告，用于调参而非提交答案。
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

from solve_sisinf import SearchConfig, solve_instances


def build_variants(base: SearchConfig) -> Dict[str, SearchConfig]:
    """预定义若干 SearchConfig 变体名称 → 配置，用于横向对比。"""
    return {
        "baseline_iir_cls": SearchConfig(
            restarts=base.restarts,
            iters=base.iters,
            delta=base.delta,
            kick_size=base.kick_size,
            kick_every=base.kick_every,
            seed=base.seed,
            max_delta=base.delta,
            candidate_count=0,
            entropy_weight=0.0,
            euclid_weight=0.0,
            overflow_weight=base.overflow_weight,
            entropy_bins=base.entropy_bins,
            dynamic_schedule=False,
            use_dual_space=False,
            use_bkz_seeds=False,
            parallel_workers=1,
        ),
        "dual_space_only": SearchConfig(
            restarts=base.restarts,
            iters=base.iters,
            delta=base.delta,
            kick_size=base.kick_size,
            kick_every=base.kick_every,
            seed=base.seed,
            max_delta=base.max_delta,
            candidate_count=base.candidate_count,
            entropy_weight=0.0,
            euclid_weight=0.0,
            overflow_weight=base.overflow_weight,
            entropy_bins=base.entropy_bins,
            dynamic_schedule=False,
            use_dual_space=True,
            use_bkz_seeds=False,
            parallel_workers=1,
        ),
        "full_hybrid_entropy": base,
    }


def summarize(results: List[Dict]) -> Dict[str, float]:
    n = len(results)
    success = sum(1 for r in results if r["success"])
    avg_time = sum(r["elapsed_sec"] for r in results) / max(n, 1)
    avg_overflow = sum(r["meta"].get("overflow_sum", 0) for r in results) / max(n, 1)
    return {
        "num_instances": float(n),
        "num_success": float(success),
        "success_rate": float(success) / max(n, 1),
        "avg_time_sec": float(avg_time),
        "avg_overflow_sum": float(avg_overflow),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline/ablation for SIS∞ solver.")
    parser.add_argument("--input", required=True, help="Input instances JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    parser.add_argument("--restarts", type=int, default=12)
    parser.add_argument("--iters", type=int, default=1200)
    parser.add_argument("--delta", type=int, default=2)
    parser.add_argument("--max-delta", type=int, default=6)
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument("--entropy-weight", type=float, default=0.25)
    parser.add_argument("--euclid-weight", type=float, default=1.5)
    parser.add_argument("--overflow-weight", type=float, default=1.0)
    parser.add_argument("--entropy-bins", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    instances = json.loads(Path(args.input).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = SearchConfig(
        restarts=args.restarts,
        iters=args.iters,
        delta=args.delta,
        kick_size=6,
        kick_every=120,
        seed=args.seed,
        max_delta=args.max_delta,
        candidate_count=args.candidate_count,
        entropy_weight=args.entropy_weight,
        euclid_weight=args.euclid_weight,
        overflow_weight=args.overflow_weight,
        entropy_bins=args.entropy_bins,
        dynamic_schedule=True,
        use_dual_space=True,
        use_bkz_seeds=False,
        parallel_workers=1,
    )

    variants = build_variants(base_cfg)
    summary_rows: List[Dict[str, float]] = []

    for idx, (name, cfg) in enumerate(variants.items()):
        cfg.seed = args.seed + idx * 1000
        t0 = time.time()
        results = solve_instances(instances, cfg)
        elapsed = time.time() - t0
        payload = {
            "variant": name,
            "config": cfg.__dict__,
            "summary": summarize(results),
            "results": results,
            "wall_time_sec": elapsed,
        }
        (out_dir / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        row = payload["summary"]
        row["variant"] = name
        row["wall_time_sec"] = elapsed
        summary_rows.append(row)
        print(f"[OK] {name}: success={int(row['num_success'])}/{int(row['num_instances'])}")

    csv_path = out_dir / "ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "num_instances",
                "num_success",
                "success_rate",
                "avg_time_sec",
                "avg_overflow_sum",
                "wall_time_sec",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"[DONE] summary -> {csv_path}")


if __name__ == "__main__":
    main()
