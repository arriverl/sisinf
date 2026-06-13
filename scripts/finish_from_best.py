"""CLI：从 incumbent 执行 ILP 收尾（委托 ``finish_core.execute_finish``）。"""

from __future__ import annotations

import argparse
import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from finish_core import execute_finish


def main() -> None:
    p = argparse.ArgumentParser(description="ILP finish (full/chunk/lex) + optional polish from incumbent.")
    p.add_argument("--instance", required=True)
    p.add_argument("--incumbent", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--ilp-mode", choices=["full", "chunk", "lex", "auto"], default="auto")
    p.add_argument("--ilp-time-limit", type=float, default=3600.0)
    p.add_argument("--ilp-workers", type=int, default=4)
    p.add_argument("--ilp-chunk-cols", type=int, default=40)
    p.add_argument("--ilp-chunk-rounds", type=int, default=12)
    p.add_argument("--ilp-chunk-stride", type=int, default=0)
    p.add_argument("--skip-ilp", action="store_true")
    p.add_argument("--skip-sub-bkz", action="store_true", default=True)
    p.add_argument("--enable-sub-bkz", action="store_true", help="覆盖默认：开启 sub-BKZ（不推荐）")
    p.add_argument("--euclid-polish", action="store_true", help="强制第三类欧氏抛光")
    p.add_argument("--no-euclid-polish", action="store_true")
    p.add_argument("--sub-bkz-rows", type=int, default=40)
    p.add_argument("--sub-bkz-cols", type=int, default=40)
    p.add_argument("--sub-bkz-beta", type=int, default=28)
    p.add_argument("--sub-bkz-seeds", type=int, default=12)
    p.add_argument("--ls-restarts", type=int, default=8)
    p.add_argument("--ls-iters", type=int, default=4000)
    args = p.parse_args()

    mode = None if args.ilp_mode == "auto" else args.ilp_mode
    polish = None
    if args.euclid_polish:
        polish = True
    if args.no_euclid_polish:
        polish = False

    execute_finish(
        args.instance,
        args.incumbent,
        args.output,
        ilp_mode=mode,
        ilp_time_limit=args.ilp_time_limit,
        ilp_workers=args.ilp_workers,
        ilp_chunk_cols=args.ilp_chunk_cols,
        ilp_chunk_rounds=args.ilp_chunk_rounds,
        ilp_chunk_stride=args.ilp_chunk_stride,
        skip_ilp=args.skip_ilp,
        skip_sub_bkz=not args.enable_sub_bkz,
        sub_bkz_rows=args.sub_bkz_rows,
        sub_bkz_cols=args.sub_bkz_cols,
        sub_bkz_beta=args.sub_bkz_beta,
        sub_bkz_seeds=args.sub_bkz_seeds,
        euclid_polish=polish,
        ls_restarts=args.ls_restarts,
        ls_iters=args.ls_iters,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
