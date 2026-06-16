"""
论文全量方案预设（``--full-max``）：不考虑算力上限，拉满所有文献路线。

文献对应
--------
- 类一：Chen–Nguyen BKZ 2.0 + Becker BDGL 筛法 (G6K) + Wang L∞ slice + Wagner + full CP-SAT
- 类二：Kannan 嵌入 + BKZ 2.0 + CVP 提升 + full CP-SAT
- 类三：Wang restricted SVP (enumerate-slice + d4f) + G6K 列表 + lex CP-SAT + 欧氏抛光
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solve_sisinf import SearchConfig


def apply_full_max_stack(cfg: "SearchConfig", sis_class: int) -> "SearchConfig":
    """在 ``apply_sis_class_defaults`` 结果上叠加全量论文参数。"""
    common = {
        "restarts": max(cfg.restarts, 160),
        "iters": max(cfg.iters, 12000),
        "max_delta": max(cfg.max_delta, 15),
        "delta": max(cfg.delta, 4),
        "parallel_workers": max(cfg.parallel_workers, 8),
        "timeout_sec": max(cfg.timeout_sec or 0, 7200.0) if cfg.timeout_sec else 7200.0,
        "cheby_weight": max(cfg.cheby_weight, 64.0),
        "cp_repair_time_limit": max(cfg.cp_repair_time_limit, 8.0),
        "block_cp_time_limit": max(cfg.block_cp_time_limit, 12.0),
        "use_g6k_sieve": True,
        "g6k_sieve_alg": "bdgl2",
        "g6k_saturation_ratio": 0.95,
        "g6k_threads": max(cfg.g6k_threads, 16),
        "g6k_max_lift_vectors": max(cfg.g6k_max_lift_vectors, 2048),
        "bkz_max_dim": 260,
        "bkz_beta": max(cfg.bkz_beta, 56),
        "bkz_max_vectors": max(cfg.bkz_max_vectors, 128),
        "bkz_combo_depth": max(cfg.bkz_combo_depth, 8),
        "bkz_combo_coeff_max": 3,
    }

    if sis_class == 1:
        return replace(
            cfg,
            **common,
            use_bkz_seeds=True,
            use_sieve_seeds=True,
            use_restricted_svp_seeds=True,
            use_kannan_seeds=False,
            use_wagner_seeds=True,
            wagner_list_cap=max(cfg.wagner_list_cap, 2400),
            wang_enum_tail_rank=max(cfg.wang_enum_tail_rank, 48),
            wang_enum_pool_size=max(cfg.wang_enum_pool_size, 4096),
            wang_enum_coeff_max=4,
            wang_enum_max_trials=50000,
            kernel_max_basis=max(cfg.kernel_max_basis, 64),
            g6k_bkz_block=56,
        )

    if sis_class == 2:
        return replace(
            cfg,
            **common,
            use_bkz_seeds=True,
            use_sieve_seeds=False,
            use_kannan_seeds=True,
            use_restricted_svp_seeds=False,
            bkz_beta=max(cfg.bkz_beta, 52),
            cvp_lift_variants=max(cfg.cvp_lift_variants, 24),
            modular_pull_variants=max(cfg.modular_pull_variants, 20),
            kannan_embedding_factor=0,
            g6k_bkz_block=52,
        )

    # class 3
    return replace(
        cfg,
        **common,
        use_bkz_seeds=True,
        use_sieve_seeds=True,
        use_restricted_svp_seeds=True,
        use_kannan_seeds=False,
        restricted_svp_samples=max(cfg.restricted_svp_samples, 2000),
        wang_enum_tail_rank=max(cfg.wang_enum_tail_rank, 56),
        wang_enum_pool_size=max(cfg.wang_enum_pool_size, 8192),
        wang_enum_coeff_max=4,
        wang_enum_max_trials=80000,
        euclid_weight=max(cfg.euclid_weight, 8.0),
        entropy_weight=max(cfg.entropy_weight, 0.65),
        g6k_bkz_block=56,
    )


def full_max_finish_kwargs(sis_class: int) -> dict:
    """``execute_finish`` 全量 ILP 参数。"""
    return {
        "ilp_time_limit": 14400.0,
        "ilp_mode": "lex" if sis_class == 3 else "full",
        "euclid_polish": sis_class == 3,
    }


PAPER_STACK_TABLE = """
| 类 | 文献路线 | 本仓库模块 | full-max 关键参数 |
|----|----------|------------|-------------------|
| 一 | Chen BKZ2.0 + Becker BDGL + Wang L∞ | lattice_bkz, lattice_g6k, lattice_sieve, lattice_restricted_svp, Wagner | β≥56, g6k bdgl2 sat=0.95, wang pool=4096 |
| 二 | Kannan CVP + BKZ2.0 | lattice_kannan, lattice_bkz, CVP lift | Kannan+β52, CVP≥24 |
| 三 | Wang restricted SVP + BDGL list | lattice_restricted_svp, lattice_g6k, lex CP-SAT | wang pool=8192, g6k+slice, ILP 4h |
"""
