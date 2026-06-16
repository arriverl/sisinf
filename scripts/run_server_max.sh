#!/usr/bin/env bash
# 论文全量方案 — 服务器拉满运行（需 Linux + fpylll + G6K）
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"

echo "=== 1. 环境自检 ==="
python3 scripts/check_fpylll.py || true
python3 scripts/check_ilp_finish.py || true
python3 scripts/check_g6k.py || echo "WARN: G6K missing — run: bash scripts/install_g6k.sh"
python3 scripts/check_algorithms.py

echo ""
echo "=== 2. 冒烟（单题）==="
python3 scripts/smoke_algorithms.py

echo ""
echo "=== 3. 全量论文栈验证（十题，full-max）==="
# batch 24 轮 × 2h timeout/restart × 4h ILP ≈ 数天/题量级，按需调整
python3 scripts/run_full_validation.py \
  --full-max \
  --batch-rounds 24 \
  --ilp-time-limit 14400 \
  --output-dir results/full_max_validation

echo ""
echo "=== 4. 分三类长跑（可选并行三台机器）==="
echo "# 类一"
echo "python3 scripts/run_class_batch.py --class 1 --full-max --max-rounds 24 --seed 20260603"
echo "# 类二"
echo "python3 scripts/run_class_batch.py --class 2 --full-max --max-rounds 24 --seed 20260603"
echo "# 类三"
echo "python3 scripts/run_class_batch.py --class 3 --full-max --max-rounds 24 --seed 20260603"
echo ""
echo "报告: results/full_max_validation/full_validation_report.json"
