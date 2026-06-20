#!/usr/bin/env bash
# 服务器全量验证脚本（Linux + fpylll + G6K 推荐）
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "=== Environment ==="
python3 scripts/sis_cli.py check all

echo ""
echo "=== Quick validation (10 problems, ~1-3h) ==="
python3 scripts/sis_cli.py --quick

echo ""
echo "=== Standard production ==="
echo "python3 scripts/sis_cli.py --batch-rounds 6 --ilp-time-limit 3600"

echo ""
echo "=== Paper full-max (G6K + Wang max, multi-day) ==="
echo "bash scripts/run_server_max.sh"
echo "# or:"
echo "python3 scripts/sis_cli.py --full-max --batch-rounds 24 --ilp-time-limit 14400"

echo ""
echo "Reports:"
echo "  results/full_validation/full_validation_report.json"
echo "  results/full_max_validation/full_validation_report.json"
echo "Docs: docs/COURSE_PAPER.md (LaTeX: docs/COURSE_PAPER.tex)"
