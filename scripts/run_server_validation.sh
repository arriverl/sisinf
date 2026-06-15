#!/usr/bin/env bash
# 服务器全量验证脚本（Linux + fpylll 推荐）
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "=== Environment ==="
python3 scripts/check_fpylll.py || true
python3 scripts/check_ilp_finish.py || true
python3 scripts/smoke_algorithms.py

echo ""
echo "=== Quick validation (10 problems, ~1-3h) ==="
python3 scripts/run_full_validation.py --quick

echo ""
echo "=== Full pipeline (production, multi-day) ==="
echo "Run manually when ready:"
echo "  python3 scripts/run_full_validation.py --batch-rounds 6 --ilp-time-limit 3600"
echo "  python3 scripts/run_all_pipeline.py --batch-rounds 6 --ilp-time-limit 3600"
echo ""
echo "Reports: results/full_validation/full_validation_report.json"
echo "         results/pipeline/all_pipeline_report.json"
