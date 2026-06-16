#!/usr/bin/env bash
# 在 Linux 服务器上安装 G6K（Becker BDGL 真筛法）
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENDOR="${ROOT}/vendor/g6k"

echo "=== Prerequisites: fpylll, gmp, mpfr ==="
python3 -c "from fpylll import BKZ" 2>/dev/null || {
  echo "Install fpylll first: conda install -c conda-forge fpylll"
  exit 1
}

if [[ ! -d "${VENDOR}/.git" ]]; then
  mkdir -p vendor
  git clone --depth 1 https://github.com/fplll/g6k.git "${VENDOR}"
fi

cd "${VENDOR}"
pip install -r requirements.txt
python3 setup.py build_ext --inplace
pip install -e .

cd "${ROOT}"
python3 scripts/check_g6k.py
echo "G6K install complete."
