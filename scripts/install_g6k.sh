#!/usr/bin/env bash
# 在 Linux 服务器上安装 G6K（Becker BDGL 真筛法）
#
# 常见错误：ModuleNotFoundError: No module named 'Cython'
# → 本脚本会先装 Cython/cysignals，再用 --no-build-isolation 编译。
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENDOR="${ROOT}/vendor/g6k"

echo "=== 1. 检查 fpylll ==="
python3 -c "from fpylll import BKZ; print('fpylll OK')" 2>/dev/null || {
  echo "请先安装 fpylll:"
  echo "  conda install -c conda-forge fpylll"
  exit 1
}

echo "=== 2. 构建依赖（Cython 必须在编译前装好）==="
# conda 与 pip 二选一或叠加均可
if command -v conda &>/dev/null; then
  conda install -y -c conda-forge cython cysignals numpy setuptools wheel 2>/dev/null || true
fi
pip install --upgrade pip setuptools wheel
pip install "Cython>=3.0" cysignals numpy

echo "=== 3. 克隆 G6K ==="
if [[ ! -d "${VENDOR}/.git" ]]; then
  mkdir -p vendor
  git clone --depth 1 https://github.com/fplll/g6k.git "${VENDOR}"
fi

cd "${VENDOR}"

echo "=== 4. 安装 G6K 可选依赖 ==="
pip install -r requirements.txt || pip install "Cython>=3.0" cysignals numpy begins

echo "=== 5. 编译 Cython 扩展（inplace）==="
python3 setup.py build_ext --inplace

echo "=== 6. 安装 Python 包（禁用 PEP517 隔离，否则找不到 Cython）==="
pip install --no-build-isolation -e .

cd "${ROOT}"
echo "=== 7. 自检 ==="
python3 scripts/sis_cli.py check g6k
echo "G6K install complete."
