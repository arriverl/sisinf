# 在 Windows 上安装「真 fpylll」（推荐 conda-forge 预编译包，避免 pip 缺 gmp.h）
# 用法（在 Anaconda/Miniconda 环境中）：
#   conda install -y -c conda-forge fpylll
#   python scripts/check_fpylll.py

Write-Host "=== fpylll 安装检查 ===" -ForegroundColor Cyan
if (Get-Command conda -ErrorAction SilentlyContinue) {
    Write-Host "检测到 conda，尝试安装 fpylll ..."
    conda install -y -c conda-forge fpylll
    if ($LASTEXITCODE -eq 0) {
        python scripts/check_fpylll.py
        exit $LASTEXITCODE
    }
}
Write-Host "未找到 conda 或安装失败。" -ForegroundColor Yellow
Write-Host "备选 1: 安装 Miniconda 后执行: conda install -c conda-forge fpylll"
Write-Host "备选 2: WSL 内: pip install fpylll，并设置环境变量 SIS_USE_WSL_BKZ=1"
Write-Host "备选 3: 继续使用启发式格种子（无 fpylll 时自动回退）"
exit 1
