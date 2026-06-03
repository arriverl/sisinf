# =============================================================================
# run_all_classes_sequential.ps1
# 按赛题三类顺序批量调用 Python 求解器（run_class_batch.py）
#
# 执行顺序：
#   1) 第一类 1,3,6,9  → results/class1/batch_report.json
#   2) 第二类 2,4,7,10 → results/class2/batch_report.json
#   3) 第三类 5,8      → results/class3/batch_report.json
#
# 环境变量（可选）：
#   SIS_MAX_ROUNDS  每题最大轮次，默认 6；设为 0 表示无限（慎用）
#   SIS_QUICK=1     启用 --quick（缩短 iters/timeout，用于快速筛查）
#
# 线程：限制 OpenBLAS/OMP/MKL 为 1，避免 Windows 上多进程+多线程内存争用
# =============================================================================

$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
Set-Location $PSScriptRoot\..

$py = 'python'
$maxRounds = if ($env:SIS_MAX_ROUNDS) { $env:SIS_MAX_ROUNDS } else { '6' }
$quick = if ($env:SIS_QUICK -eq '1') { '--quick' } else { '' }

& $py scripts/run_class_batch.py --class 1 --max-rounds $maxRounds $quick
& $py scripts/run_class_batch.py --class 2 --max-rounds $maxRounds $quick
& $py scripts/run_class_batch.py --class 3 --max-rounds $maxRounds $quick
