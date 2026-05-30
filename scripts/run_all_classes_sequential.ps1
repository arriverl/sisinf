# 依次求解三类题：先 1/3/6/9，再 2/4/7/10，再 5/8
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
