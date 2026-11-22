# T5 compute pruning (T4/T5 addendum part 5): restart the two secondary learned arms kept, on seeds 3000-3019,
# batch offset, smoothing OFF, latency 0, replan 0.4, 3 threads. The primary (split q0.15) and split q0.10 keep running.
$env:POLARS_SKIP_CPU_CHECK = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot\..
$py = "C:\Users\samco\tools\lerobot-test\.venv-x64\Scripts\python.exe"
$s20 = "--seeds " + ((3000..3019) -join " ")
$base = "run_closed.py --sim crowd --robot-repulsion 2.0 --ignore-robot 0.2 --people 24 --goal-mode cross --bank data/crowd_cross_t2 --bank-exclude-regime erratic --switch --switch-all --switch-to erratic --switch-t 5 --frozen-ctx fill --context 2000 --offset --offset-rule batch --offset-eta-b 0.05 --box-q 0.15 --threads 3 $s20"
foreach ($a in @(@("tabpfn", "t5-tabpfn-q15"), @("lgbm_all", "t5-lgbmall-q15"))) {
  Start-Process -FilePath powershell -ArgumentList "-NoProfile", "-Command", "& '$py' $base --planners $($a[0]) --tag $($a[1]) *> 'results\overnight\$($a[1]).log'" -WindowStyle Hidden
  Write-Output "started $($a[1])"
}
