# T4 offset test (amendment 2, T4 addendum part 1) + T5-config timing smoke. Seeds 5100-5105 (outside T5 3000-3039
# and T2 4100-4119), T5 config: bank data/crowd_cross_t2 without erratic, cell r2 i0.2 p24, cross, every seed switches
# into erratic at 5 s, frozen context 2,000 rows at natural frequency, box_q 0.15. Planners tabpfn_split, tabpfn, lgbm_all.
# Three processes: offset OFF (scored only, eta 0), rule (i) per row (eta 0.02), rule (ii) per batch (eta_b 0.05 per 2 s).
$env:POLARS_SKIP_CPU_CHECK = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot\..
$py = "C:\Users\samco\tools\lerobot-test\.venv-x64\Scripts\python.exe"
$seeds = (5100..5105) -join " "
$base = "run_closed.py --sim crowd --threads 3 --seeds $seeds --robot-repulsion 2.0 --ignore-robot 0.2 --people 24 --goal-mode cross --bank data/crowd_cross_t2 --bank-exclude-regime erratic --switch --switch-all --switch-to erratic --switch-t 5 --frozen-ctx fill --context 2000 --box-q 0.15 --offset --planners tabpfn_split tabpfn lgbm_all"
foreach ($s in @(@("--offset-eta 0", "t4off-off"), @("--offset-rule row --offset-eta 0.02", "t4off-row"), @("--offset-rule batch --offset-eta-b 0.05", "t4off-batch"))) {
  $cmd = "& '$py' $base $($s[0]) --tag $($s[1])"
  Start-Process -FilePath powershell -ArgumentList "-NoProfile", "-Command", "$cmd *> 'results\overnight\$($s[1]).log'" -WindowStyle Hidden
}
Write-Output "started offset off / row / batch"
