# T4 smoothing gate (D-28): heading change per metre drops >= 30% for the learned planners, with time and
# contacts no worse than D-14. Seeds 5110-5119 (outside T5 and T2 seeds), T5 config, box_q 0.15, no offset.
# Two processes: smoothing off and on; planners tabpfn_split, tabpfn, lgbm_all, cv m0.4 (seed-outer).
$env:POLARS_SKIP_CPU_CHECK = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot\..
$py = "C:\Users\samco\tools\lerobot-test\.venv-x64\Scripts\python.exe"
$seeds = (5110..5119) -join " "
$base = "run_closed.py --sim crowd --threads 3 --seeds $seeds --robot-repulsion 2.0 --ignore-robot 0.2 --people 24 --goal-mode cross --bank data/crowd_cross_t2 --bank-exclude-regime erratic --switch --switch-all --switch-to erratic --switch-t 5 --frozen-ctx fill --context 2000 --box-q 0.15 --margin 0.4 --planners tabpfn_split tabpfn lgbm_all cv"
foreach ($s in @(@("", "t4sm-off"), @("--smooth", "t4sm-on"))) {
  $cmd = "& '$py' $base $($s[0]) --tag $($s[1])"
  Start-Process -FilePath powershell -ArgumentList "-NoProfile", "-Command", "$cmd *> 'results\overnight\$($s[1]).log'" -WindowStyle Hidden
}
Write-Output "started smoothing off/on"
