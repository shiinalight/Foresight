# Overnight closed-loop run (docs/GATES.md section 2). Crowd simulator, seeds 1000-1099, odd seeds switch
# regime mid-episode. Learned / band planners: box_q 0.05 / 0.10 / 0.15 at margin 0.2, one process each.
# Constant velocity: margin 0.2 / 0.4 / 0.6 (its own caution dial), cheap, run first in its own process.
# Each seed's row is checkpointed to results/closed/<run_id>.csv as soon as it finishes.
$ErrorActionPreference = "Continue"
$env:TABPFN_TOKEN = [Environment]::GetEnvironmentVariable("TABPFN_TOKEN", "User")
$env:POLARS_SKIP_CPU_CHECK = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot\..
$py = "C:\Users\samco\tools\lerobot-test\.venv-x64\Scripts\python.exe"
$seeds = (1000..1099) -join " "
$common = "run_closed.py --sim crowd --switch --threads 4 --seeds $seeds"
$learned = "cv_band lgbm lgbm_all lgbm_live tabpfn tabpfn_live"
$jobs = @(
  @{tag = "cv-margins"; args = "$common --planners cv --margin 0.2 --tag cvm02"; then = @("$common --planners cv --margin 0.4 --tag cvm04", "$common --planners cv --margin 0.6 --tag cvm06")},
  @{tag = "q05"; args = "$common --planners $learned --box-q 0.05 --tag q05"; then = @()},
  @{tag = "q10"; args = "$common --planners $learned --box-q 0.10 --tag q10"; then = @()},
  @{tag = "q15"; args = "$common --planners $learned --box-q 0.15 --tag q15"; then = @()}
)
foreach ($j in $jobs) {
  $cmds = @($j.args) + $j.then
  $script = ($cmds | ForEach-Object { "& '$py' $_" }) -join "; "
  $log = "results\overnight\$($j.tag).log"
  Start-Process -FilePath powershell -ArgumentList "-NoProfile", "-Command", "$script *> '$log'" -WindowStyle Hidden
  Write-Output "started $($j.tag) -> $log"
}
Start-Process -FilePath powershell -ArgumentList "-NoProfile", "-File", "$PSScriptRoot\stay_awake.ps1" -WindowStyle Hidden
Write-Output "started stay_awake (holds while run_closed processes are alive)"
