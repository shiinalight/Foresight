# T1 (docs/DECISIONS.md section 3): can the plaza separate planners at all? No TabPFN.
# Frozen lgbm at box_q 0.10 / 0.15, constant velocity and reactive at margins 0.8 / 1.0, seeds 1000-1099, switch on.
$env:POLARS_SKIP_CPU_CHECK = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot\..
$py = "C:\Users\samco\tools\lerobot-test\.venv-x64\Scripts\python.exe"
$seeds = (1000..1099) -join " "
$common = "run_closed.py --sim crowd --switch --threads 3 --seeds $seeds"
$jobs = @(
  @{tag = "t1-lgbm-q10"; cmds = @("$common --planners lgbm --box-q 0.10 --tag t1lq10")},
  @{tag = "t1-lgbm-q15"; cmds = @("$common --planners lgbm --box-q 0.15 --tag t1lq15")},
  @{tag = "t1-margins"; cmds = @("$common --planners cv --margin 0.8 --tag t1cvm08", "$common --planners cv --margin 1.0 --tag t1cvm10",
                                  "$common --planners reactive --margin 0.8 --tag t1rea08", "$common --planners reactive --margin 1.0 --tag t1rea10")}
)
foreach ($j in $jobs) {
  $script = ($j.cmds | ForEach-Object { "& '$py' $_" }) -join "; "
  Start-Process -FilePath powershell -ArgumentList "-NoProfile", "-Command", "$script *> 'results\overnight\$($j.tag).log'" -WindowStyle Hidden
  Write-Output "started $($j.tag)"
}
