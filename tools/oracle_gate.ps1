# Open-loop oracle gate (D-21, amendment 1) in the chosen T2 cell: repulsion 2, ignore 0.2, people 24, cross,
# seeds 4100-4119, same switch design and bank as T2. Oracle margins 0.2/0.4/0.6 plus the baseline dial sweeps
# not already run in T2 (cv and reactive margins 0.2/0.6/0.8, vo weights 0.5/2). Four chains in parallel.
$env:POLARS_SKIP_CPU_CHECK = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot\..
$py = "C:\Users\samco\tools\lerobot-test\.venv-x64\Scripts\python.exe"
$seeds = (4100..4119) -join " "
$base = "run_closed.py --sim crowd --switch --threads 2 --seeds $seeds --robot-repulsion 2.0 --ignore-robot 0.2 --people 24 --goal-mode cross --bank data/crowd_cross"
$tag = "t2c-r20-i02-p24"
$chains = @(
  @("--planners oracle --margin 0.2 --tag $tag-orm02", "--planners cv --margin 0.2 --tag $tag-cvm02", "--planners vo --vo-w 0.5 --tag $tag-vo05"),
  @("--planners oracle --margin 0.4 --tag $tag-orm04", "--planners cv --margin 0.6 --tag $tag-cvm06", "--planners vo --vo-w 2.0 --tag $tag-vo20"),
  @("--planners oracle --margin 0.6 --tag $tag-orm06", "--planners cv --margin 0.8 --tag $tag-cvm08"),
  @("--planners reactive --margin 0.2 --tag $tag-rea02", "--planners reactive --margin 0.6 --tag $tag-rea06", "--planners reactive --margin 0.8 --tag $tag-rea08")
)
$i = 0
foreach ($c in $chains) {
  $i++
  $script = ($c | ForEach-Object { "& '$py' $base $_" }) -join "; "
  Start-Process -FilePath powershell -ArgumentList "-NoProfile", "-Command", "$script *> 'results\overnight\oracle-gate-$i.log'" -WindowStyle Hidden
}
Write-Output "started $i chains"
