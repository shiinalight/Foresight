# T5 confirmatory closed loop (pre-registration amendments 1 and 2 in RESULTS.md). Seeds 3000-3039, T5 config:
# cell r2 i0.2 p24, cross, bank data/crowd_cross_t2 without erratic, every seed switches into erratic at 5 s,
# frozen context 2,000 rows at natural frequency, TabPFN 3.5-Fast 1 estimator. Seed-outer, planner-inner, checkpointed.
# Values from the T4 addendum: -Smooth (D-28 gate), -Offset off|row|batch (offset gate), -Latency (secondary arm).
#   tools/t5.ps1 -Offset batch -Latency 0.6 [-Smooth] [-DryRun]
param([switch]$Smooth, [string]$Offset = "off", [double]$Latency = 0.4, [switch]$DryRun)
$env:POLARS_SKIP_CPU_CHECK = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot\..
$py = "C:\Users\samco\tools\lerobot-test\.venv-x64\Scripts\python.exe"
$seeds = (3000..3039) -join " "
$sm = if ($Smooth) { " --smooth" } else { "" }
$off = switch ($Offset) { "row" { " --offset --offset-rule row --offset-eta 0.02" } "batch" { " --offset --offset-rule batch --offset-eta-b 0.05" } default { "" } }
$base = "run_closed.py --sim crowd --seeds $seeds --robot-repulsion 2.0 --ignore-robot 0.2 --people 24 --goal-mode cross --bank data/crowd_cross_t2 --bank-exclude-regime erratic --switch --switch-all --switch-to erratic --switch-t 5 --frozen-ctx fill --context 2000$sm"
$procs = [ordered]@{
  # primary: one process per caution level
  "t5-split-q10" = @("--threads 2 --planners tabpfn_split --box-q 0.10$off --tag t5-split-q10")
  "t5-split-q15" = @("--threads 2 --planners tabpfn_split --box-q 0.15$off --tag t5-split-q15")
  "t5-split-q20" = @("--threads 2 --planners tabpfn_split --box-q 0.20$off --tag t5-split-q20")
  # secondary learned arms (exploratory), same offset setting
  "t5-tabpfn-q10" = @("--threads 2 --planners tabpfn --box-q 0.10$off --tag t5-tabpfn-q10")
  "t5-tabpfn-q15" = @("--threads 2 --planners tabpfn --box-q 0.15$off --tag t5-tabpfn-q15")
  "t5-lgbm-q10" = @("--threads 2 --planners lgbm_all lgbm --box-q 0.10$off --tag t5-lgbm-q10")
  "t5-lgbm-q15" = @("--threads 2 --planners lgbm_all lgbm --box-q 0.15$off --tag t5-lgbm-q15")
  # baselines over their dials, and the open-loop oracle (two chains)
  "t5-base-a" = @("--threads 1 --planners stop_go --tag t5-stopgo",
                  "--threads 1 --planners cv --margin 0.2 --tag t5-cvm02", "--threads 1 --planners cv --margin 0.4 --tag t5-cvm04",
                  "--threads 1 --planners cv --margin 0.6 --tag t5-cvm06", "--threads 1 --planners cv --margin 0.8 --tag t5-cvm08",
                  "--threads 1 --planners vo --vo-w 0.5 --tag t5-vo05", "--threads 1 --planners vo --vo-w 1.0 --tag t5-vo10",
                  "--threads 1 --planners vo --vo-w 2.0 --tag t5-vo20")
  "t5-base-b" = @("--threads 1 --planners reactive --margin 0.2 --tag t5-rea02", "--threads 1 --planners reactive --margin 0.4 --tag t5-rea04",
                  "--threads 1 --planners reactive --margin 0.6 --tag t5-rea06", "--threads 1 --planners reactive --margin 0.8 --tag t5-rea08",
                  "--threads 1 --planners oracle --margin 0.2 --tag t5-orm02", "--threads 1 --planners oracle --margin 0.4 --tag t5-orm04",
                  "--threads 1 --planners oracle --margin 0.6 --tag t5-orm06")
  # secondary loop arms: baselines replanning every simulation step (0.2 s; 0.1 s is below the sim step), and the primary at q 0.15 with a virtual-time latency
  "t5-r02" = @("--threads 1 --planners reactive --margin 0.4 --replan 0.2 --tag t5-rea04-r02",
               "--threads 1 --planners vo --vo-w 1.0 --replan 0.2 --tag t5-vo10-r02")
  "t5-split-lat" = @("--threads 2 --planners tabpfn_split --box-q 0.15$off --latency $Latency --tag t5-split-q15-lat")
}
New-Item -ItemType Directory -Force results\overnight | Out-Null
foreach ($k in $procs.Keys) {
  $script = ($procs[$k] | ForEach-Object { "& '$py' $base $_" }) -join "; "
  if ($DryRun) { Write-Output "[$k]"; $procs[$k] | ForEach-Object { Write-Output "  $base $_" }; continue }
  Start-Process -FilePath powershell -ArgumentList "-NoProfile", "-Command", "$script *> 'results\overnight\$k.log'" -WindowStyle Hidden
  Write-Output "started $k ($($procs[$k].Count) runs)"
}
