# T2 (docs/DECISIONS.md): find the mildest crowd that makes the best baseline struggle. Baselines only.
# Grid: robot_repulsion {2, 1, 0.5} x ignore_robot {0, 0.2, 0.4} x people {16, 24, 32}; 20 seeds per cell from -Seed0.
# Cross scene (D-24): -GoalMode cross -Seed0 4100 -Bank data/crowd_cross
# Baselines: stop_go, cv margin 0.4, reactive margin 0.4, vo weight 1.0 (their best pilot settings). One process per repulsion.
param([string]$GoalMode = "anywhere", [int]$Seed0 = 4000, [string]$Bank = "data/crowd")
$env:POLARS_SKIP_CPU_CHECK = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot\..
$py = "C:\Users\samco\tools\lerobot-test\.venv-x64\Scripts\python.exe"
$seeds = ($Seed0..($Seed0 + 19)) -join " "
foreach ($rep in @("2.0", "1.0", "0.5")) {
  $cmds = @()
  foreach ($ign in @("0.0", "0.2", "0.4")) {
    foreach ($ppl in @(16, 24, 32)) {
      $base = "run_closed.py --sim crowd --switch --threads 2 --seeds $seeds --robot-repulsion $rep --ignore-robot $ign --people $ppl --goal-mode $GoalMode --bank $Bank"
      $tag = "t2$($GoalMode.Substring(0,1))-r$($rep -replace '\.','')-i$($ign -replace '\.','')-p$ppl"
      $cmds += "$base --planners stop_go --tag $tag-stopgo"
      $cmds += "$base --planners cv --margin 0.4 --tag $tag-cvm04"
      $cmds += "$base --planners reactive --margin 0.4 --tag $tag-rea04"
      $cmds += "$base --planners vo --vo-w 1.0 --tag $tag-vo10"
    }
  }
  $script = ($cmds | ForEach-Object { "& '$py' $_" }) -join "; "
  Start-Process -FilePath powershell -ArgumentList "-NoProfile", "-Command", "$script *> 'results\overnight\t2-r$($rep -replace '\.','').log'" -WindowStyle Hidden
  Write-Output "started repulsion $rep ($($cmds.Count) runs)"
}
