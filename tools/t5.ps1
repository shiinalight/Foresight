# T5 confirmatory closed loop (pre-registration amendments 1 and 2, T4/T5 addendum in RESULTS.md).
# T5 config: cell r2 i0.2 p24, cross, bank data/crowd_cross_t2 without erratic, every seed switches into erratic at
# 5 s, frozen context 2,000 rows at natural frequency, TabPFN 3.5-Fast 1 estimator. Seed-outer, planner-inner,
# checkpointed. Primary and every baseline: seeds 3000-3039. Secondary arms (exploratory): seeds 3000-3019.
# Every arm using choose/choose_vo (baselines, oracle, fast-replan) runs with smoothing OFF and ON (tag suffix -sm);
# the T4 smoothing gate picks the set that enters the analysis. Learned arms: smoothing per -Smooth, plus the primary
# at q 0.15 with smoothing ON as an extra process.
#   tools/t5.ps1 -Offset batch [-Smooth] [-DryRun]
param([switch]$Smooth, [string]$Offset = "off", [double]$Latency = 0.4, [switch]$DryRun)
$env:POLARS_SKIP_CPU_CHECK = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot\..
$py = "C:\Users\samco\tools\lerobot-test\.venv-x64\Scripts\python.exe"
$s40 = "--seeds " + ((3000..3039) -join " ")
$s20 = "--seeds " + ((3000..3019) -join " ")
$sm = if ($Smooth) { " --smooth" } else { "" }
$off = switch ($Offset) { "row" { " --offset --offset-rule row --offset-eta 0.02" } "batch" { " --offset --offset-rule batch --offset-eta-b 0.05" } default { "" } }
$base = "run_closed.py --sim crowd --robot-repulsion 2.0 --ignore-robot 0.2 --people 24 --goal-mode cross --bank data/crowd_cross_t2 --bank-exclude-regime erratic --switch --switch-all --switch-to erratic --switch-t 5 --frozen-ctx fill --context 2000"
$procs = [ordered]@{
  # primary: one process per caution level (40 seeds), plus q 0.15 with smoothing ON
  "t5-split-q10" = @("$s40 --threads 2 --planners tabpfn_split --box-q 0.10$off$sm --tag t5-split-q10")
  "t5-split-q15" = @("$s40 --threads 2 --planners tabpfn_split --box-q 0.15$off$sm --tag t5-split-q15")
  "t5-split-q20" = @("$s40 --threads 2 --planners tabpfn_split --box-q 0.20$off$sm --tag t5-split-q20")
  "t5-split-q15-sm" = @("$s40 --threads 2 --planners tabpfn_split --box-q 0.15$off --smooth --tag t5-split-q15-sm")
  # secondary learned arms (exploratory, 20 seeds), same offset setting
  "t5-tabpfn" = @("$s20 --threads 2 --planners tabpfn --box-q 0.10$off$sm --tag t5-tabpfn-q10",
                  "$s20 --threads 2 --planners tabpfn --box-q 0.15$off$sm --tag t5-tabpfn-q15")
  "t5-lgbm-q10" = @("$s20 --threads 2 --planners lgbm_all lgbm --box-q 0.10$off$sm --tag t5-lgbm-q10")
  "t5-lgbm-q15" = @("$s20 --threads 2 --planners lgbm_all lgbm --box-q 0.15$off$sm --tag t5-lgbm-q15")
  "t5-split-lat" = @("$s20 --threads 2 --planners tabpfn_split --box-q 0.15$off$sm --latency $Latency --tag t5-split-q15-lat")
}
# baselines (40 seeds) and fast-replan arms (20 seeds), smoothing off and on
foreach ($v in @(@("", ""), @(" --smooth", "-sm"))) {
  $f, $x = $v
  $procs["t5-base-a$x"] = @("$s40 --threads 1 --planners stop_go --tag t5-stopgo$x",
    "$s40 --threads 1 --planners cv --margin 0.2$f --tag t5-cvm02$x", "$s40 --threads 1 --planners cv --margin 0.4$f --tag t5-cvm04$x",
    "$s40 --threads 1 --planners cv --margin 0.6$f --tag t5-cvm06$x", "$s40 --threads 1 --planners cv --margin 0.8$f --tag t5-cvm08$x",
    "$s40 --threads 1 --planners vo --vo-w 0.5$f --tag t5-vo05$x", "$s40 --threads 1 --planners vo --vo-w 1.0$f --tag t5-vo10$x",
    "$s40 --threads 1 --planners vo --vo-w 2.0$f --tag t5-vo20$x")
  $procs["t5-base-b$x"] = @("$s40 --threads 1 --planners reactive --margin 0.2$f --tag t5-rea02$x",
    "$s40 --threads 1 --planners reactive --margin 0.4$f --tag t5-rea04$x", "$s40 --threads 1 --planners reactive --margin 0.6$f --tag t5-rea06$x",
    "$s40 --threads 1 --planners reactive --margin 0.8$f --tag t5-rea08$x",
    "$s40 --threads 1 --planners oracle --margin 0.2$f --tag t5-orm02$x", "$s40 --threads 1 --planners oracle --margin 0.4$f --tag t5-orm04$x",
    "$s40 --threads 1 --planners oracle --margin 0.6$f --tag t5-orm06$x")
  $procs["t5-r02$x"] = @("$s20 --threads 1 --planners reactive --margin 0.4 --replan 0.2$f --tag t5-rea04-r02$x",
    "$s20 --threads 1 --planners vo --vo-w 1.0 --replan 0.2$f --tag t5-vo10-r02$x")
}
if ($Smooth) { $procs.Remove("t5-split-q15-sm") }       # already the primary setting
New-Item -ItemType Directory -Force results\overnight | Out-Null
foreach ($k in $procs.Keys) {
  $script = ($procs[$k] | ForEach-Object { "& '$py' $base $_" }) -join "; "
  if ($DryRun) { Write-Output "[$k]"; $procs[$k] | ForEach-Object { Write-Output "  $_" }; continue }
  Start-Process -FilePath powershell -ArgumentList "-NoProfile", "-Command", "$script *> 'results\overnight\$k.log'" -WindowStyle Hidden
  Write-Output "started $k ($($procs[$k].Count) runs)"
}
