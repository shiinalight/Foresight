# T4 offset reproduction + T5-config timing smoke. Seeds 5100-5103 (outside T5 3000-3039 and T2 4100-4119), T5 config:
# bank data/crowd_cross_t2 without erratic, cell r2 i0.2 p24, cross, every seed switches into erratic at 5 s,
# frozen context 2,000 rows at natural frequency, box_q 0.15. Three processes:
#   A: tabpfn and tabpfn_split, offset scoring only (eta 0): miss rate of the raw box, and ms per plan
#   B: tabpfn and tabpfn_split with the offset (eta 0.02, alpha 0.2 on the raw 5-95% box, d applied to the box_q box)
#   C: lgbm_all with eta 0 and with the offset
$env:POLARS_SKIP_CPU_CHECK = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot\..
$py = "C:\Users\samco\tools\lerobot-test\.venv-x64\Scripts\python.exe"
$base = "run_closed.py --sim crowd --threads 3 --seeds 5100 5101 5102 5103 --robot-repulsion 2.0 --ignore-robot 0.2 --people 24 --goal-mode cross --bank data/crowd_cross_t2 --bank-exclude-regime erratic --switch --switch-all --switch-to erratic --switch-t 5 --frozen-ctx fill --context 2000 --box-q 0.15 --offset"
$chains = @(
  @("--planners tabpfn tabpfn_split --offset-eta 0 --tag t4off-eta0"),
  @("--planners tabpfn tabpfn_split --tag t4off-eta002"),
  @("--planners lgbm_all --offset-eta 0 --tag t4off-lgbm-eta0", "--planners lgbm_all --tag t4off-lgbm-eta002")
)
$i = 0
foreach ($c in $chains) {
  $i++
  $script = ($c | ForEach-Object { "& '$py' $base $_" }) -join "; "
  Start-Process -FilePath powershell -ArgumentList "-NoProfile", "-Command", "$script *> 'results\overnight\t4off-$i.log'" -WindowStyle Hidden
}
Write-Output "started $i chains"
