# Stop the overnight learned-planner processes (q05 / q10 / q15) at a fixed time, whatever their count.
# Seed-outer checkpointing leaves complete paired seeds in results/closed/*-qXX.csv.
param([string]$At = "07:00")
$target = [datetime]::ParseExact($At, "HH:mm", $null)
if ($target -lt (Get-Date)) { $target = $target.AddDays(1) }
Start-Sleep -Seconds ([int]($target - (Get-Date)).TotalSeconds)
$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -match "run_closed" -and $_.CommandLine -match "--tag q(05|10|15)" }
foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force; "stopped $($p.ProcessId)" | Out-File -Append results\overnight\cutoff.log }
"cutoff done at $(Get-Date -Format HH:mm)" | Out-File -Append results\overnight\cutoff.log
