# Hard stop: kill the closed-loop processes whose tag matches -Pattern at a fixed time, whatever their count.
# Seed-outer checkpointing leaves complete paired seeds in results/closed/*.csv. T5: -At 08:00 -Pattern "--tag t5-".
param([string]$At = "08:00", [string]$Pattern = "--tag t5-")
$target = [datetime]::ParseExact($At, "HH:mm", $null)
if ($target -lt (Get-Date)) { $target = $target.AddDays(1) }
"cutoff armed for $($target.ToString('yyyy-MM-dd HH:mm')) on '$Pattern'" | Out-File -Append results\overnight\cutoff.log
Start-Sleep -Seconds ([int]($target - (Get-Date)).TotalSeconds)
$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -match "run_closed" -and $_.CommandLine -match [regex]::Escape($Pattern) }
foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force; "stopped $($p.ProcessId)" | Out-File -Append results\overnight\cutoff.log }
"cutoff done at $(Get-Date -Format HH:mm)" | Out-File -Append results\overnight\cutoff.log
