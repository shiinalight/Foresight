# Stop a running closed-loop process once its checkpoint file holds every seed up to -LastSeed (T5 pruning: 3019).
param([string[]]$Tags = @("t5-split-q10", "t5-split-q15"), [int]$LastSeed = 3019, [switch]$PythonOnly)
# -PythonOnly: stop only the python run, so a chained launcher moves on to its next run (baseline chains).
Set-Location $PSScriptRoot\..
$left = [System.Collections.ArrayList]@($Tags)
while ($left.Count) {
  foreach ($t in @($left)) {
    $f = Get-ChildItem results\closed -Filter "*-$t.csv" | Where-Object { $_.Name -notlike "robotfeat_*" } | Select-Object -First 1
    if ($f) {
      $seeds = (Import-Csv $f.FullName | ForEach-Object { [int]$_.seed })
      if (($seeds | Where-Object { $_ -le $LastSeed } | Sort-Object -Unique).Count -ge ($LastSeed - 2999)) {
        Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq "python.exe" -or ($_.Name -eq "powershell.exe" -and -not $PythonOnly)) -and $_.CommandLine -match "--tag $t(\s|$|')" -and $_.CommandLine -notmatch "seed_cap" } |
          ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        "$(Get-Date -Format HH:mm) capped $t at seed $LastSeed" | Out-File -Append results\overnight\seed_cap.log
        $left.Remove($t)
      }
    }
  }
  Start-Sleep -Seconds 20
}
