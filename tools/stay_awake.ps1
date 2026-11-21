# Keep Windows from idle-sleeping while overnight run_closed.py processes are alive.
# Uses SetThreadExecutionState (the same request a video player makes); changes no settings.
# A closed lid or a manual Sleep still sleeps: keep the lid open and the charger in.
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
Start-Sleep -Seconds 60
while ($true) {
  $alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run_closed' }
  if (-not $alive) { break }
  [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
  Start-Sleep -Seconds 30
}
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
