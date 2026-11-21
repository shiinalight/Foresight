# Mirror the plan documents from darshak/core to main, without touching the working checkout.
#
#   powershell -File tools/sync_docs.ps1
#
# Takes the COMMITTED versions on darshak/core (commit the docs first), so half-finished edits never
# reach main. Uses a temporary git worktree on origin/main, commits only the listed paths, and pushes
# by fast-forward (never force). Afterwards merge origin/main into darshak/core before the next code
# promotion; the files are identical on both sides, so that merge is clean.
$ErrorActionPreference = "Stop"
$paths = @("PLAN.md", "docs", "SCENARIOS.md", "INTERFACE.md", "TEAM-PLAN.md", "tools/sync_docs.ps1", "media/fig_data_*.png")
$repo = (git rev-parse --show-toplevel).Trim()
Set-Location $repo
git fetch -q origin
$src = (git rev-parse --short darshak/core).Trim()
$tmp = Join-Path $env:TEMP "foresight-main-sync"
if (Test-Path $tmp) { git worktree remove --force $tmp 2>$null; if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp } }
git worktree add -q --detach $tmp origin/main
try {
    Push-Location $tmp
    git checkout darshak/core -- $paths
    git add -- $paths
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) { Write-Output "main already has the current plan documents"; return }
    git commit -q -m "Sync plan documents from darshak/core $src"
    git push -q origin HEAD:main
    Write-Output "main updated with the plan documents from darshak/core $src"
} finally {
    Pop-Location
    git worktree remove --force $tmp
}
