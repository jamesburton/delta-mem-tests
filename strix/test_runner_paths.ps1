# Lightweight smoke test for the Windows-on-Strix runner wiring.
#
# Asserts that:
#   1. The venv exists at .venv\Scripts\Activate.ps1.
#   2. `python -m run.training_smoke --help` returns 0 -- i.e. the run
#      package is importable and the smoke CLI is wired.
#
# Does NOT run the training smoke itself (no GPU touched, no model
# download). That's the actual remote test; this is just the gate that
# says "the runner can find Python and the modules it needs".
#
# Exit codes: 0 = OK, 1 = venv missing, 2 = module discovery failed.

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot  = Resolve-Path (Join-Path $ScriptDir '..')
Set-Location $RepoRoot

Write-Host "[smoke] repo root: $RepoRoot"

$VenvActivate = Join-Path $RepoRoot '.venv\Scripts\Activate.ps1'
if (-not (Test-Path $VenvActivate)) {
    Write-Host "[smoke] FAIL: venv missing at $VenvActivate"
    Write-Host "[smoke]   fix: py -3.11 -m venv .venv"
    exit 1
}
Write-Host "[smoke] OK   : venv found at $VenvActivate"

. $VenvActivate
Write-Host "[smoke] OK   : venv activated; python = $((Get-Command python).Source)"

Write-Host "[smoke] running: python -m run.training_smoke --help"
# Use Start-Process + temp files to avoid PS 5.1's `2>&1` quirk where
# stderr lines become ErrorRecords and trip $? even at exit 0.
$stdoutFile = [System.IO.Path]::GetTempFileName()
$stderrFile = [System.IO.Path]::GetTempFileName()
try {
    $proc = Start-Process -FilePath python `
        -ArgumentList @('-m','run.training_smoke','--help') `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutFile `
        -RedirectStandardError  $stderrFile
    $Code = $proc.ExitCode
    if ($Code -ne 0) {
        Write-Host "[smoke] FAIL: training_smoke --help returned exit code $Code"
        Write-Host "[smoke]   stdout (last 30 lines):"
        Get-Content $stdoutFile -Tail 30 | ForEach-Object { Write-Host "    $_" }
        Write-Host "[smoke]   stderr (last 30 lines):"
        Get-Content $stderrFile -Tail 30 | ForEach-Object { Write-Host "    $_" }
        exit 2
    }
}
finally {
    Remove-Item $stdoutFile, $stderrFile -ErrorAction SilentlyContinue
}
Write-Host "[smoke] OK   : training_smoke CLI responds to --help"
Write-Host ""
Write-Host "[smoke] PASS -- runner wiring looks good. Actual GPU smoke is next:"
Write-Host "         python -m run.training_smoke --probe   # 5-10 min on Strix"
exit 0
