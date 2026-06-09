# Strix Halo phase-1 training driver (Windows-on-Strix variant).
#
# Same logic as run_phase1.sh -- chain smoke -> data prep -> training --
# but runs natively under Windows PowerShell with ROCm-on-Windows. Use
# this script when SSHing into the Strix box (cmd.exe lands; invoke this
# from PowerShell or via `powershell.exe -NoProfile -File ...` from cmd).
#
# Wall time on Strix Halo for phase 1: ~2-7 days for 2000 steps at 32 k
# context (see STRIX_INSTRUCTIONS.md cost table).

$ErrorActionPreference = 'Stop'

# Defaults (override via env or by editing this file before kickoff).
$CkptDir  = if ($env:CKPT_DIR)  { $env:CKPT_DIR }  else { 'checkpoints\longctx-v1-32k' }
$DataFile = if ($env:DATA_FILE) { $env:DATA_FILE } else { 'data\longctx_mix_v1.jsonl' }
$Steps    = if ($env:STEPS)     { $env:STEPS }     else { '2000' }
$Context  = if ($env:CONTEXT)   { $env:CONTEXT }   else { '32768' }

# cd to repo root (one level up from the script directory).
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot  = Resolve-Path (Join-Path $ScriptDir '..')
Set-Location $RepoRoot

# Activate the Windows venv. Activate.ps1 lives under .venv\Scripts\.
$VenvActivate = Join-Path $RepoRoot '.venv\Scripts\Activate.ps1'
if (-not (Test-Path $VenvActivate)) {
    Write-Error "venv not found at $VenvActivate -- create one first: py -3.11 -m venv .venv"
    exit 1
}
. $VenvActivate

# Tee everything to a timestamped transcript so a dropped SSH session
# doesn't lose progress.
$LogDir = Join-Path $RepoRoot 'logs'
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogPath = Join-Path $LogDir "train_$Stamp.log"
Start-Transcript -Path $LogPath -Append | Out-Null

try {
    # 1. Training-pipeline smoke (REQUIRED gate) -- catches recipe bugs at
    #    256 tokens before we burn 32 k-context hours. The --probe sweep
    #    confirms the largest fitting context too.
    Write-Host "[run_phase1] gate 1/3: training smoke + probe"
    & python -m run.training_smoke --probe
    if ($LASTEXITCODE -ne 0) { throw "training_smoke failed (exit $LASTEXITCODE)" }

    # 2. Idempotent data prep -- exits fast if data\longctx_mix_v1.jsonl
    #    already exists. First-time run downloads LongMemEval + InfBench
    #    and tokenises (a few GB of pull + an hour of CPU tokenisation).
    Write-Host ""
    Write-Host "[run_phase1] gate 2/3: prepare training mix"
    & python -m strix.prepare_data --out $DataFile
    if ($LASTEXITCODE -ne 0) { throw "prepare_data failed (exit $LASTEXITCODE)" }

    # 3. Main training. Saves every 200 steps + final checkpoint.
    Write-Host ""
    Write-Host "[run_phase1] gate 3/3: training ($Steps steps @ $Context ctx)"
    & python -m strix.train_phase1 `
        --steps $Steps `
        --context $Context `
        --data $DataFile `
        --out $CkptDir
    if ($LASTEXITCODE -ne 0) { throw "train_phase1 failed (exit $LASTEXITCODE)" }

    Write-Host ""
    Write-Host "[run_phase1] DONE. checkpoint at: $CkptDir\final"
    Write-Host ""
    Write-Host "To validate on the local 12 GB host (from THAT machine):"
    Write-Host "  1. python -m tools.strix_ssh copy-down $CkptDir"
    Write-Host "  2. .venv\Scripts\python.exe -m strix.verify_checkpoint --ckpt $CkptDir\final"
    Write-Host ""
    Write-Host "Expected: anchor (17k) ratio>=1.25, extension (25k) ratio>=1.20, stretch (32k) ratio>=1.10"
    Write-Host ""
    Write-Host "[run_phase1] full transcript: $LogPath"
}
finally {
    Stop-Transcript | Out-Null
}
