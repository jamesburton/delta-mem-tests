# Strix Halo SSH coordination

Quick-reference for the helper at `tools/strix_ssh.py`. Use this from
**this Windows host** before/during Strix training jobs.

## Topology assumption (current production target)

- The Strix box runs **Windows 11 with ROCm-on-Windows** natively (no WSL,
  no Linux dual-boot in active use).
- `ssh strix` lands on `cmd.exe` (verified).
- Remote PowerShell is reached via `powershell.exe -NoProfile -Command "..."`
  from inside cmd.
- The repo on Strix lives under `C:\Users\james\delta-mem-tests` (default).

If you ever run a Linux-on-Strix variant (e.g. Ubuntu dual-boot or a
container), set `STRIX_SHELL=bash` and `STRIX_REPO_DIR=/home/james/delta-mem-tests`
- the helper supports both.

## First-time setup

Set these environment variables in your PowerShell profile (or per-session):

```powershell
$env:STRIX_SSH_TARGET = "strix"                              # SSH alias from ~/.ssh/config
$env:STRIX_SHELL      = "cmd"                                # default; cmd | powershell | bash
$env:STRIX_REPO_DIR   = "C:\Users\james\delta-mem-tests"     # Windows-native default
```

Examples for the other shells (rare):

```powershell
# Drive Strix from PowerShell (better for multi-line commands)
$env:STRIX_SHELL = "powershell"

# Legacy Linux-on-Strix (or WSL bounce)
$env:STRIX_SHELL    = "bash"
$env:STRIX_REPO_DIR = "/home/james/delta-mem-tests"
```

Verify SSH key-based auth works (no password prompt) -- use the
**Windows-native** OpenSSH directly to avoid the msys2 `libcrypto-3-x64.dll`
error that Git-for-Windows' ssh.exe hits:

```powershell
C:\Windows\System32\OpenSSH\ssh.exe $env:STRIX_SSH_TARGET ver
```

Expected output: a Windows version string like
`Microsoft Windows [Version 10.0.26100.xxxx]`.

The helper picks the native OpenSSH automatically when running on
Windows (`tools/strix_ssh.py::_ssh_exe`), so you don't need to think
about it after the manual smoke above.

If `ssh` prompts for a password, set up an SSH key first:

```powershell
ssh-keygen -t ed25519
# Then copy the .pub into C:\Users\james\.ssh\authorized_keys on Strix.
# (OpenSSH on Windows uses the same authorized_keys format as Linux.)
```

The helper uses `BatchMode=yes` so password prompts cause it to fail
immediately rather than hang.

## Before submitting a training job -- always run this

```powershell
.venv\Scripts\python.exe -m tools.strix_ssh check
```

Output is one of:

- `Strix GPU is FREE. Safe to submit a training job.` -> go ahead
- `GPU appears in use. Coordinate before submitting.` -> check what's
  running (`tools.strix_ssh run "tasklist | findstr python"` from cmd
  shell) before launching.
- `SSH FAILED` or `TIMEOUT` -> fix connectivity first.

The "busy" threshold is **util > 5% OR memory > 10%** -- picks up
training jobs and avoids false positives from idle GPU contexts. The
check probes `rocm-smi` first (Strix is AMD); falls back to
`nvidia-smi` if absent (unlikely on a Strix Halo box).

## Send the repo up

```powershell
.venv\Scripts\python.exe -m tools.strix_ssh copy-up
```

Sends the repo to `$env:STRIX_REPO_DIR`, excluding the local `.venv\`,
`outputs\`, `report\raw\`, `checkpoints\`, and (by default) `data\`.
Pass `--include-data` if you want to push the LoCoMo fixtures too
(usually unnecessary; `strix\prepare_data.py` pulls long-context
datasets fresh from HF).

**rsync vs scp:** the helper uses rsync when available
(delete-mode keeps local & remote in sync) and falls back to a
per-entry `scp -r` loop when rsync is not on PATH. The scp fallback is
slower and does **not** delete remote-only files. Install rsync via Git
for Windows or MSYS2 on Strix for faster syncs.

Don't run rsync-mode copy-up with local work-in-progress you haven't
committed -- rsync mode will mirror deletions.

## Run something on Strix

The default `STRIX_SHELL=cmd` passes commands through to cmd.exe
unchanged. Switch to `powershell` for anything multi-line or with
PowerShell-only syntax.

```powershell
# Quick one-off (cmd.exe)
.venv\Scripts\python.exe -m tools.strix_ssh run "python -m run.training_smoke"

# Kick off phase-1 training (PowerShell driver)
.venv\Scripts\python.exe -m tools.strix_ssh run "powershell.exe -NoProfile -File strix\run_phase1.ps1"

# Or, if STRIX_SHELL=powershell, just:
$env:STRIX_SHELL = "powershell"
.venv\Scripts\python.exe -m tools.strix_ssh run "strix\run_phase1.ps1"
```

### Long-running jobs (tmux replacement on Windows)

Windows doesn't have tmux. The simplest production-grade pattern is
**`Start-Process` with output redirection** -- fire-and-forget, survives
the SSH session closing, log file is tailable:

```powershell
# From your local box (STRIX_SHELL=powershell is convenient here):
$env:STRIX_SHELL = "powershell"
.venv\Scripts\python.exe -m tools.strix_ssh run @'
$stamp = Get-Date -Format yyyyMMdd_HHmmss;
$log = "logs\train_$stamp.log";
Start-Process -FilePath powershell.exe `
  -ArgumentList "-NoProfile","-File","strix\run_phase1.ps1" `
  -WindowStyle Hidden `
  -RedirectStandardOutput $log `
  -RedirectStandardError  "logs\train_$stamp.err"
Write-Host "started; log: $log"
'@
```

That spawns a detached PowerShell process on Strix, writes stdout to
`logs\train_<stamp>.log`, and returns immediately. The PID is
discoverable later via `Get-Process powershell` or by reading the log
header.

Alternative: Windows Task Scheduler (`schtasks /create`) -- more
durable across reboots but heavier to set up. Use it if you need
scheduled recurring runs.

Note: `strix\run_phase1.ps1` already runs `Start-Transcript`, so you
get a self-contained transcript inside `logs\train_<stamp>.log` even
without the outer redirection. The outer `Start-Process` is what makes
it survive SSH disconnect.

## Watch the log

```powershell
.venv\Scripts\python.exe -m tools.strix_ssh tail-log
# or a specific pattern:
.venv\Scripts\python.exe -m tools.strix_ssh tail-log --log-pattern "logs\train_*.log"
```

On Windows targets, `tail-log` runs PowerShell's
`Get-Content -Path <newest-match> -Wait -Tail 30` -- the native Windows
equivalent of `tail -F`. Ctrl-C to stop.

On `STRIX_SHELL=bash` targets it uses the classic
`ls -t ... | head -1 | xargs tail -F`.

## Pull a checkpoint back

```powershell
.venv\Scripts\python.exe -m tools.strix_ssh copy-down checkpoints\longctx-v1-32k
```

Pulls `$env:STRIX_REPO_DIR\checkpoints\longctx-v1-32k\` to local
`.\checkpoints\longctx-v1-32k\`. Then verify on this 12 GB box:

```powershell
.venv\Scripts\python.exe -m strix.verify_checkpoint `
    --ckpt checkpoints\longctx-v1-32k\final
```

Or run `locomo_eval` directly with the adapter override (the
`--adapter-override` flag is the cross-platform deploy hook -- works
identically for Kaggle-, Strix-, or rented-H100-trained checkpoints):

```powershell
.venv\Scripts\python.exe -m run.locomo_eval --kv-cache-backend oscar --kv-cache-bits 2 `
    --max-conversations 1 --max-questions-per-conversation 10 `
    --adapter-override checkpoints\longctx-v1-32k\final `
    --output-json outputs\longctx_v1_eval.json
```

## Notes

- `BatchMode=yes` is set explicitly so the helper fails fast on auth
  issues instead of waiting for interactive input.
- `ConnectTimeout=10` is also set; network blip -> fail in 10 s.
- The helper hard-codes `C:\Windows\System32\OpenSSH\ssh.exe` on Windows
  hosts to avoid msys2-ssh's libcrypto error. The fallback is
  `shutil.which("ssh")` -- override only if you have a specific reason.
- For the legacy Linux/WSL Strix variant, set `STRIX_SHELL=bash` and
  use `strix/run_phase1.sh` instead of the PowerShell driver.
