# Strix Halo SSH coordination

Quick-reference for the helper at `tools/strix_ssh.py`. Use this from
**this Windows host** before/during Strix training jobs.

## First-time setup

Set these environment variables in your PowerShell profile (or per-session):

```powershell
$env:STRIX_SSH_TARGET = "jamesb@strix-halo.local"   # your actual SSH alias
$env:STRIX_REPO_DIR  = "/home/jamesb/delta-mem-tests"
```

Verify SSH key-based auth works (no password prompt):

```powershell
ssh $env:STRIX_SSH_TARGET uname -a
```

If that prompts for a password, set up an SSH key first (`ssh-keygen` then
`ssh-copy-id`). The helper uses `BatchMode=yes` so password prompts cause it
to fail immediately rather than hang.

## Before submitting a training job — always run this

```powershell
.venv\Scripts\python.exe -m tools.strix_ssh check
```

Output is one of:
- `Strix GPU is FREE. Safe to submit a training job.` → go ahead
- `GPU appears in use. Active processes: ...` → coordinate with whoever owns
  the running task (the helper lists PIDs + process names). Don't submit.
- `SSH FAILED` or `TIMEOUT` → fix connectivity first.

The "busy" threshold is **util > 5 % OR memory > 10 %** — picks up training
jobs and avoids false positives from idle CUDA contexts.

## Send the repo up

```powershell
.venv\Scripts\python.exe -m tools.strix_ssh copy-up
```

rsync's the repo to `$env:STRIX_REPO_DIR`, excluding the local `.venv/`,
`outputs/`, `report/raw/`, `checkpoints/`, and (by default) the `data/`
dir. Pass `--include-data` if you want to push the LoCoMo fixtures too
(usually unnecessary; Strix's `strix/prepare_data.py` pulls long-context
datasets fresh from HF).

The rsync is **delete-mode** so renamed/removed files on this host also
disappear on Strix — keeps things in sync. Don't run with local
work-in-progress you haven't committed.

## Run something on Strix

```powershell
# Quick one-off
.venv\Scripts\python.exe -m tools.strix_ssh run "python -m run.training_smoke"

# Kick off the phase-1 training (after Agent B's strix/run_phase1.sh lands)
.venv\Scripts\python.exe -m tools.strix_ssh run "bash strix/run_phase1.sh"
```

For long-running jobs, run inside `tmux` or `nohup` on the Strix box so
the SSH session can close:

```powershell
.venv\Scripts\python.exe -m tools.strix_ssh run "tmux new -d -s train 'bash strix/run_phase1.sh > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1'"
```

Then re-attach later:

```powershell
.venv\Scripts\python.exe -m tools.strix_ssh run "tmux attach -t train"
```

## Watch the log

```powershell
.venv\Scripts\python.exe -m tools.strix_ssh tail-log
# or specific pattern:
.venv\Scripts\python.exe -m tools.strix_ssh tail-log --log-pattern "logs/train_*.log"
```

Streams `tail -F` of the most recently modified matching log. Ctrl-C to stop.

## Pull a checkpoint back

```powershell
.venv\Scripts\python.exe -m tools.strix_ssh copy-down checkpoints/lora_v1_strix
```

rsync's `$env:STRIX_REPO_DIR/checkpoints/lora_v1_strix/` to local
`./checkpoints/lora_v1_strix/`. Then verify on this 12 GB box:

```powershell
.venv\Scripts\python.exe -m run.locomo_eval --kv-cache-backend oscar --kv-cache-bits 2 `
    --max-conversations 1 --max-questions-per-conversation 10 `
    --adapter-override checkpoints\lora_v1_strix `
    --output-json outputs\lora_v1_strix_eval.json
```

(The `--adapter-override` flag added in commit `a6f01aa` is the
cross-platform deploy hook — works the same for Kaggle-trained,
Strix-trained, or rented-H100-trained checkpoints.)

## Notes

- The helper assumes **nvidia-smi** on Strix. If the Strix Halo box uses
  ROCm (AMD iGPU), `check` will fail at the `nvidia-smi` step — the
  helper currently prints a hint to use `rocm-smi`. A `rocm-smi` parser
  is a small follow-up; until then, run `ssh strix rocm-smi --showuse`
  manually before submitting jobs.
- `BatchMode=yes` is set explicitly so the helper fails fast on auth
  issues instead of waiting for interactive input.
- `ConnectTimeout=10` is also set; network blip → fail in 10 s.
