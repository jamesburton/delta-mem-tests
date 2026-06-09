# Strix Halo training -- quick reference (Windows-on-Strix)

One-page entry point for someone SSH'd into the Strix Halo box (AMD
Ryzen AI Max+ 395, 96 GB unified VRAM, **ROCm-on-Windows**). For the
full plan, rationale, and the Linux-on-Strix variant, see
[`STRIX_INSTRUCTIONS.md`](../STRIX_INSTRUCTIONS.md) at the repo root
("Windows-on-Strix variant" section). The "why this checkpoint" framing
lives in [`LONG_CONTEXT_PLAN.md`](../LONG_CONTEXT_PLAN.md) (Option 1).

## Topology assumption

- Strix runs Windows 11 with ROCm-on-Windows installed natively.
- `ssh strix` lands on `cmd.exe`. PowerShell is reached via
  `powershell.exe -NoProfile -File ...` from cmd.
- Repo lives at `C:\Users\james\delta-mem-tests` (default -- adjust if
  cloned elsewhere).

For the legacy Linux/WSL variant on Strix, see
[`run_phase1.sh`](run_phase1.sh) and the corresponding section of
`STRIX_INSTRUCTIONS.md`.

## First-time setup

Follow `STRIX_INSTRUCTIONS.md` "Windows-on-Strix variant" once:

1. Install AMD's ROCm-on-Windows package (see AMD docs link in
   `STRIX_INSTRUCTIONS.md`).
2. Install Python 3.11 (use `py -3.11 -m venv .venv`).
3. Install Git for Windows (and ideally MSYS2 for `rsync`).
4. `.\.venv\Scripts\Activate.ps1` then `pip install` per the doc.
5. `pip install -e third_party\oscar-transformers`.
6. Verify Triton import. If it fails, set
   `$env:DELTA_MEM_DISABLE_TRITON='1'` (1.5x slower but correct).

## "GPU is free?" check

```powershell
rocm-smi                                          # iGPU utilisation + memory
Get-Process python -ErrorAction SilentlyContinue  # any leftover trainers?
```

If `rocm-smi` reports >5% GPU utilisation or memory in use, find and
kill the stale process before starting a new run.

(From your **local** Windows box, `python -m tools.strix_ssh check`
does this for you over SSH.)

## Recommended run order

```powershell
# Option A: drive everything with one command
powershell.exe -NoProfile -File strix\run_phase1.ps1

# Option B: run each gate by hand (for debugging)
python -m run.training_smoke --probe              # gate 1 -- pipeline smoke
python -m strix.prepare_data                       # gate 2 -- data mix
python -m strix.train_phase1 `                     # gate 3 -- 32 k training
    --steps 2000 --context 32768 `
    --data data\longctx_mix_v1.jsonl `
    --out checkpoints\longctx-v1-32k
```

After training, copy `checkpoints\longctx-v1-32k\final\` back to the
local Windows/CUDA host (RTX 3060 12 GB) and run there:

```powershell
.venv\Scripts\python.exe -m strix.verify_checkpoint `
    --ckpt checkpoints\longctx-v1-32k\final
```

The simplest copy-back is from the **local** box:

```powershell
.venv\Scripts\python.exe -m tools.strix_ssh copy-down checkpoints\longctx-v1-32k
```

## Expected wall times

| Stage | Wall time | Notes |
|-------|-----------|-------|
| `training_smoke --probe` | 5-10 min | one-time, validates the pipeline |
| `prepare_data` (cold) | 30-90 min | ~5-10 GB HF pull + tokenise on CPU |
| `prepare_data` (warm) | < 5 s | idempotent fast-exit if JSONL exists |
| `train_phase1` 32 k / 2000 steps | 2-7 days | bandwidth-bound on Strix (256 GB/s) |
| `train_phase1` 32 k / 200 steps (smoke) | 6-18 h | initial loss-curve sanity |
| Checkpoint copy back | 1-5 min | a few hundred MB |
| `verify_checkpoint` (local 3060) | 3-6 h | 3 scenarios, OSCAR INT2 backend |

Phase 2 (64 k context) -- same recipe with `--context 65536`; budget
~85 GB peak, drop `--grad-accum` to 4 if it OOMs.

## Where to look if it dies

| Symptom | Where to look | Common fix |
|---------|---------------|-----------|
| `RuntimeError: Trying to backward through the graph a second time` from `affine_scan.py` | `train_phase1.py` GRAD_CHECKPOINT_KWARGS | Must be `{"use_reentrant": True}` -- DO NOT change |
| OOM at step 1 of training | `rocm-smi` peak during forward; checkpoint log `peak=` field | Lower `--context` to 24576 or `--grad-accum` to 4; verify UMA cap in BIOS is 96 GB |
| OOM partway through training | Loss-spike preceding OOM | bf16 numeric blow-up; check `grad_norm` -- if > 10, lower `--clip` to 0.5 |
| Triton JIT compile error on import | `python -c "from deltamem.core.delta_impl import DeltaMemAttention"` | `$env:DELTA_MEM_DISABLE_TRITON='1'` and retry; eager scan path is correctness-equivalent |
| `LongMemEval load failed` during prep | `prepare_data.py` source loop | Network/HF outage; rerun with `--max-per-source 500` first |
| `verify_checkpoint` anchor (17 k) regresses | `outputs\verify_ckpt\verify_anchor_17k.json` | Training overfit; lower `--lr` to 5e-5 and increase LoCoMo share |
| `verify_checkpoint` extension (25 k) fails but anchor passes | extension JSON ratio < 1.20 | Curriculum didn't transfer; push more LongMemEval rows |
| `python` not found on Strix | confirm Windows venv activated | `.\.venv\Scripts\Activate.ps1` (run_phase1.ps1 does this for you) |

Log paths:

- Driver transcript: `logs\train_<timestamp>.log` (auto-created by
  `run_phase1.ps1` via `Start-Transcript`)
- Training log JSON: `<--out>\training_log.json`
- Per-checkpoint adapter: `<--out>\step_N\` + `<--out>\final\`
- Verify output: `outputs\verify_ckpt\verify_<scenario>.json`

## Checkpoint recovery

Each `step_N\` directory is a complete, loadable adapter checkpoint. To
resume training from the most recent step instead of re-initialising
from `declare-lab/delta-mem_qwen3_4b-instruct`, swap the
`_attach_adapter` line in `train_phase1.py` to point at the local
checkpoint directory.

## Sanity test for the runner wiring (no GPU needed)

```powershell
powershell.exe -NoProfile -File strix\test_runner_paths.ps1
```

Asserts the venv exists and `python -m run.training_smoke --help`
returns. Does NOT touch the GPU.
