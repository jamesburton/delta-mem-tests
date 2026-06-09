# Strix Halo training — quick reference

One-page entry point for someone SSH'd into the Strix Halo box (AMD
Ryzen AI Max+ 395, 96 GB unified VRAM, ROCm). For the full plan,
rationale, and ROCm setup, see
[`STRIX_INSTRUCTIONS.md`](../STRIX_INSTRUCTIONS.md) at the repo root.
The "why this checkpoint" framing lives in
[`LONG_CONTEXT_PLAN.md`](../LONG_CONTEXT_PLAN.md) (Option 1).

## First-time setup

Follow `STRIX_INSTRUCTIONS.md` "Software stack" once: ROCm 6.2+, the
`uv` venv with `torch==2.5.1` ROCm wheel, `pip install -e
third_party/oscar-transformers`, the `HSA_OVERRIDE_GFX_VERSION=11.5.1`
env, and the Triton import check. If Triton fails to JIT for delta-mem,
set `export DELTA_MEM_DISABLE_TRITON=1` (1.5x slower but correct).

## "GPU is free?" check

```bash
rocm-smi                       # show iGPU utilisation + temp + power
nvidia-smi 2>/dev/null || true # no-op on AMD; expected
fuser -v /dev/kfd 2>/dev/null  # which PIDs hold the compute device
```

If `rocm-smi` reports >5% GPU utilisation or memory in use, find and
kill the stale process before starting a new run.

## Recommended run order

```bash
# Option A: drive everything with one command
chmod +x strix/run_phase1.sh
./strix/run_phase1.sh

# Option B: run each gate by hand (for debugging)
python -m run.training_smoke --probe              # gate 1 — pipeline smoke
python -m strix.prepare_data                       # gate 2 — data mix
python -m strix.train_phase1 \                     # gate 3 — 32 k training
    --steps 2000 --context 32768 \
    --data data/longctx_mix_v1.jsonl \
    --out checkpoints/longctx-v1-32k
```

After training, copy `checkpoints/longctx-v1-32k/final/` back to the
local Windows/CUDA host (RTX 3060 12 GB) and run there:

```powershell
.venv\Scripts\python.exe -m strix.verify_checkpoint `
    --ckpt .planning\adapters\longctx-v1-32k
```

## Expected wall times

| Stage | Wall time | Notes |
|-------|-----------|-------|
| `training_smoke --probe` | 5-10 min | one-time, validates the pipeline |
| `prepare_data` (cold) | 30-90 min | ~5-10 GB HF pull + tokenise on CPU |
| `prepare_data` (warm) | < 5 s | idempotent fast-exit if JSONL exists |
| `train_phase1` 32 k / 2000 steps | 2-7 days | bandwidth-bound on Strix (256 GB/s) |
| `train_phase1` 32 k / 200 steps (smoke) | 6-18 h | for an initial loss-curve sanity |
| Checkpoint copy back | 1-5 min | a few hundred MB over rsync/scp |
| `verify_checkpoint` (local 3060) | 3-6 h | 3 scenarios, OSCAR INT2 backend |

Phase 2 (64 k context) — same recipe with `--context 65536`; budget
~85 GB peak, drop `--grad-accum` to 4 if it OOMs.

## Where to look if it dies

| Symptom | Where to look | Common fix |
|---------|---------------|-----------|
| `RuntimeError: Trying to backward through the graph a second time` from `affine_scan.py` | `train_phase1.py` GRAD_CHECKPOINT_KWARGS | Must be `{"use_reentrant": True}` — DO NOT change. See CRITICAL block in train_phase1.py |
| OOM at step 1 of training | `rocm-smi` peak during forward; checkpoint log `peak=` field | Lower `--context` to 24576 or `--grad-accum` to 4; drop other GPU users; verify UMA cap in BIOS is 96 GB |
| OOM partway through training | Loss-spike preceding OOM | bf16 numeric blow-up; check `grad_norm` in last log line — if > 10, lower `--clip` to 0.5 |
| Triton JIT compile error on import | `python -c "from deltamem.core.delta_impl import DeltaMemAttention"` | `export DELTA_MEM_DISABLE_TRITON=1` and retry; eager scan path is correctness-equivalent |
| `LongMemEval load failed` during prep | `prepare_data.py` source loop | Network/HF outage; rerun with `--max-per-source 500` first to validate, or skip the share and rebalance later |
| `verify_checkpoint` anchor (17 k) regresses | `outputs/verify_ckpt/verify_anchor_17k.json` | Training overfit to long-context; lower `--lr` to 5e-5 and increase LoCoMo share in `prepare_data.py` |
| `verify_checkpoint` extension (25 k) fails but anchor passes | extension JSON ratio < 1.20 | Curriculum didn't transfer; push more LongMemEval rows by raising `--max-per-source` and retrain |

Log paths:

- Training log JSON: `<--out>/training_log.json` (per-step loss / lr / grad_norm / peak_gb)
- Per-checkpoint adapter: `<--out>/step_N/` (every `--save-every` steps) + `<--out>/final/`
- Smoke output: stdout only — pipe to a file if you want to keep it
- Verify output: `outputs/verify_ckpt/verify_<scenario>.json`

## Checkpoint recovery

Each `step_N/` directory is a complete, loadable adapter checkpoint. To
resume training from the most recent step instead of re-initialising
from `declare-lab/delta-mem_qwen3_4b-instruct`, swap the
`_attach_adapter` line in `train_phase1.py` to point at the local
checkpoint directory. (We deliberately don't add a `--resume` CLI flag
yet — the explicit edit forces a moment of "do I actually want to keep
the optimiser-cold restart?" thought.)
