#!/usr/bin/env bash
# Strix Halo phase-1 training driver -- single entry point.
#
# NOTE: this is the Linux/WSL variant. The current production target is
# Windows-on-Strix with ROCm-on-Windows; use strix/run_phase1.ps1 there.
# This bash script is preserved for the Linux-on-Strix case (Ubuntu
# dual-boot or a future containerised setup). See STRIX_INSTRUCTIONS.md
# "Windows-on-Strix variant" for the current default.
#
# Gates training behind training_smoke (required) and the idempotent
# data prep, then kicks off the 32 k-context fine-tune with sensible
# defaults. Wall time on Strix Halo for phase 1: ~2-7 days for 2000
# steps at 32 k context (see STRIX_INSTRUCTIONS.md cost table).
set -euo pipefail

CKPT_DIR="${CKPT_DIR:-checkpoints/longctx-v1-32k}"
DATA_FILE="${DATA_FILE:-data/longctx_mix_v1.jsonl}"
STEPS="${STEPS:-2000}"
CONTEXT="${CONTEXT:-32768}"

cd "$(dirname "$0")/.."

# 1. Training-pipeline smoke (REQUIRED gate) -- catches recipe bugs at
#    256 tokens before we burn 32 k-context hours. The --probe sweep
#    confirms the largest fitting context too.
echo "[run_phase1] gate 1/3: training smoke + probe"
python -m run.training_smoke --probe

# 2. Idempotent data prep -- exits fast if data/longctx_mix_v1.jsonl
#    already exists. First-time run downloads LongMemEval + InfBench
#    and tokenises (a few GB of pull + an hour of CPU tokenisation).
echo
echo "[run_phase1] gate 2/3: prepare training mix"
python -m strix.prepare_data --out "$DATA_FILE"

# 3. Main training. Saves every 200 steps + final checkpoint.
echo
echo "[run_phase1] gate 3/3: training (${STEPS} steps @ ${CONTEXT} ctx)"
python -m strix.train_phase1 \
    --steps "$STEPS" \
    --context "$CONTEXT" \
    --data "$DATA_FILE" \
    --out "$CKPT_DIR"

echo
echo "[run_phase1] DONE. checkpoint at: ${CKPT_DIR}/final"
echo
echo "To validate on the local 12 GB host:"
echo "  1. scp -r strixhost:${PWD}/${CKPT_DIR}/final  .planning/adapters/longctx-v1-32k/"
echo "  2. .venv\\Scripts\\python.exe -m strix.verify_checkpoint --ckpt .planning\\adapters\\longctx-v1-32k"
echo
echo "Expected: anchor (17k) ratio>=1.25, extension (25k) ratio>=1.20, stretch (32k) ratio>=1.10"
