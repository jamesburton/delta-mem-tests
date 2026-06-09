# Strix Halo training plan — long-context delta-mem adapter, deploy on this host

This document is the handoff for moving delta-mem adapter **training** to a
Strix Halo box (AMD Ryzen AI Max+ 395, 96 GB allocatable VRAM via Radeon
8060S iGPU on ROCm) while keeping **inference and evaluation** on the
current Windows/CUDA host (RTX 3060, 12 GB).

## Why move training off this host

48 h of optimisation on this 12 GB box closed the headline reproduction
(see `report/tier1-summary.md` Appendix D, commit `ab1257c`):

- OSCAR INT2 + GPQA-cal rotation + delta-mem ratio **1.33×** at 17 k context
  (paper claims 1.20×).
- INT2 4-per-byte packing landed (submodule `cd17cb3`) → headline 7.1×
  memory saving vs bf16.
- `OSCAR_DISABLE_DEQUANT_SHADOW=1` env var (submodule `3835184`) +
  `--eval-batch-size 1` flag (commit `ab1257c`) let us stretch to ~25 k.
- **But** at 25 k the **published adapter** (`declare-lab/delta-mem_qwen3_4b-instruct`)
  collapses (ratio 0.60×, delta arm -62 %). Likely because it was trained
  on ~17 k-class contexts and the additional KV history is OOD for the
  delta-state computation.

The next real win is **retraining/fine-tuning the delta-mem adapter on
longer-context data** so it stays useful past 20 k. That training is
infeasible here (12 GB) but well within Strix Halo's 96 GB. Trained
adapter weights are framework-agnostic safetensors, so cross-deployment
to this CUDA host is mechanical.

## Goal in one sentence

Produce a delta-mem adapter (`delta-mem_qwen3_4b-instruct-longctx-vN`) that
extends the **quality ceiling** from ~17 k to ≥32 k, then validate on this
host that v5 (17 k) reproduces ratio ≥ 1.30 and v6c (25 k) crosses ratio
≥ 1.20 (currently 0.60).

---

## Hardware preconditions on the training box

| Spec | Required | Notes |
|------|----------|-------|
| GPU | Strix Halo (Ryzen AI Max+ 395 / 8060S iGPU, RDNA 3.5) | 96 GB unified-memory cap via BIOS UMA / Variable Graphics Memory setting |
| RAM | ≥ 128 GB unified | Strix Halo ships 64 / 128 GB variants — the 128 GB SKU is the one this plan targets |
| OS | Ubuntu 24.04 LTS (or 22.04 LTS) | ROCm 6.2+ on the recent kernels |
| Disk | ≥ 500 GB free NVMe | model + datasets + checkpoints |
| Network | Reliable for `huggingface_hub` snapshot pulls | First pull is ~10 GB |

ROCm consumer iGPU support is still less polished than discrete; pick a
kernel/driver combo from AMD's tested matrix rather than rolling-release.

---

## Software stack

```bash
# 1. ROCm (host)
#    Follow https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
#    Target ROCm 6.2 or newer; iGPU UMA exposed via amdgpu.gpu_recovery=1 and
#    set GFX_VERSION=11.5.1 (RDNA 3.5) in your shell rc:
echo 'export HSA_OVERRIDE_GFX_VERSION=11.5.1' >> ~/.bashrc
echo 'export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc

# 2. Python env (uv preferred for parity with this host)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11
source .venv/bin/activate

# 3. PyTorch (ROCm wheels). Pick the version matching your installed ROCm.
#    Example for ROCm 6.2:
uv pip install --index-url https://download.pytorch.org/whl/rocm6.2 \
    "torch==2.5.1" "torchvision==0.20.1"

# 4. Transformers + accelerate (framework-agnostic)
uv pip install "transformers==5.9.0" "accelerate>=0.34" "peft>=0.14" \
    "safetensors" "huggingface_hub" "datasets" "trl"

# 5. Triton (delta-mem scan kernels). PyTorch's ROCm wheel ships HIP-Triton.
#    Verify import; if it fails, fall back to pure-PyTorch path (see below).
python -c "import triton; print(triton.__version__)"

# 6. FlashAttention-2 ROCm port (optional; speeds up training prefill)
#    Built from source per AMD's FA2 ROCm fork; skip if it errors on first try.

# 7. Clone the repos (use the SAME submodule pins as this host)
git clone --recursive https://github.com/jamesburton/delta-mem-tests
cd delta-mem-tests
git submodule update --init --recursive
uv pip install -e third_party/oscar-transformers   # adapter loader needs this
```

### Triton kernel fallback

`delta-Mem/deltamem/runtime/session.py` and `delta_impl.py` import Triton
scan kernels (installed for this host in commit `2e9cced`). On AMD-Triton
they may fail to JIT-compile. If `python -c "from deltamem.core.delta_impl
import DeltaMemAttention"` raises a Triton compilation error, set:

```bash
export DELTA_MEM_DISABLE_TRITON=1   # forces the eager-PyTorch scan path
```

(Sub-1.5× slower per training step but correctness-equivalent. Long-context
training is bandwidth-bound on Strix Halo regardless, so the cost is mostly
hidden.)

### Sanity check before training

```bash
# Verify CUDA-API compatibility (torch.cuda.* works on ROCm)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); \
           x=torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16); \
           print((x @ x.T).sum().item())"

# Smoke-load Qwen3-4B-Instruct-2507 + the adapter
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B-Instruct-2507',
                                          dtype=torch.bfloat16).to('cuda')
print('model on', m.device, 'mem:', torch.cuda.memory_allocated() / 2**30, 'GB')
"
```

### Run the training pipeline smoke (REQUIRED before real training)

After cloning, run `run/training_smoke.py` on the Strix Halo box to
validate the training pipeline end-to-end at a tiny context length.
The same script was used on this CUDA host to de-risk the recipe — it
caught real bugs that would have wasted Strix Halo hours.

```bash
python -m run.training_smoke           # 256-token smoke (6 checks)
python -m run.training_smoke --probe   # also sweeps max-fit context
```

Findings from the local smoke run that apply to Strix Halo too:

1. **delta-mem adapter is 4.9 M trainable params (252 tensors)**, not the
   ~50 M I had estimated. Backbone freezes to 4.02 B params correctly.
2. **`gradient_checkpointing_enable(..., use_reentrant=True)` is REQUIRED.**
   The default `use_reentrant=False` fails after the first iteration
   because delta-mem's Triton scan kernel (`affine_scan.py`) saves
   tensors in a way that's incompatible with non-reentrant checkpointing.
   No checkpoint at all *also* fails on the second iteration for the
   same reason. The legacy reentrant path is the only working option.
3. **Local 12 GB ceiling under reentrant checkpointing: ~2048 tokens** —
   useful for pipeline smoke runs and hyperparameter scouts, far below
   the 32-64 k training target that requires Strix Halo.
4. **Peak memory at 256 tokens (no checkpointing): 10.05 GB** — leaves
   1.95 GB headroom. At 2048 with checkpointing: 13.2 GB (Windows
   paging into shared memory; would be a hard OOM on Linux without
   swap).
5. **Adapter save/load round-trips bit-identically** across 324 tensors
   (252 trainable + 72 buffers). Cross-platform deployment of the
   trained adapter back to this CUDA host should be mechanical.

### CRITICAL training-config requirement

When wiring `transformers.Trainer` (or any custom training loop) on
Strix Halo, set:

```python
TrainingArguments(
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},  # <- non-negotiable
    ...
)
```

Or equivalently:

```python
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": True}
)
```

Without `use_reentrant=True` the training crashes on the second forward
pass with `RuntimeError: Trying to backward through the graph a second
time (or directly access saved tensors after they have already been
freed)` originating in
`deltamem/kernels/affine_scan.py:378`. This is a real blocker, not a
warning to ignore.

---

## Training plan

### Data: long-context conversational memory

The published adapter was trained on the OSAR/LoCoMo-style mixture that
peaks around 17 k tokens. To push the quality ceiling we need data at the
target context length **with the same dialogue+QA structure**.

Two sources, both already permissively licensed:

1. **LongMemEval** (`xiaowu0162/long-mem-eval`) — multi-session
   conversations at 30-50 k tokens, with retrieval-style QA. Direct match
   for our LoCoMo distribution but longer.
2. **InfBench** (`xinrongzhang2022/infbench`) — covers 64 k - 128 k
   conversation/document QA. Strong stress-test data.

For curriculum, mix at:
- 50 % LoCoMo originals (anchors prior performance) at 8-18 k
- 30 % LongMemEval at 20-32 k
- 20 % InfBench-mem at 32-64 k

Curriculum lets the optimiser see the easier 17 k anchor early and avoid
catastrophic forgetting of the published adapter's strengths.

### Model setup

- Backbone: `Qwen/Qwen3-4B-Instruct-2507` (matches this host; frozen)
- Init: load published `declare-lab/delta-mem_qwen3_4b-instruct` adapter
  weights and **continue training**, don't restart from random. This
  protects the 17 k-class quality already there.

### Hyperparameters (starting point — tune)

```python
TRAIN = dict(
    max_seq_len = 32768,            # phase 1 target; phase 2 -> 65536
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate = 1e-4,           # 10x lower than initial training; we're fine-tuning
    lr_scheduler = "cosine",
    warmup_steps = 200,
    num_train_epochs = 2,
    weight_decay = 0.01,
    bf16 = True,
    gradient_checkpointing = True,  # required at 32 k+ on 96 GB
    gradient_checkpointing_kwargs = {"use_reentrant": True},  # REQUIRED — see smoke findings
    optim = "adamw_torch_fused",
    eval_steps = 500,
    save_steps = 1000,
    logging_steps = 25,
    # delta-mem-specific
    freeze_backbone = True,
    train_adapter_only = True,
    state_update_mode = "online",   # match inference-time setting
)
```

### VRAM budget (estimate at 32 k context, batch=1, grad-checkpoint)

| Component | Size |
|-----------|------|
| Qwen3-4B bf16 weights (frozen, no grad) | ~7.5 GB |
| Adapter weights (trainable) | ~0.2 GB |
| Adam optimiser states (m, v for adapter) | ~0.6 GB |
| Activations with grad checkpointing | ~30-40 GB |
| KV cache (bf16, batch=1 × 32 k × 36 layers × 8 heads × 128 × 2) | ~9.7 GB |
| Buffers / scratch | ~5-8 GB |
| **Total** | **~55-65 GB** of 96 GB |

At 64 k context, push the same numbers up; expect ~85 GB peak. Should fit
but close — drop to `gradient_accumulation_steps=4` if it OOMs.

### Phase 1 (target 32 k) command sketch

```bash
# inside the cloned delta-mem-tests on Strix Halo
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# The published delta-mem repo doesn't ship a training script. Use the
# delta-Mem submodule's training entry-point (see `delta-Mem/deltamem/training`).
# Pseudocode skeleton:
python -m deltamem.training.train_adapter \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --adapter-init declare-lab/delta-mem_qwen3_4b-instruct \
    --dataset-mix data/longctx_mix_v1.jsonl \
    --max-seq-len 32768 \
    --output-dir checkpoints/longctx-v1-32k \
    --bf16 --gradient-checkpointing \
    --per-device-batch 1 --grad-accum 8 \
    --lr 1e-4 --epochs 2
```

If the in-repo training script needs adapting (likely — the published
adapter's training code may not match the submodule exactly), the simplest
fallback is a `trl` `SFTTrainer` wrapper around the delta-mem
`DeltaMemAttention` module with only the adapter parameters
`requires_grad=True`. About 80-100 LOC.

### Phase 2 (push to 64 k)

If phase 1 hits ratio ≥ 1.20 on a 32 k held-out set, repeat the same
recipe with `max_seq_len=65536` and the heavier InfBench tail of the data
mix. Phase 1 first so you have a working checkpoint to fall back to.

---

## Adapter export → cross-deploy here

### What to ship back

A delta-mem adapter checkpoint is a directory like
`declare-lab/delta-mem_qwen3_4b-instruct/`. Required files:

```
adapter_model.safetensors      # the trained weights
adapter_config.json            # rank, alpha, target modules, etc.
config.json                    # parent model reference
tokenizer.json                 # (optional but easier to keep alongside)
```

Total size: a few hundred MB. Zip and copy via scp / rsync / cloud bucket.

### Verify portability on Strix Halo before shipping

```bash
# Round-trip the adapter through a CPU load + bf16 save to strip any
# accidental device tensors or ROCm-specific dtypes from the safetensors.
python - <<'PY'
import torch
from safetensors.torch import load_file, save_file
sd = load_file("checkpoints/longctx-v1-32k/adapter_model.safetensors")
sd_clean = {k: v.detach().to("cpu").to(torch.bfloat16).contiguous() for k, v in sd.items()}
save_file(sd_clean, "checkpoints/longctx-v1-32k/adapter_model.safetensors")
print(f"clean: {len(sd_clean)} tensors, sizes ok")
PY
```

### Drop onto this host

```powershell
# On this CUDA host (E:\Development\delta-mem-tests)
# Place the adapter dir under .planning/adapters/ (gitignored data) — e.g.:
#   .planning/adapters/delta-mem_longctx-v1-32k/

# Re-point the eval to the local adapter dir
$env:PYTHONIOENCODING='utf-8'
$env:KV_CACHE_BACKEND='oscar'
$env:KV_CACHE_BITS='2'
$env:OSCAR_K_ROTATION_PATH='data\oscar\rotations\instruct_gpqa\k_rotation_qqt_r_h_pbr.pt'
$env:OSCAR_V_ROTATION_PATH='data\oscar\rotations\instruct_gpqa\v_rotation_sst_r_h_pbr.pt'

# locomo_eval reads adapter from EVAL_CONFIG["adapter"]; either edit
# run/locomo_eval.py to point at the local dir or use the existing
# --adapter-override flag if added (TODO if not present).
.venv\Scripts\python.exe -m run.locomo_eval --kv-cache-backend oscar --kv-cache-bits 2 \
    --max-conversations 1 --max-questions-per-conversation 10 \
    --output-json outputs\longctx_v1_conv0_smoke.json
```

### Success criteria for the round-trip

1. **Anchor preserved**: conv-26 / 10 q at 17 k context — `delta ≥ 0.34`,
   `ratio ≥ 1.25` (i.e. doesn't lose what the published adapter already
   does well).
2. **Extension achieved**: conv-41 / 10 q at 25 k context — `delta ≥
   0.30`, `ratio ≥ 1.20` (the published adapter scored 0.139 / 0.60 here).
3. **Hard target**: synthetic conv-26 x2 at ~32 k — `delta ≥ 0.25`,
   `ratio ≥ 1.10` (any positive ratio is a win at this length).

If (1) fails, the training regressed the anchor; lower LR, increase the
anchor share in the data mix.

If (1) passes and (2) regresses to ratio < 1.0, the curriculum didn't
transfer; push more LongMemEval data and re-train.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| ROCm Triton kernels fail to JIT for delta-mem scan | Medium | `DELTA_MEM_DISABLE_TRITON=1` falls back to eager-PyTorch path; 1.5× slower but correctness-equivalent |
| FlashAttention-2 ROCm port unstable | Medium | Fall back to `attn_implementation="sdpa"` (default); training ~2× slower at 32 k+ |
| Numerical drift between bf16 on ROCm and CUDA shifts adapter quality | Low | bf16 has a single IEEE definition; rounding mode is consistent. Use the round-trip clean-save step above to strip any device-bound state |
| Long-context QA datasets don't match LoCoMo dialogue structure → adapter overfits to wrong distribution | Medium | The 50 / 30 / 20 mix anchors on LoCoMo; tune the ratio after eval (1) |
| 96 GB cap actually applies to discrete vs unified differently | Low | Verify with `rocm-smi` early; if iGPU exposes less, scale `max_seq_len` down accordingly |
| Strix Halo memory bandwidth (256 GB/s) makes training prohibitively slow | Known | Bandwidth-bound but feasible; estimate ~12-36 h per phase 1 epoch on a 10 k-sample mix. Plan for 2-7 days per phase, not hours |

---

## Cost / time comparison (to keep honest)

| Option | Wall time per training run | $$ | Iteration cost |
|--------|---------------------------|-----|----------------|
| Strix Halo (owned) | 2-7 days | $2-3 k one-time | $0 per run after hardware |
| Rented cloud H100 80 GB | 6-12 h | $2-4 / h → ~$50-200 / run | Per-run fees, faster turnaround |
| Rented cloud A100 80 GB | 18-30 h | $1-2 / h → ~$30-100 / run | Per-run fees, mid speed |
| H200 / B200 (cloud) | 4-8 h | $5-10 / h → ~$50-200 / run | Per-run fees, fastest |

Break-even at ~15-30 runs depending on cloud SKU. If you expect to iterate
adapter rank, data mix, curriculum, and context-length sweep across
several configurations, Strix Halo wins on TCO.

**Recommended first step before committing the hardware**: rent an H100
for one day, run a 10 k-sample LongMemEval-only sanity training to confirm
the data + recipe produces a ratio improvement at 25 k. If yes, then
either continue on cloud or buy the box.

---

## What this host's role becomes

After Strix Halo is set up:

| Concern | Host (this box) | Strix Halo |
|---------|-----------------|------------|
| Training delta-mem | ❌ (insufficient VRAM) | ✅ |
| Adapter evaluation on LoCoMo | ✅ | possible but not required |
| OSCAR rotation calibration (3-phase) | ✅ already done | not needed unless distribution changes |
| KV-cache backend development | ✅ | not relevant |
| Inference / demo | ✅ | ✅ (but bandwidth-limited tokens/s) |

This host stays the **eval gold-standard**: every adapter checkpoint
shipped back gets validated here against the LoCoMo conv-0/10 q anchor
before promotion.

---

## Open questions to settle before training starts

1. **Adapter format compatibility.** The published `declare-lab` adapter
   uses a specific delta-Mem config (rank, target modules, head fan-out).
   Confirm the training entry-point reads it back identically before
   spending hours on the first epoch. Run a 100-step dry-run, save, load
   on this host, eval on conv-0/3 q (10 min). Iterate config until the
   round-trip matches the published-adapter baseline within noise.

2. **State-update-mode parity.** Delta-mem has multiple `state_update_mode`
   options. Inference here uses the default. Training MUST match or the
   trained corrections will look correct in training and wrong at
   inference. Lock this on day one.

3. **Triton kernel parity.** If `DELTA_MEM_DISABLE_TRITON=1` is needed on
   ROCm, also run training with the eager path on CUDA once for a
   100-step sanity comparison — confirms the gradient is identical
   between eager and Triton (it should be) and rules out a silent bug
   showing up only post-training.

4. **Tokenizer pad-side.** delta-mem training is sensitive to whether the
   tokenizer pads on left or right; the published adapter assumes
   right-pad with `pad_token_id=eos_token_id`. Set the same way on Strix
   Halo before any data is loaded.

---

## Windows-on-Strix variant (current production target)

The recipe above assumes Ubuntu 24.04 with ROCm on Linux. The **current
production target is Windows 11 with ROCm-on-Windows installed natively**
-- no WSL, no Linux dual-boot. This section documents the deltas; the
data, hyperparameter, and adapter sections above are unchanged.

### Topology

- OS: Windows 11 Pro on the Strix Halo box.
- GPU stack: AMD ROCm-on-Windows (see AMD's "Install ROCm on Windows":
  https://rocm.docs.amd.com/projects/install-on-windows/en/latest/).
- SSH: OpenSSH server on Windows. `ssh strix` lands on `cmd.exe`.
- Python: native Windows Python 3.11 in a per-repo venv at
  `.venv\Scripts\` (vs `.venv/bin/` on POSIX).
- Repo: `C:\Users\james\delta-mem-tests` (default; adjust to taste).
- Driver script: `strix\run_phase1.ps1` (PowerShell), not the bash variant.
- Long-running launcher: `Start-Process -WindowStyle Hidden
  -RedirectStandardOutput` (the tmux equivalent on Windows).

### Prerequisites (one-time, on the Strix box)

1. **ROCm on Windows** -- install per AMD's docs (link above). After
   install, confirm `rocm-smi` runs from PowerShell and reports the
   Radeon 8060S iGPU.
2. **Python 3.11** -- install from python.org (NOT the Microsoft Store
   variant; the Store variant has restricted filesystem semantics that
   break HuggingFace caching).
3. **Git for Windows** -- provides `git` and a bash shell if you want
   one. (We use Windows-native OpenSSH instead of Git's bundled ssh; see
   `tools/STRIX_SSH_USAGE.md`.)
4. **(Optional) rsync** -- via MSYS2 or Git for Windows; without it the
   `tools/strix_ssh.py copy-up` / `copy-down` commands fall back to
   slower `scp -r`.

### Software stack (PowerShell)

```powershell
# 1. ROCm -- already installed via the AMD installer above. Verify:
rocm-smi --showuse

# 2. Clone and venv
cd C:\Users\james
git clone --recursive https://github.com/jamesburton/delta-mem-tests
cd delta-mem-tests
git submodule update --init --recursive

py -3.11 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 3. PyTorch (ROCm-on-Windows wheels). Pick the version matching your
#    installed ROCm. Example for ROCm 6.2:
pip install --index-url https://download.pytorch.org/whl/rocm6.2 `
    "torch==2.5.1" "torchvision==0.20.1"

# 4. Transformers + accelerate (framework-agnostic)
pip install "transformers==5.9.0" "accelerate>=0.34" "peft>=0.14" `
    "safetensors" "huggingface_hub" "datasets" "trl"

# 5. Triton -- PyTorch's ROCm-on-Windows wheel ships HIP-Triton. Verify:
python -c "import triton; print(triton.__version__)"
# If this fails, set $env:DELTA_MEM_DISABLE_TRITON='1' before training
# (1.5x slower, correctness-equivalent).

# 6. OSCAR loader (required for adapter inference / verify)
pip install -e third_party\oscar-transformers

# 7. Smoke
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
powershell.exe -NoProfile -File strix\test_runner_paths.ps1  # no-GPU wiring smoke
python -m run.training_smoke --probe                          # GPU smoke
```

The `HSA_OVERRIDE_GFX_VERSION=11.5.1` and
`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` env vars from the
Linux recipe are **not currently needed on ROCm-Windows** -- the
Windows driver advertises the RDNA 3.5 capabilities directly. Keep
them in mind as a fallback if a future ROCm-Windows release misbehaves
(set with `$env:HSA_OVERRIDE_GFX_VERSION='11.5.1'`).

### Running training

```powershell
# Driver (on Strix, in cmd.exe after ssh):
powershell.exe -NoProfile -File strix\run_phase1.ps1

# Or from the local box (drops into cmd.exe on Strix):
.\.venv\Scripts\python.exe -m tools.strix_ssh run `
    "powershell.exe -NoProfile -File strix\run_phase1.ps1"
```

The PowerShell driver mirrors `run_phase1.sh` exactly: smoke -> data
prep -> train. It uses `Start-Transcript` to capture all output to
`logs\train_<timestamp>.log` automatically.

### Long-running jobs (tmux replacement on Windows)

Windows has no tmux. Use `Start-Process` with output redirection from
the local box -- fire-and-forget, survives the SSH session closing:

```powershell
$env:STRIX_SHELL = "powershell"
.venv\Scripts\python.exe -m tools.strix_ssh run @'
$stamp = Get-Date -Format yyyyMMdd_HHmmss;
Start-Process -FilePath powershell.exe `
  -ArgumentList "-NoProfile","-File","strix\run_phase1.ps1" `
  -WindowStyle Hidden `
  -RedirectStandardOutput "logs\train_$stamp.log" `
  -RedirectStandardError  "logs\train_$stamp.err"
Write-Host "started; log: logs\train_$stamp.log"
'@
```

Tail-follow from the local box:

```powershell
.venv\Scripts\python.exe -m tools.strix_ssh tail-log
```

### Cross-deploy back to this CUDA host

Unchanged from the Linux section -- adapter safetensors are
framework-agnostic. Pull with:

```powershell
# from this Windows/CUDA dev box
.venv\Scripts\python.exe -m tools.strix_ssh copy-down checkpoints\longctx-v1-32k
.venv\Scripts\python.exe -m strix.verify_checkpoint --ckpt checkpoints\longctx-v1-32k\final
```

### Why we kept the Linux/bash variant around

`strix\run_phase1.sh` and the `STRIX_SHELL=bash` mode of
`tools/strix_ssh.py` are retained for:

- Future Ubuntu dual-boot on Strix (if ROCm-on-Windows regresses).
- Containerised training (Docker on Linux is much smoother than on
  Windows for ROCm).
- Anyone reusing this repo on a rented Linux GPU host.

The Windows variant supersedes Linux for the active production target
but the bash recipe still works end-to-end.

---

## Continuation pointer

When Strix Halo training is set up and a checkpoint is ready, drop a note
in `.planning/.continue-here.md` pointing at:

- The Strix Halo training session ID / commit on the trainer's branch
- The local adapter dir path on this host
- The conv-0/10 q anchor eval result with the new adapter
- Next experiment: conv-41 / 10 q at 25 k

The handoff back to this CUDA host is then mechanical — replace the
`EVAL_CONFIG["adapter"]` path and re-run the existing v5 / v6c eval
commands.
