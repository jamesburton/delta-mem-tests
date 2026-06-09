# Kaggle delta-mem adapter training — T4 (16 GB) + T4×2

This is the "rent free GPU hours, train a checkpoint" path that sits
between the 12 GB local box (only validates the pipeline) and Strix Halo
(handles 32-64 k context). Kaggle T4 16 GB can usefully fine-tune the
adapter at **4-8 k context** with proper checkpointing, and T4×2 with
DeepSpeed ZeRO-3 can push to **~16 k context** by sharding the frozen
backbone weights.

It does **not** fully replace Strix Halo for the 32-64 k targets, but it
runs ~10-30× faster than this local box for any context that fits (no
shared-memory paging penalty), so it's the right place to do hyperparameter
scouts and interim long-context checkpoints.

---

## TL;DR — which Kaggle GPU

| GPU | Memory | Best for | Why |
|-----|--------|----------|-----|
| **T4 single ★** | 16 GB | 4-8 k context, validation runs, hyperparam sweeps | Best supported, simple pipeline, 30 h/week quota |
| **T4 × 2** | 16 GB × 2 = 32 GB | 8-16 k context with DeepSpeed ZeRO-3 | Doubles fit; harder setup |
| P100 single | 16 GB | Avoid for this workload | Older arch, no bf16 TensorCores; some Triton kernels mis-tune |
| L4 (when available) | 24 GB | 16-24 k context single-card | Newer Ada arch, much better bf16 throughput. Spotty availability on free tier. |

**Recommended first run**: **T4 single** to validate everything (the
training_smoke gate, then a 500-step fine-tune at 8 k context). Once that's
clean and you have an output you want to push further, switch to **T4 × 2**
for 12-16 k context.

---

## VRAM budget on T4 16 GB

Numbers measured locally at 12 GB extrapolated to 16 GB. Reentrant
gradient checkpointing assumed (required per our findings).

| Context | Activations | KV cache | Weights | Adapter+opt | Total | Fits on T4? |
|---------|-------------|----------|---------|-------------|-------|-------------|
| 2 k | 2.5 GB | 0.3 GB | 7.5 GB | 0.06 GB | ~10.4 GB | ✓ comfortable |
| 4 k | 4 GB | 0.6 GB | 7.5 GB | 0.06 GB | ~12.2 GB | ✓ comfortable |
| 8 k | 7 GB | 1.2 GB | 7.5 GB | 0.06 GB | ~15.8 GB | ✓ tight (margin ~200 MB) |
| 16 k | 13 GB | 2.4 GB | 7.5 GB | 0.06 GB | ~23 GB | ✗ OOM on single T4 |
| 16 k with T4×2 + ZeRO-3 | 13 GB | 2.4 GB | 3.75 GB/card | 0.06 GB | ~12 GB / card | ✓ fits |

Numbers are bf16 throughout. fp32 master weights or fp32 Adam states would
push the budget up by ~1 GB; use 8-bit Adam (`bitsandbytes`) if it gets tight.

---

## Setup — first run (no cached wheels yet)

### 1. Create the notebook

1. Sign in to kaggle.com
2. Code → New Notebook
3. Settings (right sidebar):
   - **Accelerator**: `GPU T4 x1` (recommended start) or `GPU T4 x2`
   - **Internet**: ON (required for `pip install` and the HF model pull)
   - **Persistence**: Variables and files (keeps `/kaggle/working/` between
     cells within a session)

### 2. Paste the notebook content

Use [`notebook_t4.ipynb`](notebook_t4.ipynb) in this repo as the source.
Or copy the cells from this doc's [Notebook contents](#notebook-contents)
section below.

### 3. Run the smoke cell first

The first run of the smoke cell (cell #3 in the notebook) takes ~5-7 min:
~2 min for pip installs + repo clone + submodule init, ~2 min for HF model
pull (Qwen3-4B + adapter ≈ 8 GB), ~30 s for the 6 smoke checks. If the
smoke passes, the rest of the notebook is safe to run.

### 4. Run the training cell

At 8 k context for 200 steps on a T4 single, expect ~30-50 min wall time
(no Windows paging penalty — should be ~10-20 s/step instead of our local
212 s/step). The training cell saves the checkpoint to
`/kaggle/working/checkpoints/lora_v0_kaggle/` and downloads cleanly via
the notebook output panel.

---

## Wheel caching across sessions

Kaggle's `/kaggle/working/` does NOT persist across notebook sessions —
each new session starts fresh. To avoid re-downloading wheels every time:

### Strategy 1 — Kaggle Dataset for wheels (recommended)

After your first successful run, save the pip cache as a Kaggle Dataset:

```python
# Run this once after a successful install in the notebook
import shutil, os
os.makedirs("/kaggle/working/wheel_cache", exist_ok=True)
shutil.copytree("/root/.cache/pip", "/kaggle/working/wheel_cache/pip",
                 dirs_exist_ok=True)
shutil.copytree("/root/.triton/cache", "/kaggle/working/wheel_cache/triton",
                 dirs_exist_ok=True)
```

Then in the notebook UI: **File → Save Version → "Save & Run All"**, then
**Save → "Output" tab → New Dataset**. Name it e.g.
`<your-username>/delta-mem-wheel-cache`.

On subsequent sessions, attach that dataset via **Data → Add data** and
the notebook's first cell restores from it:

```python
import shutil
from pathlib import Path
cache_src = Path("/kaggle/input/delta-mem-wheel-cache")
if cache_src.exists():
    shutil.copytree(cache_src / "pip", "/root/.cache/pip", dirs_exist_ok=True)
    shutil.copytree(cache_src / "triton", "/root/.triton/cache", dirs_exist_ok=True)
    print("wheel + triton kernel cache restored")
```

Net effect: pip install drops from ~2 min to ~30 s, and Triton kernel
JIT recompiles are skipped (saves ~30-60 s on first training step).

### Strategy 2 — pre-built wheels in a Dataset

If a specific wheel needs to be built from source on every session (rare
for our stack but possible for some Triton versions), build it once
locally and upload the `.whl` file to a Kaggle Dataset. The first cell
then `pip install`s from that dataset path.

---

## T4×2 with DeepSpeed ZeRO-3 (the 16 k context configuration)

For training at 16 k context on dual T4, use DeepSpeed Stage 3 to shard
the frozen backbone weights across both cards. The adapter (4.9 M params)
is too small to benefit from sharding but the backbone (4 B params) takes
the win.

`deepspeed_zero3_config.json` is shipped alongside the notebook; the T4×2
cell launches with:

```bash
deepspeed --num_gpus=2 run/local_lora_train.py \
    --steps 200 --context 16384 --out /kaggle/working/checkpoints/lora_v0_kaggle_t4x2 \
    --deepspeed_config kaggle/deepspeed_zero3_config.json
```

(Note: `--deepspeed_config` is not currently wired into `local_lora_train.py`.
The notebook's T4×2 cell either monkey-patches it in or falls back to
`accelerate launch`. See the notebook for the chosen approach.)

---

## Notebook contents

The actual notebook is [`notebook_t4.ipynb`](notebook_t4.ipynb). Cells
in summary:

| # | Cell | Purpose |
|---|------|---------|
| 1 | Setup + cache restore | nvidia-smi, env vars, restore wheel + Triton cache from dataset (if attached) |
| 2 | Install deps | `pip install` with the cached wheels |
| 3 | Clone repo | `git clone` + `git submodule update --init --recursive` |
| 4 | Smoke test | `python -m run.training_smoke` — gate; if this fails, stop |
| 5 | Train | `python -m run.local_lora_train --steps 200 --context 8192 ...` |
| 6 | Eval round-trip | `python -m run.locomo_eval --adapter-override ...` (3 q smoke) |
| 7 | Save cache + checkpoint | snapshot caches for the next session, zip the checkpoint for download |

---

## Common gotchas

| Issue | Fix |
|-------|-----|
| `git clone` slow on Kaggle | Use `--depth 1` and `--shallow-submodules` (already in notebook) |
| HF download hangs | `huggingface_hub.snapshot_download(... etag_timeout=60)`; or pre-stage as a Kaggle Dataset |
| Triton kernel JIT fails on T4 | T4 is compute capability 7.5 — most Triton kernels work but watch for fp8/bf16 ops; fall back via `DELTA_MEM_DISABLE_TRITON=1` |
| `optimum-quanto` errors at import | Optional dep; safe to skip via `pip install ... --no-deps` for transformers if needed (we don't use quanto on Kaggle) |
| Notebook idle timeout | Kaggle kills sessions after ~12 h idle and ~9 h continuous compute on free tier; long training runs need to be `nohup`-ed or checkpointed frequently |
| GPU memory not freed between cells | `torch.cuda.empty_cache()` + restart kernel; or use `subprocess.run(...)` to spawn training as a child process so its memory is reclaimed cleanly |

---

## After training: pull the checkpoint back

The notebook saves to `/kaggle/working/checkpoints/lora_vN_kaggle/`. To
get it back to this host:

1. In the Kaggle notebook UI, **File → Save Version → Save & Run All**.
2. After the version saves, go to the **Output** tab on the notebook page.
3. Download `checkpoints/lora_vN_kaggle/` as a zip.
4. Extract to `E:\Development\delta-mem-tests\checkpoints\lora_vN_kaggle\`.
5. Evaluate:

```powershell
. .\env\vsenv.ps1
$env:KV_CACHE_BACKEND='oscar'; $env:KV_CACHE_BITS='2'
$env:OSCAR_K_ROTATION_PATH='data\oscar\rotations\instruct_gpqa\k_rotation_qqt_r_h_pbr.pt'
$env:OSCAR_V_ROTATION_PATH='data\oscar\rotations\instruct_gpqa\v_rotation_sst_r_h_pbr.pt'
.venv\Scripts\python.exe -m run.locomo_eval --kv-cache-backend oscar --kv-cache-bits 2 \
    --max-conversations 1 --max-questions-per-conversation 10 \
    --adapter-override checkpoints\lora_vN_kaggle \
    --output-json outputs\lora_vN_kaggle_eval.json
```

The `--adapter-override` flag is the cross-platform deploy hook —
identical contract regardless of where the checkpoint was trained.

---

## When to graduate from Kaggle to Strix Halo

| Reason | Move to Strix Halo |
|--------|-------------------|
| Need 32-64 k context training | YES — Kaggle T4×2 caps at ~16 k |
| Iterating on data mix daily | YES — Kaggle's 30 h/week quota limits iteration |
| Need multi-day continuous training | YES — Kaggle kills at 9 h |
| One-off experiment at 4-8 k | NO — Kaggle T4 is faster + free |

Kaggle is also useful as a **second opinion** — train the same recipe on
Kaggle T4 and Strix Halo independently; if they produce equivalent
checkpoints (eval score within noise on this host), the recipe is
reproducible and platform-portable.
