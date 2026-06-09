# SmolLM3 path — preparation notes

Companion to `LONG_CONTEXT_PLAN.md` (Option 2). These notes record what was
found while staging the SmolLM3-3B training/eval pipeline, so the next
agent (or future-me) doesn't re-derive surprises.

## Backbone

- **HF model**: <https://hf.co/HuggingFaceTB/SmolLM3-3B>
  - The instruct-tuned release lives at the bare `SmolLM3-3B` repo; there is
    **no separate `-Instruct` suffix**, unlike Qwen3's `Qwen3-4B-Instruct-2507`.
  - 3.08 B parameters, 36 transformer layers, bf16 weights ~ 5.6 GB on disk.
  - License: Apache 2.0.
- Model card: <https://hf.co/HuggingFaceTB/SmolLM3-3B>
- The base (pre-instruct) checkpoint is at `HuggingFaceTB/SmolLM3-3B-Base`.

## What ships out of the box in delta-mem

- `delta-Mem/deltamem/core/backbone_compat.py` already imports
  `SmolLM3Attention`, `apply_rotary_pos_emb` and `eager_attention_forward`
  from `transformers.models.smollm3.modeling_smollm3`, so attach paths
  treat SmolLM3 as a first-class peer of Qwen3.
- `delta-Mem/deltamem/core/delta_impl.py` extends
  `SUPPORTED_BASE_ATTENTION_TYPES` with `SmolLM3Attention` when the import
  succeeds, and `DeltaMemAttention.__init__` branches on
  `isinstance(base, SmolLM3Attention)` to swap in the correct
  `eager_attention_forward`.
- Smoke (`run/smollm3_smoke.py`) confirmed:
  - `HAS_SMOLLM3 == True`
  - All 36 `self_attn` modules are `SmolLM3Attention`
  - `attach_delta_mem` wraps all 36 with `DeltaMemAttention`
  - `freeze_non_delta_mem_params` returns **216 trainable tensors / 2.9 M
    params** (6 trainable tensors per layer x 36 layers — matches Qwen3's
    per-layer adapter shape).

## Differences vs Qwen3 that were caught during prep

1. **No `q_norm` / `k_norm`.** Qwen3 attention applies per-head QK-Norm
   between the projection and RoPE; SmolLM3 does not. This breaks OSCAR's
   rotation port — see "Open questions" below.
2. **Per-layer `use_rope` flag.** SmolLM3 has some layers with
   `self.use_rope=False` (controlled by `config.no_rope_layers`). Any
   dump/rotation/patched-forward code that unconditionally applies RoPE
   will produce wrong activations on those layers. The dump path in
   `run/oscar_calibrate_smollm3.py` already honours `self.use_rope`.
3. **Sliding-attention layers.** `config.layer_types` marks some layers
   as `sliding_attention`, setting `self.sliding_window`. delta-mem
   forwards `self.sliding_window` to the attention-interface call (already
   wired in `DeltaMemAttention.__init__`), so this should be transparent —
   but rotations on sliding-window layers haven't been validated yet.
4. **Chat template defaults to thinking mode.** SmolLM3 templates ship
   `/think` enabled; `delta-Mem/deltamem/chat_templates.py`'s
   `smollm3_enable_thinking_override` already forces it to False for
   benchmark/training prompts. Nothing else to do here.
5. **Different head_dim?** Confirm at calibration time — the
   `--head-dim 128` default in `run/oscar_calibrate_smollm3.py` matches
   SmolLM3-3B's published config (2048 hidden / 16 heads), but a variant
   could ship with 64.

## Open questions

- **OSCAR port assumes Qwen3 attention.**
  `third_party/oscar-transformers/oscar_transformers/rotation.py`'s
  `_build_patched_forward` calls `self.q_norm(...)` / `self.k_norm(...)`
  and unconditionally applies RoPE. On SmolLM3 the first forward will
  raise `AttributeError: 'SmolLM3Attention' object has no attribute
  'q_norm'`. **Prerequisite for Phase C of
  `oscar_calibrate_smollm3.py` and for any inference with
  `KV_CACHE_BACKEND=oscar` on SmolLM3.** Resolution: add a
  SmolLM3-shaped patched_forward (no q_norm/k_norm, honour
  `self.use_rope`) and dispatch on class in `apply_rotations` /
  `_PATCHED_CLASSES`. Estimate: ~50 LOC in `rotation.py` mirroring the
  existing Qwen3 path. The Phase C smoke catches the AttributeError and
  prints a clear diagnostic so the failure mode is unambiguous.

- **`freeze_non_delta_mem_params` parameter-naming.** Verified empirically
  by the smoke — returns the expected 216 tensors with names like
  `model.layers.0.self_attn.memory_q_proj`. **No issue.** (Worth keeping
  here as a "we checked" note since the original ask flagged it as an
  open question.)

- **Rotations on `use_rope=False` layers.** The covariance is computed
  from raw (un-rotated) Q/K on those layers because that's what the
  patched dumper captures. The runtime patched_forward must also skip
  RoPE on those layers, otherwise the basis won't match. Phase C smoke
  will catch this if it slips.

- **Sliding-attention layer rotation correctness.** Untested. Phase 1
  training proceeds even if the windowed layers' rotations are
  sub-optimal (training compensates), but a stand-alone OSCAR-only smoke
  on those layers would be prudent before Phase 2.

## Files added in this prep

| Path | Purpose |
|------|---------|
| `run/oscar_calibrate_smollm3.py` | 3-phase OSCAR rotation calibration for SmolLM3 (dump → compute → smoke) |
| `run/smollm3_smoke.py` | Cheap integration smoke (PASSED on dev host) |
| `run/SMOLLM3_NOTES.md` | This file |
| `run/locomo_eval.py` | `--model-override` flag added (~20 LOC patch) |
| `strix/train_smollm3_phase1.py` | Strix Halo / H100 Phase-1 training script |

## Suggested execution order (when GPU time is available)

1. Fix the OSCAR port to support SmolLM3 (open question #1 above).
2. `python -m run.oscar_calibrate_smollm3 --phase A --num-prompts 32`
   (Strix Halo or dev box; ~30-60 min on the 3060)
3. `python -m run.oscar_calibrate_smollm3 --phase B` (CPU, ~20 min)
4. `python -m run.oscar_calibrate_smollm3 --phase C` (validates the
   port + rotations)
5. `python -m strix.train_smollm3_phase1 --steps 1000 --context 32768
   --out /workspace/checkpoints/smollm3_phase1_v0/` (Strix Halo;
   ~2-3 days)
6. Copy the adapter back to the dev box and run the verification
   command at the bottom of `strix/train_smollm3_phase1.py`.
