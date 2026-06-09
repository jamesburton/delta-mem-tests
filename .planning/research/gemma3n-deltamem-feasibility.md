# Gemma 3n E4B + delta-mem: Feasibility Research

**Status:** Research only — no code changes.
**Author:** Claude (Opus 4.7) | **Date:** 2026-06-09
**Scope:** Option 3 in `LONG_CONTEXT_PLAN.md` — adding Gemma 3n E4B as a third
supported backbone alongside Qwen3 and SmolLM3.

**Companion doc:** `.planning/research/gemma3n-implementation-plan.md` —
"if we ever decide to do it, here's exactly how" (5-phase execution plan
with file paths, LOC, risk register, and a revised recommendation that
takes the April 2026 Gemma 4 release into account).

---

## TL;DR

**Recommendation: DEFER INDEFINITELY (revisit only if multimodal becomes a
hard requirement).** Gemma 3n E4B is a fundamentally different beast from the
Qwen3/SmolLM3 attention class our delta-mem fork is wired for: it carries
AltUp, Laurel blocks, MatFormer-style per-layer activation sparsity, KV-shared
layers, a 4:1 sliding:full attention ratio at a 512-token window, PLE, and
vision/audio towers — and on top of that the bf16 weights (~14.6 GB) don't
fit on a 12 GB card without INT4 quant of the backbone, which itself
invalidates the OSCAR rotation calibration. Qwen3-4B + retrained adapter
(Option 1) or SmolLM3-3B + new adapter (Option 2) deliver the 32 k goal with
a fraction of the engineering and infra risk.

---

## 1. Gemma 3n E4B architecture summary

All numbers below come from the official HF `config.json` for the released
model (the unsloth-republished, content-identical mirror is used because the
google org repo is gated):
[`unsloth/gemma-3n-E4B-it/config.json`](https://huggingface.co/unsloth/gemma-3n-E4B-it/raw/main/config.json).

### Parameter count
- **Total params (loaded weights):** 7.85 B (per HF model-card metadata,
  [google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it)).
- **"Effective" params at inference:** Google markets it as ~4 B effective
  because PLE embeddings can stay in CPU RAM and the MatFormer-style nested
  FFN can be sliced down ([Google AI for Developers, *Gemma 3n model
  overview*](https://ai.google.dev/gemma/docs/gemma-3n)). The 4 B figure
  refers to *accelerator memory footprint with PLE offloaded*, **not** the
  number of FLOPs or the number of params that must hit the GPU during a
  training step.
- The community-quoted "5.44 B" comes from an older HF blog
  ([rishiraj — Understanding Gemma 3n](https://huggingface.co/blog/rishiraj/matformer-in-gemma-3n))
  and predates the final release.

### MatFormer / nested architecture
- The shipped E4B does **not** in fact vary `intermediate_size` per layer in
  its public config — every entry in the `intermediate_size` list is
  `16384`, repeated for all 35 layers. The MatFormer training trick is
  *latent* in the weights (FFN sub-matrices were jointly optimised), and
  the user is expected to extract an E2B sub-model post-hoc rather than the
  E4B forward pass dynamically choosing widths ([HF blog](https://huggingface.co/blog/rishiraj/matformer-in-gemma-3n)).
- However, the config does include **per-layer activation sparsity** that
  *does* vary by layer index: layers 0-9 have
  `activation_sparsity_pattern = 0.95`; layers 10-34 have `0.0`. That is a
  real per-layer behaviour difference (gated FFN top-k=5% activations on
  early layers).
- **AltUp** (`altup_num_inputs: 4`, `altup_active_idx: 0`,
  `altup_correct_scale: true`) and **Laurel** (`laurel_rank: 64`) are
  additional Gemma 3n-specific block types that wrap each transformer layer
  ([modeling_gemma3n.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3n/modeling_gemma3n.py)).
  Delta-mem has zero awareness of these.

### Per-Layer Embedding (PLE) cache
- `hidden_size_per_layer_input: 256`, `vocab_size_per_layer_input: 262144`,
  35 layers ⇒ **PLE table = 35 × 256 × 262144 ≈ 2.35 B params**.
- At bf16 that is **~4.7 GB**. Google's claim is that this table can live on
  CPU RAM and be streamed per-token via PCIe — but in our use-case (training
  a delta-mem adapter on a single GPU, then evaluating long-context QA),
  PLE residency policy is a non-trivial knob we'd have to tune. If it goes
  on GPU, we lose 4.7 GB of the 12 GB budget before *any* attention KV.

### Attention pattern (5:1 ratio at 512 window)
From the config `layer_types` list (35 entries):
```
[S, S, S, S, F,  S, S, S, S, F,  S, S, S, S, F,
 S, S, S, S, F,  S, S, S, S, F,  S, S, S, S, F,
 S, S, S, S, F]
```
where `S = sliding_attention`, `F = full_attention`.

- Ratio is **4:1 sliding-to-full per block** (every 5th layer is full), not
  the 1:5 mentioned in the Gemma 3 paper's text. **7 full-attention layers
  out of 35.**
- `sliding_window: 512` tokens (not 4096 as for Gemma 3 27B). This is the
  *local* window for sliding layers.
- `num_kv_shared_layers: 15` — the top 15 layers re-use KV states from a
  shared middle layer (the modeling code branches on
  `self.is_kv_shared_layer` and reads `shared_kv_states[kv_shared_layer_index]`
  inside `forward`; source:
  [modular_gemma3n.py — Gemma3nTextAttention](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3n/modular_gemma3n.py)).
  Those layers **never compute** their own K/V projections at inference.

### Head structure (GQA, very narrow)
- `num_attention_heads: 8`
- `num_key_value_heads: 2` (so GQA group size = 4)
- `head_dim: 256`
- `hidden_size: 2048`
- Q/K/V projections include `q_norm`, `k_norm`, `v_norm` (all RMSNorm) —
  delta-mem already handles `k_norm`/`v_norm` via
  `_normalize_key_states`/`_normalize_value_states`, so this part is fine.

### Native context window
- `max_position_embeddings: 32768` — **native 32 k**. (Gemma 3 4B/12B/27B
  are 128 k; Gemma 3n is 32 k.)
- Two rope freqs: `rope_theta: 1000000.0` (global), `rope_local_base_freq:
  10000.0` (sliding) — different rope per layer type.

### Multimodality
- Text + vision (MobileNet-v5-300M, `vocab_offset: 262144`) + audio
  (Conformer-style, `vocab_offset: 262272`, 12 layers, hidden 1536). All
  loaded into the same `Gemma3nForConditionalGeneration` checkpoint by
  default.
- Tokens 262144-262271 are vision soft tokens; 262273+ are audio. Text-only
  use still pays the multimodal tower cost unless we surgically delete them
  at load.

### HuggingFace IDs and licence
- Base: [`google/gemma-3n-E4B`](https://huggingface.co/google/gemma-3n-E4B)
- Instruction-tuned: [`google/gemma-3n-E4B-it`](https://huggingface.co/google/gemma-3n-E4B-it)
- Licence: **Gemma Terms of Use** (free for commercial use with a usage
  policy / responsible use clause — *not* OSI-approved open source).
- Both repos are **gated** — requires accepting the Gemma licence on the HF
  account before download. unsloth republishes the same weights ungated as
  [`unsloth/gemma-3n-E4B-it`](https://huggingface.co/unsloth/gemma-3n-E4B-it).

### Tokenizer
- `vocab_size: 262400` (262144 text + 128 vision + 128 audio).
- BOS=2, EOS=106 (`<end_of_turn>` in instruct), PAD=0, image_token=262145,
  audio_token=262273.

---

## 2. What delta-mem assumes about the base attention class

From `delta-Mem/deltamem/core/delta_impl.py:500-600` and `:640-665` and
`:2200-2280` (read in main repo since the worktree submodule is uninitialised).

### Interface contract (`base` must provide)
- **Linear projections** (must be `nn.Linear`-like with `in_features`,
  `out_features`, `weight` attributes):
  - `base.q_proj`, `base.k_proj`, `base.v_proj`, `base.o_proj`
  - Optional packed: `base.qkv_proj` (Gemma 3n does **not** pack)
- **Norm modules** (optional, applied if present):
  - `base.q_norm`, `base.k_norm`, `base.v_norm` — used by
    `_normalize_{query,key,value}_states`
- **Layer metadata**:
  - `base.layer_idx`, `base.head_dim`, `base.num_key_value_groups`,
    `base.scaling`, `base.attention_dropout`, `base.is_causal`,
    `base.config` (must expose `_attn_implementation`)
- **Optional sliding / sparsity hints** (defaulted via `getattr`):
  - `base.sliding_window`, `base.layer_type`, `base.is_sliding`,
    `base.is_kv_shared_layer`, `base.kv_shared_layer_index`,
    `base.store_full_length_kv`
- **Class type check**:
  ```python
  SUPPORTED_BASE_ATTENTION_TYPES = (Qwen3Attention,)
  if HAS_SMOLLM3:
      SUPPORTED_BASE_ATTENTION_TYPES += (SmolLM3Attention,)
  ```
  `attach_delta_mem` filters `isinstance(module, (Qwen3Attention,
  SmolLM3Attention))`. Gemma's class would be rejected outright.

### Lifecycle assumptions
- `forward(hidden_states, position_embeddings, attention_mask, past_key_values,
  cache_position, ...)` — signature matches Qwen3 & SmolLM3.
  - **Gap:** Gemma3n's signature adds `shared_kv_states: dict[int, tuple]`
    as an explicit positional/keyword arg passed by the decoder layer.
    Delta-mem's wrapper does not forward this, so KV-shared layers
    (15 of 35) would lose their KV-share behaviour unless the wrapper is
    explicitly threaded.
- Delta-mem mutates internal state (`self.delta_state`, `read_context_mask`,
  `last_*_norm` stats) on every forward — needs `reset_state()` between
  conversations. Same lifecycle as before, no Gemma-specific issue.
- Past-KV update is via the standard HF `Cache` API
  (`past_key_values.update(key_states, value_states, layer_idx,
  cache_kwargs)`). Gemma3n uses the standard `HybridCache` /
  `DynamicCache` — compatible at the surface, but the sliding-window
  variant truncates older keys per-layer, which interacts with delta-mem's
  position-of-read computations.

### `_apply_standard_rotary` and `_normalize_value_states` hooks (OSCAR uses)
- `_apply_standard_rotary` (lines 653-664) branches on
  `self.is_smollm3_attention`. For Gemma3n we'd need a third branch that
  calls `gemma3n_apply_rotary_pos_emb` and — critically — uses the
  *correct* rope (`rope_theta=1e6` for full layers, `rope_local_base_freq=1e4`
  for sliding layers). Currently the rotary cos/sin tensors come in via
  `position_embeddings`; we'd inherit whichever the parent decoder layer
  computed, so this *might* be transparent. Needs verification.
- `_normalize_value_states` (lines 647-651) just defers to `base.v_norm`.
  Gemma3n has `v_norm` ⇒ works as-is.
- OSCAR's rotation calibration (`run/compute_kv_rotation.py`) was fitted on
  Qwen3-4B's K and V distributions. Gemma3n's K/V distributions are
  different — different head_dim (256 vs Qwen3's 128), different head count
  (2 KV heads vs Qwen3's 8), different normalization → **rotations must be
  re-calibrated from scratch on Gemma3n activations.**

### GQA vs MQA specifics
- Gemma3n: 8 Q heads, 2 KV heads, `head_dim=256` ⇒ GQA group size 4, KV
  width 2 × 256 = 512.
- Delta-mem computes `num_key_value_heads = base.k_proj.out_features //
  self.head_dim` (line 582) — works with any GQA ratio. **OK in principle**,
  but the delta head ranks (`config.rank`) and the
  `base_v_out_features` calculation assume `v_proj` exists (Gemma3n has
  one — fine for the 20 non-KV-shared layers; the 15 KV-shared layers have
  `v_proj` defined but never call it).

---

## 3. Gap analysis: Gemma 3n vs delta-mem assumptions

| Assumption | Holds? | Notes |
|---|---|---|
| Separate `q_proj`/`k_proj`/`v_proj`/`o_proj` exist | YES | All four are `nn.Linear`. No `qkv_proj` packing. |
| `q_norm`/`k_norm`/`v_norm` (optional) | YES | All three RMSNorms present; delta-mem already handles them. |
| `layer_idx`, `head_dim`, `scaling` on base | YES | Standard HF attributes. |
| `forward(hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, **kwargs)` signature | PARTIAL | Gemma3n adds `shared_kv_states` as an extra arg the decoder always passes. Delta-mem wrapper would need to accept-and-forward it. |
| KV is computed inside this attention module | NO (for 15 of 35 layers) | `is_kv_shared_layer=True` layers read K/V from `shared_kv_states[kv_shared_layer_index]` — they do not call `k_proj`/`v_proj`. **Delta-mem's whole premise of adding `delta_k`/`delta_v` to projected states is meaningless for these layers**, because the K/V the layer uses came from a different layer entirely. |
| Uniform per-layer attention type | NO | 28 sliding (window=512), 7 full. Delta-mem attaches to *all* attention modules in `attach_delta_mem`; we'd want to skip sliding-window layers (or at minimum the KV-shared subset) and only attach to the 7 full layers — otherwise delta-mem's "global memory" reads compete with a 512-token local window that throws away most of the context anyway. |
| Single rope per model | NO | Two rope basis frequencies (1e6 global, 1e4 local). Already handled at decoder layer level via `position_embeddings`, but the OSCAR rotations would need to be calibrated separately per layer type or pooled carefully. |
| MatFormer per-layer width | YES (in shipped E4B) | `intermediate_size` is uniform 16384 across all 35 layers in the public config. The MatFormer story is for users who derive an E2B sub-model; for E4B inference it's a non-issue. (Confirmed against config.json.) |
| Activation sparsity uniform | NO | Layers 0-9 are 95% sparse, 10-34 are dense. Doesn't directly affect delta-mem (it wraps attention, not FFN) but means layer 0-9 attention sits in a very different residual stream than the rest — adapter capacity per layer should probably be allocated accordingly. |
| AltUp/Laurel transparent to attention wrapper | UNKNOWN | AltUp maintains 4 parallel residual streams; the active one (`altup_active_idx=0`) is what attention sees. Laurel is a parallel block at the layer level. Delta-mem wraps attention only, so it *should* see the same input shape, but the inter-layer state coupling that AltUp/Laurel introduce means delta-mem's "memory of layer k informs layer k+1" assumption is more entangled than for Qwen/SmolLM3. **Needs experimental verification before any training run.** |
| PLE doesn't interact with cross-forward state | YES | PLE is purely an embedding-table fetch per token per layer; once the layer's input hidden state is built, PLE is done. Does not affect what delta-mem tracks. **However**, PLE adds 4.7 GB of weights, which is a hard memory cost. |

### Specific question: should delta-mem attach to sliding-window layers?
**Recommendation: no.** Sliding layers see a 512-token window of context;
delta-mem's value-add is *additional* memory beyond the local window. If we
attach to a sliding layer, the cross-attention computation reads a tiny
window and the delta-mem read injects "global" content — which is *exactly*
what delta-mem is for, so in theory it's a perfect match. **However**, the
training-data distribution Qwen3's published adapter learned on assumes the
base attention can see the entire context. Repurposing the same recipe for
sliding-window layers would require a retraining curriculum specifically
designed for that mismatch — none of which is in delta-mem's published
literature.

A safer first pass: attach delta-mem **only to the 7 full-attention layers**
(layer indices 4, 9, 14, 19, 24, 29, 34), exactly as the existing
`config.target_layers` filter supports. This loses 80% of the per-layer
adapter capacity vs Qwen3 (where we attach to all 28 layers) — a real
quality hit that compounds the risks below.

---

## 4. Memory analysis

### Weights at bf16

| Component | Param count | bf16 GB |
|---|---|---|
| Text transformer (35 layers, hidden 2048, FFN 16384, 8/2 heads × 256) | ~3.6 B | 7.2 |
| PLE table (35 × 256 × 262144) | 2.35 B | 4.7 |
| Token embedding (262400 × 2048) | 0.54 B | 1.08 |
| Vision tower (MobileNet-v5-300M → hidden 2048) | ~0.4 B | 0.8 |
| Audio encoder (12 Conformer layers, hidden 1536) | ~0.3 B | 0.6 |
| AltUp + Laurel block parameters | ~0.1 B | 0.2 |
| **Total checkpoint at bf16** | **~7.85 B (HF reported)** | **~14.6 GB** |

The LONG_CONTEXT_PLAN.md estimate of ~13 GB was close but slightly low —
the actual checkpoint footprint is **~14.6 GB** at bf16, with **PLE
contributing 4.7 GB** of that. **This does not fit on a 12 GB or 16 GB
card.**

If we drop the vision + audio towers at load (text-only use case):
**~13.0 GB bf16** — still over the 12 GB ceiling.

### Per-token KV at OSCAR INT2 packed
- KV per layer per token, full bf16 reference: 2 KV heads × 256 head_dim ×
  2 (K and V) × 2 bytes = **2048 bytes/layer/token = 2 KB**.
- 35 layers × 2 KB = **70 KB/token at bf16** — but this ignores
  KV-sharing (only 20 layers actually store K/V; the other 15 reuse). Real
  bf16 per-token KV is **20 × 2 KB = 40 KB/token**.
- At OSCAR INT2 packed (16x compression vs bf16): **40 / 16 = 2.5 KB/token**.
- Compared to Qwen3-4B (~20 KB/tok INT2 in the plan's table): Gemma3n is
  **8× cheaper per token** thanks to GQA-2 + KV sharing.

### At 32 k context

| Component | Size |
|---|---|
| Backbone weights bf16 (text only, with PLE on GPU) | ~13.0 GB |
| Backbone weights INT4 NF4 (text only, PLE bf16) | ~6.0 GB |
| Backbone weights INT4 with PLE offloaded to CPU | ~1.6 GB (text transformer only) |
| OSCAR INT2 KV at 32k (20 effective layers × 2.5 KB/tok × 32768) | **2.6 GB** |
| Activations at 32 k (hidden_size 2048, layer 35, AltUp 4×) | ~2.5 GB |
| `_assemble` peak transient | ~0.15 GB |
| **Total (INT4 weights + PLE offload + OSCAR INT2 KV @ 32k)** | **~6.85 GB** |
| **Total (bf16 weights + OSCAR INT2 KV @ 32k)** | **~18.3 GB ✗** |

**Conclusion: bf16 weights do not fit on a 12 GB or 16 GB card at any
context length.** INT4 weight quantization is **mandatory**, *and* PLE must
be offloaded to CPU (which costs PCIe bandwidth per forward — measurable
latency hit). With INT4 + PLE offload + OSCAR INT2 KV, ~6.85 GB at 32 k —
fits on a 12 GB card with reasonable headroom, but only after **two
quantization layers (NF4 weights + OSCAR INT2 KV) are applied to a model
the OSCAR rotation has never seen.** That stack has never been calibrated
or measured together.

### PLE memory cost at 32 k
PLE is a *parameter*, not a per-token activation — its cost is constant
regardless of context length: **4.7 GB bf16**, or ~1.2 GB if we INT4-quant
the PLE table too (untested by Google; not in any published recipe). The
4.7 GB sits entirely in the embedding table and does not scale with context.

---

## 5. Effort estimate

### LOC breakdown (engineer-only, no training)

| Area | Est. LOC | Notes |
|---|---|---|
| `backbone_compat.py` — add `Gemma3nTextAttention` import, `HAS_GEMMA3N` flag, gemma rotary import, gemma eager-attention-forward, `ensure_attention_compat_views` updates to wire AltUp/Laurel inputs if needed | **80-150** | Worktree's `backbone_compat.py` is currently 14 lines; SmolLM3 support was added by importing 3 symbols. Gemma3n is heavier because of `shared_kv_states` plumbing. |
| `delta_impl.py` — extend `SUPPORTED_BASE_ATTENTION_TYPES`, new branches in `_apply_standard_rotary` for gemma rope (global vs local), `is_gemma3n_attention` checks, `shared_kv_states` forwarding through delta `forward()`, gating so delta-mem on KV-shared layers no-ops cleanly | **150-300** | The KV-shared-layer no-op alone is delicate: we'd want to *not* wrap those layers in the first place, but `attach_delta_mem` currently iterates *all* attention modules. |
| `delta.py` (mainline wrapper) — update isinstance checks and union types | **20-40** | Mostly mechanical. |
| OSCAR rotation calibration adaptation (`run/compute_kv_rotation.py`, `run/oscar_dump_qkv.py`) — Gemma3n-specific dump path, GQA-2 head-dim-256 handling, decide per-layer-type vs pooled rotation | **150-250** | Existing scripts assume one rotation per K, one per V; Gemma3n's two rope frequencies argue for two rotation pairs (or two separate calibrations). |
| Training script (`delta-Mem/deltamem/train/delta_sft.py`) — `--base-model google/gemma-3n-E4B`, handle gated repo auth, special-token mask updates for vision/audio tokens, `freeze_non_delta_mem_params` extensions for AltUp/Laurel | **100-200** | Has to thread through HF Gemma3n loader with `attn_implementation` set carefully, plus the multimodal token IDs. |
| Eval harness (`run/locomo_eval.py`, `run/_chunked_eval_runner.py`) — gemma3n loader path, `--quantize-backbone-int4` interaction, PLE-offload toggle, text-only stripping | **80-150** | Need to land NF4-weight quant flag here anyway (also in LONG_CONTEXT_PLAN Option 4). |
| Test additions (port the SmolLM3 unit tests to gemma3n; round-trip tests for `attach_delta_mem` + reset + save/load; rotation-calibration sanity) | **200-400** | Existing test suite mostly already cross-backbone parameterised; we'd add a `gemma3n` parameter. |
| **Subtotal (code only)** | **~780-1490 LOC** | |

### Wall-clock time (one engineer)

- **Phase A: attach** — get `attach_delta_mem` to wrap exactly the 7
  full-attention layers without errors, forward pass returns sane shapes,
  generation produces grammatical output: **3-5 days**.
- **Phase B: rotation calibration** — dump Q/K/V on Gemma3n, fit OSCAR
  rotations, sanity-check that the bf16→INT2 round trip preserves logits
  to within tolerance: **3-5 days** (and assumes we can afford bf16 weights
  for the calibration step on a larger machine, since 12 GB box can't load
  bf16).
- **Phase C: integration with INT4-weight backbone + PLE offload** —
  re-validate that the calibration done in B is still valid when weights
  are NF4 (likely not — see risk 1). Re-do calibration on top of NF4 if
  required. **5-10 days**.
- **Phase D: adapter training** — from-scratch delta-mem adapter on
  Gemma3n, on cloud H100/A100. **3-5 days wall + $200-600 cloud.**
- **Phase E: evaluation + iteration** — LoCoMo + LongMemEval + InfBench at
  17 k / 25 k / 32 k. **5-10 days**.

**Wall total: 4-7 weeks** for one engineer, **+ $200-600 cloud training**.

### Risk areas (ranked highest first)

1. **NF4 + OSCAR INT2 KV double-quant on an uncalibrated model** —
   blocking. The whole 12 GB budget requires NF4 weights, but the OSCAR
   rotation was designed for bf16 attention statistics. We have no
   measurement of how rotation quality degrades when the *backbone itself*
   is INT4. Risk shared with Option 4 but compounded because Gemma3n's
   activation distribution is also unfamiliar.
2. **KV-shared layers (15 of 35)** — delta-mem's value-add evaporates on
   layers that don't compute their own K/V. Cleanly skipping them is
   straightforward (`target_layers=[0,1,2,...,19]`), but means the
   effective capacity of delta-mem is roughly halved vs Qwen3 (28 layers).
3. **Sliding-window layers (28 of 35)** — if we attach delta-mem to them,
   we mix global memory with a 512-token local window, an untested
   training regime. If we *don't*, we attach only to the 7 full-attention
   layers — even less capacity. Either way, *expected quality below
   Qwen3 + adapter*, before we've spent the training budget.
4. **AltUp + Laurel inter-layer coupling** — Gemma3n maintains 4 parallel
   residual streams (AltUp) and runs Laurel blocks in parallel to each
   layer. Delta-mem's "the residual at layer k flows to layer k+1
   unmolested" assumption is no longer simple. Empirical investigation
   required before we can predict adapter behaviour.
5. **PLE memory + PCIe latency** — offloading PLE to CPU costs ~10-30 ms
   per layer per forward in PCIe Gen3 land, multiplied by 35 layers
   ⇒ added latency that dominates the inference loop. Not a blocker for
   training (PLE stays GPU there) but reshapes the deploy story.
6. **Gated weights + multimodal weight bloat** — we'd ship adapter
   checkpoints designed for a gated base, complicating the publish path.
   Also need to strip vision/audio towers at load to free 1.4 GB. Solvable
   but a project-management overhead.

---

## 6. Recommendation

**DEFER INDEFINITELY.** The argument:

1. The promise of Gemma 3n is multimodal + 32 k native context; delta-mem
   gives us memory *compression*, not new modalities. Qwen3-4B + adapter
   already hits 32 k VRAM (LONG_CONTEXT_PLAN.md, Option 1) and SmolLM3-3B
   already hits 64 k VRAM (Option 2), both with existing delta-mem
   integration.
2. Gemma3n's bf16 weights don't fit on a 12 GB card; we'd be forced to
   stack NF4 weight quant *and* OSCAR INT2 KV quant *and* untrained
   AltUp/Laurel awareness *and* PLE offload *and* sliding-window
   delta-mem — five untested compounding risks for a model whose unique
   benefit (multimodality) delta-mem doesn't enhance.
3. KV-shared + sliding layers mean delta-mem's effective per-layer
   capacity drops to ~7-20 attached layers (vs Qwen3's 28), a real
   quality ceiling we cannot raise with more training budget.
4. The 4-7 engineer-week cost + $200-600 cloud training is comparable to
   Options 1+2 *combined* and delivers strictly less than Option 1's
   demonstrated path.
5. **Build only if:** (a) we get a concrete multimodal long-context
   requirement (e.g., LoCoMo-with-images), (b) Option 1 or 2 has shipped
   and is *insufficient* on a specific real workload, and (c) we have a
   16+ GB card to develop on without the NF4-on-Gemma3n risk.

---

## 7. Alternatives to building Gemma support ourselves

> **Deep-dive companion doc:** EpiCache and LoCoCo are evaluated in detail
> in [`long-context-alternatives.md`](./long-context-alternatives.md)
> (side-by-side comparison, integration cost analysis, 3-day pilot
> proposal). Summary: **EpiCache** is the strongest stacking candidate
> against our existing OSCAR+delta-mem (training-free, MIT, fresh
> Oct-2025 Apple release, evaluates natively on LoCoMo, Qwen2.5/Llama-3
> already wired); the Qwen3 port is ~50 LOC. **LoCoCo** is not
> competitive (dead repo since Sep-2024, Llama-2-7B-only intrusive
> fork, no GQA support, training required, no published checkpoint,
> no LoCoMo eval).

1. **EpiCache (ArXiv 2509.17396, Sep 2025)** — episodic KV cache
   management for long-term conversation on resource-constrained
   environments. Works as a drop-in on any HF attention class via
   monkey-patch of the cache layer, not the attention layer. Likely Gemma
   3-compatible without modeling-code changes
   ([EpiCache paper](https://arxiv.org/html/2509.17396)). Closest
   spiritual match to delta-mem but is **eviction-based**, not delta-write
   based — fundamentally less expressive on multi-session memory.
2. **LoCoCo (ArXiv 2406.05317)** — convolutional KV compression that
   advertises "universal compatibility with existing LLM architectures",
   architecture-agnostic ([LoCoCo paper](https://arxiv.org/pdf/2406.05317)).
   Less learned-adapter, more drop-in compression. Worth a smoke test if
   the multimodal long-context requirement materializes.
3. **Gemma 3 native long-context training (Unsloth + LoRA)** — Unsloth's
   fork of Gemma 3 supports LoRA fine-tunes at 6× longer sequences than
   FA2 on a 48 GB GPU ([Unsloth — Fine-tune Gemma
   3](https://unsloth.ai/blog/gemma3)). Not memory-adapter territory, but
   if the goal is "more useful context on Gemma 3n specifically" and we
   have a 48 GB card available, this is the lowest-friction path. Trades
   delta-mem's compression story for raw long-context fine-tuning.

No existing delta-mem-like adapter for Gemma 3 was found on the HF Hub
(`hub_repo_search query="delta-mem gemma"` returned zero results). The
closest reusable artifact is the Gemma3n model code in
[`transformers`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3n/modular_gemma3n.py) — i.e., we'd be the first to do this.

---

## Sources

- [`unsloth/gemma-3n-E4B-it/config.json`](https://huggingface.co/unsloth/gemma-3n-E4B-it/raw/main/config.json) — authoritative architecture config (layer counts, head_dim, layer_types, KV sharing, PLE fields)
- [`google/gemma-3n-E4B-it` HF model card](https://huggingface.co/google/gemma-3n-E4B-it) — official 7.85 B param count, gated, gemma licence
- [`google/gemma-3n-E4B` HF model card](https://huggingface.co/google/gemma-3n-E4B) — base (non-instruct) variant
- [Google AI for Developers — Gemma 3n model overview](https://ai.google.dev/gemma/docs/gemma-3n) — official "effective 1.91 B" claim, multimodality, 32 k context
- [Google Developers Blog — Introducing Gemma 3n: The developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/) — MatFormer + PLE overview
- [HF blog — Understanding Gemma 3n: How MatFormer Gives You Many Models in One](https://huggingface.co/blog/rishiraj/matformer-in-gemma-3n) — MatFormer FFN nesting, PLE CPU offload story
- [`transformers/src/transformers/models/gemma3n/modular_gemma3n.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3n/modular_gemma3n.py) — `Gemma3nTextAttention` source: projections, q/k/v norms, `is_kv_shared_layer`, `shared_kv_states`, rotary
- [Gemma 3 Technical Report (ArXiv 2503.19786)](https://arxiv.org/abs/2503.19786) — 5:1 sliding:full attention ratio rationale (note: shipped E4B is 4:1)
- [Gemma explained: What's new in Gemma 3 (Google blog)](https://developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/) — sliding/global mix design intent
- [EpiCache paper (ArXiv 2509.17396)](https://arxiv.org/html/2509.17396) — alternative cache-eviction long-context strategy
- [LoCoCo paper (ArXiv 2406.05317)](https://arxiv.org/pdf/2406.05317) — alternative convolutional KV compression
- [Unsloth — Fine-tune Gemma 3](https://unsloth.ai/blog/gemma3) — alternative LoRA-only long-context path

### Internal sources cross-referenced
- `LONG_CONTEXT_PLAN.md` (this branch) — Option 3 framing and budget table
- `delta-Mem/deltamem/core/delta_impl.py:480-700, 2200-2280` — interface contract
- `delta-Mem/deltamem/core/backbone_compat.py:1-14` — SmolLM3 compat pattern
- `delta-Mem/deltamem/core/delta.py:1-148` — mainline wrapper / `attach_delta_mem`
