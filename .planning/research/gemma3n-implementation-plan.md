# Gemma 3n E4B + delta-mem: Implementation Plan

**Status:** Plan only — no code changes. Companion to
`.planning/research/gemma3n-deltamem-feasibility.md`.
**Author:** Claude (Opus 4.7) | **Date:** 2026-06-09
**Branch:** `gemma3n-plan`
**Scope:** "If we ever decide to add Gemma 3n support, here's exactly how."

---

## Pre-flight: A material change since the feasibility doc

The feasibility doc was written 2026-06-09 morning. Between that doc and this
plan, the following was confirmed via web search:

- **Gemma 4 E4B exists and was released April 2026** (~2 months ago at the
  time of writing). It is the named successor to Gemma 3n's edge-model line.
  ([HuggingFace blog — Welcome Gemma 4](https://huggingface.co/blog/gemma4),
  [google/gemma-4-E4B-it on HF](https://huggingface.co/google/gemma-4-E4B-it),
  [botmonster.com — Gemma 4 architecture](https://botmonster.com/posts/gemma-4-architecture-per-layer-embeddings-shared-kv-cache-dual-rope/))
- Gemma 4 E4B specs (from those sources):
  - **128 k native context** (vs Gemma 3n's 32 k)
  - **42 layers**, 24 compute KV + 18 KV-shared
  - **Drops AltUp and Laurel** ("highly compatible across libraries and
    devices" — quoted from the HF blog)
  - Keeps **PLE** (parameter-efficient)
  - Keeps **shared-KV** (now 18 of 42 layers vs Gemma 3n's 15 of 35)
  - Keeps **alternating sliding (512) + global** attention
  - Adds **dual RoPE** (pruned p-RoPE for global layers)
  - Multimodal: text + image + audio (USM-style conformer carried over)

**Implication:** any work targeting Gemma 3n now has a strictly-better
target one model-generation away. Two of the three blocking risks from the
feasibility doc — AltUp/Laurel coupling and the 32 k native limit —
are *resolved by Gemma 4*. The remaining hard risks (KV-shared layers,
sliding-window interaction, NF4+OSCAR compounding, PLE residency) are
**shared between Gemma 3n and Gemma 4**.

**Therefore this plan is written so that ~90 % of the work is reusable for
Gemma 4 E4B as well**, with Gemma-3n-specific code clearly marked and
swappable. Phases 1-7 are nearly identical for either target. Phase 8
diverges (training-data context lengths, AltUp/Laurel awareness).

The phases below are written for Gemma 3n; the Recommendation section
(#10) discusses whether to even start.

---

## 1. Decision gate — when to pick this plan up

### Build IF **ALL** of the following:
1. **Option 1 has shipped.** A Qwen3-4B + retrained adapter checkpoint exists
   and has been verified on the local 12 GB host at 32 k (`strix/verify_checkpoint.py`
   anchor ≥ 1.25, extension ≥ 1.20, stretch ≥ 1.10).
2. **A concrete real workload demands what Gemma 3n uniquely offers** — one of:
   - Multimodal long-context (e.g., LoCoMo-with-image-attachments, video
     QA over hour-long meetings) where dropping vision/audio is unacceptable.
   - 140+ language coverage where Qwen3 / SmolLM3 underperform on a measured
     non-English benchmark (LMSYS multilingual, MMLU-multilingual, …).
3. **At least one of**:
   - A 16+ GB development card is available (avoids the NF4+OSCAR risk
     compounding on first integration).
   - An H100/A100 cloud budget of **$400-800** is approved for calibration +
     training (single-engineer 3-4 weeks).
4. **The Gemma 4 alternative has been considered and rejected** — either
   because the work targets the production-deployed Gemma 3n ecosystem
   (mobile / Edge TPU) or because Gemma 4's hybrid attention is *more*
   complex (e.g. dual RoPE) and Gemma 3n's simpler architecture is preferred
   as a stepping-stone. See Recommendation (#10).

### Skip permanently IF **ANY** of:
1. Option 1 (Qwen3 + retrained adapter) closes the 32 k quality gap to
   ratio ≥ 1.20 AND we have no multimodal requirement. (Doing nothing is the
   right call.)
2. EpiCache or LoCoCo integration (see §9 and the parallel
   "Gemma-alternatives" agent output) lands as a drop-in for **any** HF
   attention class on Gemma 3n with acceptable quality. We then route
   "want Gemma 3n long context" through *that*, not through adding it as a
   delta-mem backbone.
3. A new model in the same family (Gemma 4 has happened — Gemma 5 might
   happen) supersedes Gemma 3n before we start.
4. Google ships an *official* long-context adapter or extended-context
   variant of Gemma 3n.

### Defer (reassess in 3 months) IF:
- The decision gate "Build IF all" is partially met (e.g. Option 1 shipped,
  but no concrete multimodal workload yet) — the trigger is the workload.

---

## 2. Phase plan (5 phases — collapsed from the feasibility doc's 8)

Five phases instead of eight: the feasibility doc's separate AltUp,
Laurel, and quantization phases are folded into Phase 2 (architecture
adaptation) and Phase 4 (memory budget). The reason for the collapse is
that AltUp + Laurel + NF4 are **only meaningful together** — none of them
have isolated acceptance criteria worth a separate phase. Calibration is
also folded with adapter training because they share a data pipeline.

### Phase 1 — Repository scaffolding & attach plumbing
**Goal in one sentence:** `attach_delta_mem(gemma3n_model, config)` runs
without error and wraps exactly the 7 full-attention layers; forward pass
returns sane shapes; no Gemma-3n-specific math yet.

**Deliverables (file path | est. LOC):**
- `delta-Mem/deltamem/core/backbone_compat.py` | **+30 LOC** — add
  `Gemma3nTextAttention` import wrapped in `try/except ImportError` exactly
  mirroring the SmolLM3 pattern; expose `HAS_GEMMA3N` flag; re-export
  `gemma3n_apply_rotary_pos_emb` and `gemma3n_eager_attention_forward`;
  extend `ensure_attention_compat_views` to no-op on Gemma3n attention
  modules.
- `delta-Mem/deltamem/core/delta.py` | **+12 LOC** — extend the
  `isinstance` filter in `attach_delta_mem` to include `Gemma3nTextAttention`
  (guarded by `HAS_GEMMA3N`); update the type union in `DeltaMemAttention.__init__`.
- `delta-Mem/deltamem/core/delta_impl.py` | **+40 LOC** — add
  `is_gemma3n_attention` flag in `DeltaMemAttention.__init__` (line ~514
  pattern); add a Gemma3n branch in `_apply_standard_rotary`; extend
  `SUPPORTED_BASE_ATTENTION_TYPES` (line 501-ish union); thread
  `shared_kv_states` through `forward(...)` as `**kwargs`-acceptable.
- `run/gemma3n_smoke.py` | **+120 LOC NEW** — mirror of
  `run/smollm3_smoke.py`: import test, CPU bf16 load test (skippable via
  `GEMMA3N_SMOKE_SKIP_LOAD=1` for RAM-tight hosts), isinstance check on
  every `model.language_model.layers[*].self_attn`, `attach_delta_mem` wrap
  with a stub config, `freeze_non_delta_mem_params` non-empty trainable
  list assertion.

**Acceptance criteria (testable, in order):**
1. `python -m run.gemma3n_smoke` exits 0 on the dev host (CPU only).
2. Imports of `HAS_GEMMA3N` and `Gemma3nTextAttention` resolve.
3. `attach_delta_mem` returns a non-empty list of replaced module names.
   That list **must equal exactly** the 7 full-attention layer indices
   (4, 9, 14, 19, 24, 29, 34) when `target_layers` is left to default.
4. Forward pass on a 16-token random-input prompt completes without error
   and returns logits of shape `(1, 16, 262400)`.
5. SmolLM3 + Qwen3 smoke tests still pass — no regression to existing
   backbones.

**Time estimate:** 3 engineer-days.

**Dependencies:** none.

---

### Phase 2 — Sliding-window + KV-shared layer handling
**Goal in one sentence:** delta-mem skips sliding-window layers and
KV-shared layers cleanly, attaches only to the 7 full-attention non-shared
layers, and Gemma's `shared_kv_states` dict is threaded through the
wrapper so non-attached KV-shared layers still get their shared K/V.

**Deliverables:**
- `delta-Mem/deltamem/core/delta.py` | **+30 LOC** — extend `attach_delta_mem`
  filter logic: if `module.is_sliding` and `config.gemma3n_skip_sliding=True`
  (new config field, defaulted True), skip. If `module.is_kv_shared_layer`
  and `config.gemma3n_skip_kv_shared=True` (new config field, defaulted
  True), skip. Emit a `logging.info` summary per layer with the reason for
  skip vs wrap.
- `delta-Mem/deltamem/core/delta_impl.py` | **+25 LOC** — in
  `DeltaMemAttention.__init__`, store `self.gemma3n_skip_sliding` and
  `self.gemma3n_skip_kv_shared`; in `forward(...)`, accept and forward
  `shared_kv_states` to `self.base(...)` if present.
- `delta-Mem/deltamem/core/delta_impl.py:476-484` (HFDeltaMemConfig
  fields) | **+10 LOC** — add the two new bool config fields, with safe
  defaults.
- `tests/test_gemma3n_attach.py` | **+150 LOC NEW** — parametrized tests:
  with default config, exactly the 7 full-attention non-shared layers are
  wrapped (asserted by inspecting the returned list). With
  `target_layers=None` and both skip flags False, all 35 layers are wrapped.
  With `gemma3n_skip_kv_shared=True` and `gemma3n_skip_sliding=False`,
  layers 0-19 are wrapped (the non-shared layers). All asserts use
  Gemma3n's published `layer_types` array.

**Acceptance criteria:**
1. Under default config, exactly 7 modules wrap. List is
   `[4, 9, 14, 19, 24, 29, 34]`. Confirmed by parsing the wrapped-layer log.
2. `forward` over a 1k-token random prompt produces non-NaN logits and
   matches a no-delta-mem `forward` on the **same** Gemma3n model within
   `atol=1e-3` at every position (because delta-mem with zero-init starts
   identity — same property already covered by Qwen3/SmolLM3 unit tests).
3. KV-shared layers 20-34 still read their K/V from
   `shared_kv_states[kv_shared_layer_index]` correctly; verified by
   instrumenting the base `forward` and asserting the KV-share path was
   taken for each of layers 20-34.
4. The `is_sliding` layers still see the 512-token window
   (`sliding_window=512` attribute preserved through the wrapper).

**Time estimate:** 4 engineer-days.

**Dependencies:** Phase 1.

---

### Phase 3 — OSCAR rotation calibration for Gemma 3n
**Goal in one sentence:** Two pairs of OSCAR K/V rotations are computed
for Gemma 3n (one pair for the 7 full-attention layers, one pair for the
non-KV-shared sliding layers), saved under
`data/oscar/rotations/gemma3n_gpqa/`, and the bf16→INT2 round-trip
preserves logits to within tolerance on a held-out 1 k-token prompt.

**Deliverables:**
- `run/oscar_dump_qkv.py` | **+50 LOC modified** — generalize the
  attention-class detection from "Qwen3 or SmolLM3" to also accept
  Gemma3n; honour the dual-rope (`rope_theta=1e6` global,
  `rope_local_base_freq=1e4` local) by **separating dumps by layer type**:
  output `gemma3n_gpqa_full/` (7 layers) and
  `gemma3n_gpqa_sliding/` (13 sliding non-shared layers).
- `run/compute_kv_rotation.py` | **+20 LOC modified** — no functional
  change, just verify it works on head_dim=256 (vs Qwen3's 128); add a CLI
  smoke-print so we catch any matrix-size assumption at load.
- `run/oscar_calibrate_gemma3n.py` | **+150 LOC NEW** — mirror of
  `run/oscar_calibrate_smollm3.py`: orchestrates dump → fit → save for
  both layer-type partitions; emits two pairs of .pt rotation files.
- `third_party/oscar-transformers/oscar_transformers/rotation.py` |
  **+80 LOC** — in `_build_patched_forward`, add a Gemma3n branch that
  detects `Gemma3nTextAttention`, looks up the right rotation pair for
  the layer (full vs sliding) from a class-level dict, and applies it
  with the Gemma3n-specific RoPE call.
- `data/oscar/rotations/gemma3n_gpqa/` | new directory holding
  `{k,v}_rotation_{full,sliding}_*.pt` files (~20 MB total).

**Acceptance criteria:**
1. Calibration script completes in < 4 h on a 16+ GB card (need bf16
   weights for calibration; 12 GB card will not fit Gemma 3n bf16 — see §6).
2. Round-trip logit error on a 1 k-token GPQA-style prompt: KL divergence
   between bf16-baseline and OSCAR-INT2 outputs ≤ 0.10 nats per token at
   95th percentile (same bar we hit for Qwen3).
3. The two rotation pairs differ measurably (Frobenius norm of difference
   > 0.01) — proving the separate calibration is meaningful, not a
   collapse to one global rotation.
4. `run/oscar_smoke.py --backbone gemma3n` exits 0 (extends the existing
   smoke harness; ~15 LOC of extra branching there).

**Time estimate:** 4 engineer-days (assumes one-off access to a 16+ GB
card for calibration).

**Dependencies:** Phase 1 (need attach to work for the dump phase to
hook the projections).

---

### Phase 4 — Memory: NF4 backbone + PLE offload + 12 GB fit-check
**Goal in one sentence:** Gemma 3n E4B loads on the 12 GB dev host with
NF4 weights and PLE offloaded to CPU, runs a 4 k smoke-eval with the
calibrated rotations, and stays under 11 GB peak VRAM.

**Deliverables:**
- `run/_chunked_eval_runner.py` | **+60 LOC modified** — add a
  `--ple-offload` flag (Gemma3n-only); when set, after model load, walk
  `model.language_model.embed_tokens_per_layer` and pin it to CPU; rely on
  HF's standard `device_map` to route per-token PLE lookups through PCIe.
  Wire `--quantize-backbone-int4` so Gemma3n loads through
  `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.bfloat16)`.
- `run/locomo_eval.py` | **+15 LOC modified** — accept the two new
  flags above and forward to the runner. Same surface for SmolLM3/Qwen3
  (no change to their behaviour because both default to no-quant).
- `run/gemma3n_memory_probe.py` | **+150 LOC NEW** — mirror of the
  immediate-action verification snippet in `LONG_CONTEXT_PLAN.md` lines
  275-298. Builds a 4 k, 16 k, 32 k synthetic prompt; runs Gemma3n with
  NF4 weights + PLE offload + OSCAR INT2 KV + delta-mem zero-init; logs
  peak VRAM at each length. Emits a JSON with the budget table.

**Acceptance criteria:**
1. On the 12 GB local host: 4 k peak ≤ 7.0 GB; 16 k peak ≤ 9.0 GB;
   32 k peak ≤ 11.0 GB. (Tighter ceilings than the feasibility doc's
   ~6.85 GB estimate because OOM under load is sneakier than the
   accounting suggests.)
2. PLE offload genuinely lives on CPU — verified by `torch.cuda.memory_allocated()`
   delta < 1.0 GB after loading PLE on a CPU-only run vs a GPU-only run.
3. Generation produces grammatical English on a "Tell me a story about a
   cat" prompt at 4 k context. (Smoke quality; not a benchmark.)
4. The NF4 + OSCAR INT2 + delta-mem zero-init combo doesn't NaN. (Tested
   by running 256 forward passes and asserting no NaN in logits.)

**Time estimate:** 5 engineer-days (PLE offload mechanics likely need
debugging; HF's PLE handling has rough edges).

**Dependencies:** Phase 3 (rotations) and Phase 2 (attach plumbing).

---

### Phase 5 — Adapter training + end-to-end eval
**Goal in one sentence:** A delta-mem adapter for Gemma 3n is trained from
scratch on the LoCoMo + LongMemEval + InfBench mix at 32 k context, and
`gemma3n/verify_checkpoint.py` reports anchor ≥ 1.15, extension ≥ 1.10
on the local 12 GB host.

**Note on quality bars:** Lower than Qwen3 (1.25 / 1.20) because:
- Only 7 attached layers (vs Qwen3's 28) = ~25 % of the per-layer
  capacity.
- AltUp's 4 parallel residual streams mean delta-mem's correction is
  partially "diluted" — we receive one of four streams, write to it, and
  the AltUp combiner mixes our correction with three streams that delta-mem
  never touched.
- We have no published starter checkpoint to fine-tune from; full from-scratch.

**Deliverables:**
- `delta-Mem/deltamem/train/delta_sft.py` | **+50 LOC modified** —
  accept `--base-model google/gemma-3n-E4B-it` (or `unsloth/gemma-3n-E4B-it`
  ungated); update `freeze_non_delta_mem_params` to also freeze AltUp and
  Laurel block params (currently they would be discovered as unfrozen
  because they don't match delta-mem name patterns; need explicit pattern
  matching for `altup_*` and `laurel_*` weight names).
- `delta-Mem/deltamem/train/delta_sft.py` | **+30 LOC** — add multimodal
  token id awareness: special-token mask must include vision soft tokens
  (262145-262272) and audio tokens (262273+) as "do not count toward loss",
  so the adapter training stays text-only.
- `gemma3n/` (new top-level dir mirroring `strix/`) | new dir
  - `gemma3n/__init__.py` | **+5 LOC**
  - `gemma3n/train_phase1.py` | **+200 LOC** — adapted from
    `strix/train_phase1.py` for Gemma 3n base model id + the multimodal
    token masking + the two-pair OSCAR rotation env vars.
  - `gemma3n/verify_checkpoint.py` | **+150 LOC** — adapted from
    `strix/verify_checkpoint.py`; lower thresholds per above; emits the
    same kind of pass/fail table.
  - `gemma3n/README.md` | **+50 LOC** — how to run end-to-end on Strix
    Halo (96 GB) or a rented H100.
- `run/training_smoke.py` | **+30 LOC modified** — add a
  `--backbone gemma3n` flag mirroring the existing Qwen3/SmolLM3 paths.

**Acceptance criteria:**
1. `python -m gemma3n.train_phase1 --max-steps 100 --backbone-id ...`
   completes 100 training steps on a 24+ GB cloud GPU (sanity smoke).
   Loss decreases monotonically over the first 50 steps (no exploding
   gradient from AltUp coupling).
2. Full training run (Phase 1, 32 k context, ~10 k steps) completes in
   < 5 days on a single H100 / < 7 days on Strix Halo.
3. On the 12 GB local host, `python -m gemma3n.verify_checkpoint
   --ckpt path/to/final`:
   - Anchor (conv-26 / 10 q at ~17 k): ratio ≥ 1.15
   - Extension (conv-41 / 10 q at ~25 k): ratio ≥ 1.10
   - Stretch (conv-26 × 2 at ~32 k): ratio ≥ 1.05 (delta-mem at least not
     hurting)
4. Multimodal regression: a tiny image-prompted forward (single image,
   "What colour is this?") still completes without error and produces
   non-degenerate text. (Confirms our `freeze_non_delta_mem_params` change
   didn't break the vision tower.)

**Time estimate:** 8-12 engineer-days **plus** 3-5 wall days of remote
training, **plus** $400-600 cloud cost (H100).

**Dependencies:** Phases 1-4.

---

### Total time / cost
- **Engineer-days:** 24-30 (3-4 weeks for one engineer)
- **Wall-clock:** 4-5 weeks if no parallel work
- **Cloud:** $400-800 (H100 for training; calibration on Strix Halo
  or rented A6000 for free)
- **New + modified LOC:** ~1,200 (see §3)

---

## 3. Per-file change summary

Format: `path | LOC | what | risk | how-to-test`

### Mainline delta-mem (`delta-Mem/` submodule)

| Path | LOC | What | Risk | How to test |
|---|---|---|---|---|
| `deltamem/core/backbone_compat.py` | +30 (new symbols) | `try-import` Gemma3nTextAttention; expose `HAS_GEMMA3N`; re-export rotary + eager-attn helpers; no-op `ensure_attention_compat_views` for Gemma3n | LOW | Import in `run/gemma3n_smoke.py` |
| `deltamem/core/delta.py` | +12 (modified) | Extend isinstance tuple in `attach_delta_mem`; add to `DeltaMemAttention` type union | LOW | Smoke wrap test |
| `deltamem/core/delta_impl.py` | +75 (modified) | `is_gemma3n_attention` branch in `__init__`; Gemma3n rotary branch in `_apply_standard_rotary`; `shared_kv_states` kwarg forwarding in `forward`; two new config fields (`gemma3n_skip_sliding`, `gemma3n_skip_kv_shared`) | **MEDIUM** — `_apply_standard_rotary` is hot path; bug here corrupts every forward | Compare wrapped vs unwrapped logits on identical input with zero-init delta-mem (must be within atol=1e-3) |
| `deltamem/train/delta_sft.py` | +80 (modified) | `--base-model` path for `google/gemma-3n-E4B-it`; AltUp/Laurel freeze pattern; multimodal special-token mask | **MEDIUM** — wrong freeze list silently leaves AltUp params trainable, wastes ~50 MB grad memory and corrupts the base model | Unit test that asserts `freeze_non_delta_mem_params(gemma3n_model)` returns ≤ 200 trainable parameter tensors and they all match `delta_*\|memory_*\|beta_*\|lambda_*` |

### OSCAR submodule (`third_party/oscar-transformers/`)

| Path | LOC | What | Risk | How to test |
|---|---|---|---|---|
| `oscar_transformers/rotation.py` | +80 (modified) | Add Gemma3n class detection in `_build_patched_forward`; layer-type-aware rotation pair lookup; respect Gemma3n's dual RoPE call | **HIGH** — wrong rotation pair on a sliding vs full layer creates undetectable quality drift, not a crash | Per-layer logit diff test: for every wrapped layer, the bf16→INT2-rotated→un-rotated output must match the bf16 reference output within atol=1e-2 |

### Top-level repo (`E:/Development/delta-mem-tests/`)

| Path | LOC | What | Risk | How to test |
|---|---|---|---|---|
| `run/gemma3n_smoke.py` | +120 (NEW) | Import + load + attach test, mirrors `smollm3_smoke.py` | LOW | Run on dev host CPU |
| `run/oscar_dump_qkv.py` | +50 (modified) | Generalize attention-class detection; split dump by layer-type for Gemma3n | LOW | Run with `--backbone gemma3n` and inspect output dir layout |
| `run/oscar_calibrate_gemma3n.py` | +150 (NEW) | End-to-end calibration orchestration | LOW | Run on 16+ GB box; assert two rotation pairs differ |
| `run/_chunked_eval_runner.py` | +60 (modified) | `--quantize-backbone-int4` and `--ple-offload` flags | **MEDIUM** — quantization + PLE offload interaction is untested | `run/gemma3n_memory_probe.py` |
| `run/locomo_eval.py` | +15 (modified) | Pass-through for the two new flags | LOW | Smoke run with the flags |
| `run/gemma3n_memory_probe.py` | +150 (NEW) | 4 k / 16 k / 32 k peak VRAM budget run | LOW | Run on 12 GB host |
| `run/training_smoke.py` | +30 (modified) | `--backbone gemma3n` flag | LOW | CI |
| `gemma3n/__init__.py` | +5 (NEW) | Package marker | LOW | n/a |
| `gemma3n/train_phase1.py` | +200 (NEW) | Adapted from `strix/train_phase1.py` | **MEDIUM** — divergence from Strix path is a maintenance liability | First training step's loss matches a hand-computed expected value (anchored unit test) |
| `gemma3n/verify_checkpoint.py` | +150 (NEW) | Adapted from `strix/verify_checkpoint.py` with relaxed thresholds | LOW | Run after first checkpoint exists |
| `gemma3n/README.md` | +50 (NEW) | How-to-run docs | n/a | n/a |
| `tests/test_gemma3n_attach.py` | +150 (NEW) | Unit tests for wrap filter logic | LOW | `pytest tests/test_gemma3n_attach.py` |

### Data assets (not code)

| Path | What |
|---|---|
| `data/oscar/rotations/gemma3n_gpqa/k_rotation_full_*.pt` | ~5 MB |
| `data/oscar/rotations/gemma3n_gpqa/v_rotation_full_*.pt` | ~5 MB |
| `data/oscar/rotations/gemma3n_gpqa/k_rotation_sliding_*.pt` | ~5 MB |
| `data/oscar/rotations/gemma3n_gpqa/v_rotation_sliding_*.pt` | ~5 MB |
| `data/oscar/qkv_dumps/gemma3n_gpqa_full/` | ~2 GB transient |
| `data/oscar/qkv_dumps/gemma3n_gpqa_sliding/` | ~2 GB transient |
| `checkpoints/deltamem_gemma3n_e4b_p1_final/` | final adapter (~200 MB) |

### Totals
- **New LOC (ours):** ~1,005
- **Modified LOC (ours):** ~95
- **Modified LOC (submodules):** ~155 (delta-Mem ~75, OSCAR ~80)
- **Total LOC touched:** **~1,255**

The feasibility doc estimated 780-1490 LOC — this plan lands at 1,255,
inside the higher half of the range. The increase relative to the
feasibility doc's *lower* estimate is from explicit test files and the
new `gemma3n/` Strix-equivalent directory (the feasibility doc was
inventory-only, this is execution).

---

## 4. Risk register

Severity × Likelihood = 1-25; sorted desc by product.

### R1 — NF4 + OSCAR INT2 KV double-quant on a model never measured at this stack (S=5, L=4, **=20**)
**Description:** Gemma 3n bf16 weights don't fit on 12 GB; we are forced
into NF4. OSCAR's rotation calibration uses bf16 attention statistics.
Stacking them is novel. Per the feasibility doc, shared risk with Option 4
but compounded by Gemma 3n's unfamiliar activation distribution.

**Mitigation:**
- Calibrate OSCAR rotations on the **bf16** model first (Phase 3) — requires
  a 16+ GB card for calibration, not just dev work.
- Measure logit-KL between (bf16 + OSCAR) and (NF4 + OSCAR) on a held-out
  set before committing to training. If KL is more than 2× the
  Qwen3-equivalent KL on the same prompts, abort and reconsider.

**Fallback if mitigation fails:** Drop the 12 GB target. The 16+ GB card
becomes a hard prerequisite for inference too, not just calibration. Move
the whole project to cloud A100 deploys. This roughly halves the project's
*usefulness* (the whole point of delta-mem on this hardware is the 12 GB
fit), so failing R1 is close to a project-kill.

### R2 — KV-shared layers (15 of 35) halve delta-mem's effective capacity (S=4, L=5, **=20**)
**Description:** Per the feasibility doc, the 15 KV-shared layers never
compute K/V — delta-mem's "add `delta_k`/`delta_v` to projected states" is
meaningless for them. We resolve this by *not attaching* delta-mem to
KV-shared layers (skip flag in Phase 2). Net result: only 7 wrapped layers
(or 20 if we also wrap sliding non-shared layers), vs Qwen3's 28.

**Mitigation:** Phase 5's quality bars (anchor 1.15, extension 1.10) are
already lowered to account for this. The risk is they're *still* too high
because 7 layers genuinely cannot match 28 layers of capacity.

**Fallback:** Attach to sliding-window layers as well (set
`gemma3n_skip_sliding=False`). This breaks the assumption that delta-mem
adds context *beyond* the local window — instead delta-mem is adding
*global* context to layers whose base computation only sees 512 tokens.
That's an untrained regime; behaviour is unknown. Could go either way.
A short hyperparameter sweep (attach 7 / 20 / 27 layers, train a tiny
1 k-step adapter on each, look at loss curves) would resolve in 2 days
of GPU time.

### R3 — OSCAR rotation per-layer-type fitting (S=4, L=4, **=16**)
**Description:** Two RoPE frequencies (1e6 global, 1e4 sliding) means the
K-distributions for the two layer types live in different "rotational
phases". Pooling them into one rotation pair compromises quality on both;
separating them (Phase 3) is the right answer but is **untested by us** —
all our prior OSCAR work was single-rotation-per-model.

**Mitigation:** Phase 3 acceptance criterion 3 ("two rotation pairs
differ measurably") is the canary. If they don't differ, pooling would have
worked and we waste a small amount of complexity. If they differ a lot,
the separate calibration was correct and we have proof.

**Fallback:** If the separate calibration introduces other bugs (e.g. our
rotation dispatcher in `rotation.py` mis-routes a layer), fall back to
single-pooled rotations and accept a quality hit. Measurable in evals.

### R4 — AltUp+Laurel coupling poisons delta-mem corrections (S=4, L=3, **=12**)
**Description:** Feasibility doc R4. Delta-mem writes to *one* of AltUp's
4 parallel residual streams; the AltUp combiner mixes our correction with
the other 3 streams. Adapter training has to learn to push corrections
that *survive* this mixing, or learn to operate within the mixed-output
distribution. Neither has been measured.

**Mitigation:**
- Phase 5's smoke training (100 steps) catches catastrophic divergence
  (NaN, exploding loss). The full training run reveals whether useful
  signal exists.
- Long-term: a delta-mem variant that writes to ALL 4 AltUp streams
  (4× parameter cost) might recover quality; out of scope for this plan
  but flagged as Phase 6 future-work.

**Fallback:** If after a full Phase 1 training run the verify_checkpoint
extension scenario shows ratio < 0.95 (delta-mem is actively *hurting*),
the result is "Gemma 3n + delta-mem is fundamentally incompatible". Walk
away with the negative-result writeup. ~$400-600 wasted; not catastrophic.

### R5 — PLE PCIe latency dominates inference (S=3, L=4, **=12**)
**Description:** PLE offload to CPU saves 4.7 GB VRAM but costs PCIe Gen3
~10-30 ms per layer per token. 35 layers × ~20 ms = ~700 ms of PCIe per
token, on top of compute. At 32 k context and batch=1 we're looking at
~7-23 minutes for prompt prefill PLE alone — likely intolerable.

**Mitigation:**
- Measure in Phase 4 with `run/gemma3n_memory_probe.py`. If prefill is
  > 5 min at 4 k context, PLE-offload is non-viable; we need PLE on GPU.
- PLE on GPU adds 4.7 GB to the budget; with NF4 weights (3.6 B params
  × 0.5 byte ≈ 1.8 GB) + PLE (4.7 GB) + OSCAR INT2 KV at 32 k (2.6 GB) +
  activations (2.5 GB) = **~11.6 GB** — just barely fits, no
  margin. Real-world OOM likely.

**Fallback:** INT4-quantize the PLE table too. Not in any published
recipe; we'd be the first. If the PLE INT4 quality cost is small
(< 5 % MMLU), we have ~3.5 GB of headroom back. If it's large, project
returns to "needs 16 GB minimum" — same as R1 fallback.

### R6 — Gemma 4 release makes Gemma 3n obsolete mid-build (S=3, L=3, **=9**)
**Description:** Already-released. By the time someone picks up this plan,
Gemma 5 may also have shipped. Building for a yesterday-model.

**Mitigation:** Plan is written to be ~90 % reusable for Gemma 4 (see
Pre-flight). The fork point is Phase 5's adapter training (different
context lengths, no AltUp/Laurel awareness needed).

**Fallback:** Pivot to Gemma 4 at any phase boundary. Phase 1 work is
~50 % reusable, Phase 2-4 ~80 %, Phase 5 ~30 %.

### R7 — Gated weights + multimodal weight bloat (S=2, L=4, **=8**)
**Description:** Gemma terms-of-use gating. Multimodal tower adds 1.4 GB
we don't use for text adapter training.

**Mitigation:** Use `unsloth/gemma-3n-E4B-it` ungated mirror (same weights;
already published as such on HF). Strip vision/audio towers at load via
`del model.vision_tower; del model.audio_tower; torch.cuda.empty_cache()`
*after* loading the multimodal-aware checkpoint.

**Fallback:** Stop publishing the adapter. Use internally only.

### R8 — Adapter checkpoint format compatibility with Strix tooling (S=2, L=3, **=6**)
**Description:** Strix scripts assume Qwen3 module names. The new
`gemma3n/` dir solves this for Gemma 3n specifically, but a shared
checkpoint-loader bug between Strix's expectations and Gemma 3n's reality
could surface late.

**Mitigation:** Phase 5 verify_checkpoint script is a near-clone of Strix
verify_checkpoint; share the load helper to catch divergence early.

**Fallback:** Diverge the loaders; document the difference.

### Risk-register summary
- **2 risks at S×L = 20** (NF4+OSCAR compounding, KV-shared capacity loss)
- **1 risk at 16** (per-layer-type rotation fitting)
- **2 risks at 12** (AltUp coupling, PLE PCIe latency)
- **3 risks at ≤ 9** (Gemma 4 obsolescence, gating, checkpoint format)

**Highest-risk phase: Phase 4** — it surfaces R1, R5, and the practical
fit on the dev host all at once. The project's "go/no-go" pivot point.

---

## 5. Testing strategy

### Unit tests (Gemma-3n-specific aspects)
- `tests/test_gemma3n_attach.py` — wrap filter logic (which layers get
  wrapped under various config combos); 7 test cases.
- `tests/test_gemma3n_rotary_branch.py` (~80 LOC NEW) — `_apply_standard_rotary`
  on a Gemma3n attention vs the upstream `gemma3n_apply_rotary_pos_emb` —
  must match within atol=1e-5 (numerical identity).
- `tests/test_gemma3n_shared_kv_forward.py` (~100 LOC NEW) — `forward(...)`
  on a KV-shared layer that **isn't** wrapped by delta-mem still uses
  `shared_kv_states[kv_shared_layer_index]` correctly. Asserted by
  instrumenting the base attention.
- `tests/test_gemma3n_freeze_pattern.py` (~50 LOC NEW) — assert that
  `freeze_non_delta_mem_params` leaves AltUp/Laurel/PLE frozen.

### Smoke tests (cheap; run on dev host)
- `run/gemma3n_smoke.py` — mirrors `run/smollm3_smoke.py`. Import + load
  (skippable) + wrap + freeze test.
- `run/oscar_smoke.py --backbone gemma3n` — extended smoke that runs a
  4 k prompt through OSCAR INT2 + Gemma 3n + delta-mem zero-init and
  asserts logit KL ≤ threshold vs bf16 baseline.
- `run/gemma3n_memory_probe.py` — Phase 4 VRAM ceiling.

### Quality eval gates (mirror `strix/verify_checkpoint.py`)
`gemma3n/verify_checkpoint.py` — three scenarios:
- **Anchor** (conv-26 / 10 q at ~17 k): ratio ≥ **1.15** (vs Strix's 1.25)
- **Extension** (conv-41 / 10 q at ~25 k): ratio ≥ **1.10** (vs Strix's 1.20)
- **Stretch** (conv-26 × 2 at ~32 k): ratio ≥ **1.05** (vs Strix's 1.10)

Lower thresholds because of the 7-attached-layer ceiling (R2). If the
extension scenario shows ≥ 1.20, that's a *positive surprise* worth
investigating (does delta-mem on 7 full-attention layers somehow leverage
Gemma 3n's native 32 k context better than expected?).

### Multimodal regression tests
- `tests/test_gemma3n_multimodal_smoke.py` (~80 LOC NEW) — load Gemma 3n
  *with* vision tower intact; pass a small embedded image through; assert
  text output is non-degenerate ("the image shows" or similar). Confirms
  our `freeze_non_delta_mem_params` and `attach_delta_mem` changes don't
  inadvertently break the vision path.
- Audio path: lower priority (rarely exercised); a single CI smoke that
  audio tokens still produce text would be enough.

### CI integration
Add a `gemma3n-ci.yml` job to the existing test matrix:
- CPU-only smoke (no model load): always run.
- 24 GB GPU smoke (full Gemma 3n load + Phase 1-3 functional tests): nightly.

---

## 6. Estimated total

### LOC
- New (ours): **~1,005**
- Modified (ours): **~95**
- Modified (submodules — delta-Mem + OSCAR): **~155**
- **Total: ~1,255 LOC**

### Engineer-time
- Phase 1: 3 days
- Phase 2: 4 days
- Phase 3: 4 days
- Phase 4: 5 days
- Phase 5: 8-12 days
- **Total: 24-28 engineer-days, ≈ 4 weeks for one engineer**

Confidence interval (assuming a senior IC familiar with delta-mem and
HF transformers):
- 50 % chance: 4 weeks
- 80 % chance: 5 weeks
- 95 % chance: 7 weeks (slips due to PLE-offload debugging, AltUp
  coupling investigation, or NF4+OSCAR calibration loop)

### GPU-hours and cost
| Task | GPU | Hours | Cost (rented) |
|---|---|---|---|
| Phase 3 calibration | 1× A6000 48 GB | ~6 | ~$5 |
| Phase 4 VRAM probe | Local 12 GB | ~4 | $0 |
| Phase 5 smoke training (100 steps) | 1× H100 | ~1 | ~$3 |
| Phase 5 Phase-1 full training | 1× H100 | ~96-144 | **$300-450** |
| Phase 5 evaluation | Local 12 GB | ~20 | $0 |
| Buffer for retries | — | — | **$100-200** |
| **Total cloud** | | | **$400-650** |

Strix Halo alternative for Phase 5 training: ~5-7 wall days at ~$0
ongoing cost (assuming the box is already owned). Adds 2-3 days vs H100
but cheaper.

Kaggle T4×2 is **not viable** for Gemma 3n training — bf16 weights don't
fit on T4 16 GB, and the project's NF4 path is for inference, not the
training loop.

### Total cost summary
- **Engineer-time:** 4 weeks (1 engineer)
- **Cloud cost:** $400-650 OR Strix Halo time (~7 wall days, ~free)
- **Calendar:** 4-5 weeks if no parallel work

---

## 7. What we'd give up by NOT doing this

The do-nothing cost. Concretely:

1. **Multimodal:** No image / audio / video understanding in any
   delta-mem-backed pipeline. We stay text-only on Qwen3 and SmolLM3.
   *Cost: variable — zero if no multimodal product requirement; high if
   it shows up.*

2. **140+ language coverage:** Gemma 3n was trained on 140+ languages
   (HF model card). Qwen3 is strong on Chinese + English; SmolLM3 is
   English-focused. *Cost: zero for our current LoCoMo / English-only
   use; non-zero for any internationalization push.*

3. **MatFormer / Mix-and-Match sub-model selection:** Gemma 3n's
   nesting trick (E2B inside E4B) gives a "free" size knob. We never
   exploit this for Qwen3 / SmolLM3. *Cost: low — we already pick
   model sizes explicitly.*

4. **First-mover credibility:** No-one has published a delta-mem
   adapter for Gemma 3 family (`hub_repo_search query="delta-mem
   gemma"` returns zero). Publishing one would be the first.
   *Cost: variable. If we want a community / research-credibility win,
   this is real. For internal use, zero.*

5. **An on-paper-correct architecture-research portfolio:** Some
   reviewers / collaborators expect any "memory adapter" project to
   eventually support the dominant open-weight family of its era.
   Gemma 3n is one of those (and Gemma 4 even more so).

**NOT-doing cost is dominated by item 1 (multimodal).** Items 2-5 are
"nice to have" outside of a specific external requirement.

---

## 8. What we'd give up by doing this

The opportunity cost. 4 engineer-weeks + $400-650 means:

1. **The same 4 weeks could go to:**
   - **Option 1 + Option 4 in production** (`LONG_CONTEXT_PLAN.md`):
     ship the longer-context Qwen3 adapter + NF4 weight quant. Likely
     the single highest-leverage thing we could do for the local 12 GB
     box. *Cost of skipping: a measurable quality regression on
     long-context evals we're already running.*
   - **EpiCache or LoCoCo integration** (see §9): potentially Gemma3n
     long-context compatibility *and* generalisation to any HF model,
     for much less work. Note: parallel-running "Gemma-alternatives"
     agent is producing depth on this — defer to that output.
   - **64 k context push on SmolLM3** (Option 2 in `LONG_CONTEXT_PLAN.md`):
     proven hardware fit, smaller adapter retrain.

2. **Risk of partial success: high.** Phase 5 verify_checkpoint failing
   at ratio < 1.0 is a real possibility (R2 + R4 compounded). After
   3-4 weeks and $500 we have either:
   - A working Gemma 3n adapter (big win), OR
   - A negative-result writeup ("Gemma 3n + delta-mem isn't useful at
     12 GB"). Negative results have *some* publication value but not
     enough to justify the spend.

3. **Branch-rot on the SmolLM3 / Qwen3 production paths.** Every week
   on Gemma 3n is a week the SmolLM3 calibration sits stale, the
   Qwen3 adapter doesn't get retrained, and the evaluation infra
   doesn't accumulate runs.

4. **Maintenance debt forever.** Adding a third backbone means every
   future change to `delta_impl.py` must be tested on three models, not
   two. Real recurring cost.

5. **Gemma 4 supersession risk** (R6). If we ship Gemma 3n support and
   the community moves to Gemma 4 within 6 months, ~50 % of the work
   needs porting, but the "first-mover credibility" item (§7.4) evaporates.

**Doing-it cost is dominated by items 1 and 2.** The opportunity cost
is concrete; the partial-success risk is moderate-to-high.

---

## 9. Alternative cheaper paths

The parallel-running "Gemma-alternatives" agent is producing depth on
EpiCache and LoCoCo. Keeping this section brief and pointing there.

### A — EpiCache ([ArXiv 2509.17396](https://arxiv.org/html/2509.17396))
Episodic KV cache management. Drop-in via cache-layer monkey-patch.
Likely Gemma-3-compatible without modeling-code changes.
**Sketch:** estimate ~3-5 days to wire (vs this plan's 4 weeks); covers
the long-context goal but not the multimodal goal. Worth a smoke test
before committing to anything bigger. See parallel agent output for
detailed integration steps.

### B — LoCoCo ([ArXiv 2406.05317](https://arxiv.org/pdf/2406.05317))
Convolutional KV compression, "architecture-agnostic". Pure inference
trick.
**Sketch:** ~2-4 days to integrate. Less expressive than delta-mem for
multi-session memory; better than nothing for raw long context. See
parallel agent output.

### C — Pure native Gemma 3n (no delta-mem; just use the 32 k window)
**Sketch:** ~0-1 days. Load Gemma 3n with `transformers` standard pipeline
+ NF4 quant + standard KV cache; use the 32 k native window directly.
What we give up: delta-mem's memory compression and multi-session memory.
What we gain: zero engineering cost; multimodal works out of the box.
Reasonable answer if the question is "we just want a multimodal
long-context box, who cares about delta-mem".

### D — Wait for Gemma 4 and target that instead
**Sketch:** This plan minus AltUp/Laurel work plus dual-RoPE work plus
4 k native context. Net: ~3 weeks instead of 4. Plus we get 128 k native
context, which arguably removes the need for delta-mem on Gemma 4 at all
(see option C applied to Gemma 4).

---

## 10. Recommendation

**SKIP — permanently — in favour of Option C applied to Gemma 4 (no
delta-mem, use native 128 k context directly).**

The Gemma 4 release that landed in April 2026 fundamentally undermines
the case for adding Gemma 3n to delta-mem. Gemma 4 E4B has:
- **128 k native context** (vs 32 k on Gemma 3n) — removes the headline
  motivation for putting Gemma in delta-mem at all, which was "extend a
  short-context model".
- **No AltUp, no Laurel** — eliminates feasibility risk R4 entirely.
- **Same shared-KV, PLE, hybrid-attention story** — the genuinely
  novel/risky bits we'd need to solve are *also* present in Gemma 4,
  meaning Gemma 3n is not even a useful stepping stone toward Gemma 4
  (it doesn't simplify the hard parts).

Spending 4 engineer-weeks + $500 to build delta-mem support for a
yesterday-model whose successor already has 4× the native context is
the wrong move. If a concrete multimodal long-context requirement
materializes (the only "must build" trigger from §1), the right answer
is **Gemma 4 + native 128 k**, not Gemma 3n + delta-mem. If 128 k still
isn't enough on Gemma 4 for some workload that materializes, *then* and
only then re-open this plan and re-target it at Gemma 4 (the plan is
~90 % portable per the Pre-flight section).

**Concrete trigger to revisit this plan:** a specific multimodal
long-context workload appears that needs > 128 k useful context AND
EpiCache/LoCoCo don't solve it on Gemma 4. Until then, this plan sits.

---

## Cross-references
- Feasibility analysis: `.planning/research/gemma3n-deltamem-feasibility.md`
- Higher-level options summary: `LONG_CONTEXT_PLAN.md` Option 3
- Existing backbone integration template (SmolLM3):
  `delta-Mem/deltamem/core/backbone_compat.py`
- Existing wrapper: `delta-Mem/deltamem/core/delta.py`,
  `delta-Mem/deltamem/core/delta_impl.py:500-700`
- OSCAR rotation patching: `third_party/oscar-transformers/oscar_transformers/rotation.py:131-200`
- Strix training/verification mirror: `strix/train_phase1.py`,
  `strix/verify_checkpoint.py`
- SmolLM3 smoke template: `run/smollm3_smoke.py`

## External sources
- [unsloth/gemma-3n-E4B-it config.json](https://huggingface.co/unsloth/gemma-3n-E4B-it/raw/main/config.json) — authoritative architecture config
- [google/gemma-3n-E4B-it model card](https://huggingface.co/google/gemma-3n-E4B-it) — official 8 B / 4 B-effective claim, 32 k native context
- [google/gemma-4-E4B-it model card](https://huggingface.co/google/gemma-4-E4B-it) — 128 k context, 42 layers, dropped AltUp, kept PLE
- [HuggingFace blog — Welcome Gemma 4](https://huggingface.co/blog/gemma4) — explicit "leaves out complex or inconclusive features such as Altup"
- [botmonster.com — Gemma 4 Architecture Explained](https://botmonster.com/posts/gemma-4-architecture-per-layer-embeddings-shared-kv-cache-dual-rope/) — per-layer-embeddings, shared KV cache, dual RoPE details
- [transformers — modular_gemma3n.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3n/modular_gemma3n.py) — `Gemma3nTextAttention` source: projections, q/k/v norms, `is_kv_shared_layer`, `shared_kv_states`, rotary
- [EpiCache (ArXiv 2509.17396)](https://arxiv.org/html/2509.17396) — alternative cache-eviction long-context
- [LoCoCo (ArXiv 2406.05317)](https://arxiv.org/pdf/2406.05317) — alternative convolutional KV compression
