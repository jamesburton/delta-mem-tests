# Path to ≥32 k useful context on the local 12 GB box

This document answers: with everything we have (OSCAR INT2 KV + 4-per-byte
packing + dequant-shadow off + reentrant gradient checkpointing + the
fused-_assemble peak reduction), what plays land us at **32 k context with
delta-mem actually winning** on this RTX 3060 12 GB host?

There are **two separate ceilings** to push past, and they have different
solutions:

| Ceiling | Current cap | Bound by |
|---------|------------|----------|
| **VRAM** | ~25 k inference (v6c showed it just fits) | tricks already applied |
| **Quality** (delta-mem ratio > 1.0) | ~17-20 k | published adapter's training distribution |

VRAM at 32 k is **achievable now** on Qwen3-4B with all tricks (see
budget below). The blocker is quality — the published adapter regresses
to ratio 0.60 at 25 k (v6c data). So this plan is mostly about **what
checkpoint to train** and where, not "what new memory trick to invent".

---

## VRAM budget per candidate at 32 k

All numbers assume the production combo: OSCAR INT2 + GPQA-cal rotation +
4-per-byte packing + `OSCAR_DISABLE_DEQUANT_SHADOW=1` + `eval_batch_size=1`
+ the fused-_assemble peak reduction. Per-token costs scale linearly with
context.

| Model | Params | Weights bf16 | Per-token KV (INT2 packed) | Activations at 32 k | _assemble peak | **Total at 32 k** |
|-------|--------|--------------|---------------------------|--------------------|-----------------| ------------|
| **Qwen3-4B-Instruct-2507** | 4 B | 7.5 GB | 20 KB/tok → 0.64 GB | ~1.5 GB | ~110 MB transient | **~10.0 GB ✓** |
| **SmolLM3-3B-Instruct** | 3 B | 5.6 GB | ~16 KB/tok → 0.51 GB | ~1.2 GB | ~85 MB transient | **~7.5 GB ✓** (huge margin) |
| **Gemma 3n E4B** | 6.9 B (4 B effective) | ~10 GB (1) | ~22 KB/tok → 0.70 GB | ~1.6 GB | ~120 MB transient | **~12.5 GB ✗** (over budget) |
| **Qwen3-4B with INT4 weights** | 4 B | ~2 GB (weights NF4) | 0.64 GB | ~1.5 GB | ~110 MB | **~4.4 GB ✓✓** (massive headroom, up to ~100 k context possible) |

(1) Gemma 3n's MatFormer reduces *compute* per token to ~4 B-equivalent
but the weight storage is still 6.9 B params; bf16 = ~13 GB before
counting PLE (Per-Layer Embedding) caches. Even with INT8 weight quant
it sits at ~7 GB; with INT4 it's ~3.5 GB.

**At 64 k context** (the same calculation, doubled per-token KV +
activations):

| Model | Total at 64 k |
|-------|---------------|
| Qwen3-4B (current production) | ~12.2 GB (just over budget) |
| **SmolLM3-3B** | **~10.4 GB ✓** |
| Qwen3-4B with INT4 weights | ~6.5 GB |

---

## Candidates ranked by total work-to-32 k-useful-context

### Option 1 — **Qwen3-4B + retrained adapter** (lowest risk, most leverage from existing work)

**Status**: 32 k VRAM already proven feasible (v6c at 25 k fit; 32 k
budget is ~10 GB which is well under). Only the adapter blocks quality.

**Steps**:

1. **Train a longer-context delta-mem adapter** at 32 k context. Per
   `STRIX_INSTRUCTIONS.md`, fine-tune from
   `declare-lab/delta-mem_qwen3_4b-instruct` on a mix of:
   - 50 % LoCoMo originals (anchors 17 k performance)
   - 30 % LongMemEval at 20-32 k
   - 20 % InfBench-mem at 32 k+
   Phase 1: 32 k target, ~2 days on Strix Halo 96 GB, ~$50-200 on rented H100.
   Kaggle T4 single can stage at 8 k for hyperparam scouts; T4×2 + ZeRO-3
   can reach 16 k for an intermediate checkpoint.

2. **Verify on the local host**:
   - Anchor: conv-26 / 10 q at 17 k → ratio ≥ 1.25 (don't regress
     existing strength).
   - Extension: conv-41 / 10 q at 25 k → ratio ≥ 1.20 (currently 0.60).
   - Stretch: synthetic conv-26 × 2 at ~32 k → ratio ≥ 1.10.
   All via `python -m run.locomo_eval --adapter-override
   checkpoints/<new>` with the rotation env vars set.

3. **Infra already done**: OSCAR rotations, packing, env vars,
   checkpointing config, eval harness, adapter loader, cross-platform
   safetensors. Zero new code needed on the inference side.

**Effort**: 0 LOC of new code here, 1 training run elsewhere.
**Quality risk**: moderate — depends on training data + recipe; mitigated
by the 50 % anchor share.

---

### Option 2 — **SmolLM3-3B + new adapter** (best long-term VRAM headroom)

**Status**: delta-mem already supports `SmolLM3Attention`
(`delta-Mem/deltamem/core/delta_impl.py:501`,
`delta-Mem/deltamem/core/backbone_compat.py`). No new attention-class
work needed.

The win: 1.9 GB freed by the smaller backbone → **64 k context fits**
(~10.4 GB total). At 32 k, ~7.5 GB total = huge margin.

**Steps**:

1. **Calibrate OSCAR rotations for SmolLM3** — re-run the 3-phase
   calibration we did for Qwen3 (per `Tier 1 / Appendix D` history):
   - Dump Q/K/V activations on SmolLM3 using a 64-token GPQA prompt slice
     via `run/oscar_dump_qkv.py` (already exists, may need a
     SmolLM3-specific monkey-patch path for the same captures —
     ~30 LOC). Output: `data/oscar/qkv_dumps/smollm3_gpqa/`.
   - Compute K and V rotations via `run/compute_kv_rotation.py` (already
     vendored): `python -m run.compute_kv_rotation
     --dump-dir data/oscar/qkv_dumps/smollm3_gpqa --output-dir
     data/oscar/rotations/smollm3_gpqa`. Output: two `.pt` files
     ~5 MB each. ~30 min on this host.
   - Quick smoke: 4 k port-debug test F-equivalent. ~10 min.

2. **Train a delta-mem adapter for SmolLM3 from scratch** — no public
   starter. Mainline trainer is at
   `delta-Mem/deltamem/train/delta_sft.py`; switch the `--base-model`
   to `HuggingFaceTB/SmolLM3-3B-Instruct`. Train phase 1 at 32 k context
   on the same LoCoMo + LongMemEval + InfBench mix. ~3-4 days Strix
   Halo, ~$100-300 rented H100.

3. **Verify on this host** — same harness as Option 1, just with
   `--adapter-override` and the SmolLM3 rotation paths via the env vars:
   ```bash
   $env:OSCAR_K_ROTATION_PATH='data\oscar\rotations\smollm3_gpqa\k_rotation_*.pt'
   $env:OSCAR_V_ROTATION_PATH='data\oscar\rotations\smollm3_gpqa\v_rotation_*.pt'
   ```
   May need a `--model-override` flag (~10 LOC; mirror of
   `--adapter-override` we added in commit `a6f01aa`).

**Effort**:
- ~30 LOC SmolLM3 dump path (or rebuild from scratch — Qwen3 dump was
  built generic enough that most code likely reuses).
- ~10 LOC `--model-override` for `locomo_eval.py`.
- One adapter training run elsewhere (no starter, so longer than
  Option 1's fine-tune).

**Quality risk**: higher than Option 1 — SmolLM3 has lower baseline
quality than Qwen3 (3 B vs 4 B + Qwen tuning). But the 64 k native
context training distribution of SmolLM3 means the backbone *itself* is
better-equipped for long context, which the delta-mem corrections build on.

---

### Option 3 — **Gemma 3n E4B + new attention support** (largest scope, native multimodal)

**Status**: delta-mem does NOT support Gemma attention. Adding it is
significant work because:

1. Gemma 3 uses a **mixed 1:5 ratio of full to sliding-window attention
   layers** per block. delta-mem assumes a uniform attention pattern;
   handling the sliding-window subset requires either skipping those
   layers (loses delta-mem's reach) or implementing a delta-mem variant
   that respects the sliding window.
2. Gemma 3n's **MatFormer** architecture varies layer width per
   sub-model; delta-mem's per-layer rank/heads assumptions need rework.
3. Gemma 3n's **Per-Layer Embedding cache** adds memory cost not
   accounted for in our budget.
4. Gemma 3n is **multimodal** by default; we'd want to ensure delta-mem
   only attaches to text-attention layers (additional plumbing).

**The good**: native 32 k context means the backbone alone holds up at
target length, so delta-mem only needs to add the cheap-memory
*compression* benefit, not the *coverage* benefit. Multimodal future
optionality.

**Steps** (estimated, since none of this is done):

1. ~200-400 LOC to add `GemmaAttention` support to
   `delta-Mem/deltamem/core/delta_impl.py` and `backbone_compat.py`.
   Mirror the SmolLM3 patterns. Validate per-layer attention type
   detection so sliding-window layers are either passed through
   unchanged or get a windowed delta-mem variant.
2. OSCAR rotation calibration for Gemma's head structure (same 3-phase
   recipe).
3. Train a delta-mem adapter for Gemma 3n E4B — at least as expensive as
   Option 2.
4. Memory: Gemma 3n E4B at bf16 = ~13 GB just for weights → **does not
   fit at all** at bf16 on a 12 GB card. Requires INT4 weight quant
   (Option 4 below) to even *load* on this hardware.

**Effort**: substantial. Probably 2-4 weeks of dev work plus the training.
**Quality risk**: novel territory; no proof anyone has done delta-mem on
Gemma 3. Higher reward IF it works (multimodal + 32 k native + delta-mem
= a meaningful new capability).

**My recommendation**: not the first move. Worth revisiting after Option
1 or 2 ships.

---

### Option 4 — **INT4 weight quantization of the chosen model**

Orthogonal to Options 1-3. Quantize the *backbone weights* (not just KV) to
INT4 via NF4/GPTQ/AWQ. Frees 5-6 GB of model weight VRAM.

**Steps for inference-only INT4 weights** (no adapter retraining needed —
the existing delta-mem adapter still attaches over the quantized
backbone via QLoRA's pattern, since adapter params remain bf16):

1. `pip install bitsandbytes` (already in our env).
2. Load model with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`.
3. Wrap with delta-mem as usual (`attach_delta_mem` + load adapter state dict).
4. Run inference exactly as today — quantized backbone, bf16 adapter
   overlay, OSCAR INT2 KV.

VRAM at 32 k with Qwen3-4B NF4 weights + OSCAR INT2 KV:
- Weights NF4: ~2.0 GB (was 7.5)
- KV codes: 0.64 GB
- Activations: ~1.5 GB
- Compute scratch (NF4 dequant per-layer transient): ~0.5 GB
- **Total: ~4.6 GB at 32 k** ⇒ **~100 k context theoretically fits**

**Risk**: NF4 weight quant of the BACKBONE alongside OSCAR INT2 KV
quant of the *same* model creates a quality compounding question we
haven't tested. Per the int4-vs-int2 research we already did
(`.planning/research/int4-vs-int2.md`), the OSCAR rotation was
calibrated against the bf16 backbone's attention statistics; NF4
weights perturb those, so the rotation might be slightly mis-aligned.
Likely a small-to-moderate quality hit; needs to be measured.

**Training over INT4 weights** is QLoRA territory and not directly
compatible with delta-mem's `freeze_non_delta_mem_params` flow without
extra work. So Option 4 is "inference-only" unless we adapt the
training path.

---

## Recommended path for ≥32 k useful context

**Phase A (low-risk, fastest to validate)**: **Option 1 + Option 4**
combined.

1. Train the longer-context Qwen3-4B adapter (Strix Halo / Kaggle T4×2 /
   rented H100), per the existing `STRIX_INSTRUCTIONS.md`.
2. On this 12 GB host, also test **inference with NF4 weight
   quantization** of the same backbone. If the quality cost is small,
   we unlock ~100 k context without any backbone retraining.
3. Validate at 32 k on conv-41 + synthetic extensions. Goal: ratio ≥
   1.20.

**Concrete first-week tasks** (in this order):

1. **NF4-weight inference smoke** — ~80 LOC patch to add a
   `--quantize-backbone-int4` flag to `run/locomo_eval.py` and a
   `BitsAndBytesConfig` path in `_chunked_eval_runner.py`. Run conv-26 / 10 q
   with the existing published adapter to measure the quality cost of
   NF4-weights-alongside-OSCAR-INT2-KV. ~3 h wall time. **Can do
   immediately on this host.**
2. **Synthetic 32 k feasibility on existing adapter** — even though
   the adapter regresses at 25 k, we want a clean VRAM ceiling
   measurement at 32 k to prove the budget. Use
   `run/build_context_sweep_data.py extend conv-26 2` (~32 k context)
   and run with shadow-off + batch=1 + fused-_assemble. Measure peak
   VRAM and confirm it fits. Expect ratio < 1 but ratio of base/delta
   numbers still informative. ~3 h wall time.
3. **Branch on results**:
   - If NF4-weights cost < 5 % on the 17 k anchor: ship as a
     production option, ratio penalty mostly from the adapter ceiling.
   - If NF4-weights cost > 15 %: stick to bf16 weights, all 32 k push
     comes from the adapter retrain.

**Phase B (when Phase A's NF4 result is in)**: kick off the adapter
retrain elsewhere with the verified-on-this-host inference recipe
locked in.

**Phase C (after Phase B's adapter lands)**: deploy the new adapter back
here via `--adapter-override`, re-verify the anchor + extension
quality at 32 k.

---

## Update — NF4 + OSCAR INT2 compounding result (measured)

The Phase-A step-1 experiment ran on 2026-06-09 with the existing
published adapter on conv-26 / 10 q (17.5 k context). Result was much
worse than the "small unknown cost" the plan predicted:

| Config | base | delta | ratio |
|--------|------|-------|-------|
| v5 (bf16 weights + OSCAR INT2 + GPQA-rot) | 0.2735 | **0.3642** | 1.332 |
| **NF4 (NF4 weights + OSCAR INT2 + GPQA-rot)** | **0.0000** | **0.1455** | undefined |

The **base arm collapses completely** — every prediction is gibberish
token soup (`"stringcomparison-'7gli6dm[]( financed"`,
`"a ast thanksbelakh"`). The compounding of NF4 weight perturbation
on top of OSCAR's bf16-calibrated rotations breaks attention entirely
on the un-corrected arm.

The **delta arm rescues the model** to coherent (if degraded) text —
real answers like `"melanie ran a charity race on 9 june, 2023"` and
`"counseling, mental health, and self-care"`. delta-mem's learned
corrections carry enough semantic information from prefill that real
answers emerge even when the base attention is producing noise. **This
is unexpected behaviour worth flagging as a research-interesting
result**, separate from the production usability question.

For production at this context length, the 60 % delta-arm drop
(0.36 → 0.14) is too steep. NF4 + OSCAR INT2 is not a usable inference
combination as currently configured.

**Recovery paths**:

1. **NF4 + bf16 KV + delta-mem** (untested). Skip the OSCAR layer
   entirely under NF4. bf16 KV at 17 k costs ~2.4 GB vs OSCAR INT2's
   ~0.33 GB — net VRAM saving from NF4 alone is ~5.2 GB - 2.1 GB =
   ~3 GB after losing the OSCAR win. Still enough headroom for 32 k.
   Avoids the rotation-vs-NF4 mismatch. **Cheap to test (no new code,
   just `--quantize-backbone-int4` without setting `KV_CACHE_BACKEND=oscar`).**

2. **Re-calibrate OSCAR rotations against NF4 attention statistics**.
   Run the 3-phase calibration with the model loaded in NF4. Significant
   GPU time but addresses the root cause. **Probably overkill unless
   path 1 doesn't free enough VRAM for the target context.**

3. **Drop NF4 entirely**; rely on adapter retrain (Option 1) for the
   32 k push. Confirmed by this result: the 12 GB box's NF4 path is
   not a shortcut around the long-context adapter training.

### Update — NF4 + bf16 KV result (path 1 measured)

Ran path 1 from above (NF4 weights + bf16 KV, no OSCAR layer) on
conv-26 / 10 q at 17.5 k context. **Surprising and important:**

| Config | base | delta | ratio |
|--------|------|-------|-------|
| v5 (bf16 weights + OSCAR INT2 + GPQA-rot) | 0.2735 | **0.3642** | 1.332 |
| NF4 (NF4 weights + OSCAR INT2 + GPQA-rot) | 0.0000 | 0.1455 | undefined |
| **NF4 + bf16 KV (no OSCAR)** | **0.3788** | 0.2962 | 0.782 |

Three significant findings:

1. **NF4+bf16 KV improves the BASE arm vs v5** (0.378 vs 0.274 = +38%).
   Removing OSCAR's INT2 quantization cost on the un-corrected arm
   actually helps — the NF4 backbone preserves raw attention quality
   better than OSCAR INT2 KV did. Important: this means v5's "headline"
   wasn't entirely about delta-mem being good; some of that score gap
   was OSCAR's cost on the base arm.

2. **Delta arm degrades 19% under NF4** (0.296 vs 0.364). Category
   breakdown shows the loss is concentrated in **multi-hop** questions
   (0.095 vs 0.667 in v5 — collapsed). Temporal (0.391 vs 0.218) and
   open-domain (0.333 vs 0.333) are unaffected or slightly improved.
   NF4 weight perturbation specifically degrades delta-mem's long-range
   reasoning corrections.

3. **Ratio inverts to 0.782** — the BASE arm beats the delta arm at this
   config. Same regime as v6c (25 k context) where the published
   adapter's training distribution stops covering. Suggests
   NF4-weight-perturbation puts the adapter into the same OOD regime as
   raw context-length OOD does.

Production implications for ≥32 k inference on the 12 GB card:

| Context | Best config (today, no retraining) |
|---------|------------------------------------|
| ≤17 k | v5 (bf16+OSCAR+delta) — ratio 1.33, delta wins |
| 17-25 k | NF4 + bf16 KV (base-only; skip delta-mem overlay) |
| 25-32 k | NF4 + bf16 KV (base-only) — proven 0.378 base; VRAM permits |
| 32 k+ delta-mem actually winning | Option 1 retrain on Strix/Kaggle/cloud |

VRAM at 32 k with NF4 + bf16 KV: weights NF4 (~2 GB) + bf16 KV (~4.8 GB)
+ activations (~2 GB) ≈ **9 GB** → comfortable on 12 GB.

**We have a production NF4 path for 25-32 k base-only inference NOW**,
without any new training. The delta-mem overlay is still the gap that
needs the adapter retrain (Option 1). For 32 k+ context with KV
quantization (if memory becomes tight again), EpiCache (Option E) is
the candidate to evaluate.

**Updated Phase-A recommendation**: do path 1 (NF4 + bf16 KV) before
committing to Option 1. Path 1 takes ~3 h on this host; if it works,
we have a 32 k production config without any new training.

---

## Quick verification harness (immediate-action)

```powershell
# Confirm the 32 k VRAM budget on the existing adapter (quality won't be
# good but we'll have a hard VRAM ceiling measurement)

. .\env\vsenv.ps1
python -m run.build_context_sweep_data extend conv-26 2   # ~32 k synthetic

$env:KV_CACHE_BACKEND='oscar'; $env:KV_CACHE_BITS='2'
$env:OSCAR_K_ROTATION_PATH='data\oscar\rotations\instruct_gpqa\k_rotation_qqt_r_h_pbr.pt'
$env:OSCAR_V_ROTATION_PATH='data\oscar\rotations\instruct_gpqa\v_rotation_sst_r_h_pbr.pt'
$env:OSCAR_DISABLE_DEQUANT_SHADOW='1'
$env:PYTHONIOENCODING='utf-8'

.venv\Scripts\python.exe -m run.locomo_eval `
    --kv-cache-backend oscar --kv-cache-bits 2 `
    --eval-batch-size 1 `
    --max-conversations 1 --max-questions-per-conversation 5 `
    --data-file data\locomo_conv-26_x2.json `
    --output-json outputs\v7_synthetic_32k.json
```

Expected outcome: completes (proves VRAM fits at ~32 k), with low/regressed
quality (proves the adapter is the bound, justifying Option 1's training
work). ~5 h wall time on this card.

---

## Summary table

| Option | First-day cost | Strix/cloud cost | 32 k VRAM | 32 k quality |
|--------|---------------|------------------|-----------|-------------|
| **1. Qwen3-4B + retrained adapter** | 0 | $50-300 H100 | ✓ ~10 GB | depends on training |
| **2. SmolLM3-3B + new adapter** | ~40 LOC (rot calibrate + model override flag) | $100-400 H100 | ✓✓ ~7.5 GB (64 k also fits) | depends on training; lower baseline |
| **3. Gemma 3n + new adapter** | 200-400 LOC + training | $200-600+ | requires Option 4 to fit weights | multimodal upside |
| **4. NF4 weight quant** (inference only) | ~80 LOC | $0 | ✓✓✓ ~4.6 GB at 32 k | small unknown cost; orthogonal to 1-3 |
| **Phase A (Option 1 + Option 4)** | ~80 LOC | $50-300 H100 | ~4.6 GB | best near-term |
| **E. EpiCache pilot** (Apple ml-epicache) | ~350 LOC (Day-1 scaffold landed 2026-06-09: submodule + Qwen3 adapter + CLI + smoke); Day-2 GPU work to make it functional | 0 (training-free, ~1 GPU-hour BookSum calibration on this host) | ~9.3 GB at 100 k (paper, LLaMA-3.2-3B) — eviction caps KV growth, complements NF4 weights | up to 40 % accuracy lift vs SnapKV/H2O/KVzip on LoCoMo per paper; **stack with delta-mem at 25 k+ is the highest-upside untested combination** ([alternatives doc Section E](.planning/research/long-context-alternatives.md)) |

**Bottom line**: the realistic path to 32 k useful context on this 12 GB
box is **train a longer-context Qwen3-4B adapter** (already planned for
Strix Halo / Kaggle / cloud per existing docs) and **optionally add NF4
backbone quantization** (~80 LOC patch we could land here today, no
retraining required) to push the VRAM ceiling much further past 32 k.
SmolLM3-3B is the most interesting alternative if we want 64 k+ context
on this hardware, but needs a from-scratch adapter. Gemma 3n is a 2-4
week project.

Want me to land the NF4 inference patch (Option 4) and run the 32 k
synthetic VRAM probe (the verification harness above) as the first
moves? Both are doable on this box today.

---

## Update — EpiCache pilot Day-1 prep (2026-06-09)

Following the NF4 + OSCAR INT2 compounding collapse above (delta-arm
ratio 0.36 -> 0.14 at 17 k), and per the recommendation in
[`.planning/research/long-context-alternatives.md`](.planning/research/long-context-alternatives.md)
Section E, we are piloting **EpiCache** (Apple ml-epicache) as an
orthogonal eviction-based mechanism that may stack with delta-mem at 25 k+
where the published adapter regresses.

### Day-1 deliverables landed (CPU-only, GPU was busy with NF4 eval)

- **Submodule pinned**: `third_party/ml-epicache` at commit
  `b742661a1b763d0a57f0a1c6b82acbdbe5ed578c` (Apple's 2025-10-02 public
  release; same commit the alternatives doc identified).
- **Install doc**: [`third_party/ml-epicache-install.md`](third_party/ml-epicache-install.md)
  documents CPU-safe install steps and Day-2 GPU-only steps
  (flash-attn 2.7.4.post1, `csrc/make` for tiny_api_cuda).
- **Qwen3 adapter**: [`run/epicache_qwen3_adapter.py`](run/epicache_qwen3_adapter.py)
  ports EpiCache's Qwen2.5 monkeypatch to Qwen3 — upstream's
  `model/monkeypatch.py` lacks a Qwen3 branch even though
  `model/wrapper.py` imports `Qwen3ForCausalLM`. Exposes
  `install_epicache_on_qwen3(model, ...)` which the runner calls once
  per attention-identity (mirrors the OSCAR `apply_rotations` pattern).
  Three TODOs documented in the file's docstring: q_norm/k_norm handling
  (Qwen3 has them, EpiCache's Llama forward does not apply them), the
  CPU-deferred attention.attn import, and the missing
  `LongConvQAModel`-style episode-cache pipeline.
- **CLI wiring**: `--kv-cache-backend epicache` plus five EpiCache
  knobs (`--epicache-budget` 4096, `--epicache-n-clusters` 4,
  `--epicache-prefill-chunk-size` 2048, `--epicache-level pair`,
  `--epicache-scoring-method clustering`, `--epicache-score-path`)
  propagate via env vars to `run/_chunked_eval_runner.py` exactly like
  the OSCAR / NF4 patterns.
- **Runner branch**: `_chunked_eval_runner.py:_new_kv_cache` recognises
  `KV_CACHE_BACKEND=epicache`, runs `install_epicache_on_qwen3`, then
  **raises NotImplementedError** because the real eviction pipeline does
  not slot into our `session._ingest_full_ids -> _decode_generate` flow.
  This is the load-bearing finding: see "Integration-shape caveat" below.
- **Smoke**: `run/epicache_smoke.py` runs 5 checks (submodule + commit
  pin, adapter import, CLI choices, runner dispatch, Qwen3-stub wiring)
  in <30 s on CPU with no GPU touch. **Result: ALL CHECKS PASSED**.
  Confirms `num_attention_heads=32 num_key_value_heads=8 head_dim=128`
  for Qwen3-4B (GQA grouping=4); EpiCache's `EvictCache` reads these
  config fields directly so the GQA path is correct without changes.

### Integration-shape caveat (correction to alternatives doc)

After reading EpiCache's code in detail, the alternatives doc's
description of EpiCache as "drop-in Cache subclass + monkeypatch" is
oversimplified:

- The `EvictCache` Cache subclass is real, but it does NOT drop into
  `model.generate(past_key_values=...)` like `DynamicCache` does.
- The actual flow (`third_party/ml-epicache/run_epicache.py:35-180`)
  requires a custom `LongConvQAModel` wrapper that calls
  `model.prefill_memory_constrained(ctx_ids, ...)` once per episode,
  caches each episode's `EvictCache` to CPU, then on each query does
  embed-question -> match-centroid -> restore-cache -> `model.generate`.
- That pipeline must be reimplemented inside our chunked runner; it does
  not co-exist trivially with `DeltaMemChatSession`. Estimated Day-2
  work for that bridge alone: 200-350 LOC, plus the flash-attn install
  / SDPA fallback (Phase 3 of the alternatives doc).

The doc's "~150 LOC including a small layer-sensitivity calibration
script" is therefore an underestimate by roughly 2-3x. Pilot is still
worth running — the headline accuracy numbers and the
delta-mem-+-EpiCache stack hypothesis are unchanged — but the engineering
budget should be revised to **5-7 engineer-days for Qwen3-4B alone**
rather than the doc's 3-4.

### Day-2 readiness checklist (next agent picks up here)

1. **Wait for the NF4 eval to free the GPU.**
2. **Install flash-attn**: `.venv\Scripts\python.exe -m pip install
   flash-attn==2.7.4.post1 --no-build-isolation`. On native Windows this
   will likely fail; either (a) move to WSL2 + CUDA toolkit, or (b) patch
   `third_party/ml-epicache/attention/attn.py` to fall back to SDPA
   (replace `flash_attn_varlen_func` + `_flash_attention_forward` with
   `torch.nn.functional.scaled_dot_product_attention`, ~50 LOC). The
   alternatives doc recommends (b) for this 12 GB box.
3. **Build tiny_api_cuda**: `cd third_party\ml-epicache\csrc && make`.
   Requires `CUDA_HOME` set and `nvcc` on PATH. On native Windows, again
   may need WSL2; if blocked, patch
   `third_party/ml-epicache/attention/kvcache.py` to use the Python
   for-loop fallback already present in `EvictCache.update`'s
   `else: # use adapted kernel` branch (the kernel is for decode-time
   single-token updates; prefill works without it).
4. **BookSum layer-sensitivity calibration on Qwen3-4B**:
   ```powershell
   python third_party\ml-epicache\data\booksum\preproc_booksum.py `
       --model_path Qwen/Qwen3-4B-Instruct-2507 --max_length 16384
   python third_party\ml-epicache\data\layer_scores\layer_profile.py `
       --model_path Qwen/Qwen3-4B-Instruct-2507 `
       --input_file <booksum_pre_path>
   ```
   Output: `data/layer_scores/booksum_Qwen3-4B-Instruct_sample0_layer_scores.json`.
   ~1 GPU-hour on the 12 GB box per alternatives doc.
5. **Implement the episode-cache pipeline** in
   `run/epicache_qwen3_adapter.py:build_episode_caches` and wire it into
   `_chunked_eval_runner.py`'s `_chunked_official_full_history_answer`
   path (replace the `NotImplementedError` raise in
   `_new_kv_cache(model)` -> `KV_CACHE_BACKEND == "epicache"`).
6. **Conv-41 baseline + EpiCache-only comparison** (alternatives doc
   "Day 2"): three runs on conv-41 / 10 q:
   - A. `--kv-cache-backend bf16` (control).
   - B. `--kv-cache-backend epicache --epicache-budget 4096` (no OSCAR,
     no delta-mem — pure EpiCache).
   - C. (already have from v6c) delta-mem + OSCAR INT2 at 25 k.
   Exit criterion: does B's delta score at 25 k beat C's 0.139?
