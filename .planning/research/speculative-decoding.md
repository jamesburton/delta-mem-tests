# Speculative Decoding for Qwen3-4B-Instruct-2507 + OSCAR INT2 KV on RTX 3060 12 GB

Research date: 2026-06-02. Target: cut LoCoMo conv-0/10q wall time from ~9.5 h on a single 3060 12 GB.
Constraint: keep HF transformers + custom `OSCARCache` (sink + INT2 middle + recent), do not retrain target.

---

## 1. Methods that actually ship in HF transformers today

`generate()` exposes speculative decoding through a single `CandidateGenerator` interface (see `transformers.generation.candidate_generator`). Four concrete generators ship in v5.x:

| Method | API | Pros | Cons / fit for us |
|---|---|---|---|
| **Assisted generation** (Leviathan/Chen-style spec sampling) | `generate(..., assistant_model=draft)` | Lossless under greedy/sampling; works with **any** `Cache` because the draft has its own cache and the target is just called once per round with `K+1` query positions | Best when draft is 10–20x smaller, same tokenizer. Single-sequence only (no batching). This is our baseline option. |
| **Universal assisted decoding (UAD)** | adds `tokenizer=` + `assistant_tokenizer=` | Allows different tokenizers via LCS re-alignment | Slower per-round than same-tokenizer; irrelevant for us since Qwen3-0.6B shares the Qwen3-4B tokenizer. |
| **Prompt lookup decoding** | `prompt_lookup_num_tokens=N` | Zero draft model, free n-gram copy from prompt | Acceptance ~0 on free-form QA over LoCoMo conversations. Worth a 30-min experiment only. |
| **Self-speculative early-exit** | `assistant_early_exit=L` | No second model; shares KV | **Requires LayerSkip-trained checkpoint** (e.g. `facebook/layerskip-llama3.2-1B`). Qwen3-4B-Instruct-2507 is not LayerSkip-trained, so this is out without retraining. |

What is **not** native to HF transformers (despite hype):

- **EAGLE / EAGLE-2 / EAGLE-3** — first-class in vLLM and SGLang only. The published Qwen3 EAGLE3 weights (e.g. `AngelSlim/Qwen3-4B_eagle3`) ship a tiny 218 M "feature predictor" plus a token head that consumes the target's hidden states from layer -2. Plugging this into HF transformers means writing the EAGLE tree-draft loop, attention-mask trick, and feature-extraction hook ourselves. There is no `assistant_model=EAGLE-head` shortcut in the current `generate()` code path.
- **Medusa** — only legacy TGI <=2.x support; no path in modern transformers.
- **SpecExec** — research code on a fork; not mainlined.
- **MTP heads (DeepSeek-V3 / Qwen3-Next style)** — Qwen3-4B-Instruct-2507 has no MTP head; the only Qwen variants that ship MTP are the Qwen3-Next "Thinking" line.

Bottom line for our stack: in pure HF transformers, the realistic options are **assisted generation with a Qwen3-0.6B draft** and **prompt lookup**. EAGLE3 weights exist for Qwen3-4B but need a custom runner — see §3.

---

## 2. Draft models / heads available for Qwen3-4B on the Hub

Same-tokenizer Qwen3 candidates (verified shared tokenizer with `Qwen/Qwen3-4B-Instruct-2507`):

- **`Qwen/Qwen3-0.6B`** — 752 M params, ~5.4x parameter ratio vs the 4B target. This is the pairing vLLM's docs officially recommend for `Qwen3-4B-*-2507` speculative decoding. Drop-in usable as `assistant_model` in HF `generate()`.
- **`Qwen/Qwen3-1.7B`** — 2.0 B params, ratio only ~2x; acceptance will be higher but each draft step is too expensive to net a win on a 3060.

EAGLE3 heads for the Qwen3 family (all from AngelSlim, BF16 safetensors, llama-format, arXiv:2509.24248):

- **`AngelSlim/Qwen3-4B_eagle3`** — 218 M params, ~9.9k downloads. Trained against `Qwen/Qwen3-4B` (not the `-Instruct-2507` post-train). Worth a quick acceptance probe; if the post-train shift is small the head should still draft well. Reported by AngelSlim's docs: ~2.08x speedup, ~2.07 mean accept length on vLLM.
- `AngelSlim/Qwen3-1.7B_eagle3`, `_8B_eagle3`, `_14B_eagle3`, `_32B_eagle3`, `_a3B_eagle3` — adjacent sizes if we ever change target.
- `Tengyunw/qwen3_8b_eagle2_v0` and `_eagle3` — community EAGLE-2/3 for the 8B; not directly compatible with 4B.

No published **MTP**, **Medusa**, or **draft-tuned** weights for Qwen3-4B-Instruct-2507 specifically.

---

## 3. Interaction with the OSCAR mixed cache

The HF `generate()` spec-decoding loop calls the target with `input_ids` shape `(B, K+1)` for verification, which in turn calls each attention layer's `OSCARCache.update(key_states, value_states, ...)` with `n_new = K+1` instead of 1. Reading `oscar_transformers/cache.py` (`OSCARCacheLayer.update`) line by line, the existing code path **already handles `n_new > 1`**:

- The sink-fill branch uses `min(sink_remaining, n_new)` and an `offset`, so a multi-token write at prefill time would fill sink+overflow correctly (and prefill already does exactly this).
- The recent-FIFO branch slices `key_states[:, :, offset:, :]` and concatenates — independent of `n_new`.
- The spill-to-INT2 branch is keyed off `recent_k.shape[2] > recent_tokens`, not off any "single-token" assumption. A `K+1`-token write at steady state with full recent buffer will trigger a single spill of `K+1` tokens, which `quantize_per_token` and the incremental `_middle_k_dq` cache both handle as a chunk (concat along dim=2).
- `get_mask_sizes(query_length)` correctly returns `(seq_len + query_length, 0)`, so the causal mask is built for the full `K+1` queries against `seq_len+K+1` keys.

So **the cache needs no API changes**. Two soft caveats:

1. **Rejection rollback.** Spec decoding writes all `K+1` keys/values to the cache, then `generate()` calls `cache.crop(accepted_len)` to discard rejected positions. `OSCARCache` does not implement `crop` (the class docstring even calls this out). Two paths:
   - **Add `crop` for the recent + dequant-middle regions only.** In practice rejections almost never reach back into the INT2 middle (you would have to reject more than `recent_tokens=256` tokens in one step). A safe `crop(n)` can `assert n >= self.get_seq_length() - self.recent_tokens` and then trim `recent_k/v` and the dequant mirror in lockstep. ~30 lines per layer, no quant code touched. **Low complexity.**
   - **Disable rejection rollback** by setting the assistant to always accept (greedy + matching first-token check only). Loses the lossless property — not recommended for an eval run.
2. **Sink ordering during verification batched prefill.** Not a concern: the offset-based sink/recent split inside `update()` is already idempotent under multi-token writes.

Net: shipping spec decoding on top of OSCAR is roughly "implement `OSCARCacheLayer.crop` and load a Qwen3-0.6B draft." Estimate **0.5–1 day** to working code, plus tuning.

---

## 4. Realistic speedup ceiling on a 3060 12 GB

This is the crux. Standard spec-decoding intuition: it pays off when verifying `K+1` tokens costs ~1x a single-token decode, because the bottleneck is *weight* memory bandwidth, not compute. The 3060's 360 GB/s bandwidth and ~13 TF/s of fp16 throughput put us firmly in the memory-bound regime for the 4B target weights (~8 GB bf16) — so verification's marginal compute cost is essentially free. Good news for spec-D in principle.

But our cache is exotic, so the per-step memory traffic decomposes differently:

| Cost per decode step | Bytes/step now | With spec-D verifying K+1 |
|---|---|---|
| Target weight read | ~8 GB | ~8 GB (unchanged — verification is one pass) |
| Dequant middle assemble (already cached as bf16 via `_middle_k_dq`) | ~17k * 2 * 8 (heads) * 128 * 2 B ≈ 70 MB per layer * 36 layers ≈ 2.5 GB | Same (cache is reused for all K+1 queries — the assemble result is a single tensor) |
| Recent + sink concat | small | small |
| **Draft model (Qwen3-0.6B)** weight read for K draft steps | 0 | K * ~1.4 GB ≈ K * 1.4 GB |

The dequant-middle fast-path is critical: because `_assemble()` is per-call and the assembled tensor is the K-side of attention for *all* K+1 query positions in a single forward, the ~2.5 GB middle traffic happens **once** per spec round, not K+1 times. **Spec-D and the dequant fast-path stack additively, not at cross purposes.**

Expected speedup envelope on this hardware, given 17k context and Qwen3-0.6B drafting at typical 60–75% acceptance for conversational QA:

- **Lower bound: 1.4–1.6x** — pessimistic acceptance (~1.5 tokens/round), draft cost ~20% of target cost.
- **Realistic: 1.7–2.1x** — same as AngelSlim's measured vLLM EAGLE3 number (~2.08x) is a soft ceiling for any spec-D method on this model size. With a vanilla draft + assisted generation we will be below EAGLE3, around 1.7–1.9x.
- **Upper bound: ~2.3x** — if we get EAGLE3 (`AngelSlim/Qwen3-4B_eagle3`) actually wired into the HF runner. Significant engineering cost.

A 1.8x wall-time win shrinks the 9.5 h conv-0/10q run to ~5.3 h.

Two important caveats specific to our setup, both from the May 2025 paper **"Speculative Decoding Meets Quantization"** (arXiv:2505.22179):

- **Tree drafts hurt on quantized models.** EAGLE-2's tree verification incurs more compute than a single linear sequence verify, and on weight-quantized targets this overhead eats the win. We are not weight-quantized (only KV), so the effect is muted, but it argues for the simpler **sequential** assisted-generation path over EAGLE tree drafts as a first step.
- **Hierarchical drafting** (small model drafts a sequence, mid-size model expands to a tree, target verifies) gives them 2.78x on Llama-3-70B INT4. Not applicable at the 4B scale — overhead dominates.

Also relevant: **QuantSpec** (arXiv:2502.10424) — self-speculative decoding where the draft is the *target model itself* running over a **more aggressively quantized** KV cache, while verification uses the full-precision KV. Acceptance >90%, ~2.5x end-to-end on 128k-context LWM. This is **architecturally close to what we have**: OSCAR's INT2 cache could itself serve as the draft KV, with a periodic full-precision verify against a separate higher-precision cache. Implementation cost is real (two KV slabs, divergence detection), but the conceptual fit with our existing pipeline is the best of all options surveyed. **Worth its own spike** as a follow-up to a baseline spec-D implementation.

---

## 5. Other 2026 speedups worth knowing (besides spec-D)

For HF transformers + 4B + long context + custom KV, ranked by effort-to-payoff for our exact setup:

1. **SDPA / FlashAttention-2 with the assembled bf16 slab.** Confirm `attn_implementation="sdpa"` (or `"flash_attention_2"` if it builds on Windows + CUDA 12.x) is active on the Qwen3-4B model. The assembled K/V from `_assemble()` is a contiguous bf16 tensor and is FA-2 eligible. Easy win if not already on.
2. **`SpecPV`** (arXiv:2512.02337, code at `github.com/TanZhendong/SpecPV) — self-speculative *partial verification*: draft uses partial KV states, full verify periodically. Reports up to 6x on Qwen3-series at long context. Built for HF transformers. Highest-payoff spec-D variant published for our exact model family.
3. **`SpecExtend`** (arXiv:2505.20776) — drop-in spec-D enhancement for long-context: hybrid tree attention + a cross-model retrieval policy. Reports significant speedups specifically at the 16k+ context regime we live in.
4. **Cache `crop` for cross-question reuse.** Currently disabled for non-bf16 backends per `run/_chunked_eval_runner.py`. Even a partial `crop` implementation that re-uses sink + INT2 middle across questions (only recents differ) saves the ~17k prefill per question. For 10q/conv this is potentially a bigger absolute win than spec-D.
5. **`torch.compile(mode="reduce-overhead")` on the decode step.** The OSCAR cache's Python-side concats are likely dispatch-overhead-bound at 17k tokens; a compiled decode step can reclaim 10–20% on consumer cards. Risk: compile may not trace through the conditional spill branch — needs `dynamic=True` and per-layer guards.
6. **Drop INT2 dequant cost via per-group on-the-fly attention** (the QuaRot / FlatQuant lineage): replace the bf16 assemble with a fused `matmul(Q, dequant(K_codes))` kernel. Hard on Windows (no Triton for sm_86 without WSL).

Two things **not** to chase here: (a) batched-input spec-D — HF `generate()` explicitly does not support it; (b) `cache_implementation="quantized"` HQQ — already evaluated, collapses at 2-bit per the prior commit log.

---

## Recommended next step

Land in this order:

1. **Implement `OSCARCacheLayer.crop`** (recent + dequant-middle only, assert no INT2 touched). ~half-day.
2. **Wire assisted generation** with `Qwen/Qwen3-0.6B` as `assistant_model` (and a separate `DynamicCache` for the draft). Measure acceptance length on 3 LoCoMo questions. If mean accept-length >= 1.8 with `num_assistant_tokens=5`, this is the path.
3. **If acceptance is weak**, port the `AngelSlim/Qwen3-4B_eagle3` head — significantly more work but the EAGLE3 head is purpose-built and the dataset shift from Qwen3-4B to Qwen3-4B-Instruct-2507 is small.
4. **Backlog**: SpecPV-style partial-verify against a second cache. Highest theoretical ceiling but biggest engineering lift.

## Sources

- HF docs — [Assisted decoding](https://huggingface.co/docs/transformers/assisted_decoding), [Cache strategies](https://huggingface.co/docs/transformers/kv_cache), [Caching](https://huggingface.co/docs/transformers/cache_explanation)
- vLLM docs — [Draft Models](https://docs.vllm.ai/en/latest/features/speculative_decoding/draft_model/) (the Qwen3-4B-2507 + Qwen3-0.6B pairing)
- Papers — [QuantSpec (2502.10424)](https://hf.co/papers/2502.10424), [Spec Decoding Meets Quantization (2505.22179)](https://hf.co/papers/2505.22179), [SpecExtend (2505.20776)](https://hf.co/papers/2505.20776), [SpecPV (2512.02337)](https://hf.co/papers/2512.02337), [EAGLE (2401.15077)](https://hf.co/papers/2401.15077), [SpecExec (2406.02532)](https://hf.co/papers/2406.02532)
- Hub — [Qwen/Qwen3-0.6B](https://hf.co/Qwen/Qwen3-0.6B), [Qwen/Qwen3-4B-Instruct-2507](https://hf.co/Qwen/Qwen3-4B-Instruct-2507), [AngelSlim/Qwen3-4B_eagle3](https://hf.co/AngelSlim/Qwen3-4B_eagle3)
- Local code — `third_party/oscar-transformers/oscar_transformers/cache.py`, `run/_chunked_eval_runner.py`
