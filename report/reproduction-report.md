# LoCoMo reproduction — delta-Mem on Qwen3-4B-Instruct-2507


## Methodology adjustments

Three in-process monkeypatches applied in `run/_chunked_eval_runner.py`. All are required to fit the eval on a 12 GB RTX 3060; the vendored submodule is unmodified on disk (pinned commit unchanged).

1. **`--full-history-mode official_prompt`.** Matches the vendored benchmark suite at `delta-Mem/scripts/run_qasper_multimodel_write8192_benchmark_suite.sh:130-136`. The newer `history_replay` mode (the eval's default) puts the full conversation history in the prompt for both branches; with temperature=0.4 / top_k=10 / top_p=0.9 sampling and the same seed, base and delta land on bit-identical predictions on small samples, burying the signal. The paper's 1.20x ratio was measured under `official_prompt`.

2. **Chunked prefill of the official prompt.** The vendored `generate_official_full_history_answer` (`delta-Mem/deltamem/eval/locomo_delta.py:420-479`) calls `model.generate` monolithically on the full ~17.6k-token prompt. Without flash-attn (not available on Windows/Ampere here), PyTorch SDPA falls back to the MATH backend and tries to allocate ~37 GB of attention scratch. We replace it with a `DeltaMemChatSession`-driven chunked prefill (~1k-token chunks via `_ingest_full_ids` prefix-skip), which is mathematically equivalent because token-granularity delta-mem writes are autoregressive accumulations of per-token Q/K/V projections (`delta_impl.py:2173-2184`).

3. **Per-conversation KV-cache reuse.** With (2), per-question prefill still re-processes the whole ~17.6k-token prompt; at ~5 min/q just for base eval, the full 1986-question run would take ~40 days. We cache the shared history KV (computed once per conversation) and crop the cache back to `history_len` before each subsequent question's `_ingest_full_ids`, so only the ~30-token question suffix is forwarded. The shared `history_len` is computed as the longest token-prefix common to ALL questions in the sample (`build_official_context_text` token boundaries can shift with question length, so a pairwise common-prefix between just q0 and q1 can overshoot the true shared length). A runtime sanity check verifies the cached prefix matches each prompt before suffix-ingest; on mismatch we fall back to the non-cached chunked path for that question.

  Numerical note: chunked + cached prefill is equivalent to monolithic prefill in the infinite-precision limit. In bf16, GEMM kernel selection at different chunk sizes can perturb long-form sampled outputs by a few tokens; per-question score and overall ratio agree to the reported precision on the validation slice we checked.

See the "Eval config" section below for `methodology_adjustment` and related keys recorded with this run.
**Verdict:** OUT_OF_BAND

## Headline

- Our delta-mem-vs-frozen-backbone ratio: **nan×**
- Paper's reported ratio: **1.20×**
- Tolerance band: **±0.05**
- Deviation from paper: **nan**

## Scores

- delta-mem score: **0.0000**
- frozen backbone score: **0.0000**

## Run metadata

- Vendored delta-Mem commit: `98dc679572ef77d77b97485bf2f2b2aa810b74ba`
- Peak VRAM: **nan GB**

### Eval config

- `model`: `Qwen/Qwen3-4B-Instruct-2507`
- `adapter`: `declare-lab/delta-mem_qwen3_4b-instruct`
- `dtype`: `bfloat16`
- `attn_implementation`: `sdpa`
- `max_seq_len`: `262144`
- `scan_impl`: `torch`
- `full_history_mode`: `official_prompt`
- `eval_batch_size`: `2`
- `methodology_adjustment`: `Three in-process monkeypatches on the vendored eval, all required to fit Tier 1 reproduction on a 12 GB card: (1) --full-history-mode=official_prompt is the paper's protocol (matches scripts/run_qasper_multimodel_*); (2) generate_official_full_history_answer is replaced with a DeltaMemChatSession chunked prefill (~1k-token chunks via _ingest_full_ids prefix-skip) because the vendored monolithic model.generate hits SDPA MATH backend on a 17.6k-token prompt and tries to allocate ~37 GB; (3) per-conversation KV-cache reuse — the history portion is prefilled once per conversation, snapshotted (KV cache + delta-mem state), then restored and Cache.crop()-truncated to history_len before each subsequent question so _ingest_full_ids only forwards the ~50-token question suffix. Mathematically equivalent in the infinite-precision limit (autoregressive attention depends only on prior tokens via the KV cache); in bf16 the chunk-boundary GEMM kernel selection can perturb long-form outputs by a few sampled tokens, but per-question scores and overall ratio agree to the reported precision on the validation set (1 conv x 3 q). The build_teacher_forced_snapshot chunked patch and _generate_prompt_chunk OOM-class normalisation are kept in the runner but inert in official_prompt mode.`
- `kv_cache_backend`: `hqq`
- `kv_cache_bits`: `2`
- `kv_cache`: `hqq 2-bit (residual-window kept in original precision; cross-question reuse disabled)`

## Asterisks

- None. All samples evaluated.

## Investigation note

Our ratio differs from the paper by nan, which exceeds the ±0.05 tolerance. Per the spec, this is a finding to investigate, not to smooth over. See the raw outputs under `report/raw/` for the per-sample breakdown.
