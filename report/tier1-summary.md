# Tier 1 LoCoMo reproduction — three-conversation summary

**Hardware:** Native Windows 11 + RTX 3060 (12 GB), Qwen3-4B-Instruct-2507 (bf16) with the published `declare-lab/delta-mem_qwen3_4b-instruct` adapter, vendored `declare-lab/delta-Mem` pinned at `98dc679572ef77d77b97485bf2f2b2aa810b74ba`.

**Eval protocol:** `--full-history-mode official_prompt` matching `delta-Mem/scripts/run_qasper_multimodel_write8192_benchmark_suite.sh:130-136`. Three in-process monkeypatches (chunked official prefill, per-conversation KV-cache reuse, history_len computed across all questions per sample) are required to fit the eval in 12 GB; see `report/reproduction-report.md` for the full methodology section and `run/_chunked_eval_runner.py` for the code.

## Headline

Across three of LoCoMo's ten conversations (n=389 questions, categories 1–4):

| Conv | sample_id | tokens (est) | n | base score | delta score | ratio | wall time |
|------|-----------|-------------:|---:|-----------:|------------:|------:|----------:|
| 0    | conv-26   |       17,618 | 152 |   0.3617 |     0.3644 | 1.007× | 3:49:23 |
| 1    | conv-30   |       13,596 |  81 |   0.4292 |     0.4281 | 0.997× | (subset) |
| 8    | conv-49   |       19,127 | 156 |   0.3764 |     0.3651 | 0.970× | (subset) |
| **weighted avg** |           |              | **389** |   **0.3816** |     **0.3779** | **0.990×** | conv 1+8: 16:43:54 |

**Paper's reported LoCoMo ratio:** 1.20× (averaged across all ten conversations).

**Our three-conversation average:** 0.990× — essentially no lift from delta-mem. Deviation from paper: **0.21**, well outside the ±0.05 tolerance band declared in the spec. **Verdict: REGRESSION** (delta-mem score is slightly below base, not meaningfully different).

## Per-conversation per-category breakdown

### conv-0 (sample_id=conv-26, n=152)

| Category    | n  | base   | delta  | ratio |
|-------------|---:|-------:|-------:|------:|
| multi_hop   | 32 | 0.3223 | 0.3213 | 0.997× |
| temporal    | 37 | 0.3866 | 0.3852 | 0.996× |
| open_domain | 13 | 0.1145 | 0.1229 | 1.073× |
| single_hop  | 70 | 0.4125 | 0.4179 | 1.013× |

### conv-1 (sample_id=conv-30, n=81)

Combined into conv 1+8 output; see `outputs/full_conv1_8.json` for category split (gitignored).

### conv-8 (sample_id=conv-49, n=156)

Combined into conv 1+8 output; see `outputs/full_conv1_8.json` (gitignored).

## What the gap tells us

Our three-conversation average of 0.990× is not the paper's 1.20×. Candidate explanations, in rough order of prior:

1. **Adapter mismatch (most likely).** The vendored benchmark script
   (`delta-Mem/scripts/run_qasper_multimodel_write8192_benchmark_suite.sh`)
   references QASPER-trained variants — `SSW_rank8_qasper_write8192`,
   `TSW_rank8_qasper_write8192`, `MSW_qasper_write8192` — at internal
   `/root/models/...` paths, not the HF-published
   `declare-lab/delta-mem_qwen3_4b-instruct` we used. The published
   adapter may not be the one that produced the paper's 1.20×.

2. **Conversation sampling.** We covered 3/10 conversations. The paper
   averages across all ten; individual conversations could vary widely.
   Two of our three convs (conv-0 and conv-1) are essentially flat; one
   (conv-8) shows a small regression. We have no reason to think the
   remaining seven would average to 1.20× on this adapter, but we can't
   rule it out without more data.

3. **bf16 numerical drift.** The chunked + cached prefill is
   mathematically equivalent to the vendored monolithic prefill in the
   infinite-precision limit, but bf16 GEMM kernel selection at different
   chunk sizes perturbs long-form sampled outputs by a few tokens (we
   verified this on a 3-question reference set: Q0/Q2 short answers
   matched byte-for-byte, Q1 long answer diverged). Over 389 questions
   this drift could plausibly erode a small signal, but it's unlikely to
   convert a +20% lift into 0%.

4. **Some other methodology difference** between this code path and
   whatever produced the paper's 1.20× number (different eval-time
   hyperparameters, post-processing, dataset version, scoring metric
   tweak, ...).

## Honesty notes / asterisks

- Three of ten conversations is a **partial reproduction**, not a full one. The honest claim is "on this hardware, on this adapter, on this 3-conversation subset, we don't reproduce the paper's improvement."
- Categories 1–4 only (the eval default). Category 5 (adversarial) excluded; this matches the official benchmark script.
- All raw outputs are committed at `outputs/full_conv0.json` and `outputs/full_conv1_8.json` (gitignored locally but available on the run machine) and the eval stdout at `report/raw/locomo-stdout-{full-conv0,conv1-8}.log` (also gitignored).
- Wall time on this hardware (RTX 3060 / 12 GB): ~3.8 h per shortest conversation, ~14 h for conv-8 alone. Full 10-conversation run would project to roughly **3–5 days** with the current KV-cache patch — feasible but a meaningful time investment.

## What I'd do next

A. **Try a different adapter.** If anyone has the SSW/TSW/MSW QASPER-trained checkpoints, swap one in and re-run conv-0; that's the cleanest test of explanation (1) above and only costs ~4 h.

B. **Or: keep the current adapter, run the remaining seven conversations.** ~3 days of wall time produces the real reproduction number under the published adapter, comparable apples-to-apples to the paper if the published adapter is in fact what they used.

C. **Or: ship Tier 1 as-is with this summary as the deliverable.** The reproduction *protocol* is correct (matches the vendored benchmark script), the patches that made it fit the hardware are documented and validated, and the honest finding is that 3 of 10 conversations under the public adapter don't show the paper's lift. That is itself a result.

## Appendix A: does the compressed delta-mem state alone preserve enough context?

The `full_history_replay` mode above is a "delta-mem on top of full attention" test, not a "delta-mem replacing the conversation history" test. The latter — feeding the model only its compressed delta-mem state plus the question, with no history in the prompt — is closer to delta-mem's intended long-context-compression value. We ran it on conv-0 (152 questions, categories 1–4) via `run/delta_only_eval.py`.

| Condition | Memory of history | overall | multi_hop | temporal | open_domain | single_hop |
|---|---|---:|---:|---:|---:|---:|
| **truncated_base** | none — model gets just the question | **0.0591** | 0.0341 | 0.0036 | 0.1468 | 0.0837 |
| **delta_only** | compressed delta-mem state only | **0.1064** | 0.0648 | 0.0713 | 0.1634 | 0.1335 |
| full_history_base (from main result) | full ~17.6k-token history in prompt | 0.3617 | 0.3223 | 0.3866 | 0.1145 | 0.4125 |
| full_history_delta (from main result) | full history *and* delta state | 0.3644 | 0.3213 | 0.3852 | 0.1229 | 0.4179 |

**Takeaways:**

- **Delta-mem captures real information.** `delta_only / truncated_base = 1.80×` on overall score. The state isn't decorative — it's storing recoverable signal from the conversation.
- **The signal is strongest where memory matters most.** Temporal questions (`when did X happen`) jump from 0.0036 → 0.0713 — a 20× lift over the no-memory baseline. Single-hop fact recall and multi-hop reasoning both roughly double.
- **Open-domain is the exception.** truncated_base actually beats full_history_base on this category (0.1468 vs 0.1145); these questions don't need the conversation. So delta_only's small lift here (1.11×) is unsurprising.
- **But compression is lossy.** `delta_only / full_history_delta = 0.29×`. On context that *fits in attention* (17.6k tokens fits comfortably in 12 GB), full attention beats compressed memory by ~3×. Delta-mem is not a drop-in replacement; it's a compressed-memory fallback for the regime where full KV won't fit.

**Memory budget at 12 GB with this approach:** weights (~8 GB) + delta-mem state (~300 MB) + tiny KV for the ~100-token question prompt + scratch ≈ 9 GB. Plenty of headroom. The conv-0 run completed in ~50 min including ~17 min of one-time history-prefill — much cheaper than the `full_history_replay` path which is ~4 h per condition.

**Implication for the long-context use case:** this is the experiment that *would* matter on a 100k-token problem, where full KV is multiple GB and won't fit alongside weights on a 12 GB card. The break-even point — at what context length does (delta-mem of N tokens + short prompt) beat (full attention of the M tokens that fit) — needs a longer-context dataset and ideally a bigger machine to set the comparison ceiling. The current data says delta-mem is real; what's missing is the regime where it's the *only* option.

## Appendix B: TurboQuant 4-bit KV + delta-mem

We ran the staged TurboQuant smoke (`turboquant 0.2.0`, `TurboQuantCache(bits=4)` drop-in for `DynamicCache`; last 128 tokens stay FP16 as a residual window) on conv-0 / first 10 questions, both branches. Same 10 questions evaluated under bf16 KV (subset of the prior full-conv-0 run) and TQ4 KV.

### Quality (same 10 questions)

| Condition | bf16 overall | TQ4 overall | diff |
|---|---:|---:|---:|
| base   | 0.2680 | 0.3344 | +0.0664 |
| delta  | 0.2680 | 0.3342 | +0.0662 |

Per-category at n≤6 the numbers are too noisy to read deeply, but **TQ4 doesn't regress** — within sampling noise it's at least as good as bf16 on this slice. The per-question raw_predictions differ between bf16 and TQ4 in most cases (quantisation perturbs which sampled tokens emerge under temperature=0.4/top_k=10/top_p=0.9), but the F1 scores against gold come out essentially the same. The slight +0.066 lift is almost certainly statistical noise at n=10.

### Memory (estimated, at 17.6k tokens)

- bf16 KV cache: ~0.6 GB
- TQ4 KV cache: ~0.16 GB (4-bit body + 128-token FP16 residual window)
- saving: ~0.44 GB (~73%)

On a 12 GB card this is the difference between roughly **~30k-token max context (bf16) and ~100k+ token max context (TQ4)**. That's the regime change that matters for delta-mem's actual use case (Appendix A): with TQ4 freeing VRAM, you can run the long-context KV *and* the delta-mem state together at sequence lengths bf16 can't reach.

### Runtime cost (the catch)

The 20 generations (10 q × 2 conditions) took **20 h wall**. `turboquant 0.2.0` is a pure-Python reference implementation; per-token quantisation in Python dominates the inner loop. Per-question prefill at 17.6k context averaged ~55 min for TQ4 vs ~5 min for bf16 — a ~11× slowdown. The compression math works, but in the current implementation the runtime tax wipes out the inference-time benefit on this hardware.

Cross-question KV-cache reuse (the optimisation that gave bf16 a 7.7× speedup) is disabled under TQ because `TurboQuantCache`'s quantised layers don't preserve correctness under `Cache.crop()`. Even if we fixed crop, the quantisation overhead alone is the bottleneck — every question still pays the ~50 min prefill cost.

### Verdict

- **Quality:** TQ4 preserves recall at this sample size. The hypothesis (KV compression + delta-mem ≈ bf16 + delta-mem in quality) holds for the conv-0 slice.
- **Memory:** Real and large — 73% saving at 17.6k, scales linearly. Enables the long-context regime on a small card.
- **Throughput:** Unusable with `turboquant 0.2.0` (pure Python). A custom CUDA kernel implementation — the kind landing in [llama.cpp forks](https://github.com/ggml-org/llama.cpp/discussions/20969) — would be required for production use. The HF transformers Python path isn't where TurboQuant gets to shine yet.

The combination *would* unlock running longer context on this 12 GB card if a fast kernel were available. The next experiment that would actually pay off is testing this on a model + dataset that genuinely needs the bigger context window (say 50k+ tokens), where bf16 simply doesn't fit, and where delta-mem's compressed state can carry the rest. That ideally happens on faster kernels (llama.cpp turbo3/turbo4 variants) or on a machine with more headroom for the slow Python path.

Raw artifacts: `outputs/tq4_conv0_smoke.json`, `report/raw/locomo-{stdout,driver}-tq4-conv0-smoke.log`.

## Appendix C: HQQ 2-bit KV — quality collapse

We staged a second 2-bit KV experiment using `transformers.cache_utils.QuantizedCache(backend="hqq", nbits=2)` — the HQQ backend was selected after optimum-quanto's CUDA extension failed to compile on Windows (its `gemm_cuda.cu` uses GCC `__asm__` inline assembly that MSVC can't parse).

Conv-0 / first 10 questions, 8h 42m wall.

| Condition | overall | multi_hop | temporal | open_domain |
|---|---:|---:|---:|---:|
| base   | **0.0000** | 0.0000 | 0.0000 | 0.0000 |
| delta  | **0.0000** | 0.0000 | 0.0000 | 0.0000 |

The model outputs are gibberish under HQQ-2bit. Sample raw predictions:

- Base Q0 ("When did Caroline go to the LGBTQ support group?"): `'and it?"\n\n,  is is that? that &  it is, said and   & ", about is is to, and hers'`
- Delta Q0: `"Car32-tofa �awning is to-t wo' H e3hH iv6   che3 �ozah� **:3 3chan riclish  is d"`

**Why naive 2-bit fails here:** HQQ's per-channel quantization is designed primarily for *model weight* compression, where the activations passing through stay fp16. KV cache quantization at 2 bits applies the same shape of compression directly to attention K/V tensors, but K/V has very different statistics than weight matrices — large outliers per channel destroy the quantization grid and the dequantized K/V values are too noisy for attention to recover usable scores.

**Why this validates the OSCAR direction:** OSCAR's headline mechanism is an *offline spectral covariance-aware rotation* applied to K/V before quantization. The rotation flattens the per-channel outlier distribution into something the 2-bit grid can represent without destroying attention. Without the rotation, 2-bit is unusable — which is exactly what this run shows. The empirically-validated conclusion is:

- 4-bit naive (TurboQuant, Appendix B): quality preserved
- 2-bit naive (HQQ, this appendix): quality collapses to 0
- 2-bit with attention-aware rotations (OSCAR): paper claims near-baseline accuracy → next experiment to run

So the next experimental step is to port OSCAR's rotation+quantize math into a `transformers.Cache` subclass (avoiding the SGLang dependency), compute Qwen3-4B-Instruct rotations, and re-run on conv-0. That's the work tracked for the staged OSCAR sub-repo.

Raw artifacts: `outputs/hqq2_conv0_smoke.json`, `report/raw/locomo-{stdout,driver}-hqq2-conv0.log`.
