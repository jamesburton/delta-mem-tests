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

## Appendix B: TurboQuant + delta-mem (planned)

A follow-up experiment is staged but not yet run. The hypothesis: TurboQuant 4-bit KV quantisation (`turboquant 0.2.0`, `TurboQuantCache` drop-in for `DynamicCache`) shrinks the per-question KV cache ~4× (last 128 tokens stay FP16 as a residual window). Delta-mem's compressed state runs alongside it. If quality holds across the four conditions (bf16/TQ4 × no-adapter/adapter), the same VRAM budget supports meaningfully longer context — which is the regime where delta-mem's value shows up (Appendix A).

Wired in:

- `pyproject.toml` adds `turboquant>=0.2.0`.
- `run/_chunked_eval_runner.py` honours `TURBOQUANT_BITS` env var; when set it disables the per-conversation KV-cache reuse path (TurboQuantCache's quantised layers don't preserve correctness under `crop()`) and seeds each fresh session with `TurboQuantCache(bits=N)`.
- `run/locomo_eval.py --turboquant-bits N` plumbs the flag through and records it in `EVAL_CONFIG`.

Runtime budget on conv-0 with cache reuse disabled is ~25 h for the full 152 × 2 conditions; a smoke at `--max-questions-per-conversation 10` (~100 min) validates quality before scaling up. Results pending.
