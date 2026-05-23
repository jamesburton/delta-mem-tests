# Task 7 — execution notes

This file captures what happened during Task 7 execution that is NOT in the
original plan (`docs/superpowers/plans/2026-05-22-delta-mem-tier1-reproduction.md`).
The plan is a stable artifact recording the original intent; this file records
the deviations that actually shipped.

## What the plan said

> If the run OOMs mid-eval, this is risk R2 materialising: lower
> `EVAL_CONFIG["max_seq_len"]` … or record the affected samples as skipped.

## What actually happened

Two OOMs occurred in the first attempt at Step 2:

1. **First OOM** — base-eval branch, ~36.93 GB allocation request. Original
   diagnosis ("base-model only, --skip-base will help") was wrong; the OOM was
   really inside `build_teacher_forced_snapshot`'s single monolithic forward on
   the full ~26k-token conversation, hitting SDPA's O(N²) attention scratch.
2. **Second OOM** — delta-eval branch, after `--skip-base` attempt. Same root
   cause; --skip-base does nothing for this code path.

`max_seq_len` lowering was rejected: the conversations *are* ~26k tokens and
truncating would change what is actually being scored, violating R3
(methodology fidelity). Sample skipping was also rejected: every conversation
would be skipped.

## Approved deviation: chunked prefill

User-approved on 2026-05-23: replace `build_teacher_forced_snapshot`'s single
`session._ingest_full_ids(full_history)` with a per-message loop that calls
`_ingest_full_ids` once per message. The session's existing prefix-skip logic
(`delta-Mem/deltamem/runtime/session.py:519-545`) means each call only
processes the new suffix — peak attention scratch shrinks to the per-message
size while the final KV cache and delta-mem state are bitwise-identical.

**Implementation:**
- `run/_chunked_eval_runner.py` applies the monkeypatch in-process and calls
  `deltamem.eval.locomo_delta.main()`.
- `run/locomo_eval.py` invokes this driver via `python -m run._chunked_eval_runner`
  instead of the raw eval module.
- The vendored submodule is NOT modified on disk; the pinned commit hash
  (`98dc679572ef77d77b97485bf2f2b2aa810b74ba`) is unchanged.
- `EVAL_CONFIG["methodology_adjustment"]` records this and `render_report`
  prepends a "Methodology adjustments" section to the report itself.
- A `rebuilt` safety check in the runner fails fast if the chat template
  emits history-dependent control tokens that defeat the prefix-skip — we
  do not want to silently degrade and OOM at the same point.

**Why this is not a numerical change:** autoregressive attention depends only
on prior tokens through the KV cache, and the KV cache is built by the same
forward pass in both versions; only the batching of that forward differs.
Same per-token logits, same delta-mem updates, same final snapshot.

## Status at end of first session

- Chunked runner committed.
- Awaiting full eval run (Step 2). The user mentioned a fallback machine
  (T5500, internal RTX 3060) if this host OOMs or crashes — the chunked
  patch should fit comfortably in 12 GB even on slower hardware. The
  invocation is the same: `uv run python -m run.locomo_eval`.

## Second OOM on T5500 (2026-05-23, internal RTX 3060)

After the smoke test passed on the T5500, the first full-eval attempt OOMed
at a *new* code path: base-eval `model.generate` inside
`_batched_generate_raw_predictions` → `_generate_prompt_chunk`, which the
chunked-prefill patch did not cover. Two root causes:

1. **Vendored default `--eval-batch-size 64`** (`locomo_delta.py:213`) is too
   large for a 12 GB card on this model + dataset.
2. The eval's halving-retry catches only `torch.OutOfMemoryError`, but the
   actual exception was a plain `RuntimeError("CUDA error: out of memory")`
   from inside `transformers/masking_utils.py:299` (SDPA causal mask), so the
   bisector never engaged.

## Second approved deviation: batch=8 + OOM-class normalisation (failed; superseded)

User-approved on 2026-05-23:

- Plumb `--eval-batch-size 8` through `run/locomo_eval.py` (recorded in
  `EVAL_CONFIG["eval_batch_size"]`).
- In `run/_chunked_eval_runner.py`, wrap `_generate_prompt_chunk` so any
  `RuntimeError` whose message contains "out of memory" is re-raised as
  `torch.OutOfMemoryError` (and `torch.cuda.empty_cache()` is called before
  re-raise). This lets the vendored bisector at `locomo_delta.py:617-665`
  engage and halve 8 → 4 → 2 → 1 when an individual sample needs it.

**Outcome:** Chunked prefill ran fine (all 20 messages, peak ~17.6k tokens),
the broadened-OOM catch engaged on the first base-eval batch (8 → 4+4), but
the CUDA OOM recovery itself **crashed the Python process** with Windows
`STATUS_STACK_BUFFER_OVERRUN` (`0xC0000409`) and a CPU-side
`memory allocation of 2289664 bytes failed` from the C runtime. The bisector
is not usable as a safety net on this host.

## Fourth observation: per-question full-prefill is the actual bottleneck

After batch=2 was running cleanly, the vendored eval's progress bar revealed
a much worse problem: each question takes ~17–25 minutes because
`model.generate` does a fresh ~17.6k-token prefill for every single question,
with no KV-cache reuse across questions in the same conversation. ETA at
batch=2 for the full 1,986-question set: ~25 days.

The vendored eval's `_generate_prompt_chunk` builds prompts as
`history_messages + [question]` and calls `model.generate(input_ids=..., past_key_values=None)`
fresh per question. Power draw is 55 W (vs 170 W TDP) — kernel is
memory-bandwidth-bound on the long-prefill SDPA path. Lowering batch
doesn't help; raising batch OOMs.

## Fifth approved deviation (planned): per-conversation KV-cache reuse via DeltaMemChatSession

**Key insight:** `DeltaMemChatSession._ingest_full_ids`
(`delta-Mem/deltamem/runtime/session.py:515-567`) already implements
prefix-skip: it computes `_common_prefix_len`, slices off the new suffix,
and runs the forward pass only on the suffix using
`past_key_values=self.past_key_values`. This is the SAME mechanism our
chunked-prefill patch already relies on for the snapshot build. It just
isn't being used for per-question generation in the eval's `_generate_prompt_chunk`.

**The catch:** after a question's generated tokens are appended to
`session.processed_input_ids`, the next question (different from question 1)
diverges from the cached prefix, and `_ingest_full_ids` hits the
`rebuilt=True` fallback that throws away the cache and re-prefills from
scratch.

**Solution:** crop the KV cache back to history length after each question.
`transformers.cache_utils.Cache.crop(max_length)` exists and truncates each
layer's K/V tensors. So per conversation:
1. Build session, chunked-ingest the history (same as snapshot patch).
2. Save the post-history state: `history_len = past_kv.get_seq_length()`.
3. For each question:
   a. Ingest `history + question` — only the question suffix gets prefilled.
   b. Iteratively generate tokens via `_decode_generate`.
   c. Crop the KV cache back to `history_len`; reset
      `processed_input_ids = history_ids`.
4. After the conversation, free the session.

**Expected speedup:** per-question prefill drops from ~17.6k tokens to
~50–100 tokens (just the question text). Per-question time should fall
from ~17 min to ~1 min — a 15–25× speedup.

**Numerical equivalence:** greedy decoding gives bitwise-identical logits
to fresh prefill (autoregressive attention only depends on prior tokens
via the KV cache, which we reconstruct exactly). Sampling paths see the
same RNG stream because the eval wraps generation in
`torch.random.fork_rng + torch.manual_seed(batch_seed)` (`locomo_delta.py:572-574`)
and our prefill is deterministic.

**Validation:** before scaling, run the same `--max-conversations 1
--max-questions-per-conversation 10` partial set with the patch enabled
and diff per-question `raw_prediction` strings against the baseline. If
they match (or differ only by sampling RNG that's already
batch-size-sensitive), enable for the full run.

## Third approved deviation: hard-set --eval-batch-size 2

User-approved on 2026-05-23: drop the initial batch to a size that NEVER OOMs
rather than relying on bisector recovery.

- 8 GB weights + 2 × ~0.6 GB KV-cache + SDPA prefill scratch ≈ 9.5-10 GB on a
  12 GB card — under budget with ~2 GB of headroom.
- The OOM-class normalisation patch in `_chunked_eval_runner.py` stays in
  place as a defence-in-depth wrapper. It should not fire at batch=2 on any
  sample; if it does, that's a separate signal to investigate (likely an
  individual conversation longer than 17.6k tokens).
- `EVAL_CONFIG["eval_batch_size"]` updated to 2; methodology section in the
  reproduction report records this change in full.

**Why this is not a numerical change for the headline scores:** the LoCoMo
`overall_score` path runs `do_sample=False` (greedy), so per-prompt logits
are batch-size-invariant. Sampling-only paths (history-replay probes) ARE
batch-RNG-sensitive in principle, but no reference batch size is committed
in the upstream repo, so the vendored default of 64 was already a free
choice; 8 is the same kind of choice with smaller VRAM.

The vendored submodule remains unmodified on disk; the pinned commit hash
(`98dc679572ef77d77b97485bf2f2b2aa810b74ba`) is unchanged. Both deviations
are in-process monkeypatches recorded in the reproduction report's
"Methodology adjustments" section and in `EVAL_CONFIG["methodology_adjustment"]`.

## If you are picking this up on the T5500

1. Verify env: `uv sync && uv run pytest tests/ -v` (all 10 should pass).
2. Verify path-lock: smoke test still works.
   `uv run python -m run.smoke_chat`
3. Run the full eval: `uv run python -m run.locomo_eval 2>&1 | Tee-Object report/raw/locomo-driver.log`
4. Watch `report/raw/locomo-stdout.log` for:
   - `[chunked-prefill i/N]` lines confirm the prefill patch is active. Each
     line should show `suffix=` much smaller than `total=`; if `suffix=` ever
     equals `total=` past the first message, the safety assertion will fire
     and abort.
   - `[locomo_delta] CUDA OOM at batch size N; retrying with ...` lines (if
     they appear) confirm the bisector engaged via the broadened-OOM catch.
5. On completion, `report/reproduction-report.md` is written and the verdict
   is one of PASS / OUT_OF_BAND / REGRESSION.
