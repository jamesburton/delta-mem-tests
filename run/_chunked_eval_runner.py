"""Subprocess entry point that monkeypatches the LoCoMo eval's prefill before
running it.

The vendored eval's `build_teacher_forced_snapshot` does a single forward pass
on the full conversation history (~26k tokens), which OOMs on the RTX 3060
(12 GB). This driver replaces it with a chunked-ingestion version that
processes the history one message at a time. The session's KV cache plus
delta-mem state accumulate identically; only the per-step attention scratch
is bounded.

This is a controller-approved methodology adjustment, NOT a numerical change.
Autoregressive attention only depends on prior tokens via the KV cache, so
chunking the prefill produces the same per-token logits as a monolithic
forward — only the peak scratch differs. Recorded in
`report/reproduction-report.md`'s "Methodology adjustments" section.

The vendored submodule stays pinned at the original commit; this monkeypatch
is applied entirely in-process from our wrapper.
"""

from __future__ import annotations

import os
# Path-lock per report/kernels-gate.md (must come before any deltamem import)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import gc
import sys

import torch
import deltamem.eval.locomo_delta as eval_mod
from deltamem.core import reset_delta_mem_states
from deltamem.runtime.session import DeltaMemChatSession


def _chunked_build_teacher_forced_snapshot(model, tokenizer, device, history):
    """Drop-in replacement for the vendored function. Processes the history
    one message at a time, relying on `session._ingest_full_ids`'s built-in
    prefix-skip logic (session.py:515-545) to compute attention only on the
    new suffix each call. Final snapshot is mathematically identical to the
    monolithic version.
    """
    reset_delta_mem_states(model)
    session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device=device)
    for i in range(1, len(history) + 1):
        session.messages = [dict(m) for m in history[:i]]
        ids_partial = session._tokenize_messages(
            session.messages, add_generation_prompt=False
        )
        session._ingest_full_ids(ids_partial)
        stats = session.last_ingest_stats
        # Safety assertion: if the tokenization is NOT prefix-preserving,
        # _ingest_full_ids will set rebuilt=True and redo the full forward from
        # scratch — providing zero VRAM benefit and OOMing at the same step as
        # the monolithic version. Fail fast here rather than silently degrading.
        if i > 1 and stats.get("rebuilt", False):
            raise RuntimeError(
                f"Chunked prefill defeated at message {i}: tokenization is not "
                f"prefix-preserving (stats={stats}). The chat template emits "
                f"history-dependent control tokens; chunking provides no VRAM "
                f"benefit. Aborting."
            )
        print(
            f"[chunked-prefill {i}/{len(history)}] "
            f"prefix={stats.get('prefix_tokens', '?')} "
            f"suffix={stats.get('suffix_tokens', '?')} "
            f"total={stats.get('full_tokens', '?')} "
            f"elapsed={stats.get('elapsed_ms', '?')}ms",
            flush=True,
        )
    return session.snapshot()


eval_mod.build_teacher_forced_snapshot = _chunked_build_teacher_forced_snapshot


# --- Second controller-approved patch: broaden the OOM-bisection trigger. ---
# The vendored _batched_generate_raw_predictions (locomo_delta.py:600-665) has
# a divide-and-conquer retry that catches torch.OutOfMemoryError. On this host
# the kernel raises a generic RuntimeError("CUDA error: out of memory") from
# inside SDPA mask construction, which is NOT a subclass of OutOfMemoryError —
# so the bisection never engages and the eval fails on the first oversized
# batch. We wrap _generate_prompt_chunk so any RuntimeError whose message
# mentions "out of memory" is re-raised as torch.OutOfMemoryError, letting the
# existing bisector kick in unchanged.
#
# This is purely an error-class normalisation: the retry behaviour, the
# halving, and the per-batch numerics are all the vendored code's.

_orig_generate_prompt_chunk = eval_mod._generate_prompt_chunk


def _oom_normalised_generate_prompt_chunk(*args, **kwargs):
    try:
        return _orig_generate_prompt_chunk(*args, **kwargs)
    except torch.OutOfMemoryError:
        raise
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda error: out of memory" in msg:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise torch.OutOfMemoryError(str(exc)) from exc
        raise


eval_mod._generate_prompt_chunk = _oom_normalised_generate_prompt_chunk


# --- Third controller-approved patch: chunked prefill for official_prompt mode ---
# The vendored generate_official_full_history_answer (locomo_delta.py:420-479)
# builds a 2-message prompt (system + user[context+question], ~17.6k tokens) and
# calls model.generate(input_ids=...) directly. PyTorch SDPA without flash-attn
# (we're on Windows / Ampere / no flash_attn package) picks the MATH backend on
# long sequences, which materialises the full O(N²) attention scratch:
# 1 × 32 heads × 17600² × 2 bytes ≈ 20 GB per layer at peak — tried to allocate
# 37 GB on first attempt, OOM on a 12 GB card.
#
# We replace it with a DeltaMemChatSession-driven chunked prefill (same
# mechanism as the snapshot patch above): _ingest_full_ids in ~1k-token chunks
# uses prefix-skip so each forward only computes attention against the new
# suffix tokens — peak scratch shrinks to chunk² + KV-extension cost. After
# prefill, we sample tokens via session._decode_generate using the same
# temperature / top_k / top_p as the vendored sampler.
#
# Numerical equivalence: token-granularity delta-mem writes (this adapter's
# config) are autoregressive accumulations of per-token Q/K/V projections —
# identical between chunked and monolithic prefill (delta_impl.py:2173-2184).
# KV cache is identical. Final-position logits are bit-identical on
# deterministic forwards. Sampling consumes RNG identically because we seed
# inside fork_rng with seed=seed+question_index, matching the vendored
# function (locomo_delta.py:458).

from deltamem.eval.locomo_protocol import (
    OFFICIAL_TEMPERATURE,
    OFFICIAL_TOP_P,
    OFFICIAL_TOP_K,
    prepare_locomo_question,
    infer_model_context_window,
    build_official_full_history_messages,
    canonicalize_locomo_prediction,
)
from deltamem.core import iter_delta_mem_modules


_OFFICIAL_PREFILL_CHUNK = 1024


# Per-conversation cache: sample_id -> ConversationCache.
# The cache holds the session prefilled with the shared history prefix
# (system + context, up to but not including the per-question text) plus
# snapshots of (processed_input_ids, delta-mem state) at the history-only
# point. Before each question we restore from these snapshots and crop the
# session's KV cache back to history_len, so the question's _ingest_full_ids
# only processes the new question-specific suffix (~50-100 tokens) instead of
# the full ~17.6k-token prompt.
_history_kv_cache: dict[str, dict] = {}


def _evict_other_caches(current_sample_id: str) -> None:
    """Free GPU memory held by stale per-conversation caches.

    The eval processes questions contiguously by sample_idx
    (locomo_delta.py:local_question_tasks), so once we see a new sample_id
    we never go back. Drop the old session(s) before allocating a new one.
    """
    stale = [k for k in _history_kv_cache if k != current_sample_id]
    if stale:
        for k in stale:
            _history_kv_cache.pop(k, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _build_full_prompt_tokens(model, tokenizer, sample, question, *, seed, question_index, answer_reserve_tokens):
    question_spec = prepare_locomo_question(
        question,
        sample_id=str(sample["sample_id"]),
        question_index=question_index,
        seed=seed,
    )
    max_context_tokens = infer_model_context_window(model, tokenizer)
    prompt_messages = build_official_full_history_messages(
        sample,
        tokenizer,
        question_spec,
        max_context_tokens=max_context_tokens,
        answer_reserve_tokens=answer_reserve_tokens,
    )
    tokens = eval_mod._chat_template_input_ids(tokenizer, prompt_messages)
    return question_spec, prompt_messages, tokens


def _compute_history_len(model, tokenizer, sample, *, seed, answer_reserve_tokens):
    """Pre-compute the common-prefix token count across questions in a sample.

    Tokenizes the FIRST TWO questions' full prompts and finds their longest
    common prefix; that's the system+context portion shared across all
    questions in this conversation. Falls back to None (no caching) when the
    sample has fewer than 2 questions.
    """
    qa_items = sample["qa"]
    if len(qa_items) < 2:
        return None
    _, _, ids0 = _build_full_prompt_tokens(
        model, tokenizer, sample, qa_items[0],
        seed=seed, question_index=0,
        answer_reserve_tokens=answer_reserve_tokens,
    )
    _, _, ids1 = _build_full_prompt_tokens(
        model, tokenizer, sample, qa_items[1],
        seed=seed, question_index=1,
        answer_reserve_tokens=answer_reserve_tokens,
    )
    min_len = min(int(ids0.shape[1]), int(ids1.shape[1]))
    eq = (ids0[0, :min_len] == ids1[0, :min_len]).to(torch.long)
    history_len = int(eq.cumprod(0).sum().item())
    return history_len


def _snapshot_delta_state(model):
    """GPU-resident snapshot of delta-mem online state (flat module -> Tensor).

    The vendored get_delta_mem_online_state in core/delta_impl.py:2727-2733
    CPU-clones each tensor, which defeats the speed of restore. We mirror its
    logic but keep tensors on the original device so restore is a fast
    in-place pointer swap (no copy).
    """
    return {
        name: module.delta_state.detach().clone()
        for name, module in iter_delta_mem_modules(model)
        if module.delta_state is not None
    }


def _restore_delta_state(model, snapshot):
    module_map = dict(model.named_modules())
    for name, tensor in snapshot.items():
        module = module_map[name]
        module.delta_state = tensor.detach().clone()


def _ensure_conversation_cache(model, tokenizer, device, sample, *, seed, answer_reserve_tokens):
    sample_id = str(sample["sample_id"])
    if sample_id in _history_kv_cache:
        return _history_kv_cache[sample_id]
    _evict_other_caches(sample_id)

    history_len = _compute_history_len(
        model, tokenizer, sample, seed=seed, answer_reserve_tokens=answer_reserve_tokens,
    )
    if history_len is None or history_len <= 0:
        # Single-question sample or degenerate; no benefit from caching.
        _history_kv_cache[sample_id] = {"disabled": True}
        return _history_kv_cache[sample_id]

    # Use q0's prompt tokens to extract the history-only prefix.
    qa_items = sample["qa"]
    _, _, full_ids = _build_full_prompt_tokens(
        model, tokenizer, sample, qa_items[0],
        seed=seed, question_index=0, answer_reserve_tokens=answer_reserve_tokens,
    )
    full_ids = full_ids.to(device)
    history_ids = full_ids[:, :history_len].contiguous()

    reset_delta_mem_states(model)
    session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device=device)

    # Chunked-ingest the history-only prefix so the session's past_kv ends
    # at exactly history_len tokens. Subsequent questions then add only their
    # question-specific suffix via _ingest_full_ids' prefix-skip.
    end = 0
    chunk_idx = 0
    while end < history_len:
        end = min(end + _OFFICIAL_PREFILL_CHUNK, history_len)
        chunk_idx += 1
        session._ingest_full_ids(history_ids[:, :end])
        stats = session.last_ingest_stats
        if chunk_idx > 1 and stats.get("rebuilt", False):
            raise RuntimeError(
                f"History prefill rebuilt at chunk {chunk_idx}; common-prefix "
                f"invariant violated for sample {sample_id}."
            )
    print(
        f"[kv-cache build sample={sample_id}] history_len={history_len} "
        f"chunks={chunk_idx} final_chunk_ms={stats.get('elapsed_ms', '?')}",
        flush=True,
    )

    cache_entry = {
        "disabled": False,
        "session": session,
        "history_len": history_len,
        "history_processed_ids": session.processed_input_ids.detach().clone(),
        "history_delta_state": _snapshot_delta_state(model),
    }
    _history_kv_cache[sample_id] = cache_entry
    return cache_entry


def _restore_session_to_history(model, session, cache_entry):
    """Roll session state back to the history-only checkpoint so the next
    question's _ingest_full_ids only processes its question suffix."""
    history_len = cache_entry["history_len"]
    if session.past_key_values is not None:
        session.past_key_values.crop(history_len)
    session.processed_input_ids = cache_entry["history_processed_ids"].detach().clone()
    _restore_delta_state(model, cache_entry["history_delta_state"])


def _chunked_official_full_history_answer(
    model,
    tokenizer,
    device,
    sample,
    question,
    *,
    question_index,
    seed,
    max_new_tokens,
    answer_reserve_tokens,
    do_sample=True,
    temperature=OFFICIAL_TEMPERATURE,
    top_p=OFFICIAL_TOP_P,
    top_k=OFFICIAL_TOP_K,
):
    question_spec, prompt_messages, prompt_ids = _build_full_prompt_tokens(
        model, tokenizer, sample, question,
        seed=seed, question_index=question_index,
        answer_reserve_tokens=answer_reserve_tokens,
    )
    prompt_ids = prompt_ids.to(device)
    total = int(prompt_ids.shape[1])

    cache_entry = _ensure_conversation_cache(
        model, tokenizer, device, sample,
        seed=seed, answer_reserve_tokens=answer_reserve_tokens,
    )

    if cache_entry.get("disabled", True):
        # Fallback: full fresh prefill per question. Matches the non-cached
        # patch behaviour we previously committed.
        reset_delta_mem_states(model)
        session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device=device)
        session.messages = [dict(m) for m in prompt_messages]
        ingest_start = 0
    else:
        session = cache_entry["session"]
        session.messages = [dict(m) for m in prompt_messages]
        _restore_session_to_history(model, session, cache_entry)
        ingest_start = cache_entry["history_len"]

    last_logits = None
    end = ingest_start
    chunk_idx = 0
    while end < total:
        end = min(end + _OFFICIAL_PREFILL_CHUNK, total)
        chunk_idx += 1
        last_logits = session._ingest_full_ids(prompt_ids[:, :end])
        stats = session.last_ingest_stats
        if chunk_idx > 1 and stats.get("rebuilt", False):
            raise RuntimeError(
                f"Chunked official prefill defeated at chunk {chunk_idx}: "
                f"stats={stats}. Common-prefix invariant violated."
            )

    if last_logits is None:
        # Edge case: history_len already equals total (no question suffix).
        # Force a single forward to obtain the next-token logits.
        last_logits = session._ingest_full_ids(prompt_ids)

    if last_logits is None:
        raise RuntimeError("Empty official-prompt; cannot generate.")
    print(
        f"[official-prefill q{question_index} sample={sample['sample_id']}] "
        f"suffix_chunks={chunk_idx} ingest_start={ingest_start} total={total} "
        f"final_chunk_ms={stats.get('elapsed_ms', '?')}",
        flush=True,
    )

    rng_devices = []
    if torch.cuda.is_available() and device.startswith("cuda"):
        rng_devices = [torch.device(device)]
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed + question_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + question_index)
        generated_ids = session._decode_generate(
            last_logits,
            max_new_tokens,
            do_sample=do_sample,
            temperature=OFFICIAL_TEMPERATURE if temperature is None else float(temperature),
            top_p=OFFICIAL_TOP_P if top_p is None else float(top_p),
            top_k=OFFICIAL_TOP_K if top_k in (None, 0) else int(top_k),
        )

    raw_prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    canonical_prediction = canonicalize_locomo_prediction(raw_prediction, question_spec)
    return raw_prediction, canonical_prediction


eval_mod.generate_official_full_history_answer = _chunked_official_full_history_answer


# Re-shape sys.argv so the eval's argparse sees the expected program name
# rather than "run._chunked_eval_runner". Cosmetic but cleaner help output.
sys.argv[0] = "deltamem.eval.locomo_delta"
sys.exit(eval_mod.main() or 0)
