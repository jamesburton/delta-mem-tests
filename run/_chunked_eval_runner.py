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


_OFFICIAL_PREFILL_CHUNK = 1024


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

    reset_delta_mem_states(model)
    session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device=device)
    session.messages = [dict(m) for m in prompt_messages]
    prompt_ids = session._tokenize_messages(prompt_messages, add_generation_prompt=True)

    total = int(prompt_ids.shape[1])
    last_logits = None
    end = 0
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
        raise RuntimeError("Empty official-prompt; cannot generate.")
    print(
        f"[official-prefill q{question_index}] chunks={chunk_idx} "
        f"total_tokens={total} final_suffix_ms={stats.get('elapsed_ms', '?')}",
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
