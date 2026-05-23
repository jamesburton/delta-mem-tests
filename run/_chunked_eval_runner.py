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


# Re-shape sys.argv so the eval's argparse sees the expected program name
# rather than "run._chunked_eval_runner". Cosmetic but cleaner help output.
sys.argv[0] = "deltamem.eval.locomo_delta"
sys.exit(eval_mod.main() or 0)
