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

# Optional KV-cache quantization. Selected by two env vars:
#   KV_CACHE_BACKEND ∈ {"bf16" (default), "turboquant", "quanto", "hqq"}
#   KV_CACHE_BITS    ∈ int (backend-default if 0)
#
# When KV_CACHE_BACKEND != "bf16":
#   * the per-conversation KV-cache reuse path is bypassed. Quantised cache
#     classes either don't override Cache.crop() correctly (turboquant) or
#     have residual+quantised-state interactions that aren't safe to crop
#     into (quanto, hqq). Re-prefilling per question is slower but correct.
#   * each fresh session is seeded with the requested quantised cache so
#     K/V are compressed in-flight; recent-token residuals stay in original
#     precision (128 tokens for turboquant, 128 for quanto/hqq via the
#     `residual_length` arg).
#
# Backwards compat: TURBOQUANT_BITS=N (the prior single-knob interface) is
# treated as `KV_CACHE_BACKEND=turboquant KV_CACHE_BITS=N`.
_legacy_tq_bits = int(os.environ.get("TURBOQUANT_BITS", "0"))
if _legacy_tq_bits > 0 and "KV_CACHE_BACKEND" not in os.environ:
    os.environ["KV_CACHE_BACKEND"] = "turboquant"
    os.environ.setdefault("KV_CACHE_BITS", str(_legacy_tq_bits))

KV_CACHE_BACKEND = os.environ.get("KV_CACHE_BACKEND", "bf16").lower()
KV_CACHE_BITS = int(os.environ.get("KV_CACHE_BITS", "0"))

_VALID_BACKENDS = {"bf16", "turboquant", "quanto", "hqq", "oscar"}
if KV_CACHE_BACKEND not in _VALID_BACKENDS:
    raise ValueError(
        f"KV_CACHE_BACKEND={KV_CACHE_BACKEND!r} not in {sorted(_VALID_BACKENDS)}"
    )

if KV_CACHE_BACKEND == "turboquant":
    # turboquant 0.2.0 calls np.trapz which was removed in numpy 2.0 (renamed
    # to np.trapezoid). Restore the alias before importing turboquant; this
    # is the minimal patch and is bit-identical to the original function.
    import numpy as _np
    if not hasattr(_np, "trapz") and hasattr(_np, "trapezoid"):
        _np.trapz = _np.trapezoid
    from turboquant import TurboQuantCache  # noqa: F401
elif KV_CACHE_BACKEND in ("quanto", "hqq"):
    # transformers ships a KIVI-style QuantizedCache that wraps either the
    # optimum-quanto or hqq backend. Quanto supports nbits ∈ {2, 4}; hqq
    # supports {1, 2, 3, 4, 8}. We default to 2-bit when KV_CACHE_BITS is 0.
    from transformers.cache_utils import QuantizedCache  # noqa: F401
elif KV_CACHE_BACKEND == "oscar":
    # OSCAR rotates Q/K/V/O projections into a basis that flattens per-channel
    # outliers, then quantizes K and V to per-token group-128 asymmetric INT2
    # in the rotated basis. The rotation is mathematically a no-op end-to-end
    # (orthogonal) but the rotated tensor distribution is friendly to INT2
    # quantization where the naive basis collapses to gibberish (Appendix C of
    # report/tier1-summary.md).
    #
    # Required env vars:
    #   OSCAR_K_ROTATION_PATH — path to k_rotation_qqt_r_h_pbr.pt
    #   OSCAR_V_ROTATION_PATH — path to v_rotation_sst_r_h_pbr.pt
    # Optional knobs (matched to RotationZoo / sglang defaults):
    #   OSCAR_SINK_TOKENS   (default 64)
    #   OSCAR_RECENT_TOKENS (default 256)
    #   OSCAR_K_CLIP        (default 0.96)
    #   OSCAR_V_CLIP        (default 0.92)
    #   OSCAR_GROUP_SIZE    (default 128)
    from oscar_transformers import (  # noqa: F401
        OSCARCache,
        apply_rotations,
        load_rotation_file,
    )

    _OSCAR_K_PATH = os.environ.get("OSCAR_K_ROTATION_PATH")
    _OSCAR_V_PATH = os.environ.get("OSCAR_V_ROTATION_PATH")
    if not _OSCAR_K_PATH or not _OSCAR_V_PATH:
        raise EnvironmentError(
            "KV_CACHE_BACKEND=oscar requires OSCAR_K_ROTATION_PATH and "
            "OSCAR_V_ROTATION_PATH to point at the per-K and per-V rotation "
            ".pt files (download from huggingface.co/Zhongzhu/OSCAR-RotationZoo)."
        )
    _OSCAR_SINK = int(os.environ.get("OSCAR_SINK_TOKENS", "64"))
    _OSCAR_RECENT = int(os.environ.get("OSCAR_RECENT_TOKENS", "256"))
    _OSCAR_K_CLIP = float(os.environ.get("OSCAR_K_CLIP", "0.96"))
    _OSCAR_V_CLIP = float(os.environ.get("OSCAR_V_CLIP", "0.92"))
    _OSCAR_GROUP = int(os.environ.get("OSCAR_GROUP_SIZE", "128"))


# Track which attention identity (by id of the first self_attn module) we
# have already rotated. The eval swaps each ``Qwen3Attention`` for a
# ``DeltaMemAttention`` wrapper between the base and delta arms; when that
# swap happens, the layer-0 ``self_attn`` is a new Python object and we need
# to re-run apply_rotations to patch the new instances. Checking object id
# is the simplest reliable signal that "the attention has been replaced".
_OSCAR_ROTATED_ATTN_ID: int = 0
_OSCAR_K_ROT_CACHE = None
_OSCAR_V_ROT_CACHE = None


def _new_kv_cache(model):
    """Construct a fresh KV cache per the env-var selection. Returns None
    for the bf16 default, in which case the caller leaves
    `session.past_key_values` unset and the model creates a DynamicCache.
    """
    global _OSCAR_ROTATED_ATTN_ID, _OSCAR_K_ROT_CACHE, _OSCAR_V_ROT_CACHE
    if KV_CACHE_BACKEND == "bf16":
        return None
    if KV_CACHE_BACKEND == "turboquant":
        bits = KV_CACHE_BITS if KV_CACHE_BITS > 0 else 4
        return TurboQuantCache(bits=bits)
    if KV_CACHE_BACKEND in ("quanto", "hqq"):
        bits = KV_CACHE_BITS if KV_CACHE_BITS > 0 else 2
        return QuantizedCache(
            backend=KV_CACHE_BACKEND,
            config=model.config,
            nbits=bits,
        )
    if KV_CACHE_BACKEND == "oscar":
        current_attn = model.model.layers[0].self_attn
        if id(current_attn) != _OSCAR_ROTATED_ATTN_ID:
            # Either first call this process, or attach_delta_adapter_in_place
            # replaced the attention modules between arms. Re-rotate.
            if _OSCAR_K_ROT_CACHE is None:
                _OSCAR_K_ROT_CACHE = load_rotation_file(_OSCAR_K_PATH)
                _OSCAR_V_ROT_CACHE = load_rotation_file(_OSCAR_V_PATH)
            attn_kind = type(current_attn).__name__
            print(
                f"[oscar] applying rotations to {attn_kind} "
                f"(objectives k='{_OSCAR_K_ROT_CACHE.objective}' "
                f"v='{_OSCAR_V_ROT_CACHE.objective}' "
                f"head_dim={_OSCAR_K_ROT_CACHE.head_dim} "
                f"layers={len(_OSCAR_K_ROT_CACHE)})",
                flush=True,
            )
            apply_rotations(
                model, k_rotations=_OSCAR_K_ROT_CACHE, v_rotations=_OSCAR_V_ROT_CACHE,
            )
            _OSCAR_ROTATED_ATTN_ID = id(current_attn)
        bits = KV_CACHE_BITS if KV_CACHE_BITS > 0 else 2
        return OSCARCache(
            config=model.config,
            sink_tokens=_OSCAR_SINK,
            recent_tokens=_OSCAR_RECENT,
            bits=bits,
            group_size=_OSCAR_GROUP,
            k_clip=_OSCAR_K_CLIP,
            v_clip=_OSCAR_V_CLIP,
        )
    raise AssertionError("unreachable")


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
    """Compute the longest token prefix shared by ALL questions in the sample.

    `build_official_context_text` computes per-question token budgets to decide
    truncation, so even when truncation isn't triggered the tokenization
    boundary between context_text and question can shift with question length
    (e.g. the tokenizer merging the final '\\n\\n' with the first character of
    the question prompt). A common prefix computed from just two questions can
    overshoot the true shared prefix and produce a `rebuilt=True` cache miss
    on a later question — which triggers a monolithic re-prefill and OOMs on
    a 12 GB card.

    To stay safe we tokenize every question's prompt up-front and compute the
    intersection. Per-question tokenization is fast (CPU/MPS-style work, no
    GPU forward) so this is a one-time per-conversation overhead well below
    the first question's prefill cost.

    Returns None for single-question samples (no benefit from caching).
    """
    qa_items = sample["qa"]
    if len(qa_items) < 2:
        return None

    all_ids = []
    for idx, qa in enumerate(qa_items):
        _, _, ids = _build_full_prompt_tokens(
            model, tokenizer, sample, qa,
            seed=seed, question_index=idx,
            answer_reserve_tokens=answer_reserve_tokens,
        )
        all_ids.append(ids[0])

    min_len = min(int(t.shape[0]) for t in all_ids)
    base = all_ids[0][:min_len]
    common = torch.ones(min_len, dtype=torch.bool)
    for t in all_ids[1:]:
        common &= t[:min_len] == base
    # Longest contiguous-from-position-0 prefix where all match.
    if not common[0]:
        return 0
    # Find first False (divergence point).
    false_positions = (~common).nonzero(as_tuple=False)
    if false_positions.numel() == 0:
        history_len = min_len
    else:
        history_len = int(false_positions[0].item())
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
    current_attn_id = id(model.model.layers[0].self_attn)
    if sample_id in _history_kv_cache:
        existing = _history_kv_cache[sample_id]
        # Cross-arm invalidation: attach_delta_adapter_in_place replaces every
        # Qwen3Attention with a DeltaMemAttention wrapper between the base and
        # delta arms. K/V values written by the two attention classes are NOT
        # interchangeable (DeltaMemAttention adds delta-mem corrections inside
        # _apply_delta_qkv). If the cache was built under a different attention
        # identity, restoring it would mix two computation paths -> word salad
        # (see outputs/oscar_gpqacal_v3b_conv0_smoke.json: delta arm = 0.0000).
        # Evict and rebuild under the current attention class.
        if existing.get("built_under_attn_id") != current_attn_id:
            _history_kv_cache.pop(sample_id, None)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"[kv-cache invalidate sample={sample_id}] attention identity "
                f"changed (likely arm switch); evicting and rebuilding.",
                flush=True,
            )
        else:
            return existing
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

    # For non-bf16 backends, seed a backend-specific cache so the history
    # prefill flows through the same quantization the per-question prefill
    # would use. Required for OSCAR (otherwise the snapshot we capture
    # below would be a DynamicCache that the OSCAR-mode questions can't
    # consume); harmless for any backend whose cache supports snapshot.
    if KV_CACHE_BACKEND != "bf16":
        seed_cache = _new_kv_cache(model)
        if seed_cache is not None:
            session.past_key_values = seed_cache

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

    # Snapshot the cache state at the history checkpoint. For OSCAR this is
    # the alternative to ``Cache.crop()`` (INT2 group-128 boundaries don't
    # allow arbitrary-length truncation); for bf16 we still use crop, so
    # the snapshot is only kept when the cache exposes ``snapshot()``.
    pkv = session.past_key_values
    kv_snapshot = pkv.snapshot() if hasattr(pkv, "snapshot") else None

    cache_entry = {
        "disabled": False,
        "session": session,
        "history_len": history_len,
        "history_processed_ids": session.processed_input_ids.detach().clone(),
        "history_delta_state": _snapshot_delta_state(model),
        "kv_snapshot": kv_snapshot,
        # Track which attention identity (base Qwen3Attention vs. delta-arm
        # DeltaMemAttention wrapper) built this cache, so the guard at the top
        # of this function can invalidate it when the eval swaps arms.
        "built_under_attn_id": id(model.model.layers[0].self_attn),
    }
    _history_kv_cache[sample_id] = cache_entry
    return cache_entry


def _restore_session_to_history(model, session, cache_entry):
    """Roll session state back to the history-only checkpoint so the next
    question's _ingest_full_ids only processes its question suffix.

    Two restore mechanisms depending on cache class:
      * cache exposes ``restore_from`` (OSCARCache): replay the snapshot
        captured at history end. INT2 quantized middle is restored verbatim,
        avoiding the group-boundary problem that makes ``crop`` impractical.
      * cache exposes ``crop`` (DynamicCache for bf16): truncate to
        history_len in place.
    """
    history_len = cache_entry["history_len"]
    pkv = session.past_key_values
    if pkv is not None:
        if cache_entry.get("kv_snapshot") is not None and hasattr(pkv, "restore_from"):
            pkv.restore_from(cache_entry["kv_snapshot"])
        elif hasattr(pkv, "crop"):
            pkv.crop(history_len)
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

    # Cross-question cache reuse:
    #   * bf16 DynamicCache: uses Cache.crop()
    #   * OSCARCache: uses snapshot()/restore_from() — supports cross-arm
    #     reuse safely now that _ensure_conversation_cache invalidates the
    #     entry when the attention class changes between arms (see the
    #     built_under_attn_id guard above). Previously this path produced a
    #     delta-arm collapse to 0.0000 because the base-arm Qwen3Attention
    #     K/V were being restored under the delta-arm DeltaMemAttention,
    #     mixing two computation paths (outputs/oscar_gpqacal_v3b_conv0_smoke.json).
    # Other quantised backends (turboquant/quanto/hqq) still don't support
    # safe reuse.
    cache_hit_attempt = (
        not cache_entry.get("disabled", True)
        and KV_CACHE_BACKEND in ("bf16", "oscar")
    )
    if cache_hit_attempt:
        # Sanity check: the cached history must be a true prefix of this
        # prompt. If not, the cache is poisoned — fall back to a fresh
        # chunked prefill rather than letting _ingest_full_ids trigger
        # `rebuilt=True` and OOM on a 12 GB card.
        history_len = cache_entry["history_len"]
        cached_prefix = cache_entry["history_processed_ids"][0].to(prompt_ids.device)
        if (
            int(prompt_ids.shape[1]) < history_len
            or not torch.equal(prompt_ids[0, :history_len], cached_prefix[:history_len])
        ):
            print(
                f"[kv-cache MISS sample={sample['sample_id']} q{question_index}] "
                f"prompt prefix diverges from cached history at len<={history_len}; "
                f"falling back to fresh chunked prefill.",
                flush=True,
            )
            cache_hit_attempt = False

    if not cache_hit_attempt:
        # Fallback: fresh chunked prefill from scratch (no cross-question
        # reuse). Matches the non-cached patch behaviour we previously
        # validated. When KV_CACHE_BACKEND != "bf16", seed the session
        # with a fresh quantised cache so the entire chunked prefill plus
        # per-token decode forwards push K/V through the chosen backend.
        reset_delta_mem_states(model)
        session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device=device)
        session.messages = [dict(m) for m in prompt_messages]
        kv_cache = _new_kv_cache(model)
        if kv_cache is not None:
            session.past_key_values = kv_cache
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
        # Trip the safety check on the FIRST chunk too if we're trying to
        # reuse the cache — `rebuilt=True` here means the cached past_kv
        # was just thrown away and a full monolithic prefill is about to
        # run (37 GB OOM on 12 GB).
        rebuilt = stats.get("rebuilt", False)
        if rebuilt and (chunk_idx > 1 or cache_hit_attempt):
            raise RuntimeError(
                f"Chunked official prefill defeated at chunk {chunk_idx} "
                f"(cache_hit={cache_hit_attempt}): stats={stats}. "
                f"Common-prefix invariant violated."
            )

    if last_logits is None:
        # Edge case: history_len already equals total (no question suffix).
        # Force a single forward to obtain the next-token logits.
        last_logits = session._ingest_full_ids(prompt_ids)

    if last_logits is None:
        raise RuntimeError("Empty official-prompt; cannot generate.")
    if KV_CACHE_BACKEND == "bf16":
        kv_tag = ""
    elif KV_CACHE_BACKEND == "turboquant":
        kv_tag = f" tq{KV_CACHE_BITS or 4}"
    else:
        kv_tag = f" {KV_CACHE_BACKEND}{KV_CACHE_BITS or 2}"
    print(
        f"[official-prefill q{question_index} sample={sample['sample_id']}{kv_tag}] "
        f"suffix_chunks={chunk_idx} ingest_start={ingest_start} total={total} "
        f"final_chunk_ms={stats.get('elapsed_ms', '?')}",
        flush=True,
    )

    rng_devices = []
    if torch.cuda.is_available() and device.startswith("cuda"):
        rng_devices = [torch.device(device)]
    # Freeze delta-mem state during answer decode. The dominant per-token
    # cost on the delta arm is _memory_affine_scan_torch (a Python for-loop
    # over generated tokens, ~9 small CUDA launches per layer per token);
    # delta-mem's *state* represents what is being remembered from the
    # prompt, not what the model is generating, so freezing on the answer
    # tokens is paper-safe. The vendored _batched_generate_raw_predictions
    # toggles this for base mode but our chunked path forgot to. Restored
    # in a try/finally so an exception path doesn't leave the model in a
    # half-frozen state.
    from deltamem.core.delta_impl import set_delta_mem_write_enabled
    set_delta_mem_write_enabled(model, False)
    try:
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
    finally:
        set_delta_mem_write_enabled(model, True)

    raw_prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    canonical_prediction = canonicalize_locomo_prediction(raw_prediction, question_spec)
    return raw_prediction, canonical_prediction


eval_mod.generate_official_full_history_answer = _chunked_official_full_history_answer


# Re-shape sys.argv so the eval's argparse sees the expected program name
# rather than "run._chunked_eval_runner". Cosmetic but cleaner help output.
sys.argv[0] = "deltamem.eval.locomo_delta"
sys.exit(eval_mod.main() or 0)
