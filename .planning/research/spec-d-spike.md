# Speculative Decoding Spike — Wiring HF `assisted_generation` into the OSCAR + delta-mem decode loop

**Status:** Research only (no source changes).
**Target model:** Qwen3-4B-Instruct-2507 (bf16, single GPU).
**Draft model candidate:** Qwen3-0.6B (same tokenizer family).
**Stack:** `transformers==5.9.0`, OSCAR INT2 KV cache (`third_party/oscar-transformers/oscar_transformers/cache.py`), `DeltaMemChatSession` (`delta-Mem/deltamem/runtime/session.py`).
**Recommendation up front:** **no-go in the 48 h window** for the assistant-model path on top of OSCAR. A targeted *prompt-lookup* spec variant is the only spec-D flavour that is cheap enough to fit, and even that is a stretch given the Tier 1 wins already on the board.

---

## 1. What the existing decode loop actually does

`DeltaMemChatSession._decode_generate` (`delta-Mem/deltamem/runtime/session.py:601-681`) is a single-token autoregressive loop:

```
next_token = sample(next_token_logits)
for _ in range(max_new_tokens):
    if next_token == eos: break
    set_delta_mem_write_message_ids(model, [assistant_message_id])
    set_delta_mem_write_sentence_ids(model, [assistant_sentence_id])
    outputs = model(input_ids=next_token,
                    past_key_values=self.past_key_values,
                    use_cache=True)
    self.past_key_values = outputs.past_key_values
    next_token = sample(outputs.logits[:, -1, :])
```

Key contract observations:

* **No `position_ids` or `cache_position` are passed** — the model fills them from the cache state on its own. This works because `past_key_values.get_seq_length()` is monotonically increasing and accurate (OSCAR includes `sink + middle + recent` in `get_seq_length()`; see `cache.py:82-89`).
* **Per-token delta-mem write hooks** are toggled with the runtime message id / sentence id. These are *frozen* during answer decode in the chunked runner (`run/_chunked_eval_runner.py:639-640` sets `set_delta_mem_write_enabled(model, False)` for the whole decode), so for speculative purposes we can pretend the side-effect machinery is inert — but it's still attached.
* **EOS check is `next_token.item() == eos_id`** — a single integer per step. Multi-token verification would need this rewritten as a vector predicate.
* **`processed_input_ids`** is the CPU-resident "everything the cache has seen" record. It's how `_ingest_full_ids` computes the next call's prefix overlap. Any spec-D loop must keep this in sync with the cache, otherwise the *next question* in the LOCOMO eval will hit the rebuild path and OOM (see the safety check at `_chunked_eval_runner.py:599-605`).

`_ingest_full_ids` (lines 515-567) already supports **multi-token forwards** through the same cache. OSCAR's `update()` (`cache.py:109-174`) handles `n_new > 1` natively — sink fill, recent FIFO append, group-128 INT2 spill of the overflow chunk. So *forward verification of K candidate tokens in one shot* is mechanically supported by both the session and the cache today.

The hard problem is **rejection rollback.**

---

## 2. What HF `_assisted_decoding` actually requires from the cache

`transformers/generation/utils.py:3462-3772` is the real method (5.x — no rename; still `_assisted_decoding`). The relevant lifecycle per iteration:

1. `candidate_generator.get_candidates(input_ids)` — `AssistedCandidateGenerator` (`generation/candidate_generator.py:80-340`) runs `assistant_model.generate(...)` for K tokens. The assistant keeps its own cache in `assistant_kwargs["past_key_values"]`.
2. Target runs a single multi-token forward over `candidate_input_ids` (line 3621). `prepare_inputs_for_generation` slices off the part already in cache; it needs the cache's `get_seq_length()` to be accurate.
3. Verification (lines 3636-3661): either Algorithm 1 from the spec-D paper (sample mode) or argmax-comparison (greedy mode) — produces `n_matches` accepted tokens.
4. **Line 3675 (unconditional):** `outputs.past_key_values.crop(new_cur_len - 1)`.
5. Inside `AssistedCandidateGenerator._update_past_and_masks` (line 299): **`self.assistant_kwargs["past_key_values"].crop(...)`** on the *assistant* cache too.
6. Line 192 forces `cache_implementation = "dynamic_full"` for the assistant. Line 3522-3526 forbids `StaticCache`. The cache class itself isn't validated against an allow-list, but `Cache.crop()` is the only API the rollback path uses.

`Cache.crop()` is defined in `cache_utils.py:1173-1176` and delegates to `DynamicLayer.crop()` at line 163-175:

```python
def crop(self, max_length: int) -> None:
    if max_length < 0:
        max_length = self.get_seq_length() - abs(max_length)
    if self.get_seq_length() <= max_length:
        return
    self.keys = self.keys[..., :max_length, :]
    self.values = self.values[..., :max_length, :]
```

That is: **`crop` is a hard requirement of the published spec-D path, called every iteration, with an arbitrary `max_length` that is `target_seq_len - 1 - (K - n_matches)`.** It will be off a group-128 boundary almost every step.

### Does OSCAR support `crop`?

No, and the file says so: `cache.py:23-25` —

> The cache is not compatible with `Cache.crop` ... Crop is disabled for non-bf16 backends in `run/_chunked_eval_runner.py`; OSCAR is no exception.

The reason is structural: per-token INT2 codes are packed into group-128 blocks with their own scale/zero. Truncating mid-block requires (a) dequantizing the tail block, (b) discarding the rejected tokens, (c) re-quantizing — and the re-quantized scale/zero will not match what the model saw in *prior* forwards. There is no per-token entry point that produces a bit-exact roll-back. The closest thing OSCAR offers is `snapshot()` / `restore_from()` (`cache.py:193-265`), which is what the chunked runner already uses for cross-question reuse.

---

## 3. Concrete options and their patches

### Option A — Just call `model.generate(..., assistant_model=ass)` from the session

This is the "obvious" option and the one to reject first, because it interacts badly with **everything** we already built:

* `generate()` builds its own `past_key_values` unless we pass one in `model_kwargs`. We *can* pass `session.past_key_values`, but step 4 above will call `.crop()` on it and explode the moment the cache is OSCAR.
* `generate()` doesn't know about delta-mem write hooks. The hooks fire on every forward; with writes frozen during decode that's fine, but the multi-token verify forward writes to the same hook context — still fine because we'd toggle `set_delta_mem_write_enabled(model, False)` around the call.
* `generate()` will also re-tokenize via the chat template if we let it. We bypass that by passing `input_ids` directly.
* Critically, it returns `sequences` but does **not** give us `processed_input_ids` updates in the form the session caches. We'd have to overwrite `session.processed_input_ids` from `sequences` and trust that the cache state matches — which is fine for bf16, broken for OSCAR.

**Verdict for OSCAR:** A is dead unless we wrap the cache in a "crop = snapshot-restore" shim. See Option C.
**Verdict for bf16:** A is the trivial 30-line patch, and the only sensible answer for the bf16 arm. Patch sketch:

```python
# session.py, new method
def decode_with_assistant(self, assistant_model, max_new_tokens, *,
                          do_sample, temperature, top_p, top_k, eos_id):
    from deltamem.core.delta_impl import set_delta_mem_write_enabled
    input_ids = self.processed_input_ids.to(self.device)
    set_delta_mem_write_enabled(self.model, False)
    try:
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                past_key_values=self.past_key_values,
                assistant_model=assistant_model,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample, temperature=temperature,
                top_p=top_p, top_k=top_k,
                use_cache=True, return_dict_in_generate=True,
                pad_token_id=eos_id,
            )
    finally:
        set_delta_mem_write_enabled(self.model, True)
    self.past_key_values = out.past_key_values
    new_ids = out.sequences[:, input_ids.shape[1]:].cpu()
    self.processed_input_ids = torch.cat([self.processed_input_ids, new_ids], dim=1)
    return new_ids
```

Call-site change in `run/_chunked_eval_runner.py:646-653`: branch on `KV_CACHE_BACKEND == "bf16"` and `os.getenv("ASSISTANT_MODEL_PATH")` — if both set, use `session.decode_with_assistant(...)`, else the existing `session._decode_generate(...)`. ~10 lines.

### Option B — Hand-rolled propose/verify inside `_decode_generate`, skipping `generate()` and `crop()` entirely

This is the "ours, not theirs" route. Sketch (intentionally rough; treat the code as pseudocode for review, not a diff to apply):

```python
def _decode_generate_speculative(
    self, next_token_logits, max_new_tokens, *,
    assistant_session,           # a second DeltaMemChatSession-lite around the draft
    K=4,                          # candidates per round
    do_sample, temperature, top_p, top_k,
):
    eos_id = self.tokenizer.eos_token_id
    accept_count = 0
    proposed_count = 0
    next_token = self._sample_next_token(next_token_logits, ...)

    while accept_count < max_new_tokens:
        # 1. Snapshot OSCAR cache + delta-mem state BEFORE the spec window.
        target_snap = self.past_key_values.snapshot() \
                      if hasattr(self.past_key_values, "snapshot") else None

        # 2. Propose K tokens from the draft (greedy or sampled).
        draft_tokens, draft_logits = assistant_session.propose(next_token, K)

        # 3. One target forward over [next_token, *draft_tokens[:-1]] (K tokens).
        cand = torch.cat([next_token, draft_tokens[:, :-1]], dim=1)  # (1, K)
        with torch.inference_mode():
            out = self.model(input_ids=cand,
                             past_key_values=self.past_key_values,
                             use_cache=True, return_dict=True)
        target_logits = out.logits[:, :, :]   # (1, K, V)
        self.past_key_values = out.past_key_values

        # 4. Compare. Greedy verification = first mismatch wins.
        target_choices = target_logits.argmax(dim=-1)  # (1, K)
        match_mask = (target_choices[0, :-1] == draft_tokens[0, :-1])
        n_matches = int((~match_mask).cumsum(dim=0).eq(0).sum())  # 0..K-1

        accepted = draft_tokens[:, :n_matches]            # the matched draft tokens
        bonus    = target_choices[:, n_matches:n_matches+1]  # 1 free target token

        # 5. ROLLBACK: cache currently holds K tokens, we want only n_matches+1.
        #    OSCAR can't crop. Restore the pre-window snapshot, then replay only
        #    the accepted prefix as a multi-token forward.
        if hasattr(self.past_key_values, "restore_from"):
            self.past_key_values.restore_from(target_snap)
            keep = torch.cat([cand[:, :n_matches], bonus], dim=1)  # (1, n_matches+1)
            with torch.inference_mode():
                replay = self.model(input_ids=keep,
                                    past_key_values=self.past_key_values,
                                    use_cache=True, return_dict=True)
            self.past_key_values = replay.past_key_values
            next_logits = replay.logits[:, -1, :]
        else:
            # bf16 DynamicCache: crop wins, 1 op vs. K replay tokens.
            self.past_key_values.crop(self.past_key_values.get_seq_length()
                                      - (cand.size(1) - n_matches - 1))
            next_logits = target_logits[:, n_matches, :]

        accept_count += int(accepted.size(1)) + 1
        ... # advance processed_input_ids, sample next_token from next_logits,
            # also drive assistant_session's cache forward by the accepted tokens.
        if accepted/bonus contains eos: break
```

Critical detail buried in step 5: **with OSCAR, every spec round pays a `snapshot()` and a `restore_from()`.** Snapshot copies the entire cache to CPU (see `OSCARCacheLayer.snapshot`, `cache.py:193-234` — `.detach().to("cpu").clone()` on `sink_k/v`, `recent_k/v`, `middle_k/v` codes/scale/zero, **and** `_middle_k_dq/_middle_v_dq` which is 2× the middle's VRAM). Restore goes the other way. For a 6-7 k token context with 36 layers × 8 KV heads × 128 head_dim in bf16, the snapshot is ~150-250 MB and the round-trip is dominated by PCIe bandwidth. **Per spec round, that is likely 50-200 ms** — i.e., wiping out any acceptance gain unless K is very large and the acceptance rate is very high.

The only way to make Option B work for OSCAR is to keep snapshots GPU-resident. That requires a new `snapshot_gpu()` variant — 4-6 lines per region, ~30 lines total in `cache.py`. Even then, the VRAM cost is real: on a 12 GB card with a baseline working set already at ~7-8 GB during decode, an extra ~150 MB snapshot per round is survivable; an extra `_middle_k_dq` shadow held across the round is also already there. So this is the *plausible* OSCAR shape, not the obvious one.

### Option C — Wrap OSCARCache to fake `crop` via snapshot/restore

A minimal `CroppableOSCARCache(OSCARCache)` that:

* on entry to spec-D decode, takes one snapshot (`self._spec_snapshot = self.snapshot()`) and records `self._spec_anchor_len = self.get_seq_length()`
* implements `crop(max_length)` as: if `max_length >= _spec_anchor_len`, restore from `_spec_snapshot`, then re-run a tiny multi-token forward to bring it up to `max_length`. If `max_length < _spec_anchor_len`, raise.

This is what makes Option A work for OSCAR. The wrapper is ~40 lines. But it still pays a full multi-token replay per spec round (whatever tokens were accepted), exactly as Option B does. There is no escape from the fact that we cannot truncate INT2 group-128 blocks in place.

**The hidden trap with Option C:** HF's loop calls `crop` *after* the target forward, then expects `prepare_inputs_for_generation` on the next iteration to feed only one new token. If our `crop` actually does a replay forward, the next iteration's target forward will *re-replay* the same tokens because `prepare_inputs_for_generation` uses cache length vs. input length to decide. We'd need to make `crop` a no-op that records the desired length, and `update` (called during the next target forward) would consult that length and replay then. This is getting into "we are now maintaining a fork of `assisted_generation`" territory.

---

## 4. The cheap escape: `PromptLookupCandidateGenerator`

`generation/candidate_generator.py:1019-1172` implements **prompt-lookup decoding**: candidates are pulled by n-gram match from the prompt itself, no draft model, no draft cache. It is invoked via `generation_config.prompt_lookup_num_tokens` (`generation/utils.py:979-987`).

The verification path is **the same `_assisted_decoding` method**, so it still calls `outputs.past_key_values.crop(...)` at line 3675. So PLD on OSCAR has the same blocker as assistant-model spec-D — `crop` is required.

PLD on bf16 is trivial (Option A with `prompt_lookup_num_tokens=10` instead of `assistant_model=...`). For LOCOMO-style long conversation prompts with lots of verbatim repetition (names, dates, facts), PLD can hit 30-50% acceptance with zero draft cost. **If you take any spec-D path in 48 h, take this one, and only for the bf16 arm.**

---

## 5. Specific file changes summary

| Change | File | Lines (approx) | Risk |
|---|---|---|---|
| Add `decode_with_assistant(...)` (Option A bf16) | `delta-Mem/deltamem/runtime/session.py` | +30 after line 681 | LOW — re-uses `generate()`, only bf16 path |
| Add `decode_with_prompt_lookup(...)` (PLD bf16) | `delta-Mem/deltamem/runtime/session.py` | +25 after line 681 | LOW — same shape, no draft model |
| Branch in chunked runner on `KV_CACHE_BACKEND == "bf16"` and an env var | `run/_chunked_eval_runner.py` near line 646 | +10 | LOW |
| Hand-rolled spec-D for OSCAR (Option B, GPU snapshot) | `delta-Mem/deltamem/runtime/session.py` + `third_party/oscar-transformers/oscar_transformers/cache.py` | +120 in session, +30 in cache | HIGH — multi-token replay path is a new code path; delta-mem hooks behaviour under multi-token write must be re-verified; OSCAR group-128 spill behaviour under repeated short replays is untested |
| OSCAR `crop`-via-snapshot wrapper (Option C) | new file `delta-Mem/deltamem/runtime/oscar_crop_shim.py` | +60 | HIGH — interacts with `prepare_inputs_for_generation`; effectively a partial fork of HF's spec-D loop |

The "smallest set of changes" for a useful experiment is the bf16 + PLD path: ~35 lines, no cache changes, no model surgery, runs against the LOCOMO eval as a parallel arm.

---

## 6. Cache-shape verification of OSCAR vs. multi-token forwards

OSCAR is confirmed multi-token-safe for the *forward* path:

* `OSCARCacheLayer.update` (cache.py:109-174) takes `n_new = key_states.shape[2]` and processes any value. It handles the three regions independently and only spills when `recent > recent_tokens`.
* `get_seq_length` (cache.py:82-89) is the simple sum, accurate after a K-token update.
* `_assemble` (cache.py:176-191) is called once per `update`, returns the full slab needed for the multi-token attention.

What it is **not** safe against is *partial undo*. Once `update` has packed N tokens into a new INT2 block (because `recent` overflowed during the multi-token forward), there is no operation in `cache.py` that can unpack a subset. Even `restore_from` works only at the snapshot granularity — there is no "snapshot every step" mode, and adding one would mean holding K snapshots per spec round at ~150-250 MB each.

---

## 7. Go / no-go in the 48 h window

Existing Tier 1 wins already on the board (per the previous research synthesis: `R_v.T` bake, snapshot/restore, frozen delta writes during decode, the assemble-cache dequant amortization): these are measured, paper-safe, and they all stack with bf16 baseline today. The question is whether spec-D adds meaningful additional throughput *for the eval that matters* (LOCOMO `conv-0`, official answer prompts, ~6-7 k token contexts, ≤512 new tokens per answer).

**OSCAR + assistant-model spec-D: NO-GO.** The crop incompatibility is structural, every workaround requires either CPU-bounce snapshots (kills the gain) or a hand-rolled verify path with multi-token replay (large new surface area, delta-mem hooks need re-validation, group-128 spill under short replays is untested). Even with everything working, the per-round replay cost likely eats most of the speed-up at K=4-6.

**bf16 + prompt-lookup decoding: YES, IF you have a free afternoon.** ~35 line patch (Option A shape, with `prompt_lookup_num_tokens=10` instead of `assistant_model=...`). LOCOMO answers reference verbatim spans of the conversation history constantly (names, dates, the question text itself). PLD pays nothing for drafting and is exactly the regime it was designed for. Worst case it matches baseline; expected case is 1.3-1.8× on the answer-decode phase, which is ~30-40% of the per-question wall clock.

**bf16 + Qwen3-0.6B as assistant: MARGINAL.** Same Option A patch, plus loading a second model (~1.2 GB bf16). Qwen3-0.6B as draft for Qwen3-4B will likely hit decent acceptance (same family, same tokenizer, same instruction style), but the 0.6B forward per K-token round is real cost and the 4B target is small enough that the speed-up may be muted (the literature suggests spec-D gains are largest when target ≫ draft, e.g., 70B/7B). On a 12 GB card the VRAM headroom is also tight: 4B bf16 (~8 GB) + 0.6B bf16 (~1.2 GB) + bf16 KV cache for 6-7 k tokens (~1 GB) + activations.

**Final recommendation:** if the next 48 h has a "spare half day" between Tier 1 ship and the OSCAR conv-0 eval, **try PLD on the bf16 arm only.** Skip the assistant-model path. Defer OSCAR spec-D to a separate workstream that can afford to fork or co-design with `_assisted_decoding`. The OSCAR work to make `crop` real is at least a week of careful engineering and validation, and the payoff is bounded by snapshot/restore latency we don't yet have measured.

---

## 8. References (file:line)

* Decode loop: `delta-Mem/deltamem/runtime/session.py:601-681`
* Multi-token prefill: `delta-Mem/deltamem/runtime/session.py:515-567`
* Chunked decode call-site: `run/_chunked_eval_runner.py:646-653`
* OSCAR snapshot/restore: `third_party/oscar-transformers/oscar_transformers/cache.py:193-330`
* OSCAR crop incompatibility note: `third_party/oscar-transformers/oscar_transformers/cache.py:23-25`
* HF `_assisted_decoding`: `.venv/Lib/site-packages/transformers/generation/utils.py:3462-3772`
* Target cache crop (unconditional): `.venv/Lib/site-packages/transformers/generation/utils.py:3675`
* Assistant cache crop: `.venv/Lib/site-packages/transformers/generation/candidate_generator.py:299`
* Cache impl forced to dynamic_full: `.venv/Lib/site-packages/transformers/generation/candidate_generator.py:192`
* `Cache.crop` definition: `.venv/Lib/site-packages/transformers/cache_utils.py:1173-1176`
* `DynamicLayer.crop` (the actual tensor slice): `.venv/Lib/site-packages/transformers/cache_utils.py:163-175`
* `PromptLookupCandidateGenerator`: `.venv/Lib/site-packages/transformers/generation/candidate_generator.py:1019-1172`
* `_get_candidate_generator` (PLD dispatch): `.venv/Lib/site-packages/transformers/generation/utils.py:979-987`
