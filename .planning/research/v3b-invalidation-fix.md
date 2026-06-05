# v3b cross-arm OSCARCache invalidation fix

## What changed

`run/_chunked_eval_runner.py` only:

1. `_ensure_conversation_cache` now records `built_under_attn_id = id(model.model.layers[0].self_attn)` on every newly-built `cache_entry`.
2. A guard at the top of `_ensure_conversation_cache` evicts and rebuilds the entry whenever the current `model.model.layers[0].self_attn` identity differs from the recorded one (i.e., the eval has swapped between the base `Qwen3Attention` and the delta-arm `DeltaMemAttention` wrapper via `attach_delta_adapter_in_place`).
3. `cache_hit_attempt` is re-enabled for `KV_CACHE_BACKEND == "oscar"` (previously tightened to bf16-only after the v3b collapse). The bf16 + oscar tuple is the safe set; turboquant/quanto/hqq still excluded.

`third_party/oscar-transformers/oscar_transformers/cache.py` and `delta-Mem/` were left untouched per constraint.

## Why it should work

The v3b failure (`outputs/oscar_gpqacal_v3b_conv0_smoke.json`: delta arm = 0.0000) was caused by the base arm filling the `OSCARCache` using raw Qwen3 K/V projections, then the delta arm restoring that snapshot under a `DeltaMemAttention` wrapper that injects delta-mem corrections in `_apply_delta_qkv`. The cache then carried K/V from two different computation paths, producing word salad. The identity-based guard catches the only moment that mismatch can be introduced (the in-place attention swap between arms) and forces a clean rebuild on the delta arm, after which snapshot/restore is internally consistent within the arm — which is exactly what `OSCARCacheLayer.snapshot()` / `restore_from()` were designed for.

## What to test next

- `python -m run.oscar_port_debug` for a coarse OSCAR-pipeline sanity check (~30 min on the 3060). It does not exercise delta-mem or cross-arm logic — it only confirms the rotation/quant pipeline itself didn't regress.
- The real validation is a fresh `conv-0/10q` LoCoMo eval with `KV_CACHE_BACKEND=oscar`: expect the delta arm to recover to a non-zero F1 comparable to base, with cache-hit reuse re-engaged (look for "kv-cache invalidate" log lines on the arm transition and absence of `cache_hit=False` on q1+).
- A full conv-0 / both-arms run will surface any subtle differences between the snapshot-replay path and a fresh prefill for OSCAR; compare base-arm F1 against the v3b reference (0.2379) to confirm no regression from re-enabling reuse.

## Subtle risks

The guard keys on `id(model.model.layers[0].self_attn)` only, so if a future change rewraps attention on a per-layer basis or recreates a wrapper object that happens to land at the same Python id after GC (rare but possible in long-lived processes), the guard could either over-evict or, worse, fail to evict — worth tightening to a tuple over all layers if we ever see anomalies.
