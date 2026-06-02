# 48h optimization plan — synthesized from research

Generated 2026-06-02 from four parallel research subagents. Bottom line:
six implementable wins identified, ranked by estimated wall-time impact on
the conv-0/10q LoCoMo eval (currently ~9.5 h end-to-end).

## Ranked priority list

Each entry: estimated wall-time saving, code size, risk, dependencies on
other entries in this list.

### Tier 1 — biggest absolute wins, low risk

1. **OSCAR snapshot/restore for cross-question KV reuse**
   - Source: `delta-mem-speedups.md` Q5
   - Saves: ~3-4 h on 10 q eval. Currently each question re-prefills 17.6 k
     history tokens (`_chunked_eval_runner.py:514-516` disables reuse for any
     non-bf16 backend). Snapshot the assembled OSCARCacheLayer at end-of-
     history and restore + extend for each new question.
   - Why snapshot, not crop: INT2 group-128 boundaries break arbitrary-length
     truncation. Snapshot (sink + middle codes/scale/zero + middle dq cache +
     recent) and restore preserves the boundary structure.
   - Code: ~80 LOC in `oscar_transformers/cache.py` + runner hookup.
   - Risk: low — cache state is in self-contained tensors, deep-copy is
     mechanical.

2. **Bake R_v.T into o_proj.weight** (rotation absorption)
   - Source: `rotation-fusion.md` §2
   - Saves: maybe 10-20% of decode time per layer × 36 layers; eliminates
     the heaviest runtime einsum (full 32 heads, not the 8 kv-heads).
   - Math: `o_proj(R_v.T @ attn_rot) = (W_o @ R_v.T) @ attn_rot`. Bake once,
     never un-rotate at runtime. Delta-mem compatible because `delta_o` is
     added AFTER `base.o_proj` (`delta_impl.py:2280-2294`), so it already
     lives in the un-rotated basis.
   - Code: ~30 LOC modification to `apply_rotations` + removal of the
     `_OSCARUnrotatingOProj` wrapper.
   - Risk: low — pure linear algebra. Validate with a needle-smoke before
     committing.

3. **Freeze delta-mem writes during answer decode**
   - Source: `delta-mem-speedups.md` Q4
   - Saves: erases the `_memory_affine_scan_torch` per-token cost (a Python
     `for token_idx in range(seq_len)` loop with 9 small CUDA launches per
     layer per token — dominant per-decoded-token cost on delta arm).
   - Mechanism: wrap `session._decode_generate` with
     `set_delta_mem_write_enabled(False)` / restore-True. Vendored eval
     already does this for base mode; our chunked runner forgot to.
   - Code: ~5 LOC wrapper in `_chunked_eval_runner.py`.
   - Risk: low. Paper framing: state is what's *remembered*, not what's
     generated; freezing answers is semantically correct.

### Tier 2 — speed wins requiring more code

4. **Pre-allocate `_assemble` output buffers**
   - Source: `rotation-fusion.md` §5a
   - Saves: maybe 20-30% of decode time. `torch.cat` of sink+middle+recent
     on every decode allocates and copies ~1.26 GB across 36 layers at 17 k
     context. Pre-allocate `(B, H_kv, max_seq, D)` buffers and return
     `narrow()` views.
   - Code: ~40 LOC in `cache.py`.
   - Risk: medium — buffer lifecycle (when to grow, when to invalidate the
     view after a spill).

5. **Install Triton on Windows host → flip scan_impl to triton**
   - Source: `delta-mem-speedups.md` Q2
   - Saves: 30-50% on prefill+decode for delta arm. The `torch` pin in
     `report/kernels-gate.md:71-74` was purely environmental (Triton
     unavailable on Windows at the time).
   - Mechanism: `pip install triton` (or `triton-windows`); unset
     `DELTA_MEM_SCAN_IMPL=torch`; "auto" picks Triton.
   - Code: 0 LOC; only env / install.
   - Risk: medium — Triton on Windows is improving but may still hit
     compilation issues with delta-mem's specific kernels. Worth trying.

### Tier 3 — quality/memory trade-offs

6. **INT4 instead of INT2**
   - Source: `int4-vs-int2.md`
   - Quality: near-bf16 across all rotation schemes (KIVI-4, Saw-INT4,
     QuaRot-INT4 all ≤0.5 PPL hit). OSCAR's eigenbasis rotation is
     INT2-specific; **at INT4 the rotation is likely overkill**.
   - Memory: +312 MB at 17 k tokens × 36 layers vs INT2. Still vastly less
     than bf16 raw.
   - Code: `--kv-cache-bits 4` already wired.
   - Risk: low. Two A/B variants: INT4 + existing rotation, and raw INT4
     (no rotation, simplest path).

### Tier 4 — bigger projects, only if time

7. **Speculative decoding with Qwen3-0.6B as draft model**
   - Source: `speculative-decoding.md`
   - Speedup: 1.7-2.1x ceiling on 3060 12 GB. **Stacks additively** with
     dequant fast-path (different memory-budget terms).
   - Code: ~100-200 LOC. OSCARCache already accepts multi-token writes
     (verified by reading cache.py); only `crop()` for rejection rollback
     is missing, ~30 LOC. Wire `Qwen/Qwen3-0.6B` as `assistant_model=`
     in `model.generate()`.
   - Risk: medium-high. HF assisted generation may not interoperate cleanly
     with OSCARCache; would need a custom verification loop if it doesn't.

## Execution plan for the 48 h window

Assumes the current v2 eval (started 16:21, finish ~02:00 local tomorrow)
lands first.

**Hour 0-6 (now → eval finish)**: research done. Eval running. Begin
implementing Tier 1 items that don't touch the live OSCARCache code path:
- Item 2 (bake R_v.T into o_proj) — affects `apply_rotations`, but only
  reads at process startup; existing eval already imported the old version
  into memory so disk-level changes don't disturb it. Safe to develop now.
- Item 3 (freeze delta writes during decode) — modifies the runner; running
  eval has already started its subprocess, so changes here also safe.

**Hour 6-12**: eval lands. Commit final result + reproduction-report
update. Write up base / delta scores. Then implement and unit-smoke Tier 1
items 1 (snapshot/restore) and 4 (pre-alloc buffers).

**Hour 12-22**: kick off conv-0/10q v3 with all Tier 1 + 2 fixes in place.
Estimated wall ~3-4 h thanks to snapshot/restore + rotation fusion + frozen
delta writes. While it runs, prepare INT4 trial branch.

**Hour 22-30**: v3 lands. Commit. Kick off v4 at INT4 (item 6). Quick A/B
on the same conv-0/10q.

**Hour 30-40**: try Triton install (item 5). If successful, kick off v5.

**Hour 40-48**: documentation pass. Update `report/tier1-summary.md` with
appendices D-F covering OSCAR-on-Instruct calibration, the continuous-zero
fix, snapshot/restore + rotation fusion wins, and the INT4 result. Commit
all. Final summary in the report.

If everything completes faster, escalate to speculative decoding (item 7).

## What we are NOT doing in this window

- **Custom Triton kernels for the rotation/dequant fast paths.** Upstream
  OSCAR's fused kernels are SGLang-paged-cache-specific and not portable.
  Would consume the full 48 h alone without a guaranteed win.
- **Re-calibrating rotations on more data.** GPQA-cal worked. The
  diminishing returns are not worth the dump+compute time.
- **Switching base model.** Sticking with Qwen3-4B-Instruct-2507 for
  reproducibility and apples-to-apples vs the Tier 1 reproduction.
- **Re-training delta-mem with OSCAR-aware K/V.** Multi-day training run on
  a 12 GB card is not realistic.
