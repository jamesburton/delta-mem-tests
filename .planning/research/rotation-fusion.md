# OSCAR Rotation Fusion — Research Report

**Target model:** Qwen3-4B-Instruct-2507 (`head_dim=128`, `num_heads=32`,
`num_kv_heads=8`, `num_layers=36`, hidden_size=2560 → q_proj 2560×4096, k_proj
& v_proj 2560×1024, o_proj 4096×2560).

**Current cost:** four einsums per layer per forward — Q-rotate, K-rotate,
V-rotate, output un-rotate. Per token at decode that is
`4 × (heads × head_dim²)` FMAs/layer ≈ `4 × 32 × 128² × 36` ≈ 75 MFMA per
token (Q/O) plus `2 × 8 × 128² × 36` ≈ 9 MFMA (K/V). The dominant cost is the
Q and O einsums because they hit the full 32 heads.

---

## 1. Bake `R_v` into `v_proj.weight` — VALID, low-effort, ~25% of overhead

The V path is `value = v_proj(x).view(B, T, H_kv, D).transpose(1,2)` followed
by `value = einsum("bhtd,de->bhte", value, R_v)`. There is no q_norm /
RoPE-style mixing on V. Linearity of `v_proj` gives:

```
v_proj(x) reshaped as (..., H_kv, D), rotated by R_v on D
   = reshape(x @ W_v^T, ..., H_kv, D) @ R_v
   = x @ (W_v^T @ block_diag_per_head(R_v))           (math is exact, fp32)
```

Concretely: reshape `v_proj.weight` to `(H_kv, D, in_features)` (PyTorch
stores `Linear.weight` as `(out, in)` so out=H_kv·D), pre-multiply the D-axis
by `R_v` per head — `W_v_new[h] = R_v.T @ W_v[h]` (because `Linear` does
`y = x @ W.T`, the rotation lands on the right of W so on the left of W.T).

**Perf:** removes the `einsum("bhtd,de->bhte", v, R_v)` entirely on every
forward (prefill *and* decode). On Qwen3-4B that's 36 × (B×T×8×128²) FMAs
saved per call. Decode-time saving per token ≈ 36 × 8 × 16384 ≈ 4.7 MFMA;
small in absolute terms vs. attention, but the einsum itself launches a
kernel and forces a contiguous copy — eliminating the kernel launch is the
real win on small-batch decode.

**Delta-mem compatibility:** safe. Delta-mem's `_normalize_value_states`
runs *before* the rotation in the patch ordering, so baking into `v_proj.weight`
is equivalent — V flows out of `v_proj` already rotated, then through the
norm (which here is `nn.Identity` for Qwen3, see `delta_impl.py` —
`_normalize_value_states` is a passthrough on Qwen3, only SmolLM3 puts a
norm there). Verify by reading `_normalize_value_states` for the target
model before enabling.

**Recommendation:** add a `bake_v_rotation_into_proj(model, v_rotations)`
helper to `rotation.py` that mutates `v_proj.weight` in-place, sets a
sentinel `attn._oscar_v_baked = True`, and makes both `patched_forward` and
`patched_normalize_value_states` skip the V einsum when the sentinel is set.

---

## 2. Bake `R_v.T` into `o_proj.weight` — VALID, biggest single win

The math: `o_proj(R_v.T @ attn_output) = attn_output @ R_v @ W_o.T`. So set
`W_o_new = W_o @ R_v.T` per head along the D axis (head_dim is contiguous in
the flattened H·D input, so reshape `o_proj.weight` to `(out, H, D)` and
right-multiply each head's D axis by `R_v`):

```
W_o.shape = (hidden, H*D)        # H = 32, D = 128, so (2560, 4096)
W_o_view  = W_o.view(hidden, H, D)
W_o_new   = einsum("ohd,de->ohe", W_o_view, R_v).view(hidden, H*D)
o_proj.weight = W_o_new
```

This **completely absorbs** the un-rotation. On Qwen3 the o-side einsum is
the most expensive of the four (operates on the full 32 heads, not 8), so
this is the highest-leverage single optimization. It also removes the
`reshape → einsum → reshape().contiguous()` round-trip, which on bf16 is
~12 MB of memory traffic per call at T=4096.

**Delta-mem compatibility — IMPORTANT NUANCE.** From `delta_impl.py:2280–2294`:

```
attn_output = attn_output.reshape(*input_shape, -1).contiguous()
base_o_output = self.base.o_proj(attn_output)        # ← un-rotation site
...
attn_output = base_o_output + delta_o_typed
```

The delta-mem path applies `base.o_proj` to the rotated `attn_output`, then
**adds** `delta_o_typed`. Because `delta_o` is computed by a separate
delta-state module and is NOT in the rotated basis, baking `R_v.T` into
`o_proj.weight` works *only because the addition happens after o_proj*. The
identity is:

```
(W_o @ R_v.T) @ attn_rot + delta_o
  = W_o @ (R_v.T @ attn_rot) + delta_o
  = W_o @ attn_unrot + delta_o                    # ✓ same as current path
```

So baking is correct **iff** `delta_o` continues to be added downstream of
o_proj in the original (un-rotated) hidden basis, which it is. The current
`_OSCARUnrotatingOProj` wrapper would become a no-op pass-through to the
baked linear; you can simply re-assign `attn.base.o_proj` back to the
(modified) original `nn.Linear` and the addition path is untouched.

**Recommendation:** bake `R_v.T` into `o_proj.weight` in `apply_rotations`
when the user opts in (`bake_into_proj=True`). This kills both the
`reshape→einsum→reshape→contiguous` chain AND the wrapper Module overhead
(one Python-level forward call per layer per token at decode is non-trivial
in eager mode).

**Combined #1 + #2 savings:** eliminates 2 of 4 einsums entirely. Expected
per-token decode latency reduction in pure-Python land: 5–15%, dominated by
removed kernel launches and intermediate allocations rather than FMA count.
Largest impact at small batch sizes where kernel-launch latency dominates.

---

## 3. Q and K cannot be baked upstream — explanation for writeup

The forward order on Qwen3 is:

```
q = q_norm(q_proj(x).view(...))              # per-channel γ on D
k = k_norm(k_proj(x).view(...))
q, k = apply_rotary_pos_emb(q, k, cos, sin)  # RoPE: block-diag 2×2 rotation
q = q @ R_k;  k = k @ R_k                    # OSCAR rotation
```

To bake `R_k` into `q_proj.weight` we would need `R_k` to commute with both
`q_norm` and RoPE. Neither commutes:

- **q_norm / k_norm.** Qwen3 RMSNorm has a learnable per-channel scale
  `γ ∈ R^D`. Writing the scale as `diag(γ)`, the operation on the normalized
  vector is `q ← diag(γ) · q̂`. Commuting requires `R_k · diag(γ) = diag(γ) · R_k`,
  which is true only when `γ` is a constant vector or `R_k` is itself
  diagonal — neither holds. (The RMS normalization itself *is* invariant to
  orthogonal rotations of the input, but the per-channel γ that follows is
  not.)

- **RoPE.** RoPE applies a block-diagonal rotation of D/2 independent 2×2
  blocks, each parameterized by token position and channel-pair frequency.
  Each 2×2 block lives in a specific channel pair `(2i, 2i+1)`. Pre-rotating
  by an arbitrary orthogonal `R_k` mixes channels across blocks, so the
  per-pair RoPE rotation no longer applies to the right subspace. There is no
  orthogonal `R_k` (other than block-diagonal with respect to RoPE's pair
  structure, AND commuting with `diag(γ)`, i.e. essentially identity on each
  RoPE block) that lets you reorder these steps.

This was already verified empirically (output collapses to repeated `?`
tokens when Q/K are baked — see `rotation.py:37`). The Q and K einsums must
stay inline. The good news: per-token at decode T=1, the Q einsum is
`H × D² = 32 × 16384 = 524k` FMAs and K is `H_kv × D² = 8 × 16384 = 131k`
FMAs per layer. After baking V and O away, these two are what remains and
they are small — well under 1% of attention itself.

---

## 4. Fused bf16 → INT2 rotate+quantize Triton kernel

The upstream OSCAR repo (FutureMLS-Lab, "Together AI Open-Sources OSCAR")
ships **fused Triton kernels integrated into SGLang's paged cache**:

- **Write path:** per token, rotate + clip + asymmetric INT2 quantize + pack
  4×2-bit values per byte in a single kernel. Operates on rows of the paged
  cache directly.
- **Read path:** unpacks bytes, dequantizes, inverse-rotates, and hands
  results to the INT2 attention kernel in one fused pass.

Source: <https://github.com/FutureMLS-Lab/OSCAR>, blog coverage at
<https://www.marktechpost.com/2026/05/25/together-ai-open-sources-oscar-...>.
No standalone llama.cpp branch by "Zhongzhu" was found — the `Zhongzhu`
namespace is the HuggingFace repo hosting the rotation files
(`Zhongzhu/OSCAR-RotationZoo`), not a kernel branch.

**Estimated perf delta vs our reference:** the upstream paper claims ~7×
KV-cache compression with kernel-bound throughput close to bf16 baseline at
long context. Our reference path has three separable costs:

1. Rotation einsums (Q, K, V, O-side) — eliminable for V/O via §1–2.
2. Per-token quantize/dequantize round-trip in `quantize_per_token` /
   `dequantize` — currently materializes `(B, H, T, G, group_size)` fp32
   intermediates, ~32× memory traffic vs the packed INT2 codes.
3. Storing codes as uint8 (one INT2 per byte, 4× memory waste) — the
   incremental `_middle_k_dq` cache then *doubles* this again.

A fused write kernel collapses (1)+(2) into one HBM round-trip and packs to
1/4 the bytes. Expected speedup over reference for the spill-and-quantize
step is 3–5×, but spill happens once per `recent_tokens` (256) tokens, so
the wall-clock gain at decode is dominated by read-side cost, not write.

**Recommendation:** vendoring the upstream Triton kernels is the right
long-term path but high cost (requires the SGLang paged-cache layout). For
delta-mem-tests, §1+§2 + the §5 micro-opts below capture most of the
practical win without writing CUDA.

---

## 5. Highest-leverage cache/quantize micro-optimizations (no custom kernel)

### 5a. Stop `torch.cat` in `_assemble` — concat is O(seq) per decoded token

`cache.py:176–191`. Every decode step rebuilds the assembled K and V slabs
by `torch.cat([sink, middle_dq, recent])` along dim=2, allocating a fresh
`(B, H_kv, total_seq, D)` tensor — at 17k tokens, B=1, H_kv=8, D=128, bf16
that is 35 MB per layer × 36 layers = 1.26 GB allocated *and copied* per
decoded token. This is almost certainly the #1 wall-clock bottleneck after
the rotation einsums.

Fix: pre-allocate a persistent `(B, H_kv, max_seq, D)` buffer per layer the
first time `_assemble` is called, then only write the *new* slice (the
recent-window growth, plus any spill-to-middle delta). The slab handed back
to the attention kernel can be a `narrow()` view into the persistent buffer.
This converts O(total_seq) memory traffic per step into O(spill+new_tokens).

### 5b. Avoid the bf16 cast inside `dequantize` on every spill

`quantize.py:113–123`: `dequantize` calls `.to(dtype)` on `codes`, `scale`,
and `zero` every time, even though the dtype is fixed for the lifetime of
the cache (set in `lazy_initialization`). Cache the cast scale/zero tensors
at the time the `QuantizedBlock` is created. Better: store `scale` and
`zero` in the target dtype (bf16) from the start — the fp16 storage choice
in `quantize_per_token` is gratuitous since the cache lives in bf16. This
removes 3 `.to()` calls and the temporary tensors they allocate, per spill.

### 5c. Pre-allocate middle/recent ring buffers and write in-place

`cache.py:142–150,165–172`: both the `recent_k/v` FIFO and the
`_middle_k_dq` accumulator grow by `torch.cat` on every spill. The recent
window has a fixed maximum size (`recent_tokens=256`); allocate it once as
`(B, H_kv, recent_tokens, D)` plus a write cursor, and shift via index ops
instead of slicing + concat. The middle dequant cache is unbounded but
grows predictably in chunks of `recent_tokens` — pre-extend by a power-of-2
growth policy (double on overflow) to amortize allocation.

### 5d. (bonus) Avoid the `.contiguous()` in `_OSCARUnrotatingOProj.forward`

Becomes moot if §2 is implemented (the wrapper goes away). Until then,
`x_back = x_unrot.reshape(b, t, -1).contiguous()` forces a 12 MB copy at
T=4096. Since the next op is a Linear which calls into cuBLAS GEMM, the
GEMM tolerates a non-contiguous input on the last two dims as long as
strides are aligned — drop the `.contiguous()` and let GEMM handle it, or
use `view` instead of `reshape` if the einsum output is already contiguous.

---

## Priority ranking for implementation

1. **§2 (bake `R_v.T` into `o_proj.weight`)** — biggest single win, removes
   the heaviest einsum AND the wrapper Module. ~1 hour to implement, ~3 hours
   to test against the existing rotation-equivalence smoke test.
2. **§5a (pre-allocated `_assemble` slab)** — likely the largest wall-clock
   improvement of all (concat-per-decode is murderous at 17k context).
3. **§1 (bake `R_v` into `v_proj.weight`)** — small but free; bundle with §2
   into one `bake_v_into_projections` helper.
4. **§5b + §5c** — incremental gains, do as one PR after §2.
5. **§4 (Triton kernel)** — out of scope for the current milestone unless
   the §1–§3 wins are insufficient.

## References

- Upstream OSCAR repo (FutureMLS-Lab, Triton kernels + SGLang integration):
  <https://github.com/FutureMLS-Lab/OSCAR>
- Together AI open-source announcement:
  <https://www.marktechpost.com/2026/05/25/together-ai-open-sources-oscar-an-attention-aware-2-bit-kv-cache-quantization-system-for-long-context-llm-serving/>
- Paper: <https://arxiv.org/abs/2605.17757>
- RotationZoo (rotation files only, not kernels):
  <https://huggingface.co/Zhongzhu/OSCAR-RotationZoo>
- Vendored port: `E:\Development\delta-mem-tests\third_party\oscar-transformers\oscar_transformers\rotation.py`
  (lines 23–54 — pre-existing acknowledgement that V-baking is valid)
- Delta-mem integration site: `E:\Development\delta-mem-tests\delta-Mem\deltamem\core\delta_impl.py:2280–2294`
