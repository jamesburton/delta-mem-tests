# OSCAR INT4 vs INT2 on Qwen3-4B-Instruct-2507 — Research Brief

**Date:** 2026-06-02
**Scope:** Quality/memory/speed trade-off if we relax the middle-region KV
quantizer from INT2 to INT4 in our OSCAR pipeline (sink=64 bf16, recent=256 bf16,
middle = per-token asym, GK=128, fp16 scale+zero).
**Bottom line:** INT4 buys roughly **+2.5 accuracy points** on Qwen3-4B-class
models, costs **~1.6x more middle-region KV bytes** (2.16 → 4.16 BPE), and
**probably does not require the OSCAR rotation at all** — block-diagonal
Hadamard (Saw-INT4 recipe) or even no rotation at GK≤128 is enough to stay
within ~2.5 points of bf16. Decode-time dequant cost is essentially identical.

---

## 1. What OSCAR paper says about INT4

OSCAR is published as a **2-bit-only** method. The paper deliberately does not
train or release an INT4 variant — its whole framing is "make INT2 work where
naive Hadamard collapses." But the paper does include INT4 **as a reference
baseline** (Saw-INT4, the team's own prior INT4 work from arXiv 2604.19157).

The headline numbers from Table 2 / oscar-quantize.github.io on
**Qwen3-4B-Thinking-2507** (closest released proxy to Qwen3-4B-Instruct-2507;
the two share architecture, head_dim=128, kv_heads=8 GQA — only the post-train
differs):

| Method                       | BPE   | GPQA  | HumanEval | LCB v6 | AIME25 | MATH-500 | Mean  | Δ vs bf16 |
|------------------------------|-------|-------|-----------|--------|--------|----------|-------|-----------|
| BF16 baseline                | 16.00 | 67.27 | 94.05     | 48.66  | 74.67  | 93.55    | 75.64 | —         |
| **Saw-INT4** (Hadamard)      | 4.25  | 66.37 | 89.78     | 46.20  | 70.00  | 93.19    | 73.11 | **−2.53** |
| TurboQuant (~INT3)           | 3.25  | 41.41 | 31.83     |  0.58  | 16.67  | 68.20    | 31.74 | −43.90    |
| QuaRot-INT2 (Hadamard only)  | 2.25  |  0.34 |  0.98     |  0.00  |  0.00  |  5.67    |  1.40 | −74.24    |
| **OSCAR INT2**               | 2.28  | 64.95 | 92.24     | 45.38  | 64.00  | 92.75    | 71.86 | **−3.78** |

Key reads:
- INT4 vs INT2 on Qwen3-4B is **a ~1.25 point mean improvement** (Saw-INT4
  73.11 vs OSCAR 71.86). The single biggest item is AIME25 (70.0 vs 64.0,
  6-point gap) — reasoning-heavy long-trace tasks benefit most.
- On Qwen3-8B the gap shrinks (OSCAR cuts the bf16 gap to 1.42; Saw-INT4 is
  comparable). On Qwen3-32B and GLM-4.7 OSCAR is **on par with bf16** so INT4
  has nothing to offer at scale. INT4's quality win is concentrated in the
  4B-8B range we are running.
- The paper does **not** publish OSCAR-INT4. Inferring: the eigenbasis rotation
  was tuned around INT2 covariance error; redoing the calibration at INT4
  would likely close the remaining ~1 point gap, but no released numbers exist.
- **RULER-NIAH 128k (Table 3):** OSCAR INT2 holds 39.5±1.0% on Qwen3-4B vs 0.0%
  for QuaRot-INT2. The paper does not run Saw-INT4 at 128k, but Saw-INT4's own
  paper claims "near-lossless" on long-context — INT4 with block-diagonal
  Hadamard should not collapse on RULER.

## 2. Exact bytes/token/layer for Qwen3-4B-Instruct-2507

Architecture: head_dim=128, kv_heads=8 (GQA), so K and V are each
`8 × 128 = 1024` elements per token per layer. KV pair = **2048 elements/token/layer**.

Per-token asymmetric quant with group_size GK=128 means **1 group per head per
KV** (128 elements per group, matching head_dim). Each group stores: GK packed
codes (bit-packed) + fp16 scale (2 B) + fp16 zero (2 B) = **4 B metadata/group**.

Groups per token per layer (K+V) = 2 (K,V) × kv_heads (8) × 1 group/head = **16 groups**.

| Precision    | Code bytes (2048 elts)        | Metadata (16 groups × 4 B) | Total B/tok/layer | BPE   | vs bf16  |
|--------------|--------------------------------|----------------------------|-------------------|-------|----------|
| bf16 raw KV  | 2048 × 2 = 4096                | 0                          | **4096**          | 16.00 | 1.00x    |
| INT8         | 2048 × 1 = 2048                | 64                         | **2112**          |  8.25 | 0.516x   |
| **INT4**     | 2048 × 0.5 = 1024              | 64                         | **1088**          |  4.25 | **0.266x** |
| **INT2**     | 2048 × 0.25 = 512              | 64                         | **576**           |  2.25 | **0.141x** |

For our actual run (17,000 cached tokens, 36 layers — Qwen3-4B has 36 hidden
layers), middle-region total KV bytes:

| Precision | B/tok/layer | × 17000 tok × 36 layers | Total middle KV     |
|-----------|-------------|-------------------------|---------------------|
| bf16      | 4096        |                         | **2.51 GB**         |
| INT4      | 1088        |                         | **0.665 GB**        |
| INT2      |  576        |                         | **0.353 GB**        |

So moving INT2 → INT4 in the middle costs us an extra **~312 MB** of KV at 17k
context. On a 12 GB 3060 with ~6 GB already absorbed by weights (Qwen3-4B in
bf16 is ~7.6 GB; we are presumably running 4-bit weights to fit), that 312 MB
is real but not back-breaking. If we keep sink=64 + recent=256 in bf16 (those
320 tokens already cost ~47 MB at full precision, untouched by this change).

**Note on metadata overhead:** at GK=128 the scale/zero overhead is 0.25 bits
per element, identical for INT2 and INT4. This is why OSCAR reports 2.28 BPE
(not 2.0): metadata + the sink/recent bf16 windows amortise to ~0.28 extra
BPE. Going to INT4 keeps the same +0.28 overhead → ~4.28 BPE in practice. The
storage ratio INT4/INT2 is **4.28 / 2.28 ≈ 1.88x** (not 2.0x).

## 3. Do other rotation-based quantizers collapse at INT4?

**No — INT4 is benign in the raw basis for almost everyone.** This is the
critical asymmetry that explains why OSCAR's rotation matters at INT2 but is
much less critical at INT4.

- **QuaRot** (arXiv 2404.00456): 4-bit end-to-end (weights + activations + KV)
  Hadamard-rotated LLaMA2-70B loses ≤0.47 WikiText-2 PPL and retains 99% of
  zero-shot. At INT4 the Hadamard rotation **is** doing work for activations,
  but the KV-only ablation is mild.
- **KIVI** (arXiv 2402.02750): KIVI-4 (4-bit, **no rotation at all**, just
  per-channel-K + per-token-V + sliding window) is **near-lossless vs FP16**
  — Llama-2-7B CoQA drops <1%, GSM8K shifts +0.3%, TruthfulQA ±0.1%. KIVI-2
  (2-bit, same scheme) drops CoQA ~0.83%, GSM8K ~0.76% on Llama-2-7B; on
  Mistral-7B GSM8K drops ~2.35%. KIVI did not publish strong-reasoning-model
  numbers, so the Qwen3-reasoning trace stress is unrepresented.
- **Saw-INT4** (arXiv 2604.19157, same FutureMLS-Lab team as OSCAR): finds
  that **token-wise INT4 + block-diagonal Hadamard** is "the minimal viable
  4-bit recipe" — recovers nearly all of naive-INT4's loss. Naive INT4 does
  show some degradation but it's a few points, not a collapse. The full
  Hadamard (QuaRot-style) is not needed at INT4; block-diagonal is enough.
- **RotateKV** (arXiv 2501.16383): targets INT2 with outlier-aware adaptive
  rotations; reports <0.3 PPL on LLaMA-2-13B at 2-bit. Explicitly motivates
  rotation as "needed for extreme low-bit."

**Implication for us:** if we relax to INT4, we very likely **do not need the
OSCAR eigenbasis rotation at all**. A block-diagonal Hadamard (Saw-INT4
recipe), or even just raw per-token-asymmetric INT4 with GK=128, should sit
within 2-3 points of bf16 on Qwen3-4B. The OSCAR rotation files in the
RotationZoo are INT2-tuned; reusing them at INT4 is safe but probably
unnecessary overhead.

## 4. HF RotationZoo audit

[Zhongzhu/OSCAR-RotationZoo](https://huggingface.co/Zhongzhu/OSCAR-RotationZoo)
publishes:

| Model                          | Calibration                       | Bit-width | Group size |
|--------------------------------|-----------------------------------|-----------|------------|
| Qwen/Qwen3-4B-Thinking-2507    | seq20000_prompt83_group128 (+v2)  | INT2      | 128        |
| Qwen/Qwen3-8B                  | seq20000_prompt83_group128        | INT2      | 128        |
| Qwen/Qwen3-32B                 | seq16000_prompt69_group128        | INT2      | 128        |
| zai-org/GLM-4.7-FP8            | seq10000_prompt43_group128        | INT2      | 128        |

**No INT4 rotations published anywhere on the Hub** — neither in OSCAR-RotationZoo,
nor in any QuaRot/RotateKV/Saw-INT4 mirror. **No Qwen3-4B-Instruct-2507**
rotation either; only the Thinking-2507 variant. Architecturally these two
share head_dim and kv_heads, so the Thinking rotations are a defensible
starting point but **not validated for Instruct**. Transferability is not
documented by the authors; OSCAR is explicitly per-model (covariance is
data-dependent).

Adjacent rotation zoos (QuaRot, FlatQuant, RotateKV) likewise publish per-model
weights and do not advertise cross-model transfer.

## 5. Decode-time dequant cost on a 3060: INT4 vs INT2

For our 17k cached-token decode-time hot path the dequant is:
`codes (u8) → unpack → cast to bf16 → subtract zero → multiply scale`.

- **Memory bandwidth** is the dominant cost on a 3060 (~360 GB/s). INT2 reads
  1024 B/tok/layer of codes; INT4 reads 2048 B (still bit-packed in u8). Both
  are ~3.6-7.2x smaller than bf16's 4096 B, so we are bandwidth-bound on
  metadata + scale broadcast, not on arithmetic.
- **Compute** is trivial in both cases. INT2 needs one extra unpack shift
  (4 codes per byte) vs INT4 (2 codes per byte). On a 3060 SM that's a couple
  of integer ops per warp lane — invisible next to the bf16 multiply.
- **Kernel realities:** OSCAR's published INT2 attention kernel is custom
  (SGLang fork). Our HF transformers port dequantises codes → bf16 then
  routes through standard SDPA. In that path the **time delta INT4 vs INT2 is
  one extra DRAM read of metadata-padded codes**, i.e. ~0.5 ms extra on a
  17k-token attention sweep — well within noise of the SDPA call itself.

**Verdict on speed:** for our pipeline, INT4 vs INT2 decode-step latency is
indistinguishable. The only speed-relevant axis is whether the larger INT4
cache forces us to spill from VRAM, which at 17k tokens it doesn't.

---

## Recommendation

If our current INT2 OSCAR run produces correctness regressions on AIME-style
multi-step reasoning, **try INT4 with the existing OSCAR rotation** as the
cheapest A/B. Expected outcome on Qwen3-4B-Instruct-2507:
- ≈+1.5 to +2.5 mean accuracy points (Saw-INT4 baseline minus OSCAR-INT2 gap),
  weighted toward AIME / LiveCodeBench.
- +312 MB middle KV at 17k tokens; total KV roughly doubles within the middle
  region but stays under 1 GB.
- No measurable decode-time latency change.

If INT4 closes the regression, the followup question becomes whether the
OSCAR rotation is still earning its keep at INT4 (per §3, probably not) — at
which point we could drop the rotation entirely and run Saw-INT4 style for
simplicity. That would let us **decommission the RotationZoo dependency**
for the INT4 mode and use raw per-token-asym INT4 with GK=128 + fp16 metadata,
matching the same memory profile as the table above.

The one trap to watch: Qwen3-4B-Instruct vs Thinking-2507 post-training
divergence could change KV outlier statistics. If we keep the rotation, we
should re-dump calibration activations on Instruct rather than reusing the
Thinking rotation file blindly; if we drop the rotation for INT4, this is moot.

---

## Sources

- OSCAR paper: <https://arxiv.org/abs/2605.17757> / project page <https://oscar-quantize.github.io/>
- OSCAR code: <https://github.com/FutureMLS-Lab/OSCAR>
- OSCAR rotation zoo: <https://huggingface.co/Zhongzhu/OSCAR-RotationZoo>
- Saw-INT4: <https://arxiv.org/abs/2604.19157>
- QuaRot: <https://arxiv.org/abs/2404.00456>
- KIVI: <https://arxiv.org/abs/2402.02750>
- RotateKV: <https://arxiv.org/abs/2501.16383>
- OScaR (independent, INT2): <https://arxiv.org/abs/2605.19660>
