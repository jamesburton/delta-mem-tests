# Long-context alternatives to building Gemma 3n + delta-mem ourselves

**Status:** Research only — no code changes.
**Author:** Claude (Opus 4.7, 1M context) | **Date:** 2026-06-09
**Branch:** `gemma3n-alternatives`
**Scope:** evaluate **EpiCache** (arXiv 2509.17396) and **LoCoCo** (arXiv
2406.05317) as alternatives to investing 4-7 engineer-weeks building
Gemma 3n + delta-mem support (per
[`.planning/research/gemma3n-deltamem-feasibility.md`](./gemma3n-deltamem-feasibility.md)).
Pairs with that doc; this is the "alternatives" half of the same decision.

---

## TL;DR

**Prefer EpiCache for a 2-3 day pilot; rule LoCoCo out.**

1. **EpiCache** is a fresh, MIT-licensed, training-free KV eviction
   framework from Apple (released
   [github.com/apple/ml-epicache](https://github.com/apple/ml-epicache)
   on 2025-10-02). It wires up out-of-the-box to **Qwen2.5** and
   **LLaMA-3.1/3.2** via a tiny monkey-patch of
   `Qwen2Attention.forward` / `LlamaAttention.forward`
   ([`model/monkeypatch.py`](https://github.com/apple/ml-epicache/blob/main/model/monkeypatch.py)).
   Adapting it to **Qwen3** would be ~50 LOC (re-attach to
   `Qwen3Attention`); adapting to **SmolLM3** ~80 LOC; **Gemma 3n** is
   the hard case (its sliding-window + KV-sharing classes need
   bespoke handling). It evaluates on **LoCoMo** (our exact eval) and
   claims up to **40 % accuracy improvement** over recent baselines
   with **4-6× compression**, **2.4× lower decode latency** and **3.5×
   lower peak memory** ([EpiCache HF
   page](https://huggingface.co/papers/2509.17396)). Could plausibly
   *stack* with OSCAR INT2 (orthogonal eviction-vs-quantization layers)
   but stacking is **untested** — the closest comparable is MiniKV
   ([arXiv 2411.18077](https://arxiv.org/html/2411.18077)) which found
   that H2O + KIVI stacks cleanly but SnapKV + KIVI degrades sharply
   (35 → 32 LongBench), so the stack carries real but bounded risk.

2. **LoCoCo** is a 2024 ICML paper from VITA-Group
   ([arXiv 2406.05317](https://arxiv.org/abs/2406.05317),
   [github.com/VITA-Group/LoCoCo](https://github.com/VITA-Group/LoCoCo))
   that compresses KV with 1-D convolutional fusion. It requires
   **training** (`--max_train_steps 1000` on RedPajama at minimum)
   and the reference implementation is an **intrusive fork of
   `modeling_llama.py`** for **Llama-2-7B only** — no GQA support, no
   Llama-3 / Qwen / Gemma path, no checkpoint download ("coming soon"
   in a [README from
   2024-09-07](https://github.com/VITA-Group/LoCoCo/commit/1be3fec)
   and still pending 21 months later). Porting LoCoCo to Qwen3-4B
   would mean reimplementing the training pipeline against a different
   modeling stack, training the conv heads from scratch on cloud GPU,
   then patching it into our pipeline — easily 3-4 engineer-weeks for
   a low-rank conv-fusion compressor with no published checkpoint and
   a 2024-vintage architectural assumption (MHA, no GQA). **Not
   competitive with EpiCache.**

3. **Pilot recommendation:** spend **2-3 days** wiring EpiCache into
   `_chunked_eval_runner.py` as an optional cache backend, run
   conv-26 (17 k) and conv-41 (25 k) on Qwen3-4B-Instruct under three
   conditions (`baseline`, `EpiCache only`, `OSCAR INT2 + EpiCache`),
   and use the resulting numbers to decide whether EpiCache replaces
   delta-mem, complements it, or gets shelved. This pilot answers
   "does the alternative stack actually buy us anything our existing
   OSCAR + delta-mem doesn't already do" with concrete data, at
   ~2 % of the cost of the Gemma 3n build-it-ourselves path.

---

## Section A — EpiCache

### A.1 Method summary

**Title:** "EpiCache: Episodic KV Cache Management for Long
Conversational Question Answering" — accepted to ICML 2026
([arXiv:2509.17396](https://arxiv.org/abs/2509.17396), latest revision
2026-05-19; [Apple ML Research
page](https://machinelearning.apple.com/research/epicache)).

**Authors:** Minsoo Kim, Arnav Kundu, Han-Byul Kim, Richa Dixit, Minsik
Cho (Apple).

**Core mechanism — eviction with episodic clustering and block-wise
prefill.** Three-stage pipeline (per
[arxiv.org/html/2509.17396](https://arxiv.org/html/2509.17396)):

1. **Conversation clustering & medoid selection.** Split the dialogue
   history into segments of `w_embed` utterances, embed each segment
   with a lightweight encoder (Qwen-0.6B, sentence-transformers
   MiniLM-L6, or the host LLM's own embedding layer), K-means cluster
   into `E=4` episodes (default), pick the segment closest to each
   centroid as that episode's "medoid".
2. **Episodic KV-cache compression.** For each episode `e`, perform
   block-wise prefill (each step adds `M_block` tokens, then evicts
   back down to budget `M`) with the episode medoid appended as a
   "patched prompt" after each block so that **attention scores for
   eviction are computed against the episode's representative
   query**. Retain top-`M` tokens by max attention score
   `s_i^max = max_t Attn(x_t → x_i)`. Store the per-episode compressed
   cache `C_KV^(e)` offline.
3. **Query-matching & decoding.** Embed each incoming query, match to
   the closest episode centroid, retrieve that episodic cache,
   decode.

**Operates on:** KV cache only. Does not touch model weights,
activations, or the attention computation itself — it monkeypatches
the attention `forward` to insert a hook before
`past_key_value.update`, and supplies a custom `EvictCache(DynamicCache)`
subclass that handles the eviction
([`attention/kvcache.py`](https://github.com/apple/ml-epicache/blob/main/attention/kvcache.py),
[`attention/attn.py`](https://github.com/apple/ml-epicache/blob/main/attention/attn.py)).

**Training requirement:** **training-free**. Explicitly: "a
training-free KV cache management framework" ([abstract,
arXiv:2509.17396](https://arxiv.org/abs/2509.17396)). One-time
**layer-sensitivity calibration** on BookSum is run per backbone to
produce `data/layer_scores/booksum_<MODEL>_sample0_layer_scores.json`,
optional but recommended (the `DO_SCORE=True` path with `POWER=1.3`
for Qwen, `1.1` for Llama-3 in
[`scripts/run_epicache_qwen.sh`](https://github.com/apple/ml-epicache/blob/main/scripts/run_epicache_qwen.sh)).
This is calibration, not training — minutes to hours on a single GPU.

**Architecture assumptions:**

- Decoder-only (LLaMA, Qwen, Mistral families confirmed in
  `monkeypatch.py`).
- **Requires Flash-Attention 2** (`pip install flash-attn==2.7.4.post1`)
  — uses `flash_attn_varlen_func`
  ([attn.py](https://github.com/apple/ml-epicache/blob/main/attention/attn.py)).
  This is a hard dependency: on Windows this means WSL2, on Strix
  Halo this means the AMD-built wheel; on RTX 3060 with current env
  we'd need to install it from source.
- **GQA-aware.** `EvictCache.__init__` reads
  `model.config.num_key_value_heads` and computes
  `n_group_kv = n_heads // n_heads_kv`
  ([kvcache.py:18-22](https://github.com/apple/ml-epicache/blob/main/attention/kvcache.py)).
  Qwen2.5 (GQA) and LLaMA-3 (GQA) work. By extension Qwen3 (GQA-2)
  should be fine.
- **No explicit sliding-window / hybrid-cache handling.** The cache is
  a `DynamicCache` subclass with custom eviction — Gemma 3 / Gemma 3n
  use `HybridCache` and per-layer sliding windows. EpiCache would
  need significant work to handle those (the `kvcache.py` import
  shows `from transformers import DynamicCache, HybridCache` but no
  visible logic for the sliding case).
- **No sliding-window-attention support published.** Not in the paper,
  not in the code.

**Headline numbers (per the published paper):**

| Metric | Full KV (LLaMA-3.2-3B) | EpiCache (6 K budget) |
|---|---|---|
| Decode latency | 68.9 ms | **30.1 ms** (2.3×) |
| Peak memory | 28.4 GB | **9.3 GB** (3.1×) |
| KV storage | 11.6 GB | **0.7 GB** (16×) |

Total wall on a 300-turn LongMemEval conversation: full KV 9 339 s,
EpiCache 545 s — **17.1× speedup**
([Table 3, arXiv:2509.17396v3](https://arxiv.org/html/2509.17396v3)).

Accuracy: "up to **40 %** accuracy improvement over recent baselines"
across **LongMemEval, RealTalk, LoCoMo**, sustaining "near-full KV
accuracy under 4-6× compression" ([Apple ML Research
page](https://machinelearning.apple.com/research/epicache)). The
paper's primary comparators are **SnapKV, H2O, StreamingLLM,
InfiniPot, KVzip, KeyDiff** — i.e. it is positioned vs. other
eviction methods, **not** vs. learned-memory adapters like delta-mem.

### A.2 Integration with our stack

**Compatibility with OSCAR INT2 KV quantization:** plausibly stackable
but untested.

- EpiCache stores its evicted cache as standard fp16/bf16 in
  `EvictCache`. Our `OSCARCache` stores packed INT2 codes. These are
  two **different `Cache` subclasses** — they cannot both be the
  `past_key_values` object simultaneously. To stack we would need a
  composite `OSCARLikeEvictCache(OSCARCache, KVScore)` that (a)
  computes attention-score statistics during the patched-prompt step
  in **dequantized** form, then (b) writes only the surviving K/V
  rows back to OSCAR INT2 packing. The dequant-during-scoring path
  costs the runtime of the snapshot's dequant-shadow that we
  measured earlier (~40 ms/layer/call per
  [`report/tier1-summary.md`](../../report/tier1-summary.md) Appendix
  D), incurred only during the few patched-prompt scoring passes per
  episode — bounded.
- **The deeper risk** is whether eviction-after-rotation produces
  meaningful attention scores. OSCAR's rotation is a unitary
  transform that preserves attention scores up to bf16 numerical
  drift, so on paper the answer is yes — scores after rotation
  should rank the same tokens. **But** OSCAR's INT2 quantization
  *does* perturb K/V; an evicted top-M set chosen on rotated/dequant
  K could fall apart on rotated/INT2 K. The closest published data
  point is MiniKV ([arXiv:2411.18077](https://arxiv.org/html/2411.18077)):
  H2O eviction + KIVI INT2 stacks cleanly, but SnapKV + KIVI loses
  ~3 LongBench points "because tokens retained by SnapKV tend to be
  more sensitive to 2-bit quantization." EpiCache uses a SnapKV-class
  patched-prompt scoring scheme, so **it inherits SnapKV's INT2
  sensitivity profile**. Expected effect: a moderate (probably 1-5
  point) quality dip from naive stacking, recoverable with a
  rotation re-calibration that bakes in the EpiCache scoring
  distribution.
- Verdict: **plausibly stackable with ~5 days of careful integration
  + 1-2 day re-calibration**, but the combination has not been
  published anywhere. Pilot is needed before committing.

**Compatibility with delta-mem adapter:** orthogonal but the two
mechanisms are partly in competition.

- delta-mem **adds** a learned per-layer write/read state that
  augments attention; it doesn't decide which tokens to keep.
- EpiCache **decides** which tokens to keep but doesn't augment the
  retained ones.
- Stacked: delta-mem keeps its delta state alongside an EpiCache-
  evicted KV cache. The delta state should be untouched (it's not
  in `past_key_values`). The attention call sees `[evicted KV] +
  [delta-mem corrections]`. Should compose, but the delta-mem
  adapter was trained against full KV; whether its corrections
  remain calibrated when only the top-M tokens are visible to base
  attention is an open question. Hypothesis: delta-mem
  *compensates* for the eviction loss because it stores compressed
  memory of evicted tokens — this is exactly the regime delta-mem
  was designed for (Appendix A of
  [`report/tier1-summary.md`](../../report/tier1-summary.md) showed
  `delta_only / truncated_base = 1.80×` — delta-mem already proves
  it can substitute for evicted context).
- **Most interesting hypothesis:** EpiCache + delta-mem could be
  **mutually amplifying** — EpiCache controls memory growth past
  17 k where delta-mem regresses today; delta-mem fills in the
  recall holes EpiCache's eviction leaves. We have no way to know
  this without running it.

**Backbone-agnosticism (LoCoMo eval reach):**

| Backbone | Native support in apple/ml-epicache | Effort to add |
|---|---|---|
| **Qwen2.5 3B / 7B-Instruct** | ✓ (`run_epicache_qwen.sh`) | none |
| **LLaMA-3.1-8B-Instruct** | ✓ (`run_epicache_llama.sh`) | none |
| **LLaMA-3.2-1B / 3B-Instruct** | ✓ | none |
| **Mistral** | ✓ (`monkeypatch.py` branch) | none |
| **Qwen3-4B-Instruct-2507** | ✗ | ~50 LOC: add `Qwen3Attention.forward = llama_flash_attn2_forward` branch + verify `apply_rotary_pos_emb` import + check `q_norm`/`k_norm` handling (Qwen3 has them; Qwen2.5 does not). The base attention forward is structurally identical so the import-swap should just work. |
| **SmolLM3-3B** | ✗ | ~80 LOC: SmolLM3's attention class lives in `transformers.models.smollm3.modeling_smollm3.SmolLM3Attention`; same monkeypatch pattern but with its own rotary import. SmolLM3 has YaRN-style RoPE at 64 k context that EpiCache's scoring may need a sliding-friendly variant for. |
| **Gemma 3n E4B** | ✗ | **400-800 LOC.** Gemma 3n has 28 sliding (window=512) + 7 full attention layers (`layer_types` in [config.json](https://huggingface.co/unsloth/gemma-3n-E4B-it/raw/main/config.json)); EpiCache's eviction logic assumes a single global cache. We'd need either to (a) apply EpiCache only to the 7 full layers (loses 80 % of the eviction benefit), or (b) build a sliding-aware EpiCache variant that respects per-layer attention type. KV-sharing across 15 of 35 layers adds another wrinkle. This is comparable scope to the delta-mem-on-Gemma3n port. |

**Code-integration effort (in our repo):**

- **Phase 1 — vendor and adapt (1-2 days).** Submodule
  `apple/ml-epicache` at the public release commit
  ([b742661](https://github.com/apple/ml-epicache/commit/b742661)) into
  `third_party/epicache/`. Add a Qwen3 branch to its
  `monkeypatch.py`. ~50 LOC.
- **Phase 2 — eval-runner integration (1 day).** Add an
  `--kv-cache-backend epicache` option to `run/locomo_eval.py` that
  (a) calls `replace_attn(model_id)` before model load, (b) passes
  the `EvictCache(model, evict_range, kv_budget=M)` as the
  `past_key_values` argument to the prefill pass. Mirrors the pattern
  we already use for `--kv-cache-backend oscar`. ~150 LOC including a
  small layer-sensitivity calibration script for Qwen3.
- **Phase 3 — flash-attn install (1 day, variable risk).** EpiCache
  hard-requires `flash-attn==2.7.4.post1`. On native Windows + RTX
  3060 this is non-trivial; either we install WSL2 + CUDA toolkit
  (~half day) or we patch EpiCache's `attn.py` to fall back to SDPA
  when flash-attn is unavailable (~50 LOC, mostly the
  `_flash_attention_forward` → SDPA replacement). The latter is the
  pragmatic path on this 12 GB box.
- **Phase 4 (optional) — stack with OSCAR (3-5 days).** New
  `OSCAREvictCache` that derives from `OSCARCache` and mixes in
  `KVScore`. Requires understanding `attention/score.py` (not yet
  read in detail) — that's the file that implements the patched-prompt
  scoring loop. **This is the experimental step**; only attempt
  after Phase 1-3 show EpiCache alone helps on Qwen3-4B at 25 k+.

**HuggingFace `pip install` availability:** No. Both the paper and
the repo are research artifacts — the apple/ml-epicache repo has 22
stars at time of writing
([gh repo view apple/ml-epicache](https://github.com/apple/ml-epicache))
and no PyPI package. We would vendor it as a submodule, exactly as
we did for `oscar-transformers`.

### A.3 What EpiCache solves vs what delta-mem solves

These are **different layers of the same problem**.

- **delta-mem** is a learned-adapter long-range memory mechanism that
  *augments* attention with a compressed, position-aware state. Its
  bottleneck is the adapter's training distribution: published
  adapter is trained around 17 k context, regresses to ratio 0.60 at
  25 k (per [`report/tier1-summary.md`](../../report/tier1-summary.md)
  v6c). Quality lift on multi-hop recall at trained range is
  substantial (1.33× overall, 4.8× on multi-hop in v5).
- **EpiCache** is an unlearned KV-cache eviction policy. Its
  bottleneck is the eviction budget M and the episode-clustering
  quality: paper reports near-full-KV accuracy at M=6 K against
  100 K-token dialogues, falling off below M=2 K. No quality lift
  vs. full KV is claimed — it's an efficiency play that *preserves*
  quality at large compression ratios.

**Could we stack them?** Yes in principle, and the use cases are
genuinely complementary:

- delta-mem already does *learned* memory of context; EpiCache does
  *attention-score* memory of context. Different signal sources.
- EpiCache controls memory growth (the hard ceiling at 25 k on the
  12 GB box that we hit at the same time as the delta-mem quality
  ceiling); delta-mem controls quality at lower context.
- The most natural pairing: **EpiCache compresses the older context
  (>17 k tokens) where delta-mem is OOD, delta-mem augments
  attention on the recent context where it's in-distribution**.
  This is roughly what the LoCoMo paper found independently —
  episodic/retrieval methods complement learned compression. Worth
  testing.

**Where does each help on LoCoMo-style multi-session-memory eval?**

| Question category | delta-mem helps | EpiCache helps | Both together |
|---|---|---|---|
| **single_hop** (fact recall) | YES — v5 base 0.27 → delta 0.36 | unknown; paper's overall ~30 % lift averages categories | likely additive |
| **multi_hop** (reasoning across sessions) | YES — flagship case (4.8× lift in v5) | YES — episodic clustering keeps cross-session topic tokens, which is exactly what multi-hop needs | most-likely synergistic; episode boundary roughly aligns with session boundary |
| **temporal** (when did X happen) | YES (5.1× lift in v5) | mixed — eviction may drop timestamp tokens unless they're attended in the patched prompt | EpiCache could hurt without delta-mem; with delta-mem the missing token is recoverable from compressed memory |
| **open_domain** (general knowledge, not memory) | small effect; doesn't need memory | small effect; eviction frees compute | unaffected by stacking |
| **adversarial (cat. 5)** | excluded from our eval | excluded from EpiCache's eval | n/a |

### A.4 Cost analysis (EpiCache)

**Engineer-time to integrate:**

- Phases 1-3 (Qwen3-4B alone, no OSCAR stack): **3-4 days** on this
  box. Most of the risk is flash-attn install — if we go SDPA
  fallback, smoother but slightly slower.
- Phase 4 (OSCAR + EpiCache stack): **+3-5 days**.
- Phase 5 (SmolLM3 also): **+2 days** (same monkeypatch pattern).
- Phase 6 (Gemma 3n): **comparable to delta-mem-on-Gemma3n port,
  3-5 weeks**. Don't do this in the pilot.

Total to "we know if EpiCache helps Qwen3-4B at 25 k+, with and
without OSCAR": **6-9 days of one engineer**, ~80 % less than
"build Gemma 3n + delta-mem from scratch" (4-7 weeks per
[`feasibility doc`](./gemma3n-deltamem-feasibility.md) §5).

**GPU time for calibration/training:**

- **0 GPU-hours for training** (training-free).
- **Layer-sensitivity calibration**: one pass over BookSum to
  generate the `<MODEL>_layer_scores.json`. On Qwen3-4B-Instruct this
  is ~1 hour on a 12 GB card.
- **No cloud cost** if we do the install on this box.

**Risk of "doesn't work on our model":**

- **Qwen3-4B:** low risk. Qwen3Attention is structurally identical
  to Qwen2.5Attention; the rotary embedding API and the
  `past_key_value.update` contract are unchanged in HF transformers.
  The `q_norm`/`k_norm` addition is the only structural difference,
  applied after projection and before rotary — won't break the
  attention monkeypatch.
- **SmolLM3-3B:** low-medium risk. SmolLM3 has 64 k native context
  with YaRN scaling; we'd want to ensure the patched-prompt scoring
  step doesn't trip over the YaRN rope. Likely fine.
- **Gemma 3n:** high risk — sliding-window layers and KV-sharing
  layers each violate EpiCache's assumption of a single
  monotonically-growing per-layer DynamicCache. Skip for pilot.
- **OSCAR INT2 + EpiCache stack:** medium risk. Closest analog
  (SnapKV + KIVI INT2 in MiniKV
  [arXiv:2411.18077](https://arxiv.org/html/2411.18077)) loses ~3
  LongBench points naively. May need a re-calibrated rotation that
  uses EpiCache-evicted distributions.

---

## Section B — LoCoCo

### B.1 Method summary

**Title:** "LoCoCo: Dropping In Convolutions for Long Context
Compression", ICML 2024
([arXiv:2406.05317](https://arxiv.org/abs/2406.05317),
[mlr.press/v235/cai24g](https://proceedings.mlr.press/v235/cai24g.html)).

**Authors:** Ruisi Cai, Yuandong Tian, Zhangyang Wang, Beidi Chen
(VITA-Group / UT Austin / Meta FAIR / CMU).

**Core mechanism — adaptive convolutional KV fusion.** A 1-D
convolution layer (kernel size 21 by default, ablation over 3-61)
inserted per attention layer reads the current KV cache + new tokens
and computes mixing weights via `Conv1d → ReLU → softmax` to produce
a fixed-size cache (e.g. 512 tokens) by **fusing old and new KV pairs
into the same fixed slots**. Each attention head shares the conv
kernel within a layer; layers have their own kernels.

**Operates on:** KV cache — but unlike EpiCache it *modifies the
cache values themselves* via the conv fusion, not just selects which
ones to keep. Closer to a learned compression than to eviction.

**Training requirement:** **not training-free.**

- **Inference / "post-hoc compression" mode:** tune the conv heads
  for **200 steps** on RedPajama-Data-1T-Sample
  ([README](https://github.com/VITA-Group/LoCoCo/blob/main/README.md)),
  base LLM frozen. Light, but not zero.
- **"Context-extension" mode** (4 K → 32 K): fine-tune conv heads +
  LoRA rank-8 adapters + modify embedding and norm layers. 1 000+
  steps, batch 128, chunk 512, learning rates `5e-5` for LoRA / `5e-2`
  for conv heads.
- **104 M training tokens** total per the paper (0.0052 % of LLaMA-2
  pretrain).
- **No published checkpoint.** The repo's README says
  "The model checkpoints is coming soon!" as of
  [commit 1be3fec, 2024-09-07](https://github.com/VITA-Group/LoCoCo/commit/1be3fec)
  and the repo has had **zero commits since** — currently 21 months
  stale.

**Architecture assumptions:**

- **Llama-2 only in the reference implementation.** The repo has
  exactly one model dir, `llama/`, containing a forked
  [`modeling_llama.py`](https://github.com/VITA-Group/LoCoCo/blob/main/llama/modeling_llama.py)
  with the conv heads woven into `LlamaAttention.forward`. This is
  an **intrusive fork**, not a monkeypatch — porting to another
  backbone means re-doing the same surgery on its modeling file.
- **No GQA / MQA support.** Llama-2-7B uses MHA; the paper does not
  test on Llama-2-70B (GQA) and there is no GQA logic in the code.
  This is a fundamental issue for our use case — every modern open
  model (Qwen3, SmolLM3, Gemma 3) is GQA.
- **No sliding-window support** discussed or implemented.
- ChatGLM3-6B-32k was also evaluated as a secondary backbone in the
  paper; that's the extent of cross-architecture testing.

**Headline numbers (per the paper):**

- Perplexity on Proof-Pile-2 at 32 K context (Llama-2-7B extended
  from 4 K): LoCoCo 3.3697 vs. H2O 3.4073 vs. full sequence 3.4012.
- LongBench at 32 K (Llama-2-13B): LoCoCo **37.4 %** vs. H2O 36.9 %
  vs. LongLoRA 34.7 %.
- SCROLLS GovReport (ChatGLM3-6B-32k): LoCoCo 0.3617 vs. H2O 0.3411
  vs. full 0.3669.
- **Compression ratio up to 32:1** (3 482 tokens into 128 cache
  slots; one experiment with default 512-slot config maps 4 K → 16 K
  → 32 K context).
- Memory: 16 K training fits in 50 GB (same as H2O, LongLoRA); full
  sequence OOM. Throughput at 16 K prefill: 33 tokens/s vs. H2O 32,
  LongLoRA 25.

These are improvements over **2024 baselines (H2O, LongLoRA)** —
neither is a current benchmark for our stack.

### B.2 Integration with our stack

**Compatibility with OSCAR INT2 KV quantization:**

- LoCoCo *writes back* into the cache via the conv fusion, replacing
  old slots with weighted combinations of old + new KV. This is
  incompatible with OSCAR's INT2 storage at a fundamental level:
  OSCAR's pack format is per-group quantized integer codes with a
  per-group scale and zero-point. Writing a *new* (weighted-sum) K/V
  value into a slot requires re-quantizing that slot, which means
  the new value gets snapped to the quantization grid at every conv
  step. Empirically that grid-snapping at INT2 destroys the conv
  fusion signal (we already know naive INT2 K/V destroys signal —
  see Appendix C of [`report/tier1-summary.md`](../../report/tier1-summary.md);
  LoCoCo's conv heads were trained against fp16 K/V).
- **Verdict: not stackable without retraining the conv heads on
  INT2-rotated K/V**, which means re-running LoCoCo's training loop
  on Qwen3 + OSCAR. That's a 1-week training run on cloud GPU +
  ~$200-500.

**Compatibility with delta-mem adapter:**

- LoCoCo's conv heads modify the K/V tensors before delta-mem's
  `_apply_delta_qkv` reads them. The delta-mem adapter was trained
  against unmodified K/V; LoCoCo perturbs the K/V distribution, so
  delta-mem's corrections will be slightly miscalibrated.
- Direction of the error: unknown — could partially cancel, could
  compound.
- Either way, **stacking would invalidate the published
  declare-lab/delta-mem_qwen3_4b-instruct adapter**, meaning we'd
  need to retrain delta-mem on top of LoCoCo's compressed
  representation. That's exactly the work delta-mem already needs
  for 25 k+ context (Option 1 in
  [`LONG_CONTEXT_PLAN.md`](../../LONG_CONTEXT_PLAN.md)), but now
  predicated on a 21-month-stale codebase with no GQA support.

**Backbone-agnosticism:**

| Backbone | Native support in VITA-Group/LoCoCo | Effort to add |
|---|---|---|
| **Llama-2-7B / 13B (MHA)** | ✓ | none |
| **ChatGLM3-6B-32k** | ✓ (paper, may not be in current code) | unknown |
| **Qwen3-4B-Instruct-2507 (GQA-8)** | ✗ | **400-800 LOC** + new training pipeline. Need to fork `modeling_qwen3.py` à la LoCoCo's `modeling_llama.py`, thread the conv heads through, handle GQA broadcasting (Q heads see K heads after `repeat_interleave` — conv heads currently assume MHA). |
| **SmolLM3-3B** | ✗ | similar to Qwen3, ~400 LOC. |
| **Gemma 3n** | ✗ | comparable to building delta-mem support; sliding/KV-shared layers make conv-fusion ill-defined. |

**Code-integration effort (in our repo):** 3-4 engineer-weeks minimum
(fork modeling file, add GQA logic, retrain conv heads on cloud
H100, integrate with our eval runner, possibly retrain delta-mem on
top). Comparable to Option 3 (Gemma 3n + delta-mem) in
[`LONG_CONTEXT_PLAN.md`](../../LONG_CONTEXT_PLAN.md) — and produces a
single-feature compression mechanism without delta-mem's
multi-session-memory benefits.

**HuggingFace `pip install` availability:** No. The repo has 17 stars,
no PyPI package, no published checkpoints, no recent maintenance.

### B.3 What LoCoCo solves vs what delta-mem solves

LoCoCo is a **fixed-size-cache compressor for streaming**.

- Its value-add: process a much longer effective context with a
  fixed memory budget. The conv-fusion learns to project new tokens
  into the existing slots without losing the most relevant info.
- It does *not* solve multi-session memory — it has no notion of
  "session" or "topic"; it's a streaming compressor.

vs. delta-mem (learned-adapter long-range memory) and EpiCache
(episode-aware eviction):

- Both delta-mem and EpiCache understand that **what to remember
  depends on what you might be asked about**. delta-mem learns it
  during adapter training; EpiCache learns it via episode clustering
  + patched-prompt scoring.
- LoCoCo has no equivalent — its conv weights are content-agnostic
  position-aware mixers. For LoCoMo's "given a dialogue history,
  answer a question about session 3's events" task, LoCoCo would
  preserve average-importance tokens at fixed cache positions and
  perform roughly like a uniform-attention-distillation baseline
  (which is what its 0.36 vs. full 0.37 score on GovReport suggests).

**Where does LoCoCo help on LoCoMo?** Probably not much. Its sweet
spot is "compress a fixed-budget cache while preserving perplexity
on next-token prediction over long-form text" — Proof-Pile-2 — not
"answer specific multi-hop questions about prior dialogue".

### B.4 Cost analysis (LoCoCo)

- **Engineer-time:** 3-4 weeks to get a Qwen3-4B port working at
  parity with the paper's Llama-2-7B numbers. Multiple unknowns
  (GQA conv-head design, the conv-head training data we don't
  have, the LoRA-extended training mode at 32 k context).
- **GPU time:** training run = 1 cloud H100 × 3-7 days ≈ $400-1 200.
- **Risk of "doesn't work on our model":** high. 2024 paper, no GQA
  support, dead repo, no checkpoint, training pipeline needs
  reimplementation per backbone.
- **Risk of "doesn't help on our eval":** also high. LoCoCo's
  evaluation is perplexity-on-long-text and SCROLLS — neither maps
  cleanly to LoCoMo's multi-session QA semantics.

---

## Section C — Side-by-side comparison

| Criterion | EpiCache | LoCoCo | Our current OSCAR+delta-mem |
|---|---|---|---|
| **Compat with OSCAR INT2 KV** | plausible but untested (eviction is orthogonal to quantization in principle; SnapKV+KIVI prior in [MiniKV](https://arxiv.org/html/2411.18077) ~3 pt drop) | broken (LoCoCo writes back into KV; INT2 snap kills conv signal) | shipped (v5 base 0.27 / delta 0.36 / ratio 1.33×) |
| **Compat with delta-mem adapter** | orthogonal; likely complementary (delta-mem fills recall holes EpiCache eviction creates) | conflicting (LoCoCo perturbs K/V distribution; delta-mem adapter trained against unmodified K/V; retrain required) | self (delta-mem is the adapter) |
| **Training cost** | **0** (training-free; ~1 GPU-hour calibration on BookSum) | ~$400-1 200 cloud H100 + 200-1 000 SGD steps per backbone | $50-300 H100 to retrain adapter for 25 k+ context |
| **Integration effort (Qwen3-4B)** | **3-4 engineer-days** (monkeypatch + eval-runner) | 3-4 engineer-weeks (intrusive modeling-file fork + training pipeline + checkpoint train) | 0 (already integrated) |
| **Headline result on relevant eval** | LoCoMo (paper): up to **40 %** accuracy lift vs. recent eviction baselines, near-full-KV at 4-6× compression; LongMemEval 17× wall-time speedup ([Apple ML Research](https://machinelearning.apple.com/research/epicache)) | LongBench at 32 K: **37.4 %** vs. H2O 36.9 % (Llama-2-13B, [arXiv:2406.05317v1](https://arxiv.org/html/2406.05317v1)) — modest; no LoCoMo number published | LoCoMo conv-26 / 10 q: **1.33× ratio** at 17 k context, dropping to **0.60× at 25 k** ([tier1-summary §v6c](../../report/tier1-summary.md)) |
| **Applies to Gemma 3n** | requires 400-800 LOC (sliding-window / KV-shared layers not handled) | requires ~equivalent to building delta-mem support from scratch | not currently — Option 3 in [LONG_CONTEXT_PLAN.md](../../LONG_CONTEXT_PLAN.md) is 4-7 weeks |
| **Applies to Qwen3-4B** | requires ~50 LOC (Qwen2.5 native; Qwen3 needs new monkeypatch branch) | requires 400+ LOC + cloud training | **shipped** |
| **Applies to SmolLM3-3B** | requires ~80 LOC | requires 400+ LOC + cloud training | requires rotation calibration + adapter training (Option 2 in [LONG_CONTEXT_PLAN.md](../../LONG_CONTEXT_PLAN.md), ~$100-400 cloud) |
| **Maintains evidence-based recall at 25 k+** | yes per paper (up to 100 k tested) | likely yes per paper (32 k tested) but on perplexity not recall | NO past 20 k with current adapter |
| **VRAM at 25 k on 12 GB box** | paper: 9.3 GB peak at 100 k context (LLaMA-3.2-3B) | not in our budget envelope (paper used 50 GB for 16 k training) | currently fits at 25 k batch=1 shadow-off only |
| **Maintenance / repo activity** | active (released 2025-10-02, MIT) | dead (last commit 2024-09-07) |
| **License** | MIT (inherited from KVzip) | not specified in repo |
| **GitHub stars** | 22 ([apple/ml-epicache](https://github.com/apple/ml-epicache)) | 17 ([VITA-Group/LoCoCo](https://github.com/VITA-Group/LoCoCo)) |
| **Bench match to LoCoMo** | direct (LoCoMo is one of three benchmarks in the paper) | none (LongBench + perplexity, not multi-session QA) |
| **Stacking with our existing OSCAR+delta-mem** | high upside, untested — **the strongest stacking candidate** | low upside, breaks OSCAR storage, retrain required |

---

## Section D — Recommendation matrix by user goal

| If the user's actual goal is... | Recommend | Why |
|---|---|---|
| **Push Qwen3-4B past 20 k useful context** | **Option 1 (retrain delta-mem adapter at 32 k)** as primary; **pilot EpiCache** as a stack-test in parallel. | Option 1 is the proven path: it's the explicit gap identified in v6c (ratio collapses at 25 k because the adapter is OOD, not because the math fails). EpiCache adds a *second* mechanism that could either replace the adapter for the OOD region or stack with a retrained adapter for a higher ceiling — but the adapter retrain alone is the lowest-risk way to hit 32 k. The EpiCache pilot answers whether we still need the retrain at all (probably yes, but maybe at smaller scale). LoCoCo: rule out — its eval doesn't match LoCoMo semantics. |
| **Get long-context Gemma 3n with reasonable engineering effort** | **Use Gemma 3n native 32 k** (its built-in context) + EpiCache for memory headroom if needed. Don't build delta-mem support and don't try to port LoCoCo. | Per the feasibility doc, building delta-mem on Gemma 3n is 4-7 engineer-weeks at compounding risk. EpiCache *could* extend Gemma 3n past native 32 k, but needs the same sliding/KV-shared rework as delta-mem (400-800 LOC). Net: Gemma 3n at native context + Apple's published EpiCache running on a sibling Llama-3.1-8B for "memory headroom" experiments is the lowest-risk way to validate whether long-context-on-Gemma actually unlocks anything we can't get from Qwen3-4B at 32 k. If validated, *then* invest in the Gemma 3n port. |
| **Run on memory-constrained hardware (this 12 GB card)** | **Stick with OSCAR INT2 + NF4 weights** (Option 4) **+ pilot EpiCache stacked on top**. Skip LoCoCo entirely. | OSCAR INT2 + NF4 already projects to ~4.6 GB at 32 k on Qwen3-4B (per [LONG_CONTEXT_PLAN.md table](../../LONG_CONTEXT_PLAN.md)). EpiCache adds a *third* compression layer (eviction): with M=4 K and OSCAR-INT2+NF4, the KV footprint at *any* context length is bounded by ~75 MB instead of growing with context. That's the change that lets a 12 GB card run a 100 k-token conversation — IF the stack works. LoCoCo: incompatible with OSCAR storage. |
| **Multimodal long-context** | **Build Gemma delta-mem** if multimodal long-context is a *hard* requirement; otherwise **Qwen3-4B + retrained adapter for text-only LoCoMo** and accept that the multimodal column stays empty. EpiCache and LoCoCo are both text-only. | EpiCache's clustering encoder is a text encoder (Qwen-0.6B / MiniLM); the paper doesn't test on multimodal contexts and the patched-prompt scoring assumes text tokens. LoCoCo same. Neither alternative addresses the multimodal-context dimension at all — it remains a Gemma 3n-only capability. |

---

## Section E — Concrete pilot proposal (EpiCache, 3 days)

**Question to answer:** does EpiCache, alone or stacked over our
existing OSCAR-INT2 + delta-mem pipeline, restore the **>1.0× ratio**
at 25 k on Qwen3-4B-Instruct that we lose past 20 k today?

### Day 1 — vendor + smoke

1. `git submodule add https://github.com/apple/ml-epicache
   third_party/epicache` at commit
   [`b742661`](https://github.com/apple/ml-epicache/commit/b742661)
   (current public release).
2. Add Qwen3 branch to `third_party/epicache/model/monkeypatch.py`:
   `transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward
   = llama_flash_attn2_forward` and verify `apply_rotary_pos_emb`
   import is qwen3-compatible (it should be — Qwen3 uses the same
   RoPE API as Qwen2.5).
3. Install `flash-attn` (try the prebuilt wheel route first; fall
   back to the SDPA-replacement patch if it fails — ~50 LOC in
   `third_party/epicache/attention/attn.py`).
4. Run the calibration: `python -m run_epicache.calibrate
   --model Qwen/Qwen3-4B-Instruct-2507 --data booksum
   --output data/layer_scores/booksum_Qwen3-4B-Instruct_layer_scores.json`
   (~1 GPU-hour on RTX 3060).
5. Smoke: run EpiCache standalone on conv-26 / 10 q at 17 k context
   with `KV_BUDGET=4096`. Expected: ratio ≈ 0.95-1.05 (near full
   KV per paper).

**Exit criterion:** EpiCache loads + runs end-to-end on Qwen3-4B
without errors.

### Day 2 — eval-runner integration + 25 k test

1. Add `--kv-cache-backend epicache` and `--epicache-kv-budget`
   flags to `run/locomo_eval.py` and the equivalent paths in
   `run/_chunked_eval_runner.py`. ~150 LOC.
2. Re-use the same `--max-conversations 1 --max-questions-per-conv
   10` recipe as the v6c run; switch to conv-41 (25 k).
3. Run **three conditions** sequentially (~3.5 h each, ~10 h total):
   - **A.** Baseline: `--kv-cache-backend bf16` (control, no
     compression).
   - **B.** EpiCache only: `--kv-cache-backend epicache
     --epicache-kv-budget 4096`. **No OSCAR.** **No delta-mem.**
   - **C.** Delta-mem only (existing v6c run; we already have this
     data; reuse it).

**Exit criterion:** condition B's base + delta scores at 25 k. We're
looking for: does EpiCache alone at M=4 K beat the OSCAR INT2 +
delta-mem stack's 0.139 delta score at 25 k? Per paper it should
("near-full KV at 4-6× compression") — meaning **EpiCache alone may
be a clean replacement for the OSCAR+delta-mem stack at 25 k**.

### Day 3 — stack test + decide

1. Implement minimal `OSCAREvictCache` (subclass `OSCARCache`, mix
   in EpiCache's `KVScore` patched-prompt scoring path; dequantize
   K from packed INT2 for the scoring step only, write-back evicted
   subset in re-packed INT2). ~200 LOC.
2. Run **condition D**: `--kv-cache-backend oscar+epicache
   --epicache-kv-budget 4096 --oscar-rotation gpqacal` on conv-41 /
   10 q.

**Exit criterion** (decision point — three outcomes):

- **Outcome 1: condition B alone beats current delta-mem at 25 k.**
  → Replace OSCAR+delta-mem with EpiCache for 25 k+ contexts. Keep
  OSCAR+delta-mem as the production path at 17 k where it wins.
  Total project value: 3 days to unlock 25-100 k contexts that we
  couldn't reach before.
- **Outcome 2: condition D (stacked) beats both B and current
  delta-mem at 25 k.** → Ship the stack as the new production
  default. Retrain the OSCAR rotation against EpiCache-evicted
  distributions over the following week (~1 day calibration). This
  is the highest-upside outcome and the most likely one if MiniKV's
  precedent generalizes.
- **Outcome 3: condition B underperforms current delta-mem at 25 k
  AND condition D underperforms.** → Confirm via second
  conversation (conv-49 at 19 k); if reproducible, EpiCache is not
  a fit for our LoCoMo distribution. Total cost: 3 engineer-days
  and a clean negative result that lets us focus on Option 1
  (adapter retrain).

**Pilot estimated cost:** **3 engineer-days** + **~20 GPU-hours** on
the 12 GB box (no cloud spend). **Yields a clear yes/no** with
quantitative numbers on the same eval slice we already use, against
the same baselines.

---

## Sources

### EpiCache
- [arXiv:2509.17396 — EpiCache: Episodic KV Cache Management for Long Conversational Question Answering](https://arxiv.org/abs/2509.17396)
- [arXiv:2509.17396v3 (HTML, latest revision)](https://arxiv.org/html/2509.17396v3)
- [arXiv:2509.17396 (HTML, "Resource-Constrained Environments" earlier title)](https://arxiv.org/html/2509.17396)
- [Apple Machine Learning Research — EpiCache project page](https://machinelearning.apple.com/research/epicache)
- [HuggingFace papers page for 2509.17396](https://huggingface.co/papers/2509.17396)
- [github.com/apple/ml-epicache](https://github.com/apple/ml-epicache)
- [`model/monkeypatch.py`](https://github.com/apple/ml-epicache/blob/main/model/monkeypatch.py) (integration mechanism: `transformers.models.{llama,qwen2,mistral}.modeling_*.{Llama,Qwen2,Mistral}Attention.forward = llama_flash_attn2_forward`)
- [`attention/kvcache.py`](https://github.com/apple/ml-epicache/blob/main/attention/kvcache.py) (`EvictCache(DynamicCache, KVScore)`)
- [`attention/attn.py`](https://github.com/apple/ml-epicache/blob/main/attention/attn.py) (FA2 forward with KV-scoring hook)
- [`scripts/run_epicache_qwen.sh`](https://github.com/apple/ml-epicache/blob/main/scripts/run_epicache_qwen.sh) (`Qwen/Qwen2.5-{3,7}B-Instruct`, `POWER=1.3`)
- [`scripts/run_epicache_llama.sh`](https://github.com/apple/ml-epicache/blob/main/scripts/run_epicache_llama.sh) (`Llama-3.{1,2}-{1,3,8}B-Instruct`)

### LoCoCo
- [arXiv:2406.05317 — LoCoCo: Dropping In Convolutions for Long Context Compression](https://arxiv.org/abs/2406.05317)
- [arXiv:2406.05317v1 (HTML)](https://arxiv.org/html/2406.05317v1)
- [PMLR v235/cai24g — ICML 2024 proceedings entry](https://proceedings.mlr.press/v235/cai24g.html)
- [HuggingFace papers page for 2406.05317](https://huggingface.co/papers/2406.05317)
- [github.com/VITA-Group/LoCoCo](https://github.com/VITA-Group/LoCoCo)
- [LoCoCo README (last commit 2024-09-07)](https://github.com/VITA-Group/LoCoCo/blob/main/README.md)
- [LoCoCo `llama/modeling_llama.py` (intrusive fork)](https://github.com/VITA-Group/LoCoCo/blob/main/llama/modeling_llama.py)

### Related (stacking precedents)
- [MiniKV: 2-Bit KV Cache via Compression and System Co-Design — arXiv:2411.18077](https://arxiv.org/html/2411.18077) (H2O + KIVI INT2 stacks; SnapKV + KIVI INT2 degrades 35 → 32 LongBench)
- [CAKE: Cascading and Adaptive KV Cache Eviction — arXiv:2503.12491](https://arxiv.org/pdf/2503.12491) (CAKE + KIVI INT4 → 32.51 at 12.5 % vs KIVI INT2 alone 32.17)
- [KIVI: Tuning-Free Asymmetric 2bit Quantization for KV Cache — arXiv:2402.02750](https://arxiv.org/html/2402.02750v2)
- [EvicPress: Joint KV-Cache Compression and Eviction — arXiv:2512.14946](https://arxiv.org/abs/2512.14946)

### Internal cross-references
- [`.planning/research/gemma3n-deltamem-feasibility.md`](./gemma3n-deltamem-feasibility.md) — "build Gemma 3n + delta-mem from scratch" cost analysis (Section 5: 4-7 engineer-weeks); also lists EpiCache + LoCoCo as alternatives (Section 7) — this doc is the deep-dive into those two.
- [`LONG_CONTEXT_PLAN.md`](../../LONG_CONTEXT_PLAN.md) — Options 1-4 (Qwen3 adapter retrain, SmolLM3 + new adapter, Gemma 3n, NF4 weights).
- [`report/tier1-summary.md`](../../report/tier1-summary.md) — Appendix D (OSCAR INT2 + delta-mem v2/v3/v5/v6 history); the v6c result (ratio 0.60 at 25 k) is the gap this alternatives evaluation aims to close.
- [`delta-Mem/deltamem/core/delta_impl.py`](../../delta-Mem/deltamem/core/delta_impl.py) — delta-mem attention wrapper (Qwen3 + SmolLM3 supported; the file EpiCache's KV-scoring hook would need to coexist with).
- [`third_party/oscar-transformers/oscar_transformers/cache.py`](../../third_party/oscar-transformers/oscar_transformers/cache.py) — `OSCARCache(DynamicCache)`; the class an `OSCAREvictCache` would derive from.
