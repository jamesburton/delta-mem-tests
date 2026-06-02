# Delta-Mem per-token decode speedup options

Scope: explain the ~2.3x decode slowdown on the delta arm vs the base arm
(Qwen3-4B-Instruct-2507, OSCAR INT2 KV cache, LoCoMo official_prompt mode)
and identify the levers that can plausibly close it. Code-only investigation
— no source modifications.

The active adapter is `outputs/full_conv0.json` lines 3624-3677:
`rank=8, num_state_heads=1, num_memory_partitions=1, multi_head_state=False,
memory_write_granularity="token", delta_heads=["q","o"], memory_readout_mode="delta"`.
This is the rank-8 Q/O TSW configuration from the README.

## 1. Where the per-token cost lives

Walking `DeltaMemAttention.forward` (`delta-Mem/deltamem/core/delta_impl.py:2083-2299`)
with `seq_len=1` (decode regime):

**Cost #1 — Per-token affine scan in pure torch (top contributor).**
Because `memory_write_granularity == "token"`, the `write_hidden` branch at
2128-2137 is skipped and execution falls into `_memory_affine_scan` at
2174-2184. With `DELTA_MEM_SCAN_IMPL=torch` pinned (see
`run/_chunked_eval_runner.py:25` and `report/kernels-gate.md:71-74`), this
dispatches to `_memory_affine_scan_torch` (`delta_impl.py:1895-1938`). At
seq_len=1 the Python `for token_idx in range(seq_len):` loop still issues
two `torch.einsum("bij,bj->bi")` ops, two outer products via
broadcasted multiplies, one fused affine update, an `unsqueeze`, a mask
multiply, and a `torch.stack` — for **every layer, every decoded token**.
On Qwen3-4B that's 36 layers × ~9 small kernel launches per token = ~324
extra CUDA launches per decoded token before any base-model work. At
decode the launch overhead — not the FLOPs of an 8×8 state — is the
dominant cost. (The Triton kernel fuses these into one launch per token
per scan call; see Cost #1's mitigation in §2.)

**Cost #2 — Two extra fused `F.linear` calls per token in `_memory_sequence_projections`.**
At `delta_impl.py:887-913`: one packed gate linear (`beta_proj` only —
`couple_lambda=True`, line 893 skips `lambda_proj`) and one packed
memory-QKV linear over `[memory_q_proj, memory_k_proj, memory_v_proj]`
with output width `3 * state_read_dim = 3 * 8 = 24`. The memory linear is
a (1, hidden_size=2560) × (2560, 24) GEMM — tiny, but it's an
extra kernel launch and a tanh+normalize pair (line 911-912 -> 805-815)
per layer per token. Followed downstream by `_compute_delta_qkv_from_reads`
(846-853) and `_apply_delta_qkv` (855-885) — three more `F.linear`s, of
which only `delta_q` and `delta_o` are non-None (delta_heads=["q","o"]
at line 567 -> `_project_delta_head` returns None for k and v at
1107-1108) but `_apply_delta_qkv` still calls `self.base.q_proj`,
`base.k_proj`, `base.v_proj` separately (876-884) since
`has_packed_qkv_proj=False` on Qwen3. That's 3 base projections + 2
delta projections per layer per token vs 3 in the base arm — **+67% of
projection launches per layer**.

**Cost #3 — `delta_o` add-on path.**
`delta_o_proj` is computed at 2200 via `_project_delta_head(reads, ...)`
— a (1, state_read_dim=8) × (8, hidden_size=2560) GEMM — and added to
`base_o_output` at 2294, plus per-token norm/ratio tracking at 2282 and
2288-2293. The norm computations (`self._masked_hidden_norm`, two
`.norm(dim=-1)` calls, one masked ratio) are tiny FLOPs but
**6 extra small reductions per layer per token** — and they're
unconditionally evaluated even when nothing reads `last_delta_o_norm`
during eval.

The qualitative ranking: scan loop overhead >> per-token projection
launches > delta_o bookkeeping. The scan is the only one that scales
with `rank²` (small at 8) but is the only one that is *unfused* on this
configuration.

## 2. Scan-impl options and why we're on the slow one

`self.scan_impl = os.environ.get("DELTA_MEM_SCAN_IMPL", "auto")` —
`delta_impl.py:633`. Valid values (`delta_impl.py:2022-2057`):

- `"torch"` — forces the per-token Python loop in
  `_memory_affine_scan_torch` (1895-1938).
- `"triton"` — forces Triton; raises if `triton_scan_support` fails
  (2034-2035).
- `"auto"` — uses Triton when `triton_scan_support` returns supported,
  else falls back to torch silently (2033, 2047).

`triton_scan_support` (`kernels/affine_scan.py:25-59`) requires: Triton
installed, all tensors on CUDA, dtype in {fp16, bf16, fp32}, square
state rank, matching 3D shapes. **Our pinning to `"torch"` was a
gate-decision (not a correctness one)**: `report/kernels-gate.md:71-74`
records that on this Windows host Triton isn't installed, and the env
var pin prevents accidental path-switch if it ever gets installed
mid-run. Algebraic equivalence between the two paths is documented at
`kernels-gate.md:35-48`.

There is no third "cuda" path — only torch and Triton. The "auto"
default would already pick Triton on a Linux/Triton host.

**Speedup expectation if Triton becomes available:** the upstream
bench script (`delta-Mem/deltamem/tools/bench_scan.py`) is the
authoritative measure. Practically, fusing ~9 small launches per token
per layer into one is the kind of optimization that erases the bulk of
this kind of overhead — on a 12 GB Ampere card the per-launch latency
floor is what's being paid here, not arithmetic. The torch path also
allocates `read_steps: list[torch.Tensor]` and stacks at the end
(1908, 1937), which under `torch.inference_mode()` is cheap but still
adds a per-call allocation.

We are avoiding Triton **only** because it isn't installed on this
Windows host. Installing `triton` (Linux/WSL2) or `triton-windows`
would unlock the faster path with no code changes — `scan_impl="auto"`
would pick it up automatically.

## 3. State shape — bounded per layer, does not grow with tokens

`_ensure_state` (`delta_impl.py:764-793`) lazily allocates
`self.delta_state` as either `[batch, num_state_heads, rank, rank]`
(multi-head) or `[batch, rank, rank]` (single-head). For our config
(`num_state_heads=1, rank=8, batch=1`) that's an 8×8 bf16 tensor per
DeltaMemAttention instance — **128 bytes per layer, 4.5 KB across 36
layers**.

It is overwritten in place on every forward at 2192
(`self.delta_state = state`). Crucially **it does not grow with token
count** — that's the whole point of compressing memory into a
fixed-rank state. So the hidden memory cost we should track is zero
beyond the constant 4.5 KB. The growth-with-tokens cost lives in the
base KV cache (OSCAR-quantized in our setup), not in delta-mem.

`reset_state` (714-732) drops `self.delta_state = None` along with all
last-* telemetry; the chunked runner snapshots/restores it across
questions via `_snapshot_delta_state` /`_restore_delta_state`
(`run/_chunked_eval_runner.py:388-407`), each snapshot also being
fixed-size.

## 4. Can state writes be disabled during decode?

Mechanically **yes**: `set_delta_mem_write_enabled(model, False)`
(`delta_impl.py:733-739, 2344-2347`) flips `write_enabled=False`, and
the forward at 2123-2191 then skips the entire write branch and runs
`_token_state_reads` directly (2186-2191), a single fused
`torch.einsum("bij,btj->bti")` per layer — eliminating Cost #1 entirely
and turning the scan into a cheap read. This is exactly what the
vendored `_batched_generate_raw_predictions` does
(`deltamem/eval/locomo_delta.py:570, 594`) for the base-eval path —
write off during decode, restored after.

**The chunked runner does NOT do this for the delta arm.**
`_chunked_official_full_history_answer` (`_chunked_eval_runner.py:478-616`)
calls `session._decode_generate` (602) which only flips
`set_delta_mem_write_message_ids`/`sentence_ids` (641-642), never
`write_enabled`. So decode is paying the full write-path cost for every
generated answer token.

**Would freezing state at end-of-prefill break correctness?** The paper
(`delta-Mem/README.md:25`, arXiv:2502.07466 abstract) frames the state as
representing the *interaction history* — i.e. tokens written into it are
the **conversation context being remembered**, not the model's own
in-progress answer. On LoCoMo, the question being answered is what the
state should help retrieve from; the answer tokens being generated are
not future "memory" the model needs to attend back to in the same QA
turn. The vendored base-eval path implicitly relies on this same
property when it sets `write_enabled=False` during `model.generate`.
The README's "long-term agent scenarios" framing (line 25) explicitly
positions writes as occurring "when a new token or interaction segment
arrives" — i.e. on inputs, not on model outputs within a single decode.

So the empirical risk reduces to: do answer tokens within a single
QA turn carry information the model needs to retrieve back from
`delta_state` via `delta_q` later in the same answer? Given answers are
short (max_new_tokens is bounded by `OFFICIAL_*` in
`deltamem/eval/locomo_protocol.py`, typically ≤256) and the state has
already absorbed the full conversation history, this is unlikely to
shift LoCoMo scores. **Recommended action:** wrap the
`session._decode_generate` call inside `_chunked_official_full_history_answer`
with `set_delta_mem_write_enabled(model, False)` /
`set_delta_mem_write_enabled(model, True)` in a try/finally — matching
the pattern already used at `locomo_delta.py:570/594`. Validate on
conv-0 first; if scores match the current delta arm to ≤1 F1 point,
roll out.

## 5. OSCARCache crop compatibility

`OSCARCache` (`third_party/oscar-transformers/oscar_transformers/cache.py`)
inherits from `transformers.Cache`. `OSCARCacheLayer` (line 37 onward)
implements `get_seq_length`, `reset`, `reorder_cache` (NotImplementedError)
and `update` — but **does not override `crop`**. The base
`Cache.crop` is a no-op stub on most transformer Cache subclasses; for
quantized caches with sink+middle+recent partitioning, the default
will crop nothing or corrupt the partitioning.

The structural problem is more fundamental than a missing override.
The layer's storage (`cache.py:62-67, 145-159`) partitions tokens into:

- `sink_k/v` — first `sink_tokens=64` tokens, full precision.
- `middle_k/v` — INT2-quantized blocks of spilled tokens (concatenated
  via `concat_blocks`), with a parallel dequantized cache
  `_middle_k_dq` / `_middle_v_dq` (167-172) kept incrementally.
- `recent_k/v` — last `recent_tokens=256` tokens, full precision.

A `crop(target_len)` operation would need to: (a) leave `sink` alone
when `target_len > sink_tokens`, (b) re-slice `middle` at a specific
INT2-block boundary — and INT2 blocks are group-128 quantized, so
crops that don't fall on group boundaries either truncate a partial
group (losing the quantization metadata for that group) or require
re-quantization, (c) re-slice `recent` and possibly *move tokens back
from middle to recent* if cropping below `total - recent_tokens`.

Plus, the dequantized parallel cache (`_middle_k_dq`/`_middle_v_dq`,
167-172) has to be kept consistent with the truncated INT2 `middle_k`
— if you crop one but not the other they drift and subsequent
`_assemble` calls (176-191) return inconsistent K/V.

**Implementability:** A correct `crop` is possible if you only ever
crop on group boundaries (multiples of `group_size=128`) and only
truncate from the end of `middle`/`recent` (never reach back into
`sink`). For our use case — cropping back to `history_len` where
`history_len` is the per-conversation shared prefix length — those
constraints can't be guaranteed: `history_len` is whatever the
tokenizer produces for the shared chat-template prefix
(`_chunked_eval_runner.py:339-385`), which has no relationship to 128.

**Recommendation:** the cleanest path is **incremental** — don't crop,
just snapshot/restore the OSCARCache at the history checkpoint. Mirror
what we already do for `delta_state` (`_chunked_eval_runner.py:388-407`).
The snapshot is fixed-size per layer (~`sink + middle_quant +
recent`-bounded), so per-conversation memory is bounded. This requires
a `_snapshot_oscar_cache` / `_restore_oscar_cache` pair that clones
each layer's six tensors plus the two `_dq` caches — straightforward,
no upstream changes. The result would restore the ~17k-token prefill
saving the bf16 path currently gets, at the cost of one fixed-size
snapshot per conversation (~bounded by sink_tokens + middle bits +
recent_tokens, far smaller than a full bf16 cache).

This is a **higher-leverage win than Triton scan**: prefill is ~17.6k
tokens per question vs a few hundred decode tokens. The current
`KV_CACHE_BACKEND != "bf16"` blanket disable
(`_chunked_eval_runner.py:514-516`) costs us a full re-prefill per
question across the OSCAR arm of the eval — almost certainly worth
more wall-clock than every other lever in this report combined.

## Suggested priority for follow-up

1. **Snapshot/restore OSCARCache across questions** — eliminates the
   per-question 17k-token re-prefill that the current crop-disable
   blocks. Highest expected impact.
2. **Disable delta-mem writes during decode** — wrap
   `session._decode_generate` in the chunked runner with
   `set_delta_mem_write_enabled(False)`. Eliminates Cost #1 in §1 for
   answer tokens; validate scores on conv-0 first.
3. **Install Triton (WSL2 or `triton-windows`)** — flips the scan path
   automatically via `scan_impl="auto"` default. No code change needed
   beyond removing the env-var pin in
   `_chunked_eval_runner.py:25`. Closes the residual prefill scan
   overhead, which still matters for the chunked-prefill path even
   after (1) and (2) land.

Items 1 and 2 are independent and can land in either order.
