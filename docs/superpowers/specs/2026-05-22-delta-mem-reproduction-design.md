# delta-Mem Reproduction — Design

**Date:** 2026-05-22
**Status:** Approved (design); Tier 1 detailed, Tiers 2–3 sketched
**Scope:** Full three-tier vision as roadmap; Tier 1 ("Reproduce & verify") designed for implementation.

---

## 1. Background

`delta-Mem` (δ-mem) is a lightweight online-memory mechanism for LLMs, published as
arXiv 2605.12357 (declare-lab, May 2026). It augments a **frozen** LLM backbone with a
compact associative-memory state, updated via the *delta rule* as tokens stream in and
read back as low-rank corrections to attention. It needs no context-window growth, no
vector index, and no backbone fine-tuning.

- **Overhead:** rank-8 LoRA-style adapters on the Q/O projections plus a small online
  memory matrix — roughly 0.12% of parameters.
- **Reported results:** ~1.20× on LoCoMo, 1.31× on MemoryAgentBench, ~1.10× average over
  the frozen backbone, with general capabilities largely preserved.
- **Released weights:** `declare-lab/delta-mem_qwen3_4b-instruct` (base
  `Qwen/Qwen3-4B-Instruct-2507`), plus Qwen3-8B and SmolLM3-3B variants.
- **Constraint:** not a mergeable PEFT adapter. The official codebase must be present at
  runtime — the adapter is attached via `deltamem.core.attach_delta_mem` /
  `load_delta_mem_adapter`, and memory read/write happens inside attention.

### Why now

The paper is weeks old. Three contributions are unclaimed and individually shareable:
an **independent reproduction**, a **non-Python runtime**, and an answer to **"does the
technique transfer to a newer Qwen model?"** This project pursues all three, in order.

### References

- Article: https://venturebeat.com/orchestration/a-0-12-parameter-add-on-gives-ai-agents-the-working-memory-rag-cant
- Code: https://github.com/declare-lab/delta-Mem
- Weights: https://huggingface.co/declare-lab/delta-mem_qwen3_4b-instruct
- Paper: https://huggingface.co/papers/2605.12357

---

## 2. Vision — three-tier roadmap

Each tier is an independently shippable artifact and gets its own spec → plan cycle.

### Tier 1 — Reproduce & verify *(this spec)*

Stand up the official repo on local hardware, load the released Qwen3-4B δ-mem adapter,
and independently reproduce the **LoCoMo** result. Output: a reproduction report. Every
later claim rests on "we reproduced it independently."

### Tier 2 — Re-host the runtime *(sketch — own spec later)*

The README's instinct is that a Python-only runtime is awkward for a .NET-centric stack.
Two paths, **not interchangeable in effort**:

- **TorchSharp port** — *realistic.* TorchSharp is PyTorch bindings for .NET; the δ-mem
  read/write path can be mirrored from the Python implementation. This is the credible
  "first .NET implementation of online associative memory" artifact.
- **LLamaSharp patch** — *much harder.* LLamaSharp wraps llama.cpp (GGUF, C++ attention
  kernels). Because δ-mem fuses low-rank corrections **into** attention, this means
  patching llama.cpp's C++ attention kernel — not a thin .NET wrapper. Treated as a
  stretch/research item, not a peer of the TorchSharp port.

A pragmatic intermediate (a Python inference service callable from C# over a thin API)
is the low-risk fallback if the TorchSharp port stalls.

### Tier 3 — Generalize to a newer model *(sketch — own spec later)*

Attach the released δ-mem adapters to the newest Qwen instruct model (e.g. Qwen3.6) and
test: do the adapters attach at all (architecture compatibility)? If not, does the
technique re-train cleanly? How reliable is it? A clean "δ-mem works on Qwen3.6, here
are the numbers" demo is the headline, world-class example. Re-training is a cloud or
Strix Halo job (the 3060's 12GB is inference-only). **Stretch:** δ-mem on AMD/ROCm using
the Strix Halo machine — its own research detour, not a Tier 3 dependency.

---

## 3. Tier 1 — detailed design

### 3.1 Approach

**Approach A — vendored repo + thin Windows-compat harness.** Vendor the official
delta-Mem repo at a pinned commit; add only a Windows-compat shim, a runner that invokes
*their* LoCoMo eval verbatim, and a reproduction report.

Rejected alternatives:

- **B — clean-room reimplementation.** Reimplementing the eval harness means any
  methodology drift makes us measure our own thing, not reproduce theirs. Forfeits the
  word "reproduction."
- **C — hybrid.** Essentially A with a thicker wrapper. We adopt C's report layer but
  keep the wrapper as thin as possible.

**Decision:** A, with a dedicated report layer borrowed from C.

### 3.2 Environment

- **Hardware:** RTX 3060 12GB (eGPU). Ampere — CUDA-capable.
- **OS:** native Windows 11 (chosen deliberately; WSL2 is the documented fallback).
- **Model:** `Qwen/Qwen3-4B-Instruct-2507` in bf16 (~8GB weights) + the δ-mem adapter.
- **Attention:** `attn_implementation="sdpa"`. FlashAttention is **not** a delta-Mem
  dependency (absent from `requirements.txt`), so no FlashAttention-on-Windows problem.
- **DeepSpeed:** a delta-Mem dependency, but only training needs it. Tier 1 is
  inference + eval — DeepSpeed is not exercised.
- **Dependency manager:** `uv` (per the repo's `setup_uv_env.sh`).

### 3.3 Components

```
delta-mem-tests/
├── delta-Mem/           # vendored official repo, pinned commit (submodule or copy)
├── env/                 # Windows bring-up: setup notes + PowerShell setup script
├── run/                 # repro runner: PowerShell wrappers + thin Python entrypoint
├── report/              # reproduction report (committed artifact)
└── docs/superpowers/specs/
```

1. **`delta-Mem/`** — the official repo at a pinned commit hash. The single source of
   truth for model loading, kernels, and the LoCoMo eval.
2. **`env/`** — documented native-Windows bring-up: `uv` environment creation, CUDA
   toolkit + MSVC setup for the `deltamem/kernels` ninja build, and the
   `attn_implementation` setting. Bash scripts from the repo are wrapped/translated to
   PowerShell.
3. **`run/`** — a thin Python entrypoint plus PowerShell wrappers that: load the base
   model + δ-mem adapter on the 3060, run the chat-demo smoke test, then invoke the
   repo's LoCoMo eval module with a recorded, exact config.
4. **`report/`** — the reproduction report: hardware, vendored commit hash, exact eval
   config, our LoCoMo δ-mem-vs-frozen-backbone ratio against the paper's 1.20×, the
   tolerance verdict, peak VRAM, and every deviation or skipped sample.

### 3.4 Data flow

```
HF Hub ── Qwen3-4B-Instruct-2507 (base)
       ── delta-mem_qwen3_4b-instruct (adapter)
       ── LoCoMo dataset (via `datasets`, pulled by the eval module)
            │
            ▼
   load base (bf16, sdpa) → attach_delta_mem → load_delta_mem_adapter   [on RTX 3060]
            │
            ▼
   chat-demo smoke test  →  LoCoMo eval loop  →  scores  →  report/
```

### 3.5 Named risks

Each risk is an explicit task in the implementation plan, not a footnote.

- **R1 — kernel build on native Windows *(the gate)*.** `deltamem/kernels/` plus the
  `ninja` dependency mean custom compiled kernels for the delta-rule write path. The
  **first task** is a go/no-go: do these kernels compile natively (MSVC + CUDA toolkit)?
  If they are Triton-based, native Windows support is shaky. On failure, the documented
  fallback order is **WSL2 (same 3060)**, then **cloud Linux GPU**. No further Tier 1
  work proceeds until this gate passes.
- **R2 — 12GB VRAM vs LoCoMo long context.** Qwen3-4B bf16 is ~8GB of weights; LoCoMo
  conversations are long, so the KV cache — not the model — is the OOM risk. The plan
  fixes a bounded evaluation window and KV-cache dtype up front (δ-mem's own premise is
  that it does not need the full context in the window). **No silent sample-skipping:**
  any sample that cannot be evaluated is recorded in the report as an explicit asterisk.
- **R3 — methodology fidelity.** The repo's eval is used verbatim at a pinned commit
  with the exact config recorded. A number that diverges beyond tolerance is a finding
  to investigate and document — never to smooth over.

### 3.6 Success criteria

- The chat demo runs on the 3060 (native Windows) with live memory read/write observed.
- **LoCoMo reproduced:** our δ-mem-vs-frozen-backbone ratio is within **±0.05** of the
  paper's reported 1.20× — or, if outside, reported honestly with investigation notes.
- The reproduction report is committed, with all asterisks (skipped samples, peak VRAM,
  config deviations) explicit.
- **Stretch:** MemoryAgentBench (paper: 1.31×) reproduced as a second data point.

### 3.7 Out of scope for Tier 1

- The full five-benchmark suite (HotpotQA, IFEval, GPQA Diamond) — verifies
  no-regression, deferred.
- Any training or re-training (no DeepSpeed exercised).
- The .NET runtime (Tier 2) and newer-model generalization (Tier 3).

---

## 4. Testing strategy

Tier 1 is a reproduction, not a feature build — "tests" are verification gates:

1. **Kernel-build gate (R1):** the vendored repo's kernels compile and import on the
   target environment.
2. **Smoke test:** the chat demo loads the model + adapter and exhibits memory
   read/write across turns.
3. **Reproduction gate:** the LoCoMo eval completes and the ratio lands within the
   ±0.05 tolerance band — or the deviation is documented.

---

## 5. Open questions for later tiers

- Tier 2: TorchSharp parity testing against the Tier 1 Python reference (the reproduction
  gives us a known-good number to diff against).
- Tier 3: which Qwen model is "newest" at the time, and whether the released adapter
  tensor shapes map to its architecture without re-training.
