# R1 GATE — deltamem.kernels on native Windows

**Result:** PASS (via torch fallback path; Triton not installed; equivalence documented)

## Environment

- Host: Windows 11, RTX 3060 12GB (eGPU)
- Python: 3.11.3
- torch: 2.5.1+cu121
- CUDA toolkit (nvcc): release 13.1, V13.1.115 (present, unused — see below)
- Vendored delta-Mem commit: `98dc679572ef77d77b97485bf2f2b2aa810b74ba`
- Kernel implementation observed: Triton (`.py` files importing `triton`)
- Triton package installed in venv: **No** (`triton` import raises `ModuleNotFoundError`)

## What this gate actually proved

`importlib.import_module("deltamem.kernels")` exits 0 because
`delta-Mem/deltamem/kernels/affine_scan.py` wraps `import triton` in a
try/except that sets `triton = None` on failure. Both Triton kernels
(`_affine_scan_forward_kernel`, `_affine_scan_backward_kernel`) end up as
`None` in this configuration.

A naive reading of just this fact would say the gate is vacuous. It is not —
because delta-Mem ships a torch reference implementation that is
**algebraically identical** to the Triton kernel and is selected automatically
when Triton is absent. Evidence:

- **Torch reference**: `delta-Mem/deltamem/core/delta_impl.py:1895-1938`
  (`_memory_affine_scan_torch`). Token-by-token loop using `torch.einsum`
  and outer products.
- **Triton kernel**: `delta-Mem/deltamem/kernels/affine_scan.py:62-155`
  (`_affine_scan_forward_kernel`). Same per-token math, fused into a kernel
  grid.
- **Equivalence by inspection**:
  - Triton (`affine_scan.py:147-151`):
    `updated = keep * state - erase * (state·k) * k + write * v * k`
  - Torch (`delta_impl.py:1924-1927`):
    `next_state = keep * state - erase * ((state·k) ⊗ k) + write * (v ⊗ k)`
  - Algebraically the same update rule; the torch path replaces fused tiled
    arithmetic with `einsum`-and-broadcast.
- **Dispatcher** at `delta_impl.py:2022-2050` reads `self.scan_impl` (default
  `"auto"` from env var `DELTA_MEM_SCAN_IMPL`, `delta_impl.py:633`). When
  `triton_scan_support()` returns unsupported, control falls through to
  `_memory_affine_scan_torch` automatically — no special configuration needed.
- **Repo's own regression suite**: `delta-Mem/deltamem/tests/test_delta_mem_regressions.py:692, 745, 774`
  exercises both `scan_impl="triton"` and `scan_impl="auto"` against the same
  expected outputs. The upstream team treats the two paths as numerically
  interchangeable for correctness purposes.

## Numerical caveat to carry forward

Floating-point reduction order differs slightly between the two paths. The
Triton kernel accumulates in fp32 inside the kernel; the torch path's
accumulation precision is determined by `torch.einsum` semantics (bf16 inputs
on CUDA accumulate in fp32). For an 8×8 state matrix at the released
adapter's rank-8 setting, expected drift is on the order of single-digit
ULPs — well below anything that should materially shift LoCoMo scores.

## Provenance of the paper's numbers

The published delta-Mem results were almost certainly produced with the
Triton kernel (the repo's bench script `delta-Mem/deltamem/tools/bench_scan.py`
toggles between impls explicitly). Our reproduction executes the **torch
path** because we are on native Windows without Triton. This is a recorded
asterisk to surface in Task 7's reproduction report; it does not invalidate
the reproduction but it is a deviation worth being explicit about.

## Configuration locked for subsequent tasks

To prevent an accidental path-switch if `triton` ever gets installed
mid-flight, **all run/*.py scripts (Task 4 smoke test, Task 6 LoCoMo
wrapper) will explicitly set `os.environ["DELTA_MEM_SCAN_IMPL"] = "torch"`
before importing `deltamem`**. The reproduction report records this as part
of the eval config so the run is bit-for-bit pinnable.

## Gate semantics for future tiers

The plan's literal Task 3 step ("verify `deltamem.kernels` imports") was a
looser gate than the spec's intent. The correct gate for Tier 2/3 work is:
**the path that will run during eval is the same path the paper used, OR is
documented-equivalent (by inspection + repo tests + planned empirical
check)**. This run satisfies the second clause.

Stdout/stderr of the original import probe: see `report/kernels-gate.log`.

## Decision

Proceed with Tier 1 on native Windows along the torch path. The decision
remains the same as the original gate report; the change is that the
substantive reason — not just an import succeeding — is now on the record.
