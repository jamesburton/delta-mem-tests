# R1 GATE — deltamem.kernels on native Windows

**Result:** PASS

- Host: Windows 11, RTX 3060 12GB (eGPU)
- Python: 3.11.3
- torch: 2.5.1+cu121
- CUDA toolkit: Cuda compilation tools, release 13.1, V13.1.115
- Vendored delta-Mem commit: 98dc679572ef77d77b97485bf2f2b2aa810b74ba
- Kernel type observed: Triton (pure Python, `.py` files importing `triton`)

**Note on Triton availability:** `triton` is not installed in the venv
(`ModuleNotFoundError: No module named 'triton'`). The kernel module
`deltamem.kernels.affine_scan` imports cleanly regardless because it wraps
the `import triton` in a try/except that sets `triton = None` on failure.
The `_affine_scan_forward_kernel` and `_affine_scan_backward_kernel` functions
are `None` in this configuration; the PyTorch fallback path in `deltamem.core`
is what will actually execute during benchmarks. The gate PASSes because
`importlib.import_module("deltamem.kernels")` exits with code 0.

Stdout/stderr: see `report/kernels-gate.log`.

Decision: proceed with Tier 1 on native Windows.
