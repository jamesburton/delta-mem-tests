# Environment bring-up

Prerequisites on native Windows 11:

1. Python 3.10–3.12 on `PATH` (`python --version`).
2. NVIDIA driver (recent) + CUDA 12.x runtime support. The CUDA Toolkit (with
   `nvcc`) is only required later, for compiling delta-mem's custom kernels in
   Task 3. For Task 2, the driver alone is sufficient because `torch==2.5.1+cu121`
   ships its own CUDA runtime libraries.
3. Visual Studio Build Tools (MSVC) installed — required by Task 3 for
   ninja-driven kernel compilation, not Task 2. **Also required for the
   2-bit KV experiments** (`optimum-quanto` JIT-compiles a CUDA extension
   the first time it dequantises). Tick the "Windows 11 SDK" individual
   component too — vcvars64.bat puts `cl.exe` on PATH but won't find the
   C runtime headers (`stddef.h` etc.) without the SDK. Quickest install:
   `winget install --id Microsoft.WindowsSDK.10.0.26100`.

   **CUDA 13.x + VS 2025 caveat:** CUDA 13.x's `host_config.h` only allows
   VS 2019–2022; with VS 2025 Build Tools, NVCC errors out unless given
   `-allow-unsupported-compiler`. `env/vsenv.ps1` injects this via
   `NVCC_PREPEND_FLAGS`. Re-source the MSVC env in a fresh PowerShell via:
   ```powershell
   . .\env\vsenv.ps1
   ```
   then run the eval as usual.

Run from the repo root in PowerShell:

```powershell
./env/setup.ps1
```

The script installs `uv`, creates `.venv`, installs project deps, replaces the
default CPU torch with `torch==2.5.1+cu121`, and registers the vendored
`delta-Mem` directory as an importable path via a `.pth` file in site-packages.

The sanity output should report `cuda=True`, the RTX 3060, and
`deltamem importable: True`.

## KV-cache backends

The chunked eval runner selects between several KV-cache implementations via
`KV_CACHE_BACKEND`:

| Backend      | `KV_CACHE_BITS` default | Extra env vars                              |
|--------------|------------------------:|---------------------------------------------|
| `bf16`       | (n/a)                   | none — model creates a `DynamicCache`       |
| `turboquant` | 4                       | none                                        |
| `quanto`     | 2                       | none (broken on Windows; see above)         |
| `hqq`        | 2                       | none                                        |
| `oscar`      | 2                       | `OSCAR_K_ROTATION_PATH`, `OSCAR_V_ROTATION_PATH` (required); `OSCAR_SINK_TOKENS` (=64), `OSCAR_RECENT_TOKENS` (=256), `OSCAR_K_CLIP` (=0.96), `OSCAR_V_CLIP` (=0.92), `OSCAR_GROUP_SIZE` (=128) (optional, defaults match RotationZoo) |

The OSCAR backend is implemented in the sibling package
`third_party/oscar-transformers` (git submodule). It pre-bakes rotation
matrices into the model's Q/K/V/O projections on first cache construction —
this mutates the in-memory model permanently for the process, so do not mix
`oscar` runs with other backends in the same Python session.

Download rotations from
[huggingface.co/Zhongzhu/OSCAR-RotationZoo](https://huggingface.co/Zhongzhu/OSCAR-RotationZoo).
We use the `Qwen3-4B-Thinking-2507` rotations on `Qwen3-4B-Instruct-2507` as a
first-pass transfer test before running our own calibration; see
`report/tier1-summary.md` Appendix C for why this stage is necessary at all.

## Why not `pip install -e ./delta-Mem`?

The vendored `delta-Mem` repo has no `pyproject.toml`/`setup.py` of its own.
Rather than modify the vendored code (which would compromise reproduction
fidelity), the setup script writes `.venv/Lib/site-packages/deltamem.pth`
containing the absolute path of `<repo>/delta-Mem`. Python loads `.pth` files
at interpreter startup, so `import deltamem` resolves to `delta-Mem/deltamem/`.
