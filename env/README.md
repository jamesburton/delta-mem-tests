# Environment bring-up

Prerequisites on native Windows 11:

1. Python 3.10–3.12 on `PATH` (`python --version`).
2. NVIDIA driver (recent) + CUDA 12.x runtime support. The CUDA Toolkit (with
   `nvcc`) is only required later, for compiling delta-mem's custom kernels in
   Task 3. For Task 2, the driver alone is sufficient because `torch==2.5.1+cu121`
   ships its own CUDA runtime libraries.
3. Visual Studio Build Tools (MSVC) installed — required by Task 3 for
   ninja-driven kernel compilation, not Task 2.

Run from the repo root in PowerShell:

```powershell
./env/setup.ps1
```

The script installs `uv`, creates `.venv`, installs project deps, replaces the
default CPU torch with `torch==2.5.1+cu121`, and registers the vendored
`delta-Mem` directory as an importable path via a `.pth` file in site-packages.

The sanity output should report `cuda=True`, the RTX 3060, and
`deltamem importable: True`.

## Why not `pip install -e ./delta-Mem`?

The vendored `delta-Mem` repo has no `pyproject.toml`/`setup.py` of its own.
Rather than modify the vendored code (which would compromise reproduction
fidelity), the setup script writes `.venv/Lib/site-packages/deltamem.pth`
containing the absolute path of `<repo>/delta-Mem`. Python loads `.pth` files
at interpreter startup, so `import deltamem` resolves to `delta-Mem/deltamem/`.
