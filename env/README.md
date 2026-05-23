# Environment bring-up

Prerequisites on native Windows 11:

1. Python 3.10–3.12 on `PATH` (`python --version`).
2. NVIDIA driver (recent) + CUDA Toolkit 12.x installed and on `PATH`
   (`nvcc --version` should report 12.x).
3. Visual Studio Build Tools (MSVC) installed for ninja-driven kernel compilation.

Run from the repo root in PowerShell:

```powershell
./env/setup.ps1
```

The script installs `uv`, creates `.venv`, installs project deps plus the
vendored `delta-Mem`, and prints a CUDA sanity line. The last line should report
`cuda=True` and your RTX 3060.
