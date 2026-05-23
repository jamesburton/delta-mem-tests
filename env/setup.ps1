# Brings up the Python env for delta-Mem Tier 1 reproduction on native Windows.
# Assumes: Windows 11, Python 3.10+ installed, NVIDIA driver + CUDA 12.x toolkit,
# Visual Studio Build Tools (MSVC) for ninja-driven kernel compilation.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)  # repo root

# 1. Install uv if missing
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    Invoke-RestMethod -Uri https://astral.sh/uv/install.ps1 | Invoke-Expression
}

# 2. Create venv (uv-managed, project-local)
uv venv --python 3.11

# 3. Install project deps (CPU index first; torch CUDA wheel added explicitly)
uv pip install -e ".[dev]"

# 4. Reinstall torch from the CUDA 12.1 index to get GPU-enabled wheels
#    (3060 is Ampere, compute capability 8.6 — fully supported.)
uv pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121

# 5. Install the vendored delta-Mem in editable mode so `import deltamem.*` works
uv pip install -e ./delta-Mem

# 6. Print a one-line CUDA sanity report
uv run python -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
