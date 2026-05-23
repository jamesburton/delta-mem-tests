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

# 3. Install project deps. Torch comes in as CPU here; step 4 replaces it.
uv pip install -e ".[dev]"

# 4. Replace torch with a CUDA build pinned to a version that exists on the
#    cu121 index. The 3060 is Ampere (sm_86) and is fully supported by cu121.
#    --reinstall is required because step 3 already installed CPU torch; without
#    --reinstall, uv compares versions across indexes and keeps the (newer) CPU
#    wheel from PyPI.
uv pip install --reinstall "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121

# 5. Make the vendored delta-Mem importable. The repo has no pyproject.toml, so
#    `uv pip install -e ./delta-Mem` cannot work. Instead drop a .pth file in the
#    venv's site-packages — Python loads .pth files at interpreter startup and
#    appends their contents to sys.path.
$absDeltaMem = (Resolve-Path "./delta-Mem").Path
$sitePackages = ".venv\Lib\site-packages"
Set-Content -Path "$sitePackages\deltamem.pth" -Value $absDeltaMem -Encoding ASCII
Write-Host "Wrote $sitePackages\deltamem.pth -> $absDeltaMem"

# 6. Sanity check: CUDA torch + importable deltamem.
uv run python -c @"
import torch, importlib
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
spec = importlib.util.find_spec('deltamem')
print(f'deltamem importable: {spec is not None} ({spec.origin if spec else None})')
"@
