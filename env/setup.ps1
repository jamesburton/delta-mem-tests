# Brings up the Python env for delta-Mem Tier 1 reproduction on native Windows.
# Assumes: Windows 11, Python 3.10+ installed, NVIDIA driver (recent),
# Visual Studio Build Tools (MSVC) for ninja-driven kernel compilation in Task 3.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)  # repo root

# 1. Install uv if missing
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    Invoke-RestMethod -Uri https://astral.sh/uv/install.ps1 | Invoke-Expression
}

# 2. Create venv (uv-managed, project-local)
uv venv --python 3.11

# 3. Install project deps. Torch comes from the pytorch-cu121 index because
#    pyproject.toml routes it there via [tool.uv.sources] — no separate
#    reinstall step is needed, and `uv run` cannot clobber the GPU wheel.
uv pip install -e ".[dev]"

# 4. Make the vendored delta-Mem importable. The vendored repo has no
#    pyproject.toml, so drop a .pth file pointing at it.
$absDeltaMem = (Resolve-Path "./delta-Mem").Path
$sitePackages = ".venv\Lib\site-packages"
Set-Content -Path "$sitePackages\deltamem.pth" -Value $absDeltaMem -Encoding ASCII
Write-Host "Wrote $sitePackages\deltamem.pth -> $absDeltaMem"

# 5. Sanity check.
uv run python -c @"
import torch, importlib
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
spec = importlib.util.find_spec('deltamem')
print(f'deltamem importable: {spec is not None} ({spec.origin if spec else None})')
"@
