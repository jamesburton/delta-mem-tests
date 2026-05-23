# R1 GATE: verify deltamem.kernels builds & imports.
# Exit code 0 = gate PASS, 1 = gate FAIL (fall back to WSL2 per the spec).

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)  # repo root

Write-Host "=== Importing deltamem.kernels (triggers JIT build if needed) ==="

$probe = @'
import sys
import importlib
import traceback

mods = [
    "deltamem",
    "deltamem.core",
    "deltamem.kernels",
]

for m in mods:
    try:
        importlib.import_module(m)
        print(f"OK  import {m}")
    except Exception:
        print(f"FAIL import {m}")
        traceback.print_exc()
        sys.exit(1)

print("ALL_KERNELS_IMPORTED")
'@

uv run python -c $probe
if ($LASTEXITCODE -ne 0) {
    Write-Host "GATE: FAIL"
    exit 1
}

Write-Host "GATE: PASS"
exit 0
