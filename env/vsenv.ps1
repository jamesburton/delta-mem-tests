# Source the Visual Studio 2025 Build Tools environment (cl.exe + INCLUDE + LIB)
# into the current PowerShell session so torch.cpp_extension's JIT compile path
# can find MSVC. Used by optimum-quanto and any other library that triggers
# torch.utils.cpp_extension.load() at runtime on Windows.
#
# Run via dot-sourcing in PowerShell so the env vars persist in the caller:
#     . .\env\vsenv.ps1
#
# Auto-detects vcvars64.bat at the known Visual Studio 18 (2025) install path.
# Override $VCVARS to point at a different vcvars64.bat (e.g. VS 17 / 2022).

$VCVARS = "${env:VCVARS}"
if (-not $VCVARS) {
    $VCVARS = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
}
if (-not (Test-Path $VCVARS)) {
    Write-Error "vcvars64.bat not found at $VCVARS. Set `$env:VCVARS` to the correct path and re-source this script."
    exit 1
}

Write-Host "Sourcing $VCVARS ..."
# Capture the env after vcvars64.bat runs and apply to this PowerShell session.
$envOutput = cmd /c "`"$VCVARS`" >nul 2>nul && set"
foreach ($line in $envOutput) {
    if ($line -match "^([^=]+)=(.*)$") {
        Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
    }
}
$cl = (Get-Command cl.exe -ErrorAction SilentlyContinue)
if ($cl) {
    Write-Host "MSVC environment loaded. cl.exe at: $($cl.Source)"
} else {
    Write-Error "Sourced vcvars64.bat but cl.exe still not on PATH. Investigate."
}
