# delta-Mem Tier 1 Reproduction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Independently reproduce the delta-Mem LoCoMo result (~1.20× over frozen Qwen3-4B-Instruct-2507) on a local RTX 3060 12GB running native Windows 11, and produce a committed reproduction report.

**Architecture:** Vendor the official `declare-lab/delta-Mem` repo at a pinned commit and add a thin Windows-compat harness (PowerShell + Python wrappers) that invokes *their* eval verbatim. A small custom report generator (the only TDD-shaped piece) reads the eval scores and produces a markdown report comparing our number to the paper's 1.20× within a ±0.05 tolerance band.

**Tech Stack:** Python 3.10+, `uv` (dependency manager), PyTorch w/ CUDA (bf16), `transformers`, HuggingFace `datasets`, pytest. Vendored `declare-lab/delta-Mem`. PowerShell for Windows bring-up.

**Companion spec:** `docs/superpowers/specs/2026-05-22-delta-mem-reproduction-design.md`

---

## File Structure

```
delta-mem-tests/
├── delta-Mem/                          # git submodule, pinned to a commit
├── env/
│   ├── setup.ps1                       # uv + venv + deps + verify-CUDA
│   └── kernels-check.ps1               # R1 GATE: build & import deltamem.kernels
├── run/
│   ├── __init__.py
│   ├── smoke_chat.py                   # smoke test: load + chat + observe memory
│   ├── locomo_eval.py                  # thin wrapper invoking the vendored LoCoMo eval
│   └── report_gen.py                   # pure-logic report generator (TDD-built)
├── tests/
│   └── test_report_gen.py              # pytest for report_gen
├── report/
│   ├── kernels-gate.md                 # R1 GATE result, committed
│   ├── smoke.md                        # smoke-test transcript + memory-state evidence
│   ├── raw/                            # eval stdout, scores JSON, peak-VRAM log
│   └── reproduction-report.md          # the final committed artifact
├── pyproject.toml                      # project metadata + uv-managed deps
├── .gitignore
└── docs/superpowers/{specs,plans}/
```

**Responsibilities (one per file):**
- `env/setup.ps1` — bring up the Python environment; verifies CUDA + GPU only.
- `env/kernels-check.ps1` — gates Tier 1 on `deltamem.kernels` building and importing.
- `run/smoke_chat.py` — proves the loaded model exhibits memory read/write across turns.
- `run/locomo_eval.py` — invokes the vendored LoCoMo eval verbatim with a recorded config.
- `run/report_gen.py` — pure function from `(eval_scores, paper_ratio, tolerance, metadata)` to a markdown string. Unit-tested.

---

## Task 1: Scaffold the repo and vendor delta-Mem

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `run/__init__.py`
- Create: `tests/__init__.py`
- Create: `report/.gitkeep`, `report/raw/.gitkeep`
- Create: `env/.gitkeep`
- Add submodule: `delta-Mem/`

- [ ] **Step 1: Add `.gitignore`**

Create `.gitignore`:

```gitignore
# Python
__pycache__/
*.pyc
.venv/
.uv/
*.egg-info/

# Reports raw outputs (large) - keep dir, not contents
report/raw/*
!report/raw/.gitkeep

# Editor
.vscode/
.idea/
*.swp
```

- [ ] **Step 2: Add `pyproject.toml`**

Create `pyproject.toml`:

```toml
[project]
name = "delta-mem-tests"
version = "0.0.1"
description = "Independent reproduction of declare-lab/delta-Mem on RTX 3060 / Windows 11"
requires-python = ">=3.10,<3.13"
dependencies = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "huggingface-hub",
    "einops",
    "pydantic",
    "tqdm",
    "ninja",
    "psutil",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["run"]
```

- [ ] **Step 3: Create empty package + dir markers**

Create:
- `run/__init__.py` — single line: `"""Tier 1 reproduction harness."""`
- `tests/__init__.py` — empty
- `report/.gitkeep` — empty
- `report/raw/.gitkeep` — empty
- `env/.gitkeep` — empty

- [ ] **Step 4: Vendor delta-Mem as a submodule pinned to a specific commit**

Add the official repo as a submodule and pin to its current HEAD so future upstream changes can't silently alter our reproduction.

Run (PowerShell, from repo root):

```powershell
git submodule add https://github.com/declare-lab/delta-Mem.git delta-Mem
cd delta-Mem
git rev-parse HEAD | Tee-Object ../report/vendored-commit.txt
cd ..
```

Expected: `delta-Mem/` populated, `.gitmodules` written, `report/vendored-commit.txt` contains a 40-char SHA.

- [ ] **Step 5: Verify the vendored repo layout matches what the spec assumes**

Confirm these directories exist in `delta-Mem/`:

```powershell
Get-ChildItem delta-Mem/deltamem -Directory | Select-Object Name
```

Expected output includes at least: `core`, `demo`, `eval`, `kernels`, `runtime`.

If `kernels/` or `eval/` is missing, STOP — the spec's risks R1 and the LoCoMo runner assume both exist. Open an issue before proceeding.

- [ ] **Step 6: Commit**

```powershell
git add .gitignore pyproject.toml run/ tests/ report/ env/ .gitmodules delta-Mem
git commit -m "scaffold: project layout and vendor delta-Mem submodule"
```

---

## Task 2: Bring up the Python environment on native Windows

**Files:**
- Create: `env/setup.ps1`
- Create: `env/README.md`

- [ ] **Step 1: Write `env/setup.ps1`**

Create `env/setup.ps1`:

```powershell
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
```

- [ ] **Step 2: Write `env/README.md`**

Create `env/README.md`:

```markdown
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
```

- [ ] **Step 3: Run setup and verify CUDA is detected**

```powershell
./env/setup.ps1
```

Expected last line resembles: `torch=2.x.x+cu121 cuda=True dev=NVIDIA GeForce RTX 3060`.

If `cuda=False`, STOP — fix the NVIDIA driver / CUDA toolkit / matching torch wheel before proceeding. There is no point gating R1 on a CPU-only torch.

- [ ] **Step 4: Commit**

```powershell
git add env/setup.ps1 env/README.md
git commit -m "env: native-Windows bring-up via uv with CUDA torch"
```

---

## Task 3: R1 GATE — `deltamem.kernels` build & import on native Windows

This task is the spec's named go/no-go gate (risk R1). If it fails, Tier 1 falls back to WSL2; no later task in this plan runs on native Windows.

**Files:**
- Create: `env/kernels-check.ps1`
- Create: `report/kernels-gate.md`

- [ ] **Step 1: Inspect kernel type (Triton vs CUDA)**

```powershell
Get-ChildItem delta-Mem/deltamem/kernels -Recurse -File | Select-Object FullName
Get-ChildItem delta-Mem/deltamem/kernels -Recurse -File | Select-String -Pattern "triton" -List | Select-Object Path, LineNumber
```

Record whether the kernel files are `.py` (likely Triton) or `.cu` / `.cpp` (CUDA + ninja). This determines fallback behaviour:

- Pure CUDA + ninja → MSVC + CUDA toolkit handle it on Windows.
- Triton → native Windows Triton support is shaky; expect failure and fall back to WSL2.

- [ ] **Step 2: Write `env/kernels-check.ps1`**

Create `env/kernels-check.ps1`:

```powershell
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
```

- [ ] **Step 3: Run the gate**

```powershell
./env/kernels-check.ps1
```

Capture full stdout + stderr to a file:

```powershell
./env/kernels-check.ps1 2>&1 | Tee-Object report/kernels-gate.log
```

- [ ] **Step 4: Record the gate result in `report/kernels-gate.md`**

If PASS — create `report/kernels-gate.md`:

```markdown
# R1 GATE — deltamem.kernels on native Windows

**Result:** PASS

- Host: Windows 11, RTX 3060 12GB (eGPU)
- Python: <version>
- torch: <torch.__version__>
- CUDA toolkit: <nvcc --version output>
- Vendored delta-Mem commit: <from report/vendored-commit.txt>
- Kernel type observed: <Triton / CUDA-via-ninja>

Stdout/stderr: see `report/kernels-gate.log`.

Decision: proceed with Tier 1 on native Windows.
```

If FAIL — create `report/kernels-gate.md`:

```markdown
# R1 GATE — deltamem.kernels on native Windows

**Result:** FAIL

- Host: Windows 11, RTX 3060 12GB (eGPU)
- Python: <version>
- torch: <torch.__version__>
- Kernel type observed: <Triton / CUDA-via-ninja>
- Failure surface: <first error from report/kernels-gate.log, 5–10 lines>

Decision per spec: fall back to **WSL2 (same 3060 via CUDA passthrough)**.
Re-run Tasks 2 and 3 inside WSL2 (`./env/setup.ps1` ported, or `bash delta-Mem/scripts/setup_uv_env.sh`).
If WSL2 also fails, escalate to a cloud Linux GPU.

Tier 1 work on native Windows is halted here.
```

- [ ] **Step 5: Commit**

```powershell
git add env/kernels-check.ps1 report/kernels-gate.md report/kernels-gate.log
git commit -m "env: R1 gate result for deltamem.kernels on native Windows"
```

- [ ] **Step 6: Decision point**

- If gate PASSED: continue to Task 4.
- If gate FAILED: stop on native Windows. Open a new plan (or amend this one) for the WSL2 fallback path. The remaining tasks in this plan still apply — only the host changes.

---

## Task 4: Smoke test — load model + adapter, observe memory state changing across turns

**Files:**
- Create: `run/smoke_chat.py`
- Create: `report/smoke.md`

This is not a TDD task — it's an integration smoke test against the real model. Its value is binary: does memory read/write actually happen?

- [ ] **Step 1: Identify the right APIs in the vendored repo**

```powershell
Select-String -Path delta-Mem/deltamem/core/*.py -Pattern "def attach_delta_mem|def load_delta_mem_adapter|HFDeltaMemConfig" -SimpleMatch | Select-Object Path, LineNumber, Line
```

Confirm `attach_delta_mem`, `load_delta_mem_adapter`, and `HFDeltaMemConfig` are importable from `deltamem.core` (the HF model card uses these exact symbols). If their names have shifted in the pinned commit, update the imports in Step 2 to match.

- [ ] **Step 2: Write `run/smoke_chat.py`**

Create `run/smoke_chat.py`:

```python
"""Smoke test: load Qwen3-4B-Instruct-2507 + delta-mem adapter, run a 3-turn
chat, and assert that the online memory state has measurably changed between
turn 1 and turn 3.

Outputs:
    - report/smoke.md (transcript + evidence)
    - exit 0 on PASS, 1 on FAIL
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamem.core import (
    HFDeltaMemConfig,
    attach_delta_mem,
    load_delta_mem_adapter,
)

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = "declare-lab/delta-mem_qwen3_4b-instruct"
REPORT_PATH = Path("report/smoke.md")


def collect_state_signature(model: torch.nn.Module) -> dict[str, float]:
    """Sum |state| over any module attribute that looks like delta-mem state.

    We don't know the exact attribute name across versions; we walk modules and
    grab tensors named like 'delta', 'mem', or 'state'. The sum-of-abs over all
    of them gives a single scalar signature that we can compare across turns.
    """
    sig: dict[str, float] = {}
    for name, module in model.named_modules():
        for attr in ("delta_state", "memory_state", "mem_state", "state"):
            tensor = getattr(module, attr, None)
            if isinstance(tensor, torch.Tensor):
                sig[f"{name}.{attr}"] = float(tensor.detach().abs().sum().item())
    return sig


def chat_turn(model, tokenizer, history: list[dict], user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    prompt = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=1.0,
        )
    reply = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    history.append({"role": "assistant", "content": reply})
    return reply


def main() -> int:
    print(f"Loading base: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    print(f"Attaching delta-mem adapter: {ADAPTER}")
    cfg = HFDeltaMemConfig.from_pretrained(ADAPTER)
    attach_delta_mem(model, cfg)
    load_delta_mem_adapter(model, ADAPTER)
    model.eval()

    sig_pre = collect_state_signature(model)

    history: list[dict] = []
    turns = [
        "My favourite colour is teal. Remember that for later.",
        "Quick aside: what is 7 times 8?",
        "What did I tell you my favourite colour was?",
    ]
    transcript: list[tuple[str, str]] = []
    for t in turns:
        reply = chat_turn(model, tokenizer, history, t)
        transcript.append((t, reply))
        print(f"\nUSER: {t}\nASSISTANT: {reply}")

    sig_post = collect_state_signature(model)

    changed = {k: (sig_pre.get(k, 0.0), sig_post.get(k, 0.0))
               for k in sig_post
               if abs(sig_post[k] - sig_pre.get(k, 0.0)) > 1e-6}

    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    pass_state = len(changed) > 0
    pass_recall = "teal" in transcript[-1][1].lower()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_render_smoke_report(
        transcript=transcript,
        changed=changed,
        peak_mem_gb=peak_mem_gb,
        pass_state=pass_state,
        pass_recall=pass_recall,
    ), encoding="utf-8")

    print(json.dumps({
        "memory_tensors_changed": len(changed),
        "peak_vram_gb": round(peak_mem_gb, 2),
        "recall_pass": pass_recall,
        "state_changed_pass": pass_state,
    }, indent=2))

    return 0 if (pass_state and pass_recall) else 1


def _render_smoke_report(*, transcript, changed, peak_mem_gb, pass_state, pass_recall) -> str:
    lines = [
        "# Smoke test — delta-mem chat",
        "",
        f"- Memory tensors that changed across the 3-turn chat: **{len(changed)}**",
        f"- Final-turn recall of 'teal': **{'PASS' if pass_recall else 'FAIL'}**",
        f"- State-change gate: **{'PASS' if pass_state else 'FAIL'}**",
        f"- Peak VRAM: **{peak_mem_gb:.2f} GB**",
        "",
        "## Transcript",
        "",
    ]
    for user, assistant in transcript:
        lines.append(f"**USER:** {user}")
        lines.append("")
        lines.append(f"**ASSISTANT:** {assistant}")
        lines.append("")
    lines.append("## Memory-state changes (per-module signature sums)")
    lines.append("")
    if changed:
        for k, (a, b) in list(changed.items())[:30]:
            lines.append(f"- `{k}`: {a:.4f} → {b:.4f}")
        if len(changed) > 30:
            lines.append(f"- ... and {len(changed) - 30} more")
    else:
        lines.append("- (none — gate FAILED)")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run the smoke test**

```powershell
uv run python -m run.smoke_chat
```

Expected: prints USER/ASSISTANT turns, then a JSON block. Both `recall_pass` and `state_changed_pass` must be `true`. Exit code 0. `report/smoke.md` is written.

If `state_changed_pass` is `false`, the heuristic in `collect_state_signature` may not have caught the right attribute names — inspect a loaded model with `[n for n, _ in model.named_modules() if 'delta' in n.lower() or 'mem' in n.lower()]` and extend the attribute list. The gate must legitimately pass before proceeding.

- [ ] **Step 4: Commit**

```powershell
git add run/smoke_chat.py report/smoke.md
git commit -m "run: smoke test for delta-mem load + chat + memory state observation"
```

---

## Task 5: Build the report generator (pure logic, TDD)

This is the one TDD-shaped piece. The report generator is a pure function — given eval scores and metadata, return a markdown string with a PASS / FAIL / OUT-OF-BAND verdict against the paper's 1.20× ±0.05.

**Files:**
- Test: `tests/test_report_gen.py`
- Create: `run/report_gen.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_gen.py`:

```python
"""Unit tests for the LoCoMo reproduction report generator."""

from run.report_gen import render_report, verdict


def test_verdict_inside_tolerance_passes():
    assert verdict(our_ratio=1.20, paper_ratio=1.20, tolerance=0.05) == "PASS"
    assert verdict(our_ratio=1.16, paper_ratio=1.20, tolerance=0.05) == "PASS"
    assert verdict(our_ratio=1.24, paper_ratio=1.20, tolerance=0.05) == "PASS"


def test_verdict_outside_tolerance_is_oob():
    assert verdict(our_ratio=1.10, paper_ratio=1.20, tolerance=0.05) == "OUT_OF_BAND"
    assert verdict(our_ratio=1.30, paper_ratio=1.20, tolerance=0.05) == "OUT_OF_BAND"


def test_verdict_below_one_is_regression():
    assert verdict(our_ratio=0.95, paper_ratio=1.20, tolerance=0.05) == "REGRESSION"


def test_render_report_contains_required_sections():
    out = render_report(
        our_ratio=1.18,
        paper_ratio=1.20,
        tolerance=0.05,
        our_delta_mem_score=0.59,
        our_frozen_score=0.50,
        peak_vram_gb=10.5,
        skipped_samples=[(7, "OOM at 12k tokens"), (42, "OOM at 14k tokens")],
        vendored_commit="abc123def",
        eval_config={
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "adapter": "declare-lab/delta-mem_qwen3_4b-instruct",
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "max_seq_len": 8192,
        },
    )
    assert "# LoCoMo reproduction" in out
    assert "PASS" in out
    assert "1.18" in out
    assert "1.20" in out
    assert "±0.05" in out
    assert "abc123def" in out
    assert "Qwen/Qwen3-4B-Instruct-2507" in out
    assert "OOM at 12k tokens" in out
    assert "10.50" in out  # peak VRAM, two decimals


def test_render_report_marks_oob_clearly():
    out = render_report(
        our_ratio=1.10,
        paper_ratio=1.20,
        tolerance=0.05,
        our_delta_mem_score=0.55,
        our_frozen_score=0.50,
        peak_vram_gb=11.2,
        skipped_samples=[],
        vendored_commit="abc123",
        eval_config={"model": "x", "adapter": "y", "dtype": "bfloat16",
                     "attn_implementation": "sdpa", "max_seq_len": 8192},
    )
    assert "OUT_OF_BAND" in out
    # No silent papering-over: the deviation magnitude must be visible
    assert "0.10" in out  # 1.20 - 1.10
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
uv run pytest tests/test_report_gen.py -v
```

Expected: ImportError or ModuleNotFoundError on `run.report_gen` (the file doesn't exist yet).

- [ ] **Step 3: Write minimal `run/report_gen.py` to make tests pass**

Create `run/report_gen.py`:

```python
"""Pure-logic generator for the LoCoMo reproduction report.

Given eval scores and run metadata, produce a markdown report with a clear
verdict against the paper's 1.20x within a ±0.05 tolerance band.
"""

from __future__ import annotations

from typing import Literal

Verdict = Literal["PASS", "OUT_OF_BAND", "REGRESSION"]


def verdict(*, our_ratio: float, paper_ratio: float, tolerance: float) -> Verdict:
    """Classify our reproduction outcome.

    - REGRESSION: our ratio is below 1.0 (delta-mem hurt rather than helped).
    - PASS: our ratio is within `tolerance` of the paper's.
    - OUT_OF_BAND: everything else (deviation > tolerance, in either direction).
    """
    if our_ratio < 1.0:
        return "REGRESSION"
    if abs(our_ratio - paper_ratio) <= tolerance:
        return "PASS"
    return "OUT_OF_BAND"


def render_report(
    *,
    our_ratio: float,
    paper_ratio: float,
    tolerance: float,
    our_delta_mem_score: float,
    our_frozen_score: float,
    peak_vram_gb: float,
    skipped_samples: list[tuple[int, str]],
    vendored_commit: str,
    eval_config: dict,
) -> str:
    """Render the LoCoMo reproduction report as markdown."""
    v = verdict(our_ratio=our_ratio, paper_ratio=paper_ratio, tolerance=tolerance)
    deviation = abs(our_ratio - paper_ratio)

    lines: list[str] = [
        "# LoCoMo reproduction — delta-Mem on Qwen3-4B-Instruct-2507",
        "",
        f"**Verdict:** {v}",
        "",
        "## Headline",
        "",
        f"- Our delta-mem-vs-frozen-backbone ratio: **{our_ratio:.2f}×**",
        f"- Paper's reported ratio: **{paper_ratio:.2f}×**",
        f"- Tolerance band: **±{tolerance:.2f}**",
        f"- Deviation from paper: **{deviation:.2f}**",
        "",
        "## Scores",
        "",
        f"- delta-mem score: **{our_delta_mem_score:.4f}**",
        f"- frozen backbone score: **{our_frozen_score:.4f}**",
        "",
        "## Run metadata",
        "",
        f"- Vendored delta-Mem commit: `{vendored_commit}`",
        f"- Peak VRAM: **{peak_vram_gb:.2f} GB**",
        "",
        "### Eval config",
        "",
    ]
    for k, val in eval_config.items():
        lines.append(f"- `{k}`: `{val}`")
    lines.append("")

    lines.append("## Asterisks")
    lines.append("")
    if skipped_samples:
        lines.append(f"{len(skipped_samples)} sample(s) skipped (recorded, not silently dropped):")
        lines.append("")
        for idx, reason in skipped_samples:
            lines.append(f"- sample #{idx}: {reason}")
    else:
        lines.append("- None. All samples evaluated.")
    lines.append("")

    if v == "OUT_OF_BAND":
        lines.append("## Investigation note")
        lines.append("")
        lines.append(
            f"Our ratio differs from the paper by {deviation:.2f}, which exceeds "
            f"the ±{tolerance:.2f} tolerance. Per the spec, this is a finding to "
            "investigate, not to smooth over. See the raw outputs under "
            "`report/raw/` for the per-sample breakdown."
        )
        lines.append("")
    elif v == "REGRESSION":
        lines.append("## Investigation note")
        lines.append("")
        lines.append(
            f"Our ratio of {our_ratio:.2f} is below 1.0 — delta-mem failed to "
            "improve over the frozen backbone in our run. Treat as a failure of "
            "the reproduction; investigate before declaring Tier 1 complete."
        )
        lines.append("")

    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
uv run pytest tests/test_report_gen.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add run/report_gen.py tests/test_report_gen.py
git commit -m "run: report generator with verdict logic (TDD)"
```

---

## Task 6: LoCoMo eval wrapper

**Files:**
- Create: `run/locomo_eval.py`

This task is integration-shaped, not TDD-shaped — it wraps the vendored eval. We rely on Task 5's report generator (already tested) to render the final number.

- [ ] **Step 1: Identify the vendored LoCoMo entry point**

```powershell
Get-ChildItem delta-Mem/deltamem/eval -Recurse -File | Select-Object FullName
Select-String -Path delta-Mem/deltamem/eval/**/*.py -Pattern "locomo|LoCoMo" -SimpleMatch | Select-Object Path, LineNumber, Line | Select-Object -First 20
```

Locate the module/function or CLI script that runs the LoCoMo eval. There are two plausible shapes:

- A Python entry point like `deltamem.eval.locomo.run(...)` or `python -m deltamem.eval.locomo`.
- A shell script in `delta-Mem/scripts/` that calls one.

Pick the *Python* entry point if available — easier to drive from Windows without translating shell. If only a shell script exists, port the essential `python -m ...` invocation it issues into our wrapper.

Record the chosen entry point and its required arguments inline in the wrapper's docstring (Step 2). Do **not** invent arguments — copy them from the vendored repo.

- [ ] **Step 2: Write `run/locomo_eval.py`**

Create `run/locomo_eval.py`. The exact entry point depends on what you found in Step 1; below is the scaffolding around it. Replace the `_invoke_vendored_eval` body with the verbatim invocation from the vendored repo.

```python
"""Run the vendored delta-Mem LoCoMo eval verbatim, then emit the reproduction
report via run.report_gen.

This wrapper deliberately does NOT reimplement scoring — it invokes the vendored
eval module/script as-is and reads the scores it produces. Methodology fidelity
is the whole point of Tier 1 (spec risk R3).

Outputs:
    - report/raw/locomo-stdout.log   (full stdout/stderr of the eval)
    - report/raw/locomo-scores.json  (the eval's own scores file, copied here)
    - report/reproduction-report.md  (via run.report_gen)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from run.report_gen import render_report

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "report" / "raw"
REPORT_PATH = REPO_ROOT / "report" / "reproduction-report.md"
COMMIT_FILE = REPO_ROOT / "report" / "vendored-commit.txt"

# --- Configuration recorded into the report. Update Step 1's findings here. ---
EVAL_CONFIG = {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "adapter": "declare-lab/delta-mem_qwen3_4b-instruct",
    "dtype": "bfloat16",
    "attn_implementation": "sdpa",
    "max_seq_len": 8192,        # delta-mem write length matches the released adapter
    "kv_cache_dtype": "bfloat16",
}
PAPER_RATIO = 1.20
TOLERANCE = 0.05


def _invoke_vendored_eval(*, output_dir: Path) -> Path:
    """Invoke the vendored LoCoMo eval. Returns the path to the scores JSON.

    REPLACE the command below with the exact invocation discovered in
    Task 6 Step 1 (e.g. `python -m deltamem.eval.locomo ...`). Do not invent
    flags — copy them from the vendored repo.
    """
    cmd = [
        sys.executable, "-m", "deltamem.eval.locomo",  # << verify in Step 1
        "--model", EVAL_CONFIG["model"],
        "--adapter", EVAL_CONFIG["adapter"],
        "--dtype", EVAL_CONFIG["dtype"],
        "--attn-impl", EVAL_CONFIG["attn_implementation"],
        "--max-seq-len", str(EVAL_CONFIG["max_seq_len"]),
        "--output-dir", str(output_dir),
    ]

    log_path = RAW_DIR / "locomo-stdout.log"
    print(f"Running: {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        sys.stderr.write(f"Vendored eval FAILED (rc={proc.returncode}); see {log_path}\n")
        sys.exit(proc.returncode)

    scores = output_dir / "scores.json"   # << verify the vendored eval's output filename
    if not scores.exists():
        sys.stderr.write(f"Expected scores at {scores} but none found; see {log_path}\n")
        sys.exit(2)

    target = RAW_DIR / "locomo-scores.json"
    shutil.copyfile(scores, target)
    return target


def _extract_ratios(scores_path: Path) -> tuple[float, float, list[tuple[int, str]]]:
    """Pull (our_delta_mem_score, our_frozen_score, skipped_samples) from the
    vendored eval's scores file.

    The vendored eval's JSON shape is treated as authoritative; if its key names
    differ from the placeholders below (`delta_mem`, `frozen`, `skipped`), edit
    this function to match — do not transform the scores themselves.
    """
    data = json.loads(scores_path.read_text(encoding="utf-8"))
    dm = float(data["delta_mem"])
    fz = float(data["frozen"])
    skipped_raw = data.get("skipped", [])
    skipped = [(int(s["index"]), str(s["reason"])) for s in skipped_raw]
    return dm, fz, skipped


def _read_vendored_commit() -> str:
    if not COMMIT_FILE.exists():
        return "<unknown — report/vendored-commit.txt missing>"
    return COMMIT_FILE.read_text(encoding="utf-8").strip()


def _peak_vram_gb_from_log(log_path: Path) -> Optional[float]:
    """Best-effort extraction of peak VRAM from the eval's stdout.

    The vendored eval may or may not print this. If absent, returns None and
    the report records 'unknown'; we will re-run with explicit VRAM capture if
    the verdict is OUT_OF_BAND.
    """
    if not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        l = line.lower()
        if "peak" in l and ("vram" in l or "gpu" in l or "memory" in l):
            for tok in line.split():
                try:
                    return float(tok.rstrip("GgBb"))
                except ValueError:
                    continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(RAW_DIR / "locomo-out"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    scores_path = _invoke_vendored_eval(output_dir=output_dir)
    dm, fz, skipped = _extract_ratios(scores_path)

    ratio = dm / fz if fz > 0 else float("nan")
    peak_vram = _peak_vram_gb_from_log(RAW_DIR / "locomo-stdout.log")

    md = render_report(
        our_ratio=ratio,
        paper_ratio=PAPER_RATIO,
        tolerance=TOLERANCE,
        our_delta_mem_score=dm,
        our_frozen_score=fz,
        peak_vram_gb=peak_vram if peak_vram is not None else float("nan"),
        skipped_samples=skipped,
        vendored_commit=_read_vendored_commit(),
        eval_config=EVAL_CONFIG,
    )
    REPORT_PATH.write_text(md, encoding="utf-8")

    print(f"Wrote {REPORT_PATH}")
    print(f"Ratio: {ratio:.3f} (paper {PAPER_RATIO:.2f}, tolerance ±{TOLERANCE:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Dry-run the wrapper logic (without running the full eval) to surface argparse / import errors**

```powershell
uv run python -c "from run.locomo_eval import _read_vendored_commit, EVAL_CONFIG; print(_read_vendored_commit()); print(EVAL_CONFIG)"
```

Expected: prints the pinned commit SHA and the eval config dict.

- [ ] **Step 4: Commit**

```powershell
git add run/locomo_eval.py
git commit -m "run: LoCoMo eval wrapper invoking vendored eval verbatim"
```

---

## Task 7: Run the LoCoMo reproduction and commit the report

This is the actual reproduction run. Expect it to take hours; queue it with status logged to `report/raw/locomo-stdout.log`.

- [ ] **Step 1: Pre-flight — confirm Tasks 3, 4, 6 all passed**

```powershell
Test-Path report/kernels-gate.md      # must exist with Result: PASS
Test-Path report/smoke.md             # must exist with both gates PASS
Test-Path report/vendored-commit.txt  # must exist with SHA
uv run pytest tests/ -v               # all green
```

If any of these checks fail, return to the matching task — do not run the eval against a broken foundation.

- [ ] **Step 2: Run the LoCoMo eval**

```powershell
uv run python -m run.locomo_eval 2>&1 | Tee-Object report/raw/locomo-driver.log
```

Expected (at completion): wrapper prints `Wrote .../reproduction-report.md` and a single ratio line. `report/reproduction-report.md` is written.

If the run OOMs mid-eval, this is risk R2 materialising:
- Check `report/raw/locomo-stdout.log` for which sample failed.
- Either lower `EVAL_CONFIG["max_seq_len"]` in `run/locomo_eval.py` (with a clear note in the report) and re-run, **or** record the affected samples as skipped — never silently drop them. The vendored eval should already surface skipped samples; if not, capture them in the wrapper before continuing.

- [ ] **Step 3: Inspect the report**

```powershell
Get-Content report/reproduction-report.md
```

Confirm the verdict line is one of: PASS / OUT_OF_BAND / REGRESSION. If OUT_OF_BAND or REGRESSION, the report itself states the next investigation step — read it, don't paper over it.

- [ ] **Step 4: Commit the report**

```powershell
git add report/reproduction-report.md report/raw/locomo-scores.json report/raw/locomo-stdout.log report/raw/locomo-driver.log
git commit -m "report: LoCoMo reproduction result on RTX 3060 / native Windows"
```

The repo now contains a committed, reproducible answer to "did delta-Mem's LoCoMo claim hold up?" That answer — whatever it is — is Tier 1's deliverable.

---

## Task 8 (Stretch): MemoryAgentBench as a second data point

Optional. Only run after Task 7 has produced a committed Tier 1 report. The structure mirrors Task 6 + 7 against MemoryAgentBench (paper: 1.31×).

**Files:**
- Modify: `run/locomo_eval.py` — extract the bits that are bench-agnostic into helpers, or
- Create: `run/mab_eval.py` — sibling of `locomo_eval.py` for MemoryAgentBench.

- [ ] **Step 1: Locate the MemoryAgentBench entry point**

```powershell
Get-ChildItem delta-Mem/deltamem/eval -Recurse -File | Select-String -Pattern "memoryagent|MemoryAgent|mab" -SimpleMatch | Select-Object Path, LineNumber, Line
```

- [ ] **Step 2: Decide: extend `run/locomo_eval.py` with a `--bench` flag, or sibling `run/mab_eval.py`**

If extending: add a `BENCHES` dict mapping `"locomo"` / `"mab"` to `{cmd_module, paper_ratio, scores_filename}`, and drive both from one wrapper. DRY.

If siblinging: copy the bench-agnostic helpers into a small `run/_eval_common.py` and import from both `locomo_eval.py` and `mab_eval.py`. Equally DRY, slightly more files.

Either is fine — pick whichever needs less code given the differences you find in Step 1.

- [ ] **Step 3: Run MAB and emit `report/mab-report.md`**

```powershell
uv run python -m run.mab_eval 2>&1 | Tee-Object report/raw/mab-driver.log
```

- [ ] **Step 4: Commit**

```powershell
git add run/ report/mab-report.md report/raw/mab-*.log report/raw/mab-scores.json
git commit -m "report: MemoryAgentBench reproduction (stretch)"
```

---

## Self-Review Summary

**Spec coverage:**
- Vision/roadmap (spec §2) → captured in plan header + spec link; T2/T3 deferred per the spec's explicit decision.
- Tier 1 components (spec §3.3) → Tasks 1 (scaffold + vendor), 2 (env), 4 (run/smoke_chat), 5–6 (run/report_gen + run/locomo_eval), 7 (report).
- R1 kernel-build gate (spec §3.5) → Task 3, with explicit pass/fail decision and WSL2 fallback.
- R2 VRAM / long-context (spec §3.5) → Task 7 Step 2 OOM handling + Task 5 report skipped-samples support.
- R3 methodology fidelity (spec §3.5) → Task 6 explicitly forbids reimplementing scoring; pinned commit recorded by Task 1.
- Success criteria — chat demo memory observation (Task 4), LoCoMo within ±0.05 (Tasks 5–7), report committed (Task 7), MAB stretch (Task 8).

**Placeholder scan:** No `TBD`, `TODO`, or "implement later" tokens. Two steps (Task 6 Steps 1–2 and Task 8 Step 1) genuinely require inspecting the vendored repo before writing the final command — those are framed as explicit verification steps with concrete inspection commands, not placeholders.

**Type consistency:** `verdict()` returns `Literal["PASS", "OUT_OF_BAND", "REGRESSION"]` consistently. `render_report` keyword arguments match the test call. `EVAL_CONFIG` keys (`model`, `adapter`, `dtype`, `attn_implementation`, `max_seq_len`) align between the wrapper and the test fixture. `_extract_ratios` returns the same shape consumed by `render_report`'s `skipped_samples` parameter.
