"""Run the vendored delta-Mem LoCoMo eval verbatim, then emit the reproduction
report via run.report_gen.

This wrapper deliberately does NOT reimplement scoring — it invokes the vendored
eval module/script as-is and reads the scores it produces. Methodology fidelity
is the whole point of Tier 1 (spec risk R3).

Entry point discovered in Subtask A: ``deltamem.eval.locomo_delta`` has an
``if __name__ == "__main__": main()`` block (locomo_delta.py:1074-1075).
Invoked as ``python -m deltamem.eval.locomo_delta``.

Output JSON schema (nested, not flat):
- frozen backbone score: data["base"]["summary"]["full_history_replay"]["overall_score"]
- delta-mem score:       data["delta"]["summary"]["full_history_replay"]["overall_score"]
No "skipped_samples" key exists in the eval output — the eval either completes
or errors; skipped_samples is passed as [] to render_report.

max_seq_len is NOT a CLI flag. The eval calls infer_model_context_window() at
runtime which reads model.config.max_position_embeddings (262144 for
Qwen3-4B-Instruct-2507). Recorded in EVAL_CONFIG for the report.

Sample limit: --max-conversations N is supported (locomo_delta.py:216-217).
Subtask C verification: wiring test (tests/test_locomo_eval_wiring.py) — a
small-slice dry-run requires GPU + model load so we verify _extract_ratios and
_read_vendored_commit with a synthetic fake-scores JSON instead.

Outputs:
    - report/raw/locomo-stdout.log   (full stdout/stderr of the eval)
    - report/raw/locomo-scores.json  (the eval's own scores file, copied here)
    - report/reproduction-report.md  (via run.report_gen)
"""

from __future__ import annotations

import os
# Path-lock per report/kernels-gate.md: lock the torch scan path before any
# deltamem import the eval (or this wrapper) may trigger. The subprocess also
# inherits this env var, ensuring the vendored eval runs the torch path too.
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import argparse
import json
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

# Qwen3-4B-Instruct-2507 max_position_embeddings = 262144 (from config.json).
# Not a CLI flag — the eval reads it via infer_model_context_window() at
# runtime (locomo_protocol.py:325-334). We record the model's actual value here
# for the reproduction report; it matches what the eval will observe.
_QWEN3_4B_MAX_POSITION_EMBEDDINGS = 262144

# Configuration recorded into the report. All values correspond to real eval
# behaviour: attn_implementation is explicitly overridden to sdpa (the eval's
# default resolve_attn_implementation(path, None) returns "flash_attention_2",
# which is unavailable on this host); scan_impl is locked to torch per the R1
# gate decision in report/kernels-gate.md.
EVAL_CONFIG = {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "adapter": "declare-lab/delta-mem_qwen3_4b-instruct",
    "dtype": "bfloat16",
    "attn_implementation": "sdpa",
    "max_seq_len": _QWEN3_4B_MAX_POSITION_EMBEDDINGS,
    "scan_impl": "torch",  # see report/kernels-gate.md
}
PAPER_RATIO = 1.20
TOLERANCE = 0.05


def _resolve_adapter_path() -> str:
    """Resolve the HF repo ID to a local snapshot path.

    The released delta-mem adapter must be loaded from a local directory
    (confirmed in Task 4: delta_impl.py:493-497, :2794-2802 accept paths only).
    The vendored eval uses local_files_only=True (locomo_delta.py:112, 118), so
    the adapter must already be present in the HF cache.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(EVAL_CONFIG["adapter"])


def _invoke_vendored_eval(
    *,
    model_path: str,
    adapter_path: str,
    output_json: Path,
    max_conversations: Optional[int] = None,
) -> Path:
    """Invoke the vendored LoCoMo eval. Returns the path to the scores JSON.

    The command is the literal Python invocation discovered in Subtask A —
    deltamem.eval.locomo_delta has a __main__ block (locomo_delta.py:1074-1075).

    Flags used (all from parse_args() in locomo_delta.py:179-267):
        --model-path         path to the base model (local snapshot)
        --adapter-dir        path to the delta-mem adapter (local snapshot)
        --dtype              bfloat16
        --attn-implementation sdpa (explicitly overriding the default which
                             resolves to flash_attention_2 via
                             resolve_attn_implementation(path, None))
        --output-json        destination for the scores JSON
        --max-conversations  (optional) cap the number of conversations for
                             partial runs / dry-run verification
    """
    cmd = [
        sys.executable, "-m", "deltamem.eval.locomo_delta",
        "--model-path", model_path,
        "--adapter-dir", adapter_path,
        "--dtype", EVAL_CONFIG["dtype"],
        "--attn-implementation", EVAL_CONFIG["attn_implementation"],
        "--output-json", str(output_json),
    ]
    if max_conversations is not None:
        cmd += ["--max-conversations", str(max_conversations)]

    log_path = RAW_DIR / "locomo-stdout.log"
    print(f"Running: {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "DELTA_MEM_SCAN_IMPL": "torch"},
        )
    if proc.returncode != 0:
        sys.stderr.write(
            f"Vendored eval FAILED (rc={proc.returncode}); see {log_path}\n"
        )
        sys.exit(proc.returncode)

    if not output_json.exists():
        sys.stderr.write(
            f"Expected scores at {output_json} but none found; see {log_path}\n"
        )
        sys.exit(2)

    target = RAW_DIR / "locomo-scores.json"
    shutil.copyfile(output_json, target)
    return target


def _extract_ratios(scores_path: Path) -> tuple[float, float, list[tuple[int, str]]]:
    """Pull (delta_mem_score, frozen_score, skipped_samples) from the scores JSON.

    Output schema (discovered in Subtask A, locomo_delta.py:977-1057):
        data["base"]["summary"]["full_history_replay"]["overall_score"]  → frozen
        data["delta"]["summary"]["full_history_replay"]["overall_score"] → delta-mem

    The eval has no "skipped" concept — it either completes or raises. We return
    an empty skipped list; any partial-run exceptions are captured in the log.
    """
    data = json.loads(scores_path.read_text(encoding="utf-8"))
    fz = float(data["base"]["summary"]["full_history_replay"]["overall_score"])
    dm = float(data["delta"]["summary"]["full_history_replay"]["overall_score"])
    return dm, fz, []


def _read_vendored_commit() -> str:
    if not COMMIT_FILE.exists():
        return "<unknown — report/vendored-commit.txt missing>"
    return COMMIT_FILE.read_text(encoding="utf-8").strip()


def _peak_vram_gb_from_log(log_path: Path) -> Optional[float]:
    """Best-effort extraction of peak VRAM from the eval's stdout."""
    if not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        low = line.lower()
        if "peak" in low and ("vram" in low or "gpu" in low or "memory" in low):
            for tok in line.split():
                try:
                    return float(tok.rstrip("GgBb"))
                except ValueError:
                    continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the vendored LoCoMo eval and emit the reproduction report."
    )
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "outputs" / "qwen3_delta_mem_locomo_eval.json"),
        help="Destination for the vendored eval's scores JSON "
             "(default matches the eval's own default output path).",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="(Partial-run only) cap eval to N conversations via --max-conversations; "
             "omit for the real Task 7 full run.",
    )
    args = parser.parse_args()

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Resolve both model and adapter to local paths before launching the
    # vendored eval, which uses local_files_only=True throughout.
    from huggingface_hub import snapshot_download

    print(f"Resolving model snapshot: {EVAL_CONFIG['model']}")
    model_path = snapshot_download(EVAL_CONFIG["model"])
    print(f"  -> {model_path}")

    print(f"Resolving adapter snapshot: {EVAL_CONFIG['adapter']}")
    adapter_path = _resolve_adapter_path()
    print(f"  -> {adapter_path}")

    scores_path = _invoke_vendored_eval(
        model_path=model_path,
        adapter_path=adapter_path,
        output_json=output_json,
        max_conversations=args.max_conversations,
    )
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
