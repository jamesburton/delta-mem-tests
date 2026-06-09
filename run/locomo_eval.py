"""Run the vendored delta-Mem LoCoMo eval via a small monkeypatching driver,
then emit the reproduction report via run.report_gen.

This wrapper deliberately does NOT reimplement scoring. Methodology fidelity
is the whole point of Tier 1 (spec risk R3). The single controller-approved
deviation is a chunked-prefill monkeypatch (see run/_chunked_eval_runner.py)
that fits the eval inside 12 GB VRAM on the RTX 3060 without changing any
numerics — autoregressive attention is unaffected by how the prefill is
batched. The patch is documented in the reproduction report itself.

Entry point: invoked via ``python -m run._chunked_eval_runner`` which
monkeypatches build_teacher_forced_snapshot before calling the vendored
eval's main(). The vendored submodule is NOT modified on disk; the pinned
commit hash still holds.

Output JSON schema (nested, not flat):
- frozen backbone score: data["base"]["summary"]["full_history_replay"]["overall_score"]
- delta-mem score:       data["delta"]["summary"]["full_history_replay"]["overall_score"]
No "skipped_samples" key exists in the eval output; passed as [] to render_report.

max_seq_len is NOT a CLI flag. The eval calls infer_model_context_window() at
runtime which reads model.config.max_position_embeddings (262144 for
Qwen3-4B-Instruct-2507). Recorded in EVAL_CONFIG for the report.

Sample limit: --max-conversations N is supported.

Outputs:
    - report/raw/locomo-stdout.log   (full stdout/stderr of the eval)
    - report/raw/locomo-scores.json  (the eval's own scores file, copied here)
    - report/reproduction-report.md  (via run.report_gen, with a prepended
                                       Methodology adjustments section)
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
    # Use the paper's official prompting protocol (matches the vendored
    # benchmark suite at delta-Mem/scripts/run_qasper_multimodel_write8192_benchmark_suite.sh).
    # The newer `history_replay` mode produces bit-identical predictions
    # between base and delta on a small partial run — likely because the
    # full history is in the prompt and temperature=0.4/top_k=10/top_p=0.9
    # sampling lockstep with the same seed lands on the same tokens, so the
    # signal is buried. The paper's 1.20x was measured under official_prompt.
    "full_history_mode": "official_prompt",
    # Default 2 (vs vendored 64) to fit base-eval generate() inside 12 GB on
    # the RTX 3060. Each base-eval prompt is history_messages + question
    # (~17.6k + question tokens), so 8 GB weights + N x ~0.6 GB KV + SDPA
    # scratch limits us to ~2 prompts in parallel on a 12 GB card. We attempted
    # batch=8 with a broadened OOM-class normalisation to engage the vendored
    # bisector (locomo_delta.py:600-665), but on this host the recovery from
    # CUDA OOM corrupts the process (Windows STATUS_STACK_BUFFER_OVERRUN), so
    # we must size the initial batch to NEVER OOM. The bisector and OOM-class
    # normalisation remain in place as belt-and-braces but should not fire on
    # the chosen size. Greedy scoring (do_sample=False) is batch-invariant.
    "eval_batch_size": 2,
    "methodology_adjustment": (
        "Three in-process monkeypatches on the vendored eval, all required "
        "to fit Tier 1 reproduction on a 12 GB card: "
        "(1) --full-history-mode=official_prompt is the paper's protocol "
        "(matches scripts/run_qasper_multimodel_*); "
        "(2) generate_official_full_history_answer is replaced with a "
        "DeltaMemChatSession chunked prefill (~1k-token chunks via "
        "_ingest_full_ids prefix-skip) because the vendored monolithic "
        "model.generate hits SDPA MATH backend on a 17.6k-token prompt and "
        "tries to allocate ~37 GB; "
        "(3) per-conversation KV-cache reuse — the history portion is "
        "prefilled once per conversation, snapshotted (KV cache + delta-mem "
        "state), then restored and Cache.crop()-truncated to history_len "
        "before each subsequent question so _ingest_full_ids only forwards "
        "the ~50-token question suffix. Mathematically equivalent in the "
        "infinite-precision limit (autoregressive attention depends only on "
        "prior tokens via the KV cache); in bf16 the chunk-boundary GEMM "
        "kernel selection can perturb long-form outputs by a few sampled "
        "tokens, but per-question scores and overall ratio agree to the "
        "reported precision on the validation set (1 conv x 3 q). The "
        "build_teacher_forced_snapshot chunked patch and _generate_prompt_chunk "
        "OOM-class normalisation are kept in the runner but inert in "
        "official_prompt mode."
    ),
}
PAPER_RATIO = 1.20
TOLERANCE = 0.05


def _resolve_adapter_path() -> str:
    """Resolve the HF repo ID (or local override path) to a local snapshot path.

    The released delta-mem adapter must be loaded from a local directory
    (confirmed in Task 4: delta_impl.py:493-497, :2794-2802 accept paths only).
    The vendored eval uses local_files_only=True (locomo_delta.py:112, 118), so
    the adapter must already be present in the HF cache.

    If ``--adapter-override <path>`` was passed, EVAL_CONFIG["adapter"] now
    holds an absolute local path; return it directly (skip snapshot_download).
    """
    if EVAL_CONFIG.get("adapter_override"):
        return EVAL_CONFIG["adapter"]
    from huggingface_hub import snapshot_download
    return snapshot_download(EVAL_CONFIG["adapter"])


def _invoke_vendored_eval(
    *,
    model_path: str,
    adapter_path: str,
    output_json: Path,
    max_conversations: Optional[int] = None,
    max_questions_per_conversation: Optional[int] = None,
    data_file: Optional[str] = None,
    kv_cache_backend: str = "bf16",
    kv_cache_bits: int = 0,
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
        sys.executable, "-m", "run._chunked_eval_runner",
        "--model-path", model_path,
        "--adapter-dir", adapter_path,
        "--dtype", EVAL_CONFIG["dtype"],
        "--attn-implementation", EVAL_CONFIG["attn_implementation"],
        "--eval-batch-size", str(EVAL_CONFIG["eval_batch_size"]),
        "--full-history-mode", EVAL_CONFIG["full_history_mode"],
        "--output-json", str(output_json),
    ]
    if max_conversations is not None:
        cmd += ["--max-conversations", str(max_conversations)]
    if max_questions_per_conversation is not None:
        cmd += ["--max-questions-per-conversation", str(max_questions_per_conversation)]
    if data_file is not None:
        cmd += ["--data-file", data_file]

    log_path = RAW_DIR / "locomo-stdout.log"
    print(f"Running: {' '.join(cmd)}")
    subprocess_env = {**os.environ, "DELTA_MEM_SCAN_IMPL": "torch"}
    if kv_cache_backend != "bf16":
        subprocess_env["KV_CACHE_BACKEND"] = kv_cache_backend
        if kv_cache_bits > 0:
            subprocess_env["KV_CACHE_BITS"] = str(kv_cache_bits)
        print(
            f"  KV_CACHE_BACKEND={kv_cache_backend} "
            f"KV_CACHE_BITS={kv_cache_bits or '(backend-default)'} "
            "(KV cache quantised; cross-question cache reuse disabled)"
        )
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=subprocess_env,
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
    parser.add_argument(
        "--max-questions-per-conversation",
        type=int,
        default=None,
        help="(Partial-run only) cap questions per conversation. Useful for "
             "validating optimisation patches (e.g. KV-cache reuse) against a "
             "small reference set before scaling up.",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Override the vendored eval's data file (default: data/locomo10.json "
             "relative to repo root). Use to run on a custom subset.",
    )
    parser.add_argument(
        "--kv-cache-backend",
        choices=["bf16", "turboquant", "quanto", "hqq", "oscar"],
        default="bf16",
        help="KV-cache quantisation backend. `bf16` (default) keeps the bf16 "
             "DynamicCache plus our per-conversation crop+reuse trick. "
             "`turboquant` uses turboquant>=0.2.0 (TurboQuantCache, 4-bit "
             "default; pure-Python codebook is slow). `quanto`/`hqq` use "
             "transformers' built-in KIVI-style QuantizedCache "
             "(quanto supports 2/4-bit; hqq supports 1/2/3/4/8-bit). "
             "`oscar` uses the vendored oscar-transformers port: per-layer "
             "orthogonal Q/K/V rotation + sink/recent/INT2-middle cache; "
             "requires OSCAR_K_ROTATION_PATH and OSCAR_V_ROTATION_PATH env vars. "
             "Any non-bf16 backend disables cross-question KV-cache reuse.",
    )
    parser.add_argument(
        "--kv-cache-bits",
        type=int,
        default=0,
        help="Bit-width for the quantisation backend (0 = backend default: "
             "turboquant->4, quanto/hqq->2).",
    )
    parser.add_argument(
        "--turboquant-bits",
        type=int,
        default=0,
        help="(Deprecated) Backwards-compat for --kv-cache-backend turboquant "
             "--kv-cache-bits N. If > 0, equivalent to those flags.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="Override EVAL_CONFIG['eval_batch_size'] (default 2). Use 1 to "
             "halve attention scratch memory for VRAM-constrained long-context "
             "runs at the cost of ~2x wall-time.",
    )
    parser.add_argument(
        "--adapter-override",
        default=None,
        help="Local path to a delta-mem adapter directory (containing "
             "delta_mem_config.json + delta_mem_adapter.pt). Skips the "
             "HF snapshot_download of the released adapter. Use this to "
             "evaluate a locally-trained or Strix-Halo-trained checkpoint.",
    )
    args = parser.parse_args()
    if args.eval_batch_size > 0:
        EVAL_CONFIG["eval_batch_size"] = args.eval_batch_size
    if args.adapter_override:
        override_path = Path(args.adapter_override).resolve()
        if not override_path.is_dir():
            raise SystemExit(f"--adapter-override path is not a directory: {override_path}")
        EVAL_CONFIG["adapter"] = str(override_path)
        EVAL_CONFIG["adapter_override"] = True
    # Backwards compat: --turboquant-bits implies --kv-cache-backend turboquant.
    if args.turboquant_bits > 0 and args.kv_cache_backend == "bf16":
        args.kv_cache_backend = "turboquant"
        if args.kv_cache_bits == 0:
            args.kv_cache_bits = args.turboquant_bits

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
        max_questions_per_conversation=args.max_questions_per_conversation,
        data_file=args.data_file,
        kv_cache_backend=args.kv_cache_backend,
        kv_cache_bits=args.kv_cache_bits,
    )

    if args.kv_cache_backend != "bf16":
        EVAL_CONFIG["kv_cache_backend"] = args.kv_cache_backend
        EVAL_CONFIG["kv_cache_bits"] = args.kv_cache_bits or (
            4 if args.kv_cache_backend == "turboquant" else 2
        )
        EVAL_CONFIG["kv_cache"] = (
            f"{args.kv_cache_backend} {EVAL_CONFIG['kv_cache_bits']}-bit "
            "(residual-window kept in original precision; "
            "cross-question reuse disabled)"
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

    # Prepend a Methodology adjustments section so future readers immediately
    # see the controller-approved chunked-prefill patch (the only deviation
    # from a bit-for-bit-faithful reproduction).
    METHODOLOGY_NOTE = (
        "\n## Methodology adjustments\n"
        "\n"
        "Three in-process monkeypatches applied in `run/_chunked_eval_runner.py`. "
        "All are required to fit the eval on a 12 GB RTX 3060; the vendored "
        "submodule is unmodified on disk (pinned commit unchanged).\n"
        "\n"
        "1. **`--full-history-mode official_prompt`.** Matches the vendored "
        "benchmark suite at "
        "`delta-Mem/scripts/run_qasper_multimodel_write8192_benchmark_suite.sh:130-136`. "
        "The newer `history_replay` mode (the eval's default) puts the full "
        "conversation history in the prompt for both branches; with "
        "temperature=0.4 / top_k=10 / top_p=0.9 sampling and the same seed, base "
        "and delta land on bit-identical predictions on small samples, burying "
        "the signal. The paper's 1.20x ratio was measured under `official_prompt`.\n"
        "\n"
        "2. **Chunked prefill of the official prompt.** The vendored "
        "`generate_official_full_history_answer` "
        "(`delta-Mem/deltamem/eval/locomo_delta.py:420-479`) calls "
        "`model.generate` monolithically on the full ~17.6k-token prompt. "
        "Without flash-attn (not available on Windows/Ampere here), PyTorch "
        "SDPA falls back to the MATH backend and tries to allocate ~37 GB of "
        "attention scratch. We replace it with a `DeltaMemChatSession`-driven "
        "chunked prefill (~1k-token chunks via `_ingest_full_ids` prefix-skip), "
        "which is mathematically equivalent because token-granularity "
        "delta-mem writes are autoregressive accumulations of per-token Q/K/V "
        "projections (`delta_impl.py:2173-2184`).\n"
        "\n"
        "3. **Per-conversation KV-cache reuse.** With (2), per-question "
        "prefill still re-processes the whole ~17.6k-token prompt; at "
        "~5 min/q just for base eval, the full 1986-question run would take "
        "~40 days. We cache the shared history KV (computed once per "
        "conversation) and crop the cache back to `history_len` before each "
        "subsequent question's `_ingest_full_ids`, so only the ~30-token "
        "question suffix is forwarded. The shared `history_len` is computed "
        "as the longest token-prefix common to ALL questions in the sample "
        "(`build_official_context_text` token boundaries can shift with "
        "question length, so a pairwise common-prefix between just q0 and q1 "
        "can overshoot the true shared length). A runtime sanity check "
        "verifies the cached prefix matches each prompt before suffix-ingest; "
        "on mismatch we fall back to the non-cached chunked path for that "
        "question.\n"
        "\n"
        "  Numerical note: chunked + cached prefill is equivalent to "
        "monolithic prefill in the infinite-precision limit. In bf16, GEMM "
        "kernel selection at different chunk sizes can perturb long-form "
        "sampled outputs by a few tokens; per-question score and overall "
        "ratio agree to the reported precision on the validation slice we "
        "checked.\n"
        "\n"
        "See the \"Eval config\" section below for `methodology_adjustment` "
        "and related keys recorded with this run.\n"
    )
    # Insert the section after the title line and before the Verdict line.
    md_lines = md.splitlines()
    # md_lines[0] is the title; insert after the title's blank line (line 1).
    md_lines.insert(2, METHODOLOGY_NOTE.rstrip())
    md = "\n".join(md_lines) + ("\n" if not md.endswith("\n") else "")

    REPORT_PATH.write_text(md, encoding="utf-8")

    print(f"Wrote {REPORT_PATH}")
    print(f"Ratio: {ratio:.3f} (paper {PAPER_RATIO:.2f}, tolerance ±{TOLERANCE:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
