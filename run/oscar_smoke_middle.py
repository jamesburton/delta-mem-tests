"""Middle-region recall diagnostic for OSCAR-INT2.

The long smoke (run/oscar_smoke_long.py) put the target fact in the LAST
sentence of a ~1.5k-token passage. That fact was in the bf16 recent window
(256 tokens), so OSCAR's INT2 middle never had to carry it. The full LoCoMo
eval then collapsed to ~0.04 quality because LoCoMo's 17.5k prompts have
facts scattered throughout, most of them deep in what becomes the INT2
middle.

This script tests middle-region recall directly:
  - Build a ~17k-token context similar in length to LoCoMo conv-26.
  - Insert a uniquely-identifiable "needle" fact at a target offset
    (default ~5k tokens from start, firmly in INT2 middle for default
    sink=64/recent=256).
  - Ask about the needle.
  - Run under multiple (sink, recent) configurations + a baseline.

Configurations tested (each takes ~2 min for 17k prefill + 60 generated
tokens):

    name              sink   recent   middle in INT2
    ----------------- ----   ------   --------------
    baseline (bf16)   N/A    N/A      0  (no quantization)
    oscar default      64       256   ~17.4k tokens
    oscar moderate    512      2048   ~14.9k tokens
    oscar generous   1024      4096   ~12.4k tokens

If quality recovers monotonically as the bf16 budget grows, the issue is
INT2 quantization eating semantic content from the middle and the right fix
is to ship at a larger bf16 budget (sacrificing some memory saving). If
quality stays flat, the rotations themselves don't transfer well from
Thinking to Instruct and we need the 3-phase calibration.

Run:
    .venv/Scripts/python.exe run/oscar_smoke_middle.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oscar_transformers import OSCARCache, apply_rotations, load_rotation_file


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
# Default to the original RotationZoo Thinking-2507 path; override with
# OSCAR_K_ROTATION_PATH / OSCAR_V_ROTATION_PATH to point at locally calibrated
# rotations (e.g. data/oscar/rotations/instruct_gpqa or instruct_locomo).
_DEFAULT_ROT_DIR = Path("data/oscar/rotations/_hf_cache/Qwen3-4B-Thinking-2507/seq20000_prompt83_group128")
K_PATH = Path(os.environ.get(
    "OSCAR_K_ROTATION_PATH",
    str(_DEFAULT_ROT_DIR / "k_rotation_qqt_r_h_pbr.pt"),
))
V_PATH = Path(os.environ.get(
    "OSCAR_V_ROTATION_PATH",
    str(_DEFAULT_ROT_DIR / "v_rotation_sst_r_h_pbr.pt"),
))

MAX_NEW = 80
# 4k prompt fits monolithic SDPA on a 12 GB card; the LoCoMo eval uses chunked
# prefill to avoid the ~35 GB SDPA-MATH allocation at 17 k, but plumbing that
# into a standalone diagnostic adds a lot of code for little extra signal.
# At 4 k with the needle near position 1500, the OSCAR(64/256) and
# OSCAR(512/2048) configs both place the needle in INT2 middle; OSCAR(1024/4096)
# spills the remainder into the bf16 recent window (control case where no
# INT2 is exercised on this prompt). This is enough to discriminate
# "INT2-quantization eats the fact" from "rotations are too lossy".
TARGET_CONTEXT_TOKENS = 4000
NEEDLE_OFFSET_TOKENS = 1500

# Distinctive, easily-checkable needle. The fact is the bracketed sentence:
# the model needs to recall the access code "QM-7194-ZULU" when asked for it.
_NEEDLE_SENTENCE = (
    "IMPORTANT BUILDING ACCESS NOTICE: The temporary security access code "
    "for the east-wing server room during the May 2026 maintenance window "
    "is QM-7194-ZULU. This code is valid only for personnel listed on the "
    "approved technician roster."
)

# A neutral filler passage. We repeat it (with mild numbering variation) to
# bracket the needle to the target offset.
_FILLER = """\
The annual operations report covers facility-wide infrastructure changes
implemented during the fiscal quarter. Maintenance teams completed scheduled
inspections of the HVAC systems, the backup generator array, and the
distributed fire-suppression network. Routine calibration of the climate
sensors in the data-floor zones proceeded on schedule. The compliance
officer confirmed that all life-safety equipment passed its quarterly
audit. Vendor coordination for the next maintenance cycle has been
initiated. The facilities team noted that the chilled-water supply lines
showed nominal pressure throughout the reporting period. Janitorial
contracts were renewed under the same terms as the prior quarter. The
security office reported no unauthorized access attempts at any controlled
ingress point during the window. Office furniture audits identified two
ergonomic chairs scheduled for replacement under the standard refresh
cycle. The mail-room renovation, originally projected to complete in the
prior quarter, finished on time and within budget. Two conference rooms
on the third floor received updated AV equipment, including replacement
projectors and updated room-control panels. Parking permit reconciliation
proceeded without exception. The vending services contract was extended
for an additional six months pending negotiation of the renewal terms."""


def build_context(tokenizer, target_tokens: int, needle_offset_tokens: int) -> tuple[str, int]:
    """Build a context with the needle inserted near ``needle_offset_tokens``.

    Returns (context_string, actual_total_tokens).
    """
    pre_target = needle_offset_tokens

    # Build pre-needle filler.
    pre_parts: list[str] = []
    pre_tokens = 0
    section_idx = 0
    while pre_tokens < pre_target:
        section_idx += 1
        marker = f"\n\n=== Operations Report Section {section_idx} ===\n\n"
        pre_parts.append(marker + _FILLER)
        joined = "".join(pre_parts)
        pre_tokens = len(tokenizer(joined, add_special_tokens=False).input_ids)
    pre_text = "".join(pre_parts)

    # Build post-needle filler to reach the total target.
    needle_block = f"\n\n=== SPECIAL NOTICE ===\n\n{_NEEDLE_SENTENCE}\n\n"
    post_parts: list[str] = []
    so_far = pre_text + needle_block
    so_far_tokens = len(tokenizer(so_far, add_special_tokens=False).input_ids)
    while so_far_tokens < target_tokens:
        section_idx += 1
        marker = f"\n\n=== Operations Report Section {section_idx} ===\n\n"
        post_parts.append(marker + _FILLER)
        so_far = pre_text + needle_block + "".join(post_parts)
        so_far_tokens = len(tokenizer(so_far, add_special_tokens=False).input_ids)
    return so_far, so_far_tokens


QUESTION = (
    "What is the temporary security access code for the east-wing server room "
    "during the May 2026 maintenance window? Reply with just the code."
)
EXPECTED_FRAGMENT = "QM-7194-ZULU"


def _generate(model, tokenizer, prompt: str, *, cache=None) -> tuple[str, int, float]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            past_key_values=cache,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    decoded = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    return decoded, ids.shape[1], elapsed


def _reset_attention_rotations(model) -> None:
    """Drop the OSCAR rotation buffers attached to each attention block.

    After a previous apply_rotations call, the patched-forward path requires
    self._oscar_k_rot / self._oscar_v_rot to be present. We can't unpatch the
    forward (it's class-level), so to run a "no OSCAR" baseline we must
    reload the model. This helper is only for re-applying with new buffers.
    """
    pass  # No-op: we reload the model for the baseline; the OSCAR runs all
    # reuse the same model object and apply_rotations() overwrites buffers.


def run_oscar(
    model, tokenizer, prompt: str, *, sink: int, recent: int, label: str,
) -> dict:
    cache = OSCARCache(
        config=model.config,
        sink_tokens=sink,
        recent_tokens=recent,
        bits=2,
        group_size=128,
    )
    pred, prompt_tokens, elapsed = _generate(model, tokenizer, prompt, cache=cache)
    ok = EXPECTED_FRAGMENT in pred.upper()
    return {
        "label": label,
        "sink": sink,
        "recent": recent,
        "prompt_tokens": prompt_tokens,
        "prediction": pred,
        "expected_in_pred": ok,
        "elapsed_sec": round(elapsed, 1),
    }


def main() -> int:
    if not K_PATH.exists() or not V_PATH.exists():
        print(f"missing rotation files at {ROT_DIR}", file=sys.stderr)
        return 1

    print(f"loading {MODEL_ID} (bf16) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Two-step load: from_pretrained(..., device_map='cuda') intermittently
    # dies at ~40% on this host (silent kill, exit 5, no Python traceback,
    # appeared after the 9.5 h overnight LoCoMo run). Loading to CPU first
    # and then .to('cuda') reliably succeeds with the same end state.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    print("  loaded to CPU; moving to CUDA ...", flush=True)
    model = model.to("cuda")
    model.eval()

    print(f"building context (~{TARGET_CONTEXT_TOKENS} tok, needle at ~{NEEDLE_OFFSET_TOKENS}) ...", flush=True)
    context, ctx_tokens = build_context(
        tokenizer, target_tokens=TARGET_CONTEXT_TOKENS, needle_offset_tokens=NEEDLE_OFFSET_TOKENS,
    )
    prompt = f"{context}\n\nQuestion: {QUESTION}"
    print(f"  context tokens (raw): {ctx_tokens}", flush=True)

    results: list[dict] = []

    # ----- Baseline (no OSCAR; no rotation applied) -----
    print("\n=== [baseline] bf16, no OSCAR ===", flush=True)
    pred, ptok, elapsed = _generate(model, tokenizer, prompt)
    ok = EXPECTED_FRAGMENT in pred.upper()
    print(f"  prompt_tokens={ptok}  elapsed={elapsed:.1f}s")
    print(f"  pred: {pred!r}")
    print(f"  needle found: {ok}")
    results.append({
        "label": "baseline-bf16", "sink": None, "recent": None,
        "prompt_tokens": ptok, "prediction": pred,
        "expected_in_pred": ok, "elapsed_sec": round(elapsed, 1),
    })

    # ----- Wire OSCAR rotations once; vary sink/recent across runs -----
    print(f"\nloading rotations:\n  K: {K_PATH}\n  V: {V_PATH}", flush=True)
    k_rot = load_rotation_file(K_PATH)
    v_rot = load_rotation_file(V_PATH)
    apply_rotations(model, k_rotations=k_rot, v_rotations=v_rot)

    configs = [
        ("oscar default (64/256)", 64, 256),
        ("oscar moderate (512/2048)", 512, 2048),
        ("oscar generous (1024/4096)", 1024, 4096),
    ]
    for label, sink, recent in configs:
        print(f"\n=== [{label}] sink={sink} recent={recent} ===", flush=True)
        r = run_oscar(model, tokenizer, prompt, sink=sink, recent=recent, label=label)
        print(f"  prompt_tokens={r['prompt_tokens']}  elapsed={r['elapsed_sec']}s")
        print(f"  pred: {r['prediction']!r}")
        print(f"  needle found: {r['expected_in_pred']}")
        results.append(r)

    # ----- Summary table -----
    print("\n=== SUMMARY ===", flush=True)
    print(f"  needle position: ~{NEEDLE_OFFSET_TOKENS} tokens from start of context")
    print(f"  expected: code containing {EXPECTED_FRAGMENT!r}")
    print()
    print(f"  {'config':<32} {'sink':>5} {'recent':>7} {'time':>7} needle?")
    for r in results:
        sink = "-" if r["sink"] is None else str(r["sink"])
        recent = "-" if r["recent"] is None else str(r["recent"])
        mark = "YES" if r["expected_in_pred"] else "NO"
        print(f"  {r['label']:<32} {sink:>5} {recent:>7} {r['elapsed_sec']:>6.1f}s {mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
