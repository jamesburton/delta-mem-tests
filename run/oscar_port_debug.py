"""Port debug script: isolate where OSCAR-INT2 quality loss comes from.

The middle-region smoke (run/oscar_smoke_middle.py) has now shown the same
needle-loss pattern under THREE rotation sources:
  - RotationZoo Thinking-2507 (originally suspected as transfer mismatch)
  - GPQA-calibrated on Instruct-2507 (matches upstream recipe exactly)
  - LoCoMo-calibrated on Instruct-2507 (matches eval distribution exactly)

If calibration source doesn't matter, the bug is more likely in OUR PORT
(third_party/oscar-transformers) than in OSCAR itself. This script decomposes
the failure into rotation-only and quantize-only paths so we can pinpoint
which component is responsible.

Tests (all on the same 4 k prompt with the needle at position ~1500):

  baseline    no rotation, bf16 DynamicCache              expected: needle YES
  test A      identity rotation, bf16 DynamicCache        expected: needle YES (identity is no-op)
                                                          fails ⇒ patched_forward wrapper bug
  test B      GPQA-cal rotation, bf16 DynamicCache        expected: needle YES (rotation is orthogonal)
                                                          fails ⇒ rotation math in bf16 is wrong
  test C      identity rotation, OSCARCache(int2)         expected: needle ? (quant noise only)
                                                          fails ⇒ INT2 itself destroys recall regardless of basis
                                                          passes ⇒ rotation+quant interaction is the bug

Reuses the build_context helper from oscar_smoke_middle to keep the prompt
identical to the prior smoke runs.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from oscar_transformers import OSCARCache, apply_rotations, load_rotation_file
from oscar_transformers.rotation import LayerRotation, RotationSet

from run.oscar_smoke_middle import (
    MODEL_ID,
    EXPECTED_FRAGMENT,
    QUESTION,
    build_context,
)

MAX_NEW = 80
TARGET_CONTEXT_TOKENS = 4000
NEEDLE_OFFSET_TOKENS = 1500
HEAD_DIM = 128
NUM_LAYERS = 36

# GPQA-cal rotations from earlier calibration step.
GPQA_K = Path("data/oscar/rotations/instruct_gpqa/k_rotation_qqt_r_h_pbr.pt")
GPQA_V = Path("data/oscar/rotations/instruct_gpqa/v_rotation_sst_r_h_pbr.pt")


def _identity_rotation_set(num_layers: int = NUM_LAYERS, head_dim: int = HEAD_DIM) -> RotationSet:
    """Build a per-layer RotationSet of identity matrices. Apply this and the
    patched forward should produce baseline-equivalent output in infinite
    precision (every rotation einsum is a no-op).
    """
    layers = {
        i: LayerRotation(
            layer_id=i,
            rotation=torch.eye(head_dim, dtype=torch.float32),
            eigenvalues=torch.ones(head_dim, dtype=torch.float32),
        )
        for i in range(num_layers)
    }
    return RotationSet(objective="identity", layers=layers)


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


def _summarize(label: str, pred: str, baseline: str | None) -> dict:
    ok = EXPECTED_FRAGMENT in pred.upper()
    matches_baseline = (pred == baseline) if baseline is not None else None
    return {
        "label": label,
        "prediction": pred,
        "needle": ok,
        "matches_baseline": matches_baseline,
    }


def main() -> int:
    # Prompts and tokenizer first (avoids the device_map='cuda' silent-kill
    # path that we hit when datasets.load_dataset runs after model is on CUDA;
    # also keeps a consistent ordering with the other smokes).
    print("[debug] tokenizer + prompt ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    context, _ = build_context(
        tokenizer, target_tokens=TARGET_CONTEXT_TOKENS,
        needle_offset_tokens=NEEDLE_OFFSET_TOKENS,
    )
    prompt = f"{context}\n\nQuestion: {QUESTION}"
    print(f"  context tokens: {len(tokenizer(context, add_special_tokens=False).input_ids)}", flush=True)

    print(f"[debug] loading {MODEL_ID} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model = model.to("cuda").eval()

    results: list[dict] = []

    # ----- baseline -----
    print("\n=== baseline (no rotation, bf16 DynamicCache) ===", flush=True)
    pred, ptok, elapsed = _generate(model, tokenizer, prompt)
    print(f"  prompt_tokens={ptok}  elapsed={elapsed:.1f}s")
    print(f"  pred: {pred!r}")
    baseline_pred = pred
    results.append(_summarize("baseline", pred, baseline=None))

    # ----- test A: identity rotation, DynamicCache -----
    # Apply identity rotation. The patched forward is now active for the rest
    # of this process; subsequent tests can update the buffers in place.
    print("\n=== test A: identity rotation, DynamicCache ===", flush=True)
    id_set = _identity_rotation_set()
    apply_rotations(model, k_rotations=id_set, v_rotations=id_set)
    pred, _, elapsed = _generate(model, tokenizer, prompt)
    print(f"  elapsed={elapsed:.1f}s")
    print(f"  pred: {pred!r}")
    print(f"  matches baseline: {pred == baseline_pred}")
    results.append(_summarize("test_A_identity_dyncache", pred, baseline=baseline_pred))

    # ----- test B: GPQA-cal rotation, DynamicCache -----
    print("\n=== test B: GPQA-cal rotation, DynamicCache ===", flush=True)
    k_rot = load_rotation_file(GPQA_K)
    v_rot = load_rotation_file(GPQA_V)
    apply_rotations(model, k_rotations=k_rot, v_rotations=v_rot)
    pred, _, elapsed = _generate(model, tokenizer, prompt)
    print(f"  elapsed={elapsed:.1f}s")
    print(f"  pred: {pred!r}")
    print(f"  matches baseline: {pred == baseline_pred}")
    results.append(_summarize("test_B_gpqa_dyncache", pred, baseline=baseline_pred))

    # ----- test C: identity rotation, OSCARCache(int2) -----
    print("\n=== test C: identity rotation, OSCARCache(int2) ===", flush=True)
    apply_rotations(model, k_rotations=id_set, v_rotations=id_set)
    cache = OSCARCache(config=model.config)  # default sink=64, recent=256, bits=2
    pred, _, elapsed = _generate(model, tokenizer, prompt, cache=cache)
    print(f"  elapsed={elapsed:.1f}s")
    print(f"  pred: {pred!r}")
    print(f"  matches baseline: {pred == baseline_pred}")
    results.append(_summarize("test_C_identity_oscar_int2", pred, baseline=baseline_pred))

    # ----- test D: GPQA-cal rotation, OSCARCache(int2) — the real thing -----
    print("\n=== test D: GPQA-cal rotation, OSCARCache(int2) ===", flush=True)
    apply_rotations(model, k_rotations=k_rot, v_rotations=v_rot)
    cache = OSCARCache(config=model.config)
    pred, _, elapsed = _generate(model, tokenizer, prompt, cache=cache)
    print(f"  elapsed={elapsed:.1f}s")
    print(f"  pred: {pred!r}")
    print(f"  matches baseline: {pred == baseline_pred}")
    results.append(_summarize("test_D_gpqa_oscar_int2", pred, baseline=baseline_pred))

    # ----- SUMMARY -----
    print("\n=== SUMMARY ===", flush=True)
    print(f"  {'test':<35} {'needle':>7} {'==baseline':>11}")
    for r in results:
        nm = "YES" if r["needle"] else "NO"
        mb = (
            "n/a" if r["matches_baseline"] is None
            else ("YES" if r["matches_baseline"] else "NO")
        )
        print(f"  {r['label']:<35} {nm:>7} {mb:>11}")

    print("\n=== interpretation ===")
    print("  test A fails  -> patched_forward wrapper bug (identity should be no-op)")
    print("  test A passes, test B fails  -> bf16 rotation precision bug")
    print("  tests A,B pass, test C fails needle  -> INT2 itself loses recall regardless of basis")
    print("  tests A,B,C pass needle, D fails  -> rotation+quant interaction is the bug")
    print("  all tests pass needle  -> port is correct; bug is elsewhere")
    return 0


if __name__ == "__main__":
    sys.exit(main())
