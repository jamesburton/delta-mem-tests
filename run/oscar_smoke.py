"""Quick smoke test for the OSCAR port on Qwen3-4B-Instruct-2507.

What it does:
  1. Loads Qwen3-4B-Instruct-2507 in bf16 on CUDA.
  2. Generates a short answer to a known-easy prompt without OSCAR
     (baseline).
  3. Bakes RotationZoo Thinking-2507 rotations into the same model,
     attaches an OSCARCache, generates the same prompt.
  4. Prints both outputs side by side.

If step 4 produces gibberish, the Thinking rotations do NOT transfer
to Instruct — we then need to run our own 3-phase calibration. If it
produces coherent (even if slightly worse) text, the path is viable
for the conv-0/10q eval.

Run:
    .venv/Scripts/python.exe run/oscar_smoke.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oscar_transformers import OSCARCache, apply_rotations, load_rotation_file


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ROT_DIR = Path("data/oscar/rotations/_hf_cache/Qwen3-4B-Thinking-2507/seq20000_prompt83_group128")
K_PATH = ROT_DIR / "k_rotation_qqt_r_h_pbr.pt"
V_PATH = ROT_DIR / "v_rotation_sst_r_h_pbr.pt"

PROMPT = "What is the capital of France? Answer in one short sentence."
MAX_NEW = 40


def _generate(model, tokenizer, prompt: str, *, cache=None) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            past_key_values=cache,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main() -> int:
    if not K_PATH.exists() or not V_PATH.exists():
        print(f"missing rotation files at {ROT_DIR}", file=sys.stderr)
        return 1

    print(f"loading {MODEL_ID} (bf16) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    print("=== baseline (no OSCAR) ===", flush=True)
    baseline = _generate(model, tokenizer, PROMPT)
    print(baseline, flush=True)

    print("\nloading rotations ...", flush=True)
    k_rot = load_rotation_file(K_PATH)
    v_rot = load_rotation_file(V_PATH)
    print(
        f"  k: layers={len(k_rot)} head_dim={k_rot.head_dim} obj={k_rot.objective!r}\n"
        f"  v: layers={len(v_rot)} head_dim={v_rot.head_dim} obj={v_rot.objective!r}",
        flush=True,
    )

    print("wiring rotations into attention forwards ...", flush=True)
    apply_rotations(model, k_rotations=k_rot, v_rotations=v_rot)

    print("=== rotation-only (baked, no quantization) ===", flush=True)
    rotonly_out = _generate(model, tokenizer, PROMPT, cache=None)
    print(rotonly_out, flush=True)

    print("=== with OSCAR (rotations baked + INT2 cache) ===", flush=True)
    cache = OSCARCache(config=model.config)
    oscar_out = _generate(model, tokenizer, PROMPT, cache=cache)
    print(oscar_out, flush=True)

    print("\n=== summary ===", flush=True)
    print(f"baseline:    {baseline!r}")
    print(f"rot-only:    {rotonly_out!r}")
    print(f"oscar(int2): {oscar_out!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
