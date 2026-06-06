"""Isolated retry of port_debug test F (GPQA-cal rotation, OSCARCache(int4)).

Test F crashed once with a transient CUDA "unknown error" — same intermittent
issue seen before. Re-running just F (not the full A-E suite) saves ~50 min.
"""
from __future__ import annotations
import sys
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from oscar_transformers import OSCARCache, apply_rotations, load_rotation_file
from run.oscar_smoke_middle import MODEL_ID, EXPECTED_FRAGMENT, QUESTION, build_context

GPQA_K = Path("data/oscar/rotations/instruct_gpqa/k_rotation_qqt_r_h_pbr.pt")
GPQA_V = Path("data/oscar/rotations/instruct_gpqa/v_rotation_sst_r_h_pbr.pt")
MAX_NEW = 80
TARGET_CONTEXT_TOKENS = 4000
NEEDLE_OFFSET_TOKENS = 1500

def main() -> int:
    print("[debug] tokenizer + prompt ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    context, _ = build_context(
        tokenizer, target_tokens=TARGET_CONTEXT_TOKENS,
        needle_offset_tokens=NEEDLE_OFFSET_TOKENS,
    )
    prompt = f"{context}\n\nQuestion: {QUESTION}"

    print(f"[debug] loading {MODEL_ID} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda").eval()

    print("\n=== test F: GPQA-cal rotation, OSCARCache(int4) ===", flush=True)
    k_rot = load_rotation_file(GPQA_K)
    v_rot = load_rotation_file(GPQA_V)
    apply_rotations(model, k_rotations=k_rot, v_rotations=v_rot)
    cache = OSCARCache(config=model.config, bits=4)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW, do_sample=False,
            past_key_values=cache, pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    pred = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    needle = EXPECTED_FRAGMENT in pred.upper()
    print(f"  elapsed={elapsed:.1f}s")
    print(f"  pred: {pred!r}")
    print(f"  needle (QM-7194-ZULU present): {needle}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
