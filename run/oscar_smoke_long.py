"""Longer-prompt OSCAR smoke that actually exercises the INT2 middle region.

The short smoke (run/oscar_smoke.py) uses a ~30-token prompt. With
sink_tokens=64 and recent_tokens=256, all tokens land in the sink or recent
(full-precision) regions; the INT2 middle is never populated. This script
constructs a synthetic ~1k-token prompt so prefill spills past 320 tokens
into the INT2 middle, then generates ~80 new tokens so the answer continues
to span recent + middle.

Outputs three answers (baseline / rot-only / OSCAR-INT2) side by side. The
rot-only run validates that apply_rotations is still a no-op end-to-end on
this prompt length. The OSCAR-INT2 run validates that the Thinking-2507
rotations transfer adequately to Instruct-2507 on a prompt that actually
uses the INT2 path.

What "passing" means here:
  - baseline produces a coherent answer to the question;
  - rot-only matches the baseline byte-for-byte (or near-byte; small
    bf16 noise is tolerable);
  - OSCAR-INT2 produces a coherent answer to the question. It may differ
    in wording from the baseline — the goal is to confirm INT2 doesn't
    collapse to gibberish/repeats, NOT to demand exact equivalence.

Run:
    .venv/Scripts/python.exe run/oscar_smoke_long.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oscar_transformers import OSCARCache, apply_rotations, load_rotation_file


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ROT_DIR = Path("data/oscar/rotations/_hf_cache/Qwen3-4B-Thinking-2507/seq20000_prompt83_group128")
K_PATH = ROT_DIR / "k_rotation_qqt_r_h_pbr.pt"
V_PATH = ROT_DIR / "v_rotation_sst_r_h_pbr.pt"

MAX_NEW = 80

# A deterministic ~1.2k-token context built from a Wikipedia-style passage
# repeated with light variation, ending in a specific factual question whose
# answer is grounded in the passage. The repetition is intentional — it gives
# us a predictable target prompt length while keeping the content semantically
# coherent. The question targets a fact that appears near the END of the
# context, so the model must attend through both the INT2 middle (older tokens)
# and the recent window (newest tokens) to answer correctly.

_PASSAGE = """\
Marie Curie was a Polish and naturalised-French physicist and chemist who
conducted pioneering research on radioactivity. She was the first woman to
win a Nobel Prize, the only woman to win the Nobel Prize twice, and the only
person to win the Nobel Prize in two scientific fields. Her husband, Pierre
Curie, was a co-winner of her first Nobel Prize, making them the first
married couple to win a Nobel Prize. The Curie family has won a total of
five Nobel Prizes. Curie was born in Warsaw, in what was then the Kingdom of
Poland, part of the Russian Empire. She studied at Warsaw's clandestine
Flying University and began her practical scientific training in Warsaw. In
1891, aged 24, she followed her elder sister Bronisława to study in Paris,
where she earned her higher degrees and conducted her subsequent scientific
work. In 1895 she married the French physicist Pierre Curie, and she shared
the 1903 Nobel Prize in Physics with him and with the physicist Henri
Becquerel for their pioneering work developing the theory of radioactivity.
In 1911 she won the Nobel Prize in Chemistry for her discovery of the
elements polonium and radium, using techniques she invented for isolating
radioactive isotopes. Under her direction, the world's first studies were
conducted into the treatment of neoplasms by the use of radioactive
isotopes. She founded the Curie Institute in Paris in 1920, and the Curie
Institute in Warsaw in 1932; both remain major medical research centres
today. During World War I she developed mobile radiography units to provide
X-ray services to field hospitals. While a French citizen, Marie Skłodowska
Curie, who used both surnames, never lost her sense of Polish identity. She
taught her daughters the Polish language and took them on visits to Poland.
She named the first chemical element she discovered, polonium, after her
native country. Marie Curie died in 1934, aged 66, at a sanatorium in
Passy, Haute-Savoie, France, of aplastic anemia from exposure to radiation
in the course of her scientific research and in the course of her
radiological work at field hospitals during World War I."""


def _build_long_context(tokenizer, target_tokens: int = 1024) -> str:
    """Build a context of approximately ``target_tokens`` tokens by repeating
    the seed passage. The final concatenation is trimmed so the encoded
    length is close to (slightly above) ``target_tokens``.
    """
    parts: list[str] = []
    total = 0
    while total < target_tokens:
        parts.append(_PASSAGE)
        joined = "\n\n".join(parts)
        total = len(tokenizer(joined, add_special_tokens=False).input_ids)
    return "\n\n".join(parts)


QUESTION = (
    "Based only on the passage above, name the disease Marie Curie died of "
    "and the year she died. Answer in one short sentence."
)


def _generate(model, tokenizer, prompt: str, *, cache=None) -> tuple[str, int]:
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
    decoded = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    return decoded, ids.shape[1]


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

    context = _build_long_context(tokenizer, target_tokens=1024)
    prompt = f"{context}\n\nQuestion: {QUESTION}"

    print(f"context tokens (raw): "
          f"{len(tokenizer(context, add_special_tokens=False).input_ids)}", flush=True)

    print("=== baseline (no OSCAR) ===", flush=True)
    baseline, prompt_tokens = _generate(model, tokenizer, prompt)
    print(f"(prompt tokens after chat template: {prompt_tokens})", flush=True)
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

    print("=== rotation-only (no quantization) ===", flush=True)
    rotonly_out, _ = _generate(model, tokenizer, prompt, cache=None)
    print(rotonly_out, flush=True)

    print("=== with OSCAR (rotations + INT2 cache) ===", flush=True)
    cache = OSCARCache(config=model.config)
    oscar_out, _ = _generate(model, tokenizer, prompt, cache=cache)
    print(oscar_out, flush=True)

    print("\n=== summary ===", flush=True)
    print(f"prompt_tokens: {prompt_tokens}")
    print(f"sink/recent/middle threshold: sink=64 recent=256, INT2 spillover starts at {64 + 256} tokens")
    print(f"baseline:    {baseline!r}")
    print(f"rot-only:    {rotonly_out!r}")
    print(f"oscar(int2): {oscar_out!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
