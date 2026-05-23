"""Smoke test: load Qwen3-4B-Instruct-2507 + delta-mem adapter, run a 3-turn
chat, and assert that the online memory state has measurably changed between
turn 1 and turn 3.

Outputs:
    - report/smoke.md (transcript + evidence)
    - exit 0 on PASS, 1 on FAIL
"""

from __future__ import annotations

import os
# Path-lock per report/kernels-gate.md: Triton is not installed on this host;
# delta-mem's torch reference path (delta_impl.py:1895-1938) is numerically
# equivalent to the Triton kernel. Setting this before importing deltamem
# prevents a path-switch if Triton ever gets installed in the future.
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
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
        for attr in ("delta_state", "memory_state", "mem_state", "state",
                     "online_state", "mem", "mem_matrix", "S", "M", "m"):
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

    print(f"Downloading delta-mem adapter: {ADAPTER}")
    adapter_path = snapshot_download(ADAPTER)
    print(f"  -> cached at: {adapter_path}")

    print(f"Attaching delta-mem adapter")
    cfg = HFDeltaMemConfig.from_pretrained(adapter_path)
    attach_delta_mem(model, cfg)
    load_delta_mem_adapter(model, adapter_path)
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
        # Replace characters not representable in the console encoding
        # (Windows cp1252 can't handle all Unicode the model may emit).
        safe_reply = reply.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8"
        )
        print(f"\nUSER: {t}\nASSISTANT: {safe_reply}")

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
        "- Scan implementation: `torch` (locked via `DELTA_MEM_SCAN_IMPL=torch`; Triton unavailable — see `report/kernels-gate.md`)",
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
