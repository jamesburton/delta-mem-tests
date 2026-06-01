"""Dump post-RoPE Q/K/V activations from a Qwen3 model for OSCAR calibration.

Reimplements OSCAR's SGLang-based dump phase in plain HF transformers so the
calibration pipeline runs natively on Windows without SGLang/WSL/cloud GPUs.

Layout (matches upstream's ``rotation/compute_kv_rotation.py`` expectations):

    <dump_dir>/layer_<id>/q/<chunk_id>.pt   shape (n_tokens, n_heads,    head_dim)
    <dump_dir>/layer_<id>/k/<chunk_id>.pt   shape (n_tokens, kv_heads,   head_dim)
    <dump_dir>/layer_<id>/v/<chunk_id>.pt   shape (n_tokens, kv_heads,   head_dim)

The tensors are written in bfloat16; ``compute_kv_rotation.py`` casts them to
float64 on load. ``chunk_id`` is the prompt index in the batch (one chunk per
prompt). Upstream's "all" mode skips ``chunk_0`` (a warmup batch in their
scheduler), so we emit chunk indices starting at 1 — this gives a free
SGLang-compat warmup-skip if we later want to mirror that convention.

The patch lives only for the duration of ``dump_qkv`` so the model class is
not left mutated; the patched forward writes to disk as a side-effect of each
forward, then defers to the unpatched forward for the actual compute.

Usage:
    python -m run.oscar_dump_qkv --calibration gpqa  --num-prompts 198 --output data/oscar/dumps/instruct_gpqa
    python -m run.oscar_dump_qkv --calibration locomo --num-prompts  10 --output data/oscar/dumps/instruct_locomo

For sanity checks pass ``--num-prompts 2`` and inspect the resulting layer_0/
shapes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_MAX_PROMPT_TOKENS = 4096  # safety cap; long prompts get truncated


def _patch_qwen3_attention_for_dump(dump_dir: Path) -> Callable[[], None]:
    """Patch ``Qwen3Attention.forward`` to dump post-RoPE q/k/v per layer.

    Returns an ``unpatch()`` callable that restores the original forward and
    drops the per-attention dump state. We do not save the unpatched forward
    on the class; the caller must call ``unpatch()`` before re-applying any
    other class-level patch (e.g. ``apply_rotations``) on the same process.
    """
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention,
        apply_rotary_pos_emb,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

    original_forward = Qwen3Attention.forward

    state = {"chunk_id": 0}

    def set_chunk_id(idx: int) -> None:
        state["chunk_id"] = int(idx)

    def patched_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # query_states: (B, n_heads, T, head_dim)
        # key/value_states: (B, kv_heads, T, head_dim)
        # Upstream expects (T_total, heads, head_dim). We assume B=1 (one
        # prompt per dump call) and emit (T, heads, head_dim) per prompt.
        b, _h, t, d = query_states.shape
        if b != 1:
            raise RuntimeError(
                f"Dumper assumes batch_size=1; got B={b}. Run the dumper one prompt at a time."
            )
        layer_id = self.layer_idx
        chunk_id = state["chunk_id"]
        for tag, tensor in (("q", query_states), ("k", key_states), ("v", value_states)):
            out_dir = dump_dir / f"layer_{layer_id}" / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            # (B, H, T, D) -> (T, H, D), keep bf16, on CPU.
            t_out = tensor[0].transpose(0, 1).contiguous().to(torch.bfloat16).cpu()
            torch.save(t_out, out_dir / f"{chunk_id}.pt")

        # Update the cache (no rotation, no INT2 — same as the unpatched
        # forward) and run the attention dot product. We replicate the
        # original forward's body verbatim from here on, sans the rotation
        # block, so the model still produces real logits and the dump only
        # adds the side-effect.
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        b2, t2 = attn_output.shape[0], attn_output.shape[1]
        attn_output = attn_output.reshape(b2, t2, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    Qwen3Attention.forward = patched_forward

    def unpatch() -> None:
        Qwen3Attention.forward = original_forward

    return set_chunk_id, unpatch


def _load_gpqa_prompts(num_prompts: int) -> list[str]:
    """Build OSCAR-style GPQA calibration prompts.

    Uses the GPQA-Diamond split (198 questions) from huggingface.co/datasets/Idavidrein/gpqa.
    The HF dataset is gated; you must accept the licence at the dataset page
    before this load call succeeds.

    The prompt format mirrors upstream's ``dump_gpqa_prompts.py`` (zero-shot
    MCQ with the four options shuffled-then-listed and a final
    ``"Answer:"`` cue). The answer doesn't matter — the model emits one token
    (max_new_tokens=1) and we only care about the prefill activations.
    """
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    prompts: list[str] = []
    for ex in list(ds)[:num_prompts]:
        q = ex["Question"]
        choices = [
            ex["Correct Answer"],
            ex["Incorrect Answer 1"],
            ex["Incorrect Answer 2"],
            ex["Incorrect Answer 3"],
        ]
        # Stable shuffle by question hash so the prompt set is reproducible.
        h = hash(q) % 24
        from itertools import permutations
        order = list(permutations(range(4)))[h]
        shuffled = [choices[i] for i in order]
        choice_block = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(shuffled))
        prompts.append(
            f"Question: {q}\n\nOptions:\n{choice_block}\n\nAnswer:"
        )
    return prompts


def _load_locomo_prompts(num_prompts: int, data_file: Path) -> list[str]:
    """Build LoCoMo-style calibration prompts.

    Each LoCoMo conversation produces multiple QA prompts. We assemble each
    prompt as a "history + question" string in the same shape the eval feeds
    the model (matches official_prompt mode), then truncate at
    ``DEFAULT_MAX_PROMPT_TOKENS`` so we don't blow VRAM during prefill on a
    12 GB card. Truncation is fine for calibration: the spectral covariance
    is averaged over many tokens and a prefix of a long conversation captures
    representative activations.

    The vendored data file is at ``delta-Mem/data/locomo10.json``.
    """
    blob = json.loads(data_file.read_text(encoding="utf-8"))
    prompts: list[str] = []
    for conv in blob[: max(1, num_prompts)]:
        sessions = conv.get("conversation", {})
        # Concatenate every speaker turn from every session into one history.
        lines: list[str] = []
        for k, msgs in sessions.items():
            if not isinstance(msgs, list):
                continue
            for m in msgs:
                speaker = m.get("speaker", "")
                text = m.get("text", "")
                if speaker and text:
                    lines.append(f"{speaker}: {text}")
        history = "\n".join(lines)
        qa = conv.get("qa", [])
        if not qa:
            continue
        q = qa[0].get("question", "")
        prompts.append(f"{history}\n\nQuestion: {q}\nAnswer:")
        if len(prompts) >= num_prompts:
            break
    return prompts


def _dump_one_prompt(model, tokenizer, prompt: str, max_tokens: int, set_chunk_id, chunk_id: int) -> int:
    """Run a single prompt through the patched model and return n_tokens fed."""
    set_chunk_id(chunk_id)
    ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_tokens,
    ).input_ids.to(model.device)
    with torch.no_grad():
        _ = model(input_ids=ids, use_cache=False)
    return ids.shape[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", choices=["gpqa", "locomo"], required=True)
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS,
                    help="Cap each prompt at this many tokens to control VRAM.")
    ap.add_argument("--output", required=True, help="Directory for the dump output.")
    ap.add_argument("--locomo-data", default="delta-Mem/data/locomo10.json",
                    help="Path to locomo10.json (used only for --calibration locomo).")
    ap.add_argument("--dry-shape-check", action="store_true",
                    help="Dump 1 prompt and print layer_0 shapes, then exit.")
    args = ap.parse_args()

    dump_dir = Path(args.output)
    dump_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dumper] calibration={args.calibration} num_prompts={args.num_prompts}", flush=True)
    print(f"[dumper] output={dump_dir}", flush=True)

    # Build calibration prompts BEFORE loading the model. On this host loading
    # the HF datasets cache for GPQA after the model is on CUDA reproducibly
    # silent-kills the process (exit 5, no traceback). Building prompts first
    # avoids the conflict and costs us nothing.
    print(f"[dumper] tokenizer + prompts ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if args.calibration == "gpqa":
        prompts = _load_gpqa_prompts(args.num_prompts)
    else:
        prompts = _load_locomo_prompts(args.num_prompts, Path(args.locomo_data))
    print(f"[dumper] built {len(prompts)} prompts", flush=True)
    if not prompts:
        print("[dumper] no prompts; aborting", file=sys.stderr)
        return 2

    print(f"[dumper] loading {MODEL_ID} (bf16) ...", flush=True)
    # Two-step load: from_pretrained(..., device_map='cuda') intermittently
    # silent-kills at ~40% on this host. CPU->.to('cuda') is reliable.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model = model.to("cuda").eval()

    set_chunk_id, unpatch = _patch_qwen3_attention_for_dump(dump_dir)
    try:
        total_tokens = 0
        t0 = time.time()
        for i, prompt in enumerate(prompts):
            # Chunk index starts at 1 to mirror upstream's "skip chunk 0" mode.
            n = _dump_one_prompt(
                model, tokenizer, prompt,
                max_tokens=args.max_prompt_tokens,
                set_chunk_id=set_chunk_id, chunk_id=i + 1,
            )
            total_tokens += n
            elapsed = time.time() - t0
            print(
                f"[dumper] prompt {i+1}/{len(prompts)} tokens={n} "
                f"total={total_tokens} elapsed={elapsed:.1f}s",
                flush=True,
            )
            if args.dry_shape_check:
                # Print shapes of the just-written layer_0 files and exit.
                from glob import glob
                for tag in ("q", "k", "v"):
                    path = dump_dir / "layer_0" / tag / f"{i+1}.pt"
                    t = torch.load(str(path), map_location="cpu")
                    print(f"  layer_0/{tag}/{i+1}.pt: shape={tuple(t.shape)} dtype={t.dtype}")
                break
    finally:
        unpatch()

    if args.dry_shape_check:
        return 0

    # Record metadata for the compute step.
    meta = {
        "model_id": MODEL_ID,
        "calibration": args.calibration,
        "num_prompts": len(prompts),
        "total_tokens": total_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (dump_dir / "dump_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[dumper] done. total_tokens={total_tokens} meta={dump_dir/'dump_meta.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
