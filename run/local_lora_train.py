"""Tiny local fine-tune of the delta-mem adapter at 2 k context.

Purpose: validate that the training recipe produces a sensible loss curve
on a slice of long-context-conversational data, BEFORE committing Strix
Halo hours. The 12 GB ceiling caps us at ~2048-token context with reentrant
gradient checkpointing (see run/training_smoke.py findings), so this is a
recipe smoke, not a full training run. The trained checkpoint round-trips
through :func:`save_delta_mem_adapter` and can be evaluated with
``run.locomo_eval --adapter-override <ckpt_dir>``.

What it does
------------

1. Load Qwen3-4B-Instruct-2507 + published delta-mem adapter (freeze
   backbone, only adapter trainable).
2. Build training examples from LoCoMo's own conversations (cheapest
   long-context source we already have on disk) — concat sessions into
   ~2 k-token chunks with the QA turns inline.
3. Fine-tune for ~100 steps (configurable) with:
   - AdamW, lr=1e-5 (10x lower than initial; we're nudging from a working
     adapter, not training from scratch)
   - gradient_checkpointing with use_reentrant=True (required per
     training_smoke findings)
   - batch=1, no grad-accum (12 GB doesn't allow more)
4. Log loss every 10 steps; save checkpoint at the end.
5. Print quick-eval command so the result can be validated.

What this does NOT do
---------------------

- Extend context-length capability (still 2 k; Strix Halo does that).
- Validate on a held-out set (just trains; eval is a separate step via
  locomo_eval).
- Touch hyperparameters that need real-scale search (rank, etc).

Usage
-----

    python -m run.local_lora_train \\
        --steps 100 \\
        --out checkpoints/local_lora_v0/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamem.core.delta import HFDeltaMemConfig, attach_delta_mem
from deltamem.core.delta_impl import (
    freeze_non_delta_mem_params,
    load_delta_mem_state_dict,
    save_delta_mem_adapter,
)

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_ID = "declare-lab/delta-mem_qwen3_4b-instruct"
LOCOMO_PATH = Path("data/locomo10.json")
CONTEXT_TOKENS = 2048
LR = 1e-5
LOG_EVERY = 10


def _format_session_text(session_messages) -> str:
    """Render a LoCoMo session's messages as plain dialogue text."""
    lines: List[str] = []
    for msg in session_messages:
        if isinstance(msg, dict):
            spk = msg.get("speaker", "?")
            txt = msg.get("text", "")
            lines.append(f"{spk}: {txt}")
        else:
            lines.append(str(msg))
    return "\n".join(lines)


def _build_training_chunks(
    tokenizer, target_tokens: int, max_chunks: int = 64,
) -> List[torch.Tensor]:
    """Read LoCoMo, concatenate session texts, slice into target_tokens-sized
    chunks. Each chunk becomes one training example.
    """
    raw = json.loads(LOCOMO_PATH.read_text())
    chunks: List[torch.Tensor] = []
    for conv in raw:
        convo = conv.get("conversation", {})
        session_keys = sorted(
            [k for k in convo if k.startswith("session_") and not k.endswith("date_time")],
            key=lambda x: int(x.split("_")[1]),
        )
        parts = [_format_session_text(convo[sk]) for sk in session_keys]
        full_text = "\n\n".join(parts)
        ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        # Slice into target_tokens-sized chunks
        for start in range(0, len(ids) - target_tokens + 1, target_tokens):
            chunk = ids[start:start + target_tokens]
            chunks.append(chunk)
            if len(chunks) >= max_chunks:
                return chunks
    return chunks


def _attach_adapter(model) -> HFDeltaMemConfig:
    """Wrap model with delta-mem and load published adapter weights."""
    adapter_dir = snapshot_download(ADAPTER_ID)
    config = HFDeltaMemConfig.from_pretrained(adapter_dir)
    attach_delta_mem(model, config)
    state = torch.load(
        Path(adapter_dir) / "delta_mem_adapter.pt",
        map_location="cpu", weights_only=True,
    )
    load_delta_mem_state_dict(model, state)
    return config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100,
                    help="Number of optimizer steps (default 100).")
    ap.add_argument("--out", type=str, required=True,
                    help="Output directory for the trained adapter checkpoint.")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--context", type=int, default=CONTEXT_TOKENS)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1

    out_dir = Path(args.out).resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[lora] loading backbone {MODEL_ID} ...", flush=True)
    model_dir = snapshot_download(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")

    print(f"[lora] attaching delta-mem adapter ...", flush=True)
    delta_config = _attach_adapter(model)

    trainable_names = freeze_non_delta_mem_params(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable_params)
    print(f"  trainable: {len(trainable_names)} tensors / {n_train/1e6:.1f} M params",
          flush=True)

    # REQUIRED per training_smoke findings — non-reentrant breaks the scan kernel
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True}
    )

    print(f"[lora] building training chunks at {args.context} tokens ...", flush=True)
    chunks = _build_training_chunks(tokenizer, target_tokens=args.context)
    print(f"  built {len(chunks)} chunks of {args.context} tokens each", flush=True)
    if not chunks:
        print("no training chunks produced — check LOCOMO_PATH", file=sys.stderr)
        return 1

    optim = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    model.train()

    loss_log: List[tuple[int, float, float]] = []  # (step, loss, peak_gb)
    t0 = time.time()
    print(f"[lora] training {args.steps} steps, lr={args.lr}, chunks_in_pool={len(chunks)}", flush=True)
    for step in range(1, args.steps + 1):
        chunk = chunks[(step - 1) % len(chunks)]
        input_ids = chunk.unsqueeze(0).to("cuda")
        labels = input_ids.clone()
        torch.cuda.reset_peak_memory_stats()
        optim.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        out.loss.backward()
        # Gradient clipping for stability with bf16 scan kernel
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optim.step()
        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        loss_val = out.loss.item()
        if not (loss_val == loss_val):  # NaN check
            print(f"  step {step}: NaN loss — aborting", flush=True)
            return 2
        if step == 1 or step % LOG_EVERY == 0 or step == args.steps:
            elapsed = time.time() - t0
            print(f"  step {step:>4}  loss={loss_val:.4f}  peak={peak_gb:.2f} GB  "
                  f"elapsed={elapsed:.1f}s", flush=True)
        loss_log.append((step, loss_val, peak_gb))
        del input_ids, labels, out

    total_s = time.time() - t0
    print(f"[lora] training done in {total_s:.1f}s ({total_s/args.steps:.1f} s/step)",
          flush=True)

    # Quick loss-curve quality check
    losses = [v for _, v, _ in loss_log]
    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    drop = early - late
    print(f"[lora] loss: early(first 5) avg={early:.4f}, late(last 5) avg={late:.4f}, "
          f"drop={drop:+.4f}", flush=True)
    if drop > 0:
        print(f"  -> loss decreased ({drop:.3f}) -- recipe is doing something useful")
    else:
        print(f"  -> loss did NOT decrease (delta {drop:+.3f}) -- LR or data may be off")

    # Save checkpoint
    print(f"[lora] saving adapter to {out_dir} ...", flush=True)
    save_delta_mem_adapter(model, str(out_dir), delta_config)
    # Also dump the loss log for later inspection
    (out_dir / "loss_log.json").write_text(json.dumps({
        "steps": args.steps,
        "lr": args.lr,
        "context_tokens": args.context,
        "trainable_tensors": len(trainable_names),
        "trainable_params": int(n_train),
        "wall_seconds": total_s,
        "loss_per_step": [{"step": s, "loss": v, "peak_gb": p} for s, v, p in loss_log],
        "early_avg": early,
        "late_avg": late,
        "drop": drop,
    }, indent=2))

    print(f"\n[lora] Evaluate via:\n"
          f"  python -m run.locomo_eval --kv-cache-backend oscar --kv-cache-bits 2 \\\n"
          f"      --max-conversations 1 --max-questions-per-conversation 10 \\\n"
          f"      --adapter-override {out_dir} \\\n"
          f"      --output-json outputs/local_lora_smoke_eval.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
