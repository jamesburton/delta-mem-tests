"""Strix Halo phase-1 delta-mem adapter fine-tune at 32 k context.

This is the production-scale companion to ``run/local_lora_train.py``
(which proved the recipe at 2 k on the 12 GB local box). The structure
mirrors that script; only the scale parameters and the data source change.

Hyperparameters follow ``STRIX_INSTRUCTIONS.md`` "Training plan":

  - max_seq_len = 32_768 (phase 1)
  - per_device_train_batch_size = 1
  - gradient_accumulation_steps = 8
  - lr = 1e-4, weight_decay = 0.01, cosine schedule, 200-step warmup
  - 1000+ optimiser steps (configurable; phase-1 default = 2000)
  - gradient checkpointing with ``use_reentrant=True`` (NON-NEGOTIABLE;
    see CRITICAL block below)

Data source: ``data/longctx_mix_v1.jsonl`` produced by
``strix.prepare_data``. One example per line, already tokenised, in the
form ``{"input_ids": [...], "labels": [...]}``.

Initialisation: continues training from the published
``declare-lab/delta-mem_qwen3_4b-instruct`` adapter — never from scratch
— per the doc's instruction to preserve the 17 k-class anchor.

Usage (Strix Halo):
    python -m strix.train_phase1 \\
        --steps 2000 \\
        --context 32768 \\
        --data data/longctx_mix_v1.jsonl \\
        --out checkpoints/longctx-v1-32k
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

from deltamem.core.delta import HFDeltaMemConfig, attach_delta_mem
from deltamem.core.delta_impl import (
    freeze_non_delta_mem_params,
    load_delta_mem_state_dict,
    save_delta_mem_adapter,
)

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_ID = "declare-lab/delta-mem_qwen3_4b-instruct"

# CRITICAL — DO NOT CHANGE. run/training_smoke.py findings:
#   The default `use_reentrant=False` PLUS no-checkpointing both fail on
#   the second iteration because delta-mem's Triton scan kernel saves
#   tensors in a way incompatible with non-reentrant graph reuse. The
#   legacy reentrant path is the ONLY working option for end-to-end
#   training. Without this, training crashes on step 2 with
#   `RuntimeError: Trying to backward through the graph a second time`
#   originating in deltamem/kernels/affine_scan.py.
GRAD_CHECKPOINT_KWARGS = {"use_reentrant": True}


def _load_examples(jsonl: Path, max_seq: int) -> List[torch.Tensor]:
    """Load tokenised examples and truncate/pad to ``max_seq`` tokens.

    Examples shorter than ``max_seq`` are dropped (we want the trainer
    seeing only at-target-length contexts to teach long-range behaviour).
    """
    out: List[torch.Tensor] = []
    too_short = 0
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ids = rec["input_ids"]
            if len(ids) < max_seq:
                too_short += 1
                continue
            out.append(torch.tensor(ids[:max_seq], dtype=torch.long))
    print(f"[train] loaded {len(out)} examples at {max_seq} tokens "
          f"({too_short} dropped as too short)", flush=True)
    return out


def _attach_adapter(model) -> HFDeltaMemConfig:
    """Wrap model with delta-mem and load published adapter weights as init."""
    adapter_dir = snapshot_download(ADAPTER_ID)
    config = HFDeltaMemConfig.from_pretrained(adapter_dir)
    attach_delta_mem(model, config)
    state = torch.load(
        Path(adapter_dir) / "delta_mem_adapter.pt",
        map_location="cpu", weights_only=True,
    )
    load_delta_mem_state_dict(model, state)
    return config


def _save_checkpoint(model, out_dir: Path, config: HFDeltaMemConfig,
                     tag: str) -> None:
    sub = out_dir / tag
    sub.mkdir(parents=True, exist_ok=True)
    save_delta_mem_adapter(model, str(sub), config)
    print(f"[train] checkpoint saved -> {sub}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000,
                    help="Number of optimiser steps (default 2000).")
    ap.add_argument("--context", type=int, default=32768,
                    help="Max sequence length in tokens (default 32768 = phase 1).")
    ap.add_argument("--data", type=Path, default=Path("data/longctx_mix_v1.jsonl"),
                    help="Pre-tokenised JSONL data file (from strix.prepare_data).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for checkpoints.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--clip", type=float, default=1.0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/ROCm not available", file=sys.stderr)
        return 1
    if not args.data.exists():
        print(f"data file missing: {args.data} — run "
              f"`python -m strix.prepare_data` first", file=sys.stderr)
        return 1

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] loading backbone {MODEL_ID} ...", flush=True)
    model_dir = snapshot_download(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")

    print(f"[train] attaching delta-mem adapter (init from {ADAPTER_ID}) ...",
          flush=True)
    delta_config = _attach_adapter(model)

    trainable_names = freeze_non_delta_mem_params(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable_params)
    print(f"[train] trainable: {len(trainable_names)} tensors / "
          f"{n_train / 1e6:.1f} M params", flush=True)

    # See CRITICAL block at top of file. Do not remove use_reentrant=True.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=GRAD_CHECKPOINT_KWARGS,
    )

    examples = _load_examples(args.data, args.context)
    if not examples:
        print(f"[train] no examples >= {args.context} tokens; "
              f"check data prep or lower --context", file=sys.stderr)
        return 1

    optim = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
        fused=True,
    )
    # Cosine schedule over the full step budget; warmup_steps applied first.
    scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup,
        num_training_steps=args.steps,
    )

    model.train()
    loss_log: List[dict] = []
    t0 = time.time()
    accum_loss = 0.0
    optim.zero_grad(set_to_none=True)
    print(f"[train] starting {args.steps} steps "
          f"(grad_accum={args.grad_accum}, context={args.context}, lr={args.lr})",
          flush=True)

    micro = 0
    step = 0
    while step < args.steps:
        ex_idx = micro % len(examples)
        input_ids = examples[ex_idx].unsqueeze(0).to("cuda")
        labels = input_ids.clone()
        torch.cuda.reset_peak_memory_stats()
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = out.loss / args.grad_accum
        loss.backward()
        accum_loss += out.loss.item()
        del out, input_ids, labels

        micro += 1
        if micro % args.grad_accum != 0:
            continue

        # An optimiser step worth of micro-batches has accumulated.
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.clip)
        optim.step()
        scheduler.step()
        optim.zero_grad(set_to_none=True)
        step += 1

        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        avg_loss = accum_loss / args.grad_accum
        accum_loss = 0.0

        if not math.isfinite(avg_loss):
            print(f"[train] step {step}: non-finite loss; aborting",
                  file=sys.stderr)
            return 2

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            sps = step / elapsed if elapsed > 0 else 0
            print(f"  step {step:>5}  loss={avg_loss:.4f}  lr={lr_now:.2e}  "
                  f"grad_norm={float(grad_norm):.3f}  peak={peak_gb:.2f} GB  "
                  f"({sps:.2f} step/s)", flush=True)
        loss_log.append({
            "step": step,
            "loss": avg_loss,
            "lr": scheduler.get_last_lr()[0],
            "grad_norm": float(grad_norm),
            "peak_gb": peak_gb,
        })

        if step % args.save_every == 0 and step < args.steps:
            _save_checkpoint(model, out_dir, delta_config, f"step_{step}")

    total_s = time.time() - t0
    print(f"[train] done in {total_s:.1f}s "
          f"({total_s / args.steps:.2f} s/step)", flush=True)

    _save_checkpoint(model, out_dir, delta_config, "final")
    (out_dir / "training_log.json").write_text(json.dumps({
        "steps": args.steps,
        "context": args.context,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_accum": args.grad_accum,
        "warmup": args.warmup,
        "wall_seconds": total_s,
        "trainable_params": int(n_train),
        "log": loss_log,
    }, indent=2))

    print(f"\n[train] next: copy {out_dir}/final back to local host and run "
          f"`python -m strix.verify_checkpoint --ckpt <local_path>`",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
