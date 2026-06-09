"""Phase-1 delta-mem adapter training for SmolLM3-3B (Strix Halo target).

This is a Qwen3-variant of ``run/local_lora_train.py`` adapted for SmolLM3:

  * Backbone: ``HuggingFaceTB/SmolLM3-3B`` (the instruct-tuned release; no
    separate -Instruct repo)
  * No starter adapter — SmolLM3 has no published delta-mem checkpoint, so we
    initialise the adapter from random (HFDeltaMemConfig + attach_delta_mem
    creates the wrappers; their non-base parameters are at their constructor
    defaults).
  * 32k context window, 1000 steps as Phase-1 hyperparams.
  * OSCAR rotation paths point at SmolLM3-specific calibration outputs
    produced by ``run.oscar_calibrate_smollm3`` (NOT the Qwen3 rotations —
    rotations are basis-aligned to a model's specific Q/K/V distribution).

DO NOT run on the 12 GB dev box — 32k training on a 3 B param model with
bf16 weights + reentrant gradient checkpointing needs ~30 GB. This script
is staged for Strix Halo (96 GB) or a rented H100.

Strix Halo run command (copy-paste, adjusting paths)::

    python -m strix.train_smollm3_phase1 \
        --steps 1000 \
        --context 32768 \
        --out /workspace/checkpoints/smollm3_phase1_v0/ \
        --lr 5e-5 \
        --oscar-k-rotation /workspace/data/oscar/rotations/smollm3_gpqa/k_rotation_qqt_r_h_pbr.pt \
        --oscar-v-rotation /workspace/data/oscar/rotations/smollm3_gpqa/v_rotation_sst_r_h_pbr.pt

Verification (back on the 12 GB box, after copying the checkpoint local)::

    $env:KV_CACHE_BACKEND='oscar'; $env:KV_CACHE_BITS='2'
    $env:OSCAR_K_ROTATION_PATH='data\oscar\rotations\smollm3_gpqa\k_rotation_qqt_r_h_pbr.pt'
    $env:OSCAR_V_ROTATION_PATH='data\oscar\rotations\smollm3_gpqa\v_rotation_sst_r_h_pbr.pt'
    python -m run.locomo_eval `
        --model-override HuggingFaceTB/SmolLM3-3B `
        --adapter-override checkpoints\smollm3_phase1_v0 `
        --kv-cache-backend oscar --kv-cache-bits 2 `
        --max-conversations 1 --max-questions-per-conversation 10 `
        --output-json outputs\smollm3_phase1_v0_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
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
    save_delta_mem_adapter,
)

MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
LOCOMO_PATH = Path("data/locomo10.json")
DEFAULT_CONTEXT_TOKENS = 32768
DEFAULT_STEPS = 1000
DEFAULT_LR = 5e-5  # higher than Qwen3 fine-tune (1e-5) because we're training from random init, not warm-starting.
LOG_EVERY = 25


def _format_session_text(session_messages) -> str:
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
    tokenizer, target_tokens: int, max_chunks: int = 256,
) -> List[torch.Tensor]:
    """Read LoCoMo, concatenate session texts per conversation, slice into
    ``target_tokens``-sized chunks. Each chunk is one training example.

    With 32k context the per-conv yields fewer chunks than at 2k, so we raise
    max_chunks to 256 to keep the pool meaningfully diverse over 1000 steps.
    Add LongMemEval / InfBench-mem data mixing later in Phase 2; Phase 1 is
    LoCoMo-only to anchor the smoke and avoid a multi-source data prep burden.
    """
    raw = json.loads(LOCOMO_PATH.read_text(encoding="utf-8"))
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
        for start in range(0, len(ids) - target_tokens + 1, target_tokens):
            chunk = ids[start:start + target_tokens]
            chunks.append(chunk)
            if len(chunks) >= max_chunks:
                return chunks
    return chunks


def _attach_fresh_adapter(model) -> HFDeltaMemConfig:
    """Construct a default HFDeltaMemConfig and wrap the model.

    No published SmolLM3 adapter to load — params live at their nn.Linear
    constructor defaults until training nudges them.
    """
    config = HFDeltaMemConfig()  # all defaults; tweak rank/heads here if Phase 2 needs them
    attach_delta_mem(model, config)
    return config


def _wire_oscar_rotations(
    model,
    k_rotation_path: Path,
    v_rotation_path: Path,
) -> None:
    """Apply OSCAR K/V rotations to the model before training so the trained
    adapter compensates for the rotation+quantization basis (matches inference
    pipeline). If the rotation files are missing, training proceeds without
    them and the warning is logged.
    """
    if not k_rotation_path.exists() or not v_rotation_path.exists():
        print(
            f"[train] WARN: OSCAR rotations not found "
            f"(k={k_rotation_path}, v={v_rotation_path}); "
            "training without rotations — adapter won't be inference-pipeline-aligned. "
            "Re-run after run.oscar_calibrate_smollm3 produces the rotations.",
            file=sys.stderr,
        )
        return
    try:
        from oscar_transformers import apply_rotations, load_rotation_file
    except ImportError as exc:
        print(f"[train] WARN: oscar_transformers not importable: {exc}; skipping rotations.",
              file=sys.stderr)
        return
    print(f"[train] applying OSCAR rotations:\n  K={k_rotation_path}\n  V={v_rotation_path}",
          flush=True)
    apply_rotations(
        model,
        k_rotations=load_rotation_file(k_rotation_path),
        v_rotations=load_rotation_file(v_rotation_path),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--context", type=int, default=DEFAULT_CONTEXT_TOKENS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--oscar-k-rotation", type=Path,
                    default=Path("data/oscar/rotations/smollm3_gpqa/k_rotation_qqt_r_h_pbr.pt"))
    ap.add_argument("--oscar-v-rotation", type=Path,
                    default=Path("data/oscar/rotations/smollm3_gpqa/v_rotation_sst_r_h_pbr.pt"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this is a Strix Halo / H100 script.", file=sys.stderr)
        return 1

    out_dir = Path(args.out).resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[train] loading backbone {MODEL_ID} ...", flush=True)
    model_dir = snapshot_download(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")

    print(f"[train] attaching fresh delta-mem adapter (defaults) ...", flush=True)
    delta_config = _attach_fresh_adapter(model)

    _wire_oscar_rotations(model, args.oscar_k_rotation, args.oscar_v_rotation)

    trainable_names = freeze_non_delta_mem_params(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable_params)
    print(f"  trainable: {len(trainable_names)} tensors / {n_train/1e6:.1f} M params",
          flush=True)
    if not trainable_names:
        print("[train] FAIL: no trainable params after freeze; "
              "check that SmolLM3Attention got wrapped.", file=sys.stderr)
        return 1

    # Reentrant required for the affine_scan kernel per local_lora_train.py notes.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True}
    )

    print(f"[train] building training chunks at {args.context} tokens ...", flush=True)
    chunks = _build_training_chunks(tokenizer, target_tokens=args.context)
    print(f"  built {len(chunks)} chunks", flush=True)
    if not chunks:
        print("[train] no chunks; aborting (LOCOMO_PATH or context too large)",
              file=sys.stderr)
        return 1

    optim = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    model.train()

    loss_log: List[tuple[int, float, float]] = []
    t0 = time.time()
    print(f"[train] training {args.steps} steps, lr={args.lr}, ctx={args.context}, "
          f"pool={len(chunks)}", flush=True)
    for step in range(1, args.steps + 1):
        chunk = chunks[(step - 1) % len(chunks)]
        input_ids = chunk.unsqueeze(0).to("cuda")
        labels = input_ids.clone()
        torch.cuda.reset_peak_memory_stats()
        optim.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optim.step()
        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        loss_val = out.loss.item()
        if not (loss_val == loss_val):  # NaN
            print(f"  step {step}: NaN loss; aborting", flush=True)
            return 2
        if step == 1 or step % LOG_EVERY == 0 or step == args.steps:
            elapsed = time.time() - t0
            print(f"  step {step:>4}  loss={loss_val:.4f}  peak={peak_gb:.2f} GB  "
                  f"elapsed={elapsed:.1f}s", flush=True)
        loss_log.append((step, loss_val, peak_gb))
        del input_ids, labels, out

    total_s = time.time() - t0
    print(f"[train] done in {total_s:.1f}s ({total_s/args.steps:.1f} s/step)",
          flush=True)

    losses = [v for _, v, _ in loss_log]
    early = sum(losses[:10]) / 10
    late = sum(losses[-10:]) / 10
    drop = early - late
    print(f"[train] loss: early={early:.4f} late={late:.4f} drop={drop:+.4f}",
          flush=True)

    print(f"[train] saving adapter to {out_dir} ...", flush=True)
    save_delta_mem_adapter(model, str(out_dir), delta_config)
    (out_dir / "loss_log.json").write_text(json.dumps({
        "model_id": MODEL_ID,
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
        "oscar_k_rotation": str(args.oscar_k_rotation),
        "oscar_v_rotation": str(args.oscar_v_rotation),
    }, indent=2))
    print(f"\n[train] Evaluate via:\n"
          f"  python -m run.locomo_eval --model-override {MODEL_ID} \\\n"
          f"      --adapter-override {out_dir} \\\n"
          f"      --kv-cache-backend oscar --kv-cache-bits 2 \\\n"
          f"      --max-conversations 1 --max-questions-per-conversation 10 \\\n"
          f"      --output-json outputs/smollm3_phase1_v0_eval.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
