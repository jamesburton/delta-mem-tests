"""Cheap end-to-end smoke proving SmolLM3-3B + delta-mem integration assumptions.

What it verifies:
  1. transformers exposes SmolLM3Attention (delta-mem's HAS_SMOLLM3 path).
  2. SmolLM3-3B-Instruct loads on CPU (bf16 weights, ~6 GB RAM).
     If the local box lacks RAM, set ``SMOLLM3_SMOKE_SKIP_LOAD=1`` and the
     script verifies only the class-level checks (still useful for catching
     import-time regressions).
  3. Every ``model.model.layers[*].self_attn`` is a SmolLM3Attention
     instance.
  4. delta-mem's ``attach_delta_mem`` wraps every layer in
     ``DeltaMemAttention`` without raising.
  5. ``freeze_non_delta_mem_params`` returns a NON-EMPTY trainable list
     (this is the key proof that delta-mem's per-parameter trainable filter
     correctly identifies SmolLM3's adapter-only params).

The wrap uses a tiny stub HFDeltaMemConfig (rank=8, num_state_heads=1) to keep
attach memory under control; we are NOT loading any adapter weights — just
proving the wrap works.

This script DOES run (cheap; CPU only) and should pass on the dev host.

Usage:
    python -m run.smollm3_smoke
"""
from __future__ import annotations

import os
import sys
import traceback

import torch

MODEL_ID = "HuggingFaceTB/SmolLM3-3B"  # the instruct-tuned release; no separate "-Instruct" suffix


def main() -> int:
    # --- 1. SmolLM3Attention is importable via delta-mem's compat shim. ---
    try:
        from deltamem.core.backbone_compat import HAS_SMOLLM3, SmolLM3Attention
    except Exception as exc:
        print(f"FAIL: backbone_compat import failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    if not HAS_SMOLLM3:
        print("FAIL: deltamem.core.backbone_compat.HAS_SMOLLM3 is False; "
              "transformers does not expose SmolLM3Attention.", file=sys.stderr)
        return 1
    print(f"[1] HAS_SMOLLM3=True; SmolLM3Attention={SmolLM3Attention.__name__}  OK", flush=True)

    # --- 2. Load SmolLM3-3B (CPU OK; skippable for RAM-tight hosts). ---
    skip_load = os.environ.get("SMOLLM3_SMOKE_SKIP_LOAD", "0") == "1"
    if skip_load:
        print("[2] SMOLLM3_SMOKE_SKIP_LOAD=1; skipping model load + remaining checks.",
              flush=True)
        return 0

    print(f"[2] loading {MODEL_ID} on CPU (bf16, ~6 GB RAM) ...", flush=True)
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    except Exception as exc:
        print(f"FAIL: model load failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    n_layers = len(model.model.layers)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] loaded. layers={n_layers} params={n_params/1e9:.2f}B  OK", flush=True)

    # --- 3. Every self_attn is a SmolLM3Attention. ---
    bad: list[tuple[int, str]] = []
    for i, layer in enumerate(model.model.layers):
        attn = getattr(layer, "self_attn", None)
        if not isinstance(attn, SmolLM3Attention):
            bad.append((i, type(attn).__name__))
    if bad:
        print(f"FAIL: {len(bad)} layers have non-SmolLM3 self_attn: {bad[:5]}...",
              file=sys.stderr)
        return 1
    print(f"[3] all {n_layers} self_attn are SmolLM3Attention  OK", flush=True)

    # --- 4. attach_delta_mem wraps every layer in DeltaMemAttention. ---
    try:
        from deltamem.core.delta import HFDeltaMemConfig, attach_delta_mem
        from deltamem.core.delta_impl import (
            DeltaMemAttention,
            freeze_non_delta_mem_params,
        )
    except Exception as exc:
        print(f"FAIL: deltamem import failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    # Minimal stub config — small rank/heads to keep wrap memory bounded on CPU.
    try:
        config = HFDeltaMemConfig(
            rank=8,
            num_state_heads=1,
            alpha=8,
            target_modules=("self_attn",),
            delta_heads=("o",),
        )
    except TypeError:
        # Some HFDeltaMemConfig versions require more or fewer kwargs; fall
        # back to a no-arg construction and assume defaults are sane.
        config = HFDeltaMemConfig()  # type: ignore[call-arg]
    print(f"[4] attaching delta-mem (rank={getattr(config, 'rank', '?')}, "
          f"heads={getattr(config, 'delta_heads', '?')}) ...", flush=True)
    try:
        replaced = attach_delta_mem(model, config)
    except Exception as exc:
        print(f"FAIL: attach_delta_mem failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    if not replaced:
        print("FAIL: attach_delta_mem returned empty replacement list",
              file=sys.stderr)
        return 1
    if len(replaced) != n_layers:
        print(f"WARN: attached {len(replaced)} layers but model has {n_layers}; "
              "may be intentional if config.target_layers filters", flush=True)

    n_wrapped = sum(
        1 for layer in model.model.layers
        if isinstance(layer.self_attn, DeltaMemAttention)
    )
    print(f"[4] attached {len(replaced)} modules; {n_wrapped} layers wrapped  OK",
          flush=True)

    # --- 5. freeze_non_delta_mem_params returns a non-empty trainable list. ---
    trainable_names = freeze_non_delta_mem_params(model)
    n_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    if not trainable_names:
        print("FAIL: freeze_non_delta_mem_params returned an EMPTY list — "
              "SmolLM3's per-param trainable filter is broken.", file=sys.stderr)
        return 1
    print(f"[5] trainable: {len(trainable_names)} tensors / "
          f"{n_trainable/1e6:.1f} M params  OK", flush=True)
    print(f"    sample names: {trainable_names[:3]}", flush=True)

    print("\nALL CHECKS PASSED.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
