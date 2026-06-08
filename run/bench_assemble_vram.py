"""Measure peak VRAM during OSCARCache._assemble() at production-realistic
sizes. Run BEFORE and AFTER the fused-dequant + pre-alloc patch to confirm
the headroom savings, with shadow OFF (the regime that benefits).
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, "e:/Development/delta-mem-tests/third_party/oscar-transformers")
import os
os.environ["OSCAR_DISABLE_DEQUANT_SHADOW"] = "1"

import importlib
import oscar_transformers.cache as cache_mod
importlib.reload(cache_mod)
from oscar_transformers.cache import OSCARCache
from types import SimpleNamespace


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1

    NUM_LAYERS = 36
    HEADS_KV = 8
    HEAD_DIM = 128
    SINK = 64
    RECENT = 256
    TARGETS = [(17590, "17.5 k (conv-26)"), (25672, "25.7 k (conv-41)")]

    cfg = SimpleNamespace(num_hidden_layers=NUM_LAYERS)

    for total_T, label in TARGETS:
        cache = OSCARCache(config=cfg, sink_tokens=SINK, recent_tokens=RECENT)
        # Build one layer's state by streaming fake KV through update.
        device = torch.device("cuda")
        layer_idx = 0
        # First push the sink chunk
        k_sink = torch.randn(1, HEADS_KV, SINK, HEAD_DIM, device=device, dtype=torch.bfloat16)
        v_sink = torch.randn(1, HEADS_KV, SINK, HEAD_DIM, device=device, dtype=torch.bfloat16)
        cache.update(k_sink, v_sink, layer_idx)
        # Now push middle + recent in chunks
        remaining = total_T - SINK
        chunk = 2048
        while remaining > 0:
            n = min(chunk, remaining)
            k = torch.randn(1, HEADS_KV, n, HEAD_DIM, device=device, dtype=torch.bfloat16)
            v = torch.randn(1, HEADS_KV, n, HEAD_DIM, device=device, dtype=torch.bfloat16)
            cache.update(k, v, layer_idx)
            remaining -= n
            del k, v
        torch.cuda.empty_cache()

        # Now measure a SINGLE _assemble call's peak alloc
        torch.cuda.reset_peak_memory_stats()
        baseline_alloc = torch.cuda.memory_allocated()
        layer = cache.layers[layer_idx]
        out_k, out_v = layer._assemble()
        peak_alloc = torch.cuda.max_memory_allocated()
        out_k_bytes = out_k.numel() * out_k.element_size()
        out_v_bytes = out_v.numel() * out_v.element_size()

        # Account
        delta = peak_alloc - baseline_alloc
        print(f"=== {label} ===")
        print(f"  cache state before _assemble: {baseline_alloc / 2**20:.1f} MB")
        print(f"  peak alloc during _assemble : {peak_alloc / 2**20:.1f} MB")
        print(f"  delta (peak - before)       : {delta / 2**20:.1f} MB")
        print(f"  out_k + out_v size           : {(out_k_bytes + out_v_bytes) / 2**20:.1f} MB")
        print(f"  transient overhead beyond out: {(delta - (out_k_bytes + out_v_bytes)) / 2**20:.1f} MB")
        print(f"  for 36 layers (if all live)  : {36 * (out_k_bytes + out_v_bytes) / 2**20:.1f} MB")
        print()

        del cache, out_k, out_v
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())
