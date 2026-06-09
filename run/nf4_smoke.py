"""NF4 backbone-quantization wiring smoke (Option 4 of LONG_CONTEXT_PLAN.md).

Validates that bitsandbytes NF4 4-bit weight loading wires cleanly with
the delta-mem adapter. NOT a quality experiment — a single 256-token forward
pass with a finite-loss assert is the entire success criterion.

What this checks (in order):
1. bitsandbytes imports and CUDA is available.
2. Qwen3-4B-Instruct-2507 loads with `BitsAndBytesConfig(load_in_4bit=True,
   bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=bf16, double_quant=True)`.
3. The published delta-mem adapter attaches over the quantized backbone
   (QLoRA-style: adapter params stay bf16 on top of NF4 weights).
4. A single 256-token forward pass produces a finite loss.
5. Peak VRAM is reported.

Expected limitations / known caveats:

* **NF4 weights + OSCAR INT2 KV is untested.** The OSCAR rotations
  (data/oscar/rotations/instruct_gpqa/*) were calibrated against the bf16
  backbone's attention statistics. NF4 weight quantization perturbs those
  statistics, so the rotation may be slightly mis-aligned. The quality
  impact has NOT been measured here — that's a follow-up via
  `python -m run.locomo_eval --quantize-backbone-int4 ...` on a small slice.
* **Training over NF4 weights is QLoRA territory.** delta-mem's
  freeze_non_delta_mem_params flow has not been validated against a
  quantized backbone. Use this for inference only until proven otherwise.
* **Adapter overlay loads in bf16** on top of the NF4-quantized backbone;
  this matches the standard QLoRA inference pattern and is what the
  delta-mem adapter expects.

Wiring caveats discovered during this smoke (handled by monkeypatches both
here and in `run/_chunked_eval_runner.py` for the full-eval path):

1. `DeltaMemAttention._init_delta_head` slices `base.q_proj.weight` as a 2D
   matrix. Under bnb NF4 the weight is a packed `Params4bit` (flat 1D);
   we dequantize on read. The seeded slice is overwritten by the loaded
   adapter state anyway.
2. `attach_delta_mem` calls `.to(dtype=base.q_proj.weight.dtype)` on the
   wrapped attention; NF4 weight dtype is `torch.uint8` which `nn.Module.to`
   rejects. We coerce non-floating dtypes to bf16 (the compute dtype).
* **bitsandbytes on Windows**: prebuilt wheels ship for win_amd64 since
  v0.43+. If `pip install bitsandbytes` fails on this host (occasionally
  happens on older Python or with corporate proxies), the wheel cache trick
  is `pip install bitsandbytes --no-cache-dir`. If the import itself fails
  at runtime with a libcudart error, ensure the venv's torch matches the
  same CUDA major version (this host: torch 2.5.1+cu121, bnb 0.49.2 OK).
  We do NOT fight the installer here; if bnb is missing, this smoke fails
  fast with a clear message.

Usage:
    .venv\\Scripts\\python.exe -m run.nf4_smoke

Wall time: < 2 min on RTX 3060 12 GB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_ID = "declare-lab/delta-mem_qwen3_4b-instruct"
CONTEXT_TOKENS = 256


def _ok(msg: str) -> None:
    print(f"  PASS  {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}", flush=True)
    sys.exit(1)


def main() -> int:
    if not torch.cuda.is_available():
        _fail("CUDA not available")

    # Fail fast with a clear message if bnb isn't installed. We do NOT try to
    # auto-install from this script.
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        _fail(
            f"bitsandbytes not installed ({exc}). Install with: "
            r".venv\Scripts\python.exe -m pip install bitsandbytes"
        )
    print(f"[smoke] bitsandbytes {bnb.__version__} OK", flush=True)

    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    import deltamem.core.delta as _delta_mod
    from deltamem.core.delta import HFDeltaMemConfig
    from deltamem.core.delta_impl import (
        DeltaMemAttention as _DMA,
        load_delta_mem_state_dict,
    )

    # Wiring caveat #1: delta-mem's `_init_delta_head` reads `base.q_proj.weight`
    # to seed `delta_q_proj` from a slice of the base weight (output_init=
    # "base_slice"). With bnb NF4, the base weight is a packed `Params4bit`
    # tensor (flat shape [N_in*N_out/2]) — slicing it as a 2D matrix crashes
    # with a shape-mismatch RuntimeError inside DeltaMemAttention.__init__.
    # We monkeypatch the init helper to dequantize bnb 4-bit weights to bf16
    # before the slice read. The init result is moot in our case (we
    # immediately load the published adapter state dict over the heads), but
    # the constructor must not crash on the way through.
    from bitsandbytes.nn import Params4bit
    from bitsandbytes.functional import dequantize_4bit

    _orig_init_delta_head = _DMA._init_delta_head

    def _nf4_init_delta_head(self, head, base_weight):
        if isinstance(base_weight, Params4bit):
            base_weight = dequantize_4bit(
                base_weight.data, quant_state=base_weight.quant_state,
            ).to(torch.bfloat16)
        return _orig_init_delta_head(self, head, base_weight)

    _DMA._init_delta_head = _nf4_init_delta_head

    # Wiring caveat #2: `attach_delta_mem` calls `DeltaMemAttention(...).to(
    # dtype=module.q_proj.weight.dtype)` after wrapping. Under bnb NF4 the
    # base q_proj.weight dtype is `torch.uint8` (packed 4-bit), which
    # `nn.Module.to` rejects ("only accepts floating point or complex dtypes").
    # We wrap attach_delta_mem so the dtype falls back to bf16 (the compute
    # dtype) when the base weight is integer-typed — adapter params remain
    # bf16 over the NF4 backbone, matching QLoRA's inference pattern.
    _orig_attach_delta_mem = _delta_mod.attach_delta_mem

    def _nf4_attach_delta_mem(model, config):
        from deltamem.core.delta import (
            DeltaMemAttention, _get_parent_module, ensure_attention_compat_views,
            Qwen3Attention, SmolLM3Attention,
        )
        if config.memory_readout_mode != "delta":
            raise ValueError("only delta readout supported")
        replaced = []
        for name, module in list(model.named_modules()):
            if not isinstance(module, (Qwen3Attention, SmolLM3Attention)):
                continue
            if name.split(".")[-1] not in config.target_modules:
                continue
            if config.target_layers and module.layer_idx not in config.target_layers:
                continue
            module = ensure_attention_compat_views(module)
            parent, attr = _get_parent_module(model, name)
            base_dtype = module.q_proj.weight.dtype
            if not base_dtype.is_floating_point:
                base_dtype = torch.bfloat16
            wrapped = DeltaMemAttention(module, config).to(
                device=module.q_proj.weight.device, dtype=base_dtype,
            )
            setattr(parent, attr, wrapped)
            replaced.append(name)
        if not replaced:
            raise RuntimeError("No target modules were replaced")
        return replaced

    _delta_mod.attach_delta_mem = _nf4_attach_delta_mem
    attach_delta_mem = _nf4_attach_delta_mem

    print("[smoke] resolving model + adapter snapshots ...", flush=True)
    model_dir = snapshot_download(MODEL_ID)
    adapter_dir = snapshot_download(ADAPTER_ID)

    print(f"[smoke] loading backbone {MODEL_ID} with NF4 ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_cfg,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    )
    model.eval()

    weight_alloc = torch.cuda.memory_allocated() / 2**30
    _ok(f"NF4 backbone load: {weight_alloc:.2f} GB allocated (vs ~7.5 GB bf16)")

    # Sanity: at least one Linear4bit module should exist if NF4 wired up.
    from bitsandbytes.nn import Linear4bit
    n_4bit = sum(1 for m in model.modules() if isinstance(m, Linear4bit))
    if n_4bit == 0:
        _fail("no Linear4bit modules — quantization config did not take effect")
    _ok(f"Linear4bit module count: {n_4bit}")

    print(f"[smoke] attaching delta-mem from {adapter_dir} ...", flush=True)
    delta_config = HFDeltaMemConfig.from_pretrained(adapter_dir)
    attach_delta_mem(model, delta_config)
    adapter_state = torch.load(
        Path(adapter_dir) / "delta_mem_adapter.pt",
        map_location="cpu", weights_only=True,
    )
    load_delta_mem_state_dict(model, adapter_state)
    _ok(f"delta-mem adapter attached + loaded ({len(adapter_state)} tensors)")

    # Forward pass at CONTEXT_TOKENS with random ids + labels.
    torch.manual_seed(0)
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (1, CONTEXT_TOKENS), device="cuda", dtype=torch.long,
    )
    labels = input_ids.clone()

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
    loss = out.loss
    if not torch.isfinite(loss):
        _fail(f"loss non-finite: {loss.item()}")
    _ok(f"forward pass at {CONTEXT_TOKENS} tok: loss={loss.item():.4f} (finite)")

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"\n[smoke] ALL CHECKS PASSED — NF4 wiring is correct.\n"
        f"  context={CONTEXT_TOKENS} tok\n"
        f"  peak alloc={peak:.2f} GB\n"
        f"  loss={loss.item():.4f}\n"
        f"  next step: small-slice quality probe via\n"
        f"    python -m run.locomo_eval --quantize-backbone-int4 \\\n"
        f"      --max-conversations 1 --max-questions-per-conversation 5",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
