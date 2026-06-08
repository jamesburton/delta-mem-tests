"""Training-pipeline smoke test for delta-mem adapter fine-tuning.

Validates the code path end-to-end at a context length that fits on this
12 GB host. Designed as a *de-risking* harness for the Strix Halo training
run — if this passes locally, the same code on a 96 GB box should at least
start cleanly. If it fails, we want to find out before burning Strix hours.

Validates:

1. Backbone + adapter load cleanly and freeze-non-delta-mem-params marks
   only the expected parameters trainable.
2. A forward pass produces a finite loss against random labels.
3. backward() runs and gradients land *only* on adapter params (not the
   frozen backbone) and are finite.
4. An optimizer step actually changes the trainable parameters.
5. The adapter state can be saved and reloaded round-trip, recovering the
   identical state_dict.
6. A second forward pass produces a finite loss with the reloaded adapter.

Context length: 256 tokens (way smaller than inference target — this is a
*pipeline* smoke, not a quality experiment). At 256 tokens the whole
training stack fits in ~9 GB on a 12 GB card.

Usage:
    .venv\\Scripts\\python.exe -m run.training_smoke
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# delta-Mem APIs
from deltamem.core.delta import HFDeltaMemConfig, attach_delta_mem
from deltamem.core.delta_impl import (
    freeze_non_delta_mem_params,
    get_delta_mem_state_dict,
    load_delta_mem_adapter,
    save_delta_mem_adapter,
)

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_ID = "declare-lab/delta-mem_qwen3_4b-instruct"
CONTEXT_TOKENS = 256
LR = 1e-4

# Optional --probe mode (after the main smoke) sweeps context lengths under
# gradient checkpointing to find the largest local training context.
PROBE_CONTEXTS = (512, 1024, 2048, 4096, 8192)


def _ok(msg: str) -> None:
    print(f"  PASS  {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}", flush=True)
    sys.exit(1)


def main() -> int:
    if not torch.cuda.is_available():
        _fail("CUDA not available")

    print(f"[smoke] resolving model + adapter snapshots ...", flush=True)
    model_dir = snapshot_download(MODEL_ID)
    adapter_dir = snapshot_download(ADAPTER_ID)

    print(f"[smoke] loading backbone {MODEL_ID} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")

    print(f"[smoke] attaching delta-mem from {adapter_dir} ...", flush=True)
    # Order matters: load config -> attach wrappers -> load weights into the
    # now-wrapped modules. load_delta_mem_adapter assumes wrappers exist.
    from deltamem.core.delta_impl import load_delta_mem_state_dict
    delta_config = HFDeltaMemConfig.from_pretrained(adapter_dir)
    attach_delta_mem(model, delta_config)
    adapter_state = torch.load(
        Path(adapter_dir) / "delta_mem_adapter.pt",
        map_location="cpu", weights_only=True,
    )
    load_delta_mem_state_dict(model, adapter_state)

    # 1) Freeze backbone
    trainable_names = freeze_non_delta_mem_params(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if not trainable_names:
        _fail("freeze_non_delta_mem_params returned an empty trainable list")
    if n_trainable == 0:
        _fail("no trainable params after freeze")
    if n_frozen < 1_000_000_000:
        _fail(f"backbone frozen mass too small ({n_frozen/1e6:.1f} M) — expected ~4 B")
    _ok(
        f"freeze: {len(trainable_names)} trainable tensors "
        f"({n_trainable/1e6:.1f} M params), backbone {n_frozen/1e9:.2f} B frozen"
    )

    # 2) Forward pass with random labels
    torch.manual_seed(0)
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (1, CONTEXT_TOKENS), device="cuda", dtype=torch.long,
    )
    labels = input_ids.clone()

    model.train()
    out = model(input_ids=input_ids, labels=labels)
    loss = out.loss
    if not torch.isfinite(loss):
        _fail(f"loss non-finite: {loss.item()}")
    _ok(f"forward: loss={loss.item():.4f} (finite)")

    # 3) Backward + grad placement check
    loss.backward()
    grad_on_trainable = sum(
        1 for p in trainable_params if p.grad is not None and p.grad.abs().sum() > 0
    )
    grad_on_frozen = sum(
        1 for p in model.parameters()
        if not p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    if grad_on_trainable == 0:
        _fail("no gradient on any trainable parameter after backward")
    if grad_on_frozen > 0:
        _fail(f"unexpected gradient on {grad_on_frozen} frozen parameters")
    # Finiteness
    for p in trainable_params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            _fail("non-finite gradient on a trainable param")
    _ok(
        f"backward: grad on {grad_on_trainable}/{len(trainable_params)} trainable params, "
        f"zero on all frozen"
    )

    # 4) Optimizer step
    optim = torch.optim.AdamW(trainable_params, lr=LR)
    snapshot_before = {
        i: p.detach().clone() for i, p in enumerate(trainable_params)
        if p.grad is not None
    }
    optim.step()
    changed = sum(
        1 for i, p_before in snapshot_before.items()
        if not torch.allclose(p_before, trainable_params[i])
    )
    if changed == 0:
        _fail("optimizer step did not change any trainable parameter")
    _ok(f"optim: AdamW step changed {changed}/{len(snapshot_before)} sampled params")

    # 5) Save + reload adapter round-trip
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "adapter_ckpt"
        save_delta_mem_adapter(model, str(out_dir), delta_config)
        saved_files = sorted(p.name for p in out_dir.iterdir())
        if "delta_mem_adapter.pt" not in saved_files:
            _fail(f"missing adapter weights in saved ckpt; got {saved_files}")
        # Capture state pre-reload
        sd_before = get_delta_mem_state_dict(model)
        # Reload into the SAME model and verify identity
        from deltamem.core.delta_impl import load_delta_mem_state_dict
        reloaded = torch.load(
            out_dir / "delta_mem_adapter.pt", map_location="cpu", weights_only=True,
        )
        load_delta_mem_state_dict(model, reloaded)
        sd_after = get_delta_mem_state_dict(model)
        mismatches = [
            k for k in sd_before
            if not torch.allclose(sd_before[k], sd_after[k], atol=0, rtol=0)
        ]
        if mismatches:
            _fail(f"save/reload round-trip mismatch on {len(mismatches)} tensors")
    _ok(f"save+reload: round-trip bit-identical across {len(sd_before)} tensors")

    # 6) Second forward pass after reload
    optim.zero_grad(set_to_none=True)
    out2 = model(input_ids=input_ids, labels=labels)
    if not torch.isfinite(out2.loss):
        _fail(f"loss non-finite after reload: {out2.loss.item()}")
    _ok(f"forward after reload: loss={out2.loss.item():.4f}")

    print("\n[smoke] ALL CHECKS PASSED — training pipeline is wired correctly.", flush=True)
    print(
        f"  context={CONTEXT_TOKENS} tok, trainable={n_trainable/1e6:.1f} M params, "
        f"peak alloc={torch.cuda.max_memory_allocated()/2**30:.2f} GB",
        flush=True,
    )

    if "--probe" in sys.argv:
        # Free everything the main smoke held — out, out2, snapshot_before
        # otherwise hang on to scan-kernel saved tensors that conflict with
        # the probe's repeated forward+backward sweep.
        del out, out2, snapshot_before
        optim.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        _probe_max_context(model, optim, trainable_params, n_trainable)
    return 0


def _try_one_forward_backward(model, optim, T: int) -> tuple[bool, str]:
    """Run one (forward, backward) at context T. Return (ok, info)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        input_ids = torch.randint(
            0, model.config.vocab_size, (1, T), device="cuda", dtype=torch.long,
        )
        labels = input_ids.clone()
        optim.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        out.loss.backward()
        peak = torch.cuda.max_memory_allocated() / 2**30
        info = f"peak={peak:.2f} GB  loss={out.loss.item():.3f}"
        return True, info
    except torch.cuda.OutOfMemoryError:
        return False, "OOM (torch)"
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg:
            return False, f"OOM ({type(e).__name__})"
        return False, f"{type(e).__name__}: {str(e)[:100]}"


def _probe_max_context(model, optim, trainable_params, n_trainable) -> None:
    """Sweep context lengths to find the largest local training context.
    Tries three configurations:
      (a) no gradient checkpointing  — baseline, fits less
      (b) gradient checkpointing use_reentrant=False  — modern path
      (c) gradient checkpointing use_reentrant=True   — legacy path,
          sometimes the only one compatible with custom autograd functions
          (e.g. delta-mem's Triton scan kernel saves_for_backward).
    """
    print("\n[probe] sweeping context lengths in three configurations ...", flush=True)
    configs = [
        ("no checkpoint", lambda m: m.gradient_checkpointing_disable() if hasattr(m, "gradient_checkpointing_disable") else None),
        ("ckpt reentrant=False", lambda m: m.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})),
        ("ckpt reentrant=True ", lambda m: m.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True})),
    ]

    results: dict[str, int] = {}
    for label, setup in configs:
        print(f"\n  [{label}]", flush=True)
        try:
            setup(model)
        except Exception as e:
            print(f"    setup failed: {e}", flush=True)
            results[label] = 0
            continue
        last_ok = 0
        for T in PROBE_CONTEXTS:
            ok, info = _try_one_forward_backward(model, optim, T)
            mark = "OK " if ok else "FAIL"
            print(f"    T={T:>5} {mark}  {info}", flush=True)
            if ok:
                last_ok = T
            else:
                break
        results[label] = last_ok
        torch.cuda.empty_cache()

    print("\n[probe] summary — largest fitting context per config:", flush=True)
    for label, T in results.items():
        print(f"  {label}: {T} tokens" if T else f"  {label}: did not fit any probe size", flush=True)
    best = max(results.values()) if results else 0
    print(f"\n[probe] best feasible local-training context: {best} tokens", flush=True)


if __name__ == "__main__":
    sys.exit(main())
