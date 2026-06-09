"""Three-phase OSCAR rotation calibration for SmolLM3-3B-Instruct.

Mirrors the Qwen3 recipe from ``report/tier1-summary.md`` Appendix D, adapted
for SmolLM3:

  Phase A — Dump post-RoPE Q/K/V activations on a GPQA prompt batch.
            Output: ``data/oscar/qkv_dumps/smollm3_gpqa/layer_<i>/{q,k,v}/<n>.pt``
  Phase B — Compute K and V rotations from those dumps via the existing
            ``run.compute_kv_rotation`` machinery.
            Output: ``data/oscar/rotations/smollm3_gpqa/{k,v}_rotation_*.pt``
  Phase C — Smoke: load SmolLM3, apply rotations, run a port-debug-F-equivalent
            test (GPQA-cal rotation + OSCARCache INT4 at 4 k context;
            expect needle-match against bf16 baseline).

DO NOT run this on the laptop without intent — phase A dumps 36 layers x N
prompts x ~4 k tokens of bf16 activations (~few GB of disk and ~30-60 min GPU
time on this card; ~10 min on Strix). Phase B is CPU-bound and adds ~20 min.

Architectural notes (open questions captured in run/SMOLLM3_NOTES.md):

1. SmolLM3Attention has NO ``q_norm`` / ``k_norm`` and applies RoPE
   conditionally (``self.use_rope`` is per-layer; some layers skip RoPE).
   OSCAR's ``apply_rotations`` in ``third_party/oscar-transformers`` currently
   hard-codes a Qwen3-style ``_build_patched_forward`` that calls
   ``self.q_norm(...)`` and unconditionally applies RoPE. That patch will
   raise ``AttributeError: 'SmolLM3Attention' object has no attribute
   'q_norm'`` on layer-0 forward. **A SmolLM3-aware patched_forward must be
   added to oscar_transformers before phase C can succeed.** The Phase C
   smoke below intentionally fails-fast on that AttributeError and prints a
   clear diagnostic so the calibration owner knows to extend the port.

2. SmolLM3-3B uses 36 layers x num_attention_heads / num_key_value_heads
   GQA layout. Run with ``--num-layers 36`` (auto-inferred from the model
   config when phase A loads the model).

3. Per-layer ``use_rope=False`` means the post-RoPE dump for those layers is
   actually post-projection raw Q/K. That is correct for OSCAR's covariance
   estimation (the cache stores whatever K/V went in), but the rotation
   chosen for those layers is data-derived from raw K/V instead of post-RoPE
   K/V. The runtime patched_forward must respect ``self.use_rope`` for the
   same layers, or the rotation basis at runtime won't match the basis the
   covariance was computed in.

Usage::

    # All three phases:
    python -m run.oscar_calibrate_smollm3 --phase all --num-prompts 32

    # Individual phases (resumable):
    python -m run.oscar_calibrate_smollm3 --phase A --num-prompts 32
    python -m run.oscar_calibrate_smollm3 --phase B
    python -m run.oscar_calibrate_smollm3 --phase C

Outputs (absolute paths printed at end of each phase):

    data/oscar/qkv_dumps/smollm3_gpqa/             (phase A)
    data/oscar/rotations/smollm3_gpqa/             (phase B)
    report/raw/smollm3_oscar_phaseC_smoke.log      (phase C)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"  # the canonical instruct-tuned release; no separate "-Instruct" suffix

DUMP_DIR = REPO_ROOT / "data" / "oscar" / "qkv_dumps" / "smollm3_gpqa"
ROT_DIR = REPO_ROOT / "data" / "oscar" / "rotations" / "smollm3_gpqa"
SMOKE_LOG = REPO_ROOT / "report" / "raw" / "smollm3_oscar_phaseC_smoke.log"

# Calibration knobs (mirror Qwen3's GPQA setup; see run/oscar_dump_qkv.py).
DEFAULT_NUM_PROMPTS = 32           # 32 is the Qwen3 default; 198 = full GPQA-Diamond.
DEFAULT_MAX_PROMPT_TOKENS = 4096   # cap per prompt to bound VRAM.
DEFAULT_METHOD = "qqt_sst"         # produces k_rotation_qqt_* and v_rotation_sst_*.
DEFAULT_COMPOSITION = "r_h_pbr"    # validated-best per Appendix D.

# Phase C smoke knobs (mirror oscar_port_debug.py test F shape).
SMOKE_CONTEXT_TOKENS = 4000
SMOKE_NEEDLE_OFFSET = 1500
SMOKE_MAX_NEW = 80


# ---------------------------------------------------------------------------
# Phase A — dump post-RoPE Q/K/V
# ---------------------------------------------------------------------------


def _patch_smollm3_attention_for_dump(dump_dir: Path) -> tuple[Callable[[int], None], Callable[[], None]]:
    """Monkey-patch ``SmolLM3Attention.forward`` to dump post-RoPE q/k/v.

    Mirrors ``run.oscar_dump_qkv._patch_qwen3_attention_for_dump`` but adapted
    for SmolLM3:
      * No q_norm/k_norm — projections feed straight into the transpose.
      * Per-layer ``self.use_rope`` controls whether RoPE is applied. The
        dump captures the same tensor that the runtime patched_forward will
        see (so identity-on-use_rope-False matches identity-on-use_rope-True
        — what matters is covariance basis consistency at runtime).
    """
    from transformers.models.smollm3.modeling_smollm3 import (
        SmolLM3Attention,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original_forward = SmolLM3Attention.forward
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

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.use_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        b, _h, t, d = query_states.shape
        if b != 1:
            raise RuntimeError(
                f"Dumper assumes batch_size=1; got B={b}. Run one prompt at a time."
            )
        layer_id = self.layer_idx
        chunk_id = state["chunk_id"]
        for tag, tensor in (("q", query_states), ("k", key_states), ("v", value_states)):
            out_dir = dump_dir / f"layer_{layer_id}" / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            # (B, H, T, D) -> (T, H, D), keep bf16, on CPU.
            t_out = tensor[0].transpose(0, 1).contiguous().to(torch.bfloat16).cpu()
            torch.save(t_out, out_dir / f"{chunk_id}.pt")

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
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    SmolLM3Attention.forward = patched_forward

    def unpatch() -> None:
        SmolLM3Attention.forward = original_forward

    return set_chunk_id, unpatch


def _build_gpqa_prompts(num_prompts: int) -> list[str]:
    """Build OSCAR-style GPQA calibration prompts. Mirrors
    ``run.oscar_dump_qkv._load_gpqa_prompts``.
    """
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    prompts: list[str] = []
    from itertools import permutations

    for ex in list(ds)[:num_prompts]:
        q = ex["Question"]
        choices = [
            ex["Correct Answer"],
            ex["Incorrect Answer 1"],
            ex["Incorrect Answer 2"],
            ex["Incorrect Answer 3"],
        ]
        h = hash(q) % 24
        order = list(permutations(range(4)))[h]
        shuffled = [choices[i] for i in order]
        choice_block = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(shuffled))
        prompts.append(f"Question: {q}\n\nOptions:\n{choice_block}\n\nAnswer:")
    return prompts


def phase_a_dump(num_prompts: int, max_prompt_tokens: int) -> int:
    """Phase A — dump post-RoPE Q/K/V activations for SmolLM3.

    Aborts cleanly if model/tokenizer load fails, dataset gate is not accepted,
    or no prompts were built.
    """
    print(f"\n=== Phase A: dump Q/K/V activations on {MODEL_ID} ===", flush=True)
    print(f"  output: {DUMP_DIR}", flush=True)
    DUMP_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("[A] CUDA unavailable; phase A will be VERY slow on CPU.", file=sys.stderr)
        # Don't abort — Strix Halo path may run via different device shimming.

    print(f"[A] loading tokenizer + GPQA prompts (num_prompts={num_prompts}) ...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as exc:
        print(f"[A] FAIL: tokenizer load failed: {exc}", file=sys.stderr)
        return 1
    try:
        prompts = _build_gpqa_prompts(num_prompts)
    except Exception as exc:
        print(
            f"[A] FAIL: GPQA prompt build failed (gated dataset? "
            f"accept the licence at huggingface.co/datasets/Idavidrein/gpqa): {exc}",
            file=sys.stderr,
        )
        return 1
    if not prompts:
        print("[A] FAIL: no prompts built; aborting", file=sys.stderr)
        return 1
    print(f"[A] built {len(prompts)} prompts", flush=True)

    print(f"[A] loading {MODEL_ID} (bf16) ...", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    except Exception as exc:
        print(f"[A] FAIL: model load failed: {exc}", file=sys.stderr)
        return 1
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    set_chunk_id, unpatch = _patch_smollm3_attention_for_dump(DUMP_DIR)
    total_tokens = 0
    t0 = time.time()
    try:
        for i, prompt in enumerate(prompts):
            set_chunk_id(i + 1)  # skip chunk 0 to mirror upstream warmup convention
            ids = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=max_prompt_tokens,
            ).input_ids.to(model.device)
            with torch.no_grad():
                _ = model(input_ids=ids, use_cache=False)
            n = ids.shape[1]
            total_tokens += n
            elapsed = time.time() - t0
            print(
                f"[A] prompt {i+1}/{len(prompts)} tokens={n} "
                f"total={total_tokens} elapsed={elapsed:.1f}s",
                flush=True,
            )
    finally:
        unpatch()

    meta = {
        "model_id": MODEL_ID,
        "num_prompts": len(prompts),
        "total_tokens": total_tokens,
        "max_prompt_tokens": max_prompt_tokens,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (DUMP_DIR / "dump_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[A] OK. total_tokens={total_tokens} meta={DUMP_DIR/'dump_meta.json'}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Phase B — compute K and V rotations
# ---------------------------------------------------------------------------


def phase_b_compute_rotations(method: str, composition: str, head_dim: int) -> int:
    """Phase B — invoke run.compute_kv_rotation on the dump dir.

    Defaults match the Qwen3 production combo: method=qqt_sst,
    composition=r_h_pbr. This produces ``k_rotation_qqt_r_h_pbr.pt`` and
    ``v_rotation_sst_r_h_pbr.pt`` under ROT_DIR.
    """
    print(f"\n=== Phase B: compute K/V rotations ({method}, {composition}) ===", flush=True)
    if not DUMP_DIR.exists() or not any(DUMP_DIR.iterdir()):
        print(f"[B] FAIL: dump dir empty/missing: {DUMP_DIR}", file=sys.stderr)
        return 1
    ROT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  output: {ROT_DIR}", flush=True)

    cmd = [
        sys.executable, "-m", "run.compute_kv_rotation",
        "--dump-path", str(DUMP_DIR),
        "--output-dir", str(ROT_DIR),
        "--method", method,
        "--composition", composition,
        "--head-dim", str(head_dim),
    ]
    print(f"[B] running: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"[B] FAIL: compute_kv_rotation rc={proc.returncode} elapsed={elapsed:.1f}s",
              file=sys.stderr)
        return proc.returncode

    # Verify the expected files landed.
    expected_k = ROT_DIR / f"k_rotation_qqt_{composition}.pt"
    expected_v = ROT_DIR / f"v_rotation_sst_{composition}.pt"
    missing = [p for p in (expected_k, expected_v) if not p.exists()]
    if missing:
        print(
            f"[B] FAIL: expected outputs missing: {[str(p) for p in missing]}",
            file=sys.stderr,
        )
        return 1

    print(f"[B] OK ({elapsed:.1f}s). K rotation: {expected_k}", flush=True)
    print(f"[B]                       V rotation: {expected_v}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Phase C — smoke: apply rotations + INT4 OSCARCache at 4k
# ---------------------------------------------------------------------------


def phase_c_smoke(composition: str) -> int:
    """Phase C — apply rotations to SmolLM3 + run a port-debug-F-equivalent
    needle-recall test (INT4 OSCARCache at 4k context).

    IMPORTANT: as of the prep date, OSCAR's ``apply_rotations``
    ``_build_patched_forward`` in ``third_party/oscar-transformers/oscar_transformers/rotation.py``
    is hard-coded for Qwen3 attention (calls ``self.q_norm(...)`` and
    unconditionally applies RoPE). SmolLM3 has no q_norm/k_norm and uses
    per-layer ``self.use_rope``. This smoke will raise
    ``AttributeError: 'SmolLM3Attention' object has no attribute 'q_norm'``
    on layer-0 forward. The error is caught here and a clear diagnostic is
    printed; extending oscar_transformers to support SmolLM3 is the
    prerequisite tracked in run/SMOLLM3_NOTES.md.
    """
    print(f"\n=== Phase C: rotation+OSCARCache(INT4) smoke at {SMOKE_CONTEXT_TOKENS} tok ===", flush=True)

    k_path = ROT_DIR / f"k_rotation_qqt_{composition}.pt"
    v_path = ROT_DIR / f"v_rotation_sst_{composition}.pt"
    if not k_path.exists() or not v_path.exists():
        print(
            f"[C] FAIL: rotation files missing under {ROT_DIR} (run phase B first).",
            file=sys.stderr,
        )
        return 1

    SMOKE_LOG.parent.mkdir(parents=True, exist_ok=True)

    try:
        from oscar_transformers import OSCARCache, apply_rotations, load_rotation_file
    except ImportError as exc:
        print(f"[C] FAIL: oscar_transformers import failed: {exc}", file=sys.stderr)
        return 1

    # Reuse the needle prompt builder from oscar_smoke_middle. The needle text
    # is generic ("QM-7194-ZULU") so it works regardless of backbone.
    from run.oscar_smoke_middle import (
        EXPECTED_FRAGMENT,
        QUESTION,
        build_context,
    )

    print(f"[C] loading {MODEL_ID} + tokenizer ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    context, _ = build_context(
        tokenizer,
        target_tokens=SMOKE_CONTEXT_TOKENS,
        needle_offset_tokens=SMOKE_NEEDLE_OFFSET,
    )
    prompt = f"{context}\n\nQuestion: {QUESTION}"

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    # Baseline first (no rotation, bf16 cache) — gives us the reference text.
    print("[C] baseline (no rotation, bf16 cache) ...", flush=True)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=SMOKE_MAX_NEW,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    baseline_pred = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    baseline_ok = EXPECTED_FRAGMENT in baseline_pred.upper()
    print(f"[C]   baseline needle={'YES' if baseline_ok else 'NO'} pred={baseline_pred!r}",
          flush=True)

    # Now apply OSCAR rotations and rerun with INT4 OSCARCache.
    print(f"[C] apply_rotations from {k_path}, {v_path} ...", flush=True)
    try:
        k_rot = load_rotation_file(k_path)
        v_rot = load_rotation_file(v_path)
        apply_rotations(model, k_rotations=k_rot, v_rotations=v_rot)
    except AttributeError as exc:
        msg = str(exc)
        if "q_norm" in msg or "k_norm" in msg:
            print(
                "[C] EXPECTED FAILURE: OSCAR port assumes Qwen3 attention "
                "with q_norm/k_norm; SmolLM3 has neither.\n"
                "    Resolution: extend third_party/oscar-transformers's "
                "_build_patched_forward to detect SmolLM3Attention and emit "
                "a SmolLM3-shaped patched forward (no q_norm/k_norm, honour "
                "self.use_rope per layer). See run/SMOLLM3_NOTES.md.",
                file=sys.stderr,
            )
            return 2  # 2 = expected-port-gap, distinguishable from infra failure
        raise

    cache = OSCARCache(config=model.config, bits=4)
    print("[C] generating with OSCARCache(INT4) ...", flush=True)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=SMOKE_MAX_NEW,
            do_sample=False,
            past_key_values=cache,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    pred = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    needle_ok = EXPECTED_FRAGMENT in pred.upper()
    matches_baseline = pred == baseline_pred

    print(f"[C] OSCAR INT4 needle={'YES' if needle_ok else 'NO'} "
          f"matches_baseline={'YES' if matches_baseline else 'NO'} "
          f"elapsed={elapsed:.1f}s", flush=True)
    print(f"[C]   pred={pred!r}", flush=True)

    SMOKE_LOG.write_text(
        json.dumps({
            "model": MODEL_ID,
            "context_tokens": SMOKE_CONTEXT_TOKENS,
            "needle_offset": SMOKE_NEEDLE_OFFSET,
            "baseline_pred": baseline_pred,
            "baseline_needle": baseline_ok,
            "oscar_int4_pred": pred,
            "oscar_int4_needle": needle_ok,
            "oscar_int4_matches_baseline": matches_baseline,
            "elapsed_sec": round(elapsed, 1),
            "k_rotation": str(k_path),
            "v_rotation": str(v_path),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[C] log: {SMOKE_LOG}", flush=True)

    if not needle_ok and baseline_ok:
        print("[C] FAIL: baseline found needle but OSCAR INT4 did not — "
              "investigate rotation correctness.", file=sys.stderr)
        return 1
    print("[C] OK.", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS,
                    help="Phase A: number of GPQA prompts to dump.")
    ap.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS,
                    help="Phase A: per-prompt token cap.")
    ap.add_argument("--method", default=DEFAULT_METHOD,
                    help="Phase B: hessian method (see run.compute_kv_rotation).")
    ap.add_argument("--composition", default=DEFAULT_COMPOSITION,
                    help="Phase B: rotation composition.")
    ap.add_argument("--head-dim", type=int, default=128,
                    help="Phase B: SmolLM3-3B head_dim. Override if a variant uses a different size.")
    args = ap.parse_args()

    if args.phase in ("A", "all"):
        rc = phase_a_dump(args.num_prompts, args.max_prompt_tokens)
        if rc != 0:
            return rc
    if args.phase in ("B", "all"):
        rc = phase_b_compute_rotations(args.method, args.composition, args.head_dim)
        if rc != 0:
            return rc
    if args.phase in ("C", "all"):
        rc = phase_c_smoke(args.composition)
        if rc != 0:
            return rc
    print("\n[done] all requested phases completed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
