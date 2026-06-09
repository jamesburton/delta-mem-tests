"""Qwen3-specific monkeypatch adapter for EpiCache.

Day-1 status (2026-06-09): port from EpiCache's Qwen2.5 monkeypatch.
EpiCache upstream (commit b742661) does NOT include a Qwen3 branch in
`third_party/ml-epicache/model/monkeypatch.py` (only `llama`, `qwen2.5`,
`mistral`), even though `third_party/ml-epicache/model/wrapper.py:8` imports
`Qwen3ForCausalLM`. Per the alternatives doc, Qwen3Attention is structurally
identical to Qwen2Attention so the same `llama_flash_attn2_forward` should
attach cleanly — the only architectural delta is the per-head
`q_norm` / `k_norm` RMSNorm layers Qwen3 applies between projection and rotary
embedding, which EpiCache's forward does NOT account for.

Open question (TODO #1): does the missing q_norm/k_norm step cause measurable
attention-score corruption in EpiCache's KV-importance scoring path? The
scoring is computed on `query_states` / `key_states` AFTER `apply_rotary_pos_emb`
but BEFORE Qwen3's `q_norm` / `k_norm` would normally run. There are two
plausible interpretations:

  (a) EpiCache's `llama_flash_attn2_forward` SKIPS q_norm/k_norm, scoring
      against unnormalised states. This is what the trivial port does and
      may be the safer choice — scoring is a relative ranking, so any
      monotone transform of K shouldn't change which tokens score highest.
      But Qwen3's norms have learned per-head weights that emphasise some
      head dims over others, so attention scores in the original Qwen3 model
      ARE computed in a different space than what EpiCache will score
      against. The drift is bounded by the norm's per-head learned weights
      (~unit variance for a healthy checkpoint) but is non-zero.

  (b) Rewrite EpiCache's forward to apply self.q_norm / self.k_norm before
      RoPE (matching Qwen3's original forward). Cleaner but requires
      sub-classing the forward, not just reassigning it. ~30 LOC for a
      Qwen3-specific forward variant.

The Day-1 port uses (a): simple monkeypatch reassignment. Validation of the
quality impact is part of the Day-2 smoke (calibration + conv-26 EpiCache-only
baseline).

Open question (TODO #2): EpiCache's `attention/attn.py:13` hard-imports
`from flash_attn import flash_attn_varlen_func`. On this Windows host we don't
have flash-attn built. The Day-1 adapter therefore CANNOT actually import
attention.attn at module load — we defer all attention.attn / attention.kvcache
imports until `install_epicache_on_qwen3` is invoked, so that the CPU smoke
test can import this module without crashing. The smoke validates wiring
shape only; live install requires GPU + flash-attn (Day-2).

Open question (TODO #3): EpiCache's flow expects E=4 per-episode caches built
via `model.prefill_memory_constrained` and a `LongConvQAModel` wrapper. Just
monkeypatching `Qwen3Attention.forward` is necessary but NOT sufficient; the
real integration also needs the episode-cache construction + per-query
dispatch wired into our `_chunked_eval_runner.py`. Day-2 work — see
`third_party/ml-epicache-install.md` "Integration-shape caveat".

GQA configuration on Qwen3-4B-Instruct-2507: 32 query heads / 8 KV heads
(grouping=4), per AutoConfig (num_attention_heads=32, num_key_value_heads=8).
EpiCache's `EvictCache.__init__` reads `num_attention_heads` and
`num_key_value_heads` to compute `n_group_kv = n_heads // n_heads_kv`
(attention/kvcache.py:20-21) — confirmed GQA-aware, will pick up Qwen3-4B's
4:1 grouping correctly without changes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EPICACHE_ROOT = REPO_ROOT / "third_party" / "ml-epicache"


def _ensure_epicache_on_path() -> None:
    """Add the EpiCache submodule directory to sys.path so `import attention`,
    `import model`, etc. resolve at the EpiCache root rather than triggering
    any unrelated top-level package collision.

    Idempotent; safe to call multiple times.
    """
    if not EPICACHE_ROOT.exists():
        raise FileNotFoundError(
            f"EpiCache submodule not found at {EPICACHE_ROOT}. Did you forget "
            f"`git submodule update --init third_party/ml-epicache`?"
        )
    epi_str = str(EPICACHE_ROOT)
    if epi_str not in sys.path:
        sys.path.insert(0, epi_str)


def install_epicache_on_qwen3(
    model: Any,
    *,
    require_flash_attn: bool = True,
    require_tiny_api_cuda: bool = True,
) -> dict[str, Any]:
    """Apply EpiCache's `llama_flash_attn2_forward` to Qwen3Attention modules
    in the given model.

    Parameters
    ----------
    model
        A loaded HF Qwen3 model (Qwen3ForCausalLM or compatible). The model
        is mutated in place: every Qwen3Attention.forward (the class method,
        not per-instance) is reassigned to EpiCache's flash-attn 2 forward.
    require_flash_attn
        When True (default), import EpiCache's `attention.attn` module which
        in turn imports `flash_attn.flash_attn_varlen_func`. Set False to
        bypass that import in CPU-only smoke contexts; the smoke test cannot
        run an actual attention forward, only validate that the wiring
        functions resolve.
    require_tiny_api_cuda
        When True (default), import EpiCache's `attention.kvcache` which
        depends on the custom `tiny_api_cuda` extension built via
        `csrc/make`. Set False for smoke.

    Returns
    -------
    dict
        Diagnostic info: `{"patched_class": str, "n_layers_patched": int,
        "n_kv_heads": int, "n_q_heads": int, "head_dim": int}`. Useful for
        log assertions and for the smoke test to verify the right classes
        were touched.

    Raises
    ------
    RuntimeError
        If no Qwen3Attention classes are found on the model (likely wrong
        backbone) or if the EpiCache imports fail when their corresponding
        require_* flags are True.
    """
    _ensure_epicache_on_path()

    # Late imports so module load is CPU-safe.
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    except ImportError as exc:
        raise RuntimeError(
            "transformers.models.qwen3.modeling_qwen3.Qwen3Attention not "
            "importable. Need transformers>=4.51 (where Qwen3 lands)."
        ) from exc

    forward_fn = None
    if require_flash_attn or require_tiny_api_cuda:
        # Importing attention.kvcache pulls in tiny_api_cuda;
        # importing attention.attn pulls in flash_attn. Both must succeed
        # before we can attach the forward.
        try:
            if require_tiny_api_cuda:
                import attention.kvcache  # noqa: F401  (registers EvictCache)
            from attention.attn import llama_flash_attn2_forward  # noqa: E402
            forward_fn = llama_flash_attn2_forward
        except ImportError as exc:
            raise RuntimeError(
                f"EpiCache attention modules unavailable: {exc}. Install "
                f"flash-attn==2.7.4.post1 and run `cd third_party/ml-epicache/csrc && make` "
                f"on the GPU host before invoking install_epicache_on_qwen3 with "
                f"require_flash_attn=True. See third_party/ml-epicache-install.md."
            ) from exc

    # Find and replace the forward on Qwen3Attention modules. EpiCache's
    # monkeypatch.py works by reassigning the CLASS method (so every instance
    # picks it up); we mirror that pattern. Counting instances is for diagnostics.
    layers_patched = 0
    seen_classes: set[type] = set()
    n_kv_heads = None
    n_q_heads = None
    head_dim = None
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            seen_classes.add(type(module))
            layers_patched += 1
            if n_kv_heads is None:
                # Pull from config to avoid relying on attribute names that
                # transformers occasionally renames between versions.
                cfg = model.config
                n_kv_heads = getattr(cfg, "num_key_value_heads", None)
                n_q_heads = getattr(cfg, "num_attention_heads", None)
                head_dim = getattr(cfg, "head_dim", None) or (
                    cfg.hidden_size // n_q_heads if n_q_heads else None
                )

    if layers_patched == 0:
        raise RuntimeError(
            "No Qwen3Attention modules found in the supplied model. Check "
            "you are passing a Qwen3 backbone (not Qwen2.5 / Llama / SmolLM3)."
        )

    # Reassign forward at the class level so future instances also pick it up.
    # If forward_fn is None (CPU smoke), we skip the actual reassignment but
    # still return the diagnostic info — confirms the wiring resolves.
    if forward_fn is not None:
        for cls in seen_classes:
            cls.forward = forward_fn

    return {
        "patched_class": ", ".join(sorted(c.__name__ for c in seen_classes)),
        "n_layers_patched": layers_patched,
        "n_kv_heads": n_kv_heads,
        "n_q_heads": n_q_heads,
        "head_dim": head_dim,
        "forward_attached": forward_fn is not None,
    }


# Convenience: an episode-cache builder placeholder that documents the Day-2
# shape so the runner branch in `_chunked_eval_runner.py` has somewhere to
# call into. Real implementation will mirror EpiCache's
# `run_epicache.py:35-120` flow:
#   - cluster the conversation into E=4 episodes via utils.cluster.ClusterManager
#   - for each episode, `model.prefill_memory_constrained(ctx_ids, ...)`
#   - move each episode cache to CPU between questions; restore on demand
def build_episode_caches(*args, **kwargs):
    raise NotImplementedError(
        "Day-2 placeholder. The episode-cache construction needs GPU + "
        "flash-attn + EpiCache's LongConvQAModel wrapper. See "
        "third_party/ml-epicache/run_epicache.py:35-120 for the reference flow."
    )
