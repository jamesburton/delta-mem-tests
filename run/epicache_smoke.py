"""CPU-only smoke test for the EpiCache Day-1 wiring.

Purpose: validate that the Python-level wiring resolves end-to-end without
touching the GPU. Specifically:

  1. The EpiCache submodule is on disk at the expected commit.
  2. `run.epicache_qwen3_adapter` imports cleanly.
  3. `install_epicache_on_qwen3` discovers Qwen3Attention modules in a
     minimal stub model and reports diagnostics.
  4. The CLI choices list in `run/locomo_eval.py` includes "epicache".
  5. `run/_chunked_eval_runner.py` recognises the new KV_CACHE_BACKEND value.

This is the "code is syntactically wired" gate, NOT the "EpiCache actually
works" gate. The actual EpiCache attention forward CANNOT run here — it
requires flash-attn and the `tiny_api_cuda` custom kernel, both of which are
Day-2 work on the GPU host.

Run:
    .venv\\Scripts\\python.exe -m run.epicache_smoke

Expected: prints OK lines for each check; exits 0 in <30 s on CPU.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EPICACHE_ROOT = REPO_ROOT / "third_party" / "ml-epicache"
EXPECTED_COMMIT = "b742661a1b763d0a57f0a1c6b82acbdbe5ed578c"


def check_submodule_present() -> None:
    if not EPICACHE_ROOT.exists():
        raise SystemExit(
            f"FAIL: {EPICACHE_ROOT} missing. Run `git submodule update --init "
            f"third_party/ml-epicache`."
        )
    head_file = EPICACHE_ROOT / ".git"
    # gitlink form: file containing `gitdir: ...`. We don't strictly need
    # the commit to match EXPECTED_COMMIT, but warn if it doesn't.
    print(f"[1/5] OK submodule present at {EPICACHE_ROOT}")
    # Verify the pin via subprocess so we don't rely on dulwich/pygit2.
    import subprocess
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(EPICACHE_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception as exc:
        print(f"      WARN: could not read submodule HEAD: {exc}")
        return
    if sha != EXPECTED_COMMIT:
        print(
            f"      WARN: submodule HEAD is {sha}; expected {EXPECTED_COMMIT}. "
            f"The Day-1 port was authored against {EXPECTED_COMMIT}; if upstream "
            f"changed Qwen2.5 attention or added Qwen3 support natively, the "
            f"adapter may need to update."
        )
    else:
        print(f"      OK   submodule pinned at {EXPECTED_COMMIT}")


def check_adapter_imports() -> None:
    try:
        from run import epicache_qwen3_adapter  # noqa: F401
    except Exception as exc:
        raise SystemExit(f"FAIL: cannot import run.epicache_qwen3_adapter: {exc}")
    print("[2/5] OK run.epicache_qwen3_adapter imported")


def check_cli_choices_extended() -> None:
    """Inspect run/locomo_eval.py source to confirm 'epicache' is in the
    --kv-cache-backend choices. We grep rather than import because importing
    locomo_eval triggers the full eval pipeline."""
    source = (REPO_ROOT / "run" / "locomo_eval.py").read_text(encoding="utf-8")
    needle = '"epicache"'
    if needle not in source:
        raise SystemExit(
            f"FAIL: {needle} not found in run/locomo_eval.py — "
            "the --kv-cache-backend choices were not extended."
        )
    print("[3/5] OK run/locomo_eval.py exposes --kv-cache-backend epicache")


def check_runner_branch_present() -> None:
    source = (REPO_ROOT / "run" / "_chunked_eval_runner.py").read_text(encoding="utf-8")
    if '"epicache"' not in source or "epicache" not in source.lower():
        raise SystemExit(
            "FAIL: run/_chunked_eval_runner.py does not reference epicache; "
            "the backend dispatch was not extended."
        )
    print("[4/5] OK run/_chunked_eval_runner.py recognises KV_CACHE_BACKEND=epicache")


def check_qwen3_adapter_against_stub() -> None:
    """Construct a tiny Qwen3 config + initialise a 2-layer model on CPU,
    then call install_epicache_on_qwen3 with require_flash_attn=False to
    confirm the adapter discovers the Qwen3Attention modules and reports
    GQA sizing.

    We avoid `from_pretrained` (network + multi-GB download) and use
    `AutoConfig` + `Qwen3ForCausalLM(config)` with hidden_size shrunk so
    the model fits in CPU RAM in <2 s.
    """
    # First, verify the real Qwen3-4B config loads (no download — needs local
    # cache; if not cached, skip silently with a note).
    real_config_ok = False
    try:
        from transformers import AutoConfig
        cfg_real = AutoConfig.from_pretrained(
            "Qwen/Qwen3-4B-Instruct-2507", local_files_only=True,
        )
        print(
            f"      Qwen3-4B config: num_attention_heads={cfg_real.num_attention_heads} "
            f"num_key_value_heads={cfg_real.num_key_value_heads} "
            f"head_dim={getattr(cfg_real, 'head_dim', None) or cfg_real.hidden_size // cfg_real.num_attention_heads}"
        )
        real_config_ok = True
    except Exception as exc:
        print(
            f"      NOTE Qwen3-4B config not in local cache ({type(exc).__name__}); "
            f"skipping AutoConfig verification. Will rely on stub-model wiring check."
        )

    # Build a 2-layer Qwen3 stub via the model class directly.
    try:
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Attention  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"FAIL: transformers Qwen3 modeling not importable: {exc}. "
            f"Need transformers>=4.51."
        )

    stub_cfg = Qwen3Config(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,  # GQA grouping = 4, mirrors Qwen3-4B's 32/8 ratio
        max_position_embeddings=1024,
        head_dim=16,
    )
    # The model build itself spins up an embedding table + 2 transformer
    # blocks on CPU; with these miniature sizes it's a sub-second op.
    import torch
    torch.set_default_dtype(torch.float32)  # avoid bf16 on CPU
    stub_model = Qwen3ForCausalLM(stub_cfg).eval()

    from run.epicache_qwen3_adapter import install_epicache_on_qwen3
    diag = install_epicache_on_qwen3(
        stub_model,
        require_flash_attn=False,
        require_tiny_api_cuda=False,
    )

    if diag["n_layers_patched"] != 2:
        raise SystemExit(
            f"FAIL: expected 2 Qwen3Attention modules in the stub; "
            f"adapter discovered {diag['n_layers_patched']}."
        )
    if diag["n_kv_heads"] != 2 or diag["n_q_heads"] != 8:
        raise SystemExit(
            f"FAIL: GQA sizing wrong: {diag}. Expected n_q_heads=8, n_kv_heads=2."
        )
    if diag["forward_attached"]:
        raise SystemExit(
            "FAIL: forward_attached=True under CPU smoke (require_flash_attn=False); "
            "should be False — we deliberately skipped the GPU-only attention import."
        )
    print(
        f"[5/5] OK install_epicache_on_qwen3 wired against Qwen3 stub: {diag}"
    )


def main() -> int:
    # Make sure we can find the repo `run` package when invoked as a script
    # rather than via `python -m`.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    print("=== EpiCache Day-1 wiring smoke (CPU only) ===")
    check_submodule_present()
    check_adapter_imports()
    check_cli_choices_extended()
    check_runner_branch_present()
    check_qwen3_adapter_against_stub()
    print("=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
