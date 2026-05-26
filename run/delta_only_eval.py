"""Delta-mem-only experiment — does the compressed delta-mem state alone
preserve enough context to answer LoCoMo questions?

The standard vendored eval feeds the full ~17.6k-token conversation history
into the prompt for both the base and delta-mem branches. That measures
"does delta-mem help on top of full attention", not "can delta-mem replace
the full history". Our prior 0.99x result on that mode says no on top.

This script measures the second, more interesting question:

  Condition A — truncated_base
    Frozen Qwen3-4B-Instruct-2507, no delta-mem adapter.
    Prompt = [system, user(question_only)].
    The model has zero memory of the conversation.

  Condition B — delta_only
    Qwen3-4B + delta-mem adapter, with the per-conversation delta-mem state
    pre-loaded (built by chunked-prefilling the full history once).
    Prompt = [system, user(question_only)].
    The model has only its compressed delta-mem state to "remember" the
    conversation; the history is NOT in the prompt.

If delta_only >> truncated_base, delta-mem is doing real memory work.
If delta_only ~= truncated_base, the compression is throwing the signal
away. If delta_only approaches the full-history scores we already
measured (~0.36), the compression preserves most of the useful signal.

Memory budget on a 12 GB card with this approach: weights (~8 GB) +
delta-mem state (~300 MB) + tiny KV for ~100-token prompt + scratch
= comfortable headroom. Should be MUCH cheaper to run than the
full-history eval and could be the basis of running real long-context
workloads (50k+) where full KV doesn't fit.

Usage:
    uv run python -m run.delta_only_eval --max-conversations 1 \\
        --output-json outputs/delta_only_conv0.json

Output JSON shape mirrors the vendored eval enough for our render_report
helper, with two conditions per question in the same record. A small
summary block at the top includes the full-history baselines for
reference.
"""

from __future__ import annotations

import os
# Path-lock per report/kernels-gate.md (must come before any deltamem import)
os.environ.setdefault("DELTA_MEM_SCAN_IMPL", "torch")

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamem.eval.locomo_delta import (
    attach_delta_adapter_in_place,
    load_locomo_samples,
)
from deltamem.eval.locomo_protocol import (
    OFFICIAL_ANSWER_RESERVE_TOKENS,
    OFFICIAL_MAX_NEW_TOKENS,
    OFFICIAL_SYSTEM_PROMPT,
    OFFICIAL_TEMPERATURE,
    OFFICIAL_TOP_K,
    OFFICIAL_TOP_P,
    build_locomo_history_messages,
    build_official_question_prompt,
    canonicalize_locomo_prediction,
    prepare_locomo_question,
    score_locomo_prediction,
    summarize_locomo_records,
)
from deltamem.runtime.session import DeltaMemChatSession
from deltamem.core import (
    iter_delta_mem_modules,
    reset_delta_mem_states,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = REPO_ROOT / "delta-Mem" / "data" / "locomo10.json"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_ID = "declare-lab/delta-mem_qwen3_4b-instruct"
DEFAULT_CATEGORIES = (1, 2, 3, 4)
SEED = 42
PREFILL_CHUNK = 1024
DEVICE = "cuda:0"
DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "sdpa"
CONDITION_TRUNCATED_BASE = "truncated_base"
CONDITION_DELTA_ONLY = "delta_only"


def _build_question_messages(question_prompt: str) -> list[dict[str, str]]:
    """Minimal prompt — no conversation history."""
    return [
        {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT},
        {"role": "user", "content": question_prompt},
    ]


def _generate_short(
    *,
    model,
    tokenizer,
    messages: list[dict[str, str]],
    seed: int,
    max_new_tokens: int,
) -> str:
    """Vendored-style generate with the same temperature/top_k/top_p as the
    official eval. No KV cache reuse — every call is fresh."""
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids.input_ids
    input_ids = input_ids.to(DEVICE)

    rng_devices = [torch.device(DEVICE)] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=True,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=OFFICIAL_TEMPERATURE,
                top_p=OFFICIAL_TOP_P,
                top_k=OFFICIAL_TOP_K,
            )
    generated_ids = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _chunked_ingest_history(session, history_messages: list[dict[str, str]]) -> None:
    """Build the delta-mem state by chunked-prefilling the full history."""
    session.messages = [dict(m) for m in history_messages]
    full_ids = session._tokenize_messages(session.messages, add_generation_prompt=False)
    total = int(full_ids.shape[1])
    end = 0
    chunk_idx = 0
    t0 = time.perf_counter()
    while end < total:
        end = min(end + PREFILL_CHUNK, total)
        chunk_idx += 1
        session._ingest_full_ids(full_ids[:, :end])
        stats = session.last_ingest_stats
        if chunk_idx > 1 and stats.get("rebuilt", False):
            raise RuntimeError(
                f"Chunked history prefill rebuilt at chunk {chunk_idx}: {stats}"
            )
    elapsed = time.perf_counter() - t0
    print(
        f"  [history-prefill] {total} tokens in {chunk_idx} chunks "
        f"({elapsed:.1f}s)",
        flush=True,
    )


def _snapshot_delta_state(model) -> dict[str, torch.Tensor]:
    return {
        name: module.delta_state.detach().clone()
        for name, module in iter_delta_mem_modules(model)
        if module.delta_state is not None
    }


def _restore_delta_state(model, snapshot: dict[str, torch.Tensor]) -> None:
    module_map = dict(model.named_modules())
    for name, tensor in snapshot.items():
        module = module_map[name]
        module.delta_state = tensor.detach().clone()


def _run_condition_truncated_base(
    *,
    samples: list[dict],
    model_path: str,
    max_new_tokens: int,
    out_records: list[dict],
) -> None:
    """Pass 1: frozen base, no delta-mem, prompt = [system, question]."""
    print(f"\n=== Condition A: truncated_base ===")
    print(f"Loading base model ({MODEL_ID}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": DEVICE},
        attn_implementation=ATTN_IMPLEMENTATION,
        local_files_only=True,
    ).eval()

    for s_idx, sample in enumerate(samples):
        sid = str(sample["sample_id"])
        rec = out_records[s_idx]
        assert rec["sample_id"] == sid
        for q_idx, qa in enumerate(sample["qa"]):
            spec = prepare_locomo_question(
                qa, sample_id=sid, question_index=q_idx, seed=SEED,
            )
            question_prompt = build_official_question_prompt(spec)
            messages = _build_question_messages(question_prompt)
            t0 = time.perf_counter()
            raw_prediction = _generate_short(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                seed=SEED + q_idx,
                max_new_tokens=max_new_tokens,
            )
            elapsed = time.perf_counter() - t0
            canonical = canonicalize_locomo_prediction(raw_prediction, spec)
            score = round(score_locomo_prediction(qa, canonical), 4)
            qa_rec = rec["qa"][q_idx]
            qa_rec.setdefault("conditions", {})[CONDITION_TRUNCATED_BASE] = {
                "prediction": canonical,
                "raw_prediction": raw_prediction,
                "score": score,
                "turn_stats": {
                    "condition_name": CONDITION_TRUNCATED_BASE,
                    "elapsed_ms": round(elapsed * 1000, 1),
                },
            }
            if q_idx % 25 == 0 or q_idx == len(sample["qa"]) - 1:
                print(
                    f"  [{sid} A {q_idx+1}/{len(sample['qa'])}] "
                    f"score={score:.4f} ({elapsed:.1f}s) "
                    f"pred={raw_prediction[:60]!r}",
                    flush=True,
                )

    # Free the base model before loading delta.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_condition_delta_only(
    *,
    samples: list[dict],
    model_path: str,
    adapter_dir: str,
    max_new_tokens: int,
    out_records: list[dict],
) -> None:
    """Pass 2: delta-adapted model with per-conversation snapshot loaded,
    prompt = [system, question] (no history)."""
    print(f"\n=== Condition B: delta_only ===")
    print(f"Loading delta model ({MODEL_ID} + {ADAPTER_ID}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": DEVICE},
        attn_implementation=ATTN_IMPLEMENTATION,
        local_files_only=True,
    ).eval()
    attach_delta_adapter_in_place(
        model,
        adapter_dir=Path(adapter_dir),
        rank=8,
        alpha=16.0,
        beta_bias_init=-1.5,
        rankwise_gates=True,
        output_init="base_slice_fixed",
        online_gain=0.05,
    )

    for s_idx, sample in enumerate(samples):
        sid = str(sample["sample_id"])
        rec = out_records[s_idx]
        assert rec["sample_id"] == sid
        print(f"\n  Conversation {sid}: building delta-mem state from history ...")
        # Build the snapshot from the full history (the same way our
        # chunked-prefill snapshot patch does).
        reset_delta_mem_states(model)
        session = DeltaMemChatSession(
            model=model, tokenizer=tokenizer, device=DEVICE,
        )
        history_messages = build_locomo_history_messages(sample)
        _chunked_ingest_history(session, history_messages)
        snapshot = _snapshot_delta_state(model)

        # Free the session's KV cache before per-question generation;
        # we only need the delta-mem state, not the long KV history.
        del session
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for q_idx, qa in enumerate(sample["qa"]):
            spec = prepare_locomo_question(
                qa, sample_id=sid, question_index=q_idx, seed=SEED,
            )
            question_prompt = build_official_question_prompt(spec)
            messages = _build_question_messages(question_prompt)

            # Restore the snapshot's delta-mem state so the model has
            # "memory" of the history during this short-prompt generation.
            _restore_delta_state(model, snapshot)

            t0 = time.perf_counter()
            raw_prediction = _generate_short(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                seed=SEED + q_idx,
                max_new_tokens=max_new_tokens,
            )
            elapsed = time.perf_counter() - t0
            canonical = canonicalize_locomo_prediction(raw_prediction, spec)
            score = round(score_locomo_prediction(qa, canonical), 4)
            qa_rec = rec["qa"][q_idx]
            qa_rec.setdefault("conditions", {})[CONDITION_DELTA_ONLY] = {
                "prediction": canonical,
                "raw_prediction": raw_prediction,
                "score": score,
                "turn_stats": {
                    "condition_name": CONDITION_DELTA_ONLY,
                    "elapsed_ms": round(elapsed * 1000, 1),
                },
            }
            if q_idx % 25 == 0 or q_idx == len(sample["qa"]) - 1:
                print(
                    f"  [{sid} B {q_idx+1}/{len(sample['qa'])}] "
                    f"score={score:.4f} ({elapsed:.1f}s) "
                    f"pred={raw_prediction[:60]!r}",
                    flush=True,
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Delta-mem-only experiment: does the compressed delta-mem state "
            "alone preserve enough context to answer LoCoMo questions?"
        )
    )
    parser.add_argument(
        "--data-file",
        default=str(DEFAULT_DATA_FILE),
        help="LoCoMo data file (default: delta-Mem/data/locomo10.json).",
    )
    parser.add_argument("--max-conversations", type=int, default=1)
    parser.add_argument("--max-questions-per-conversation", type=int, default=None)
    parser.add_argument(
        "--categories", type=int, nargs="+", default=list(DEFAULT_CATEGORIES),
    )
    parser.add_argument("--max-new-tokens", type=int, default=OFFICIAL_MAX_NEW_TOKENS)
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "outputs" / "delta_only_conv0.json"),
    )
    args = parser.parse_args()

    print(f"Resolving model: {MODEL_ID}")
    model_path = snapshot_download(MODEL_ID)
    print(f"Resolving adapter: {ADAPTER_ID}")
    adapter_path = snapshot_download(ADAPTER_ID)

    samples = load_locomo_samples(
        Path(args.data_file),
        max_conversations=args.max_conversations,
        max_questions_per_conversation=args.max_questions_per_conversation,
        categories=args.categories,
    )
    n_questions = sum(len(s["qa"]) for s in samples)
    print(
        f"Eval setup: {len(samples)} conversations, "
        f"{n_questions} questions (categories {args.categories})."
    )

    # Build per-conversation skeleton records — populated as we evaluate.
    out_records: list[dict[str, Any]] = []
    for sample in samples:
        out_records.append({
            "sample_id": str(sample["sample_id"]),
            "speakers": [
                sample["conversation"].get("speaker_a"),
                sample["conversation"].get("speaker_b"),
            ],
            "num_sessions": sum(
                1 for k in sample["conversation"]
                if k.startswith("session_") and not k.endswith("date_time")
            ),
            "qa": [
                {
                    "question": qa["question"],
                    "answer": qa.get("answer"),
                    "adversarial_answer": qa.get("adversarial_answer"),
                    "evidence": list(qa.get("evidence", [])),
                    "category": int(qa["category"]),
                    "conditions": {},
                }
                for qa in sample["qa"]
            ],
        })

    # Pass 1: truncated_base (frozen Qwen3, no delta-mem). Loading model A.
    _run_condition_truncated_base(
        samples=samples,
        model_path=model_path,
        max_new_tokens=args.max_new_tokens,
        out_records=out_records,
    )

    # Pass 2: delta_only (delta-adapted model, snapshot loaded, short prompt).
    _run_condition_delta_only(
        samples=samples,
        model_path=model_path,
        adapter_dir=adapter_path,
        max_new_tokens=args.max_new_tokens,
        out_records=out_records,
    )

    # Summarise.
    condition_names = [CONDITION_TRUNCATED_BASE, CONDITION_DELTA_ONLY]
    summary = summarize_locomo_records(out_records, condition_names=condition_names)
    payload = {
        "model": MODEL_ID,
        "adapter": ADAPTER_ID,
        "data_file": str(args.data_file),
        "num_conversations": len(samples),
        "num_questions": n_questions,
        "categories": args.categories,
        "max_new_tokens": args.max_new_tokens,
        "seed": SEED,
        "scan_impl": "torch",
        "records": out_records,
        "summary": summary,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Summary ===")
    for cond in condition_names:
        cs = summary[cond]
        print(f"  {cond}: overall_score={cs['overall_score']:.4f} (n={cs['num_questions']})")
        for cat, scores in cs["category_scores"].items():
            print(f"    cat {cat} ({scores['name']:>12}): {scores['score']:.4f}  n={scores['count']}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
