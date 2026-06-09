"""Download + tokenise the long-context training mix for Strix Halo phase 1.

Produces a single ``data/longctx_mix_v1.jsonl`` (gitignored) with one
example per line in the format::

    {"input_ids": [int, ...], "labels": [int, ...]}

The mix targets the curriculum in ``STRIX_INSTRUCTIONS.md``:

  - 50% LoCoMo originals (already in ``data/locomo10.json``) sliced into
    8-18 k token chunks. Anchors the published-adapter strengths.
  - 30% LongMemEval (``xiaowu0162/longmemeval-cleaned`` — the older
    ``long-mem-eval`` ID in the doc is now deprecated; we use the cleaned
    replacement) sliced into 20-32 k chunks.
  - 20% InfBench (``xinrongzhang2022/InfiniteBench``,
    ``longdialogue_qa_eng`` split — the closest match to "mem-specific";
    falls back to ``longbook_qa_eng`` if missing) sliced into 32 k+
    chunks.

Idempotent: if the output JSONL already exists, the script exits early.

Usage:
    python -m strix.prepare_data
    python -m strix.prepare_data --out data/longctx_mix_v1.jsonl --max-per-source 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LOCOMO_PATH = Path("data/locomo10.json")
DEFAULT_OUT = Path("data/longctx_mix_v1.jsonl")

# (dataset_id, split, target chunk size, share weight)
LOCOMO_TARGET = 16_384   # mid of the 8-18k band
LONGMEMEVAL_TARGET = 28_672  # mid of the 20-32k band
INFBENCH_TARGET = 36_864     # just above 32k


def _format_locomo_session(messages) -> str:
    lines: List[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            spk = msg.get("speaker", "?")
            txt = msg.get("text", "")
            lines.append(f"{spk}: {txt}")
        else:
            lines.append(str(msg))
    return "\n".join(lines)


def _iter_locomo_texts() -> Iterable[str]:
    """Yield one long-form conversation text per LoCoMo sample."""
    if not LOCOMO_PATH.exists():
        print(f"[prep] WARN: {LOCOMO_PATH} not found, skipping LoCoMo share",
              flush=True)
        return
    raw = json.loads(LOCOMO_PATH.read_text(encoding="utf-8"))
    for conv in raw:
        convo = conv.get("conversation", {})
        keys = sorted(
            [k for k in convo if k.startswith("session_") and not k.endswith("date_time")],
            key=lambda x: int(x.split("_")[1]),
        )
        if not keys:
            continue
        yield "\n\n".join(_format_locomo_session(convo[k]) for k in keys)


def _iter_longmemeval_texts(max_items: int) -> Iterable[str]:
    """Yield long-form session-history texts from LongMemEval (cleaned)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[prep] WARN: `datasets` not installed; skipping LongMemEval",
              flush=True)
        return
    try:
        ds = load_dataset("xiaowu0162/longmemeval-cleaned",
                          split="longmemeval_s_cleaned")
    except Exception as exc:  # noqa: BLE001
        print(f"[prep] WARN: LongMemEval load failed: {exc}", flush=True)
        return
    count = 0
    for row in ds:
        # The schema: row has `haystack_sessions` (list of list of dicts) +
        # `question` + `answer`. We assemble a dialogue from the sessions
        # and append the QA inline as the training target context.
        chunks: List[str] = []
        sessions = row.get("haystack_sessions") or row.get("sessions") or []
        for sess in sessions:
            if isinstance(sess, list):
                for turn in sess:
                    if isinstance(turn, dict):
                        role = turn.get("role", "user")
                        content = turn.get("content", "")
                        chunks.append(f"{role}: {content}")
        q = row.get("question") or ""
        a = row.get("answer") or ""
        if q:
            chunks.append(f"user: {q}")
        if a:
            chunks.append(f"assistant: {a}")
        if chunks:
            yield "\n".join(chunks)
            count += 1
            if count >= max_items:
                return


def _iter_infbench_texts(max_items: int) -> Iterable[str]:
    """Yield long-form context+QA texts from InfiniteBench long-dialogue split."""
    try:
        from datasets import load_dataset, Features, Value, Sequence
    except ImportError:
        print("[prep] WARN: `datasets` not installed; skipping InfBench",
              flush=True)
        return
    features = Features({
        "id": Value("int64"),
        "context": Value("string"),
        "input": Value("string"),
        "answer": Sequence(Value("string")),
        "options": Sequence(Value("string")),
    })
    for split in ("longdialogue_qa_eng", "longbook_qa_eng"):
        try:
            ds = load_dataset("xinrongzhang2022/InfiniteBench",
                              features=features, split=split)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[prep] InfBench split {split} not loadable: {exc}",
                  flush=True)
            ds = None
    if ds is None:
        print("[prep] WARN: no InfBench split loaded; skipping share",
              flush=True)
        return
    count = 0
    for row in ds:
        ctx = row.get("context") or ""
        q = row.get("input") or ""
        ans_list = row.get("answer") or []
        ans = ans_list[0] if ans_list else ""
        if not ctx:
            continue
        text = ctx
        if q:
            text += f"\n\nuser: {q}"
        if ans:
            text += f"\nassistant: {ans}"
        yield text
        count += 1
        if count >= max_items:
            return


def _chunk(tokenizer, text: str, target: int) -> List[List[int]]:
    """Tokenise a text and slice it into ``target``-sized chunks (drop tail)."""
    ids = tokenizer(text, add_special_tokens=False,
                    return_tensors=None).input_ids
    if isinstance(ids[0], list):  # batched form
        ids = ids[0]
    out: List[List[int]] = []
    for start in range(0, len(ids) - target + 1, target):
        out.append(ids[start:start + target])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"Output JSONL path (default {DEFAULT_OUT}).")
    ap.add_argument("--max-per-source", type=int, default=2000,
                    help="Cap raw rows pulled per HF dataset (before chunking).")
    ap.add_argument("--force", action="store_true",
                    help="Re-build even if output exists.")
    args = ap.parse_args()

    out_path = args.out.resolve()
    if out_path.exists() and not args.force:
        n = sum(1 for _ in out_path.open(encoding="utf-8"))
        print(f"[prep] {out_path} already exists with {n} examples; "
              f"skipping (use --force to rebuild)", flush=True)
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[prep] resolving tokenizer for {MODEL_ID} ...", flush=True)
    tok_dir = snapshot_download(MODEL_ID, allow_patterns=["tokenizer*", "*.json"])
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    sources = [
        ("locomo", _iter_locomo_texts(), LOCOMO_TARGET),
        ("longmemeval", _iter_longmemeval_texts(args.max_per_source),
         LONGMEMEVAL_TARGET),
        ("infbench", _iter_infbench_texts(args.max_per_source),
         INFBENCH_TARGET),
    ]

    by_source: dict[str, List[List[int]]] = {}
    for name, it, target in sources:
        print(f"[prep] processing {name} (target {target} tokens/chunk) ...",
              flush=True)
        bucket: List[List[int]] = []
        for i, text in enumerate(it):
            chunks = _chunk(tokenizer, text, target)
            bucket.extend(chunks)
            if (i + 1) % 50 == 0:
                print(f"  {name}: {i + 1} raw rows -> {len(bucket)} chunks",
                      flush=True)
        by_source[name] = bucket
        print(f"  {name}: final {len(bucket)} chunks", flush=True)

    # Compose the 50/30/20 mix; we let LoCoMo cap the totals and downsample
    # the others to hit the ratio.
    n_locomo = len(by_source.get("locomo", []))
    if n_locomo == 0:
        print("[prep] FATAL: no LoCoMo chunks produced; check data/locomo10.json",
              file=sys.stderr)
        return 1
    n_total_target = int(n_locomo / 0.5)
    n_long = min(len(by_source.get("longmemeval", [])),
                 int(n_total_target * 0.30))
    n_inf = min(len(by_source.get("infbench", [])),
                int(n_total_target * 0.20))
    print(f"[prep] mix target: locomo={n_locomo} longmemeval={n_long} "
          f"infbench={n_inf} (total {n_locomo + n_long + n_inf})", flush=True)

    selected: List[List[int]] = []
    selected.extend(by_source["locomo"][:n_locomo])
    selected.extend(by_source.get("longmemeval", [])[:n_long])
    selected.extend(by_source.get("infbench", [])[:n_inf])

    # Interleave so the trainer sees a curriculum mix, not blocks.
    # Simple stride interleave: rotate sources round-robin.
    import random
    random.seed(0)
    random.shuffle(selected)

    print(f"[prep] writing {len(selected)} examples to {out_path} ...", flush=True)
    lengths: List[int] = []
    with out_path.open("w", encoding="utf-8") as f:
        for ids in selected:
            ids_list = list(ids) if not isinstance(ids, list) else ids
            f.write(json.dumps({"input_ids": ids_list, "labels": ids_list}))
            f.write("\n")
            lengths.append(len(ids_list))

    if lengths:
        lengths.sort()
        p50 = lengths[len(lengths) // 2]
        p10 = lengths[len(lengths) // 10] if len(lengths) >= 10 else lengths[0]
        p90 = lengths[(len(lengths) * 9) // 10] if len(lengths) >= 10 else lengths[-1]
        print(f"[prep] token-length distribution: "
              f"min={min(lengths)} p10={p10} p50={p50} p90={p90} max={max(lengths)}",
              flush=True)
    print(f"[prep] done. {len(selected)} examples ready at {out_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
