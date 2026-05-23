"""Pure-stdlib scan of LoCoMo conversation lengths.

Avoids importing transformers/torch entirely (host is under memory pressure
from a concurrently-running eval; loading torch DLLs fails). Mirrors the
exact text-construction of locomo_protocol.build_locomo_history_messages
and back-computes a tokens/char ratio from the conv-0 token count we
already saw in the eval log (17,600 tokens for sample 0).

Safe to run alongside an active GPU eval — pure CPU, stdlib only.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "delta-Mem" / "data" / "locomo10.json"

# Verbatim from locomo_protocol.OFFICIAL_SYSTEM_PROMPT
OFFICIAL_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant whose job is to understand "
    "the following conversation and answer questions based on the conversation. "
    "If you don't know the answer to a question, please don't share false information."
)

# Conv 0 had history=17,600 tokens per the chunked-prefill log line:
#   "[chunked-prefill 20/20] prefix=16920 suffix=680 total=17600"
KNOWN_CONV0_TOKENS = 17600


def render_turn(dialog: dict) -> str:
    """Mirror locomo_protocol.render_locomo_turn."""
    turn = f'{dialog["speaker"]} said, "{dialog["text"]}"\n'
    if dialog.get("blip_caption"):
        turn += f' and shared {dialog["blip_caption"]}.'
    turn += "\n"
    return turn


def build_session_text(conversation: dict, session_num: int) -> str:
    session_key = f"session_{session_num}"
    date_key = f"{session_key}_date_time"
    turns = "".join(
        render_turn(dialog) for dialog in conversation[session_key]
    ).rstrip()
    return f"DATE: {conversation[date_key]}\nCONVERSATION:\n{turns}"


def conversation_text(sample: dict) -> tuple[str, int]:
    """Concatenate all session messages exactly as build_locomo_history_messages
    would, including the system prompt. Returns (full_text, num_sessions).
    """
    conversation = sample["conversation"]
    session_nums = sorted(
        int(key.split("_")[-1])
        for key in conversation
        if key.startswith("session_") and not key.endswith("date_time")
    )
    parts: list[str] = [OFFICIAL_SYSTEM_PROMPT]
    n_sessions = 0
    for n in session_nums:
        if not conversation[f"session_{n}"]:
            continue
        parts.append(build_session_text(conversation, n))
        n_sessions += 1
    return "\n".join(parts), n_sessions


def main() -> int:
    print(f"Loading data: {DATA_FILE}")
    samples = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    print(f"Found {len(samples)} conversations\n")

    rows: list[tuple[int, int, int, int]] = []  # (idx, chars, sessions, questions)
    for idx, sample in enumerate(samples):
        text, n_sessions = conversation_text(sample)
        n_questions = len(sample.get("qa", []))
        rows.append((idx, len(text), n_sessions, n_questions))

    # Back-compute tokens/char ratio from conv 0.
    ratio = KNOWN_CONV0_TOKENS / rows[0][1]
    print(f"Conv 0: {rows[0][1]} chars -> {KNOWN_CONV0_TOKENS} tokens (observed)")
    print(f"Estimated tokens/char ratio: {ratio:.4f}\n")

    print(f"{'conv':>4}  {'chars':>7}  {'est_tok':>8}  {'sess':>4}  {'q':>4}")
    for idx, chars, sess, q in rows:
        est_tok = int(chars * ratio)
        print(f"{idx:>4}  {chars:>7}  {est_tok:>8}  {sess:>4}  {q:>4}")

    est_tokens = [int(c * ratio) for _, c, _, _ in rows]
    questions = [r[3] for r in rows]
    sessions = [r[2] for r in rows]
    total_q = sum(questions)
    print("\n--- Summary ---")
    print(f"Conversations:   {len(rows)}")
    print(f"Total questions: {total_q}")
    print(f"Est tokens:      min={min(est_tokens)}  median={int(statistics.median(est_tokens))}  "
          f"mean={int(statistics.mean(est_tokens))}  max={max(est_tokens)}  sum={sum(est_tokens)}")
    print(f"Sessions:        min={min(sessions)}  median={int(statistics.median(sessions))}  "
          f"max={max(sessions)}")
    print(f"Questions:       min={min(questions)}  median={int(statistics.median(questions))}  "
          f"mean={statistics.mean(questions):.1f}  max={max(questions)}")

    # Flag conversations longer than conv 0 — those are batch=2 OOM risks.
    threshold = est_tokens[0]
    longer = [(i, t) for i, t in enumerate(est_tokens) if t > threshold]
    if longer:
        print(f"\nConvs longer than conv 0 ({threshold} tokens) — potential batch=2 OOM:")
        for i, t in sorted(longer, key=lambda x: -x[1]):
            print(f"  conv {i}: ~{t} tokens ({t - threshold:+d} vs conv 0)")
    else:
        print(f"\nNo conv exceeds conv 0 ({threshold} tokens) — batch=2 should hold.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
