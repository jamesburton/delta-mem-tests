"""Build custom LoCoMo data files for context-size scaling experiments.

Two modes:
  - single CONV-ID    write data/locomo_<CONV>.json with just that conversation,
                      so `--data-file ... --max-conversations 1` runs only it.
  - extend CONV-ID N  write data/locomo_<CONV>_x<N>.json with the chosen
                      conversation's sessions duplicated N times (preserving QA
                      against original session timestamps; we keep the original
                      session_1..K and append session_(K+1)..(K*N) clones).

Usage:
    python -m run.build_context_sweep_data single conv-41
    python -m run.build_context_sweep_data extend conv-26 2     # ~2x sessions
    python -m run.build_context_sweep_data extend conv-26 3     # ~3x sessions

The extender preserves all QA items (still answerable from the original
sessions). Extra duplicated sessions add filler context — they push the
prefill token count up without changing the gold answers.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


def _load_all() -> list:
    return json.loads(Path("data/locomo10.json").read_text())


def _find_by_id(data: list, sample_id: str) -> dict:
    for conv in data:
        if conv.get("sample_id") == sample_id:
            return conv
    raise SystemExit(f"sample_id {sample_id!r} not found")


def _session_keys(conv: dict) -> list[str]:
    convo = conv["conversation"]
    return sorted(
        [k for k in convo if k.startswith("session_") and not k.endswith("date_time")],
        key=lambda x: int(x.split("_")[1]),
    )


def cmd_single(sample_id: str) -> Path:
    data = _load_all()
    conv = _find_by_id(data, sample_id)
    out = Path(f"data/locomo_{sample_id}.json")
    out.write_text(json.dumps([conv]))
    return out


def cmd_extend(sample_id: str, repeats: int) -> Path:
    if repeats < 1:
        raise SystemExit("repeats must be >= 1")
    data = _load_all()
    conv = copy.deepcopy(_find_by_id(data, sample_id))
    convo = conv["conversation"]
    orig_keys = _session_keys(conv)
    K = len(orig_keys)
    last_idx = int(orig_keys[-1].split("_")[1])

    for rep in range(1, repeats):
        for i, k in enumerate(orig_keys):
            new_idx = last_idx + rep * K + i + 1
            new_key = f"session_{new_idx}"
            new_dt_key = f"session_{new_idx}_date_time"
            convo[new_key] = copy.deepcopy(convo[k])
            dt_key = f"{k}_date_time"
            if dt_key in convo:
                convo[new_dt_key] = convo[dt_key]

    out = Path(f"data/locomo_{sample_id}_x{repeats}.json")
    out.write_text(json.dumps([conv]))
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    mode = sys.argv[1]
    if mode == "single":
        p = cmd_single(sys.argv[2])
    elif mode == "extend":
        p = cmd_extend(sys.argv[2], int(sys.argv[3]))
    else:
        print(f"unknown mode {mode!r}")
        return 2
    print(f"wrote {p} ({p.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
