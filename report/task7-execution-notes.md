# Task 7 — execution notes

This file captures what happened during Task 7 execution that is NOT in the
original plan (`docs/superpowers/plans/2026-05-22-delta-mem-tier1-reproduction.md`).
The plan is a stable artifact recording the original intent; this file records
the deviations that actually shipped.

## What the plan said

> If the run OOMs mid-eval, this is risk R2 materialising: lower
> `EVAL_CONFIG["max_seq_len"]` … or record the affected samples as skipped.

## What actually happened

Two OOMs occurred in the first attempt at Step 2:

1. **First OOM** — base-eval branch, ~36.93 GB allocation request. Original
   diagnosis ("base-model only, --skip-base will help") was wrong; the OOM was
   really inside `build_teacher_forced_snapshot`'s single monolithic forward on
   the full ~26k-token conversation, hitting SDPA's O(N²) attention scratch.
2. **Second OOM** — delta-eval branch, after `--skip-base` attempt. Same root
   cause; --skip-base does nothing for this code path.

`max_seq_len` lowering was rejected: the conversations *are* ~26k tokens and
truncating would change what is actually being scored, violating R3
(methodology fidelity). Sample skipping was also rejected: every conversation
would be skipped.

## Approved deviation: chunked prefill

User-approved on 2026-05-23: replace `build_teacher_forced_snapshot`'s single
`session._ingest_full_ids(full_history)` with a per-message loop that calls
`_ingest_full_ids` once per message. The session's existing prefix-skip logic
(`delta-Mem/deltamem/runtime/session.py:519-545`) means each call only
processes the new suffix — peak attention scratch shrinks to the per-message
size while the final KV cache and delta-mem state are bitwise-identical.

**Implementation:**
- `run/_chunked_eval_runner.py` applies the monkeypatch in-process and calls
  `deltamem.eval.locomo_delta.main()`.
- `run/locomo_eval.py` invokes this driver via `python -m run._chunked_eval_runner`
  instead of the raw eval module.
- The vendored submodule is NOT modified on disk; the pinned commit hash
  (`98dc679572ef77d77b97485bf2f2b2aa810b74ba`) is unchanged.
- `EVAL_CONFIG["methodology_adjustment"]` records this and `render_report`
  prepends a "Methodology adjustments" section to the report itself.
- A `rebuilt` safety check in the runner fails fast if the chat template
  emits history-dependent control tokens that defeat the prefix-skip — we
  do not want to silently degrade and OOM at the same point.

**Why this is not a numerical change:** autoregressive attention depends only
on prior tokens through the KV cache, and the KV cache is built by the same
forward pass in both versions; only the batching of that forward differs.
Same per-token logits, same delta-mem updates, same final snapshot.

## Status at end of session

- Chunked runner committed.
- Awaiting full eval run (Step 2). The user mentioned a fallback machine
  (T5500, internal RTX 3060) if this host OOMs or crashes — the chunked
  patch should fit comfortably in 12 GB even on slower hardware. The
  invocation is the same: `uv run python -m run.locomo_eval`.

## If you are picking this up on the T5500

1. Verify env: `uv sync && uv run pytest tests/ -v` (all 10 should pass).
2. Verify path-lock: smoke test still works.
   `uv run python -m run.smoke_chat`
3. Run the full eval: `uv run python -m run.locomo_eval 2>&1 | Tee-Object report/raw/locomo-driver.log`
4. Watch `report/raw/locomo-stdout.log` for `[chunked-prefill i/N]` lines —
   they confirm the patch is active. Each line should show `suffix=` much
   smaller than `total=`; if `suffix=` ever equals `total=` past the first
   message, the safety assertion will fire and abort.
5. On completion, `report/reproduction-report.md` is written and the verdict
   is one of PASS / OUT_OF_BAND / REGRESSION.
