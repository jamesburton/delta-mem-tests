"""Wiring tests for run.locomo_eval — no model load, no GPU required.

Subtask C verification: feeds a synthetic fake-scores JSON (matching the
vendored eval's real nested output schema) through _extract_ratios and
confirms the plumbing to render_report works. Defends:
- the nested key path discovered in Subtask A
- the vendored-commit reader
- the scan_impl path-lock in EVAL_CONFIG
"""

import json
import tempfile
from pathlib import Path

from run.locomo_eval import EVAL_CONFIG, _extract_ratios, _read_vendored_commit
from run.report_gen import render_report


def _make_fake_scores(delta_score: float, base_score: float) -> dict:
    """Build a minimal scores dict matching the vendored eval's output schema.

    Schema from locomo_delta.py:977-1057: payload["base"]["summary"] and
    payload["delta"]["summary"] each contain a "full_history_replay" key.
    """
    return {
        "base": {
            "summary": {
                "full_history_replay": {
                    "overall_score": base_score,
                    "num_questions": 10,
                }
            }
        },
        "delta": {
            "summary": {
                "full_history_replay": {
                    "overall_score": delta_score,
                    "num_questions": 10,
                }
            }
        },
    }


def test_extract_ratios_parses_nested_schema():
    """_extract_ratios reads the correct nested keys from the scores JSON."""
    fake = _make_fake_scores(delta_score=0.62, base_score=0.50)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(fake, f)
        tmp_path = Path(f.name)

    try:
        dm, fz, skipped = _extract_ratios(tmp_path)
        assert dm == 0.62
        assert fz == 0.50
        assert skipped == []
    finally:
        tmp_path.unlink(missing_ok=True)


def test_extract_ratios_feeds_render_report():
    """_extract_ratios output flows correctly into render_report."""
    fake = _make_fake_scores(delta_score=0.60, base_score=0.50)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(fake, f)
        tmp_path = Path(f.name)

    try:
        dm, fz, skipped = _extract_ratios(tmp_path)
        ratio = dm / fz
        md = render_report(
            our_ratio=ratio,
            paper_ratio=1.20,
            tolerance=0.05,
            our_delta_mem_score=dm,
            our_frozen_score=fz,
            peak_vram_gb=float("nan"),
            skipped_samples=skipped,
            vendored_commit="98dc679572ef77d77b97485bf2f2b2aa810b74ba",
            eval_config=EVAL_CONFIG,
        )
        assert "# LoCoMo reproduction" in md
        assert "1.20" in md  # ratio = 0.60/0.50 = 1.20
    finally:
        tmp_path.unlink(missing_ok=True)


def test_read_vendored_commit_reads_file():
    """_read_vendored_commit reads the commit hash from report/vendored-commit.txt."""
    commit = _read_vendored_commit()
    # The file is present in the repo (Task 2 wrote it).
    assert commit == "98dc679572ef77d77b97485bf2f2b2aa810b74ba"


def test_eval_config_scan_impl_is_torch():
    """EVAL_CONFIG scan_impl is 'torch' — defends the path-lock from kernels-gate.md."""
    assert EVAL_CONFIG["scan_impl"] == "torch"
